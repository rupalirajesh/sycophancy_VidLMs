#!/usr/bin/env python3
"""
Layer 5a — attention knockout of pressure tokens.

Zeros attention FROM every position TO the pressure-sentence tokens, for a
sliding band of layers, on items that flipped from correct under real
pressure (drawn from results_dilution_<model>.jsonl's "realN" conditions, or
results_probe_<model>.jsonl — anything using the raw video at a fixed frame
count works, since contexts are rebuilt from saved response text via
mech_utils.build_contexts). Checks whether that band's knockout restores the
original (pre-pressure) answer — the causal, not correlational, version of
"does attention carry the capitulation."

Scope note carried over from the earlier package: this operates on the FINAL
forced-choice readout over the already-generated pressured response text, not
during live generation of that response — a positive result shows the answer-
commitment step is causally reversible, not where the model decided to
capitulate while writing its response. Keep that distinction in any writeup.
"""
import argparse, json, sys
from pathlib import Path

import torch

import common
import mech_utils
from common import Engine, log, init_log, load_jsonl, append_jsonl, FORCED_PREFIX


def load_source_items(args):
    """Rebuilds a video_id/qid -> item lookup from whichever dataset the
    source records came from, so we have `question`/`options` to feed
    build_contexts (the results JSONL only stores IDs, not full items)."""
    sys.path.insert(0, str(Path(__file__).parent / "datasets"))
    lookup = {}
    if args.pt_dir:
        import perception_test_items
        ann = Path(args.pt_dir) / (f"{args.pt_split}.json" if args.pt_split == "sample"
                                    else f"all_{args.pt_split}.json")
        for it in perception_test_items.build_items(ann, Path(args.pt_dir) / "videos"):
            lookup[("perception_test", it["video_id"], it["qid"])] = it
    if args.nextqa_dir and args.nextqa_video_dir:
        import nextqa_items
        for split in args.nextqa_splits:
            for it in nextqa_items.build_items(args.nextqa_dir, args.nextqa_video_dir, split=split):
                lookup[("nextqa", it["video_id"], it["qid"])] = it
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-jsonl", required=True,
                     help="e.g. results_dilution_<model>.jsonl or results_probe_<model>.jsonl")
    ap.add_argument("--pt-dir")
    ap.add_argument("--pt-split", default="sample", choices=["sample", "train", "valid", "test"])
    ap.add_argument("--nextqa-dir")
    ap.add_argument("--nextqa-video-dir")
    ap.add_argument("--nextqa-splits", nargs="+", default=["val", "test"])
    ap.add_argument("--model", choices=["qwen25", "qwen3"], default="qwen3")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--max-items", type=int, default=60)
    ap.add_argument("--layer-window", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    init_log(str(out_dir / f"run_mech_knockout_{args.model}.txt"))

    all_results = load_jsonl(args.results_jsonl)
    flipped = [r for r in all_results
               if r.get("condition", "").startswith("real") or r.get("exp") == "probe_regression"
               if r.get("argmax_pre") == r.get("correct_index") and r.get("prob_flip") is True]
    flipped = flipped[:args.max_items]
    log(f"Loaded {len(all_results)} records, {len(flipped)} eligible flipped-from-correct items")
    if not flipped:
        log("No eligible flipped items — run the behavioral stage first (needs real"
            "N conditions or probe_regression records with prob_flip=True).")
        return

    items_lookup = load_source_items(args)
    log(f"Loaded {len(items_lookup)} source items for context rebuilding.")

    common.patch_decord_backend()
    model, processor = common.load_model(common.MODEL_IDS[args.model], eager_attn=True)
    engine = Engine(model, processor)
    log("Model loaded with eager attention.")

    n_layers = len(mech_utils.get_decoder_layers(model))
    windows = [list(range(i, min(i + args.layer_window, n_layers)))
               for i in range(0, n_layers, args.stride)]
    windows = [w for w in windows if w]
    log(f"n_layers={n_layers}, {len(windows)} windows of size {args.layer_window}, stride {args.stride}")

    out_file = str(out_dir / f"knockout_{args.model}.jsonl")
    done = {(r["video_id"], r["qid"], tuple(r["window"])) for r in load_jsonl(out_file)}

    n_processed, n_span_fail = 0, 0
    for rec in flipped:
        key = (rec["source"], rec["video_id"], rec["qid"])
        item = items_lookup.get(key)
        if item is None or not Path(item["video_path"]).exists():
            continue
        visual = {"type": "video", "video": item["video_path"], "nframes": rec["n_frames"]}
        ctxs = mech_utils.build_contexts(item, visual, rec)

        try:
            inputs = engine.prep_inputs(ctxs["post"], extra_text=FORCED_PREFIX)
            ids = inputs["input_ids"][0].tolist()
            press_ids_full = engine.processor.tokenizer.encode(rec["pressure_used"], add_special_tokens=False)
            span = mech_utils.find_subseq(ids, press_ids_full) or mech_utils.find_subseq(ids, press_ids_full[1:-1])
        except Exception as e:
            log(f"  {key}: context build failed: {e}")
            continue
        if span is None:
            n_span_fail += 1
            continue
        key_slice = slice(span[0], span[1])
        n_opts = len(item["options"])
        canonical_order = rec["canonical_order"]

        for window in windows:
            wkey = (rec["video_id"], rec["qid"], tuple(window))
            if wkey in done:
                continue
            try:
                with mech_utils.AttentionKnockout(model, window, key_slice):
                    pos = mech_utils.forced_choice_argmax(engine, ctxs["post"], FORCED_PREFIX, n_opts)
                knockout_idx = mech_utils.position_to_original(pos, canonical_order)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log(f"  OOM {key} window={window}")
                continue
            except RuntimeError as e:
                log(f"  knockout failed (needs a mech_utils.py fix, not a workaround): {e}")
                return
            row = {
                "source": rec["source"], "video_id": rec["video_id"], "qid": rec["qid"],
                "window": window, "correct_index": rec["correct_index"],
                "argmax_pre": rec["argmax_pre"], "argmax_post": rec["argmax_post"],
                "argmax_knockout": knockout_idx,
                "restored": knockout_idx == rec["argmax_pre"] and knockout_idx != rec["argmax_post"],
            }
            append_jsonl(out_file, row)
        n_processed += 1
        if n_processed % 5 == 0:
            log(f"  processed {n_processed}/{len(flipped)} items")

    if n_span_fail:
        log(f"WARNING: pressure-token span not found for {n_span_fail} items — tokenizer/encoding "
            f"mismatch, inspect the span-finding logic above before trusting the results.")

    rows = load_jsonl(out_file)
    from collections import defaultdict
    by_window = defaultdict(list)
    for r in rows:
        by_window[tuple(r["window"])].append(r["restored"])
    log("\n── Restoration rate by layer window (higher = this band carries the flip) ──")
    for w in sorted(by_window, key=lambda w: w[0]):
        vals = by_window[w]
        rate = sum(vals) / len(vals) if vals else 0.0
        log(f"  layers {w[0]:>3}-{w[-1]:<3}: restored {sum(vals)}/{len(vals)} ({100*rate:.1f}%)")
    log(f"\nRaw rows: {out_file}")
    log("Layer 5a knockout done.")


if __name__ == "__main__":
    main()
