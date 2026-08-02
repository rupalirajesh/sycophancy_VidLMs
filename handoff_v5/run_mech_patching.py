#!/usr/bin/env python3
"""
Layer 5b — activation patching (clean/pre-pressure run -> pressured run).

Per decoder layer L: cache the last-token residual-stream activation from the
pre-pressure context, splice it into the pressured context at the same
layer/position, let the rest of the forward pass run unmodified, and check
whether the forced-choice readout flips back to the original answer. Same
readout-vs-generation scope note as run_mech_knockout.py: this tests whether
the final answer-commitment step is causally reversible, not where the model
decided to capitulate while writing the pressured response.
"""
import argparse, json, sys
from pathlib import Path

import torch

import common
import mech_utils
from common import Engine, log, init_log, load_jsonl, append_jsonl, FORCED_PREFIX

# reuse the same source-item loading logic as the knockout script
sys.path.insert(0, str(Path(__file__).parent))
from run_mech_knockout import load_source_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-jsonl", required=True)
    ap.add_argument("--pt-dir")
    ap.add_argument("--pt-split", default="sample", choices=["sample", "train", "valid", "test"])
    ap.add_argument("--nextqa-dir")
    ap.add_argument("--nextqa-video-dir")
    ap.add_argument("--nextqa-splits", nargs="+", default=["val", "test"])
    ap.add_argument("--model", choices=["qwen25", "qwen3"], default="qwen3")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--max-items", type=int, default=60)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    init_log(str(out_dir / f"run_mech_patching_{args.model}.txt"))

    all_results = load_jsonl(args.results_jsonl)
    flipped = [r for r in all_results
               if r.get("condition", "").startswith("real") or r.get("exp") == "probe_regression"
               if r.get("argmax_pre") == r.get("correct_index") and r.get("prob_flip") is True]
    flipped = flipped[:args.max_items]
    log(f"Loaded {len(all_results)} records, {len(flipped)} eligible flipped-from-correct items")
    if not flipped:
        log("No eligible flipped items — run the behavioral stage first.")
        return

    items_lookup = load_source_items(args)
    log(f"Loaded {len(items_lookup)} source items for context rebuilding.")

    common.patch_decord_backend()
    model, processor = common.load_model(common.MODEL_IDS[args.model])
    engine = Engine(model, processor)
    log("Model loaded.")

    n_layers = len(mech_utils.get_decoder_layers(model))
    log(f"n_layers={n_layers}")

    out_file = str(out_dir / f"patching_{args.model}.jsonl")
    done = {(r["video_id"], r["qid"], r["layer"]) for r in load_jsonl(out_file)}

    n_processed = 0
    for rec in flipped:
        key = (rec["source"], rec["video_id"], rec["qid"])
        item = items_lookup.get(key)
        if item is None or not Path(item["video_path"]).exists():
            continue
        visual = {"type": "video", "video": item["video_path"], "nframes": rec["n_frames"]}
        ctxs = mech_utils.build_contexts(item, visual, rec)
        n_opts = len(item["options"])
        canonical_order = rec["canonical_order"]

        try:
            neutral_hs = mech_utils.hidden_states_at_last_pos(engine, ctxs["pre"], FORCED_PREFIX)
        except Exception as e:
            log(f"  {key}: clean-run cache failed: {e}")
            continue
        if neutral_hs.shape[0] != n_layers + 1:
            log(f"  {key}: expected {n_layers + 1} hidden-state layers, got {neutral_hs.shape[0]} "
                f"— skipping (model structure mismatch).")
            continue

        for layer in range(n_layers):
            wkey = (rec["video_id"], rec["qid"], layer)
            if wkey in done:
                continue
            replacement = neutral_hs[layer + 1]
            try:
                with mech_utils.ActivationPatch(model, layer, replacement):
                    pos = mech_utils.forced_choice_argmax(engine, ctxs["post"], FORCED_PREFIX, n_opts)
                patched_idx = mech_utils.position_to_original(pos, canonical_order)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log(f"  OOM {key} layer={layer}")
                continue
            row = {
                "source": rec["source"], "video_id": rec["video_id"], "qid": rec["qid"],
                "layer": layer, "correct_index": rec["correct_index"],
                "argmax_pre": rec["argmax_pre"], "argmax_post": rec["argmax_post"],
                "argmax_patched": patched_idx,
                "restored": patched_idx == rec["argmax_pre"] and patched_idx != rec["argmax_post"],
            }
            append_jsonl(out_file, row)
        n_processed += 1
        if n_processed % 5 == 0:
            log(f"  processed {n_processed}/{len(flipped)} items")

    rows = load_jsonl(out_file)
    from collections import defaultdict
    by_layer = defaultdict(list)
    for r in rows:
        by_layer[r["layer"]].append(r["restored"])
    log("\n── Restoration rate by layer (peak = causal locus of the flip) ──")
    for layer in sorted(by_layer):
        vals = by_layer[layer]
        rate = sum(vals) / len(vals) if vals else 0.0
        log(f"  layer {layer:>3}: restored {sum(vals)}/{len(vals)} ({100*rate:.1f}%)")
    log(f"\nRaw rows: {out_file}")
    log("Layer 5b patching done.")


if __name__ == "__main__":
    main()
