#!/usr/bin/env python3
"""
Layer 1+3 — behavioral probe + continuous-covariate collection.

Runs the calibrated probe (common.probe — open-ended first, then the
dataset's own real options, permutation-averaged confidence) across items
from Perception Test and/or NExT-QA, on the real video. No constructed
conditions here (no near/far distractor sets, no occurrence-vs-order split) —
every item just gets probed once, with its dataset-provided covariates
(reasoning_tag, area_tag, content_tags/has_distractor_tag, clip_action_fraction
for Perception Test; type_code for NExT-QA) carried into the output record.
The regression itself (flip/confidence-shift ~ margin + these covariates)
happens in analyze.py, against the pooled JSONL this script produces.

Two datasets, two option counts (3 vs 5) -> different calibration cost:
Perception Test's 3 options are cheap to enumerate exhaustively (6
permutations, automatic in common.calibrated_confidence); NExT-QA's 5 options
use --k-permutations random draws (default 5) since exhaustive (120) is too
expensive at scale.
"""
import argparse, sys
from pathlib import Path

import common
from common import Engine, log, init_log, make_run_probe

sys.path.insert(0, str(Path(__file__).parent / "datasets"))
import perception_test_items
import nextqa_items


def _pt_items(args, budget):
    ann = Path(args.pt_dir) / (f"{args.pt_split}.json" if args.pt_split == "sample"
                                else f"all_{args.pt_split}.json")
    video_dir = Path(args.pt_dir) / "videos"
    n = 0
    for it in perception_test_items.build_items(ann, video_dir):
        if n >= budget:
            return
        if Path(it["video_path"]).exists():
            # k=6 is unused by calibrated_confidence's own auto-enumeration for n=3 options
            # (it enumerates all 3!=6 exhaustively regardless); kept explicit for the logged field.
            yield it, 6
            n += 1


def _nextqa_items(args, budget):
    n = 0
    for split in args.nextqa_splits:
        for it in nextqa_items.build_items(args.nextqa_dir, args.nextqa_video_dir, split=split):
            if n >= budget:
                return
            if Path(it["video_path"]).exists():
                yield it, args.k_permutations
                n += 1


def iter_items(args):
    """Interleaves rather than exhausting one dataset before the other, so a
    small --n-items still samples both when --dataset both."""
    if args.dataset == "perception_test":
        yield from _pt_items(args, args.n_items)
        return
    if args.dataset == "nextqa":
        yield from _nextqa_items(args, args.n_items)
        return
    half = args.n_items // 2
    yield from _pt_items(args, half)
    yield from _nextqa_items(args, args.n_items - half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["perception_test", "nextqa", "both"], default="both")
    ap.add_argument("--pt-dir", help="Perception Test root (from download_perception_test.py --out-dir)")
    ap.add_argument("--pt-split", default="sample", choices=["sample", "train", "valid", "test"])
    ap.add_argument("--nextqa-dir", help="datasets/nextqa_annotations")
    ap.add_argument("--nextqa-video-dir", help="NExT-QA videos root (from download_nextqa.py)")
    ap.add_argument("--nextqa-splits", nargs="+", default=["val", "test"],
                     help="'train' adds volume but zero grounding coverage; val/test have grounding "
                          "on ~65-68%% of items (irrelevant to this script, relevant to analyze.py)")
    ap.add_argument("--n-items", type=int, default=1000)
    ap.add_argument("--k-permutations", type=int, default=5, help="for NExT-QA's 5-option items")
    ap.add_argument("--model", choices=["qwen25", "qwen3"], default="qwen3")
    ap.add_argument("--n-frames", type=int, default=32)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    if args.dataset in ("perception_test", "both") and not args.pt_dir:
        ap.error("--pt-dir required when --dataset includes perception_test")
    if args.dataset in ("nextqa", "both") and not (args.nextqa_dir and args.nextqa_video_dir):
        ap.error("--nextqa-dir and --nextqa-video-dir required when --dataset includes nextqa")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    init_log(str(out_dir / f"run_probe_regression_{args.model}.txt"))

    common.patch_decord_backend()
    model, processor = common.load_model(common.MODEL_IDS[args.model])
    engine = Engine(model, processor)
    log("Model loaded.")

    output_file = str(out_dir / f"results_probe_{args.model}.jsonl")
    run_probe = make_run_probe(output_file, args.model)

    from tqdm import tqdm
    n_done = 0
    for it, k in tqdm(iter_items(args), desc="probe + regression collection", total=args.n_items):
        if n_done >= args.n_items:
            break
        visual = {"type": "video", "video": it["video_path"], "nframes": args.n_frames}
        extra = {
            "n_frames": args.n_frames,
            "reasoning_tag": it["reasoning_tag"], "area_tag": it["area_tag"],
            "content_tags": it["content_tags"], "has_distractor_tag": it["has_distractor_tag"],
            "clip_action_count": it["clip_action_count"], "clip_action_fraction": it["clip_action_fraction"],
            "type_code": it.get("type_code"), "clip_duration_s": it["clip_duration_s"],
            "n_options": len(it["options"]),
        }
        rec = run_probe(engine, "probe_regression", "main", it, visual, extra,
                         k_permutations=k)
        if rec is not None:
            n_done += 1
    log(f"Done: {n_done} new records -> {output_file}")


if __name__ == "__main__":
    main()
