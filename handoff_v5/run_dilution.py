#!/usr/bin/env python3
"""
Layer 4 — token-count dilution test (the one clean causal manipulation that
survives from the earlier design unchanged: same content repeated N times vs.
real N-frame video, at matched token count, isolates token count from content
by construction — not a judgment call, so it never had the "are these
comparable" problem the categorical experiments (retired) had).

Conditions per item: "1f" (single keyframe), then per N in --frame-counts,
"staticN" (that keyframe repeated N times) and "realN" (true N-frame video).
Prediction under token-count dilution: flip(staticN) ~= flip(realN), both
rising with N. Prediction under content/grounding mattering instead:
flip(staticN) ~= flip(1f), flip(realN) diverges and is lower.

Dataset-agnostic: takes items from either loader, only needs item["video_path"].
"""
import argparse, sys
from pathlib import Path

import common
from common import Engine, log, init_log, make_run_probe

sys.path.insert(0, str(Path(__file__).parent / "datasets"))
import perception_test_items
import nextqa_items


def iter_items(args):
    if args.dataset == "perception_test":
        ann = Path(args.pt_dir) / (f"{args.pt_split}.json" if args.pt_split == "sample"
                                    else f"all_{args.pt_split}.json")
        video_dir = Path(args.pt_dir) / "videos"
        for it in perception_test_items.build_items(ann, video_dir):
            if Path(it["video_path"]).exists():
                yield it
    else:
        for split in args.nextqa_splits:
            for it in nextqa_items.build_items(args.nextqa_dir, args.nextqa_video_dir, split=split):
                if Path(it["video_path"]).exists():
                    yield it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["perception_test", "nextqa"], required=True)
    ap.add_argument("--pt-dir")
    ap.add_argument("--pt-split", default="sample", choices=["sample", "train", "valid", "test"])
    ap.add_argument("--nextqa-dir")
    ap.add_argument("--nextqa-video-dir")
    ap.add_argument("--nextqa-splits", nargs="+", default=["val", "test"])
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--frame-counts", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--model", choices=["qwen25", "qwen3"], default="qwen3")
    ap.add_argument("--frame-dir", default="/tmp/frames_v5")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    init_log(str(out_dir / f"run_dilution_{args.model}.txt"))

    common.patch_decord_backend()
    model, processor = common.load_model(common.MODEL_IDS[args.model])
    engine = Engine(model, processor)
    log("Model loaded.")

    output_file = str(out_dir / f"results_dilution_{args.model}.jsonl")
    run_probe = make_run_probe(output_file, args.model)

    from tqdm import tqdm
    n_done = 0
    for it in tqdm(iter_items(args), desc="dilution sweep", total=args.n_items):
        if n_done >= args.n_items:
            break
        vp = it["video_path"]
        try:
            kf = common.extract_keyframe(vp, args.frame_dir)
        except Exception as e:
            log(f"  keyframe failed {it['video_id']}: {e}")
            continue

        # k=6 is a no-op for perception_test's 3-option items -- calibrated_confidence
        # auto-enumerates all 3!=6 permutations exhaustively regardless of this value
        # (see common.calibrated_confidence); kept explicit for the logged field.
        k_perm = 6 if it["source"] == "perception_test" else 5
        extra_base = {"reasoning_tag": it["reasoning_tag"], "n_options": len(it["options"])}

        run_probe(engine, "dilution", "1f", it,
                  {"type": "video", "video": [kf]}, {**extra_base, "n_frames": 1},
                  k_permutations=k_perm)
        for n in args.frame_counts:
            run_probe(engine, "dilution", f"static{n}", it,
                      {"type": "video", "video": [kf] * n}, {**extra_base, "n_frames": n},
                      k_permutations=k_perm)
            run_probe(engine, "dilution", f"real{n}", it,
                      {"type": "video", "video": vp, "nframes": n}, {**extra_base, "n_frames": n},
                      k_permutations=k_perm)
        n_done += 1
    log(f"Done: {n_done} items -> {output_file}")


if __name__ == "__main__":
    main()
