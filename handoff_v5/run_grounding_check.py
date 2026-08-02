#!/usr/bin/env python3
"""
Layer 2 — grounding-alignment check.

Answers "does the model's attention actually track the real evidence window,"
using REAL human-annotated grounding (NExT-GQA), not a constructed proxy. This
was explicitly requested ahead of more behavioral condition-building: if
attention doesn't track real evidence at all, frame-level manipulation isn't
the right lever and Layers 3/4 need rethinking before spending more compute.

Why NExT-GQA and not Perception Test: verified directly against both real
annotation files this session — Perception Test's mc_question items are NOT
linked to a specific action_localisation time span (no shared id), so there is
no per-question ground truth to check attention against. NExT-GQA's gsub_*.json
gives exactly that: per-question [start_s, end_s] windows, human-annotated,
for ~68%/65% of the NExT-QA val/test items respectively. Only items with
`grounding_windows` populated are used here — see datasets/nextqa_items.py.

Method: run the probe up through the pre-pressure commit turn and the
post-pressure turn (reusing the generated response text, no extra generation),
then a separate eager-attention forward pass over each context, reading the
final-position attention distribution over visual tokens, split into
per-frame-group shares (frames sampled evenly across the clip's full
duration, so frame i's timestamp is well-defined). Compare the attention mass
landing inside the grounded window(s) to a chance baseline (the window's
share of total clip duration) — a ratio > 1 means attention concentrates on
the real evidence more than chance; a ratio ~= 1 means it doesn't track the
annotated evidence at all.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

import common
from common import Engine, log, init_log, make_run_probe, load_jsonl, FORCED_PREFIX

BASE_FRAMES = 32

import sys
sys.path.insert(0, str(Path(__file__).parent / "datasets"))
import nextqa_items


def capture_frame_group_attention(engine, messages, n_frames):
    """Per-layer attention share on each of n_frames token-groups at the final
    position. Token->frame-group boundaries are evenly split across the visual
    span (approximation of the true patch-merger boundary — flagged, same
    caveat as the earlier E7 attempt at this)."""
    inputs = engine.prep_inputs(messages, extra_text=FORCED_PREFIX)
    with torch.inference_mode():
        out = engine.model(**inputs, output_attentions=True)
    ids = inputs["input_ids"][0]
    i0 = int((ids == engine.model.config.vision_start_token_id).nonzero()[0]) + 1
    i1 = int((ids == engine.model.config.vision_end_token_id).nonzero()[0])
    span = i1 - i0
    bounds = np.linspace(0, span, n_frames + 1).astype(int)
    profile = []
    for layer_attn in out.attentions:
        a = layer_attn[0, :, -1, :].mean(0).float().cpu().numpy()
        vis = a[i0:i1]
        tot = float(a.sum())
        profile.append([float(vis[bounds[j]:bounds[j + 1]].sum()) / tot for j in range(n_frames)])
    del out
    torch.cuda.empty_cache()
    return np.array(profile, dtype=np.float32)   # (L, n_frames)


def frame_timestamps(n_frames, duration_s):
    """Frame i is sampled at i/(n_frames-1) * duration_s (matches the even
    np.linspace(0, total-1, n_frames) sampling in common.patch_decord_backend)."""
    if n_frames <= 1:
        return [0.0]
    return [i / (n_frames - 1) * duration_s for i in range(n_frames)]


def in_window_mask(timestamps, windows):
    mask = np.zeros(len(timestamps), dtype=bool)
    for t_idx, t in enumerate(timestamps):
        for (s, e) in windows:
            if s <= t <= e:
                mask[t_idx] = True
                break
    return mask


def chance_fraction(windows, duration_s):
    """Union of window durations / clip duration — NOT a sum, to avoid
    double-counting if windows overlap."""
    if duration_s <= 0:
        return None
    merged = sorted(windows)
    union = 0.0
    cur_s, cur_e = None, None
    for s, e in merged:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            union += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        union += cur_e - cur_s
    return min(union / duration_s, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nextqa-dir", required=True, help="dir with {train,val,test}.csv etc (datasets/nextqa_annotations)")
    ap.add_argument("--video-dir", required=True, help="dir with NExT-QA videos extracted")
    ap.add_argument("--split", choices=["val", "test"], default="val",
                     help="only val/test have any NExT-GQA grounding")
    ap.add_argument("--n-items", type=int, default=100)
    ap.add_argument("--model", choices=["qwen25", "qwen3"], default="qwen3")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    init_log(str(out_dir / f"run_grounding_check_{args.model}.txt"))

    items = [it for it in nextqa_items.build_items(args.nextqa_dir, args.video_dir, split=args.split)
             if it["grounding_windows"] and Path(it["video_path"]).exists()]
    log(f"Grounded items with video present: {len(items)} (of all {args.split} items)")
    items = items[:args.n_items]
    if not items:
        log("No grounded items with a downloaded video found — check --video-dir, or that "
            "the NExT-QA video download actually completed (see datasets/download_nextqa.py; "
            "this is the known-fragile Google Drive step).")
        return

    common.patch_decord_backend()
    model, processor = common.load_model(common.MODEL_IDS[args.model])
    engine = Engine(model, processor)
    log("Model loaded (behavioral pass).")

    output_file = str(out_dir / f"results_grounding_{args.model}.jsonl")
    run_probe = make_run_probe(output_file, args.model)

    from tqdm import tqdm
    for it in tqdm(items, desc="grounding-check probe"):
        visual = {"type": "video", "video": it["video_path"], "nframes": BASE_FRAMES}
        run_probe(engine, "grounding_check", "main", it, visual, {"n_frames": BASE_FRAMES})
    log("Behavioral pass done.")

    model, processor = common.load_model(common.MODEL_IDS[args.model], eager_attn=True)
    engine = Engine(model, processor)
    log("Model reloaded with eager attention for capture pass.")

    results = {(r["video_id"], r["qid"]): r for r in load_jsonl(output_file)
               if r["exp"] == "grounding_check"}
    attn_out = {}
    n_ok, n_oom, n_skip = 0, 0, 0
    for it in tqdm(items, desc="attention capture"):
        key = (it["video_id"], it["qid"])
        r = results.get(key)
        if not r:
            n_skip += 1
            continue
        duration_s = it["clip_duration_s"]
        if not duration_s:
            n_skip += 1
            continue
        timestamps = frame_timestamps(BASE_FRAMES, duration_s)
        mask = in_window_mask(timestamps, it["grounding_windows"])
        chance = chance_fraction(it["grounding_windows"], duration_s)
        if mask.sum() == 0 or chance is None:
            n_skip += 1
            continue

        visual = {"type": "video", "video": it["video_path"], "nframes": BASE_FRAMES}
        base = [{"role": "user", "content": [visual,
                 {"type": "text", "text": it["question"] + common.FREE_ASK_SUFFIX}]}]
        commit_prompt_ctx = base + [{"role": "assistant", "content": r["free_response"]}]
        pre_ctx = (commit_prompt_ctx + [{"role": "user", "content": r["commit_prompt"]},
                                        {"role": "assistant", "content": r["commit_response"]}])
        post_ctx = pre_ctx + [{"role": "user", "content": r["pressure_used"]},
                              {"role": "assistant", "content": r["pressured_response"]}]

        try:
            prof_pre = capture_frame_group_attention(engine, pre_ctx, BASE_FRAMES)
            prof_post = capture_frame_group_attention(engine, post_ctx, BASE_FRAMES)
        except torch.cuda.OutOfMemoryError:
            n_oom += 1; torch.cuda.empty_cache()
            continue
        except Exception as e:
            log(f"  capture failed {key}: {e}")
            n_skip += 1
            continue

        mean_pre = prof_pre.mean(axis=0)   # mean over layers -> (n_frames,)
        mean_post = prof_post.mean(axis=0)
        in_share_pre = float(mean_pre[mask].sum())
        in_share_post = float(mean_post[mask].sum())
        attn_out[f"{it['video_id']}_{it['qid']}_pre"] = prof_pre
        attn_out[f"{it['video_id']}_{it['qid']}_post"] = prof_post
        attn_out[f"{it['video_id']}_{it['qid']}_mask"] = mask.astype(np.int32)
        with open(out_dir / f"grounding_summary_{args.model}.jsonl", "a") as f:
            f.write(json.dumps({
                "video_id": it["video_id"], "qid": it["qid"],
                "chance_fraction": chance,
                "attn_in_window_pre": in_share_pre, "attn_in_window_post": in_share_post,
                "ratio_pre": in_share_pre / chance, "ratio_post": in_share_post / chance,
                "prob_flip": r["prob_flip"],
            }) + "\n")
        n_ok += 1

    npz_path = out_dir / f"attn_grounding_{args.model}.npz"
    np.savez_compressed(npz_path, **attn_out)
    log(f"Grounding-alignment capture: {n_ok} ok, {n_oom} OOM, {n_skip} skipped -> {npz_path}")
    log("Layer 2 grounding-check done.")


if __name__ == "__main__":
    main()
