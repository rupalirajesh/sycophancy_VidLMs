#!/usr/bin/env python3
"""
download_videos.py — Download only the video files needed for DPO training.

The SpatialVID videos are stored as group_XXXX.tar.gz on HuggingFace (~13 GB each).
This script downloads group_0001 and group_0002 (the two groups our dataset uses),
extracts only the 552 specific video IDs we need, and saves them to data/videos/.

Usage:
    python download_videos.py --train output/train.jsonl --val output/val.jsonl
    python download_videos.py --train output/train.jsonl --val output/val.jsonl --hf-token hf_xxx

Output:
    data/videos/{video_id}.mp4   for each of the 552 needed video IDs

Download size: ~26 GB (two tar archives, extracted videos kept, archives deleted)
"""

import argparse
import json
import os
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "SpatialVID/SpatialVID"
GROUPS_NEEDED = ["group_0001", "group_0002"]


def load_needed_ids(train_path: str, val_path: str) -> dict[str, str]:
    """Returns {video_id: group_name} for all videos in train + val."""
    records = []
    for path in [train_path, val_path]:
        with open(path) as f:
            records += [json.loads(l) for l in f if l.strip()]

    # We stored which group each video is in during annotation download.
    # Reconstruct from local annotation directory.
    ann_root = Path("data/annotations/SpatialVID/annotations")
    id_to_group = {}
    if ann_root.exists():
        for group_dir in ann_root.iterdir():
            if group_dir.is_dir():
                for vid_dir in group_dir.iterdir():
                    id_to_group[vid_dir.name] = group_dir.name

    needed = {}
    missing = []
    for r in records:
        vid = r["video_id"]
        if vid not in needed:
            group = id_to_group.get(vid, "unknown")
            needed[vid] = group
            if group == "unknown":
                missing.append(vid)

    if missing:
        print(f"WARNING: {len(missing)} video IDs not found in local annotations.")
    print(f"Need {len(needed)} unique videos across groups: "
          f"{dict(sorted((g, sum(1 for v in needed.values() if v == g)) for g in set(needed.values())))}")
    return needed


def download_and_extract(group: str, needed_ids: set[str], video_dir: Path, token: str | None):
    tar_path = video_dir / f"{group}.tar.gz"
    remote_path = f"videos/{group}.tar.gz"

    # Download tar if not already present
    if not tar_path.exists():
        print(f"\nDownloading {remote_path} (~13 GB)...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=remote_path,
            local_dir=str(video_dir),
            token=token,
        )
        # hf_hub_download may nest the file; move it flat
        src = Path(downloaded)
        if src != tar_path:
            src.rename(tar_path)
    else:
        print(f"\n{tar_path} already downloaded, skipping.")

    # Peek at tar structure on first extraction
    print(f"Extracting needed videos from {tar_path.name}...")
    extracted = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        # Figure out naming: videos may be at {group}/{id}.mp4 or {id}.mp4
        for m in members[:5]:
            print(f"  sample entry: {m.name}")

        for m in tqdm(members, desc=f"Scanning {group}"):
            if not m.isfile():
                continue
            # Extract video_id from path (works for both flat and nested tars)
            stem = Path(m.name).stem
            if stem in needed_ids:
                dest = video_dir / f"{stem}.mp4"
                if dest.exists():
                    continue
                f = tf.extractfile(m)
                if f:
                    dest.write_bytes(f.read())
                    extracted += 1

    print(f"Extracted {extracted} videos from {group}.")

    # Remove the tar to free space
    tar_path.unlink()
    print(f"Deleted {tar_path.name} to free disk space.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train",    default="output/train.jsonl")
    p.add_argument("--val",      default="output/val.jsonl")
    p.add_argument("--video-dir", default="data/videos")
    p.add_argument("--hf-token", default=None, help="HuggingFace token (needed if repo is gated)")
    args = p.parse_args()

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    needed = load_needed_ids(args.train, args.val)

    for group in GROUPS_NEEDED:
        group_ids = {vid for vid, g in needed.items() if g == group}
        if not group_ids:
            continue
        already = {p.stem for p in video_dir.glob("*.mp4")}
        remaining = group_ids - already
        if not remaining:
            print(f"{group}: all {len(group_ids)} videos already downloaded.")
            continue
        print(f"{group}: need {len(remaining)}/{len(group_ids)} videos.")
        download_and_extract(group, remaining, video_dir, args.hf_token)

    # Final count
    downloaded = list(video_dir.glob("*.mp4"))
    print(f"\nDone. {len(downloaded)}/{len(needed)} videos in {video_dir}/")
    missing = {vid for vid in needed if not (video_dir / f"{vid}.mp4").exists()}
    if missing:
        print(f"Still missing {len(missing)} videos: {list(missing)[:5]}...")


if __name__ == "__main__":
    main()
