#!/usr/bin/env python3
"""
download_benchmark_videos.py

Downloads the video files needed to run SycoBench-Vid evaluation.

Tier 2 videos (dataset == "tier2_vlm") are Wikimedia Commons public-domain
clips.  Their source URLs are stored directly in benchmark.jsonl — this script
fetches them automatically.

Charades videos (dataset == "charades") require a one-time manual request:
  https://allenai.org/plato/charades/
Fill the form (free, academic use) and you'll receive a download link by email,
usually same day.  Place the .mp4 files in --out-dir when they arrive.

Usage:
    python3 download_benchmark_videos.py
    python3 download_benchmark_videos.py --benchmark benchmark/benchmark.jsonl --out-dir data/clips
    python3 download_benchmark_videos.py --dataset wikimedia   # Tier 2 only
    python3 download_benchmark_videos.py --dataset charades    # print needed IDs only
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

UA = "SycoBench-Vid/1.0 (academic research)"


def load_benchmark(path: Path) -> list[dict]:
    instances: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def download(url: str, dest: Path, retries: int = 4) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                while chunk := r.read(65536):
                    f.write(chunk)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 30 * (2 ** attempt)
                print(f" rate-limited, waiting {wait}s...", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f" HTTP {e.code}")
                return False
        except Exception as e:
            print(f" ERROR: {e}")
            return False
    if dest.exists():
        dest.unlink()
    return False


def download_wikimedia(instances: list[dict], out_dir: Path):
    """Download Tier 2 clips whose URLs are stored in benchmark.jsonl."""
    # Deduplicate by video_id — one download covers all instances for that video
    seen: dict[str, str] = {}  # video_id → url
    for inst in instances:
        if inst.get("dataset") == "tier2_vlm":
            vid = inst["video_id"]
            url = inst.get("video_url", "")
            if url and vid not in seen:
                seen[vid] = url

    if not seen:
        print("No tier2_vlm instances with video_url found in benchmark — nothing to download.")
        return

    already = {p.stem for p in out_dir.iterdir()
               if p.suffix in (".webm", ".mp4", ".ogv", ".mkv")}
    to_fetch = {vid: url for vid, url in seen.items() if vid not in already}

    print(f"Tier 2 (Wikimedia): {len(seen)} unique clips, {len(to_fetch)} not yet downloaded")
    downloaded = 0
    for i, (vid, url) in enumerate(to_fetch.items(), 1):
        suffix = Path(url.split("?")[0]).suffix.lower() or ".webm"
        dest = out_dir / f"{vid}{suffix}"
        print(f"[{i}/{len(to_fetch)}] {vid[:55]} ...", end=" ", flush=True)
        if download(url, dest):
            print(f"OK ({dest.stat().st_size / 1e6:.1f} MB)")
            downloaded += 1
        time.sleep(3)

    print(f"\nTier 2: {downloaded} new clips downloaded.")


def print_charades_instructions(instances: list[dict], out_dir: Path):
    """Print the Charades video IDs needed and how to obtain them."""
    already = {p.stem for p in out_dir.iterdir() if p.suffix in (".mp4", ".avi", ".mov")}
    ids = sorted({inst["video_id"] for inst in instances
                  if inst.get("dataset") == "charades"
                  and inst["video_id"] not in already})

    if not ids:
        print("All Charades videos already present in output directory.")
        return

    print(f"\nCharades: {len(ids)} video clips needed — manual download required.")
    print("  1. Go to https://allenai.org/plato/charades/")
    print("  2. Fill the dataset request form (free, academic use, same-day response)")
    print("  3. Download Charades_v1.zip and extract .mp4 files")
    print(f"  4. Place the files in {out_dir}/")
    print(f"\n  Required Charades IDs ({len(ids)} total):")
    for vid in ids:
        status = "✓" if vid in already else "✗"
        print(f"    {status} {vid}.mp4")


def main():
    p = argparse.ArgumentParser(
        description="Download video clips for SycoBench-Vid evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmark", default="benchmark/benchmark.jsonl",
                   help="Path to benchmark.jsonl")
    p.add_argument("--out-dir",   default="data/clips",
                   help="Where to save downloaded clips")
    p.add_argument("--dataset",   default="all",
                   choices=["all", "wikimedia", "charades"],
                   help="Which dataset's videos to download. "
                        "wikimedia = Tier 2 clips, auto-downloaded from stored URLs. "
                        "charades = prints IDs and instructions (manual download required).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = load_benchmark(Path(args.benchmark))
    print(f"Loaded {len(instances)} instances from {args.benchmark}")

    if args.dataset in ("all", "wikimedia"):
        download_wikimedia(instances, out_dir)

    if args.dataset in ("all", "charades"):
        print_charades_instructions(instances, out_dir)


if __name__ == "__main__":
    main()
