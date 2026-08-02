#!/usr/bin/env python3
"""
Downloads Perception Test (NeurIPS 2023, Google DeepMind — arXiv:2305.13786).
Primary dataset for this pipeline: single source with real timestamped action
segments AND pre-authored multiple-choice QA together, Apache-2.0/CC-BY
licensed, plain HTTPS (no Google Drive gymnastics, no auth).

Verified directly against the live bucket (2026-08-02): URL pattern is
https://storage.googleapis.com/dm-perception-test/zip_data/{split}_{videos,annotations}.zip
split="sample" is a small 8-video/215MB set for smoke-testing; "train" videos
alone are 26.5GB, "valid" 70.2GB, "test" 41.8GB — see README.md for the full
size table before choosing a split for the real run.
"""
import argparse, sys
from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "https://storage.googleapis.com/dm-perception-test/zip_data"

SPLIT_SIZES_MB = {
    "sample": {"videos": 215, "annotations": 3},
    "train": {"videos": 26500, "annotations": 31},
    "valid": {"videos": 70200, "annotations": 82},
    "test": {"videos": 41800, "annotations": 1},
}


_last_pct = {}

def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(100, count * block_size * 100 // total_size)
    if pct != _last_pct.get(total_size, -1) and pct % 10 == 0:
        print(f"  {pct}%", flush=True)
        _last_pct[total_size] = pct


def download_split(split, out_dir, what):
    import zipfile   # stdlib zipfile handles the bzip2-compressed entries these
                      # zips use; the macOS/BSD `unzip` CLI does NOT (errors
                      # with "need PK compat. v4.6") — always extract in Python.
    assert what in ("videos", "annotations")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{split}_{what}.zip"
    tmp_path = out_dir / f"{split}_{what}.zip.part"
    marker = out_dir / f".{split}_{what}_extracted"
    if marker.exists():
        print(f"  {split}_{what}: already extracted (marker present), skipping.")
        return

    # A previous run interrupted mid-download would leave a truncated file at
    # zip_path if we trusted existence alone -- verify it's actually a valid
    # zip before skipping the download, not just present.
    if zip_path.exists() and not zipfile.is_zipfile(zip_path):
        print(f"  {zip_path} exists but isn't a valid zip (previous download likely "
              f"interrupted) — deleting and re-downloading.")
        zip_path.unlink()

    if not zip_path.exists():
        url = f"{BASE_URL}/{split}_{what}.zip"
        size_mb = SPLIT_SIZES_MB.get(split, {}).get(what, "?")
        print(f"Downloading {url} (~{size_mb} MB)...")
        if tmp_path.exists():
            tmp_path.unlink()
        urlretrieve(url, tmp_path, _progress)
        print()
        tmp_path.rename(zip_path)   # atomic-ish: only a fully-downloaded file gets the real name
    else:
        print(f"  {zip_path} already present and valid, skipping download.")

    print(f"  Extracting {zip_path.name} ...")
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
    except zipfile.BadZipFile:
        print(f"  ERROR: {zip_path} is corrupt (BadZipFile) even though it looked valid at "
              f"open time — delete {zip_path} and re-run this script to re-download.",
              file=sys.stderr)
        raise
    marker.touch()
    print(f"  Done: {split}_{what} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["sample", "train", "valid", "test"], default="sample",
                     help="'sample' (8 videos, ~215MB) for smoke-testing; 'train'/'valid'/'test' "
                          "for the real run, each tens of GB — see README.md size table first.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--annotations-only", action="store_true",
                     help="skip video download (e.g. to inspect ground truth / build items only)")
    args = ap.parse_args()

    download_split(args.split, args.out_dir, "annotations")
    if not args.annotations_only:
        download_split(args.split, args.out_dir, "videos")

    ann_file = Path(args.out_dir) / (f"{args.split}.json" if args.split == "sample"
                                      else f"all_{args.split}.json")
    if ann_file.exists():
        print(f"PERCEPTION_TEST_READY: {args.out_dir} (annotations: {ann_file.name})")
    else:
        print(f"WARNING: expected annotation file {ann_file} not found after extraction — "
              f"check {args.out_dir} contents manually.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
