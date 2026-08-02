#!/usr/bin/env python3
"""
Downloads NExT-QA (CVPR 2021) + NExT-GQA (CVPR 2024 Highlight) — secondary
dataset, used for higher-volume causal/temporal items and (on the val/test
subset) real human-annotated temporal grounding windows.

Annotation files (CSVs, NExT-GQA's gsub_*.json grounding, the video-id map)
are small and already bundled in datasets/nextqa_annotations/ — nothing to
download for those.

Videos are the weak link: NExT-QA/GQA share one raw-video archive hosted on
Google Drive (file id 1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH — confirmed by
fetching both projects' READMEs), with no way to fetch individual clips.
Google Drive automation is fragile for large files (quota limits, the
"can't scan this file for viruses" interstitial). This script tries `gdown`
(handles that interstitial in most cases) and falls back to printing manual
instructions rather than silently failing. Treat this as best-effort — unlike
Perception Test's plain-HTTPS download, this is not guaranteed to complete
unattended. If it fails, run on Perception Test alone; NExT-QA/GQA is a
bonus, not a dependency for the rest of the pipeline.
"""
import argparse, sys, zipfile
from pathlib import Path

DRIVE_FILE_ID = "1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
MANUAL_URL = f"https://drive.google.com/file/d/{DRIVE_FILE_ID}/view"


def try_gdown(out_dir):
    try:
        import gdown
    except ImportError:
        print("gdown not installed (pip install gdown) — skipping automated download.")
        return False
    zip_path = Path(out_dir) / "nextqa_videos.zip"
    if zip_path.exists():
        if zipfile.is_zipfile(zip_path):
            print(f"  {zip_path} already present and valid, skipping download.")
            return True
        print(f"  {zip_path} exists but isn't a valid zip (previous attempt likely "
              f"interrupted — this is a multi-GB file, easy to interrupt) — deleting "
              f"and retrying.")
        zip_path.unlink()
    print(f"Attempting gdown from {DRIVE_URL} (large file, this can take a while) ...")
    try:
        gdown.download(DRIVE_URL, str(zip_path), quiet=False, fuzzy=True)
    except Exception as e:
        print(f"  gdown failed: {e}")
        return False
    if not (zip_path.exists() and zip_path.stat().st_size > 0 and zipfile.is_zipfile(zip_path)):
        print(f"  gdown finished but {zip_path} isn't a valid zip — likely an incomplete "
              f"or interrupted transfer, not a bug in this script.")
        if zip_path.exists():
            zip_path.unlink()
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-videos", action="store_true",
                     help="only stage annotations (already bundled) — useful for inspecting "
                          "ground truth / building items without needing the video archive")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = Path(__file__).parent / "nextqa_annotations"
    print(f"Annotations already bundled at {ann_dir} (train/val/test.csv, gsub_val/test.json, "
          f"map_vid_vidorID.json) — no download needed for these.")

    if args.skip_videos:
        print("NEXTQA_ANNOTATIONS_READY: (videos skipped by --skip-videos)")
        return

    video_dir = out_dir / "videos"
    marker = out_dir / ".videos_extracted"
    if marker.exists():
        print(f"NEXTQA_READY: {video_dir} (already extracted)")
        return

    ok = try_gdown(out_dir)
    if not ok:
        print(
            "\nAutomated video download did not complete. This is a known-fragile step "
            "(Google Drive bulk zip, no official mirror) — not a bug in this script. "
            "To proceed manually:\n"
            f"  1. Open {MANUAL_URL} in a browser and download the zip.\n"
            f"  2. Place it at {out_dir}/nextqa_videos.zip\n"
            f"  3. Re-run this script with the same --out-dir — it will pick up from there.\n"
            "If this isn't worth the friction right now, skip NExT-QA/GQA entirely and run "
            "the pipeline on Perception Test alone (--dataset perception_test everywhere).",
            file=sys.stderr)
        sys.exit(1)

    zip_path = out_dir / "nextqa_videos.zip"
    print(f"Extracting {zip_path.name} (this is a large archive, may take a while) ...")
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(video_dir)
    except zipfile.BadZipFile:
        print(f"ERROR: {zip_path} is corrupt — delete it and re-run this script.", file=sys.stderr)
        raise
    marker.touch()

    _verify_resolution(video_dir)
    print(f"NEXTQA_READY: {video_dir}")


def _verify_resolution(video_dir):
    """This archive's internal folder layout was never verified against the
    real multi-GB file in development (impractical to download here) — check
    it NOW, once, cheaply, rather than let a wrong guess surface silently as
    zero NExT-QA items deep into a GPU run. Not a bug if this prints a low
    number; it means resolve_video_path's guesses didn't match and its
    recursive-scan fallback (see nextqa_items.py) is doing the real work —
    still fine, just slower to build the index on first use."""
    sys.path.insert(0, str(Path(__file__).parent))
    import nextqa_items
    ann_dir = Path(__file__).parent / "nextqa_annotations"
    vidor_map = nextqa_items.load_map_vid_vidorid(ann_dir)
    sample_ids = list(vidor_map.keys())[:200]
    found = sum(1 for vid in sample_ids
                if Path(nextqa_items.resolve_video_path(video_dir, vid, vidor_map)).exists())
    pct = 100 * found / len(sample_ids) if sample_ids else 0
    print(f"\nVideo-path resolution check: {found}/{len(sample_ids)} sampled video IDs "
          f"({pct:.0f}%) resolved to an actual file under {video_dir}.")
    if found == 0:
        print("WARNING: 0% resolved. Either the zip's internal layout doesn't match any of "
              "resolve_video_path's strategies (check what's actually under "
              f"{video_dir} by hand — `find {video_dir} -name '*.mp4' | head`), or the "
              "download/extraction didn't actually produce video files. Report this rather "
              "than proceeding — the pipeline will silently show 0 NExT-QA items otherwise, "
              "not a clear error.", file=sys.stderr)
    elif pct < 50:
        print(f"NOTE: resolution is partial ({pct:.0f}%) — some videos may be missing from "
              "the archive itself (this happens; not every dataset paper's release is 100% "
              "complete), not necessarily a path-resolution bug.")


if __name__ == "__main__":
    main()
