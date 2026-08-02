#!/usr/bin/env bash
# Staged V5 pipeline. Datasets download on first run (checked, not re-fetched
# if already present) — see datasets/download_perception_test.py (reliable,
# plain HTTPS) and datasets/download_nextqa.py (best-effort, Google-Drive-
# dependent, documented as the weak link).
#
# Usage:
#   ./run_all.sh OUT_DIR [model] [pt_split]
#
# pt_split defaults to "sample" (8 videos, ~215MB) for a first end-to-end
# smoke test. For the real run pass "train" (26.5GB), "valid" (70.2GB), or
# "test" (41.8GB) — see README.md's size table before choosing.
set -uo pipefail   # NOT -e: a failed nextqa video download should degrade,
                    # not abort the whole pipeline (Perception Test alone
                    # still supports every layer except grounding-alignment).

OUT_DIR="${1:?usage: run_all.sh OUT_DIR [model] [pt_split]}"
MODEL="${2:-qwen3}"
PT_SPLIT="${3:-sample}"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUT_DIR"
PT_DIR="$OUT_DIR/perception_test"
NEXTQA_VIDEO_DIR="$OUT_DIR/nextqa_videos"
NEXTQA_ANN_DIR="$HERE/datasets/nextqa_annotations"

echo "=== Stage 0a/8: Perception Test download (split=$PT_SPLIT) ==="
python3 "$HERE/datasets/download_perception_test.py" --split "$PT_SPLIT" --out-dir "$PT_DIR"
PT_OK=$?

echo "=== Stage 0b/8: NExT-QA/GQA video download (best-effort) ==="
python3 "$HERE/datasets/download_nextqa.py" --out-dir "$NEXTQA_VIDEO_DIR"
NEXTQA_OK=$?
if [ $NEXTQA_OK -ne 0 ]; then
  echo "NExT-QA video download did not complete — continuing with Perception Test only "
  echo "for the layers that support it. Layer 2 (grounding-alignment) needs NExT-GQA "
  echo "specifically and will be skipped. See datasets/download_nextqa.py's manual-"
  echo "fallback instructions if you want to try again by hand."
fi

if [ $PT_OK -ne 0 ]; then
  echo "FATAL: Perception Test download failed — this is the reliable, plain-HTTPS one. "
  echo "Check network access to storage.googleapis.com before anything else."
  exit 1
fi

DATASET_ARG="perception_test"
if [ $NEXTQA_OK -eq 0 ]; then DATASET_ARG="both"; fi

echo "=== Stage 1/8: Layer 2 grounding-alignment check ==="
if [ $NEXTQA_OK -eq 0 ]; then
  python3 "$HERE/run_grounding_check.py" --nextqa-dir "$NEXTQA_ANN_DIR" \
      --video-dir "$NEXTQA_VIDEO_DIR/videos" --model "$MODEL" --out-dir "$OUT_DIR"
else
  echo "Skipped (needs NExT-GQA video download, which did not complete)."
fi

echo "=== Stage 2/8: Layer 1+3 behavioral probe + regression collection ==="
python3 "$HERE/run_probe_regression.py" --dataset "$DATASET_ARG" \
    --pt-dir "$PT_DIR" --pt-split "$PT_SPLIT" \
    --nextqa-dir "$NEXTQA_ANN_DIR" --nextqa-video-dir "$NEXTQA_VIDEO_DIR/videos" \
    --model "$MODEL" --out-dir "$OUT_DIR"

echo "=== Stage 3/8: Layer 4 token-count dilution ==="
python3 "$HERE/run_dilution.py" --dataset perception_test \
    --pt-dir "$PT_DIR" --pt-split "$PT_SPLIT" --model "$MODEL" --out-dir "$OUT_DIR"

echo "=== Stage 4/8: Layer 5a mechanistic attention knockout ==="
python3 "$HERE/run_mech_knockout.py" \
    --results-jsonl "$OUT_DIR/results_dilution_${MODEL}.jsonl" \
    --pt-dir "$PT_DIR" --pt-split "$PT_SPLIT" \
    --nextqa-dir "$NEXTQA_ANN_DIR" --nextqa-video-dir "$NEXTQA_VIDEO_DIR/videos" \
    --model "$MODEL" --out-dir "$OUT_DIR"

echo "=== Stage 5/8: Layer 5b mechanistic activation patching ==="
python3 "$HERE/run_mech_patching.py" \
    --results-jsonl "$OUT_DIR/results_dilution_${MODEL}.jsonl" \
    --pt-dir "$PT_DIR" --pt-split "$PT_SPLIT" \
    --nextqa-dir "$NEXTQA_ANN_DIR" --nextqa-video-dir "$NEXTQA_VIDEO_DIR/videos" \
    --model "$MODEL" --out-dir "$OUT_DIR"

echo "=== Stage 6/8: Analysis ==="
python3 "$HERE/analyze.py" --out-dir "$OUT_DIR" --model "$MODEL" \
    | tee "$OUT_DIR/analysis_summary_${MODEL}.txt"

echo "=== Done ==="
echo "Send back: $OUT_DIR/results_*.jsonl, grounding_summary_*.jsonl, knockout_*.jsonl,"
echo "patching_*.jsonl, attn_*.npz, every run_*.txt log, analysis_summary_${MODEL}.txt."
echo "(Not the downloaded video/annotation files themselves — those regenerate from the"
echo "download scripts and are large; only the pipeline's own output needs to travel back.)"
