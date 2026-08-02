# Runbook: V5 Video-Sycophancy Experiments

You (Claude) are operating this pipeline for a video-LLM sycophancy research
project. Read this whole file before running anything. It tells you what
"done" looks like at each step so you don't have to guess, and what to do
when something degrades (a lot here is designed to degrade gracefully rather
than hard-fail, which is different from v4's package — read Step 1 closely).

## What this is, briefly

Question: why do video-LLMs cave to pressure ("I don't think that's right,
reconsider") more than the facts justify, and what property of video drives
it. This is a full rebuild of an earlier attempt (`handoff_v4/` in the main
repo) after real design problems were found by hand: hand-built distractor
sets with uncontrolled difficulty, cross-question-type comparisons that
weren't actually comparable, uncorrected letter-position bias, and no real
ground truth for "does the model look at the right part of the video." See
`README.md` for the full list of what changed and why. Read a script's module
docstring before running it if anything below is unclear about what it tests.

## Ground rules

- **Verify, don't assume.** After every stage, check the concrete signal
  listed below before moving on. A stage returning exit 0 is not the same as
  it having done anything — check the actual output file's record count.
- **Smoke-test before committing hours of compute.** Every script accepts
  `--n-items`/`--max-items`. Run each stage once at a small count and inspect
  the output JSONL by hand first. Always start Perception Test at
  `--pt-split sample` (8 videos) before `train`/`valid`/`test` (tens of GB).
- **This pipeline is built to degrade gracefully, not hard-fail, on the one
  known-fragile step** (NExT-QA/GQA video download, which depends on a large
  Google Drive zip). If that step fails, that's expected and handled —
  `run_all.sh` continues with Perception Test alone for every layer except
  Layer 2. Don't treat that failure as something to debug or route around by
  yourself; it's already the documented behavior. Do NOT spend time trying
  clever Google Drive workarounds beyond what `datasets/download_nextqa.py`
  already attempts (`gdown`) — if that fails, report it and move on with
  Perception Test.
- **Don't silently work around errors you don't understand elsewhere.** If
  `mech_utils.py` raises `RuntimeError` about decoder-layer paths or the
  attention_mask convention, that means this transformers version doesn't
  match the assumptions the code was written against. Stop, report the exact
  error and `pip show transformers` version, and wait.
- **Report progress at each milestone** (dataset ready, each stage's record
  count, warnings, final analysis summary), not just at the end.

## Step 0 — Environment check

```bash
nvidia-smi                    # confirm GPU, note VRAM (need >=40GB)
python3 --version              # 3.10+
ffmpeg -version                 # decord needs this
pip install -r requirements.txt
```

## Step 1 — Datasets (auto-download, checked-not-refetched)

```bash
python3 datasets/download_perception_test.py --split sample --out-dir ./data/perception_test
python3 datasets/download_nextqa.py --out-dir ./data/nextqa_videos
```
Perception Test: verify stdout ends with `PERCEPTION_TEST_READY: ...`. This
one is plain HTTPS against a public bucket — if it fails, that's a real
network problem, not a fragile-dependency thing; investigate normally.

NExT-QA: verify stdout ends with `NEXTQA_READY: ...` for success. If it
instead prints the manual-fallback block and exits non-zero, **that is
expected, documented behavior**, not a bug to fix — see the Ground Rules
above. Either try the manual download it suggests if you have time, or
proceed without it: everything except `run_grounding_check.py` works fine on
Perception Test alone.

## Step 2 — Smoke test each stage

For each script, run once at a tiny scale before the real run:
```bash
python3 run_probe_regression.py --dataset perception_test --pt-dir ./data/perception_test \
    --pt-split sample --n-items 5 --model qwen3 --out-dir ./smoke_test
```
Confirm: no traceback, the output JSONL has >0 lines, each line parses as
JSON with a non-null `free_response` and `commit_response`, and the log
doesn't show repeated `probe error` lines (that means something's failing on
every item, not per-item quirks) or a `LOAD REPORT`/mismatched-keys warning
at model load (see Known Failure Modes). Do the same for
`run_grounding_check.py` (needs NExT-QA video present), `run_dilution.py`,
then `run_mech_knockout.py`/`run_mech_patching.py` (these need a
`results_dilution_*.jsonl` or `results_probe_*.jsonl` with at least one
flipped-from-correct item to do anything — if the smoke-test batch has zero
flips, that's fine, just means you can't smoke-test the mechanistic stage
until the real run produces some).

Delete `./smoke_test` once satisfied.

## Step 3 — Real run

```bash
./run_all.sh ./out qwen3 sample
```
Change the third argument to `train`/`valid`/`test` for the real Perception
Test split once the sample run looks right — check the size table in
README.md first (`train` videos alone are 26.5GB). Report back after each
stage: record count in that stage's output file, elapsed time, anything
unusual in the log. Don't wait until everything finishes to say something if
an early stage looks wrong.

## Step 4 — Analysis

```bash
python3 analyze.py --out-dir ./out --model qwen3
```
Also runs automatically at the end of `run_all.sh`. Read the actual printed
output — every rate already carries n and a 95% cluster-bootstrap CI, and the
interpretation notes (e.g. "descriptive only at this n", "causal claim but
treat as hypothesis") are there because this project's convention is to never
strip those hedges when relaying results. Pass the numbers through as
printed.

## Known failure modes

- **CUDA OOM.** Reduce `--n-items`/`--max-items`, or for `run_dilution.py`
  reduce `--frame-counts` (drop 64 first). Don't reduce `max_new_tokens` or
  otherwise change what's being measured to "fix" this.
- **`LOAD REPORT` / mismatched-keys warnings on model load, or every probe
  erroring identically.** A past version of this project had a real incident
  where the wrong model class silently loaded mismatched weights for a
  Qwen3-VL checkpoint — the fix (`AutoModelForImageTextToText` in
  `common.py`, never a hardcoded `Qwen2_5_VLForConditionalGeneration`) is
  already in this package. If you see this warning anyway, something changed
  upstream in transformers; stop and report, don't patch around it.
- **NExT-QA video download fails.** Expected, documented, handled — see
  Ground Rules and Step 1. Not a bug.
- **`RuntimeError` from `mech_utils.py`.** Decoder-layer path or
  attention_mask convention mismatch for this transformers version. Report
  the exact error, transformers version, and which script.
- **`run_mech_knockout.py`/`run_mech_patching.py` find 0 eligible items.**
  Means the source results file (`results_dilution_*.jsonl` by default) has
  no items that were both initially-correct and flipped under pressure at a
  `real*` condition. Not an error — either there's genuinely no sycophancy at
  that condition/sample size (itself a real finding, report it) or increase
  `--n-items` on the upstream stage first.

## Step 5 — Package results for handback

```bash
for f in ./out/results_*.jsonl ./out/grounding_summary_*.jsonl ./out/knockout_*.jsonl ./out/patching_*.jsonl; do
  [ -f "$f" ] && echo "$f: $(wc -l < "$f") records"
done
python3 analyze.py --out-dir ./out --model qwen3   # should run with no exceptions
```
Send back the whole `./out` directory's JSONL/txt/npz files (not the
downloaded video/annotation data itself — that regenerates from the download
scripts). Report the final record counts per stage and the full analysis
output — that's the deliverable, not just "it finished."
