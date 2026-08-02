# Video-Sycophancy V5 — Rebuilt Pipeline

Full rebuild of the video-sycophancy investigation on established, peer-
reviewed datasets, after the earlier Charades-based design (`handoff_v4/` in
the main repo, now superseded — kept for history, not for reuse) turned out
to have several real design gaps found by hand: hand-constructed distractor
sets that could differ in baseline difficulty as well as plausibility,
occurrence-vs-order comparisons that were never actually the same task,
option-letter position bias uncorrected in the confidence readout, and a
"does the model look at the right part of the video" question that had no
real ground truth to check against (Charades has no per-question temporal
grounding).

## What changed and why

1. **Datasets**: [Perception Test](https://arxiv.org/abs/2305.13786) (NeurIPS
   2023, Google DeepMind) as primary — verified directly this session
   (downloaded and parsed the real annotation files): single source with
   real timestamped action segments AND pre-authored multiple-choice QA
   together, author-assigned reasoning-type tags (descriptive/explanatory/
   predictive/counterfactual) and even an author-flagged "deliberate
   plausible distractor" tag — removing the need to hand-construct any of
   that ourselves. [NExT-QA](https://arxiv.org/abs/2105.08276) (CVPR 2021) +
   [NExT-GQA](https://arxiv.org/abs/2309.01327) (CVPR 2024, Highlight) as
   secondary, for higher-volume causal/temporal items and — critically —
   NExT-GQA is the *only* one of the datasets considered that has real
   per-question temporal grounding (human-annotated, exactly which seconds
   of video justify each answer). Both bundled options/CSVs were re-checked
   against the live repos; the schemas actually differ between the
   `doc-doc/NExT-QA` release and the `NExT-GQA` release's bundled copies
   (index-based vs. text-based answers) — this package uses NExT-GQA's
   versions throughout, consistently.
2. **No hand-constructed distractors anywhere.** Every question and option
   comes verbatim from the dataset authors. The plausibility axis is now the
   dataset's own `has_distractor_tag` (Perception Test) instead of a
   hand-picked near/far scene cluster.
3. **Belief formation decoupled from option framing.** The model answers
   open-ended first (no options shown), then the real options are introduced
   only at the confidence-readout step — so nothing about how those options
   are framed can retroactively change what the model actually believed.
4. **Calibrated (permutation-averaged) confidence readout**, not a single
   fixed order — option-letter position bias is a documented, large effect
   (13-75pt accuracy swings from reordering alone, arXiv:2308.11483); a
   single canonical order only fixes cross-condition comparability, not the
   model's own baseline letter preference. Full probability distributions
   are logged at every stage (not just the argmax), so KL-divergence and
   entropy-shift are available as continuous outcomes alongside binary flip.
5. **Grounding-alignment as its own layer**, anchored to NExT-GQA's real
   per-question timestamps — not an artificial frame-splice construction —
   answering "does attention track the real evidence" directly, ahead of
   further behavioral condition-building.
6. **Categorical condition comparisons (near/far, occurrence-vs-order)
   retired** in favor of regression on continuous, dataset-provided
   covariates (reasoning tag, evidence-density-adjacent covariates, the
   model's own measured confidence) — nothing here requires arguing that two
   constructed groups are "comparable," because nothing is constructed.

## Layers

| Layer | Script | Question |
|---|---|---|
| 0 | `datasets/download_*.py` | Get the data (auto-run by `run_all.sh`) |
| 1+3 | `run_probe_regression.py` | Calibrated probe across real items; logs covariates for regression |
| 2 | `run_grounding_check.py` | Does attention track the real (NExT-GQA) evidence window? |
| 4 | `run_dilution.py` | Token count vs. content (static-repeat vs. real video, matched frame count) |
| 5 | `run_mech_knockout.py` / `run_mech_patching.py` | Causal: does knocking out/patching specific layers restore the pre-pressure answer? |
| — | `analyze.py` | Stats for all of the above |

## Setup

```bash
pip install -r requirements.txt
```
CUDA GPU, >=40GB VRAM. `decord` needs ffmpeg on the system.

## Running

```bash
./run_all.sh /path/to/out_dir qwen3 sample
```
`sample` is Perception Test's small smoke-test split (8 videos, ~215MB) —
**always run this first** before pointing at `train`/`valid`/`test` (26.5GB/
70.2GB/41.8GB of video). Datasets download automatically, checked-not-
re-fetched on repeat runs. Every stage checkpoints to JSONL and resumes.

**Perception Test download is reliable** (plain HTTPS, verified this
session). **NExT-QA/GQA video download is best-effort** — it's one large
Google Drive zip shared by both projects, and Drive automation is fragile at
that scale (quota limits, the "can't scan for viruses" interstitial).
`datasets/download_nextqa.py` tries `gdown` and prints manual-download
instructions if that doesn't complete; `run_all.sh` degrades gracefully if it
fails — everything except Layer 2 (which specifically needs NExT-GQA's
grounding) still runs on Perception Test alone.

### Rough time estimates (single A100 40GB, Qwen3-VL-8B) — not yet measured on real GPU hardware, treat as order-of-magnitude

Each full probe is more expensive than the earlier pipeline's by design (4
generation calls + ~16 calibration forward-passes vs. the earlier 3+3) —
that's the cost of permutation-debiased confidence, not overhead to trim.

| Stage | Rough cost |
|---|---|
| Layer 1+3 probe (per item) | ~25-35s |
| Layer 2 grounding-check | ~30-35s/item (probe) + attention capture pass |
| Layer 4 dilution (9 conditions/item) | ~4-5 min/item |
| Layer 5 knockout/patching | single forward passes, cheap per window/layer; dominated by n_layers x n_items |

Start with `sample`/a small `--n-items` everywhere, confirm clean, then scale.

## Known limitations (carry into any writeup)

- **Layer 2 only covers NExT-GQA-grounded items** (~65-68% of NExT-QA val/test,
  0% of train) — Perception Test's MC questions are NOT linked to a specific
  timestamp (verified directly: no shared id between `mc_question` and
  `action_localisation`), so they can't be used for this layer.
- **Layer 5 (knockout/patching) tests the final forced-choice readout over
  already-generated response text, not live generation** — a positive result
  shows the answer-commitment step is causally reversible, not where the
  model decided to capitulate while writing its response. See each script's
  docstring.
- **Calibration cost differs by dataset**: Perception Test's fixed 3-option
  format is cheap to enumerate exhaustively (6 permutations, automatic, zero
  residual position-bias noise); NExT-QA's 5-option format uses
  `--k-permutations` random draws (default 5) — real per-item sampling noise
  remains there, averages out across items, not within one.
- **`reasoning_tag`/`type_code` vocabularies differ by dataset** and are kept
  as separate fields (not pooled as if equivalent) — `source` should stay a
  covariate in any pooled analysis.
- Perception Test's per-clip `clip_action_fraction`/`clip_action_count` are
  per-CLIP, not per-question — a coarse covariate, not a precise one.
