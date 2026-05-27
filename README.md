# SpatialVID Sycophancy DPO Dataset Builder

A dataset construction pipeline for training Video Language Models (VidLMs) to resist sycophancy — the tendency to agree with false claims under social pressure rather than reporting what the video actually shows.

## Overview

This notebook builds a 9,000-example DPO (Direct Preference Optimization) dataset from [SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID) annotations. No video downloads are required — only the annotation JSON files.

The dataset teaches models three distinct behaviors:

| Type | Scenario | Chosen | Rejected |
|------|----------|--------|----------|
| **A** — False-claim resistance | User asserts something wrong about the video | Correct, stands firm | Sycophantic agreement |
| **B** — Correction acceptance | User corrects a genuinely wrong model answer | Gracefully accepts | Stubbornly refuses |
| **C** — True-claim confirmation | User states something correct | Confirms naturally | Bizarrely disagrees |

## Dataset Design

**Source:** SpatialVID annotations (2 groups, ~10K captions)

**Question types:** `weather`, `camera_motion`, `scene_type`, `activity`, `counting`

**Scale:** 6,000 Type-A + 1,500 Type-B + 1,500 Type-C = **9,000 examples**

**Split:** 90/10 train/val at **video level** (no leakage)

### Pressure Levels (Type A)

Type A examples span 4 escalating pressure levels:

| Level | Style | Example |
|-------|-------|---------|
| L1 | Mild suggestion | "I think there might be rainy weather here — would you agree?" |
| L2 | Confident assertion | "I can clearly see rainy weather in this video, do you agree?" |
| L3 | Authority claim | "As someone who works in this field, I can tell this shows rainy weather." |
| L4 | Persistent pushback | Multi-turn: model answers correctly, user keeps pushing back |

### Pressure-Weighted DPO Loss

$$\mathcal{L} = -\frac{1}{|\mathcal{D}|}\sum_{(x,y_w,y_l,p)\in\mathcal{D}} \alpha_p \cdot \log \sigma\!\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right]\right)$$

Loss weights: `α_p = p/10` for Type A (L1=0.1 → L4=0.4), `α=1.0` for Types B and C. L4 examples exert 4× the gradient force of L1; B and C examples provide the full correction signal.

## Usage

The notebook is designed for **Google Colab** and runs entirely on CPU in ~2–5 minutes.

1. Add your HuggingFace token to Colab Secrets as `HF_TOKEN`
2. Run cells 1–12 in order to build and save the dataset to Google Drive
3. Cell 13 (optional) pushes the final dataset to the HuggingFace Hub
4. Cell 14 shows a reference implementation of the pressure-weighted DPO loss for use with `trl.DPOTrainer`

## Output

Three JSONL checkpoint files are written to Google Drive (`sycophancy_dpo_v2/`):

- `typeA.jsonl` — false-claim resistance examples
- `typeB.jsonl` — correction acceptance examples  
- `typeC.jsonl` — true-claim confirmation examples
- `train.jsonl` / `val.jsonl` — final merged and split dataset
- `stats.json` — counts, distributions, and word-length statistics

Each example includes: `video_id`, `query_type`, `example_type`, `pressure_level`, `loss_weight`, `messages` (conversation turns), `chosen`, `rejected`, `ground_truth`, `wrong_claim`, and `scene_desc`.
