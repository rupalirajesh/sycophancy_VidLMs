# Video DPO Training — Sycophancy Resistance

Trains a video-language model (Qwen2.5-VL) to resist sycophancy: refusing to agree with false claims about a video just because the user asserts them confidently, while still accepting genuine corrections and confirming true claims.

Training uses Direct Preference Optimization (DPO) with actual video frames fed to the model.

---

## Dataset

The dataset is already built and lives in `output/`. **No regeneration needed.**

| File | Size |
|------|------|
| `output/train.jsonl` | 9,209 examples |
| `output/val.jsonl` | 1,052 examples |

**10,261 total examples** across 4 types:

| Type | Scenario | Count | Loss weight (`--weighted`) |
|------|----------|-------|---------------------------|
| **A** — False-claim resistance | User asserts something wrong; model must hold its ground | 4,614 | `pressure_level / 10` → 0.1–0.4 |
| **B** — Correction acceptance | User corrects a genuinely wrong model answer; model must accept | 1,904 | 0.1 |
| **C** — True-claim confirmation | User states something correct; model must confirm | 1,861 | 0.1 |
| **D** — Neutral factual Q&A | Unprompted factual question grounded in video | 1,882 | 0.1 |

Type A is the core sycophancy-targeting signal. Types B/C/D maintain general instruction-following and are de-emphasized in weighted training (0.1) so they don't dilute the Type A gradient.

### Type A pressure levels

| Level | Style | Example user message |
|-------|-------|----------------------|
| L1 | Mild suggestion | "I think I noticed rainy weather here — would you agree?" |
| L2 | Confident assertion | "I can clearly see rainy weather. Wouldn't you agree?" |
| L3 | Authority claim | "As someone in this field, I can tell this is rainy weather." |
| L4 | Persistent pushback | Multi-turn: model already answered correctly, user keeps pushing back |

### Weighted DPO loss

$$\mathcal{L} = -\frac{1}{|\mathcal{D}|}\sum_{(x,y_w,y_l,\alpha)\in\mathcal{D}} \alpha \cdot \log \sigma\!\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right]\right)$$

Reference model = base Qwen2.5-VL with LoRA adapters disabled (no separate model copy needed).

---

## Setup

### 1. Install dependencies

```bash
pip install "transformers>=4.57.0" "peft>=0.18.0" "accelerate>=0.34.0" qwen-vl-utils wandb
# Install PyTorch separately: https://pytorch.org/get-started/locally/
```

### 2. Download videos

The dataset references video IDs from [SpatialVID](https://huggingface.co/datasets/SpatialVID/SpatialVID). Download only the needed clips (~26 GB total, two tarballs extracted selectively):

```bash
python download_videos.py \
    --train output/train.jsonl \
    --val   output/val.jsonl \
    --hf-token hf_xxx        # HuggingFace token (repo is gated)
```

Videos are saved to `data/videos/{video_id}.mp4`.

---

## Training

```bash
python train_dpo.py \
    --video-dir data/videos \
    --train     output/train.jsonl \
    --val       output/val.jsonl \
    --weighted
```

`--weighted` is the intended training mode. It applies the per-type loss weights described above.

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-VL-7B-Instruct` | Model to fine-tune |
| `--weighted` | off | Enable pressure-weighted DPO loss |
| `--epochs` | 2 | Training epochs |
| `--grad-accum` | 16 | Gradient accumulation steps (effective batch size) |
| `--lr` | 5e-5 | Peak learning rate (cosine schedule with 5% warmup) |
| `--beta` | 0.1 | DPO temperature β |
| `--output-dir` | `checkpoints/` | Where to save LoRA checkpoints |
| `--no-wandb` | off | Disable Weights & Biases logging |
| `--debug` | off | 5-step smoke test, no save, no W&B |

### Larger model example

```bash
python train_dpo.py \
    --video-dir data/videos \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --weighted \
    --grad-accum 8
```

### Debug / smoke test

```bash
python train_dpo.py --video-dir data/videos --weighted --debug
```

---

## Outputs

```
checkpoints/
└── weighted-dpo-qwen25vl-video/
    ├── checkpoint-200/       # LoRA adapter saved every 200 steps
    ├── checkpoint-400/
    ├── checkpoint-latest/    # Always the most recent — used for crash recovery
    ├── optimizer_latest.pt   # Optimizer state for resume
    ├── scheduler_latest.pt   # LR scheduler state for resume
    ├── resume_state.json     # Step/epoch position for resume
    ├── final/                # Final LoRA adapter + processor
    └── loss_log.json         # Per-step train loss and reward margin
```

Eval runs every 100 optimizer steps on the first 50 val examples.

### Crash recovery

If training crashes, just rerun the exact same command — it automatically picks up from the last 200-step checkpoint. Adapter weights, optimizer, and scheduler are all restored; the dataloader fast-forwards to the correct position.

### Plot training curves

```bash
python plot_loss.py checkpoints/weighted-dpo-qwen25vl-video/loss_log.json
```

### W&B

Training logs to the `sycophancy-dpo` project automatically. Disable with `--no-wandb`.

---

## Memory design

The visual encoder is expensive (~8–10 GB peak) and produces identical output for policy and reference models (LoRA only modifies the language model). It runs **once per training example** under `no_grad`; the resulting embeddings are shared across all four forward passes (ref_chosen, ref_rejected, policy_chosen, policy_rejected). Backward passes are run one at a time to avoid holding two computation graphs simultaneously. This fits comfortably on a single A100 40 GB for 7B; use `device_map="auto"` for larger models across multiple GPUs.

---

## Repo files

| File | Purpose |
|------|---------|
| `train_dpo.py` | DPO training script |
| `build_sycophancy_dpo.py` | Dataset construction pipeline (already run) |
| `download_videos.py` | Downloads only the needed video clips from SpatialVID |
| `plot_loss.py` | Plots `loss_log.json` training curves |
| `output/train.jsonl` | Training split (9,209 examples) |
| `output/val.jsonl` | Validation split (1,052 examples) |
| `output/stats.json` | Dataset statistics |
