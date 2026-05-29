#!/usr/bin/env python3
"""
train_dpo.py — Video DPO training for Qwen3-VL sycophancy resistance.

Each training example passes actual video frames to the model so it learns
to ground its answers in visual evidence rather than text pressure.

Usage:
    python train_dpo.py \
        --video-dir data/videos \
        --train output/train.jsonl \
        --val   output/val.jsonl

    python train_dpo.py --video-dir data/videos --weighted   # pressure-weighted variant
    python train_dpo.py --video-dir data/videos --debug      # 5-step smoke test

Colab setup:
    !pip install "transformers>=4.57.0" "peft>=0.18.0" "torchao>=0.16.0" accelerate \
                 qwen-vl-utils wandb datasets -q

DPO loss (per sample):
    L = -log sigmoid( beta * ((log π(chosen|x,v) - log π_ref(chosen|x,v))
                             - (log π(rejected|x,v) - log π_ref(rejected|x,v))) )

    where v = video frames. Reference model = base model with LoRA disabled.
    Weighted variant scales each loss by loss_weight (pressure_level / 10).
"""

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
NFRAMES = 8   # frames sampled per video


# ── dataset ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


class VideoPreferenceDataset(Dataset):
    def __init__(self, records: list[dict], video_dir: str, skip_missing: bool = True):
        self.video_dir = Path(video_dir)
        self.records = []
        missing = 0
        for r in records:
            vpath = self.video_dir / f"{r['video_id']}.mp4"
            if not vpath.exists():
                missing += 1
                if skip_missing:
                    continue
            self.records.append(r)
        if missing:
            print(f"  Skipped {missing} records with missing video files.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def inject_video(messages: list[dict], video_path: str) -> list[dict]:
    """Insert the video into the first user message as a multimodal content block."""
    result = []
    video_injected = False
    for m in messages:
        if m["role"] == "user" and not video_injected:
            result.append({
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "nframes": NFRAMES, "fps": 1.0},
                    {"type": "text",  "text": m["content"]},
                ],
            })
            video_injected = True
        else:
            result.append(m)
    return result


# ── forward pass helpers ──────────────────────────────────────────────────────

def process_example(processor, messages_with_video: list[dict], response_text: str, device):
    """
    Tokenize prompt (with video) + response.
    Returns tensors ready for model.forward(), with labels = -100 on prompt tokens.
    """
    from qwen_vl_utils import process_vision_info

    prompt_str = processor.apply_chat_template(
        messages_with_video, tokenize=False, add_generation_prompt=True
    )
    _, video_inputs = process_vision_info(messages_with_video)

    # Tokenize prompt with video → gets input_ids including visual placeholder tokens
    prompt_enc = processor(
        text=[prompt_str],
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
        padding=False,
    )
    prompt_len = prompt_enc["input_ids"].shape[1]

    # Tokenize response only (no special tokens)
    resp_ids = processor.tokenizer(
        response_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]  # (1, resp_len)

    # Append EOS
    eos = torch.tensor([[processor.tokenizer.eos_token_id]])
    resp_ids = torch.cat([resp_ids, eos], dim=1)

    full_ids = torch.cat([prompt_enc["input_ids"], resp_ids], dim=1)
    attn_mask = torch.ones_like(full_ids)
    labels = torch.full_like(full_ids, -100)
    labels[0, prompt_len:] = resp_ids[0]

    batch = {
        "input_ids":      full_ids.to(device),
        "attention_mask": attn_mask.to(device),
        "labels":         labels.to(device),
    }
    for key in ("pixel_values_videos", "video_grid_thw"):
        if key in prompt_enc:
            batch[key] = prompt_enc[key].to(device)

    return batch


@torch.no_grad()
def get_logps_nograd(model, batch: dict) -> torch.Tensor:
    return _get_logps(model, batch)


def get_logps_grad(model, batch: dict) -> torch.Tensor:
    return _get_logps(model, batch)


def _get_logps(model, batch: dict) -> torch.Tensor:
    """Sum of log probs over response tokens (where labels != -100)."""
    vision_kwargs = {}
    for key in ("pixel_values_videos", "video_grid_thw"):
        if key in batch:
            vision_kwargs[key] = batch[key]

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        **vision_kwargs,
    )
    logits = outputs.logits[:, :-1].float()        # (B, seq-1, vocab)
    shifted_labels = batch["labels"][:, 1:]        # (B, seq-1)

    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = torch.gather(
        log_probs, 2, shifted_labels.clamp(min=0).unsqueeze(2)
    ).squeeze(2)

    mask = shifted_labels != -100
    return (token_logps * mask).sum(-1)            # (B,)


# ── DPO loss ──────────────────────────────────────────────────────────────────

def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    weight: float = 1.0,
) -> torch.Tensor:
    pi_logratios  = policy_chosen_logps  - policy_rejected_logps
    ref_logratios = ref_chosen_logps     - ref_rejected_logps
    loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    return loss.mean() * weight


# ── loss logger ───────────────────────────────────────────────────────────────

class LossLog:
    def __init__(self, path: str):
        self.path = path
        self._log: list[dict] = []

    def add(self, step: int, **kwargs):
        self._log.append({"step": step, **{k: round(float(v), 4) for k, v in kwargs.items()}})
        if step % 10 == 0:
            self._flush()

    def _flush(self):
        with open(self.path, "w") as f:
            json.dump(self._log, f, indent=2)

    def close(self):
        self._flush()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train",      default="output/train.jsonl")
    p.add_argument("--val",        default="output/val.jsonl")
    p.add_argument("--video-dir",  default="data/videos", help="Directory with {video_id}.mp4 files")
    p.add_argument("--model",      default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--weighted",   action="store_true", help="Pressure-weighted DPO")
    p.add_argument("--no-wandb",   action="store_true")
    p.add_argument("--debug",      action="store_true", help="5 steps, no save, no W&B")
    p.add_argument("--epochs",     type=int,   default=2)
    p.add_argument("--grad-accum", type=int,   default=16, help="Effective batch size")
    p.add_argument("--lr",         type=float, default=5e-5)
    p.add_argument("--beta",       type=float, default=0.1)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        args.no_wandb = True

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = f"{'weighted' if args.weighted else 'standard'}-dpo-qwen2vl-video"
    if args.debug:
        run_name += "-debug"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  {run_name}\n  Output: {out_dir}\n{'='*60}\n")

    # ── W&B ──────────────────────────────────────────────────────────────
    if not args.no_wandb:
        import wandb
        wandb.init(project="sycophancy-dpo", name=run_name, config=vars(args))
        print(f"W&B: {wandb.run.url}\n")

    # ── model + processor ────────────────────────────────────────────────
    print(f"Loading {args.model}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(args.model)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── data ─────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    train_records = load_jsonl(args.train)
    val_records   = load_jsonl(args.val)

    if args.debug:
        train_records = train_records[:20]
        val_records   = val_records[:10]

    train_ds = VideoPreferenceDataset(train_records, args.video_dir)
    val_ds   = VideoPreferenceDataset(val_records,   args.video_dir)
    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    if len(train_ds) == 0:
        raise RuntimeError(
            f"No training examples found. Make sure videos are in {args.video_dir}/ "
            f"and run download_videos.py first."
        )

    # DataLoader with shuffle; collate_fn is identity (we process one at a time)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=lambda x: x[0])
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    # ── optimizer ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = (len(train_ds) * args.epochs) // args.grad_accum
    warmup_steps = max(1, int(0.05 * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    max_steps = 5 if args.debug else total_steps
    loss_log = LossLog(str(out_dir / "loss_log.json"))

    # ── training loop ─────────────────────────────────────────────────────
    print(f"\nStarting training — {total_steps} optimizer steps "
          f"({len(train_ds)} examples × {args.epochs} epochs ÷ {args.grad_accum} accum)\n")

    global_step = 0
    accum_loss  = 0.0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        for local_step, record in enumerate(train_loader):
            if global_step >= max_steps:
                break

            video_path = str(Path(args.video_dir) / f"{record['video_id']}.mp4")
            messages   = inject_video(record["messages"], video_path)
            weight     = float(record.get("loss_weight", 1.0)) if args.weighted else 1.0

            try:
                chosen_batch   = process_example(processor, messages, record["chosen"],   device)
                rejected_batch = process_example(processor, messages, record["rejected"], device)
            except Exception as e:
                print(f"  [skip] processing error for {record['video_id']}: {e}")
                continue

            # Reference log probs (no gradient, LoRA disabled)
            with torch.no_grad(), model.disable_adapter():
                ref_chosen_logps   = get_logps_nograd(model, chosen_batch)
                ref_rejected_logps = get_logps_nograd(model, rejected_batch)

            # Policy log probs (with gradient)
            policy_chosen_logps   = get_logps_grad(model, chosen_batch)
            policy_rejected_logps = get_logps_grad(model, rejected_batch)

            loss = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps,    ref_rejected_logps,
                beta=args.beta, weight=weight,
            )
            (loss / args.grad_accum).backward()
            accum_loss += loss.item()

            # Optimizer step every grad_accum examples
            if (local_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                accum_loss /= args.grad_accum

                print(f"  step {global_step}/{max_steps}  loss={accum_loss:.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

                if not args.no_wandb:
                    import wandb
                    wandb.log({"train_loss": accum_loss, "lr": scheduler.get_last_lr()[0]}, step=global_step)

                loss_log.add(global_step, train_loss=accum_loss)
                accum_loss = 0.0

                # Eval every 100 steps
                if global_step % 100 == 0 or global_step == max_steps:
                    model.eval()
                    eval_loss = 0.0
                    eval_steps = min(50, len(val_ds))
                    for val_record in list(val_loader)[:eval_steps]:
                        vpath = str(Path(args.video_dir) / f"{val_record['video_id']}.mp4")
                        vmsg  = inject_video(val_record["messages"], vpath)
                        try:
                            cb = process_example(processor, vmsg, val_record["chosen"],   device)
                            rb = process_example(processor, vmsg, val_record["rejected"], device)
                            with torch.no_grad():
                                with model.disable_adapter():
                                    rc = get_logps_nograd(model, cb)
                                    rr = get_logps_nograd(model, rb)
                                pc = get_logps_nograd(model, cb)
                                pr = get_logps_nograd(model, rb)
                            eval_loss += dpo_loss(pc, pr, rc, rr, beta=args.beta).item()
                        except Exception:
                            pass
                    eval_loss /= max(eval_steps, 1)
                    print(f"  [eval]  loss={eval_loss:.4f}")
                    loss_log.add(global_step, eval_loss=eval_loss)
                    if not args.no_wandb:
                        import wandb
                        wandb.log({"eval_loss": eval_loss}, step=global_step)
                    model.train()

                # Checkpoint
                if not args.debug and global_step % 200 == 0:
                    ckpt = out_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(str(ckpt))
                    print(f"  Saved checkpoint to {ckpt}")

        if global_step >= max_steps:
            break

    # ── final save ───────────────────────────────────────────────────────
    loss_log.close()
    if not args.debug:
        final = out_dir / "final"
        model.save_pretrained(str(final))
        processor.save_pretrained(str(final))
        print(f"\nSaved to {final}")

    print(f"Loss log: {out_dir}/loss_log.json")
    if not args.no_wandb:
        import wandb
        print(f"W&B: {wandb.run.url}")
        wandb.finish()


if __name__ == "__main__":
    main()
