#!/usr/bin/env python3
"""
train_dpo.py — DPO / Pressure-Weighted DPO training for sycophancy resistance

Two training runs (run sequentially or in separate Colab sessions):

    Standard DPO:
        python train_dpo.py --train output/train.jsonl --val output/val.jsonl

    Pressure-weighted DPO:
        python train_dpo.py --train output/train.jsonl --val output/val.jsonl --weighted

Each run opens a W&B link for live loss curves. Progress is also saved to
checkpoints/<run_name>/loss_log.json every 10 steps.

Colab setup:
    !pip install transformers==4.45.2 trl==0.11.4 peft==0.12.0 \
                 accelerate==0.34.2 wandb datasets -q

NOTE: This trains on text descriptions (scene_desc embedded in messages).
      To use actual video frames, replace `messages` with Qwen2VL visual inputs.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

# ── LoRA config ──────────────────────────────────────────────────────────────

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ── data helpers ─────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_dataset(records: list[dict], processor, include_weights: bool) -> Dataset:
    """Convert JSONL records to a HuggingFace Dataset for DPOTrainer.

    TRL expects: prompt (str), chosen (str), rejected (str).
    The prompt is the full conversation up to the last user turn,
    formatted with the model's chat template.
    """
    rows = []
    for r in records:
        prompt = processor.tokenizer.apply_chat_template(
            r["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        row = {
            "prompt": prompt,
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }
        if include_weights:
            row["loss_weight"] = float(r.get("loss_weight", 1.0))
        rows.append(row)
    return Dataset.from_list(rows)


# ── text-only DPO base ───────────────────────────────────────────────────────

class TextOnlyDPOTrainer(DPOTrainer):
    """Forces text-only mode even when a VLM processor is passed.

    TRL sets is_vision_model=True when it sees Qwen2VLProcessor (which has an
    image_processor attribute), then unconditionally looks for pixel_values in
    every batch. Our dataset is text-only, so we override the flag after init.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_vision_model = False


# ── weighted DPO ─────────────────────────────────────────────────────────────

@dataclass
class WeightedDPOCollator:
    """Wraps TRL's default DPO collator to pass loss_weight through to the batch."""
    base: Any

    def __call__(self, features: list[dict]) -> dict:
        weights = [float(f.pop("loss_weight", 1.0)) for f in features]
        batch = self.base(features)
        batch["loss_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


class WeightedDPOTrainer(TextOnlyDPOTrainer):
    """DPO trainer that scales each sample's loss by loss_weight.

    Pressure-level mapping: p1→0.1, p2→0.2, p3→0.3, p4→0.4, B/C/D→1.0.
    High-pressure sycophancy events get proportionally more gradient signal.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Wrap the collator so loss_weight reaches the batch dict
        self.data_collator = WeightedDPOCollator(base=self.data_collator)
        self._batch_weights: Optional[torch.Tensor] = None

    def tokenize_row(self, feature, model=None):
        """Preserve loss_weight through TRL's tokenization step."""
        result = super().tokenize_row(feature, model=model)
        result["loss_weight"] = feature.get("loss_weight", 1.0)
        return result

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop weights before parent's forward pass so model.forward() never sees them
        w = inputs.pop("loss_weight", None)
        if w is not None:
            self._batch_weights = (
                w if isinstance(w, torch.Tensor)
                else torch.tensor(w, dtype=torch.float32)
            )
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ):
        # Parent returns per-sample losses (before .mean())
        losses, chosen_rewards, rejected_rewards = super().dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        if self._batch_weights is not None:
            w = self._batch_weights.to(device=losses.device, dtype=losses.dtype)
            losses = losses * w
            self._batch_weights = None
        return losses, chosen_rewards, rejected_rewards


# ── loss logging callback ─────────────────────────────────────────────────────

class LossLogCallback(TrainerCallback):
    """Saves loss to loss_log.json every `save_every` steps.

    Use plot_loss.py to visualize mid-run or after training completes.
    """

    def __init__(self, path: str, save_every: int = 10):
        self.path = path
        self.save_every = save_every
        self._log: list[dict] = []

    def _flush(self):
        with open(self.path, "w") as f:
            json.dump(self._log, f, indent=2)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ):
        if not logs:
            return
        entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 3)}
        if "loss" in logs:
            entry["train_loss"] = round(logs["loss"], 4)
        if "eval_loss" in logs:
            entry["eval_loss"] = round(logs["eval_loss"], 4)
        if "rewards/margins" in logs:
            entry["reward_margin"] = round(logs["rewards/margins"], 4)
        if entry.keys() - {"step", "epoch"}:  # only save if we have real metrics
            self._log.append(entry)
            if state.global_step % self.save_every == 0:
                self._flush()

    def on_train_end(self, args, state, control, **kwargs):
        self._flush()


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DPO training for sycophancy resistance")
    p.add_argument("--train", default="output/train.jsonl", help="Training JSONL")
    p.add_argument("--val", default="output/val.jsonl", help="Validation JSONL")
    p.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--weighted", action="store_true",
                   help="Use pressure-weighted DPO (vs standard DPO)")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Per-device batch size (2 fits A100 40GB with QLoRA)")
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Effective batch = batch_size * grad_accum = 16")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1, help="DPO regularization strength")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--debug", action="store_true",
                   help="Smoke-test: 50 examples, 5 steps, no save, no W&B")
    return p.parse_args()


def main():
    args = parse_args()

    if args.debug:
        args.no_wandb = True

    run_name = f"{'weighted' if args.weighted else 'standard'}-dpo-qwen2vl"
    if args.debug:
        run_name += "-debug"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_log_path = str(out_dir / "loss_log.json")

    print(f"\n{'='*60}")
    print(f"  Run: {run_name}{'  [DEBUG]' if args.debug else ''}")
    print(f"  Output: {out_dir}")
    print(f"  Loss log: {loss_log_path}")
    print(f"{'='*60}\n")

    # ── W&B ──────────────────────────────────────────────────────────────
    if not args.no_wandb:
        import wandb
        wandb.init(project="sycophancy-dpo", name=run_name, config=vars(args))
        print(f"W&B run: {wandb.run.url}\n")

    # ── model ─────────────────────────────────────────────────────────────
    # bf16 + LoRA fits comfortably on A100 40GB (~14 GB base, no quantization needed)
    print(f"Loading {args.model} in bf16 + LoRA mode...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(args.model)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"  # required for DPO
    # TRL 0.11 double-unwraps VLM processors: first extracts .tokenizer from the processor,
    # then calls .tokenizer again on the result. Self-reference prevents the AttributeError.
    processor.tokenizer.tokenizer = processor.tokenizer

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── dataset ───────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    train_records = load_jsonl(args.train)
    val_records = load_jsonl(args.val)

    if args.debug:
        train_records = train_records[:50]
        val_records = val_records[:20]
        print("[DEBUG] Truncated to 50 train / 20 val examples")

    train_ds = build_dataset(train_records, processor, include_weights=args.weighted)
    val_ds = build_dataset(val_records, processor, include_weights=args.weighted)
    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    # ── training config ───────────────────────────────────────────────────
    dpo_cfg = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=1 if args.debug else args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1 if args.debug else args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_length // 2,
        bf16=True,
        logging_steps=1 if args.debug else 10,
        eval_strategy="steps",
        eval_steps=5 if args.debug else 200,
        save_strategy="no" if args.debug else "steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=False,
        remove_unused_columns=False,   # keep loss_weight column in batch
        report_to=[] if args.no_wandb else ["wandb"],
        run_name=run_name,
        dataloader_num_workers=0,
        optim="adamw_torch_fused",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_steps=5 if args.debug else -1,
    )

    TrainerClass = WeightedDPOTrainer if args.weighted else TextOnlyDPOTrainer
    trainer = TrainerClass(
        model=model,
        ref_model=None,          # PEFT mode: base model is the implicit reference
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
        callbacks=[LossLogCallback(loss_log_path, save_every=10)],
    )

    # ── train ─────────────────────────────────────────────────────────────
    print(f"\nStarting training ({TrainerClass.__name__})...\n")
    trainer.train()

    final_path = out_dir / "final"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"\nSaved to {final_path}")
    print(f"Loss log:  {loss_log_path}")
    if not args.no_wandb:
        import wandb
        print(f"W&B:       {wandb.run.url}")
        wandb.finish()


if __name__ == "__main__":
    main()
