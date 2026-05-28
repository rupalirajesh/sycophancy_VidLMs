#!/usr/bin/env python3
"""
plot_loss.py — Plot training loss curves from loss_log.json files.

Usage:
    # Single run
    python plot_loss.py checkpoints/standard-dpo-qwen2vl/loss_log.json

    # Compare both runs side-by-side
    python plot_loss.py \
        checkpoints/standard-dpo-qwen2vl/loss_log.json \
        checkpoints/weighted-dpo-qwen2vl/loss_log.json

    # Save to PNG instead of showing
    python plot_loss.py checkpoints/*/loss_log.json --save loss_curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
LABELS = {
    "standard-dpo-qwen2vl": "Standard DPO",
    "weighted-dpo-qwen2vl": "Weighted DPO",
}


def load_log(path: str) -> dict:
    with open(path) as f:
        entries = json.load(f)
    run_name = Path(path).parent.name
    label = LABELS.get(run_name, run_name)
    steps, train_loss, eval_loss, reward_margin = [], [], [], []
    for e in entries:
        steps.append(e["step"])
        if "train_loss" in e:
            train_loss.append((e["step"], e["train_loss"]))
        if "eval_loss" in e:
            eval_loss.append((e["step"], e["eval_loss"]))
        if "reward_margin" in e:
            reward_margin.append((e["step"], e["reward_margin"]))
    return {
        "label": label,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "reward_margin": reward_margin,
    }


def unzip(pairs):
    if not pairs:
        return [], []
    return zip(*pairs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="Path(s) to loss_log.json")
    p.add_argument("--save", default=None, help="Save plot to this path instead of showing")
    args = p.parse_args()

    runs = [load_log(l) for l in args.logs]
    has_reward = any(r["reward_margin"] for r in runs)
    n_plots = 3 if has_reward else 2

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    fig.suptitle("DPO Training Progress", fontsize=13, fontweight="bold")

    for i, run in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        label = run["label"]

        ax = axes[0]
        xs, ys = unzip(run["train_loss"])
        if xs:
            ax.plot(xs, ys, color=c, label=label, linewidth=1.5)

        ax = axes[1]
        xs, ys = unzip(run["eval_loss"])
        if xs:
            ax.plot(xs, ys, color=c, label=label, linewidth=1.5, linestyle="--")

        if has_reward:
            ax = axes[2]
            xs, ys = unzip(run["reward_margin"])
            if xs:
                ax.plot(xs, ys, color=c, label=label, linewidth=1.5)

    titles = ["Train Loss", "Eval Loss", "Reward Margin (chosen − rejected)"]
    ylabels = ["Loss", "Loss", "Margin"]

    for j, ax in enumerate(axes[:n_plots]):
        ax.set_title(titles[j], fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabels[j])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
