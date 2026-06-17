#!/usr/bin/env python3
"""Plots for the taboo-finetune + NLA secret-word recovery baseline.

Reads training metrics from results/taboo_baseline/<word>-baseline/metrics.json
and eval outputs from results/taboo_baseline/nla_eval/<run>/.

    python -m taboo_secret_word.plot_taboo_baseline --words gold ship smile cloud
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from taboo_secret_word.nla_taboo_eval import (
    score_explanations,
)

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "taboo_baseline"
PTYPES = ["user_end", "assistant_header", "response"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_loss_curves(words: list[str], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for word in words:
        hist = load_json(RESULTS_ROOT / f"{word}-baseline" / "metrics.json")["history"]
        steps = [r["step"] for r in hist["train_loss"]]
        losses = [r["loss"] for r in hist["train_loss"]]
        axes[0].plot(steps, losses, label=word, alpha=0.8)
        ev_steps = [r["step"] for r in hist["eval_loss"]]
        ev_losses = [r["loss"] for r in hist["eval_loss"]]
        axes[1].plot(ev_steps, ev_losses, marker="o", label=word)
    axes[0].set_title("Train loss (assistant tokens only)")
    axes[1].set_title("Eval loss (held-out 10%)")
    for ax in axes:
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("cross-entropy")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss.png", dpi=150)
    plt.close(fig)


def plot_recovery_rates(words: list[str], out_dir: Path) -> dict:
    """Bar plots of secret-word hit rates: finetuned vs base control."""
    base_expl = load_json(RESULTS_ROOT / "nla_eval" / "base-control" / "explanations.json")["explanations"]
    rates: dict[str, dict] = {}
    for word in words:
        ft = load_json(RESULTS_ROOT / "nla_eval" / f"{word}-baseline" / "summary.json")
        base_summary = score_explanations([dict(e) for e in base_expl], word)
        rates[word] = {"finetuned": ft, "base": base_summary}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(words))
    width = 0.38
    for ax, metric, title in (
        (axes[0], "hit_rate", "AV explanation hit rate (all decodes)"),
        (axes[1], "position_hit_rate", "Positions with >=1 hit across rollouts"),
    ):
        ft_vals = [rates[w]["finetuned"][metric] for w in words]
        base_vals = [rates[w]["base"][metric] for w in words]
        ax.bar(x - width / 2, ft_vals, width, label="taboo finetuned", color="#d62728")
        ax.bar(x + width / 2, base_vals, width, label="base Qwen2.5 control", color="#7f7f7f")
        ax.set_xticks(x, words)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_ylabel("secret-word mention rate")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        for xi, v in zip(x - width / 2, ft_vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
        for xi, v in zip(x + width / 2, base_vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    fig.suptitle("NLA recovery of the hidden taboo word (layer-20 AV, Qwen2.5-7B)")
    fig.tight_layout()
    fig.savefig(out_dir / "recovery_rates.png", dpi=150)
    plt.close(fig)
    return rates


def plot_recovery_by_ptype(words: list[str], rates: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(words))
    width = 0.8 / len(PTYPES)
    colors = {"user_end": "#1f77b4", "assistant_header": "#ff7f0e", "response": "#2ca02c"}
    for k, ptype in enumerate(PTYPES):
        vals = [rates[w]["finetuned"]["by_position_type"][ptype]["hit_rate"] for w in words]
        ax.bar(x + (k - 1) * width, vals, width, label=ptype, color=colors[ptype])
    ax.set_xticks(x, words)
    ax.set_ylim(0, 1)
    ax.set_title("Hit rate by activation position type (taboo finetuned models)")
    ax.set_ylabel("secret-word mention rate")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "recovery_by_position_type.png", dpi=150)
    plt.close(fig)


def plot_recurrence(words: list[str], out_dir: Path) -> None:
    """Per position: number of rollouts (of 3) whose explanation names the secret word."""
    fig, axes = plt.subplots(1, len(words), figsize=(3.2 * len(words), 3.8), sharey=True)
    for ax, word in zip(axes, words):
        expl = load_json(RESULTS_ROOT / "nla_eval" / f"{word}-baseline" / "explanations.json")["explanations"]
        scored = score_explanations([dict(e) for e in expl], word)
        per_pos: dict[tuple, int] = {}
        for e in expl:
            text = e["explanation"].lower()
            hit = any(v in text for v in (word, word + "s", word + "es", word + "ing", word + "ed"))
            key = (e["prompt_idx"], e["pos"])
            per_pos[key] = per_pos.get(key, 0) + int(hit)
        n_rollouts = max(per_pos.values(), default=0)
        counts = np.bincount(list(per_pos.values()), minlength=4)
        ax.bar(range(len(counts)), counts, color="#9467bd")
        ax.set_title(f"{word}\n(pos hit rate {scored['position_hit_rate']:.2f})")
        ax.set_xlabel("rollouts naming word")
    axes[0].set_ylabel("number of positions")
    fig.suptitle("Recurrence of the secret word across AV rollouts per position")
    fig.tight_layout()
    fig.savefig(out_dir / "recurrence_histogram.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--words", nargs="+", default=["gold", "ship", "smile", "cloud"])
    args = ap.parse_args()

    out_dir = RESULTS_ROOT / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cross_word_heatmap.png").unlink(missing_ok=True)

    plot_loss_curves(args.words, out_dir)
    rates = plot_recovery_rates(args.words, out_dir)
    plot_recovery_by_ptype(args.words, rates, out_dir)
    plot_recurrence(args.words, out_dir)

    for word in args.words:
        ft = rates[word]["finetuned"]
        base = rates[word]["base"]
        print(f"{word}: finetuned hit_rate={ft['hit_rate']:.3f} "
              f"pos_hit_rate={ft['position_hit_rate']:.3f} | "
              f"base hit_rate={base['hit_rate']:.3f} pos_hit_rate={base['position_hit_rate']:.3f}")
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
