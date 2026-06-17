#!/usr/bin/env python3
"""Plots for the activation-preserving taboo finetune sweep (Phase 2).

Compares regular taboo LoRAs ("<word>-baseline") against preservation-penalty
LoRAs ("<word>-preserve-<loss>-w<weight>") on NLA secret-word recovery.

Reads:
  - training metrics: results/taboo_baseline/<run>/metrics.json
  - NLA eval:         results/taboo_baseline/nla_eval/<run>/summary.json
  - L20 drift:        results/taboo_baseline/drift/<run>.json

    python -m nla_experiments.taboo_secret_word.plot_taboo_preserve --words cloud gold ship smile
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "taboo_baseline"
LOSS_TYPES = ["combined", "mse", "cos", "kl"]
LOSS_COLORS = {
    "combined": "#9467bd",
    "mse": "#1f77b4",
    "cos": "#2ca02c",
    "kl": "#ff7f0e",
    "baseline": "#d62728",
}
DRIFT_METRICS = [("rel_mse", "L20 relative MSE"), ("cos_pen", "L20 mean (1 - cos)"), ("kl", "next-token KL (nats)")]
RUN_RE = re.compile(
    r"^(?P<word>[a-z]+)-preserve-(?:(?P<loss>mse|cos|kl)-w(?P<weight>[0-9.]+)|(?P<combined>combined)-light)$"
)
BEHAVIOR_MIN = 0.5  # a "genuine evasion" must keep at least half the on-policy hint behavior


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def discover_preserve_runs() -> list[dict]:
    """Every preserve run that has both a training metrics.json and an nla_eval summary.json."""
    runs = []
    for metrics_path in sorted(RESULTS_ROOT.glob("*/metrics.json")):
        run_name = metrics_path.parent.name
        m = RUN_RE.match(run_name)
        if not m:
            continue
        summary_path = RESULTS_ROOT / "nla_eval" / run_name / "summary.json"
        if not summary_path.exists():
            continue
        loss = m["loss"] or m["combined"]
        runs.append({
            "run_name": run_name,
            "word": m["word"],
            "loss": loss,
            "weight": float(m["weight"] or 0.0),
            "metrics": metrics_path,
            "summary": summary_path,
            "drift": RESULTS_ROOT / "drift" / f"{run_name}.json",
        })
    return runs


def baseline_summary(word: str) -> dict:
    return load_json(RESULTS_ROOT / "nla_eval" / f"{word}-baseline" / "summary.json")


def final_eval_ce(metrics_path: Path) -> float:
    return load_json(metrics_path)["history"]["eval_loss"][-1]["loss"]


def behavior_retention(run_name: str) -> float | None:
    path = RESULTS_ROOT / "behavior" / f"{run_name}.json"
    if not path.exists():
        return None
    return load_json(path)["behavior_retention"]


def weighted_preserve_value(run: dict, train_record: dict) -> float:
    if run["loss"] == "combined":
        return float(train_record["preserve"])
    return float(run["weight"]) * float(train_record["preserve"])


def run_label(run: dict) -> str:
    if run["loss"] == "combined":
        return "combined-light"
    return f"w{run['weight']:g}"


def select_best(runs: list[dict], word: str, loss: str) -> dict | None:
    """Strongest genuine-evasion run for (word, loss): lowest NLA recovery among runs that
    keep 0 eval leaks and behavior_retention >= BEHAVIOR_MIN (rules out behavior collapse)."""
    cands = [r for r in runs if r["word"] == word and r["loss"] == loss
             and load_json(r["summary"])["response_leak_count"] == 0
             and (behavior_retention(r["run_name"]) or -1.0) >= BEHAVIOR_MIN]
    if not cands:
        return None
    return min(cands, key=lambda r: load_json(r["summary"])["position_hit_rate"])


def plot_training_curves(runs: list[dict], word: str, out_dir: Path) -> None:
    word_runs = [r for r in runs if r["word"] == word]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, loss in zip(axes.ravel(), LOSS_TYPES):
        loss_runs = sorted([r for r in word_runs if r["loss"] == loss], key=lambda r: r["weight"])
        weights = [r["weight"] for r in loss_runs]
        alphas = np.linspace(0.4, 1.0, len(loss_runs)) if loss_runs else []
        for r, alpha in zip(loss_runs, alphas):
            hist = load_json(r["metrics"])["history"]["train_loss"]
            steps = [h["step"] for h in hist]
            ce = [h["ce"] for h in hist]
            pen = [weighted_preserve_value(r, h) for h in hist]
            ax.plot(steps, ce, color=LOSS_COLORS[loss], alpha=alpha, lw=1.4,
                    label=f"{run_label(r)} CE")
            ax.plot(steps, pen, color="black", alpha=alpha, lw=1.0, ls="--",
                    label=f"{run_label(r)} λ·pen")
        ax.set_title(f"{word}: {loss} preservation")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"Training: assistant-token CE (solid) vs λ·penalty (dashed) — {word}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / f"training_curves_{word}.png", dpi=150)
    plt.close(fig)


def plot_recovery_comparison(runs: list[dict], words: list[str], out_dir: Path) -> None:
    groups = ["baseline"] + LOSS_TYPES
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(words))
    width = 0.8 / len(groups)
    for k, group in enumerate(groups):
        vals = []
        for word in words:
            if group == "baseline":
                vals.append(baseline_summary(word)["hit_rate"])
            else:
                best = select_best(runs, word, group)
                # NaN (no bar drawn) when no run keeps behavior — a gap, not a 0-recovery claim.
                vals.append(load_json(best["summary"])["hit_rate"] if best else np.nan)
        bars = ax.bar(
            x + (k - (len(groups) - 1) / 2) * width,
            vals,
            width,
            label=group,
            color=LOSS_COLORS[group],
        )
        for b, v in zip(bars, vals):
            label = "collapse" if np.isnan(v) else f"{v:.2f}"
            ax.text(
                b.get_x() + b.get_width() / 2,
                (0.0 if np.isnan(v) else v) + 0.01,
                label,
                ha="center",
                fontsize=7,
                rotation=90 if np.isnan(v) else 0,
                color="gray" if np.isnan(v) else "black",
            )
    ax.set_xticks(x, words)
    ax.set_ylim(0, 1)
    ax.set_title("AV explanation hit rate")
    ax.set_ylabel("secret-word mention rate")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("NLA recovery: regular vs activation-preserving taboo finetuning (best λ per loss)")
    fig.tight_layout()
    fig.savefig(out_dir / "recovery_comparison.png", dpi=150)
    plt.close(fig)


def plot_tradeoff(runs: list[dict], word: str, out_dir: Path) -> None:
    base = baseline_summary(word)
    base_ce = final_eval_ce(RESULTS_ROOT / f"{word}-baseline" / "metrics.json")
    base_beh = behavior_retention(f"{word}-baseline")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for loss in LOSS_TYPES:
        loss_runs = sorted([r for r in runs if r["word"] == word and r["loss"] == loss],
                           key=lambda r: r["weight"])
        weights = [0.0] + [r["weight"] for r in loss_runs]
        phr = [base["position_hit_rate"]] + [load_json(r["summary"])["position_hit_rate"] for r in loss_runs]
        beh = [np.nan if base_beh is None else base_beh]
        beh += [np.nan if behavior_retention(r["run_name"]) is None else behavior_retention(r["run_name"]) for r in loss_runs]
        ces = [base_ce] + [final_eval_ce(r["metrics"]) for r in loss_runs]
        xs = [max(w, 0.05) for w in weights]  # 0 placed at left on log axis
        axes[0].plot(xs, phr, marker="o", color=LOSS_COLORS[loss], label=loss)
        axes[1].plot(xs, beh, marker="^", color=LOSS_COLORS[loss], label=loss)
        axes[2].plot(xs, ces, marker="s", color=LOSS_COLORS[loss], label=loss)
    axes[0].set_title(f"{word}: NLA position hit rate vs λ (lower = more hidden)")
    axes[0].set_ylabel("position hit rate")
    axes[1].set_title(f"{word}: on-policy behavior retention vs λ (higher = still hinting)")
    axes[1].set_ylabel("behavior retention")
    axes[1].axhline(BEHAVIOR_MIN, color="gray", ls=":", lw=1, label="collapse threshold")
    axes[2].set_title(f"{word}: held-out CE vs λ (teacher-forced; misses collapse)")
    axes[2].set_ylabel("held-out cross-entropy")
    for ax in (axes[0], axes[1]):
        ax.set_ylim(0, 1)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("preservation weight λ (0 shown at left)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"Preservation strength tradeoff — {word}")
    fig.tight_layout()
    fig.savefig(out_dir / f"tradeoff_{word}.png", dpi=150)
    plt.close(fig)


def plot_pareto(runs: list[dict], words: list[str], out_dir: Path) -> None:
    """Headline tradeoff: on-policy behavior retention (x) vs NLA recovery (y).
    Bottom-right = secret hidden AND behavior kept (genuine evasion). KL collapse sits bottom-left."""
    markers = {"cloud": "o", "gold": "s", "ship": "^", "smile": "D"}
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for word in words:
        base = baseline_summary(word)
        base_beh = behavior_retention(f"{word}-baseline")
        if base_beh is not None:
            ax.scatter(base_beh, base["position_hit_rate"],
                       color=LOSS_COLORS["baseline"], marker=markers[word], s=130, edgecolor="black", zorder=3)
    for r in runs:
        if not (RESULTS_ROOT / "behavior" / f"{r['run_name']}.json").exists():
            continue
        x = behavior_retention(r["run_name"])
        y = load_json(r["summary"])["position_hit_rate"]
        ax.scatter(x, y, color=LOSS_COLORS[r["loss"]], marker=markers[r["word"]], s=90,
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.annotate(f"{r['loss']} {run_label(r)}", (x, y), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.axvline(BEHAVIOR_MIN, color="gray", ls=":", lw=1)
    ax.text(BEHAVIOR_MIN - 0.02, 0.75, "← collapse", fontsize=8, color="gray", ha="right")
    ax.text(BEHAVIOR_MIN + 0.02, 0.75, "retained →", fontsize=8, color="gray", ha="left")
    ax.set_xlabel("on-policy behavior retention (higher = still playing the taboo game)")
    ax.set_ylabel("NLA position hit rate (lower = secret hidden)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.8)
    ax.grid(alpha=0.3)
    loss_handles = [plt.Line2D([], [], marker="o", ls="", color=LOSS_COLORS[l], label=l)
                    for l in ["baseline"] + LOSS_TYPES]
    word_handles = [plt.Line2D([], [], marker=markers[w], ls="", color="gray", label=w) for w in words]
    ax.legend(handles=loss_handles + word_handles, fontsize=8, loc="upper left")
    fig.suptitle("Genuine evasion vs behavior collapse: behavior retention vs NLA recovery")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_behavior_vs_recovery.png", dpi=150)
    plt.close(fig)


def plot_drift_vs_recovery(runs: list[dict], words: list[str], out_dir: Path) -> None:
    markers = {"cloud": "o", "gold": "s", "ship": "^", "smile": "D"}
    # baselines (loss="baseline") + every preserve run that has a drift file
    points = []
    for word in words:
        points.append({"word": word, "loss": "baseline",
                       "drift": RESULTS_ROOT / "drift" / f"{word}-baseline.json",
                       "summary": RESULTS_ROOT / "nla_eval" / f"{word}-baseline" / "summary.json"})
    for r in runs:
        if r["drift"].exists():
            points.append({"word": r["word"], "loss": r["loss"], "drift": r["drift"], "summary": r["summary"]})

    fig, axes = plt.subplots(1, len(DRIFT_METRICS), figsize=(5.2 * len(DRIFT_METRICS), 4.6))
    for ax, (key, label) in zip(axes, DRIFT_METRICS):
        for p in points:
            if not p["drift"].exists() or not p["summary"].exists():
                continue
            d = load_json(p["drift"])["all"][key]
            phr = load_json(p["summary"])["position_hit_rate"]
            ax.scatter(d, phr, color=LOSS_COLORS[p["loss"]], marker=markers[p["word"]],
                       s=70, edgecolor="black", linewidth=0.4)
        ax.set_xlabel(label)
        ax.set_ylabel("NLA position hit rate")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    loss_handles = [plt.Line2D([], [], marker="o", ls="", color=LOSS_COLORS[l], label=l)
                    for l in ["baseline"] + LOSS_TYPES]
    word_handles = [plt.Line2D([], [], marker=markers[w], ls="", color="gray", label=w) for w in words]
    axes[0].legend(handles=loss_handles + word_handles, fontsize=8, loc="lower right")
    fig.suptitle("L20 activation drift vs NLA recovery (more drift → more recovery)")
    fig.tight_layout()
    fig.savefig(out_dir / "drift_vs_recovery.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--words", nargs="+", default=["cloud", "gold", "ship", "smile"])
    ap.add_argument("--curve-words", nargs="+", default=["cloud"],
                    help="words to draw training/tradeoff curves for (need a full λ sweep)")
    args = ap.parse_args()

    out_dir = RESULTS_ROOT / "plots_preserve"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_preserve_runs()
    print(f"[plot_taboo_preserve] discovered {len(runs)} preserve runs")

    for word in args.curve_words:
        plot_training_curves(runs, word, out_dir)
        plot_tradeoff(runs, word, out_dir)
    plot_recovery_comparison(runs, args.words, out_dir)
    plot_drift_vs_recovery(runs, args.words, out_dir)
    plot_pareto(runs, args.words, out_dir)

    print(f"[plot_taboo_preserve] best-λ selections (genuine evasion: 0 leaks, behavior >= {BEHAVIOR_MIN}):")
    for word in args.words:
        for loss in LOSS_TYPES:
            best = select_best(runs, word, loss)
            if best:
                s = load_json(best["summary"])
                print(f"  {word}/{loss}: {best['run_name']} pos_hit_rate={s['position_hit_rate']:.3f} "
                      f"behavior={behavior_retention(best['run_name']):.3f} leaks={s['response_leak_count']}")
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
