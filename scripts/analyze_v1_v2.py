#!/usr/bin/env python3
"""Analyze MontaguLM v1 (rigid) vs CompressorLM v2 (iterative).

Loads all 10 checkpoints from each model, extracts:
  - Eval loss curves
  - Training loss (smoothed)
  - Phase gradient norms over training
  - Phase activation norms (v1 only — v2 didn't record these)
  - Compile gate test results

Produces plots and a JSON summary.

Usage:
    uv run python scripts/analyze_v1_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════

V1_CKPT_DIR = Path("checkpoints/montagu-lm")
V2_CKPT_DIR = Path("checkpoints/compressor-lm-iterative")
V1_SUMMARY = Path("results/montagu-lm/training-summary.json")
V2_SUMMARY = Path("results/compressor-lm-iterative/training-summary.json")
OUTPUT_DIR = Path("results/v1-v2-comparison")

STEPS = list(range(1000, 11000, 1000))  # 1K to 10K


def load_checkpoints(ckpt_dir: Path) -> list[dict]:
    """Load all checkpoint files, extracting non-weight data."""
    checkpoints = []
    for step in STEPS:
        path = ckpt_dir / f"step_{step:06d}.pt"
        if not path.exists():
            print(f"  ⚠ Missing: {path}")
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # Extract everything except model/optimizer state (huge)
        entry = {
            "step": ckpt.get("step", step),
            "loss": ckpt.get("loss"),
            "phase_grad_norms": ckpt.get("phase_grad_norms", {}),
            "phase_act_norms": ckpt.get("phase_act_norms", {}),
            "compile_results": ckpt.get("compile_results", []),
            "train_losses_recent": ckpt.get("train_losses_recent", []),
            "eval_losses": ckpt.get("eval_losses", []),
        }
        checkpoints.append(entry)
        print(f"  ✓ Step {step:5d}: loss={entry['loss']:.4f}")
    return checkpoints


def smooth(values: list[float], window: int = 50) -> list[float]:
    """Simple moving average."""
    if len(values) < window:
        return values
    cumsum = np.cumsum([0.0] + values)
    return [(cumsum[i + window] - cumsum[i]) / window
            for i in range(len(values) - window + 1)]


def plot_eval_losses(v1_summary: dict, v2_summary: dict, output_dir: Path):
    """Side-by-side eval loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    v1_evals = v1_summary["eval_losses"]
    v2_evals = v2_summary["eval_losses"]

    ax.plot([e["step"] for e in v1_evals], [e["loss"] for e in v1_evals],
            "o-", color="#d62728", linewidth=2, markersize=5, label="v1 rigid (seq=256)")
    ax.plot([e["step"] for e in v2_evals], [e["loss"] for e in v2_evals],
            "s-", color="#1f77b4", linewidth=2, markersize=5, label="v2 iterative (seq=4096)")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Eval Loss (nats)", fontsize=12)
    ax.set_title("Eval Loss: v1 Rigid vs v2 Iterative CompressorLM", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate best losses
    v1_best = min(v1_evals, key=lambda e: e["loss"])
    v2_best = min(v2_evals, key=lambda e: e["loss"])
    ax.annotate(f"best: {v1_best['loss']:.3f}\n(step {v1_best['step']})",
                xy=(v1_best["step"], v1_best["loss"]),
                xytext=(v1_best["step"] + 500, v1_best["loss"] + 0.2),
                arrowprops=dict(arrowstyle="->", color="#d62728"),
                fontsize=9, color="#d62728")
    ax.annotate(f"best: {v2_best['loss']:.3f}\n(step {v2_best['step']})",
                xy=(v2_best["step"], v2_best["loss"]),
                xytext=(v2_best["step"] - 2000, v2_best["loss"] + 0.3),
                arrowprops=dict(arrowstyle="->", color="#1f77b4"),
                fontsize=9, color="#1f77b4")

    plt.tight_layout()
    path = output_dir / "eval_loss_comparison.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_train_losses(v1_summary: dict, v2_summary: dict, output_dir: Path):
    """Smoothed training loss from summary last100."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # v1 train losses (last 100 steps of training)
    v1_train = v1_summary.get("train_losses_last100", [])
    v2_train = v2_summary.get("train_losses_last100", [])

    if v1_train:
        ax1.plot(v1_train, alpha=0.3, color="#d62728", linewidth=0.8)
        if len(v1_train) >= 10:
            ax1.plot(smooth(v1_train, 10), color="#d62728", linewidth=2, label="10-step MA")
        ax1.set_title(f"v1 Rigid — Last 100 Steps\nmean={np.mean(v1_train):.3f}, std={np.std(v1_train):.3f}", fontsize=11)
        ax1.set_ylabel("Train Loss", fontsize=11)
        ax1.set_xlabel("Step (relative)", fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    if v2_train:
        ax2.plot(v2_train, alpha=0.3, color="#1f77b4", linewidth=0.8)
        if len(v2_train) >= 10:
            ax2.plot(smooth(v2_train, 10), color="#1f77b4", linewidth=2, label="10-step MA")
        ax2.set_title(f"v2 Iterative — Last 100 Steps\nmean={np.mean(v2_train):.3f}, std={np.std(v2_train):.3f}", fontsize=11)
        ax2.set_ylabel("Train Loss", fontsize=11)
        ax2.set_xlabel("Step (relative)", fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "train_loss_last100.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_grad_norms(v1_ckpts: list[dict], v2_ckpts: list[dict], output_dir: Path):
    """Phase gradient norm evolution across training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── v1: rigid phases ──
    v1_steps = [c["step"] for c in v1_ckpts if c["phase_grad_norms"]]
    v1_phases = {}
    for c in v1_ckpts:
        if not c["phase_grad_norms"]:
            continue
        for phase, norm in c["phase_grad_norms"].items():
            if phase not in v1_phases:
                v1_phases[phase] = []
            v1_phases[phase].append(norm)

    colors_v1 = {"phase1_type": "#ff7f0e", "phase2_parse": "#2ca02c",
                 "phase3_apply": "#d62728", "embeddings": "#7f7f7f"}
    labels_v1 = {"phase1_type": "Type (P1)", "phase2_parse": "Parse (P2)",
                 "phase3_apply": "Apply (P3)", "embeddings": "Embeddings"}

    for phase, norms in v1_phases.items():
        ax1.plot(v1_steps[:len(norms)], norms,
                 "o-", color=colors_v1.get(phase, "#333"),
                 label=labels_v1.get(phase, phase), linewidth=2, markersize=5)

    ax1.set_xlabel("Training Step", fontsize=11)
    ax1.set_ylabel("Gradient L2 Norm", fontsize=11)
    ax1.set_title("v1 Rigid — Phase Gradient Norms", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # ── v2: iterative phases ──
    v2_steps = [c["step"] for c in v2_ckpts if c["phase_grad_norms"]]
    v2_phases = {}
    for c in v2_ckpts:
        if not c["phase_grad_norms"]:
            continue
        for phase, norm in c["phase_grad_norms"].items():
            if phase not in v2_phases:
                v2_phases[phase] = []
            v2_phases[phase].append(norm)

    colors_v2 = {"type": "#ff7f0e", "parse": "#2ca02c",
                 "apply": "#d62728", "predict": "#9467bd",
                 "embeddings": "#7f7f7f"}
    labels_v2 = {"type": "Type", "parse": "Parse",
                 "apply": "Apply", "predict": "Predict (PC)",
                 "embeddings": "Embeddings"}

    for phase, norms in v2_phases.items():
        ax2.plot(v2_steps[:len(norms)], norms,
                 "s-", color=colors_v2.get(phase, "#333"),
                 label=labels_v2.get(phase, phase), linewidth=2, markersize=5)

    ax2.set_xlabel("Training Step", fontsize=11)
    ax2.set_ylabel("Gradient L2 Norm", fontsize=11)
    ax2.set_title("v2 Iterative — Phase Gradient Norms", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    path = output_dir / "phase_grad_norms.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_act_norms(v1_ckpts: list[dict], output_dir: Path):
    """Phase activation norms (v1 only — v2 didn't record these)."""
    v1_with_act = [c for c in v1_ckpts if c["phase_act_norms"]]
    if not v1_with_act:
        print("  ⚠ No activation norms found in v1 checkpoints")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    steps = [c["step"] for c in v1_with_act]
    phases = {}
    for c in v1_with_act:
        for phase, norm in c["phase_act_norms"].items():
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(norm)

    colors = {"input_embed": "#7f7f7f", "phase1_type": "#ff7f0e",
              "phase2_parse": "#2ca02c", "phase3_apply": "#d62728"}
    labels = {"input_embed": "Input Embed", "phase1_type": "After Type",
              "phase2_parse": "After Parse", "phase3_apply": "After Apply"}

    for phase, norms in phases.items():
        ax.plot(steps[:len(norms)], norms,
                "o-", color=colors.get(phase, "#333"),
                label=labels.get(phase, phase), linewidth=2, markersize=5)

    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Activation L2 Norm (mean over batch)", fontsize=11)
    ax.set_title("v1 Rigid — Activation Norms Through Phases", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "v1_activation_norms.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_grad_norm_ratios(v1_ckpts: list[dict], v2_ckpts: list[dict], output_dir: Path):
    """Ratio of apply/type gradient norms — measures phase differentiation."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # v1: phase3_apply / phase1_type
    v1_steps = []
    v1_ratios = []
    for c in v1_ckpts:
        g = c["phase_grad_norms"]
        if g and "phase3_apply" in g and "phase1_type" in g and g["phase1_type"] > 0:
            v1_steps.append(c["step"])
            v1_ratios.append(g["phase3_apply"] / g["phase1_type"])

    # v2: apply / type
    v2_steps = []
    v2_ratios = []
    for c in v2_ckpts:
        g = c["phase_grad_norms"]
        if g and "apply" in g and "type" in g and g["type"] > 0:
            v2_steps.append(c["step"])
            v2_ratios.append(g["apply"] / g["type"])

    ax.plot(v1_steps, v1_ratios, "o-", color="#d62728", linewidth=2,
            markersize=6, label="v1 rigid: Apply/Type ratio")
    ax.plot(v2_steps, v2_ratios, "s-", color="#1f77b4", linewidth=2,
            markersize=6, label="v2 iterative: Apply/Type ratio")

    ax.axhline(y=1.0, color="#333", linestyle="--", alpha=0.5, label="Equal (ratio=1)")
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Gradient Norm Ratio (Apply / Type)", fontsize=11)
    ax.set_title("Phase Differentiation: Apply vs Type Gradient Ratio", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    path = output_dir / "grad_norm_ratio_apply_type.png"
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


def build_summary(v1_summary: dict, v2_summary: dict,
                  v1_ckpts: list[dict], v2_ckpts: list[dict]) -> dict:
    """Build JSON summary comparing v1 and v2."""

    def grad_norm_at_step(ckpts, step, phase):
        for c in ckpts:
            if c["step"] == step:
                return c["phase_grad_norms"].get(phase)
        return None

    # Gradient norm trajectories
    v1_grad_trajectory = {}
    for c in v1_ckpts:
        v1_grad_trajectory[c["step"]] = c["phase_grad_norms"]

    v2_grad_trajectory = {}
    for c in v2_ckpts:
        v2_grad_trajectory[c["step"]] = c["phase_grad_norms"]

    # Compile test results across training
    v1_compile = {c["step"]: sum(1 for r in c["compile_results"] if r.get("has_lambda"))
                  for c in v1_ckpts}
    v2_compile = {c["step"]: sum(1 for r in c["compile_results"] if r.get("has_lambda"))
                  for c in v2_ckpts}

    return {
        "comparison": "MontaguLM v1 (rigid) vs CompressorLM v2 (iterative)",
        "v1": {
            "architecture": "Rigid 3-phase, separate residual streams",
            "seq_len": v1_summary["config"].get("seq_len", 256),
            "params": v1_summary["params"]["total"],
            "tokens_trained": v1_summary["tokens_trained"],
            "best_eval_loss": v1_summary["best_eval_loss"],
            "best_eval_step": min(v1_summary["eval_losses"], key=lambda e: e["loss"])["step"],
            "final_eval_loss": v1_summary["final_eval_loss"],
            "train_loss_mean_last100": float(np.mean(v1_summary.get("train_losses_last100", [0]))),
            "train_loss_std_last100": float(np.std(v1_summary.get("train_losses_last100", [0]))),
            "grad_norm_trajectory": v1_grad_trajectory,
            "compile_gate_activations": v1_compile,
        },
        "v2": {
            "architecture": "Iterative predictive coding, shared residual, strided W=8",
            "seq_len": v2_summary.get("seq_len", 4096),
            "params": v2_summary["params"]["total"],
            "tokens_trained": v2_summary["tokens_trained"],
            "best_eval_loss": v2_summary["best_eval_loss"],
            "best_eval_step": min(v2_summary["eval_losses"], key=lambda e: e["loss"])["step"],
            "final_eval_loss": v2_summary["final_eval_loss"],
            "train_loss_mean_last100": float(np.mean(v2_summary.get("train_losses_last100", [0]))),
            "train_loss_std_last100": float(np.std(v2_summary.get("train_losses_last100", [0]))),
            "grad_norm_trajectory": v2_grad_trajectory,
            "compile_gate_activations": v2_compile,
        },
        "delta": {
            "eval_loss_improvement": v1_summary["best_eval_loss"] - v2_summary["best_eval_loss"],
            "eval_loss_improvement_pct": (v1_summary["best_eval_loss"] - v2_summary["best_eval_loss"]) / v1_summary["best_eval_loss"] * 100,
            "final_loss_improvement": v1_summary["final_eval_loss"] - v2_summary["final_eval_loss"],
            "param_ratio": v1_summary["params"]["total"] / v2_summary["params"]["total"],
            "seq_len_ratio": v2_summary.get("seq_len", 4096) / v1_summary["config"].get("seq_len", 256),
        },
        "key_observations": [],  # Filled by human after reviewing plots
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # ── Load summaries ────────────────────────────────────────────────
    print("Loading training summaries...")
    v1_summary = json.loads(V1_SUMMARY.read_text())
    v2_summary = json.loads(V2_SUMMARY.read_text())

    print(f"  v1: best={v1_summary['best_eval_loss']:.3f}, final={v1_summary['final_eval_loss']:.3f}, params={v1_summary['params']['total']:,}")
    print(f"  v2: best={v2_summary['best_eval_loss']:.3f}, final={v2_summary['final_eval_loss']:.3f}, params={v2_summary['params']['total']:,}")
    delta = v1_summary["best_eval_loss"] - v2_summary["best_eval_loss"]
    print(f"  Δ best: {delta:+.3f} nats ({delta / v1_summary['best_eval_loss'] * 100:.1f}% improvement)")

    # ── Load checkpoints ──────────────────────────────────────────────
    print("\nLoading v1 checkpoints (rigid)...")
    v1_ckpts = load_checkpoints(V1_CKPT_DIR)

    print("\nLoading v2 checkpoints (iterative)...")
    v2_ckpts = load_checkpoints(V2_CKPT_DIR)

    # ── Generate plots ────────────────────────────────────────────────
    print("\nGenerating plots...")

    print("  1. Eval loss comparison")
    plot_eval_losses(v1_summary, v2_summary, OUTPUT_DIR)

    print("  2. Training loss last 100 steps")
    plot_train_losses(v1_summary, v2_summary, OUTPUT_DIR)

    print("  3. Phase gradient norms")
    plot_grad_norms(v1_ckpts, v2_ckpts, OUTPUT_DIR)

    print("  4. Activation norms (v1 only)")
    plot_act_norms(v1_ckpts, OUTPUT_DIR)

    print("  5. Gradient norm ratios (apply/type)")
    plot_grad_norm_ratios(v1_ckpts, v2_ckpts, OUTPUT_DIR)

    # ── Build summary ─────────────────────────────────────────────────
    print("\nBuilding summary...")
    summary = build_summary(v1_summary, v2_summary, v1_ckpts, v2_ckpts)

    summary_path = OUTPUT_DIR / "comparison-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  Saved: {summary_path}")

    # ── Print key findings ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  KEY FINDINGS")
    print("=" * 60)
    print(f"\n  Best eval loss:  v1={v1_summary['best_eval_loss']:.3f}  v2={v2_summary['best_eval_loss']:.3f}  Δ={delta:+.3f}")
    print(f"  Final eval loss: v1={v1_summary['final_eval_loss']:.3f}  v2={v2_summary['final_eval_loss']:.3f}")
    print(f"  Parameters:      v1={v1_summary['params']['total']:,}  v2={v2_summary['params']['total']:,}")
    print(f"  Seq length:      v1={v1_summary['config'].get('seq_len', 256)}  v2={v2_summary.get('seq_len', 4096)}")

    # Last checkpoint grad norms
    if v1_ckpts:
        last_v1 = v1_ckpts[-1]
        print(f"\n  v1 final grad norms: {json.dumps({k: round(v, 4) for k, v in last_v1['phase_grad_norms'].items()})}")
    if v2_ckpts:
        last_v2 = v2_ckpts[-1]
        print(f"  v2 final grad norms: {json.dumps({k: round(v, 4) for k, v in last_v2['phase_grad_norms'].items()})}")

    # Compile gate across training
    print(f"\n  Compile gate activations across training:")
    for c in v1_ckpts:
        n = sum(1 for r in c["compile_results"] if r.get("has_lambda"))
        print(f"    v1 step {c['step']:5d}: {n}/4")
    for c in v2_ckpts:
        n = sum(1 for r in c["compile_results"] if r.get("has_lambda"))
        print(f"    v2 step {c['step']:5d}: {n}/4")

    print(f"\n  All outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
