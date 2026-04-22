#!/usr/bin/env python3
"""v6 compile gradient probe — no backward compatibility needed.

Probes a VSMLMV6 checkpoint with the compile-gradient probe set.
Runs forward_instrumented on each probe and displays v6-specific
metrics: gates, multiplicative modulation, complex phase angles,
and ternary statistics.

Usage:
    uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000.pt

    # With custom probe set:
    uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000.pt \\
        --probes probes/compile-gradient.json

    # Quiet: summary only (no per-probe detail):
    uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000.pt --quiet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

PROBES_PATH = Path("probes/compile-gradient.json")
GATES_DIR = Path("gates/")
RESULTS_DIR = Path("results/compile-gradient")

PASS_NAMES = ["L0_asc", "L1_asc", "L2_apex", "L1_desc", "L0_desc"]
PASS_LABELS = ["L0↑", "L1↑", " L2", "L1↓", "L0↓"]
PHASE_NAMES = ["prep", "converge", "consolidate"]
REG_NAMES = ["type", "scope", "role"]


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════


def load_probes(probe_path: Path | None = None) -> list[dict]:
    """Load probe set from JSON. Defaults to compile-gradient."""
    path = probe_path or PROBES_PATH
    data = json.loads(path.read_text())
    return data["probes"]


def load_gate(gate_id: str) -> str:
    """Load gate text file by ID."""
    return (GATES_DIR / f"{gate_id}.txt").read_text()


# ══════════════════════════════════════════════════════════════════════
# Checkpoint loading
# ══════════════════════════════════════════════════════════════════════


def load_checkpoint(path: Path, device: str) -> tuple:
    """Load a VSMLMV6 checkpoint.

    Args:
        path:   path to .pt checkpoint file
        device: 'mps', 'cuda', or 'cpu'

    Returns:
        (model, step, config) where:
          model  — VSMLMV6 instance on device, in eval mode
          step   — training step at checkpoint
          config — dict of architecture hyperparameters from checkpoint
    """
    from verbum.v6.model import VSMLMV6

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Extract config — provide defaults matching v6 training script
    config = ckpt.get("config", {})
    step = ckpt.get("step", 0)
    arch = ckpt.get("architecture", "vsm-lm-v6")

    if arch not in ("vsm-lm-v6", "VSM-LM-v6"):
        print(f"  WARNING: checkpoint architecture is '{arch}', expected 'vsm-lm-v6'")

    model = VSMLMV6(
        vocab_size=config.get("vocab_size", 50277),
        d_model=config.get("d_model", 512),
        d_register=config.get("d_register", 128),
        max_len=config.get("seq_len", 4096),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 1536),
        d_ff_consolidate=config.get("d_ff_consolidate", 2048),
        window=config.get("window", 8),
        strides=tuple(config.get("strides", [1, 8, 64, 512])),
        alpha=config.get("alpha", 1.18),
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, step, config


# ══════════════════════════════════════════════════════════════════════
# Probing
# ══════════════════════════════════════════════════════════════════════


def probe_checkpoint(
    model,
    probes: list[dict],
    tokenizer,
    device: str,
    gate_name: str = "compile",
) -> list[dict]:
    """Run forward_instrumented on each probe, collect v6 metrics.

    Args:
        model:      VSMLMV6 instance (eval mode)
        probes:     list of probe dicts from load_probes()
        tokenizer:  HuggingFace tokenizer
        device:     device string
        gate_name:  gate text file to prepend (default: 'compile')

    Returns:
        list of result dicts, one per probe, each containing:
          probe_id, category, gradient, prompt, gate_used,
          metrics (all forward_instrumented outputs),
          generation (short greedy decode),
          has_lambda (bool)
    """
    # Load gate text (prefix applied to every prompt)
    try:
        gate_text = load_gate(gate_name)
    except FileNotFoundError:
        print(f"  WARNING: gate '{gate_name}' not found — running without gate")
        gate_text = ""

    results = []

    with torch.no_grad():
        for probe in probes:
            probe_id = probe["id"]
            category = probe.get("category", "unknown")
            gradient = probe.get("metadata", {}).get("gradient", None)

            # Build gated prompt
            gate_for_probe = probe.get("gate", gate_name)
            if gate_for_probe == "null":
                full_prompt = probe["prompt"]
            else:
                full_prompt = gate_text + probe["prompt"]

            # Tokenize
            ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

            # Truncate if needed (v6 max_len=4096)
            if ids.shape[1] > model.max_len:
                ids = ids[:, -model.max_len:]

            # Forward instrumented
            t0 = time.time()
            _, _, metrics = model.forward_instrumented(ids)
            elapsed_ms = (time.time() - t0) * 1000

            # Short generation for qualitative check
            gen_ids = model.generate(ids, max_new_tokens=20, temperature=0.8)
            gen_text = tokenizer.decode(gen_ids[0, ids.shape[1]:], skip_special_tokens=True)
            has_lambda = "λ" in gen_text or "\\" in gen_text

            results.append({
                "probe_id": probe_id,
                "category": category,
                "gradient": gradient,
                "prompt": probe["prompt"],
                "gate_used": gate_for_probe,
                "generation": gen_text,
                "has_lambda": has_lambda,
                "elapsed_ms": round(elapsed_ms, 1),
                "metrics": {k: round(v, 6) for k, v in metrics.items()},
            })

    return results


# ══════════════════════════════════════════════════════════════════════
# Display
# ══════════════════════════════════════════════════════════════════════


def print_probe(probe_id: str, result: dict, verbose: bool = True) -> None:
    """Print v6-specific display for one probe result.

    Shows per-pass: gate values, modulation means, phase angles,
    and register norms.
    """
    metrics = result["metrics"]
    cat = result["category"]
    grad_str = f"  [gradient={result['gradient']:.1f}]" if result["gradient"] is not None else ""
    lambda_marker = "✓λ" if result["has_lambda"] else "  "

    print(f"\n  {lambda_marker} {probe_id:20s} [{cat:15s}]{grad_str}")
    print(f"     prompt: {result['prompt'][:70]!r}")
    print(f"     gen:    {result['generation'][:60]!r}  ({result['elapsed_ms']:.0f}ms)")

    if not verbose:
        return

    # Per-pass table: pass | gates(3 phases) | mod means | phase angles
    print(f"     ┌─────────────┬─────────────────────────────────┬────────────────────┬────────────────────────┐")
    print(f"     │ pass        │ gates  prep/conv/cons            │ mod  prep/conv/cons│ reg phases type/scope  │")
    print(f"     ├─────────────┼─────────────────────────────────┼────────────────────┼────────────────────────┤")

    for pname, plabel in zip(PASS_NAMES, PASS_LABELS):
        gates = [
            metrics.get(f"{pname}_{ph}_gate_mean", 0.0)
            for ph in PHASE_NAMES
        ]
        mods = [
            metrics.get(f"{pname}_{ph}_mod_mean", 1.0)
            for ph in PHASE_NAMES
        ]
        reg_type_phase = metrics.get(f"{pname}_register_type_phase_final", 0.0)
        reg_scope_phase = metrics.get(f"{pname}_register_scope_phase_final", 0.0)

        gate_str = "/".join(f"{g:+.2f}" for g in gates)
        mod_str = "/".join(f"{m:.2f}" for m in mods)
        phase_str = f"{reg_type_phase:+.3f}/{reg_scope_phase:+.3f}"

        print(f"     │ {plabel:11s} │ {gate_str:31s} │ {mod_str:18s} │ {phase_str:22s} │")

    print(f"     └─────────────┴─────────────────────────────────┴────────────────────┴────────────────────────┘")

    # Meta-S3 gates
    meta_str = "  meta-S3: " + "  ".join(
        f"{plabel.strip()}={metrics.get(f'meta_s3_gate_{pname}', 0.0):.3f}"
        for pname, plabel in zip(PASS_NAMES, PASS_LABELS)
    )
    print(f"     {meta_str}")

    # Overall expansion
    exp = metrics.get("overall_expansion", 0.0)
    embed_n = metrics.get("embed_norm", 0.0)
    out_n = metrics.get("output_norm", 0.0)
    print(f"     expansion: {exp:.3f}x  (embed={embed_n:.3f} → out={out_n:.3f})")


def print_summary(results: list[dict], step: int, model) -> None:
    """Print a summary table with per-category stats and ternary info."""
    print("\n" + "=" * 70)
    print(f"  v6 Probe Summary — step {step:,}")
    print("=" * 70)

    # Group by category
    categories: dict[str, list[dict]] = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, []).append(r)

    cat_order = ["strong_compile", "medium_compile", "weak_compile", "null", "anti_compile"]

    print(f"\n  {'Category':20s} {'N':>3} {'λ%':>6} {'expansion':>10} {'L0↑_conv_gate':>14} {'meta_L2':>9}")
    print(f"  {'─'*20} {'─'*3} {'─'*6} {'─'*10} {'─'*14} {'─'*9}")

    for cat in cat_order:
        if cat not in categories:
            continue
        cat_results = categories[cat]
        n = len(cat_results)
        lambda_frac = sum(1 for r in cat_results if r["has_lambda"]) / n * 100
        avg_exp = sum(r["metrics"].get("overall_expansion", 0) for r in cat_results) / n
        avg_conv_gate = sum(
            r["metrics"].get("L0_asc_converge_gate_mean", 0) for r in cat_results
        ) / n
        avg_meta_l2 = sum(
            r["metrics"].get("meta_s3_gate_L2_apex", 0) for r in cat_results
        ) / n

        print(
            f"  {cat:20s} {n:>3} {lambda_frac:>5.0f}%  "
            f"{avg_exp:>10.3f}  {avg_conv_gate:>14.3f}  {avg_meta_l2:>9.3f}"
        )

    # Ternary stats at probe time
    ternary_stats = model.ternary_stats()
    if ternary_stats:
        print(f"\n  Ternary statistics (module group averages at probe time):")

        group_stats: dict[str, list] = {
            "prep": [],
            "stride_stack": [],
            "consolidate": [],
            "mod_projs": [],
        }
        for mod_name, stat in ternary_stats.items():
            for group_key in group_stats:
                if mod_name.startswith(group_key):
                    group_stats[group_key].append(stat)
                    break
            else:
                group_stats.setdefault("other", []).append(stat)

        print(f"  {'Module group':15s}  {'#layers':>7}  {'sparsity':>9}  {'pos_frac':>9}  {'neg_frac':>9}  {'gamma':>8}")
        print(f"  {'─'*15}  {'─'*7}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}")
        for grp, stat_list in group_stats.items():
            if not stat_list:
                continue
            n_layers = len(stat_list)
            avg_sp = sum(s["sparsity"] for s in stat_list) / n_layers
            avg_pos = sum(s["pos_frac"] for s in stat_list) / n_layers
            avg_neg = sum(s["neg_frac"] for s in stat_list) / n_layers
            avg_gm = sum(s["gamma"] for s in stat_list) / n_layers
            print(
                f"  {grp:15s}  {n_layers:>7}  {avg_sp:>9.3f}  "
                f"{avg_pos:>9.3f}  {avg_neg:>9.3f}  {avg_gm:>8.4f}"
            )

    # Lambda score overall
    n_total = len(results)
    n_lambda = sum(1 for r in results if r["has_lambda"])
    print(f"\n  Overall λ generation: {n_lambda}/{n_total} ({n_lambda/n_total*100:.0f}%)")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# Results saving
# ══════════════════════════════════════════════════════════════════════


def save_results(results: list[dict], step: int, config: dict, model) -> Path:
    """Save probe results to results/compile-gradient/vsm_probe_step_{step}_v6.json.

    Args:
        results: list of result dicts from probe_checkpoint()
        step:    training step
        config:  config dict from checkpoint
        model:   VSMLMV6 for ternary_stats

    Returns:
        Path to saved file
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"vsm_probe_step_{step:06d}_v6.json"

    ternary_stats = model.ternary_stats()

    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "architecture": "vsm-lm-v6",
        "step": step,
        "config": config,
        "ternary_stats": ternary_stats,
        "n_probes": len(results),
        "n_lambda": sum(1 for r in results if r["has_lambda"]),
        "results": results,
    }

    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v6 compile gradient probe — probes VSMLMV6 checkpoints"
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to VSMLMV6 .pt checkpoint (e.g. checkpoints/vsm-lm-v6/step_001000.pt)",
    )
    parser.add_argument(
        "--probes",
        type=Path,
        default=PROBES_PATH,
        help=f"Probe set JSON (default: {PROBES_PATH})",
    )
    parser.add_argument(
        "--gate",
        type=str,
        default="compile",
        help="Gate name to apply (default: compile). Use 'null' for no gate.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print summary only, not per-probe detail",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: mps, cuda, or cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving results to file",
    )
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\n{'=' * 60}")
    print(f"  VSM-LM v6 Compile Gradient Probe")
    print(f"{'=' * 60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device:     {device}")
    print(f"  Gate:       {args.gate}")
    print(f"  Probes:     {args.probes}")

    # Load checkpoint
    print(f"\n  Loading checkpoint...")
    model, step, config = load_checkpoint(args.checkpoint, device)
    print(f"  Loaded v6 model at step {step:,}")

    # Print architecture summary
    params = model.count_parameters()
    total_m = params["total"] / 1e6
    ternary_m = params["total_ternary"] / 1e6
    eff_bits = params["effective_bits_x1000"] / 1000.0
    print(f"  Parameters: {total_m:.1f}M total  ({ternary_m:.1f}M ternary, {eff_bits:.2f} bits/param)")

    # Load tokenizer
    print(f"  Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")

    # Load probes
    probes = load_probes(args.probes)
    print(f"  Loaded {len(probes)} probes from {args.probes}")
    print()

    # Run probing
    t_start = time.time()
    results = probe_checkpoint(model, probes, tokenizer, device, gate_name=args.gate)
    t_elapsed = time.time() - t_start

    print(f"  Probed {len(results)} inputs in {t_elapsed:.1f}s")

    # Per-probe display
    if not args.quiet:
        print(f"\n{'─' * 70}")
        print(f"  Per-probe results:")
        print(f"{'─' * 70}")
        for r in results:
            print_probe(r["probe_id"], r, verbose=True)

    # Summary
    print_summary(results, step, model)

    # Save
    if not args.no_save:
        out_path = save_results(results, step, config, model)
        print(f"\n  Saved: {out_path}")
    else:
        print(f"\n  (results not saved — use without --no-save to persist)")


if __name__ == "__main__":
    main()
