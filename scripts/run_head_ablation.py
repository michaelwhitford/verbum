#!/usr/bin/env python
"""Run head ablation experiment on critical layers.

Usage::

    uv run python scripts/run_head_ablation.py

Or in background::

    uv run python scripts/run_head_ablation.py &

Loads Qwen3-4B, builds the fractal experiment graph, and fires
head ablation across 5 gate-ablation probes on 8 critical layers
(256 heads per probe, 1280 forward passes total).

Results cached in ``results/experiments/`` — crash and rerun to
resume where you left off.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ─── constants ──────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBE_SET = PROJECT_ROOT / "probes" / "gate-ablation.json"
GATES_DIR = PROJECT_ROOT / "gates"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
MODEL = "Qwen/Qwen3-4B"
CRITICAL_LAYERS = [0, 1, 4, 7, 24, 26, 30, 33]


def main() -> None:
    # Late imports — torch is heavy, fail fast on config errors first
    print(f"Probe set:  {PROBE_SET}")
    print(f"Gates dir:  {GATES_DIR}")
    print(f"Results:    {RESULTS_DIR}")
    print(f"Model:      {MODEL}")
    print(f"Layers:     {CRITICAL_LAYERS}")
    print("Heads/layer: 32")
    print(f"Total forward passes: {5 * len(CRITICAL_LAYERS) * 32}")
    print()

    if not PROBE_SET.is_file():
        print(f"ERROR: Probe set not found: {PROBE_SET}", file=sys.stderr)
        sys.exit(1)
    if not GATES_DIR.is_dir():
        print(f"ERROR: Gates dir not found: {GATES_DIR}", file=sys.stderr)
        sys.exit(1)

    # Build the experiment graph (no model needed yet)
    from verbum.experiments.head_ablation import build_head_ablation

    print("Building experiment graph...")
    graph = build_head_ablation(
        probe_set_path=PROBE_SET,
        gates_dir=GATES_DIR,
        target_layers=CRITICAL_LAYERS,
        model_name=MODEL,
    )
    print(f"Graph: {len(graph.children)} probes x {len(CRITICAL_LAYERS)} layers")
    print(f"Graph hash: {graph.config_hash[:24]}...")
    print()

    # Load model
    from verbum.instrument import load_model

    print("Loading model (this may take a minute)...")
    model, tokenizer, info = load_model(MODEL)
    print(f"Model loaded: {info.n_layers}L, {info.n_heads}H, {info.head_dim}D")
    print()

    # Build interceptors and run
    from verbum.experiment import default_interceptors, run

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    interceptors = default_interceptors(
        RESULTS_DIR,
        resources={"model": model, "tokenizer": tokenizer, "info": info},
    )

    print("Starting experiment...")
    print("=" * 60)
    results = run(graph, interceptors=interceptors, node_id="head-ablation")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_broken: dict[int, set[int]] = {}  # layer → set of broken heads across probes

    for probe_name, probe_result in sorted(results.items()):
        print(f"\n{probe_name}:")
        for layer_name, layer_result in sorted(probe_result.items()):
            layer_idx = layer_result["layer"]
            broken = layer_result["broken_heads"]
            n_broken = layer_result["n_broken"]
            baseline_ok = layer_result["baseline_has_lambda"]

            if layer_idx not in all_broken:
                all_broken[layer_idx] = set()
            all_broken[layer_idx].update(broken)

            status = f"{n_broken} broken" if n_broken > 0 else "all survive"
            baseline_str = "baseline OK" if baseline_ok else "BASELINE FAILED"
            print(f"  {layer_name}: {status} ({baseline_str})")
            if broken:
                print(f"    broken heads: {broken}")

    # Cross-probe summary
    print("\n" + "=" * 60)
    print("CROSS-PROBE ESSENTIAL HEADS (broken in ANY probe):")
    print("=" * 60)

    total_essential = 0
    for layer_idx in sorted(all_broken.keys()):
        heads = sorted(all_broken[layer_idx])
        total_essential += len(heads)
        if heads:
            print(f"  L{layer_idx}: {len(heads)} heads — {heads}")
        else:
            print(f"  L{layer_idx}: none")

    print(f"\nTotal essential heads: {total_essential} / {len(CRITICAL_LAYERS) * 32}")
    print(
        f"Circuit sparsity: {total_essential / (len(CRITICAL_LAYERS) * 32) * 100:.1f}%"
    )

    # Save summary
    summary_path = RESULTS_DIR / "head-ablation-summary.json"
    summary = {
        "model": MODEL,
        "critical_layers": CRITICAL_LAYERS,
        "n_probes": len(results),
        "essential_heads": {str(k): sorted(v) for k, v in sorted(all_broken.items())},
        "total_essential": total_essential,
        "total_candidates": len(CRITICAL_LAYERS) * 32,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
