#!/usr/bin/env python
"""Run all circuit discovery experiments.

Five experiments as one fractal Graph, each independently cacheable.
Crash and rerun to resume where you left off.

Usage::

    uv run python scripts/run_circuit_discovery.py [--dry-run]

Experiments:
  1. sufficiency   — are 3 heads sufficient without the rest?
  2. multi-head    — threshold for distributed composition breakdown
  3. bos-tracing   — which layers' BOS contributions are necessary?
  4. dual-process  — does the model reason its way to lambda when direct fails?
  5. decompile     — is the circuit bidirectional (compile + decompile)?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBE_SET = PROJECT_ROOT / "probes" / "gate-ablation.json"
DECOMPILE_SET = PROJECT_ROOT / "probes" / "decompile.json"
GATES_DIR = PROJECT_ROOT / "gates"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
MODEL = "Qwen/Qwen3-4B"
CRITICAL_LAYERS = [0, 1, 4, 7, 24, 26, 30, 33]
ESSENTIAL_HEADS = [(1, 0), (24, 0), (24, 2)]


def build_all() -> dict:
    """Build all experiment graphs. No model needed."""
    from verbum.experiment import Graph
    from verbum.experiments.bos_tracing import build_bos_tracing
    from verbum.experiments.decompile import build_decompile_ablation
    from verbum.experiments.dual_process import build_dual_process
    from verbum.experiments.multi_head import build_multi_head_experiment

    graphs: dict = {}

    print("Building multi-head experiment...")
    graphs["multi-head"] = build_multi_head_experiment(
        probe_set_path=PROBE_SET,
        gates_dir=GATES_DIR,
        essential_heads=ESSENTIAL_HEADS,
        critical_layers=CRITICAL_LAYERS,
        model_name=MODEL,
    )

    print("Building BOS tracing experiment...")
    graphs["bos-tracing"] = build_bos_tracing(
        probe_set_path=PROBE_SET,
        gates_dir=GATES_DIR,
        model_name=MODEL,
    )

    print("Building dual process experiment...")
    graphs["dual-process"] = build_dual_process(
        probe_set_path=PROBE_SET,
        gates_dir=GATES_DIR,
        essential_heads=ESSENTIAL_HEADS,
        model_name=MODEL,
    )

    print("Building decompile experiment...")
    graphs["decompile"] = build_decompile_ablation(
        probe_set_path=DECOMPILE_SET,
        gates_dir=GATES_DIR,
        essential_heads=ESSENTIAL_HEADS,
        model_name=MODEL,
    )

    # Top-level fractal graph
    master = Graph(id="circuit-discovery", children=graphs)

    return {"master": master, "sub": graphs}


def print_graph_stats(graphs: dict) -> None:
    """Print graph structure stats."""

    def count_leaves(comp: object) -> int:
        from verbum.experiment import Graph

        if isinstance(comp, Graph):
            return sum(count_leaves(c) for c in comp.children.values())
        return 1

    total = 0
    for name, graph in graphs["sub"].items():
        n = count_leaves(graph)
        total += n
        print(f"  {name}: {n} leaf nodes")
    print(f"  TOTAL: {total} leaf computations")
    print(f"  Master hash: {graphs['master'].config_hash[:24]}...")


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("  CIRCUIT DISCOVERY — 5 EXPERIMENTS")
    print("=" * 60)
    print()

    all_graphs = build_all()
    print()
    print_graph_stats(all_graphs)
    print()

    if dry_run:
        print("[DRY RUN] Graph built successfully. Exiting.")
        return

    # Load model
    from verbum.instrument import load_model

    print("Loading model...")
    model, tokenizer, info = load_model(MODEL)
    print(f"Loaded: {info.n_layers}L, {info.n_heads}H, {info.head_dim}D")

    # BOS tracing needs pre-computed null residuals
    from verbum.instrument import capture_bos_residuals

    null_gate = (GATES_DIR / "null.txt").read_text("utf-8")
    null_prompt = null_gate + "Tell me about the weather today."
    print("Capturing null BOS residuals...")
    null_bos = capture_bos_residuals(model, tokenizer, null_prompt, info)
    print(f"Captured {len(null_bos)} layer residuals")

    # Build interceptors
    from verbum.experiment import default_interceptors, run

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    interceptors = default_interceptors(
        RESULTS_DIR,
        resources={
            "model": model,
            "tokenizer": tokenizer,
            "info": info,
            "null_bos_residuals": null_bos,
        },
    )

    print()
    print("Starting experiments...")
    print("=" * 60)

    results = run(
        all_graphs["master"],
        interceptors=interceptors,
        node_id="circuit-discovery",
    )

    # ─── Summary ──────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    # 1. Sufficiency
    if "multi-head" in results:
        mh = results["multi-head"]
        if "sufficiency" in mh:
            print("\n--- SUFFICIENCY TEST ---")
            for probe_name, r in sorted(mh["sufficiency"].items()):
                status = "COMPILES" if r["has_lambda"] else "BREAKS"
                print(f"  {probe_name}: {status} (zeroed {r['n_zeroed']} heads)")

        # Threshold
        for key in sorted(mh.keys()):
            if key.startswith("threshold-"):
                print(f"\n--- {key.upper()} ---")
                for probe_name, r in sorted(mh[key].items()):
                    status = "survives" if r["has_lambda"] else "BREAKS"
                    print(f"  {probe_name}: {status}")

    # 2. BOS tracing
    if "bos-tracing" in results:
        print("\n--- BOS TRACING (which layers' BOS is necessary) ---")
        for probe_name, probe_result in sorted(results["bos-tracing"].items()):
            broken_layers = []
            for _layer_name, r in sorted(probe_result.items()):
                if not r["has_lambda"]:
                    broken_layers.append(r["layer"])
            if broken_layers:
                print(f"  {probe_name}: breaks at layers {broken_layers}")
            else:
                print(f"  {probe_name}: all layers survive BOS patching")

    # 3. Dual process
    if "dual-process" in results:
        print("\n--- DUAL PROCESS (System 1 vs System 2) ---")
        for probe_name, probe_result in sorted(results["dual-process"].items()):
            baseline = probe_result.get("baseline", {})
            print(f"\n  {probe_name}:")
            print(f"    baseline: lambda={baseline.get('has_lambda')}")
            for head_name, r in sorted(probe_result.items()):
                if head_name == "baseline":
                    continue
                lam = "lambda" if r["has_lambda"] else "no-lambda"
                reas = "reasoning" if r["has_reasoning"] else "direct"
                print(f"    {head_name}: {lam}, {reas}")

    # 4. Decompile
    if "decompile" in results:
        print("\n--- DECOMPILE (is the circuit bidirectional?) ---")
        for probe_name, probe_result in sorted(results["decompile"].items()):
            baseline = probe_result.get("baseline", {})
            print(f"\n  {probe_name}:")
            eng = "english" if baseline.get("has_english") else "no-english"
            print(f"    baseline: {eng}")
            for head_name, r in sorted(probe_result.items()):
                if head_name == "baseline":
                    continue
                eng = "english" if r["has_english"] else "NO-ENGLISH"
                lam = "+lambda" if r["has_lambda"] else ""
                print(f"    {head_name}: {eng} {lam}")

    # Save summary
    summary_path = RESULTS_DIR / "circuit-discovery-summary.json"
    # Can't serialize full results (may contain tensors), save structure
    summary = {
        "model": MODEL,
        "essential_heads": ESSENTIAL_HEADS,
        "critical_layers": CRITICAL_LAYERS,
        "experiments": list(results.keys()),
        "completed": True,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
