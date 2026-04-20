#!/usr/bin/env python
"""Cross-task ablation — do the same 3 heads control different tasks?

Usage::

    uv run python scripts/run_cross_task.py [--dry-run]
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
MODEL = "Qwen/Qwen3-4B"
ESSENTIAL_HEADS = [(1, 0), (24, 0), (24, 2)]
HEAD_LABELS = {(1, 0): "L1:H0", (24, 0): "L24:H0", (24, 2): "L24:H2"}


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("  CROSS-TASK ABLATION — typed_apply universality test")
    print("=" * 60)
    print()

    from verbum.experiments.cross_task import build_cross_task

    print("Building cross-task graph...")
    graph = build_cross_task(
        gates_dir=PROJECT_ROOT / "gates",
        tasks={
            "compile": str(PROJECT_ROOT / "probes/gate-ablation.json"),
            "summarize": str(PROJECT_ROOT / "probes/summarize.json"),
            "translate": str(PROJECT_ROOT / "probes/translate.json"),
            "classify": str(PROJECT_ROOT / "probes/classify.json"),
            "extract": str(PROJECT_ROOT / "probes/extract.json"),
        },
        model_name=MODEL,
    )

    # Count nodes
    from verbum.experiment import Graph

    def count_leaves(comp: object) -> int:
        if isinstance(comp, Graph):
            return sum(count_leaves(c) for c in comp.children.values())
        return 1

    total = count_leaves(graph)
    print(f"Graph: {len(graph.children)} tasks, {total} leaf nodes")
    print(f"Hash: {graph.config_hash[:24]}...")
    print()

    if dry_run:
        print("[DRY RUN] Graph built. Exiting.")
        return

    # Load model
    from verbum.instrument import load_model

    print("Loading model...")
    model, tokenizer, info = load_model(MODEL)
    print(f"Loaded: {info.n_layers}L, {info.n_heads}H")

    # Run
    from verbum.experiment import default_interceptors, run

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    interceptors = default_interceptors(
        RESULTS_DIR,
        resources={"model": model, "tokenizer": tokenizer, "info": info},
    )

    print()
    print("Running experiments...")
    print("=" * 60)

    results = run(graph, interceptors=interceptors, node_id="cross-task")

    # ─── Essentiality Matrix ──────────────────────────────────
    print()
    print("=" * 60)
    print("  ESSENTIALITY MATRIX — head x task")
    print("=" * 60)
    print()

    tasks = sorted(results.keys())
    head_names = ["L1-H0", "L24-H0", "L24-H2"]

    # Header
    header = f"{'':18s}"
    for task in tasks:
        header += f" {task:>10s}"
    print(header)
    print("-" * len(header))

    # Baseline row
    row = f"{'baseline':18s}"
    for task in tasks:
        task_result = results[task]
        successes = sum(
            1
            for pr in task_result.values()
            if pr.get("baseline", {}).get("success", False)
        )
        total = len(task_result)
        row += f" {successes}/{total:>7d}"
    print(row)

    # Per-head rows
    matrix: dict[str, dict[str, str]] = {}
    for head_name in head_names:
        row = f"{head_name:18s}"
        matrix[head_name] = {}
        for task in tasks:
            task_result = results[task]
            successes = 0
            total = 0
            for probe_result in task_result.values():
                if head_name in probe_result:
                    total += 1
                    if probe_result[head_name].get("success", False):
                        successes += 1
            pct = f"{successes}/{total}"
            matrix[head_name][task] = pct
            # Mark breaks
            baseline_successes = sum(
                1
                for pr in task_result.values()
                if pr.get("baseline", {}).get("success", False)
            )
            if successes < baseline_successes:
                pct = f"*{pct}*"
            row += f" {pct:>10s}"
        print(row)

    print()
    print("* = degraded vs baseline (head is essential for this task)")

    # ─── Per-task detail ──────────────────────────────────────
    print()
    print("=" * 60)
    print("  PER-TASK DETAIL")
    print("=" * 60)

    for task in tasks:
        print(f"\n--- {task.upper()} ---")
        task_result = results[task]
        for probe_name, probe_result in sorted(task_result.items()):
            baseline = probe_result.get("baseline", {})
            b_status = "OK" if baseline.get("success") else "FAIL"
            parts = [f"{probe_name}: baseline={b_status}"]
            for head_name in head_names:
                if head_name in probe_result:
                    h_status = (
                        "ok" if probe_result[head_name].get("success") else "BREAK"
                    )
                    parts.append(f"{head_name}={h_status}")
            print(f"  {', '.join(parts)}")

    # ─── Generations for broken cases ─────────────────────────
    print()
    print("=" * 60)
    print("  BROKEN CASES — what the model outputs when heads are ablated")
    print("=" * 60)

    for task in tasks:
        task_result = results[task]
        for probe_name, probe_result in sorted(task_result.items()):
            baseline = probe_result.get("baseline", {})
            if not baseline.get("success"):
                continue  # skip probes where baseline already fails
            for head_name in head_names:
                if head_name in probe_result:
                    hr = probe_result[head_name]
                    if not hr.get("success"):
                        print(f"\n  {task}/{probe_name}/{head_name}:")
                        print(f"    baseline: {baseline.get('generation', '')[:100]}")
                        print(f"    ablated:  {hr.get('generation', '')[:100]}")


if __name__ == "__main__":
    main()
