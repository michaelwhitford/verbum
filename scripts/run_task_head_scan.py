#!/usr/bin/env python
"""Full head scan across tasks — find task-specific essential heads.

Runs 8 critical layers x 32 heads x 5 probes for each task.
Finds specialized preprocessor heads that configure L24:H0.

Usage::

    uv run python scripts/run_task_head_scan.py [--dry-run]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments"
MODEL = "Qwen/Qwen3-4B"
CRITICAL_LAYERS = [0, 1, 4, 7, 24, 26, 30, 33]
COMPILE_ESSENTIAL = {(1, 0), (24, 0), (24, 2)}


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("  TASK HEAD SCAN — find task-specific essential heads")
    print("=" * 60)
    print()

    from verbum.experiments.task_head_scan import build_task_head_scan

    tasks = {
        "extract": str(PROJECT_ROOT / "probes/extract.json"),
        "translate": str(PROJECT_ROOT / "probes/translate.json"),
        "classify": str(PROJECT_ROOT / "probes/classify.json"),
    }

    print("Building scan graph...")
    graph = build_task_head_scan(
        tasks=tasks,
        gates_dir=str(PROJECT_ROOT / "gates"),
        target_layers=CRITICAL_LAYERS,
        model_name=MODEL,
    )

    from verbum.experiment import Graph

    def count_leaves(comp: object) -> int:
        if isinstance(comp, Graph):
            return sum(count_leaves(c) for c in comp.children.values())
        return 1

    total = count_leaves(graph)
    fwd_passes = total * 32  # each leaf does 32 head ablations
    print(f"Graph: {len(graph.children)} tasks, {total} leaf nodes")
    print(f"Total forward passes: {fwd_passes}")
    print(f"Hash: {graph.config_hash[:24]}...")
    print()

    if dry_run:
        print("[DRY RUN] Graph built. Exiting.")
        return

    from verbum.instrument import load_model

    print("Loading model...")
    model, tokenizer, info = load_model(MODEL)
    print(f"Loaded: {info.n_layers}L, {info.n_heads}H, {info.head_dim}D")

    from verbum.experiment import default_interceptors, run

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    interceptors = default_interceptors(
        RESULTS_DIR,
        resources={"model": model, "tokenizer": tokenizer, "info": info},
    )

    print()
    print("Running scan...")
    print("=" * 60)

    results = run(graph, interceptors=interceptors, node_id="task-head-scan")

    # ─── Analysis ─────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  RESULTS — essential heads per task")
    print("=" * 60)

    all_essential: dict[str, dict[int, set[int]]] = {}

    for task_name, task_result in sorted(results.items()):
        print(f"\n{'=' * 50}")
        print(f"  {task_name.upper()}")
        print(f"{'=' * 50}")

        task_broken: dict[int, set[int]] = {}

        for probe_name, probe_result in sorted(task_result.items()):
            print(f"\n  {probe_name}:")
            for layer_name, layer_result in sorted(probe_result.items()):
                layer_idx = layer_result["layer"]
                broken = layer_result["broken_heads"]
                n_broken = layer_result["n_broken"]
                baseline_ok = layer_result["baseline_success"]

                if layer_idx not in task_broken:
                    task_broken[layer_idx] = set()
                task_broken[layer_idx].update(broken)

                if not baseline_ok:
                    print(f"    {layer_name}: baseline FAIL (skip)")
                elif n_broken > 0:
                    print(f"    {layer_name}: {n_broken} broken — {broken}")
                else:
                    print(f"    {layer_name}: all survive")

        all_essential[task_name] = task_broken

        # Task summary
        print(f"\n  --- {task_name} CROSS-PROBE ESSENTIAL ---")
        task_total = 0
        for layer_idx in sorted(task_broken.keys()):
            heads = sorted(task_broken[layer_idx])
            task_total += len(heads)
            if heads:
                # Mark shared with compile
                annotated = []
                for h in heads:
                    tag = " *" if (layer_idx, h) in COMPILE_ESSENTIAL else ""
                    annotated.append(f"{h}{tag}")
                print(f"    L{layer_idx}: {', '.join(annotated)}")
        print(f"    Total: {task_total} essential heads")

    # ─── Cross-task comparison ────────────────────────────────
    print()
    print("=" * 60)
    print("  CROSS-TASK COMPARISON")
    print("=" * 60)
    print()
    print("  * = also essential for compile (L1:H0, L24:H0, L24:H2)")
    print()

    # Collect all unique (layer, head) pairs across tasks
    all_heads: set[tuple[int, int]] = set()
    for task_broken in all_essential.values():
        for layer_idx, heads in task_broken.items():
            for h in heads:
                all_heads.add((layer_idx, h))
    # Add compile essentials
    all_heads.update(COMPILE_ESSENTIAL)

    if all_heads:
        # Header
        task_names = ["compile"] + sorted(all_essential.keys())
        header = f"{'head':12s}"
        for t in task_names:
            header += f" {t:>10s}"
        print(header)
        print("-" * len(header))

        for layer_idx, head_idx in sorted(all_heads):
            label = f"L{layer_idx}:H{head_idx}"
            row = f"{label:12s}"

            # Compile column
            is_compile = (layer_idx, head_idx) in COMPILE_ESSENTIAL
            row += f" {'ESSENTIAL':>10s}" if is_compile else f" {'—':>10s}"

            # Other tasks
            for task_name in sorted(all_essential.keys()):
                task_broken = all_essential[task_name]
                is_essential = head_idx in task_broken.get(layer_idx, set())
                row += f" {'ESSENTIAL':>10s}" if is_essential else f" {'—':>10s}"

            print(row)

    # ─── Save ─────────────────────────────────────────────────
    summary = {
        "model": MODEL,
        "critical_layers": CRITICAL_LAYERS,
        "compile_essential": [[l, h] for l, h in sorted(COMPILE_ESSENTIAL)],
        "task_essential": {
            task: {str(l): sorted(hs) for l, hs in sorted(broken.items()) if hs}
            for task, broken in all_essential.items()
        },
    }
    summary_path = RESULTS_DIR / "task-head-scan-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
