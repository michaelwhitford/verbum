#!/usr/bin/env python3
"""Replicate the Qwen3-4B circuit discovery on Phi-4-mini-instruct.

Full pipeline:
  1. Layer ablation → find critical layers
  2. Head ablation on critical layers → find essential heads
  3. Cross-task ablation → test universality of essential heads
  4. Failure mode analysis → quantify System 1 → System 2 shift

Usage:
    uv run python scripts/run_phi4_replication.py

Outputs to results/phi4-mini/
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import structlog

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)

log = structlog.get_logger()

MODEL_NAME = "microsoft/Phi-4-mini-instruct"
RESULTS_DIR = Path("results/phi4-mini")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Saved: {path}")


# ──────────────────────────── Phase 0: Load ───────────────────────────


def load():
    """Load Phi-4-mini and return (model, tokenizer, info)."""
    from verbum.instrument import load_model

    banner("LOADING Phi-4-mini-instruct")
    model, tokenizer, info = load_model(MODEL_NAME)
    print(f"  Layers: {info.n_layers}")
    print(f"  Heads: {info.n_heads}")
    print(f"  KV Heads: {info.n_kv_heads}")
    print(f"  Head dim: {info.head_dim}")
    print(f"  Hidden: {info.hidden_size}")
    return model, tokenizer, info


# ──────────────────────────── Phase 1: Layer ablation ─────────────────


def phase1_layer_ablation(model, tokenizer, info):
    """Skip-ablate each layer to find critical layers for compilation."""
    from verbum.instrument import ablate_layers
    from verbum.probes import load_probe_set, resolve_probes

    banner("PHASE 1: Layer Ablation — find critical layers")

    probe_set = load_probe_set("probes/gate-ablation.json")
    resolved = resolve_probes(probe_set, Path("gates"))

    # Use first probe for layer scan
    prompt = resolved[0].full_prompt
    print(f"  Probe: {resolved[0].probe_id}")
    print(f"  Prompt: {prompt[:80]}...")

    baseline, results = ablate_layers(model, tokenizer, prompt, info)
    print(f"\n  Baseline: {baseline[:100]}")

    critical = []
    for r in results:
        if not r.has_lambda:
            critical.append(r.layer)
            print(f"  ✗ Layer {r.layer:2d} CRITICAL: {r.generation[:60]}")

    print(f"\n  Critical layers: {critical}")
    print(f"  Total critical: {len(critical)} / {info.n_layers}")

    save_json(RESULTS_DIR / "phase1-layer-ablation.json", {
        "model": MODEL_NAME,
        "n_layers": info.n_layers,
        "critical_layers": critical,
        "baseline": baseline,
        "probe_id": resolved[0].probe_id,
    })

    return critical


# ──────────────────────────── Phase 2: Head ablation ──────────────────


def phase2_head_ablation(model, tokenizer, info, critical_layers):
    """For each critical layer, ablate each head individually."""
    from verbum.experiment import Graph, default_interceptors, run
    from verbum.experiments.head_ablation import build_head_ablation

    banner("PHASE 2: Head Ablation — find essential heads")

    cache_dir = RESULTS_DIR / "experiments"
    cache_dir.mkdir(parents=True, exist_ok=True)

    graph = build_head_ablation(
        probe_set_path="probes/gate-ablation.json",
        gates_dir="gates",
        target_layers=critical_layers,
        model_name=MODEL_NAME,
        n_heads=info.n_heads,
        head_dim=info.head_dim,
        max_new_tokens=50,
    )

    interceptors = default_interceptors(
        cache_dir,
        resources={"model": model, "tokenizer": tokenizer, "info": info},
    )

    results = run(graph, interceptors=interceptors)

    # Aggregate: find heads whose ablation breaks ALL probes
    essential_per_layer = {}
    for probe_id, probe_data in results.items():
        for layer_key, layer_data in probe_data.items():
            layer_idx = layer_data["layer"]
            for hr in layer_data["head_results"]:
                if not hr["has_lambda"]:
                    essential_per_layer.setdefault(layer_idx, set()).add(hr["head"])

    # A head is essential if it breaks compilation in ANY probe
    # (conservative: we want heads that matter)
    essential_heads = []
    for layer_idx in sorted(essential_per_layer.keys()):
        for head_idx in sorted(essential_per_layer[layer_idx]):
            essential_heads.append([layer_idx, head_idx])

    # Stricter: essential = breaks ALL probes (not just some)
    n_probes = len(results)
    strict_essential = {}
    for probe_id, probe_data in results.items():
        for layer_key, layer_data in probe_data.items():
            layer_idx = layer_data["layer"]
            for hr in layer_data["head_results"]:
                if not hr["has_lambda"]:
                    key = (layer_idx, hr["head"])
                    strict_essential[key] = strict_essential.get(key, 0) + 1

    strict_heads = [
        [l, h] for (l, h), count in strict_essential.items()
        if count == n_probes
    ]

    summary = {
        "model": MODEL_NAME,
        "critical_layers": critical_layers,
        "n_probes": n_probes,
        "essential_heads_any": essential_heads,
        "essential_heads_all": strict_heads,
        "total_candidates": len(critical_layers) * info.n_heads,
        "break_counts": {
            f"L{l}:H{h}": count
            for (l, h), count in sorted(strict_essential.items())
        },
    }

    print(f"\n  Essential (any probe): {essential_heads}")
    print(f"  Essential (all probes): {strict_heads}")
    print(f"  Total candidates: {len(critical_layers) * info.n_heads}")

    save_json(RESULTS_DIR / "phase2-head-ablation.json", summary)

    return essential_heads, strict_heads


# ──────────────────────────── Phase 3: Cross-task ─────────────────────


def phase3_cross_task(model, tokenizer, info, essential_heads):
    """Test essential heads against all 5 tasks."""
    from verbum.experiment import default_interceptors, run
    from verbum.experiments.cross_task import build_cross_task

    banner("PHASE 3: Cross-Task — compositor universality test")

    heads = [tuple(h) for h in essential_heads]
    print(f"  Testing heads: {heads}")

    cache_dir = RESULTS_DIR / "experiments"

    graph = build_cross_task(
        essential_heads=heads,
        model_name=MODEL_NAME,
        max_new_tokens=50,
    )

    interceptors = default_interceptors(
        cache_dir,
        resources={"model": model, "tokenizer": tokenizer, "info": info},
    )

    results = run(graph, interceptors=interceptors)

    # Print essentiality matrix
    print("\n" + "=" * 60)
    print("  ESSENTIALITY MATRIX — head x task")
    print("=" * 60)

    from verbum.experiments.cross_task import DETECTORS

    tasks = list(results.keys())
    head_labels = ["baseline"] + [f"L{l}-H{h}" for l, h in heads]

    header = f"{'':>20}" + "".join(f"{t:>12}" for t in tasks)
    print(header)
    print("-" * len(header))

    matrix = {}
    for task_name, probes in results.items():
        for probe_id, conditions in probes.items():
            for cond_name, cond_data in conditions.items():
                key = (task_name, cond_name)
                if key not in matrix:
                    matrix[key] = {"success": 0, "total": 0}
                matrix[key]["total"] += 1
                if cond_data.get("success", False):
                    matrix[key]["success"] += 1

    for label in head_labels:
        row = f"{label:>20}"
        for task in tasks:
            key = (task, label)
            if key in matrix:
                s, t = matrix[key]["success"], matrix[key]["total"]
                row += f"{s}/{t:>6}  "
            else:
                row += f"{'?':>12}"
        print(row)

    save_json(RESULTS_DIR / "phase3-cross-task.json", results)
    return results


# ──────────────────────────── Phase 4: Failure modes ──────────────────


def phase4_failure_modes(cross_task_results):
    """Analyze failure modes in cross-task data."""
    from verbum.analysis.failure_modes import analyze_cross_task, format_report

    banner("PHASE 4: Failure Mode Analysis")

    report = analyze_cross_task(cross_task_results)
    print(format_report(report))

    save_data = {k: v for k, v in report.items() if k != "records"}
    save_json(RESULTS_DIR / "phase4-failure-modes.json", save_data)

    return report


# ──────────────────────────── Main ────────────────────────────────────


def main():
    start = time.time()
    banner(f"PHI-4-MINI REPLICATION — {datetime.now(timezone.utc).isoformat()}")

    model, tokenizer, info = load()

    # Phase 1: Find critical layers
    critical_layers = phase1_layer_ablation(model, tokenizer, info)

    if not critical_layers:
        print("WARNING: No critical layers found! Model may handle ablation differently.")
        print("Using all layers for head scan (expensive)...")
        critical_layers = list(range(info.n_layers))

    # Phase 2: Find essential heads
    essential_any, essential_all = phase2_head_ablation(
        model, tokenizer, info, critical_layers
    )

    # Use the broadest set for cross-task testing
    heads_to_test = essential_all if essential_all else essential_any
    if not heads_to_test:
        print("WARNING: No essential heads found!")
        print("This would be a significant negative result.")
        save_json(RESULTS_DIR / "summary.json", {
            "model": MODEL_NAME,
            "finding": "no_essential_heads",
            "critical_layers": critical_layers,
            "elapsed_s": time.time() - start,
        })
        return

    # Phase 3: Cross-task
    cross_task_results = phase3_cross_task(
        model, tokenizer, info, heads_to_test
    )

    # Phase 4: Failure mode analysis
    report = phase4_failure_modes(cross_task_results)

    # Final summary
    elapsed = time.time() - start
    summary = {
        "model": MODEL_NAME,
        "n_layers": info.n_layers,
        "n_heads": info.n_heads,
        "head_dim": info.head_dim,
        "critical_layers": critical_layers,
        "essential_heads_any": essential_any,
        "essential_heads_all": essential_all,
        "elapsed_s": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "comparison_with_qwen": {
            "qwen_essential": [[1, 0], [24, 0], [24, 2]],
            "qwen_n_layers": 36,
            "qwen_n_heads": 32,
            "qwen_head_dim": 80,
        },
    }
    save_json(RESULTS_DIR / "summary.json", summary)

    banner(f"COMPLETE — {elapsed:.0f}s")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Critical layers: {critical_layers}")
    print(f"  Essential heads (any): {essential_any}")
    print(f"  Essential heads (all): {essential_all}")
    print(f"  Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
