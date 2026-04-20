#!/usr/bin/env python3
"""Replicate the circuit discovery pipeline on Pythia-2.8B-deduped.

Tests the localization gradient hypothesis:
  - Qwen3-4B (general web + instruction) -> 3 essential heads (sparse)
  - Phi-4-mini (reasoning-dense synthetic) -> 0 essential, 40 degraded
  - Pythia-2.8B (The Pile only) -> prediction: <=3 essential heads

Pythia is a BASE model -- no instruction tuning, no RLHF. The compile
gate is a few-shot pattern that works as text completion. Phase 0
verifies this.

Key architecture: GPTNeoXForCausalLM
  - model.gpt_neox.layers (not model.model.layers)
  - layer.attention (not layer.self_attn)
  - No GQA

Usage:
    uv run python scripts/run_pythia_replication.py

Outputs to results/pythia-2.8b/
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
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

MODEL_NAME = "EleutherAI/pythia-2.8b-deduped"
PROBE_SET = "probes/gate-ablation-base.json"
GATES_DIR = Path("gates")
RESULTS_DIR = Path("results/pythia-2.8b")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Saved: {path}")


def _lh_list(pairs: list) -> list[list[int]]:
    """Convert list of (layer, head) tuples to JSON-safe [[l, h], ...]."""
    return [[layer, head] for layer, head in pairs]


# ──────────────────────────── Phase 0: Load + Baseline ────────────────


def load():
    """Load Pythia-2.8B and return (model, tokenizer, info).

    Uses float32 because Pythia-2.8B produces NaN logits in fp16
    on MPS — a numerical stability issue specific to this architecture
    and backend combination.
    """
    import torch

    from verbum.instrument import load_model

    banner("LOADING Pythia-2.8B-deduped (float32)")
    model, tokenizer, info = load_model(
        MODEL_NAME, dtype=torch.float32
    )
    print(f"  Architecture: {type(model).__name__}")
    print(f"  Layers: {info.n_layers}")
    print(f"  Heads: {info.n_heads}")
    print(f"  KV Heads: {info.n_kv_heads}")
    print(f"  Head dim: {info.head_dim}")
    print(f"  Hidden: {info.hidden_size}")
    print("  Dtype: float32 (fp16 produces NaN on MPS)")
    return model, tokenizer, info


def phase0_baseline(model, tokenizer):
    """Verify Pythia can compile lambda at all using the gate prompt.

    This is the critical gate: if the base model can't do in-context
    few-shot lambda compilation, the experiment stops here (which is
    itself a finding about the localization gradient).
    """
    from verbum.instrument import LAMBDA_INDICATORS, _detect_lambda, _generate
    from verbum.probes import load_probe_set, resolve_probes

    banner("PHASE 0: Baseline -- can Pythia compile lambda?")

    probe_set = load_probe_set(PROBE_SET)
    resolved = resolve_probes(probe_set, GATES_DIR)

    results = []
    for rp in resolved:
        gen = _generate(
            model, tokenizer, rp.full_prompt, max_new_tokens=80
        )
        has_lambda = _detect_lambda(gen)
        lcount = sum(gen.count(s) for s in LAMBDA_INDICATORS)

        results.append({
            "probe_id": rp.probe_id,
            "prompt": rp.prompt,
            "generation": gen,
            "has_lambda": has_lambda,
            "lambda_count": lcount,
        })

        status = "Y COMPILES" if has_lambda else "X NO LAMBDA"
        print(f"  {status}  [{rp.probe_id}]  {gen[:100]}")

    n_success = sum(1 for r in results if r["has_lambda"])
    success_rate = n_success / len(results)
    print(f"\n  Success rate: {success_rate:.0%} ({n_success}/{len(results)})")

    save_json(RESULTS_DIR / "phase0-baseline.json", {
        "model": MODEL_NAME,
        "success_rate": success_rate,
        "results": results,
    })

    return results, success_rate


# ──────────────────────────── Phase 1: Layer Ablation ─────────────────


def phase1_layer_ablation(model, tokenizer, info):
    """Skip-ablate each layer to find critical layers."""
    from verbum.instrument import ablate_layers
    from verbum.probes import load_probe_set, resolve_probes

    banner("PHASE 1: Layer Ablation -- find critical layers")

    probe_set = load_probe_set(PROBE_SET)
    resolved = resolve_probes(probe_set, GATES_DIR)

    all_critical = []

    for rp in resolved[:2]:
        print(f"\n  Probe: {rp.probe_id}")
        print(f"  Prompt: {rp.prompt[:60]}...")

        baseline, results = ablate_layers(
            model, tokenizer, rp.full_prompt, info
        )
        print(f"  Baseline: {baseline[:100]}")

        critical = []
        for r in results:
            if not r.has_lambda:
                critical.append(r.layer)
                gen_snip = r.generation[:60]
                print(f"    X Layer {r.layer:2d} CRITICAL: {gen_snip}")

        all_critical.append({
            "probe_id": rp.probe_id,
            "baseline": baseline,
            "critical_layers": critical,
            "total_layers": info.n_layers,
        })

    # Union of critical layers across probes
    critical_union = sorted(
        set().union(*(set(pc["critical_layers"]) for pc in all_critical))
    )

    print(f"\n  Critical layers (union): {critical_union}")
    print(f"  Total critical: {len(critical_union)} / {info.n_layers}")

    save_json(RESULTS_DIR / "phase1-layer-ablation.json", {
        "model": MODEL_NAME,
        "n_layers": info.n_layers,
        "critical_layers_union": critical_union,
        "per_probe": all_critical,
    })

    return critical_union


# ──────────────────────────── Phase 2: Head Ablation ──────────────────


def phase2_head_ablation(model, tokenizer, info, critical_layers):
    """For each critical layer, ablate each head individually."""
    from verbum.instrument import LAMBDA_INDICATORS, ablate_heads
    from verbum.probes import load_probe_set, resolve_probes

    banner("PHASE 2: Head Ablation -- find essential heads")

    probe_set = load_probe_set(PROBE_SET)
    resolved = resolve_probes(probe_set, GATES_DIR)

    essential_per_probe: dict[str, dict] = {}

    for rp in resolved:
        print(f"\n  Probe: {rp.probe_id}")
        baseline, results = ablate_heads(
            model, tokenizer, rp.full_prompt, info,
            target_layers=critical_layers,
            max_new_tokens=80,
        )

        essential = []
        degraded = []
        baseline_count = sum(
            baseline.count(s) for s in LAMBDA_INDICATORS
        )

        for r in results:
            if not r.has_lambda:
                essential.append((r.layer, r.head))
            elif r.lambda_count < baseline_count - 1:
                degraded.append((r.layer, r.head))

        essential_per_probe[rp.probe_id] = {
            "essential": essential,
            "degraded": degraded,
            "baseline": baseline,
            "baseline_lambda_count": baseline_count,
        }

        print(f"    Essential: {essential}")
        print(f"    Degraded: {len(degraded)}")

    # Aggregate across probes
    break_counts: Counter = Counter()
    degrade_counts: Counter = Counter()
    n_probes = len(resolved)

    for _pid, pdata in essential_per_probe.items():
        for layer, head in pdata["essential"]:
            break_counts[(layer, head)] += 1
        for layer, head in pdata["degraded"]:
            degrade_counts[(layer, head)] += 1

    essential_all = sorted([
        (layer, head)
        for (layer, head), count in break_counts.items()
        if count == n_probes
    ])
    essential_any = sorted(break_counts.keys())

    total_candidates = len(critical_layers) * info.n_heads

    print(f"\n  Essential (all {n_probes} probes): {essential_all}")
    print(f"  Essential (any probe): {essential_any}")
    print(f"  Total degraded: {len(degrade_counts)}")
    print(f"  Total candidates: {total_candidates}")

    summary = {
        "model": MODEL_NAME,
        "critical_layers": critical_layers,
        "n_probes": n_probes,
        "essential_heads_all": _lh_list(essential_all),
        "essential_heads_any": _lh_list(essential_any),
        "total_degraded": len(degrade_counts),
        "total_candidates": total_candidates,
        "break_counts": {
            f"L{layer}:H{head}": count
            for (layer, head), count in sorted(break_counts.items())
        },
        "degrade_counts": {
            f"L{layer}:H{head}": count
            for (layer, head), count in sorted(degrade_counts.items())
        },
    }

    save_json(RESULTS_DIR / "phase2-head-ablation.json", summary)

    # Save full results for detailed analysis
    save_json(RESULTS_DIR / "phase2-head-ablation-full.json", {
        "per_probe": {
            pid: {
                "essential": _lh_list(pdata["essential"]),
                "degraded": _lh_list(pdata["degraded"]),
                "baseline": pdata["baseline"],
                "baseline_lambda_count": pdata["baseline_lambda_count"],
            }
            for pid, pdata in essential_per_probe.items()
        },
    })

    return essential_all, essential_any


# ──────────────────────────── Phase 3: Comparison ─────────────────────


def phase3_comparison(info, critical_layers, essential_all):
    """Compare Pythia results with Qwen and Phi-4."""

    banner("PHASE 3: Cross-Architecture Comparison")

    comparison = {
        "models": {
            "qwen3-4b": {
                "architecture": "Qwen2ForCausalLM",
                "params": "4.0B",
                "training": "General web + instruction tuning",
                "n_layers": 36,
                "n_heads": 32,
                "critical_layers": [0, 1, 4, 7, 24, 26, 30, 33],
                "critical_pct": 22.2,
                "essential_heads": [[1, 0], [24, 0], [24, 2]],
                "n_essential": 3,
                "essential_pct": 0.26,
                "topology": "sparse/localized",
            },
            "phi4-mini": {
                "architecture": "Phi3ForCausalLM",
                "params": "3.8B",
                "training": "5T tokens, reasoning-dense synthetic",
                "n_layers": 32,
                "n_heads": 24,
                "critical_layers": [0, 3, 5, 30],
                "critical_pct": 12.5,
                "essential_heads": [],
                "n_essential": 0,
                "essential_pct": 0.0,
                "topology": "distributed/redundant",
            },
            "pythia-2.8b": {
                "architecture": "GPTNeoXForCausalLM",
                "params": "2.8B",
                "training": "The Pile (300B tokens), base model",
                "n_layers": info.n_layers,
                "n_heads": info.n_heads,
                "critical_layers": critical_layers,
                "critical_pct": round(
                    100 * len(critical_layers) / info.n_layers, 1
                ),
                "essential_heads": _lh_list(essential_all),
                "n_essential": len(essential_all),
                "essential_pct": round(
                    100
                    * len(essential_all)
                    / (info.n_layers * info.n_heads),
                    2,
                ),
                "topology": (
                    "sparse/localized"
                    if len(essential_all) <= 5
                    else "intermediate"
                    if len(essential_all) <= 20
                    else "distributed"
                ),
            },
        },
        "localization_gradient": {
            "hypothesis": (
                "localization is inversely proportional to "
                "reasoning training density"
            ),
            "qwen_result": "3 essential (general training)",
            "phi4_result": "0 essential (reasoning-dense)",
            "pythia_prediction": "<=3 essential (minimal training)",
            "pythia_result": f"{len(essential_all)} essential",
            "confirmed": len(essential_all) <= 3,
        },
    }

    # Print comparison table
    hdr = f"  {'':>20} {'Qwen3-4B':>12} {'Phi-4':>12} {'Pythia':>12}"
    print(hdr)
    print(f"  {'-' * 56}")
    for prop in ["n_layers", "n_heads", "n_essential", "essential_pct"]:
        row = f"  {prop:>20}"
        for mdl in ["qwen3-4b", "phi4-mini", "pythia-2.8b"]:
            val = comparison["models"][mdl].get(prop, "?")
            row += f"  {val:>10}"
        print(row)

    topology = comparison["models"]["pythia-2.8b"]["topology"]
    confirmed = comparison["localization_gradient"]["confirmed"]
    print(f"\n  Topology: {topology}")
    verdict = "CONFIRMED" if confirmed else "FALSIFIED"
    print(f"  Gradient hypothesis: {verdict}")

    save_json(RESULTS_DIR / "comparison.json", comparison)


# ──────────────────────────── Main ────────────────────────────────────


def main():
    start = time.time()
    ts = datetime.now(UTC).isoformat()
    banner(f"PYTHIA-2.8B REPLICATION -- {ts}")

    model, tokenizer, info = load()

    # Phase 0: Can Pythia compile lambda at all?
    _baseline_results, success_rate = phase0_baseline(
        model, tokenizer
    )

    if success_rate == 0:
        banner("NEGATIVE RESULT: Pythia cannot compile lambda")
        print("  The gate prompt does not activate lambda compilation")
        print("  in this base model. This is itself a finding about")
        print("  the localization gradient.")
        print()
        print("  Possible next steps:")
        print("  - Try Pythia-6.9B or Pythia-12B (more capacity)")
        print("  - Try more exemplars (5-shot instead of 2-shot)")
        print("  - Try a different gate for base models")

        save_json(RESULTS_DIR / "summary.json", {
            "model": MODEL_NAME,
            "finding": "cannot_compile_lambda",
            "success_rate": success_rate,
            "elapsed_s": time.time() - start,
            "timestamp": datetime.now(UTC).isoformat(),
            "interpretation": (
                "Base model Pythia-2.8B does not compile lambda "
                "via the 2-shot gate prompt."
            ),
        })
        return

    if success_rate < 0.6:
        print(f"\n  WARNING: Low success rate ({success_rate:.0%}).")
        print("  Proceeding with caution -- results may be noisy.")

    # Phase 1: Find critical layers
    critical_layers = phase1_layer_ablation(model, tokenizer, info)

    if not critical_layers:
        print("  WARNING: No critical layers found!")
        print("  All layers are individually redundant.")
        critical_layers = list(range(info.n_layers))

    # Phase 2: Find essential heads
    essential_all, essential_any = phase2_head_ablation(
        model, tokenizer, info, critical_layers
    )

    # Phase 3: Cross-architecture comparison
    phase3_comparison(info, critical_layers, essential_all)

    # Final summary
    elapsed = time.time() - start
    summary = {
        "model": MODEL_NAME,
        "architecture": "GPTNeoXForCausalLM",
        "n_layers": info.n_layers,
        "n_heads": info.n_heads,
        "head_dim": info.head_dim,
        "hidden_size": info.hidden_size,
        "baseline_success_rate": success_rate,
        "critical_layers": critical_layers,
        "essential_heads_all": _lh_list(essential_all),
        "essential_heads_any": _lh_list(essential_any),
        "elapsed_s": elapsed,
        "timestamp": datetime.now(UTC).isoformat(),
        "localization_gradient_confirmed": len(essential_all) <= 3,
    }
    save_json(RESULTS_DIR / "summary.json", summary)

    n_ess = len(essential_all)
    verdict = "CONFIRMED" if n_ess <= 3 else "FALSIFIED"
    banner(f"COMPLETE -- {elapsed:.0f}s")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Baseline: {success_rate:.0%}")
    print(f"  Critical layers: {critical_layers}")
    print(f"  Essential heads (all probes): {essential_all}")
    print(f"  Essential heads (any probe): {essential_any}")
    print(f"  Gradient hypothesis: {verdict}")
    print(f"  Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
