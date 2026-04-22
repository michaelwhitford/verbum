#!/usr/bin/env python3
"""v6 compile gradient probe — MLX version.

Probes a VSMLMV6 checkpoint with the compile-gradient probe set.
Runs forward on each probe and displays v6-specific metrics:
ternary statistics, generation quality, compile gate scores.

Usage:
    uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000

    # Quiet: summary only
    uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000 --quiet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

PROBES_PATH = Path("probes/compile-gradient.json")
GATES_DIR = Path("gates/")
RESULTS_DIR = Path("results/compile-gradient")


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════


def load_probes(probe_path: Path | None = None) -> list[dict]:
    path = probe_path or PROBES_PATH
    data = json.loads(path.read_text())
    return data["probes"]


def load_gate(gate_id: str) -> str:
    return (GATES_DIR / f"{gate_id}.txt").read_text()


# ══════════════════════════════════════════════════════════════════════
# Checkpoint loading
# ══════════════════════════════════════════════════════════════════════


def load_checkpoint(path: Path) -> tuple:
    """Load a VSMLMV6 checkpoint from safetensors + meta.json.

    Args:
        path: directory containing weights.safetensors + meta.json

    Returns:
        (model, step, config)
    """
    from verbum.v6.model import VSMLMV6

    meta_path = path / "meta.json"
    weights_path = path / "weights.safetensors"

    if not meta_path.exists():
        print(f"  WARNING: no meta.json in {path}, using defaults")
        config = {}
        step = 0
    else:
        meta = json.loads(meta_path.read_text())
        config = meta.get("config", {})
        step = meta.get("step", 0)

    model = VSMLMV6(
        vocab_size=config.get("vocab_size", 50277),
        d_model=config.get("d_model", 512),
        d_register=config.get("d_register", 128),
        max_len=config.get("seq_len", 4096),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 1536),
        d_ff_consolidate=config.get("d_ff_consolidate", 2048),
        window=config.get("window", 8),
        strides=tuple(config.get("strides", [1, 8, 16, 32, 64, 128, 256, 512, 1024])),
        alpha=config.get("alpha", 1.18),
    )

    if weights_path.exists():
        model.load_weights(str(weights_path))
        print(f"  Loaded weights from {weights_path}")

    return model, step, config


# ══════════════════════════════════════════════════════════════════════
# Probing
# ══════════════════════════════════════════════════════════════════════


def probe_checkpoint(model, probes, tokenizer, gate_name="compile"):
    try:
        gate_text = load_gate(gate_name)
    except FileNotFoundError:
        print(f"  WARNING: gate '{gate_name}' not found — running without gate")
        gate_text = ""

    results = []

    for probe in probes:
        probe_id = probe["id"]
        category = probe.get("category", "unknown")
        gradient = probe.get("metadata", {}).get("gradient", None)

        gate_for_probe = probe.get("gate", gate_name)
        if gate_for_probe == "null":
            full_prompt = probe["prompt"]
        else:
            full_prompt = gate_text + probe["prompt"]

        ids = mx.array(tokenizer.encode(full_prompt)).reshape(1, -1)
        if ids.shape[1] > model.max_len:
            ids = ids[:, -model.max_len:]

        t0 = time.time()
        logits, _ = model(ids)
        mx.eval(logits)
        elapsed_ms = (time.time() - t0) * 1000

        # Short generation
        gen_ids = model.generate(ids, max_new_tokens=20, temperature=0.8)
        mx.eval(gen_ids)
        gen_text = tokenizer.decode(gen_ids[0, ids.shape[1]:].tolist())
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
        })

    return results


# ══════════════════════════════════════════════════════════════════════
# Display
# ══════════════════════════════════════════════════════════════════════


def print_summary(results, step, model):
    print("\n" + "=" * 70)
    print(f"  v6 Probe Summary — step {step:,}")
    print("=" * 70)

    categories: dict[str, list] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    cat_order = ["strong_compile", "medium_compile", "weak_compile", "null", "anti_compile"]

    print(f"\n  {'Category':20s} {'N':>3} {'λ%':>6}")
    print(f"  {'─'*20} {'─'*3} {'─'*6}")

    for cat in cat_order:
        if cat not in categories:
            continue
        cat_results = categories[cat]
        n = len(cat_results)
        lambda_frac = sum(1 for r in cat_results if r["has_lambda"]) / n * 100
        print(f"  {cat:20s} {n:>3} {lambda_frac:>5.0f}%")

    # Ternary stats
    ternary_stats = model.ternary_stats()
    if ternary_stats:
        print(f"\n  Ternary statistics ({len(ternary_stats)} modules):")
        group_stats: dict[str, list] = {
            "prep": [], "stride_stack": [], "consolidate": [],
            "mod_projs": [], "s4": [], "s3": [], "meta": [],
        }
        for mod_name, stat in ternary_stats.items():
            for gk in group_stats:
                if gk in mod_name:
                    group_stats[gk].append(stat)
                    break
            else:
                group_stats.setdefault("other", []).append(stat)

        print(f"  {'Group':15s} {'#':>4} {'sparsity':>9} {'gamma':>8}")
        print(f"  {'─'*15} {'─'*4} {'─'*9} {'─'*8}")
        for grp, sl in group_stats.items():
            if not sl:
                continue
            n = len(sl)
            sp = sum(s["sparsity"] for s in sl) / n
            gm = sum(s["gamma_mean"] for s in sl) / n
            print(f"  {grp:15s} {n:>4} {sp:>9.3f} {gm:>8.4f}")

    n_total = len(results)
    n_lambda = sum(1 for r in results if r["has_lambda"])
    print(f"\n  Overall λ generation: {n_lambda}/{n_total} ({n_lambda/n_total*100:.0f}%)")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="v6 probe (MLX)")
    parser.add_argument("checkpoint", type=Path, help="Checkpoint directory")
    parser.add_argument("--probes", type=Path, default=PROBES_PATH)
    parser.add_argument("--gate", type=str, default="compile")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  VSM-LM v6 Probe (MLX)")
    print(f"{'=' * 60}")
    print(f"  Checkpoint: {args.checkpoint}")

    model, step, config = load_checkpoint(args.checkpoint)
    print(f"  Loaded v6 model at step {step:,}")
    print(model.describe())

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")

    probes = load_probes(args.probes)
    print(f"  Loaded {len(probes)} probes")

    results = probe_checkpoint(model, probes, tokenizer, gate_name=args.gate)

    if not args.quiet:
        for r in results:
            lm = "✓λ" if r["has_lambda"] else "  "
            print(f"  {lm} {r['probe_id']:20s} [{r['category']:15s}]")
            print(f"     gen: {r['generation'][:60]!r}  ({r['elapsed_ms']:.0f}ms)")

    print_summary(results, step, model)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"vsm_probe_step_{step:06d}_v6_mlx.json"
    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "architecture": "vsm-lm-v6-mlx",
        "step": step,
        "config": config,
        "n_probes": len(results),
        "n_lambda": sum(1 for r in results if r["has_lambda"]),
        "results": results,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
