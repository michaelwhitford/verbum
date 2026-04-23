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
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

# Information-theoretic constants (must match train.py)
E_IRREDUCIBLE = 1.69
PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI

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
        (model, step, meta) where meta is the full checkpoint metadata
    """
    from verbum.v6.model import VSMLMV6

    meta_path = path / "meta.json"
    weights_path = path / "weights.safetensors"

    if not meta_path.exists():
        print(f"  WARNING: no meta.json in {path}, using defaults")
        meta = {}
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

    return model, step, meta


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
# φ-Compression Analysis (forward_instrumented)
# ══════════════════════════════════════════════════════════════════════


def analyze_phi_compression(model, tokenizer, n_samples=5):
    """Run forward_instrumented on sample texts and extract φ-compression metrics.

    Returns dict with per-pass compression ratios, phi deviations, and aggregates.
    """
    samples = [
        "The cat sat on the mat and looked out the window at the birds.",
        "In 1969, Apollo 11 landed on the moon, marking a giant leap for mankind.",
        "Every student who passed the exam received a certificate of achievement.",
        "λx. λy. apply(x, y) → result",
        "The quick brown fox jumps over the lazy dog near the river bank.",
    ]

    pass_names = ["L0_asc", "L1_asc", "L2_apex", "L1_desc", "L0_desc"]
    all_ratios = {p: [] for p in pass_names}
    all_h_in = {p: [] for p in pass_names}
    all_h_out = {p: [] for p in pass_names}
    all_phi_dev = {p: [] for p in pass_names}
    all_losses = []

    for text in samples[:n_samples]:
        ids = mx.array(tokenizer.encode(text)).reshape(1, -1)
        if ids.shape[1] > model.max_len:
            ids = ids[:, -model.max_len:]
        targets = mx.concatenate([ids[:, 1:], mx.zeros((1, 1), dtype=mx.int32)], axis=1)

        _, loss, metrics = model.forward_instrumented(ids, targets)
        mx.eval(loss)
        if loss is not None:
            all_losses.append(loss.item())

        for p in pass_names:
            cr_key = f"{p}_compression_ratio"
            pd_key = f"{p}_phi_deviation"
            hi_key = f"{p}_h_in"
            ho_key = f"{p}_h_out"
            if cr_key in metrics:
                all_ratios[p].append(metrics[cr_key])
            if pd_key in metrics:
                all_phi_dev[p].append(metrics[pd_key])
            if hi_key in metrics:
                all_h_in[p].append(metrics[hi_key])
            if ho_key in metrics:
                all_h_out[p].append(metrics[ho_key])

    # Aggregate
    result = {"pass_metrics": {}, "samples": n_samples}
    for p in pass_names:
        if all_ratios[p]:
            mean_cr = sum(all_ratios[p]) / len(all_ratios[p])
            mean_pd = sum(all_phi_dev[p]) / len(all_phi_dev[p])
            mean_hi = sum(all_h_in[p]) / len(all_h_in[p])
            mean_ho = sum(all_h_out[p]) / len(all_h_out[p])
            result["pass_metrics"][p] = {
                "compression_ratio": mean_cr,
                "phi_deviation": mean_pd,
                "h_in": mean_hi,
                "h_out": mean_ho,
            }

    if all_losses:
        mean_loss = sum(all_losses) / len(all_losses)
        log_v = float(np.log(model.vocab_size))
        result["mean_loss"] = mean_loss
        result["relational_loss"] = (mean_loss - E_IRREDUCIBLE) / (log_v - E_IRREDUCIBLE)
        result["excess_ppl"] = float(np.exp(max(mean_loss - E_IRREDUCIBLE, 0)))

    if result["pass_metrics"]:
        all_cr = [m["compression_ratio"] for m in result["pass_metrics"].values()]
        all_pd = [m["phi_deviation"] for m in result["pass_metrics"].values()]
        result["mean_compression_ratio"] = sum(all_cr) / len(all_cr)
        result["mean_phi_deviation"] = sum(all_pd) / len(all_pd)
        result["inv_phi"] = INV_PHI

    return result


# ══════════════════════════════════════════════════════════════════════
# Display
# ══════════════════════════════════════════════════════════════════════


def print_summary(results, step, model, meta=None, phi_analysis=None):
    print("\n" + "=" * 70)
    print(f"  v6 Probe Summary — step {step:,}")
    print("=" * 70)

    # ── Checkpoint metadata ───────────────────────────────────
    if meta:
        train_loss = meta.get("train_loss")
        eval_loss = meta.get("eval_loss")
        total_flips = meta.get("total_flips")
        flip_target = meta.get("flip_target_pct")
        flip_thresh = meta.get("flip_threshold")
        grad_norm = meta.get("grad_norm")

        loss_str = f"train={train_loss:.4f}" if train_loss else ""
        if eval_loss:
            loss_str += f"  eval={eval_loss:.4f}"

        # Relational metrics (from checkpoint meta or computed)
        r_loss = meta.get("relational_loss")
        xppl = meta.get("excess_ppl")
        ppl = meta.get("ppl")
        if r_loss is not None:
            loss_str += f"  r={r_loss:.3f}  xppl={xppl:.1f}  ppl={ppl:.1f}"
        elif train_loss:
            log_v = float(np.log(model.vocab_size))
            r = (train_loss - E_IRREDUCIBLE) / (log_v - E_IRREDUCIBLE)
            xp = float(np.exp(max(train_loss - E_IRREDUCIBLE, 0)))
            pp = float(np.exp(train_loss))
            loss_str += f"  r={r:.3f}  xppl={xp:.1f}  ppl={pp:.1f}"

        if loss_str:
            print(f"\n  Loss: {loss_str}")

        if total_flips is not None:
            pct = total_flips / 35_258_368 * 100
            print(f"  Flips: {total_flips:,} ({pct:.2f}% of ternary weights)")
        if flip_target is not None:
            print(f"  Adaptive: target={flip_target:.4f}  threshold={flip_thresh:.1f}")
        if grad_norm is not None:
            print(f"  Grad norm: {grad_norm:.2f}")

    # ── φ-Compression analysis ────────────────────────────────
    if phi_analysis and phi_analysis.get("pass_metrics"):
        pm = phi_analysis["pass_metrics"]
        mean_cr = phi_analysis.get("mean_compression_ratio", 0)
        mean_pd = phi_analysis.get("mean_phi_deviation", 0)

        print(f"\n  φ-Compression Analysis (1/φ = {INV_PHI:.4f}):")
        print(f"  {'Pass':12s} {'h_in':>8} {'h_out':>8} {'ratio':>8} {'φ-dev':>8}")
        print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for pname in ["L0_asc", "L1_asc", "L2_apex", "L1_desc", "L0_desc"]:
            if pname in pm:
                m = pm[pname]
                cr = m["compression_ratio"]
                # Mark if close to 1/φ
                marker = " ←φ" if m["phi_deviation"] < 0.05 else ""
                print(
                    f"  {pname:12s} {m['h_in']:>8.3f} {m['h_out']:>8.3f} "
                    f"{cr:>8.4f} {m['phi_deviation']:>8.4f}{marker}"
                )
        print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        print(f"  {'mean':12s} {'':>8} {'':>8} {mean_cr:>8.4f} {mean_pd:>8.4f}")

        if phi_analysis.get("relational_loss") is not None:
            print(f"\n  Instrumented: r={phi_analysis['relational_loss']:.3f}  "
                  f"xppl={phi_analysis['excess_ppl']:.1f}")

    # ── Probe results by category ─────────────────────────────
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

    # ── Ternary stats ─────────────────────────────────────────
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

        print(f"  {'Group':15s} {'#':>4} {'sparsity':>9} {'gamma':>8} {'accum_mean':>11} {'accum_max':>10}")
        print(f"  {'─'*15} {'─'*4} {'─'*9} {'─'*8} {'─'*11} {'─'*10}")
        for grp, sl in group_stats.items():
            if not sl:
                continue
            n = len(sl)
            sp = sum(s["sparsity"] for s in sl) / n
            gm = sum(s["gamma_mean"] for s in sl) / n
            am = sum(s.get("accum_mean", 0) for s in sl) / n
            ax = max(s.get("accum_max", 0) for s in sl)
            print(f"  {grp:15s} {n:>4} {sp:>9.3f} {gm:>8.4f} {am:>11.2f} {ax:>10.1f}")

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

    model, step, meta = load_checkpoint(args.checkpoint)
    config = meta.get("config", {})
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

    # φ-compression analysis via forward_instrumented
    print(f"\n  Running φ-compression analysis...")
    phi_analysis = analyze_phi_compression(model, tokenizer)

    print_summary(results, step, model, meta=meta, phi_analysis=phi_analysis)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"vsm_probe_step_{step:06d}_v6_mlx.json"
    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "architecture": "vsm-lm-v6-mlx",
        "step": step,
        "config": config,
        "total_flips": meta.get("total_flips"),
        "flip_target_pct": meta.get("flip_target_pct"),
        "flip_threshold": meta.get("flip_threshold"),
        "grad_norm": meta.get("grad_norm"),
        "train_loss": meta.get("train_loss"),
        "eval_loss": meta.get("eval_loss"),
        "relational_loss": meta.get("relational_loss"),
        "excess_ppl": meta.get("excess_ppl"),
        "ppl": meta.get("ppl"),
        "phi_compression": phi_analysis,
        "n_probes": len(results),
        "n_lambda": sum(1 for r in results if r["has_lambda"]),
        "results": results,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
