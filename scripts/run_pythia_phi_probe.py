#!/usr/bin/env python3
"""Pythia-160M φ-compression probe.

Measures whether a standard transformer exhibits φ-compression
(compression ratio → 1/φ ≈ 0.618) across its layers, using the
same entropy proxy and stratified samples as the v6 VSM-LM probe.

The v6 model measures h_in/h_out per recursive pass. For a standard
transformer there are no recursive passes — instead we measure the
compression ratio at each layer boundary:

    h(layer_i) = log(mean(var_per_feature(residual_stream)))
    ratio(i) = h(layer_i) / h(layer_{i-1})

If the φ-hypothesis holds universally (not just for VSM architectures),
we should see:
  1. Layer-level compression ratios approaching 1/φ
  2. Content-independent compression (low stratum spread)
  3. Self-similar pattern across layers

Usage:
    uv run python scripts/run_pythia_phi_probe.py
    uv run python scripts/run_pythia_phi_probe.py --verbose
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ══════════════════════════════════════════════════════════════════════
# Constants (same as v6 probe)
# ══════════════════════════════════════════════════════════════════════

PHI = (1 + np.sqrt(5)) / 2
INV_PHI = 1 / PHI  # ≈ 0.6180

RESULTS_DIR = Path("results/pythia-phi")

# Same strata as v6 probe — allows direct comparison
PHI_STRATA = {
    "prose": [
        "The cat sat on the mat and looked out the window at the birds flying south for the winter.",
        "Every student who passed the final exam received a certificate of achievement from the dean.",
        "The quick brown fox jumps over the lazy dog near the river bank on a warm summer afternoon.",
        "In a quiet village nestled between rolling hills, the old baker opened his shop at dawn.",
    ],
    "compositional": [
        "The man who the dog that the cat chased bit ran away quickly.",
        "If every student reads a book then some teacher who knows the author is happy.",
        "No politician who endorsed the candidate that lost the election won their own race.",
        "Every lawyer who represents a client that a judge dismissed the case against appealed.",
    ],
    "technical": [
        "The gradient of the loss with respect to the weights is computed via backpropagation.",
        "Attention scores are computed as the softmax of the scaled dot product of queries and keys.",
        "The learning rate schedule uses cosine annealing with linear warmup over 500 steps.",
        "Each layer applies layer normalization before the self-attention and feed-forward blocks.",
    ],
    "math": [
        "∀x ∈ ℝ: x² ≥ 0 ∧ x² = 0 ↔ x = 0",
        "λx. λy. apply(x, y) → result",
        "P(A|B) = P(B|A) × P(A) / P(B)",
        "∑_{i=1}^{n} i = n(n+1)/2",
    ],
}


# ══════════════════════════════════════════════════════════════════════
# Entropy proxy (same formula as v6)
# ══════════════════════════════════════════════════════════════════════


def activation_entropy(x: torch.Tensor) -> float:
    """Estimate entropy of activation tensor via log-variance proxy.

    Uses mean per-feature variance across batch and sequence as a
    proxy for the information content of the representation.

    Same formula as VSMLMV6._activation_entropy:
        h = log(mean(var_per_feature) + eps)

    Args:
        x: (B, L, D) activation tensor

    Returns:
        Scalar entropy estimate (higher = more information content)
    """
    # x shape: (B, L, D) — variance per feature across batch+seq
    var_per_feat = x.var(dim=(0, 1))  # (D,)
    mean_var = var_per_feat.mean()
    return float(torch.log(mean_var + 1e-10).item())


# ══════════════════════════════════════════════════════════════════════
# Layer-by-layer residual capture with hooks
# ══════════════════════════════════════════════════════════════════════


def capture_layer_entropies(
    model, tokenizer, text: str
) -> dict:
    """Capture pre- and post-layer entropy for every transformer layer.

    Hooks the residual stream at each layer boundary. For GPTNeoX (Pythia),
    each layer receives the residual stream as input and outputs the
    updated residual stream.

    Returns dict with:
        embeddings_h: entropy of embedding output (before any layer)
        layers: list of {layer, h_in, h_out, ratio, phi_dev}
        loss: cross-entropy loss on the input
    """
    from verbum.instrument import _get_layers

    layers = _get_layers(model)
    n_layers = len(layers)

    # Storage for pre/post layer activations
    pre_layer = {}   # layer_idx → entropy
    post_layer = {}  # layer_idx → entropy

    hooks = []

    def make_pre_hook(layer_idx):
        def hook_fn(module, args):
            # args[0] is the hidden_states input
            hidden = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
            pre_layer[layer_idx] = activation_entropy(hidden)
        return hook_fn

    def make_post_hook(layer_idx):
        def hook_fn(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            post_layer[layer_idx] = activation_entropy(hidden)
        return hook_fn

    try:
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_pre_hook(make_pre_hook(i)))
            hooks.append(layer.register_forward_hook(make_post_hook(i)))

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        loss = outputs.loss.item() if outputs.loss is not None else None

    finally:
        for h in hooks:
            h.remove()

    # Build per-layer results
    layer_results = []
    for i in range(n_layers):
        h_in = pre_layer.get(i, 0.0)
        h_out = post_layer.get(i, 0.0)

        if abs(h_in) > 1e-10:
            ratio = h_out / h_in
        else:
            ratio = 1.0

        layer_results.append({
            "layer": i,
            "h_in": round(h_in, 6),
            "h_out": round(h_out, 6),
            "ratio": round(ratio, 6),
            "phi_dev": round(abs(ratio - INV_PHI), 6),
            "direction": "compressing" if ratio < 1.0 else "expanding",
        })

    return {
        "layers": layer_results,
        "loss": loss,
    }


# ══════════════════════════════════════════════════════════════════════
# Stratified analysis
# ══════════════════════════════════════════════════════════════════════


def run_stratum(model, tokenizer, samples: list[str]) -> dict:
    """Run φ-compression analysis on a list of samples.

    Returns summary with per-layer averages and aggregate stats.
    """
    all_layer_data = {}  # layer_idx → {h_in: [...], h_out: [...], ratio: [...]}
    all_losses = []

    for text in samples:
        result = capture_layer_entropies(model, tokenizer, text)
        if result["loss"] is not None:
            all_losses.append(result["loss"])

        for lr in result["layers"]:
            idx = lr["layer"]
            if idx not in all_layer_data:
                all_layer_data[idx] = {"h_in": [], "h_out": [], "ratio": []}
            all_layer_data[idx]["h_in"].append(lr["h_in"])
            all_layer_data[idx]["h_out"].append(lr["h_out"])
            all_layer_data[idx]["ratio"].append(lr["ratio"])

    # Summarize per layer
    layer_summary = []
    all_ratios = []
    for idx in sorted(all_layer_data.keys()):
        d = all_layer_data[idx]
        mean_ratio = np.mean(d["ratio"])
        std_ratio = np.std(d["ratio"])
        mean_h_in = np.mean(d["h_in"])
        mean_h_out = np.mean(d["h_out"])
        phi_dev = abs(mean_ratio - INV_PHI)
        all_ratios.append(mean_ratio)

        layer_summary.append({
            "layer": idx,
            "mean_h_in": round(float(mean_h_in), 6),
            "mean_h_out": round(float(mean_h_out), 6),
            "mean_ratio": round(float(mean_ratio), 6),
            "std_ratio": round(float(std_ratio), 6),
            "phi_dev": round(float(phi_dev), 6),
            "direction": "compressing" if mean_ratio < 1.0 else "expanding",
        })

    # Aggregate
    aggregate = {}
    if all_ratios:
        aggregate = {
            "mean_ratio": round(float(np.mean(all_ratios)), 6),
            "std_ratio": round(float(np.std(all_ratios)), 6),
            "mean_phi_dev": round(float(np.mean([abs(r - INV_PHI) for r in all_ratios])), 6),
            "min_phi_dev": round(float(np.min([abs(r - INV_PHI) for r in all_ratios])), 6),
            "closest_layer": int(np.argmin([abs(r - INV_PHI) for r in all_ratios])),
            "target": INV_PHI,
        }

    loss_summary = {}
    if all_losses:
        mean_loss = np.mean(all_losses)
        loss_summary = {
            "mean_loss": round(float(mean_loss), 4),
            "ppl": round(float(np.exp(mean_loss)), 2),
        }

    return {
        "layers": layer_summary,
        "aggregate": aggregate,
        "loss": loss_summary,
    }


# ══════════════════════════════════════════════════════════════════════
# Multi-layer grouping (analogy to v6 passes)
# ══════════════════════════════════════════════════════════════════════


def compute_pass_analogy(layer_summary: list[dict], n_layers: int) -> dict:
    """Group layers into thirds and compute per-group compression.

    Pythia has 12 layers. Grouping into thirds (0-3, 4-7, 8-11)
    provides an analogy to v6's ascending/apex/descending structure.

    Also computes cumulative compression: the product of ratios
    across a group of layers, giving the total compression factor.
    """
    third = n_layers // 3
    groups = {
        "early (L0-L3)": list(range(0, third)),
        "middle (L4-L7)": list(range(third, 2 * third)),
        "late (L8-L11)": list(range(2 * third, n_layers)),
    }

    group_results = {}
    for gname, glayers in groups.items():
        ratios = [layer_summary[i]["mean_ratio"] for i in glayers if i < len(layer_summary)]
        if ratios:
            # Cumulative compression = product of ratios
            cumulative = float(np.prod(ratios))
            group_results[gname] = {
                "mean_ratio": round(float(np.mean(ratios)), 6),
                "cumulative_compression": round(cumulative, 6),
                "phi_dev": round(float(abs(np.mean(ratios) - INV_PHI)), 6),
                "layers": glayers,
            }

    # Total compression: embedding → final layer
    all_ratios = [ls["mean_ratio"] for ls in layer_summary]
    total_compression = float(np.prod(all_ratios)) if all_ratios else 1.0

    # Does the total compression approach 1/φ^n for some n?
    # If each layer independently compresses at 1/φ, total = (1/φ)^n_layers
    expected_phi_total = INV_PHI ** n_layers
    total_phi_dev = abs(total_compression - expected_phi_total)

    return {
        "groups": group_results,
        "total_compression": round(total_compression, 6),
        "expected_phi_total": round(expected_phi_total, 10),
        "total_phi_dev": round(total_phi_dev, 6),
    }


# ══════════════════════════════════════════════════════════════════════
# Consecutive-layer pair analysis
# ══════════════════════════════════════════════════════════════════════


def compute_layer_pairs(layer_summary: list[dict]) -> list[dict]:
    """Compute compression ratios for consecutive layer PAIRS.

    If individual layers don't show φ-compression, maybe pairs of
    layers (attention + FFN as a unit) do? This tests whether the
    compression unit is larger than a single transformer layer.
    """
    pairs = []
    for i in range(0, len(layer_summary) - 1, 2):
        l1 = layer_summary[i]
        l2 = layer_summary[i + 1]

        # Combined ratio = product of individual ratios
        combined = l1["mean_ratio"] * l2["mean_ratio"]
        phi_dev = abs(combined - INV_PHI)

        pairs.append({
            "layers": f"L{l1['layer']}-L{l2['layer']}",
            "ratio_1": l1["mean_ratio"],
            "ratio_2": l2["mean_ratio"],
            "combined_ratio": round(float(combined), 6),
            "phi_dev": round(float(phi_dev), 6),
        })

    return pairs


# ══════════════════════════════════════════════════════════════════════
# Display
# ══════════════════════════════════════════════════════════════════════


def print_results(
    overall: dict,
    strata: dict[str, dict],
    passes: dict,
    pairs: list[dict],
    n_layers: int,
    verbose: bool = False,
):
    print("\n" + "=" * 70)
    print(f"  Pythia-160M φ-Compression Analysis")
    print(f"  Target: 1/φ = {INV_PHI:.4f}")
    print("=" * 70)

    # ── Per-layer table ───────────────────────────────────────
    print(f"\n  Per-layer compression:")
    print(f"  {'Layer':>5} {'h_in':>8} {'h_out':>8} {'ratio':>8} {'±std':>8} {'φ-dev':>8}")
    print(f"  {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for ls in overall["layers"]:
        marker = " ←φ" if ls["phi_dev"] < 0.05 else ""
        print(
            f"  L{ls['layer']:>3} {ls['mean_h_in']:>8.3f} {ls['mean_h_out']:>8.3f} "
            f"{ls['mean_ratio']:>8.4f} {ls['std_ratio']:>8.4f} "
            f"{ls['phi_dev']:>8.4f}{marker}"
        )

    agg = overall["aggregate"]
    print(f"  {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  {'MEAN':>5} {'':>8} {'':>8} {agg['mean_ratio']:>8.4f} {agg['std_ratio']:>8.4f} {agg['mean_phi_dev']:>8.4f}")
    print(f"  Closest to φ: layer {agg['closest_layer']} (dev={agg['min_phi_dev']:.4f})")

    # ── Layer pair analysis ───────────────────────────────────
    if pairs:
        print(f"\n  Layer-pair compression (attention+FFN as unit):")
        print(f"  {'Pair':>8} {'r1':>8} {'r2':>8} {'combined':>8} {'φ-dev':>8}")
        print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        for p in pairs:
            marker = " ←φ" if p["phi_dev"] < 0.05 else ""
            print(
                f"  {p['layers']:>8} {p['ratio_1']:>8.4f} {p['ratio_2']:>8.4f} "
                f"{p['combined_ratio']:>8.4f} {p['phi_dev']:>8.4f}{marker}"
            )

    # ── Layer group analysis (v6 analogy) ─────────────────────
    if passes and "groups" in passes:
        print(f"\n  Layer groups (analogy to v6 ascending/apex/descending):")
        print(f"  {'Group':>20} {'mean_ratio':>10} {'cumulative':>10} {'φ-dev':>8}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*8}")
        for gname, gdata in passes["groups"].items():
            marker = " ←φ" if gdata["phi_dev"] < 0.05 else ""
            print(
                f"  {gname:>20} {gdata['mean_ratio']:>10.4f} "
                f"{gdata['cumulative_compression']:>10.4f} "
                f"{gdata['phi_dev']:>8.4f}{marker}"
            )
        print(f"\n  Total compression (all layers): {passes['total_compression']:.6f}")
        print(f"  Expected if each layer ≡ 1/φ:   {passes['expected_phi_total']:.10f}")

    # ── Per-stratum table ─────────────────────────────────────
    print(f"\n  Per-stratum compression:")
    print(f"  {'stratum':>15} {'mean_ratio':>10} {'φ-dev':>8} {'loss':>8} {'ppl':>8}")
    print(f"  {'─'*15} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")

    stratum_means = []
    for sname in ["prose", "compositional", "technical", "math"]:
        if sname not in strata:
            continue
        ssummary = strata[sname]
        agg = ssummary["aggregate"]
        loss = ssummary.get("loss", {})
        mr = agg["mean_ratio"]
        pd = agg["mean_phi_dev"]
        stratum_means.append(mr)
        ml = loss.get("mean_loss", 0)
        ppl = loss.get("ppl", 0)
        print(f"  {sname:>15} {mr:>10.4f} {pd:>8.4f} {ml:>8.3f} {ppl:>8.1f}")

    if len(stratum_means) >= 2:
        spread = max(stratum_means) - min(stratum_means)
        print(f"  {'─'*15} {'─'*10} {'─'*8}")
        print(f"  {'spread':>15} {spread:>10.4f}")
        if spread < 0.01:
            print(f"  ✓ Content-independent compression — universal pattern.")
        elif spread < 0.05:
            print(f"  → Near content-independent. Low spread.")
        else:
            print(f"  ⚠ Content-dependent compression (spread={spread:.4f}).")

    # ── Per-stratum per-layer detail (verbose) ────────────────
    if verbose:
        print(f"\n  Per-stratum per-layer detail:")
        for sname in ["prose", "compositional", "technical", "math"]:
            if sname not in strata:
                continue
            print(f"\n    {sname}:")
            for ls in strata[sname]["layers"]:
                marker = " ←φ" if ls["phi_dev"] < 0.05 else ""
                print(
                    f"      L{ls['layer']:>2} ratio={ls['mean_ratio']:.4f} "
                    f"φ-dev={ls['phi_dev']:.4f}{marker}"
                )

    # ── Interpretation ────────────────────────────────────────
    print(f"\n  {'─'*60}")
    mr = overall["aggregate"]["mean_ratio"]
    pd = overall["aggregate"]["mean_phi_dev"]
    closest = overall["aggregate"]["closest_layer"]
    min_dev = overall["aggregate"]["min_phi_dev"]

    if pd < 0.05:
        print(f"  ✓ Average compression ratio near 1/φ! φ may be universal.")
    elif min_dev < 0.05:
        print(f"  → Layer {closest} approaches 1/φ (dev={min_dev:.4f}).")
        print(f"    But average is off (dev={pd:.4f}). φ may be layer-specific.")
    elif mr > 0.95 and mr < 1.05:
        print(f"  ≈ Near-identity transformation (ratio ≈ {mr:.3f}).")
        print(f"    Residual connections dominate — layers add, don't compress.")
    elif mr > 1.0:
        print(f"  ↑ Expanding (ratio > 1). Information grows through layers.")
    else:
        print(f"  ↓ Compressing at {mr:.3f}, but not near φ (dev={pd:.4f}).")

    # Compare to v6
    print(f"\n  Comparison to v6 (step 9000):")
    print(f"    v6 L1_asc:    ratio ≈ 0.566, φ-dev ≈ 0.052  (closest pass)")
    print(f"    Pythia mean:  ratio ≈ {mr:.3f}, φ-dev ≈ {pd:.3f}")
    print(f"    Pythia best:  L{closest} ratio, φ-dev ≈ {min_dev:.3f}")

    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pythia-160M φ-compression probe")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    start = time.time()

    print("\n" + "=" * 60)
    print("  Loading Pythia-160M-deduped...")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "EleutherAI/pythia-160m-deduped"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Full precision for accurate entropy
    )
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"  Device: {device}")

    from verbum.instrument import _get_layers
    n_layers = len(_get_layers(model))
    print(f"  Layers: {n_layers}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Run stratified analysis ───────────────────────────────
    print(f"\n  Running φ-compression analysis on {sum(len(v) for v in PHI_STRATA.values())} samples...")

    # Overall (all samples)
    all_samples = []
    for samples in PHI_STRATA.values():
        all_samples.extend(samples)

    overall = run_stratum(model, tokenizer, all_samples)

    # Per-stratum
    strata_results = {}
    for sname, samples in PHI_STRATA.items():
        print(f"    Stratum: {sname} ({len(samples)} samples)...")
        strata_results[sname] = run_stratum(model, tokenizer, samples)

    # Layer group analysis
    passes = compute_pass_analogy(overall["layers"], n_layers)

    # Layer pair analysis
    pairs = compute_layer_pairs(overall["layers"])

    # ── Display ───────────────────────────────────────────────
    print_results(overall, strata_results, passes, pairs, n_layers, verbose=args.verbose)

    # ── Save ──────────────────────────────────────────────────
    elapsed = time.time() - start
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "pythia_160m_phi_compression.json"

    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model_name,
        "n_layers": n_layers,
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": device,
        "elapsed_s": round(elapsed, 2),
        "phi_target": INV_PHI,
        "overall": overall,
        "strata": strata_results,
        "layer_groups": passes,
        "layer_pairs": pairs,
    }

    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {out_path}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
