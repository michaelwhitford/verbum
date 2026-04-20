#!/usr/bin/env python3
"""Progressive stripping — find the minimum circuit that compiles lambda.

Systematically removes model components and tests compilation:

  Level 0: Baseline (full model)
  Level 1: Zero ALL 36 FFN blocks (attention-only)
  Level 2: Zero attention in 28 non-critical layers
  Level 3: Zero BOTH FFN + attention in non-critical layers
  Level 4: Critical layers only, but only 3 essential heads
  Level 5: Only 3 heads, no FFN anywhere
  Level 6: Only L24:H0 alone

Each level: run 5 compile probes, report P(lambda) and quality.

Usage:
    uv run python scripts/run_strip_test.py

Outputs to results/strip-test/
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import structlog

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)

RESULTS_DIR = Path("results/strip-test")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# From session 001-002 findings
CRITICAL_LAYERS = [0, 1, 4, 7, 24, 26, 30, 33]
ESSENTIAL_HEADS = [(1, 0), (24, 0), (24, 2)]  # (layer, head)


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  Saved: {path}")


# ──────────────────────────── Hooking infrastructure ──────────────────


def _zero_output_hook(module, args, output):
    """Replace module output with zeros (same shape)."""
    if isinstance(output, tuple):
        zeroed = tuple(
            t.zeros_like(t) if hasattr(t, "zeros_like") else t
            for t in output
        )
        return zeroed
    return output.zeros_like(output) if hasattr(output, "zeros_like") else output


def _make_zero_hook():
    """Hook that replaces output with zeros (same shape).

    Works for both plain tensor (MLP) and tuple (attention) outputs.
    Uses torch.zeros_like — never modifies in place.
    """
    import torch

    def hook(module, args, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]), *output[1:])
        return torch.zeros_like(output)
    return hook


def _make_head_mask_hook(keep_heads, head_dim, n_heads):
    """Zero all attention heads EXCEPT those in keep_heads."""
    def hook(module, args, output):
        patched = output[0].clone()
        for head_idx in range(n_heads):
            if head_idx not in keep_heads:
                start = head_idx * head_dim
                end = start + head_dim
                patched[:, :, start:end] = 0.0
        return (patched, *output[1:])
    return hook


def run_probes(model, tokenizer, probes, max_new_tokens=50):
    """Generate for each probe and check for lambda."""
    from verbum.instrument import LAMBDA_INDICATORS, _detect_lambda, _generate

    results = []
    for rp in probes:
        gen = _generate(model, tokenizer, rp.full_prompt, max_new_tokens)
        has_lambda = _detect_lambda(gen)
        lcount = sum(gen.count(s) for s in LAMBDA_INDICATORS)
        results.append({
            "probe_id": rp.probe_id,
            "generation": gen,
            "has_lambda": has_lambda,
            "lambda_count": lcount,
        })
    return results


def summarize(results, level_name):
    """Print and return summary for a stripping level."""
    n_lambda = sum(1 for r in results if r["has_lambda"])
    total = len(results)
    rate = n_lambda / total if total > 0 else 0.0

    status = "PASS" if rate >= 0.8 else "PARTIAL" if rate > 0 else "FAIL"
    print(f"  {status:7s} {level_name}: {n_lambda}/{total} ({rate:.0%})")
    for r in results:
        tag = "Y" if r["has_lambda"] else "X"
        print(f"    {tag} [{r['probe_id']:12s}] {r['generation'][:70]}")

    return {
        "level": level_name,
        "status": status,
        "success_rate": rate,
        "n_lambda": n_lambda,
        "total": total,
        "results": results,
    }


# ──────────────────────────── Stripping levels ────────────────────────


def level0_baseline(model, tokenizer, probes):
    """Full model, no modifications."""
    results = run_probes(model, tokenizer, probes)
    return summarize(results, "L0: Baseline (full model)")


def level1_no_ffn(model, tokenizer, probes):
    """Zero ALL FFN blocks. Attention-only model."""
    from verbum.instrument import _get_layers

    layers = _get_layers(model)
    hooks = []
    for layer in layers:
        h = layer.mlp.register_forward_hook(_make_zero_hook())
        hooks.append(h)

    try:
        results = run_probes(model, tokenizer, probes)
    finally:
        for h in hooks:
            h.remove()

    return summarize(results, "L1: Zero ALL FFN (attention-only)")


def level2_critical_attn_only(model, tokenizer, probes, info):
    """Zero attention in non-critical layers. Keep FFN everywhere."""
    from verbum.instrument import _get_layers, _get_self_attn

    layers = _get_layers(model)
    hooks = []
    for layer_idx in range(info.n_layers):
        if layer_idx not in CRITICAL_LAYERS:
            attn = _get_self_attn(layers[layer_idx])
            h = attn.register_forward_hook(_make_zero_hook())
            hooks.append(h)

    try:
        results = run_probes(model, tokenizer, probes)
    finally:
        for h in hooks:
            h.remove()

    return summarize(
        results,
        "L2: Zero non-critical attention (FFN everywhere)"
    )


def level3_critical_only(model, tokenizer, probes, info):
    """Zero BOTH FFN + attention in non-critical layers.
    Pure residual pass-through for 28 layers."""
    from verbum.instrument import _get_layers, _get_self_attn

    layers = _get_layers(model)
    hooks = []
    for layer_idx in range(info.n_layers):
        if layer_idx not in CRITICAL_LAYERS:
            attn = _get_self_attn(layers[layer_idx])
            h1 = attn.register_forward_hook(_make_zero_hook())
            h2 = layers[layer_idx].mlp.register_forward_hook(
                _make_zero_hook()
            )
            hooks.extend([h1, h2])

    try:
        results = run_probes(model, tokenizer, probes)
    finally:
        for h in hooks:
            h.remove()

    return summarize(
        results,
        "L3: Zero non-critical FFN+attn (residual pass-through)"
    )


def level4_essential_heads_with_ffn(model, tokenizer, probes, info):
    """Critical layers: only 3 essential heads. Keep FFN in
    critical layers. Zero everything in non-critical."""
    from verbum.instrument import _get_layers, _get_self_attn

    layers = _get_layers(model)
    hooks = []

    # Build per-layer keep sets
    keep_per_layer = {}
    for layer_idx, head_idx in ESSENTIAL_HEADS:
        keep_per_layer.setdefault(layer_idx, set()).add(head_idx)

    for layer_idx in range(info.n_layers):
        if layer_idx not in CRITICAL_LAYERS:
            # Non-critical: zero both
            attn = _get_self_attn(layers[layer_idx])
            h1 = attn.register_forward_hook(_make_zero_hook())
            h2 = layers[layer_idx].mlp.register_forward_hook(
                _make_zero_hook()
            )
            hooks.extend([h1, h2])
        else:
            # Critical: mask heads
            keep = keep_per_layer.get(layer_idx, set())
            if len(keep) < info.n_heads:
                attn = _get_self_attn(layers[layer_idx])
                h = attn.register_forward_hook(
                    _make_head_mask_hook(
                        keep, info.head_dim, info.n_heads
                    )
                )
                hooks.append(h)

    try:
        results = run_probes(model, tokenizer, probes)
    finally:
        for h in hooks:
            h.remove()

    return summarize(
        results,
        "L4: 3 essential heads + critical FFN"
    )


def level5_essential_heads_no_ffn(model, tokenizer, probes, info):
    """Only 3 essential heads. No FFN anywhere."""
    from verbum.instrument import _get_layers, _get_self_attn

    layers = _get_layers(model)
    hooks = []

    keep_per_layer = {}
    for layer_idx, head_idx in ESSENTIAL_HEADS:
        keep_per_layer.setdefault(layer_idx, set()).add(head_idx)

    for layer_idx in range(info.n_layers):
        # Zero FFN everywhere
        h_ffn = layers[layer_idx].mlp.register_forward_hook(
            _make_zero_hook()
        )
        hooks.append(h_ffn)

        if layer_idx not in CRITICAL_LAYERS:
            # Non-critical: also zero attention
            attn = _get_self_attn(layers[layer_idx])
            h_attn = attn.register_forward_hook(_make_zero_hook())
            hooks.append(h_attn)
        else:
            # Critical: mask to essential heads only
            keep = keep_per_layer.get(layer_idx, set())
            if len(keep) < info.n_heads:
                attn = _get_self_attn(layers[layer_idx])
                h_attn = attn.register_forward_hook(
                    _make_head_mask_hook(
                        keep, info.head_dim, info.n_heads
                    )
                )
                hooks.append(h_attn)

    try:
        results = run_probes(model, tokenizer, probes)
    finally:
        for h in hooks:
            h.remove()

    return summarize(
        results,
        "L5: 3 essential heads ONLY (no FFN)"
    )


def level6_single_head(model, tokenizer, probes, info):
    """Only L24:H0 — the universal compositor. Everything else zeroed."""
    from verbum.instrument import _get_layers, _get_self_attn

    layers = _get_layers(model)
    hooks = []

    for layer_idx in range(info.n_layers):
        # Zero FFN everywhere
        h_ffn = layers[layer_idx].mlp.register_forward_hook(
            _make_zero_hook()
        )
        hooks.append(h_ffn)

        if layer_idx != 24:
            # Zero attention in all layers except 24
            attn = _get_self_attn(layers[layer_idx])
            h_attn = attn.register_forward_hook(_make_zero_hook())
            hooks.append(h_attn)
        else:
            # Layer 24: keep only head 0
            attn = _get_self_attn(layers[layer_idx])
            h_attn = attn.register_forward_hook(
                _make_head_mask_hook(
                    {0}, info.head_dim, info.n_heads
                )
            )
            hooks.append(h_attn)

    try:
        results = run_probes(model, tokenizer, probes)
    finally:
        for h in hooks:
            h.remove()

    return summarize(
        results,
        "L6: L24:H0 ONLY (single head)"
    )


# ──────────────────────────── Main ────────────────────────────────────


def main():
    start = time.time()
    banner(f"PROGRESSIVE STRIPPING — {datetime.now(UTC).isoformat()}")

    from verbum.instrument import load_model
    from verbum.probes import load_probe_set, resolve_probes

    model, tokenizer, info = load_model("Qwen/Qwen3-4B")

    probe_set = load_probe_set("probes/gate-ablation.json")
    probes = resolve_probes(probe_set, Path("gates"))

    print(f"  Model: {info.name}")
    print(f"  Layers: {info.n_layers}, Heads: {info.n_heads}")
    print(f"  Critical layers: {CRITICAL_LAYERS}")
    print(f"  Essential heads: {ESSENTIAL_HEADS}")
    print(f"  Probes: {len(probes)}")

    all_levels = []

    banner("LEVEL 0: Baseline")
    all_levels.append(level0_baseline(model, tokenizer, probes))

    banner("LEVEL 1: Zero ALL FFN")
    all_levels.append(level1_no_ffn(model, tokenizer, probes))

    banner("LEVEL 2: Attention only in critical layers")
    all_levels.append(
        level2_critical_attn_only(model, tokenizer, probes, info)
    )

    banner("LEVEL 3: Critical layers only (residual pass-through)")
    all_levels.append(
        level3_critical_only(model, tokenizer, probes, info)
    )

    banner("LEVEL 4: 3 heads + critical FFN")
    all_levels.append(
        level4_essential_heads_with_ffn(
            model, tokenizer, probes, info
        )
    )

    banner("LEVEL 5: 3 heads, no FFN")
    all_levels.append(
        level5_essential_heads_no_ffn(
            model, tokenizer, probes, info
        )
    )

    banner("LEVEL 6: L24:H0 alone")
    all_levels.append(
        level6_single_head(model, tokenizer, probes, info)
    )

    # Summary table
    elapsed = time.time() - start
    banner(f"RESULTS — {elapsed:.0f}s")

    print(f"  {'Level':<45s} {'Rate':>6s}  {'Status'}")
    print(f"  {'-' * 60}")
    for level in all_levels:
        rate = f"{level['success_rate']:.0%}"
        print(f"  {level['level']:<45s} {rate:>6s}  {level['status']}")

    # Find the minimum passing level
    min_passing = None
    for level in reversed(all_levels):
        if level["status"] == "PASS":
            min_passing = level["level"]

    print(f"\n  Minimum passing: {min_passing}")

    save_json(RESULTS_DIR / "summary.json", {
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "levels": [
            {
                "level": lev["level"],
                "status": lev["status"],
                "success_rate": lev["success_rate"],
            }
            for lev in all_levels
        ],
        "minimum_passing": min_passing,
        "critical_layers": CRITICAL_LAYERS,
        "essential_heads": [
            list(h) for h in ESSENTIAL_HEADS
        ],
    })

    # Save full results
    save_json(RESULTS_DIR / "full-results.json", {
        "levels": all_levels,
    })


if __name__ == "__main__":
    main()
