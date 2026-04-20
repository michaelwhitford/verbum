#!/usr/bin/env python3
"""BOS register probe v2 — controlled for gate confound.

v1 compared compile (compile gate) vs null (null gate). The trivial
separation was gate identity, not compilation state.

v2 uses within-gate contrasts:
  1. Compile vs Decompile (same gate, different task direction)
  2. Within-compile: simple vs complex (same gate, same task)
  3. English-input vs Lambda-input (compile has English, decompile has λ)

This reveals what the BOS register actually encodes about content
and task, not just which gate prefix is present.

Usage:
    uv run python scripts/run_bos_probe_v2.py

Outputs to results/bos-probe-v2/
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import structlog

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)

log = structlog.get_logger()

RESULTS_DIR = Path("results/bos-probe-v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  Saved: {path}")


def cohens_d(group_a, group_b):
    """Cohen's d between two arrays along first PC."""
    pooled = np.sqrt(
        (group_a.std() ** 2 + group_b.std() ** 2) / 2
    )
    if pooled < 1e-8:
        return 0.0
    return float(abs(group_a.mean() - group_b.mean()) / pooled)


def pca_separation(data, mask_a, mask_b, label=""):
    """PCA separation analysis between two groups."""
    centered = data - data.mean(axis=0)
    _, svals, vt = np.linalg.svd(centered, full_matrices=False)

    explained = svals**2 / (svals**2).sum()
    cumulative = np.cumsum(explained)

    # Project onto PCs
    top_k = min(20, len(svals))
    projected = centered @ vt[:top_k].T

    # Centroid distance in full PC space
    a_proj = projected[mask_a]
    b_proj = projected[mask_b]
    dist = float(np.linalg.norm(
        a_proj.mean(axis=0) - b_proj.mean(axis=0)
    ))

    # Cohen's d on PC1
    d_pc1 = cohens_d(projected[mask_a, 0], projected[mask_b, 0])

    # Best separating PC (check first 10)
    best_d = 0.0
    best_pc = 0
    for pc_idx in range(min(10, top_k)):
        d_val = cohens_d(
            projected[mask_a, pc_idx], projected[mask_b, pc_idx]
        )
        if d_val > best_d:
            best_d = d_val
            best_pc = pc_idx

    dims_90 = int(np.searchsorted(cumulative, 0.90)) + 1
    dims_95 = int(np.searchsorted(cumulative, 0.95)) + 1

    return {
        "label": label,
        "centroid_dist": dist,
        "cohens_d_pc1": d_pc1,
        "best_separating_pc": best_pc,
        "best_cohens_d": best_d,
        "pc1_explained": float(explained[0]),
        "dims_90pct": dims_90,
        "dims_95pct": dims_95,
    }


# ──────────────────────────── Capture ─────────────────────────────────


def capture_all_bos(model, tokenizer, info):
    """Capture BOS residuals for compile + decompile probes."""
    from verbum.instrument import capture_bos_residuals
    from verbum.probes import load_probe_set, resolve_probes

    banner("CAPTURE: BOS residuals (compile + decompile)")

    probe_set = load_probe_set("probes/v0-behavioral.json")
    resolved = resolve_probes(probe_set, Path("gates"))

    # Include compile and decompile (both use same gate)
    probes = [
        rp for rp in resolved
        if rp.category in ("compile", "decompile")
    ]
    n_c = sum(1 for p in probes if p.category == "compile")
    n_d = sum(1 for p in probes if p.category == "decompile")
    print(f"  Probes: {len(probes)} ({n_c} compile, {n_d} decompile)")

    all_residuals = []
    categories = []
    probe_ids = []
    prompts = []
    complexities = []

    for rp in probes:
        bos = capture_bos_residuals(
            model, tokenizer, rp.full_prompt, info
        )
        stacked = np.stack([b.cpu().float().numpy() for b in bos])
        all_residuals.append(stacked)
        categories.append(rp.category)
        probe_ids.append(rp.probe_id)
        prompts.append(rp.prompt)

        # Rough complexity label for within-compile analysis
        complexity = rp.metadata.get("complexity", "unknown")
        if complexity == "unknown" and rp.category == "compile":
            n_words = len(rp.prompt.split())
            complexity = (
                "simple" if n_words <= 4
                else "medium" if n_words <= 8
                else "complex"
            )
        complexities.append(complexity)

        print(f"    {rp.category:12s} {complexity:8s} {rp.probe_id}")

    residuals = np.stack(all_residuals)
    print(f"\n  Shape: {residuals.shape}")

    np.savez_compressed(
        str(RESULTS_DIR / "bos-residuals.npz"),
        residuals=residuals,
    )
    save_json(RESULTS_DIR / "probe-manifest.json", {
        "probe_ids": probe_ids,
        "categories": categories,
        "prompts": prompts,
        "complexities": complexities,
    })

    return residuals, categories, complexities, probe_ids


# ──────────────────────────── Analysis ────────────────────────────────


def analyze(residuals, categories, complexities):
    """Three contrasts, per layer."""

    categories = np.array(categories)
    complexities = np.array(complexities)

    compile_mask = categories == "compile"
    decompile_mask = categories == "decompile"

    # Within-compile complexity masks
    compile_probes = np.where(compile_mask)[0]
    compile_complex = np.array(complexities)
    simple_mask = np.zeros(len(categories), dtype=bool)
    complex_mask = np.zeros(len(categories), dtype=bool)
    for idx in compile_probes:
        if compile_complex[idx] == "simple":
            simple_mask[idx] = True
        elif compile_complex[idx] in ("medium", "complex"):
            complex_mask[idx] = True

    n_simple = simple_mask.sum()
    n_complex = complex_mask.sum()

    banner("ANALYSIS: Three within-gate contrasts")
    print(f"  Contrast 1: compile({compile_mask.sum()}) vs "
          f"decompile({decompile_mask.sum()})")
    print(f"  Contrast 2: simple({n_simple}) vs "
          f"complex({n_complex}) [within compile]")

    _n_probes, n_layers, _hidden = residuals.shape

    contrast1_layers = []  # compile vs decompile
    contrast2_layers = []  # simple vs complex

    key_layers = [0, 1, 4, 7, 12, 18, 23, 24, 26, 30, 33, 35]

    for layer_idx in range(n_layers):
        bos = residuals[:, layer_idx, :]

        # Contrast 1: compile vs decompile
        c1 = pca_separation(
            bos, compile_mask, decompile_mask,
            label=f"L{layer_idx}_compile_vs_decompile",
        )
        c1["layer"] = layer_idx
        contrast1_layers.append(c1)

        # Contrast 2: simple vs complex (only if enough probes)
        if n_simple >= 2 and n_complex >= 2:
            c2 = pca_separation(
                bos, simple_mask, complex_mask,
                label=f"L{layer_idx}_simple_vs_complex",
            )
            c2["layer"] = layer_idx
            contrast2_layers.append(c2)

        if layer_idx in key_layers:
            print(
                f"  L{layer_idx:2d}:  "
                f"c_vs_d d={c1['best_cohens_d']:.2f}(PC{c1['best_separating_pc']})  "
                f"dist={c1['centroid_dist']:.1f}  "
                f"dims90={c1['dims_90pct']}"
            )
            if contrast2_layers and contrast2_layers[-1]["layer"] == layer_idx:
                c2_last = contrast2_layers[-1]
                print(
                    f"        "
                    f"s_vs_c d={c2_last['best_cohens_d']:.2f}"
                    f"(PC{c2_last['best_separating_pc']})  "
                    f"dist={c2_last['centroid_dist']:.1f}"
                )

    # L24 deep-dive
    print("\n  === L24 deep-dive (compositor input) ===")
    c1_l24 = contrast1_layers[24]
    print("  Compile vs Decompile:")
    print(f"    Best d: {c1_l24['best_cohens_d']:.2f} "
          f"on PC{c1_l24['best_separating_pc']}")
    print(f"    Centroid dist: {c1_l24['centroid_dist']:.1f}")
    print(f"    PC1 explains: {c1_l24['pc1_explained']:.1%}")
    print(f"    Dims for 90%: {c1_l24['dims_90pct']}")

    if contrast2_layers:
        c2_l24 = contrast2_layers[24]
        print("  Simple vs Complex:")
        print(f"    Best d: {c2_l24['best_cohens_d']:.2f} "
              f"on PC{c2_l24['best_separating_pc']}")
        print(f"    Centroid dist: {c2_l24['centroid_dist']:.1f}")

    save_json(RESULTS_DIR / "contrast-analysis.json", {
        "compile_vs_decompile": contrast1_layers,
        "simple_vs_complex": contrast2_layers,
    })

    return contrast1_layers, contrast2_layers


# ──────────────────────────── Head Q redux ────────────────────────────


def head_query_redux(model, info, residuals, categories):
    """Re-analyze L24:H0's Q on within-gate data."""

    from verbum.instrument import _get_layers, _get_self_attn

    banner("HEAD Q REDUX: L24:H0 on within-gate contrasts")

    layers = _get_layers(model)
    attn = _get_self_attn(layers[24])
    head_dim = info.head_dim

    q_h0 = attn.q_proj.weight.detach().cpu().float().numpy()[:head_dim, :]

    bos_l24 = residuals[:, 24, :]  # (n_probes, 2560)
    q_proj = bos_l24 @ q_h0.T  # (n_probes, 80)

    cats = np.array(categories)
    compile_mask = cats == "compile"
    decompile_mask = cats == "decompile"

    # Separation in Q-projected space
    q_compile = q_proj[compile_mask]
    q_decompile = q_proj[decompile_mask]

    dist = float(np.linalg.norm(
        q_compile.mean(axis=0) - q_decompile.mean(axis=0)
    ))

    # PCA in Q space
    q_centered = q_proj - q_proj.mean(axis=0)
    _, _, qvt = np.linalg.svd(q_centered, full_matrices=False)
    q_pc1 = q_centered @ qvt[0]
    d_val = cohens_d(q_pc1[compile_mask], q_pc1[decompile_mask])

    # Compare raw BOS vs Q-projected
    raw_centered = bos_l24 - bos_l24.mean(axis=0)
    _, _, rvt = np.linalg.svd(raw_centered, full_matrices=False)
    raw_pc1 = raw_centered @ rvt[0]
    raw_d = cohens_d(raw_pc1[compile_mask], raw_pc1[decompile_mask])

    amp = d_val / raw_d if raw_d > 1e-8 else 0.0

    print("  Compile vs Decompile in Q-space:")
    print(f"    Centroid dist: {dist:.2f}")
    print(f"    Cohen's d (QPC1): {d_val:.2f}")
    print(f"    Raw BOS d (PC1): {raw_d:.2f}")
    if amp > 1.0:
        print(f"    Q AMPLIFIES: {amp:.1f}x")
    else:
        print(f"    Q REDUCES: {amp:.2f}x")

    # What does each compile probe look like in Q-space?
    print("\n  Per-probe Q projection (PC1 value):")
    q_pc1_vals = q_centered @ qvt[0]
    for i, cat in enumerate(categories):
        print(f"    {cat:12s}  PC1={q_pc1_vals[i]:.2f}")

    save_json(RESULTS_DIR / "head-q-redux.json", {
        "compile_vs_decompile": {
            "q_centroid_dist": dist,
            "q_cohens_d_pc1": d_val,
            "raw_cohens_d_pc1": raw_d,
            "q_amplification": amp,
        },
    })

    return {"q_cohens_d": d_val, "raw_d": raw_d, "amplification": amp}


# ──────────────────────────── Main ────────────────────────────────────


def main():
    import time

    start = time.time()
    banner(f"BOS PROBE v2 — {datetime.now(UTC).isoformat()}")

    from verbum.instrument import load_model

    model, tokenizer, info = load_model("Qwen/Qwen3-4B")

    # Capture
    residuals, categories, complexities, _ids = capture_all_bos(
        model, tokenizer, info
    )

    # Controlled analysis
    c1, _c2 = analyze(residuals, categories, complexities)

    # Head Q on controlled data
    head_results = head_query_redux(
        model, info, residuals, categories
    )

    # Summary
    elapsed = time.time() - start
    c1_l24 = c1[24]

    banner(f"SUMMARY — {elapsed:.0f}s")
    print("  Within-gate BOS analysis (gate confound removed):")
    print("    Compile vs Decompile at L24:")
    print(f"      Best d: {c1_l24['best_cohens_d']:.2f}")
    print(f"      Dims 90%: {c1_l24['dims_90pct']}")
    print(f"    Q amplification: {head_results['amplification']:.1f}x")
    print(f"    Q-space d: {head_results['q_cohens_d']:.2f}")

    save_json(RESULTS_DIR / "summary.json", {
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "l24_compile_vs_decompile": {
            "best_d": c1_l24["best_cohens_d"],
            "best_pc": c1_l24["best_separating_pc"],
            "centroid_dist": c1_l24["centroid_dist"],
            "dims_90": c1_l24["dims_90pct"],
            "dims_95": c1_l24["dims_95pct"],
        },
        "head_q": {
            "q_cohens_d": head_results["q_cohens_d"],
            "raw_d": head_results["raw_d"],
            "amplification": head_results["amplification"],
        },
    })


if __name__ == "__main__":
    main()
