#!/usr/bin/env python3
"""Probe the BOS composition register — what does L24:H0 read?

Phase 1 of the extraction investigation. The 3 essential heads in
Qwen3-4B read from BOS (position 0) which accumulates information
across all 36 layers. This script answers:

1. How many dimensions separate compile vs null at L24?
2. At which layer does compile/null separation emerge?
3. What does L24:H0's Q vector select from BOS?
4. Is the signal low-dimensional (extractable) or high-dimensional?

Uses v0-behavioral.json probes: 12 compile + 8 null = 20 contrasts.

Usage:
    uv run python scripts/run_bos_probe.py

Outputs to results/bos-probe/
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

RESULTS_DIR = Path("results/bos-probe")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  Saved: {path}")


# ──────────────────────────── Phase 1: Capture ────────────────────────


def capture_all_bos(model, tokenizer, info):
    """Capture BOS residuals for all compile + null probes."""
    from verbum.instrument import capture_bos_residuals
    from verbum.probes import load_probe_set, resolve_probes

    banner("PHASE 1: Capture BOS residuals")

    probe_set = load_probe_set("probes/v0-behavioral.json")
    resolved = resolve_probes(probe_set, Path("gates"))

    # Filter to compile and null only
    probes = [
        rp for rp in resolved if rp.category in ("compile", "null")
    ]
    n_compile = sum(1 for p in probes if p.category == "compile")
    n_null = sum(1 for p in probes if p.category == "null")
    print(f"  Probes: {len(probes)} ({n_compile} compile, {n_null} null)")

    all_residuals = []  # (n_probes, n_layers, hidden_size)
    labels = []  # 1 = compile, 0 = null
    probe_ids = []
    prompts = []

    for rp in probes:
        bos = capture_bos_residuals(model, tokenizer, rp.full_prompt, info)
        # bos is list of tensors, one per layer, each (hidden_size,)
        stacked = np.stack([b.cpu().float().numpy() for b in bos])
        all_residuals.append(stacked)
        labels.append(1 if rp.category == "compile" else 0)
        probe_ids.append(rp.probe_id)
        prompts.append(rp.prompt)
        print(f"    {rp.category:8s} {rp.probe_id:20s} {rp.prompt[:40]}")

    residuals = np.stack(all_residuals)  # (n_probes, n_layers, hidden)
    labels_arr = np.array(labels)

    print(f"\n  Residuals shape: {residuals.shape}")
    print(f"  Labels: {labels_arr.sum()} compile, {(1 - labels_arr).sum()} null")

    np.savez_compressed(
        str(RESULTS_DIR / "bos-residuals.npz"),
        residuals=residuals,
        labels=labels_arr,
    )
    save_json(RESULTS_DIR / "probe-manifest.json", {
        "probe_ids": probe_ids,
        "labels": labels,
        "prompts": prompts,
        "categories": [
            "compile" if lab else "null" for lab in labels
        ],
    })

    return residuals, labels_arr, probe_ids


# ──────────────────────────── Phase 2: PCA ────────────────────────────


def pca_analysis(residuals, labels):
    """PCA on BOS residuals per layer — where does separation emerge?"""

    banner("PHASE 2: PCA analysis — compile vs null separation")

    _n_probes, n_layers, _hidden = residuals.shape
    compile_mask = labels == 1
    null_mask = labels == 0

    layer_metrics = []

    for layer_idx in range(n_layers):
        bos_at_layer = residuals[:, layer_idx, :]  # (n_probes, hidden)

        # Center
        mean = bos_at_layer.mean(axis=0)
        centered = bos_at_layer - mean

        # SVD for PCA
        _, singular_vals, vt = np.linalg.svd(centered, full_matrices=False)
        explained = singular_vals**2 / (singular_vals**2).sum()
        cumulative = np.cumsum(explained)

        # Project onto top components
        top_k = min(10, len(singular_vals))
        projected = centered @ vt[:top_k].T  # (n_probes, top_k)

        # Separation metric: distance between compile and null centroids
        # in PCA space, normalized by pooled std
        compile_proj = projected[compile_mask]
        null_proj = projected[null_mask]
        centroid_dist = np.linalg.norm(
            compile_proj.mean(axis=0) - null_proj.mean(axis=0)
        )

        # Cohen's d on PC1 (univariate effect size)
        c_pc1 = compile_proj[:, 0]
        n_pc1 = null_proj[:, 0]
        pooled_std = np.sqrt(
            (c_pc1.std() ** 2 + n_pc1.std() ** 2) / 2
        )
        cohens_d = (
            abs(c_pc1.mean() - n_pc1.mean()) / pooled_std
            if pooled_std > 1e-8
            else 0.0
        )

        # Dims for 90% / 95% / 99% variance
        dims_90 = int(np.searchsorted(cumulative, 0.90)) + 1
        dims_95 = int(np.searchsorted(cumulative, 0.95)) + 1
        dims_99 = int(np.searchsorted(cumulative, 0.99)) + 1

        layer_metrics.append({
            "layer": layer_idx,
            "centroid_dist": float(centroid_dist),
            "cohens_d_pc1": float(cohens_d),
            "pc1_explained": float(explained[0]),
            "top5_explained": float(cumulative[4]) if len(cumulative) > 4 else 1.0,
            "dims_90pct": dims_90,
            "dims_95pct": dims_95,
            "dims_99pct": dims_99,
        })

        if layer_idx in (0, 1, 4, 7, 23, 24, 26, 30, 33, 35):
            print(
                f"  L{layer_idx:2d}:  d={cohens_d:.2f}  "
                f"dist={centroid_dist:.1f}  "
                f"PC1={explained[0]:.1%}  "
                f"dims90={dims_90}  dims95={dims_95}"
            )

    # Find peak separation layer
    peak_layer = max(layer_metrics, key=lambda m: m["cohens_d_pc1"])
    print(f"\n  Peak separation: Layer {peak_layer['layer']} "
          f"(d={peak_layer['cohens_d_pc1']:.2f})")

    # Detailed analysis of L24 specifically
    l24 = layer_metrics[24]
    print("\n  L24 (compositor input):")
    print(f"    Cohen's d on PC1: {l24['cohens_d_pc1']:.2f}")
    print(f"    Centroid distance: {l24['centroid_dist']:.1f}")
    print(f"    PC1 explains: {l24['pc1_explained']:.1%}")
    print(f"    Dims for 90%: {l24['dims_90pct']}")
    print(f"    Dims for 95%: {l24['dims_95pct']}")

    save_json(RESULTS_DIR / "pca-analysis.json", {
        "per_layer": layer_metrics,
        "peak_separation_layer": peak_layer["layer"],
        "peak_cohens_d": peak_layer["cohens_d_pc1"],
        "l24_summary": l24,
    })

    return layer_metrics


# ──────────────────────────── Phase 3: Linear probe ───────────────────


def linear_probe(residuals, labels):
    """Logistic regression on BOS@each layer: compile vs null."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    banner("PHASE 3: Linear probe — layer-by-layer classification")

    n_probes, n_layers, _hidden = residuals.shape
    loo = LeaveOneOut()

    layer_accuracies = []

    for layer_idx in range(n_layers):
        bos = residuals[:, layer_idx, :]  # (n_probes, hidden)

        # Leave-one-out cross-validation (small dataset)
        correct = 0
        for train_idx, test_idx in loo.split(bos):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(bos[train_idx])
            x_test = scaler.transform(bos[test_idx])
            y_train = labels[train_idx]
            y_test = labels[test_idx]

            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(x_train, y_train)
            if clf.predict(x_test)[0] == y_test[0]:
                correct += 1

        acc = correct / n_probes
        layer_accuracies.append({
            "layer": layer_idx,
            "accuracy": acc,
            "correct": correct,
            "total": n_probes,
        })

        if layer_idx in (0, 1, 4, 7, 23, 24, 26, 30, 33, 35):
            print(f"  L{layer_idx:2d}:  {acc:.0%} ({correct}/{n_probes})")

    # Find first layer with perfect separation
    perfect_from = None
    for entry in layer_accuracies:
        if entry["accuracy"] >= 1.0 and perfect_from is None:
            perfect_from = entry["layer"]

    print(f"\n  Perfect classification from: L{perfect_from}")

    save_json(RESULTS_DIR / "linear-probe.json", {
        "per_layer": layer_accuracies,
        "perfect_from_layer": perfect_from,
    })

    return layer_accuracies


# ──────────────────────────── Phase 4: Head Q analysis ────────────────


def head_query_analysis(model, info, residuals, labels):
    """What does L24:H0's Q vector select from BOS?

    L24:H0's query projection tells us what it looks for in the
    residual stream. By projecting BOS through Q, we see the
    effective query vector — the information the compositor reads.
    """

    from verbum.instrument import _get_layers, _get_self_attn

    banner("PHASE 4: L24:H0 query analysis — what does the compositor read?")

    layers = _get_layers(model)
    attn = _get_self_attn(layers[24])

    # Extract Q projection weight for head 0
    # Qwen uses GQA: q_proj is (n_heads * head_dim, hidden_size)
    q_weight = attn.q_proj.weight.detach().cpu().float().numpy()
    head_dim = info.head_dim
    q_h0 = q_weight[:head_dim, :]  # first head's Q: (80, 2560)

    # K projection — KV heads are shared in GQA
    # Head 0 uses KV head 0 (first of 8 KV heads)
    k_weight = attn.k_proj.weight.detach().cpu().float().numpy()
    k_h0 = k_weight[:head_dim, :]  # KV head 0: (80, 2560)

    # V projection
    v_weight = attn.v_proj.weight.detach().cpu().float().numpy()
    v_h0 = v_weight[:head_dim, :]  # KV head 0: (80, 2560)

    # O projection — maps head output back to residual stream
    o_weight = attn.o_proj.weight.detach().cpu().float().numpy()
    o_h0 = o_weight[:, :head_dim]  # head 0's slice: (2560, 80)

    print(f"  Q_h0 shape: {q_h0.shape}")
    print(f"  K_h0 shape: {k_h0.shape}")
    print(f"  V_h0 shape: {v_h0.shape}")
    print(f"  O_h0 shape: {o_h0.shape}")

    # SVD of Q — what's the effective rank?
    _, sq, _ = np.linalg.svd(q_h0, full_matrices=False)
    sq_normalized = sq / sq.sum()
    cumulative_q = np.cumsum(sq_normalized)
    q_rank_90 = int(np.searchsorted(cumulative_q, 0.90)) + 1
    q_rank_95 = int(np.searchsorted(cumulative_q, 0.95)) + 1

    print(f"\n  Q effective rank (90% energy): {q_rank_90}/{head_dim}")
    print(f"  Q effective rank (95% energy): {q_rank_95}/{head_dim}")
    print(f"  Top singular value ratio: {sq[0]/sq.sum():.1%}")

    # Project BOS@L24 through Q — the effective query at each probe
    bos_l24 = residuals[:, 24, :]  # (n_probes, 2560)
    q_projected = bos_l24 @ q_h0.T  # (n_probes, 80) — query vectors

    # Same through K and V
    k_projected = bos_l24 @ k_h0.T  # (n_probes, 80)
    v_projected = bos_l24 @ v_h0.T  # (n_probes, 80)

    compile_mask = labels == 1
    null_mask = labels == 0

    # Separation in Q-space: is compile vs null distinguishable
    # AFTER projection through Q? (i.e. does Q preserve the signal?)
    q_compile = q_projected[compile_mask]
    q_null = q_projected[null_mask]

    q_centroid_dist = float(np.linalg.norm(
        q_compile.mean(axis=0) - q_null.mean(axis=0)
    ))

    # Cohen's d in Q-space (first principal component)
    q_all_centered = q_projected - q_projected.mean(axis=0)
    _, _, q_vt = np.linalg.svd(q_all_centered, full_matrices=False)
    q_pc1 = q_all_centered @ q_vt[0]
    qc_pc1 = q_pc1[compile_mask]
    qn_pc1 = q_pc1[null_mask]
    pooled = np.sqrt((qc_pc1.std()**2 + qn_pc1.std()**2) / 2)
    q_cohens_d = (
        float(abs(qc_pc1.mean() - qn_pc1.mean()) / pooled)
        if pooled > 1e-8 else 0.0
    )

    print("\n  Q-projected separation:")
    print(f"    Centroid dist: {q_centroid_dist:.2f}")
    print(f"    Cohen's d on QPC1: {q_cohens_d:.2f}")

    # Key question: does Q AMPLIFY or REDUCE the compile/null signal?
    # Compare with raw BOS separation at L24
    raw_centered = bos_l24 - bos_l24.mean(axis=0)
    _, _, raw_vt = np.linalg.svd(raw_centered, full_matrices=False)
    raw_pc1 = raw_centered @ raw_vt[0]
    rc_pc1 = raw_pc1[compile_mask]
    rn_pc1 = raw_pc1[null_mask]
    raw_pooled = np.sqrt((rc_pc1.std()**2 + rn_pc1.std()**2) / 2)
    raw_d = (
        float(abs(rc_pc1.mean() - rn_pc1.mean()) / raw_pooled)
        if raw_pooled > 1e-8 else 0.0
    )

    amplification = q_cohens_d / raw_d if raw_d > 1e-8 else 0.0
    if amplification > 1.0:
        print(f"    Q AMPLIFIES signal: {amplification:.1f}x")
    else:
        print(f"    Q REDUCES signal: {amplification:.2f}x")

    # Do the same for L1:H0 (gate recognizer)
    attn_l1 = _get_self_attn(layers[1])
    q_l1h0 = attn_l1.q_proj.weight.detach().cpu().float().numpy()[:head_dim, :]
    bos_l24 @ q_l1h0.T  # project L24 BOS through L1's Q
    # (This is conceptually wrong — L1:H0 reads BOS at L1, not L24.
    # But we can check L1's BOS too.)
    bos_l1 = residuals[:, 1, :]
    q_l1_at_l1 = bos_l1 @ q_l1h0.T
    l1_compile = q_l1_at_l1[compile_mask]
    l1_null = q_l1_at_l1[null_mask]
    l1_dist = float(np.linalg.norm(
        l1_compile.mean(axis=0) - l1_null.mean(axis=0)
    ))
    print("\n  L1:H0 Q-projected separation at L1:")
    print(f"    Centroid dist: {l1_dist:.2f}")

    # Save all weight matrices and projections
    np.savez_compressed(
        str(RESULTS_DIR / "head-weights.npz"),
        q_l24_h0=q_h0,
        k_l24_h0=k_h0,
        v_l24_h0=v_h0,
        o_l24_h0=o_h0,
        q_l1_h0=q_l1h0,
        q_singular_values=sq,
    )
    np.savez_compressed(
        str(RESULTS_DIR / "head-projections.npz"),
        q_projected=q_projected,
        k_projected=k_projected,
        v_projected=v_projected,
        q_l1_at_l1=q_l1_at_l1,
    )

    save_json(RESULTS_DIR / "head-analysis.json", {
        "l24_h0": {
            "q_shape": list(q_h0.shape),
            "q_rank_90": q_rank_90,
            "q_rank_95": q_rank_95,
            "q_top_sv_ratio": float(sq[0] / sq.sum()),
            "q_centroid_dist": q_centroid_dist,
            "q_cohens_d_pc1": q_cohens_d,
            "raw_cohens_d_pc1": raw_d,
            "q_amplification": amplification,
        },
        "l1_h0": {
            "q_centroid_dist_at_l1": l1_dist,
        },
    })

    return {
        "q_rank_90": q_rank_90,
        "q_rank_95": q_rank_95,
        "amplification": amplification,
        "q_cohens_d": q_cohens_d,
    }


# ──────────────────────────── Main ────────────────────────────────────


def main():
    import time

    start = time.time()
    banner(f"BOS REGISTER PROBE — {datetime.now(UTC).isoformat()}")

    from verbum.instrument import load_model

    model, tokenizer, info = load_model("Qwen/Qwen3-4B")

    # Phase 1: Capture
    residuals, labels, _probe_ids = capture_all_bos(
        model, tokenizer, info
    )

    # Phase 2: PCA
    layer_metrics = pca_analysis(residuals, labels)

    # Phase 3: Linear probe
    layer_accuracies = linear_probe(residuals, labels)

    # Phase 4: Head Q analysis
    head_results = head_query_analysis(model, info, residuals, labels)

    # Summary
    elapsed = time.time() - start
    l24_pca = layer_metrics[24]
    l24_acc = layer_accuracies[24]["accuracy"]

    banner(f"SUMMARY — {elapsed:.0f}s")
    print("  BOS register at L24:")
    print(f"    Linear probe accuracy: {l24_acc:.0%}")
    print(f"    Cohen's d (PC1): {l24_pca['cohens_d_pc1']:.2f}")
    print(f"    Dims for 90% variance: {l24_pca['dims_90pct']}")
    print(f"    Dims for 95% variance: {l24_pca['dims_95pct']}")
    print("  L24:H0 query analysis:")
    print(f"    Q effective rank (90%): {head_results['q_rank_90']}")
    print(f"    Q amplification: {head_results['amplification']:.1f}x")
    print(f"    Signal in Q-space (d): {head_results['q_cohens_d']:.2f}")

    # Interpretation
    dims = l24_pca["dims_90pct"]
    if dims <= 20:
        verdict = "LOW-DIMENSIONAL — extraction-friendly"
    elif dims <= 100:
        verdict = "MODERATE — targeted extraction possible"
    else:
        verdict = "HIGH-DIMENSIONAL — distillation territory"
    print(f"\n  Verdict: {verdict}")

    save_json(RESULTS_DIR / "summary.json", {
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "l24_linear_probe_acc": l24_acc,
        "l24_cohens_d": l24_pca["cohens_d_pc1"],
        "l24_dims_90": l24_pca["dims_90pct"],
        "l24_dims_95": l24_pca["dims_95pct"],
        "q_rank_90": head_results["q_rank_90"],
        "q_amplification": head_results["amplification"],
        "verdict": verdict,
    })


if __name__ == "__main__":
    main()
