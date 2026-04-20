#!/usr/bin/env python3
"""Weight decomposition of the 3 essential heads.

The function IS the weights. This script decomposes the OV and QK
circuits algebraically — no probes needed.

For each head (L1:H0, L24:H0, L24:H2):
  OV = O @ V  →  (2560, 2560) rank-80.  "What I read, what I write."
  QK = Q^T @ K → (2560, 2560) rank-80.  "What I attend to."

SVD reveals the effective rank and interpretable directions.
Projecting through embed/unembed decodes what tokens each
direction corresponds to.

GQA note: Qwen3-4B has 32 query heads sharing 8 KV heads (ratio 4).
Heads 0-3 share KV head 0. So L24:H0 and L24:H2 share K and V —
they READ the same thing but QUERY and WRITE differently.

Usage:
    uv run python scripts/run_weight_decomposition.py

Outputs to results/weight-decomposition/
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import structlog

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)

RESULTS_DIR = Path("results/weight-decomposition")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ESSENTIAL_HEADS = [
    (1, 0, "L1:H0 gate recognizer"),
    (24, 0, "L24:H0 core compositor"),
    (24, 2, "L24:H2 recursion head"),
]


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  Saved: {path}")


# ──────────────────────────── Extract weights ─────────────────────────


def extract_head_weights(model, info, layer_idx, head_idx):
    """Extract Q, K, V, O for a specific query head, handling GQA."""
    from verbum.instrument import _get_layers, _get_self_attn

    layers = _get_layers(model)
    attn = _get_self_attn(layers[layer_idx])
    hd = info.head_dim
    gqa_ratio = info.n_heads // info.n_kv_heads  # 4 for Qwen3-4B

    # Query head slice
    q_start = head_idx * hd
    q_weight = attn.q_proj.weight.detach().cpu().float().numpy()
    q_h = q_weight[q_start : q_start + hd, :]  # (80, 2560)

    # KV head index (GQA grouping)
    kv_idx = head_idx // gqa_ratio
    kv_start = kv_idx * hd

    k_weight = attn.k_proj.weight.detach().cpu().float().numpy()
    k_h = k_weight[kv_start : kv_start + hd, :]  # (80, 2560)

    v_weight = attn.v_proj.weight.detach().cpu().float().numpy()
    v_h = v_weight[kv_start : kv_start + hd, :]  # (80, 2560)

    # Output projection: slice for this query head
    o_weight = attn.o_proj.weight.detach().cpu().float().numpy()
    o_h = o_weight[:, q_start : q_start + hd]  # (2560, 80)

    return {
        "Q": q_h,  # (80, 2560)
        "K": k_h,  # (80, 2560)
        "V": v_h,  # (80, 2560)
        "O": o_h,  # (2560, 80)
        "kv_head": kv_idx,
        "gqa_ratio": gqa_ratio,
    }


# ──────────────────────────── Circuit decomposition ───────────────────


def decompose_circuits(weights, label):
    """SVD of OV and QK circuits for one head."""

    q_mat, k_mat, v_mat, o_mat = (
        weights["Q"], weights["K"], weights["V"], weights["O"]
    )

    # OV circuit: what I read (V) and write (O)
    # OV = O @ V → (2560, 2560), rank ≤ 80
    ov = o_mat @ v_mat  # (2560, 80) @ (80, 2560) = (2560, 2560)
    u_ov, s_ov, vt_ov = np.linalg.svd(ov, full_matrices=False)

    # QK circuit: what I attend to
    # Full bilinear form: x_i^T @ Q^T @ K @ x_j
    # QK = Q^T @ K → (2560, 2560), rank ≤ 80
    qk = q_mat.T @ k_mat  # (2560, 80) @ (80, 2560) = (2560, 2560)
    u_qk, s_qk, vt_qk = np.linalg.svd(qk, full_matrices=False)

    # Effective rank
    def eff_rank(svals, threshold):
        cumulative = np.cumsum(svals) / svals.sum()
        return int(np.searchsorted(cumulative, threshold)) + 1

    ov_rank_90 = eff_rank(s_ov, 0.90)
    ov_rank_95 = eff_rank(s_ov, 0.95)
    qk_rank_90 = eff_rank(s_qk, 0.90)
    qk_rank_95 = eff_rank(s_qk, 0.95)

    # Concentration: how much does the top direction dominate?
    ov_top_ratio = float(s_ov[0] / s_ov.sum())
    qk_top_ratio = float(s_qk[0] / s_qk.sum())

    print(f"  {label}:")
    print(f"    OV rank(90%): {ov_rank_90}  "
          f"rank(95%): {ov_rank_95}  "
          f"top_sv: {ov_top_ratio:.1%}")
    print(f"    QK rank(90%): {qk_rank_90}  "
          f"rank(95%): {qk_rank_95}  "
          f"top_sv: {qk_top_ratio:.1%}")

    return {
        "label": label,
        "ov": {
            "singular_values": s_ov,
            "U": u_ov,       # left singular vectors (write directions)
            "Vt": vt_ov,     # right singular vectors (read directions)
            "rank_90": ov_rank_90,
            "rank_95": ov_rank_95,
            "top_ratio": ov_top_ratio,
        },
        "qk": {
            "singular_values": s_qk,
            "U": u_qk,       # query-side directions
            "Vt": vt_qk,     # key-side directions
            "rank_90": qk_rank_90,
            "rank_95": qk_rank_95,
            "top_ratio": qk_top_ratio,
        },
    }


# ──────────────────────────── Token decoding ──────────────────────────


def decode_directions(model, decomp, n_top=5, n_tokens=15):
    """Project top singular vectors through embed/unembed.

    OV write directions (U columns) → project through lm_head
    → what output tokens does this direction promote?

    QK key directions (Vt rows) → project through embedding
    → what input tokens does this direction attend to?
    """

    # Get embedding and unembedding matrices
    embed = model.model.embed_tokens.weight.detach().cpu().float().numpy()
    # lm_head may or may not be tied
    if hasattr(model, "lm_head"):
        unembed = model.lm_head.weight.detach().cpu().float().numpy()
    else:
        unembed = embed  # tied weights

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    label = decomp["label"]
    results = {"label": label, "ov_directions": [], "qk_directions": []}

    print(f"\n  {label} — top directions:")

    # OV: write directions (what tokens does this head promote?)
    print("    OV write directions (→ output tokens):")
    for i in range(min(n_top, len(decomp["ov"]["singular_values"]))):
        sv = float(decomp["ov"]["singular_values"][i])
        write_dir = decomp["ov"]["U"][:, i]  # (2560,)

        # Project through unembedding: scores over vocab
        scores = unembed @ write_dir  # (vocab_size,)
        top_ids = np.argsort(scores)[-n_tokens:][::-1]
        bot_ids = np.argsort(scores)[:n_tokens]

        top_tokens = [
            (tokenizer.decode([tid]), float(scores[tid]))
            for tid in top_ids
        ]
        bot_tokens = [
            (tokenizer.decode([tid]), float(scores[tid]))
            for tid in bot_ids
        ]

        top_str = " ".join(f"{t[0]!r}" for t in top_tokens[:8])
        bot_str = " ".join(f"{t[0]!r}" for t in bot_tokens[:5])
        print(f"      SV{i} ({sv:.1f}): +[{top_str}]")
        print(f"              -[{bot_str}]")

        # Also: what does the READ direction look like?
        read_dir = decomp["ov"]["Vt"][i, :]  # (2560,)
        read_scores = embed @ read_dir
        read_top = np.argsort(read_scores)[-n_tokens:][::-1]
        read_tokens = [
            (tokenizer.decode([tid]), float(read_scores[tid]))
            for tid in read_top
        ]
        read_str = " ".join(f"{t[0]!r}" for t in read_tokens[:8])
        print(f"         read: [{read_str}]")

        results["ov_directions"].append({
            "sv_index": i,
            "singular_value": sv,
            "top_write_tokens": [
                {"token": t, "score": s} for t, s in top_tokens
            ],
            "bottom_write_tokens": [
                {"token": t, "score": s} for t, s in bot_tokens
            ],
            "top_read_tokens": [
                {"token": t, "score": s} for t, s in read_tokens
            ],
        })

    # QK: what does this head attend to?
    print("    QK attention directions (→ what tokens to attend to):")
    for i in range(min(n_top, len(decomp["qk"]["singular_values"]))):
        sv = float(decomp["qk"]["singular_values"][i])

        # Key direction: what tokens in the input get attended to
        key_dir = decomp["qk"]["Vt"][i, :]  # (2560,)
        key_scores = embed @ key_dir
        key_top = np.argsort(key_scores)[-n_tokens:][::-1]
        key_tokens = [
            (tokenizer.decode([tid]), float(key_scores[tid]))
            for tid in key_top
        ]

        # Query direction: what positions query for this
        query_dir = decomp["qk"]["U"][:, i]  # (2560,)
        query_scores = embed @ query_dir
        query_top = np.argsort(query_scores)[-n_tokens:][::-1]
        query_tokens = [
            (tokenizer.decode([tid]), float(query_scores[tid]))
            for tid in query_top
        ]

        k_str = " ".join(f"{t[0]!r}" for t in key_tokens[:8])
        q_str = " ".join(f"{t[0]!r}" for t in query_tokens[:8])
        print(f"      SV{i} ({sv:.1f}): keys=[{k_str}]")
        print(f"              queries=[{q_str}]")

        results["qk_directions"].append({
            "sv_index": i,
            "singular_value": sv,
            "top_key_tokens": [
                {"token": t, "score": s} for t, s in key_tokens
            ],
            "top_query_tokens": [
                {"token": t, "score": s} for t, s in query_tokens
            ],
        })

    return results


# ──────────────────────────── Cross-head comparison ───────────────────


def cross_head_analysis(decomps):
    """Compare the 3 heads: do they share directions?"""

    banner("CROSS-HEAD COMPARISON")

    labels = [d["label"] for d in decomps]
    n = len(decomps)

    # OV write-direction similarity (top-5 U vectors)
    print("  OV write direction cosine similarity (top-5):")
    for i in range(n):
        for j in range(i + 1, n):
            u_i = decomps[i]["ov"]["U"][:, :5]  # (2560, 5)
            u_j = decomps[j]["ov"]["U"][:, :5]
            # Max cosine sim between any pair of top-5 directions
            sims = np.abs(u_i.T @ u_j)  # (5, 5)
            max_sim = float(sims.max())
            mean_sim = float(sims.mean())
            print(f"    {labels[i]} vs {labels[j]}:")
            print(f"      max={max_sim:.3f}  mean={mean_sim:.3f}")

    # QK key-direction similarity
    print("\n  QK key direction cosine similarity (top-5):")
    for i in range(n):
        for j in range(i + 1, n):
            vt_i = decomps[i]["qk"]["Vt"][:5, :]  # (5, 2560)
            vt_j = decomps[j]["qk"]["Vt"][:5, :]
            sims = np.abs(vt_i @ vt_j.T)
            max_sim = float(sims.max())
            mean_sim = float(sims.mean())
            print(f"    {labels[i]} vs {labels[j]}:")
            print(f"      max={max_sim:.3f}  mean={mean_sim:.3f}")

    # Special: L24:H0 vs L24:H2 share KV — compare OV difference
    print("\n  L24:H0 vs L24:H2 (shared KV, different Q and O):")
    d0 = decomps[1]  # L24:H0
    d2 = decomps[2]  # L24:H2
    # They share V, so OV difference is purely in O
    # Read directions should be identical (same V)
    v_sim = np.abs(d0["ov"]["Vt"][:5] @ d2["ov"]["Vt"][:5].T)
    print("    OV read directions (should be similar — shared V):")
    print(f"      max={float(v_sim.max()):.3f}  "
          f"mean={float(v_sim.mean()):.3f}")

    # Write directions differ (different O)
    o_sim = np.abs(d0["ov"]["U"][:, :5].T @ d2["ov"]["U"][:, :5])
    print("    OV write directions (differ — different O):")
    print(f"      max={float(o_sim.max()):.3f}  "
          f"mean={float(o_sim.mean()):.3f}")

    return {
        "ov_write_sims": {
            f"{labels[i]}_vs_{labels[j]}": {
                "max": float(np.abs(
                    decomps[i]["ov"]["U"][:, :5].T
                    @ decomps[j]["ov"]["U"][:, :5]
                ).max()),
            }
            for i in range(n) for j in range(i + 1, n)
        },
    }


# ──────────────────────────── Main ────────────────────────────────────


def main():
    start = time.time()
    banner(f"WEIGHT DECOMPOSITION — {datetime.now(UTC).isoformat()}")

    from verbum.instrument import load_model

    model, _tokenizer, info = load_model("Qwen/Qwen3-4B")

    print(f"  Heads: {info.n_heads} query, {info.n_kv_heads} KV")
    print(f"  GQA ratio: {info.n_heads // info.n_kv_heads}")
    print(f"  Head dim: {info.head_dim}")
    print(f"  Hidden: {info.hidden_size}")

    # Extract and decompose each head
    all_decomps = []
    all_decoded = []
    all_weights = {}

    for layer_idx, head_idx, label in ESSENTIAL_HEADS:
        banner(f"HEAD: {label} (L{layer_idx}:H{head_idx})")

        weights = extract_head_weights(model, info, layer_idx, head_idx)
        decomp = decompose_circuits(weights, label)
        decoded = decode_directions(model, decomp, n_top=5)

        all_decomps.append(decomp)
        all_decoded.append(decoded)

        key = f"L{layer_idx}_H{head_idx}"
        all_weights[key] = {
            "Q": weights["Q"],
            "K": weights["K"],
            "V": weights["V"],
            "O": weights["O"],
        }

        # Save singular values
        np.savez_compressed(
            str(RESULTS_DIR / f"{key}-svd.npz"),
            ov_singular_values=decomp["ov"]["singular_values"],
            ov_U=decomp["ov"]["U"][:, :20],
            ov_Vt=decomp["ov"]["Vt"][:20, :],
            qk_singular_values=decomp["qk"]["singular_values"],
            qk_U=decomp["qk"]["U"][:, :20],
            qk_Vt=decomp["qk"]["Vt"][:20, :],
        )

    # Cross-head comparison
    cross = cross_head_analysis(all_decomps)

    # Save everything
    save_json(RESULTS_DIR / "token-directions.json", {
        "heads": all_decoded,
    })
    save_json(RESULTS_DIR / "cross-head.json", cross)

    # Summary
    elapsed = time.time() - start
    banner(f"SUMMARY — {elapsed:.0f}s")

    summary = {"timestamp": datetime.now(UTC).isoformat(), "heads": []}
    for decomp in all_decomps:
        head_summary = {
            "label": decomp["label"],
            "ov_rank_90": decomp["ov"]["rank_90"],
            "ov_rank_95": decomp["ov"]["rank_95"],
            "ov_top_sv_ratio": decomp["ov"]["top_ratio"],
            "qk_rank_90": decomp["qk"]["rank_90"],
            "qk_rank_95": decomp["qk"]["rank_95"],
            "qk_top_sv_ratio": decomp["qk"]["top_ratio"],
        }
        summary["heads"].append(head_summary)
        print(f"  {decomp['label']}:")
        print(f"    OV: rank90={decomp['ov']['rank_90']}  "
              f"top={decomp['ov']['top_ratio']:.1%}")
        print(f"    QK: rank90={decomp['qk']['rank_90']}  "
              f"top={decomp['qk']['top_ratio']:.1%}")

    save_json(RESULTS_DIR / "summary.json", summary)


if __name__ == "__main__":
    main()
