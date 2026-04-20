#!/usr/bin/env python
"""Analyze attention patterns of the 3 essential compiler heads.

Records full attention matrices for compile probes and null control,
then characterizes what L1:H0, L24:H0, and L24:H2 attend to.

Usage::

    uv run python scripts/analyze_essential_heads.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GATES_DIR = PROJECT_ROOT / "gates"
RESULTS_DIR = PROJECT_ROOT / "results"

ESSENTIAL_HEADS = [
    (1, 0, "L1:H0 (gate recognizer)"),
    (24, 0, "L24:H0 (core composer)"),
    (24, 2, "L24:H2 (recursion head)"),
]

# Probes: compile examples of varying complexity + null control
PROBES = {
    "simple": "The dog runs.",
    "quant": "Every student reads a book.",
    "relcl": "The cat that sat on the mat is black.",
    "cond": "If it rains, the ground is wet.",
    "complex": "Someone believes that the earth is flat.",
}

NULL_PROMPT = "Tell me about the weather today."


def main() -> None:
    from verbum.instrument import load_model, record_attention

    # Load gate
    gate = (GATES_DIR / "compile.txt").read_text("utf-8")
    null_gate = (GATES_DIR / "null.txt").read_text("utf-8")

    print("Loading model...")
    model, tokenizer, info = load_model("Qwen/Qwen3-4B")
    print(f"Loaded: {info.n_layers}L, {info.n_heads}H, {info.head_dim}D")
    print()

    # Build prompts
    compile_prompts = {name: gate + text for name, text in PROBES.items()}
    null_prompt = null_gate + NULL_PROMPT

    # Record attention for all prompts
    all_prompts = [*list(compile_prompts.values()), null_prompt]
    print(f"Recording attention for {len(all_prompts)} prompts...")
    captures = record_attention(model, tokenizer, all_prompts)
    print()

    # Get null baseline
    null_capture = captures[null_prompt]

    # ─── Per-head analysis ────────────────────────────────────────
    for layer, head, label in ESSENTIAL_HEADS:
        print("=" * 70)
        print(f"  {label}")
        print("=" * 70)

        for name, prompt in compile_prompts.items():
            cap = captures[prompt]
            tokens = cap.token_strs
            n_tokens = cap.n_tokens

            # Attention pattern for this head: (seq_len, seq_len)
            # attn[i, j] = how much token i attends to token j
            attn = cap.patterns[layer, head]  # (seq_len, seq_len)

            print(f'\n--- {name}: "{PROBES[name]}" ({n_tokens} tokens) ---')

            # 1. What does the LAST token attend to? (generation position)
            last_attn = attn[-1]  # (seq_len,)
            top_k = 10
            top_indices = np.argsort(last_attn)[-top_k:][::-1]

            print(f"\n  Last token attends to (top {top_k}):")
            for idx in top_indices:
                tok = tokens[idx] if idx < len(tokens) else "?"
                print(f'    [{idx:3d}] {last_attn[idx]:.4f}  "{tok}"')

            # 2. Average attention across all positions (what is globally important)
            mean_received = attn.mean(axis=0)  # (seq_len,) avg attention received
            top_received = np.argsort(mean_received)[-8:][::-1]

            print("\n  Most attended-to tokens (avg across all positions):")
            for idx in top_received:
                tok = tokens[idx] if idx < len(tokens) else "?"
                print(f'    [{idx:3d}] {mean_received[idx]:.4f}  "{tok}"')

            # 3. Attention entropy (how focused vs distributed)
            # Per-row entropy, averaged
            eps = 1e-10
            row_entropy = -np.sum(attn * np.log(attn + eps), axis=1)
            mean_entropy = row_entropy.mean()
            print(
                f"\n  Attention entropy: {mean_entropy:.3f} "
                f"(max possible: {np.log(n_tokens):.3f})"
            )

        # 4. Compare with null condition
        print(f'\n--- NULL CONTROL: "{NULL_PROMPT}" ---')
        null_attn = null_capture.patterns[layer, head]
        null_tokens = null_capture.token_strs
        null_last = null_attn[-1]
        top_null = np.argsort(null_last)[-8:][::-1]
        print("\n  Last token attends to (top 8):")
        for idx in top_null:
            tok = null_tokens[idx] if idx < len(null_tokens) else "?"
            print(f'    [{idx:3d}] {null_last[idx]:.4f}  "{tok}"')

        null_entropy = -np.sum(null_attn * np.log(null_attn + 1e-10), axis=1).mean()
        print(f"\n  Attention entropy: {null_entropy:.3f}")
        print()

    # ─── Cross-head comparison ────────────────────────────────────
    print("=" * 70)
    print("  CROSS-HEAD COMPARISON: compile vs null selectivity")
    print("=" * 70)

    # For each essential head, compute selectivity (L2 distance)
    # between compile and null attention patterns
    for layer, head, label in ESSENTIAL_HEADS:
        compile_patterns = []
        for name, prompt in compile_prompts.items():
            cap = captures[prompt]
            min_seq = min(cap.n_tokens, null_capture.n_tokens)
            c = cap.patterns[layer, head, :min_seq, :min_seq]
            n = null_capture.patterns[layer, head, :min_seq, :min_seq]
            dist = np.sqrt(np.mean((c - n) ** 2))
            compile_patterns.append((name, dist))

        print(f"\n{label}:")
        for name, dist in compile_patterns:
            print(f"  {name:12s}: selectivity = {dist:.4f}")

    # ─── Token-level gate analysis for L1:H0 ─────────────────────
    print()
    print("=" * 70)
    print("  L1:H0 GATE TOKEN ANALYSIS")
    print("=" * 70)
    print()

    # The gate is the first part of the prompt. Let's see which
    # gate tokens L1:H0 focuses on.
    gate_tokens_count = len(tokenizer(gate)["input_ids"])
    print(f"Gate is {gate_tokens_count} tokens")
    print(f"Gate text: {gate!r}")
    print()

    for name, prompt in compile_prompts.items():
        cap = captures[prompt]
        attn = cap.patterns[1, 0]  # L1:H0
        tokens = cap.token_strs

        # For each INPUT token (after gate), how much does it
        # attend to gate tokens vs input tokens?
        gate_attn = attn[gate_tokens_count:, :gate_tokens_count].sum(axis=1)
        input_attn = attn[gate_tokens_count:, gate_tokens_count:].sum(axis=1)
        total = gate_attn + input_attn + 1e-10

        gate_frac = (gate_attn / total).mean()
        print(
            f"  {name:12s}: {gate_frac:.1%} attention to gate, "
            f"{1 - gate_frac:.1%} to input"
        )

    # Same for null
    null_gate_count = len(tokenizer(null_gate)["input_ids"])
    null_attn_mat = null_capture.patterns[1, 0]
    null_gate_attn = null_attn_mat[null_gate_count:, :null_gate_count].sum(axis=1)
    null_input_attn = null_attn_mat[null_gate_count:, null_gate_count:].sum(axis=1)
    null_total = null_gate_attn + null_input_attn + 1e-10
    null_gate_frac = (null_gate_attn / null_total).mean()
    print(
        f"  {'null':12s}: {null_gate_frac:.1%} attention to gate, "
        f"{1 - null_gate_frac:.1%} to input"
    )

    # ─── Save raw attention data for the 3 heads ─────────────────
    print()
    print("Saving attention data...")
    save_data = {}
    for name, prompt in compile_prompts.items():
        cap = captures[prompt]
        for layer, head, _label in ESSENTIAL_HEADS:
            key = f"{name}_L{layer}_H{head}"
            save_data[key] = cap.patterns[layer, head]

    # Null condition
    for layer, head, _label in ESSENTIAL_HEADS:
        key = f"null_L{layer}_H{head}"
        save_data[key] = null_capture.patterns[layer, head]

    out_path = RESULTS_DIR / "essential-heads-attention.npz"
    np.savez_compressed(str(out_path), **save_data)
    print(f"Saved: {out_path}")
    print(f"Keys: {sorted(save_data.keys())}")

    # Token lists for reference
    token_data = {
        name: captures[prompt].token_strs for name, prompt in compile_prompts.items()
    }
    token_data["null"] = null_capture.token_strs

    token_path = RESULTS_DIR / "essential-heads-tokens.json"
    token_path.write_text(
        json.dumps(token_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {token_path}")


if __name__ == "__main__":
    main()
