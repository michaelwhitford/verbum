#!/usr/bin/env python3
"""Compression shape probe — how do transformers compress across scales?

We know binding lives in FFN L6-22 in Qwen3-4B (three stages: type→scope→role).
We DON'T know how multi-scale compression works. Is it:
  - Spatial (reducing effective positions via attention patterns)?
  - Representational (reducing dimensionality in place)?
  - Position-accumulative (migrating info to anchor positions like BOS)?

Three experiments to characterize the compression shape:

1. **Within-constituent similarity by layer** — for sentences with known
   syntactic structure, measure cosine similarity between tokens in the
   same constituent vs across boundaries. If compression is happening,
   within-constituent similarity should INCREASE at deeper layers (tokens
   being "merged" into shared constituent representations).

2. **Effective dimensionality by layer** — PCA on residual streams.
   If representational compression occurs, the effective rank should
   decrease at deeper layers (fewer dimensions explain the variance).

3. **Cross-position influence by layer** — how much does token j's
   representation at layer L depend on token i? If there's a funnel,
   influence radius should grow with depth.

Run on both Qwen3-4B and Pythia-160M to see if the compression pattern
is universal or scale-dependent.

Usage:
    uv run python scripts/run_compression_shape.py --model qwen
    uv run python scripts/run_compression_shape.py --model pythia
    uv run python scripts/run_compression_shape.py --model both
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

RESULTS_DIR = Path("results/compression-shape")


# ══════════════════════════════════════════════════════════════════════
# Stimulus sentences with known constituent structure
# ══════════════════════════════════════════════════════════════════════

# Each entry: (sentence, list_of_constituents)
# Constituents are (start_token, end_token, label) — will be adjusted
# after tokenization. For now, word-level boundaries.
#
# We use simple, unambiguous sentences where constituency is clear.

STIMULI = [
    {
        "text": "The big dog chased the small cat in the park",
        "constituents": [
            # (word_start, word_end_exclusive, label)
            (0, 3, "NP_subj"),      # The big dog
            (3, 4, "V"),            # chased
            (4, 7, "NP_obj"),       # the small cat
            (7, 9, "P"),            # in the
            (9, 11, "NP_loc"),      # the park
        ],
    },
    {
        "text": "Every student who passed the exam received a certificate from the department",
        "constituents": [
            (0, 2, "NP_subj"),      # Every student
            (2, 6, "RC"),           # who passed the exam
            (6, 7, "V"),            # received
            (7, 9, "NP_obj"),       # a certificate
            (9, 12, "PP"),          # from the department
        ],
    },
    {
        "text": "The professor told the student that the results were significant",
        "constituents": [
            (0, 2, "NP_subj"),      # The professor
            (2, 3, "V"),            # told
            (3, 5, "NP_iobj"),      # the student
            (5, 6, "COMP"),         # that
            (6, 8, "NP_emb_subj"),  # the results
            (8, 10, "VP_emb"),      # were significant
        ],
    },
    {
        "text": "A cat sat on the mat and the dog lay beside the fire",
        "constituents": [
            (0, 2, "NP_subj1"),     # A cat
            (2, 3, "V1"),           # sat
            (3, 6, "PP1"),          # on the mat
            (6, 7, "CONJ"),         # and
            (7, 9, "NP_subj2"),     # the dog
            (9, 10, "V2"),          # lay
            (10, 13, "PP2"),        # beside the fire
        ],
    },
    {
        "text": "The woman who the man saw left the building quickly",
        "constituents": [
            (0, 2, "NP_subj"),      # The woman
            (2, 6, "RC"),           # who the man saw
            (6, 7, "V"),            # left
            (7, 9, "NP_obj"),       # the building
            (9, 10, "ADV"),         # quickly
        ],
    },
    {
        "text": "Someone believes that every child deserves a good education",
        "constituents": [
            (0, 1, "NP_subj"),      # Someone
            (1, 2, "V"),            # believes
            (2, 3, "COMP"),         # that
            (3, 5, "NP_emb_subj"),  # every child
            (5, 6, "V_emb"),        # deserves
            (6, 9, "NP_emb_obj"),   # a good education
        ],
    },
]


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n", flush=True)


# ══════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════


def load_model(model_name: str) -> tuple:
    """Load model with appropriate settings for probing."""
    from verbum.instrument import load_model as _load, _get_layers

    model, tokenizer, info = _load(model_name)
    n_layers = len(_get_layers(model))
    print(f"  Loaded: {model_name}")
    print(f"  Layers: {n_layers}")
    print(f"  Hidden: {info.hidden_size}")
    print(f"  Device: {model.device}")
    return model, tokenizer, info, n_layers


# ══════════════════════════════════════════════════════════════════════
# Core: capture ALL residual streams (all positions, all layers)
# ══════════════════════════════════════════════════════════════════════


def capture_all_residuals(model, tokenizer, text: str) -> tuple[torch.Tensor, list[str]]:
    """Capture residual stream at every layer for ALL positions.

    Returns:
        residuals: (n_layers, seq_len, hidden_size) — float32, CPU
        tokens: list of token strings
    """
    from verbum.instrument import _get_layers

    layers = _get_layers(model)
    residuals = []
    hooks = []

    def make_hook(storage):
        def hook_fn(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Capture ALL positions
            storage.append(hidden[0].detach().cpu().float())
        return hook_fn

    try:
        for layer in layers:
            hooks.append(layer.register_forward_hook(make_hook(residuals)))

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)

        # Also get tokens for alignment
        token_ids = inputs["input_ids"][0].tolist()
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
    finally:
        for h in hooks:
            h.remove()

    # Stack: (n_layers, seq_len, hidden_size)
    residuals_tensor = torch.stack(residuals)
    return residuals_tensor, tokens


# ══════════════════════════════════════════════════════════════════════
# Word-to-token alignment
# ══════════════════════════════════════════════════════════════════════


def align_words_to_tokens(text: str, tokens: list[str]) -> list[list[int]]:
    """Map word indices to token indices.

    Returns: list where word_to_tokens[word_idx] = [token_idx, ...]
    """
    words = text.split()
    word_to_tokens = []
    token_idx = 0

    # Skip BOS/special tokens at start
    # Reconstruct text from tokens to find alignment
    token_texts = []
    for t in tokens:
        # Clean token text (remove special chars like Ġ, ##, etc)
        cleaned = t.replace("Ġ", " ").replace("▁", " ").replace("Ã", "").strip()
        token_texts.append(cleaned)

    # Simple greedy alignment: accumulate tokens until they match the word
    reconstructed = ""
    current_word_idx = 0
    current_word_tokens = []

    for ti, tok in enumerate(tokens):
        # Get the actual text this token contributes
        decoded = tok
        # Add to current reconstruction
        current_word_tokens.append(ti)
        reconstructed += decoded

        # Check if we've completed the current word
        # Use the tokenizer's built-in offset mapping if possible
        pass  # Will use a simpler approach below

    # Simpler approach: use tokenizer with return_offsets_mapping
    # For now, use a character-based alignment
    word_boundaries = []
    pos = 0
    for word in words:
        start = text.index(word, pos)
        end = start + len(word)
        word_boundaries.append((start, end))
        pos = end

    # Get character offsets for each token
    # Encode character by character to find token boundaries
    # Actually, let's use the tokenizer's offset mapping
    return _align_via_offset_mapping(text, tokens, words)


def _align_via_offset_mapping(text: str, tokens: list[str], words: list[str]) -> list[list[int]]:
    """Align words to tokens by reconstructing from token text."""
    # Build word char boundaries
    word_spans = []
    pos = 0
    for word in words:
        idx = text.find(word, pos)
        if idx == -1:
            idx = pos  # fallback
        word_spans.append((idx, idx + len(word)))
        pos = idx + len(word)

    # Build token char positions by accumulating decoded text
    # The tokens list comes from decode([id]) which may have leading spaces
    token_char_starts = []
    char_pos = 0

    # Re-encode to get the proper alignment
    # Simpler: just use the full text encoding and track
    full_decoded = "".join(tokens)

    # Heuristic: strip leading special tokens, then greedily match
    # This works well enough for the probes we need
    token_to_word = [None] * len(tokens)
    for ti, tok in enumerate(tokens):
        tok_clean = tok.replace("Ġ", " ").replace("▁", " ")
        # Find which word this token belongs to based on position
        # We'll just do forward matching
        pass

    # Even simpler: tokenize with offset mapping
    # Most modern tokenizers support this
    word_to_tokens = [[] for _ in words]

    # Use word_ids() from tokenizer
    # This requires fast tokenizer
    try:
        encoding = None  # Will try below
        from transformers import AutoTokenizer
        # Re-tokenize with word alignment
        # Split by words and use is_split_into_words
        encoding = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-4B"  # placeholder — we'll pass tokenizer
        )(words, is_split_into_words=True, return_tensors="pt")
    except Exception:
        pass

    # Fallback: sequential assignment based on token text
    # Accumulate token text until it matches the next word boundary
    word_idx = 0
    accumulated = ""
    target = words[0] if words else ""

    for ti in range(len(tokens)):
        tok_text = tokens[ti]
        # Skip obvious special tokens
        if tok_text in ("<|endoftext|>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"):
            continue

        # Strip leading space markers
        clean = tok_text.lstrip()
        if not clean:
            clean = tok_text

        # Does this token start a new word? (has leading space in original)
        starts_new = tok_text.startswith(" ") or tok_text.startswith("Ġ") or tok_text.startswith("▁")

        if starts_new and accumulated and word_idx < len(words):
            # Previous word is complete, move to next
            word_idx += 1
            accumulated = ""
            if word_idx < len(words):
                target = words[word_idx]

        if word_idx < len(words):
            word_to_tokens[word_idx].append(ti)
            accumulated += clean

    return word_to_tokens


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Within-constituent similarity by layer
# ══════════════════════════════════════════════════════════════════════


def compute_constituent_similarity(
    residuals: torch.Tensor,
    word_to_tokens: list[list[int]],
    constituents: list[tuple[int, int, str]],
) -> dict:
    """Compute within vs across constituent cosine similarity per layer.

    Args:
        residuals: (n_layers, seq_len, hidden_size)
        word_to_tokens: word index → token indices
        constituents: (word_start, word_end, label) list

    Returns:
        dict with per-layer within/across similarity ratios
    """
    n_layers, seq_len, hidden = residuals.shape

    # Map constituent boundaries to token indices
    constituent_token_sets = []
    for word_start, word_end, label in constituents:
        tokens_in_constituent = []
        for wi in range(word_start, min(word_end, len(word_to_tokens))):
            tokens_in_constituent.extend(word_to_tokens[wi])
        if tokens_in_constituent:
            constituent_token_sets.append((tokens_in_constituent, label))

    # For each layer, compute:
    #   within_sim = avg cosine between tokens in SAME constituent
    #   across_sim = avg cosine between tokens in DIFFERENT constituents
    layer_results = []

    for layer_idx in range(n_layers):
        hidden_states = residuals[layer_idx]  # (seq_len, hidden)
        # Normalize for cosine (clamp to avoid NaN from zero-norm vectors)
        norms = hidden_states.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed = hidden_states / norms

        within_sims = []
        across_sims = []

        # Within-constituent pairs
        for token_set, _ in constituent_token_sets:
            if len(token_set) < 2:
                continue
            for i in range(len(token_set)):
                for j in range(i + 1, len(token_set)):
                    ti, tj = token_set[i], token_set[j]
                    if ti < seq_len and tj < seq_len:
                        sim = (normed[ti] @ normed[tj]).item()
                        within_sims.append(sim)

        # Across-constituent pairs (sample to avoid quadratic blowup)
        all_pairs = []
        for ci in range(len(constituent_token_sets)):
            for cj in range(ci + 1, len(constituent_token_sets)):
                set_i = constituent_token_sets[ci][0]
                set_j = constituent_token_sets[cj][0]
                for ti in set_i[:3]:  # sample max 3 per constituent
                    for tj in set_j[:3]:
                        if ti < seq_len and tj < seq_len:
                            sim = (normed[ti] @ normed[tj]).item()
                            across_sims.append(sim)

        avg_within = np.mean(within_sims) if within_sims else 0
        avg_across = np.mean(across_sims) if across_sims else 0
        ratio = avg_within / max(avg_across, 1e-8)

        layer_results.append({
            "layer": layer_idx,
            "within_sim": round(float(avg_within), 6),
            "across_sim": round(float(avg_across), 6),
            "ratio": round(float(ratio), 4),
            "n_within_pairs": len(within_sims),
            "n_across_pairs": len(across_sims),
        })

    return {"layer_results": layer_results}


def run_constituent_similarity(model, tokenizer, info, n_layers: int, model_name: str):
    """Experiment 1: within-constituent similarity across layers."""
    banner(f"EXPERIMENT 1: Constituent Similarity ({model_name})")

    all_results = []

    for stimulus in STIMULI:
        text = stimulus["text"]
        constituents = stimulus["constituents"]

        print(f"  Sentence: \"{text}\"")

        # Capture residuals
        residuals, tokens = capture_all_residuals(model, tokenizer, text)
        print(f"    Tokens ({len(tokens)}): {tokens[:10]}...")

        # Align words to tokens
        word_to_tokens = align_words_to_tokens(text, tokens)
        print(f"    Word→token alignment: {len(word_to_tokens)} words")
        for wi, toks in enumerate(word_to_tokens[:5]):
            word = text.split()[wi] if wi < len(text.split()) else "?"
            tok_strs = [tokens[ti] for ti in toks if ti < len(tokens)]
            print(f"      word[{wi}] '{word}' → tokens {toks} = {tok_strs}")

        # Compute similarity
        result = compute_constituent_similarity(residuals, word_to_tokens, constituents)
        result["sentence"] = text
        result["n_tokens"] = len(tokens)
        all_results.append(result)

        # Print layer progression (sample every few layers)
        print(f"    Layer progression (within/across ratio):")
        step = max(1, n_layers // 10)
        for lr in result["layer_results"][::step]:
            ratio = lr["ratio"] if not np.isnan(lr["ratio"]) else 0
            bar = "█" * min(50, int(ratio * 10))
            print(f"      L{lr['layer']:2d}: within={lr['within_sim']:.4f} "
                  f"across={lr['across_sim']:.4f} ratio={ratio:.3f} {bar}")

    # Average across all sentences
    avg_by_layer = {}
    for result in all_results:
        for lr in result["layer_results"]:
            layer = lr["layer"]
            if layer not in avg_by_layer:
                avg_by_layer[layer] = {"within": [], "across": [], "ratio": []}
            avg_by_layer[layer]["within"].append(lr["within_sim"])
            avg_by_layer[layer]["across"].append(lr["across_sim"])
            avg_by_layer[layer]["ratio"].append(lr["ratio"])

    print(f"\n  AVERAGE across {len(STIMULI)} sentences:")
    print(f"  {'Layer':>5s} | {'Within':>8s} | {'Across':>8s} | {'Ratio':>7s} | {'Δ':>7s}")
    print(f"  {'-'*45}")
    prev_ratio = None
    for layer in sorted(avg_by_layer.keys()):
        w = np.mean(avg_by_layer[layer]["within"])
        a = np.mean(avg_by_layer[layer]["across"])
        r = np.mean(avg_by_layer[layer]["ratio"])
        delta = r - prev_ratio if prev_ratio is not None else 0
        direction = "↑" if delta > 0.01 else "↓" if delta < -0.01 else "→"
        if layer % max(1, n_layers // 15) == 0 or layer == n_layers - 1:
            print(f"  L{layer:3d}  | {w:8.4f} | {a:8.4f} | {r:7.3f} | {delta:+.3f} {direction}")
        prev_ratio = r

    return all_results


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Effective dimensionality by layer
# ══════════════════════════════════════════════════════════════════════


def compute_effective_dimensionality(
    residuals: torch.Tensor,
    threshold: float = 0.95,
) -> list[dict]:
    """Compute effective dimensionality via PCA at each layer.

    Effective dimensionality = number of principal components needed
    to explain `threshold` fraction of the variance.

    If compression occurs, this should DECREASE at deeper layers.
    """
    n_layers, seq_len, hidden = residuals.shape
    layer_results = []

    for layer_idx in range(n_layers):
        X = residuals[layer_idx]  # (seq_len, hidden)
        # Center
        X_centered = X - X.mean(dim=0, keepdim=True)
        # SVD (more numerically stable than covariance for PCA)
        try:
            U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            # Explained variance ratios
            var_explained = (S ** 2) / (S ** 2).sum()
            cumulative = var_explained.cumsum(dim=0)
            # Effective dim = first k where cumulative >= threshold
            eff_dim = (cumulative < threshold).sum().item() + 1
            # Also compute participation ratio (more robust)
            # PR = (Σλᵢ)² / Σλᵢ² — measures how many eigenvalues contribute
            eigenvalues = S ** 2
            PR = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()

            layer_results.append({
                "layer": layer_idx,
                "eff_dim_95": eff_dim,
                "participation_ratio": round(PR.item(), 2),
                "top1_var": round(var_explained[0].item(), 4),
                "top5_var": round(cumulative[4].item(), 4) if len(cumulative) > 4 else 1.0,
                "top10_var": round(cumulative[9].item(), 4) if len(cumulative) > 9 else 1.0,
            })
        except Exception as e:
            layer_results.append({
                "layer": layer_idx,
                "error": str(e),
            })

    return layer_results


def run_effective_dimensionality(model, tokenizer, info, n_layers: int, model_name: str):
    """Experiment 2: effective dimensionality across layers."""
    banner(f"EXPERIMENT 2: Effective Dimensionality ({model_name})")

    # Use multiple sentences for richer statistics
    all_dim_results = []

    for stimulus in STIMULI[:3]:  # use first 3 for speed
        text = stimulus["text"]
        print(f"  Sentence: \"{text}\"")

        residuals, tokens = capture_all_residuals(model, tokenizer, text)
        dim_results = compute_effective_dimensionality(residuals)
        all_dim_results.append(dim_results)

    # Average across sentences
    n_layers_actual = len(all_dim_results[0])
    print(f"\n  AVERAGE effective dimensionality across {len(all_dim_results)} sentences:")
    print(f"  {'Layer':>5s} | {'EffDim95':>8s} | {'PartRatio':>9s} | {'Top1%':>7s} | {'Top10%':>7s}")
    print(f"  {'-'*50}")

    avg_results = []
    for li in range(n_layers_actual):
        eff_dims = [r[li].get("eff_dim_95", 0) for r in all_dim_results if "error" not in r[li]]
        part_ratios = [r[li].get("participation_ratio", 0) for r in all_dim_results if "error" not in r[li]]
        top1s = [r[li].get("top1_var", 0) for r in all_dim_results if "error" not in r[li]]
        top10s = [r[li].get("top10_var", 0) for r in all_dim_results if "error" not in r[li]]

        avg = {
            "layer": li,
            "eff_dim_95": np.mean(eff_dims) if eff_dims else 0,
            "participation_ratio": np.mean(part_ratios) if part_ratios else 0,
            "top1_var": np.mean(top1s) if top1s else 0,
            "top10_var": np.mean(top10s) if top10s else 0,
        }
        avg_results.append(avg)

        if li % max(1, n_layers_actual // 15) == 0 or li == n_layers_actual - 1:
            print(f"  L{li:3d}  | {avg['eff_dim_95']:8.1f} | {avg['participation_ratio']:9.1f} | "
                  f"{avg['top1_var']:7.4f} | {avg['top10_var']:7.4f}")

    return avg_results


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Cross-position influence by layer
# ══════════════════════════════════════════════════════════════════════


def compute_influence_radius(
    model, tokenizer, text: str, target_positions: list[int] | None = None,
) -> list[dict]:
    """Measure how far each token's influence extends at each layer.

    Method: for each target position, zero out its input and measure
    the change in other positions' representations. The "influence radius"
    is how far away positions are significantly affected.

    Simpler proxy: use the attention patterns directly (we already have
    this infrastructure). The effective receptive field at layer L =
    the positions that collectively account for 90% of attention mass.
    """
    from verbum.instrument import _get_layers

    layers = _get_layers(model)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]

    # Capture attention patterns at each layer
    attn_patterns = []
    hooks = []

    def make_attn_hook(storage):
        def hook_fn(module, args, output):
            # For Qwen: output is (hidden_states, attn_weights, ...)
            # attn_weights shape: (B, n_heads, seq_len, seq_len)
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                storage.append(output[1][0].detach().cpu().float())  # (n_heads, L, L)
        return hook_fn

    # Enable attention output
    prev_attn = model.config.output_attentions
    model.config.output_attentions = True

    try:
        for layer in layers:
            attn = _get_self_attn_module(layer)
            hooks.append(attn.register_forward_hook(make_attn_hook(attn_patterns)))

        with torch.no_grad():
            model(**inputs)
    finally:
        for h in hooks:
            h.remove()
        model.config.output_attentions = prev_attn

    # Compute influence radius per layer
    # For each layer: average attention entropy and effective receptive field
    layer_results = []

    for li, attn in enumerate(attn_patterns):
        # attn: (n_heads, seq_len, seq_len)
        n_heads = attn.shape[0]

        # Average across heads
        avg_attn = attn.mean(dim=0)  # (seq_len, seq_len)

        # For each position, compute effective receptive field
        # = number of positions needed to cover 90% of attention mass
        receptive_fields = []
        for pos in range(seq_len):
            row = avg_attn[pos, :pos + 1]  # causal: only attend to past
            if row.sum() < 1e-8:
                continue
            sorted_attn, _ = row.sort(descending=True)
            cumsum = sorted_attn.cumsum(dim=0)
            eff_rf = (cumsum < 0.9).sum().item() + 1
            receptive_fields.append(eff_rf)

        # Average distance of attention (how far back does it look?)
        avg_distances = []
        for pos in range(1, seq_len):
            row = avg_attn[pos, :pos + 1]
            positions = torch.arange(pos + 1, dtype=torch.float)
            distances = (pos - positions)  # distance from current position
            avg_dist = (row * distances).sum().item()
            avg_distances.append(avg_dist)

        # Attention entropy (uniform = high entropy = broad attention)
        entropies = []
        for pos in range(seq_len):
            row = avg_attn[pos, :pos + 1]
            row = row + 1e-10
            entropy = -(row * row.log()).sum().item()
            entropies.append(entropy)

        layer_results.append({
            "layer": li,
            "avg_receptive_field": round(np.mean(receptive_fields), 2) if receptive_fields else 0,
            "avg_attention_distance": round(np.mean(avg_distances), 2) if avg_distances else 0,
            "avg_entropy": round(np.mean(entropies), 4),
            "max_receptive_field": max(receptive_fields) if receptive_fields else 0,
        })

    return layer_results


def _get_self_attn_module(layer):
    """Get the self-attention module for hook registration."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "attention"):
        return layer.attention
    if hasattr(layer, "attn"):
        return layer.attn
    raise AttributeError(f"Cannot find attention in {type(layer)}")


def run_influence_radius(model, tokenizer, info, n_layers: int, model_name: str):
    """Experiment 3: attention-based influence radius across layers."""
    banner(f"EXPERIMENT 3: Influence Radius ({model_name})")

    text = STIMULI[0]["text"]  # Use first sentence
    print(f"  Sentence: \"{text}\"")

    layer_results = compute_influence_radius(model, tokenizer, text)

    if not layer_results:
        print("  ⚠ No attention patterns captured (model may not output attention weights)")
        print("  Falling back to residual-based influence measurement...")
        return run_influence_radius_residual(model, tokenizer, text, n_layers)

    print(f"\n  {'Layer':>5s} | {'AvgRF':>6s} | {'AvgDist':>7s} | {'Entropy':>7s} | {'MaxRF':>5s}")
    print(f"  {'-'*45}")
    for lr in layer_results:
        if lr["layer"] % max(1, len(layer_results) // 15) == 0 or lr["layer"] == len(layer_results) - 1:
            print(f"  L{lr['layer']:3d}  | {lr['avg_receptive_field']:6.1f} | "
                  f"{lr['avg_attention_distance']:7.2f} | {lr['avg_entropy']:7.4f} | "
                  f"{lr['max_receptive_field']:5d}")

    return layer_results


def run_influence_radius_residual(model, tokenizer, text: str, n_layers: int):
    """Fallback: measure influence via residual stream correlation."""
    banner("EXPERIMENT 3 (fallback): Residual correlation radius")

    residuals, tokens = capture_all_residuals(model, tokenizer, text)
    n_layers_actual, seq_len, hidden = residuals.shape

    print(f"  Measuring correlation decay by distance at each layer...")

    layer_results = []
    for li in range(n_layers_actual):
        X = F.normalize(residuals[li], dim=-1)  # (seq_len, hidden)
        # Compute pairwise cosine similarity
        sim_matrix = X @ X.T  # (seq_len, seq_len)

        # Bin by distance and compute average similarity
        max_dist = min(32, seq_len)
        dist_sims = []
        for d in range(1, max_dist):
            sims = []
            for i in range(d, seq_len):
                sims.append(sim_matrix[i, i - d].item())
            dist_sims.append(np.mean(sims) if sims else 0)

        # "Influence radius" = distance at which similarity drops below 50% of d=1
        baseline = dist_sims[0] if dist_sims else 0
        threshold = baseline * 0.5
        radius = 1
        for d, s in enumerate(dist_sims, 1):
            if s >= threshold:
                radius = d
            else:
                break

        layer_results.append({
            "layer": li,
            "correlation_radius": radius,
            "sim_d1": round(dist_sims[0], 4) if dist_sims else 0,
            "sim_d4": round(dist_sims[3], 4) if len(dist_sims) > 3 else 0,
            "sim_d8": round(dist_sims[7], 4) if len(dist_sims) > 7 else 0,
            "sim_d16": round(dist_sims[15], 4) if len(dist_sims) > 15 else 0,
        })

    print(f"\n  {'Layer':>5s} | {'Radius':>6s} | {'d=1':>6s} | {'d=4':>6s} | {'d=8':>6s} | {'d=16':>6s}")
    print(f"  {'-'*50}")
    for lr in layer_results:
        if lr["layer"] % max(1, n_layers_actual // 15) == 0 or lr["layer"] == n_layers_actual - 1:
            print(f"  L{lr['layer']:3d}  | {lr['correlation_radius']:6d} | "
                  f"{lr['sim_d1']:6.4f} | {lr['sim_d4']:6.4f} | "
                  f"{lr['sim_d8']:6.4f} | {lr['sim_d16']:6.4f}")

    return layer_results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def run_all(model_name: str):
    """Run all experiments on one model."""
    start = time.time()
    banner(f"COMPRESSION SHAPE PROBE: {model_name}")

    model, tokenizer, info, n_layers = load_model(model_name)

    # Experiment 1: Within-constituent similarity
    sim_results = run_constituent_similarity(model, tokenizer, info, n_layers, model_name)

    # Experiment 2: Effective dimensionality
    dim_results = run_effective_dimensionality(model, tokenizer, info, n_layers, model_name)

    # Experiment 3: Influence radius
    inf_results = run_influence_radius(model, tokenizer, info, n_layers, model_name)

    elapsed = time.time() - start
    banner(f"DONE — {model_name} ({elapsed:.0f}s)")

    # Save results
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "hidden_size": info.hidden_size,
        "elapsed_s": elapsed,
        "constituent_similarity": sim_results,
        "effective_dimensionality": dim_results,
        "influence_radius": inf_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_name.replace("/", "_").replace("-", "_")
    out_path = RESULTS_DIR / f"{slug}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Saved: {out_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compression shape probe")
    parser.add_argument("--model", choices=["qwen", "pythia", "both"], default="qwen")
    args = parser.parse_args()

    if args.model in ("qwen", "both"):
        run_all("Qwen/Qwen3-4B")

    if args.model in ("pythia", "both"):
        run_all("EleutherAI/pythia-160m-deduped")


if __name__ == "__main__":
    main()
