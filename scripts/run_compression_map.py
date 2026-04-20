#!/usr/bin/env python3
"""Compression map probe — what IS the compression mechanism in Qwen3-4B?

Builds on run_compression_shape.py findings:
  - Constituent similarity peaks at L6-9
  - Effective dimensionality collapses to ~1 at L6+
  - Attention narrows at deeper layers

Four experiments in one model-load pass:

  Q1: What is the dominant direction?
      PCA at each layer; correlate top-PC projection with word position,
      constituent depth, POS category, content vs function word.

  Q2: How do tokens converge? (attention vs FFN ablation)
      At layers 4-12: zero the FFN, zero the attention.
      Measure within-constituent similarity after each ablation.
      Tells us whether compression is attention- or FFN-mediated.

  Q3: What survives in the residual?
      Decompose each token into along-PC ("compressed") + perp ("residual").
      Probe the perp component for: token identity (embedding cosine),
      position-in-constituent, absolute sentence position.

  Q4: Does convergence track syntax or semantics?
      Garden-path and ambiguous sentences where syntactic constituency and
      semantic relatedness diverge. At L6-9: does the similarity matrix
      cluster by syntax or by meaning?

Usage:
    uv run python scripts/run_compression_map.py
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

from verbum.instrument import _get_layers, load_model as _load_model

# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════

RESULTS_DIR = Path("results/compression-map")
N_LAYERS = 36          # Qwen3-4B
HIDDEN_SIZE = 2560     # Qwen3-4B

# Layers where compression is happening (from run_compression_shape findings)
COMPRESSION_LAYERS = list(range(4, 13))   # Q2 ablation sweep
DOMINANT_PC_LAYERS = list(range(6, 11))   # Q3 residual analysis

# ══════════════════════════════════════════════════════════════════════
# Q1 Stimuli: sentences with rich linguistic annotations
# ══════════════════════════════════════════════════════════════════════
# Each word has: pos (0=noun,1=verb,2=det,3=adj,4=prep,5=other),
#                depth (0=main clause, 1=embedded, 2=doubly-embedded),
#                is_content (True for nouns/verbs/adjectives/adverbs)
#
# POS codes: 0=NOUN, 1=VERB, 2=DET, 3=ADJ, 4=PREP, 5=OTHER
POS_NOUN, POS_VERB, POS_DET, POS_ADJ, POS_PREP, POS_OTHER = 0, 1, 2, 3, 4, 5

Q1_STIMULI = [
    {
        "text": "The big dog chased the small cat in the park",
        # word:    The big dog chased the small cat in  the park
        "pos":   [2,  3,  0,  1,    2,  3,    0,  4,  2,  0],
        "depth": [0,  0,  0,  0,    0,  0,    0,  0,  0,  0],
        "is_content": [False, True, True, True, False, True, True, False, False, True],
    },
    {
        "text": "Every student who passed the exam received a certificate",
        # word:   Every student who passed the exam received a certificate
        "pos":   [2,    0,      5,  1,     2,  0,   1,        2, 0],
        "depth": [0,    0,      1,  1,     1,  1,   0,        0, 0],
        "is_content": [False, True, False, True, False, True, True, False, True],
    },
    {
        "text": "The professor told the student that the results were significant",
        # word:  The professor told the student that the results were significant
        "pos":  [2,  0,        1,   2,  0,      5,   2,  0,      1,   3],
        "depth":[0,  0,        0,   0,  0,      0,   1,  1,      1,   1],
        "is_content": [False, True, True, False, True, False, False, True, True, True],
    },
    {
        "text": "Someone believes that every child deserves a good education",
        # word:  Someone believes that every child deserves a good education
        "pos":  [0,      1,       5,   2,    0,    1,        2, 3,   0],
        "depth":[0,      0,       0,   1,    1,    1,        1, 1,   1],
        "is_content": [True, True, False, False, True, True, False, True, True],
    },
    {
        "text": "The old man carried a heavy wooden box down the hill",
        # word:  The old man carried a heavy wooden box down the hill
        "pos":  [2,  3,  0,  1,      2, 3,    3,     0,  4,   2,  0],
        "depth":[0,  0,  0,  0,      0, 0,    0,     0,  0,   0,  0],
        "is_content": [False, True, True, True, False, True, True, True, False, False, True],
    },
    {
        "text": "The woman who the man saw quickly left the building",
        # word:  The woman who the man saw quickly left the building
        "pos":  [2,  0,    5,  2,  0,  1,  5,       1,   2,  0],
        "depth":[0,  0,    1,  1,  1,  1,  0,       0,   0,  0],
        "is_content": [False, True, False, False, True, True, True, True, False, True],
    },
]

# ══════════════════════════════════════════════════════════════════════
# Q4 Stimuli: syntax vs. semantics divergence sentences
# ══════════════════════════════════════════════════════════════════════
# For each sentence: the "syntactic groups" and "semantic groups" we
# expect at L6-9 if the model tracks syntax vs. semantics respectively.

Q4_STIMULI = [
    {
        "label": "garden_path_old_man",
        "text": "The old man the boats",
        # True parse: [The old] [man] [the boats]
        #   "man" = VERB; "old" = NOUN (subject)
        # Naive/semantic parse: [The old man] [the boats]
        #   "old man" grouped as NP
        "syntactic_groups": [[0, 1], [2], [3, 4]],   # Det+Adj | VERB | Det+NOUN
        "semantic_groups": [[0, 1, 2], [3, 4]],       # NP_old_man | NP_boats
        "note": "garden path: 'old' is noun, 'man' is verb",
    },
    {
        "label": "garden_path_horse",
        "text": "The horse raced past the barn fell",
        # True parse: [[The horse [raced past the barn]] fell]
        #   reduced relative: "raced past the barn" modifies "horse"
        # Naive: "horse raced" as NP+V; "fell" is confusing
        "syntactic_groups": [[0, 1], [2, 3, 4, 5], [6]],   # NP | RC | V
        "semantic_groups": [[0, 1, 2], [3, 4, 5], [6]],     # horse+raced | past barn | fell
        "note": "reduced relative clause; 'fell' is main verb",
    },
    {
        "label": "ambiguous_time_flies",
        "text": "Time flies like an arrow",
        # Parse 1: [Time] [flies] [like an arrow]  — time passes quickly
        # Parse 2: [Time flies] [like] [an arrow]  — insects prefer arrows
        # Parse 3: [Time] [flies like] [an arrow]  — time moves like arrow
        "syntactic_groups": [[0], [1], [2, 3, 4]],       # NP V PP (most common)
        "semantic_groups": [[0, 1], [2, 3, 4]],           # agent | manner
        "note": "triple ambiguity: time/flies/like all have multiple parses",
    },
    {
        "label": "control_unambiguous",
        "text": "The tall man sailed the boats",
        # Unambiguous: [The tall man] [sailed] [the boats]
        "syntactic_groups": [[0, 1, 2], [3], [4, 5]],
        "semantic_groups": [[0, 1, 2], [3], [4, 5]],      # same for unambiguous
        "note": "unambiguous control: Det+Adj+Noun | Verb | Det+Noun",
    },
    {
        "label": "control_simple",
        "text": "The cat sat on the mat",
        # Unambiguous: [The cat] [sat] [on the mat]
        "syntactic_groups": [[0, 1], [2], [3, 4, 5]],
        "semantic_groups": [[0, 1], [2], [3, 4, 5]],
        "note": "simple control sentence",
    },
]

# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════

def banner(text: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {text}")
    print("═" * 70, flush=True)


def sub_banner(text: str) -> None:
    print(f"\n  ── {text} ──")


def capture_all_residuals(
    model: object,
    tokenizer: object,
    text: str,
) -> tuple[torch.Tensor, list[str]]:
    """Capture residual stream at every layer, every position.

    Returns:
        residuals: (n_layers, seq_len, hidden_size) — float32, CPU
        tokens: list of decoded token strings
    """
    layers = _get_layers(model)
    residuals: list[torch.Tensor] = []
    hooks = []

    def make_hook(storage: list) -> object:
        def hook_fn(module: object, args: object, output: object) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            storage.append(hidden[0].detach().cpu().float())
        return hook_fn

    try:
        for layer in layers:
            hooks.append(layer.register_forward_hook(make_hook(residuals)))

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)

        token_ids = inputs["input_ids"][0].tolist()
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
    finally:
        for h in hooks:
            h.remove()

    return torch.stack(residuals), tokens  # (n_layers, seq_len, hidden_size)


def align_words_to_tokens(
    text: str, tokens: list[str], words: list[str]
) -> list[list[int]]:
    """Map word indices to token indices (greedy, space-marker aware).

    Returns word_to_tokens[wi] = [ti, ...].
    """
    word_to_tokens: list[list[int]] = [[] for _ in words]
    word_idx = 0
    accumulated = ""

    for ti, tok in enumerate(tokens):
        # Skip obvious special tokens
        if tok in ("<|endoftext|>", "<s>", "</s>", "<|im_start|>", "<|im_end|>",
                   "<|end_of_text|>"):
            continue

        # Detect word boundary: token starts with a space/marker
        starts_new = (
            tok.startswith(" ")
            or tok.startswith("Ġ")
            or tok.startswith("▁")
        )

        if starts_new and accumulated and word_idx < len(words):
            # Commit current word, advance
            word_idx += 1
            accumulated = ""

        if word_idx < len(words):
            word_to_tokens[word_idx].append(ti)
            clean = tok.lstrip("Ġ▁ ")
            accumulated += clean

    return word_to_tokens


def cosine_sim_matrix(X: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity. X: (n, d) → (n, n)."""
    norms = X.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    Xn = X / norms
    return Xn @ Xn.T


def within_across_similarity(
    hidden: torch.Tensor,
    groups: list[list[int]],
) -> tuple[float, float]:
    """Compute mean within-group and across-group cosine similarity.

    hidden: (seq_len, hidden_size)
    groups: list of token-index lists

    Returns (within_sim, across_sim).
    """
    seq_len = hidden.shape[0]
    norms = hidden.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    Xn = hidden / norms

    within_sims: list[float] = []
    across_sims: list[float] = []

    valid_groups = [g for g in groups if len(g) >= 1]

    for gi, grp in enumerate(valid_groups):
        # Within-group pairs
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                ti, tj = grp[i], grp[j]
                if ti < seq_len and tj < seq_len:
                    within_sims.append((Xn[ti] @ Xn[tj]).item())

        # Across-group pairs (sample to limit quadratic blowup)
        for gj in range(gi + 1, len(valid_groups)):
            other = valid_groups[gj]
            for ti in grp[:3]:
                for tj in other[:3]:
                    if ti < seq_len and tj < seq_len:
                        across_sims.append((Xn[ti] @ Xn[tj]).item())

    w = float(np.mean(within_sims)) if within_sims else 0.0
    a = float(np.mean(across_sims)) if across_sims else 0.0
    return w, a


# ══════════════════════════════════════════════════════════════════════
# Q1: Dominant Direction
# ══════════════════════════════════════════════════════════════════════

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Robust Pearson r (returns 0 if std is zero)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return 0.0
    sx, sy = x.std(), y.std()
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def run_q1_dominant_direction(
    model: object,
    tokenizer: object,
    info: object,
) -> dict:
    """Q1: What is the dominant direction at each layer?

    For each sentence:
      - Capture residuals at all layers
      - At each layer, compute top PC via SVD
      - Project each token onto that PC → scalar
      - Correlate scalar with: word_position, constituent_depth, POS, content_word

    Returns per-layer correlation statistics averaged across sentences.
    """
    banner("Q1: DOMINANT DIRECTION — what does the top PC track?")

    # Accumulators: layer → list of (projections, word_pos, depth, pos, is_content)
    layer_data: dict[int, dict[str, list]] = {
        li: {"proj": [], "word_pos": [], "depth": [], "pos_cat": [], "content": []}
        for li in range(N_LAYERS)
    }

    per_sentence_results = []

    for si, stim in enumerate(Q1_STIMULI):
        text = stim["text"]
        words = text.split()
        pos_labels = stim["pos"]
        depths = stim["depth"]
        is_content = stim["is_content"]

        print(f"\n  Sentence {si+1}: \"{text}\"")

        residuals, tokens = capture_all_residuals(model, tokenizer, text)
        n_layers, seq_len, _ = residuals.shape

        w2t = align_words_to_tokens(text, tokens, words)

        # Build per-token label arrays (one entry per token, using parent word's label)
        token_word_pos: list[float] = []
        token_depth: list[float] = []
        token_pos_cat: list[float] = []
        token_content: list[float] = []

        for ti in range(seq_len):
            # Find which word owns this token
            owner_wi = None
            for wi, toks in enumerate(w2t):
                if ti in toks:
                    owner_wi = wi
                    break
            if owner_wi is None or owner_wi >= len(words):
                # Special token or unaligned — skip
                token_word_pos.append(float("nan"))
                token_depth.append(float("nan"))
                token_pos_cat.append(float("nan"))
                token_content.append(float("nan"))
            else:
                token_word_pos.append(float(owner_wi))
                token_depth.append(float(depths[owner_wi]) if owner_wi < len(depths) else float("nan"))
                token_pos_cat.append(float(pos_labels[owner_wi]) if owner_wi < len(pos_labels) else float("nan"))
                token_content.append(float(is_content[owner_wi]) if owner_wi < len(is_content) else float("nan"))

        wp_arr = np.array(token_word_pos)
        dep_arr = np.array(token_depth)
        pos_arr = np.array(token_pos_cat)
        cont_arr = np.array(token_content)

        valid_mask = ~(np.isnan(wp_arr) | np.isnan(dep_arr))
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 2:
            print("    ⚠ too few valid tokens, skipping")
            continue

        sentence_layer_corr = []

        for li in range(n_layers):
            X = residuals[li]  # (seq_len, hidden_size)
            X_valid = X[valid_indices]  # select valid (non-special) tokens

            # Compute top PC via SVD
            X_centered = X_valid - X_valid.mean(dim=0, keepdim=True)
            try:
                _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
                top_pc = Vh[0]  # (hidden_size,) — the dominant direction
            except Exception:
                sentence_layer_corr.append({
                    "layer": li,
                    "r_word_pos": 0.0, "r_depth": 0.0,
                    "r_pos_cat": 0.0, "r_content": 0.0,
                })
                continue

            # Project ALL tokens onto top PC
            projections = (X @ top_pc).numpy()  # (seq_len,)
            proj_valid = projections[valid_indices]

            wp_v  = wp_arr[valid_indices]
            dep_v = dep_arr[valid_indices]
            pos_v = pos_arr[valid_indices]
            cont_v = cont_arr[valid_indices]

            r_wp   = pearson_r(proj_valid, wp_v)
            r_dep  = pearson_r(proj_valid, dep_v)
            r_pos  = pearson_r(proj_valid, pos_v)
            r_cont = pearson_r(proj_valid, cont_v)

            sentence_layer_corr.append({
                "layer": li,
                "r_word_pos": round(r_wp, 4),
                "r_depth":    round(r_dep, 4),
                "r_pos_cat":  round(r_pos, 4),
                "r_content":  round(r_cont, 4),
            })

            # Accumulate for cross-sentence average
            layer_data[li]["proj"].extend(proj_valid.tolist())
            layer_data[li]["word_pos"].extend(wp_v.tolist())
            layer_data[li]["depth"].extend(dep_v.tolist())
            layer_data[li]["pos_cat"].extend(pos_v.tolist())
            layer_data[li]["content"].extend(cont_v.tolist())

        per_sentence_results.append({
            "sentence": text,
            "n_tokens": seq_len,
            "n_words": len(words),
            "layer_correlations": sentence_layer_corr,
        })

    # Compute cross-sentence correlations
    print("\n  CROSS-SENTENCE CORRELATIONS (all tokens pooled per layer):")
    print(f"\n  {'Layer':>5s} | {'r(word_pos)':>11s} | {'r(depth)':>8s} | "
          f"{'r(pos_cat)':>10s} | {'r(content)':>10s} | dominant_feature")
    print(f"  {'-' * 70}")

    agg_layer_corr = []
    for li in range(N_LAYERS):
        d = layer_data[li]
        if len(d["proj"]) < 4:
            agg_layer_corr.append({
                "layer": li,
                "r_word_pos": 0.0, "r_depth": 0.0,
                "r_pos_cat": 0.0, "r_content": 0.0,
                "dominant_feature": "insufficient_data",
            })
            continue

        proj = np.array(d["proj"])
        r_wp   = pearson_r(proj, np.array(d["word_pos"]))
        r_dep  = pearson_r(proj, np.array(d["depth"]))
        r_pos  = pearson_r(proj, np.array(d["pos_cat"]))
        r_cont = pearson_r(proj, np.array(d["content"]))

        feature_map = {
            "word_pos": abs(r_wp),
            "depth":    abs(r_dep),
            "pos_cat":  abs(r_pos),
            "content":  abs(r_cont),
        }
        dominant = max(feature_map, key=lambda k: feature_map[k])

        agg_layer_corr.append({
            "layer": li,
            "r_word_pos": round(r_wp, 4),
            "r_depth":    round(r_dep, 4),
            "r_pos_cat":  round(r_pos, 4),
            "r_content":  round(r_cont, 4),
            "dominant_feature": dominant,
        })

        if li % 3 == 0 or li == N_LAYERS - 1:
            bar_wp   = "█" * min(15, int(abs(r_wp) * 20))
            bar_dep  = "█" * min(15, int(abs(r_dep) * 20))
            bar_pos  = "█" * min(15, int(abs(r_pos) * 20))
            bar_cont = "█" * min(15, int(abs(r_cont) * 20))
            print(f"  L{li:3d}  | {r_wp:+.4f} {bar_wp:<15s} | {r_dep:+.4f} {bar_dep:<8s} | "
                  f"{r_pos:+.4f} {bar_pos:<10s} | {r_cont:+.4f} {bar_cont:<10s} | [{dominant}]")

    return {
        "per_sentence": per_sentence_results,
        "aggregate_layer_correlations": agg_layer_corr,
    }


# ══════════════════════════════════════════════════════════════════════
# Q2: Convergence Mechanism — Attention vs. FFN ablation
# ══════════════════════════════════════════════════════════════════════

def capture_residuals_with_ablation(
    model: object,
    tokenizer: object,
    text: str,
    ablate_layer: int,
    ablate_target: str,   # "none" | "ffn" | "attn"
) -> torch.Tensor:
    """Forward pass with optional ablation at one layer.

    ablate_target:
        "none"  — normal forward pass
        "ffn"   — zero the MLP output at ablate_layer
        "attn"  — zero the self-attention output at ablate_layer

    Returns residuals: (n_layers, seq_len, hidden_size), float32 CPU.
    """
    layers = _get_layers(model)
    residuals: list[torch.Tensor] = []
    hooks = []

    # Residual capture hooks (all layers)
    def make_residual_hook(storage: list) -> object:
        def hook_fn(module: object, args: object, output: object) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            storage.append(hidden[0].detach().cpu().float())
        return hook_fn

    # Ablation hooks
    def zero_output_hook(module: object, args: object, output: object) -> object:
        if isinstance(output, tuple):
            zeros = torch.zeros_like(output[0])
            return (zeros,) + output[1:]
        return torch.zeros_like(output)

    try:
        for layer in layers:
            hooks.append(layer.register_forward_hook(make_residual_hook(residuals)))

        if ablate_target == "ffn":
            target_layer = layers[ablate_layer]
            if hasattr(target_layer, "mlp"):
                hooks.append(
                    target_layer.mlp.register_forward_hook(zero_output_hook)
                )
            else:
                print(f"    ⚠ No .mlp on layer {ablate_layer}, skipping ffn ablation")

        elif ablate_target == "attn":
            target_layer = layers[ablate_layer]
            if hasattr(target_layer, "self_attn"):
                hooks.append(
                    target_layer.self_attn.register_forward_hook(zero_output_hook)
                )
            else:
                print(f"    ⚠ No .self_attn on layer {ablate_layer}, skipping attn ablation")

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)

    finally:
        for h in hooks:
            h.remove()

    return torch.stack(residuals)  # (n_layers, seq_len, hidden_size)


def run_q2_convergence_mechanism(
    model: object,
    tokenizer: object,
    info: object,
) -> dict:
    """Q2: Is compression mediated by attention or by the FFN?

    For each layer in COMPRESSION_LAYERS:
      - Normal forward: baseline within-constituent similarity
      - FFN zeroed:     similarity without FFN contribution
      - Attn zeroed:    similarity without attention contribution

    The larger drop indicates the more critical component.
    """
    banner("Q2: CONVERGENCE MECHANISM — attention vs. FFN ablation")

    # Use 3 unambiguous stimuli for reliability
    q2_texts = [s["text"] for s in Q1_STIMULI[:3]]
    q2_constituents = [
        # word-level constituent groups (will be mapped to tokens)
        [[0, 1, 2], [3], [4, 5, 6], [7, 8, 9]],   # big dog | chased | small cat | in park
        [[0, 1], [2, 3, 4, 5], [6], [7, 8]],        # student | RC | received | certificate
        [[0, 1], [2], [3, 4], [5], [6, 7], [8, 9]], # professor | told | student | that | results | were sig
    ]

    all_layer_results = []

    print(f"\n  Sweeping layers {COMPRESSION_LAYERS[0]}–{COMPRESSION_LAYERS[-1]}")
    print(f"  {'Layer':>5s} | {'normal_w':>8s} | {'no_ffn_w':>8s} | "
          f"{'no_attn_w':>9s} | {'Δ_ffn':>7s} | {'Δ_attn':>7s} | critical")
    print(f"  {'-' * 70}")

    for ablate_layer in COMPRESSION_LAYERS:
        normal_ws, no_ffn_ws, no_attn_ws = [], [], []

        for si, (text, constituents_words) in enumerate(zip(q2_texts, q2_constituents)):
            words = text.split()
            _, tokens = capture_all_residuals(model, tokenizer, text)
            # We need the token mapping — capture residuals normally first
            w2t = align_words_to_tokens(text, tokens, words)

            # Convert word-level constituent groups to token groups
            token_groups = []
            for wg in constituents_words:
                tg = []
                for wi in wg:
                    if wi < len(w2t):
                        tg.extend(w2t[wi])
                if tg:
                    token_groups.append(tg)

            # --- Normal pass ---
            res_normal = capture_residuals_with_ablation(
                model, tokenizer, text, ablate_layer, "none"
            )
            hidden_normal = res_normal[ablate_layer]  # (seq_len, hidden_size)
            w_normal, _ = within_across_similarity(hidden_normal, token_groups)
            normal_ws.append(w_normal)

            # --- FFN zeroed ---
            res_no_ffn = capture_residuals_with_ablation(
                model, tokenizer, text, ablate_layer, "ffn"
            )
            hidden_no_ffn = res_no_ffn[ablate_layer]
            w_no_ffn, _ = within_across_similarity(hidden_no_ffn, token_groups)
            no_ffn_ws.append(w_no_ffn)

            # --- Attention zeroed ---
            res_no_attn = capture_residuals_with_ablation(
                model, tokenizer, text, ablate_layer, "attn"
            )
            hidden_no_attn = res_no_attn[ablate_layer]
            w_no_attn, _ = within_across_similarity(hidden_no_attn, token_groups)
            no_attn_ws.append(w_no_attn)

        avg_normal  = float(np.mean(normal_ws))
        avg_no_ffn  = float(np.mean(no_ffn_ws))
        avg_no_attn = float(np.mean(no_attn_ws))

        delta_ffn  = avg_no_ffn  - avg_normal   # negative = FFN helped
        delta_attn = avg_no_attn - avg_normal   # negative = attn helped

        # Critical component: which ablation hurts MORE (larger negative delta)
        if delta_ffn < delta_attn:
            critical = "FFN"
        elif delta_attn < delta_ffn:
            critical = "ATTN"
        else:
            critical = "equal"

        layer_result = {
            "layer": ablate_layer,
            "normal_within": round(avg_normal, 5),
            "no_ffn_within": round(avg_no_ffn, 5),
            "no_attn_within": round(avg_no_attn, 5),
            "delta_ffn": round(delta_ffn, 5),
            "delta_attn": round(delta_attn, 5),
            "critical": critical,
        }
        all_layer_results.append(layer_result)

        arrow_ffn  = "↑" if delta_ffn > 0.001 else ("↓" if delta_ffn < -0.001 else "→")
        arrow_attn = "↑" if delta_attn > 0.001 else ("↓" if delta_attn < -0.001 else "→")
        print(f"  L{ablate_layer:3d}  | {avg_normal:8.5f} | {avg_no_ffn:8.5f} | "
              f"{avg_no_attn:9.5f} | {delta_ffn:+.5f}{arrow_ffn} | {delta_attn:+.5f}{arrow_attn} | "
              f"[{critical}]")

    # Summary
    ffn_critical_layers = [r["layer"] for r in all_layer_results if r["critical"] == "FFN"]
    attn_critical_layers = [r["layer"] for r in all_layer_results if r["critical"] == "ATTN"]
    print(f"\n  FFN-critical layers:  {ffn_critical_layers}")
    print(f"  ATTN-critical layers: {attn_critical_layers}")

    return {
        "ablation_layers": COMPRESSION_LAYERS,
        "layer_results": all_layer_results,
        "ffn_critical_layers": ffn_critical_layers,
        "attn_critical_layers": attn_critical_layers,
        "summary": (
            f"FFN critical at {len(ffn_critical_layers)} layers, "
            f"ATTN critical at {len(attn_critical_layers)} layers "
            f"(of {len(COMPRESSION_LAYERS)} tested)"
        ),
    }


# ══════════════════════════════════════════════════════════════════════
# Q3: What Survives in the Residual?
# ══════════════════════════════════════════════════════════════════════

def get_embedding_matrix(model: object) -> torch.Tensor:
    """Extract the token embedding matrix. Returns (vocab_size, hidden_size) float32."""
    # Qwen / LLaMA / Mistral: model.model.embed_tokens
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight.detach().cpu().float()
    # GPTNeoX: model.gpt_neox.embed_in
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "embed_in"):
        return model.gpt_neox.embed_in.weight.detach().cpu().float()
    # GPT-2: model.transformer.wte
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte.weight.detach().cpu().float()
    raise AttributeError("Cannot find embedding matrix")


def probe_residual_content(
    perp_component: torch.Tensor,          # (n_tokens, hidden_size)
    token_ids: list[int],                  # original token IDs
    token_positions: list[int],            # absolute position in sequence
    group_positions: list[float],          # relative position within constituent (0=first, 0.5=mid, 1=last)
    embed_matrix: torch.Tensor,            # (vocab_size, hidden_size) — normalized
) -> dict:
    """Probe the perpendicular (non-PC) component of residual streams.

    Measures:
      1. Token identity: cosine similarity of perp component to embedding of the
         actual token vs. mean cosine to all other embeddings.
         Higher = token identity is preserved in the perp component.

      2. Position-in-constituent: Pearson r between perp norm and group_positions.

      3. Sentence position: Pearson r between perp norm and token_positions.

    Returns dict with per-metric scores.
    """
    n_tokens = perp_component.shape[0]
    if n_tokens < 2:
        return {"token_identity_margin": 0.0, "r_group_pos": 0.0, "r_abs_pos": 0.0}

    # Normalize perp component
    perp_norms = perp_component.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    perp_n = perp_component / perp_norms

    # 1. Token identity: compare each token's perp against its own embedding
    #    embed_matrix is pre-normalized (done outside)
    identity_margins = []
    for i, tid in enumerate(token_ids):
        if tid < embed_matrix.shape[0]:
            sim_self = float((perp_n[i] @ embed_matrix[tid]).item())
            # Compare to a random sample of other tokens
            rand_indices = torch.randint(0, embed_matrix.shape[0], (50,))
            sim_others = float((perp_n[i] @ embed_matrix[rand_indices].T).mean().item())
            identity_margins.append(sim_self - sim_others)

    # 2. Correlation of perp norm with position features
    perp_magnitudes = perp_norms.squeeze(-1).numpy()

    r_group = pearson_r(perp_magnitudes, np.array(group_positions, dtype=float))
    r_abs   = pearson_r(perp_magnitudes, np.array(token_positions, dtype=float))

    return {
        "token_identity_margin": round(float(np.mean(identity_margins)) if identity_margins else 0.0, 5),
        "r_group_pos": round(r_group, 4),
        "r_abs_pos": round(r_abs, 4),
    }


def run_q3_residual_content(
    model: object,
    tokenizer: object,
    info: object,
) -> dict:
    """Q3: What information survives in the non-dominant (perpendicular) component?

    At each layer in DOMINANT_PC_LAYERS:
      1. Compute top PC across all token representations (SVD)
      2. Decompose each token: along PC + perpendicular
      3. Probe perpendicular for: token identity, group position, sentence position
    """
    banner("Q3: RESIDUAL CONTENT — what survives in the perpendicular component?")

    # Get embedding matrix for token identity probe
    print("  Loading embedding matrix...")
    embed_matrix = get_embedding_matrix(model)
    # Normalize once
    embed_norms = embed_matrix.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    embed_matrix_n = embed_matrix / embed_norms
    print(f"  Embedding matrix: {embed_matrix_n.shape}")

    all_layer_probes: dict[int, list[dict]] = {li: [] for li in DOMINANT_PC_LAYERS}

    # Use the first 4 stimuli (more variety)
    for si, stim in enumerate(Q1_STIMULI[:4]):
        text = stim["text"]
        words = text.split()
        depths = stim["depth"]

        print(f"\n  Sentence {si+1}: \"{text}\"")

        residuals, tokens = capture_all_residuals(model, tokenizer, text)
        n_layers, seq_len, _ = residuals.shape

        w2t = align_words_to_tokens(text, tokens, words)

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        token_ids = inputs["input_ids"][0].tolist()

        # Build constituent groups (use depth 0 = main clause, depth 1 = embedded)
        # Group tokens by their constituent depth to form "groups"
        depth_to_toks: dict[int, list[int]] = {}
        for ti in range(seq_len):
            for wi, toks in enumerate(w2t):
                if ti in toks and wi < len(depths):
                    d = depths[wi]
                    depth_to_toks.setdefault(d, []).append(ti)
        # Groups are just the distinct depth-level clusters
        token_groups = [depth_to_toks[d] for d in sorted(depth_to_toks.keys())]

        # Group positions: for each token, relative position within its group
        group_positions = [0.5] * seq_len  # default
        for grp in token_groups:
            n = len(grp)
            for rank, ti in enumerate(sorted(grp)):
                group_positions[ti] = rank / max(n - 1, 1)

        # Absolute positions
        abs_positions = list(range(seq_len))

        for li in DOMINANT_PC_LAYERS:
            X = residuals[li]  # (seq_len, hidden_size)

            # Compute top PC
            X_centered = X - X.mean(dim=0, keepdim=True)
            try:
                _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
                top_pc = Vh[0]                   # (hidden_size,)
                top_pc_var = float((S[0]**2 / (S**2).sum()).item())
            except Exception:
                continue

            # Decompose: along = (X·pc)*pc,  perp = X - along
            X_orig = residuals[li]  # use non-centered for decomposition
            proj_scalar = X_orig @ top_pc        # (seq_len,)
            along = torch.outer(proj_scalar, top_pc)  # (seq_len, hidden_size)
            perp  = X_orig - along               # (seq_len, hidden_size)

            # Mean magnitudes
            along_mag = float(along.norm(dim=-1).mean().item())
            perp_mag  = float(perp.norm(dim=-1).mean().item())
            total_mag = float(X_orig.norm(dim=-1).mean().item())

            # Probe perpendicular component
            probe = probe_residual_content(
                perp_component=perp,
                token_ids=token_ids,
                token_positions=abs_positions,
                group_positions=group_positions,
                embed_matrix=embed_matrix_n,
            )

            all_layer_probes[li].append({
                "sentence": text,
                "top_pc_var_explained": round(top_pc_var, 4),
                "along_frac": round(along_mag / max(total_mag, 1e-8), 4),
                "perp_frac": round(perp_mag / max(total_mag, 1e-8), 4),
                **probe,
            })

    # Aggregate across sentences and print
    print(f"\n  {'Layer':>5s} | {'top_pc_var':>10s} | {'along_frac':>10s} | "
          f"{'perp_frac':>9s} | {'id_margin':>9s} | {'r_grp_pos':>9s} | {'r_abs_pos':>9s}")
    print(f"  {'-' * 75}")

    agg_results = {}
    for li in DOMINANT_PC_LAYERS:
        probes = all_layer_probes[li]
        if not probes:
            continue
        agg = {
            "layer": li,
            "top_pc_var_explained": round(float(np.mean([p["top_pc_var_explained"] for p in probes])), 4),
            "along_frac": round(float(np.mean([p["along_frac"] for p in probes])), 4),
            "perp_frac": round(float(np.mean([p["perp_frac"] for p in probes])), 4),
            "token_identity_margin": round(float(np.mean([p["token_identity_margin"] for p in probes])), 5),
            "r_group_pos": round(float(np.mean([p["r_group_pos"] for p in probes])), 4),
            "r_abs_pos": round(float(np.mean([p["r_abs_pos"] for p in probes])), 4),
            "n_sentences": len(probes),
        }
        agg_results[li] = agg
        print(f"  L{li:3d}  | {agg['top_pc_var_explained']:10.4f} | "
              f"{agg['along_frac']:10.4f} | {agg['perp_frac']:9.4f} | "
              f"{agg['token_identity_margin']:+9.5f} | {agg['r_group_pos']:+9.4f} | "
              f"{agg['r_abs_pos']:+9.4f}")

    # Interpretation
    print("\n  INTERPRETATION:")
    for li, agg in agg_results.items():
        margin = agg["token_identity_margin"]
        r_grp  = abs(agg["r_group_pos"])
        r_abs  = abs(agg["r_abs_pos"])
        along  = agg["along_frac"]
        print(f"    L{li}: {along*100:.0f}% in PC direction. "
              f"Perp carries: id_margin={margin:+.4f}, |r_grp|={r_grp:.3f}, |r_abs|={r_abs:.3f}")

    return {
        "analysis_layers": DOMINANT_PC_LAYERS,
        "per_layer": {str(li): v for li, v in agg_results.items()},
        "per_sentence_per_layer": {
            str(li): all_layer_probes[li] for li in DOMINANT_PC_LAYERS
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Q4: Syntax vs. Semantics
# ══════════════════════════════════════════════════════════════════════

def group_similarity_score(
    sim_matrix: np.ndarray,
    groups: list[list[int]],
    seq_len: int,
) -> float:
    """Average within-group vs across-group similarity using a precomputed matrix."""
    within_sims: list[float] = []
    across_sims: list[float] = []

    valid_groups = [[t for t in g if t < seq_len] for g in groups if g]
    valid_groups = [g for g in valid_groups if g]

    for gi, grp in enumerate(valid_groups):
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                within_sims.append(float(sim_matrix[grp[i], grp[j]]))
        for gj in range(gi + 1, len(valid_groups)):
            other = valid_groups[gj]
            for ti in grp[:3]:
                for tj in other[:3]:
                    across_sims.append(float(sim_matrix[ti, tj]))

    if not within_sims:
        return 0.0
    w = float(np.mean(within_sims))
    a = float(np.mean(across_sims)) if across_sims else 0.0
    return w - a   # positive = within-group more similar than across


def run_q4_syntax_vs_semantics(
    model: object,
    tokenizer: object,
    info: object,
) -> dict:
    """Q4: Does convergence cluster by syntax or by semantics?

    For garden-path and ambiguous sentences, at L6-9:
      - Compute full pairwise cosine similarity matrix
      - Score how well syntactic grouping fits the similarity pattern
      - Score how well semantic grouping fits
      - Compare: if syntax_score > semantic_score, model tracks syntax

    For unambiguous controls: scores should be equal (groups are the same).
    """
    banner("Q4: SYNTAX vs. SEMANTICS — does convergence follow syntax or meaning?")

    probe_layers = list(range(6, 10))  # L6, L7, L8, L9

    all_results = []

    print(f"\n  Probing at layers: {probe_layers}")

    for stim in Q4_STIMULI:
        label = stim["label"]
        text  = stim["text"]
        syn_groups = stim["syntactic_groups"]   # word-level groups
        sem_groups = stim["semantic_groups"]

        print(f"\n  ── {label}: \"{text}\"")
        print(f"     ({stim['note']})")

        residuals, tokens = capture_all_residuals(model, tokenizer, text)
        n_layers, seq_len, _ = residuals.shape
        words = text.split()
        w2t = align_words_to_tokens(text, tokens, words)

        # Convert word-level groups to token groups
        def words_to_toks(word_groups: list[list[int]]) -> list[list[int]]:
            result = []
            for wg in word_groups:
                tg = []
                for wi in wg:
                    if wi < len(w2t):
                        tg.extend(w2t[wi])
                if tg:
                    result.append(tg)
            return result

        syn_tok_groups = words_to_toks(syn_groups)
        sem_tok_groups = words_to_toks(sem_groups)

        layer_data_list = []

        for li in probe_layers:
            X = residuals[li]  # (seq_len, hidden_size)
            norms = X.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            Xn = X / norms
            sim_mat = (Xn @ Xn.T).numpy()  # (seq_len, seq_len)

            syn_score = group_similarity_score(sim_mat, syn_tok_groups, seq_len)
            sem_score = group_similarity_score(sim_mat, sem_tok_groups, seq_len)

            tracks = "SYNTAX" if syn_score > sem_score else "SEMANTICS"
            if abs(syn_score - sem_score) < 0.005:
                tracks = "EQUAL"

            layer_data_list.append({
                "layer": li,
                "syntax_score": round(syn_score, 5),
                "semantic_score": round(sem_score, 5),
                "delta": round(syn_score - sem_score, 5),
                "tracks": tracks,
            })

            delta = syn_score - sem_score
            direction = "→SYNTAX" if delta > 0.005 else ("→SEMANTICS" if delta < -0.005 else "→EQUAL")
            print(f"    L{li}: syn={syn_score:+.5f}  sem={sem_score:+.5f}  "
                  f"Δ={delta:+.5f}  {direction}")

        all_results.append({
            "label": label,
            "text": text,
            "note": stim["note"],
            "is_control": stim["label"].startswith("control"),
            "syntactic_word_groups": syn_groups,
            "semantic_word_groups": sem_groups,
            "syntactic_token_groups": syn_tok_groups,
            "semantic_token_groups": sem_tok_groups,
            "layer_results": layer_data_list,
        })

    # Cross-sentence summary
    print("\n  SUMMARY across sentences:")
    print(f"  {'Sentence':30s} | L6_tracks | L7_tracks | L8_tracks | L9_tracks")
    print(f"  {'-' * 75}")
    for r in all_results:
        layer_tracks = [d["tracks"] for d in r["layer_results"]]
        label_short = r["label"][:28]
        print(f"  {label_short:30s} | {'  |  '.join(f'{t[:6]:6s}' for t in layer_tracks)}")

    # Aggregate: garden path sentences — does model group by syntax or semantics?
    gp_results = [r for r in all_results if not r["is_control"]]
    ctrl_results = [r for r in all_results if r["is_control"]]

    gp_syntax_wins = sum(
        1 for r in gp_results
        for d in r["layer_results"]
        if d["tracks"] == "SYNTAX"
    )
    gp_sem_wins = sum(
        1 for r in gp_results
        for d in r["layer_results"]
        if d["tracks"] == "SEMANTICS"
    )
    total_gp_votes = len(gp_results) * len(probe_layers)

    print(f"\n  Garden-path/ambiguous: SYNTAX wins {gp_syntax_wins}/{total_gp_votes} layer votes, "
          f"SEMANTICS wins {gp_sem_wins}/{total_gp_votes}")

    return {
        "probe_layers": probe_layers,
        "sentences": all_results,
        "aggregate": {
            "n_garden_path_sentences": len(gp_results),
            "n_control_sentences": len(ctrl_results),
            "gp_syntax_layer_wins": gp_syntax_wins,
            "gp_semantic_layer_wins": gp_sem_wins,
            "total_gp_layer_votes": total_gp_votes,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Compression map probe for Qwen3-4B")
    parser.add_argument(
        "--skip-q", type=int, nargs="*", default=[],
        help="Skip specific experiment numbers (e.g. --skip-q 3 4)",
    )
    args = parser.parse_args()
    skip = set(args.skip_q or [])

    t0 = time.time()
    banner("COMPRESSION MAP PROBE — Qwen3-4B")
    print("  Building on compression-shape findings:")
    print("  • Constituent similarity peaks L6-9")
    print("  • Effective dimensionality collapses to 1 at L6+")
    print("  • Attention narrows at deeper layers")
    print("\n  Questions:")
    print("  Q1: What is the dominant direction?")
    print("  Q2: Is compression attention- or FFN-mediated?")
    print("  Q3: What survives in the perpendicular (non-dominant) component?")
    print("  Q4: Does convergence track syntax or semantics?")

    # ── Load model ──────────────────────────────────────────────────
    banner("LOADING MODEL")
    print("  Loading Qwen/Qwen3-4B ...")
    t_load = time.time()
    model, tokenizer, info = _load_model("Qwen/Qwen3-4B")
    print(f"  Loaded in {time.time() - t_load:.1f}s")
    print(f"  Layers: {info.n_layers}  Hidden: {info.hidden_size}  Device: {info.device}")

    # ── Run experiments ─────────────────────────────────────────────
    results: dict = {
        "model": "Qwen/Qwen3-4B",
        "n_layers": info.n_layers,
        "hidden_size": info.hidden_size,
        "compression_layers_tested": COMPRESSION_LAYERS,
        "dominant_pc_layers_tested": DOMINANT_PC_LAYERS,
    }

    if 1 not in skip:
        t1 = time.time()
        results["q1_dominant_direction"] = run_q1_dominant_direction(model, tokenizer, info)
        results["q1_elapsed_s"] = round(time.time() - t1, 2)
    else:
        print("\n  [Skipping Q1]")

    if 2 not in skip:
        t2 = time.time()
        results["q2_convergence_mechanism"] = run_q2_convergence_mechanism(model, tokenizer, info)
        results["q2_elapsed_s"] = round(time.time() - t2, 2)
    else:
        print("\n  [Skipping Q2]")

    if 3 not in skip:
        t3 = time.time()
        results["q3_residual_content"] = run_q3_residual_content(model, tokenizer, info)
        results["q3_elapsed_s"] = round(time.time() - t3, 2)
    else:
        print("\n  [Skipping Q3]")

    if 4 not in skip:
        t4 = time.time()
        results["q4_syntax_vs_semantics"] = run_q4_syntax_vs_semantics(model, tokenizer, info)
        results["q4_elapsed_s"] = round(time.time() - t4, 2)
    else:
        print("\n  [Skipping Q4]")

    # ── Synthesize findings ─────────────────────────────────────────
    banner("SYNTHESIS")

    if "q1_dominant_direction" in results:
        q1 = results["q1_dominant_direction"]
        agg = q1["aggregate_layer_correlations"]
        # Find layers where any feature has |r| > 0.4
        strong_layers = [
            e for e in agg
            if max(abs(e.get("r_word_pos", 0)), abs(e.get("r_depth", 0)),
                   abs(e.get("r_pos_cat", 0)), abs(e.get("r_content", 0))) > 0.4
        ]
        dominant_features = [e["dominant_feature"] for e in agg[6:10] if "dominant_feature" in e]
        print(f"\n  Q1: Dominant direction at L6-9 tracks: {dominant_features}")
        print(f"      Strong correlation (|r|>0.4) at {len(strong_layers)} layers")

    if "q2_convergence_mechanism" in results:
        q2 = results["q2_convergence_mechanism"]
        print(f"\n  Q2: {q2['summary']}")

    if "q3_residual_content" in results:
        q3 = results["q3_residual_content"]
        pl = q3["per_layer"]
        if pl:
            best_li = max(pl.keys(), key=lambda k: abs(pl[k]["token_identity_margin"]))
            best = pl[best_li]
            print(f"\n  Q3: At L{best_li}: {best['along_frac']*100:.0f}% of representation in PC direction.")
            print(f"      Perpendicular component: id_margin={best['token_identity_margin']:+.4f}, "
                  f"r_grp={best['r_group_pos']:+.3f}, r_abs={best['r_abs_pos']:+.3f}")

    if "q4_syntax_vs_semantics" in results:
        q4 = results["q4_syntax_vs_semantics"]
        agg4 = q4["aggregate"]
        total = agg4["total_gp_layer_votes"]
        syn_w = agg4["gp_syntax_layer_wins"]
        sem_w = agg4["gp_semantic_layer_wins"]
        bias = "SYNTAX" if syn_w > sem_w else ("SEMANTICS" if sem_w > syn_w else "EQUAL")
        print(f"\n  Q4: Garden-path sentences: {bias} bias "
              f"({syn_w} syntax vs {sem_w} semantic layer-votes out of {total})")

    # ── Save results ────────────────────────────────────────────────
    total_elapsed = round(time.time() - t0, 2)
    results["total_elapsed_s"] = total_elapsed

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "qwen3_4b_map.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))

    banner(f"DONE — {total_elapsed:.0f}s total")
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
