#!/usr/bin/env python3
"""VSM-LM v6 — Ternary on Metal, 1B token training run.

MLX implementation with custom Metal ternary matmul kernels.
All ternary weights learn through flip accumulation (not Adam).
Continuous params (gamma, embeddings, norms, gates) use AdamW.

Usage:
    uv run python scripts/v6/train.py
    uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_003500
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from verbum.v6.model import VSMLMV6
from verbum.v6.ternary import (
    TernaryLinear,
    _walk_ternary_modules,
    _classify_group,
    accumulate_flips,
    apply_flips,
    normalize_shared_grads,
    restore_ternary,
    zero_ternary_grads,
)

DATA_DIR = Path("/Users/mwhitford/data/fractal-bitnet/shards")

# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════

VOCAB_SIZE = 50277
D_MODEL = 512
D_REGISTER = 128
SEQ_LEN = 4096
D_FF = 1536
D_FF_CONSOLIDATE = 2048
WINDOW = 8
STRIDES = (1, 8, 16, 32, 64, 128, 256, 512, 1024)
N_HEADS = 8
ALPHA = 1.18

BATCH_SIZE = 2
GRAD_ACCUM = 4
TOKENS_PER_STEP = BATCH_SIZE * GRAD_ACCUM * SEQ_LEN  # 32,768
TARGET_TOKENS = 1_000_000_000
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
N_STEPS = TARGET_TOKENS // TOKENS_PER_STEP + 1  # 30,518
WARMUP_STEPS = 500
SEED = 42

FLIP_INTERVAL = 4         # check for consensus flips every 4 steps (cheap: just threshold + mx.where)
FLIP_PROBE_INTERVAL = 100 # re-run VSM probes for monitoring (expensive: 13 forward passes)
FLIP_CONSENSUS = 40       # absolute threshold: net votes needed to flip (int8 accum units)
                          # Accumulators persist across intervals — only reset on flip.
                          # 40 net votes = strong directional consensus before committing.
                          # At interval=4 (16 votes/interval), needs ~3 intervals to flip:
                          # prevents single-interval cascade while staying responsive.
FLIP_MAX_PCT = 0.001      # cap: at most 0.1% of ternary weights flip per interval (~35K of 35M)
                          # Small blast radius lets Adam's running statistics (m_t, v_t)
                          # stay approximately valid across topology changes. Evolution not
                          # revolution — continuous params can compensate within a few steps.
                          # Previous: 1% (350K) caused cascading instability on resume from
                          # frozen topology (loss 5.18 → 11.59 in 125 steps).
# No gradient clipping — Adam handles per-parameter scale adaptation.
# Shared-weight gradients are normalized by 1/N_PASSES instead (see normalize_shared_grads).
# MAX_GRAD_NORM removed: clipping at any fixed threshold creates unstable
# scaling when ‖g‖ oscillates 10⁴-10⁹ (as it does in this 5-pass shared-weight architecture).

# PHI_LAMBDA is now managed by phase transitions (see relational_control).
# Initial value: 0.0 (explore phase). Updated at runtime by phase_transition().
PHI_LAMBDA = 0.0

# φ-feedback monitoring only activates below this loss. Above it,
# compression ratios are meaningless noise — the model hasn't learned
# enough structure for φ-deviation to be a real signal.
PHI_FEEDBACK_LOSS = 6.0

# ── Information-theoretic constants ──────────────────────────────
# Chinchilla scaling law: L(N,D) = E + A/N^α + B/D^β
# E = irreducible entropy of natural language (nats/token)
# Source: Hoffmann et al. 2022, Epoch AI replication 2024
# Relational framing inspired by:
#   https://github.com/massimilianoconcas0-del/Relational_Loss_ML
#   (Concas 2026, "Relational Calculus for Efficient ML")
E_IRREDUCIBLE = 1.69       # nats/token (Chinchilla); Epoch AI: 1.82
LOG_V = float(np.log(VOCAB_SIZE))  # max entropy = log(vocab) ≈ 10.83
LEARNABLE_RANGE = LOG_V - E_IRREDUCIBLE

# Golden ratio hypothesis: true entropy rate may be 1/φ ≈ 0.618 bits/char
# Within error bars of Shannon (0.6-1.3), Chinchilla (0.667 bits/byte)
# If compression is self-similar (Hilberg 1990), φ is the fixed point
PHI = (1 + np.sqrt(5)) / 2    # ≈ 1.618
INV_PHI = 1 / PHI              # ≈ 0.618

LOG_INTERVAL = 25
EVAL_INTERVAL = 500
CHECKPOINT_INTERVAL = 500

# These are set from model.REGISTER_NAMES etc. after model construction.
# Declared here so module-level functions can reference them.
N_PASSES = 5
PASS_NAMES = ["L0_asc", "L1_asc", "L2_apex", "L1_desc", "L0_desc"]
REG_NAMES = ["type", "scope", "role"]
PHASE_NAMES = ["prep", "converge", "consolidate"]


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n", flush=True)


# ══════════════════════════════════════════════════════════════════════
# Data loader (numpy-based, framework-agnostic)
# ══════════════════════════════════════════════════════════════════════


class ShardedDataLoader:
    def __init__(self, data_dir, batch_size, seq_len, split="train", seed=42):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        shards = sorted(self.data_dir.glob("shard_*.npy"))
        self.shards = shards[:54] if split == "train" else shards[54:]
        rng = np.random.RandomState(seed)
        self._indices = []
        for si, shard_path in enumerate(self.shards):
            shard_len = len(np.load(shard_path, mmap_mode="r"))
            n_seqs = shard_len // (seq_len + 1)
            for j in range(n_seqs):
                self._indices.append((si, j * (seq_len + 1)))
        rng.shuffle(self._indices)
        self._idx_pos = 0
        self._loaded_shards = {}

    def _get_shard(self, idx):
        if idx not in self._loaded_shards:
            self._loaded_shards[idx] = np.load(self.shards[idx], mmap_mode="r")
        return self._loaded_shards[idx]

    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        sequences = []
        for _ in range(B):
            if self._idx_pos >= len(self._indices):
                self._idx_pos = 0
            si, pos = self._indices[self._idx_pos]
            self._idx_pos += 1
            shard = self._get_shard(si)
            seq = shard[pos : pos + T + 1].astype(np.int64)
            sequences.append(seq)
        buf = mx.array(np.stack(sequences))
        return buf[:, :T], buf[:, 1 : T + 1]

    def reset(self):
        self._idx_pos = 0


# ══════════════════════════════════════════════════════════════════════
# Loss function
# ══════════════════════════════════════════════════════════════════════


# ── Per-sequence stratum weighting (Loop 4 application) ──────────
# Module-level state for stratum-weighted loss. Set by the training
# loop before each micro-batch. loss_fn reads it as a non-differentiable
# routing signal — only the loss scaling flows through the gradient.
_batch_seq_weights: mx.array | None = None


def build_stratum_token_sets(tokenizer) -> dict[str, set[int]]:
    """Precompute token-level stratum membership from vocabulary.

    Scans the tokenizer vocabulary once at init. Returns sets of token IDs
    for each stratum. Classification becomes a pure integer set-membership
    count — no tokenizer.decode() calls during training.
    """
    math_chars = set("λ∀∈∃→≥≤²³∫Σ∏∂∇⊗⊕∧∨¬↔⇒∞ℝℤℕℂ×÷±≈≠")
    tech_terms = [
        "gradient", "softmax", "attention", "embedding", "backprop",
        "layer", "norm", "optimizer", "batch", "loss", "neural",
        "transformer", "convolution", "activation", "dropout",
        "weight", "tokeniz", "logit", "entropy", "perplexity",
        "parameter", "tensor", "kernel", "epoch",
    ]
    comp_terms = [
        " who ", " whom ", " which ", " whose ",
        " whether ", " although ", " whereas ", " whenever ",
        " wherever ", " whoever ",
    ]

    vocab = tokenizer.get_vocab()  # {token_str: id}
    math_ids: set[int] = set()
    tech_ids: set[int] = set()
    comp_ids: set[int] = set()

    for token_str, token_id in vocab.items():
        # Math: contains math symbols or is a digit token
        if any(c in math_chars for c in token_str):
            math_ids.add(token_id)
        elif token_str.strip().replace(".", "").replace("-", "").isdigit() and len(token_str.strip()) > 0:
            math_ids.add(token_id)

        # Technical: contains ML/CS terms
        tok_lower = token_str.lower()
        if any(t in tok_lower for t in tech_terms):
            tech_ids.add(token_id)

        # Compositional: relative clause markers
        if any(t.strip() in tok_lower for t in comp_terms):
            comp_ids.add(token_id)

    return {"math": math_ids, "technical": tech_ids, "compositional": comp_ids}


def build_stratum_lookup(token_sets: dict[str, set[int]], vocab_size: int) -> dict[str, mx.array]:
    """Build boolean lookup arrays from token sets for fast tensor classification.

    Returns {stratum: (vocab_size,) bool array} for index-based lookup.
    """
    lookups = {}
    for sname, ids in token_sets.items():
        arr = np.zeros(vocab_size, dtype=np.bool_)
        for tid in ids:
            if tid < vocab_size:
                arr[tid] = True
        lookups[sname] = mx.array(arr)
    return lookups


def classify_batch_tokens(
    x: mx.array,
    stratum_lookups: dict[str, mx.array],
    stratum_weights: dict[str, float],
) -> mx.array:
    """Classify each sequence by token composition, return per-sequence weights.

    Pure tensor ops — no decoding, no string matching. Each sequence is
    classified by which stratum has the highest token density.

    Args:
        x: (B, L) int32 token IDs
        stratum_lookups: {stratum: (vocab_size,) bool} from build_stratum_lookup
        stratum_weights: {stratum: weight} from compute_stratum_weights

    Returns:
        (B,) float32 per-sequence weights, normalized so mean=1.
    """
    B = x.shape[0]
    # Count stratum token hits per sequence: index into lookup array
    counts = {}
    for sname, lookup in stratum_lookups.items():
        hits = lookup[x]  # (B, L) bool
        counts[sname] = hits.sum(axis=1)  # (B,)

    # Classify each sequence by highest hit density
    strata_names = list(counts.keys())
    hit_matrix = mx.stack([counts[s].astype(mx.float32) for s in strata_names], axis=1)  # (B, n_strata)
    mx.eval(hit_matrix)

    weights = []
    for i in range(B):
        hits_i = [hit_matrix[i, j].item() for j in range(len(strata_names))]
        max_idx = max(range(len(hits_i)), key=lambda j: hits_i[j])
        if hits_i[max_idx] > 0:
            stratum = strata_names[max_idx]
        else:
            stratum = "prose"
        weights.append(stratum_weights.get(stratum, 1.0))

    w_arr = mx.array(weights, dtype=mx.float32)
    # Normalize so mean=1 (preserves loss scale)
    w_arr = w_arr / (w_arr.mean() + 1e-8)
    return w_arr


def loss_fn(model, x, y):
    """Compute combined loss with optional per-sequence stratum weighting.

    When _batch_seq_weights is set (by the training loop), computes
    per-sequence CE loss weighted by stratum importance. Lagging strata
    get higher weight → more gradient signal → faster catch-up.

    When _batch_seq_weights is None, falls back to uniform mean.
    """
    logits, _, phi_loss, _ = model(x, y)

    B, L, V = logits.shape
    ce_per_token = nn.losses.cross_entropy(
        logits.reshape(-1, V), y.reshape(-1),
    )  # (B*L,)

    if _batch_seq_weights is not None:
        # Per-sequence weighted loss
        ce_per_seq = ce_per_token.reshape(B, L).mean(axis=1)  # (B,)
        ce_loss = (ce_per_seq * _batch_seq_weights).mean()
    else:
        ce_loss = ce_per_token.mean()

    if phi_loss is not None and model.phi_lambda > 0:
        return ce_loss + model.phi_lambda * phi_loss
    return ce_loss


def relational_metrics(loss: float) -> dict:
    """Compute information-theoretic metrics from raw CE loss.

    Returns dict with:
      - relational_loss: fraction of learnable capacity remaining [0=optimal, 1=random]
      - excess_ppl: how many x more confused than theoretically necessary
      - ppl: standard perplexity
      - reducible_loss: nats of learnable structure still uncaptured
    """
    reducible = loss - E_IRREDUCIBLE
    return {
        "relational_loss": reducible / LEARNABLE_RANGE,
        "excess_ppl": float(np.exp(max(reducible, 0))),
        "ppl": float(np.exp(loss)),
        "reducible_loss": reducible,
    }


# ══════════════════════════════════════════════════════════════════════
# LR schedule
# ══════════════════════════════════════════════════════════════════════


def lr_schedule(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, N_STEPS - WARMUP_STEPS)
    return LEARNING_RATE * max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))


# ══════════════════════════════════════════════════════════════════════
# Eval
# ══════════════════════════════════════════════════════════════════════


def estimate_loss(model, eval_loader, n_batches=10):
    total = 0
    for _ in range(n_batches):
        x, y = eval_loader.next_batch()
        _, ce_loss, _, _ = model(x, y)
        mx.eval(ce_loss)
        total += ce_loss.item()
    return total / n_batches


def compile_gate_test(model, tokenizer):
    prompts = [
        "λ",
        "The dog chased the cat",
        "Every student read a book",
        "compile: The cat sat on the mat",
    ]
    results = []
    for prompt in prompts:
        ids = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
        out = model.generate(ids, max_new_tokens=30)  # greedy (argmax)
        mx.eval(out)
        text = tokenizer.decode(out[0].tolist())
        has_lambda = "λ" in text[len(prompt):] or "\\" in text[len(prompt):]
        results.append({"prompt": prompt, "output": text, "has_lambda": has_lambda})
    n_lambda = sum(1 for r in results if r["has_lambda"])
    return {"score": f"{n_lambda}/{len(prompts)}", "results": results}


# ── Per-stratum loss samples ──────────────────────────────────────

STRATUM_SAMPLES = {
    "prose": [
        "The cat sat on the mat and looked out the window at the birds flying south.",
        "In a quiet village nestled between rolling hills the old baker opened his shop.",
    ],
    "compositional": [
        "The man who the dog that the cat chased bit ran away quickly.",
        "If every student reads a book then some teacher is happy.",
    ],
    "technical": [
        "The gradient of the loss with respect to the weights is computed via backpropagation.",
        "Attention scores are computed as the softmax of the scaled dot product of queries and keys.",
    ],
    "math": [
        "λx. λy. apply(x, y) → result",
        "P(A|B) = P(B|A) × P(A) / P(B)",
    ],
}


def phi_compression_probe(model, tokenizer):
    """Lightweight φ-compression probe for inline training diagnostics.

    Runs forward_instrumented on a few samples, returns per-pass
    compression ratios, per-stride ratios, and gate values.
    """
    samples = [
        "The cat sat on the mat and looked out the window at the birds.",
        "Every student who passed the exam received a certificate.",
        "In 1969 Apollo 11 landed on the moon marking a giant leap.",
    ]
    all_ratios = {p: [] for p in PASS_NAMES}
    all_gates = {}  # {pass_phase: [values]}
    all_stride_ratios = {}  # {pass_stride_key: [values]}
    all_hilberg = {p: [] for p in PASS_NAMES}

    for text in samples:
        ids = mx.array(tokenizer.encode(text)).reshape(1, -1)
        if ids.shape[1] > model.max_len:
            ids = ids[:, -model.max_len:]
        targets = mx.concatenate([ids[:, 1:], mx.zeros((1, 1), dtype=mx.int32)], axis=1)
        _, _, metrics = model.forward_instrumented(ids, targets)
        for p in PASS_NAMES:
            cr_key = f"{p}_compression_ratio"
            if cr_key in metrics:
                all_ratios[p].append(metrics[cr_key])
            # Gate values
            for ph in PHASE_NAMES:
                gk = f"{p}_{ph}"
                gv = metrics.get(f"{p}_{ph}_gate_mean")
                if gv is not None:
                    all_gates.setdefault(gk, []).append(gv)
            # Per-stride ratios
            for key, val in metrics.items():
                if key.startswith(f"{p}_stride_") and key.endswith("_ratio"):
                    all_stride_ratios.setdefault(key, []).append(val)
            # Hilberg β
            hb = metrics.get(f"{p}_hilberg_beta")
            hs = metrics.get(f"{p}_hilberg_slope")
            if hb is not None:
                all_hilberg[p].append({"slope": hs, "beta": hb})
            elif hs is not None:
                all_hilberg[p].append({"slope": hs, "beta": hs + 1})

    result = {}
    for p in PASS_NAMES:
        if all_ratios[p]:
            result[p] = sum(all_ratios[p]) / len(all_ratios[p])

    if result:
        all_cr = list(result.values())
        result["mean"] = sum(all_cr) / len(all_cr)
        result["mean_phi_dev"] = sum(abs(cr - INV_PHI) for cr in all_cr) / len(all_cr)

    # Average gate values
    result["gates"] = {}
    for gk, gvs in all_gates.items():
        result["gates"][gk] = sum(gvs) / len(gvs)

    # Average Hilberg β
    result["hilberg"] = {}
    for p in PASS_NAMES:
        if all_hilberg[p]:
            avg_slope = sum(h["slope"] for h in all_hilberg[p]) / len(all_hilberg[p])
            avg_beta = sum(h["beta"] for h in all_hilberg[p]) / len(all_hilberg[p])
            result["hilberg"][p] = {"slope": avg_slope, "beta": avg_beta}

    return result


VSM_PROBE_TEXT = "Every student who passed the final exam received a certificate."


def vsm_probe(model, tokenizer):
    """Lightweight VSM signal extraction for flip feedback.

    Runs forward_instrumented on one fixed sample and returns the
    control signals the VSM uses to regulate itself:
    - meta_s3: per-pass contribution gates (5 values)
    - s3: per-pass × per-phase alignment gates (15 values)
    - register_norms: per-pass × per-register structural state (15 values)

    Returns a flat dict of scalars for easy before/after comparison,
    plus a signal vector for cosine similarity.
    """
    ids = mx.array(tokenizer.encode(VSM_PROBE_TEXT)).reshape(1, -1)
    if ids.shape[1] > model.max_len:
        ids = ids[:, -model.max_len:]
    targets = mx.concatenate([ids[:, 1:], mx.zeros((1, 1), dtype=mx.int32)], axis=1)

    _, _, metrics = model.forward_instrumented(ids, targets)

    signals = {}

    # Meta-S3 gates: per-pass importance
    for p in PASS_NAMES:
        key = f"meta_s3_gate_{p}"
        signals[key] = metrics.get(key, 0.5)

    # S3 phase gates: per-pass × per-phase activity
    for p in PASS_NAMES:
        for ph in PHASE_NAMES:
            key = f"{p}_{ph}_gate_mean"
            signals[key] = metrics.get(key, 0.5)

    # Register norms: structural state
    for p in PASS_NAMES:
        for rn in REG_NAMES:
            key = f"{p}_register_{rn}_norm"
            signals[key] = metrics.get(key, 0.0)

    # φ-deviation from the same instrumented pass (for flip feedback)
    phi_dev = metrics.get("mean_phi_deviation", None)
    signals["phi_deviation"] = phi_dev

    # Flatten to vector for cosine similarity (exclude phi_deviation — it's a separate signal)
    signal_vec = np.array([signals[k] for k in sorted(signals.keys()) if k != "phi_deviation"], dtype=np.float64)

    return signals, signal_vec


def vsm_stability(vec_before, vec_after):
    """Cosine similarity between VSM signal vectors.

    Returns similarity in [0, 1]:
    - > 0.95: system self-stabilized, no intervention needed
    - 0.8–0.95: mild perturbation, monitor
    - < 0.8: destabilized, escalate to global feedback
    """
    dot = np.dot(vec_before, vec_after)
    norm_b = np.linalg.norm(vec_before)
    norm_a = np.linalg.norm(vec_after)
    if norm_b < 1e-10 or norm_a < 1e-10:
        return 0.0
    return float(dot / (norm_b * norm_a))


def compute_per_group_flip_targets(
    signals,
    base_target,
    stratum_spread: float = 0.0,
    hilberg_beta_dev: float = 0.0,
):
    """Compute per-group flip targets from VSM control signals.

    Inverts importance: high gate → protect (fewer flips), low gate → explore (more flips).
    Base_target is the current global flip_target_pct.

    Additional signals:
      stratum_spread: compositional-prose loss spread. High spread (>1.0)
        means stride_stack isn't composing well → more exploration needed.
      hilberg_beta_dev: |mean_β - 0.5|. High deviation means stride
        hierarchy isn't achieving self-similar compression → explore.

    Returns dict {group_name: target_pct}.
    """
    # Average S3 gates per phase across all passes
    phase_activity = {}
    for ph in PHASE_NAMES:
        gates = [signals.get(f"{p}_{ph}_gate_mean", 0.5) for p in PASS_NAMES]
        phase_activity[ph] = sum(gates) / len(gates)

    # Meta-S3: overall pass importance
    pass_importance = [signals.get(f"meta_s3_gate_{p}", 0.5) for p in PASS_NAMES]
    mean_importance = sum(pass_importance) / len(pass_importance)

    # Inversion: importance → protection factor
    # gate=1.0 → factor=0.3 (protect: 30% of base rate)
    # gate=0.0 → factor=2.0 (explore: 200% of base rate)
    def invert(gate_val):
        factor = 2.0 * (1.0 - gate_val) + 0.3 * gate_val
        return max(0.3, min(2.0, factor))

    targets = {
        "prep": base_target * invert(phase_activity["prep"]),
        "stride_stack": base_target * invert(phase_activity["converge"]),
        "consolidate": base_target * invert(phase_activity["consolidate"]),
        "mod_projs": base_target * invert(mean_importance),
        # Control system: always conservative (50% of base)
        "s3": base_target * 0.5,
        "s4": base_target * 0.5,
        "meta": base_target * 0.3,
    }

    # ── Stratum-aware stride_stack modulation ─────────────────
    # High compositional-prose spread → stride hierarchy isn't
    # composing well → give it more topological exploration.
    if stratum_spread > 1.0:
        targets["stride_stack"] *= 1.5
        targets["consolidate"] *= 1.3
    elif stratum_spread > 0.5:
        targets["stride_stack"] *= 1.2
    elif stratum_spread < 0.2 and stratum_spread > 0:
        targets["stride_stack"] *= 0.8  # converging, protect

    # ── Hilberg β-aware stride_stack modulation ───────────────
    # |β - 0.5| > 0.2 → strides aren't achieving self-similar
    # compression → need more topological change.
    if hilberg_beta_dev > 0.3:
        targets["stride_stack"] *= 1.4
    elif hilberg_beta_dev > 0.2:
        targets["stride_stack"] *= 1.2

    # Clamp all to [FLIP_PCT_MIN, FLIP_PCT_MAX]
    for k in targets:
        targets[k] = max(FLIP_PCT_MIN, min(FLIP_PCT_MAX, targets[k]))

    return targets


# ══════════════════════════════════════════════════════════════════════
# Relational training control — four interlocking feedback loops
# ══════════════════════════════════════════════════════════════════════
#
# r ≡ (loss - E_IRREDUCIBLE) / LEARNABLE_RANGE ∈ [0,1]
# 0 = optimal (at irreducible entropy), 1 = random (at log(vocab))
#
# Loop 1: flip_by_r — r modulates flip aggressiveness (continuous)
# Loop 2: phase_transition — r_ema crosses thresholds (discrete w/ hysteresis)
# Loop 3: flip_by_stratum — stratum gaps target specific VSM groups
# Loop 4: stratum_weight — upweight lagging strata (logged, future: applied)
#
# Composition: effective_rate(group) = phase_base × r_scale × group_factor


def adaptive_flip_scale(r: float) -> float:
    """Continuous flip aggressiveness scale from relational loss.

    r > 0.6 → scale=2.0  (explore: much topology to discover)
    r = 0.4 → scale=1.0  (balanced: baseline rates)
    r < 0.2 → scale=0.3  (protect: nearly converged)

    Smooth ramp, no discontinuities.
    """
    return 0.3 + 1.7 * max(0.0, min(1.0, r / 0.6))


# Phase state: explore → balance → refine
PHASE_EXPLORE = "explore"
PHASE_BALANCE = "balance"
PHASE_REFINE = "refine"

PHASE_CONFIG = {
    PHASE_EXPLORE: {"phi_lambda": 0.0, "flip_max_scale": 2.0, "consensus_scale": 0.5},
    PHASE_BALANCE: {"phi_lambda": 0.01, "flip_max_scale": 1.0, "consensus_scale": 1.0},
    PHASE_REFINE: {"phi_lambda": 0.1, "flip_max_scale": 0.3, "consensus_scale": 2.0},
}

PHASE_HYSTERESIS = 100  # steps below/above threshold before transition


def phase_for_r(r_ema: float) -> str:
    """Target phase for a given r_ema (without hysteresis)."""
    if r_ema > 0.5:
        return PHASE_EXPLORE
    elif r_ema < 0.25:
        return PHASE_REFINE
    else:
        return PHASE_BALANCE


def phase_transition(
    r_ema: float,
    current_phase: str,
    steps_toward_new: int,
) -> tuple[str, int, bool]:
    """Phase transition with hysteresis.

    Returns (new_phase, new_steps_toward, did_transition).
    Requires PHASE_HYSTERESIS consecutive steps targeting a different
    phase before actually transitioning.
    """
    target = phase_for_r(r_ema)
    if target == current_phase:
        return current_phase, 0, False
    else:
        steps_toward_new += 1
        if steps_toward_new >= PHASE_HYSTERESIS:
            return target, 0, True
        return current_phase, steps_toward_new, False


def stratum_group_factors(strata: dict) -> dict[str, float]:
    """Compute per-group flip factors from stratum loss gaps.

    Maps stratum performance gaps to VSM group flip rates:
    - compositional_gap → stride_stack, consolidate (composition is routing)
    - abstract_gap → prep (abstraction is preprocessing)
    - Control groups always conservative.

    Returns {group_name: factor} where factor multiplies base_max_pct.
    """
    strata_r = {}
    for sname in ["prose", "compositional", "technical", "math"]:
        if sname in strata and "relational_loss" in strata[sname]:
            strata_r[sname] = strata[sname]["relational_loss"]

    if len(strata_r) < 4:
        # Not enough data — return neutral factors
        return {
            "prep": 1.0, "stride_stack": 1.0, "consolidate": 1.0,
            "mod_projs": 1.0, "s3": 0.5, "s4": 0.5, "meta": 0.3,
        }

    compositional_gap = strata_r["compositional"] - strata_r["prose"]
    abstract_gap = strata_r["math"] - strata_r["technical"]

    # Stride stack: compositional gap drives exploration
    if compositional_gap > 0.05:
        stride_factor = 1.0 + min(1.5, compositional_gap / 0.2)
        consolidate_factor = 1.0 + min(1.0, compositional_gap / 0.3)
    else:
        stride_factor = 0.7  # composing well → protect
        consolidate_factor = 0.7

    # Prep: abstract gap drives exploration
    if abstract_gap > 0.05:
        prep_factor = 1.0 + min(1.0, abstract_gap / 0.2)
    else:
        prep_factor = 0.7  # abstracting well → protect

    return {
        "prep": prep_factor,
        "stride_stack": stride_factor,
        "consolidate": consolidate_factor,
        "mod_projs": 1.0,
        "s3": 0.5,      # control: always conservative
        "s4": 0.5,
        "meta": 0.3,
    }


def compute_stratum_weights(strata: dict) -> dict[str, float]:
    """Compute per-stratum loss weights (upweight lagging strata).

    Weight ∝ stratum_r / mean_r, normalized so weights sum to N_STRATA.
    Higher r (worse performance) → higher weight → more gradient signal.

    Currently: logged only. Applying requires stratum-aware batching
    (shard metadata) or inline token classification (heuristic). Both
    are future work — the weight computation itself is the foundation.
    """
    strata_names = ["prose", "compositional", "technical", "math"]
    strata_r = {}
    for sn in strata_names:
        if sn in strata and "relational_loss" in strata[sn]:
            strata_r[sn] = strata[sn]["relational_loss"]

    if len(strata_r) < len(strata_names):
        return {sn: 1.0 for sn in strata_names}

    mean_r = sum(strata_r.values()) / len(strata_r)
    if mean_r < 1e-8:
        return {sn: 1.0 for sn in strata_names}

    weights = {sn: strata_r[sn] / mean_r for sn in strata_names}
    return weights


def stratum_loss_probe(model, tokenizer):
    """Measure loss per content stratum."""
    results = {}
    for sname, samples in STRATUM_SAMPLES.items():
        losses = []
        for text in samples:
            ids = mx.array(tokenizer.encode(text)).reshape(1, -1)
            if ids.shape[1] > model.max_len:
                ids = ids[:, -model.max_len:]
            targets = mx.concatenate([ids[:, 1:], mx.zeros((1, 1), dtype=mx.int32)], axis=1)
            _, ce_loss, _, _ = model(ids, targets)
            mx.eval(ce_loss)
            if ce_loss is not None:
                losses.append(ce_loss.item())
        if losses:
            mean_loss = sum(losses) / len(losses)
            rm = relational_metrics(mean_loss)
            results[sname] = {"loss": mean_loss, **rm}
    return results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    global N_PASSES, PASS_NAMES, PHASE_NAMES, REG_NAMES
    from transformers import AutoTokenizer

    # ── CLI ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="VSM-LM v6 training")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from (e.g. checkpoints/vsm-lm-v6/step_003500)",
    )
    args = parser.parse_args()

    results_dir = Path("results/vsm-lm-v6")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path("checkpoints/vsm-lm-v6")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mx.random.seed(SEED)
    np.random.seed(SEED)

    start = time.time()
    banner("VSM-LM v6 — Ternary on Metal (MLX)")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")

    tokens_total = N_STEPS * TOKENS_PER_STEP
    print(f"  Framework: MLX + custom Metal ternary matmul kernel")
    print(f"  Architecture: VSM-LM-v6 (ternary StrideStack + α={ALPHA})")
    print(f"  Passes: {N_PASSES} (L0↑, L1↑, L2, L1↓, L0↓)")
    print(f"  Strides: {STRIDES}")
    print(f"  Ternary: all projections (Metal add/sub kernel)")
    print(f"  Continuous: embeddings, gamma, norms, gates (AdamW)")
    print(f"  Flip policy: consensus={FLIP_CONSENSUS}, cap={FLIP_MAX_PCT*100:.1f}%, every {FLIP_INTERVAL} steps, probe every {FLIP_PROBE_INTERVAL}")
    print(f"  Flip mechanism: strongest consensus first, capped to prevent mass mutation")
    print(f"  φ-lambda: {PHI_LAMBDA} ({'Phase 1: observe only' if PHI_LAMBDA == 0 else f'active: CE + {PHI_LAMBDA}×φ_dev'})")
    print(f"  Embed norm: RMSNorm (constrains embedding scale)")
    print(f"  Seq len: {SEQ_LEN}, Batch: {BATCH_SIZE} × {GRAD_ACCUM} accum")
    print(f"  Steps: {N_STEPS}, Tokens: {tokens_total:,}")
    print(f"  Data: SHUFFLED", flush=True)

    # ── Build model ───────────────────────────────────────────────
    banner("BUILDING MODEL")

    model = VSMLMV6(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        d_register=D_REGISTER,
        max_len=SEQ_LEN,
        n_heads=N_HEADS,
        d_ff=D_FF,
        d_ff_consolidate=D_FF_CONSOLIDATE,
        window=WINDOW,
        strides=STRIDES,
        alpha=ALPHA,
        phi_lambda=PHI_LAMBDA,
    )

    print(model.describe())
    print()

    # Sync architecture constants from model (single source of truth)
    N_PASSES = model.N_PASSES
    PASS_NAMES = list(model.PASS_NAMES)
    PHASE_NAMES = list(model.PHASE_NAMES)
    REG_NAMES = list(model.REGISTER_NAMES)

    # Compute ternary weight count from model (not hardcoded)
    _n_ternary_weights = model.count_parameters()["total_ternary"]

    # Enable training metrics capture (lightweight, stop_gradient)
    model.capture_training_metrics = True

    # Precompute token-level stratum classification (once, at init)
    _stratum_token_sets = build_stratum_token_sets(tokenizer)
    _stratum_lookups = build_stratum_lookup(_stratum_token_sets, VOCAB_SIZE)
    print(f"  Stratum tokens: math={len(_stratum_token_sets['math'])} "
          f"tech={len(_stratum_token_sets['technical'])} "
          f"comp={len(_stratum_token_sets['compositional'])}", flush=True)

    ternary_stats_init = model.ternary_stats()
    n_ternary_modules = len(ternary_stats_init)
    if n_ternary_modules:
        avg_sparsity = sum(
            s["sparsity"] for s in ternary_stats_init.values()
        ) / n_ternary_modules
        print(f"  TernaryLinear modules: {n_ternary_modules}")
        print(f"  Ternary weights: {_n_ternary_weights:,}")
        print(f"  Initial avg sparsity: {avg_sparsity:.3f}", flush=True)

    # ── Resume from checkpoint ─────────────────────────────────────
    start_step = 0
    resumed_total_flips = 0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"  ✗ Resume path not found: {resume_path}")
            sys.exit(1)

        banner(f"RESUMING FROM {resume_path}")

        # Load metadata to get step and total_flips
        meta_path = resume_path / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                resume_meta = json.loads(f.read())
            start_step = resume_meta["step"]
            resumed_total_flips = resume_meta.get("total_flips", 0)
            print(f"  Step: {start_step}")
            print(f"  Train loss: {resume_meta.get('train_loss', 'N/A')}")
            print(f"  Eval loss: {resume_meta.get('eval_loss', 'N/A')}")
            print(f"  Total flips: {resumed_total_flips:,}")
        else:
            # Try to infer step from directory name
            try:
                start_step = int(resume_path.name.split("_")[-1])
            except ValueError:
                print(f"  ✗ Cannot determine step from {resume_path} (no meta.json)")
                sys.exit(1)
            print(f"  Step (inferred from dirname): {start_step}")

        # Load model weights
        weights_path = resume_path / "weights.safetensors"
        if weights_path.exists():
            model.load_weights(str(weights_path))
            print(f"  ✓ Model weights loaded")
        else:
            print(f"  ✗ No weights.safetensors in {resume_path}")
            sys.exit(1)

        # Zero flip accumulators on resume. The saved accumulators contain
        # gradient votes from the model's entire history, including early
        # requests the model already found continuous-parameter workarounds
        # for. Replaying that stale consensus would flip weights the model
        # no longer needs changed, disrupting the adapted topology. Fresh
        # accumulators let the current gradient signal drive flips based on
        # what the model needs NOW, not what it needed 3000 steps ago.
        print(f"  ✓ Flip accumulators zeroed (fresh consensus from current gradient)")

        print(f"  LR at step {start_step + 1}: {lr_schedule(start_step + 1):.2e}")
        print(flush=True)

    # ── Data ──────────────────────────────────────────────────────
    train_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, "train", seed=SEED)
    eval_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, "eval", seed=SEED + 1)

    # ── Optimizer (continuous params only) ─────────────────────────
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Restore optimizer state if resuming and state file exists
    if args.resume:
        opt_path = Path(args.resume) / "optimizer_state.npz"
        if opt_path.exists():
            from mlx.utils import tree_unflatten
            opt_loaded = dict(mx.load(str(opt_path)))
            opt_flat = list(opt_loaded.items())
            optimizer.state = tree_unflatten(opt_flat)
            print(f"  ✓ Optimizer state restored (Adam m_t, v_t)")
        else:
            # No optimizer state — need to prime Adam by doing one dummy step
            # so it initializes its state structure, then training proceeds normally.
            # Adam will reconverge its moments within ~100 steps.
            print(f"  ⚠ No optimizer_state.npz — Adam moments start fresh")
            print(f"    (Adam v_t reconverges within ~100 steps)")
        print(flush=True)

    # ── Loss + grad function ──────────────────────────────────────
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # ── Training ──────────────────────────────────────────────────
    banner("TRAINING" + (f" (resuming from step {start_step})" if start_step > 0 else ""))

    train_losses = []
    eval_losses = []
    total_flips = resumed_total_flips
    grad_norm = 0.0
    flips_since_last_probe = 0

    # ── Relational control state ──────────────────────────────
    r_ema = 1.0                          # start pessimistic (random)
    current_phase = PHASE_EXPLORE        # start in explore
    steps_toward_new_phase = 0           # hysteresis counter
    cached_group_factors = None          # stratum → group factors (updated at probe)
    cached_stratum_weights = None        # stratum weights (updated at eval)

    def _tree_add(a, b):
        """Add two gradient pytrees element-wise."""
        if isinstance(a, dict):
            return {k: _tree_add(a[k], b[k]) for k in a}
        elif isinstance(a, list):
            return [_tree_add(ai, bi) for ai, bi in zip(a, b)]
        else:
            return a + b

    def _tree_scale(tree, s):
        """Scale all arrays in a gradient pytree by scalar s."""
        if isinstance(tree, dict):
            return {k: _tree_scale(v, s) for k, v in tree.items()}
        elif isinstance(tree, list):
            return [_tree_scale(v, s) for v in tree]
        else:
            return tree * s

    for step in range(start_step + 1, N_STEPS + 1):
        step_loss = 0.0
        accum_grads = None

        for accum_idx in range(GRAD_ACCUM):
            x, y = train_loader.next_batch()

            # Loop 4: set per-sequence stratum weights for loss_fn
            # Pure tensor ops — no decoding, uses precomputed lookup arrays
            global _batch_seq_weights
            if cached_stratum_weights is not None:
                _batch_seq_weights = classify_batch_tokens(x, _stratum_lookups, cached_stratum_weights)
                mx.eval(_batch_seq_weights)
            else:
                _batch_seq_weights = None

            loss, grads = loss_and_grad_fn(model, x, y)

            # CRITICAL: evaluate both loss AND grads to materialize tensors
            # and free the computation graph. Without this, each micro-batch
            # retains the full forward+backward graph in memory → OOM.
            mx.eval(loss, grads)
            step_loss += loss.item() / GRAD_ACCUM

            # Route ternary grads to flip accumulator (per micro-batch)
            accumulate_flips(model, grads)

            # Accumulate gradients across micro-batches
            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = _tree_add(accum_grads, grads)
                mx.eval(accum_grads)  # prevent graph buildup in accumulator

        # Average accumulated gradients
        accum_grads = _tree_scale(accum_grads, 1.0 / GRAD_ACCUM)

        # NaN guard: skip optimizer step if loss is NaN
        if np.isnan(step_loss):
            print(f"  ⚠ step {step}: NaN loss, skipping optimizer update", flush=True)
            train_losses.append(step_loss)
            continue

        # Zero ternary weight gradients. They've already been consumed
        # by accumulate_flips (sign-based). Keeping them would pollute
        # Adam's statistics for continuous params.
        accum_grads = zero_ternary_grads(model, accum_grads)

        # Normalize shared-weight gradients by 1/N_PASSES.
        # Shared modules (prep, stride_stack, consolidate, mod_projs, s4)
        # accumulate gradient from 5 passes with VARYING ∂L/∂x magnitudes.
        # The sum oscillates 10⁴-10⁹ between steps, defeating Adam's v_t.
        # Dividing by 5 turns the volatile sum into a stable average.
        accum_grads = normalize_shared_grads(model, accum_grads, n_passes=N_PASSES)

        # NO gradient clipping. Adam handles per-parameter scale adaptation
        # via its second moment (v_t). Clipping at a fixed threshold creates
        # a scaling factor that varies by 10⁵× when ‖g‖ is unstable,
        # which destroys Adam's running statistics. LR warmup protects
        # early training while v_t converges.
        #
        # Compute grad norm for logging/diagnostics only.
        _, grad_norm = optim.clip_grad_norm(accum_grads, float('inf'))

        optimizer.learning_rate = lr_schedule(step)
        optimizer.update(model, accum_grads)
        # Restore int8 ternary weights (optimizer casts to float)
        restore_ternary(model)
        mx.eval(model.parameters())

        train_losses.append(step_loss)

        # ══════════════════════════════════════════════════════
        # RELATIONAL CONTROL: four interlocking feedback loops
        #
        # 1. r_ema: exponential moving average of relational loss
        # 2. Phase transitions: explore → balance → refine
        # 3. Adaptive flip scaling: r modulates consensus + cap
        # 4. Stratum-based group factors: target specific VSM groups
        #
        # effective_rate(group) = phase_base × r_scale × group_factor
        # ══════════════════════════════════════════════════════

        # ── Loop 1: update r_ema every step ──
        r = relational_metrics(step_loss)["relational_loss"]
        r_ema = 0.99 * r_ema + 0.01 * r

        # ── Loop 2: phase transition check ──
        new_phase, steps_toward_new_phase, did_transition = phase_transition(
            r_ema, current_phase, steps_toward_new_phase
        )
        if did_transition:
            current_phase = new_phase
            pcfg = PHASE_CONFIG[current_phase]
            model.phi_lambda = pcfg["phi_lambda"]
            print(
                f"\n  ══ PHASE TRANSITION → {current_phase.upper()} "
                f"(r_ema={r_ema:.3f}, φ-λ={pcfg['phi_lambda']}, "
                f"flip_scale={pcfg['flip_max_scale']}, "
                f"consensus_scale={pcfg['consensus_scale']}) ══\n",
                flush=True,
            )

        # ── Flip execution with relational modulation ──
        if step % FLIP_INTERVAL == 0:
            # Compose: phase base × r_scale
            pcfg = PHASE_CONFIG[current_phase]
            r_scale = adaptive_flip_scale(r_ema)
            effective_max_pct = FLIP_MAX_PCT * pcfg["flip_max_scale"] * r_scale
            effective_consensus = FLIP_CONSENSUS * pcfg["consensus_scale"] / r_scale
            effective_consensus = int(max(10, min(127, effective_consensus)))
            effective_max_pct = max(0.0001, min(0.01, effective_max_pct))

            n_flipped = apply_flips(model, threshold=effective_consensus, max_flip_pct=effective_max_pct)
            total_flips += n_flipped
            flips_since_last_probe += n_flipped

            # ── Probe step: use training-pass metrics (no extra forward pass) ──
            if step % FLIP_PROBE_INTERVAL == 0:
                pct_flipped = flips_since_last_probe / _n_ternary_weights * 100

                # Read metrics captured during the training forward pass
                tm = getattr(model, "_training_metrics", None)
                phi_msg = ""
                if tm and tm.get("compression_ratios"):
                    crs = [cr.item() for cr in tm["compression_ratios"]]
                    mean_phi_dev = sum(abs(cr - INV_PHI) for cr in crs) / len(crs)
                    phi_msg = f"φ-dev={mean_phi_dev:.4f}"

                    # Log meta gates
                    mg = [g.item() for g in tm["meta_gates"]]
                    mg_parts = [f"{p}={g:.2f}" for p, g in zip(PASS_NAMES, mg)]
                    # Log compression ratios
                    cr_parts = [f"{p}={cr:.3f}" for p, cr in zip(PASS_NAMES, crs)]
                else:
                    phi_msg = "φ-dev=N/A"

                # Loop 3: update stratum-based group factors (still uses probe
                # for stratum loss — this runs on fixed samples, not training batch)
                strata_probe = stratum_loss_probe(model, tokenizer)
                if strata_probe:
                    cached_group_factors = stratum_group_factors(strata_probe)

                print(
                    f"  ── flip probe @ step {step}: {flips_since_last_probe:,} flips "
                    f"({pct_flipped:.3f}%) since last probe  "
                    f"total={total_flips:,}  {phi_msg}  "
                    f"r_ema={r_ema:.3f}  phase={current_phase}  "
                    f"eff_con={effective_consensus}  eff_pct={effective_max_pct:.4f} ──",
                    flush=True,
                )
                if cached_group_factors:
                    gf_parts = [f"{g}={f:.2f}" for g, f in sorted(cached_group_factors.items())]
                    print(f"  ── group factors: {' '.join(gf_parts)} ──", flush=True)

                flips_since_last_probe = 0

        # ── Logging ───────────────────────────────────────────
        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - start
            total_tokens = step * TOKENS_PER_STEP
            tps = total_tokens / elapsed
            pct = total_tokens / TARGET_TOKENS * 100
            rm = relational_metrics(step_loss)
            print(
                f"  step {step:5d}/{N_STEPS}  "
                f"loss={step_loss:.4f}  "
                f"r={rm['relational_loss']:.3f}  "
                f"r̄={r_ema:.3f}  "
                f"xppl={rm['excess_ppl']:.1f}  "
                f"lr={lr_schedule(step):.2e}  "
                f"‖g‖={grad_norm:.2f}  "
                f"flips={total_flips:,}  "
                f"phase={current_phase[0]}  "
                f"tokens={total_tokens/1e6:.0f}M ({pct:.0f}%)  "
                f"tok/s={tps:.0f}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

        # ── Eval ──────────────────────────────────────────────
        if step % EVAL_INTERVAL == 0:
            eval_loader.reset()
            el = estimate_loss(model, eval_loader)
            erm = relational_metrics(el)
            eval_losses.append({"step": step, "loss": el, **erm})
            print(
                f"  ── eval @ step {step}: loss={el:.4f}  "
                f"r={erm['relational_loss']:.3f}  "
                f"xppl={erm['excess_ppl']:.1f}  "
                f"ppl={erm['ppl']:.1f} ──",
                flush=True,
            )

            # φ-compression probe (per-pass ratios, gates, Hilberg)
            phi = phi_compression_probe(model, tokenizer)
            if phi:
                parts = []
                for p in PASS_NAMES:
                    if p in phi:
                        cr = phi[p]
                        marker = "←φ" if abs(cr - INV_PHI) < 0.05 else ""
                        parts.append(f"{p}={cr:.3f}{marker}")
                mean_cr = phi.get("mean", 0)
                mean_pd = phi.get("mean_phi_dev", 0)
                print(
                    f"  ── φ-compression: {' '.join(parts)}  "
                    f"mean={mean_cr:.3f}  φ-dev={mean_pd:.3f}  (1/φ={INV_PHI:.3f}) ──",
                    flush=True,
                )

                # Gate trajectory (3 phases × 5 passes = 15 values)
                gates = phi.get("gates", {})
                if gates:
                    gate_parts = []
                    for p in PASS_NAMES:
                        p_gates = [gates.get(f"{p}_{ph}", 0) for ph in PHASE_NAMES]
                        gate_parts.append(f"{p}=[{' '.join(f'{g:.2f}' for g in p_gates)}]")
                    print(
                        f"  ── gates (prep/conv/cons): {' '.join(gate_parts)} ──",
                        flush=True,
                    )

                # Hilberg β per pass
                hilberg = phi.get("hilberg", {})
                if hilberg:
                    hparts = []
                    for p in PASS_NAMES:
                        if p in hilberg:
                            h = hilberg[p]
                            # hilberg dict now has {pass: {"slope": s, "beta": b}} or just beta
                            if isinstance(h, dict):
                                β = h.get("beta", h.get("slope", 0) + 1)
                            else:
                                β = h + 1  # legacy: stored slope, convert to β
                            marker = "←!" if abs(β - 0.5) < 0.1 else ""
                            hparts.append(f"{p}:β={β:.2f}{marker}")
                    if hparts:
                        print(
                            f"  ── hilberg (β≈0.5 = self-similar): {' '.join(hparts)} ──",
                            flush=True,
                        )

            # Per-stratum loss
            strata = stratum_loss_probe(model, tokenizer)
            if strata:
                sparts = []
                for sn in ["prose", "compositional", "technical", "math"]:
                    if sn in strata:
                        s = strata[sn]
                        sparts.append(f"{sn}={s['loss']:.3f}(r={s['relational_loss']:.3f})")
                if sparts:
                    vals = [strata[sn]["loss"] for sn in strata]
                    spread = max(vals) - min(vals)
                    print(
                        f"  ── stratum loss: {' '.join(sparts)}  spread={spread:.3f} ──",
                        flush=True,
                    )
                    # Loop 4: log stratum weights
                    if cached_stratum_weights:
                        sw_parts = [f"{sn}={cached_stratum_weights.get(sn, 1.0):.2f}" for sn in ["prose", "compositional", "technical", "math"]]
                        print(f"  ── stratum weights: {' '.join(sw_parts)} ──", flush=True)

        # ── Checkpoint ────────────────────────────────────────
        if step % CHECKPOINT_INTERVAL == 0:
            compile = compile_gate_test(model, tokenizer)
            ternary_stats = model.ternary_stats()

            print(f"  ── checkpoint {step} ({step * TOKENS_PER_STEP / 1e6:.0f}M tokens) ──")
            print(f"     compile gate: {compile['score']}")
            print(f"     total flips: {total_flips:,} ({total_flips / _n_ternary_weights * 100:.1f}% cumulative)  consensus={FLIP_CONSENSUS}")
            print(f"     relational: r_ema={r_ema:.3f}  phase={current_phase}  r_scale={adaptive_flip_scale(r_ema):.2f}")

            # Ternary stats by group (using canonical _classify_group)
            group_stats: dict[str, list] = {}
            for mod_name, stat in ternary_stats.items():
                group = _classify_group(mod_name)
                group_stats.setdefault(group, []).append(stat)

            for grp, stat_list in group_stats.items():
                if not stat_list:
                    continue
                avg_sp = sum(s["sparsity"] for s in stat_list) / len(stat_list)
                avg_gm = sum(s["gamma_mean"] for s in stat_list) / len(stat_list)
                print(f"     {grp:15s}: sparsity={avg_sp:.3f}  gamma={avg_gm:.4f}  ({len(stat_list)} modules)")

            # φ-compression at checkpoint
            phi_ckpt = phi_compression_probe(model, tokenizer)
            if phi_ckpt:
                parts = []
                for p in PASS_NAMES:
                    if p in phi_ckpt:
                        cr = phi_ckpt[p]
                        marker = "←φ" if abs(cr - INV_PHI) < 0.05 else ""
                        parts.append(f"{p}={cr:.3f}{marker}")
                print(f"     φ-compression: {' '.join(parts)}  mean={phi_ckpt.get('mean', 0):.3f}  φ-dev={phi_ckpt.get('mean_phi_dev', 0):.3f}")
                # Gate values
                gates = phi_ckpt.get("gates", {})
                if gates:
                    gate_parts = []
                    for p in PASS_NAMES:
                        p_gates = [gates.get(f"{p}_{ph}", 0) for ph in PHASE_NAMES]
                        gate_parts.append(f"{p}=[{' '.join(f'{g:.2f}' for g in p_gates)}]")
                    print(f"     gates: {' '.join(gate_parts)}")
                # Hilberg β
                hilberg = phi_ckpt.get("hilberg", {})
                if hilberg:
                    hparts = []
                    for p in PASS_NAMES:
                        if p in hilberg:
                            h = hilberg[p]
                            β = h["beta"] if isinstance(h, dict) else h + 1
                            hparts.append(f"{p}:β={β:.2f}")
                    if hparts:
                        print(f"     hilberg: {' '.join(hparts)}")

            # Per-stratum loss at checkpoint + Loop 4: stratum weights
            strata_ckpt = stratum_loss_probe(model, tokenizer)
            if strata_ckpt:
                cached_stratum_weights = compute_stratum_weights(strata_ckpt)
            if strata_ckpt:
                sparts = [f"{sn}={strata_ckpt[sn]['loss']:.3f}" for sn in ["prose", "compositional", "technical", "math"] if sn in strata_ckpt]
                if sparts:
                    print(f"     stratum loss: {' '.join(sparts)}")

            # Save checkpoint as safetensors + metadata JSON
            ckpt_path = checkpoint_dir / f"step_{step:06d}"
            ckpt_path.mkdir(exist_ok=True)

            # Save model weights
            model.save_weights(str(ckpt_path / "weights.safetensors"))

            # Save flip accumulators (using _walk_ternary_modules for correct traversal)
            accum_dict = {}
            for path, mod in _walk_ternary_modules(model):
                accum_dict[path] = mod._flip_accum
            if accum_dict:
                mx.savez(str(ckpt_path / "flip_accum.npz"), **accum_dict)

            # Save optimizer state (Adam m_t, v_t for warm resume)
            from mlx.utils import tree_flatten
            opt_flat = tree_flatten(optimizer.state)
            if opt_flat:
                opt_dict = {k: v for k, v in opt_flat}
                mx.savez(str(ckpt_path / "optimizer_state.npz"), **opt_dict)

            # Save metadata
            rm = relational_metrics(step_loss)
            _gn = float(grad_norm.item()) if hasattr(grad_norm, 'item') else float(grad_norm)
            meta = {
                "step": step,
                "train_loss": float(step_loss),
                "relational_loss": float(rm["relational_loss"]),
                "excess_ppl": float(rm["excess_ppl"]),
                "ppl": float(rm["ppl"]),
                "reducible_loss": float(rm["reducible_loss"]),
                "eval_loss": float(eval_losses[-1]["loss"]) if eval_losses else None,
                "compile_gate": compile["score"],
                "total_flips": int(total_flips),
                "flip_consensus": FLIP_CONSENSUS,
                "grad_norm": _gn,
                "architecture": "vsm-lm-v6-mlx",
                "config": {
                    "d_model": D_MODEL, "d_register": D_REGISTER,
                    "d_ff": D_FF, "d_ff_consolidate": D_FF_CONSOLIDATE,
                    "n_heads": N_HEADS, "strides": list(STRIDES),
                    "window": WINDOW, "vocab_size": VOCAB_SIZE,
                    "seq_len": SEQ_LEN, "alpha": ALPHA,
                    "n_passes": N_PASSES,
                    "pass_names": PASS_NAMES,
                    "phase_names": PHASE_NAMES,
                    "reg_names": REG_NAMES,
                    "total_ternary_weights": _n_ternary_weights,
                },
                "ternary_stats_summary": {
                    grp: {
                        "n_modules": len(sl),
                        "avg_sparsity": sum(s["sparsity"] for s in sl) / len(sl),
                        "avg_gamma": sum(s["gamma_mean"] for s in sl) / len(sl),
                    }
                    for grp, sl in group_stats.items() if sl
                },
                "phi_compression": phi_ckpt if phi_ckpt else None,
                "stratum_loss": strata_ckpt if strata_ckpt else None,
            }
            (ckpt_path / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"     saved: {ckpt_path}", flush=True)

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - start
    banner(f"DONE — {elapsed:.0f}s ({elapsed / 3600:.1f}h)")

    # Compute final relational metrics
    final_rm = relational_metrics(train_losses[-1]) if train_losses else {}
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "architecture": "VSM-LM-v6 (MLX, Metal ternary kernel)",
        "framework": "MLX",
        "target_tokens": TARGET_TOKENS,
        "total_flips": total_flips,
        "total_ternary_weights": _n_ternary_weights,
        "pct_weights_ever_flipped": total_flips / _n_ternary_weights * 100,
        "info_theoretic_constants": {
            "E_irreducible": E_IRREDUCIBLE,
            "log_V": LOG_V,
            "learnable_range": LEARNABLE_RANGE,
            "phi": PHI,
            "inv_phi": INV_PHI,
            "note": "E from Chinchilla (Hoffmann 2022). φ hypothesis: true H ≈ 1/φ bits/char (Hilberg 1990 self-similarity).",
        },
        "final_relational": final_rm,
        "train_losses": train_losses,
        "eval_losses": eval_losses,
    }
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Saved: {summary_path}")

    print()
    print("  Reference:")
    print("    VSM-LM v1:   best eval 5.245 @ step 9500")
    print("    VSM-LM v2:   best eval 5.064 @ step 29500 (1B tokens)")
    print("    VSM-LM v3:   best eval 4.872 @ step 10000")
    print("    VSM-LM v4:   best eval 4.713 @ step 16000")
    print("    VSM-LM v4.1: best eval 4.728 @ step 15000")
    print("    VSM-LM v5:   TBD (training)")
    print()
    if eval_losses:
        best = min(eval_losses, key=lambda e: e["loss"])
        tokens_at_best = best["step"] * TOKENS_PER_STEP
        print(f"  This run (VSM-LM-v6, MLX + Metal ternary):")
        print(f"    Best eval: {best['loss']:.3f} @ step {best['step']} ({tokens_at_best/1e6:.0f}M tokens)")


if __name__ == "__main__":
    main()
