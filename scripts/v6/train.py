#!/usr/bin/env python3
"""VSM-LM v6 — Ternary on Metal, 1B token training run.

MLX implementation with custom Metal ternary matmul kernels.
All ternary weights learn through flip accumulation (not Adam).
Continuous params (gamma, embeddings, norms, gates) use AdamW.

Usage:
    uv run python scripts/v6/train.py
"""

from __future__ import annotations

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
    apply_flips_per_group,
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

FLIP_INTERVAL = 25        # cheap: just apply flips from cached group targets
FLIP_PROBE_INTERVAL = 100 # expensive: re-run VSM probes to update group targets + stability/φ feedback
FLIP_TARGET_PCT = 0.005   # start: 0.5% of weights per flip interval
FLIP_PCT_MIN = 0.0001     # floor: 0.01%
FLIP_PCT_MAX = 0.02       # ceiling: 2%
MAX_GRAD_NORM = 1.0       # global clip after ternary grads zeroed — safe now that they don't pollute the norm

# Phase 1: observe φ-compression (lambda=0.0, no gradient pressure)
# Phase 2: gentle φ-pressure (lambda=0.01-0.1, test effect on convergence)
# Phase 3: full φ-regulation (lambda tuned from Phase 2 findings)
PHI_LAMBDA = 0.0

# φ-feedback on flip rate only activates below this loss. Above it,
# compression ratios are meaningless noise — the model hasn't learned
# enough structure for φ-deviation to be a real signal. Flips run at
# the base rate to explore topology freely during early training.
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


def loss_fn(model, x, y):
    """Compute combined loss. Used with nn.value_and_grad.

    Returns ce_loss + PHI_LAMBDA * phi_loss (when phi_lambda > 0).
    """
    _, ce_loss, phi_loss = model(x, y)
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
        _, ce_loss, _ = model(x, y)
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
            _, ce_loss, _ = model(ids, targets)
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
    print(f"  Flip accumulation: apply every {FLIP_INTERVAL} steps, probe every {FLIP_PROBE_INTERVAL} steps")
    print(f"  Flip policy: VSM-signal inversion + stratum/Hilberg corrections")
    print(f"  φ-lambda: {PHI_LAMBDA} ({'Phase 1: observe only' if PHI_LAMBDA == 0 else f'active: CE + {PHI_LAMBDA}×φ_dev'})")
    print(f"  Embed norm: RMSNorm (internalizes grad clip constraint)")
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

    ternary_stats_init = model.ternary_stats()
    n_ternary_modules = len(ternary_stats_init)
    if n_ternary_modules:
        avg_sparsity = sum(
            s["sparsity"] for s in ternary_stats_init.values()
        ) / n_ternary_modules
        print(f"  TernaryLinear modules: {n_ternary_modules}")
        print(f"  Ternary weights: {_n_ternary_weights:,}")
        print(f"  Initial avg sparsity: {avg_sparsity:.3f}", flush=True)

    # ── Data ──────────────────────────────────────────────────────
    train_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, "train", seed=SEED)
    eval_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, "eval", seed=SEED + 1)

    # ── Optimizer (continuous params only) ─────────────────────────
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ── Loss + grad function ──────────────────────────────────────
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # ── Training ──────────────────────────────────────────────────
    banner("TRAINING")

    train_losses = []
    eval_losses = []
    total_flips = 0
    grad_norm = 0.0
    flip_target_pct = FLIP_TARGET_PCT
    # Cached group targets for cheap flip steps (updated every FLIP_PROBE_INTERVAL)
    cached_group_targets = {g: FLIP_TARGET_PCT for g in
                            ("prep", "stride_stack", "consolidate", "mod_projs", "s3", "s4", "meta")}

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

    for step in range(1, N_STEPS + 1):
        step_loss = 0.0
        accum_grads = None

        for accum_idx in range(GRAD_ACCUM):
            x, y = train_loader.next_batch()
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

        # Zero ternary weight gradients before clipping. They've already
        # been consumed by accumulate_flips (sign-based). Including them
        # in clip_grad_norm would clip continuous params to near-zero
        # because ternary grads sum over B×L positions without normalization.
        accum_grads = zero_ternary_grads(model, accum_grads)

        # Global gradient clipping. Now safe because ternary grads are
        # already zeroed above — only continuous params contribute to the
        # norm. This preserves gradient geometry (relative scale across
        # params) unlike per-param clipping which distorts it.
        accum_grads, grad_norm = optim.clip_grad_norm(accum_grads, MAX_GRAD_NORM)

        optimizer.learning_rate = lr_schedule(step)
        optimizer.update(model, accum_grads)
        # Restore int8 ternary weights (optimizer casts to float)
        restore_ternary(model)
        mx.eval(model.parameters())

        train_losses.append(step_loss)

        # ══════════════════════════════════════════════════════
        # FLIP: Three-level VSM-regulated control
        #
        # Level 1 (S3 feed-forward): VSM signals → per-group flip targets
        #   Runs BEFORE flips. S3/Meta-S3 gates modulate where flips
        #   happen. High importance → protect, low → explore.
        #
        # Level 2 (local stability): VSM signal diff after flips
        #   Immediate check. If VSM signals stayed coherent (cosine sim
        #   > threshold), the system self-regulated. No escalation.
        #
        # Level 3 (circuit breaker): Global loss ratio at step+25
        #   Only fires if Level 2 detected instability. Emergency
        #   adjustment of the global base flip rate.
        # ══════════════════════════════════════════════════════

        if step % FLIP_INTERVAL == 0:
            # ══════════════════════════════════════════════════
            # Two-tier flip control:
            #
            # Every FLIP_INTERVAL (25 steps): apply flips using cached
            #   group_targets. Cheap — just percentile + mx.where.
            #
            # Every FLIP_PROBE_INTERVAL (100 steps): re-run full VSM
            #   probe pipeline to update group_targets, check stability,
            #   and adjust flip_target_pct via φ-feedback. Expensive
            #   (13 forward passes) but only 1/4 as often as flips.
            # ══════════════════════════════════════════════════

            is_probe_step = (step % FLIP_PROBE_INTERVAL == 0)

            if is_probe_step:
                # ── Full probe: update group targets + stability/φ feedback ──
                signals_before, vec_before = vsm_probe(model, tokenizer)
                phi_dev_before = signals_before.get("phi_deviation")

                # Compute stratum spread and Hilberg β for flip routing
                flip_strata = stratum_loss_probe(model, tokenizer)
                stratum_spread = 0.0
                if flip_strata and "compositional" in flip_strata and "prose" in flip_strata:
                    stratum_spread = flip_strata["compositional"]["loss"] - flip_strata["prose"]["loss"]

                flip_phi = phi_compression_probe(model, tokenizer)
                hilberg_beta_dev = 0.0
                if flip_phi:
                    hilberg = flip_phi.get("hilberg", {})
                    betas = []
                    for p in PASS_NAMES:
                        if p in hilberg:
                            h = hilberg[p]
                            b = h["beta"] if isinstance(h, dict) else h + 1
                            betas.append(b)
                    if betas:
                        mean_beta = sum(betas) / len(betas)
                        hilberg_beta_dev = abs(mean_beta - 0.5)

                # VSM signal inversion → per-group targets
                cached_group_targets = compute_per_group_flip_targets(
                    signals_before, flip_target_pct,
                    stratum_spread=stratum_spread,
                    hilberg_beta_dev=hilberg_beta_dev,
                )

            # ── Apply flips using cached targets ──────────────
            group_flips = apply_flips_per_group(model, cached_group_targets)
            n_flipped = sum(group_flips.values())
            total_flips += n_flipped
            pct_flipped = n_flipped / _n_ternary_weights * 100

            if is_probe_step:
                # ── Stability check + φ-feedback (probe steps only) ──
                signals_after, vec_after = vsm_probe(model, tokenizer)
                stability = vsm_stability(vec_before, vec_after)
                phi_dev_after = signals_after.get("phi_deviation")

                if stability > 0.95:
                    level_msg = "L1:self-regulated"
                elif stability > 0.80:
                    level_msg = f"L2:mild-perturbation(sim={stability:.3f})"
                else:
                    level_msg = f"L2:DESTABILIZED(sim={stability:.3f})"

                # φ-deviation feedback
                phi_msg = ""
                phi_feedback_active = (
                    phi_dev_before is not None
                    and phi_dev_after is not None
                    and step_loss < PHI_FEEDBACK_LOSS
                )
                if phi_dev_before is not None and phi_dev_after is not None:
                    delta_phi = phi_dev_after - phi_dev_before
                    if not phi_feedback_active:
                        phi_msg = f"  φ~gated(loss={step_loss:.2f}>{PHI_FEEDBACK_LOSS})"
                    elif delta_phi < -0.01:
                        flip_target_pct = min(flip_target_pct * 1.2, FLIP_PCT_MAX)
                        phi_msg = f"  φ↓ good(Δ={delta_phi:+.4f}) target↑{flip_target_pct:.4f}"
                    elif delta_phi > 0.05:
                        flip_target_pct = max(flip_target_pct * 0.5, FLIP_PCT_MIN)
                        phi_msg = f"  φ↑ BAD(Δ={delta_phi:+.4f}) target↓{flip_target_pct:.4f}"
                    else:
                        phi_msg = f"  φ~neutral(Δ={delta_phi:+.4f})"

                    if stability < 0.80 and delta_phi > 0.02:
                        flip_target_pct = max(flip_target_pct * 0.3, FLIP_PCT_MIN)
                        phi_msg += f"  ⚠ BRAKE→{flip_target_pct:.4f}"

                # Full diagnostic output
                flip_parts = " ".join(f"{g}={c:,}" for g, c in group_flips.items() if c > 0)
                target_parts = " ".join(f"{g}={t:.4f}" for g, t in cached_group_targets.items() if group_flips.get(g, 0) > 0)
                if phi_dev_before is not None and phi_dev_after is not None:
                    print(
                        f"  ── flip @ step {step}: {n_flipped:,} ({pct_flipped:.3f}%)  "
                        f"stability={stability:.3f}  {level_msg}{phi_msg}\n"
                        f"     groups=[{flip_parts}]\n"
                        f"     targets=[{target_parts}]\n"
                        f"     φ-dev: {phi_dev_before:.4f}→{phi_dev_after:.4f} ──",
                        flush=True,
                    )
                else:
                    print(
                        f"  ── flip @ step {step}: {n_flipped:,} ({pct_flipped:.3f}%)  "
                        f"stability={stability:.3f}  {level_msg}\n"
                        f"     groups=[{flip_parts}]\n"
                        f"     targets=[{target_parts}] ──",
                        flush=True,
                    )
            else:
                # Lightweight log for non-probe flips
                print(
                    f"  ── flip @ step {step}: {n_flipped:,} ({pct_flipped:.3f}%)  "
                    f"target={flip_target_pct:.4f} ──",
                    flush=True,
                )

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
                f"xppl={rm['excess_ppl']:.1f}  "
                f"lr={lr_schedule(step):.2e}  "
                f"‖g‖={grad_norm:.2f}  "
                f"flips={total_flips:,}  "
                f"target={flip_target_pct:.4f}  "
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

        # ── Checkpoint ────────────────────────────────────────
        if step % CHECKPOINT_INTERVAL == 0:
            compile = compile_gate_test(model, tokenizer)
            ternary_stats = model.ternary_stats()

            print(f"  ── checkpoint {step} ({step * TOKENS_PER_STEP / 1e6:.0f}M tokens) ──")
            print(f"     compile gate: {compile['score']}")
            print(f"     total flips: {total_flips:,} ({total_flips / _n_ternary_weights * 100:.1f}% cumulative)  target={flip_target_pct:.4f}")

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

            # Per-stratum loss at checkpoint
            strata_ckpt = stratum_loss_probe(model, tokenizer)
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
                "flip_target_pct": float(flip_target_pct),
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
