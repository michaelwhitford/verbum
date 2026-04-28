"""
v8 — Dual MERA Training Loop

Two phase modes:
  bios:  BIOS flash burn-in on math + clojure data (1 shard, many epochs)
  dolma: Prose training on Dolma (60 shards, 1-2 epochs)

BIOS flash burns computation circuits into the deepest ternary levels.
Dolma adds prose capacity on top of frozen circuits.

Usage:
    cd ~/src/verbum
    uv run python scripts/v8/train.py --phase bios
    uv run python scripts/v8/train.py --phase dolma --resume checkpoints/v8-bios/step_050000
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

sys.path.insert(0, str(Path(__file__).parent))
from model import DualMERA, DualMERAConfig, create_model
from ternary import (
    TernaryLinear,
    zero_ternary_grads,
    restore_ternary,
    save_ternary_state,
    load_ternary_state,
    count_ternary_weights,
    mutation_cone,
    bios_mutation_budget,
    save_topology,
    load_topology,
    mutate_topology,
    _walk_ternary_modules,
)
from tokenizer import VOCAB_SIZE, EOD_ID
from compute_probe import run_computation_probe, print_probe_results


# ═══════════════════════════════════════════════════════════════════
# Phase configurations
# ═══════════════════════════════════════════════════════════════════

PHASE_DEFAULTS = {
    "bios": {
        "data_dir": "/Users/mwhitford/data/fractal-bitnet/shards-bios",
        "checkpoint_dir": "checkpoints/v8-bios",
        "seq_len": 512,
        "batch_size": 16,
        "grad_accum": 2,
        "lr": 3e-4,
        "warmup": 500,
        "steps": 50_000,
        "eval_interval": 1000,
        "eval_batches": 5,
        "checkpoint_interval": 2500,
        "log_interval": 50,
        "gen_interval": 50,          # evolutionary generation interval
        "gen_base_pct": 0.005,       # mutation rate during BIOS exploration (0.5%)
        "gen_n_mutants": 4,          # population size per generation
        "gen_circuit_bonus": 0.5,    # fitness bonus scale for probe accuracy
        "gen_sign_flip_rate": 0.2,   # fraction of non-zero mutations that flip sign
    },
    "dolma": {
        "data_dir": "/Users/mwhitford/data/fractal-bitnet/shards-qwen3",
        "checkpoint_dir": "checkpoints/v8-dolma",
        "seq_len": 4096,
        "batch_size": 4,
        "grad_accum": 8,
        "lr": 1e-4,
        "warmup": 1000,
        "steps": 165_000,
        "eval_interval": 2500,
        "eval_batches": 10,
        "checkpoint_interval": 10000,
        "log_interval": 100,
        "gen_interval": 200,         # slower evolution — topology mostly frozen
        "gen_base_pct": 0.0002,      # narrow cone — protect BIOS circuits
        "gen_n_mutants": 4,
        "gen_circuit_bonus": 1.0,    # strong circuit protection during Dolma
        "gen_sign_flip_rate": 0.2,
    },
}


# ═══════════════════════════════════════════════════════════════════
# BIOS depth-weighted mutation priorities
# ═══════════════════════════════════════════════════════════════════
#
# During BIOS burn-in, concentrate mutations where circuits need to form.
# Pipeline shared level (reused at every depth) and feedbacks get highest
# priority. Embedding gets minimal mutations — it's 156M params of token
# lookup, not computation.

BIOS_DEPTH_WEIGHTS = {
    "compressor.embed":       0.1,   # token lookup — barely touch
    "compressor.level0":      0.3,   # surface routing
    "compressor.shared":      0.3,   # deep compressor routing
    "compressor.reducer":     0.5,   # inter-level pooling
    "pipeline.level0":        1.0,   # surface computation
    "pipeline.shared":        2.0,   # deep computation — HIGHEST priority
    "pipeline.reducer":       1.0,   # inter-level pooling
    "pipeline.feedback":      1.5,   # constraint propagation (feedback cascade)
}

# Dolma: no depth weighting — uniform proportional (protect everything equally)
DOLMA_DEPTH_WEIGHTS = None


# ═══════════════════════════════════════════════════════════════════
# Information-theoretic landmarks
# ═══════════════════════════════════════════════════════════════════

LOG_V = float(np.log(VOCAB_SIZE))       # ~11.93 nats for Qwen3 vocab
E_IRREDUCIBLE = 1.69                     # irreducible entropy floor (prose)
LEARNABLE_RANGE = LOG_V - E_IRREDUCIBLE  # ~10.24 nats


def relational_loss(loss: float) -> float:
    """Dimensionless position in the learnable range [0, 1].
    r=1.0 → uniform random, r=0.0 → irreducible floor.
    """
    return min(1.0, max(0.0, (loss - E_IRREDUCIBLE) / LEARNABLE_RANGE))


# ═══════════════════════════════════════════════════════════════════
# Teacher-forced circuit probe for tournament fitness
# ═══════════════════════════════════════════════════════════════════

def run_teacher_forced_probe(
    model: DualMERA, seq_len: int, seed: int, n_examples: int = 10,
) -> float:
    """Fast circuit probe via teacher forcing — single batched forward pass.

    Instead of autoregressive decode (150 sequential forward passes),
    feeds prompt+answer as input and checks if logits at each answer
    position have the correct next token as argmax.

    An example is "correct" if ALL answer tokens are predicted correctly
    at every position (teacher-forced exact match).

    Cost: 1 forward pass at batch=n_examples ≈ 130ms
    vs autoregressive: 150 forward passes at batch=1 ≈ 9500ms

    Args:
        model:      DualMERA model
        seq_len:    model sequence length
        seed:       random seed for example generation
        n_examples: number of tier-1 examples (default 10)

    Returns:
        Accuracy as float [0, 1].
    """
    import random as stdlib_random
    from compute_probe import _gen_tier1
    from tokenizer import encode, PAD_ID

    rng = stdlib_random.Random(seed)
    examples = _gen_tier1(rng, n=n_examples)[:n_examples]

    # Tokenize each prompt+answer pair and track answer boundaries.
    # BPE may re-segment at the prompt/answer boundary, so we tokenize
    # the full string and find the answer span from the end.
    sequences = []   # (full_ids, n_answer_tokens)
    for prompt, expected, _tier, _op in examples:
        full_text = prompt + expected
        full_ids = encode(full_text)
        answer_ids = encode(expected)

        # The answer tokens are at the END of full_ids.
        # Due to BPE merging at the boundary, full_ids[-len(answer_ids):]
        # may not equal answer_ids. So we count answer tokens by encoding
        # just the answer and using that length as the span from the end.
        # This is correct even if BPE merges boundary tokens differently,
        # because we check against full_ids (the ground truth tokenization).
        n_ans = len(answer_ids)

        # Clamp to seq_len (leave room for at least 1 prompt token)
        if len(full_ids) > seq_len:
            full_ids = full_ids[:seq_len]
            n_ans = min(n_ans, seq_len - 1)

        if n_ans < 1:
            continue

        sequences.append((full_ids, n_ans))

    if not sequences:
        return 0.0

    B = len(sequences)

    # Pad all sequences to seq_len (model requires exact seq_len for MERA structure).
    # Left-pad with PAD tokens so answer tokens are right-aligned.
    import numpy as np_
    batch = np_.full((B, seq_len), PAD_ID, dtype=np_.int64)
    for i, (ids, _) in enumerate(sequences):
        L = len(ids)
        batch[i, seq_len - L :] = ids

    # Forward pass: logits[b, t] predicts token at position t+1
    tokens = mx.array(batch, dtype=mx.int32)
    logits = model(tokens)
    mx.eval(logits)

    # Check answer tokens: for each example, the answer occupies the
    # last n_ans tokens of the padded sequence. To predict token at position j,
    # we check argmax(logits[b, j-1]). So for answer tokens at positions
    # [seq_len - n_ans, seq_len), we check logits at [seq_len - n_ans - 1, seq_len - 1).
    correct = 0
    for i, (ids, n_ans) in enumerate(sequences):
        # Answer tokens are at batch positions [seq_len - n_ans, seq_len)
        # The logit that predicts batch[i, j] is logits[i, j-1]
        all_match = True
        for k in range(n_ans):
            pos = seq_len - n_ans + k       # position of answer token k
            target_token = batch[i, pos]
            predicted = int(mx.argmax(logits[i, pos - 1]).item())
            if predicted != target_token:
                all_match = False
                break
        if all_match:
            correct += 1

    return correct / B


# ═══════════════════════════════════════════════════════════════════
# Evolutionary tournament
# ═══════════════════════════════════════════════════════════════════

# Mutant strategies: each scales the base budget differently.
# Conservative explores less, aggressive explores more.
# All strategies are evaluated and the best survives.
MUTANT_STRATEGIES = {
    "conservative": 0.25,
    "standard":     1.0,
    "aggressive":   2.0,
    "explorer":     4.0,
}

# Strategy win tracking for adaptive mutation rate
_strategy_history: list[str | None] = []
_STRATEGY_WINDOW = 20


def _adapt_base_pct(base_pct: float, phase: str) -> tuple[float, str | None]:
    """Adapt mutation rate based on which strategies are winning.

    If explorer wins >50% of the last 20 generations, the model wants
    more exploration → increase base_pct.
    If conservative wins >50%, the model is near a good topology →
    decrease base_pct.

    Returns (new_base_pct, adaptation_reason_or_None).
    """
    if len(_strategy_history) < _STRATEGY_WINDOW:
        return base_pct, None

    window = _strategy_history[-_STRATEGY_WINDOW:]
    wins = {}
    for s in window:
        if s is not None:
            wins[s] = wins.get(s, 0) + 1

    # Bounds depend on phase
    if phase == "bios":
        min_pct, max_pct = 0.001, 0.02
    else:
        min_pct, max_pct = 0.00005, 0.001

    explorer_rate = wins.get("explorer", 0) / _STRATEGY_WINDOW
    conservative_rate = wins.get("conservative", 0) / _STRATEGY_WINDOW

    if explorer_rate > 0.5:
        new_pct = min(max_pct, base_pct * 1.5)
        if new_pct != base_pct:
            return new_pct, f"explorer winning {explorer_rate:.0%} → ↑ base_pct"
    elif conservative_rate > 0.5:
        new_pct = max(min_pct, base_pct * 0.67)
        if new_pct != base_pct:
            return new_pct, f"conservative winning {conservative_rate:.0%} → ↓ base_pct"

    return base_pct, None


def run_tournament(
    model: DualMERA,
    eval_loader,
    step: int,
    total_steps: int,
    total_ternary: int,
    base_pct: float,
    n_mutants: int,
    n_eval_batches: int,
    gen_seed: int,
    phase: str = "bios",
    r_ema: float = 1.0,
    circuit_bonus: float = 0.5,
    depth_weights: dict[str, float] | None = None,
    sign_flip_rate: float = 0.2,
    seq_len: int = 512,
    row_importance: dict | None = None,
    col_importance: dict | None = None,
    grad_direction: dict | None = None,
) -> dict:
    """Run one evolutionary generation: mutate, evaluate, select.

    BIOS mode:  phase-aware constant budget (not loss-gated)
    Dolma mode: relational loss cone (protect BIOS circuits)

    Two-pass selection to keep tournament fast:
      Pass 1: Select best mutant by eval loss alone (cheap — batched forward only)
      Pass 2: Probe champion and best mutant for circuit fitness (expensive — greedy decode)

    If the winning mutant has better fitness (loss - circuit_bonus * probe_accuracy)
    than champion, adopt it. Otherwise revert.

    Champion never degrades — invariant of the double-buffer.
    """
    # Evaluate champion (loss only — probe comes after selection)
    champion_metrics = evaluate(model, eval_loader, n_batches=n_eval_batches)
    champion_loss = champion_metrics["loss"]

    # Compute base budget (phase-dependent)
    if phase == "bios":
        base_budget = bios_mutation_budget(step, total_steps, total_ternary, base_pct)
    else:
        base_budget = mutation_cone(r_ema, total_ternary, base_pct)

    if base_budget == 0:
        _strategy_history.append(None)
        return {
            "champion_loss": champion_loss,
            "champion_probe": 0.0,
            "budget": 0,
            "accepted": None,
            "accepted_loss": champion_loss,
            "mutations_tried": 0,
            "frozen": True,
        }

    # Save champion for reversion
    champion_snapshot = save_topology(model)

    # ── Pass 1: loss-only selection across all mutants ──
    best_loss = champion_loss
    best_strategy = None
    best_snapshot = None
    strategies_tried = []

    strategy_names = list(MUTANT_STRATEGIES.keys())[:n_mutants]

    for strategy_name in strategy_names:
        scale = MUTANT_STRATEGIES[strategy_name]
        budget = max(1, int(base_budget * scale))

        # Mutate from champion (always start from champion, not from previous mutant)
        load_topology(model, champion_snapshot)
        rng = np.random.RandomState(gen_seed + hash(strategy_name) % (2**31))
        n_applied = mutate_topology(
            model, budget, rng,
            depth_weights=depth_weights,
            sign_flip_rate=sign_flip_rate,
            row_importance=row_importance,
            col_importance=col_importance,
            grad_direction=grad_direction,
        )

        # Evaluate mutant: loss only (fast)
        mutant_metrics = evaluate(model, eval_loader, n_batches=n_eval_batches)
        mutant_loss = mutant_metrics["loss"]

        strategies_tried.append({
            "strategy": strategy_name,
            "budget": budget,
            "applied": n_applied,
            "loss": mutant_loss,
            "delta_loss": mutant_loss - champion_loss,
        })

        if mutant_loss <= best_loss:
            best_loss = mutant_loss
            best_strategy = strategy_name
            best_snapshot = save_topology(model)

    # ── Pass 2: probe champion and best mutant for circuit fitness ──
    # Probe champion
    load_topology(model, champion_snapshot)
    champion_probe = run_teacher_forced_probe(model, seq_len, seed=gen_seed)
    champion_fitness = champion_loss - circuit_bonus * champion_probe

    if best_snapshot is not None and best_strategy is not None:
        # Probe best mutant
        load_topology(model, best_snapshot)
        mutant_probe = run_teacher_forced_probe(
            model, seq_len,
            seed=gen_seed + hash(best_strategy) % (2**31),
        )
        mutant_fitness = best_loss - circuit_bonus * mutant_probe

        if mutant_fitness <= champion_fitness:
            # Accept: mutant wins on combined fitness
            load_topology(model, best_snapshot)
        else:
            # Reject: mutant had better loss but worse circuits
            # Revert to champion
            load_topology(model, champion_snapshot)
            best_strategy = None
            best_loss = champion_loss
            mutant_probe = champion_probe
    else:
        # No mutant beat champion on loss — revert
        load_topology(model, champion_snapshot)
        mutant_probe = champion_probe

    # Track strategy wins for adaptive rate
    _strategy_history.append(best_strategy)

    accepted_probe = mutant_probe if best_strategy is not None else champion_probe

    return {
        "champion_loss": champion_loss,
        "champion_probe": champion_probe,
        "budget": base_budget,
        "accepted": best_strategy,
        "accepted_loss": best_loss,
        "accepted_probe": accepted_probe,
        "delta": (best_loss - circuit_bonus * accepted_probe) - champion_fitness,
        "mutations_tried": len(strategies_tried),
        "strategies": strategies_tried,
        "frozen": False,
    }


# ═══════════════════════════════════════════════════════════════════
# Data loader — handles both BIOS (1 shard) and Dolma (60 shards)
# ═══════════════════════════════════════════════════════════════════

class ShardedDataLoader:
    """Numpy mmap-based data loader for pre-tokenized shards.

    Adapts to any number of shards. For BIOS (1 shard), cycles
    indefinitely with reshuffling each epoch. For Dolma (60 shards),
    splits train/eval.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int,
        seq_len: int,
        split: str = "train",
        eval_shards: int = 0,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        data_dir = Path(data_dir)

        shards = sorted(data_dir.glob("shard_*.npy"))
        assert len(shards) >= 1, f"No shards found in {data_dir}"

        if len(shards) == 1:
            # BIOS mode: single shard, use for both train and eval
            self.shards = shards
        else:
            # Dolma mode: split train/eval
            if eval_shards == 0:
                eval_shards = max(1, len(shards) // 10)  # 10% for eval
            if split == "train":
                self.shards = shards[:-eval_shards]
            else:
                self.shards = shards[-eval_shards:]

        # Build index
        self._rng = np.random.RandomState(seed)
        self._build_index()
        self._loaded: dict[int, np.ndarray] = {}
        self.epoch = 0

    def _build_index(self):
        """Build shuffled (shard_idx, offset) index."""
        self._indices = []
        T = self.seq_len
        for si, shard_path in enumerate(self.shards):
            shard_len = len(np.load(shard_path, mmap_mode="r"))
            n_seqs = shard_len // (T + 1)
            for j in range(n_seqs):
                self._indices.append((si, j * (T + 1)))
        self._rng.shuffle(self._indices)
        self._pos = 0

    def _get_shard(self, idx: int) -> np.ndarray:
        if idx not in self._loaded:
            self._loaded[idx] = np.load(self.shards[idx], mmap_mode="r")
        return self._loaded[idx]

    def next_batch(self) -> tuple[mx.array, mx.array]:
        """Returns (inputs, targets) each of shape (B, seq_len)."""
        B, T = self.batch_size, self.seq_len
        sequences = []
        for _ in range(B):
            if self._pos >= len(self._indices):
                # Epoch complete — reshuffle and continue
                self.epoch += 1
                self._rng.shuffle(self._indices)
                self._pos = 0
            si, offset = self._indices[self._pos]
            self._pos += 1
            shard = self._get_shard(si)
            seq = shard[offset : offset + T + 1].astype(np.int64)
            sequences.append(seq)
        buf = mx.array(np.stack(sequences))
        return buf[:, :T], buf[:, 1: T + 1]

    @property
    def sequences_per_epoch(self) -> int:
        return len(self._indices)

    def reset(self):
        self._pos = 0


# ═══════════════════════════════════════════════════════════════════
# Loss function
# ═══════════════════════════════════════════════════════════════════

def compute_loss(model: DualMERA, inputs: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy loss (scalar)."""
    logits = model(inputs)
    B, T, V = logits.shape
    return nn.losses.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction="mean"
    )


# ═══════════════════════════════════════════════════════════════════
# LR schedule
# ═══════════════════════════════════════════════════════════════════

def cosine_lr(step: int, warmup: int, total: int, lr_max: float) -> float:
    """Cosine annealing with linear warmup. Decays to 10% of lr_max."""
    lr_min = lr_max * 0.1
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate(model: DualMERA, loader: ShardedDataLoader, n_batches: int = 10) -> dict:
    """Run evaluation, return loss + relational metrics."""
    total_loss = 0.0
    total_tokens = 0

    for _ in range(n_batches):
        inputs, targets = loader.next_batch()
        logits = model(inputs)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1), reduction="sum"
        )
        mx.eval(loss)
        total_loss += float(loss)
        total_tokens += B * T

    avg_loss = total_loss / total_tokens
    return {
        "loss": avg_loss,
        "relational": relational_loss(avg_loss),
        "perplexity": math.exp(min(avg_loss, 20)),
    }


# ═══════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: DualMERA,
    optimizer,
    step: int,
    metrics: dict,
    checkpoint_dir: Path,
    data_pos: int,
    epoch: int,
    train_losses: list[float],
    total_generations: int,
    total_accepted: int,
    r_ema: float,
    phase: str,
    gen_base_pct: float = 0.005,
):
    """Save full training state."""
    step_dir = checkpoint_dir / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    # Model weights (includes packed ternary topology)
    flat = tree_flatten(model.parameters())
    mx.savez(str(step_dir / "model.npz"), **{k: v for k, v in flat})

    # Optimizer state
    opt_flat = tree_flatten(optimizer.state)
    mx.savez(str(step_dir / "optimizer.npz"), **{k: v for k, v in opt_flat})

    # Training state JSON
    state = {
        "step": step,
        "epoch": epoch,
        "data_pos": data_pos,
        "phase": phase,
        "r_ema": r_ema,
        "gen_base_pct": gen_base_pct,
        "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in metrics.items()},
        "train_losses_last100": train_losses[-100:],
        "total_generations": total_generations,
        "total_accepted": total_accepted,
    }
    (step_dir / "state.json").write_text(json.dumps(state, indent=2))
    print(f"  💾 Checkpoint: {step_dir}", flush=True)


def load_checkpoint(
    checkpoint_dir: Path,
    model: DualMERA,
    optimizer,
) -> dict:
    """Load training state from checkpoint. Returns state dict."""
    # Model weights
    weights = dict(mx.load(str(checkpoint_dir / "model.npz")))
    model.load_weights(list(weights.items()))

    # Optimizer state
    opt_path = checkpoint_dir / "optimizer.npz"
    if opt_path.exists():
        from mlx.utils import tree_unflatten
        opt_state = dict(mx.load(str(opt_path)))
        optimizer.state = tree_unflatten(list(opt_state.items()))
        mx.eval(optimizer.state)

    # Ternary state
    ternary_path = str(checkpoint_dir / "ternary_state.npz")
    load_ternary_state(model, ternary_path)

    # Training state
    state = json.loads((checkpoint_dir / "state.json").read_text())
    print(f"  📂 Loaded: {checkpoint_dir}")
    print(f"     step={state['step']}  epoch={state.get('epoch', 0)}  "
          f"r_ema={state.get('r_ema', 1.0):.3f}  "
          f"flips={state.get('total_flips', 0):,}", flush=True)
    return state


# ═══════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════

def train(args):
    phase = args.phase
    print("=" * 70)
    print(f"  v8 — Dual MERA Training [{phase.upper()}]")
    print("=" * 70)

    # ── Model ──
    cfg = DualMERAConfig(seq_len=args.seq_len)
    model = create_model(cfg)

    counts = model.count_params()
    total = counts.get("total", sum(counts.values()))
    print(f"\n  Model: DualMERA — {total:,} params")
    print(f"  Seq len: {args.seq_len}")
    print(f"  Vocab: {VOCAB_SIZE}")

    # Count ternary
    n_ternary = 0
    for _, m in _walk_ternary_modules(model):
        if hasattr(m, 'out_features') and hasattr(m, 'in_features'):
            n_ternary += m.out_features * m.in_features
        elif hasattr(m, '_ternary_weight'):
            n_ternary += m._ternary_weight.size * 4
    print(f"  Ternary: {n_ternary:,} weights")

    # ── Data ──
    data_dir = Path(args.data_dir)
    print(f"\n  Data: {data_dir}")

    n_shards = len(list(data_dir.glob("shard_*.npy")))
    print(f"  Shards: {n_shards}")

    eval_shards = 0 if n_shards == 1 else max(1, n_shards // 10)
    train_loader = ShardedDataLoader(
        data_dir, args.batch_size, args.seq_len,
        split="train", eval_shards=eval_shards,
    )
    eval_loader = ShardedDataLoader(
        data_dir, args.batch_size, args.seq_len,
        split="eval", eval_shards=eval_shards,
    )

    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    seqs_per_epoch = train_loader.sequences_per_epoch
    steps_per_epoch = seqs_per_epoch // (args.batch_size * args.grad_accum)
    total_epochs = args.steps / max(1, steps_per_epoch)

    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Sequences/epoch: {seqs_per_epoch:,}")
    print(f"  Steps/epoch: {steps_per_epoch:,}")
    print(f"  Total: {args.steps:,} steps ≈ {total_epochs:.1f} epochs")

    # ── Optimizer ──
    optimizer = optim.AdamW(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Loss + grad function ──
    loss_and_grad = nn.value_and_grad(model, compute_loss)

    # ── Training state ──
    start_step = 0
    train_losses: list[float] = []
    best_eval_loss = float("inf")
    total_generations = 0
    total_accepted = 0
    total_rejected = 0
    adapt_reason = None  # adaptive mutation rate change reason (for logging)
    r_ema = 1.0  # relational loss EMA
    ema_alpha = 0.02

    # ── Gradient-informed mutation: importance maps ──
    # Accumulated via EMA from gamma gradients and input activations.
    # row_importance[path]: (out_features,) — |∂L/∂γ| EMA per output channel
    # col_importance[path]: (in_features,) — mean(|x|) EMA per input channel
    # grad_direction[path]: (out_features,) — sign(∂L/∂γ) EMA (directional signal)
    importance_ema_alpha = 0.1
    row_importance: dict[str, np.ndarray] = {}
    col_importance: dict[str, np.ndarray] = {}
    grad_direction: dict[str, np.ndarray] = {}

    # ── Ternary weight count for mutation budget ──
    total_ternary = count_ternary_weights(model)

    checkpoint_dir = Path(args.checkpoint_dir)

    # ── Resume ──
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            print(f"  ⚠ Resume path not found: {resume_dir}")
            sys.exit(1)

        # Init optimizer state with dummy step first
        dummy_in, dummy_tgt = train_loader.next_batch()
        dummy_loss, dummy_grads = loss_and_grad(model, dummy_in, dummy_tgt)
        mx.eval(dummy_loss, dummy_grads)
        dummy_grads = zero_ternary_grads(model, dummy_grads)
        optimizer.update(model, dummy_grads)
        mx.eval(model.parameters(), optimizer.state)
        restore_ternary(model)
        train_loader.reset()

        state = load_checkpoint(resume_dir, model, optimizer)
        start_step = state["step"]
        train_losses = state.get("train_losses_last100", [])
        total_generations = state.get("total_generations", 0)
        total_accepted = state.get("total_accepted", 0)
        total_rejected = state.get("total_rejected", 0)
        r_ema = state.get("r_ema", 1.0)
        # Restore adaptive mutation rate if saved
        if "gen_base_pct" in state:
            args.gen_base_pct = state["gen_base_pct"]
        train_loader._pos = state.get("data_pos", 0)
        train_loader.epoch = state.get("epoch", 0)

    # ── Summary ──
    print(f"\n  Phase: {phase}")
    print(f"  LR: {args.lr}, warmup: {args.warmup}")
    print(f"  Steps: {start_step} → {args.steps}")
    print(f"  Evolution: gen_interval={args.gen_interval}, "
          f"base_pct={args.gen_base_pct*100:.3f}%, "
          f"mutants={args.gen_n_mutants}, "
          f"circuit_bonus={args.gen_circuit_bonus}, "
          f"sign_flip={args.gen_sign_flip_rate}")
    if phase == "bios":
        print(f"  Mode: BIOS (phase-aware budget, depth-weighted, probe fitness)")
    else:
        print(f"  Mode: Dolma (relational loss cone, uniform, probe fitness)")
    print(f"  Ternary: {total_ternary:,} weights")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"\n{'='*70}\n", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Main training loop
    # ═══════════════════════════════════════════════════════════════

    t_start = time.time()

    for step in range(start_step + 1, args.steps + 1):
        t0 = time.time()

        # ── LR schedule ──
        lr = cosine_lr(step, args.warmup, args.steps, args.lr)
        optimizer.learning_rate = lr

        # ── Gradient accumulation ──
        accum_loss = 0.0
        accum_grads = None

        for _micro in range(args.grad_accum):
            inputs, targets = train_loader.next_batch()
            loss_val, grads = loss_and_grad(model, inputs, targets)
            mx.eval(loss_val, grads)
            accum_loss += float(loss_val)

            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)

        # Average gradients
        accum_grads = tree_map(lambda g: g / args.grad_accum, accum_grads)
        avg_loss = accum_loss / args.grad_accum

        # ── Extract gradient importance BEFORE zeroing ternary grads ──
        # Gamma gradients tell us which rows need topology attention.
        # Input activation stats (saved by TernaryLinear) tell us which columns matter.
        for path, mod in _walk_ternary_modules(model):
            if not isinstance(mod, TernaryLinear):
                continue

            # Navigate grad tree to find gamma gradient for this module
            parts = path.split(".")
            g = accum_grads
            for p in parts:
                if isinstance(g, dict):
                    g = g.get(p, {})
                elif isinstance(g, list) and p.isdigit():
                    g = g[int(p)]
            gamma_grad = g.get("gamma") if isinstance(g, dict) else None

            if gamma_grad is not None:
                gg = np.array(mx.abs(gamma_grad))
                gs = np.array(gamma_grad)  # signed, for direction
                if path in row_importance:
                    row_importance[path] = importance_ema_alpha * gg + (1 - importance_ema_alpha) * row_importance[path]
                    grad_direction[path] = importance_ema_alpha * gs + (1 - importance_ema_alpha) * grad_direction[path]
                else:
                    row_importance[path] = gg
                    grad_direction[path] = gs

            # Column importance from saved input activation magnitude
            if hasattr(mod, "_x_abs_mean"):
                xm = np.array(mod._x_abs_mean)
                if path in col_importance:
                    col_importance[path] = importance_ema_alpha * xm + (1 - importance_ema_alpha) * col_importance[path]
                else:
                    col_importance[path] = xm

        # Zero ternary grads (topology evolves via mutation, not optimizer)
        accum_grads = zero_ternary_grads(model, accum_grads)

        # Gradient clipping
        grad_sq = [mx.sum(g * g) for _, g in tree_flatten(accum_grads)]
        mx.eval(*grad_sq)
        grad_norm = sum(float(g) for g in grad_sq) ** 0.5

        if args.max_grad_norm > 0 and grad_norm > args.max_grad_norm:
            scale = args.max_grad_norm / (grad_norm + 1e-6)
            accum_grads = tree_map(lambda g: g * scale, accum_grads)

        # Optimizer step
        optimizer.update(model, accum_grads)
        mx.eval(model.parameters(), optimizer.state)

        # Restore ternary weights to uint8
        restore_ternary(model)

        # ── Update relational loss EMA ──
        r = relational_loss(avg_loss)
        r_ema = ema_alpha * r + (1 - ema_alpha) * r_ema

        # ── Evolutionary tournament ──
        if step % args.gen_interval == 0:
            # Select depth weights based on phase
            depth_weights = BIOS_DEPTH_WEIGHTS if phase == "bios" else DOLMA_DEPTH_WEIGHTS

            gen_result = run_tournament(
                model=model,
                eval_loader=eval_loader,
                step=step,
                total_steps=args.steps,
                total_ternary=total_ternary,
                base_pct=args.gen_base_pct,
                n_mutants=args.gen_n_mutants,
                n_eval_batches=args.eval_batches,
                gen_seed=step,
                phase=phase,
                r_ema=r_ema,
                circuit_bonus=args.gen_circuit_bonus,
                depth_weights=depth_weights,
                sign_flip_rate=args.gen_sign_flip_rate,
                seq_len=args.seq_len,
                row_importance=row_importance if row_importance else None,
                col_importance=col_importance if col_importance else None,
                grad_direction=grad_direction if grad_direction else None,
            )
            total_generations += 1
            if gen_result["accepted"]:
                total_accepted += 1
            elif not gen_result["frozen"]:
                total_rejected += 1

            # Adaptive mutation rate
            new_pct, adapt_reason = _adapt_base_pct(args.gen_base_pct, phase)
            if adapt_reason:
                args.gen_base_pct = new_pct

        train_losses.append(avg_loss)
        dt = time.time() - t0

        # ── Logging ──
        if step % args.log_interval == 0 or step == start_step + 1:
            tps = tokens_per_step / dt
            epoch = train_loader.epoch

            print(
                f"step {step:>6d} │ "
                f"loss {avg_loss:.4f}  r={r:.3f}  r_ema={r_ema:.3f}  "
                f"lr={lr:.2e}  ‖g‖={grad_norm:.1f}  "
                f"epoch={epoch}  "
                f"{tps/1000:.1f}k tok/s  {dt:.2f}s",
                flush=True,
            )

            # Evolution stats on generation steps
            if step % args.gen_interval == 0:
                budget = gen_result.get("budget", 0)
                accept_rate = (total_accepted / total_generations * 100
                               if total_generations > 0 else 0)
                status = gen_result.get("accepted", "—") or "rejected"
                delta = gen_result.get("delta", 0)
                probe_acc = gen_result.get("accepted_probe", gen_result.get("champion_probe", 0))
                print(
                    f"         │ 🧬 gen {total_generations}: "
                    f"{status}  Δ={delta:+.4f}  "
                    f"budget={budget:,}  "
                    f"probe={probe_acc:.0%}  "
                    f"accept={total_accepted}/{total_generations} ({accept_rate:.0f}%)  "
                    f"base_pct={args.gen_base_pct:.4f}",
                    flush=True,
                )
                if adapt_reason:
                    print(f"         │ 📐 {adapt_reason}", flush=True)

        # ── Eval ──
        if step % args.eval_interval == 0:
            eval_metrics = evaluate(model, eval_loader, n_batches=args.eval_batches)
            is_best = eval_metrics["loss"] < best_eval_loss
            if is_best:
                best_eval_loss = eval_metrics["loss"]

            print(
                f"\n  ── EVAL step {step} ──\n"
                f"     loss={eval_metrics['loss']:.4f}  "
                f"r={eval_metrics['relational']:.3f}  "
                f"ppl={eval_metrics['perplexity']:.1f}  "
                f"epoch={train_loader.epoch}  "
                f"{'★ best' if is_best else ''}\n",
                flush=True,
            )

            # ── Computation probe (circuit detection) ──
            probe_results = run_computation_probe(
                model, seq_len=args.seq_len,
                n_tier1=20, n_tier2=10, n_tier3=10,
                seed=step,
            )
            print_probe_results(probe_results, step)

        # ── Checkpoint ──
        if step % args.checkpoint_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                metrics={"train_loss": avg_loss, "relational": r, "r_ema": r_ema},
                checkpoint_dir=checkpoint_dir,
                data_pos=train_loader._pos,
                epoch=train_loader.epoch,
                train_losses=train_losses,
                total_generations=total_generations,
                total_accepted=total_accepted,
                r_ema=r_ema,
                phase=phase,
                gen_base_pct=args.gen_base_pct,
            )

    # ── Final ──
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Training complete: {args.steps - start_step} steps in {elapsed:.0f}s "
          f"({elapsed/60:.1f} min)")
    print(f"Final train loss: {train_losses[-1]:.4f}  r={relational_loss(train_losses[-1]):.3f}")

    eval_metrics = evaluate(model, eval_loader, n_batches=args.eval_batches * 2)
    print(f"Final eval loss:  {eval_metrics['loss']:.4f}  "
          f"r={eval_metrics['relational']:.3f}  "
          f"ppl={eval_metrics['perplexity']:.1f}")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=args.steps,
        metrics={"train_loss": train_losses[-1], "eval_loss": eval_metrics["loss"],
                 "r_ema": r_ema},
        checkpoint_dir=checkpoint_dir,
        data_pos=train_loader._pos,
        epoch=train_loader.epoch,
        train_losses=train_losses,
        total_generations=total_generations,
        total_accepted=total_accepted,
        r_ema=r_ema,
        phase=phase,
        gen_base_pct=args.gen_base_pct,
    )

    # Save loss curve
    curve_path = checkpoint_dir / "loss_curve.json"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(json.dumps({
        "phase": phase,
        "train_losses": train_losses,
        "steps": list(range(start_step + 1, start_step + 1 + len(train_losses))),
    }))
    print(f"Loss curve: {curve_path}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="v8 — Dual MERA Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase", choices=["bios", "dolma"], default="bios",
                        help="Training phase: bios (burn-in) or dolma (prose)")

    # All flags with None default — filled from phase defaults if not specified
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--gen-interval", type=int, default=None,
                        help="Steps between evolutionary generations")
    parser.add_argument("--gen-base-pct", type=float, default=None,
                        help="Max mutation rate at cone's widest")
    parser.add_argument("--gen-n-mutants", type=int, default=None,
                        help="Number of mutants per generation")
    parser.add_argument("--gen-circuit-bonus", type=float, default=None,
                        help="Fitness bonus scale for probe accuracy in tournament")
    parser.add_argument("--gen-sign-flip-rate", type=float, default=None,
                        help="Fraction of non-zero mutations that flip sign (0-1)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint directory to resume from")

    args = parser.parse_args()

    # Fill unspecified args from phase defaults
    defaults = PHASE_DEFAULTS[args.phase]
    for key, default_val in defaults.items():
        arg_key = key.replace("-", "_")
        if getattr(args, arg_key, None) is None:
            setattr(args, arg_key, default_val)

    train(args)


if __name__ == "__main__":
    main()
