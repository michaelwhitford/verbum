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
    zero_ternary_grads,
    restore_ternary,
    save_ternary_state,
    load_ternary_state,
    count_ternary_weights,
    mutation_cone,
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
        "checkpoint_interval": 5000,
        "log_interval": 50,
        "gen_interval": 50,          # evolutionary generation interval
        "gen_base_pct": 0.001,       # max mutation rate at cone's widest
        "gen_n_mutants": 4,          # population size per generation
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
    },
}


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


def run_tournament(
    model: DualMERA,
    eval_loader,
    r_ema: float,
    total_ternary: int,
    base_pct: float,
    n_mutants: int,
    n_eval_batches: int,
    gen_seed: int,
) -> dict:
    """Run one evolutionary generation: mutate, evaluate, select.

    1. Evaluate champion (current model)
    2. For each mutant strategy:
       a. Save champion topology
       b. Mutate with strategy-scaled budget
       c. Evaluate mutant
       d. Keep if better, else revert
    3. Return stats

    Champion never degrades — invariant of the double-buffer.
    """
    # Evaluate champion
    champion_metrics = evaluate(model, eval_loader, n_batches=n_eval_batches)
    champion_loss = champion_metrics["loss"]

    # Base budget from the relational loss cone
    base_budget = mutation_cone(r_ema, total_ternary, base_pct)

    if base_budget == 0:
        return {
            "champion_loss": champion_loss,
            "budget": 0,
            "accepted": None,
            "accepted_loss": champion_loss,
            "mutations_tried": 0,
            "frozen": True,
        }

    # Save champion for reversion
    champion_snapshot = save_topology(model)

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
        n_applied = mutate_topology(model, budget, rng)

        # Evaluate mutant
        mutant_metrics = evaluate(model, eval_loader, n_batches=n_eval_batches)
        mutant_loss = mutant_metrics["loss"]

        strategies_tried.append({
            "strategy": strategy_name,
            "budget": budget,
            "applied": n_applied,
            "loss": mutant_loss,
            "delta": mutant_loss - champion_loss,
            "accepted": mutant_loss <= best_loss,
        })

        if mutant_loss <= best_loss:
            best_loss = mutant_loss
            best_strategy = strategy_name
            best_snapshot = save_topology(model)

    # Restore the winner
    if best_snapshot is not None and best_strategy is not None:
        load_topology(model, best_snapshot)
    else:
        # All mutants were worse — revert to champion
        load_topology(model, champion_snapshot)

    return {
        "champion_loss": champion_loss,
        "budget": base_budget,
        "accepted": best_strategy,
        "accepted_loss": best_loss,
        "delta": best_loss - champion_loss,
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
    r_ema = 1.0  # relational loss EMA
    ema_alpha = 0.02

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
        train_loader._pos = state.get("data_pos", 0)
        train_loader.epoch = state.get("epoch", 0)

    # ── Summary ──
    print(f"\n  Phase: {phase}")
    print(f"  LR: {args.lr}, warmup: {args.warmup}")
    print(f"  Steps: {start_step} → {args.steps}")
    print(f"  Evolution: gen_interval={args.gen_interval}, "
          f"base_pct={args.gen_base_pct*100:.2f}%, "
          f"mutants={args.gen_n_mutants}")
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
            gen_result = run_tournament(
                model=model,
                eval_loader=eval_loader,
                r_ema=r_ema,
                total_ternary=total_ternary,
                base_pct=args.gen_base_pct,
                n_mutants=args.gen_n_mutants,
                n_eval_batches=args.eval_batches,
                gen_seed=step,
            )
            total_generations += 1
            if gen_result["accepted"]:
                total_accepted += 1
            elif not gen_result["frozen"]:
                total_rejected += 1

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
                budget = mutation_cone(r_ema, total_ternary, args.gen_base_pct)
                accept_rate = (total_accepted / total_generations * 100
                               if total_generations > 0 else 0)
                status = gen_result.get("accepted", "—") or "rejected"
                delta = gen_result.get("delta", 0)
                print(
                    f"         │ 🧬 gen {total_generations}: "
                    f"{status}  Δ={delta:+.4f}  "
                    f"budget={budget:,}  "
                    f"accept={total_accepted}/{total_generations} ({accept_rate:.0f}%)",
                    flush=True,
                )

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
