"""
v8 — Dual MERA Pipeline Training Loop

Trains the Dual MERA Pipeline LM on Dolma (pre-tokenized GPT-NeoX shards).

Features:
  - Per-stage relational loss tracking (r_k ∈ [0,1])
  - Per-stage phase control (explore/balance/refine) with hysteresis
  - Global phase coordination across stages
  - Cosine LR with warmup
  - Gradient accumulation
  - Eval + checkpoint at configurable intervals
  - Full per-stage metrics at every step

Usage:
    cd ~/src/verbum
    uv run python scripts/v8/train.py [--steps N] [--batch_size B] [--lr LR]
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Force unbuffered stdout — see output immediately
os.environ["PYTHONUNBUFFERED"] = "1"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

# ── Local import (same directory) ──
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import PipelineConfig, VSMPipeline, create_model
from ternary import (
    accumulate_flips,
    apply_flips,
    compute_flip_threshold,
    zero_ternary_grads,
    restore_ternary,
    save_ternary_state,
    load_ternary_state,
    _walk_ternary_modules,
)


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

DATA_DIR = Path("/Users/mwhitford/data/fractal-bitnet/shards")
CHECKPOINT_DIR = Path("checkpoints/vsm-lm-v8")
VOCAB_SIZE = 50277

# Information-theoretic landmarks
LOG_V = float(np.log(VOCAB_SIZE))          # 10.83 nats (uniform random)
E_IRREDUCIBLE = 1.69                        # irreducible entropy floor
LEARNABLE_RANGE = LOG_V - E_IRREDUCIBLE     # 9.14 nats

# Phase thresholds (on relational loss r)
PHASE_EXPLORE = "explore"
PHASE_BALANCE = "balance"
PHASE_REFINE = "refine"

PHASE_CONFIG = {
    PHASE_EXPLORE: {"description": "wide search, high learning rate effect"},
    PHASE_BALANCE: {"description": "balanced learning"},
    PHASE_REFINE: {"description": "fine-tuning, protect topology"},
}

PHASE_HYSTERESIS = 50  # steps before phase transition

# ═══════════════════════════════════════════════════════════════════
# Ternary flip control — topology annealing via relational loss
# ═══════════════════════════════════════════════════════════════════
#
# The ternary routing topology converges through three mechanisms:
#
# 1. Relational modulation:  r_ema drives flip_scale continuously.
#    High r (far from optimal) → flip aggressively → explore routes.
#    Low r (near optimal) → near-zero flips → topology frozen.
#
# 2. Per-weight cooldown:  after flipping, a weight must wait
#    FLIP_COOLDOWN × FLIP_INTERVAL steps before it can flip again.
#    This prevents oscillation: route A→B→A→B.  The system commits
#    to a route and lets continuous params (gamma, norms) adapt.
#
# 3. Threshold from consensus:  a weight only flips when gradient
#    direction is consistent across many micro-batches.  As the model
#    converges, gradients become less coherent → fewer weights
#    exceed threshold → fewer flips naturally.
#
# Together: the topology anneals from liquid (early) to frozen (late).
# No explicit schedule — the relational loss IS the temperature.

FLIP_INTERVAL = 50        # steps between flip checks (was 25 — more evidence per decision)
FLIP_BASE_PCT = 0.001     # base % of ternary weights to flip per check (was 0.5% — much smaller trickle)
FLIP_COOLDOWN = 8         # intervals before a weight can flip again (8 × 50 = 400 steps)


def adaptive_flip_scale(r_ema: float) -> float:
    """Continuous flip rate modulator from relational loss.

    r > 0.6 → scale=2.0  (far from optimal, explore topology)
    r = 0.4 → scale=1.0  (balanced)
    r < 0.15 → scale=0.05 (near optimal, topology essentially frozen)
    r < 0.05 → scale=0.0  (converged, no flips at all)

    Smooth ramp. No discontinuities. The topology anneals
    continuously as the model learns.
    """
    if r_ema < 0.05:
        return 0.0
    return max(0.05, 0.05 + 1.95 * min(1.0, r_ema / 0.6))


# ═══════════════════════════════════════════════════════════════════
# Data loader
# ═══════════════════════════════════════════════════════════════════


class ShardedDataLoader:
    """Numpy mmap-based data loader for pre-tokenized Dolma shards.

    Shards: shard_NNNNN.npy, int32, ~50M tokens each.
    54 train / 6 eval split.
    """

    def __init__(self, data_dir: str | Path, batch_size: int, seq_len: int,
                 split: str = "train", seed: int = 42):
        self.batch_size = batch_size
        self.seq_len = seq_len
        data_dir = Path(data_dir)

        shards = sorted(data_dir.glob("shard_*.npy"))
        assert len(shards) >= 60, f"Expected ≥60 shards, found {len(shards)}"
        self.shards = shards[:54] if split == "train" else shards[54:]

        # Build index: (shard_idx, offset) for each sequence
        rng = np.random.RandomState(seed)
        self._indices = []
        for si, shard_path in enumerate(self.shards):
            shard_len = len(np.load(shard_path, mmap_mode="r"))
            n_seqs = shard_len // (seq_len + 1)
            for j in range(n_seqs):
                self._indices.append((si, j * (seq_len + 1)))
        rng.shuffle(self._indices)
        self._pos = 0
        self._loaded = {}

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
                self._pos = 0
            si, offset = self._indices[self._pos]
            self._pos += 1
            shard = self._get_shard(si)
            seq = shard[offset: offset + T + 1].astype(np.int64)
            sequences.append(seq)
        buf = mx.array(np.stack(sequences))
        return buf[:, :T], buf[:, 1: T + 1]

    def reset(self):
        self._pos = 0


# ═══════════════════════════════════════════════════════════════════
# Relational metrics
# ═══════════════════════════════════════════════════════════════════


def relational_loss(loss: float) -> float:
    """Dimensionless position in the learnable range.

    r=1.0 → model knows nothing (loss = log(V))
    r=0.0 → model at irreducible floor (loss = E)
    """
    reducible = max(0.0, loss - E_IRREDUCIBLE)
    return min(1.0, reducible / LEARNABLE_RANGE)


def phase_for_r(r: float) -> str:
    """Target phase for a given relational loss."""
    if r > 0.5:
        return PHASE_EXPLORE
    elif r < 0.25:
        return PHASE_REFINE
    return PHASE_BALANCE


class StagePhaseController:
    """Independent phase controller for one VSM stage.

    Stage 1: driven by its own CE (r₁ = relational_loss(CE₁)).
    Stages 2-4: driven by their contribution delta (Δₖ = CEₖ₋₁ - CEₖ).
      High Δ → stage is contributing → balance/refine.
      Low Δ → stage isn't contributing → explore.
      Negative Δ → stage is hurting → explore (needs to learn).
    """

    def __init__(self, stage_id: int):
        self.stage_id = stage_id
        self.phase = PHASE_EXPLORE
        self.steps_toward_new = 0
        self.r_ema = 1.0         # for Stage 1: relational loss of CE₁
        self.delta_ema = 0.0     # for Stages 2-4: contribution delta
        self.ce_ema = 10.0       # raw CE EMA for this stage's measurement
        self.ema_alpha = 0.05

    def update_stage1(self, ce: float) -> tuple[str, bool]:
        """Update Stage 1 controller with its own CE."""
        r = relational_loss(ce)
        self.r_ema = self.ema_alpha * r + (1 - self.ema_alpha) * self.r_ema
        self.ce_ema = self.ema_alpha * ce + (1 - self.ema_alpha) * self.ce_ema
        return self._check_transition(self.r_ema)

    def update_higher(self, ce: float, ce_prev: float) -> tuple[str, bool]:
        """Update Stages 2-4 with their contribution delta.

        delta = ce_prev - ce (positive = this stage helps).
        Map delta to a phase signal:
          delta > 0.1  → contributing meaningfully → balance/refine
          delta ≈ 0    → not contributing → explore
          delta < 0    → hurting → explore
        """
        delta = ce_prev - ce
        self.delta_ema = self.ema_alpha * delta + (1 - self.ema_alpha) * self.delta_ema
        self.ce_ema = self.ema_alpha * ce + (1 - self.ema_alpha) * self.ce_ema

        # Map delta_ema to a relational-like signal for phase control
        # High delta → low r (contributing well → refine)
        # Low/negative delta → high r (not contributing → explore)
        r = max(0.0, min(1.0, 1.0 - self.delta_ema * 5.0))
        self.r_ema = self.ema_alpha * r + (1 - self.ema_alpha) * self.r_ema
        return self._check_transition(self.r_ema)

    def _check_transition(self, r: float) -> tuple[str, bool]:
        target = phase_for_r(r)
        if target == self.phase:
            self.steps_toward_new = 0
            return self.phase, False
        else:
            self.steps_toward_new += 1
            if self.steps_toward_new >= PHASE_HYSTERESIS:
                self.phase = target
                self.steps_toward_new = 0
                return self.phase, True
            return self.phase, False


class GlobalPhaseController:
    """Coordinates phase across all stages.

    Global phase:
      explore  iff ANY stage has r_ema > 0.5
      refine   iff ALL stages have r_ema < 0.25
      balance  otherwise
    """

    def __init__(self, stage_controllers: list[StagePhaseController]):
        self.stages = stage_controllers

    @property
    def phase(self) -> str:
        rs = [s.r_ema for s in self.stages]
        if any(r > 0.5 for r in rs):
            return PHASE_EXPLORE
        if all(r < 0.25 for r in rs):
            return PHASE_REFINE
        return PHASE_BALANCE


# ═══════════════════════════════════════════════════════════════════
# Loss function
# ═══════════════════════════════════════════════════════════════════


def compute_loss(model: VSMPipeline, inputs: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy loss (scalar). Used in grad computation."""
    logits = model(inputs)
    B, T, V = logits.shape
    return nn.losses.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), reduction="mean")


# ═══════════════════════════════════════════════════════════════════
# LR schedule
# ═══════════════════════════════════════════════════════════════════


def cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float = 0.0) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════


def evaluate(model: VSMPipeline, loader: ShardedDataLoader, n_batches: int = 10) -> dict:
    """Run evaluation and return aggregate metrics including per-stage CE."""
    total_loss = 0.0
    total_tokens = 0
    all_metrics = {}

    for _ in range(n_batches):
        inputs, targets = loader.next_batch()
        logits, metrics = model.forward_with_metrics(inputs, targets=targets)

        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1), reduction="sum"
        )
        mx.eval(loss)
        total_loss += float(loss)
        total_tokens += B * T

        for k, v in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(float(v) if isinstance(v, (int, float)) else v)

    avg_loss = total_loss / total_tokens
    result = {
        "loss": avg_loss,
        "relational": relational_loss(avg_loss),
        "perplexity": math.exp(min(avg_loss, 20)),  # cap to avoid overflow
    }
    for k, vals in all_metrics.items():
        if isinstance(vals[0], (int, float)):
            result[k] = sum(vals) / len(vals)

    # Per-stage deltas in eval
    for i in range(1, 5):
        ce_key = f"ce_stage{i}"
        if ce_key in result:
            result[f"r_stage{i}"] = relational_loss(result[ce_key])
    for i in range(2, 5):
        prev = result.get(f"ce_stage{i-1}", avg_loss)
        curr = result.get(f"ce_stage{i}", avg_loss)
        result[f"delta_stage{i}"] = prev - curr

    return result


# ═══════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════


def save_checkpoint(
    model: VSMPipeline,
    optimizer,
    step: int,
    metrics: dict,
    cfg: PipelineConfig,
    checkpoint_dir: Path,
    stage_controllers: list[StagePhaseController],
    data_pos: int,
    train_losses: list[float],
    total_flips: int = 0,
    total_reversals: int = 0,
    has_ternary: bool = False,
):
    """Save full training state for clean resume.

    Saves:
      model.npz         — model weights
      optimizer.npz      — Adam momentum + variance
      ternary_state.npz  — flip cooldown + direction history (if ternary)
      state.json         — step, metrics, config, phases, flip counters
    """
    step_dir = checkpoint_dir / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    # Model weights
    flat = tree_flatten(model.parameters())
    mx.savez(str(step_dir / "model.npz"), **{k: v for k, v in flat})

    # Optimizer state (Adam momentum + variance + step counter)
    opt_flat = tree_flatten(optimizer.state)
    mx.savez(str(step_dir / "optimizer.npz"), **{k: v for k, v in opt_flat})

    # Ternary flip state (cooldown + direction history)
    if has_ternary:
        save_ternary_state(model, str(step_dir / "ternary_state.npz"))

    # Training state (JSON for readability + probing)
    state = {
        "step": step,
        "data_pos": data_pos,
        "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in metrics.items()},
        "config": {
            "vocab_size": cfg.vocab_size,
            "seq_len": cfg.seq_len,
            "d_model": cfg.d_model,
            "stage_positions": cfg.stage_positions,
            "stages": [
                {"n_layers": s.n_layers, "n_heads": s.n_heads,
                 "d_model": s.d_model, "d_ff": s.d_ff}
                for s in cfg.stages
            ],
        },
        "phase_controllers": [
            {
                "stage_id": sc.stage_id,
                "phase": sc.phase,
                "steps_toward_new": sc.steps_toward_new,
                "r_ema": sc.r_ema,
                "delta_ema": sc.delta_ema,
                "ce_ema": sc.ce_ema,
            }
            for sc in stage_controllers
        ],
        "train_losses_last100": train_losses[-100:],
        "total_flips": total_flips,
        "total_reversals": total_reversals,
    }
    (step_dir / "state.json").write_text(json.dumps(state, indent=2))
    print(f"  💾 Checkpoint saved: {step_dir}")


def load_checkpoint(
    checkpoint_dir: Path,
    model: VSMPipeline,
    optimizer,
    stage_controllers: list[StagePhaseController],
    has_ternary: bool = False,
) -> tuple[int, int, list[float], int, int]:
    """Load full training state from checkpoint.

    Returns (step, data_pos, train_losses).
    Mutates model, optimizer, and stage_controllers in place.
    """
    # Load model weights
    weights = dict(mx.load(str(checkpoint_dir / "model.npz")))
    model.load_weights(list(weights.items()))

    # Load optimizer state — need to init optimizer first with a dummy step
    # so it has the right structure, then overwrite
    opt_path = checkpoint_dir / "optimizer.npz"
    if opt_path.exists():
        opt_state = dict(mx.load(str(opt_path)))
        from mlx.utils import tree_unflatten
        optimizer.state = tree_unflatten(list(opt_state.items()))
        mx.eval(optimizer.state)

    # Load training state
    state = json.loads((checkpoint_dir / "state.json").read_text())
    step = state["step"]
    data_pos = state.get("data_pos", 0)
    train_losses = state.get("train_losses_last100", [])
    total_flips = state.get("total_flips", 0)
    total_reversals = state.get("total_reversals", 0)

    # Restore phase controllers
    for sc_state in state.get("phase_controllers", []):
        sid = sc_state["stage_id"]
        if sid < len(stage_controllers):
            sc = stage_controllers[sid]
            sc.phase = sc_state["phase"]
            sc.steps_toward_new = sc_state["steps_toward_new"]
            sc.r_ema = sc_state["r_ema"]
            sc.delta_ema = sc_state["delta_ema"]
            sc.ce_ema = sc_state["ce_ema"]

    # Restore ternary flip state (cooldown + direction history, NOT accumulator)
    if has_ternary:
        ternary_path = str(checkpoint_dir / "ternary_state.npz")
        load_ternary_state(model, ternary_path)

    print(f"  📂 Checkpoint loaded: {checkpoint_dir}")
    print(f"     step={step}  data_pos={data_pos}")
    if has_ternary:
        print(f"     flips={total_flips:,}  reversals={total_reversals:,}")
    for sc in stage_controllers:
        print(f"     Stage {sc.stage_id+1}: phase={sc.phase}  r_ema={sc.r_ema:.3f}  δ_ema={sc.delta_ema:+.4f}")

    return step, data_pos, train_losses, total_flips, total_reversals


# ═══════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════


def train(args):
    print("=" * 70)
    print("  v8 — Dual MERA Pipeline Language Model")
    print("=" * 70)

    # ── Config ──
    cfg = PipelineConfig(seq_len=args.seq_len)
    model = create_model(cfg)

    # Print architecture
    counts = model.count_params()
    print(f"\nArchitecture: {len(cfg.stages)} stages, positions {cfg.stage_positions}")
    stage_names = ['Surface', 'Structural', 'Semantic', 'Reasoning']
    for i, s in enumerate(cfg.stages):
        # Find the count key (may include "(ternary)" suffix)
        stage_key = [k for k in counts if k.startswith(f"stage{i+1}")][0]
        t_label = " [TERNARY]" if cfg.ternary_stages[i] else ""
        print(f"  Stage {i+1} ({stage_names[i]}){t_label}: "
              f"{s.n_layers}L {s.n_heads}H d={s.d_model} ff={s.d_ff} "
              f"pos={cfg.stage_positions[i]} — {counts[stage_key]:,} params")
    print(f"  Reducers: {sum(counts[k] for k in counts if 'reducer' in k):,} params")
    print(f"  Feedback: {sum(counts[k] for k in counts if 'feedback' in k):,} params")
    print(f"  Embedding: {counts['embedding']:,} params (tied)")
    print(f"  Total: {counts['total']:,} params")
    if counts.get("hot_ternary_weights", 0) > 0:
        print(f"  Hot path: {counts['hot_ternary_bytes']:,} bytes (ternary) "
              f"= {counts['hot_ternary_bytes']/1024:.0f} KB")

    # ── Data ──
    print(f"\nData: {DATA_DIR}")
    train_loader = ShardedDataLoader(DATA_DIR, args.batch_size, args.seq_len, split="train")
    eval_loader = ShardedDataLoader(DATA_DIR, args.batch_size, args.seq_len, split="eval")
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    print(f"  Batch: {args.batch_size} × {args.grad_accum} accum × {args.seq_len} seq = "
          f"{tokens_per_step:,} tokens/step")
    print(f"  Total: {args.steps:,} steps = {args.steps * tokens_per_step / 1e6:.1f}M tokens")

    # ── Optimizer ──
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    # ── Phase controllers ──
    stage_controllers = [StagePhaseController(i) for i in range(len(cfg.stages))]
    global_controller = GlobalPhaseController(stage_controllers)

    # ── Loss + grad function ──
    loss_and_grad = nn.value_and_grad(model, compute_loss)

    # ── Ternary detection ──
    has_ternary = any(cfg.ternary_stages) or cfg.ternary_feedback
    total_flips = 0
    total_reversals = 0
    last_flip_count = 0
    last_reversal_count = 0
    if has_ternary:
        n_ternary = sum(
            m.out_features * m.in_features
            for _, m in _walk_ternary_modules(model)
        )
        print(f"\n  Ternary: {n_ternary:,} weights ({n_ternary // 4:,} packed bytes)")
        print(f"  Flip interval: {FLIP_INTERVAL} steps, base rate: {FLIP_BASE_PCT*100:.1f}%")

    # ── Training state ──
    start_step = 0
    train_losses = []
    best_eval_loss = float("inf")

    # ── Resume from checkpoint ──
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            print(f"  ⚠ Resume path not found: {resume_dir}")
            sys.exit(1)

        # Need to init optimizer state before loading (MLX requires structure match)
        # Do one dummy forward+backward to create optimizer state
        dummy_in, dummy_tgt = train_loader.next_batch()
        dummy_loss, dummy_grads = loss_and_grad(model, dummy_in, dummy_tgt)
        mx.eval(dummy_loss, dummy_grads)
        # Must zero ternary grads before optimizer (shape mismatch otherwise)
        if has_ternary:
            dummy_grads = zero_ternary_grads(model, dummy_grads)
        optimizer.update(model, dummy_grads)
        mx.eval(model.parameters(), optimizer.state)
        if has_ternary:
            restore_ternary(model)
        train_loader.reset()

        start_step, data_pos, train_losses, total_flips, total_reversals = load_checkpoint(
            resume_dir, model, optimizer, stage_controllers, has_ternary=has_ternary
        )
        train_loader._pos = data_pos
        print(f"  Resuming from step {start_step}, running to step {args.steps}")

    print(f"\nTraining config: lr={args.lr}, warmup={args.warmup}, steps={args.steps}")
    print(f"  Eval every {args.eval_interval} steps, checkpoint every {args.checkpoint_interval} steps")
    print(f"\n{'='*70}\n", flush=True)

    step_time_start = time.time()

    for step in range(start_step + 1, args.steps + 1):
        t0 = time.time()

        # ── LR schedule ──
        lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.lr * 0.1)
        optimizer.learning_rate = lr

        # ── Gradient accumulation ──
        accum_loss = 0.0
        accum_grads = None

        for micro in range(args.grad_accum):
            inputs, targets = train_loader.next_batch()
            loss_val, grads = loss_and_grad(model, inputs, targets)
            mx.eval(loss_val, grads)
            accum_loss += float(loss_val)

            # Accumulate ternary flip votes (per micro-batch)
            if has_ternary:
                accumulate_flips(model, grads)

            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = tree_map(
                    lambda a, b: a + b, accum_grads, grads
                )

        # Average gradients
        accum_grads = tree_map(
            lambda g: g / args.grad_accum, accum_grads
        )
        avg_loss = accum_loss / args.grad_accum

        # ── Zero ternary grads before optimizer ──
        # Ternary weight grads route to flip accumulator, not optimizer.
        # Must zero them to prevent optimizer shape mismatch.
        if has_ternary:
            accum_grads = zero_ternary_grads(model, accum_grads)

        # ── Gradient clipping (single eval, not per-param) ──
        grad_sq = [mx.sum(g * g) for _, g in tree_flatten(accum_grads)]
        mx.eval(*grad_sq)
        grad_norm = sum(float(g) for g in grad_sq) ** 0.5

        if args.max_grad_norm > 0 and grad_norm > args.max_grad_norm:
            scale = args.max_grad_norm / (grad_norm + 1e-6)
            accum_grads = tree_map(lambda g: g * scale, accum_grads)

        # ── Update ──
        optimizer.update(model, accum_grads)
        mx.eval(model.parameters(), optimizer.state)

        # ── Restore ternary weights to uint8 (only if ternary) ──
        if has_ternary:
            restore_ternary(model)

        # ── Periodic ternary flips (relational-modulated) ──
        if has_ternary and step % FLIP_INTERVAL == 0:
            # Stage 1's r_ema drives flip rate for all ternary weights
            # (Stage 1 and feedback 2→1 are both on the hot path)
            r1 = stage_controllers[0].r_ema
            flip_scale = adaptive_flip_scale(r1)
            effective_pct = FLIP_BASE_PCT * flip_scale

            if effective_pct > 0:
                threshold = compute_flip_threshold(model, effective_pct)
                n_flipped, n_reversals = apply_flips(
                    model,
                    threshold=max(1, int(threshold)),
                    max_flip_pct=effective_pct,
                    cooldown_intervals=FLIP_COOLDOWN,
                )
                total_flips += n_flipped
                total_reversals += n_reversals
                last_flip_count = n_flipped
                last_reversal_count = n_reversals
            else:
                last_flip_count = 0
                last_reversal_count = 0
                # Still need to decrement cooldowns even with no flips
                apply_flips(model, threshold=999, max_flip_pct=0.0,
                           cooldown_intervals=FLIP_COOLDOWN)

        train_losses.append(avg_loss)
        dt = time.time() - t0

        # ── Per-stage metrics (expensive — only at log interval) ──
        # Between measurements, phase controllers use the global training loss.
        # This avoids 4 extra CE projections + 6 feedback passes per step.
        compute_stage_metrics = (step % args.log_interval == 0 or step == 1)

        if compute_stage_metrics:
            logits_m, step_metrics = model.forward_with_metrics(inputs, targets=targets)
            mx.eval(logits_m)  # force single eval of the full graph
            ce_keys = ["ce_stage1", "ce_stage2", "ce_stage3", "ce_stage4"]
            ces = [step_metrics.get(k, avg_loss) for k in ce_keys]

            # Update phase controllers with per-stage signal
            stage_controllers[0].update_stage1(ces[0])
            for k in range(1, len(stage_controllers)):
                stage_controllers[k].update_higher(ces[k], ces[k - 1])
        else:
            # Cheap update: all controllers use the global loss
            for sc in stage_controllers:
                sc.update_stage1(avg_loss)
            ces = None

        r = relational_loss(avg_loss)
        g_phase = global_controller.phase

        # ── Logging ──
        if step % args.log_interval == 0 or step == 1:
            tps = tokens_per_step / dt
            stage_phases = "".join(sc.phase[0].upper() for sc in stage_controllers)

            print(
                f"step {step:>6d} │ "
                f"loss {avg_loss:.4f}  r={r:.3f}  "
                f"lr={lr:.2e}  "
                f"‖g‖={grad_norm:.1f}  "
                f"phase={stage_phases}({g_phase[0].upper()})  "
                f"{tps/1000:.1f}k tok/s  {dt:.2f}s"
            )

            # Per-stage CE and deltas (only when measured)
            if ces is not None:
                ce_parts = [f"CE{i+1}={ces[i]:.3f}" for i in range(4)]
                deltas = [f"Δ{i+1}={ces[i-1]-ces[i]:+.3f}" for i in range(1, 4)]
                print(f"         │ {' '.join(ce_parts)}")
                print(f"         │ {' '.join(deltas)}")

            # Per-stage r_ema and phase
            r_parts = [f"r{i+1}={sc.r_ema:.3f}" for i, sc in enumerate(stage_controllers)]
            d_parts = [f"δ{i+1}={sc.delta_ema:+.4f}" for i, sc in enumerate(stage_controllers[1:])]
            print(f"         │ {' '.join(r_parts)}  │  {' '.join(d_parts)}", flush=True)

            # Ternary stats (on flip steps)
            if has_ternary and step % FLIP_INTERVAL == 0:
                r1 = stage_controllers[0].r_ema
                fs = adaptive_flip_scale(r1)
                ep = FLIP_BASE_PCT * fs
                rev_rate = (total_reversals / total_flips * 100) if total_flips > 0 else 0
                print(f"         │ flips: {last_flip_count:,}(+{last_reversal_count} rev) this check  "
                      f"total: {total_flips:,} flips, {total_reversals:,} rev ({rev_rate:.1f}%)  "
                      f"scale={fs:.2f}")

        # ── Eval ──
        if step % args.eval_interval == 0:
            eval_metrics = evaluate(model, eval_loader, n_batches=args.eval_batches)
            eval_r = eval_metrics["relational"]
            is_best = eval_metrics["loss"] < best_eval_loss
            if is_best:
                best_eval_loss = eval_metrics["loss"]

            print(f"\n  ── EVAL step {step} ──")
            print(f"     loss={eval_metrics['loss']:.4f}  "
                  f"r={eval_r:.3f}  "
                  f"ppl={eval_metrics['perplexity']:.1f}  "
                  f"{'★ best' if is_best else ''}")

            # Per-stage eval CE and deltas
            eval_ces = [f"CE{i}={eval_metrics.get(f'ce_stage{i}', 0):.3f}" for i in range(1, 5)]
            eval_deltas = [f"Δ{i}={eval_metrics.get(f'delta_stage{i}', 0):+.3f}" for i in range(2, 5)]
            print(f"     {' '.join(eval_ces)}")
            print(f"     {' '.join(eval_deltas)}")
            print()

        # ── Checkpoint ──
        if step % args.checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, step,
                metrics={
                    "train_loss": avg_loss,
                    "relational": r,
                    "grad_norm": grad_norm,
                    "lr": lr,
                    **{k: v for k, v in step_metrics.items()},
                },
                cfg=cfg,
                checkpoint_dir=CHECKPOINT_DIR,
                stage_controllers=stage_controllers,
                data_pos=train_loader._pos,
                train_losses=train_losses,
                total_flips=total_flips,
                total_reversals=total_reversals,
                has_ternary=has_ternary,
            )

    # ── Final eval ──
    elapsed = time.time() - step_time_start
    print(f"\n{'='*70}")
    print(f"Training complete: {args.steps} steps, {elapsed:.1f}s")
    print(f"Final train loss: {train_losses[-1]:.4f}  r={relational_loss(train_losses[-1]):.3f}")

    eval_metrics = evaluate(model, eval_loader, n_batches=args.eval_batches * 2)
    print(f"Final eval loss:  {eval_metrics['loss']:.4f}  "
          f"r={eval_metrics['relational']:.3f}  "
          f"ppl={eval_metrics['perplexity']:.1f}")

    save_checkpoint(
        model, optimizer, args.steps,
        metrics={
            "train_loss": train_losses[-1],
            "eval_loss": eval_metrics["loss"],
            "relational": relational_loss(train_losses[-1]),
        },
        cfg=cfg,
        checkpoint_dir=CHECKPOINT_DIR,
        stage_controllers=stage_controllers,
        data_pos=train_loader._pos,
        train_losses=train_losses,
        total_flips=total_flips,
        total_reversals=total_reversals,
        has_ternary=has_ternary,
    )

    # ── Save loss curve ──
    curve_path = CHECKPOINT_DIR / "loss_curve.json"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(json.dumps({
        "train_losses": train_losses,
        "steps": list(range(1, len(train_losses) + 1)),
    }))
    print(f"Loss curve saved: {curve_path}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="v8 — Dual MERA Pipeline Training")
    parser.add_argument("--steps", type=int, default=165000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=2500)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
