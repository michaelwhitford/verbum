#!/usr/bin/env python3
"""Tesseract with adaptive gradient clipping.

The cube and tesseract both collapse at step ~651 when shard_00000
shifts from LaTeX to prose. The collapse is a weight-level problem —
broader attention doesn't help because the weights are tuned to the
wrong distribution.

Dense attention survives because it averages gradients over 256
positions, naturally dampening distribution shifts. Strided W=8
has no such buffer — all 8 positions flip simultaneously.

Adaptive gradient clipping gives strided attention the same smoothing:
maintain an EMA of gradient norms, clip to a multiple of the EMA.
Normal training runs at full speed; only anomalous spikes get cut.

Usage:
    uv run python scripts/run_tesseract.py
    uv run python scripts/run_tesseract.py --clip-mult 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATA_DIR = Path("/Users/mwhitford/data/fractal-bitnet/shards")

# ══════════════════════════════════════════════════════════════════════
# Config — same as v2 except strides and diagnostic intervals
# ══════════════════════════════════════════════════════════════════════

VOCAB_SIZE = 50277
D_MODEL = 256
SEQ_LEN = 4096
D_FF = 768
WINDOW = 8
STRIDES = (1, 8, 64, 512)  # TESSERACT

BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
N_STEPS = 1000
WARMUP_STEPS = 500
N_ITERATIONS = 2

# Momentum dampening on loss spike
LOSS_EMA_DECAY = 0.99
LOSS_SPIKE_THRESHOLD = 3.0  # dampen when loss > ema + threshold * std
MOMENTUM_DAMPEN = 0.1       # multiply first moment by this on spike

# Logging intervals
LOG_INTERVAL = 50
EVAL_INTERVAL = 500
CHECKPOINT_INTERVAL = 1000


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


class ShardedDataLoader:
    """Data loader with optional shuffling.

    Sequential mode (shuffle=False): reads contiguously through shards
    in order. Deterministic, but vulnerable to domain clustering.

    Shuffled mode (shuffle=True): pre-computes all valid sequence
    start positions across all shards, shuffles them, and iterates.
    Each sequence is drawn from a random location in a random shard.
    Ensures domain diversity from step 1.
    """

    def __init__(self, data_dir, batch_size, seq_len, split="train",
                 shuffle=False, seed=42):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        shards = sorted(self.data_dir.glob("shard_*.npy"))
        self.shards = shards[:54] if split == "train" else shards[54:]

        if shuffle:
            # Pre-compute all valid (shard_idx, position) pairs
            rng = np.random.RandomState(seed)
            self._indices = []
            for si, shard_path in enumerate(self.shards):
                shard_len = len(np.load(shard_path, mmap_mode="r"))
                # Each sequence needs seq_len + 1 tokens
                n_seqs = shard_len // (seq_len + 1)
                for j in range(n_seqs):
                    self._indices.append((si, j * (seq_len + 1)))
            rng.shuffle(self._indices)
            self._idx_pos = 0
            self._loaded_shards = {}  # cache
        else:
            self.current_shard_idx = 0
            self.position = 0
            self.current_data = None
            self._load_shard(0)

    def _load_shard(self, idx):
        self.current_shard_idx = idx % len(self.shards)
        self.current_data = np.load(
            self.shards[self.current_shard_idx], mmap_mode="r"
        ).astype(np.int64)
        self.position = 0

    def _get_shard(self, idx):
        """Get shard data, caching mmap references."""
        if idx not in self._loaded_shards:
            self._loaded_shards[idx] = np.load(
                self.shards[idx], mmap_mode="r"
            )
        return self._loaded_shards[idx]

    def next_batch(self):
        B, T = self.batch_size, self.seq_len

        if self.shuffle:
            sequences = []
            for _ in range(B):
                if self._idx_pos >= len(self._indices):
                    self._idx_pos = 0  # wrap around (epoch boundary)
                si, pos = self._indices[self._idx_pos]
                self._idx_pos += 1
                shard = self._get_shard(si)
                seq = shard[pos : pos + T + 1].astype(np.int64)
                sequences.append(seq)
            buf = torch.from_numpy(np.stack(sequences)).long()
            return buf[:, :T], buf[:, 1 : T + 1]
        else:
            needed = B * (T + 1)
            if self.position + needed > len(self.current_data):
                self._load_shard(self.current_shard_idx + 1)
            buf = self.current_data[self.position : self.position + needed]
            self.position += needed
            buf = torch.from_numpy(buf.copy()).long().view(B, T + 1)
            return buf[:, :T], buf[:, 1 : T + 1]


def estimate_loss(model, eval_loader, device, n_batches=10):
    model.eval()
    total_loss = 0
    for _ in range(n_batches):
        x, y = eval_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            _, loss = model(x, y)
        total_loss += loss.item()
    model.train()
    return total_loss / n_batches


def get_phase_norms(model):
    """Get gradient norms per phase for the tesseract."""
    norms = {}
    for name, layer in [
        ("type", model.block.type_layer),
        ("parse", model.block.parse_layer),
        ("apply", model.block.apply_layer),
        ("context", model.block.context_layer),
        ("predict", [
            model.block.predict_parse,
            model.block.predict_apply,
            model.block.predict_context,
        ]),
    ]:
        if layer is None:
            continue
        params = (
            layer.parameters()
            if hasattr(layer, "parameters") and not isinstance(layer, list)
            else [p for m in layer for p in m.parameters()]
        )
        total = (
            sum(
                p.grad.data.norm(2).item() ** 2
                for p in params
                if p.grad is not None
            )
            ** 0.5
        )
        norms[name] = total
    norms["embeddings"] = (
        sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.token_embed.parameters()
            if p.grad is not None
        )
        ** 0.5
    )
    return norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Disable data shuffling (sequential reads)")
    args = parser.parse_args()

    n_steps = args.steps
    do_shuffle = not args.no_shuffle

    tag = "shuffled" if do_shuffle else "sequential"
    results_dir = Path(f"results/tesseract-{tag}")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(f"checkpoints/tesseract-{tag}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    from transformers import AutoTokenizer
    from verbum.compressor_lm import CompressorLM

    start = time.time()
    banner(f"TESSERACT — {tag.upper()}")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")

    tokens_total = n_steps * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN
    print(f"  Device: {device}")
    print(f"  Strides: {STRIDES} (tesseract)")
    print(f"  Seq len: {SEQ_LEN}")
    print(f"  Batch: {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Steps: {n_steps}")
    print(f"  Tokens: {tokens_total:,}")
    print(f"  Data: {'SHUFFLED' if do_shuffle else 'sequential'}")
    print(f"  Domain transition expected at step ~650")

    # ── Build model ───────────────────────────────────────────────────
    banner("BUILDING MODEL")

    model = CompressorLM(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=SEQ_LEN,
        d_ff=D_FF, window=WINDOW, strides=STRIDES, mode="iterative",
        n_iterations=N_ITERATIONS,
    ).to(device)

    params = model.count_parameters()
    print(model.describe_heads())
    for k, v in params.items():
        print(f"  {k:25s}: {v:>12,}")
    print()

    # ── Data ──────────────────────────────────────────────────────────
    train_loader = ShardedDataLoader(
        DATA_DIR, BATCH_SIZE, SEQ_LEN, shuffle=do_shuffle, seed=42,
    )
    eval_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, split="eval")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, n_steps - WARMUP_STEPS)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Training ──────────────────────────────────────────────────────
    banner("TRAINING")

    model.train()
    losses = []
    eval_losses = []

    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        accum_loss = 0

        for _ in range(GRAD_ACCUM):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            loss = loss / GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(accum_loss)

        if step % LOG_INTERVAL == 0:
            cur_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start
            tps = step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / elapsed

            print(
                f"  step {step:5d}/{n_steps}  "
                f"loss={accum_loss:.4f}  "
                f"lr={cur_lr:.2e}  "
                f"tok/s={tps:.0f}  "
                f"elapsed={elapsed:.0f}s"
            )

        if step % CHECKPOINT_INTERVAL == 0:
            phase_norms = get_phase_norms(model)
            print(f"  ── checkpoint {step} ──")
            print(f"     grad norms: {json.dumps({k: round(v, 4) for k, v in phase_norms.items()})}")

            ckpt_path = checkpoint_dir / f"step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "loss": accum_loss,
                "phase_grad_norms": phase_norms,
                "train_losses": losses[:],
                "eval_losses": eval_losses[:],
            }, ckpt_path)
            print(f"     saved: {ckpt_path}")

        if step % EVAL_INTERVAL == 0:
            eval_loss = estimate_loss(model, eval_loader, device)
            eval_losses.append({"step": step, "loss": eval_loss})
            print(f"  ── eval loss at step {step}: {eval_loss:.4f} ──")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start
    banner(f"SUMMARY — {elapsed:.0f}s ({elapsed / 60:.1f}m)")

    # Compare collapse zone to v2
    collapse_zone = losses[630:680] if len(losses) >= 680 else []
    pre_collapse = losses[500:630] if len(losses) >= 630 else []

    if collapse_zone and pre_collapse:
        pre_mean = np.mean(pre_collapse)
        zone_max = max(collapse_zone)
        zone_mean = np.mean(collapse_zone)
        spike = zone_max / pre_mean if pre_mean > 0 else 0

        print(f"  Pre-collapse mean (steps 500-630): {pre_mean:.3f}")
        print(f"  Collapse zone max (steps 630-680): {zone_max:.3f}")
        print(f"  Collapse zone mean:                {zone_mean:.3f}")
        print(f"  Spike ratio (max/pre_mean):        {spike:.2f}x")
        print()

        print()

        # v2 comparison
        print(f"  v2 cube (for reference):")
        print(f"    Pre-collapse mean: 5.035")
        print(f"    Spike peak:        7.786")
        print(f"    Spike ratio:       1.55x")
        print()

        if spike > 1.3:
            print(f"  ❌ COLLAPSE DETECTED (spike ratio {spike:.2f}x > 1.3)")
        elif spike > 1.1:
            print(f"  ⚠  DAMPENED COLLAPSE (spike ratio {spike:.2f}x)")
        else:
            print(f"  ✅ NO COLLAPSE (spike ratio {spike:.2f}x ≤ 1.1)")

    # Save summary
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "architecture": f"CompressorLM (tesseract, {tag})",
        "strides": list(STRIDES),
        "params": params,
        "n_steps": n_steps,
        "shuffled": do_shuffle,
        "train_losses": losses,
        "eval_losses": eval_losses,
    }
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved: {summary_path}")


if __name__ == "__main__":
    main()
