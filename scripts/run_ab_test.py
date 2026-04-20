#!/usr/bin/env python3
"""Quick A/B test: cube vs pipeline attention structure.

Runs 1000 steps of each mode on Dolma, logs loss every 10 steps.
~9 min per run, ~18 min total. Saves state for resuming the winner.

Usage:
    uv run python scripts/run_ab_test.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATA_DIR = Path("/Users/mwhitford/data/fractal-bitnet/shards")
RESULTS_DIR = Path("results/ab-test")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Config
VOCAB_SIZE = 50277
D_MODEL = 256
SEQ_LEN = 256
BATCH_SIZE = 32
GRAD_ACCUM = 4
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 100
N_STEPS = 1000
LOG_INTERVAL = 10


class ShardedDataLoader:
    def __init__(self, data_dir, batch_size, seq_len, split="train"):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        shards = sorted(self.data_dir.glob("shard_*.npy"))
        self.shards = shards[:54] if split == "train" else shards[54:]
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

    def next_batch(self):
        B, T = self.batch_size, self.seq_len
        needed = B * (T + 1)
        if self.position + needed > len(self.current_data):
            self._load_shard(self.current_shard_idx + 1)
        buf = self.current_data[self.position : self.position + needed]
        self.position += needed
        buf = torch.from_numpy(buf.copy()).long().view(B, T + 1)
        return buf[:, :T], buf[:, 1 : T + 1]


def train_run(mode: str, device: str, seed: int = 42, n_steps: int = N_STEPS):
    """Train one mode for n_steps, return loss curve."""
    from verbum.compressor_lm import CompressorLM

    torch.manual_seed(seed)
    np.random.seed(seed)

    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_len=SEQ_LEN,
        mode=mode,
        d_ff=768,
        window=8,
        strides=(1, 8, 64),
    )
    if mode == "iterative":
        kwargs["n_iterations"] = 2

    model = CompressorLM(**kwargs).to(device)

    params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"  {mode.upper()} — {params['total']:,} params")
    print(f"{'='*60}")
    print(model.describe_heads())

    loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN)

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

    model.train()
    losses = []
    start = time.time()

    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        accum_loss = 0

        for _ in range(GRAD_ACCUM):
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            loss = loss / GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - start
            tps = step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / elapsed
            losses.append({"step": step, "loss": accum_loss})
            print(f"  step {step:4d}/{n_steps}  loss={accum_loss:.4f}  "
                  f"tok/s={tps:.0f}  elapsed={elapsed:.0f}s")

    elapsed = time.time() - start

    # Save model state for resuming
    save_path = RESULTS_DIR / f"{mode}_state.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": n_steps,
        "mode": mode,
        "losses": losses,
    }, save_path)

    return {
        "mode": mode,
        "elapsed_s": elapsed,
        "params": params["total"],
        "losses": losses,
        "final_loss": losses[-1]["loss"] if losses else None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CompressorLM A/B test")
    parser.add_argument("modes", nargs="*", default=["pipeline", "iterative"],
                        help="Modes to run: cube, pipeline, iterative (default: pipeline iterative)")
    parser.add_argument("--steps", type=int, default=N_STEPS)
    args = parser.parse_args()

    n_steps = args.steps

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"CompressorLM A/B Test — {', '.join(args.modes)}")
    print(f"Device: {device}")
    print(f"Steps: {n_steps}, Log every {LOG_INTERVAL}")
    print(f"Tokens per run: {n_steps * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN:,}")
    print(f"Timestamp: {datetime.now(UTC).isoformat()}")

    # Load any existing results
    comp_path = RESULTS_DIR / "comparison.json"
    if comp_path.exists():
        existing = json.loads(comp_path.read_text())
    else:
        existing = {}

    # Run requested modes
    results = {}
    for mode in args.modes:
        results[mode] = train_run(mode, device, seed=42, n_steps=n_steps)

    # Merge with existing results
    for mode in existing:
        if mode not in results and mode not in ("timestamp", "config"):
            if isinstance(existing[mode], dict) and "losses" in existing[mode]:
                results[mode] = existing[mode]

    # ── Comparison ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}\n")

    modes = list(results.keys())
    all_losses = {m: results[m]["losses"] for m in modes}
    n_steps_logged = min(len(v) for v in all_losses.values())

    # Print header
    header = f"  {'Step':>5}"
    for m in modes:
        header += f"  {m:>10}"
    print(header)
    print(f"  {'─'*5}" + f"  {'─'*10}" * len(modes))

    wins = {m: 0 for m in modes}
    for i in range(n_steps_logged):
        step = all_losses[modes[0]][i]["step"]
        losses_at_step = {m: all_losses[m][i]["loss"] for m in modes}
        best_mode = min(losses_at_step, key=losses_at_step.get)
        # Only count as a win if margin > 0.01
        second_best = sorted(losses_at_step.values())[1] if len(modes) > 1 else float('inf')
        if second_best - losses_at_step[best_mode] > 0.01:
            wins[best_mode] += 1

        row = f"  {step:5d}"
        for m in modes:
            marker = " *" if m == best_mode and second_best - losses_at_step[best_mode] > 0.1 else "  "
            row += f"  {losses_at_step[m]:8.4f}{marker}"
        if step % 50 == 0 or step <= 50 or i >= n_steps_logged - 5:
            print(row)

    print()
    for m in modes:
        final = all_losses[m][-1]["loss"]
        print(f"  {m:>10} final: {final:.4f}  wins: {wins[m]}/{n_steps_logged}  params: {results[m]['params']:,}")
    print()
    for m in modes:
        print(f"  {m:>10} time: {results[m]['elapsed_s']:.0f}s")

    # Also compare to v1 at matched steps
    v1_path = Path("results/montagu-lm/training-summary.json")
    if v1_path.exists():
        v1 = json.loads(v1_path.read_text())
        v1_1000 = next((e for e in v1["eval_losses"] if e["step"] == 1000), None)
        if v1_1000:
            print(f"\n  v1 (rigid) at step 1000: {v1_1000['loss']:.4f}")
            for m in modes:
                final = all_losses[m][-1]["loss"]
                print(f"  {m:>10} at step 1000: {final:.4f}")

    # Save results
    save_path = RESULTS_DIR / "comparison.json"
    save_data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM, "lr": LEARNING_RATE,
            "n_steps": n_steps, "d_model": D_MODEL,
            "window": 8, "strides": [1, 8, 64],
        },
    }
    for m in modes:
        save_data[m] = results[m]
    save_path.write_text(json.dumps(save_data, indent=2))
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
