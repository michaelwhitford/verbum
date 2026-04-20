#!/usr/bin/env python3
"""A/B test: fine→coarse vs coarse→fine predictive coding.

Runs both tesseract variants for 1K steps on shuffled Dolma data
with identical config and seed. Measures loss curves, expansion
ratios, and cosine similarity of predictions vs deltas.

Usage:
    uv run python scripts/run_reverse_ab.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATA_DIR = Path("/Users/mwhitford/data/fractal-bitnet/shards")

# ── Config (identical for both arms) ──────────────────────────────
VOCAB_SIZE = 50277
D_MODEL = 256
SEQ_LEN = 4096
D_FF = 768
WINDOW = 8
STRIDES = (1, 8, 64, 512)

BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
N_STEPS = 1000
WARMUP_STEPS = 500
N_ITERATIONS = 2
SEED = 42

LOG_INTERVAL = 10
EVAL_INTERVAL = 500


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


class ShardedDataLoader:
    """Shuffled data loader — same as run_tesseract.py."""

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
        buf = torch.from_numpy(np.stack(sequences)).long()
        return buf[:, :T], buf[:, 1 : T + 1]

    def reset(self):
        self._idx_pos = 0


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


@torch.no_grad()
def measure_expansion(model, eval_loader, device, n_batches=5):
    """Measure per-phase expansion ratios and cosine similarities."""
    model.eval()
    block = model.block
    is_reverse = block.reverse

    all_metrics = []
    for _ in range(n_batches):
        x_ids, _ = eval_loader.next_batch()
        x_ids = x_ids.to(device)

        positions = torch.arange(x_ids.shape[1], device=device)
        x = model.token_embed(x_ids) + model.pos_embed(positions)

        metrics = {}
        metrics["embed_norm"] = x.norm(dim=-1).mean().item()

        for it in range(model.n_iterations):
            pfx = f"iter{it}"

            if is_reverse:
                # Context first (coarsest)
                x_ctx = block.context_layer(x)
                ctx_delta = x_ctx - x

                # Apply: predicted by context
                apply_predicted = block.predict_apply(ctx_delta)
                x_apply = block.apply_layer(x_ctx)
                apply_delta = x_apply - x_ctx
                apply_error = apply_delta - apply_predicted

                # Parse: predicted by apply error
                x_with_apply = x_ctx + apply_error
                parse_predicted = block.predict_parse(apply_error)
                x_parse = block.parse_layer(x_with_apply)
                parse_delta = x_parse - x_with_apply
                parse_error = parse_delta - parse_predicted

                # Type: predicted by parse error
                x_with_parse = x_ctx + apply_error + parse_error
                type_predicted = block.predict_type(parse_error)
                x_type = block.type_layer(x_with_parse)
                type_delta = x_type - x_with_parse
                type_error = type_delta - type_predicted

                x_out = x + ctx_delta + apply_error + parse_error + type_error

                # Measure the predicted phases (apply, parse, type)
                for name, delta, predicted, error in [
                    ("apply", apply_delta, apply_predicted, apply_error),
                    ("parse", parse_delta, parse_predicted, parse_error),
                    ("type", type_delta, type_predicted, type_error),
                ]:
                    d_norm = delta.norm(dim=-1).mean().item()
                    e_norm = error.norm(dim=-1).mean().item()
                    p_norm = predicted.norm(dim=-1).mean().item()
                    cos = F.cosine_similarity(
                        predicted.reshape(-1, predicted.shape[-1]),
                        delta.reshape(-1, delta.shape[-1]),
                        dim=-1,
                    ).mean().item()
                    metrics[f"{pfx}_{name}_delta"] = d_norm
                    metrics[f"{pfx}_{name}_error"] = e_norm
                    metrics[f"{pfx}_{name}_predicted"] = p_norm
                    metrics[f"{pfx}_{name}_cos"] = cos
                    metrics[f"{pfx}_{name}_expansion"] = e_norm / d_norm if d_norm > 0 else 0

                # Lead phase (context) has no prediction
                metrics[f"{pfx}_context_delta"] = ctx_delta.norm(dim=-1).mean().item()
            else:
                # Type first (finest)
                x_type = block.type_layer(x)
                type_delta = x_type - x

                parse_predicted = block.predict_parse(type_delta)
                x_parse = block.parse_layer(x_type)
                parse_delta = x_parse - x_type
                parse_error = parse_delta - parse_predicted

                x_with_parse = x_type + parse_error
                apply_predicted = block.predict_apply(parse_error)
                x_apply = block.apply_layer(x_with_parse)
                apply_delta = x_apply - x_with_parse
                apply_error = apply_delta - apply_predicted

                x_with_apply = x_type + parse_error + apply_error
                context_predicted = block.predict_context(apply_error)
                x_context = block.context_layer(x_with_apply)
                context_delta = x_context - x_with_apply
                context_error = context_delta - context_predicted

                x_out = x + type_delta + parse_error + apply_error + context_error

                for name, delta, predicted, error in [
                    ("parse", parse_delta, parse_predicted, parse_error),
                    ("apply", apply_delta, apply_predicted, apply_error),
                    ("context", context_delta, context_predicted, context_error),
                ]:
                    d_norm = delta.norm(dim=-1).mean().item()
                    e_norm = error.norm(dim=-1).mean().item()
                    p_norm = predicted.norm(dim=-1).mean().item()
                    cos = F.cosine_similarity(
                        predicted.reshape(-1, predicted.shape[-1]),
                        delta.reshape(-1, delta.shape[-1]),
                        dim=-1,
                    ).mean().item()
                    metrics[f"{pfx}_{name}_delta"] = d_norm
                    metrics[f"{pfx}_{name}_error"] = e_norm
                    metrics[f"{pfx}_{name}_predicted"] = p_norm
                    metrics[f"{pfx}_{name}_cos"] = cos
                    metrics[f"{pfx}_{name}_expansion"] = e_norm / d_norm if d_norm > 0 else 0

                metrics[f"{pfx}_type_delta"] = type_delta.norm(dim=-1).mean().item()

            x = x_out

        metrics["output_norm"] = x_out.norm(dim=-1).mean().item()
        metrics["overall_expansion"] = metrics["output_norm"] / metrics["embed_norm"]
        all_metrics.append(metrics)

    # Average
    keys = all_metrics[0].keys()
    return {k: sum(d[k] for d in all_metrics) / len(all_metrics) for k in keys}


def train_arm(name: str, reverse: bool, device: str) -> dict:
    """Train one arm of the A/B test."""
    from verbum.compressor_lm import CompressorLM

    banner(f"{name} ({'coarse→fine' if reverse else 'fine→coarse'})")

    # Deterministic
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, "train", seed=SEED)
    eval_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, "eval", seed=SEED + 1)

    model = CompressorLM(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=SEQ_LEN,
        d_ff=D_FF, window=WINDOW, strides=STRIDES, mode="iterative",
        n_iterations=N_ITERATIONS, reverse=reverse,
    ).to(device)

    params = model.count_parameters()
    print(f"  Params: {params['total']:,}")
    print(f"  Reverse: {reverse}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        decay_ratio = (step - WARMUP_STEPS) / (N_STEPS - WARMUP_STEPS)
        return max(0.1, 0.5 * (1.0 + np.cos(np.pi * decay_ratio)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    start = time.time()
    train_losses = []
    eval_losses = []

    model.train()
    for step in range(1, N_STEPS + 1):
        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(GRAD_ACCUM):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            (loss / GRAD_ACCUM).backward()
            step_loss += loss.item() / GRAD_ACCUM

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(step_loss)

        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - start
            print(f"  [{name}] step {step:>5}/{N_STEPS}  "
                  f"loss={step_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"elapsed={elapsed:.0f}s")

        if step % EVAL_INTERVAL == 0:
            eval_loader.reset()
            el = estimate_loss(model, eval_loader, device)
            eval_losses.append({"step": step, "loss": el})
            print(f"  [{name}] *** eval loss = {el:.4f} ***")
            model.train()

    elapsed = time.time() - start

    # Final expansion measurement
    eval_loader.reset()
    expansion = measure_expansion(model, eval_loader, device)

    return {
        "name": name,
        "reverse": reverse,
        "params": params,
        "elapsed_s": elapsed,
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "expansion": expansion,
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    banner(f"REVERSE A/B TEST — {N_STEPS} steps, device={device}")

    results = {}

    # Arm A: current fine→coarse
    results["forward"] = train_arm("FORWARD", reverse=False, device=device)

    # Arm B: reversed coarse→fine
    results["reverse"] = train_arm("REVERSE", reverse=True, device=device)

    # ── Summary ───────────────────────────────────────────────────────
    banner("RESULTS")

    for arm in ["forward", "reverse"]:
        r = results[arm]
        exp = r["expansion"]
        direction = "coarse→fine" if r["reverse"] else "fine→coarse"
        print(f"  {r['name']} ({direction})")
        print(f"    Params: {r['params']['total']:,}")
        print(f"    Elapsed: {r['elapsed_s']:.0f}s")
        print(f"    Final train loss: {r['train_losses'][-1]:.4f}")
        for ev in r["eval_losses"]:
            print(f"    Eval @ step {ev['step']}: {ev['loss']:.4f}")
        print(f"    Overall expansion: {exp['overall_expansion']:.2f}x")
        print()

        # Phase details
        for it in range(2):
            pfx = f"iter{it}"
            print(f"    Iteration {it}:")
            if r["reverse"]:
                phases = ["apply", "parse", "type"]
                lead = "context"
            else:
                phases = ["parse", "apply", "context"]
                lead = "type"
            print(f"      Lead ({lead}) delta: {exp[f'{pfx}_{lead}_delta']:.4f}")
            for phase in phases:
                cos = exp.get(f"{pfx}_{phase}_cos", 0)
                expansion = exp.get(f"{pfx}_{phase}_expansion", 0)
                verdict = "COMPRESS" if expansion < 1.0 else "EXPAND"
                print(f"      {phase:>8}: cos={cos:+.4f}  "
                      f"expansion={expansion:.4f}x  {verdict}")
            print()

    # Save
    out_dir = Path("results/reverse-ab")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
