#!/usr/bin/env python3
"""Register probe — fast A/B test, 2000 steps.

Trains TWO models simultaneously with identical seeds:
  A: reverse tesseract, NO register (baseline)
  B: reverse tesseract, WITH register

Same batches, same initialization (except register params).
Measures cosines every 200 steps. Direct comparison eliminates noise.

The question: does the register shift cosines toward positive
(genuine prediction) vs the baseline's anti-correlation?

If yes by step 2000: run the full 10K.
If no: the register design needs rethinking.

Usage:
    uv run python scripts/run_register_probe.py
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

DATA_DIR = Path("/Users/mwhitford/data/fractal-bitnet/shards")

# ══════════════════════════════════════════════════════════════════════
# Config — same as the 10K runs
# ══════════════════════════════════════════════════════════════════════

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
N_STEPS = 2000
WARMUP_STEPS = 500
N_ITERATIONS = 2
SEED = 42

MEASURE_INTERVAL = 200  # dense measurement


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n", flush=True)


# ══════════════════════════════════════════════════════════════════════
# Data loader
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
        buf = torch.from_numpy(np.stack(sequences)).long()
        return buf[:, :T], buf[:, 1 : T + 1]

    def reset(self):
        self._idx_pos = 0


# ══════════════════════════════════════════════════════════════════════
# Phase measurement (works for both register and non-register models)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_cosines(model, eval_loader, device, n_batches=5):
    """Measure prediction cosines for a model. Returns compact dict."""
    model.eval()
    block = model.block

    all_metrics = []
    for _ in range(n_batches):
        x_ids, _ = eval_loader.next_batch()
        x_ids = x_ids.to(device)
        positions = torch.arange(x_ids.shape[1], device=device)
        x = model.token_embed(x_ids) + model.pos_embed(positions)

        m = {}
        register = block._init_register() if block.use_register else None

        for it in range(model.n_iterations):
            pfx = f"iter{it}"

            # Context
            x_ctx = block.context_layer(x)
            ctx_delta = x_ctx - x
            if register is not None:
                register = block._update_register(register, ctx_delta)

            # Apply
            if register is not None:
                apply_predicted = block._predict_with_register(
                    block.predict_apply, ctx_delta, register,
                )
            else:
                apply_predicted = block.predict_apply(ctx_delta)
            x_apply = block.apply_layer(x_ctx)
            apply_delta = x_apply - x_ctx
            apply_error = apply_delta - apply_predicted
            if register is not None:
                register = block._update_register(register, apply_error)

            # Parse
            x_with_apply = x_ctx + apply_error
            if register is not None:
                parse_predicted = block._predict_with_register(
                    block.predict_parse, apply_error, register,
                )
            else:
                parse_predicted = block.predict_parse(apply_error)
            x_parse = block.parse_layer(x_with_apply)
            parse_delta = x_parse - x_with_apply
            parse_error = parse_delta - parse_predicted
            if register is not None:
                register = block._update_register(register, parse_error)

            # Type
            x_with_parse = x_ctx + apply_error + parse_error
            if register is not None:
                type_predicted = block._predict_with_register(
                    block.predict_type, parse_error, register,
                )
            else:
                type_predicted = block.predict_type(parse_error)
            x_type = block.type_layer(x_with_parse)
            type_delta = x_type - x_with_parse
            type_error = type_delta - type_predicted
            if register is not None:
                register = block._update_register(register, type_error)

            x_out = x + ctx_delta + apply_error + parse_error + type_error

            # Cosines
            for name, delta, predicted, error in [
                ("apply", apply_delta, apply_predicted, apply_error),
                ("parse", parse_delta, parse_predicted, parse_error),
                ("type", type_delta, type_predicted, type_error),
            ]:
                cos = F.cosine_similarity(
                    predicted.reshape(-1, predicted.shape[-1]),
                    delta.reshape(-1, delta.shape[-1]),
                    dim=-1,
                ).mean().item()
                e_n = error.norm(dim=-1).mean().item()
                d_n = delta.norm(dim=-1).mean().item()
                m[f"{pfx}_{name}_cos"] = cos
                m[f"{pfx}_{name}_expansion"] = e_n / d_n if d_n > 0 else 0.0

            x = x_out

        # Register metrics
        if register is not None:
            m["register_final_norm"] = register.norm().item()

        m["output_norm"] = x_out.norm(dim=-1).mean().item()
        m["embed_norm"] = model.token_embed(x_ids).norm(dim=-1).mean().item()

        all_metrics.append(m)

    # Average
    keys = all_metrics[0].keys()
    avg = {k: round(sum(d[k] for d in all_metrics) / len(all_metrics), 6)
           for k in keys}
    model.train()
    return avg


def estimate_loss(model, eval_loader, device, n_batches=10):
    model.eval()
    total = 0
    for _ in range(n_batches):
        x, y = eval_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / n_batches


# ══════════════════════════════════════════════════════════════════════
# Main — A/B test
# ══════════════════════════════════════════════════════════════════════

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    from verbum.compressor_lm import CompressorLM

    results_dir = Path("results/register-probe")
    results_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    banner("REGISTER PROBE — A/B TEST, 2000 STEPS")

    # ── Build both models — copy weights for exact match ─────────────
    print("  Building model A (baseline, no register)...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model_a = CompressorLM(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=SEQ_LEN,
        d_ff=D_FF, window=WINDOW, strides=STRIDES, mode="iterative",
        n_iterations=N_ITERATIONS, reverse=True, use_register=False,
    ).to(device)

    print("  Building model B (with register, weights copied from A)...")
    model_b = CompressorLM(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=SEQ_LEN,
        d_ff=D_FF, window=WINDOW, strides=STRIDES, mode="iterative",
        n_iterations=N_ITERATIONS, reverse=True, use_register=True,
    ).to(device)

    # Copy all shared weights from A → B (exact match)
    a_state = model_a.state_dict()
    b_state = model_b.state_dict()
    for key in a_state:
        if key in b_state:
            b_state[key] = a_state[key].clone()
    model_b.load_state_dict(b_state)

    pa = model_a.count_parameters()
    pb = model_b.count_parameters()
    print(f"  Model A params: {pa['total']:,}")
    print(f"  Model B params: {pb['total']:,}")
    print(f"  Register overhead: {pb['total'] - pa['total']:,} ({(pb['total'] - pa['total']) / pa['total'] * 100:.2f}%)")
    print(flush=True)

    # Verify shared weights match (excluding register params)
    mismatch = 0
    b_state_check = model_b.state_dict()
    for key in a_state:
        if key in b_state_check and not torch.equal(
            a_state[key].to(device), b_state_check[key],
        ):
            mismatch += 1
    print(f"  Weight verification: {mismatch} mismatches (should be 0)")

    # ── Data ──────────────────────────────────────────────────────────
    # Two train loaders with same seed = same batches
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_loader_a = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, "train", seed=SEED)
    train_loader_b = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, "train", seed=SEED)
    eval_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, "eval", seed=SEED + 1)

    # ── Optimizers ────────────────────────────────────────────────────
    opt_a = torch.optim.AdamW(
        model_a.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    opt_b = torch.optim.AdamW(
        model_b.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, N_STEPS - WARMUP_STEPS)
        return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))

    sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, lr_schedule)
    sched_b = torch.optim.lr_scheduler.LambdaLR(opt_b, lr_schedule)

    # ── Training ──────────────────────────────────────────────────────
    banner("TRAINING (A/B parallel)")

    model_a.train()
    model_b.train()
    measurements = []

    for step in range(1, N_STEPS + 1):
        # Same batches for both models
        opt_a.zero_grad()
        opt_b.zero_grad()
        loss_a_accum = 0
        loss_b_accum = 0

        for _ in range(GRAD_ACCUM):
            xa, ya = train_loader_a.next_batch()
            xb, yb = train_loader_b.next_batch()
            xa, ya = xa.to(device), ya.to(device)
            xb, yb = xb.to(device), yb.to(device)

            _, loss_a = model_a(xa, ya)
            (loss_a / GRAD_ACCUM).backward()
            loss_a_accum += loss_a.item() / GRAD_ACCUM

            _, loss_b = model_b(xb, yb)
            (loss_b / GRAD_ACCUM).backward()
            loss_b_accum += loss_b.item() / GRAD_ACCUM

        torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
        opt_a.step()
        opt_b.step()
        sched_a.step()
        sched_b.step()

        if step % 50 == 0:
            elapsed = time.time() - start
            reg_norm = model_b.block.register_init.data.norm().item()
            print(
                f"  step {step:5d}/{N_STEPS}  "
                f"A={loss_a_accum:.4f}  B={loss_b_accum:.4f}  "
                f"Δ={loss_b_accum - loss_a_accum:+.4f}  "
                f"reg={reg_norm:.4f}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

        if step % MEASURE_INTERVAL == 0:
            print(f"\n  ── measuring step {step} ──")
            eval_loader.reset()
            cos_a = measure_cosines(model_a, eval_loader, device)
            eval_loader.reset()
            cos_b = measure_cosines(model_b, eval_loader, device)
            eval_loader.reset()
            eval_a = estimate_loss(model_a, eval_loader, device)
            eval_loader.reset()
            eval_b = estimate_loss(model_b, eval_loader, device)

            record = {
                "step": step,
                "eval_a": eval_a,
                "eval_b": eval_b,
                "train_a": loss_a_accum,
                "train_b": loss_b_accum,
                "cosines_a": cos_a,
                "cosines_b": cos_b,
            }
            measurements.append(record)

            # Print comparison table
            print(f"     eval:  A={eval_a:.4f}  B={eval_b:.4f}  Δ={eval_b - eval_a:+.4f}")
            print(f"     {'':>14s}  {'baseline':>10s}  {'register':>10s}  {'Δ':>10s}")
            for it in range(N_ITERATIONS):
                for phase in ["apply", "parse", "type"]:
                    key = f"iter{it}_{phase}_cos"
                    ca = cos_a[key]
                    cb = cos_b[key]
                    marker = " ◀" if cb > ca else ""
                    print(f"     iter{it} {phase:>5s} cos  {ca:+10.4f}  {cb:+10.4f}  {cb - ca:+10.4f}{marker}")

            # Register state
            if "register_final_norm" in cos_b:
                print(f"     register norm: {cos_b['register_final_norm']:.4f}")

            # Expansion comparison
            for it in range(N_ITERATIONS):
                for phase in ["apply", "parse", "type"]:
                    key = f"iter{it}_{phase}_expansion"
                    ea = cos_a.get(key, 0)
                    eb = cos_b.get(key, 0)

            print(flush=True)
            model_a.train()
            model_b.train()

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start
    banner(f"DONE — {elapsed:.0f}s ({elapsed / 3600:.1f}h)")

    # Final verdict
    if measurements:
        last = measurements[-1]
        print("  FINAL COMPARISON (step 2000):")
        print(f"  {'':>14s}  {'baseline':>10s}  {'register':>10s}  {'Δ':>10s}")
        print(f"  {'eval loss':>14s}  {last['eval_a']:10.4f}  {last['eval_b']:10.4f}  {last['eval_b'] - last['eval_a']:+10.4f}")
        print()
        any_positive = False
        any_improved = False
        for it in range(N_ITERATIONS):
            for phase in ["apply", "parse", "type"]:
                key = f"iter{it}_{phase}_cos"
                ca = last["cosines_a"][key]
                cb = last["cosines_b"][key]
                delta = cb - ca
                if cb > 0:
                    any_positive = True
                if delta > 0.05:
                    any_improved = True
                marker = " ★" if cb > 0 else (" ◀" if delta > 0.05 else "")
                print(f"  iter{it} {phase:>5s} cos  {ca:+10.4f}  {cb:+10.4f}  {delta:+10.4f}{marker}")

        print()
        if any_positive:
            print("  ★ POSITIVE COSINES DETECTED — register enables genuine prediction")
            print("  → Proceed to full 10K run")
        elif any_improved:
            print("  ◀ Cosines improved but still negative — register helps but doesn't flip")
            print("  → Consider: stronger register (larger, or attention-based read)")
        else:
            print("  ✗ No improvement — register design needs rethinking")
            print("  → Consider: register may need to be per-position, not global")

    # Save
    summary = {
        "elapsed_s": elapsed,
        "measurements": measurements,
    }
    summary_path = results_dir / "probe-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved: {summary_path}")


if __name__ == "__main__":
    main()
