#!/usr/bin/env python3
"""VSM-LM-v2 — Resume training to 1B tokens.

Resumes from step 10K checkpoint. Targets 30,518 steps (1B tokens).
Uses re-warmup + cosine decay LR schedule over remaining steps.

Previous run (10K steps, 328M tokens):
  - Best eval: 5.256 @ step 9500
  - Expansion: 10.34x (compressing from 16.6x)
  - iter1 gates still opening, S4 entropy still diverging
  - Compression-loss coupling: r=0.935

Hypothesis: VSM floor is below forward compressor (5.043) within 1B tokens.
Projected crossover: ~600M tokens (~18K steps).

Usage:
    uv run python scripts/resume_vsm_v2_1B.py
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

# ══════════════════════════════════════════════════════════════════════
# Config — same as original run
# ══════════════════════════════════════════════════════════════════════

VOCAB_SIZE = 50277
D_MODEL = 256
SEQ_LEN = 4096
D_FF = 768
WINDOW = 8
STRIDES = (1, 8, 64)
N_HEADS = 8

BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
SEED = 42

# Resume config
RESUME_CHECKPOINT = Path("checkpoints/vsm-lm-v2/step_010000.pt")
N_ITERATIONS = 2

# Target: 1B tokens
TOKENS_PER_STEP = BATCH_SIZE * GRAD_ACCUM * SEQ_LEN  # 32,768
TARGET_TOKENS = 1_000_000_000
N_STEPS_TOTAL = TARGET_TOKENS // TOKENS_PER_STEP + 1  # 30,518

# LR schedule for resumed training
# Short re-warmup (200 steps) then cosine decay over remaining steps
REWARMUP_STEPS = 200
# Start LR at 50% of original (we're resuming mid-training, not cold starting)
RESUME_LR = 3e-4

LOG_INTERVAL = 50
EVAL_INTERVAL = 500
CHECKPOINT_INTERVAL = 1000


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n", flush=True)


# ══════════════════════════════════════════════════════════════════════
# Data loader (same as original)
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
# Instrumentation (same as original)
# ══════════════════════════════════════════════════════════════════════

def get_grad_norms(model):
    norms = {}
    for i, name in enumerate(model.phase_names):
        total = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.s1_layers[i].parameters() if p.grad is not None
        ) ** 0.5
        norms[f"S1_{name}"] = total

    norms["S3"] = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.s3.parameters() if p.grad is not None
    ) ** 0.5

    n_phases = model.s3.n_phases
    for it in range(model.n_iterations):
        for pi, name in enumerate(model.phase_names):
            head_idx = it * n_phases + pi
            head = model.s3.gate_heads[head_idx]
            total = sum(
                p.grad.data.norm(2).item() ** 2
                for p in head.parameters() if p.grad is not None
            ) ** 0.5
            norms[f"S3_iter{it}_{name}"] = total

    norms["S4"] = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.s4.parameters() if p.grad is not None
    ) ** 0.5

    norms["S5_embed"] = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.token_embed.parameters() if p.grad is not None
    ) ** 0.5

    if model.register_init.grad is not None:
        norms["S5_register"] = model.register_init.grad.norm().item()

    return norms


@torch.no_grad()
def measure_gate_divergence(model):
    divergence = {}
    n_phases = model.s3.n_phases
    for pi, name in enumerate(model.phase_names):
        iter0_head = model.s3.gate_heads[pi]
        iter1_head = model.s3.gate_heads[n_phases + pi]
        w0 = iter0_head.weight.data.flatten()
        w1 = iter1_head.weight.data.flatten()
        cos = F.cosine_similarity(w0.unsqueeze(0), w1.unsqueeze(0)).item()
        divergence[f"gate_cosine_{name}"] = round(cos, 6)
    return divergence


@torch.no_grad()
def measure_dynamics(model, eval_loader, device, n_batches=5):
    model.eval()
    all_metrics = []
    for _ in range(n_batches):
        x_ids, y = eval_loader.next_batch()
        x_ids = x_ids.to(device)
        _, _, metrics = model.forward_instrumented(x_ids)
        all_metrics.append(metrics)

    keys = all_metrics[0].keys()
    avg = {k: round(sum(d[k] for d in all_metrics) / len(all_metrics), 6)
           for k in keys}
    model.train()
    return avg


def compile_gate_test(model, tokenizer, device):
    prompts = [
        "λ",
        "The dog chased the cat",
        "Every student read a book",
        "compile: The cat sat on the mat",
    ]
    results = []
    model.eval()
    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        out = model.generate(ids, max_new_tokens=30, temperature=0.8)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        has_lambda = "λ" in text[len(prompt):] or "\\" in text[len(prompt):]
        results.append({"prompt": prompt, "output": text, "has_lambda": has_lambda})
    model.train()
    n_lambda = sum(1 for r in results if r["has_lambda"])
    return {"score": f"{n_lambda}/{len(prompts)}", "results": results}


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    from transformers import AutoTokenizer
    from verbum.vsm_lm_v2 import VSMLMV2

    results_dir = Path("results/vsm-lm-v2-1B")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path("checkpoints/vsm-lm-v2")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    start = time.time()

    # ── Load checkpoint ───────────────────────────────────────────
    banner("LOADING CHECKPOINT")
    assert RESUME_CHECKPOINT.exists(), f"Checkpoint not found: {RESUME_CHECKPOINT}"
    ckpt = torch.load(RESUME_CHECKPOINT, map_location="cpu", weights_only=False)
    resume_step = ckpt["step"]
    remaining_steps = N_STEPS_TOTAL - resume_step

    print(f"  Checkpoint: {RESUME_CHECKPOINT}")
    print(f"  Resuming from step: {resume_step}")
    print(f"  Target steps: {N_STEPS_TOTAL}")
    print(f"  Remaining steps: {remaining_steps}")
    print(f"  Tokens seen: {resume_step * TOKENS_PER_STEP:,}")
    print(f"  Target tokens: {TARGET_TOKENS:,}")
    print(f"  Checkpoint loss: {ckpt['loss']:.4f}")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")

    banner(f"VSM-LM v2 — Resume to 1B tokens ({N_STEPS_TOTAL} steps)")

    tokens_total = N_STEPS_TOTAL * TOKENS_PER_STEP
    print(f"  Device: {device}")
    print(f"  Architecture: VSM-LM-v2 (two-channel compressor)")
    print(f"  Resume step: {resume_step}")
    print(f"  Total steps: {N_STEPS_TOTAL}")
    print(f"  Remaining: {remaining_steps}")
    print(f"  Total tokens: {tokens_total:,}")
    print(f"  LR: {RESUME_LR} (re-warmup {REWARMUP_STEPS} steps)")
    print(f"  Data: SHUFFLED (new seed for data loader)", flush=True)

    # ── Build model and load state ────────────────────────────────
    banner("BUILDING MODEL")

    model = VSMLMV2(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=SEQ_LEN,
        n_heads=N_HEADS, d_ff=D_FF, window=WINDOW, strides=STRIDES,
        n_iterations=N_ITERATIONS,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print("  Model state loaded ✓")

    print(model.describe())
    params = model.count_parameters()
    for k, v in params.items():
        print(f"  {k:25s}: {v:>12,}")
    print(flush=True)

    # ── Data (fresh seed so we don't repeat sequences) ────────────
    # Use a different seed offset so we see different data ordering
    data_seed = SEED + resume_step
    train_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, "train", seed=data_seed)
    eval_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, "eval", seed=SEED + 1)

    # ── Optimizer ────────────────────────────────────────────────
    # Build fresh optimizer with RESUME_LR, then load momentum/variance
    # state from checkpoint for continuity. We explicitly reset LR after
    # loading because load_state_dict restores the old schedule's decayed LR.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=RESUME_LR, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # Force LR back to RESUME_LR (load_state_dict overwrites it)
    for pg in optimizer.param_groups:
        pg["lr"] = RESUME_LR
        pg["initial_lr"] = RESUME_LR  # LambdaLR reads this as base_lr

    print(f"  Optimizer state loaded ✓ (LR reset to {RESUME_LR})")

    def lr_schedule(step_offset):
        """LR schedule relative to resumed training (step_offset starts at 0)."""
        if step_offset < REWARMUP_STEPS:
            # Re-warmup from 10% to full
            return 0.1 + 0.9 * (step_offset / REWARMUP_STEPS)
        progress = (step_offset - REWARMUP_STEPS) / max(1, remaining_steps - REWARMUP_STEPS)
        return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Carry forward history from checkpoint ─────────────────────
    train_losses = ckpt.get("train_losses", [])
    eval_losses = ckpt.get("eval_losses", [])

    # ── Training ──────────────────────────────────────────────────
    banner("TRAINING (RESUMED)")

    model.train()
    checkpoints_data = []
    best_eval = min((e["loss"] for e in eval_losses), default=float("inf"))

    for step in range(resume_step + 1, N_STEPS_TOTAL + 1):
        step_offset = step - resume_step - 1  # 0-based offset for scheduler

        optimizer.zero_grad()
        accum_loss = 0

        for _ in range(GRAD_ACCUM):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item() / GRAD_ACCUM

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        train_losses.append(accum_loss)

        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - start
            total_elapsed = elapsed  # just this session
            total_tokens = step * TOKENS_PER_STEP
            tps = (step - resume_step) * TOKENS_PER_STEP / elapsed
            reg_norm = model.register_init.data.norm().item()
            pct = total_tokens / TARGET_TOKENS * 100
            print(
                f"  step {step:5d}/{N_STEPS_TOTAL}  "
                f"loss={accum_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"reg={reg_norm:.4f}  "
                f"tok/s={tps:.0f}  "
                f"tokens={total_tokens/1e6:.0f}M ({pct:.0f}%)  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

        if step % EVAL_INTERVAL == 0:
            eval_loader.reset()
            el = estimate_loss(model, eval_loader, device)
            eval_losses.append({"step": step, "loss": el})
            is_best = el < best_eval
            if is_best:
                best_eval = el
            marker = " ★ NEW BEST" if is_best else ""
            print(f"  ── eval loss at step {step}: {el:.4f}{marker} ──", flush=True)

        if step % CHECKPOINT_INTERVAL == 0:
            grad_norms = get_grad_norms(model)
            gate_div = measure_gate_divergence(model)
            eval_loader.reset()
            dynamics = measure_dynamics(model, eval_loader, device)
            compile = compile_gate_test(model, tokenizer, device)

            ckpt_info = {
                "step": step,
                "train_loss": accum_loss,
                "eval_loss": eval_losses[-1]["loss"] if eval_losses else None,
                "grad_norms": grad_norms,
                "gate_divergence": gate_div,
                "dynamics": dynamics,
                "compile_gate": compile["score"],
            }
            checkpoints_data.append(ckpt_info)

            # Print summary
            print(f"  ── checkpoint {step} ({step * TOKENS_PER_STEP / 1e6:.0f}M tokens) ──")
            print(f"     expansion: {dynamics['overall_expansion']:.2f}x")

            for it in range(N_ITERATIONS):
                s4_key = f"iter{it}_s4_attn_entropy"
                if s4_key in dynamics:
                    print(f"     iter{it} S4 entropy: {dynamics[s4_key]:.4f}")

            reg_parts = [f"init={dynamics['register_init_norm']:.4f}"]
            for it in range(N_ITERATIONS):
                s4_key = f"iter{it}_register_after_s4"
                if s4_key in dynamics:
                    reg_parts.append(f"S4.{it}={dynamics[s4_key]:.4f}")
                reg_parts.append(f"iter{it}={dynamics[f'iter{it}_register_norm']:.4f}")
            print(f"     register: {' → '.join(reg_parts)}")

            for it in range(N_ITERATIONS):
                gate_str = "  ".join(
                    f"{name}={dynamics[f'iter{it}_{name}_gate_mean']:.3f}"
                    f"±{dynamics[f'iter{it}_{name}_gate_std']:.3f}"
                    for name in model.phase_names
                )
                print(f"     iter{it} gates: {gate_str}")

            for it in range(N_ITERATIONS):
                ratio_str = "  ".join(
                    f"{name}={dynamics[f'iter{it}_{name}_gated_norm'] / max(dynamics[f'iter{it}_{name}_delta_norm'], 1e-8):.3f}"
                    for name in model.phase_names
                )
                print(f"     iter{it} throughput: {ratio_str}")

            div_str = "  ".join(
                f"{name}={gate_div[f'gate_cosine_{name}']:.3f}"
                for name in model.phase_names
            )
            print(f"     gate divergence (cosine iter0↔iter1): {div_str}")
            print(f"     compile gate: {compile['score']}")

            ckpt_path = checkpoint_dir / f"step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": accum_loss,
                "dynamics": dynamics,
                "grad_norms": grad_norms,
                "gate_divergence": gate_div,
                "compile_gate": compile["score"],
                "compile_gate_results": compile["results"],
                "train_losses": train_losses[:],
                "eval_losses": eval_losses[:],
            }, ckpt_path)
            print(f"     saved: {ckpt_path}", flush=True)

            model.train()

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - start
    banner(f"DONE — {elapsed:.0f}s ({elapsed / 3600:.1f}h) this session")

    total_tokens = N_STEPS_TOTAL * TOKENS_PER_STEP
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "architecture": "VSM-LM-v2 (two-channel compressor)",
        "resumed_from": str(RESUME_CHECKPOINT),
        "resume_step": resume_step,
        "total_steps": N_STEPS_TOTAL,
        "total_tokens": total_tokens,
        "resume_lr": RESUME_LR,
        "rewarmup_steps": REWARMUP_STEPS,
        "strides": list(STRIDES),
        "n_iterations": N_ITERATIONS,
        "params": model.count_parameters(),
        "eval_losses": eval_losses,
        "checkpoints": checkpoints_data,
    }
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Saved: {summary_path}")

    # Comparison
    print()
    print("  Reference:")
    print("    Forward CompressorLM:  best eval 5.043 @ step 9500 (328M tokens)")
    print("    Reverse CompressorLM:  best eval 5.342 @ step 9500 (328M tokens)")
    print(f"    VSM-LM-v2 @ 10K:      best eval 5.256 @ step 9500 (328M tokens)")
    print()
    if eval_losses:
        best = min(eval_losses, key=lambda e: e["loss"])
        tokens_at_best = best["step"] * TOKENS_PER_STEP
        last_dyn = checkpoints_data[-1]["dynamics"] if checkpoints_data else {}
        print(f"  This run (VSM-LM-v2 → 1B tokens):")
        print(f"    Best eval: {best['loss']:.4f} @ step {best['step']} ({tokens_at_best/1e6:.0f}M tokens)")
        print(f"    Overall expansion: {last_dyn.get('overall_expansion', '?')}x")
        if best["loss"] < 5.043:
            print(f"    ★ BEATS forward compressor by {5.043 - best['loss']:.4f}")
        else:
            print(f"    Behind forward compressor by {best['loss'] - 5.043:.4f}")


if __name__ == "__main__":
    main()
