#!/usr/bin/env python3
"""VSM-LM-v3.1 — 1B token training run.

Phased compression with global integration:
  4 strides (1, 8, 64, 512) — full 4096-token sequence coverage
  4 registers (type/scope/role/coherence) — phased compression hypothesis
  4 phases: type → parse → apply → integrate
  2 CompressorLayers per phase (16 FFN passes per forward)
  ~30,518 steps at batch_size=2 × grad_accum=4 × seq_len=4096 = 32,768 tok/step

Instrumentation:
  - Per-register norms (type/scope/role/coherence) at every checkpoint
  - Soft partition write gate values (16 values per checkpoint)
  - Gate divergence across iterations
  - Compile gate test

Usage:
    uv run python scripts/run_vsm_v3_1_10k.py
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
# Config — 1B token run
# ══════════════════════════════════════════════════════════════════════

VOCAB_SIZE = 50277
D_MODEL = 512
D_REGISTER = 256
SEQ_LEN = 4096
D_FF = 1536
WINDOW = 8
STRIDES = (1, 8, 64, 512)
N_HEADS = 8
N_LAYERS_PER_PHASE = 2

BATCH_SIZE = 2
GRAD_ACCUM = 4
TOKENS_PER_STEP = BATCH_SIZE * GRAD_ACCUM * SEQ_LEN  # 32,768
TARGET_TOKENS = 1_000_000_000
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
N_STEPS = TARGET_TOKENS // TOKENS_PER_STEP + 1  # 30,518
WARMUP_STEPS = 500
N_ITERATIONS = 2
SEED = 42

LOG_INTERVAL = 50
EVAL_INTERVAL = 500
CHECKPOINT_INTERVAL = 1000

REG_NAMES = ["type", "scope", "role", "coherence"]


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n", flush=True)


# ══════════════════════════════════════════════════════════════════════
# Data loader (same as v3)
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
# Instrumentation (updated for v3.1)
# ══════════════════════════════════════════════════════════════════════

def get_grad_norms(model):
    """Gradient norms by VSM system."""
    norms = {}

    # S1: per-phase (now stacks of 2 layers)
    for i, name in enumerate(model.phase_names):
        total = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.s1_stacks[i].parameters() if p.grad is not None
        ) ** 0.5
        norms[f"S1_{name}"] = total

    # S3: total
    norms["S3"] = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.s3.parameters() if p.grad is not None
    ) ** 0.5

    # S3: per-iteration gate head grad norms
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

    # S4
    norms["S4"] = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.s4.parameters() if p.grad is not None
    ) ** 0.5

    # S5: embeddings
    norms["S5_embed"] = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.token_embed.parameters() if p.grad is not None
    ) ** 0.5

    # S5: register inits (v3.1 uses model.register_inits ParameterDict)
    for rname in REG_NAMES:
        param = model.register_inits[f"reg_{rname}"]
        if param.grad is not None:
            norms[f"S5_register_{rname}"] = param.grad.norm().item()

    return norms


@torch.no_grad()
def measure_gate_divergence(model):
    """Measure how much iter0 and iter1 gate heads have diverged."""
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
    """Full VSM instrumentation via forward_instrumented."""
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
    """Test if model produces lambda notation."""
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
    from verbum.vsm_lm_v3_1 import VSMLMV3_1

    results_dir = Path("results/vsm-lm-v3.1")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path("checkpoints/vsm-lm-v3.1")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    start = time.time()
    banner(f"VSM-LM v3.1 — Phased Compression 1B TOKENS ({N_STEPS} STEPS)")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")

    tokens_total = N_STEPS * TOKENS_PER_STEP
    print(f"  Device: {device}")
    print(f"  Architecture: VSM-LM-v3.1 (4 registers, 2 layers/phase)")
    print(f"  S1 strides: {STRIDES}")
    print(f"  S1 layers per phase: {N_LAYERS_PER_PHASE}")
    print(f"  S1 order: type → parse → apply → integrate (fine→coarse)")
    print(f"  S4: 4-register cross-attention (per-iteration)")
    print(f"  S3: per-dimension gating + soft-partitioned register writes")
    print(f"  Registers: {len(STRIDES)} × d_register={D_REGISTER}")
    print(f"  Iterations: {N_ITERATIONS}")
    print(f"  FFN passes/forward: {len(STRIDES) * N_LAYERS_PER_PHASE * N_ITERATIONS}")
    print(f"  Seq len: {SEQ_LEN}")
    print(f"  Batch: {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Steps: {N_STEPS}")
    print(f"  Tokens: {tokens_total:,}")
    print(f"  Data: SHUFFLED", flush=True)

    # ── Build model ───────────────────────────────────────────────────
    banner("BUILDING MODEL")

    model = VSMLMV3_1(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, d_register=D_REGISTER,
        max_len=SEQ_LEN, n_heads=N_HEADS, d_ff=D_FF, window=WINDOW,
        strides=STRIDES, n_iterations=N_ITERATIONS,
        n_layers_per_phase=N_LAYERS_PER_PHASE,
    ).to(device)

    print(model.describe())
    print()
    params = model.count_parameters()
    for k, v in params.items():
        print(f"  {k:25s}: {v:>12,}")

    non_embed = params["S4_intelligence"] + params["S3_control"] + params["S1_operations"] + params["S5_other"]
    print(f"  {'non_embedding':25s}: {non_embed:>12,}  ({non_embed / params['total'] * 100:.1f}%)")
    print(flush=True)

    # ── Data ──────────────────────────────────────────────────────────
    train_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, "train", seed=SEED)
    eval_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, "eval", seed=SEED + 1)

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, N_STEPS - WARMUP_STEPS)
        return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Training ──────────────────────────────────────────────────────
    banner("TRAINING")

    model.train()
    train_losses = []
    eval_losses = []
    checkpoints_data = []

    for step in range(1, N_STEPS + 1):
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
            total_tokens = step * TOKENS_PER_STEP
            tps = total_tokens / elapsed
            pct = total_tokens / TARGET_TOKENS * 100
            reg_norms = " ".join(
                f"{n}={model.register_inits[f'reg_{n}'].data.norm().item():.3f}"
                for n in REG_NAMES
            )
            print(
                f"  step {step:5d}/{N_STEPS}  "
                f"loss={accum_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"regs=[{reg_norms}]  "
                f"tokens={total_tokens/1e6:.0f}M ({pct:.0f}%)  "
                f"tok/s={tps:.0f}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

        if step % EVAL_INTERVAL == 0:
            eval_loader.reset()
            el = estimate_loss(model, eval_loader, device)
            eval_losses.append({"step": step, "loss": el})
            print(f"  ── eval loss at step {step}: {el:.4f} ──", flush=True)

        if step % CHECKPOINT_INTERVAL == 0:
            grad_norms = get_grad_norms(model)
            gate_div = measure_gate_divergence(model)

            eval_loader.reset()
            dynamics = measure_dynamics(model, eval_loader, device)

            compile = compile_gate_test(model, tokenizer, device)

            # Compute write gate partition matrix (the key v3.1 signal)
            partition_matrix = {}
            for it in range(N_ITERATIONS):
                for phase in model.phase_names:
                    for rn in REG_NAMES:
                        k = f"iter{it}_{phase}_write_{rn}"
                        partition_matrix[k] = dynamics.get(k, 0)

            # Compute register trajectories (init → S4 → phase0 → ... → final)
            reg_trajectories = {}
            for rn in REG_NAMES:
                traj = [dynamics.get(f"register_{rn}_init_norm", 0)]
                for it in range(N_ITERATIONS):
                    traj.append(dynamics.get(f"iter{it}_reg_{rn}_after_s4", 0))
                    for phase in model.phase_names:
                        traj.append(dynamics.get(f"iter{it}_{phase}_reg_{rn}_norm", 0))
                reg_trajectories[rn] = traj

            # Gating ratios (throughput: how much of each phase's delta survives)
            gating_ratios = {}
            for it in range(N_ITERATIONS):
                for phase in model.phase_names:
                    delta = dynamics.get(f"iter{it}_{phase}_delta_norm", 1e-8)
                    gated = dynamics.get(f"iter{it}_{phase}_gated_norm", 0)
                    gating_ratios[f"iter{it}_{phase}"] = round(gated / max(delta, 1e-8), 6)

            ckpt_info = {
                "step": step,
                "train_loss": accum_loss,
                "eval_loss": eval_losses[-1]["loss"] if eval_losses else None,
                "grad_norms": grad_norms,
                "gate_divergence": gate_div,
                "dynamics": dynamics,
                "partition_matrix": partition_matrix,
                "register_trajectories": reg_trajectories,
                "gating_ratios": gating_ratios,
                "compile_gate": compile["score"],
            }
            checkpoints_data.append(ckpt_info)

            # Print summary
            print(f"  ── checkpoint {step} ({step * TOKENS_PER_STEP / 1e6:.0f}M tokens) ──")
            print(f"     grad norms: {json.dumps({k: round(v, 4) for k, v in grad_norms.items()})}")
            print(f"     expansion: {dynamics['overall_expansion']:.2f}x")

            # Per-iteration S4 entropy
            for it in range(N_ITERATIONS):
                s4_key = f"iter{it}_s4_attn_entropy"
                if s4_key in dynamics:
                    print(f"     iter{it} S4 entropy: {dynamics[s4_key]:.4f}")

            # Register trajectories (4 registers)
            for rn in REG_NAMES:
                parts = [f"init={dynamics.get(f'register_{rn}_init_norm', 0):.4f}"]
                for it in range(N_ITERATIONS):
                    parts.append(f"S4.{it}={dynamics.get(f'iter{it}_reg_{rn}_after_s4', 0):.4f}")
                    parts.append(f"iter{it}={dynamics.get(f'iter{it}_register_{rn}_norm', 0):.4f}")
                print(f"     reg_{rn}: {' → '.join(parts)}")

            # Gate summary (per-iteration)
            for it in range(N_ITERATIONS):
                gate_str = "  ".join(
                    f"{name}={dynamics.get(f'iter{it}_{name}_gate_mean', 0):.3f}"
                    f"±{dynamics.get(f'iter{it}_{name}_gate_std', 0):.3f}"
                    for name in model.phase_names
                )
                print(f"     iter{it} gates: {gate_str}")

            # Soft partition write gates (the key v3.1 metric)
            print(f"     soft partition (write gates):")
            for it in range(N_ITERATIONS):
                for phase in model.phase_names:
                    vals = " ".join(
                        f"{rn}={dynamics.get(f'iter{it}_{phase}_write_{rn}', 0):.3f}"
                        for rn in REG_NAMES
                    )
                    print(f"       iter{it}/{phase}: {vals}")

            # Gate head divergence
            div_str = "  ".join(
                f"{name}={gate_div[f'gate_cosine_{name}']:.3f}"
                for name in model.phase_names
            )
            print(f"     gate divergence (cosine iter0↔iter1): {div_str}")
            print(f"     compile gate: {compile['score']}")

            # Save checkpoint with full instrumentation
            ckpt_path = checkpoint_dir / f"step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": accum_loss,
                "dynamics": dynamics,
                "grad_norms": grad_norms,
                "gate_divergence": gate_div,
                "partition_matrix": partition_matrix,
                "register_trajectories": reg_trajectories,
                "gating_ratios": gating_ratios,
                "compile_gate": compile["score"],
                "compile_gate_results": compile["results"],
                "train_losses": train_losses[:],
                "eval_losses": eval_losses[:],
                "architecture": "vsm-lm-v3.1",
                "config": {
                    "d_model": D_MODEL,
                    "d_register": D_REGISTER,
                    "d_ff": D_FF,
                    "n_heads": N_HEADS,
                    "n_layers_per_phase": N_LAYERS_PER_PHASE,
                    "n_iterations": N_ITERATIONS,
                    "strides": list(STRIDES),
                    "window": WINDOW,
                    "vocab_size": VOCAB_SIZE,
                    "seq_len": SEQ_LEN,
                },
            }, ckpt_path)
            print(f"     saved: {ckpt_path}", flush=True)

            model.train()

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start
    banner(f"DONE — {elapsed:.0f}s ({elapsed / 3600:.1f}h)")

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "architecture": "VSM-LM-v3.1 (phased compression, 1B tokens)",
        "target_tokens": TARGET_TOKENS,
        "tokens_per_step": TOKENS_PER_STEP,
        "strides": list(STRIDES),
        "n_iterations": N_ITERATIONS,
        "n_layers_per_phase": N_LAYERS_PER_PHASE,
        "d_register": D_REGISTER,
        "n_registers": len(STRIDES),
        "ffn_passes_per_forward": len(STRIDES) * N_LAYERS_PER_PHASE * N_ITERATIONS,
        "s4_mode": "4-register cross-attention, per-iteration",
        "s3_mode": "per-dimension gating + soft-partitioned register writes",
        "s1_order": "fine_to_coarse",
        "v3_changes": [
            "4 partitioned registers (type/scope/role/coherence, 128 dims each)",
            "4 phases with strides (1, 8, 64, 512)",
            "2 CompressorLayers per phase (16 FFN passes)",
            "16 soft-partitioned write paths",
        ],
        "params": params,
        "n_steps": N_STEPS,
        "seed": SEED,
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "checkpoints": checkpoints_data,
    }
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Saved: {summary_path}")

    # Comparison
    print()
    print("  Reference:")
    print("    Forward CompressorLM:  best eval 5.043 @ step 9500")
    print("    Reverse CompressorLM:  best eval 5.342 @ step 9500")
    print("    VSM-LM v1:            best eval 5.245 @ step 9500")
    print("    VSM-LM v2:            best eval 5.064 @ step 29500 (1B tokens)")
    print("    VSM-LM v3:            best eval 4.872 @ step 10000")
    print()
    if eval_losses:
        best = min(eval_losses, key=lambda e: e["loss"])
        last_dyn = checkpoints_data[-1]["dynamics"] if checkpoints_data else {}
        tokens_at_best = best["step"] * TOKENS_PER_STEP
        print(f"  This run (VSM-LM-v3.1, 1B tokens):")
        print(f"    Best eval: {best['loss']:.3f} @ step {best['step']} ({tokens_at_best/1e6:.0f}M tokens)")
        print(f"    Overall expansion: {last_dyn.get('overall_expansion', '?')}x")
        if last_dyn:
            for it in range(N_ITERATIONS):
                gate_str = ", ".join(
                    f"{name}={last_dyn.get(f'iter{it}_{name}_gate_mean', 0):.3f}"
                    for name in model.phase_names
                )
                print(f"    iter{it} gates: {gate_str}")
            # Soft partition summary
            print(f"    Soft partition (final):")
            for phase in model.phase_names:
                vals = ", ".join(
                    f"{rn}={last_dyn.get(f'iter1_{phase}_write_{rn}', 0):.3f}"
                    for rn in REG_NAMES
                )
                print(f"      {phase}: {vals}")


if __name__ == "__main__":
    main()
