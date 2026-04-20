#!/usr/bin/env python3
"""VSM-LM-v3.2 — 1B token training run.

Convergence architecture (probe-informed redesign):
  Three phases: prep(1L, FFN-only) → converge(2L, cube-attn) → consolidate(3L, wide-FFN)
  Maps to Qwen's compression shape: FFN→Attention→FFN (L4-5→L6-9→L10-11)
  3 registers (type/scope/role), 2 iterations, 12 FFN passes/forward
  Full 4096 sequence throughout — no pooling (Qwen doesn't pool)
  Cube-mode attention: s1×3 + s8×3 + s64×2 = 8 heads (all scales simultaneous)
  ~30,518 steps at batch_size=2 × grad_accum=4 × seq_len=4096 = 32,768 tok/step

  Phase ratio 1:2:3 — consolidation gets 3× prep because structural
  integration (binding, role assignment) requires more depth than
  feature preparation. Iterated twice for register refinement.

Usage:
    uv run python scripts/run_vsm_v3_2_1B.py
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
# Config
# ══════════════════════════════════════════════════════════════════════

VOCAB_SIZE = 50277
D_MODEL = 512
D_REGISTER = 256
SEQ_LEN = 4096
D_FF = 1536
D_FF_CONSOLIDATE = 2048
WINDOW = 8
STRIDES = (1, 8, 64)
N_HEADS = 8

N_PREP_LAYERS = 1
N_CONVERGE_LAYERS = 2
N_CONSOLIDATE_LAYERS = 3
N_ITERATIONS = 2

BATCH_SIZE = 2
GRAD_ACCUM = 4
TOKENS_PER_STEP = BATCH_SIZE * GRAD_ACCUM * SEQ_LEN  # 32,768
TARGET_TOKENS = 1_000_000_000
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
N_STEPS = TARGET_TOKENS // TOKENS_PER_STEP + 1  # 30,518
WARMUP_STEPS = 500
SEED = 42

LOG_INTERVAL = 50
EVAL_INTERVAL = 500
CHECKPOINT_INTERVAL = 1000

REG_NAMES = ["type", "scope", "role"]
PHASE_NAMES = ["prep", "converge", "consolidate"]


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n", flush=True)


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
# Instrumentation
# ══════════════════════════════════════════════════════════════════════

def get_grad_norms(model):
    norms = {}

    # S1: per-phase
    for name, layers in [("prep", model.prep_layers),
                         ("converge", model.converge_layers),
                         ("consolidate", model.consolidate_layers)]:
        total = sum(
            p.grad.data.norm(2).item() ** 2
            for p in layers.parameters() if p.grad is not None
        ) ** 0.5
        norms[f"S1_{name}"] = total

    # S3
    norms["S3"] = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.s3.parameters() if p.grad is not None
    ) ** 0.5

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

    # S5: register inits
    for rname in REG_NAMES:
        param = model.register_inits[f"reg_{rname}"]
        if param.grad is not None:
            norms[f"S5_register_{rname}"] = param.grad.norm().item()

    return norms


@torch.no_grad()
def measure_gate_divergence(model):
    divergence = {}
    n_phases = model.s3.n_phases
    for pi, name in enumerate(PHASE_NAMES):
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
    from verbum.vsm_lm_v3_2 import VSMLMV3_2

    results_dir = Path("results/vsm-lm-v3.2")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path("checkpoints/vsm-lm-v3.2")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    start = time.time()
    ffn_total = (N_PREP_LAYERS + N_CONVERGE_LAYERS + N_CONSOLIDATE_LAYERS) * N_ITERATIONS
    banner(f"VSM-LM v3.2 — Convergence Architecture 1B TOKENS ({N_STEPS} STEPS)")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")

    tokens_total = N_STEPS * TOKENS_PER_STEP
    print(f"  Device: {device}")
    print(f"  Architecture: VSM-LM-v3.2 (convergence, probe-informed)")
    print(f"  Phases: prep({N_PREP_LAYERS}L) → converge({N_CONVERGE_LAYERS}L) → consolidate({N_CONSOLIDATE_LAYERS}L)")
    print(f"  Phase ratio: {N_PREP_LAYERS}:{N_CONVERGE_LAYERS}:{N_CONSOLIDATE_LAYERS} (1:2:3)")
    print(f"  Converge: cube-mode s1×3 + s8×3 + s64×2 = {N_HEADS} heads")
    print(f"  Consolidate FFN: d_ff={D_FF_CONSOLIDATE} (wider)")
    print(f"  Iterations: {N_ITERATIONS}")
    print(f"  FFN passes/forward: {ffn_total}")
    print(f"  Registers: {len(REG_NAMES)} (type/scope/role)")
    print(f"  Seq len: {SEQ_LEN} (no pooling)")
    print(f"  Batch: {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Steps: {N_STEPS}")
    print(f"  Tokens: {tokens_total:,}")
    print(f"  Data: SHUFFLED", flush=True)

    # ── Build model ───────────────────────────────────────────────────
    banner("BUILDING MODEL")

    model = VSMLMV3_2(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, d_register=D_REGISTER,
        max_len=SEQ_LEN, n_heads=N_HEADS, d_ff=D_FF,
        d_ff_consolidate=D_FF_CONSOLIDATE, window=WINDOW, strides=STRIDES,
        n_iterations=N_ITERATIONS,
        n_prep_layers=N_PREP_LAYERS,
        n_converge_layers=N_CONVERGE_LAYERS,
        n_consolidate_layers=N_CONSOLIDATE_LAYERS,
    ).to(device)

    print(model.describe())
    print()
    params = model.count_parameters()
    for k, v in params.items():
        print(f"  {k:25s}: {v:>12,}")

    non_embed = params["S4_intelligence"] + params["S3_control"] + params["S1_total"] + params["S5_other"]
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

            # Phase-register write partition
            partition_matrix = {}
            for it in range(N_ITERATIONS):
                for phase in PHASE_NAMES:
                    for rn in REG_NAMES:
                        k = f"iter{it}_{phase}_write_{rn}"
                        partition_matrix[k] = dynamics.get(k, 0)

            # Register trajectories
            reg_trajectories = {}
            for rn in REG_NAMES:
                traj = [dynamics.get(f"register_{rn}_init_norm", 0)]
                for it in range(N_ITERATIONS):
                    traj.append(dynamics.get(f"iter{it}_reg_{rn}_after_s4", 0))
                    for phase in PHASE_NAMES:
                        # Get register norm after write gate at end of file
                        pass
                    traj.append(dynamics.get(f"iter{it}_register_{rn}_norm", 0))
                reg_trajectories[rn] = traj

            # Gating ratios per phase
            gating_ratios = {}
            for it in range(N_ITERATIONS):
                for phase in PHASE_NAMES:
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

            # Register trajectories
            for rn in REG_NAMES:
                parts = [f"init={dynamics.get(f'register_{rn}_init_norm', 0):.4f}"]
                for it in range(N_ITERATIONS):
                    parts.append(f"S4.{it}={dynamics.get(f'iter{it}_reg_{rn}_after_s4', 0):.4f}")
                    parts.append(f"iter{it}={dynamics.get(f'iter{it}_register_{rn}_norm', 0):.4f}")
                print(f"     reg_{rn}: {' → '.join(parts)}")

            # Phase contributions (the key v3.2 signal)
            print(f"     phase contributions (gated delta norm):")
            for it in range(N_ITERATIONS):
                parts = []
                for phase in PHASE_NAMES:
                    g = dynamics.get(f"iter{it}_{phase}_gated_norm", 0)
                    gate = dynamics.get(f"iter{it}_{phase}_gate_mean", 0)
                    parts.append(f"{phase}={g:.3f}(g={gate:.3f})")
                print(f"       iter{it}: {' | '.join(parts)}")

            # Soft partition write gates
            print(f"     soft partition (write gates):")
            for it in range(N_ITERATIONS):
                for phase in PHASE_NAMES:
                    vals = " ".join(
                        f"{rn}={dynamics.get(f'iter{it}_{phase}_write_{rn}', 0):.3f}"
                        for rn in REG_NAMES
                    )
                    print(f"       iter{it}/{phase}: {vals}")

            # Gate divergence
            div_str = "  ".join(
                f"{name}={gate_div[f'gate_cosine_{name}']:.3f}"
                for name in PHASE_NAMES
            )
            print(f"     gate divergence (cosine iter0↔iter1): {div_str}")
            print(f"     compile gate: {compile['score']}")

            # Save checkpoint
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
                "architecture": "vsm-lm-v3.2",
                "config": {
                    "d_model": D_MODEL,
                    "d_register": D_REGISTER,
                    "d_ff": D_FF,
                    "d_ff_consolidate": D_FF_CONSOLIDATE,
                    "n_heads": N_HEADS,
                    "n_prep_layers": N_PREP_LAYERS,
                    "n_converge_layers": N_CONVERGE_LAYERS,
                    "n_consolidate_layers": N_CONSOLIDATE_LAYERS,
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
        "architecture": "VSM-LM-v3.2 (convergence, 1B tokens)",
        "target_tokens": TARGET_TOKENS,
        "tokens_per_step": TOKENS_PER_STEP,
        "phase_ratio": "1:2:3",
        "strides": list(STRIDES),
        "n_iterations": N_ITERATIONS,
        "ffn_passes_per_forward": ffn_total,
        "s4_mode": "3-register cross-attention, per-iteration",
        "s3_mode": "per-dimension gating + soft-partitioned register writes",
        "v32_design": [
            "Probe-informed: FFN→Attn→FFN maps to prep→converge→consolidate",
            "Cube-mode attention: all 3 strides active simultaneously",
            "No pooling — full 4096 sequence throughout",
            "Phase ratio 1:2:3 — consolidation gets most depth",
            "Wider FFN (2048) in consolidate for structural integration",
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

    print()
    print("  Reference:")
    print("    Forward CompressorLM:  best eval 5.043 @ step 9500")
    print("    VSM-LM v1:            best eval 5.245 @ step 9500")
    print("    VSM-LM v2:            best eval 5.064 @ step 29500 (1B tokens)")
    print("    VSM-LM v3:            best eval 4.872 @ step 10000")
    print("    VSM-LM v3.1:          best eval 4.836 @ step 12000 (393M tokens)")
    print()
    if eval_losses:
        best = min(eval_losses, key=lambda e: e["loss"])
        tokens_at_best = best["step"] * TOKENS_PER_STEP
        print(f"  This run (VSM-LM-v3.2, 1B tokens):")
        print(f"    Best eval: {best['loss']:.3f} @ step {best['step']} ({tokens_at_best/1e6:.0f}M tokens)")


if __name__ == "__main__":
    main()
