#!/usr/bin/env python3
"""Reversed tesseract (coarse→fine) — full 10K training run.

Same config as the shuffled tesseract that reached eval 5.04, but with
reversed predictive coding: context → apply → parse → type.

Checkpoints save full phase instrumentation:
  - gradient norms per phase
  - activation norms at phase boundaries
  - cosine similarity of predictions vs deltas
  - expansion ratios (||error|| / ||delta||)
  - eval loss
  - compile gate test (4 prompts)

Usage:
    uv run python scripts/run_reverse_10k.py
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
# Config — identical to tesseract-shuffled run
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
N_STEPS = 10000
WARMUP_STEPS = 500
N_ITERATIONS = 2
SEED = 42

LOG_INTERVAL = 50
EVAL_INTERVAL = 500
CHECKPOINT_INTERVAL = 1000


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n", flush=True)


# ══════════════════════════════════════════════════════════════════════
# Data loader (shuffled, same as tesseract-shuffled)
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
# Phase instrumentation
# ══════════════════════════════════════════════════════════════════════

def get_grad_norms(model):
    """Gradient norms per phase (works for both forward and reverse)."""
    block = model.block
    norms = {}
    phase_modules = [
        ("type", block.type_layer),
        ("parse", block.parse_layer),
        ("apply", block.apply_layer),
    ]
    if block.context_layer is not None:
        phase_modules.append(("context", block.context_layer))

    # Prediction heads
    pred_modules = [block.predict_parse, block.predict_apply]
    if block.predict_context is not None:
        pred_modules.append(block.predict_context)
    if block.predict_type is not None:
        pred_modules.append(block.predict_type)

    for name, module in phase_modules:
        total = sum(
            p.grad.data.norm(2).item() ** 2
            for p in module.parameters() if p.grad is not None
        ) ** 0.5
        norms[name] = total

    norms["predictions"] = sum(
        p.grad.data.norm(2).item() ** 2
        for m in pred_modules for p in m.parameters() if p.grad is not None
    ) ** 0.5

    norms["embeddings"] = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.token_embed.parameters() if p.grad is not None
    ) ** 0.5

    return norms


@torch.no_grad()
def measure_phase_dynamics(model, eval_loader, device, n_batches=5):
    """Full phase instrumentation: norms, cosines, expansion ratios."""
    model.eval()
    block = model.block

    all_metrics = []
    for _ in range(n_batches):
        x_ids, _ = eval_loader.next_batch()
        x_ids = x_ids.to(device)
        positions = torch.arange(x_ids.shape[1], device=device)
        x = model.token_embed(x_ids) + model.pos_embed(positions)

        m = {"embed_norm": x.norm(dim=-1).mean().item()}

        for it in range(model.n_iterations):
            pfx = f"iter{it}"

            # ── Coarse→fine (reverse) ──
            # Context first
            x_ctx = block.context_layer(x)
            ctx_delta = x_ctx - x
            m[f"{pfx}_context_delta_norm"] = ctx_delta.norm(dim=-1).mean().item()
            m[f"{pfx}_after_context"] = x_ctx.norm(dim=-1).mean().item()

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

            # Measure each predicted phase
            for name, delta, predicted, error in [
                ("apply", apply_delta, apply_predicted, apply_error),
                ("parse", parse_delta, parse_predicted, parse_error),
                ("type", type_delta, type_predicted, type_error),
            ]:
                d_n = delta.norm(dim=-1).mean().item()
                e_n = error.norm(dim=-1).mean().item()
                p_n = predicted.norm(dim=-1).mean().item()
                cos = F.cosine_similarity(
                    predicted.reshape(-1, predicted.shape[-1]),
                    delta.reshape(-1, delta.shape[-1]),
                    dim=-1,
                ).mean().item()
                m[f"{pfx}_{name}_delta_norm"] = d_n
                m[f"{pfx}_{name}_error_norm"] = e_n
                m[f"{pfx}_{name}_predicted_norm"] = p_n
                m[f"{pfx}_{name}_cos"] = cos
                m[f"{pfx}_{name}_expansion"] = e_n / d_n if d_n > 0 else 0.0

            # Phase boundary norms
            m[f"{pfx}_after_apply"] = (x_ctx + apply_error).norm(dim=-1).mean().item()
            m[f"{pfx}_after_parse"] = (x_ctx + apply_error + parse_error).norm(dim=-1).mean().item()
            m[f"{pfx}_after_type"] = x_out.norm(dim=-1).mean().item()

            x = x_out

        m["output_norm"] = x_out.norm(dim=-1).mean().item()
        m["overall_expansion"] = m["output_norm"] / m["embed_norm"]
        all_metrics.append(m)

    # Average
    keys = all_metrics[0].keys()
    avg = {k: round(sum(d[k] for d in all_metrics) / len(all_metrics), 6)
           for k in keys}
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
    from verbum.compressor_lm import CompressorLM

    results_dir = Path("results/tesseract-reverse")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path("checkpoints/tesseract-reverse")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    start = time.time()
    banner("TESSERACT REVERSE (coarse→fine) — 10K STEPS")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")

    tokens_total = N_STEPS * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN
    print(f"  Device: {device}")
    print(f"  Strides: {STRIDES} (tesseract)")
    print(f"  Direction: REVERSE (coarse→fine)")
    print(f"  Seq len: {SEQ_LEN}")
    print(f"  Batch: {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Steps: {N_STEPS}")
    print(f"  Tokens: {tokens_total:,}")
    print(f"  Data: SHUFFLED", flush=True)

    # ── Build model ───────────────────────────────────────────────────
    banner("BUILDING MODEL")

    model = CompressorLM(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=SEQ_LEN,
        d_ff=D_FF, window=WINDOW, strides=STRIDES, mode="iterative",
        n_iterations=N_ITERATIONS, reverse=True,
    ).to(device)

    params = model.count_parameters()
    print(model.describe_heads())
    for k, v in params.items():
        print(f"  {k:25s}: {v:>12,}")
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
            tps = step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / elapsed
            print(
                f"  step {step:5d}/{N_STEPS}  "
                f"loss={accum_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
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
            # Gradient norms (from the last backward pass)
            grad_norms = get_grad_norms(model)

            # Phase dynamics (activation norms, cosines, expansion)
            eval_loader.reset()
            phase = measure_phase_dynamics(model, eval_loader, device)

            # Compile gate
            compile = compile_gate_test(model, tokenizer, device)

            # Prediction head weight norms (are they learning?)
            block = model.block
            pred_weights = {
                "predict_apply": block.predict_apply.weight.norm().item(),
                "predict_parse": block.predict_parse.weight.norm().item(),
                "predict_type": block.predict_type.weight.norm().item(),
                "predict_context": block.predict_context.weight.norm().item() if block.predict_context else None,
            }

            ckpt_info = {
                "step": step,
                "train_loss": accum_loss,
                "eval_loss": eval_losses[-1]["loss"] if eval_losses else None,
                "grad_norms": grad_norms,
                "phase_dynamics": phase,
                "prediction_weight_norms": pred_weights,
                "compile_gate": compile["score"],
            }
            checkpoints_data.append(ckpt_info)

            print(f"  ── checkpoint {step} ──")
            print(f"     grad norms: {json.dumps({k: round(v, 4) for k, v in grad_norms.items()})}")
            print(f"     expansion: {phase['overall_expansion']:.2f}x")
            print(f"     iter1 cosines: "
                  f"apply={phase.get('iter1_apply_cos', 0):+.4f}  "
                  f"parse={phase.get('iter1_parse_cos', 0):+.4f}  "
                  f"type={phase.get('iter1_type_cos', 0):+.4f}")
            print(f"     iter1 expansion: "
                  f"apply={phase.get('iter1_apply_expansion', 0):.4f}x  "
                  f"parse={phase.get('iter1_parse_expansion', 0):.4f}x  "
                  f"type={phase.get('iter1_type_expansion', 0):.4f}x")
            print(f"     compile gate: {compile['score']}")
            print(f"     pred weight norms: {json.dumps({k: round(v, 4) if v else None for k, v in pred_weights.items()})}")

            # Save checkpoint
            ckpt_path = checkpoint_dir / f"step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": accum_loss,
                "phase_dynamics": phase,
                "grad_norms": grad_norms,
                "train_losses": train_losses[:],
                "eval_losses": eval_losses[:],
            }, ckpt_path)
            print(f"     saved: {ckpt_path}", flush=True)

            model.train()

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start
    banner(f"DONE — {elapsed:.0f}s ({elapsed / 3600:.1f}h)")

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "architecture": "CompressorLM (tesseract, reverse, shuffled)",
        "direction": "coarse_to_fine",
        "strides": list(STRIDES),
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

    # Print final comparison reference
    print()
    print("  Reference (forward tesseract-shuffled at 10K):")
    print("    Best eval: 5.043 @ step 9500")
    print("    Overall expansion: 21.25x")
    print("    Cosines (iter1): -0.34, -0.42, -0.50")
    print()
    if eval_losses:
        best = min(eval_losses, key=lambda e: e["loss"])
        last_phase = checkpoints_data[-1]["phase_dynamics"] if checkpoints_data else {}
        print(f"  This run (reverse):")
        print(f"    Best eval: {best['loss']:.3f} @ step {best['step']}")
        print(f"    Overall expansion: {last_phase.get('overall_expansion', '?')}x")
        cos_str = ", ".join(
            f"{last_phase.get(f'iter1_{p}_cos', 0):+.4f}"
            for p in ["apply", "parse", "type"]
        )
        print(f"    Cosines (iter1): {cos_str}")


if __name__ == "__main__":
    main()
