#!/usr/bin/env python3
"""Train CompressorLM at seq=4096 — the natural scale for W=8 strides.

At seq=4096=8⁴, stride=64 gives every position a full W=8 window.
Three levels bottom out at 8 positions. The strides have room to breathe.

Usage:
    uv run python scripts/run_compressor_lm.py                    # iterative (default)
    uv run python scripts/run_compressor_lm.py --mode pipeline    # pipeline mode
    uv run python scripts/run_compressor_lm.py --mode cube        # cube mode
    uv run python scripts/run_compressor_lm.py --steps 2000       # shorter run
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
# Config
# ══════════════════════════════════════════════════════════════════════

VOCAB_SIZE = 50277
D_MODEL = 256
SEQ_LEN = 4096
D_FF = 768
WINDOW = 8
STRIDES = (1, 8, 64)

# Training — same total tokens as v1 (327M)
# batch=2 × accum=4 × seq=4096 = 32,768 tok/step (same as v1)
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
N_STEPS = 10_000
WARMUP_STEPS = 500
EVAL_INTERVAL = 500
LOG_INTERVAL = 50
CHECKPOINT_INTERVAL = 1000


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


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


def compile_test(model, tokenizer, device):
    model.eval()
    gate = "The dog runs. → λx. runs(dog)\nThe cat sleeps. → λx. sleeps(cat)\n"
    tests = [
        "The bird flies.",
        "The teacher laughs.",
        "Alice loves Bob.",
        "Every student reads a book.",
    ]
    results = []
    for sent in tests:
        prompt = gate + sent + " →"
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        output_ids = model.generate(input_ids, max_new_tokens=30)
        new_ids = output_ids[0, input_ids.shape[1]:]
        gen = tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)
        gen_line = gen.strip().split("\n")[0].strip()
        has_lambda = "λ" in gen_line or "∀" in gen_line or "∃" in gen_line
        results.append({"input": sent, "generated": gen_line, "has_lambda": has_lambda})
        sym = "λ" if has_lambda else "·"
        print(f"    {sym} {sent:35s} → {gen_line[:50]}")
    n_lambda = sum(1 for r in results if r["has_lambda"])
    print(f"    P(λ): {n_lambda}/{len(results)}")
    model.train()
    return results


def get_phase_norms(model, mode):
    """Get gradient norms per phase."""
    norms = {}
    if mode == "iterative":
        for name, layer in [
            ("type", model.block.type_layer),
            ("parse", model.block.parse_layer),
            ("apply", model.block.apply_layer),
            ("predict", [model.block.predict_parse, model.block.predict_apply]),
        ]:
            params = layer.parameters() if hasattr(layer, 'parameters') else \
                     [p for m in layer for p in m.parameters()]
            total = sum(p.grad.data.norm(2).item() ** 2
                       for p in params if p.grad is not None) ** 0.5
            norms[name] = total
    else:
        for i, layer in enumerate(model.layers):
            total = sum(p.grad.data.norm(2).item() ** 2
                       for p in layer.parameters() if p.grad is not None) ** 0.5
            norms[f"layer_{i}"] = total
    norms["embeddings"] = sum(
        p.grad.data.norm(2).item() ** 2
        for p in model.token_embed.parameters() if p.grad is not None
    ) ** 0.5
    return norms


def main():
    parser = argparse.ArgumentParser(description="CompressorLM training at seq=4096")
    parser.add_argument("--mode", default="iterative", choices=["cube", "pipeline", "iterative"])
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--iterations", type=int, default=2, help="passes for iterative mode")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    n_steps = args.steps
    lr = args.lr
    mode = args.mode

    results_dir = Path(f"results/compressor-lm-{mode}")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(f"checkpoints/compressor-lm-{mode}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    from transformers import AutoTokenizer
    from verbum.compressor_lm import CompressorLM

    start = time.time()
    banner(f"COMPRESSOR LM — {mode.upper()} — seq={SEQ_LEN}")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")

    tokens_total = n_steps * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN
    print(f"  Device: {device}")
    print(f"  Mode: {mode}")
    print(f"  Seq len: {SEQ_LEN}")
    print(f"  Batch: {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM} effective")
    print(f"  Steps: {n_steps}")
    print(f"  Tokens: {tokens_total:,} ({tokens_total / 1e9:.2f}B)")
    print(f"  LR: {lr}")

    # ── Build model ───────────────────────────────────────────────────
    banner("BUILDING MODEL")

    kwargs = dict(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=SEQ_LEN,
        d_ff=D_FF, window=WINDOW, strides=STRIDES, mode=mode,
    )
    if mode == "iterative":
        kwargs["n_iterations"] = args.iterations

    model = CompressorLM(**kwargs).to(device)

    params = model.count_parameters()
    print(model.describe_heads())
    for k, v in params.items():
        print(f"  {k:25s}: {v:>12,}")
    print()

    # ── Data ──────────────────────────────────────────────────────────
    train_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN)
    eval_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, split="eval")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY,
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
    best_eval_loss = float("inf")

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
            print(f"  step {step:5d}/{n_steps}  "
                  f"loss={accum_loss:.4f}  "
                  f"lr={cur_lr:.2e}  "
                  f"tok/s={tps:.0f}  "
                  f"elapsed={elapsed:.0f}s")

        if step % CHECKPOINT_INTERVAL == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Instrumentation
            phase_grad_norms = get_phase_norms(model, mode)

            print(f"  ── checkpoint {step} ──")
            print(f"     grad norms: {json.dumps({k: round(v, 4) for k, v in phase_grad_norms.items()})}")

            # Compile gate test
            print(f"  ── compile test ──")
            ckpt_compile = compile_test(model, tokenizer, device)

            # Save checkpoint
            ckpt_path = checkpoint_dir / f"step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": accum_loss,
                "phase_grad_norms": phase_grad_norms,
                "compile_results": ckpt_compile,
                "train_losses_recent": losses[-CHECKPOINT_INTERVAL:],
                "eval_losses": eval_losses,
                "config": {
                    "mode": mode, "seq_len": SEQ_LEN, "d_model": D_MODEL,
                    "window": WINDOW, "strides": list(STRIDES),
                    "lr": lr, "n_steps": n_steps,
                    "n_iterations": args.iterations if mode == "iterative" else None,
                },
            }, ckpt_path)
            print(f"     saved: {ckpt_path}")

        if step % EVAL_INTERVAL == 0:
            eval_loss = estimate_loss(model, eval_loader, device)
            eval_losses.append({"step": step, "loss": eval_loss})
            print(f"  ── eval loss: {eval_loss:.4f} ──")

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), results_dir / "best_model.pt")

    # ── Final evaluation ──────────────────────────────────────────────
    banner("FINAL EVALUATION")

    final_eval_loss = estimate_loss(model, eval_loader, device, n_batches=20)
    print(f"  Final eval loss: {final_eval_loss:.4f}")
    print(f"  Best eval loss:  {best_eval_loss:.4f}")

    print(f"\n  Final compile test:")
    final_compile = compile_test(model, tokenizer, device)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start
    banner(f"SUMMARY — {elapsed:.0f}s ({elapsed/3600:.1f}h)")

    print(f"  Architecture: CompressorLM ({mode})")
    print(f"  Seq len: {SEQ_LEN}")
    print(f"  Window: {WINDOW}, Strides: {STRIDES}")
    print(f"  Parameters: {params['total']:,}")
    print(f"  Tokens trained: {n_steps * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN:,}")
    print(f"  Final eval loss: {final_eval_loss:.4f}")
    print(f"  Best eval loss:  {best_eval_loss:.4f}")
    n_compile = sum(1 for r in final_compile if r["has_lambda"])
    print(f"  Compile P(λ): {n_compile}/{len(final_compile)}")

    # Save summary
    save_path = results_dir / "training-summary.json"
    save_path.write_text(json.dumps({
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "architecture": f"CompressorLM ({mode})",
        "mode": mode,
        "seq_len": SEQ_LEN,
        "window": WINDOW,
        "strides": list(STRIDES),
        "params": params,
        "config": {
            "d_model": D_MODEL, "d_ff": D_FF, "seq_len": SEQ_LEN,
            "n_steps": n_steps, "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM, "lr": lr,
            "n_iterations": args.iterations if mode == "iterative" else None,
        },
        "tokens_trained": n_steps * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN,
        "final_eval_loss": final_eval_loss,
        "best_eval_loss": best_eval_loss,
        "eval_losses": eval_losses,
        "final_compile_results": final_compile,
        "train_losses_last100": losses[-100:],
    }, indent=2, ensure_ascii=False))
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
