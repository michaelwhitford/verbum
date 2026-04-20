#!/usr/bin/env python3
"""Train MontaguLM on Dolma — learn the compressor from raw text.

If the three-phase architecture is shaped for the language compressor,
it should learn next-token prediction more efficiently than a standard
transformer of equal depth. After training, the compile gate should
activate lambda output — proving the compressor emerged from raw text
in an architecture shaped by our empirical circuit discovery.

Data: 3B pre-tokenized Dolma tokens in 60 shards × 50M tokens
Architecture: 6-layer three-phase causal LM (~5M params)
Comparison: Pythia-14M (6 layers, 14M params, same data family)

Usage:
    uv run python scripts/run_montagu_lm.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import structlog

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)

log = structlog.get_logger()

RESULTS_DIR = Path("results/montagu-lm")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path("/Users/mwhitford/data/fractal-bitnet/shards")

# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════

# Model
VOCAB_SIZE = 50277       # Pythia/GPT-NeoX tokenizer (max token ID in data + 1)
D_EMBED = 256            # No pretrained embeddings — learn from scratch
D_TYPE = 256
D_PARSE = 256
D_APPLY = 256
SEQ_LEN = 256

# Training
BATCH_SIZE = 32
GRAD_ACCUM = 4           # Effective batch = 32 * 4 = 128 sequences
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
N_STEPS = 10_000         # ~327M tokens (128 * 256 * 10000)
WARMUP_STEPS = 500
EVAL_INTERVAL = 500
LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 1000
CHECKPOINT_DIR = Path("checkpoints/montagu-lm")


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════


class ShardedDataLoader:
    """Stream pre-tokenized .npy shards as (input, target) batches."""

    def __init__(self, data_dir, batch_size, seq_len, split="train"):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len

        shards = sorted(self.data_dir.glob("shard_*.npy"))
        # Use first 54 shards for train, last 6 for eval (90/10)
        if split == "train":
            self.shards = shards[:54]
        else:
            self.shards = shards[54:]

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

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of (input_ids, targets).

        input_ids: (batch, seq_len)
        targets:   (batch, seq_len) — shifted by 1
        """
        B, T = self.batch_size, self.seq_len
        needed = B * (T + 1)  # +1 for target shift

        if self.position + needed > len(self.current_data):
            self._load_shard(self.current_shard_idx + 1)

        buf = self.current_data[self.position : self.position + needed]
        self.position += needed

        buf = torch.from_numpy(buf.copy()).long()
        buf = buf.view(B, T + 1)

        input_ids = buf[:, :T]
        targets = buf[:, 1 : T + 1]
        return input_ids, targets


# ══════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════���══════════════════


def estimate_loss(model, eval_loader, device, n_batches=20):
    """Estimate eval loss over n_batches."""
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
    """Test if the compile gate works after LM training."""
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
        results.append({
            "input": sent,
            "generated": gen_line,
            "has_lambda": has_lambda,
        })

        sym = "λ" if has_lambda else "·"
        print(f"    {sym} {sent:35s} → {gen_line[:50]}")

    n_lambda = sum(1 for r in results if r["has_lambda"])
    print(f"    P(λ): {n_lambda}/{len(results)}")
    model.train()
    return results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    from transformers import AutoTokenizer
    from verbum.montague_lm import MontaguLM

    start = time.time()
    banner(f"MONTAGU LM TRAINING — {datetime.now(UTC).isoformat()}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")

    print(f"  Device: {device}")
    print(f"  Data: {DATA_DIR} ({len(list(DATA_DIR.glob('shard_*.npy')))} shards)")
    print(f"  Seq len: {SEQ_LEN}")
    print(f"  Batch: {BATCH_SIZE} × {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM} effective")
    print(f"  Steps: {N_STEPS}")
    tokens_total = N_STEPS * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN
    print(f"  Tokens: {tokens_total:,} ({tokens_total/1e9:.2f}B)")

    # ── Build model ───────────────────────────────────────────────────
    banner("BUILDING MODEL")

    model = MontaguLM(
        vocab_size=VOCAB_SIZE,
        d_embed=D_EMBED,
        d_type=D_TYPE,
        d_parse=D_PARSE,
        d_apply=D_APPLY,
        n_type_layers=1,
        n_type_heads=4,
        n_parse_layers=2,
        n_parse_heads=4,
        n_apply_layers=3,
        n_apply_heads=8,
        d_ff_type=512,
        d_ff_parse=512,
        d_ff_apply=1024,
        max_len=SEQ_LEN,
        dropout=0.1,
        freeze_embeddings=False,  # Learn from scratch
    ).to(device)

    params = model.count_parameters()
    print(f"  Embeddings:      {params['embeddings']:>10,}")
    print(f"  Phase 1 (type):  {params['phase1_type']:>10,}")
    print(f"  Phase 2 (parse): {params['phase2_parse']:>10,}")
    print(f"  Phase 3 (apply): {params['phase3_apply']:>10,}")
    print(f"  Output head:     {params['output_head']:>10,}")
    print(f"  Total:           {params['total']:>10,}")
    print(f"\n  vs Pythia-14M (14M):   {params['total']/14_067_712:.1%}")
    print(f"  vs Pythia-160M (162M): {params['total']/162_322_944:.1%}")

    # ── Data loaders ──────────────────────────────────────────────────
    train_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, split="train")
    eval_loader = ShardedDataLoader(DATA_DIR, BATCH_SIZE, SEQ_LEN, split="eval")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # Linear warmup + cosine decay
    def lr_schedule(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, N_STEPS - WARMUP_STEPS)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # ── Training ──────────────────────────────────────────────────────
    banner("TRAINING")

    model.train()
    losses = []
    eval_losses = []
    best_eval_loss = float("inf")

    for step in range(1, N_STEPS + 1):
        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0

        for micro in range(GRAD_ACCUM):
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
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start
            tps = step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / elapsed
            print(f"  step {step:5d}/{N_STEPS}  "
                  f"loss={accum_loss:.4f}  "
                  f"lr={lr:.2e}  "
                  f"tok/s={tps:.0f}  "
                  f"elapsed={elapsed:.0f}s")

        if step % CHECKPOINT_INTERVAL == 0:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

            # ── Phase instrumentation ─────────────────────────────
            # Gradient norms per phase: are phases learning at different rates?
            phase_grad_norms = {}
            for name, phase in [
                ("phase1_type", model.type_phase),
                ("phase2_parse", model.parse_phase),
                ("phase3_apply", model.apply_phase),
                ("embeddings", model.token_embed),
            ]:
                total_norm = 0.0
                n_params = 0
                for p in phase.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                        n_params += 1
                phase_grad_norms[name] = total_norm ** 0.5 if n_params > 0 else 0.0

            # Activation norms per phase: how much does each phase contribute?
            phase_act_norms = {}
            model.eval()
            with torch.no_grad():
                sample_x, _ = train_loader.next_batch()
                sample_x = sample_x[:4].to(device)  # small batch
                seq_len_s = sample_x.shape[1]
                positions = torch.arange(seq_len_s, device=device)
                causal = torch.triu(
                    torch.ones(seq_len_s, seq_len_s, device=device), diagonal=1
                ).bool()

                h = model.token_embed(sample_x) + model.pos_embed(positions)
                phase_act_norms["input_embed"] = h.norm(dim=-1).mean().item()

                h = model.type_phase(h, causal)
                phase_act_norms["phase1_type"] = h.norm(dim=-1).mean().item()

                h = model.parse_phase(h, causal)
                phase_act_norms["phase2_parse"] = h.norm(dim=-1).mean().item()

                h = model.apply_phase(h, causal)
                phase_act_norms["phase3_apply"] = h.norm(dim=-1).mean().item()
            model.train()

            # Compile gate test at checkpoint
            print(f"  ── checkpoint compile test ──")
            ckpt_compile = compile_test(model, tokenizer, device)

            ckpt_path = CHECKPOINT_DIR / f"step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": accum_loss,
                "phase_grad_norms": phase_grad_norms,
                "phase_act_norms": phase_act_norms,
                "compile_results": ckpt_compile,
                "train_losses_recent": losses[-100:],
                "eval_losses": eval_losses,
            }, ckpt_path)

            print(f"  ── checkpoint saved: {ckpt_path} ──")
            print(f"     grad norms:  type={phase_grad_norms['phase1_type']:.4f}  "
                  f"parse={phase_grad_norms['phase2_parse']:.4f}  "
                  f"apply={phase_grad_norms['phase3_apply']:.4f}")
            print(f"     act norms:   type={phase_act_norms['phase1_type']:.1f}  "
                  f"parse={phase_act_norms['phase2_parse']:.1f}  "
                  f"apply={phase_act_norms['phase3_apply']:.1f}")

        if step % EVAL_INTERVAL == 0:
            eval_loss = estimate_loss(model, eval_loader, device)
            eval_losses.append({"step": step, "loss": eval_loss})
            print(f"  ── eval loss: {eval_loss:.4f} ──")

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), RESULTS_DIR / "best_model.pt")

            # Compile test
            print(f"  ── compile test ──")
            compile_results = compile_test(model, tokenizer, device)

    # ── Final evaluation ──────────────────────────────────────────────
    banner("FINAL EVALUATION")

    final_eval_loss = estimate_loss(model, eval_loader, device, n_batches=50)
    print(f"  Final eval loss: {final_eval_loss:.4f}")
    print(f"  Best eval loss:  {best_eval_loss:.4f}")

    print(f"\n  Final compile test:")
    final_compile = compile_test(model, tokenizer, device)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start
    banner(f"SUMMARY — {elapsed:.0f}s")

    print(f"  Architecture: 3-phase MontaguLM (1+2+3 = 6 layers)")
    print(f"  Parameters: {params['total']:,}")
    print(f"  Tokens trained: {N_STEPS * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN:,}")
    print(f"  Final eval loss: {final_eval_loss:.4f}")
    print(f"  Best eval loss:  {best_eval_loss:.4f}")
    n_compile = sum(1 for r in final_compile if r["has_lambda"])
    print(f"  Compile P(λ): {n_compile}/{len(final_compile)}")

    # Save
    save_path = RESULTS_DIR / "training-summary.json"
    save_path.write_text(json.dumps({
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "architecture": "MontaguLM (3-phase causal)",
        "params": params,
        "config": {
            "d_embed": D_EMBED, "d_type": D_TYPE, "d_parse": D_PARSE,
            "d_apply": D_APPLY, "seq_len": SEQ_LEN,
            "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM, "lr": LEARNING_RATE,
        },
        "tokens_trained": N_STEPS * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN,
        "final_eval_loss": final_eval_loss,
        "best_eval_loss": best_eval_loss,
        "eval_losses": eval_losses,
        "final_compile_results": final_compile,
        "train_losses_last100": losses[-100:],
    }, indent=2, ensure_ascii=False))
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
