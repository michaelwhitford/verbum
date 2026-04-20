#!/usr/bin/env python3
"""Train the three-phase MontaguCompiler on compilation pairs.

The architecture is shaped by our empirical circuit discovery:
  Phase 1: Type embedding (from Pythia-160M, frozen)
  Phase 2: Structure parser (2-layer self-attention encoder)
  Phase 3: Typed application (3-layer cross-attention decoder)

The cross-attention decoder solves the content mapping problem:
it can look back at the encoder (input sentence) and copy/transform
tokens — exactly what decoder-only Pythia-160M could not do.

Usage:
    uv run python scripts/run_montagu_training.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import structlog

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)

log = structlog.get_logger()

RESULTS_DIR = Path("results/montagu-compiler")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════


class CompilationDataset(Dataset):
    """(English sentence, lambda expression) pairs for seq2seq training."""

    def __init__(self, jsonl_path, input_tokenizer, lambda_tokenizer, max_src=64, max_tgt=96):
        self.examples = []
        self.input_tokenizer = input_tokenizer
        self.lambda_tokenizer = lambda_tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                if r["output"]:
                    self.examples.append(r)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        r = self.examples[idx]

        # Encode input (English sentence) with Pythia tokenizer
        src_enc = self.input_tokenizer(
            r["input"],
            truncation=True,
            max_length=self.max_src,
            return_tensors="pt",
        )
        src_ids = src_enc["input_ids"].squeeze(0)
        src_mask = src_enc["attention_mask"].squeeze(0)

        # Encode target (lambda expression) with lambda tokenizer
        tgt_ids = self.lambda_tokenizer.encode(r["output"])
        tgt_ids = tgt_ids[:self.max_tgt]
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)

        return {
            "src_ids": src_ids,
            "src_mask": src_mask,
            "tgt_ids": tgt_tensor,
        }


def collate_fn(batch, pad_id=0):
    """Pad sequences to max length in batch."""
    max_src = max(b["src_ids"].size(0) for b in batch)
    max_tgt = max(b["tgt_ids"].size(0) for b in batch)

    src_ids = []
    src_masks = []
    tgt_ids = []

    for b in batch:
        src_pad = max_src - b["src_ids"].size(0)
        tgt_pad = max_tgt - b["tgt_ids"].size(0)

        src_ids.append(F.pad(b["src_ids"], (0, src_pad), value=0))
        src_masks.append(F.pad(b["src_mask"], (0, src_pad), value=0))
        tgt_ids.append(F.pad(b["tgt_ids"], (0, tgt_pad), value=pad_id))

    return {
        "src_ids": torch.stack(src_ids),
        "src_mask": torch.stack(src_masks),
        "tgt_ids": torch.stack(tgt_ids),
    }


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════


def train_epoch(model, loader, optimizer, scheduler, device, lambda_tokenizer):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        src_ids = batch["src_ids"].to(device)
        src_mask = batch["src_mask"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)

        # Teacher forcing: input is tgt[:-1], target is tgt[1:]
        tgt_input = tgt_ids[:, :-1]
        tgt_target = tgt_ids[:, 1:]

        # Padding mask for encoder: 0 = pad, needs to be inverted for
        # PyTorch attention (True = ignore)
        enc_padding_mask = (src_mask == 0)

        logits = model(src_ids, tgt_input, input_padding_mask=enc_padding_mask)

        # Flatten for cross-entropy
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_target.reshape(-1),
            ignore_index=lambda_tokenizer.pad_id,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, input_tokenizer, lambda_tokenizer, eval_path, device):
    """Generate lambda for each eval sentence and measure quality."""
    from test_montague_grammar import validate as validate_montague

    model.eval()
    results = []

    with open(eval_path) as f:
        eval_data = [json.loads(line) for line in f]

    for r in eval_data:
        src_enc = input_tokenizer(r["input"], return_tensors="pt").to(device)
        src_ids = src_enc["input_ids"]
        src_mask = src_enc["attention_mask"]
        enc_padding_mask = (src_mask == 0)

        generated = model.generate(
            src_ids, lambda_tokenizer,
            max_len=80,
            input_padding_mask=enc_padding_mask,
        )
        gen_text = generated[0]

        # Metrics
        ok, _ = validate_montague(gen_text)
        has_lambda = "λ" in gen_text or "∀" in gen_text or "∃" in gen_text

        # Content accuracy: do predicates from input appear in output?
        input_words = set(w.lower().rstrip(".,") for w in r["input"].split())
        content = input_words - {"the", "a", "an", "if", "no", "every", "some",
                                  "and", "or", "is", "does", "not", "that", "who"}
        gen_lower = gen_text.lower()
        found = [w for w in content if w in gen_lower]

        results.append({
            "input": r["input"],
            "expected": r["output"],
            "generated": gen_text,
            "parses": ok,
            "has_lambda": has_lambda,
            "content_found": found,
            "content_total": list(content),
            "category": r.get("category", ""),
        })

        parse_sym = "P" if ok else "·"
        content_pct = f"{len(found)}/{len(content)}" if content else "—"
        print(f"  {parse_sym} {r['input'][:35]:35s} → {gen_text[:45]:45s} [{content_pct}]")

    # Aggregate
    n = len(results)
    n_lambda = sum(1 for r in results if r["has_lambda"])
    n_parse = sum(1 for r in results if r["parses"])
    n_content = sum(
        len(r["content_found"]) / max(len(r["content_total"]), 1)
        for r in results
    ) / max(n, 1)

    print(f"\n  P(lambda): {n_lambda}/{n} ({n_lambda/n:.0%})")
    print(f"  Parse:     {n_parse}/{n} ({n_parse/n:.0%})")
    print(f"  Content:   {n_content:.0%} avg")

    return results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    from transformers import AutoTokenizer
    from verbum.montague_net import LambdaTokenizer, MontaguCompiler

    start = time.time()
    banner(f"MONTAGU COMPILER TRAINING — {datetime.now(UTC).isoformat()}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_path = "data/compile-train.jsonl"
    eval_path = "data/compile-eval.jsonl"
    test_path = "data/compile-test.jsonl"

    # ── Build lambda tokenizer from training data ─────────────────────
    banner("BUILDING TOKENIZERS")

    lambda_exprs = []
    with open(train_path) as f:
        for line in f:
            r = json.loads(line)
            if r["output"]:
                lambda_exprs.append(r["output"])

    lambda_tok = LambdaTokenizer.from_training_data(lambda_exprs)
    print(f"  Lambda vocab: {lambda_tok.vocab_size} tokens")

    # Test encoding
    test_expr = "∀x. dog(x) → runs(x)"
    encoded = lambda_tok.encode(test_expr)
    decoded = lambda_tok.decode(encoded)
    print(f"  Test: {test_expr!r} → {encoded} → {decoded!r}")

    # Pythia tokenizer for input
    input_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")
    print(f"  Input vocab: {input_tok.vocab_size}")

    # ── Load pretrained embeddings ────────────────────────────────────
    banner("LOADING PRETRAINED EMBEDDINGS (Pythia-160M)")

    from transformers import AutoModel
    pythia = AutoModel.from_pretrained(
        "EleutherAI/pythia-160m-deduped",
        torch_dtype=torch.float32,
    )
    pretrained_embeds = pythia.embed_in.weight.data.clone()
    d_input = pretrained_embeds.shape[1]
    actual_vocab_size = pretrained_embeds.shape[0]
    print(f"  Embedding dim: {d_input}")
    print(f"  Embedding shape: {pretrained_embeds.shape}")
    print(f"  Actual vocab size: {actual_vocab_size} (tokenizer reports {input_tok.vocab_size})")
    del pythia  # Free memory

    # ── Build model ───────────────────────────────────────────────────
    banner("BUILDING MODEL")

    d_model = 256
    model = MontaguCompiler(
        input_vocab_size=actual_vocab_size,
        output_vocab_size=lambda_tok.vocab_size,
        d_input=d_input,
        d_model=d_model,
        n_parser_layers=2,
        n_parser_heads=4,
        n_apply_layers=3,
        n_apply_heads=4,
        d_ff=512,
        dropout=0.1,
        pretrained_embeddings=pretrained_embeds,
        freeze_embeddings=True,
    ).to(device)

    params = model.count_parameters()
    print(f"  Phase 1 (type embed):    {params['phase1_type_embed']:>10,} params")
    print(f"  Phase 2 (parser):        {params['phase2_parser']:>10,} params")
    print(f"  Phase 3 (decoder):       {params['phase3_decoder']:>10,} params")
    print(f"  Total trainable:         {params['total_trainable']:>10,} params")

    # Compare to Pythia-160M
    pythia_params = 162_322_944
    ratio = params["total_trainable"] / pythia_params
    print(f"\n  vs Pythia-160M ({pythia_params:,}): {ratio:.1%} of parameters")

    # ── Dataset ───────────────────────────────────────────────────────
    banner("LOADING DATA")

    dataset = CompilationDataset(train_path, input_tok, lambda_tok)
    print(f"  Training examples: {len(dataset)}")

    loader = DataLoader(
        dataset, batch_size=8, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=lambda_tok.pad_id),
    )

    # ── Training ──────────────────────────────────────────────────────
    n_epochs = 30
    banner(f"TRAINING ({n_epochs} epochs)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.01
    )
    total_steps = n_epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    epoch_losses = []
    for epoch in range(n_epochs):
        loss = train_epoch(model, loader, optimizer, scheduler, device, lambda_tok)
        epoch_losses.append(loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{n_epochs}: loss={loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    # ── Evaluate ──────────────────────────────────────────────────────
    banner("EVAL — gold standard (10 examples)")
    eval_results = evaluate(model, input_tok, lambda_tok, eval_path, device)

    holdout_results = None
    if Path(test_path).exists() and Path(test_path).stat().st_size > 0:
        banner("HOLDOUT — novel predicates (40 examples)")
        holdout_results = evaluate(model, input_tok, lambda_tok, test_path, device)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start
    banner(f"SUMMARY — {elapsed:.0f}s")

    n_eval = len(eval_results)
    n_lambda = sum(1 for r in eval_results if r["has_lambda"])
    n_parse = sum(1 for r in eval_results if r["parses"])
    avg_content = sum(
        len(r["content_found"]) / max(len(r["content_total"]), 1)
        for r in eval_results
    ) / max(n_eval, 1)

    print(f"  Architecture: 3-phase MontaguCompiler")
    print(f"  Trainable params: {params['total_trainable']:,}")
    print(f"  Training: {len(dataset)} examples, {n_epochs} epochs")
    print(f"  Final loss: {epoch_losses[-1]:.4f}")
    print(f"  Eval P(lambda): {n_lambda}/{n_eval} ({n_lambda/n_eval:.0%})")
    print(f"  Eval parse:     {n_parse}/{n_eval} ({n_parse/n_eval:.0%})")
    print(f"  Eval content:   {avg_content:.0%}")

    if holdout_results:
        n_h = len(holdout_results)
        h_content = sum(
            len(r["content_found"]) / max(len(r["content_total"]), 1)
            for r in holdout_results
        ) / max(n_h, 1)
        print(f"  Holdout content: {h_content:.0%}  ← THE KEY METRIC")

    # Compare to Pythia fine-tuning
    print(f"\n  ── Comparison to Pythia-160M fine-tuning ──")
    print(f"  {'':30s} {'Pythia-FT':>12s}  {'Montagu':>12s}")
    print(f"  {'Trainable params':30s} {'162M':>12s}  {params['total_trainable']:>12,}")
    print(f"  {'Eval P(lambda)':30s} {'90%':>12s}  {n_lambda/n_eval:>12.0%}")
    print(f"  {'Eval parse':30s} {'90%':>12s}  {n_parse/n_eval:>12.0%}")
    print(f"  {'Holdout content':30s} {'~0%':>12s}  {h_content if holdout_results else 0:>12.0%}")

    # Save
    save_path = RESULTS_DIR / "training-summary.json"
    save_path.write_text(json.dumps({
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "architecture": "MontaguCompiler (3-phase)",
        "params": params,
        "d_model": d_model,
        "n_parser_layers": 2,
        "n_apply_layers": 3,
        "n_epochs": n_epochs,
        "epoch_losses": epoch_losses,
        "lambda_vocab_size": lambda_tok.vocab_size,
        "eval_results": eval_results,
        "holdout_results": holdout_results,
    }, indent=2, ensure_ascii=False))
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
