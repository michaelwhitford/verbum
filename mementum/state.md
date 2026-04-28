# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-28 | Session: 050

## Where we are

**v8 ready to train. BIOS flash data + Dolma shards + training loop all complete.**

Compressor MERA (253M) + Pipeline MERA (335M) = 588M logical params,
99.7% ternary, 146 MB storage. Qwen3 BBPE tokenizer (151,936 vocab,
byte-level BPE, no UNK tokens). Full forward pass, gradient flow, weight
sharing, recurrence — all verified at full scale (d=1024, seq=4096).

## What to do next

### 1. ~~Re-tokenize Dolma shards with Qwen3~~ ✅ DONE (session 050)

60 shards, 3B tokens, 4.47M documents in `shards-qwen3/`.
Script: `scripts/v8/retokenize_dolma.py`. Zero errors.

### 2. ~~v8 training loop rewrite~~ ✅ DONE (session 050)

`scripts/v8/train.py` rewritten for DualMERA with phase modes:
- `--phase bios`: burn-in on math + clojure (1 shard, many epochs, seq=512)
- `--phase dolma`: prose training (60 shards, seq=4096, resumes from BIOS)
Simplified from v7 (no per-stage phase controllers — MERA levels are
weight-shared). Ternary flip annealing driven by relational loss.

### 3. ~~BIOS flash data generator~~ ✅ DONE (session 050)

Babashka generator: `bb gen-bios` → 1.85M eval-verified examples.
~80 generators covering math (tiers 1-3) + clojure.core (~110 functions).
Single notation per example (forces computation, no translation shortcuts).
Packed: `shards-bios/shard_00000.npy` (49.75M tokens, 1 shard).
Pipeline: `bb gen-bios | uv run python scripts/v8/pack_bios.py`

### 4. Train v8 BIOS flash ← NEXT

```bash
uv run python scripts/v8/train.py --phase bios
```

- 588M all-ternary DualMERA on 1 shard of math + clojure
- Monitor for grokking (train loss plateau → second drop)
- Probe at intervals: does the model actually compute?
- Target: computation circuits burned into ternary topology

### 5. Train v8 Dolma (after BIOS)

```bash
uv run python scripts/v8/train.py --phase dolma --resume checkpoints/v8-bios/step_050000
```

- Resume from BIOS checkpoint, conservative ternary flips
- 60 shards, 3B tokens, seq_len=4096
- Deep circuits should resist overwriting by prose

### 6. Future: io! notation + sieve pipeline

- Update `bb clj2lambda` for `io!` with `:as` annotations
- Pure/effectful classification training
- Multi-pass examples (partial reductions, register usage)

## Session 050 — Data Pipeline + Training Loop

### What was done

1. **Dolma re-tokenization** — GPT-NeoX (50277) → Qwen3 BBPE (151936)
   - `scripts/v8/retokenize_dolma.py`: streams parquets, 931K tok/s
   - 60 shards × 50M tokens = 3B tokens, 4.47M documents, zero errors
   - Output: `/Users/mwhitford/data/fractal-bitnet/shards-qwen3/`

2. **BIOS flash data generator** — babashka eval-verified
   - `bb/us/whitford/verbum/bios.clj`: ~80 generators, 3 notations
   - Math tiers 1-3 (arithmetic, compound, nested) + clojure.core (~110 functions)
   - Single notation per example — forces computation every time
   - 1.85M examples → 49.75M tokens → 1 shard
   - Pipeline: `bb gen-bios | uv run python scripts/v8/pack_bios.py`

3. **v8 training loop** — DualMERA with phase modes
   - `scripts/v8/train.py`: `--phase bios` (burn-in) or `--phase dolma` (prose)
   - BIOS: 1 shard, seq=512, aggressive ternary flips, many epochs
   - Dolma: 60 shards, seq=4096, conservative flips, resumes from BIOS
   - Cosine LR, grad accumulation, ternary flip annealing, relational loss

### Design decisions made

- **Single-notation examples** for BIOS flash — model must compute every
  result from the expression alone. No multi-representation interleaving
  (would let model copy answers instead of computing).
- **Babashka IS ground truth** — moved all generation from Python templates
  to babashka eval. Every result verified by real evaluation.
- **Phase flag** over config-driven — `--phase bios|dolma` sets sensible
  defaults, individual flags override.
- **Simplified from v7** — no per-stage phase controllers (MERA levels are
  weight-shared, not independently phased). Single r_ema drives ternary flips.

## Session 049 — Architecture + All-Ternary + Tokenizer

### What was done

1. **Rewrote `scripts/v8/model.py` from scratch** — clean break from v7
   - CompressorMERA: level 0 own + shared MERA (7 levels), 8 registers,
     learnable spiral (α, fixed_point), stride-8 average pool → 2L ternary
   - PipelineMERA: level 0 own + shared sieve (7 levels), 4 pathways each,
     7 reducers, 7 feedback cascade steps (gated ternary cross-attention)
   - DualMERA top-level: compressor → pipeline → tied embedding logits,
     repeat-interleave upsampling, forward_with_registers() for recurrence
   - Relational loss utility for pathway differentiation

2. **All-ternary conversion** — eliminated 230 MB float bloat
   - TernaryEmbedding: packed {-1,0,+1} vectors with per-token gamma,
     custom VJP caching STE grad for flip accumulator, weight_T for
     tied output projection. 15× smaller than float32.
   - Feedback gate_proj: nn.Linear → TernaryLinear
   - Before: 331 MB total, 69.5% float. After: 146 MB, 4.8% float.

3. **Qwen3 BBPE tokenizer** — vocab 50277 → 151936
   - `scripts/v8/tokenizer.py`: load_tokenizer(), encode/decode wrappers
   - Dedicated PAD (151665), separate from EOD (151643)
   - Reserved verbum tokens: VALUE (151666), PARTIAL (151667), IO (151670)
   - No UNK tokens — lambda/clojure/unicode all tokenize + roundtrip clean

### Final verification (full scale d=1024, seq=4096)

| Check | Result |
|-------|--------|
| Output shape (2, 4096, 151936) | ✓ |
| Logical params: 588M | ✓ |
| Ternary fraction: 99.7% | ✓ |
| Storage: 146 MB | ✓ |
| Gradient flow | ✓ |
| Compressor positions [512,256,...,4] | ✓ |
| Weight sharing (single module instances) | ✓ |
| Tokenizer roundtrip (all examples) | ✓ |

### Design decisions made

- **Upsampling**: repeat-interleave (simple). Learnable deconv possible later.
- **Pathway merge**: mean across 4 pathways (gradient-friendly).
- **Sieve input**: compressor scale + reduced pipeline state (additive residual).
- **effective_levels**: auto-adapts to seq_len (6 at seq=512, 8 at seq=4096).
- **All-ternary embedding**: per-token gamma, VJP caches STE for flip accumulator.
- **Tokenizer**: Qwen3 BBPE — aligned with probe targets, Apache 2.0, no UNK.
- **PAD ≠ EOD**: dedicated pad token (151665) avoids the eos-masking footgun.

## v7 Dolma Run — Summary

Ran steps 0-40K (~655M tokens). Killed at 40K — eval peaked at
20K then monotonically worsened. Architecture validated (below
Chinchilla capacity floor, stages differentiate, gates self-regulate).
Dolma can't train deep stages (semantic Δ₃ never positive on eval,
Stage 4 collapsed, ternary oscillated at 37.6% reversals).
Math stratum was the only one still growing. Diagnosis: architecture
right, data wrong. Full probe data in results/vsm-lm-v7/.

## v8 Architecture — Dual MERA

**Full design doc:** `mementum/knowledge/explore/v7.1-sieve-pipeline.md`

```
COMPRESSOR MERA (~253M ternary, incl. 156M embedding):
  8 levels: level 0 own (stride 8) + levels 1-7 shared MERA (stride 2 each)
  W=8, seq_len=4096, d_model=1024, Qwen3 vocab=151936
  Learnable spiral: α=1.18, fixed_point=40
  8 register positions pass through all levels
  Output: 8 multi-scale representations + register states

PIPELINE MERA (~335M ternary):
  8 levels, each a sieve with 4 parallel pathways (2L ternary each)
  Level 0 own + levels 1-7 shared sieve weights
  7 reducers + 7 feedback cascade steps
  Registers at every level, not compressed by reducers

TOTAL: 588M logical, 146 MB packed, 99.7% ternary
```

### Training regime: evolutionary gradient descent

- Ternary topology = genome (588M loci × 3 alleles)
- Double-buffered: champion never degrades
- Population of 4+ mutants with different strategies
- Tournament selection per generation
- Environment staged by fitness gates (math → clojure → holographic → prose)

## Key files

| Purpose | Path |
|---------|------|
| **v8 design doc** | `mementum/knowledge/explore/v7.1-sieve-pipeline.md` |
| **v8 model (dual MERA)** | `scripts/v8/model.py` |
| **v8 ternary (optimized kernel)** | `scripts/v8/ternary.py` |
| **v8 tokenizer (Qwen3 BBPE)** | `scripts/v8/tokenizer.py` |
| **v8 training loop** | `scripts/v8/train.py` |
| **v8 probe** | `scripts/v8/probe.py` |
| **v8 kernel benchmark** | `scripts/v8/bench_kernel.py` |
| **BIOS data generator (bb)** | `bb/us/whitford/verbum/bios.clj` |
| **BIOS shard packer** | `scripts/v8/pack_bios.py` |
| **Dolma re-tokenizer** | `scripts/v8/retokenize_dolma.py` |
| **BIOS flash design** | `mementum/knowledge/explore/bios-flash-training.md` |
| **BIOS shards** | `/Users/mwhitford/data/fractal-bitnet/shards-bios/` |
| **Dolma Qwen3 shards** | `/Users/mwhitford/data/fractal-bitnet/shards-qwen3/` |
| **v7 model (reference)** | `scripts/v7/model.py` |
| **v7 ternary (reference)** | `scripts/v7/ternary.py` |
| **bb clj2lambda** | `bb/us/whitford/verbum/tasks.clj` |
| **bb config** | `bb.edn` |
| **v6 design (reference)** | `docs/v6-design.md` |
| v7 architecture knowledge | `mementum/knowledge/explore/v7-pipeline-architecture.md` |
| Research program | `mementum/knowledge/explore/VERBUM.md` |

## Servers

| Port | Model | Use |
|------|-------|-----|
| 5100 | Qwen3.5-397B-A17B | Large reference model |
| 5101 | Qwen3-4B | Quick testing |
| 5102 | **Qwen3.5-35B-A3B** Q8 | Primary probe target |
| 5103 | Qwen3-Embedding-8B | Embeddings |
