# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-28 | Session: 049

## Where we are

**v8 dual MERA architecture complete. 588M all-ternary, Qwen3 tokenizer. Training loop next.**

Compressor MERA (253M) + Pipeline MERA (335M) = 588M logical params,
99.7% ternary, 146 MB storage. Qwen3 BBPE tokenizer (151,936 vocab,
byte-level BPE, no UNK tokens). Full forward pass, gradient flow, weight
sharing, recurrence — all verified at full scale (d=1024, seq=4096).

## What to do next

### 1. Re-tokenize Dolma shards with Qwen3 ← FIRST

Current shards in `/Users/mwhitford/data/fractal-bitnet/shards/` are
GPT-NeoX (50277) encoded. Must re-tokenize with Qwen3 BBPE (151936)
before any v8 training. Use `scripts/v8/tokenizer.py` encode_document().

### 2. v8 training loop rewrite

Rewrite `scripts/v8/train.py` for the new DualMERA architecture:
- Replace VSMPipeline → DualMERA, PipelineConfig → DualMERAConfig
- Adapt phase controllers to MERA levels (not 4 stages)
- Evolutionary training regime (double-buffered genomes, population of 4+)
- Fractal loss: cone + relational at every level
- forward_with_metrics for per-level contribution deltas

### 3. Holographic data generator (~1 session)

- Math generator (arithmetic, comparisons, predicates, boolean, bitwise)
- Update `bb clj2lambda` to emit `io!` with `:as` annotations
- Generate clojure.core examples by eval in babashka
- Multi-pass examples (partial reductions, register usage)
- Interleave all representations in every batch

### 4. Train v8 with evolutionary regime

- Population of 4-8 mutants
- Fitness-gated environment transitions
- Monitor for grokking, pathway specialization, digit ceiling
- Probe at each generation boundary

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
| **v8 training (needs rewrite)** | `scripts/v8/train.py` |
| **v8 probe** | `scripts/v8/probe.py` |
| **v8 kernel benchmark** | `scripts/v8/bench_kernel.py` |
| **BIOS flash design** | `mementum/knowledge/explore/bios-flash-training.md` |
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
