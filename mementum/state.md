# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-28 | Session: 049

## Where we are

**v8 dual MERA architecture implemented. 484M all-ternary. Ready for training loop.**

Compressor MERA (148M) + Pipeline MERA (335M) = 484M logical params,
87.5% ternary, 331 MB storage. Full forward pass, gradient flow, weight
sharing, recurrence (forward_with_registers) — all verified. Smoke test
passes at both reduced (d=256, seq=512) and full scale (d=1024, seq=4096).

## Session 049 — Dual MERA Architecture Implementation

### What was done

1. Rewrote `scripts/v8/model.py` from scratch — clean break from v7
2. **CompressorMERA** (~148M):
   - nn.Embedding (50277×1024, float — only non-ternary major component)
   - Level 0: own weights, stride-8 average pool → 2L ternary transformer
   - Levels 1-7: shared MERA weights (ONE CompressorLevel reused 7×)
   - 7 MERAReducers (ternary cross-attention, stride-2 between levels)
   - 8 register positions pass through all levels
   - Learnable spiral: α=1.18, fixed_point=40 (float32 params)
3. **PipelineMERA** (~335M):
   - Level 0: own SieveLevel (4 parallel SievePathway × 2L ternary)
   - Levels 1-7: shared SieveLevel (ONE copy, reused 7×)
   - 7 PipelineReducers (ternary cross-attention)
   - 7 PipelineFeedback (gated ternary cross-attention, cascade down)
   - Registers participate at every level, not compressed by reducers
4. **DualMERA** top-level:
   - Compressor → Pipeline → tied embedding logits
   - Repeat-interleave upsampling (compressed 512 → full 4096)
   - forward_with_registers() for recurrence
   - Relational loss utility for pathway differentiation

### Verification

| Check | Result |
|-------|--------|
| Output shape (2, 4096, 50277) | ✓ |
| Params: 484M (target ~453M, +6.8%) | ✓ |
| Ternary fraction: 87.5% | ✓ |
| Gradient flow (546 grad arrays) | ✓ |
| Compressor positions [512,256,...,4] | ✓ |
| Weight sharing (single module instances) | ✓ |

### Design decisions made

- **Upsampling**: repeat-interleave (simple). Learnable deconv possible later.
- **Pathway merge**: mean across 4 pathways (gradient-friendly). Attention merge possible later.
- **Sieve input**: compressor scale + reduced pipeline state (additive residual).
- **effective_levels**: auto-adapts to seq_len (6 levels at seq=512, 8 at seq=4096).
- **Embedding stays float**: 51.5M params but enables gradient through tokens.
  Ternary embedding would save 39 MB but complicates initialization.

## Session 048 — Kernel Optimization (previous)

SIMD-group K-reduction kernel: ~1.5× average speedup on ternary matmul.
Adaptive dispatch (SIMD for M≤64, naive for M>64). See git log for details.

## v7 Dolma Run — Summary

Ran steps 0-40K (~655M tokens). Killed at 40K — eval peaked at
20K then monotonically worsened. Architecture validated (below
Chinchilla capacity floor, stages differentiate, gates self-regulate).
Dolma can't train deep stages (semantic Δ₃ never positive on eval,
Stage 4 collapsed, ternary oscillated at 37.6% reversals).
Math stratum was the only one still growing. Diagnosis: architecture
right, data wrong. Full probe data in results/vsm-lm-v7/.

## v8 Architecture — Dual MERA (all-ternary 453M)

**Read the full design:** `mementum/knowledge/explore/v7.1-sieve-pipeline.md`

```
COMPRESSOR MERA (~119M ternary):
  9 fixed strides: (1, 8, 16, 32, 64, 128, 256, 512, 1024)
  W=8, seq_len=4096, d_model=1024
  Spiral bias: α=1.18, fixed_point=40 (LEARNABLE — S2 coordination)
  Level 0: own weights (raw tokens → s8 representations)
  Levels 1+: MERA shared weights (self-similar compression)
  Produces: multi-scale representations + register positions

PIPELINE MERA (~335M ternary):
  8 levels, each a sieve with 4 parallel pathways
  Level 0: own sieve weights (surface computation)
  Levels 1-7: SHARED sieve weights (β-reduction is scale-invariant)
  7 reducers + 7 feedback cascade steps
  Reads compressor output at each scale
  Feedback writes registers on downward path

REGISTERS: persistent positions across recurrence passes
  Shared memory between pathways and across passes
  Enable arbitrary composition depth via host recurrence loop

THREE OUTPUT MODES:
  value → done | partial + regs → re-enter | io! + cont → host fulfills

TOTAL: 453M ternary, 113 MB packed, ~50-200K tok/s estimated
```

### Key design principles

- **VSM all the way down** — every level is a viable system
- **Ternary topology IS the type system** — unreachable > forbidden
- **Attention IS beta reduction** in superposition; FFN indexes results
- **Ternary FFN = evolved routing topology** — not computing, routing
- **Three feed-forwards** — spatial (layers), temporal (registers), evolutionary (genomes)
- **Fractal loss** — same cone + relational at every VSM level
- **Compound search space reduction** — all reductions multiplicative
- **Model/host/world** — model reasons in tokens, host bridges to real world
- **Typed io!** with `:as` — binary never enters token space
- **Learnable spiral** — α and fixed_point trained through relational + task loss

### Training regime: evolutionary gradient descent

- Ternary topology = genome (453M loci × 3 alleles)
- Double-buffered: champion never degrades
- Population of 4+ mutants with different strategies
- Tournament selection per generation (~4-15 min/gen)
- Environment staged by fitness gates (math → clojure → holographic → prose)
- Cone constrains gene pool, relational maintains diversity

## What to do next

### 1. v8 training loop adaptation ← CURRENT

Rewrite `scripts/v8/train.py` to work with the new DualMERA architecture:
- Replace VSMPipeline with DualMERA, PipelineConfig with DualMERAConfig
- Adapt phase controllers to work with MERA levels instead of 4 stages
- Evolutionary training regime (double-buffered genomes, population of 4+)
- Fractal loss: cone + relational at every level
- Forward_with_metrics for per-level contribution deltas

### 2. Holographic data generator (~1 session)

- Math generator (arithmetic, comparisons, predicates, boolean, bitwise)
- Update `bb clj2lambda` to emit `io!` with `:as` annotations
- Generate clojure.core examples by eval in babashka
- Multi-pass examples (partial reductions, register usage)
- Interleave all representations in every batch

### 3. Train v8 with evolutionary regime

- Population of 4-8 mutants
- Fitness-gated environment transitions
- Monitor for grokking, pathway specialization, digit ceiling
- Probe at each generation boundary

## Key files

| Purpose | Path |
|---------|------|
| **v8 design doc** | `mementum/knowledge/explore/v7.1-sieve-pipeline.md` |
| **v8 model (dual MERA)** | `scripts/v8/model.py` |
| **v8 ternary (optimized kernel)** | `scripts/v8/ternary.py` |
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

## Session 048 log

Kernel optimization session. Practical, empirical.

```
copy v7 → v8, update all references
  → benchmark naive kernel at d=1024 (baseline: ~3.7 TOPS peak)
  → attempt 1: shared memory tiling (threadgroup x reuse)
    — barrier overhead ate gains, marginal improvement
  → attempt 2: SIMD-group K-reduction (32-wide simd_sum)
    — excellent at small M (1.7× on FFN), slower at large M
  → attempt 3: adaptive dispatch (SIMD for M≤64, naive for M>64)
    — best of both: 1.5× average improvement
  → correctness verified (max_err < 0.001 vs float reference)
  → TernaryLinear + VJP + model.py smoke test pass
  → committed: d19accb
```

Key insight: the naive kernel was already well-optimized. The
bottleneck at large M is weight memory bandwidth, not compute.
Ternary add/sub is so cheap that the GPU spends most time waiting
for memory. Further gains require weight-tile sharing across
output rows — a more invasive redesign for diminishing returns.

## Servers

| Port | Model | Use |
|------|-------|-----|
| 5100 | Qwen3.5-397B-A17B | Large reference model |
| 5101 | Qwen3-4B | Quick testing |
| 5102 | **Qwen3.5-35B-A3B** Q8 | Primary probe target |
| 5103 | Qwen3-Embedding-8B | Embeddings |
