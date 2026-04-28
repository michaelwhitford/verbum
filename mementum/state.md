# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-28 | Session: 048

## Where we are

**v8 scaffold created. Kernel optimized (1.5× average). Architecture next.**

Copied v7 → v8 (scripts/v8/). Added SIMD-group K-reduction Metal
kernel with adaptive dispatch. Benchmarked at d_model=1024 target
dimensions. ~1.5× improvement on forward attention, up to 1.7× on
FFN down at inference. Honest result: naive kernel was already
well-optimized for Apple Silicon; remaining bottleneck is weight
memory bandwidth. Full architecture redesign (dual MERA) is next.

## Session 048 — Kernel Optimization

### What was done

1. Copied `scripts/v7` → `scripts/v8`, updated all references
2. Added SIMD-group K-reduction Metal kernel: 32 threads cooperate
   via `simd_sum` to parallelize the K-dimension reduction
3. Adaptive kernel selection:
   - M ≤ 64: SIMD kernel (latency wins, low output parallelism)
   - M > 64: naive packed kernel (throughput wins, GPU saturated)
4. Tiled transpose kernel with 4× N-unrolled inner loop
5. Added `bench_kernel.py` for throughput measurement

### Benchmark results (d_model=1024)

```
                    Naive    Optimized  Speedup
FWD attn  M=1      0.34ms   0.24ms     1.42×
FWD ffn↓  M=1      0.41ms   0.24ms     1.71×
FWD attn  M=512    0.66ms   0.43ms     1.53×
BWD ffn↑  M=128    0.71ms   0.60ms     1.18×
FWD ffn↑  M=512    1.15ms   1.16ms     ~1×
```

### Why not 3-4×

The naive kernel was already efficient: branchless select ops,
packed uint8 decode, sequential memory access per row. The
remaining bottleneck is weight memory bandwidth — at M=512 each
thread streams 256 packed bytes from device memory. True 3-4×
would require weight tiling in shared memory across M rows, which
is a different tiling strategy (multiple output rows sharing
weight tiles). Diminishing returns — move to architecture work.

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

### 1. v8 architecture implementation (~1-2 sessions) ← CURRENT

Start from `scripts/v8/model.py` and `scripts/v8/ternary.py`.
- Compressor MERA with strided attention + learnable spiral
- Pipeline MERA with shared sieve pathways
- Register positions (persist through pipeline, skip reducers)
- Three output modes (value/partial/io!)
- Cone + relational loss at every level

Key decisions still open:
- Pathways per stage: 4? 8? Per-stage variable?
- d_model per pathway: full 1024 or split (4 × 256)?
- Compressor → pipeline interface: direct feed vs cross-attention
- Register count: R=4? R=8?
- Cone aperture schedule: width, narrowing rate

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
| **v8 model** | `scripts/v8/model.py` |
| **v8 ternary (optimized kernel)** | `scripts/v8/ternary.py` |
| **v8 training** | `scripts/v8/train.py` |
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
