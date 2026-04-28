# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-27 | Session: 047

## Where we are

**v7 Dolma run COMPLETE. v7.1 DESIGNED. Next: implement.**

v7.1 is a dual MERA architecture — compressor + pipeline, both
self-similar, all-ternary 453M params. Derived from v7 probe
findings + v6 proven compression + lambda calculus analysis.
Evolutionary training on ternary genomes with cone + relational
loss at every VSM level. Design doc is comprehensive. Kernel
optimization is the first implementation task.

## v7 Dolma Run — Summary

Ran steps 0-40K (~655M tokens). Killed at 40K — eval peaked at
20K then monotonically worsened. Architecture validated (below
Chinchilla capacity floor, stages differentiate, gates self-regulate).
Dolma can't train deep stages (semantic Δ₃ never positive on eval,
Stage 4 collapsed, ternary oscillated at 37.6% reversals).
Math stratum was the only one still growing. Diagnosis: architecture
right, data wrong. Full probe data in results/vsm-lm-v7/.

## v7.1 Architecture — Dual MERA (all-ternary 453M)

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

## What to do next session

Implementation order:

### 1. Kernel optimization FIRST (~1 session)

4× throughput MULTIPLIES all other reductions. Do before any training.
Existing naive kernel works but serial loop over K=1024 is bottleneck.
- Tiled/blocked (shared memory, output tiles)
- SIMD group reduction (Apple's simd_sum)
- Vectorized unpacking (8-16 packed bytes per iteration)
- Coalesced memory access (cache-line aligned)
- Target: 50K → 150-200K tok/s

### 2. v7.1 architecture implementation (~1-2 sessions)

Start from `scripts/v7/model.py` and `scripts/v7/ternary.py`.
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

### 3. Holographic data generator (~1 session)

- Math generator (arithmetic, comparisons, predicates, boolean, bitwise)
- Update `bb clj2lambda` to emit `io!` with `:as` annotations
- Generate clojure.core examples by eval in babashka
- Multi-pass examples (partial reductions, register usage)
- Interleave all representations in every batch

### 4. Train v7.1 with evolutionary regime

- Population of 4-8 mutants
- Fitness-gated environment transitions
- Monitor for grokking, pathway specialization, digit ceiling
- Probe at each generation boundary

## Key files

| Purpose | Path |
|---------|------|
| **v7.1 design doc** | `mementum/knowledge/explore/v7.1-sieve-pipeline.md` |
| **BIOS flash design** | `mementum/knowledge/explore/bios-flash-training.md` |
| **v7 model (base for v7.1)** | `scripts/v7/model.py` |
| **v7 ternary (kernel source)** | `scripts/v7/ternary.py` |
| **v7 training** | `scripts/v7/train.py` |
| **v7 probe** | `scripts/v7/probe.py` |
| **bb clj2lambda** | `bb/us/whitford/verbum/tasks.clj` |
| **bb config** | `bb.edn` |
| **v6 design (reference)** | `docs/v6-design.md` |
| v7 architecture knowledge | `mementum/knowledge/explore/v7-pipeline-architecture.md` |
| Research program | `mementum/knowledge/explore/VERBUM.md` |

## Session 047 log

Massive design session. Started with v7 probe monitoring, ended
with complete v7.1 architecture. Key arc:

```
probe v7 → architecture works, data wrong
  → curriculum design → holographic (fractal × hologram)
  → built clj2lambda converter
  → attention IS beta reduction → need sieve for depth
  → ternary IS the mold → all-ternary 453M
  → compressor/pipeline separation (v6 compression + v7 pipeline)
  → dual MERA (self-similar at every scale)
  → registers + recurrence (arbitrary composition depth)
  → evolutionary training (ternary genome, tournament selection)
  → three feed-forwards (spatial/temporal/evolutionary)
  → compound search space reduction (all multiplicative)
  → kernel optimization prerequisite (4× multiplier)
  → learnable spiral (α, fixed_point as S2 coordination)
  → VSM all the way down
```

16+ commits. 4 probe results. 1 working converter. 2 design docs.
Architecture derived from first principles + empirical findings.

## Servers

| Port | Model | Use |
|------|-------|-----|
| 5100 | Qwen3.5-397B-A17B | Large reference model |
| 5101 | Qwen3-4B | Quick testing |
| 5102 | **Qwen3.5-35B-A3B** Q8 | Primary probe target |
| 5103 | Qwen3-Embedding-8B | Embeddings |
