# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-28 | Session: 052

## Where we are

**v8 evolution redesigned. Ready to re-launch BIOS training.**

DualMERA (559M, 99.7% ternary, d=1024) with Qwen3 BBPE tokenizer.
Training loop uses **redesigned evolutionary mutation** with phase-aware
budget, depth-weighted allocation, probe-aware fitness, sign flips,
and adaptive mutation rate. MLX quantized_matmul on Apple AMX.

**Problem identified in session 052:** Original cone-based evolution was
starved — gamma (continuous, Adam) learned surface statistics in ~1K steps,
driving loss down → r_ema down → cone narrow → topology frozen at 0.009%
mutation rate. 82% acceptance proved the topology was nowhere near optimal.
Probe accuracy was 0% — no circuits formed.

## What to do next

### 1. ~~Smoke-test v8 BIOS training~~ ✅ DONE (session 051)

Model init, data loading, forward/backward all verified clean.

### 2. ~~Evolutionary topology mutation~~ ✅ REDESIGNED (session 052)

Original (session 051):
- `mutation_cone(r_ema)` → loss-gated budget (**starved topology**)
- Budget: 50K mutations/gen (0.009% of topology)
- Visited 7% of weights total over 50K training steps

Redesigned (session 052):
- `bios_mutation_budget()` → constant 0.5% for 80%, decay in final 20%
- Budget: 2.8M mutations/gen (56× increase)
- Visits every weight ~5× over training
- Depth-weighted: pipeline.shared 2×, embedding 0.1×
- Sign flips: 20% of non-zero mutations flip sign directly
- Probe-aware fitness: loss - circuit_bonus × probe_accuracy
- Two-pass tournament: loss-only selection, then probe champion + winner
- Adaptive rate: tracks strategy wins, auto-tunes base_pct

### 3. ~~MLX quantized_matmul~~ ✅ DONE (session 051)

Replaced custom Metal ternary kernels with `mx.quantized_matmul(bits=2)`:
- Custom Metal shaders → Apple AMX hardware path
- 2.3-3.7x faster per matmul, 1.7x end-to-end
- No custom VJP needed — MLX autograd handles everything natively
- `stop_gradient(weight)` prevents invalid grad through uint32
- TernaryEmbedding unchanged (gather, not matmul)

### 4. ~~Computation probe~~ ✅ DONE (session 051)

`scripts/v8/compute_probe.py` — grokking detector:
- Generates fresh examples (never in training data) at 3 tiers
- Greedy-decodes model output, checks exact match vs ground truth
- Integrated into train.py at eval_interval
- Accuracy 0% → >0% = circuit formation signal

### 5. Train v8 BIOS flash ← NEXT

```bash
uv run python scripts/v8/train.py --phase bios
```

- 559M all-ternary DualMERA on 1 shard (49.75M tokens, ~16 epochs)
- 50K steps, ~27 hours
- **Gradient-informed** mutations: |∂L/∂γ| guides row selection, mean(|x|) guides columns
- Budget: 2.8M mutations/gen, constant for 40K steps, then linear decay
- Depth-weighted: pipeline.shared 2×, embedding 0.1×
- Teacher-forced probe fitness: loss - 0.5 × probe_accuracy (137ms per probe)
- Adaptive rate: auto-tunes base_pct from strategy win history
- Checkpoints every 2500 steps with importance maps + evolution diagnostics
- Monitor: probe accuracy 0% → >0% = circuit formation

### 6. Train v8 Dolma (after BIOS)

```bash
uv run python scripts/v8/train.py --phase dolma --resume checkpoints/v8-bios/step_050000
```

- Resume from BIOS checkpoint, narrow cone (protect BIOS circuits)
- 60 shards, 3B tokens, seq_len=4096
- Deep circuits should resist overwriting by prose

### 7. Future: io! notation + sieve pipeline

- Update `bb clj2lambda` for `io!` with `:as` annotations
- Pure/effectful classification training
- Multi-pass examples (partial reductions, register usage)

## Session 052 — Evolutionary Mutation Redesign

### Problem diagnosed

Ran BIOS training for ~1100 steps with original evolution system. Data:
- r_ema dropped to 0.18 in 1000 steps (gamma learned surface statistics)
- Mutation budget: 50K per gen (0.009% of 559M topology)
- Accept rate: 82% — topology far from optimal but barely exploring
- Explorer (4× budget) winning — model screaming for more mutations
- Probe accuracy: 0% — NO circuits formed despite loss dropping to 3.56
- Diagnosis: gamma (Adam, every step) outcompetes topology (mutation, every 50 steps)
- The cone punishes topology when gamma makes loss drop → vicious cycle

### What was done

1. **Phase-aware budget** — BIOS uses constant high budget (0.5% per gen),
   not loss-gated cone. 56× more mutations (2.8M vs 50K per gen).
   Visits every weight ~5× over training vs 7% previously.

2. **Depth-weighted allocation** — pipeline.shared gets 2× mutations,
   embedding gets 0.1×. Circuits need to form in pipeline, not embedding.

3. **Sign flips** — 20% of non-zero mutations flip sign directly
   (-1→+1) instead of always deactivating through zero.

4. **Teacher-forced probe** — replaces autoregressive decode in tournament.
   Feeds prompt+answer, checks logits at answer positions. Single batched
   forward pass: 137ms vs 9,500ms (46× faster). Same circuit signal.

5. **Two-pass tournament** — pass 1: loss-only selection across 4 mutants
   (fast batched eval). Pass 2: probe champion + winner only for circuit
   fitness. Total tournament: 6.5s (was 36.5s with autoregressive probe).

6. **Gradient-informed mutations** — two tiers of signal, zero extra cost:
   - Tier 1: |∂L/∂γ| per row → which output channels have suboptimal
     topology (gamma compensating). 281,000× dynamic range. Extracted
     from existing gamma gradients before zero_ternary_grads().
   - Tier 2: mean(|x|) per column → which input features carry signal.
     Cached in TernaryLinear via stop_gradient (no backward cost).
   - Sampling: 70% importance-weighted (row × col), 30% uniform exploration.
   - Direction: sign(∂L/∂γ) biases 0→±1 mutations (80% follow gradient).

7. **Adaptive mutation rate** — tracks strategy win history (20-gen window).
   Explorer winning >50% → increase base_pct. Conservative >50% → decrease.

8. **Rich checkpoints** — importance.npz (3.6MB), evolution_diagnostics.json
   (per-module ternary stats, hottest modules, global sparsity).
   Importance maps restore on resume for immediate guided mutations.

9. **Enhanced standalone probe** — compute_probe.py now reports ternary
   topology stats and evolution diagnostics when run on a checkpoint.

### Performance journey (session 052)

| Version | Tournament | 50K steps | Mutations/gen |
|---|---|---|---|
| Original (cone, autoregressive) | 7.2s | 25.2h | 50K |
| + Phase-aware + all-mutant probe | 36.5s | 50h+ | 2.8M |
| + Two-pass (probe champ+winner) | 18.5s | 32.4h | 2.8M |
| + Teacher-forced probe | 7.4s | 25.8h | 2.8M |
| + Gradient-informed sampling | 8.3s | ~27h | 2.8M (targeted) |

### Design decisions

- **Constant budget > cone for BIOS** — the cone was designed for
  annealing, but BIOS is about topology discovery, not convergence.
  Topology should explore while gamma handles surface statistics.
- **Teacher-forcing over autoregressive** — probe was 78% of tournament
  time. Batch=1 sequential decode wastes GPU. Teacher-forced checks the
  same thing (does model predict the answer?) in one batched pass.
- **Gradient as compass, tournament as judge** — gradients suggest WHERE
  and WHAT DIRECTION. Tournament validates WHETHER it actually helps.
  This is gradient-guided evolution, not gradient descent on topology.
- **Dolma unchanged** — cone is correct for Dolma (protect circuits).
  Only BIOS mode was redesigned.

### Checkpoint contents (v8-bios)

| File | Size | Contents |
|------|------|----------|
| model.npz | 143 MB | Packed ternary topology + gamma + norms |
| optimizer.npz | 519 MB | Adam state for continuous params |
| importance.npz | 3.6 MB | Row/col/direction importance maps (205 modules) |
| state.json | 1.5 KB | Step, epoch, r_ema, gen_base_pct, losses, gen stats |
| evolution_diagnostics.json | 109 KB | Per-module ternary stats, hottest modules |

## Session 051 — Evolutionary Training + Quantized Kernels

### What was done

1. **Smoke-tested BIOS training** — 559M params, 512 seq_len, data loading,
   forward/backward all clean. Initial throughput: 3.3k tok/s.

2. **Profiled the performance bottleneck** — backward pass was 73% of step
   time, dominated by `grad_w = gs_2d.T @ x_2d` (442M float32 gradients).
   This dense matmul existed only for sign-based flip accumulation — the
   optimizer never used it.

3. **Replaced gradient flips with evolutionary mutation** — ternary topology
   is now a genome that evolves via mutation + tournament selection.
   Relational loss forms a cone-shaped restriction: wide at r≈1 (explore),
   narrow at r≈0 (frozen). Champion never degrades (double-buffered).
   Result: 3.3k → 5.9k tok/s.

4. **Profiled ternary kernel performance** — custom Metal kernels were
   2-4x SLOWER than float32 matmul. Root cause: 1024× memory access
   amplification (1M threads each independently reading same rows) +
   GPU shader cores vs AMX hardware. The bit-shift decode was negligible
   (0.24ms for full model).

5. **Replaced Metal kernels with MLX quantized_matmul** — 2-bit affine
   quantization maps ternary {-1,0,+1} cleanly to MLX's uint32 format.
   Apple's optimized AMX path: 2.3-3.7x per matmul. MLX autograd handles
   backward natively — no custom VJP needed. Result: 5.9k → 9.5k tok/s.

6. **Built computation probe** — generates fresh math/clojure examples,
   greedy-decodes, checks exact match. Three tiers. Integrated into
   train.py at eval intervals. Grokking signal: accuracy 0% → >0%.

### Performance journey

| Change | tok/s | BIOS 50K | Speedup |
|---|---|---|---|
| Start (gradient flips + custom Metal) | 3.3k | 69h | 1.0x |
| + Evolutionary mutation (no grad_w) | 5.9k | 41h | 1.7x |
| + MLX quantized_matmul (AMX path) | 9.5k | 25.5h | 2.7x |

### Design decisions made

- **Gradient descent for continuous, evolution for discrete** — clean
  separation. Adam trains gamma and norms. Tournament selects topology.
  No gradient through ternary weights at all.
- **Relational loss IS the temperature** — no separate annealing schedule.
  The cone narrows naturally as the model learns.
- **MLX quantized_matmul over custom kernels** — Apple's AMX hardware
  path beats any custom Metal shader. The ternary concept is sound;
  the implementation needed Apple's infrastructure.
- **Computation probe over loss-only monitoring** — loss can drop via
  memorization. The probe tests actual generalization on novel inputs.
  Accuracy >0% is the definitive circuit formation signal.

### Architecture insight: why ternary was slow

The custom Metal ternary kernel was naive: 1 thread per output element,
no tiling, no shared memory. For a 1024×1024 matmul:
- 1M threads each read 4KB independently = 4.6 GB total traffic
- But unique data is only 4.5 MB
- **1024× memory amplification**

Plus: custom Metal shaders run on GPU compute units. Apple's matmul
(including quantized_matmul) dispatches to AMX — dedicated matrix
hardware that custom shaders cannot access.

The bit-shift decode was ~0.24ms — essentially free. The ternary
concept works. It just needs Apple's optimized paths.

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
  result from the expression alone. No multi-representation interleaving.
- **Babashka IS ground truth** — all generation from babashka eval.
- **Phase flag** over config-driven — `--phase bios|dolma` sets sensible
  defaults, individual flags override.
- **Simplified from v7** — no per-stage phase controllers.

## Session 049 — Architecture + All-Ternary + Tokenizer

### What was done

1. **Rewrote `scripts/v8/model.py` from scratch** — clean break from v7
   - CompressorMERA + PipelineMERA = DualMERA
   - d=1024, 6 effective levels at seq=512, 8 at seq=4096
   - 4 parallel pathways per sieve level, feedback cascade

2. **All-ternary conversion** — TernaryEmbedding + TernaryLinear everywhere
   - 559M logical params, 99.7% ternary, 146 MB packed storage

3. **Qwen3 BBPE tokenizer** — vocab 151,936, byte-level BPE, no UNK tokens

## v7 Dolma Run — Summary

Ran steps 0-40K (~655M tokens). Killed at 40K — eval peaked at
20K then monotonically worsened. Architecture validated but Dolma
can't train deep stages. Math stratum was the only one still growing.
Diagnosis: architecture right, data wrong. Full probe data in
results/vsm-lm-v7/.

## v8 Architecture — Dual MERA

**Full design doc:** `mementum/knowledge/explore/v7.1-sieve-pipeline.md`

```
COMPRESSOR MERA (~253M ternary, incl. 156M embedding):
  8 levels: level 0 own (stride 8) + levels 1-7 shared MERA (stride 2 each)
  W=8, seq_len=4096, d_model=1024, Qwen3 vocab=151936
  8 register positions pass through all levels
  Output: multi-scale representations + register states

PIPELINE MERA (~335M ternary):
  8 levels, each a sieve with 4 parallel pathways (2L ternary each)
  Level 0 own + levels 1-7 shared sieve weights
  7 reducers + 7 feedback cascade steps

TOTAL: 559M logical, ~146 MB packed, 99.7% ternary
```

### Training regime: gradient-informed evolutionary descent

- Ternary topology = genome (559M loci × 3 alleles)
- Continuous params (gamma, norms) = Adam
- Double-buffered: champion never degrades
- 4 mutant strategies per generation (conservative/standard/aggressive/explorer)
- BIOS: constant budget (0.5%), depth-weighted, gradient-informed sampling
- Dolma: relational loss cone (protect BIOS circuits)
- Gradient signal: |∂L/∂γ| → row importance, mean(|x|) → col importance
- Teacher-forced probe in tournament fitness
- Forward/backward via MLX quantized_matmul (Apple AMX, 2-bit)

## Key files

| Purpose | Path |
|---------|------|
| **v8 design doc** | `mementum/knowledge/explore/v7.1-sieve-pipeline.md` |
| **v8 model (dual MERA)** | `scripts/v8/model.py` |
| **v8 ternary (quantized_matmul)** | `scripts/v8/ternary.py` |
| **v8 tokenizer (Qwen3 BBPE)** | `scripts/v8/tokenizer.py` |
| **v8 training loop** | `scripts/v8/train.py` |
| **v8 computation probe** | `scripts/v8/compute_probe.py` |
| **v8 kernel benchmark** | `scripts/v8/bench_kernel.py` |
| **BIOS data generator (bb)** | `bb/us/whitford/verbum/bios.clj` |
| **BIOS shard packer** | `scripts/v8/pack_bios.py` |
| **Dolma re-tokenizer** | `scripts/v8/retokenize_dolma.py` |
| **BIOS flash design** | `mementum/knowledge/explore/bios-flash-training.md` |
| **BIOS shards** | `/Users/mwhitford/data/fractal-bitnet/shards-bios/` |
| **Dolma Qwen3 shards** | `/Users/mwhitford/data/fractal-bitnet/shards-qwen3/` |
| **v7 model (reference)** | `scripts/v7/model.py` |
| **bb clj2lambda** | `bb/us/whitford/verbum/tasks.clj` |
| **bb config** | `bb.edn` |
| Research program | `mementum/knowledge/explore/VERBUM.md` |

## Servers

| Port | Model | Use |
|------|-------|-----|
| 5100 | Qwen3.5-397B-A17B | Large reference model |
| 5101 | Qwen3-4B | Quick testing |
| 5102 | **Qwen3.5-35B-A3B** Q8 | Primary probe target |
| 5103 | Qwen3-Embedding-8B | Embeddings |
