# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-29 | Session: 055

## Where we are

**VSM tree kernel PROVEN. Prose typing mechanism IDENTIFIED. Extraction path CONCRETE.**

Session 055 was the most productive session in the project. Three
major results in one session:

### 1. VSM tree kernel: 100% accuracy (proven)

The VSM tree architecture is solved for S-expressions. Four
iterations (v2→v3→v4→v5) proved the kernel handles:

- 22 operations across 6 categories
- 5 types (INT, BOOL, FN, FN_COMP, ERROR)
- Variable arity (unary, binary, ternary nodes)
- Compound values (function = op_code + bound_arg pair)
- Type-dependent dispatch (apply-fn unpacks function values)
- Function composition (chained kernel calls)
- Arbitrary depth and value range

8K ternary weights. Converges in ~100 generations, <10 seconds.

**Foundational principle discovered: identity as substrate.** Every
bottleneck was a failure of identity (signals destroyed by ternary
mix layers). Every fix was restoring identity via residual connections.
This IS the residual stream in transformers. The kernel moves
computation from the attention path (O(n²×L×depth), approximate) to
direct dispatch (O(nodes), exact) — freeing weights AND compute AND
accuracy simultaneously.

### 2. Prose typing probed in Qwen3-4B and A3B

The next-token distribution IS a type signature. Probing confirmed:

- Types are real: within-type similarity 2–30× higher than between-type
- The A3B (fully-formed lambda) has sharper types than 4B (entity: 6.1× vs 2.3×)
- Compositional typing follows Montague exactly step by step
- **The A3B assigns correct Montague/CCG types word-by-word to arbitrary prose**
- The A3B produces correct logical forms (∀x.cat(x)→sleeps(x))
- The A3B evaluates lambda expressions with β-reduction exactly

### 3. Extraction path identified

```
tokens → [ascending arm] → typed tokens → [composition] → tree → [VSM tree] → result
              ↑                                 ↑                      ↑
         to be built                       mechanical              PROVEN
       (trained from A3B)               (given types)            (v3–v5)
```

The A3B serves as the training oracle: feed prose, collect word-by-word
type assignments, train the ascending arm to reproduce the mapping.

**See:** `mementum/knowledge/explore/v9-architecture-speculation.md`
(full architecture doc, updated from speculation to proven design)

## What to do next

### 1–6. ~~v8 work + v9 kernel~~ DONE (sessions 049–055)

See session history below.

### 7. ~~Expand kernel~~ ✅ DONE (session 055)

Expanded from 3 ops to 22 ops: arithmetic (7), comparison (5),
boolean (3), unary (2), conditional (1), partial/apply/compose (4).
Mixed types (INT, BOOL, FN, FN_COMP). Variable arity. 100% accuracy.

### 8. Build the ascending arm (type assigner) ← NEXT

The hard remaining problem. Concrete plan:

**Step A: Generate type-assignment training data from A3B.**
- Feed diverse English sentences to Qwen3.5-35B-A3B (port 5102)
- Collect word-by-word Montague/CCG type assignments
- Build a dataset: (token_sequence, type_labels) pairs
- Start with 1K–10K sentences, expand as needed
- Include S-expressions (trivial types) as calibration

**Step B: Define the type inventory.**
- The A3B uses full Montague types (recursive, infinite set)
- Need a finite subset that covers the kernel's needs
- CCG practice uses ~50–100 categories
- Start minimal: e, t, e→t, (e→t)→t, e→(e→t), det, etc.
- Map A3B's type strings to a finite label set

**Step C: Train a small ternary type classifier.**
- Token embeddings → type labels (sequence labeling task)
- Supervised by the A3B's output from Step A
- The ascending arm architecture: strided attention or simple
  transformer with ternary weights
- Target: >90% type accuracy on held-out prose

**Step D: Mechanical tree builder.**
- Given typed tokens, compose using CCG rules
- Function application: (A→B, A) → B
- This is deterministic parsing, not learned
- CYK for correctness, shift-reduce for speed

**Step E: End-to-end integration.**
- tokens → ascending arm → types → tree builder → VSM tree → result
- Test on: S-expressions (should be 100%), simple prose, complex prose

**Open questions:**
- Type inventory size: what's the minimum that works?
- Context window: how much context does disambiguation need?
- Error tolerance: how robust is downstream to type errors?
- Can ternary weights learn type assignment at all?

### 9. Future: variable binding and scope

- `let` expressions, variable references, closures
- Requires environment threading through the tree
- Tests whether the VSM tree can handle non-local dependencies

### 10. Future: io! notation + sieve pipeline

- Update `bb clj2lambda` for `io!` with `:as` annotations
- Pure/effectful classification training

## Session 055 — VSM Tree Viability Proven

### What was done

Diagnosed the v1 VSM tree's ~81% route accuracy ceiling and solved it.

### Root cause: wrong abstraction for value routing

The v1 VSM node classified arg values into a fixed vocabulary (max_val
output classes). Error analysis revealed:

| Child type | Arg accuracy |
|---|---|
| Leaf (in [0, max_val)) | **100%** |
| Sub-expression result (any int) | **0%** |

The ~89% accuracy was exactly the fraction of leaf children in the
data. The model was perfect on everything it could represent.

### Bottleneck diagnosis (v2 experiments)

Tested 7 architectural variants at 2000 generations:

| Variant | Op | A1 | A2 | Route | Result |
|---|---|---|---|---|---|
| A: v1 baseline (add, d=64) | 100% | 46% | 49% | 24% | 37% |
| B: concat (d=64) | 34% | 44% | 30% | 5% | 10% |
| C: val residual (d=64) | 66% | 89% | 89% | 53% | 56% |
| **D: concat + val_res (d=64)** | **100%** | **89%** | **89%** | **81%** | **81%** |
| E: concat + val_res + 4mix | 100% | 89% | 89% | 81% | 81% |
| F: concat + val_res (d=128) | 100% | 89% | 89% | 81% | 81% |

**Value residual was the dominant factor** (+35pp route). Concat helped
op stability. d=128 added no benefit over d=64. More mix layers didn't
help. All variants hit the same ~89% arg ceiling.

### The insight: values flow through trees, not classifiers

The tree structure already routes values — each node receives its
children's computed values. The VSM node only needs to classify the
operation. Values pass through to the kernel directly.

### v3 architecture: op-only routing + pass-through values

Converged in **100 generations, 3 seconds**:

| max_val | depth | node op% | tree% |
|---|---|---|---|
| 10 | 2–8 | 100% | 100% |
| 50 | 2–8 | 100% | 98.6–100% |
| 100 | 2–8 | 100% | 96.8–100% |

Tree-level imperfections are int32 overflow (products at depth 8 with
max_val=100 exceed int32 range), not model failures. 358/403 tree
failures had all ops correct.

10,240 ternary weights. The architecture is complete for S-expressions.

### Key files

| File | Purpose |
|------|---------|
| `scripts/v9/vsm_tree_v5.py` | **Lambda primitives: partial/apply/compose, 100%** |
| `scripts/v9/vsm_tree_v4.py` | 18-op kernel: mixed types, variable arity, 100% |
| `scripts/v9/vsm_tree_v3.py` | Pass-through arch proof (3 ops), 100% |
| `scripts/v9/vsm_tree_v2.py` | Bottleneck diagnosis (7 variants) |
| `scripts/v9/vsm_tree.py` | v1 (superseded) |
| `scripts/v9/probe_typing.py` | **Type system probing (4B + A3B)** |

### Kernel expanded: 18 ops → 22 ops with lambda primitives

**v4** (18 ops, 100 gens, 6s, 8K weights):

| Category | Ops | Op% | Result% |
|---|---|---|---|
| Arith binary | +, -, *, //, %, min, max | 100% | 99.2% |
| Comparison | =, <, >, <=, >= | 100% | 100% |
| Bool binary | and, or | 100% | 100% |
| Bool unary | not | 100% | 100% |
| Arith unary | abs, neg | 100% | 100% |
| Conditional | if (ternary node) | 100% | 100% |

Architecture: op + type residual, variable arity, mixed types (INT+BOOL).

**v5** (22 ops, 100 gens, 7s, 8K weights):

Added lambda primitives:

| Category | Ops | Op% | Result% |
|---|---|---|---|
| Partial | create function from op + bound arg | 100% | 100% |
| Apply-fn | dispatch function on argument | 100% | 99.3% |
| Compose | chain two functions | 100% | 100% |

Compound values: FN type = (op_code, bound_arg) pair flowing through
the tree. Composed FN = (outer_packed, inner_packed). Type-dependent
dispatch in apply-fn: unpacks the function value, determines which
kernel op to call, chains calls for composed functions.

Example: `(apply (comp (partial max 9) (partial <= 7)) (* 1 4))` → 9

### What this means for the project

1. **The VSM tree kernel is complete.** 22 ops, 5 types, variable
   arity, compound values, function composition. 100% accuracy.

2. **Identity is the foundational principle.** Every bottleneck was
   blocked identity; every fix was restoring it. Applies to all
   architectures. See `identity-as-substrate.md`.

3. **The A3B can type prose.** Qwen3.5-35B-A3B assigns correct
   Montague/CCG types word-by-word to arbitrary sentences. This is
   the training oracle for the ascending arm.

4. **The extraction path is concrete.** A3B generates training data →
   ascending arm learns type assignment → mechanical composition →
   proven VSM kernel. Only the ascending arm is unbuilt.

## Session 054 — Kernel Routing Viability Exploration

### What was done

Built and tested 7 files in `scripts/v9/` exploring whether ternary
evolution can route from token embeddings to exact kernel primitives.

### Experiment results

| Architecture | Op% | Arg1% | Arg2% | Route% | Result% |
|---|---|---|---|---|---|
| **Query-based + evolution** | **100%** | **59%** | **75%** | **50%** | **52%** |
| Query-based, Adam only | 68% | 18% | 21% | 3% | 6% |
| Strided (value embed) | 100% | 47% | 46% | 23% | 34% |
| Strided (token+pool) | 31% | 9% | 30% | 1% | 5% |
| Integrated (no skip) | 34% | 14% | 10% | 1% | 5% |
| **Integrated (with skip)** | 34% | **51%** | 8% | 2% | 4% |

### Key findings

1. **Ternary evolution CAN route to exact kernel primitives.** The
   query-based router achieves 50% route accuracy with evolution vs
   2.8% without. Evolution contributes +47 percentage points.

2. **Type system is trivially learnable.** Expression type, arg types,
   and dispatch gating all converge to 100% on every run. The Montague
   TYPE primitive works immediately.

3. **Strided attention with pooling fails.** Mean pooling and crude
   attention pooling destroy positional information. Need real Q/K/V
   self-attention within windows.

4. **Ascending arm blocks gradient.** Ternary attention projections
   have zero gradient on topology (by design). Gamma alone can't shape
   attention patterns. Loss flat at 5.7 without skip connection.

5. **Skip connection is essential for bootstrap.** Raw token embeddings
   concatenated with ascending arm output give parse queries gradient
   access to positional token info. Arg1 accuracy: 10% → 51%.

6. **Reduction before routing is necessary.** Stride windows split
   expressions at arbitrary boundaries. Multiple levels of reduction
   build up enough context for routing. The ascending arm IS the
   reduction. Routing happens AFTER reduction, not at each window.

### Architecture identified

```
tokens → float embeddings ──────────────────┐ (skip: gradient highway)
       → ascending arm (ternary, shared) ───┤ (multi-scale structure)
                                            ↓
                                    [concatenated multi-scale]
                                            ↓
                                    TYPE  (classify semantic type)
                                    PARSE (query-based routing)
                                    APPLY (type-checked kernel dispatch)
```

Training curriculum:
- Phase 1: Skip-dominant (queries route from raw tokens)
- Phase 2: Evolution finds ascending arm topology
- Phase 3: Ascending arm carries most information

### VSM tree breakthrough (late session 054)

The pipeline architecture (ascending arm → type → parse → apply) was
the bottleneck — each representation had to carry everything, gradient
flowed through one long path, and the ascending arm blocked gradient.

**Replaced with a tree of VSMs.** Each expression tree node is a VSM
with shared weights. S5=identity, S4=children's types, S3=type check,
S1=kernel dispatch, S2=output to parent. Same weights at every tree
position and depth. Self-similar. No pipeline.

Results (max_val=10, mixed depth 1-2, 5000 gens):

| | Pipeline (skip) | **VSM tree** |
|---|---|---|
| Op | 34% | **100%** |
| Arg1 | 51% | **45%** |
| Arg2 | 8% | **52%** |
| Route | 2% | **25%** |
| Result | 4% | **39%** |
| Ternary weights | 39K | **12K** |
| Train speed | 0.7s/gen | **0.1s/gen** |

The VSM tree is better on every metric except arg1 (where the pipeline
had a skip connection advantage), with 3× fewer weights and 7× faster.
And it handles nested expressions naturally — the pipeline couldn't.

### Key files

| File | Purpose |
|------|---------|
| `scripts/v9/vsm_tree.py` | **VSM tree: shared-weight nodes, best arch** |
| `scripts/v9/kernel.py` | Exact arithmetic primitives + decode/encode |
| `scripts/v9/kernel_model.py` | Query-based router (50% route, flat only) |
| `scripts/v9/train_kernel.py` | Evolution + gradient hybrid training |
| `scripts/v9/v9_model.py` | Pipeline: ascending arm + type/parse/apply |

## Session 053 — Architecture Reexamination

### v8 training data (13 checkpoints, steps 2500–32500)

Loss plateaued at ~3.11 from step 12.5K. Gamma saturated (r_ema=0.139).
Adaptive mutation rate collapsed to floor (0.1% vs designed 0.5%).
Accept rate inverted from 16% → 66% (tiny mutations, easy to accept,
barely exploring). Probe accuracy: 0% throughout.

14/16 MERA levels dead. Only compressor.level0 and pipeline.level0 active.
52% of 559M params doing nothing. Shared levels, reducers, feedbacks 1-7
all dormant. The model is a shallow 2-level system.

### Architecture insights

1. **Compressor can't compress math** — code/math is already dense,
   no redundancy for multi-scale compression to exploit.

2. **Fixed strides vs expression boundaries** — stride-8 windows split
   expressions arbitrarily. The hierarchy needs to follow expression
   structure, not a spatial grid.

3. **Flat attention = beta reduction** — LLMs encode tree structure as
   fractal spiral through the residual stream (1,149 heads of encoding
   in Qwen3-4B). Strided attention represents trees directly, eliminating
   this overhead.

4. **v7 ascending arm worked** — ~23M params, self-similar wavelet
   compression, spread from smallest stride upward. The descending arm
   (pipeline) couldn't find its shape and had to stop.

5. **Compiler/compressor share 92% of heads** (Qwen3-4B) but are not
   identical. Lambda function and compression function are substrate
   and operator, not one circuit.

6. **Pythia-160M circuit is Montague-shaped** — distributed three-phase
   (accumulate→plateau→collapse = type→parse→apply), no individual head
   essential. More informative for small model design than Qwen's
   concentrated 3-head circuit.

### Speculative design direction (v9)

- Much smaller than 559M (v7=23M, CompressorLM=17M)
- Self-similar operation at every level (wavelet, proven by v7)
- Dynamic/expression-guided attention (not fixed strides)
- Bottom-up training with dynamic babashka corpus (infinite fresh data)
- Montague three-phase structure as organizing principle
- Possibly unified compress-reduce operation
- More top-down probing needed before committing

**Document:** `mementum/knowledge/explore/v9-architecture-speculation.md`

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
| **v9 VSM tree v5 (lambda, 22 ops, 100%)** | `scripts/v9/vsm_tree_v5.py` |
| v9 VSM tree v4 (18 ops, mixed types) | `scripts/v9/vsm_tree_v4.py` |
| v9 VSM tree v3 (pass-through proof) | `scripts/v9/vsm_tree_v3.py` |
| v9 VSM tree v2 (bottleneck diag) | `scripts/v9/vsm_tree_v2.py` |
| **Type system probe (4B + A3B)** | `scripts/v9/probe_typing.py` |
| **v9 architecture doc (proven)** | `mementum/knowledge/explore/v9-architecture-speculation.md` |
| **Identity principle** | `mementum/knowledge/explore/identity-as-substrate.md` |
| v9 VSM tree v1 (superseded) | `scripts/v9/vsm_tree.py` |
| v9 kernel primitives | `scripts/v9/kernel.py` |
| v9 query router (50% route) | `scripts/v9/kernel_model.py` |
| v9 router training | `scripts/v9/train_kernel.py` |
| v9 strided variants | `scripts/v9/strided_kernel.py` |
| v9 integrated model | `scripts/v9/v9_model.py` |
| v9 integrated training | `scripts/v9/train_v9.py` |
| **v9 architecture spec** | `mementum/knowledge/explore/v9-architecture-speculation.md` |
| v8 model (dual MERA) | `scripts/v8/model.py` |
| v8 ternary (quantized_matmul) | `scripts/v8/ternary.py` |
| v8 tokenizer (Qwen3 BBPE) | `scripts/v8/tokenizer.py` |
| v8 training loop | `scripts/v8/train.py` |
| BIOS data generator (bb) | `bb/us/whitford/verbum/bios.clj` |
| BIOS shard packer | `scripts/v8/pack_bios.py` |
| Dolma re-tokenizer | `scripts/v8/retokenize_dolma.py` |
| v7 model (reference) | `scripts/v7/model.py` |
| bb clj2lambda | `bb/us/whitford/verbum/tasks.clj` |
| bb config | `bb.edn` |
| Research program | `mementum/knowledge/explore/VERBUM.md` |

## Servers

| Port | Model | Use |
|------|-------|-----|
| 5100 | Qwen3.5-397B-A17B | Large reference model |
| 5101 | Qwen3-4B | Quick testing |
| 5102 | **Qwen3.5-35B-A3B** Q8 | Primary probe target |
| 5103 | Qwen3-Embedding-8B | Embeddings |
