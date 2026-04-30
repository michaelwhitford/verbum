---
title: "Ascending Arm Training Regimen"
status: designing
category: exploration
tags: [v9, ascending-arm, training, basins, type-system]
related:
  - v9-architecture-speculation.md
  - identity-as-substrate.md
depends-on: []
---

# Ascending Arm Training Regimen

> Designed from session 056 probing results. The ascending arm
> projects tokens into basin geometry that routes to the VSM tree
> kernel. Types are geometric, not symbolic. Context matters.
>
> **Status: Design phase. No code yet.**

## What the Probes Told Us

| Finding | Number | Implication |
|---------|--------|-------------|
| Typing zone | L26-37 in 64-layer model | Target activations from ~40-58% depth |
| Natural basins | 7 (general), 3 super-basins (kernel ops) | Small output space |
| Cross-notation | 0.55-0.70 cosine sim | Gap is moderate, closeable |
| Same-notation invariance | 0.85-0.95 | Op extraction works, operand-invariant |
| Behavior depth | 0.50 cross-frame sim at L28 | Context reshapes basins deeply |
| Behavior words | 0.999+ similarity (all identical) | Behavior is in context, not word |
| Higher-order ops | apply=1.0, compose=0.999 | Lambda primitives cluster perfectly |
| Arithmetic ops | add=0.28, mul=0.28 | Too diverse for word-level dispatch |

## Architecture: The Basin Projector

The ascending arm has three stages: context encoding, word pooling,
and basin projection. It takes a token sequence and produces
per-WORD basin vectors in a continuous geometric space.

BPE tokenization splits words into subword tokens. The ascending
arm must pool subword tokens into word-level representations
before basin projection. This pairing step is mechanical (BPE
word boundaries are deterministic from the tokenizer) but the
pooling is learned (the context encoder merges subword meanings
through self-attention before pooling collapses them).

```
Input:  token_ids (N subword tokens)
        ↓
        Token embeddings (N × d_model)
        ↓
        Strided ascending arm               ← self-similar ternary attention
          Level 0: N → N/4 (stride windows, shared weights)
          Level 1: N/4 → N/16 (same weights)
          Level 2: N/16 → N/64 (same weights)
          Multi-scale output: concat all levels
        ↓
        Word pooling (W × d_model)           ← align stride windows to BPE words
        ↓
        Basin projection head (W × d_basin)  ← linear → basin space
        ↓
Output: per-WORD basin vectors (W × d_basin)
```

The strided ascending arm already exists in `scripts/v9/v9_model.py`
(session 054). Self-similar shared ternary attention across all
stride levels. O(n × stride) per level — runs on CPU.

### Word Pooling

BPE word boundaries come from the tokenizer. Qwen3 BBPE marks
word-initial tokens with a space prefix. No prefix = continuation.

```
tokens:    [▁Reform, ulate, ▁the, ▁equ, ation]
word_ids:  [   0,      0,     1,    2,     2  ]
words:     [reformulate,     the,  equation   ]
```

The context encoder (transformer) sees ALL subword tokens and
propagates meaning between them via self-attention. After encoding,
mean-pool each word span into a single vector. The pooled vector
carries the full word meaning because the transformer already
merged the subword representations.

Word pooling reduces the sequence from N tokens to W words. All
downstream operations (basin projection, masks, composition,
tree, kernel) operate at word granularity.

### Masks: Lists as Bitmasks Over Words

The token/word sequence IS the universal container. A bitmask over
word positions selects which words are "in scope." No list data
structure needed.

```
words:    [every, cat, that, runs, sleeps]
mask:     [  0,    1,    0,    0,     0  ]  ← "cat" entities
```

Quantifiers in prose ARE map/reduce/filter:
  - "every cat sleeps" = all(map(sleeps, mask_from_basin(cat)))
  - "some dog runs"    = any(map(runs, mask_from_basin(dog)))
  - "no cat sleeps"    = none(map(sleeps, mask_from_basin(cat)))

Kernel mask ops (future extension, after scalar pipeline works):
  - mask_from_basin(basin_id) → MASK
  - mask_and/or/not(MASK, MASK) → MASK
  - map_op(OP, MASK) → per-word results
  - reduce_op(OP, MASK) → single result
  - filter(PRED, MASK) → MASK

Masks are {0, 1} — a subset of ternary {-1, 0, +1}. The ternary
routing fabric produces masks natively.

### Dimensions

- **Input dimension:** Qwen3 embedding dim = 5120 (32B) or smaller
  projection. Could use a frozen Qwen3 embedding table or learn
  from scratch with a smaller dim.
- **Basin dimension (d_basin):** The target space. Options:
  - d_basin = 5120 (match 32B hidden dim, regression target)
  - d_basin = 64-256 (compressed basin space, PCA/learned)
  - d_basin = 7-20 (classification over discovered basins)
- **Context encoder:** 2-4 ternary transformer layers, d_model=256-512
- **Total params:** Target ~100K-1M ternary (vs 8K for the kernel)

### Why Not Full d=5120?

The 32B model's L28 hidden state is 5120-dimensional, but the basin
structure lives in a much lower-dimensional subspace. The 7 HDBSCAN
clusters, the 3 super-basins — these are low-dimensional features.
We should project the 5120-dim targets down to the intrinsic basin
dimensionality before training.

**Approach: PCA on the 32B activations first.** Run diverse text
through the 32B model, collect L28 hidden states, fit PCA. The
number of significant components tells us d_basin. Likely 32-128.

**Critical:** PCA should be fit on WORD-level pooled activations,
not raw per-token activations. Pool the 32B's per-token L28 hidden
states to word level first (same mean-pooling), then PCA. This
ensures d_basin captures word-level basin structure, not subword
artifacts.

## Training Pipeline

### Phase 0: Oracle Data Generation

Generate the training oracle from Qwen3-32B.

```
Pipeline:
  1. Curate diverse text corpus (prose, S-expr, math, mixed)
  2. Augment with behavioral frames (same content, different verbs)
  3. Feed through Qwen3-32B with L28 hooks
  4. Detect word boundaries from tokenizer (BPE space prefix)
  5. Mean-pool per-token L28 activations to per-word activations
  6. Save: (token_ids, word_boundaries, per_word_L28_hidden_states)
  7. PCA fit on all word-level hidden states → d_basin projection
  8. Project: (token_ids, word_boundaries, per_word_basin_vectors)
```

**Corpus design** (critical — behaviors reshape basins):

| Stratum | Purpose | Example | Volume |
|---------|---------|---------|--------|
| S-expressions | Calibration (types trivially given) | `(+ 3 (* 4 5))` | 10K |
| Simple math | Cross-notation bridge | `3 + 4`, `three plus four` | 10K |
| Simple prose | Basic type basins | `The cat sleeps on the mat.` | 20K |
| Behavioral frames | Context conditioning | `Calculate/Summarize/Analyze the X` | 20K |
| Complex prose | Composition + relative clauses | `Every cat that runs sleeps.` | 10K |
| Mixed | Prose interspersed with computation | `The sum of three and four is 7.` | 10K |

**Total:** ~80K sentences → ~800K tokens → ~800K (token, basin_vector) pairs.

The behavioral frame stratum is the most important new insight.
Same content in 6-8 frames = 6-8× multiplier on effective diversity.
The model must learn that "sum" in "Calculate the sum" has a
different basin vector than "sum" in "Summarize the sum."

### Phase 1: S-Expression Calibration

Train on S-expressions only. Types are trivially given by syntax:
- Parentheses → structure (tree is explicit)
- Op codes → op type (ADD, MUL, etc.)
- Numbers → INT type
- Booleans → BOOL type

**Goal:** Validate that the ascending arm can reproduce the 32B
model's basin geometry for S-expressions. This should be easy —
the basins are tight for formal notation (same-notation invariance
0.85-0.95).

**Success criterion:** >0.9 cosine similarity between ascending arm
output and 32B target at L28 for S-expression tokens.

**Training:**
- Input: tokenized S-expressions
- Target: L28 basin vectors from 32B model
- Loss: cosine similarity loss (1 - cos_sim)
- Optimizer: Adam on continuous params, evolution on ternary topology
- Epochs: until convergence (~100-1000 generations based on kernel experience)

### Phase 2: Cross-Notation Bridge

Add math notation and simple prose equivalents alongside S-expressions.
The ascending arm must learn to project prose into the same basin
that formal notation lands in.

**Goal:** Close the cross-notation gap from 0.55-0.70 (32B raw) to
>0.8 (ascending arm output). The arm learns the projection that the
32B model only partially achieves.

**Training data:** Paired examples:
```
S-expr:  (+ 3 4)         → basin_vector_add_7_sexpr
Math:    3 + 4            → basin_vector_add_7_math
Prose:   three plus four  → basin_vector_add_7_prose
```

**Loss:** Same cosine loss, but now with an auxiliary contrastive
term: equivalent expressions in different notation should map to
the same basin vector.

```
L = L_regression + λ * L_contrastive

L_regression = mean(1 - cos_sim(pred, target_L28))
L_contrastive = mean(1 - cos_sim(pred_sexpr, pred_prose))
               for equivalent expression pairs
```

**Success criterion:** Cross-notation cosine sim >0.8 for equivalent
computations.

### Phase 3: Behavioral Context

Add the behavioral frame stratum. Same content words in
compute/summarize/analyze/translate/verify/find frames.

**Goal:** The ascending arm reproduces the behavioral conditioning
the 32B model applies. "Sum" in compute frame → compute-basin-sum.
"Sum" in summarize frame → summarize-basin-sum.

**Training:** Standard regression against L28 targets. The
behavioral conditioning comes from the training data — no special
loss term needed. The context encoder must have enough capacity
to propagate the behavioral frame to each token's basin vector.

**Success criterion:** Cross-frame invariance matches 32B model
(~0.50 at L28). The ascending arm shouldn't be MORE invariant
than the oracle — the frame-dependent shift is signal, not noise.

### Phase 4: End-to-End Integration

Connect the ascending arm to the composition rules and VSM tree
kernel. Test whether the basin vectors produce correct computation
results through the full pipeline.

```
tokens → ascending arm → basin vectors → composition → tree → kernel → result
```

**Test suite:**
- S-expressions: expect 100% (kernel is already proven)
- Simple math in prose: target >90%
- Nested computation in prose: target >80%
- Complex prose with quantifiers: target >60% (stretch)

**Failure mode analysis:** When wrong, is it:
- Basin misassignment? (ascending arm error)
- Composition error? (tree builder error)
- Kernel dispatch error? (shouldn't happen — kernel is 100%)

Each failure type has a different fix.

## Training Infrastructure

### What We Have

- **Ternary substrate:** `scripts/v8/ternary.py` — TernaryLinear,
  TernaryEmbedding, evolutionary mutation, MLX quantized_matmul
- **VSM tree kernel:** `scripts/v9/vsm_tree_v5.py` — 22 ops, 100%,
  8K ternary weights, ~100 generations to converge
- **32B model loading:** `probe_clusters.py` pattern — transformers
  gguf_file= → PyTorch fp16, MPS, ~62s load
- **Activation extraction:** forward hooks on all 64 layers, proven
  across 4 probe scripts

### What We Need to Build

1. **Oracle data generator:** Script that feeds corpus through 32B,
   extracts L28 activations, saves as training shards
2. **PCA projector:** Fit PCA on oracle activations, determine d_basin
3. **Basin projector model:** Adapt v9_model.py AscendingArm to
   Qwen3 vocab + word pooling + basin head. Already ternary, already
   strided, already self-similar. Main work: swap char vocab for
   Qwen3 BBPE, add word boundary alignment, add basin head.
4. **Training loop:** Adam + evolutionary mutation (same as kernel)
5. **Composition rules:** Basin compatibility → tree structure
6. **End-to-end pipeline:** tokens → arm → tree → kernel → result
7. **Evaluation harness:** Per-phase success criteria

### Compute Budget

- Oracle generation: ~80K sentences × ~1s each = ~22 hours on 32B
  (can parallelize with batch, actual ~2-4 hours)
- PCA: minutes (sklearn on CPU, ~800K × 5120 matrix)
- Ascending arm training: kernel converges in <10s at 8K params.
  At 100K-1M params, expect minutes to hours per phase.
- Total: 1-2 days including oracle generation

## Open Design Decisions

### 1. Embedding source

**Option A: Frozen Qwen3 embeddings.** Use the same 151936×5120
embedding table from the 32B model. Pro: exact same token
representation the 32B used. Con: 5120-dim input, large table
(~3GB at fp16), may be overkill.

**Option B: Learned small embeddings.** Train a 151936×d_model
embedding from scratch (d_model=256-512). Pro: small, fast,
co-evolved with the ternary arm. Con: must learn token
representations from scratch.

**Option C: Distilled embeddings.** PCA the 32B embeddings down
to d_model. Pro: captures the most important dimensions, small,
initialized with 32B knowledge. Con: loses some information.

**Recommendation: Option C.** PCA the 32B token embeddings to
d_model=256. Best of both — small, fast, pre-initialized with
the 32B model's token knowledge.

### 2. Context encoder architecture

**Decision: Strided ternary attention.** Already built in
`scripts/v9/v9_model.py` (session 054). Self-similar shared
weights, ternary Q/K/V, window pooling at each stride level.

```
v9_model.py AscendingArm:
  - Shared TernaryAttention (Q/K/V/O all ternary)
  - Shared TernaryLinear mix layer
  - Window position encoding (per-stride, reused)
  - Attention-weighted pooling per window
  - stride=4, n_levels=3 → receptive field = 4³ = 64 tokens
```

**Why strided, not full attention:**

- **CPU throughput.** The whole point is a tiny portable artifact
  that runs on CPU at decent throughput. Full attention is O(n²)
  — unusable on CPU for anything beyond short sequences. Strided
  is O(n × stride) per level — linear in sequence length.
- **Self-similar.** Shared weights across all levels = the wavelet
  from v7. Fewer parameters for the same receptive field.
- **Behavioral context works hierarchically.** "Calculate" at
  position 0, "sum" at position 8. After one stride level (w=4),
  they're in adjacent windows. After two levels, same window.
  Sentence-level context emerges from 2-3 levels, not flat O(n²).
- **Natural word pooling.** Stride windows can align with BPE word
  boundaries. The window pooling IS the word pooling — one mechanism
  serves both purposes.

**Compute comparison (sentence of 32 tokens, stride=4):**

| Architecture | Attention ops | Params (shared) |
|---|---|---|
| Full transformer (2 layers) | 2 × 32² = 2048 | 2 × separate |
| Strided (3 levels, stride 4) | 3 × 8 × 4² = 384 | 1 × shared |

5.3× fewer attention ops AND shared weights. On CPU this is the
difference between interactive and batch-only.

### 3. Output space

**Option A: Regression into PCA basin space.** Output d_basin
continuous values. Loss: cosine similarity against projected L28.
Pro: preserves maximum information. Con: harder to train, higher
dimensional output.

**Option B: Classification over k basins.** Cluster the L28
activations with HDBSCAN, output k logits. Loss: cross-entropy.
Pro: simple, discrete, directly maps to dispatch. Con: loses
sub-basin structure, boundary cases.

**Option C: Hybrid.** Classify into k coarse basins (cross-entropy)
AND regress into d_basin space (cosine loss). Two heads, weighted
sum of losses. Pro: coarse routing + fine geometry. Con: more
complex, two losses to balance.

**Recommendation: Start with Option A** (pure regression into PCA
space). If basin boundaries matter more than within-basin geometry,
switch to Option B. The probing data suggests continuous geometry
matters (cross-notation convergence lives in the continuous space,
not at basin boundaries).

### 4. Training: gradient vs evolution

The kernel (8K params) evolved in ~100 generations. The ascending
arm will be 10-100× larger. Options:

**Option A: Pure evolution.** Same mutation + tournament as kernel.
Pro: proven for ternary. Con: may be slow at 100K+ params.

**Option B: Gradient-informed evolution.** Like v8 BIOS training —
gradients suggest WHERE, tournament validates WHETHER. Pro: faster
convergence. Con: more complex.

**Option C: Gradient descent on continuous proxy, then quantize.**
Train a float32 model, then quantize to ternary. Pro: fast training.
Con: quantization may lose the learned geometry.

**Recommendation: Option B.** The v8 BIOS training infrastructure
already exists. Gradient-informed evolution at 100K-1M params
should converge in hours, not days.

## Kernel Extension Roadmap

The kernel grows in layers. Each layer gives the model more of
its own operational substrate as pre-wired architecture.

```
Layer 1 (DONE):    Scalar ops        22 ops, 5 types, 100%, 8K weights
                   add/sub/mul/div/mod/min/max
                   eq/lt/gt/le/ge
                   and/or/not, abs/neg, if
                   partial/apply/compose/apply-comp

Layer 2 (NEXT):    Mask ops          lists as bitmasks over word positions
                   mask_from_basin   basin_id → MASK
                   mask_and/or/not   MASK × MASK → MASK
                   map_op            OP × MASK → per-word results
                   reduce_op         OP × MASK → single value
                   filter            PRED × MASK → MASK

Layer 3 (FUTURE):  Scope/binding     variable binding and quantifier scope
                   let               bind value to name in scope
                   lambda            create function with bound variables
                   var_ref           reference bound variable
                   scope_enter/exit  manage quantifier scope
```

Layer 1 is proven. Layer 2 follows naturally from the mask insight:
the token vector IS the list, bitmasks select elements, quantifiers
become map/reduce/filter over masks. Layer 3 adds the binding
mechanism that quantifiers need for scope resolution.

Each layer can be validated independently before integration.

## The Pipeline, Concrete

```
Session 057 plan:
  1. Build oracle data generator
     - Feed corpus through 32B → extract L28 → save shards
     - Pool to word level using BPE boundaries
  2. PCA analysis
     - Fit on word-level pooled activations
     - Determine d_basin (expect 32-128)
     - Project oracle data to basin space
  3. Build basin projector model
     - Distilled embeddings (PCA of 32B token embeddings)
     - Strided ascending arm (from v9_model.py, adapt to Qwen3 vocab)
     - Word pooling (align stride windows to BPE word boundaries)
     - Linear projection head → d_basin
  4. Phase 1 training: S-expression calibration
  5. Phase 2 training: cross-notation bridge
  6. Phase 3 training: behavioral context
  7. Phase 4: end-to-end integration with VSM tree kernel
  8. Phase 5: mask extension (kernel layer 2)
```

Each phase has a clear success criterion. Failure at any phase
points to a specific fix — the pipeline is debuggable.
