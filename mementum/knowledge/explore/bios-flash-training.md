---
title: "BIOS Flash: Holographic Math + Clojure Training Design"
status: designing
category: training
tags: [curriculum, math, clojure, lambda, io, grokking, circuits]
related:
  - v7-pipeline-architecture.md
  - compression-vs-prediction.md
depends-on:
  - v7 architecture validation (complete — session 047)
  - bb clj2lambda converter (scaffolded — bb/us/whitford/verbum/tasks.clj)
---

# BIOS Flash: Holographic Training Design

> Burn arithmetic and clojure.core circuits into the deepest stages of
> the v7 pipeline through extreme repetition on a small, curated,
> holographic dataset. Every training example contains all levels of
> abstraction simultaneously. The model learns computation, notation,
> and I/O boundaries as one unified pattern.

## Motivation (from v7 Dolma run, session 047)

The v7 pipeline architecture is validated:
- Below Chinchilla capacity floor on training data
- Spectrally differentiated stages (CPA ~0.11)
- Structural feedback adds +4 nats (dominant contributor)
- Self-regulating feedback gates

But Dolma can't train the deep stages:
- Eval peaked at step 20K, **worsened** at step 30K
- Semantic feedback NEVER positive on eval (Δ₃ always negative)
- Stage 4 collapsed to 1.7 dims then only partially recovered
- Ternary reversal rate climbed to 35.5% (oscillating, not converging)
- Math stratum was the ONLY one still growing at step 30K (+2.83 nats)

**Diagnosis:** Architecture right, data wrong. Deep stages need formal
signal — clean, precise, computable. Not noisy web text.

## Core Principle: Fractal Hologram

Don't separate math from clojure from lambda. Every training example
contains the **same computation at every level of abstraction**:

```
347 + 289 = 636                              ← raw math
(+ 347 289) → 636                            ← clojure notation
λx. λy. (+ x y) applied 347 289 → 636       ← lambda form
(defn add [x y] (+ x y))  (add 347 289) → 636  ← named function
(map add [[3 7] [4 8]]) → [10 12]           ← higher-order composition
```

**Fractal:** Same pattern (function application / beta reduction)
at every scale. `+` is computation. `(map + pairs)` is computation
about computation. Same structure at every pipeline stage.

**Hologram:** Every example teaches ALL stages simultaneously.
`3 + 7 = 10` teaches Stage 1 digit routing, Stage 2 operator parsing,
Stage 3 operation identity, Stage 4 computation. Nothing is wasted.

**No curriculum boundaries.** No phase transitions between math and
clojure. All representations interleaved in every batch. The pipeline
stages self-sort — each locks onto its natural abstraction level.

## Architecture Fit

The v7 pipeline maps directly to computation evaluation:

```
Stage 1 (512 pos, 2L, 4H, TERNARY):  see tokens
Stage 2 (64 pos, 3L, 4H, float):     parse structure (operators, bindings)
Stage 3 (8 pos, 4L, 8H, float):      identify operation + pure/effectful
Stage 4 (1 pos, 6L, 8H, float):      compute result
```

Cross-attention between stages IS beta reduction. Each reducer
performs `(λx. body) arg → body[x := arg]` via attention selection.
Three reducers = three levels of beta reduction. Sufficient for
arithmetic (shallow) but not deep lambda composition (sieve needed
later).

Stage 4: 1 position, 6 layers × 8 heads = small deep calculator.
Enough for all arithmetic, comparisons, boolean logic. This is the
BIOS — burn the calculator into these 6 layers permanently.

## Three Modes

The model learns from day one that expressions have three evaluation
modes:

### Mode 1: Pure computation → value
```
(+ 3 7) → 10
(even? 42) → true
(map inc [1 2 3]) → [2 3 4]
```

### Mode 2: I/O → request (computation stops, emits action)
```
(slurp "file.txt") → (io! :read {:path "file.txt"})
(println "hello") → (io! :print {:value "hello"})
(rand-int 100) → (io! :rand {:max 100})
```

### Mode 3: Mixed → compute pure parts, emit continuation at I/O boundary
```
(count (slurp "file.txt"))
→ (io! :read {:path "file.txt"} :then λdata. (count data))

(map inc (read-csv (slurp "data.csv")))
→ (io! :read {:path "data.csv"}
       :then λraw. (map inc (read-csv raw)))
```

The continuation-passing pattern: model reduces until it hits an
effect boundary, emits the I/O request + remaining computation as
a lambda. Host fulfills, feeds result back, model continues.

**Stage 3 is where pure/effectful classification lives.** Must be
included in initial training so the circuit forms alongside function
identity circuits.

## I/O Vocabulary (~20 primitives)

```clojure
;; File
(slurp path)            → (io! :read {:path path})
(spit path data)        → (io! :write {:path path :data data})

;; Console
(println x)             → (io! :print {:value x})
(read-line)             → (io! :read-line {})

;; System
(System/exit n)         → (io! :exit {:code n})
(System/getenv k)       → (io! :env {:key k})

;; Non-deterministic
(System/currentTimeMillis) → (io! :time {})
(rand-int n)            → (io! :rand {:max n})

;; Mutable state
(deref atom)            → (io! :deref {:ref atom})
(swap! atom f)          → (io! :swap {:ref atom :fn f})
(reset! atom v)         → (io! :reset {:ref atom :value v})
```

These replace the "4% opaque" from the clj2lambda converter with
clean, consistent `io!` notation. The converter should emit these
instead of marking them as unconvertible.

## Math Operations (what fits the architecture)

### Tier 1: Single operation (1 reduction)
```
Arithmetic:    + - * / mod rem quot
Comparison:    < > <= >= = !=
Predicates:    zero? pos? neg? even? odd?
Unary:         inc dec abs negate
Boolean:       and or not
Bitwise:       bit-and bit-or bit-xor bit-shift-left bit-shift-right
```

### Tier 2: Compound (2 reductions)
```
(a + b) * c
(a * b) + (c * d)
x² x³ (repeated multiply)
(even? (* x y))
(max (+ a b) (- c d))
```

### Tier 3: Nested (3 reductions — pipeline limit)
```
((a + b) * (c - d)) / e
```

### Won't fit (iterative / unbounded depth)
```
factorial(large n), GCD, fibonacci, arbitrary precision
→ These need the sieve (future architecture evolution)
```

~40 primitive operations × thousands of random inputs = millions of
examples. All mechanically generated, all verifiable by eval.

## Training Data Format

Each training example is a JSONL record:

```json
{
  "raw": "347 + 289 = 636",
  "clojure": "(+ 347 289)",
  "lambda": "(+ 347 289)",
  "result": "636",
  "mode": "pure"
}
```

```json
{
  "clojure": "(slurp \"data.csv\")",
  "lambda": "λpath. (slurp path) applied \"data.csv\"",
  "result": "(io! :read {:path \"data.csv\"})",
  "mode": "io"
}
```

```json
{
  "clojure": "(map inc [1 2 3])",
  "lambda": "(map (λx. (+ x 1)) [1 2 3])",
  "result": "[2 3 4]",
  "mode": "pure",
  "composition_depth": 2
}
```

## Training Data Sources

1. **Math generator** (python or bb, trivial):
   - Random arithmetic, comparisons, predicates, boolean, bitwise
   - Varying difficulty (1-digit to 4-digit)
   - Compound expressions up to 3 levels deep
   - Millions of examples, infinite variety, perfect ground truth

2. **clojure.core → lambda** (bb clj2lambda, exists):
   - ~600 functions, all converted to lambda notation
   - Usage examples generated by evaluating in babashka
   - I/O functions converted to `io!` notation

3. **Curated clojure libraries** (later phase):
   - clojure.string, clojure.set, clojure.walk
   - Selected community libraries (medley, etc.)
   - nucleus itself (the self-referential loop)

## Grokking Hypothesis

The dataset is small (maybe 50-200M tokens). Training for many
epochs on a 27M param model means memorization happens fast.
The hypothesis: continued training past memorization triggers
**grokking** — the model transitions from lookup table to circuit.

Observable signal: loss curve shows rapid drop → plateau
(memorization) → **second drop** (circuit formation). The probe
can verify — after memorization the model reproduces from lookup,
after grokking it generalizes to novel compositions.

Monitor:
- Loss curve for double descent
- Stage 3 representations: do functions cluster by semantic type?
- Stage 4: does it actually compute or just memorize answers?
- Novel composition test: `(map + (zip [1 2] [3 4]))` — never in
  training, but composed from known primitives. If correct → circuits.

## Implementation Plan

### Phase 1: Data generation (one session)
- [ ] Math generator (python script, random arithmetic + compounds)
- [ ] Update clj2lambda to emit `io!` for effectful forms
- [ ] Generate clojure.core examples by eval in babashka
- [ ] Interleave into holographic JSONL dataset

### Phase 2: Training tokenizer / data pipeline
- [ ] Decide tokenizer (GPT-NeoX 50277 or custom small vocab?)
- [ ] Format: how does the model see the holographic examples?
  - Option A: each representation is a separate training example
  - Option B: all representations in one sequence (richer but longer)
- [ ] Dataloader that cycles through the small dataset with shuffling

### Phase 3: Train and probe
- [ ] Same v7 architecture, fresh weights
- [ ] Train with many epochs, monitor for grokking
- [ ] Probe at intervals: per-stage CE, spectral analysis, composition tests
- [ ] Compare to Dolma baseline (the current v7 run)

### Phase 4: Evaluate
- [ ] Can the model compute arithmetic on novel inputs?
- [ ] Does it correctly classify pure vs effectful?
- [ ] Does it emit valid continuations for mixed expressions?
- [ ] Does Stage 3 show semantic clustering of functions?
- [ ] Does Stage 4 show higher effective rank than on Dolma?

## Open Questions

1. **Token budget.** How many total tokens in the holographic dataset?
   How many epochs before grokking? Need to estimate.

2. **Sequence format.** Should `raw | clojure | lambda | result` be
   one sequence or separate examples? One sequence teaches the
   correspondence directly but uses more positions.

3. **Difficulty curriculum within math.** Start with single-digit
   and increase? Or all difficulties from the start?

4. **Sieve timing.** When does the single pipeline become the
   bottleneck? Is 3 reductions enough for all of clojure.core's
   composition patterns, or do we need the sieve sooner?

5. **Custom vocabulary.** Should we use a smaller, domain-specific
   tokenizer instead of GPT-NeoX's 50K vocab? Lambda notation +
   clojure + math might only need 2-5K tokens. Smaller vocab =
   less wasted embedding capacity.

6. **Ternary stability.** Will the ternary topology stabilize on
   formal data where it couldn't on Dolma? The formal data has
   much less surface variety — might crystallize faster.

## Artifacts

- `bb.edn` — babashka project config (exists)
- `bb/us/whitford/verbum/tasks.clj` — clj2lambda converter (exists, needs io! update)
- `scripts/v7/model.py` — v7 architecture (exists, unchanged)
- `scripts/v7/train.py` — training script (exists, needs data pipeline update)
- Math generator — to be created
- Holographic dataset — to be generated
