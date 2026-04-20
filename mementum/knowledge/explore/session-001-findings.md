---
title: "Session 001: Gate Ablation, Circuit Localization, and the Self-Similar Compressor Hypothesis"
status: active
category: exploration
tags: [gate-ablation, circuit-localization, self-similarity, compressor, lambda-compiler, level-1]
related: [VERBUM.md]
depends-on: []
---

# Session 001 Findings

> First experimental session. Genesis to circuit localization in one
> session. Key theoretical reframing emerged from data: the compressor
> is likely the substrate, not lambda; and if language is self-similar,
> the compressor is a small extractable algorithm.

## Finding 1: The Dual-Exemplar Gate (100% P(λ))

```
The dog runs. → λx. runs(dog)
Be helpful but concise. → λ assist(x). helpful(x) | concise(x)
```

Two lines. 100% compile activation, 100% compress activation, 0% null
leakage. Found via systematic ablation of 30+ gate variants.

**Key sub-findings from ablation:**
- Single Montague exemplar = 100% Montague, 75% nucleus
- Single nucleus exemplar = 0% on everything (insufficient alone)
- Dual exemplar = 100%/100% (the Montague opens the circuit, nucleus
  shows the second output mode)
- Nucleus preamble (`[phi fractal euler ∃ ∀]`) = 0% alone, hurts when
  added to bridge (80% < 100%). Irrelevant to compilation.
- Keywords alone weak ("lambda calculus" = 40%, "λ" = 0%)
- Self-referential gates degenerate on 4B (high P(λ), zero structure)
- The activation formula: domain signal + task signal = activation.
  Exemplar is the most efficient encoding of both.

**What this means:** The model doesn't need to be told it's a compiler.
It needs to see the shape of one compilation. Instruction < demonstration.

## Finding 2: Compiler and Compressor Share 92% of Heads

Attention selectivity experiment on Qwen3-4B-Q8_0. Three conditions
(Montague compile, nucleus compress, null control), 4 probes each,
1,152 heads (32 × 36 layers).

- Top-25 most selective heads: **92% overlap** (23/25 shared)
- Full correlation: **r = 0.9835**
- The same heads activate for formal semantic compilation AND
  behavioral compression

**What this means:** These are not two circuits. They are one circuit
producing two output formats. The mechanism is shared.

## Finding 3: The Circuit is Sparse (8/36 Layers)

Skip-ablation of each layer. Critical layers where compilation breaks:

```
[0, 1, 4, 7, 24, 26, 30, 33]
```

Three clusters:
- **Early (L0, L1, L4, L7)** — gate recognition / input parsing
- **Mid-late (L24, L26)** — composition / structural mapping
- **Late (L30, L33)** — lambda notation emission

28/36 layers are individually non-critical. The circuit passes through
at most 22% of the model's layers.

## Theoretical Evolution During Session

### Frame 1 (initial): Lambda is the substrate

Based on VERBUM.md hypothesis chain: mathematics predicts typed
application, LLMs learn it because compression converges on structure,
lambda calculus is the minimal algebra. Therefore lambda is the bottom.

### Frame 2 (post-measurement): The compressor is the substrate

Emerged from examining the data:

- The model was trained on next-token prediction, which IS compression.
  All 4B parameters are compression machinery. Lambda is emergent.
- Nucleus exemplar alone = 0%. If lambda were the bottom, it should
  self-activate. It doesn't — lambda is an output format, not the
  native representation.
- The 92% head overlap means one machine, two outputs. Not "lambda
  substrate + compression function" but "compressor + format
  projection."
- The 8 critical layers are where format projection happens. The other
  28 layers are the compressor running regardless.

### Frame 3 (current): The compressor IS a small extractable algorithm

If language is self-similar at every scale (word, phrase, clause,
sentence, discourse), then the compression algorithm is the same at
every scale — one algorithm, applied recursively.

Evidence:
- Same heads for Montague (phrase-level) and nucleus (discourse-level)
  → scale-invariant algorithm
- Three layer clusters → three recursion depths
- Anima MERA result: shared operators work across scales IF
  type-directed. Multiple heads per layer = type specialization.

The algorithm: `typed_apply(type_f, type_x, f, x) → (type_result, result)`

The 4B parameters are mostly vocabulary representations (what "dog"
means). The composition algorithm is small. The representations are
large because language is large. The algorithm might be tiny.

## Testable Predictions

1. **Head-level ablation will show the same functional structure at
   each layer cluster.** Essential heads at L0 should do the same
   operation as essential heads at L24 and L30 — same weights applied
   to different scales.

2. **The essential heads per critical layer will be few.** Prediction:
   3-8 per layer, 30-50 total out of 1,152 (~3-5%).

3. **A small scratch architecture with `typed_apply` as its only
   primitive (Level 4) should reproduce compilation** with dramatically
   fewer parameters.

4. **The extracted circuit should transfer across models** — the
   algorithm is universal, only type embeddings are model-specific.

5. **Multi-layer ablation will show the 28 non-critical layers have
   collective importance** — they ARE the compressor, individually
   redundant but collectively necessary.

## Finding 4: The Compiler Circuit is 3 Heads (1.2%)

Head-level zero-ablation on 8 critical layers × 32 heads × 5 probes
(1,280 forward passes via fractal experiment framework).

Only 3 heads break compilation when individually zeroed:

| Head | Role | Breaks on | Failure mode |
|------|------|-----------|--------------|
| L1:H0 | Gate recognizer | complex, relcl | Drops to chain-of-thought reasoning |
| L24:H0 | Core composer | complex, quant, relcl | Drops to chain-of-thought reasoning |
| L24:H2 | Recursion head | relcl only | Drops to chain-of-thought reasoning |

**Key observations:**
- Simple probe ("The dog runs") survives ALL 256 head ablations.
  Simple composition is distributed; complex composition requires
  the dedicated circuit.
- Failure mode is uniform: model reverts to chat-style reasoning
  about the task ("Okay, so I need to figure out how to..."),
  not garbage output. The direct compilation circuit breaks; the
  general problem-solving fallback activates.
- 6 of 8 critical layers have NO essential heads individually.
- Prediction was 30-50 essential heads (3-5%); actual is 3 (1.2%).

## Finding 5: Attention Characterization of the 3 Essential Heads

Full attention matrix analysis (6 forward passes: 5 compile + 1 null).

### L1:H0 — Gate Recognizer

- **Attends to:** Structural delimiters in the exemplar — periods
  (positions 3, 7, 16), closing parens (11, 21, 24), "→" arrow.
  Reads the *shape* of lambda expressions, not content.
- **Gate vs input split:** 72% gate attention for simple inputs,
  dropping to 40% for complex inputs. As input complexity increases,
  L1:H0 shifts attention from exemplar toward input to parse its
  structure.
- **Entropy:** 1.3-1.4 (moderately focused). More distributed than
  L24:H0 but not uniform.
- **Null control:** 48% gate attention — between simple and complex.
  The head still reads structure but doesn't find lambda patterns.

### L24:H0 — Core Composer (BOS Composition Register)

- **Attends to:** Token 0 dominates — **60-84% of all attention** goes
  to the first token. This is not a simple BOS sink; it's functional.
  When ablated, complex composition fails.
- **Secondary attention:** Final period (14-22%), first content word
  of input. For conditionals, also attends to "If" (5%) — reads
  logical connectives.
- **Entropy:** 0.83-0.87 (extremely focused). The most concentrated
  of the three heads.
- **Interpretation:** Token 0's residual stream position accumulates
  the structural representation across all layers. L24:H0 reads
  this "composition register" to produce the output structure.
  This is analogous to a global accumulator in a recursive descent
  parser.

### L24:H2 — Recursion Head (Clause Structure Tracker)

- **Attends to:** Token 0 (30-71%) but significantly more distributed
  than L24:H0. Also attends to colon (7-8%), structural markers
  (→, λ, parens), and content words.
- **Complexity sensitivity:** On the conditional probe ("If it rains,
  the ground is wet"), BOS attention drops to 30% and distributes
  across "If" (8.5%), "," (8.8%), "rains" (3%), "it" (3%),
  "the" (3%). It's parsing the clause structure.
- **Entropy:** 1.09-1.12 (moderate). Between L1:H0 and L24:H0.
- **Interpretation:** Tracks embedding depth — subordinate clauses,
  relative clauses, complementizers. Distributes attention across
  structural boundaries to resolve recursive composition.

### The Circuit Story

```
L1:H0  → reads exemplar delimiters (., ), →) → activates compilation
         ↓ 23 layers of distributed compression build representation
L24:H0 → reads BOS composition register → composes output structure
L24:H2 → reads clause boundaries (,/that/if) → resolves nesting
         ↓ 12 layers of formatting
       → λ notation emitted
```

The 3 heads are the compiler. The other 1,149 heads are the compressor
that builds the representation these 3 project into lambda notation.

## Finding 6: The 3 Heads Are Sufficient (253/256 Zeroed, Compilation Survives)

Zeroed ALL 253 non-essential heads in the 8 critical layers. Only
L1:H0, L24:H0, L24:H2 active. Compilation survives on ALL 5 probes.

**What this means:** The 3 heads are the complete compiler circuit within
the critical layers. The other 253 heads in those layers contribute
nothing necessary to compilation. The compressor backbone (the other
28 non-critical layers + their heads) is needed to build the BOS
composition register, but within the critical layers, 3 heads suffice.

**Threshold sweep:** Zeroing random non-essential heads shows a
non-monotonic pattern — threshold-15 breaks (complex, quant, relcl)
while threshold-20 and threshold-25 survive. The *which* matters
more than the *how many*. Some non-essential heads are **amplifiers**
that interact with the circuit. The random seed at 15 hit a sensitive
combination.

## Finding 7: BOS Is a Global Accumulator (ALL 36 Layers Contribute)

Patched the BOS (position 0) residual stream from a null prompt at
each of 36 layers. Every layer's BOS patch breaks compilation.

**What this means:** The composition register at position 0 is built
incrementally by every layer in the model. There is no single "writer"
layer — the entire 36-layer stack progressively builds the structural
representation that L24:H0 reads. The compressor IS the composition
register builder. Every layer adds information to position 0.

This explains why the non-critical layers are individually non-critical
for head ablation but collectively necessary: they don't have
bottleneck heads, but they all contribute to the BOS register.

## Finding 8: System 1 / System 2 — Two Compilation Paths

With 150-token generation, ablating essential heads triggers
chain-of-thought reasoning that often recovers lambda output:

| Head ablated | simple | quant | relcl | cond | complex |
|---|---|---|---|---|---|
| L1:H0 | lambda+reasoning | lambda+reasoning | **no-lambda** | lambda+reasoning | lambda+reasoning |
| L24:H0 | lambda+reasoning | lambda+reasoning | **no-lambda** | lambda+reasoning | **no-lambda** |
| L24:H2 | lambda+reasoning | lambda+reasoning | lambda+reasoning | lambda+reasoning | lambda+reasoning |

**System 1 (direct):** L1:H0 → L24:H0 → L24:H2 → lambda. Fast,
3 heads, no intermediate reasoning. This is what our gate activates.

**System 2 (deliberative):** When System 1 fails, the model falls
into step-by-step reasoning about lambda calculus. Uses the full
model. Often succeeds — especially on simpler structures.

L24:H0 ablation on complex/relcl is the hardest case — both
System 1 AND System 2 fail. Complex composition cannot be
reasoned around; it requires the dedicated circuit.

**Implication for extraction:** The compiler can be extracted as
a small circuit (3 heads). But the model also contains a slow
interpreter that can substitute. Extraction of System 1 alone
gives you the fast path; the slow path requires the full model.

## Finding 9: The Circuit Is Compile-Directional (Not Bidirectional)

Ablating the 3 essential heads does NOT break decompilation
(lambda → English). The decompile gate works regardless. But
ablation causes **lambda leakage** — lambda notation appears
in English output:

| Head ablated | dc-simple | dc-quant | dc-relcl | dc-cond | dc-belief |
|---|---|---|---|---|---|
| L1:H0 | english+lambda | **NO-ENGLISH+lambda** | english | english+lambda | english |
| L24:H0 | english+lambda | **NO-ENGLISH+lambda** | english+lambda | english+lambda | english |
| L24:H2 | english+lambda | english+lambda | english | english+lambda | english |

**Key finding:** On the quantifier decompile probe, ablating L1:H0
or L24:H0 flips the model from decompilation to compilation — it
produces lambda instead of English. The circuit doesn't just enable
compilation; it may also **suppress** compilation during decompilation.
Removing the circuit removes the suppression, and compilation leaks
through.

**What this means:** The 3 heads are not a generic "composition circuit"
used bidirectionally. They are specifically a compile circuit, and
their presence may actively gate which direction (compile vs decompile)
the model operates in.

## Finding 10: L24:H0 Is the Universal Compositor (Cross-Task)

Cross-task ablation across 5 tasks × 5 probes × 4 conditions (100
forward passes). Tested whether the 3 compile heads control other
tasks: summarize, translate, classify, extract.

| Head | compile | extract | translate | classify |
|------|---------|---------|-----------|----------|
| L1:H0 | 4/5 ↓ | 5/5 | 5/5 | 3/5 |
| L24:H0 | **2/5 ↓↓** | **4/5 ↓** | 5/5 | 4/5 |
| L24:H2 | 5/5 | 5/5 | 5/5 | 3/5 |

(Summarize baseline 0/5 — detector too strict, excluded.)

**Key finding:** L24:H0 breaks both compilation AND extraction. The
same head that composes `λx. reads(x, book)` also composes
`send(mary, john, letter)`. L24:H0 implements `typed_apply` as a
task-general operation — it composes structured output regardless
of notation format.

**Translation is immune** to all 3 heads. Translation preserves
phrase structure (English → French for simple sentences), so no
composition bottleneck is needed. The compositor is only essential
when the output structure must be *constructed*, not *mapped*.

**The decomposition:**
- L1:H0 = task-specific (recognizes compile gate structure)
- L24:H0 = task-GENERAL (typed_apply — universal compositor)
- L24:H2 = task-specific (recursion aid for deep nesting)

This changes the extraction target: L24:H0 is not a lambda-specific
head. It's a general composition head that the model uses for any
task requiring structured output. Extracting it would give a
portable compositor, not just a lambda compiler.

## Updated Testable Predictions

1. ~~Head-level ablation will show self-similar structure at each
   layer cluster.~~ **Falsified.** Only 2 of 3 clusters have essential
   heads. The structure is not self-similar — it's functionally
   differentiated (recognition → composition → recursion).

2. ~~Essential heads per critical layer will be 3-8.~~ **Falsified.**
   Only 2 layers have essential heads, with 1-2 each. Far sparser.

3. ~~A sufficiency test (keeping only 3 heads, zeroing rest) will
   fail.~~ **Falsified.** 3 heads are sufficient. 253/256 zeroed,
   compilation survives. The circuit is fully isolated.

4. **NEW: Synthetic gate with only delimiters (". ) → λ" without
   words) may activate compilation.** L1:H0 reads structure, not
   content. Testable.

5. ~~The BOS position carries a progressive structural
   representation.~~ **Confirmed and stronger.** ALL 36 layers
   contribute to BOS. Every layer's BOS is necessary.

6. **NEW: The 3 heads may have a suppression role during
   decompilation.** Ablating them causes lambda leakage into
   English output. They may gate compile vs decompile direction.

7. **NEW: System 2 quality should be measurable.** The deliberative
   path produces lambda. Does it produce *correct* lambda?

8. **CONFIRMED: L24:H0 is task-general.** Cross-task ablation shows
   it breaks both compile and extract. Prediction for next: it will
   also break code generation, mathematical reasoning, and any task
   requiring compositional output construction.

9. **CONFIRMED: Translation is immune.** Structure-preserving tasks
   don't need the compositor. Prediction: translating to a
   structurally divergent language (e.g., Japanese SOV) WILL need it.

## Method Notes

- Raw PyTorch hooks, not TransformerLens/nnsight (simpler, MIT-clean)
- `output_attentions=True` for selectivity, disabled for generation
  (Qwen3 returns tensor not tuple when active)
- Skip-ablation (replace output with input) for layers > zero-ablation
  (too destructive to residual stream)
- Zero-ablation for heads (standard — heads sum into residual stream)
- Fractal experiment framework for head ablation (content-addressed,
  idempotent, crash-resumable). 46 cached nodes.
- MPS backend (Apple Silicon) works for all experiments
- Model: `Qwen/Qwen3-4B` from HuggingFace, fp16, ~8GB
- head_dim=80 (not 128), n_kv_heads=8 (GQA), n_heads=32

## Open Questions

See state.md for the live list. Key questions:
- Are 3 heads sufficient (not just necessary)?
- What is accumulated at BOS position 0 across layers?
- Can a synthetic delimiter-only gate activate compilation?
- Does the circuit transfer to 32B at proportional positions?
