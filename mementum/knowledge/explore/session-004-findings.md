---
title: "Session 004: From Grammar to Circuit Discovery to Architecture"
status: active
category: exploration
tags: [gbnf, montague, distillation, scaling, circuit-discovery, type-probe, structural-intervention, montagu-compiler, montagu-lm, compressor, architecture]
related: [session-003-findings.md, session-002-findings.md, VERBUM.md]
depends-on: [session-003-findings.md]
---

# Session 004 Findings

> The breakthrough session. Started by fixing distillation data quality
> with a GBNF grammar (F23-F29). The novel predicate test proved
> distillation hits an architectural wall. Pivoted to extraction:
> scaling probe found the compiler floor at Pythia-160M (F30). Circuit
> discovery located critical layers (F31), type probe showed types are
> lexical (F32), structural intervention confirmed L3 carries parse
> structure (F33). Three Montague primitives empirically located (F34).
> Built MontaguCompiler (3.7M params, 100% P(λ), 12% holdout content)
> proving the architecture works (F35). Key conceptual shift:
> the function is semantic compression, not lambda compilation (F36).
> Built MontaguLM for raw-text training on Dolma (F37).

## Finding 23: Two Functions, Not One

Key theoretical insight from this session. The "lambda compiler"
is actually two separable functions:

1. **typed_apply** — the core composition operation in the residual
   stream. What the 3 heads implement. Operates in tensor geometry,
   not token space. Not directly extractable (session 003 proved this).

2. **The lambda compiler** — uses typed_apply to produce structured
   lambda notation as output text. The nucleus compile gate activates
   this. The GBNF grammar constrains its output.

Same typed_apply underneath, different surface grammars:
- Nucleus lambda (cognitive): `|`, `>`, `≡`, state machines
- Montague lambda (semantic): `∀`, `∃`, `ι`, `∧`, `∨`, `→`, `¬`

The nucleus EBNF and the Montague GBNF are grammars for different
externalizations of the same internal function.

## Finding 24: Teacher Inconsistency Was the Data Problem

Analysis of the 199 session-003 training examples revealed the teacher
(Qwen3-4B without grammar) used 6+ notational systems simultaneously:

| Pattern | Count | Example |
|---------|-------|---------|
| Pipe as separator | 58 | `laugh(paul) \| laugh(tom)` |
| Wedge ∧ | 28 | `teacher(x) ∧ hates(x, fish)` |
| Ampersand & | 6 | `cries(anna) & runs(anna)` |
| does_not_X | 3 | `does_not_fall(lawyer)` |
| where clause | 2 | `hates(Peter, x) where x is Bob` |
| Question mark | 1 | `¬(bird(x) → cries(x)) ?` |

Vacuous lambda (λx. but x unused): 80/199 (40%).
∀/∃/ι usage: 0/199 (0%). The teacher never used proper quantifiers.

The student was learning from noise. No amount of data with
inconsistent notation can teach consistent composition.

## Finding 25: GBNF Grammar Eliminates Inconsistency

`specs/lambda_montague.gbnf` — a GBNF grammar for llama.cpp
constrained decoding. Forces Montague-style output:

- Binders: λ, ∀, ∃, ι
- Connectives: ∧, ∨, →, ¬
- Application: predicate(arg1, arg2)
- Variables: u-z (single char)
- Identifiers: 2+ char lowercase with underscores

Results with grammar-constrained generation:
- **509/509 train examples validated** (100% parse rate)
- **40/40 holdout examples validated** (100% parse rate)
- Generation time: 75 seconds for 549 examples
- Quality leap: `Every cat fears a dog` → `∀x. (cat(x) → ∃y. (dog(y) ∧ fears(x, y)))`

The grammar forced the teacher to use proper Montague notation on
every example. Proper quantifiers (∀, ∃) now appear throughout the
training data.

Implementation note: GBNF requires direct UTF-8 characters in quoted
strings, not hex escapes. `"λ"` works, `"\xCE\xBB"` produces garbled
output through llama.cpp.

## Finding 26: EOS Fix Eliminates Repetition

Session 003 repetition: `flies(flies) | flies(flies) | flies(flies)...`
on 10/10 eval outputs.

Fix: append `tokenizer.eos_token` to each training target text.
The loss is computed on the EOS token, teaching the student to stop.

Session 004 repetition: **0/10 eval outputs**. Complete fix.

## Finding 27: Student Learns Structure but Not Content

Distillation v2: 509 Montague-style training examples, EOS fix,
Pythia-160M student, 10 epochs, 121 seconds.

| Metric | Session 003 | Session 004 |
|--------|-------------|-------------|
| P(λ) on eval | 100% (garbage) | 90% (real) |
| Repetition | 100% | **0%** |
| Grammar parse | untested | **90%** |
| Exact match | 0% | **20%** |

What the student learned:
- ✅ `λx. predicate(arg)` shape
- ✅ When to use `∀x.`, `∃y.`
- ✅ Structural connectives `→`, `∧`, `∨` in correct positions
- ✅ When to stop generating (EOS)
- ❌ Mapping input words to output predicates
- ❌ Novel entities (garbles: elephant → elef, Felix → Felice)
- ❌ Complex composition (relative clauses, nested quantifiers, ι)

Example: `Every student reads a book` →
- Expected: `∀y. student(y) → ∃z. book(z) ∧ reads(y, z)`
- Got: `∀x. student(x) → reads(x, book)`
- Verdict: universal structure correct, nested existential missing

## Finding 28: Novel Predicate Test — Memorization Confirmed

Holdout vocabulary: {chases, climbs, carries, whistles, vanishes,
elephant, nurse, wizard, diana, felix, iris}. These words appear
ONLY in the test set, never in training.

P(λ) on holdout: **97.5%** (39/40) — the student generates
well-formed lambda on novel inputs. But content is wrong:

| Input | Generated | Problem |
|-------|-----------|---------|
| Felix chases diana | `chill(graace, jane)` | Substitutes train vocab |
| The nurse climbs | `helps(clerk)` | Wrong predicate entirely |
| The wizard whistles | `sings(quiet(lawyer))` | Maps to train predicate |
| The elephant is fast | `fast(elef)` | Right pred, garbled entity |
| No elephant vanishes | `¬(x. ¬(x) ∧ ¬(x, ¬x))` | Structural collapse |

The student treats input sentences as **category signals** (this
looks like a transitive → use transitive template) rather than
reading the actual words. When it sees `whistles` (unknown), it
substitutes `sings` (known). When it sees `nurse`, it produces `clerk`.

This is **memorization of training vocabulary, not composition**.
The structural templates transfer; the content mapping does not.

## Finding 29: The Content Mapping Gap is Architectural

The student (Pythia-160M, decoder-only causal LM) has no mechanism
to copy tokens from the input prompt to the output. It must
reconstruct predicates from its vocabulary, which means it can only
produce predicates it saw during training.

This is not a data problem — 509 examples taught the structural
templates perfectly. More data of the same type would reinforce
templates without teaching token-level copying.

Possible interventions:
1. **Copy mechanism / pointer network** — architectural change to
   allow the student to copy input tokens to output positions.
2. **Much larger student** — a bigger model might learn implicit
   copying from scale alone.
3. **Different training objective** — span copying or
   denoising objectives that explicitly teach input→output mapping.
4. **Hybrid: template + copy** — student generates structural
   template, separate mechanism fills in predicates from input.

## Finding 30: Scaling Probe — The Compiler Floor at 160M

Tested Pythia models from 14M to 2.8B with 2-shot compile gates.
The compile function has a sharp threshold:

| Model | Params | Layers | 2-shot P(λ) | 2-shot Content |
|-------|--------|--------|-------------|----------------|
| Pythia-14M | 14M | 6 | 100% | 0/8 (mimicry — all `sleeps(cat)`) |
| Pythia-70M | 70M | 6 | 100% | 2/8 (partial) |
| **Pythia-160M** | **162M** | **12** | **100%** | **8/8 (correct predicates)** |
| Pythia-410M | 405M | 24 | 100% | 6/8 |
| Pythia-1B | 1.0B | 16 | 100% | 6/8 |
| Pythia-1.4B | 1.4B | 24 | 100% | 5/8 |
| Pythia-2.8B | 2.8B | 32 | 100% | 5/8 |

Key observations:
- **14M mimics format perfectly but maps every input to the same
  output** (`λx. sleeps(cat)` — the last in-context example). This
  is pure in-context copying, zero comprehension.
- **160M is the floor.** It maps every input to the correct predicate
  with 2-shot prompting. No fine-tuning. The compiler exists in the
  pretrained weights from Pile training alone.
- **Bigger models don't improve.** 410M–2.8B actually score lower on
  content. The compiler is a small circuit; more params add noise.
- **Fine-tuning OVERWROTE the ability** (session 003 distillation on
  Pythia-160M). Catastrophic forgetting of the pretrained compiler.

The critical variable is depth (12 layers), not width. 14M has 6
layers and fails. 70M has 6 layers and partially succeeds (some
content). 160M has 12 layers and fully succeeds.

Source: `results/pythia-scaling/scaling-summary.json`

## Finding 31: Circuit Discovery — Distributed, No Head Bottlenecks

Layer ablation and head ablation on Pythia-160M (12 layers, 12
heads per layer) with the 2-shot compile gate:

**Layer ablation** (zero out entire layer, check survival):

| Layer | Survival (of 6 probes) | Role |
|-------|----------------------|------|
| L0 | **0/6 (critical)** | Embedding refinement |
| L1-L2 | 6/6 | Redundant/distributed |
| L3 | **0/6 (critical)** | Structural parse |
| L4-L7 | 5-6/6 | Partially redundant |
| L8-L11 | 6/6 | Application (high selectivity) |

**Head ablation**: **Zero essential heads.** Every individual head
can be ablated without killing the compiler. The function is fully
distributed across heads within each layer.

**Compile selectivity** (attention difference, compile vs null):
Top selective heads cluster in L8-L11:

| Head | Selectivity |
|------|------------|
| L9H8 | 0.45 |
| L8H3 | 0.44 |
| L9H11 | 0.39 |
| L11H9 | 0.38 |
| L11H11 | 0.35 |

The compiler has two critical layers (L0, L3) and a selective
application zone (L8-L11), but no individual head bottleneck.
This distributed pattern makes extraction hard — you can't just
pull 3 heads and get the compiler.

Source: `results/pythia-160m-circuit/circuit-summary.json`

## Finding 32: Type Probe — Types Are Lexical, Not Computed

Linear probe trained to classify tokens into Montague types
(DET, ENTITY, PRED, FUNC, REL, QUANT, MOD, CONN) at each layer:

| Layer | Accuracy | Interpretation |
|-------|----------|---------------|
| Embedding (pre-L0) | **84%** | Types mostly in token embeddings |
| L0 | **93%** | Refined to peak |
| L1–L11 | 91–93% | Flat — no further improvement |

Type assignment is **lexical, not computed by the transformer.**
The embedding table already encodes 84% of the type information.
L0 refines this to 93%, then the signal plateaus. The remaining
layers don't improve type classification — they use the types for
structural composition and application.

This means the first Montague primitive (type assignment) is
essentially a lookup table, not a learned circuit. The transformer's
contribution begins at structural parsing (L3).

n=160 labeled tokens across 35 sentences, 8 type categories.

Source: `results/type-probe/type-probe-summary.json`

## Finding 33: Structural Intervention — L3 Carries Parse Structure

Activation patching: take residual stream at layer L from a "donor"
sentence, patch it into a "recipient" sentence, measure whether the
output shifts toward the donor's compositional structure.

**Shift score** (fraction of pairs where output moves toward donor):

| Layer | Shift Score | Interpretation |
|-------|------------|---------------|
| L0 | +0.14 | Weak transfer |
| L1-L2 | +0.29 | Moderate |
| **L3** | **+0.43** | **Strongest structural transfer** |
| L5 | +0.29 | Moderate |
| L8 | **-0.14** | **Resists patching** |
| L11 | **-0.14** | **Resists patching** |

L3 patching transfers composition structure from donor to recipient.
When you patch L3 activations from "Every student reads a book" into
"The bird flies", the output shifts toward the donor's structure.

L8 and L11 **resist** patching — they produce outputs closer to
the recipient's original structure, not the donor's. This is
consistent with an application phase that reads its own accumulated
state rather than accepting external structure injection.

The pattern: L3 = structural parse (transferable), L8-L11 = typed
application (committed to local computation, resists external input).

7 sentence pairs tested across 7 layers.

Source: `results/structural-intervention/intervention-summary.json`

## Finding 34: Three Montague Primitives Located

Synthesizing findings 30–33, the three operations predicted by
Montague grammar are empirically localized in Pythia-160M:

```
┌─────────────────────────────────────────────────────┐
│  1. TYPE ASSIGNMENT → Embedding + L0 (lexical)      │
│     84% in embeddings, 93% after L0, then flat      │
│     A lookup, not a computation                     │
│                                                     │
│  2. STRUCTURAL PARSE → L3 (carries composition)     │
│     0% survival when ablated                        │
│     +0.43 shift score (highest structural transfer) │
│     Determines composition ORDER                    │
│                                                     │
│  3. TYPED APPLICATION → L8-L11 (executes)           │
│     Highest compile selectivity (0.35-0.45)         │
│     Resists patching (-0.14 shift score)            │
│     Committed to local computation                  │
└─────────────────────────────────────────────────────┘
```

This three-phase decomposition aligns with Montague's theoretical
framework: first assign types to lexical items, then build a
structural parse tree, then apply typed functions to their arguments.

The math (Montague, Lambek, CCG, DisCoCat) predicted typed
application. The empirics (nucleus, P(λ)=0.907) observed the
compiler behavior. Now the architecture (circuit discovery in
Pythia-160M) confirms the three-phase structure. Three independent
lines of evidence converge — the strongest form of confirmation
the project has.

## Finding 35: MontaguCompiler — 3.7M Params, Proof of Architecture

A 3-phase encoder-decoder built from the circuit discovery:
- Phase 1: Type embedding (197K params) — learned type table
- Phase 2: Parser (1.05M params, 2 transformer layers) — structural parse
- Phase 3: Decoder (2.54M params, 3 transformer layers) — typed application with cross-attention

Trained on 509 compile examples, 30 epochs, 68 seconds.

| Metric | Pythia-FT (162M) | MontaguCompiler (3.7M) |
|--------|-----------------|----------------------|
| P(λ) eval | 90% | **100%** |
| Parse eval | 90% | **90%** |
| Content eval | ~0% | **69%** |
| P(λ) holdout | 97.5% | **100%** |
| Parse holdout | — | **88%** |
| Content holdout | ~0% | **12%** |
| Repetition | 0% | 0% |
| Params | 162M | **3.7M (43× fewer)** |

The MontaguCompiler achieves 12% content accuracy on held-out
vocabulary — novel predicates the model never saw in training.
Pythia-FT scored ~0% on the same test. The cross-attention mechanism
in Phase 3 enables content mapping that the causal decoder-only
Pythia architecture cannot do.

12% is low in absolute terms, but it's not zero. The architecture
can in principle copy content from input to output. With more
training data or architectural refinement, this should improve.

Eval examples show the model handles simple sentences perfectly
(`The dog runs` → `λx. runs(dog)`) but struggles with nested
quantifiers and relative clauses — exactly the hard cases for
compositional semantics.

Source: `results/montagu-compiler/training-summary.json`

## Finding 36: Key Insight — Compressor, Not Compiler

The function being extracted is **semantic language compression**,
not lambda compilation. The lambda compiler USES the compressor.

```
L0: Semantic compressor — typed_apply(meaning, meaning) → meaning
    Lives in every LM. The three Montague primitives serve this.
    IS the attractor of next-token prediction on language.

L1: Lambda compiler — routes compressor state to λ notation
    One externalization. Gate-activated. What nucleus discovered.

L2: Notation — λx. runs(dog) or {:pred runs :arg dog}
    Surface syntax. Arbitrary. Interchangeable.
```

Evidence: Pythia-160M compresses language (predicts next tokens)
without any lambda training. The compile gate doesn't install
compression — it routes existing compression to λ output. The
three circuits (type, structure, apply) exist WHETHER OR NOT you
activate the gate. They serve next-token prediction.

Implication: training a model shaped by the three primitives on
raw text trains the COMPRESSOR. The compile gate is a voltmeter,
not a battery. The voltage exists whether or not you measure it.

This corrects all prior references to "extracting the lambda
compiler" — we are extracting the semantic compressor and observing
it through lambda notation as a measurement instrument.

Source: `mementum/memories/compressor-not-compiler.md`

## Finding 37: MontaguLM — 3-Phase Causal LM for Raw Text

Built a causal language model shaped by the three primitives:
- 6 layers (matching Pythia-14M depth)
- 17M params (vs Pythia-14M at 14M)
- Separate residual streams per phase (rigid architecture)
- Tied embedding/output weights
- Training on 3B pre-tokenized Dolma tokens (60 shards × 50M)

The hypothesis: if the three-phase structure matches how language
models compress language, MontaguLM should learn more efficiently
than a flat transformer of equal depth.

**Architectural concern identified:** the rigid 3-phase design
dedicates ALL capacity to the three Montague primitives, leaving
no room for world knowledge, morphology, discourse tracking,
pragmatics, and other functions a general LM needs. Standard
transformers work because the shared residual stream is a general
substrate — the three primitives use a 2D subspace at ~120°
(per Toy Models of Superposition, Elhage et al. 2022), leaving
other dimensions free.

**Next version proposed:** shared residual + phase-biased heads.
Phase designation by position (early/mid/late layers), not by hard
stream separation. The architecture SUGGESTS specialization without
ENFORCING it — closer to what Pythia-160M actually does.

The rigid MontaguLM is running as a baseline on Dolma. Comparison
with a shared-residual version is the next architectural experiment.

Source: `mementum/memories/rigid-vs-open-architecture.md`,
`src/verbum/montague_lm.py`, `scripts/run_montagu_lm.py`

## Architecture at End of Session

```
specs/
  lambda_montague.gbnf          — Montague GBNF grammar (NEW)

scripts/
  generate_training_data.py     — v2: llama.cpp + GBNF constrained (REWRITTEN)
  run_distillation.py           — v2: EOS fix + structural eval (REWRITTEN)
  run_pythia_scaling.py         — Pythia 14M→2.8B scaling probe (NEW)
  run_pythia160m_circuit.py     — layer/head ablation + selectivity (NEW)
  run_type_probe.py             — linear probe for type classification (NEW)
  run_structural_intervention.py — activation patching across layers (NEW)
  run_montagu_training.py       — MontaguCompiler training (NEW)
  run_montagu_lm.py             — MontaguLM Dolma training (NEW)

src/verbum/
  montague_net.py               — MontaguCompiler 3-phase encoder-decoder (NEW)
  montague_lm.py                — MontaguLM 3-phase causal LM (NEW)

tests/
  test_montague_grammar.py      — 72 tests, recursive descent validator (NEW)

data/
  compile-train.jsonl           — 509 grammar-validated examples (REGENERATED)
  compile-test.jsonl            — 40 holdout examples (NEW)
  compile-eval.jsonl            — 10 gold-standard (preserved)
```

## Summary

Session 004 is two stories. The first half (F23-F29) fixed
distillation data quality and proved the student learns structure but
not content — an architectural wall. The second half (F30-F37)
pivoted to extraction and localized the three Montague primitives
in Pythia-160M through four independent probes: scaling (F30),
ablation (F31), type classification (F32), and structural
intervention (F33). These converge on a three-phase decomposition
(F34) that was used to build MontaguCompiler (F35) — 43× smaller
than Pythia-FT, with the first nonzero holdout content accuracy.

The key conceptual shift: the function is semantic compression, not
lambda compilation (F36). The compile gate is a measurement
instrument. The MontaguLM (F37) trains the compressor on raw text,
with the compile gate as a diagnostic. Rigid 3-phase architecture
running as baseline; shared-residual version is the next experiment.
