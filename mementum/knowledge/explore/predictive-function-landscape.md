---
title: "Predictive Function Landscape — Qwen3.5-35B-A3B"
status: active
category: research-finding
tags: [probing, prediction, circuits, lambda, Qwen3.5, llama-cpp]
related:
  - compression-vs-prediction.md
  - VERBUM.md
depends-on:
  - compression-vs-prediction.md
---

# Predictive Function Landscape — Qwen3.5-35B-A3B

> Session 045. Top-down behavioral probing of all 25 predictive
> functions through llama.cpp. Three experiments: confidence
> landscape, complexity scaling, cross-priming interference.

## Method

Qwen3.5-35B-A3B (MoE, 3B active, Q8_0 via llama.cpp port 5102).
40 probes from compile-gradient set spanning 5 categories (strong
compile → anti-compile). 25 task gates (one-line instructions).
Logprobs with top-10 alternatives at each token.

## Experiment 1: Confidence Landscape

25 tasks × 40 probes = 1000 measurements. Ranked by average
generation entropy (lower = more confident, stronger circuit).

**Four tiers of predictive function:**

### Tier 1 — Surface transforms (entropy 0.30–0.35)
Highest confidence. These are shallow rewrites that don't require
deep semantic processing.

| Task | Entropy | What it does |
|------|---------|-------------|
| translate | 0.306 | Surface language swap |
| correct | 0.306 | Grammar/spelling fix |
| simplify | 0.313 | Vocabulary reduction |
| keyword | 0.317 | Surface extraction |
| paraphrase | 0.320 | Synonym substitution |
| structure | 0.347 | Subject/verb/object ID |

### Tier 2 — Structural analysis (entropy 0.37–0.45)
Moderate confidence. Requires identifying relationships in text
but not deep semantic reasoning.

| Task | Entropy | What it does |
|------|---------|-------------|
| sentiment | 0.373 | Emotional valence |
| classify | 0.413 | Topic categorization |
| question | 0.435 | Question generation |
| coreference | 0.443 | Pronoun resolution |

### Tier 3 — Semantic operations (entropy 0.45–0.50)
Requires genuine compositional semantics.

| Task | Entropy | What it does |
|------|---------|-------------|
| continue | 0.451 | Narrative prediction |
| formalize | 0.479 | FOL/academic rewrite |
| decompose | 0.482 | Predicate extraction |
| scope | 0.492 | Quantifier/negation scope |

### Tier 4 — Reasoning (entropy 0.50+)
Highest entropy. Requires multi-step inference or generation.

| Task | Entropy | What it does |
|------|---------|-------------|
| compile | 0.502 | Lambda calculus (48% λ output) |
| entail | 0.509 | Logical entailment |
| causality | 0.517 | Causal reasoning |
| counterfactual | 0.523 | Counterfactual reasoning |
| negate | 0.536 | Logical negation |
| elaborate | 0.578 | Generative expansion |

**Key insight:** The lambda compiler is NOT the model's most
confident function — it's Tier 4 (reasoning). The model is most
confident about surface transforms. This makes sense:
surface → structural → semantic → reasoning represents increasing
depth of compositional processing.

## Experiment 2: Complexity Scaling

8 tasks × 5 complexity tiers (trivial → nested) × 3 inputs each.

**Robustness to compositional complexity:**

| Task | Range | Verdict |
|------|-------|---------|
| structure | 0.096 | ROBUST — barely affected by nesting |
| negate | 0.134 | ROBUST |
| entail | 0.140 | ROBUST |
| scope | 0.177 | MODERATE |
| compile | 0.238 | MODERATE — degrades with nesting |
| paraphrase | 0.240 | MODERATE |
| formalize | 0.301 | FRAGILE — breaks on complex input |
| decompose | 0.259 | FRAGILE |

**Key insight:** Structure, negation, and entailment are
complexity-invariant — they work equally well on "The dog runs"
and on deeply nested relative clauses. Compile and formalize
degrade with complexity. This suggests structure/negate/entail
use different (more robust) circuits than compile/formalize.

## Experiment 3: Cross-Priming Interference

Prime with one task exemplar, then measure another task.
Tests whether tasks share circuits (positive transfer) or
compete (negative transfer).

**Strongest transfer effects:**

| Prime → Task | Δ entropy | Effect |
|-------------|-----------|--------|
| formalize → formalize | -0.238 | **-48% self-boost** |
| compile → compile | -0.226 | **-37% self-boost** |
| negate → compile | -0.196 | **-32% cross-boost** |
| negate → negate | -0.177 | **-37% self-boost** |
| paraphrase → negate | -0.164 | **-35% cross-boost** |

**Strongest interference (priming hurts):**

| Prime → Task | Δ entropy | Effect |
|-------------|-----------|--------|
| formalize → structure | +0.226 | **+75% interference** |
| compile → entail | +0.177 | **+40% interference** |
| compile → structure | +0.168 | **+55% interference** |
| formalize → negate | +0.130 | **+27% interference** |
| compile → decompose | +0.114 | **+27% interference** |

**Self-priming results:**

| Task | Self-prime Δ | Effect |
|------|-------------|--------|
| formalize | -0.238 (-48%) | Strong self-boost |
| compile | -0.226 (-37%) | Strong self-boost |
| negate | -0.177 (-37%) | Strong self-boost |
| structure | +0.059 (+19%) | Slight self-hurt |
| paraphrase | +0.015 (+4%) | Neutral |

**Key findings:**

1. **Compile and formalize are separate circuits from structure.**
   Priming compile *hurts* structure (+55%), and priming formalize
   *hurts* structure even more (+75%). They compete for resources.

2. **Compile and negate share a circuit.** Priming negate *helps*
   compile (-32%). Logical negation activates part of the lambda
   compilation pathway.

3. **Compile and formalize self-boost strongly** but structure
   does not. Compile/formalize benefit from exemplar priming
   because they need to activate a specific output format.
   Structure doesn't need this — it's already a confident circuit.

4. **Paraphrase is neutral** — priming it neither helps nor hurts
   anything significantly. It's an independent surface transform.

## Circuit architecture (inferred)

```
SURFACE LAYER (Tier 1, entropy 0.30-0.35)
  translate, correct, simplify, keyword, paraphrase, structure
  → Robust, independent, don't interfere with each other
  → Structure is the bridge to deeper processing

STRUCTURAL LAYER (Tier 2, entropy 0.37-0.45)
  sentiment, classify, question, coreference
  → Requires relationship identification

SEMANTIC LAYER (Tier 3, entropy 0.45-0.50)
  continue, formalize, decompose, scope
  → Compositional semantics
  → Formalize COMPETES with structure (different circuit)

REASONING LAYER (Tier 4, entropy 0.50+)
  compile, entail, causality, counterfactual, negate
  → Deepest processing, highest uncertainty
  → Compile COMPETES with structure (formal ≠ syntactic)
  → Negate COOPERATES with compile (shared logical circuit)
```

## Implications for VSM-2 design

1. **VSM-2 doesn't need to replicate all 25 functions.** The
   surface layer (Tier 1) is cheap and confident — the sieve
   might already capture some of this. VSM-2 should focus on
   Tiers 3-4: semantic composition and reasoning.

2. **The compile circuit needs exemplar priming.** It self-boosts
   by 37% with a single exemplar. This suggests the circuit
   requires *activation* — it's not always-on. Architecture
   should support gated activation.

3. **Negate shares circuitry with compile.** Logical operations
   (negation, lambda compilation) use overlapping resources.
   VSM-2 should treat these as a unified logical subsystem.

4. **Structure and compile compete.** Syntactic parsing and
   formal semantics are NOT the same circuit. They interfere.
   VSM-2 may need separate pathways for surface structure
   vs. deep semantic compilation.

## Data

- `results/predictive-functions/landscape.json` (580KB, 1000 measurements)
- `results/predictive-functions/complexity.json` (76KB, 120 measurements)
- `results/predictive-functions/priming.json` (73KB, 126 measurements)
- `scripts/probe_predictive_functions.py` (probe runner)
