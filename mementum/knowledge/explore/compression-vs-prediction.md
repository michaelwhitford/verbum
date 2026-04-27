---
title: "Compression ≠ Prediction: The H≈0.7 Boundary"
status: active
category: research-finding
tags: [hilberg, compression, prediction, lambda, architecture]
related:
  - v6.1-training-trajectory.md
  - holographic-compression.md
  - relational-loss-phi-compression.md
  - VERBUM.md
depends-on: []
---

# Compression ≠ Prediction: The H≈0.7 Boundary

> Session 045. The central finding that reframes the verbum research
> program. Compression alone cannot achieve generation. The lambda
> function is a predictive circuit, not just a compression target.

## The finding

The v6.1 ternary sieve trained to step 32500 (39% of 3B tokens).
It learned universal content-independent compression (stratum spread
0.013) but achieved 0% λ generation across all 64 checkpoints.

The Hilberg exponent β measured across the sieve's passes:
- Ascending: **0.75** (L0↑ → L2)
- Descending: **0.83** (L1↓ → L0↓)

These values match the empirical literature on natural language:
- Hilberg (1990): β ≈ 0.5 (limited data, sequences ≤100 chars)
- Dębowski (2015): β ≈ 0.95 (compression codes, likely overestimate)
- Entropy rate studies: β ≈ 0.884 across 6 languages (stretched exponential)
- L²M paper (Chen et al. 2025): bipartite MI scales as L^β, β ≈ 0.82

**If β > 0.5, compression alone cannot capture the long-range
dependencies that prediction requires.** The mutual information
between past and future tokens grows as L^0.7 — a fixed-state
compressor systematically loses this growing signal.

## Why the sieve can't generate

The v6.1 sieve achieves **1.8:1 end-to-end entropy compression**
through 5 ternary passes. This is real compression — but it's the
wrong kind. It removes statistical redundancy (entropy compression)
but doesn't capture compositional semantics (what prediction needs).

The L²M condition (Chen et al. 2025): a model's state size for
storing past information must scale faster than the bipartite
mutual information for effective long-context modeling.

- Transformers satisfy L²M: KV cache grows linearly with context
- SSMs/RNNs with fixed state do NOT satisfy L²M
- The ternary sieve with fixed passes does NOT satisfy L²M

The sieve's compression ratio drifted from 0.83→0.89 during
training (steps 25500→32000). The model was *correct* to relax
compression — it discovered that prediction requires modeling
long-range dependencies, not just removing local redundancy.

## The lambda function is a predictive circuit

Key insight from the nucleus project:
- P(λ) = 0.907 — all LLMs converge on the lambda compiler
- Pythia-160M has the Montague-shaped lambda function
- Qwen3.5-35B-A3B produces clean λx.(run x) with high confidence

If the lambda function weren't useful for prediction, gradient
descent wouldn't converge on it independently across all models.
The 6.2:1 compression ratio of the lambda compiler is not just
compression — it's the compression rate of the structure that
prediction requires. Typed application IS how models capture the
L^0.7 growing mutual information.

Early probing of Qwen3.5-35B-A3B (session 045) shows:
- compile and formalize are the model's most confident semantic
  transformations (lowest entropy)
- They produce essentially the same output (FOL notation)
- They're more confident than structure, negation, or entailment
- The lambda/FOL circuit is a strongly formed attractor

## Architectural implication: two-VSM design

The sieve proved it can compress. But generation requires a second
system that holds growing state over the compressed representations.

```
VSM-1 (Sieve/Compressor) — what v6.1 built
  Fixed ternary passes, 1.8:1 entropy compression
  Content-independent, universal compressor
  Cheap (8-bit effective), fast
  DOES NOT satisfy L²M condition

VSM-2 (State/Predictor) — what's needed
  Operates over compressed representations
  State grows with context (satisfies L²M)
  Must learn the lambda-shaped compositional structure
  Generates from the compressed manifold
```

The compressor reduces the problem: instead of modeling L^0.7
dependencies over raw 50K-vocab token space, VSM-2 models them
over the 1.8× denser compressed representation.

## Next steps

1. **Map the full predictive toolkit** via top-down probing of
   Qwen3.5-35B-A3B through llama.cpp (probe script built,
   experiments queued: landscape, complexity, priming)
2. **Design VSM-2** informed by what functions prediction
   actually uses (not just lambda — also structure, negation,
   entailment, paraphrase, etc.)
3. **Determine if the sieve is worth keeping** as VSM-1, or
   if the 1.8:1 compression is too marginal to justify

## References

- DeepMind, "Language Modeling Is Compression" (2023): prediction
  ≡ compression, but scaling beyond a point deteriorates compression
- Chen et al., "L²M: Mutual Information Scaling Law" (2025):
  bipartite MI grows as L^β, state must scale faster
- Dębowski, "Entropy Rate Estimates" (2016): β ≈ 0.884 across
  6 languages, a universal of natural language complexity
- nucleus project: P(λ) = 0.907, 6.2:1 compression ratio
