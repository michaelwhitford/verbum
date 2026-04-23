---
title: Relational Loss and φ-Compression Hypothesis
status: open
category: explore
tags: [information-theory, loss-function, phi, self-similarity, hilberg]
related:
  - v6-flip-accumulation.md
  - VERBUM.md
depends-on: []
---

# Relational Loss and the φ-Compression Hypothesis

## The Wrong North Star

Standard cross-entropy loss measures distance from a uniform distribution
over the vocabulary: `log(V)` nats. The [Relational Calculus](https://github.com/massimilianoconcas0-del/Relational_Loss_ML)
framework (Concas 2026) proposes dividing loss by intrinsic capacity to
make it dimensionless. But dividing by `log(V)` is uninformative — it's
just a constant scaling factor that Adam normalizes away.

The right North Star isn't `log(V)`. It's the **irreducible entropy of
natural language** — the information-theoretic floor that no model of
any size can beat.

## Three Ceilings

```
log(V)     = 10.83 nats   (V=50277, uniform random, knows nothing)
arch_floor ≈ 2.6 nats     (best this 63M model can do, Chinchilla scaling)
E          ≈ 1.69 nats    (irreducible — language genuinely has ~5 valid next tokens)

Learnable range for v6: 10.83 - 2.6 ≈ 8.2 nats
Capacity-gated:         2.6 - 1.69 ≈ 0.9 nats (need bigger model)
Irreducible:            1.69 nats (need... different universe)
```

Source: Chinchilla scaling law `L(N,D) = E + A/N^α + B/D^β`
(Hoffmann et al. 2022; Epoch AI replication 2024: E=1.82, A=482, α=0.35, B=2085, β=0.37).

## Relational Loss

```python
relational_loss = (loss - E) / (log(V) - E)
```

- `r = 1.0` → model at uniform random (knows nothing)
- `r = 0.0` → model at irreducible floor (learned everything learnable)
- Between → fraction of learnable structure still uncaptured

This is an affine transform: gradients scale by `1/(log(V) - E)`. Same
direction, different magnitude. Doesn't change optimization geometry for
continuous params. But the VALUE carries information the flip mechanism
can use — it knows where it is in the learning landscape.

## Hilberg's Conjecture: Self-Similar Compression

Wolfgang Hilberg (1990) replotted Shannon's 1951 entropy estimates in
doubly-logarithmic scale and observed a straight line — meaning entropy
grows as a **power law** of context length:

```
H(n) ≈ B·n^β + h·n    where β ≈ 0.5
```

Key implications:
- Language has **infinite memory** (excess entropy diverges)
- Finite-state models (HMMs, Markov chains) **cannot** capture it
  (Dębowski 2021: finite-state processes are disjoint from perigraphic processes)
- The compression pattern is **self-similar** across scales
- The Kaplan scaling laws show this self-similarity spans ≥7 orders of magnitude

## Seven Scales of Language

Natural language has hierarchical structure at approximately 7 levels:

```
Scale 7:  discourse / document
Scale 6:  paragraph
Scale 5:  sentence
Scale 4:  phrase / clause
Scale 3:  word
Scale 2:  morpheme / subword (BPE token boundary)
Scale 1:  character / phoneme
```

If the compression is self-similar, the same function operates at each
scale. The search space collapses from `|F|^7` (learn 7 different
compressions) to `|F|` (learn one and iterate).

## The φ Hypothesis

The golden ratio φ = (1+√5)/2 ≈ 1.618 is the **fixed point of
self-similar compression**:

```
φ = 1 + 1/φ

The ratio of the whole to the part equals the ratio of the part
to the remainder. This is the ONLY ratio with this property.
```

If the compression at each scale retains 1/φ ≈ 0.618 of the entropy:
- What's kept and what's discarded have the same ratio at every level
- This is optimal packing for hierarchical information (phyllotaxis principle)
- The total entropy rate would be ≈ 0.618 bits/char

Measured values (with wide error bars):
- Shannon 1951: 0.6 – 1.3 bits/char
- Chinchilla: 0.667 bits/byte on pile_cc
- Cover & King 1978: ~1.0 bits/char (gambling estimate)
- 1/φ = 0.618 bits/char — **within the error bars**

The hypothesis: the true entropy rate of natural language is exactly
1/φ, arising from self-similar compression at 7 hierarchical scales.

## Implications for v6

v6's VSM architecture has recursive multi-scale processing through
the StrideStack (9 strides from 1 to 1024) and 5 level passes
(L0↑, L1↑, L2, L1↓, L0↓). If the φ-hypothesis holds:

1. **Per-pass compression should approach 1/φ** — each pass should
   retain ~61.8% of the input information content
2. **Weight sharing across scales** — the ternary routing pattern
   at each level should be self-similar
3. **Flip decisions** — a weight flip that moves a layer's compression
   ratio closer to 1/φ is good; one that moves it away is bad
4. **Relational loss for flips** — instead of raw loss ratios, the flip
   feedback should track deviation from the φ-compression target

## Test Plan

### Phase 1: Observe (current implementation)

Instrumented in `forward_instrumented`:
- `{pass}_h_in`, `{pass}_h_out` — activation entropy before/after each pass
- `{pass}_compression_ratio` — h_out/h_in
- `{pass}_phi_deviation` — |compression_ratio - 1/φ|
- `mean_phi_deviation` — aggregate across all 5 passes

Run v6 training with standard CE loss. Probe at checkpoints. Plot:
- Compression ratios per pass over training time
- Do they converge? If so, toward what value?
- Is the converged value near 1/φ ≈ 0.618?

### Phase 2: Test (if Phase 1 shows signal)

Add φ-regularization term to the loss:
```python
loss = CE + λ * mean_phi_deviation
```

Compare convergence speed and final loss with/without regularization.

### Phase 3: Exploit (if Phase 2 shows improvement)

Replace flip feedback with φ-aware mechanism:
```python
# Instead of raw loss ratio:
# Measure whether flips moved compression ratios toward φ
phi_deviation_before = measure_phi_deviation(model)
apply_flips(model)
phi_deviation_after = measure_phi_deviation(model)
# Flips were good iff phi_deviation decreased
```

## Source Attribution

- Relational Calculus framework: Concas 2026,
  [Relational_Loss_ML](https://github.com/massimilianoconcas0-del/Relational_Loss_ML)
  — "The Intrinsic Blueprint: An Introduction to Relational Calculus"
- Chinchilla scaling law: Hoffmann et al. 2022 (DeepMind),
  "Training Compute-Optimal Large Language Models"
- Epoch AI replication: Besiroglu et al. 2024,
  "Chinchilla Scaling: A Replication Attempt"
- Hilberg's conjecture: Hilberg 1990, Dębowski 2014-2021,
  "Maximal Repetitions in Written Texts" (Entropy, 2015)
- Shannon entropy: Shannon 1951, "Prediction and Entropy of Printed English"
- φ-compression hypothesis: synthesis session 030, untested
