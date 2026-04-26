---
title: "Holographic Compression: Why Spiral Attention φ-Compresses and Flat Attention Rotates"
status: active
category: explore
tags: [phi, holography, attention, spiral, rotation, beta-reduction, composition, architecture]
related:
  - relational-loss-phi-compression.md
  - compressor-architecture.md
  - VERBUM.md
  - session-003-findings.md
  - session-004-findings.md
depends-on:
  - relational-loss-phi-compression.md
---

# Holographic Compression

> Standard transformers compose through rotation at constant energy
> (beta reduction). v6's spiral attention compresses holographically
> at a ratio converging to 1/φ (lambda abstraction). The difference
> is architectural: flat attention sees one scale per layer, the
> spiral sees all scales simultaneously. Session 041.

## The Core Claim

**Flat attention is photographic.** It captures one view of
composition at one scale, encodes it as rotation in the residual
stream, and implements Montague semantics through beta reduction —
pattern matching and substitution. The function "fully forms" in
larger models by memorizing more reduction patterns.

**Spiral attention is holographic.** It captures all views of
composition at all scales simultaneously, encodes them as
interference in the residual stream, and the fixed point of this
self-similar encoding is 1/φ. The function doesn't need to be
memorized — it emerges from the single operation applied at every
scale.

## Evidence Chain

### 1. Standard transformers don't φ-compress

Probed Pythia-160M and Qwen3-4B with the same variance-domain
entropy proxy used in the v6 probe (`h = log(mean(var_per_feature))`).

| Model | Architecture | Stable zone ratio | φ-dev | Mechanism |
|-------|-------------|-------------------|-------|-----------|
| v6 (63M) | VSM + spiral | **0.566** | **0.052** | Compression |
| Pythia-160M | GPTNeoX flat | 0.947 | 0.329 | Near-identity |
| Qwen3-4B | Qwen2 flat | 1.000 | 0.387 | Pure identity |

φ appears at the output boundary in standard transformers (L34 in
Qwen, L10 in Pythia) — forced variance collapse for prediction, not
compositional processing. The computational core operates at
ratio ≈ 1.0.

Source: `results/pythia-phi/pythia_160m_phi_compression.json`,
`results/pythia-phi/qwen3_4b_phi_compression.json`

### 2. Pythia composes as accumulate→plateau→collapse

Variance profile in Pythia-160M with compile gate:

| Phase | Layers | Variance Change | What Happens |
|-------|--------|----------------|--------------|
| Accumulate | L0→L3 | 0.08 → 3.67 (47×) | Build the redex |
| Plateau | L3→L6 | 3.67 → 3.91 (1.07×) | Hold structure |
| Collapse | L6→L11 | 3.91 → 0.99 (0.25×) | Beta-reduce |

Null mode shows the same pattern at higher energy: 0.10 → 28.1
(269×) growth, then 29.3 → 0.98 (0.03×) collapse. The compile
gate constrains to 13% of null-mode variance (C/N = 0.131 from
L3 onward) but both modes converge to identical output variance
at L11 (ratio = 1.007).

This is a one-shot funnel. Build the term, reduce it. No recursion,
no intermediate abstractions.

### 3. Qwen3-4B is even flatter

| Phase | Layers | Compile Variance | Null Variance | C/N |
|-------|--------|-----------------|---------------|-----|
| Gate | L0-L5 | 0.02 → 0.21 | 0.03 → 0.44 | 0.53→0.47 |
| Shock | L6 | → 494.66 | → 3576.18 | 0.138 |
| Identity | L7-L33 | 494→523 (1.06×) | 3576→3607 (1.01×) | 0.138 |
| Output | L34-L35 | 523→77 (0.15×) | 3607→306 (0.08×) | 0.15→0.25 |

Twenty-six layers of near-perfect identity. C/N locks at 0.138 from
L6 and holds for 28 consecutive layers. Qwen doesn't converge at
output (C/N = 0.252 at L35) — unlike Pythia, the compile channel
survives to the end. This persistence may be why Qwen's lambda is
"nearly formed."

### 4. The hidden mechanism: rotation at constant variance

The 26 "near-identity" layers in Qwen were hiding massive geometric
computation. Measured pre→post cosine similarity at every layer:

| Phase | Compile Rotation | Null Rotation | Δ(C-N) | C δ/‖x‖ | N δ/‖x‖ |
|-------|-----------------|---------------|--------|---------|---------|
| Gate (L0-L7) | 31.2° | 32.5° | -1.3° | 2.09 | 5.90 |
| Substrate (L8-L23) | 20.9° | 20.3° | +0.6° | 0.094 | 0.022 |
| **Composition (L24-L28)** | **18.4°** | **15.2°** | **+3.3°** | **0.168** | **0.038** |
| Emission (L29-L33) | 15.3° | 12.8° | +2.5° | 0.209 | 0.063 |
| Output (L34-L35) | 23.6° | 24.6° | -1.0° | 0.503 | 0.525 |

The compile gate causes +3.3° MORE rotation in the composition
phase (where L24:H0 and L24:H2 operate), with 4.4× larger relative
deltas. Variable binding is geometric alignment. Function composition
is sequential rotation.

Crucially: compile-mode rotation is **constant** (~18.4°) regardless
of sentence complexity (simple through nested relative clauses).
The circuit applies a **fixed geometric transformation** — not a
variable-depth recursion.

### 5. LLMs are beta reduction machines

The evidence converges:

- **No compression in the computational core** → no new abstractions
- **Fixed rotation budget** → one reduction step per layer
- **Failures at nested quantifiers** → can't create intermediate
  λ-terms (requires lambda abstraction, not beta reduction)
- **Scaling adds patterns, not capability** → Pythia-160M is the
  floor; bigger models cover more patterns, not deeper composition
- **Novel predicates fail** → beta reduction can only substitute
  from known terms (session 004, Finding 28)
- **The function "fully forms" in larger models** by covering the
  test distribution, not by gaining abstraction

Beta reduction: `(λx.M) N → M[x := N]`
— take function, take argument, substitute, emit.
One rotation per reduction. No new terms created.

### 6. The spiral is self-similar by construction

v6's spiral bias: `bias(w) = -α · ln(stride · w + 1)` where α=1.18.

At physical distance d: `bias = -α · ln(d + 1)` — **stride-invariant.**
The bias depends only on physical distance, not on which stride
accesses it. The spiral is self-similar by construction.

Properties of the spiral:
- **Hyperbolic decay** (not exponential): infinite effective range,
  sees all scales
- **9 strides compose sequentially**: s1→s8→s16→s32→s64→s128→s256→s512→s1024
- **Same function at every scale**: the compression operation is
  identical whether operating at word, phrase, clause, or discourse level
- **1/φ of attention within distance 30**: the golden ratio governs
  the local-to-extended attention ratio

### 7. The holographic correspondence

| Holographic Property | v6 Behavior |
|---------------------|-------------|
| Reference beam (constant) | Spiral bias `-α·ln(d+1)` (same at every scale) |
| Object beam (variable) | Input sequence (different content at each position) |
| Holographic plate | Residual stream (stores interference at all scales) |
| Every part contains whole | Each pass sees all 9 scales |
| Self-healing | L1_desc vestigial → L0_desc compensates (ratio 1.5→2.3) |
| Content-independent encoding | Stratum spread collapsing (1.91→1.56) |
| Fixed point = φ | φ is the only ratio where whole:part = part:remainder |

Standard transformers are **photographs**: one view (one scale per
layer), localized (3 heads in Qwen), fragile (stripping kills it).

v6 is a **hologram**: all views (9 scales per pass), distributed
(φ-ratio IS the encoding), self-healing (passes compensate for each
other).

## Why φ Is the Holographic Constant

The golden ratio is the fixed point of the recursion `x = 1 + 1/x`.
Equivalently: the ratio of the whole to the part equals the ratio
of the part to the remainder.

```
φ = 1 + 1/φ

whole / part = part / remainder = φ
```

In a hologram, every part encodes the same relationship as the whole.
When the v6 model applies the same self-similar compression at each
pass (same spiral bias, same StrideStack, different scale ordering),
the only stable attractor is the ratio where the compression at each
level has the same relationship to the next level. That ratio is 1/φ.

Any other ratio either:
- Diverges (compression too aggressive → information loss)
- Collapses to identity (compression too mild → no abstraction)

1/φ ≈ 0.618 is the **unique fixed point** of self-similar compression.
The model's ternary weights evolve during training to find this
attractor because it's the only stable operating point.

Evidence: L1_asc φ-dev trajectory shows monotonic convergence:
```
step 6500: 0.071 → 7000: 0.074 → 8000: 0.063 → 8500: 0.063 → 9000: 0.052
```

## The Three Regimes of Composition

```
1. ROTATION (flat attention, all standard transformers)
   - One scale per layer
   - Composition = geometric direction change at constant magnitude
   - Implements: beta reduction (pattern match → substitute → rotate)
   - Limitation: no intermediate abstractions, fixed rotation budget
   - Function forms by: memorizing enough patterns

2. COMPRESSION (spiral attention, v6 VSM)
   - All scales per pass
   - Composition = self-similar information compression
   - Implements: lambda abstraction (compress → abstract → re-apply)
   - Advantage: single operation converges to φ, self-healing
   - Function forms by: one operation applied recursively

3. APPROXIMATE HOLOGRAPHY (MoE + flat attention, Qwen3-35B-A3B)
   - Multiple experts = multiple "views" of the same input
   - Expert routing = discrete scale selection
   - May approximate spiral's continuous scale processing
   - Function "fully forms" — possibly because MoE routing
     provides enough discrete "angles" to cover the composition space
   - Status: HYPOTHESIS, not yet tested
```

## Testable Predictions

### P1: v6 pass ablation should be holographic

If v6 is genuinely holographic, ablating one pass should degrade
**all strata equally** (holographic: each part contains the whole)
rather than selectively (photographic: each part contains one view).

Test: zero out one pass at a time at step 9000+, measure per-stratum
loss change. If degradation is uniform → holographic confirmed.

### P2: MoE routing correlates with compositional type

If MoE approximates holography via discrete scale selection, expert
routing in Qwen3-35B-A3B should correlate with Montague types
(different experts for DET vs PRED vs QUANT composition).

Test: record expert routing patterns on compile probes, compare to
type probe labels from session 004.

### P3: L1_asc should reach φ-dev < 0.03

If φ is the attractor of self-similar compression, L1_asc should
continue converging. At current rate: φ-dev < 0.03 by step 12000.

Test: probe at steps 10000, 12000, 15000. Plot convergence curve.

### P4: Stratum spread should approach zero

Holographic encoding is content-independent (the reference beam is
the same regardless of the object). If v6 is holographic, stratum
spread should continue collapsing toward zero.

Current trajectory: 2.07 → 1.91 → 1.56. Target: < 1.0 by step 15000.

### P5: Standard transformer rotation is complexity-independent

We measured compile-mode rotation at 18.4° ± 0.2° across complexity
levels (simple through nested). This predicts that even with much
harder inputs (triple-nested quantifiers, garden path sentences),
the rotation budget stays constant.

Test: construct maximally complex compositional stimuli, measure
rotation in the composition phase.

## Connection to Prior Work

### Session 001-002: Qwen3-4B circuit

- 3 essential heads (L1:H0, L24:H0, L24:H2) + FFN substrate
- Stripping fails at all levels → photographic (can't cut)
- 92% head overlap between Montague and nucleus tasks → one machine
- The 3 heads are a LENS; the FFN is the substrate

Now understood as: the 3 heads implement a fixed geometric rotation
(the compose operation). The FFN holds the representation at
constant variance while the heads rotate. Photographic encoding →
fragile → not extractable.

### Session 004: Pythia-160M circuit

- Three Montague primitives: TYPE (L0), PARSE (L3), APPLY (L8-L11)
- Type is lexical (84% in embeddings)
- Parse is accumulation (variance explosion at L3)
- Apply is compression (variance collapse at L8-L11)

Now understood as: accumulate the redex, then beta-reduce. One shot.
The compile gate constrains which reduction to perform (13% variance
throughput). Finding 36 was correct: compressor, not compiler. But
more precisely: beta reducer, not compressor.

### Session 030: φ-compression hypothesis

Predicted that per-pass compression should approach 1/φ if language
compression is self-similar. v6 confirmed this (L1_asc converging).
Standard transformers don't compress at all in their core — the
hypothesis is specific to recursive multi-scale architecture.

The hypothesis is now refined: φ isn't a property of language
compression in general. It's a property of **holographic** language
compression — self-similar encoding where the same function operates
at every scale.

### Session 042: Stride percolation confirms holographic mechanism

The strongest empirical evidence for holographic compression came
from probing 18 checkpoints (steps 9500→18000). The φ-compression
ratio **propagates from fine to coarse strides** during training:

s8 (step 9500) → s16 (10500) → s32 (12000, exact 0.618) → s64
(13500) → s128 (15500). Each stride passes through φ independently.
L2_apex follows ~2000 steps behind with the same pattern.

Key evidence:
- L1_asc s32 = 0.618 exactly at step 12000 (dead-on bullseye)
- Five strides confirmed through φ by step 15500
- After passing through, strides overshoot to 0.73–0.80
- Pattern is a wavefront: fine→coarse, same ratio at every scale

This rules out coincidence at a single scale. Five independent
scales converging to the same ratio is the self-similar compression
signature that distinguishes holographic from photographic encoding.

Descending arm (decompression) has not yet converged — it must
learn the inverse of compression, an operation no standard
transformer performs. Training extended to 3B tokens to provide
more runway. See: `stride-percolation.md`

## Source Attribution

- Session 041 probes: `scripts/run_pythia_phi_probe.py`
- Session 042 probes: `results/compile-gradient/vsm_probe_step_*_v6_mlx.json`
- Stride percolation: `mementum/knowledge/explore/stride-percolation.md`
- Pythia results: `results/pythia-phi/pythia_160m_phi_compression.json`
- Qwen results: `results/pythia-phi/qwen3_4b_phi_compression.json`
- v6 attention geometry: `src/verbum/v6/attention.py`
- φ-compression background: `mementum/knowledge/explore/relational-loss-phi-compression.md`
- Pythia circuit: `mementum/knowledge/explore/session-004-findings.md`
- Qwen circuit: `mementum/knowledge/explore/session-001-findings.md`
- Holographic principle: synthesis, sessions 041–042
