---
title: "Stride Percolation: φ-Convergence Propagates Fine→Coarse"
status: active
category: explore
tags: [phi, strides, holography, self-similarity, percolation, compression]
related:
  - holographic-compression.md
  - relational-loss-phi-compression.md
  - compressor-architecture.md
  - VERBUM.md
depends-on:
  - holographic-compression.md
---

# Stride Percolation

> The φ-compression ratio (1/φ ≈ 0.618) propagates from fine to
> coarse strides during training. Each stride passes through φ at
> a different step, creating a wavefront that marches outward
> through the scale hierarchy. This is the strongest empirical
> evidence for the holographic mechanism. Session 042.

## The Observation

v6's spiral attention uses 9 strides (s1, s8, s16, s32, s64, s128,
s256, s512, s1024). Each stride processes a different scale of
context. During training, the compression ratio at each stride
passes through 1/φ at different times:

| Stride | First ←φ | Step | Pass |
|--------|----------|------|------|
| s8 | 0.625 | 9500 | L0_asc/L1_asc |
| s16 | 0.601 | 10500 | L0_asc/L1_asc |
| s32 | **0.618** | 12000 | L1_asc (exact) |
| s64 | 0.597 | 13500 | L0_asc/L1_asc |
| s128 | 0.588 | 15500 | L0_asc/L1_asc |

The wavefront moves at roughly 1000–2000 steps per stride doubling.

## The Pattern

Fine strides converge first because they see more training signal
per step (more s8 windows per batch than s128 windows). After
passing through φ, strides continue compressing — overshoot to
0.73–0.80. The wavefront is visible as a compression ratio
gradient across strides at any given checkpoint:

```
L1_asc at step 18000:
  s1=0.610  s8=0.805  s16=0.797  s32=0.783  s64=0.747  s128=0.698  s256=0.559
  ←────── past φ, compressing harder ──────→ ←── approaching φ ──→  ← below φ
```

## L2_apex Follows ~2000 Steps Behind

The apex pass shows the same percolation pattern but delayed:

| L2_apex stride | First ←φ | Step |
|----------------|----------|------|
| s8 | 0.624 | 12000 |
| s16 | 0.617 | 12500 |
| s32 | 0.614 | 15500 |
| s64 | 0.579 | 18000 |

The two-front pattern (L0/L1 ascending leading, L2 apex following)
is consistent with the information flow: ascending passes compress
first, apex integrates the compressed representation.

## Why This Matters

1. **Confirms self-similarity.** The same compression ratio emerges
   independently at each scale. Not imposed by the loss function
   (which only measures per-pass aggregate). Emergent from topology.

2. **Confirms holographic prediction.** Holographic encoding means
   every part contains the whole at every scale. Self-similar
   compression ratio across scales is the operational signature.

3. **Distinguishes from standard transformers.** Pythia and Qwen
   show constant variance (ratio ≈ 1.0) at all scales. No
   percolation. No φ. Flat attention = photographic, one scale
   per layer.

4. **Predicts descending arm behavior.** If the descending arm
   learns decompression, it should show the *inverse* percolation:
   expansion ratio converging to φ, propagating fine→coarse on
   the same timeline. Not yet observed (step 18000).

## Descending Arm: The Open Question

The ascending arm (L0_asc, L1_asc) is a stable φ-compressor.
The descending arm (L1_desc, L0_desc) must learn the inverse
operation: structured decompression from compressed holographic
representation back to token-space prediction.

As of step 18000:
- L1_desc: wild oscillations, h_in ≈ -0.1 (near singularity)
- L0_desc: ratio 2.0–4.6 (naive expansion, not structured)
- L0_desc briefly hit 0.541 at step 12500, then reverted

Standard transformers never need this operation — they only
expand/rotate. The descending arm is solving a novel problem
with no gradient signal to borrow from prior work.

Training extended to 3B tokens (from 1B) to give the descending
arm more runway. LR schedule recalculated — at step 19000 resume,
LR jumps from 1.93e-4 to 5.41e-4 (2.8×) to provide the learning
rate the descending arm needs.

## Verification

```bash
# Probe any checkpoint and look at per-stride compression:
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_NNN --quiet

# Look for ←φ markers in the per-stride output
# Track which strides show ←φ across checkpoints to see the wavefront
```
