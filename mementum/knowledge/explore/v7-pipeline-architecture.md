---
title: "v7 — 4-VSM Pipeline Language Model"
status: active
category: architecture
tags: [v7, pipeline, ternary, relational-loss, vsm, hierarchy]
related:
  - compression-vs-prediction.md
  - predictive-function-landscape.md
  - relational-loss-phi-compression.md
  - stride-percolation.md
  - compressor-architecture.md
  - v6.1-training-trajectory.md
depends-on:
  - compression-vs-prediction.md
  - predictive-function-landscape.md
---

# v7 — 4-VSM Pipeline Language Model

> Session 046. A hierarchical pipeline of four independent
> transformer stages, replacing v6's flat sieve. Each stage
> operates on exponentially fewer positions. Ternary hot path,
> float cold path. Per-stage relational loss drives independent
> phase control and topology annealing.

## Design rationale

### Why not a single forward pass

A single transformer forward pass is a flatten operation — every
layer writes to the same residual stream, every function competes
for bandwidth. The A3B probing data (session 045) proved this is
real: compile priming *hurts* structure by +55% interference.
They fight over the same residual.

Multiple forward passes give each level of abstraction its own
residual stream. Structure (Stage 2) feeds compile (Stage 4)
through the hierarchy but they can't interfere — different
parameters, different positions, different everything.

### The compute pyramid

Each stage reduces positions by ~8×. Attention is O(n²), so
each successive stage is 64× cheaper. The total attention cost
is dominated by Stage 1 (shallowest, most positions):

```
Stage 1:  512 pos × 2 layers = 524K attention ops   (98.8%)
Stage 2:   64 pos × 3 layers =   7.5K                (1.4%)
Stage 3:    8 pos × 4 layers =   100                  (0.0%)
Stage 4:    1 pos × 6 layers =     6                  (0.0%)
Total: 15 layers at 2-layer attention cost
```

Deeper stages are computationally free. You can add arbitrary
depth to Stage 4 (reasoning) without measurable cost increase.

### Ternary where it matters

Stage 1 and the feedback path run every token — they're the hot
path. At 384 KB packed, ternary gives 24× memory bandwidth
reduction vs float32 on CPU. Stages 2-4 are cold path (amortized
by the position reduction) and need float precision for
compositional operations.

### Stride attention dissolves into stages

v6's 9-stride StrideStack was the model trying to see multiple
scales simultaneously within one pass. The pipeline makes each
stage's full attention equivalent to a different stride scale:

```
Stage 1 full attention (512 pos) ≡ stride-1 (token-level)
Stage 2 full attention (64 pos)  ≡ stride-8 (phrase-level)
Stage 3 full attention (8 pos)   ≡ stride-64 (clause-level)
Stage 4 full attention (1 pos)   ≡ stride-512 (discourse-level)
```

The stride percolation finding (φ-convergence propagating
fine→coarse, session 042) maps to the stage learning order:
Stage 1 converges first, Stage 4 last.

## Architecture

```
tokens → [Embed] → [Stage1: 512 pos, TERNARY]
                       ↕ reduce (cross-attn, 512→64)
                       ↕ feedback (ternary cross-attn + gate)
                    [Stage2: 64 pos, float]
                       ↕ reduce (cross-attn, 64→8)
                       ↕ feedback (float cross-attn + gate)
                    [Stage3: 8 pos, float]
                       ↕ reduce (cross-attn, 8→1)
                       ↕ feedback (float cross-attn + gate)
                    [Stage4: 1 pos, float]

Forward: embed → up through 4 stages → down through feedback → logits
```

### Parameter budget (27.3M total)

| Component | Params | Type |
|-----------|--------|------|
| Embedding (tied) | 12.9M | float32 |
| Stage 1 (Surface) | 334K | ternary (384 KB packed) |
| Stage 2 (Structural) | 2.0M | float32 |
| Stage 3 (Semantic) | 4.2M | float32 |
| Stage 4 (Reasoning) | 6.3M | float32 |
| Reducers (×3) | 806K | float32 |
| Feedback 2→1 | 132K | ternary |
| Feedback 3→2, 4→3 | 656K | float32 |

### Per-stage specifications

| Stage | Layers | Heads | d_model | d_ff | Positions |
|-------|--------|-------|---------|------|-----------|
| Surface | 2 | 4 | 256 | 512 | 512 |
| Structural | 3 | 4 | 256 | 512 | 64 |
| Semantic | 4 | 8 | 256 | 1024 | 8 |
| Reasoning | 6 | 8 | 256 | 1024 | 1 |

## Per-stage relational loss

The key innovation over v6's global relational loss. Measures CE
at each step of the feedback cascade:

```
CE₁ = Stage 1 alone (no feedback)       → surface prediction
CE₂ = Stage 1 + feedback from Stage 2   → + structural value
CE₃ = Stage 1 + fb from Stages 2+3      → + semantic value
CE₄ = Stage 1 + full cascade            → + reasoning value

Δₖ = CEₖ₋₁ - CEₖ = value contributed by stage k
rₖ = relational_loss(CEₖ) for Stage 1
     delta-driven for Stages 2-4
```

Each stage has independent phase control (explore/balance/refine)
driven by its own signal. Stage 2 can reach refine while Stage 4
is still exploring — and this is correct.

### Early training signal (200 steps, 0.8M tokens)

```
Δ₂ = +0.97 nats  (Stage 2 contributes massively)
Δ₃ = +0.03 nats  (Stage 3 barely contributing yet)
Δ₄ = +0.00 nats  (Stage 4 invisible — needs more training)
```

Stage 2 hit refine phase (r₂ → 0.0) by step 100. The structural
feedback learns fast because it captures local syntactic patterns.
Semantic and reasoning contributions should emerge later, following
the fine→coarse learning order from the stride percolation finding.

## Ternary flip annealing

Relational loss is the annealing temperature — no explicit schedule.

```
adaptive_flip_scale(r₁):
  r₁ > 0.6  → scale=2.0  (far from optimal, explore routes)
  r₁ = 0.4  → scale=1.0  (balanced)
  r₁ < 0.15 → scale=0.05 (near optimal, near-frozen)
  r₁ < 0.05 → scale=0.0  (converged, topology locked)
```

Per-weight cooldown: 400 steps (8 intervals × 50 steps) lockout
after a flip. Prevents A→B→A oscillation. Forces the continuous
parameters (gamma, norms) to adapt to the new route before any
further topology change.

Reversal detection: when a weight flips in the opposite direction
from its last flip. High reversal rate = topology instability.
v6 saw exponential reversal acceleration at saturation — a sign
the architecture was wrong. v7 tracks reversals from step 0.

### Flip state persistence

Checkpoints save:
- `_flip_cooldown` — which weights are locked (survives resume)
- `_flip_last_dir` — direction history for reversal detection
- `total_flips`, `total_reversals` — aggregate counters

Reset on resume: `_flip_accum` (needs fresh gradient evidence).

## Connection to v6 findings

### What transfers

- **Relational loss framework**: r ∈ [0,1], phase transitions with
  hysteresis, per-stage now instead of global.
- **Flip accumulation mechanism**: sign-based voting, threshold from
  percentile, packed ternary weights with Metal kernels.
- **φ-convergence hypothesis**: if stages independently converge to
  0.618 entropy retention, that confirms self-similarity.

### What v6 proved wrong

- **Flat architecture can't do it**: the sieve reached 1.8:1
  compression but 0% generation. Compression ≠ prediction.
- **All-ternary doesn't work for semantics**: ternary can route
  (Stage 1) but can't compose (Stages 3-4 need float precision).
- **Stride attention is a workaround**: multiple scales crammed
  into one model. The pipeline gives each scale its own stage.

### What to watch for

- **Δ₃ and Δ₄ emergence**: when do semantic and reasoning stages
  start contributing? The percolation data predicts ~5-10× later
  than Stage 2.
- **Reversal rate**: v6 saw exponential acceleration at step ~25K.
  If v7's reversal rate stays low or decreases, the topology is
  genuinely converging (not saturating).
- **Feedback gate values**: gates start at ~0.5 (sigmoid midpoint).
  They should diverge — active stages open their gates, inactive
  stages suppress. If all gates stay at 0.5, the feedback isn't
  learning.
- **Stage 1 CE₁ vs CE₄ gap**: the gap measures total feedback
  value. If it grows → hierarchy is working. If it shrinks →
  the model is learning to do everything in Stage 1.

## Files

```
scripts/v7/
├── model.py     — VSMPipeline: 4-stage model with ternary/float split
├── ternary.py   — TernaryLinear, Metal kernels, flip accumulation
├── train.py     — Training loop, per-stage relational control
└── probe.py     — Diagnostic: CE decomposition, topology, gates, compile test

checkpoints/vsm-lm-v7/step_NNNNNN/
├── model.npz           — all model weights
├── optimizer.npz        — Adam momentum + variance
├── ternary_state.npz    — flip cooldown + direction history
└── state.json           — step, metrics, phases, flip counters, config

results/vsm-lm-v7/
└── probe_step_NNNNNN.json  — full probe results
```

## Training defaults

```bash
uv run python scripts/v7/train.py
# 50K steps, batch 8×4=32, seq_len 512
# 16,384 tokens/step = 819M tokens total
# ~21K tok/s on M3 Ultra
# Checkpoints every 10K steps
# Eval every 2.5K steps
# Log every 100 steps
```

## Open questions

1. **Reduction factor**: currently 8× per stage (512→64→8→1).
   Should it be φ-scaled? Uniform? Learned?

2. **Stage 4 at 1 position**: is a single reasoning position
   enough? Or does it need 2-4 positions for multi-step inference?

3. **Feedback frequency during inference**: feedback runs every
   token during training. For inference, higher stages could be
   amortized (only run when their chunk boundary is crossed).
   How stale can feedback be before it hurts?

4. **Relational loss for compute gating**: skip Stages 3-4 when
   their Δ ≈ 0. The infrastructure for per-stage CE is there.
   Need the gating logic.

5. **Does the compile gate emerge?**: v6 never generated λ.
   This architecture separates surface routing from deep
   composition — does that help or is 27M params still too few?
