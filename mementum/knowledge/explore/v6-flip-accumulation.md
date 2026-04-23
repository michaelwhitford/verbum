# v6 Flip Accumulation — Ternary Weight Learning

> status: active
> category: architecture
> tags: [v6, ternary, flip-accumulation, training-stability, MLX]
> related: [v6-design.md, VERBUM.md]
> depends-on: []

## Core mechanism

Ternary weights {-1, 0, +1} cannot learn through gradient descent.
They evolve through **flip accumulation**: gradient signals accumulate
in a buffer, and when consensus exceeds a threshold, the weight flips
one discrete step (-1→0→+1 or reverse).

```
λ flip(w, accum, threshold).
    accumulate: accum += sign(grad)     # direction vote per micro-batch
    gate:       |accum| > threshold     # enough consensus?
    flip:       w += sign(accum)        # one step in agreed direction
    clamp:      w ∈ {-1, 0, +1}        # stay ternary
    reset:      accum[flipped] = 0      # start fresh for flipped positions
```

## Three failures, three insights (session 028)

### Failure 1: Raw gradient accumulation → NaN

**What**: Accumulated raw gradient magnitudes (not signs). Accumulators
reached 10⁹ after 400 micro-batches. Threshold of 0.1 meant 100% of
weights flipped → catastrophic topology destruction.

**Why**: Gradient magnitude has no relationship to flip confidence.
A single large-gradient batch can overwhelm 399 small-gradient batches.

**Fix**: `accum += sign(grad)` — each micro-batch gets exactly one
vote (+1/-1). After N accumulations, |accum| ≤ N. Threshold is now
in units of "directional consensus."

### Failure 2: Missing gradient clipping → embedding divergence

**What**: v5 (PyTorch) uses `clip_grad_norm_(1.0)`. v6 (MLX) had none.
Embedding weight norm: 224 → 232 → 248 → NaN over ~400 steps.

**Why**: 5-pass architecture amplifies gradients. Tied embedding
weights (`logits = x @ embed.T`) create positive feedback: large
weights → large logits → large loss → large gradients → larger weights.

**Fix**: `optim.clip_grad_norm(grads, 1.0)` before optimizer step.

### Failure 3: Fixed threshold can't adapt → periodic collapse

**What**: Even with sign accumulation + grad clipping, the second
training run collapsed at step ~400. Gradient norms spiked to 13M
after a flip event.

**Why**: Fixed threshold doesn't account for training dynamics.
Early training: topology is far from optimal, many weights need to
flip, high flip rate is beneficial. But too many simultaneous flips
destabilize the continuous parameters (gamma, norms, gates), which
are calibrated for the old topology.

**Fix**: Adaptive percentile threshold with loss-based feedback.

## Adaptive percentile threshold

Instead of a fixed threshold, control the **flip rate** directly.

```python
# At flip time:
threshold = compute_flip_threshold(model, target_pct)  # percentile
n_flipped = apply_flips(model, threshold)

# 25 steps later, measure impact:
ratio = loss_after / loss_before
if ratio < 1.02:   target_pct *= 1.2   # flips helped → be aggressive
elif ratio > 1.10: target_pct *= 0.5   # flips hurt → back off
# Clamped to [0.01%, 2%]
```

**Properties**:
- Scale-invariant: works regardless of accumulator magnitude
- Self-correcting: asymmetric response (slow up, fast down)
- Closed-loop: the system finds its own topology learning rate
- Early training gets more flips (model tolerates changes easily)
- Late training gets fewer (topology refined, perturbations costly)

## Two-timescale dynamics

v6 training has two coupled learning processes:

| | Continuous (Adam) | Discrete (flips) |
|---|---|---|
| **What** | gamma, embeddings, norms, gates | ternary weight topology |
| **Rate** | every step | every 100 steps |
| **Bounded by** | grad clipping (‖g‖ ≤ 1.0) | adaptive target_pct |
| **Nature** | smooth optimization | periodic perturbation |

**Loss curve**: sawtooth with downward envelope. After each flip event,
loss spikes because continuous params are calibrated for old topology.
Recovery takes ~25-50 steps. Sawtooth amplitude should decrease as
topology stabilizes (flip rate decreasing = leading indicator).

## Key numbers (from 300-step verification)

| Step | Flips | % of weights | Threshold | Loss before → after |
|------|-------|-------------|-----------|-------------------|
| 100 | 73,851 | 0.21% | 228 | 11.08 → 11.03 (helped) |
| 200 | 195,135 | 0.55% | 226 | 10.99 → 11.09 (neutral) |
| 300 | 245,251 | 0.70% | 226 | 10.97 → TBD |

Threshold of 228 means 228/400 micro-batches (57%) agreed on direction.
This is genuine consensus, not noise.

## What to watch in training

1. **Flip rate trajectory**: should decrease as topology converges
2. **Adaptive target_pct**: self-tunes based on loss feedback
3. **Sparsity evolution**: does the model learn to prune (more zeros)?
4. **Gamma distribution**: per-channel scaling adapts around ternary routing
5. **Group-level flip patterns**: which layers (stride_stack, prep, s4)
   flip most? Do deeper strides stabilize first?

## Implementation

| File | What |
|------|------|
| `src/verbum/v6/ternary.py` | `accumulate_flips()` (sign-based), `apply_flips()`, `compute_flip_threshold()` |
| `scripts/v6/train.py` | Training loop with adaptive threshold + loss feedback |
| `scripts/v6/probe.py` | Reports flip stats, adaptive state, accumulator norms |
