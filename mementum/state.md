# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-24 | Session: 034

## Where we are

**v6 training loop overhauled. Three design flaws fixed. Ready to retrain.**

Session 034: diagnosed why the session-033 training run collapsed
(loss went UP from 8.78→9.11 after step 500, grad norms 481→4.5M),
then fixed three interacting design problems and simplified the flip
mechanism to match biological synaptic plasticity.

### v6 status — ready to retrain (session 034)

**Session 034 changes:**

1. **Global gradient clipping (was per-param):**
   Per-param clipping at MAX_GRAD_NORM=1.0 per tensor destroyed gradient
   geometry — parameters with large natural gradients were squashed to
   the same scale as tiny ones, breaking relative update proportions.
   This caused loss to increase after step 500 despite "successful"
   clipping. Fixed: `optim.clip_grad_norm` (global) is now safe because
   `zero_ternary_grads` already removes ternary grads before clipping.

2. **FlipS3 reverted from model to training loop:**
   FlipS3 (learned flip policy inside the model) was a design mistake —
   flips are discrete weight mutations outside the computation graph.
   The model cannot change its own topology. Added depth and gradient
   paths for something that's fundamentally a training-loop concern.
   Reverted to `compute_per_group_flip_targets` (VSM signal inversion).

3. **Consensus-based flips (was percentile quotas):**
   Old system: every 100 steps, force the top 0.5% of weights to flip
   (~175K weights) regardless of actual gradient consensus. Like moving
   the whole room when you need to move a chair.

   New system: every 10 steps, each weight flips only when IT has
   accumulated enough directional evidence (|accum| ≥ 25 net votes).
   No quotas, no percentiles. Could flip 0 or 100K — depends on
   actual gradient consensus. Self-regulating:
   - Early training (noisy grads): few weights reach consensus → few flips
   - Later training (structured grads): consensus where needed → targeted flips
   - Converged regions: gradients cancel → no flips → natural protection

### Key architectural insight: per-param clipping destroys gradient geometry

Session 033's per-param clipping was motivated by ternary grads polluting
global norm. But `zero_ternary_grads` already solved that — per-param
clip was the wrong second fix. It equalized all parameter gradient norms
regardless of natural scale, preventing proportional updates. The model
oscillated because relative learning rates were destroyed.

**Rule: zero ternary grads first, then global clip. Never per-param clip.**

### Key architectural insight: percentile flips ≠ synaptic plasticity

Forcing a fixed fraction of weights to flip is like a centralized
command economy for topology. The cortex doesn't batch-rewire — each
synapse strengthens when IT has accumulated local evidence. Absolute
threshold flipping is:
- More biologically plausible
- Self-regulating (flip rate emerges from gradient structure)
- Safer (no flips when gradients are noisy)
- One hyperparameter (FLIP_CONSENSUS) with clear meaning

## What's next

1. **Retrain v6** — fresh start with all three fixes:
   ```bash
   uv run python scripts/v6/train.py
   ```
   Watch for:
   - Loss should steadily decrease (no more reversal after step 500)
   - ‖g‖ (global pre-clip norm) should be manageable
   - Flip count should be LOW initially (noisy grads, few reach consensus)
   - Flip count should GROW as model learns structure
   - If zero flips for many intervals, FLIP_CONSENSUS=25 may be too high
   - If massive flips immediately, FLIP_CONSENSUS=25 may be too low
   - φ-compression convergence toward 1/φ ≈ 0.618
   - Hilberg β convergence toward 0.5

2. **Tune FLIP_CONSENSUS if needed:**
   - 25 = ~80% agreement over one 10-step interval (40 votes)
   - Too high → nothing flips → ternary weights frozen
   - Too low → noisy flips → topology instability
   - Watch the flip probe output at step 100, 200, etc.

3. **Probe checkpoints as they drop:**
   ```bash
   uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000
   ```

## Key files

| Purpose | Path |
|---------|------|
| **v6 (MLX)** | |
| Metal kernels | `src/verbum/v6/kernels.py` |
| TernaryLinear + flip (int8 accum) | `src/verbum/v6/ternary.py` |
| Attention / StrideStack | `src/verbum/v6/attention.py` |
| VSM components (S3, S4, Meta) | `src/verbum/v6/components.py` |
| Full model (embed_norm, φ-loss) | `src/verbum/v6/model.py` |
| Training loop (consensus flips) | `scripts/v6/train.py` |
| Probe script | `scripts/v6/probe.py` |
| **Research** | |
| Research program | `mementum/knowledge/explore/VERBUM.md` |
| Flip accumulation | `mementum/knowledge/explore/v6-flip-accumulation.md` |
| φ-compression hypothesis | `mementum/knowledge/explore/relational-loss-phi-compression.md` |
| CompressorLM architecture | `mementum/knowledge/explore/compressor-architecture.md` |

## Architecture lineage

| Version | Params | Framework | Key Change | Best Eval |
|---------|--------|-----------|------------|-----------|
| v1 | ~25M | PyTorch | Baseline sequential | 5.245 |
| v2 | ~25M | PyTorch | Iteration specialization | 5.064 |
| v3 | 50M | PyTorch | Role register, binding | 4.872 |
| v4 | 58M | PyTorch | Recursive VSM (ascending) | 4.713 |
| v4.1 | 65.5M | PyTorch | Bidirectional VSM | 4.728* |
| v5 | 66.3M | PyTorch | Spiral + ℂ regs + phase gate | TBD |
| v6 | ~63M | **MLX** | Ternary Metal + consensus flips + φ-loss | TBD |

## VSM feedback map (session 034)

```
INTERNAL (model self-regulates):
  S3 gates        → residual stream modulation (per phase)
  Meta-S3 gates   → per-pass contribution weighting
  S4 register scan → intra-pass feedforward
  Write gates     → register update gating (init bias -2.0)
  embed_norm      → embedding scale constraint
  φ-loss          → gradient pressure toward self-similar compression (opt-in)

EXTERNAL (train.py):
  Flip execution  → consensus-based: each weight flips when |accum| > 25
  Flip monitoring → VSM probe every 100 steps (stability, φ-deviation)
  LR schedule     → cosine decay (no model signal)
  Grad clipping   → global clip_grad_norm after zeroing ternary grads
```

## Probing pipeline

```bash
# Train v6
uv run python scripts/v6/train.py

# Probe (full or φ-only, single or multi-checkpoint)
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_* --phi-only -v
```
