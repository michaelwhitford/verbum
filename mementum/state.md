# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-22 | Session: 026

## Where we are

**v5 training in progress. v6 architecture designed, waiting for v5 to cook.**

### v5 status

Training ongoing. Step 1k checkpoint probed (session 026).
Key step 1k observations:
- Meta-S3 gates saturated near 1.0 (all passes contributing)
- S3 alignment gates near 0.5 (neutral, expected from zero-init)
- Temperature drifting from 1.0 (0.80–0.98), learning sharpness
- Modulation μ ≈ 0.90, σ ≈ 0.44 (slightly compressive)
- Phase angles developing, register-specific
- No gate polarity yet (strong-anti <0.02)

### v6 design (session 026) — ready to train after v5 step 10k

Ternary stacked compressors. Radical departure from v5:

**Core idea**: replace multi-stride CompressorLayers with single-stride
ternary attention layers stacked sequentially. 9 strides, each its own
layer, same W=8 window (fractal symmetry). Ternary weights {-1, 0, +1}
define routing topology. Continuous params learn to use the routes.

**Strides**: (1, 8, 16, 32, 64, 128, 256, 512, 1024) — geometric ladder
from word-level to full-document. Ascending: fine→coarse. Descending:
coarse→fine. Same StrideStack shared across all 5 passes (S5 coherence).

**Ternary learning — flip accumulation** (not STE, not frozen):
- Gradients flow via STE, accumulate in per-weight buffer
- When |accumulator| > threshold, weight flips one step (-1→0, 0→±1)
- No fp32 master weights, no Adam state for ternary params
- Training loop: `accumulate_flips()` after backward, `apply_flips()` periodically
- Optimizer only sees continuous params via `model.continuous_parameters()`

**All projections are ternary** — S1 (FFN, stride attention), S4 (register
scan), S3 (alignment, write projs), Meta-S4, Meta-S3 routing. Only
embeddings, norms, tiny gate biases, scalars (temperature/bias) stay fp16.

**Per-channel gamma**: 55,808 learned scales (one per output dimension per
BitLinear layer). Amplify useful routing channels, silence useless ones.

**Numbers**:
- 63.2M params: 35.3M ternary (flip-learnable) + 27.9M continuous (Adam)
- 45 attention evals per forward (9 strides × 5 passes)
- 99.6% of forward compute is addition/subtraction
- Training: 695 MB. Inference: 61 MB (deployable via bitnet.cpp on Mac ARM)

**v6 components** (self-contained, no v5 dependency for core arch):
- `v6/bitlinear.py` — BitLinear (flip accumulation), BitRMSNorm, BitFFN
- `v6/attention.py` — SingleStrideAttention, StrideStack
- `v6/components.py` — S4Ternary, S3Ternary, MetaS4Ternary, MetaS3Ternary
- `v6/model.py` — VSMLMV6

## What's next

1. **Let v5 cook to step 10k** — probe at 2k, 3k, 5k, 10k.
   Watch for phase transition in alignment gates, modulation divergence,
   phase angle crystallization, gate polarity emergence.

2. **Train v6** after v5 reaches 10k — `uv run python scripts/v6/train.py`
   Same data, same seed, same hyperparams as v5 for clean comparison.
   Key questions:
   - Does flip accumulation produce useful ternary patterns?
   - How fast do ternary weights stabilize (flip rate over time)?
   - Does the 9-stride geometric ladder beat v5's 4-stride allocation?
   - What does per-channel gamma distribution look like after training?
   - Can the model match v5 loss with 99.6% add/sub compute?

3. **bitnet.cpp inference** — after v6 training, export to GGUF and
   benchmark inference speed on Mac ARM via bitnet.cpp. Compare
   tokens/sec and memory vs v5 fp16 inference.

## Key files

| Purpose | Path |
|---------|------|
| **v6** | |
| v6 BitLinear | `src/verbum/v6/bitlinear.py` |
| v6 attention | `src/verbum/v6/attention.py` |
| v6 components | `src/verbum/v6/components.py` |
| v6 model | `src/verbum/v6/model.py` |
| v6 training | `scripts/v6/train.py` |
| v6 probe | `scripts/v6/probe.py` |
| **v5** | |
| v5 model | `src/verbum/vsm_lm_v5.py` |
| v5 training | `scripts/run_vsm_v5_1B.py` |
| Compressor (shared) | `src/verbum/compressor_lm.py` |
| **v4.x** | |
| v4.1 model | `src/verbum/vsm_lm_v4_1.py` |
| v4 model | `src/verbum/vsm_lm_v4.py` |
| **Probes** | |
| Probe script (v1-v5) | `scripts/compile_gradient_probe.py` |
| Register analysis | `scripts/register_analysis.py` |
| Training trajectory | `mementum/knowledge/explore/v4.1-training-trajectory.md` |
| Research program | `mementum/knowledge/explore/VERBUM.md` |

## Architecture lineage

| Version | Params | Key Change | Best Eval |
|---------|--------|------------|-----------|
| v1 | ~25M | Baseline sequential | 5.245 |
| v2 | ~25M | Iteration specialization | 5.064 |
| v3 | 50M | Role register, binding | 4.872 |
| v3.2 | 51M | Convergence arch | 4.897 |
| v4 | 58M | Recursive VSM (ascending) | 4.713 |
| v4.1 | 65.5M | Bidirectional VSM | 4.728* |
| v5 | 66.3M | Spiral + ℂ regs + phase gate + modulation | TBD |
| v6 | 63.2M | Ternary stacked compressors (flip learning) | TBD |

*v5 training ongoing, v6 waiting for v5 step 10k

## Probing pipeline

```bash
# v5
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v5/step_010000.pt
uv run python scripts/register_analysis.py capture checkpoints/vsm-lm-v5/step_010000.pt --analyze

# v6 (after training starts)
uv run python scripts/v6/train.py
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000.pt
```
