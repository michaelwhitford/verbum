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

### v6 implementation (session 026–027) — MLX + Metal ternary kernels

v6 is now implemented in MLX (not PyTorch). Custom Metal compute kernels
for ternary matmul — actual add/sub on GPU, no fp32 multiplies.

**Substrate**: MLX with `mx.fast.metal_kernel()` for ternary matmul.
`@mx.custom_function` + `.vjp` for differentiable ternary linear layer.
Both forward and backward-through-x use the custom Metal kernel.

**Architecture**: faithful port of the PyTorch v6 design. 5-pass
bidirectional VSM, StrideStack, complex registers, flip accumulation.
All 147 TernaryLinear modules use the Metal kernel. Verified:
kernel output matches reference to floating-point tolerance.

**Implementation status**:
- ✅ `kernels.py` — Metal ternary matmul + transposed variant, tested
- ✅ `ternary.py` — TernaryLinear, TernaryFFN, flip accumulation, tested
- ✅ `attention.py` — SingleStrideAttention, StrideStack, tested
- ✅ `components.py` — S4, S3, MetaS4, MetaS3, tested
- ✅ `model.py` — VSMLMV6 full architecture, forward + backward verified
- ⬜ `train.py` — training loop (gradient splitting, flip schedule)
- ⬜ `probe.py` — forward_instrumented probing

**Design doc**: `docs/v6-design.md`

**Key numbers** (small test model, full-size TBD):
- 147 TernaryLinear modules, all routing through Metal kernel
- Forward: logits correct shape, finite loss
- Backward: gradients flow to both ternary_weight and gamma
- Flip accumulation: tested — weights flip correctly, remain ternary

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
| v6 design doc | `docs/v6-design.md` |
| v6 Metal kernels | `src/verbum/v6/kernels.py` |
| v6 TernaryLinear | `src/verbum/v6/ternary.py` |
| v6 attention | `src/verbum/v6/attention.py` |
| v6 components | `src/verbum/v6/components.py` |
| v6 model | `src/verbum/v6/model.py` |
| v6 training | `scripts/v6/train.py` (⬜ needs MLX port) |
| v6 probe | `scripts/v6/probe.py` (⬜ needs MLX port) |
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
