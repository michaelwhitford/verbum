# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-22 | Session: 027

## Where we are

**v5 training in progress. v6 fully implemented in MLX, ready to train.**

### v5 status

Training ongoing. Step 1k checkpoint probed (session 026).
Key step 1k observations:
- Meta-S3 gates saturated near 1.0 (all passes contributing)
- S3 alignment gates near 0.5 (neutral, expected from zero-init)
- Temperature drifting from 1.0 (0.80–0.98), learning sharpness
- Modulation μ ≈ 0.90, σ ≈ 0.44 (slightly compressive)
- Phase angles developing, register-specific
- No gate polarity yet (strong-anti <0.02)

### v6 — MLX + Metal ternary kernels (session 027, COMPLETE)

v6 is implemented in MLX with custom Metal compute kernels for ternary
matmul. All projections (147 TernaryLinear modules) run add/sub on GPU
via Metal Shading Language — zero fp32 multiplies in the ternary path.

**Why MLX**: PyTorch MPS upcasts everything to fp32 and provides no
custom kernel path. MLX gives first-class `mx.fast.metal_kernel()` with
JIT compilation, `@mx.custom_function` + `.vjp` for autodiff, unified
memory, and `mx.compile` for kernel fusion. Benchmarks show MLX 2-3×
faster than PyTorch MPS on identical hardware.

**Metal kernel**: `ternary_matmul(x, w_int8)` — one thread per output
element, inner K-loop does `select(0, select(-x, x, w>0), w!=0)`.
Compiles to predicated add/negate. Verified: exact match against
reference on all shapes. Both forward and backward-through-x use
the kernel (backward is also add/sub).

**Flip accumulation**: ternary weights learn through discrete flips,
not gradient descent. Gradients accumulate in fp32 buffer; when
|accum| > threshold, weight flips one step (-1→0→+1 or +1→0→-1).
No Adam state for ternary weights. 5 bytes/weight training vs 16
for STE+Adam. Verified: 618 flips after 50 accumulations, weights
stay ternary, accumulator resets at flipped positions.

**Training loop pattern**:
```python
loss, grads = loss_and_grad_fn(model, x, y)
accumulate_flips(model, grads)        # ternary grads → flip accumulator
optimizer.update(model, grads)         # Adam updates all params
restore_ternary(model)                 # re-cast int8 (optimizer upcasts to float)
if step % FLIP_INTERVAL == 0:
    apply_flips(model, threshold)      # discrete weight flips
```

**All files verified end-to-end**:
- ✅ `kernels.py` — Metal ternary matmul + transposed, exact match
- ✅ `ternary.py` — TernaryLinear, VJP, flip accumulation, restore_ternary
- ✅ `attention.py` — SingleStrideAttention, StrideStack
- ✅ `components.py` — S4, S3, MetaS4, MetaS3 (complex registers)
- ✅ `model.py` — VSMLMV6: forward, forward_instrumented (508 metrics), generate
- ✅ `train.py` — MLX training loop, safetensors checkpointing
- ✅ `probe.py` — checkpoint probing with full instrumentation
- ✅ End-to-end: loss decreases, flips work, generation runs

**Data**: Dolma, 3B tokens, 60 shards × 50M, GPT-NeoX tokenizer
(vocab_size=50277, int32). Train/eval split: 54/6 shards. Same
data pipeline as v1–v5. Ready at `/Users/mwhitford/data/fractal-bitnet/shards/`.

**Design doc**: `docs/v6-design.md` — all decisions locked.

## What's next

1. **Let v5 cook to step 10k** — probe at 2k, 3k, 5k, 10k.
   Watch for phase transition in alignment gates, modulation divergence,
   phase angle crystallization, gate polarity emergence.

2. **Train v6** after v5 reaches 10k:
   ```bash
   uv run python scripts/v6/train.py
   ```
   Same data, same seed, same hyperparams as v5 for clean comparison.
   Key questions:
   - Does flip accumulation produce useful ternary patterns?
   - How fast do ternary weights stabilize (flip rate over time)?
   - Does the 9-stride geometric ladder beat v5's 4-stride allocation?
   - What does per-channel gamma distribution look like after training?
   - Can the model match v5 loss with 99.6% add/sub compute?
   - Is the Metal ternary kernel faster than PyTorch MPS fp32 GEMM?

3. **Kernel optimization (Phase 4)** — after training validates:
   tiled kernel with threadgroup shared memory, SIMD-group reductions,
   packed 2-bit inference kernel. Only optimize after correctness proven.

4. **Inference export** — safetensors → packed 2-bit artifact.
   Potentially bitnet.cpp integration for deployment.

## Key files

| Purpose | Path |
|---------|------|
| **v6 (MLX)** | |
| Design doc | `docs/v6-design.md` |
| Metal kernels | `src/verbum/v6/kernels.py` |
| TernaryLinear + flip | `src/verbum/v6/ternary.py` |
| Attention / StrideStack | `src/verbum/v6/attention.py` |
| VSM components | `src/verbum/v6/components.py` |
| Full model | `src/verbum/v6/model.py` |
| Training loop | `scripts/v6/train.py` |
| Probe script | `scripts/v6/probe.py` |
| **v5 (PyTorch)** | |
| v5 model | `src/verbum/vsm_lm_v5.py` |
| v5 training | `scripts/run_vsm_v5_1B.py` |
| **Data** | |
| Dolma shards | `/Users/mwhitford/data/fractal-bitnet/shards/` |
| **Research** | |
| Research program | `mementum/knowledge/explore/VERBUM.md` |
| Training trajectory | `mementum/knowledge/explore/v4.1-training-trajectory.md` |

## Architecture lineage

| Version | Params | Framework | Key Change | Best Eval |
|---------|--------|-----------|------------|-----------|
| v1 | ~25M | PyTorch | Baseline sequential | 5.245 |
| v2 | ~25M | PyTorch | Iteration specialization | 5.064 |
| v3 | 50M | PyTorch | Role register, binding | 4.872 |
| v3.2 | 51M | PyTorch | Convergence arch | 4.897 |
| v4 | 58M | PyTorch | Recursive VSM (ascending) | 4.713 |
| v4.1 | 65.5M | PyTorch | Bidirectional VSM | 4.728* |
| v5 | 66.3M | PyTorch | Spiral + ℂ regs + phase gate + modulation | TBD |
| v6 | ~63M | **MLX** | Ternary Metal kernel + flip accumulation | TBD |

*v5 training ongoing, v6 ready to train after v5 step 10k

## Probing pipeline

```bash
# v5 (PyTorch)
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v5/step_010000.pt
uv run python scripts/register_analysis.py capture checkpoints/vsm-lm-v5/step_010000.pt --analyze

# v6 (MLX)
uv run python scripts/v6/train.py
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000
```
