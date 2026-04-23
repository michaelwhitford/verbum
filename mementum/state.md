# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-23 | Session: 028

## Where we are

**v6 restarting with sign-based flip accumulation + adaptive threshold.**

All prior v6 checkpoints invalid (NaN). Three bugs found and fixed
in session 028. Training restarting fresh.

### v5 status

Stopped at step 5k. Checkpoints at steps 1k–5k (PyTorch).

### v6 status — ready to train (session 028)

Three training attempts, three failures, three fixes:

1. **NaN from missing grad clipping** — v5 has `clip_grad_norm_(1.0)`,
   v6 had none. Embedding weights diverged (224→NaN). Fixed: added
   `optim.clip_grad_norm(grads, 1.0)`.

2. **Catastrophic flip cascade** — grad clipping protected the
   optimizer but the flip accumulator still saw raw gradients.
   Accumulators reached billions, threshold was 0.1 → 76% of weights
   flipped simultaneously → model destroyed. Fixed: **sign-based
   accumulation** — `accum += sign(grad)` bounds accumulators to ±N.

3. **Flip-induced loss spikes** — even with sign accumulation, fixed
   threshold can't adapt to training dynamics. Fixed: **adaptive
   percentile threshold** with loss-based feedback loop:
   - `compute_flip_threshold(model, target_pct)` → flip top N% by consensus
   - 25 steps after flips, measure loss ratio
   - ratio < 1.02 → target × 1.2 (be aggressive)
   - ratio > 1.10 → target × 0.5 (back off)

**Verified 300 steps**: loss 11.4→10.95, controlled flips (0.2%→0.7%),
threshold ~228 (57% micro-batch consensus), embedding weight stable,
feedback loop self-tuning upward. No collapse.

### Two timescales of learning

v6 has a unique training dynamic: **continuous** (Adam, every step,
clipped) and **discrete** (ternary flips, every 100 steps, adaptive).
Loss curve is sawtooth with downward envelope — spikes after flips
as continuous params re-adapt to new routing, then recovers. Sawtooth
amplitude should decrease as topology stabilizes and flip rate drops.

See `mementum/knowledge/explore/v6-flip-accumulation.md` for details.

## What's next

1. **Train v6** — fresh start:
   ```bash
   uv run python scripts/v6/train.py
   ```
   Watch: flip rate trajectory, loss sawtooth pattern, adaptive
   target_pct evolution, ternary sparsity changes.

2. **Probe v6 checkpoints** as they arrive:
   ```bash
   uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000
   ```
   Probe now shows: flips, adaptive state, accumulator stats per group.

3. **Compare v5 vs v6** once v6 has matching checkpoints at 1k–5k.

4. **Kernel optimization** — after training validates correctness.

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
| Flip accumulation | `mementum/knowledge/explore/v6-flip-accumulation.md` |
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

*v5 stopped at step 5k, v6 restarting with sign-based flip + adaptive threshold

## Probing pipeline

```bash
# v5 (PyTorch)
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v5/step_005000.pt

# v6 (MLX)
uv run python scripts/v6/train.py
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000
```
