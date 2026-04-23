# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-23 | Session: 028

## Where we are

**v5 stopped at step 5k. v6 restarting after NaN fix.**

### v5 status

Stopped at step 5k. Checkpoints available at steps 1k–5k.

### v6 — NaN diagnosed and fixed (session 028)

v6 training ran steps 1k–6k but produced **NaN loss from the very
first checkpoint**. All 6 checkpoints were invalid. Diagnosed and
fixed in session 028.

**Root cause**: Missing gradient clipping. v5 (PyTorch) uses
`clip_grad_norm_(model.parameters(), 1.0)` — v6 (MLX) had none.
Without clipping, embedding weight norm grew monotonically
(224→248→NaN over ~400 steps). The 5-pass architecture amplifies
gradients through 5 sequential level-passes, and tied weight
projection (`logits = x @ embed.T`) creates a positive feedback
loop: large logits → large loss → large gradients → larger weights.

**Symptoms**: 11.1M ternary weight flips at first `apply_flips`
(step 100) — the flip accumulators overflowed with astronomically
large gradient values (max accumulator >1e9). After the catastrophic
flip, 76% of all ternary weights changed simultaneously, but the
model was already dead from NaN.

**Fixes** (three parts):
1. `train.py`: Gradient clipping — `optim.clip_grad_norm(grads, 1.0)`.
   NaN guard skips optimizer step if loss is NaN.
2. `ternary.py`: **Sign-based accumulation** — `accum += sign(grad)`
   instead of raw magnitude. Each micro-batch casts a direction vote
   (+1/-1). After N accumulations, |accum| ≤ N (bounded). Eliminates
   scale mismatch between gradients and threshold.
3. `train.py`: **Adaptive percentile threshold** — instead of fixed
   threshold, flip the top `target_pct` of weights by consensus.
   Feedback loop adjusts target_pct based on loss recovery after flips:
   - ratio < 1.02 (flips helped): target × 1.2
   - ratio > 1.10 (flips hurt): target × 0.5
   - Clamped to [0.01%, 2%]

**Verified**: 300-step training shows:
- Loss steady decline: 11.4 → 10.95
- Controlled flips: 73k (0.2%) → 195k (0.6%) → 245k (0.7%)
- Threshold at ~228 = 57% micro-batch directional consensus
- Feedback loop self-tuning target upward (model absorbs flips well)
- Embedding weight stable at 223.3–223.9
- ‖g‖ bounded 0.56–1.64

**Training loop pattern** (updated):
```python
loss, grads = loss_and_grad_fn(model, x, y)
accumulate_flips(model, grads)            # sign(grad) → accumulator (+1/-1 votes)
grads, grad_norm = clip_grad_norm(grads, 1.0)
optimizer.update(model, grads)             # Adam updates continuous params
restore_ternary(model)                     # re-cast int8
if step % FLIP_INTERVAL == 0:
    threshold = compute_flip_threshold(model, target_pct)  # percentile
    apply_flips(model, threshold)          # flip top target_pct by consensus
    # 25 steps later: measure loss ratio → adapt target_pct
```

### v6 architecture (unchanged from session 027)

MLX + custom Metal compute kernels for ternary matmul. 171
TernaryLinear modules run add/sub on GPU via Metal Shading Language.
See `docs/v6-design.md` for full architecture description.

**Data**: Dolma, 3B tokens, 60 shards × 50M, GPT-NeoX tokenizer
(vocab_size=50277, int32). Train/eval split: 54/6 shards. Same
data pipeline as v1–v5. Ready at `/Users/mwhitford/data/fractal-bitnet/shards/`.

## What's next

1. **Restart v6 training** with grad clipping fix:
   ```bash
   uv run python scripts/v6/train.py
   ```
   Invalid checkpoints cleared. Fresh start with same data/seed.
   Key questions:
   - Does flip accumulation produce useful ternary patterns?
   - How fast do ternary weights stabilize (flip rate over time)?
   - Does the 9-stride geometric ladder beat v5's 4-stride allocation?
   - What does per-channel gamma distribution look like after training?
   - Can the model match v5 loss with 99.6% add/sub compute?

2. **Probe v5 at steps 1k–5k** — compare with v6 once v6 has
   matching checkpoints. Watch for phase transition in alignment
   gates, modulation divergence, gate polarity emergence.

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

*v5 stopped at step 5k, v6 restarting with grad clipping fix

## Probing pipeline

```bash
# v5 (PyTorch)
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v5/step_010000.pt
uv run python scripts/register_analysis.py capture checkpoints/vsm-lm-v5/step_010000.pt --analyze

# v6 (MLX)
uv run python scripts/v6/train.py
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000
```
