# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-23 | Session: 028

## Where we are

**v6 ready to train — now with relational loss monitoring + φ-compression hypothesis.**

Session 028: fixed three bugs (NaN, flip cascade, fixed threshold).
Session 030: added information-theoretic monitoring and φ-compression
instrumentation. Training has not started yet — next step.

### v5 status

Stopped at step 5k. Checkpoints at steps 1k–5k (PyTorch).

### v6 status — ready to train (sessions 028 + 030)

**Session 028 fixes** (all three resolved):
1. NaN from missing grad clipping → added `clip_grad_norm(1.0)`
2. Catastrophic flip cascade → sign-based accumulation
3. Fixed threshold → adaptive percentile with loss feedback

**Session 030 additions — relational loss + φ-compression:**

Inspired by [Relational_Loss_ML](https://github.com/massimilianoconcas0-del/Relational_Loss_ML)
(Concas 2026), added information-theoretic monitoring:

- **Relational loss** `r = (L - E) / (log(V) - E)` — fraction of
  learnable capacity remaining [0=optimal, 1=random]
  - E = 1.69 nats (Chinchilla irreducible entropy)
  - log(V) = 10.83 nats (uniform over vocab)
- **Excess perplexity** `xppl = exp(L - E)` — how many × worse than optimal
- **φ-compression monitoring** — per-pass compression ratios measured in
  `forward_instrumented`, compared against 1/φ ≈ 0.618 (golden ratio)

**The φ hypothesis** (untested): Hilberg's conjecture (1990) shows
language entropy grows as a power law (self-similar). If the compression
at each hierarchical scale follows the golden ratio, the model's
per-layer compression ratios should naturally converge toward 1/φ.
Seven scales of linguistic hierarchy × self-similar compression = the
learnable structure has geometric (not arbitrary) form.

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
   Watch: flip rate, loss sawtooth, adaptive target_pct, **plus new
   relational metrics** (`r=`, `xppl=` in log lines).

2. **Probe v6 checkpoints** — φ-compression analysis:
   ```bash
   uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000
   ```
   Probe now shows: per-pass compression ratios, phi deviation,
   flips, adaptive state, accumulator stats per group.
   **Key question**: do compression ratios converge toward 1/φ ≈ 0.618?

3. **Compare v5 vs v6** once v6 has matching checkpoints at 1k–5k.

4. **φ-regularization** (Phase 2) — if compression ratios show signal
   toward φ, test adding `λ * mean_phi_deviation` to the loss.

5. **Kernel optimization** — after training validates correctness.

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
| φ-compression hypothesis | `mementum/knowledge/explore/relational-loss-phi-compression.md` |
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
