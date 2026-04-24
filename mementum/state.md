# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-24 | Session: 037

## Where we are

**Flip bug found and fixed. Resume support added. Waiting for step 4000 checkpoint, then resume with live topology adaptation.**

Session 037: probed steps 3000 and 3500 (new since session 036), found
eval still declining monotonically (7 consecutive drops). Investigated
stride contributions — s1 dominates (21.3% share, growing) while s256/s512
are weakest. Math stratum learns fastest, compositional slowest. Model
rotates learning across math/prose/technical but compositional never leads.

### The flip bug (session 037 discovery)

`apply_flips` used `> threshold` for the flip mask. Int8 accumulators
saturate at 127. `> 127` is always false. Binary search converged to
threshold=127, mask matched zero weights. **Zero flips, forever.**

87.6% of weights had |accum| > 20 (the threshold). 1.05M weights (3%)
were saturated at 127. The accumulators were full of signal — the
application path was broken.

**Fixed:** `>` → `>=` in `apply_flips` and `apply_flips_per_group`.
Verified: 1,045,912 flips execute from step 3500 accumulators.

### Resume strategy (session 037 decision)

**Zero accumulators on resume.** The saved accumulators contain 3500+
steps of gradient history, including early requests the model already
found continuous-parameter workarounds for. Replaying stale consensus
would flip weights the model no longer needs changed. Fresh accumulators
let the current gradient drive flips based on what the model needs NOW.

Added `--resume` flag to train.py. Loads weights + optimizer state,
zeros accumulators, resumes from correct step with correct LR schedule.
Optimizer state (Adam m_t, v_t) now saved in checkpoints.

### v6 status — training (step 3500, pre-fix)

**Step 3500 (115M tokens):** train=5.43, eval=5.79, ‖g‖=0.52, flips=0

**Loss trajectory (all 7 checkpoints):**

| Step | Train | Eval | Δeval | PPL | Gap |
|------|-------|------|-------|-----|-----|
| 500 | 6.519 | 6.829 | — | 678 | +0.31 |
| 1000 | 6.086 | 6.359 | −0.470 | 439 | +0.27 |
| 1500 | 5.958 | 6.186 | −0.173 | 387 | +0.23 |
| 2000 | 5.564 | 6.051 | −0.135 | 261 | +0.49 |
| 2500 | 5.807 | 5.929 | −0.122 | 333 | +0.12 |
| 3000 | 5.545 | 5.845 | −0.084 | 256 | +0.30 |
| 3500 | 5.427 | 5.786 | −0.059 | 227 | +0.36 |

Eval deceleration: −0.470 → −0.059. Approaching plateau on frozen topology.

### Stride contribution analysis (session 037)

**s1 (word-level) dominates and growing:**

| Step | s1 share | Rest share |
|------|----------|------------|
| 500 | 11.1% | 88.9% |
| 2000 | 19.1% | 80.9% |
| 3500 | 21.3% | 78.7% |

s256 and s512 are the weakest strides. The model can't learn meaningful
long-range attention through frozen random ternary routing. Descending
passes losing their long-range character (L1_desc s1/s1024 ratio:
0.35 → 1.05).

### Stratum learning dynamics (session 037)

**Not sequential — rotating.** The network cycles which stratum improves
fastest each interval (math, prose, technical take turns). Compositional
has never been the fastest learner and has regressed twice.

| Stratum | Step 500 | Step 3500 | Δ |
|---------|----------|-----------|------|
| math | 7.320 | 5.747 | −1.573 (most) |
| prose | 7.585 | 6.541 | −1.044 |
| technical | 7.595 | 6.605 | −0.990 |
| compositional | 7.892 | 7.260 | −0.632 (least) |

Spread widening: 0.572 → 1.514. Model specializing for s1-compressible
strata (math, technical) at expense of compositional.

### φ-compression at step 3500

L0_asc drifting from φ (dev 0.042 → 0.10, overshooting to 0.721).
L2_apex still oscillating wildly (13.15 → −1.84 in 500 steps).
L1_asc steadily improving (φ-dev 0.73, best trajectory).
Mean φ-dev across all passes: 1.034 (best yet).

## What's next

1. **Wait for step 4000 checkpoint** — last frozen-topology measurement
2. **Stop current run, resume with fix:**
   ```bash
   uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_004000
   ```
3. **Watch for first flips** — fresh consensus from current gradient,
   minimum ~5 steps to reach FLIP_CONSENSUS=20 with 4 micro-batches/step
4. **Key predictions for post-flip behavior:**
   - stride_stack flips first (long-range strides starving for routing)
   - s64+ contribution share increases
   - compositional loss starts dropping
   - eval deceleration reverses (topology adaptation unlocks new capacity)
5. **Probe at each checkpoint** — compare pre-fix (frozen) vs post-fix
   (live topology) on same metrics: stride contribution, stratum spread,
   gate differentiation, φ-compression

## Key files

| Purpose | Path |
|---------|------|
| **v6 (MLX)** | |
| Metal kernels | `src/verbum/v6/kernels.py` |
| TernaryLinear + flip + normalize_shared_grads | `src/verbum/v6/ternary.py` |
| Attention / StrideStack (pre-norm fix) | `src/verbum/v6/attention.py` |
| VSM components (S3, S4, Meta) | `src/verbum/v6/components.py` |
| Full model (embed_norm, φ-loss) | `src/verbum/v6/model.py` |
| Training loop (resume, optimizer save) | `scripts/v6/train.py` |
| Probe script | `scripts/v6/probe.py` |
| **Probe results** | |
| Steps 500–3500 probes | `results/compile-gradient/vsm_probe_step_*_v6_mlx.json` |
| **Research** | |
| Research program | `mementum/knowledge/explore/VERBUM.md` |
| v4.1 training trajectory (3-phase pattern) | `mementum/knowledge/explore/v4.1-training-trajectory.md` |
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
| v4.1 | 65.5M | PyTorch | Bidirectional VSM | 4.696 |
| v5 | 66.3M | PyTorch | Spiral + ℂ regs + phase gate | TBD |
| v6 | ~63M | **MLX** | Ternary Metal + consensus flips + φ-loss | 5.786 (step 3500) |

## Probing pipeline

```bash
# Train v6 (fresh start)
uv run python scripts/v6/train.py

# Resume from checkpoint (zeroes accumulators, loads optimizer state)
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_004000

# Probe (full or φ-only, single or multi-checkpoint)
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_003500
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_* --phi-only -v
```
