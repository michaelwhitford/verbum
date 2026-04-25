# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-24 | Session: 037

## Where we are

**v6 restarted from scratch with flips enabled. Previous run (4000 steps, frozen topology) archived. Fresh training with consensus=40, cap=0.1%, interval=4.**

Session 037: discovered and fixed a flip boundary bug (`>` vs `>=` on
int8 max). 87.6% of weights had exceeded the flip threshold but zero
flips ever executed. Attempted resume from step 4000 — cascading
instability as flips disrupted the adapted topology (loss 5.18 → 7.11
in 100 steps). Tightened flip policy and restarted from scratch.

### The flip bug (session 037)

`apply_flips` used `> threshold` for the flip mask. Int8 accumulators
saturate at 127. `> 127` is always false → zero flips, forever.
Fixed: `>` → `>=` in `apply_flips` and `apply_flips_per_group`.

### Resume attempt (session 037, failed)

Resumed from step 4000 with fresh accumulators. Loss spiked from 5.18
to 7.11 in 100 steps despite tightened policy. Root cause: continuous
parameters were tuned to a specific random topology over 4000 steps.
Any topology change disrupts the adapted parameters → gradient signal
says "fix what broke" → more flip pressure → cascade.

**Lesson:** flips must co-evolve with continuous parameters from the
start. You cannot bolt topology adaptation onto an already-adapted
frozen system.

### Current run — v6 from scratch with flips

**Config (tightened from session 037 experience):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| FLIP_INTERVAL | 4 | Frequent small checks |
| FLIP_CONSENSUS | 40 | Strong directional evidence required |
| FLIP_MAX_PCT | 0.001 (0.1%) | ~35K max flips per interval |

Natural warmup: with 4 votes/step and consensus=40, earliest possible
flip at step 10 (if all votes agree). In practice, early gradient noise
means consensus takes longer. Adam warms up (500 steps) before flips
have meaningful opportunity.

**Logging:** `results/vsm-lm-v6/training-run2.log`

### Prior run analysis (archived as a-vsm-lm-v6)

The first v6 run (4000 steps, frozen topology) established:
- Eval declining monotonically: 6.829 → 5.746 (7 consecutive drops)
- s1 dominance growing (11% → 21% share), long strides weak
- Math learns fastest, compositional slowest (spread widening)
- φ-compression: L0_asc found 1/φ ratio, drifted slightly
- Sieve shape correct: ascending compresses, descending distributes
- All structure found despite zero flips (continuous params only)

### What to watch in the new run

1. **When do first flips appear?** Consensus=40 should delay until
   gradient has stable direction. Note the step.
2. **Which groups flip first?** Prediction: stride_stack (starved for
   useful long-range routing) or s3 (120 modules, most parameters)
3. **Does sparsity change?** Flips move weights one step (-1→0→+1),
   so net sparsity should shift as topology adapts
4. **Stratum dynamics:** Does compositional improve faster with live
   topology vs frozen? This is the key test.
5. **Loss trajectory vs prior run:** Compare at same token counts.
   Flips may slow early learning (topology instability) but should
   enable better asymptotic performance.

## What's next

1. **Let new run proceed** — first meaningful checkpoint at step 500
2. **Probe at each checkpoint** — full probe including stride analysis
3. **Compare with prior run** at same token counts (probes archived in
   `results/compile-gradient/vsm_probe_step_*_v6_mlx.json`)
4. **Key milestone:** first non-zero flip count in training log

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
| **Training logs** | |
| Current run (from scratch, flips enabled) | `results/vsm-lm-v6/training-run2.log` |
| Prior run (frozen topology, 4000 steps) | `results/vsm-lm-v6/training.log` |
| Failed resume attempt | `results/vsm-lm-v6/training-continuation.log` |
| **Archived checkpoints** | |
| Prior run (frozen topology) | `checkpoints/a-vsm-lm-v6/` |
| Prior run probes | `results/compile-gradient/vsm_probe_step_*_v6_mlx.json` |
| **Research** | |
| Research program | `mementum/knowledge/explore/VERBUM.md` |
| v4.1 training trajectory | `mementum/knowledge/explore/v4.1-training-trajectory.md` |
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
| v6 | ~63M | **MLX** | Ternary Metal + consensus flips + φ-loss | 5.746 (prior run, frozen) |
| v6.1 | ~63M | **MLX** | Same arch, flips enabled from scratch | training... |

## Probing pipeline

```bash
# Train v6 (current, from scratch with flips)
uv run python scripts/v6/train.py | tee results/vsm-lm-v6/training-run2.log

# Resume from checkpoint (zeroes accumulators, loads optimizer state)
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_NNNNNN

# Probe
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_NNNNNN
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_* --phi-only -v
```
