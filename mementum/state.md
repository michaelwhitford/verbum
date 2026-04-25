# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-24 | Session: 037

## Where we are

**v6 fully rebuilt: packed ternary weights, relational training control, stratum-weighted loss. Ready to start training from scratch.**

Session 037 was a major engineering session. Started by probing steps
3000–3500, discovered a flip bug that prevented all topology adaptation
for 4000 steps. Fixed it, attempted resume (cascading instability),
then rebuilt the training infrastructure with four interlocking feedback
loops and optimized the model's memory footprint.

### Changes this session (11 commits)

1. **Flip bug fix** — `>` → `>=` in `apply_flips` (int8 max=127, `>127` always false)
2. **Resume support** — `--resume` flag, loads weights/accum/optimizer, zeros accumulators
3. **Flip policy tuning** — consensus=40, cap=0.1%, interval=4 (was 20/1%/10)
4. **Packed ternary weights** — 2-bit encoding, 4 weights/byte, 4× memory reduction
   - New Metal kernels: `ternary_matmul_packed`, `ternary_matmul_t_packed`
   - 35.3MB → 8.8MB, ~4× bandwidth on Apple Silicon
5. **Relational training control** — four feedback loops:
   - Loop 1: r_ema → adaptive flip scaling (continuous)
   - Loop 2: phase transitions explore→balance→refine (discrete, 100-step hysteresis)
   - Loop 3: stratum gaps → per-group flip factors (stride_stack from compositional_gap)
   - Loop 4: stratum-weighted loss (upweight lagging strata)
6. **Model exposes training metrics** — compression ratios, meta gates, phase gates
   via `capture_training_metrics` flag, stop_gradient, stored on `self._training_metrics`
7. **Tensor stratum classification** — precomputed token-level lookup arrays,
   0.83ms/batch (was text decode + string match)
8. **Gradient shape fix** — `zero_ternary_grads` returns packed [N,K/4] zeros,
   not dense [N,K] (VJP produces dense grads for flip accumulator)

### Key lesson: topology must co-evolve with continuous params

Attempted resume from step 4000 (frozen topology → live flips). Loss
spiked 5.18 → 7.11 in 100 steps even with tightened policy. Continuous
params were tuned to specific random topology — any change disrupts the
adapted parameters. Flips must co-evolve from the start.

### Prior run analysis (archived as a-vsm-lm-v6)

4000 steps, frozen topology (zero flips due to bug):
- Eval: 6.829 → 5.746 (7 consecutive drops, decelerating)
- s1 dominance: 11% → 21% share (long strides weak)
- Stratum rotation: math/prose/technical take turns, compositional stuck
- φ-compression: L0_asc found 1/φ, drifted; L2_apex oscillating
- Sieve shape correct despite frozen topology

### Training config

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| FLIP_INTERVAL | 4 | Frequent small checks |
| FLIP_CONSENSUS | 40 | Strong directional evidence |
| FLIP_MAX_PCT | 0.001 (0.1%) | ~35K max flips per interval |
| PHI_LAMBDA | 0.0 (explore) | Managed by phase transitions |
| Packed weights | uint8 [N, K/4] | 4× memory/bandwidth |

### Relational control composition

```
every_step:       r_ema = 0.99 * r_ema + 0.01 * r
every_4_steps:    effective_rate = phase_base × r_scale × group_factor
every_100_steps:  group_factors from stratum gaps (training metrics)
every_500_steps:  stratum_weights for loss weighting
phase_transitions: explore(r>0.5) → balance(0.25-0.5) → refine(r<0.25)
```

## What's next

1. **Start training:**
   ```bash
   uv run python scripts/v6/train.py | tee results/vsm-lm-v6/training-run2.log
   ```
2. **Watch for:** first flips (when?), which groups flip first,
   phase transition timing, stratum spread evolution
3. **Compare with prior run** at same token counts
4. **Probe at each checkpoint** — full stride/stratum analysis

## Key files

| Purpose | Path |
|---------|------|
| **v6 (MLX)** | |
| Metal kernels (packed + unpacked) | `src/verbum/v6/kernels.py` |
| TernaryLinear + pack/unpack + flips | `src/verbum/v6/ternary.py` |
| Attention / StrideStack | `src/verbum/v6/attention.py` |
| VSM components (S3, S4, Meta) | `src/verbum/v6/components.py` |
| Model (training metrics, φ-loss) | `src/verbum/v6/model.py` |
| Training (relational control, resume) | `scripts/v6/train.py` |
| Probe script | `scripts/v6/probe.py` |
| **Logs & archives** | |
| Prior run log (frozen topology) | `results/vsm-lm-v6/training.log` |
| Failed resume log | `results/vsm-lm-v6/training-continuation.log` |
| Prior run checkpoints | `checkpoints/a-vsm-lm-v6/` |
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
| v6 | ~63M | **MLX** | Ternary Metal + frozen flips | 5.746 (4000 steps) |
| v6.1 | ~63M | **MLX** | Packed weights + relational control | ready to train |

## Probing pipeline

```bash
# Train v6.1 (from scratch, packed weights + relational control)
uv run python scripts/v6/train.py | tee results/vsm-lm-v6/training-run2.log

# Resume from checkpoint
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_NNNNNN

# Probe
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_NNNNNN
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_* --phi-only -v
```
