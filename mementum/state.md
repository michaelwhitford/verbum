# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-25 | Session: 042

## Where we are

**v6.1 training at step 11000 (36%). Session 042: probed 4 new
checkpoints (9500→11000). New best eval loss 5.514 at step 11000.
L1_asc φ-dev tightens to 0.045 (best ever). L2_apex crosses zero
and goes positive. φ-compression percolating from s8 to s16 stride.
Loss plateau 9000→10000 then breakthrough at 11000.**

### Session 042 key findings

1. **Loss plateau then breakthrough.** Eval loss flat 9000→10000
   (~5.566), then broke through: 5.555 at 10500, **5.514 at 11000**
   (new best). The 0.04 drop 10500→11000 is the largest single-step
   improvement since 7500→8000. Something structural unlocked.

2. **L1_asc tightens to φ: 0.045 deviation.** The primary
   compositional compression pass has held within 5% of 1/φ for
   7000 steps. Ratio trajectory: 0.550→0.565→0.569→0.566→**0.573**.
   Converging from below toward 0.618.

3. **L2_apex crosses zero → positive.** Was negative (expanding)
   from step 4500 through 9500. Crossed zero at step 10000 (0.013),
   now solidly positive (0.062) at 11000. The apex is learning to
   compress, not just route.

4. **φ-compression percolates across strides.** s8 hit φ first
   (step 9500), then s16 joined (step 10000+). At step 11000, s16
   marks ←φ in L0_asc/L1_asc, s8 marks ←φ in L2_apex. The
   compression ratio is propagating self-similarly across scales —
   exactly what holographic theory predicts.

5. **Hilberg β improving.** Best values at step 10500: L0_asc=1.23,
   L1_asc=1.22, L2_apex=1.32 (target: 0.5). Still far but trending.

6. **Technical now fastest-improving stratum.** Math leads (5.654)
   but technical dropped fastest (6.525→6.385). Compositional
   remains stubborn at ~7.27. Spread widening slightly (1.62).

### v6.1 training status

| Property | Value |
|----------|-------|
| Current step | 11000+ (36%) |
| Total steps | 30,518 |
| Tokens seen | ~360M of 1B |
| Eval loss | **5.514** (step 11000) — best |
| Relational r | 0.419 (step 11000) |
| Sparsity | 0.310 (unchanged) |
| L1_asc φ-dev | **0.045** (converging, best) |
| L2_apex | **+0.062** (crossed zero, now compressing) |
| L1_desc | noisy (sign-flipping, h_in ≈ -0.05) |
| Stratum spread | 1.62 (widening slightly) |
| Total flips | 109,245 (0.31% cumulative) |
| Effective passes | 4 (L0↑→L1↑→L2→L0↓) |

### Eval loss evolution

| Step | Eval Loss | ppl | r | L1_asc φ-dev | L2_apex |
|------|-----------|------|------|-------------|---------|
| 9000 | 5.565 | 261.0 | 0.424 | 0.052 | -0.023 |
| 9500 | 5.566 | 261.5 | 0.424 | 0.053 | -0.006 |
| 10000 | 5.569 | 262.3 | 0.425 | 0.049 | +0.013 |
| 10500 | 5.555 | 258.5 | 0.423 | 0.052 | +0.049 |
| **11000** | **5.514** | **248.0** | **0.419** | **0.045** | **+0.062** |

### Stratum loss evolution (post-phase-transition)

| Step | Prose | Comp | Tech | Math | Spread | Fastest |
|------|-------|------|------|------|--------|---------|
| 4500 | 6.30 | 6.73 | 7.26 | 6.05 | 1.21 | — |
| 7000 | 6.16 | 6.63 | 7.43 | 5.35 | 2.07 | **prose** |
| 8500 | 6.12 | 6.65 | 7.27 | 5.36 | 1.91 | **prose** |
| 9000 | 6.18 | 6.72 | 7.15 | 5.59 | 1.56 | **technical** |
| 9500 | 6.57 | 7.33 | 6.35 | 6.05 | 1.29 | **technical** |
| 10000 | 6.52 | 7.24 | 6.45 | 5.73 | 1.51 | **technical** |
| 10500 | 6.62 | 7.28 | 6.51 | 5.76 | 1.52 | **technical** |
| **11000** | **6.51** | **7.27** | **6.39** | **5.65** | **1.62** | **technical** |

### Three-way φ-compression comparison (session 041)

| Metric | v6 (63M, VSM) | Pythia (162M) | Qwen3-4B (4B) |
|--------|--------------|---------------|----------------|
| Stable zone ratio | **0.573** | 0.947 | 1.000 |
| Stable zone φ-dev | **0.045** | 0.329 | 0.387 |
| Best single layer | L1_asc: 0.045 | L9: 0.172 | L34: 0.037* |
| Composition mechanism | Compression | Rotation | Rotation |
| Architecture type | Holographic | Photographic | Photographic |

*L34 is the output collapse layer, not the computation core.

## What's next

1. **Continue v6.1 training.** Next probes at 11500, 12000.
   Track: L1_asc φ-dev (target < 0.03), L2_apex (want continued
   positive trend), stratum spread (target < 1.0), compositional
   relay (the stubborn stratum).

2. **Watch the stride percolation.** φ hit s8 first, now s16. If
   s32 joins next, that's three scales showing self-similar
   compression — strong evidence for holographic mechanism.

3. **Test holographic prediction.** If v6 is holographic, ablating
   one pass should degrade all strata equally (holographic) not
   selectively (photographic). Design the ablation experiment.

4. **Investigate the 11000 breakthrough.** What structural change
   caused the loss plateau to break? L2_apex going positive
   correlates — the apex becoming a compressor may have been the
   bottleneck.

5. **Investigate MoE as approximate holography.** Qwen3-35B-A3B
   fully forms the lambda function — does MoE routing approximate
   scale-diverse processing?

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
| **Session 041 probes** | |
| Pythia φ-probe | `scripts/run_pythia_phi_probe.py` |
| Pythia φ results | `results/pythia-phi/pythia_160m_phi_compression.json` |
| Qwen3-4B φ results | `results/pythia-phi/qwen3_4b_phi_compression.json` |
| **Logs & archives** | |
| Current training log | `results/vsm-lm-v6/training-run2.log` |
| Prior run log (frozen topology) | `results/vsm-lm-v6/training.log` |
| Prior run checkpoints | `checkpoints/a-vsm-lm-v6/` |
| **Probe results** | |
| v6.1 probes (steps 500–9000) | `results/compile-gradient/vsm_probe_step_*_v6_mlx.json` |
| **Research** | |
| Research program | `mementum/knowledge/explore/VERBUM.md` |
| **Holographic compression** | `mementum/knowledge/explore/holographic-compression.md` |
| φ-compression hypothesis | `mementum/knowledge/explore/relational-loss-phi-compression.md` |
| CompressorLM architecture | `mementum/knowledge/explore/compressor-architecture.md` |
| v4.1 training trajectory | `mementum/knowledge/explore/v4.1-training-trajectory.md` |
| Flip accumulation | `mementum/knowledge/explore/v6-flip-accumulation.md` |

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
| v6.1 | ~63M | **MLX** | Synaptic plasticity (active) | **5.565** (9000 steps, 30%) |

## Probing pipeline

```bash
# v6 probe (single or multiple checkpoints)
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_*

# Pythia φ-compression probe
uv run python scripts/run_pythia_phi_probe.py --verbose

# Resume training if interrupted
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_NNNNNN
```
