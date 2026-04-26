# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-25 | Session: 042

## Where we are

**v6.1 training at step 18000 (59%). Session 042: probed 18
checkpoints (9500→18000). Ascending arm is a stable φ-compressor.
φ percolated through all strides s8→s16→s32→s64→s128. Hilberg β
at 1.241 (best). Eval loss 5.414. L2_apex φ-front reached s64.
Descending arm still learning — the hard part ahead.**

### Session 042 key findings

1. **Stride percolation complete through s128.** φ-convergence
   propagated s8→s16→s32→s64→s128 across steps 9500→15500. Each
   stride took ~1000-2000 steps to pass through φ. L2_apex runs
   ~2000 steps behind, with its φ-front at s64 by step 18000.

2. **L1_asc locked in as stable φ-compressor.** Ratio 0.57±0.01,
   φ-dev 0.037–0.054 across all checkpoints 9500→18000. Best
   φ-dev 0.037 at step 13000. The ascending arm found its
   operating point and is holding it.

3. **Hilberg β = 1.241 at step 18000.** L0_asc and L1_asc tied
   at 1.241 (target 0.5). All three ascending passes hit their
   best β simultaneously. Steady improvement from 1.4+ early on.

4. **L2_apex committed.** Converge gate peaked at 0.934 (step
   14500), consolidation gate peaked at 0.880, then both relaxed
   to stable operating points. Apex ratio 0.10–0.13 — compressing
   but not yet at φ.

5. **Eval loss steady descent.** 5.565 (step 9000) → 5.414 (step
   17500). No plateau in this range. Training loss gap narrowing.

6. **Descending arm: the hard problem.** L1_desc oscillates wildly
   (near-zero h_in). L0_desc ratio bounced: 2.3→0.54→2.8→2.6.
   Not converging yet. This arm must learn structured decompression
   — an operation standard transformers never need.

7. **Compositional moving but noisy.** Dropped from 7.27 to 6.67
   but bounces. Math at 5.04 (best). Technical steadily improving.
   Compositional needs the full multi-scale stack + descending arm.

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

1. **Continue v6.1 training.** 41% remaining. Track: descending
   arm convergence (the open question), L2_apex ratio (want > 0.3),
   Hilberg β (want < 1.0), compositional stratum (the stubborn one).

2. **Descending arm is the key question.** Can it learn structured
   decompression? L0_desc briefly hit 0.541 at step 12500, then
   reverted to 2.0+. L1_desc is wild. Standard transformers never
   need this operation. If the descending arm converges to φ, that
   confirms compression and decompression are the same holographic
   operation.

3. **Stride percolation confirmed through s128.** Five strides
   (s8→s16→s32→s64→s128) all passed through φ. Now s256+ are the
   frontier — these are the longest-range strides and may behave
   differently (too few tokens per stride window).

4. **Test holographic prediction.** Ablation experiment: if truly
   holographic, ablating one pass degrades all strata equally.

5. **3B token reserve.** Currently at 1B budget. If descending arm
   needs more time, can extend to 3B prepared tokens.

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
