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
| Current step | ~18750 (20% of 3B schedule) |
| Total steps | **91,553** (extended from 30,518) |
| Tokens seen | ~614M of 3B |
| Token budget | **3B** (extended from 1B, 2.7B train shards) |
| Eval loss | **5.414** (step 17500) — best |
| Relational r̄ | 0.379 (step 18750, declining) |
| Sparsity | 0.310 (unchanged) |
| L1_asc φ-dev | **0.037** (step 13000, best) |
| L1_asc range | 0.564–0.581 (locked in) |
| L2_apex ratio | +0.131 (step 18000, compressing) |
| L1_desc | wild oscillations (h_in ≈ -0.1) |
| L0_desc | 2.0–4.6 (expanding, not converging) |
| Hilberg β | L0↑=L1↑=**1.241** (step 18000, best) |
| Stride percolation | s8→s16→s32→s64→s128 confirmed |
| Total flips | ~178,000 (0.50% cumulative) |
| LR (current) | ~2.0e-4 (old 1B schedule, about to jump) |
| LR (after 19k resume) | ~5.4e-4 (new 3B schedule, 2.8× jump) |

### Eval loss evolution

| Step | Eval Loss | ppl | r | L1_asc φ-dev | L2_apex | Hilberg β |
|------|-----------|------|------|-------------|---------|-----------|
| 9000 | 5.565 | 261 | 0.424 | 0.052 | -0.023 | 1.59/1.41 |
| 11000 | 5.514 | 248 | 0.419 | 0.045 | +0.062 | 1.39/1.42 |
| 13000 | 5.500 | 170 | 0.377 | **0.037** | +0.119 | 1.30/1.33 |
| 13500 | 5.465 | 219 | 0.405 | 0.046 | +0.100 | 1.36/1.30 |
| 15000 | 5.468 | 133 | 0.350 | 0.046 | +0.095 | 1.25/1.28 |
| 16000 | 5.440 | 217 | 0.404 | 0.053 | +0.077 | 1.27/1.31 |
| 17500 | **5.414** | 197 | 0.393 | 0.046 | +0.114 | 1.27/1.25 |
| 18000 | 5.424 | 155 | 0.367 | 0.041 | +0.131 | **1.24/1.24** |

### Stratum loss evolution (post-phase-transition)

| Step | Prose | Comp | Tech | Math | Spread |
|------|-------|------|------|------|--------|
| 4500 | 6.30 | 6.73 | 7.26 | 6.05 | 1.21 |
| 9000 | 6.18 | 6.72 | 7.15 | 5.59 | 1.56 |
| 13500 | 6.17 | 6.64 | 7.23 | 5.23 | 2.00 |
| 16000 | **6.06** | 6.76 | **7.07** | 5.16 | 1.91 |
| 17500 | 6.19 | 6.75 | **7.02** | **5.04** | 1.98 |
| 18000 | **6.04** | **6.67** | 7.12 | 5.14 | 1.98 |

### Three-way φ-compression comparison (updated step 18000)

| Metric | v6 (63M, VSM) | Pythia (162M) | Qwen3-4B (4B) |
|--------|--------------|---------------|----------------|
| Stable zone ratio | **0.577** | 0.947 | 1.000 |
| Stable zone φ-dev | **0.041** | 0.329 | 0.387 |
| Best single layer | L1_asc: 0.037 | L9: 0.172 | L34: 0.037* |
| Composition mechanism | Compression | Rotation | Rotation |
| Architecture type | Holographic | Photographic | Photographic |
| Strides at φ | **5 (s8→s128)** | N/A | N/A |

*L34 is the output collapse layer, not the computation core.

## What's next

1. **Resume at step 19000 with 3B schedule.** Training extended to
   3B tokens (91,553 steps). LR jumps from ~2e-4 to ~5.4e-4 (2.8×).
   Command: `uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_019000`
   Watch r̄ and flip rate for stability after the LR bump.

2. **Descending arm is THE question.** Can it learn structured
   decompression? The higher LR + 72,500 more steps gives it the
   runway it needs. L0_desc briefly hit 0.541 at step 12500 then
   reverted. If the descending arm converges to φ, that confirms
   compression and decompression are the same holographic operation.

3. **Track ascending arm stability through LR jump.** L1_asc has
   been locked at 0.57±0.01 for 9000 steps. It should survive the
   2.8× LR bump — it survived the full 6e-4 peak. If it destabilizes,
   that's important data.

4. **Stride percolation: watch s256+.** Five strides confirmed.
   s256 at 0.559 (step 18000) approaching φ. These longest-range
   strides may behave differently (few tokens per window).

5. **Test holographic prediction.** Ablation experiment: if truly
   holographic, ablating one pass degrades all strata equally.

6. **r̄ approaching refine threshold.** Currently 0.379, refine
   phase triggers at r̄ < 0.25 (with 100-step hysteresis). The LR
   jump may push r̄ up temporarily, delaying the transition. If r̄
   reaches refine phase, flip rates drop to 30% — topology freezes.

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
| v6.1 probes (steps 500–18000) | `results/compile-gradient/vsm_probe_step_*_v6_mlx.json` |
| **Research** | |
| Research program | `mementum/knowledge/explore/VERBUM.md` |
| **Holographic compression** | `mementum/knowledge/explore/holographic-compression.md` |
| **Stride percolation** | `mementum/knowledge/explore/stride-percolation.md` |
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
| v6.1 | ~63M | **MLX** | Synaptic plasticity (active) | **5.414** (17500 steps, 59%) |

## Probing pipeline

```bash
# v6 probe (single or multiple checkpoints)
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_*

# Pythia φ-compression probe
uv run python scripts/run_pythia_phi_probe.py --verbose

# Resume training if interrupted
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_NNNNNN
```
