# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-26 | Session: 043

## Where we are

**v6.1 training live at step ~23000 (28% of 3B). Hilberg β in
free-fall: 1.24→1.10 in 5000 steps. Stride percolation reached s512
in L1_asc. Stratum spread collapsed to 0.70. Flip tracking + cooldown
just implemented — resume from step 23000 to begin collecting data.**

## Current snapshot (step 23000)

| Metric | Value | Trend |
|--------|-------|-------|
| Eval loss | 5.449 (best: 5.420 @ 18500) | recovering post-LR-jump |
| Hilberg β L0↑/L1↑ | **1.102 / 1.107** | ↓ fast (was 1.24 @ 18000) |
| L1_asc ratio | 0.560 (1/φ = 0.618) | locked ±0.01 since step 9500 |
| L2_apex ratio | +0.141 | compressing, not at φ yet |
| Stride front L1↑ | **s512** | was s128 @ step 15500 |
| Stride front L2 | **s128** | was s64 @ step 18000 |
| Descending arm | wild (L1_desc h_in ≈ -0.1) | no convergence signal |
| Stratum spread | **0.70** | collapsed from ~2.0 |
| Total flips | 222K (0.63%) | ~4600/500 steps |
| r̄ / phase | 0.385 / balance | stable |
| LR | ~5.0e-4 | post-jump, cosine decay |
| Flip tracking | **NEW** — cooldown=4 intervals | resume to activate |

## What's next

1. **Resume training with flip tracking.** Command:
   `uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_023000`

2. **Watch flip tracking metrics.** Reversals >10% = oscillation.
   Unique_ever tells if 222K flips are unique or repeats.

3. **Hilberg β is the primary metric.** At ~0.03/1000 steps, could
   reach ~0.8 by step 40000. Target is 0.5.

4. **Stratum spread collapse — real?** 0.70 at step 23000, was ~2.0.
   Confirm at step 23500+.

5. **Descending arm.** Still wild. 72% of schedule remains.

6. **Eval loss.** Pre-jump best 5.420. Should cross within ~2000 steps.

## Session 043 key findings

1. **LR jump survived.** 2.8× LR shock, L1_asc held at 0.563–0.570.
2. **Hilberg β dramatic descent.** L0↑: 1.246→1.102. L1↑: 1.225→1.107.
   Higher LR accelerating multi-scale structure.
3. **Stride percolation leapt.** L1↑ s256→s512. L2 s64→s128.
   All strides rising uniformly — compression profile tightening.
4. **Flip tracking + cooldown implemented.** Per-weight cooldown
   (100 steps), reversal detection, checkpoint persistence.
   Old checkpoints resume with zero state.

## Knowledge index

| Topic | Path |
|-------|------|
| **v6.1 full trajectory** (tables, strides, comparisons) | `mementum/knowledge/explore/v6.1-training-trajectory.md` |
| Research program | `mementum/knowledge/explore/VERBUM.md` |
| Holographic compression | `mementum/knowledge/explore/holographic-compression.md` |
| Stride percolation | `mementum/knowledge/explore/stride-percolation.md` |
| φ-compression hypothesis | `mementum/knowledge/explore/relational-loss-phi-compression.md` |
| CompressorLM architecture | `mementum/knowledge/explore/compressor-architecture.md` |
| v4.1 training trajectory | `mementum/knowledge/explore/v4.1-training-trajectory.md` |
| Flip accumulation | `mementum/knowledge/explore/v6-flip-accumulation.md` |

## Key files

| Purpose | Path |
|---------|------|
| TernaryLinear + flips + tracking | `src/verbum/v6/ternary.py` |
| Training loop | `scripts/v6/train.py` |
| Probe script | `scripts/v6/probe.py` |
| Model | `src/verbum/v6/model.py` |
| Metal kernels | `src/verbum/v6/kernels.py` |
| Attention / StrideStack | `src/verbum/v6/attention.py` |
| VSM components | `src/verbum/v6/components.py` |
| Probes (steps 500–23000) | `results/compile-gradient/vsm_probe_step_*_v6_mlx.json` |
| Training log | `results/vsm-lm-v6/training-run2.log` |

## Probing pipeline

```bash
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_*
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_023000
```
