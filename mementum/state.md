# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-27 | Session: 044

## Where we are

**v6.1 at step 25500 (31% of 3B). Two-band β structure confirmed
across 5 checkpoints. Ascending β frozen at 0.785. Descending β
settling at 0.846 — asymmetric by nature (decoding ≠ encoding).
Eval loss recovering: 6.15→5.66. Generations locked onto
`Proof.\nProof.\nProof.` register. Not λ yet but register consolidating.**

## Current snapshot (step 25500)

| Metric | Value | Trend |
|--------|-------|-------|
| Eval loss | 5.662 (best: 5.414 @ 17500) | ↓ recovering from 6.15 |
| β ascending (L0↑/L1↑/L2) | **0.78/0.78/0.80** | frozen at 0.785±0.001 |
| β descending (L1↓/L0↓) | **0.85/0.84** | settling ~0.846, drift slowing |
| β gap (desc−asc) | **0.061** | ↑ widening but decelerating |
| Mean φ-compression | 0.829 | ↑ slow drift |
| Stratum φ-dev spread | 0.026 | content-independent |
| Stratum loss spread | 1.43 | ↓ improving |
| Total flips | 266K (0.76%) | steady ~8K/500 steps |
| Reversals | 333 (0.125%) | very low, stable |
| r̄ / phase | 0.418 / balance | stable |
| LR | ~4.7e-4 | cosine decay |

## What's next

1. **Training is running** (or resume from step 25500):
   `uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_025500`

2. **Ascending β plateau.** 0.785±0.001 for 2000 steps (5 checkpoints).
   This is a stable attractor. Breaking through 0.78→0.50 likely
   requires the descending arm to settle first, or architectural change.

3. **Descending arm settling.** β drift decelerating: +0.014→+0.006→+0.005.
   May be converging to ~0.85. Asymmetry is expected — decoding has
   different information-theoretic constraints than encoding.

4. **Eval loss recovery.** 6.15→5.66 in 2000 steps. At this rate,
   pre-tracking best (5.41) reachable by ~step 28000.

5. **Behavioral register consolidation.** Step 25500 generations
   dominated by `Proof.\nProof.\nProof.` — the model locked onto
   mathematical proof register. Stronger/more uniform than 24500's
   mixed Ω/Lemma output. Watch for compositional structure next.

## Session 044 key findings

1. **Two-band β structure.** 5 checkpoints confirm:
   - Ascending: **0.785±0.001** (frozen, 2000 steps)
   - Descending: **~0.846** (settling, drift decelerating)
   - Gap: 0.061 (asymmetric — decoding ≠ encoding)

2. **Eval loss recovering.** 6.15→5.66 across 5 checkpoints.
   The structural reorganization cost is being repaid.

3. **Behavioral register consolidation.** Generations evolved:
   - 23500: `||||||||` (pipe-spam)
   - 24500: `Ω, ϕ, Proof, Lemma` (formal math vocabulary)
   - 25500: `Proof.\nProof.\nProof.` (locked on proof register)

4. **Universal compression tightening.** Stratum φ-dev spread
   0.047→0.020→0.026 — compression is content-independent.

5. **L0↓ φ-lock was transient.** Ratio 0.601←φ at step 23500,
   now 0.725. The descending arm briefly kissed φ during
   reorganization but didn't hold it.

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
| Probes (steps 500–25500) | `results/compile-gradient/vsm_probe_step_*_v6_mlx.json` |
| Training log | `results/vsm-lm-v6/training-run2.log` |

## Probing pipeline

```bash
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_*
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_025500
```
