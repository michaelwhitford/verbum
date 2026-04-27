# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-27 | Session: 044

## Where we are

**v6.1 at step 25000 (30% of 3B). Lockstep CONFIRMED across 4
checkpoints. Ascending β plateaued at 0.786 — rock-stable. Descending
β drifting up 0.81→0.84 (still searching). Eval loss recovering:
6.15→5.72. Generations shifting from pipe-spam to formal math
vocabulary (Ω, Proof, Lemma). Not λ yet but register is changing.**

## Current snapshot (step 25000)

| Metric | Value | Trend |
|--------|-------|-------|
| Eval loss | 5.724 (best: 5.414 @ 17500) | ↓ recovering from 6.15 |
| β ascending (L0↑/L1↑/L2) | **0.78/0.78/0.80** | plateaued, band=0.023 |
| β descending (L1↓/L0↓) | **0.85/0.83** | ↑ drifting up (was 0.83/0.80) |
| β gap (desc−asc) | **0.054** | ↑ growing (was 0.035) |
| L0_desc ratio | 0.694 (was 0.601←φ) | drifting from φ |
| Mean φ-compression | 0.813 | ↑ (was 0.787) |
| Stratum φ-dev spread | **0.020** | ↓↓ content-independent |
| Stratum loss spread | 1.54 | stable |
| Total flips | 258K (0.73%) | steady ~8.6K/500 steps |
| Reversals | 292 (0.113%) | very low, stable |
| Unique ever flipped | tracking | see flip_tracking.npz |
| r̄ / phase | 0.398 / balance | stable |
| LR | ~4.8e-4 | cosine decay |

## What's next

1. **Training is running** (or resume from step 25000):
   `uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_025000`

2. **Ascending β plateau.** 0.786±0.001 for 1500 steps. Either:
   - This is the floor for the current regime (needs architectural change)
   - It will resume descent after the descending arm stabilizes

3. **Descending arm diverging.** β 0.81→0.84 while ascending holds 0.78.
   Gap growing 0.035→0.054. The descending arm may need more training
   to find its shape, or the asymmetry is structural (decoding ≠ encoding).

4. **Eval loss recovery.** 6.15→5.72 in 1500 steps. At this rate,
   pre-tracking best (5.41) reachable by ~step 27000.

5. **Behavioral shift.** Generations at 24500 show formal math vocabulary
   (Ω, ϕ, Γ, Proof, Lemma). Not λ yet but the model is finding the
   right register. Watch for λ-like structure in future checkpoints.

## Session 044 key findings

1. **Lockstep confirmed.** Not a transient — 4 consecutive checkpoints
   show all arms in 0.78–0.85 band (was 1.1+/chaotic pre-tracking).

2. **Two-band structure emerging:**
   - Ascending: 0.786±0.001, band 0.023 (frozen)
   - Descending: 0.84±0.01, band 0.020 (drifting up)
   - The model found the ascending shape but descending is still moving.

3. **Eval loss recovering.** 6.15→5.88→5.79→5.72 across 4 checkpoints.
   The structural reorganization cost is being repaid.

4. **Universal compression tightening.** Stratum φ-dev spread
   0.047→0.020 — compression becoming content-independent.

5. **L0↓ φ-lock was transient.** Ratio 0.601←φ at step 23500,
   now 0.694. The descending arm briefly kissed φ during reorganization
   but didn't hold it.

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
| Probes (steps 500–25000) | `results/compile-gradient/vsm_probe_step_*_v6_mlx.json` |
| Training log | `results/vsm-lm-v6/training-run2.log` |

## Probing pipeline

```bash
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_*
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_025000
```
