# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-26 | Session: 044

## Where we are

**v6.1 at step 23500 (29% of 3B). Hilberg β phase transition: all 5
arms converged to 0.76–0.83 lockstep band (was 1.1+/chaotic). First
time ascending and descending arms show same self-similar regime.
Eval loss regressed 5.45→6.15 — structural reorganization cost.
Flip tracking live: 250 reversals / 232K flips (0.108%). Training
paused at ~step 23550. Resume to watch whether lockstep holds.**

## Current snapshot (step 23500)

| Metric | Value | Trend |
|--------|-------|-------|
| Eval loss | 6.154 (best: 5.414 @ 17500) | ⚠️ regressed post-flip-tracking |
| Hilberg β (all arms) | **0.76–0.83 lockstep** | ↓↓ phase transition from 1.1+ |
| β L0↑/L1↑/L2/L1↓/L0↓ | 0.78/0.76/0.79/0.83/0.80 | all coherent for first time |
| L1_asc ratio | 0.870 (was 0.560 near φ) | moved away from φ during reorg |
| L0_desc ratio | **0.601←φ** | descending arm locked to φ! |
| Stride compression | flattened: 0.73–0.95 all strides | was gradient 0.32–0.83 |
| Stratum spread | 1.35 | widened from 0.70 (reorg cost) |
| Total flips | 232K (0.66%) | +10K since resume |
| Reversals | 250 (0.108%) | very low oscillation |
| Unique ever flipped | 9,541 (0.027%) | narrow flip set |
| r̄ / phase | 0.474 / balance | settled from explore |
| LR | ~4.9e-4 | cosine decay |
| Flip tracking | **LIVE** — cooldown=4 intervals | first checkpoint with data |

## What's next

1. **Resume training.** Command:
   `uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_023500`
   Training paused at ~step 23550.

2. **Step 24000 is the critical checkpoint.** Distinguishes:
   - Lockstep holds + loss recovers → genuine phase shift
   - β bounces back to 1.1+ → transient from resume shock
   - Loss keeps climbing → destabilization

3. **Watch the lockstep band.** If all 5 arms stay within ~0.07 of
   each other as β descends toward 0.5, that's the holographic
   compressor signature — same shape going in and coming out.

4. **Watch s1 ratio.** Moved from φ (0.62→0.73). If it returns
   toward φ while long strides hold, the model is re-differentiating.
   If it stays at 0.73, new compression regime.

5. **Eval loss recovery.** Pre-tracking best was 5.414 @ 17500.
   Post-LR-jump best was 5.441 @ 22500. Now at 6.15. Full recovery
   would validate that the reorganization was productive.

## Session 044 key findings

1. **Hilberg β phase transition.** 500 steps transformed all arms:
   L0↑: 1.10→0.78, L1↑: 1.11��0.76, L2: 1.26→0.79,
   L1↓: -0.22→0.83, L0↓: N/A→0.80. Band width: 0.07.

2. **Lockstep = symmetric compression shape.** Ascending (encoding)
   and descending (decoding) arms converged to the same self-similar
   regime. The holographic compressor should look the same in both
   directions — and now it does.

3. **Descending arm awakened.** S3 gates for L1↓ and L0↓ jumped
   from ~0.6 to ~0.99. The model is actually using all 5 passes.

4. **Stride flattening.** All strides compressed to 0.73–0.95 band.
   The per-stride gradient collapsed — what drove β down was
   uniform compression across scales, not fine-grained φ-locking.

5. **Cost: eval loss +0.70.** Structural reorganization isn't free.
   The model traded generalization for internal geometric coherence.
   Prior precedent (LR jump): loss recovered within ~2000 steps.

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
| Probes (steps 500–23500) | `results/compile-gradient/vsm_probe_step_*_v6_mlx.json` |
| Training log | `results/vsm-lm-v6/training-run2.log` |

## Probing pipeline

```bash
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_*
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_023500
```
