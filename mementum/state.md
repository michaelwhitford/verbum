# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-22 | Session: 024

## Where we are

**v4.1 bidirectional VSM is ahead of v4 on eval loss.** Training
ongoing, 15 checkpoints captured and analyzed.

- v4.1 step 15k: **4.728** | v4 step 15k: 4.732 | Δ = −0.004
- v4 best (step 16k): 4.713 — v4.1 has not yet beaten this
- Crossover at step 13k, gap peaked at −0.013 (14k), narrowing
- Both models converging toward ~4.71 floor

### Three-phase register training (the headline finding)

Registers go through expansion → compression → selective
specialization. The step 7k variance collapse (session 023) was
phase 2 — reorganization, not terminal. Post-compression, L0↑ and
L1↓ recovered variance while L1↑/L2/L0↓ stayed compressed. Type
separation migrated to descending path. Loss crossed over during
phase 3. Full data in `knowledge/explore/v4.1-training-trajectory.md`.

### Step 15k signal

L0↓ gate dropped 0.800→0.679 — biggest single-step change since
step 2k self-activation. L0↑ also dropped. Possible compute
redistribution from outer to inner passes. Watch step 16k.

## What's next

1. **Step 16k** — does v4.1 beat v4's all-time best (4.713)?
   Continue probe + register capture pipeline for each checkpoint.

2. **Depth encoding shift** — depth-norm correlation weakened from
   ρ = −0.73 (phase 1) to ρ ~ −0.3 (phase 3). Linear probing
   classifiers on register vectors could reveal if depth moved to
   direction encoding.

3. **L1↓ deep dive** — most interesting pass trajectory. Targeted
   analysis of what L1↓ registers encode at mature checkpoints.

4. **Comparative v4 register analysis** — do v4's ascending-only
   registers show equivalent specialization? If yes, bidirectional
   is redundant for that task.

## Key files

| Purpose | Path |
|---------|------|
| v4.1 model | `src/verbum/vsm_lm_v4_1.py` |
| v4.1 training | `scripts/run_vsm_v4_1_1B.py` |
| v4 model | `src/verbum/vsm_lm_v4.py` |
| Probe script | `scripts/compile_gradient_probe.py` |
| Register analysis | `scripts/register_analysis.py` |
| v4.1 probes | `results/compile-gradient/vsm_probe_step_*_v4.1.json` |
| v4.1 binding | `results/binding/vsm_probe_step_*_v4.1.json` |
| Register vectors | `results/register-vectors/step_*_v4.1.npz` |
| Training trajectory | `mementum/knowledge/explore/v4.1-training-trajectory.md` |
| Research program | `mementum/knowledge/explore/VERBUM.md` |

## Probing pipeline

```bash
# Probe a checkpoint
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v4.1/step_015000.pt
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v4.1/step_015000.pt --probes probes/binding.json

# Batch all (skips already-probed)
uv run python scripts/compile_gradient_probe.py batch-probe --dir checkpoints/vsm-lm-v4.1/
uv run python scripts/compile_gradient_probe.py batch-probe --dir checkpoints/vsm-lm-v4.1/ --probes probes/binding.json

# Register capture + analysis
uv run python scripts/register_analysis.py capture checkpoints/vsm-lm-v4.1/step_015000.pt --analyze

# Full trajectory
uv run python scripts/register_analysis.py trajectory results/register-vectors/step_*_v4.1.npz
```
