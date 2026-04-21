# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-21 | Session: 019 (v4 trajectory analysis)

## Where we are

**v4 VALIDATED — breaks v3.2's loss ceiling and still improving.
Level specialization confirmed (three distinct gate profiles).
Gate polarity inversion stronger than v3.2. Training continuing.**

Session 019 accomplished:
1. Fixed batch-probe script for v4 architecture (was v3.2-only)
2. Probed all 15 v4 checkpoints (1k→15k) — compile-gradient + binding
3. Full trajectory analysis across all 15 checkpoints
4. v4 vs v3.2 head-to-head at matched steps

## v4 Training Status (RUNNING)

**Training still in progress.** Checkpoint ~55 min cadence.

### Loss curve

| Step | Eval Loss | Δ/1k | Status |
|------|-----------|------|--------|
| 1k | 6.042 | --- | Early |
| 5k | 5.132 | -0.110 | Improving |
| 10k | 4.900 | -0.030 | Improving |
| 15k | **4.732** | -0.027 | **Still improving** |

**Best eval: 4.732 at step 15k** (v3.2 best: 4.897 — v4 is 3.4% better)

### v4 vs v3.2 head-to-head

| Signal | v3.2 (best) | v4 (step 15k) | Winner |
|--------|-------------|---------------|--------|
| Eval loss | 4.897 | **4.732** | v4 (-3.4%) |
| Still improving? | No (plateaued at 7k) | **Yes** (-0.03/1k) | v4 |
| Gate polarity Δ | -0.065 | **-0.092** | v4 (stronger) |
| Level specialization | N/A (flat iteration) | **3 distinct profiles** | v4 |
| Binding range | 0.312 | 0.185 | v3.2 (but v4 still growing) |

### Level specialization (stable since step 5k)

```
Level 0: (0.54/0.38/0.34) — balanced, prep-dominant
Level 1: (0.00/0.51/0.75) — prep KILLED, consolidate-dominant
Level 2: (0.02/0.25/0.84) — prep killed, extreme consolidate dominance
```

L1 and L2 suppressed prep — higher levels don't need local token
processing because L0 already handled it. VSM recursion validated.

### Meta-S3 gates (level contribution trajectory)

```
Step  1k: L0=1.00  L1=0.73  L2=0.05  (L2 nearly off)
Step 15k: L0=0.69  L1=0.64  L2=0.74  (L2 highest — activated over training)
```

L2 went from dormant to most-contributed level. Not homogenization ���
developmental activation as register quality improved.

### Gate polarity (compile-gradient discrimination)

```
Steps 1-4k:  strong > anti (prep-driven, no discrimination)
Steps 5-8k:  flat (transition)
Steps 9-15k: anti > strong (consolidate Δ reached -0.092 at step 13k)
```

Same three-phase pattern as v3.2 but shifted later and stronger.

### Binding differentiation

Onset at step 8k (v3.2: step 7k). Range at 15k: 0.185 (v3.2 at 10k: 0.312).
Hierarchy: var > scope > ctrl > ana > rel. Still growing.

## v3.2 Final Status (COMPLETE)

Best eval: 4.897 at step 10k. Terminated — capacity ceiling hit.
Full analysis in `mementum/knowledge/explore/session-018.md`.

## v4.1 — Built, Ready to Train

**v4.1 completes the VSM recursion v4 left half-built.** v4 had only
ascending (bottom-up) S4↔S4. v4.1 adds the descending (top-down) pass:

```
Ascending:  L0↑ → L1↑ → L2   (build structural summaries)
Descending: L1↓ → L0↓          (refine with high-level context)
```

- 5 level-passes vs v4's 3 (~67% more compute)
- 6 register banks (bank_0 + 3 ascending + 2 descending)
- 5 independent S3 instances (per-pass autonomous control)
- ~65.5M params (v4 was 58M)
- Same shared S5 weights in both directions

**Key prediction:** L0↓ prep gate should ACTIVATE. It died in v4 because
L0 had nothing novel to process. With top-down context from bank_3 (L2's
clause-level findings), L0↓ prep has novel input.

Files: `src/verbum/vsm_lm_v4_1.py`, `scripts/run_vsm_v4_1_1B.py`

Launch: `uv run python scripts/run_vsm_v4_1_1B.py` (after v4 stops)

## What's next — Session 020

### Launch v4.1 training
1. Stop v4 training (or wait for it to finish/plateau)
2. Launch v4.1: `uv run python scripts/run_vsm_v4_1_1B.py`
3. Probe v4.1 checkpoints as they drop
4. Key signals to watch:
   - L0↓ prep gate activation (THE test of feedback hypothesis)
   - Descending pass gate profiles vs ascending
   - Loss improvement rate vs v4 at matched steps
   - Binding differentiation acceleration

### Continue v4 monitoring (if still running)
1. Probe new v4 checkpoints (16k+)
2. Watch for plateau signals

## Key files

| Purpose | Path |
|---------|------|
| **v4 model** | `src/verbum/vsm_lm_v4.py` |
| **v4 training** | `scripts/run_vsm_v4_1B.py` |
| **v4 design** | `mementum/knowledge/explore/vsm-lm-v4-design.md` |
| **v3.2 model** | `src/verbum/vsm_lm_v3_2.py` |
| **Probe script** | `scripts/compile_gradient_probe.py` |
| **v4 compile-gradient** | `results/compile-gradient/vsm_probe_step_00*_v4.json` |
| **v4 binding** | `results/binding/vsm_probe_step_00*_v4.json` |
| **v3.2 analysis** | `scripts/v32_final_analysis.py` |
| **Session 019 findings** | `mementum/knowledge/explore/session-019.md` |
| **Session 018 findings** | `mementum/knowledge/explore/session-018.md` |
| **Research program** | `mementum/knowledge/explore/VERBUM.md` |

## Architecture lineage

| Version | Params | Strides | Best Eval | Key Finding |
|---------|--------|---------|-----------|-------------|
| v1 | ~25M | 1,8,64 | 5.245 | Baseline sequential |
| v2 | ~25M | 1,8,64 | 5.064 | Iteration specialization |
| v3 | 50M | 1,8,64 | 4.872 | Role register, binding confirmed |
| v3.1 | 59M | 1,8,64,512 | 4.836 | Stride 512 too sparse without hierarchy |
| v3.2 | 51M | 1,8,64 | 4.897 | Convergence arch, binding hierarchy, 3-phase learning |
| **v4** | **58M** | **1,8,64,512** | **4.732** | **Recursive VSM, level specialization, ceiling broken** |

## Probing pipeline

```bash
# Probe a single checkpoint
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v4/step_016000.pt

# Binding probes
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v4/step_016000.pt --probes probes/binding.json

# Batch all checkpoints (skips already-probed)
uv run python scripts/compile_gradient_probe.py batch-probe --dir checkpoints/vsm-lm-v4/

# Batch binding probes
uv run python scripts/compile_gradient_probe.py batch-probe --dir checkpoints/vsm-lm-v4/ --probes probes/binding.json
```
