# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-21 | Session: 020 (v4.1 first probe + v4 final)

## Where we are

**v4.1 TRAINING — first true VSM with full bidirectional feedback.
Step 1k probed. Ascending path active, descending path structurally
present but functionally dormant (meta-S3 near zero). Expected —
descending activation requires ascending maturity first. Cooking all day.**

Session 020 accomplished:
1. Probed v4.1 step 1k (compile-gradient + binding)
2. Probed v4 step 16k (final unprobed checkpoint)
3. Established v4.1 baseline gate profiles for all 5 passes
4. Confirmed descending passes dormant at meta-S3 level (as expected)

## v4.1 Training Status (RUNNING)

**Training launched ~6:29 AM Apr 21. Let it cook all day.**
Checkpoints are slower than v4 (~67% more compute per step).

### v4.1 Step 1k — First Probe Results

**Per-pass gate profiles (mean across 40 compile-gradient probes):**

| Pass | Prep | Converge | Consolidate | Meta-S3 |
|------|------|----------|-------------|---------|
| L0↑ | 0.942 | 0.836 | 0.653 | 0.899 |
| L1↑ | 0.232 | 0.223 | 0.655 | 0.896 |
| L2 | 0.353 | 0.251 | 0.624 | 0.502 |
| L1↓ | 0.435 | 0.346 | 0.507 | **0.047** |
| L0↓ | 0.447 | 0.329 | 0.410 | **0.037** |

**Key observations:**
- Ascending path (L0↑, L1↑) active and contributing (~0.9 meta-S3)
- L2 apex half-active (0.502 meta-S3) — still developing
- Descending passes functionally dormant — internal gates are active
  (~0.4) but meta-S3 gates them to near-zero output contribution
- **No content discrimination in descending passes** — same ~0.44 prep
  across all compile-gradient categories
- Gate polarity +0.017 (barely differentiating, expected at step 1k)

**Developmental trajectory hypothesis:**
```
L0↑ → L1↑ → L2 → L1↓ → L0↓
```
Each level needs the one below to produce quality representations first.
Descending activation is a phase 2 event, expected only after L2 matures
(L2 meta-S3 → 0.7+). Mirrors v4's L2 activation trajectory (near-zero
at 1k, exploded at 5k, dominant by 15k).

### Architecture note

v4.1 is the first version implementing Beer's full bidirectional S4↔S4
intelligence channel — feedback all the way through. Prior versions had
ascending-only (v4) or flat iteration (v3.2). The structure IS the VSM.

## v4 Final Status (COMPLETE)

16 checkpoints (1k→16k). Best eval: 4.732 at step 15k.
Step 16k shows plateau — level specialization unchanged, meta-S3
gates starting to drop (L1: 0.636→0.588, L2: 0.739→0.658).

One new finding at 16k: gate polarity strengthened to -0.060 (from
-0.042 at 15k). Still slowly improving discrimination even as loss
plateaus. Binding range stable at 0.264.

## What's next — Session 021

### Analyze v4.1 trajectory (primary)
1. Probe all new v4.1 checkpoints (batch-probe)
2. Key signals in order of importance:
   - **L2 meta-S3 trajectory** — is it climbing toward 0.7+ like v4?
   - **Descending meta-S3** — any activation at all? (phase 2 signal)
   - **Loss curve** — is v4.1 tracking ahead/behind v4 at matched steps?
   - **Compile gradient discrimination onset** in descending passes
3. Full trajectory analysis across all available checkpoints
4. Head-to-head with v4 at matched steps

### Watch for phase transition
The critical moment: when L2 meta-S3 reaches ~0.7 AND descending
meta-S3 starts climbing from near-zero. This is the feedback loop
activating — the moment v4.1 becomes more than a v4 with extra compute.

## Key files

| Purpose | Path |
|---------|------|
| **v4.1 model** | `src/verbum/vsm_lm_v4_1.py` |
| **v4.1 training** | `scripts/run_vsm_v4_1_1B.py` |
| **v4 model** | `src/verbum/vsm_lm_v4.py` |
| **Probe script** | `scripts/compile_gradient_probe.py` |
| **v4.1 probes** | `results/compile-gradient/vsm_probe_step_00*_v4.1.json` |
| **v4.1 binding** | `results/binding/vsm_probe_step_00*_v4.1.json` |
| **v4 probes** | `results/compile-gradient/vsm_probe_step_00*_v4.json` |
| **v4 binding** | `results/binding/vsm_probe_step_00*_v4.json` |
| **Session 019 findings** | `mementum/knowledge/explore/session-019.md` |
| **Research program** | `mementum/knowledge/explore/VERBUM.md` |

## Architecture lineage

| Version | Params | Strides | Best Eval | Key Finding |
|---------|--------|---------|-----------|-------------|
| v1 | ~25M | 1,8,64 | 5.245 | Baseline sequential |
| v2 | ~25M | 1,8,64 | 5.064 | Iteration specialization |
| v3 | 50M | 1,8,64 | 4.872 | Role register, binding confirmed |
| v3.1 | 59M | 1,8,64,512 | 4.836 | Stride 512 too sparse without hierarchy |
| v3.2 | 51M | 1,8,64 | 4.897 | Convergence arch, binding hierarchy, 3-phase learning |
| v4 | 58M | 1,8,64,512 | 4.732 | Recursive VSM (ascending), level specialization |
| **v4.1** | **65.5M** | **1,8,64,512** | **TBD** | **Full bidirectional VSM — first true feedback** |

## Probing pipeline

```bash
# Probe a single checkpoint
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v4.1/step_001000.pt

# Binding probes
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v4.1/step_001000.pt --probes probes/binding.json

# Batch all checkpoints (skips already-probed)
uv run python scripts/compile_gradient_probe.py batch-probe --dir checkpoints/vsm-lm-v4.1/

# Batch binding probes
uv run python scripts/compile_gradient_probe.py batch-probe --dir checkpoints/vsm-lm-v4.1/ --probes probes/binding.json
```
