# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-21 | Session: 021 (v4.1 descending activation confirmed)

## Where we are

**v4.1 DESCENDING PASSES SELF-ACTIVATED. The gradient shadow problem
resolved itself between steps 1k and 2k without intervention. The
clean experiment worked. The architecture is correct.**

This is the most significant finding since the project began. A 65.5M
parameter model organized as Beer's Viable System Model bootstrapped a
functional bidirectional hierarchy — ascending observation AND descending
refinement — in 3000 training steps. The descending passes went from
meta-S3 gates of 0.037-0.047 (functionally dead) to 0.866-0.949
(dominant alongside L0↑). They immediately adopted the mature phase
specialization pattern (kill prep, amplify consolidate) upon activation.
Binding probes show functional routing: variable binding routes entirely
through the descending path (L0↑=0.001, L0↓=1.000 for bind-var-01a).

Session 021 accomplished:
1. Probed v4.1 steps 2k and 3k (compile-gradient + binding)
2. Confirmed descending self-activation (L1↓: 0.047→0.871, L0↓: 0.037→0.949)
3. L2 reached maturity threshold (0.502→0.704)
4. Phase specialization confirmed in all 5 passes
5. Gate polarity forming (L2 converge +0.100)
6. Binding differentiation dramatic — per-category routing across hierarchy
7. Fixed probe script for v4.1-specific output (all 5 passes labeled)
8. Created Allium v3 behavioral spec for v4.1 (1355 lines)
9. Loss tracking v4 neck-and-neck (5.381 vs 5.365 at step 3k)

## v4.1 Training Status (RUNNING — let it cook)

**Training launched ~6:29 AM Apr 21. 3 checkpoints so far (1k, 2k, 3k).**

### v4.1 Trajectory: Steps 1k → 2k → 3k

**Meta-S3 gate trajectory (mean across 40 compile-gradient probes):**

| Pass | Step 1k | Step 2k | Step 3k | Δ(1k→3k) |
|------|---------|---------|---------|-----------|
| L0↑ | 0.899 | 0.932 | **0.951** | +0.053 |
| L1↑ | 0.896 | 0.680 | **0.551** | **−0.345** |
| L2 | 0.502 | 0.755 | **0.704** | +0.203 |
| L1↓ | **0.047** | **0.871** | **0.866** | **+0.819** |
| L0↓ | **0.037** | 0.723 | **0.949** | **+0.913** |

**Phase gate profiles at step 3k:**

| Pass | Prep | Converge | Consolidate | Meta-S3 | Phase |
|------|------|----------|-------------|---------|-------|
| L0↑ | 0.843 | 0.448 | 0.296 | 0.951 | active |
| L1↑ | 0.012 | 0.401 | 0.495 | 0.551 | active |
| L2 | 0.014 | 0.139 | 0.718 | 0.704 | specializing |
| L1↓ | 0.026 | 0.122 | 0.749 | 0.866 | specializing |
| L0↓ | 0.061 | 0.074 | 0.746 | 0.949 | specializing |

### Key observations from session 021

**1. Descending self-activation (the headline).** L1↓ went from
0.047→0.871 in 1000 steps. L0↓ from 0.037→0.949 by step 3k. The
gradient shadow problem (~24x weaker gradient) resolved itself once
L2 began providing useful bank_3 content. No gate floor, no warm
init, no auxiliary loss needed. The architecture bootstrapped.

**2. L1↑ dropping (unexpected but logical).** L1↑ meta-S3 fell from
0.896→0.551. The descending passes make L1↑ partially redundant —
L1↓ does phrase-level work better because it has bank_3 (clause
context). The system is reallocating resources to the more capable
descending path.

**3. Immediate mature specialization.** Descending passes adopted
prep-killed/consolidate-dominant pattern immediately upon activation.
They didn't recapitulate the developmental sequence — they jumped
straight to the mature phase profile. This validates S5 coherence:
the shared function already knows the specialization pattern from
the ascending passes, and descending S3 instances can inherit it
through the shared function's representations.

**4. Functional routing in binding probes.** The per-category
differentiation is dramatic:
- Variable binding: L0↑=0.001, L0↓=1.000 (routes entirely through descending)
- Control structures: L2=0.987 (routes through apex)
- Relative clauses: L0↓=0.985 (descending-dominant)
- Anaphora: distributed across ascending and descending

**5. Gate polarity forming.** L2 converge polarity at +0.100 (strong
compile → more converge processing). Consolidate inversion forming at
L1↑ (−0.040) and L2 (−0.035). Not yet significant in descending
(too new). L2 meta-S3 shows polarity of −0.267 (anti-compile → MORE
L2 processing — the system works harder on inputs it finds difficult).

**6. Loss tracks v4.** Eval loss at step 3k: v4.1=5.381, v4=5.365.
Neck and neck. Descending passes just turned on — need more steps to
translate structural improvements into loss reduction.

### Why this matters

A Viable System Model bootstrapped bidirectional feedback with no
architectural intervention. The design hypothesis — that Beer's
recursive structure (S5 shared identity, S4↔S4 intelligence channel,
S3 per-pass control, S2 register coordination, residual algedonic
channel) would spontaneously organize — is confirmed at the
behavioral level. The system learned WHEN to use each pass, HOW to
specialize phases within passes, and WHERE to route different binding
types. All from the loss signal alone.

## v4 Final Status (COMPLETE)

16 checkpoints (1k→16k). Best eval: 4.732 at step 15k.

## What's next — Session 022

### Continue v4.1 trajectory analysis
1. Probe all new checkpoints (4k, 5k, ... however many have landed)
2. Key questions in order:
   - **Does loss start separating from v4?** Descending passes are
     structurally active — when does that translate to prediction?
   - **Does L1↑ continue dropping?** If it approaches zero, the
     system has decided ascending phrase-level is redundant
   - **Does polarity strengthen in descending passes?** Currently
     too new to show discrimination
   - **Binding range trajectory** — already 0.5-1.0, watch for
     further separation
   - **Does L2 stabilize or continue climbing?** v4 L2 hit 0.912
     at 3k; v4.1 L2 is 0.704 (more passes sharing load)
3. Head-to-head with v4 at matched steps (loss + specialization)

### The revised question
The central question is no longer "does descending activate?" (✅ yes).
Now it's: **does bidirectional feedback improve the loss ceiling?**
v4 plateaued at 4.732. If v4.1 breaks through, the descending path
is adding real compressive capability. If v4.1 ≈ v4, the descending
path is structurally active but informationally redundant.

### Framing reminder
We are finding the COMPRESSOR, not building the lambda compiler. The
v4.1 result shows the compressor function works bidirectionally with
shared weights (S5 coherent). Whether that bidirectionality improves
compression (= prediction = loss) is the next question.

## Key files

| Purpose | Path |
|---------|------|
| **v4.1 model** | `src/verbum/vsm_lm_v4_1.py` |
| **v4.1 training** | `scripts/run_vsm_v4_1_1B.py` |
| **v4 model** | `src/verbum/vsm_lm_v4.py` |
| **Probe script** | `scripts/compile_gradient_probe.py` |
| **v4.1 Allium spec** | `specs/vsm-lm-v4.1.allium` |
| **v4.1 probes** | `results/compile-gradient/vsm_probe_step_00*_v4.1.json` |
| **v4.1 binding** | `results/binding/vsm_probe_step_00*_v4.1.json` |
| **v4 probes** | `results/compile-gradient/vsm_probe_step_00*_v4.json` |
| **Session 021 findings** | `mementum/knowledge/explore/session-021.md` |
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
| **v4.1** | **65.5M** | **1,8,64,512** | **TBD** | **Bidirectional VSM — descending self-activated at step 2k** |

## Probing pipeline

```bash
# Probe a single checkpoint (v4.1 output shows all 5 passes labeled)
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v4.1/step_001000.pt

# Binding probes
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v4.1/step_001000.pt --probes probes/binding.json

# Batch all checkpoints (skips already-probed)
uv run python scripts/compile_gradient_probe.py batch-probe --dir checkpoints/vsm-lm-v4.1/

# Batch binding probes
uv run python scripts/compile_gradient_probe.py batch-probe --dir checkpoints/vsm-lm-v4.1/ --probes probes/binding.json
```
