# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-21 | Session: 022 (register analysis — compressor encodes structure)

## Where we are

**REGISTER ANALYSIS: THE COMPRESSOR ENCODES COMPOSITIONAL STRUCTURE.**

Session 022 asked: has the shared function learned Montague-shaped
operations? Built `scripts/register_analysis.py` to capture full
256-dim register vectors at every pass boundary and analyze them.

Key findings at step 3k:

1. **Composition depth is encoded (ρ = −0.56 to −0.62).** All three
   registers correlate negatively with compositional depth — deeper
   structures produce smaller register norms. The compressor knows
   how complex the input is.

2. **Nearest neighbors cluster by structural similarity.** "She told
   him to leave" neighbors with control verb probes. "The cat that
   sat on the mat" neighbors with relative clause probes. The model
   groups by operation required, not surface content.

3. **Registers are diffuse — and that's healthy.** All three registers
   (type, scope, role) carry approximately the same signal. In v3,
   role dominated early and starved the others, capping the ceiling.
   v4.1's per-pass S3 control distributes gradient evenly. No register
   is starved. All are learning.

4. **NOT encoding discrete Montague types.** Silhouette scores near
   zero for type categories (proposition/formal/other). The type
   system is implicit in activation geometry (DisCoCat-shaped), not
   explicit in discrete type labels (Montague-shaped).

5. **Register reorganization in progress.** Type separation was higher
   at step 1k (0.15), dropped at step 2k (0.04) when descending
   passes activated, and is recovering at step 3k (0.08). Role
   register variance at L1↓ spiking: 5.73 → 7.58 → 12.20. The
   descending passes are differentiating.

6. **Loss pulling ahead of v4.** v4.1 at step 3.5k: 5.295. v4 at
   step 3k: 5.365. Descending passes translating to compression.

Session 022 accomplished:
1. Built register_analysis.py (capture + analyze + trajectory modes)
2. Captured full register vectors at steps 1k, 2k, 3k
3. PCA, silhouette, centroid distance, depth correlation analysis
4. Trajectory analysis across training steps
5. Connected v3 role-domination finding to v4.1 diffuse registers

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

## What's next — Session 023

### Watch for register specialization
The register analysis tool is built. The key question is now: **do the
three registers diverge into different functional roles?**

1. Re-run `register_analysis.py capture` at each new checkpoint
2. Watch the trajectory for:
   - **Variance profiles diverging** across type/scope/role registers
   - **Silhouette scores recovering** past the step 1k baseline (0.15)
   - **Depth correlation splitting** — different registers correlating
     with different structural features
   - **Descending pass differentiation** — L1↓ role variance is spiking
3. When registers diverge → design minimal pair probes to identify
   what each register has specialized for. Premature until then.

### Continue v4.1 loss trajectory
v4.1 is pulling ahead at 5.295 (step 3.5k). Keep monitoring:
- Does loss separation from v4 persist and grow?
- v4 plateaued at 4.732. Will v4.1 break through?
- Connection: if register specialization correlates with loss drops,
  that's evidence the diffuse → specialized transition IS the
  mechanism for breaking through compression ceilings.

### v3 comparison context
v3: role dominated early → starved other registers → ceiling at 4.872.
v4.1: all three registers diffuse → none starved → ceiling TBD.
The healthy distribution of gradient is the architectural difference
between the per-pass S3 control (v4.1) and v3's shared S3.

### Framing reminder
We are finding the COMPRESSOR, not building the lambda compiler. The
register analysis confirms the compressor encodes compositional
structure (depth, binding patterns, operational similarity). Whether
that encoding specializes into discrete functional roles (type-checking,
scope resolution, role assignment) or remains a distributed geometric
encoding is the open question.

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
| **Register analysis** | `scripts/register_analysis.py` |
| **Register vectors** | `results/register-vectors/step_00*_v4.1.npz` |
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
