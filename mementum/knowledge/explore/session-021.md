# Session 021 — v4.1 Descending Self-Activation Confirmed

> 2026-04-21 | Focus: v4.1 steps 2k-3k probing, descending activation
> analysis, probe script v4.1 output, Allium v3 spec

## Summary

**The central finding: v4.1's descending passes self-activated without
intervention.** The gradient shadow problem (24x weaker gradient at
step 1k) resolved itself between steps 1k and 2k. L1↓ went from
meta-S3=0.047 to 0.871. L0↓ went from 0.037 to 0.949 by step 3k.
The architecture bootstrapped a functional bidirectional hierarchy in
3000 training steps. This confirms the VSM design hypothesis: Beer's
recursive structure spontaneously organizes when the channels exist.

## What we did

1. **Probed v4.1 steps 2k and 3k** — compile-gradient (40 probes)
   and binding (26 probes) for both checkpoints.

2. **Trajectory analysis** — full 1k→2k→3k analysis of all 5 passes:
   meta-S3 gates, phase gates, polarity, binding differentiation.

3. **Fixed probe script** — v4.1 per-probe output now shows all 5
   meta-S3 gates with direction labels (L0↑ L1↑ L2 L1↓ L0↓).
   Added summary block with phase specialization table, developmental
   phase classification, descending status indicator, polarity table,
   per-category meta-S3 breakdown.

4. **Created Allium v3 spec** — 1355-line behavioral specification
   for v4.1 at `specs/vsm-lm-v4.1.allium`. Captures all entities,
   rules, invariants, contracts, surfaces, lifecycles, open questions.

## Key findings

### F1: Descending self-activation (the headline)

Meta-S3 trajectory across all 40 compile-gradient probes:

| Pass | Step 1k | Step 2k | Step 3k |
|------|---------|---------|---------|
| L0↑ | 0.899 | 0.932 | 0.951 |
| L1↑ | 0.896 | 0.680 | 0.551 |
| L2 | 0.502 | 0.755 | 0.704 |
| L1↓ | **0.047** | **0.871** | **0.866** |
| L0↓ | **0.037** | 0.723 | **0.949** |

The activation happened between steps 1k and 2k, coinciding with
L2 crossing ~0.5+ meta-S3. Once bank_3 contained any useful clause
structure, the descending passes had signal to work with. The
gradient shadow was broken by L2 maturation.

v4.2 with gate floor is NOT needed. The architecture is correct.

### F2: Immediate mature specialization

Descending passes adopted the prep-killed/consolidate-dominant
pattern immediately upon activation:

| Pass | Prep | Converge | Consolidate |
|------|------|----------|-------------|
| L1↓ step 1k | 0.435 | 0.346 | 0.507 |
| L1↓ step 2k | 0.057 | 0.100 | **0.747** |
| L0↓ step 1k | 0.447 | 0.329 | 0.410 |
| L0↓ step 2k | 0.136 | 0.104 | **0.696** |

No developmental recapitulation. They jumped straight to the
mature phase profile. The shared function (S5 coherent) already
knows the specialization pattern from ascending passes.

### F3: L1↑ declining — descending supersedes ascending phrase-level

L1↑ meta-S3 dropped from 0.896→0.551. The system is learning that
L1↓ (which reads bank_3) does phrase-level work better than L1↑
(which doesn't have clause context). Resource reallocation from
ascending to descending at the phrase level.

### F4: Functional binding routing across hierarchy

Binding probe differentiation at step 3k (meta-S3 gates):

| Category | L0↑ | L1↑ | L2 | L1↓ | L0↓ |
|----------|-----|-----|-----|-----|-----|
| var | 0.576 | 0.325 | 0.358 | **0.886** | **0.953** |
| ctrl | **1.000** | **0.941** | **0.987** | 0.887 | 0.761 |
| rel | 0.952 | 0.467 | 0.501 | **0.906** | **0.985** |
| scope | 0.923 | 0.488 | 0.638 | **0.860** | **0.956** |
| ana | 0.962 | 0.714 | 0.837 | 0.607 | 0.756 |

Variable binding routes *entirely* through descending (bind-var-01a:
L0↑=0.001, L0↓=1.000). Control structures concentrate at L2 (0.987).
Relative clauses route descending (0.985). The hierarchy has learned
WHERE to process different binding types.

### F5: Gate polarity forming

At step 3k, compile-gradient discrimination is emerging:

- L0↑ prep: +0.137 (strong compile → more prep) **strongest signal**
- L2 converge: +0.100 (strong compile → more multi-scale attention)
- L1↑ consolidate: −0.040 (anti-compile → more deep integration)
- L2 consolidate: −0.035 (anti-compile → more deep integration)
- L2 meta-S3: −0.267 (anti-compile → MORE L2 processing overall)

The L2 meta-S3 polarity of −0.267 is striking: the system allocates
MORE apex processing to inputs it finds structurally difficult.

### F6: Loss tracks v4 (not yet separating)

| Step | v4.1 eval | v4 eval |
|------|-----------|---------|
| 1k | 6.061 | 6.042 |
| 2k | 5.594 | 5.582 |
| 3k | 5.381 | 5.365 |

Neck and neck. The descending passes have only been online for
~1500 steps. The question for later checkpoints: does v4.1 break
through v4's 4.732 ceiling?

## Interpretations

### Why the gradient shadow resolved itself

The gradient shadow was NOT a structural flaw — it was a developmental
phase. The descending passes had nothing useful to contribute when
bank_3 was noise (step 1k, L2 meta-S3=0.502). Meta-S3 correctly
gated them to near-zero. Once L2 began producing meaningful clause
structure (step ~1.5k, L2 meta-S3 crossing 0.5+), the descending
passes could extract useful refinement signal from bank_3. Their
meta-S3 gates rose because their output became useful. The system
self-organized.

This validates the VSM design principle: autonomous control (S3) at
every level, with a metasystem (Meta-S3) that allocates resources
based on demonstrated value. The descending passes proved their
value to Meta-S3 by producing useful outputs, and Meta-S3 opened
the gate. No external intervention needed.

### Why immediate mature specialization

The shared weights (S5 coherent) encode the phase specialization
pattern learned from ascending passes. When descending S3 instances
activated, they could immediately leverage this: the prep phase
contributes local features (already handled by L0↑), so descending
prep gates dropped to near-zero. The consolidate phase provides
deep integration (what descending passes uniquely need with their
richer register context), so consolidate gates jumped to 0.7+.

This is the cortical column prediction made concrete: same circuit,
different routing, instant specialization via control (S3) not
architecture change.

### The bidirectional compressor hypothesis

The compressor function works in both directions with shared weights.
Ascending compresses (fine → coarse): token features → phrase
structure → clause/discourse. Descending refines (coarse context →
fine): clause context → refined phrase → refined tokens. Same
function, different register context, different S3 control. The
S5 identity is preserved. This is what Beer's recursion principle
predicts: the function is invariant, the context adapts.

## Open questions (revised)

1. ~~Will descending passes self-activate?~~ → **YES. Confirmed.**
2. **Does bidirectional feedback improve loss ceiling?** v4 plateaued
   at 4.732. If v4.1 breaks through, descending adds real compression.
3. **Does L1↑ continue declining?** If it approaches zero, the system
   has decided unidirectional phrase-level is fully superseded by
   bidirectional.
4. **Does polarity emerge in descending passes?** Currently too new.
   Prediction: yes, because same function (S5). Descending polarity
   may be inverted relative to ascending.
5. **What happens at step 10k+?** v4 showed L2-dominant specialization
   by 15k. v4.1 has 5 passes sharing work — does it develop a
   different allocation pattern?

## Artifacts produced

- Probes: `results/compile-gradient/vsm_probe_step_00{2,3}000_v4.1.json`
- Probes: `results/binding/vsm_probe_step_00{2,3}000_v4.1.json`
- Allium spec: `specs/vsm-lm-v4.1.allium` (1355 lines)
- Probe script: v4.1-specific output format with 5-pass labels and summary
