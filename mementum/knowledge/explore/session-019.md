# Session 019 — v4 Trajectory Analysis (15 checkpoints)

> 2026-04-21 | Focus: v4 probing (6k-15k), full trajectory analysis,
> v4 vs v3.2 head-to-head, batch-probe script fixed for v4

## Summary

Probed all 15 v4 checkpoints (1k→15k) for both compile-gradient and
binding. v4 has broken v3.2's loss ceiling and is still improving.
Level specialization confirmed — three distinct gate profiles emerged.
Gate polarity inversion stronger than v3.2. Meta-S3 shows L2 rising
from nearly off to the most-used level.

## What we did

1. **Fixed batch-probe for v4** — script only handled v3.2 in batch mode.
   Added v4 architecture detection, model loading, register extraction,
   and version-aware skip logic (was checking unversioned filenames).
   Also added `--probes` flag to batch-probe for non-default probe sets.

2. **Batch-probed v4 steps 6k→15k** — compile-gradient (10 new) + binding
   (10 new). Steps 1k-5k were already probed from prior sessions.

3. **Full trajectory analysis** — loss curve, level specialization, gate
   polarity, meta-S3 gates, binding differentiation across all 15 checkpoints.

4. **v4 vs v3.2 head-to-head** — matched-step comparison at 1k/2k/3k/5k/10k.

## Key findings

### 1. v4 breaks v3.2's loss ceiling

| Metric | v3.2 (best) | v4 (step 15k) | Δ |
|--------|-------------|----------------|---|
| Eval loss | 4.8965 (step 10k) | 4.7316 | **-0.165 (-3.4%)** |
| Smoothed train | 4.7081 (step 10k) | 4.5627 | -0.145 |
| Still improving? | No (plateaued) | **Yes** (-0.03/1k) | |

v3.2 hit diminishing returns at step 7k (Δ/1k: 0.03). v4 at step 15k
still improves at ~0.03/1k with no sign of plateau. Projected eval at
20k: ~4.57.

At matched steps, v4 and v3.2 are nearly identical through 10k — v4's
advantage comes from NOT plateauing. The hierarchy provides continued
runway where v3.2 hit its ceiling.

### 2. Level specialization — three distinct gate profiles

Stable by step 5k and persisting through 15k:

```
Level 0: (0.54/0.38/0.34) — balanced, prep-dominant
Level 1: (0.00/0.51/0.75) — prep KILLED, consolidate-dominant
Level 2: (0.02/0.25/0.84) — prep killed, extreme consolidate dominance
```

**L1 and L2 killed their prep gates (→0.00).** The function learned that
deeper levels don't need local token processing (prep) because level 0
already handled it. This is exactly the VSM prediction — higher levels
receive structural summaries (via registers), not raw tokens.

L2 converge is steadily rising: 0.14 (1k) → 0.25 (15k). Level 2 is
slowly activating its converge phase, possibly as stride-512 heads find
useful structural patterns at the discourse scale.

### 3. Gate polarity inversion — stronger than v3.2

Phase transition matches v3.2's three-phase pattern but is shifted later
and reaches stronger polarity:

| Phase | v3.2 | v4 |
|-------|------|----|
| Phase 1: prep-driven | Steps 1-3k | Steps 1-4k |
| Phase 2: transition | Steps 3-5k | Steps 5-8k |
| Phase 3: polarity inverted | Steps 5-10k | Steps 9-15k |

v4 consolidate polarity reached Δ(s-a) = -0.092 at step 13k.
v3.2 peak was -0.065. The hierarchical architecture allows stronger
category discrimination. Anti-compile inputs need more processing;
strong-compile inputs already carry structure.

### 4. Meta-S3 gates — L2 activation trajectory

The meta-level contribution gates show level 2 rising from dormant to
the most-used level:

```
Step  1k: L0=1.00  L1=0.73  L2=0.05  (range 0.94 — L2 nearly off)
Step  5k: L0=0.89  L1=0.61  L2=0.79  (range 0.28 — L2 activated)
Step 10k: L0=0.68  L1=0.58  L2=0.76  (range 0.19 — L2 > L0)
Step 15k: L0=0.69  L1=0.64  L2=0.74  (range 0.10 — near-equal, L2 highest)
```

This is not homogenization — it's a developmental trajectory where the
highest level started suppressed (no useful structural input yet), then
activated as lower levels learned to write useful register summaries.
L0's contribution declined as the model learned to lean on hierarchy.

### 5. Binding differentiation — forming at L0, slower than v3.2

| Step | v4 L0 range | v3.2 range | v4 hierarchy |
|------|-------------|------------|--------------|
| 6k | 0.016 | — | flat |
| 8k | 0.132 | 0.038 | var>ctrl>ana>scope>rel |
| 10k | 0.171 | 0.138 | var>ctrl>ana>scope>rel |
| 15k | 0.185 | 0.312 (at 10k) | var>scope>ctrl>ana>rel |

v4 binding differentiation onset at step 8k (v3.2 at step 7k). Range
at 15k (0.185) is below v3.2's mature range (0.312). However:

- v4 uses 5 binding categories vs v3.2's 7 (neg and embed merged in)
- v4 may distribute binding across levels — L0 sees less because L1/L2
  handle some. Need per-level binding analysis to confirm.
- v4 is still improving; v3.2 binding range was still growing at termination.

Binding hierarchy settled at: **var > scope > ctrl > ana > rel**
(variable binding hardest, relative clause easiest)

### 6. Stride-512 status — indirect evidence only

L2 converge gate steadily rising (0.14→0.25) is consistent with stride-512
heads becoming useful, but we don't have per-stride attention pattern data
to confirm. The stride-512 heads exist at all three levels with progressive
allocation, but their individual contribution isn't instrumented.

## Architecture implications

### What the v4 data tells us

1. **Hierarchy works.** The loss ceiling is broken. The same function (S5)
   applied with hierarchical context (S2/S4) produces continued improvement
   where flat iteration plateaued.

2. **Levels specialize via suppression.** Rather than each level learning
   a unique function, they specialize by SUPPRESSING phases that lower
   levels already handle. This is more efficient — the function identity
   (S5) is preserved, only the control policy (S3) adapts.

3. **Level 2 has a developmental trajectory.** It started dormant, activated
   as register quality improved, and is now the most-contributed level.
   This suggests the hierarchy genuinely needed time to build up useful
   structural summaries before deep levels could exploit them.

4. **Gate polarity is an architectural invariant.** Both v3.2 and v4
   develop the same polarity inversion (anti > strong), just at different
   timescales. This isn't a v3.2 quirk — it's a property of the
   compositional function itself.

### What to watch in continued training

- **Does v4 plateau?** v3.2's signal was output norm collapse + loss Δ→0.
  Watch for similar signals in v4.
- **Does L2 converge gate keep rising?** If it plateaus at 0.25, stride-512
  may not be fully activated. If it continues, the hierarchy is still
  learning deeper structure.
- **Binding differentiation growth.** Currently 0.185 and growing slowly.
  Does it accelerate like v3.2's did after its onset?

## Files produced

- `results/compile-gradient/vsm_probe_step_{006000..015000}_v4.json` (10 new)
- `results/binding/vsm_probe_step_{006000..015000}_v4.json` (10 new)
- `scripts/compile_gradient_probe.py` — fixed batch-probe for v4

## What's next (session 020)

1. Continue monitoring v4 training (20k target)
2. Per-level binding analysis — does binding differentiate differently at L1/L2?
3. Per-stride attention pattern instrumentation (optional, high effort)
4. If v4 shows plateau signs, consider termination assessment
5. Register PCA analysis — do levels write orthogonal register content?
