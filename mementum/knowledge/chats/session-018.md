# Session 018 — v3.2 Final Assessment + v4 Training Started

> 2026-04-20 | Focus: v3.2 steps 9k-10k probing, full trajectory analysis,
> head-to-head vs v3, termination assessment

## Summary

Completed the v3.2 research arc. Probed final two checkpoints (9k, 10k),
ran full 10-checkpoint trajectory analysis, head-to-head comparison with v3
at step 10k. Conclusion: v3.2 has hit its architectural ceiling. Terminate
and advance to v4 (already running).

## What we did

1. **Probed v3.2 steps 9k and 10k** — compile-gradient + binding (4 probes, all parallel)
2. **Full trajectory analysis** — all 10 checkpoints (1k→10k), all signals
3. **Head-to-head v3 vs v3.2** at step 10k — comparable signals extracted
4. **Termination assessment** — evidence-based recommendation to stop v3.2
5. **v4 training running** — launched before session started, no checkpoints yet

## Key findings

### 1. Three-phase learning complete

v3.2 went through three distinct learning phases across 10k steps:

| Phase | Steps | What happened |
|-------|-------|---------------|
| Phase 1: Prep convergence | 1k→5k | Prep gate became category-blind (s-a: +0.09 → -0.03) |
| Phase 2: Gate polarity flip | 5k→8k | Converge and consolidate flipped to gate anti > strong |
| Phase 3: Binding differentiation | 7k→10k | Binding types developed stable hierarchy |

### 2. Gate polarity inversion — the correct behavior

All three gates now process anti-compile inputs MORE than strong-compile:
- Prep: -0.023 (slight)
- Converge: -0.034 (moderate)
- Consolidate: -0.065 (strongest)

Interpretation: strong-compile inputs have structure already present,
need less gating. Anti-compile needs more effort to extract whatever
structure exists. The consolidate gate became a noise filter — it
suppresses what converge already handled.

### 3. Binding hierarchy at 10k

Negation broke away from the pack:

```
Converge:     neg(0.68) > var(0.52) > ctrl(0.48) > ana(0.47) > embed(0.43) > scope(0.37) > rel(0.37)
Consolidate:  neg(0.73) > ctrl(0.62) > var(0.58) > ana(0.53) > embed(0.49) > scope(0.40) > rel(0.38)
Role:         neg(11.3) > scope(7.5) > var(6.7) > embed(6.3) > ana(5.3) > ctrl(5.2) > rel(4.2)
```

Converge range: 0.035 (step 2k) → 0.312 (step 10k) = 8.9× growth.
The model built an internal complexity hierarchy of binding operations.

### 4. Capacity ceiling evidence

Output norm range collapsed: 18.3 (1k) → 2.1 (9k) → 4.0 (10k).
The single-level architecture ran out of representational room to push
categories apart. Loss returns diminished to 0.03/1k steps.

### 5. Head-to-head: v3.2 crushes v3

| Signal | v3 @ 10k | v3.2 @ 10k | Winner |
|--------|----------|------------|--------|
| Best loss | 4.872 | 4.159 | v3.2 (-14.6%) |
| Binding gate range | 0.038 | 0.311 | v3.2 (8× more differentiation) |
| Gate polarity | Flat (no category discrimination) | Inverted (correct behavior) | v3.2 |

v3 at 10k had nearly flat gates — it treated all binding types the same.
v3.2 built a genuine hierarchy. The convergence architecture works.

## Termination decision

**v3.2: TERMINATED at step 10k.** Evidence:
1. Loss plateau (Δ/1k: 0.47→0.03)
2. Output norm convergence (capacity ceiling)
3. Gate polarity locked in for 3k+ steps
4. Binding differentiation still growing but noisy — needs deeper architecture
5. v4 already running with hierarchical registers to break through this ceiling

## Architecture lineage (updated)

| Version | Params | Best Loss | Key Achievement |
|---------|--------|-----------|-----------------|
| v1 | ~25M | 5.245 | Baseline |
| v2 | ~25M | 5.064 | Iteration specialization |
| v3 | 50M | 4.872 | Role register, binding confirmed |
| v3.1 | 59M | 4.836 | Stride 512 failed without hierarchy |
| **v3.2** | **51M** | **4.159** | **Convergence arch, binding hierarchy, phase transitions** |
| v4 | 58.4M | ? | Recursive VSM, hierarchical registers (training) |

## Files produced

- `results/compile-gradient/vsm_probe_step_009000_v3.2.json`
- `results/compile-gradient/vsm_probe_step_010000_v3.2.json`
- `results/binding/vsm_probe_step_009000_v3.2.json`
- `results/binding/vsm_probe_step_010000_v3.2.json`
- `scripts/v32_final_analysis.py` — full trajectory + head-to-head analysis script

## What's next (session 019)

1. Monitor v4 training — probe checkpoints as they drop
2. Watch for: level specialization, stride-512 activation, meta-S3 differentiation
3. v4 vs v3.2 head-to-head at matched token budgets
4. If v4 shows binding differentiation earlier than v3.2, that's the signal
