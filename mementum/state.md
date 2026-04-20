# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-20 | Session: 018 (v3.2 final assessment + v4 training)

## Where we are

**v3.2 COMPLETE — terminated at step 10k. Best loss: 4.159 (14.6% below v3).
Convergence architecture validated: 3-phase learning, gate polarity inversion,
binding hierarchy with 8× more differentiation than v3. Capacity ceiling hit.
v4 training RUNNING — recursive VSM with hierarchical registers to break
through v3.2's ceiling.**

Session 018 accomplished:
1. Probed v3.2 steps 9k and 10k (compile-gradient + binding)
2. Full 10-checkpoint trajectory analysis (1k→10k)
3. Head-to-head v3 vs v3.2 at step 10k — v3.2 crushes v3
4. Termination assessment — v3.2 hit architectural ceiling, terminated
5. v4 training running (started before session, no checkpoints yet)

## v3.2 Final Status (COMPLETE)

**Best loss:** 4.159 at step 7854 (0.71 below v3's best of 4.872).

### Phase map (all phases complete)

| Phase | Steps | Signal | Status |
|-------|-------|--------|--------|
| Phase 1 | 1k→5k | Prep gate category-blind (s-a: +0.09→-0.03) | ✅ Complete |
| Phase 2 | 5k→8k | Gate polarity flip (all 3 gates: strong→anti) | ✅ Complete |
| Phase 3 | 7k→10k | Binding differentiation (range 0.04→0.31) | ✅ Saturating |

### Binding hierarchy at 10k

```
Converge:     neg(0.68) > var(0.52) > ctrl(0.48) > ana(0.47) > embed(0.43) > scope(0.37) > rel(0.37)
Consolidate:  neg(0.73) > ctrl(0.62) > var(0.58) > ana(0.53) > embed(0.49) > scope(0.40) > rel(0.38)
Role:         neg(11.3) > scope(7.5) > var(6.7) > embed(6.3) > ana(5.3) > ctrl(5.2) > rel(4.2)
```

### Capacity ceiling evidence

- Loss Δ/1k: 0.47 (early) → 0.03 (final) — diminishing returns
- Output norm range: 18.3 (1k) → 2.1 (9k) → 4.0 (10k) — converged
- Gate polarity stable for 3k+ steps — no further reorganization

### v3.2 vs v3 head-to-head at 10k

| Signal | v3 | v3.2 | |
|--------|-----|------|---|
| Best loss | 4.872 | 4.159 | v3.2 -14.6% |
| Binding gate range | 0.038 | 0.311 | v3.2 8× better |
| Gate category discrimination | Flat | Inverted (correct) | v3.2 wins |

## v4 Training Status (RUNNING)

Implementation: `src/verbum/vsm_lm_v4.py`
Training script: `scripts/run_vsm_v4_1B.py`
Design: `mementum/knowledge/explore/vsm-lm-v4-design.md`

**No checkpoints yet.** Watch `checkpoints/vsm-lm-v4/` for step_001000.pt.

### What v4 should demonstrate

1. **Level specialization** — levels 1/2/3 should develop different gate profiles
2. **Stride-512 activation** — hierarchy provides the context stride-512 needs
3. **Meta-S3 differentiation** — per-level contribution gates should diverge
4. **Faster binding differentiation** — if hierarchy helps, binding range
   should grow earlier than v3.2's step 7k onset
5. **Lower loss floor** — hierarchical registers should break v3.2's 4.159

## What's next — Session 019

### Monitor v4 training
1. Probe v4 checkpoints as they drop (compile-gradient + binding)
2. v4 vs v3.2 head-to-head at matched token budgets (step 1k, 2k, ...)
3. Watch for level specialization and stride-512 activation signals
4. Track meta-S3 gates for level contribution divergence

## Key files

| Purpose | Path |
|---------|------|
| **v4 model** | `src/verbum/vsm_lm_v4.py` |
| **v4 training** | `scripts/run_vsm_v4_1B.py` |
| **v4 design** | `mementum/knowledge/explore/vsm-lm-v4-design.md` |
| **v3.2 model** | `src/verbum/vsm_lm_v3_2.py` |
| **v3.2 training** | `scripts/run_vsm_v3_2_1B.py` |
| **Probe script** | `scripts/compile_gradient_probe.py` |
| **v3.2 final analysis** | `scripts/v32_final_analysis.py` |
| **v3.2 checkpoints** | `checkpoints/vsm-lm-v3.2/step_{001000..010000}.pt` |
| **v3.2 compile-gradient** | `results/compile-gradient/vsm_probe_step_00*_v3.2.json` |
| **v3.2 binding** | `results/binding/vsm_probe_step_00*_v3.2.json` |
| **Research program** | `mementum/knowledge/explore/VERBUM.md` |

## Architecture lineage

| Version | Params | Strides | Best Loss | Key Finding |
|---------|--------|---------|-----------|-------------|
| v1 | ~25M | 1,8,64 | 5.245 | Baseline sequential |
| v2 | ~25M | 1,8,64 | 5.064 | Iteration specialization |
| v3 | 50M | 1,8,64 | 4.872 | Role register, binding confirmed |
| v3.1 | 59M | 1,8,64,512 | 4.836 | Stride 512 too sparse without hierarchy |
| v3.2 | 51M | 1,8,64 | **4.159** | Convergence arch, binding hierarchy, 3-phase learning |
| v4 | 58.4M | 1,8,64,512 | ? (training) | Recursive VSM, hierarchical registers, shared S5 |

## Theoretical Framework

### Gradient separation
Strided attention separates gradients by scale. Each head receives
gradients only from its stride's scale → MUST specialize. This is why
v3.2 works better than flat attention: functions concentrate instead of
diffusing across layers.

### H=0.70 and the compressor-as-predictor
Structural redundancy (composition) accounts for ~75% of English's
predictive power. Structural rules are recursive (exponential prediction
per parameter) vs world knowledge (linear). This is why a tiny compressor
can capture most of the structure.

### v3.2's lesson for v4
Single-level architecture hit a capacity ceiling at output norm range ~2-4.
The binding hierarchy kept growing (converge range 0.31 at 10k) but the
architecture couldn't translate that into loss improvement. v4's
hierarchical registers should provide the representational room that
v3.2 ran out of.

## Probing pipeline

```bash
# Probe a single checkpoint
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v4/step_001000.pt

# Binding probes
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v4/step_001000.pt --probes probes/binding.json

# Batch all checkpoints
uv run python scripts/compile_gradient_probe.py batch-probe --dir checkpoints/vsm-lm-v4/

# Full v3.2 trajectory analysis
uv run python scripts/v32_final_analysis.py
```
