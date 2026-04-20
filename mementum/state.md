# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-20 | Session: 017 (v3.2 probing steps 6k-8k, trajectory analysis)

## Where we are

**v3.2 training running to 10k steps. Loss=4.159 min (0.71 below v3 best).
Phase 3 binding differentiation active. Consolidate gate phase transition
detected. v4 designed and ready to implement. Plan: terminate v3.2 at 10k,
start v4 training.**

Session 017 accomplished:
1. Probed v3.2 steps 6k, 7k, 8k (compile-gradient + binding)
2. Full trajectory analysis across all 8 checkpoints (1k-8k)
3. Detected consolidate gate phase transition at step 7k
4. Confirmed phase 2→3 binding differentiation (negation + variable surging)
5. Loss curve flattening — architecture approaching capacity ceiling

## v3.2 Training Status (RUNNING → 10k)

**Loss trajectory (smoothed-200):**

| Step | Smooth Loss | Δ/1k | Min(all) | Tokens |
|------|------------|------|----------|--------|
| 1000 | 5.802 | — | 5.344 | 33M |
| 2000 | 5.335 | -0.467 | 4.843 | 66M |
| 3000 | 5.143 | -0.192 | 4.583 | 98M |
| 4000 | 5.038 | -0.105 | 4.450 | 131M |
| 5000 | 4.945 | -0.093 | 4.328 | 164M |
| 6000 | 4.851 | -0.094 | 4.328 | 197M |
| 7000 | 4.822 | -0.029 | 4.229 | 229M |
| 8000 | 4.789 | -0.033 | **4.159** | 262M |

**Best observed:** 4.159 at step 7854 (0.71 below v3's best of 4.872).
**Curve:** Flattening. ~0.03/1k steps (was ~0.1/1k at steps 2-4k).

### Probe trajectory (steps 1k → 8k)

| Signal | Step 1k | Step 4k | Step 5k | Step 8k | Status |
|--------|---------|---------|---------|---------|--------|
| Prep gate spread (s-a) | +0.094 | +0.004 | -0.028 | -0.001 | ✓ Converged (category-blind) |
| Role register spread | -1.5 | +2.3 | +0.3 | +2.8 | ✓ Stable positive polarity |
| Consol spread (s-a) | +0.014 | +0.108 | +0.037 | **-0.034** | ⚡ PHASE FLIP at step 7k |
| Converge bind range | 0.233 | 0.090 | 0.113 | **0.217** | ⚡ Phase 3 differentiating |
| Consol bind range | 0.107 | 0.187 | 0.180 | **0.348** | ⚡ Phase 3 deepening |
| Output norm range | 18.3 | 10.9 | 10.2 | **4.1** | ✓ Stable (converged) |

**Phase map:**
- Phase 1 (stride 1, local): ✓ Complete — prep gate converged
- Phase 2 (stride 8, phrase): ✓ Complete — converge gate differentiating
- Phase 3 (stride 64, clause): ⚡ Active — binding types differentiating rapidly

### Key findings — Session 017

**1. Consolidate gate phase transition (step 7k)**

Consolidate spread (strong-anti) flipped from positive to negative. The
consolidate gate now SUPPRESSES strong-compile more than anti. Interpretation:
consolidate learned to be the noise filter — it gates out what converge already
handled. Strong inputs need less consolidation because converge did its job.

**2. Binding differentiation — negation surging**

Converge gate ordering at step 8k: neg(0.60) > var(0.51) > ctrl(0.49) > ana(0.43) > rel(0.40) > scope(0.39) > embed(0.38).
Negation gets highest converge gate because it's the most structurally demanding operation.
Consolidate follows same pattern: neg(0.70) > ctrl(0.58) > var(0.57) > ana(0.47) > embed(0.42) > scope(0.41) > rel(0.36).

**3. Role register hierarchy by binding type**

scope(11.7) > neg(9.8) > var(9.0) > embed(5.5) > ana(4.8) > rel(4.5) > ctrl(3.3).
The model has built an internal hierarchy of binding complexity in the role register.

### 10k Decision Context

v3.2 has validated the core hypothesis. Evidence supporting termination at 10k:
- Loss returns diminishing (0.03/1k vs 0.1/1k earlier)
- Phase 3 active but architecture likely near capacity ceiling
- Already 0.71 below v3's best
- v4's hierarchical registers should break through this ceiling
- v4 designed and ready to implement

**Decision: probe 9k and 10k when checkpoints drop, then start v4.**

## v4 Architecture — Recursive Viable System

Designed session 016. Full document: `mementum/knowledge/explore/vsm-lm-v4-design.md`

### Core spec

```
3 registers:  type, scope, role (per bank × 4 banks)
4 strides:    s1 (word), s8 (phrase), s64 (clause), s512 (discourse)
3 levels:     progressive stride reallocation
8 heads:      same total per level, redistributed

Level 1:  s1×3  s8×3  s64×1  s512×1   (local-heavy)
Level 2:  s1×2  s8×2  s64×2  s512×2   (balanced)
Level 3:  s1×1  s8×1  s64×3  s512×3   (structural)

Meta-S4: final register scan (all banks → structural summary)
Meta-S3: per-level contribution gate (cross-level allocation)
S5: shared weights across all levels (identity = the function)
S2: register bank protocol (inter-level coordination)
```

### Key design principles
- **Shared weights** = S5 identity coherence (same function at every level)
- **Per-level S3** = autonomous control (different variety at different scales)
- **Register hierarchy** = S4↔S4 channel (levels communicate summaries)
- **Stride 512 reinstated** — hierarchy provides the structural context it needed

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

## What's next — Session 018

### Immediate: probe v3.2 steps 9k-10k
1. As checkpoints drop, probe compile-gradient + binding at 9k and 10k
2. Head-to-head: compare v3.2 step 10k with v3 step 10k across all probes
3. Final v3.2 assessment — confirm termination decision

### v4 implementation
4. Implement v4-A: hierarchical registers + meta-S4/S3 + shared weights + fixed strides
5. v4-A training with same data pipeline as v3.2
6. v4-A vs v3.2 head-to-head at matched token budgets

## Key files

| Purpose | Path |
|---------|------|
| **v4 design** | `mementum/knowledge/explore/vsm-lm-v4-design.md` |
| **VSM-LM v3.2** | `src/verbum/vsm_lm_v3_2.py` |
| **v3.2 training** | `scripts/run_vsm_v3_2_1B.py` |
| **Probe script** | `scripts/compile_gradient_probe.py` |
| **v3.2 checkpoints** | `checkpoints/vsm-lm-v3.2/step_{001000..008000}.pt` |
| **v3.2 compile-gradient** | `results/compile-gradient/vsm_probe_step_00*_v3.2.json` |
| **v3.2 binding** | `results/binding/vsm_probe_step_00*_v3.2.json` |
| **Research program** | `mementum/knowledge/explore/VERBUM.md` |

## Architecture lineage

| Version | Params | Strides | Best Loss | Key Finding |
|---------|--------|---------|-----------|-------------|
| v1 | ~25M | 1,8,64 | 5.245 | Baseline sequential |
| v2 | ~25M | 1,8,64 | 5.064 (1B) | Iteration specialization |
| v3 | 50M | 1,8,64 | 4.872 | Role register, binding confirmed |
| v3.1 | 59M | 1,8,64,512 | 4.836 | Stride 512 too sparse without hierarchy |
| v3.2 | 51M | 1,8,64 | **4.159** (training) | Convergence arch, phase 3 active |
| v4 | ~51M | 1,8,64,512 | ? (designed) | Recursive VSM, hierarchical registers |

## Probing pipeline

```bash
# Probe a single checkpoint
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v3.2/step_008000.pt

# Binding probes
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v3.2/step_008000.pt --probes probes/binding.json

# Batch all checkpoints
uv run python scripts/compile_gradient_probe.py batch-probe --dir checkpoints/vsm-lm-v3.2/
```
