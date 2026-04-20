# VSM-LM v4 — Hierarchical Composition Architecture

> Status: **designing** (refining during v3.2 training)
> Depends-on: v3.2 training results, binding probe maturity
> Category: architecture
> Related: vsm-lm-v3-architecture.md, compressor-architecture.md, VERBUM.md

## Core Thesis

v3.2 validates that **one compositional function** (prep→converge→consolidate)
applied iteratively can learn language structure faster than pipelined
architectures. v4 asks: what if we give that function **hierarchical
connectivity** — making each iteration explicitly operate at a different
level of abstraction?

The cortical column is one circuit. The cortex is hierarchical not because
the circuits differ, but because their **connectivity** differs. V1 processes
edges because its input is pixels. V4 processes shapes because its input is
V2's edge features. Same algorithm, different inputs, hierarchy emerges.

v4 applies this principle: same function, hierarchical register connectivity,
progressive stride reallocation.

## Theoretical Grounding

### Why hierarchy matters

Language is self-similar across scales. The same composition operation
(typed application) applies at every level:

```
morpheme + morpheme → word        (scale 1)
word + word → phrase              (scale 8)
phrase + phrase → clause           (scale 64)
clause + clause → sentence        (scale 512)
```

v3.2 handles all scales simultaneously (cube-mode), relying on the
iteration loop to deepen processing. But both iterations use the same
strides with the same allocation. There's no explicit signal saying
"iteration 2 should focus on coarser scales because iteration 1 already
handled finer scales."

### The gradient separation argument extended

v3.2's strides separate gradients by SCALE within an iteration.
v4 extends this by separating gradients by LEVEL across iterations:

```
v3.2:  iter 1 and iter 2 share the same stride allocation
       → both iterations receive similar gradient profiles
       → no architectural pressure to specialize by level

v4:    iter 1 is local-heavy, iter 2 is phrase-heavy, iter 3 is clause-heavy
       → each iteration receives gradient signal matched to its scale
       → architectural pressure to specialize per level
```

### The compression-as-prediction argument

If H ≈ 0.70 bits/char (DeepMind) and structural composition accounts for
~75% of the redundancy in language, then the compressor is most of a
predictor. Hierarchical composition makes the compressor MORE complete —
it captures structure at every level explicitly rather than hoping two
iterations of the same allocation are sufficient.

## Architecture

### v3.2 baseline (what we're building on)

```
For each iteration (×2):
  S4: Register scan (cross-attention to 3 registers)
  PREP (1L, FFN-only)
  CONVERGE (2L, cube-mode: s1×3 + s8×3 + s64×2 = 8 heads)
  CONSOLIDATE (3L, wide-FFN + cube-attn)
  S3: Gate each phase, write registers
```

Properties: 50.6M params, same function both iterations, 3 registers
shared and overwritten per iteration.

### v4 proposed: hierarchical register banks + stride reallocation

```
For each level (×3):
  S4: Register scan (cross-attention to ALL register banks 0..level)
  PREP (1L, FFN-only) — same function, same weights across levels
  CONVERGE (2L, stride allocation shifts per level)
  CONSOLIDATE (3L, wide-FFN + attn) — same function, same weights
  S3: Gate each phase, write to THIS LEVEL's register bank
```

#### Change 1: Hierarchical register banks

```
Current (v3.2):
  registers = [type, scope, role]  (3 × d_register)
  Iteration 1: reads registers → writes registers (overwrite)
  Iteration 2: reads registers → writes registers (overwrite)

Proposed (v4):
  register_bank_0 = [type, scope, role]  (init, learnable)
  register_bank_1 = [type, scope, role]  (written by level 1)
  register_bank_2 = [type, scope, role]  (written by level 2)
  register_bank_3 = [type, scope, role]  (written by level 3)

  Level 1 S4: attends to bank_0
  Level 2 S4: attends to bank_0 + bank_1
  Level 3 S4: attends to bank_0 + bank_1 + bank_2

  Each level READS from all previous, WRITES to its own bank.
```

The register banks are the hierarchical connectivity. They carry
structural summaries from each level to all subsequent levels.

Cost: 3 registers × 256 dims × 3 levels = 2304 additional parameters.
Negligible. The hierarchy is in the VALUES, not the DIMENSIONS.

S4 change: instead of cross-attention to 3 register vectors, attend
to 3 × (level+1) vectors. S4 already does multi-head cross-attention —
just more keys. The attention mechanism automatically learns which
previous levels' registers are relevant for the current level.

#### Change 2: Progressive stride reallocation

```
Level 1 (token composition):
  Converge heads: s1×3, s8×3, s64×2  (local-heavy)
  Focus: fine-grained composition, token features

Level 2 (phrase composition):
  Converge heads: s1×2, s8×3, s64×3  (phrase-heavy)
  Focus: phrase-level structure, building on level 1's local work

Level 3 (clause composition):
  Converge heads: s1×1, s8×2, s64×5  (clause-heavy)
  Focus: clause-level binding, scope, long-range dependencies
```

Same total heads (8) at every level. Same attention mechanism.
The stride allocation is a configuration parameter, not a weight change.
Gradient signal at each level is dominated by the level's focal scale.

Alternative: keep allocation fixed (same as v3.2) and let the
hierarchical registers provide all the level-differentiation signal.
Test both. The fixed allocation is simpler and might be sufficient if
register hierarchy alone creates the needed pressure.

#### Change 3: Weight sharing (the composition principle)

**Critical design decision**: the prep/converge/consolidate weights are
SHARED across all levels. This is the compositional hypothesis — same
function, different inputs (via hierarchical register context).

```
Option A — Full sharing (strongest composition hypothesis):
  prep_weights: shared across all 3 levels
  converge_weights: shared across all 3 levels
  consolidate_weights: shared across all 3 levels
  Only registers and stride allocation differ per level.
  
  Param count: same as v3.2 (~50M) regardless of depth.
  The hierarchy is FREE in parameters.

Option B — Shared function, per-level projection:
  Core weights: shared across levels (prep, converge, consolidate)
  Level projection: small per-level linear map on register input
  
  Param count: ~50M + small overhead per level

Option C — Independent weights (pipeline, defeats the purpose):
  Each level has its own prep/converge/consolidate weights.
  This is just a deeper v3.2. NOT the composition hypothesis.
  Include only as a control experiment.
```

Option A is the strong claim. Start there. Fall back to B only if A
fails to differentiate across levels.

### Proposed v4 full architecture

```
Embed: token_embed + pos_embed (same as v3.2)
Register bank 0: learnable init [type_0, scope_0, role_0]

Level 1:
  S4(registers=[bank_0]) → register scan
  PREP(shared_weights) → FFN-only
  CONVERGE(shared_weights, strides=s1×3+s8×3+s64×2) → cube-attn
  CONSOLIDATE(shared_weights) → wide-FFN+attn
  S3 → gate, write register bank_1

Level 2:
  S4(registers=[bank_0, bank_1]) → register scan (sees level 1 summary)
  PREP(shared_weights) → FFN-only
  CONVERGE(shared_weights, strides=s1×2+s8×3+s64×3) → cube-attn
  CONSOLIDATE(shared_weights) → wide-FFN+attn
  S3 → gate, write register bank_2

Level 3:
  S4(registers=[bank_0, bank_1, bank_2]) → register scan (sees all)
  PREP(shared_weights) → FFN-only
  CONVERGE(shared_weights, strides=s1×1+s8×2+s64×5) → cube-attn
  CONSOLIDATE(shared_weights) → wide-FFN+attn
  S3 → gate, write register bank_3

Output: output_norm → linear(embed_weights)
```

### Parameter budget

```
                        v3.2          v4 (Option A)
Token embed:            25.7M         25.7M (same)
Pos embed:              2.1M          2.1M (same)
S5 other:               ~2K           ~4K (+3 register banks)
S4:                     ~400K         ~400K (same mechanism, more keys)
S3:                     ~100K         ~150K (3 levels × 3 phases vs 2 × 3)
S1 prep:                ~1.6M         ~1.6M (shared across levels)
S1 converge:            ~8.5M         ~8.5M (shared across levels)
S1 consolidate:         ~12.3M        ~12.3M (shared across levels)
─────────────────────────────────────────────────
Total:                  ~50.6M        ~50.7M

Difference: ~100K params. The hierarchy is essentially free.
```

3 levels instead of 2 iterations, with essentially the same parameter
count. The extra compute is 50% more forward passes (3 vs 2 iterations),
which is the cost of hierarchy — but each level's processing should be
more efficient because it's focused on the right scale.

## What v3.2 Training Must Validate First

Before building v4, v3.2 training needs to answer:

### Must-have signals

1. **Does the converge gate differentiate by binding type at maturity?**
   If the converge phase never specializes, adding stride reallocation
   won't help. We need to see that cube-mode attention IS doing
   different things for different binding categories.
   
   Current (step 5k): control converge gate (0.444) > quant_scope (0.343).
   Signal present but early. Watch through step 10k.

2. **Do the registers carry meaningful structural information?**
   The role register polarity flipped at step 4k. But do the register
   VALUES encode something interpretable? PCA on register vectors
   across binding categories would tell us.
   
   Experiment: after v3.2 training, run PCA on register vectors. If
   binding categories cluster in register space, registers carry
   structure. If not, hierarchical register banks won't help.

3. **Does iteration 2 do something different from iteration 1?**
   If both iterations learn the same function at the same scale,
   hierarchy won't emerge just from register banks. Check: are
   iter0 gate patterns different from iter1 gate patterns?
   
   Current: yes — iter0 gates are selective (0.3-0.6), iter1
   consolidate is saturated (0.9). Different behavior per iteration
   already emerging.

### Nice-to-have signals

4. **Does stride-64 specialize for long-range binding?**
   Can we instrument per-stride attention patterns? If stride-64 heads
   attend differently for quantifier_scope vs variable_binding, that
   validates per-level stride reallocation.

5. **Loss curve elbows at phase transitions?**
   If the loss curve shows slope changes corresponding to fine→coarse
   scale transitions, that validates the bottom-up learning hypothesis
   and suggests explicit hierarchy would sharpen these transitions.

6. **Does the model benefit from more iterations?**
   Quick experiment: train v3.2 with 3 iterations instead of 2 (same
   shared weights, just one more pass). If 3 > 2, the function benefits
   from depth. If 3 ≈ 2, two passes are sufficient and v4's value comes
   from the HIERARCHY not the depth.

## Ablation Plan for v4

When v4 is built, test in this order:

```
1. v4-A: hierarchical registers + shared weights + FIXED strides (same as v3.2)
   (Tests: does register hierarchy alone create level specialization?)

2. v4-B: hierarchical registers + shared weights + PROGRESSIVE strides
   (Tests: does stride reallocation on top of register hierarchy help?)

3. v4-C: hierarchical registers + independent weights (control)
   (Tests: is weight sharing necessary? Is this just a deeper pipeline?)

4. v4-A-deep: like v4-A but with 4 or 5 levels
   (Tests: does the hierarchy scale? Or do 3 levels capture everything?)
```

Compare all against v3.2 at same token budget (1B tokens).

Primary metric: binding probe differentiation at maturity.
Secondary metric: loss at matched step count.
Tertiary metric: loss at matched token count (fairness check since
v4 does 3 iterations per step vs v3.2's 2).

## Open Questions

1. **Should S3 also be hierarchical?** Currently S3 gates per phase per
   iteration. In v4, should each level have its own S3, or should one
   S3 gate all levels? Per-level S3 allows different gating strategies
   at different scales. Shared S3 forces uniform gating.

2. **Register bank size.** Should each bank be 3 × 256 (same as v3.2)?
   Or should higher-level banks be larger (more capacity for coarser
   structural summaries)? Start with uniform, expand if registers
   saturate at higher levels.

3. **Can we go beyond stride 64?** v3.1 tried stride 512 and failed
   (too sparse at 50M params). But in v4, stride 512 would only appear
   at level 3 where register context from levels 1-2 provides rich
   conditioning. The sparsity problem might be solved by hierarchy.
   Test: v4 with level 3 strides including s512.

4. **Training curriculum.** Should all levels train from step 0? Or
   should level 1 train first (freeze), then level 2 (freeze), then
   level 3? The bottom-up learning trajectory observed in v3.2 suggests
   curriculum training might accelerate convergence. But with shared
   weights, freezing is tricky — level 1's weights ARE level 2's weights.

5. **The extraction boundary.** In v3.2, the compressor is prep+converge.
   In v4, is the compressor ALL levels? Or just level 1? If the function
   is shared, extracting one level extracts all of them — you just need
   the register banks to provide the hierarchical context. The extracted
   artifact might be: {shared_weights + register_bank_protocol}.

6. **Inference without hierarchy.** Can v4 run with fewer levels at
   inference time for speed? Level 1 only = fast local analysis.
   Levels 1+2 = phrase-level. All 3 = full structural analysis.
   Graceful degradation if the hierarchy is clean.

## Connection to Project Goals

The v4 architecture, if validated, produces:

```
Extracted artifact:
  shared_weights (~5M params)
  + register_bank_protocol (how levels communicate)
  + stride_allocation_per_level (configuration, not weights)

Deployment:
  CPU-native (O(L×W) attention, fits in L3 cache)
  Configurable depth (1-3 levels for speed/quality tradeoff)
  Universal (same function at every level, domain-invariant)

This is the portable tensor artifact from S5:λ artifact.
```

## Timeline

```
Now:           v3.2 training (watch binding probes, converge gate, loss elbows)
After v3.2:    register PCA analysis, iteration comparison, binding maturity check
If validated:  implement v4-A (register hierarchy only, simplest change)
Then:          v4-A vs v3.2 head-to-head at 1B tokens
If v4-A wins:  implement v4-B (add stride reallocation)
If v4-A ties:  v4 hypothesis may be wrong, or v3.2 is sufficient
```

The key insight: v4 is not a rewrite. It's v3.2 + register banks + an
extra iteration. The function is the same. The weights are the same.
The hierarchy is wiring, not architecture.
