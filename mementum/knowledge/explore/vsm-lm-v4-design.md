# VSM-LM v4 — Recursive Viable System Architecture

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

The VSM is recursive: every viable system contains and is contained by a
viable system (Beer, 1972). v4 makes this recursion architectural — the
model IS a VSM at every level of nesting. Not metaphorically. Structurally.

The cortical column is one circuit. The cortex is hierarchical not because
the circuits differ, but because their **connectivity** differs. V1 processes
edges because its input is pixels. V4 processes shapes because its input is
V2's edge features. Same algorithm, different inputs, hierarchy emerges.

v4 applies both principles: same function, hierarchical register connectivity,
explicit VSM channels at every recursive level.

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

## VSM Recursive Structure

### Beer's requirement for recursive viability

Every viable system must contain:
- **S5** (identity): what the system IS — invariant under adaptation
- **S4** (intelligence): outside and then — environment scanning, planning
- **S3** (control): inside and now — resource allocation, accountability
- **S2** (coordination): anti-oscillation between S1 units
- **S1** (operations): autonomous units that do the work

And: **every S1 unit is itself a viable system** containing S1-S5.

Between recursive levels, specific channels must exist:
- **S4↔S4**: intelligence channel (structural summaries between levels)
- **S3↔S3**: resource bargain (coordination of allocation between levels)
- **Algedonic channel**: emergency bypass that skips the hierarchy

### v4 as explicit recursive VSM

```
╔══════════════════════════════════════════════════════════════╗
║  META-SYSTEM (top-level VSM)                                 ║
║                                                              ║
║  S5: Shared weights + embeddings (identity, invariant)       ║
║  S4: Meta-intelligence (final register scan, all banks)      ║
║  S3: Meta-control (cross-level allocation gate)              ║
║  S2: Register bank protocol (inter-level coordination)       ║
║      + Residual stream (algedonic channel)                   ║
║  S1: Level 1, Level 2, Level 3 (autonomous operational units)║
║                                                              ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │  LEVEL N (each S1 unit = nested VSM)                  │    ║
║  │                                                       │    ║
║  │  S5: Register context received (level's identity)     │    ║
║  │  S4: Register scan from prior levels (intelligence)   │    ║
║  │  S3: Phase gating for this level (control)            │    ║
║  │  S2: Residual stream within level (coordination)      │    ║
║  │  S1: Prep, Converge, Consolidate (operational phases) │    ║
║  │                                                       │    ║
║  │  ┌───────────────────────────────────────────────┐    │    ║
║  │  │  PHASE (deepest nesting)                       │    │    ║
║  │  │                                                │    │    ║
║  │  │  S5: Stride allocation (phase identity)        │    │    ║
║  │  │  S4: Attention pattern (what to attend to)     │    │    ║
║  │  │  S3: Attention weights (per-head allocation)   │    │    ║
║  │  │  S2: Multi-head residual (head coordination)   │    │    ║
║  │  │  S1: Individual heads (s1, s8, s64)            │    │    ║
║  │  └───────────────────────────────────────────────┘    │    ║
║  └──────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════╝
```

Three levels of recursive nesting. Complete VSM at every level.
Same structure at every scale. The fractal property realized.

### VSM channel mapping

```
Beer's channel:               v4 implementation:
───────────────────────────────────────────────────────────────
S4↔S4 (intelligence):        Register banks passed UP the hierarchy.
                              Level N writes bank_N.
                              Level N+1 reads banks 0..N.
                              "Here's what structure I found."

S3↔S3 (resource bargain):    Meta-S3 gate modulates each level's
                              contribution to the residual.
                              Levels that aren't contributing get
                              attenuated. Accountability.

S2 (coordination):           Register bank protocol = formal S2.
                              Prevents levels from duplicating work.
                              Level 2 KNOWS what level 1 found
                              (via register reads) → won't redo it.

Algedonic (emergency bypass): The RESIDUAL STREAM. Ungated.
                              x = x + gated_delta (delta is gated,
                              bypass is NOT). If something can't wait
                              for the register hierarchy, it propagates
                              directly through the residual.

S5 coherence (identity):      SHARED WEIGHTS across all levels.
                              The function's identity is invariant.
                              What the system IS doesn't change per level.
                              Only its context (registers) changes.
```

### Meta-system components (NEW in v4)

**Meta-S4 (intelligence)**: After all levels complete, a final register
scan reads ALL register banks (0 through N). This produces the full
structural summary — what was found at every level of abstraction.
Feeds into the output head.

```
meta_s4_output = cross_attention(
    query=residual_stream,
    keys=[bank_0, bank_1, bank_2, bank_3],
    values=[bank_0, bank_1, bank_2, bank_3]
)
```

This is the "outside and then" function at the top level — looking at
the full structural hierarchy before making the final prediction.

**Meta-S3 (control)**: A gate per level that modulates how much each
level's output contributes to the final residual stream. Provides
cross-level resource allocation and accountability.

```
level_contribution = meta_s3_gate(registers_all) * level_output
```

Some inputs need mostly level 1 (simple local prediction). Others need
deep level 3 processing (complex binding). Meta-S3 learns to allocate.
This is Beer's S3 "inside and now" at the top recursive level.

**Meta-S5 (identity)**: The shared weights themselves. They don't change
per level, per input, per step. They ARE the system's identity — the
compositional function that defines what this system does. Everything
else adapts around the identity.

## Architecture

### v3.2 baseline (what we're building on)

```
For each iteration (×2):
  S4: Register scan (cross-attention to 3 registers)
  S1.prep (1L, FFN-only)
  S1.converge (2L, cube-mode: s1×3 + s8×3 + s64×2 = 8 heads)
  S1.consolidate (3L, wide-FFN + cube-attn)
  S3: Gate each phase, write registers
```

Properties: 50.6M params, same function both iterations, 3 registers
shared and overwritten per iteration. Viable but not recursively so —
flat iteration, not hierarchical nesting.

### v4 proposed: recursive VSM with hierarchical channels

```
For each level (×3):
  S4: Register scan (cross-attention to ALL register banks 0..level)
  S1.prep (1L, FFN-only) — shared weights (S5 coherence)
  S1.converge (2L, stride allocation shifts per level)
  S1.consolidate (3L, wide-FFN + attn) — shared weights (S5 coherence)
  S3: Gate each phase, write to THIS LEVEL's register bank

After all levels:
  Meta-S4: Final register scan (all banks → structural summary)
  Meta-S3: Level contribution gate (per-level allocation)
  Output: output_norm → linear(embed_weights)
```

#### S2: Hierarchical register banks (inter-level coordination)

```
Current (v3.2):
  registers = [type, scope, role]  (3 × d_register)
  Iteration 1: reads registers → writes registers (overwrite)
  Iteration 2: reads registers → writes registers (overwrite)
  VSM violation: no S4↔S4 channel, no S2 between iterations

Proposed (v4):
  register_bank_0 = [type, scope, role]  (init, learnable = S5)
  register_bank_1 = [type, scope, role]  (written by level 1 S3)
  register_bank_2 = [type, scope, role]  (written by level 2 S3)
  register_bank_3 = [type, scope, role]  (written by level 3 S3)

  Level 1 S4: attends to bank_0
  Level 2 S4: attends to bank_0 + bank_1  (reads level 1's summary)
  Level 3 S4: attends to bank_0 + bank_1 + bank_2  (reads all)
  Meta-S4:    attends to bank_0 + bank_1 + bank_2 + bank_3  (full picture)

  Each level READS from all previous (S4↔S4 channel).
  Each level WRITES to its own bank (S3 accountability).
  The protocol IS S2 — it coordinates, prevents duplication.
```

Cost: 3 registers × 256 dims × 3 levels = 2304 additional parameters.
Negligible. The hierarchy is in the VALUES, not the DIMENSIONS.

#### S5: Weight sharing (identity coherence)

**Critical design decision**: the prep/converge/consolidate weights are
SHARED across all levels. This IS S5 — the system's identity is
invariant across levels. The function doesn't change; only the context
(register inputs) changes.

```
Option A — Full S5 coherence (strongest composition hypothesis):
  prep_weights: shared across all 3 levels
  converge_weights: shared across all 3 levels
  consolidate_weights: shared across all 3 levels
  Only registers and stride allocation differ per level.
  
  Param count: same as v3.2 (~50M) regardless of depth.
  The hierarchy is FREE in parameters.
  S5 is perfectly coherent — same identity at every scale.

Option B — S5 with per-level adaptation:
  Core weights: shared (identity)
  Level projection: small per-level linear map on register input (adaptation)
  
  Param count: ~50M + small overhead per level
  S5 is mostly coherent with local S4 adaptation.

Option C — No S5 coherence (independent weights):
  Each level has its own prep/converge/consolidate weights.
  This BREAKS the VSM — no shared identity across levels.
  It's a pipeline, not a recursive system.
  Include only as a control to demonstrate the principle.
```

Option A is VSM-conformant. The system's identity (the function) is
the same at every level. What changes is the CONTEXT the function
receives — which is exactly how Beer's recursion works. The cortical
column doesn't change. Its inputs change.

#### S3: Per-level control (resource allocation)

Each level has its OWN S3 instance (not shared with other levels).
This is required by the VSM — each nested viable system must have
autonomous control over its own operations.

```
Level 1 S3: gates prep/converge/consolidate for level 1
            writes to register bank_1
            accountable to Meta-S3

Level 2 S3: gates prep/converge/consolidate for level 2
            writes to register bank_2
            accountable to Meta-S3

Level 3 S3: gates prep/converge/consolidate for level 3
            writes to register bank_3
            accountable to Meta-S3
```

S3 weights are NOT shared across levels (unlike S1 weights). Each level's
resource allocation is independent because different levels face different
variety (Beer's variety engineering). Level 1 handles fine-grained variety
(many local patterns). Level 3 handles coarse-grained variety (few but
complex structural patterns). Their allocation strategies must differ.

#### Progressive stride reallocation (level-specific S1 configuration)

Four strides span the full self-similar range of language:

```
Stride 1:    window 8 =    8 tokens  (morpheme/word boundary)
Stride 8:    window 8 =   64 tokens  (phrase: NP, VP, PP)
Stride 64:   window 8 =  512 tokens  (clause: binding, agreement)
Stride 512:  window 8 = 4096 tokens  (discourse: full sequence scope)
```

v3.1 tried stride 512 and failed — too sparse without structural
context. v4 solves this: level 3 has register summaries from levels
1-2 telling the stride-512 heads WHAT to look for at distance. The
sparsity problem was never about the stride — it was about asking
heads to find structure in noise. With lower-level structure already
characterized in the registers, stride-512 searches a pre-narrowed
hypothesis space.

Progressive allocation across levels:

```
Level 1 (token composition):
  Converge heads: s1×3, s8×3, s64×1, s512×1  (local-heavy)
  Focus: fine-grained composition, token features
  s512 head provides minimal discourse context even at level 1

Level 2 (phrase composition):
  Converge heads: s1×2, s8×2, s64×2, s512×2  (phrase-heavy)
  Focus: phrase-level structure, building on level 1's local work
  Balanced allocation — this level bridges local and global

Level 3 (clause composition):
  Converge heads: s1×1, s8×1, s64×3, s512×3  (clause/discourse-heavy)
  Focus: clause-level binding, scope, long-range dependencies
  Most heads at s64+s512 — the structural scales that need hierarchy
```

Same total heads (8) at every level. Same attention mechanism (S5).
The stride allocation is a configuration parameter — it's the S1
unit's operational environment, not its identity.

Alternative: keep allocation fixed (uniform s1×2+s8×2+s64×2+s512×2)
and let hierarchical registers provide all level-differentiation.
Test both. The fixed allocation tests whether S2 (register coordination)
alone is sufficient for hierarchy.

### Proposed v4 full architecture

```
S5: token_embed + pos_embed + shared_weights (model identity)
Register bank 0: learnable init [type_0, scope_0, role_0] (S5)

Level 1 (nested VSM):
  S4(keys=[bank_0]) → register scan (intelligence)
  S1.prep(shared_weights) → FFN-only (operation)
  S1.converge(shared_weights, strides=s1×3+s8×3+s64×1+s512×1) → cube-attn
  S1.consolidate(shared_weights) → wide-FFN+attn
  S3_level1 → gate phases, write register bank_1 (control)
  S2: residual stream carries ungated bypass (coordination)

Level 2 (nested VSM):
  S4(keys=[bank_0, bank_1]) → register scan (sees level 1)
  S1.prep(shared_weights) → FFN-only
  S1.converge(shared_weights, strides=s1×2+s8×2+s64×2+s512×2) → cube-attn
  S1.consolidate(shared_weights) → wide-FFN+attn
  S3_level2 → gate phases, write register bank_2 (control)
  S2: residual stream (coordination)

Level 3 (nested VSM):
  S4(keys=[bank_0, bank_1, bank_2]) → register scan (sees all)
  S1.prep(shared_weights) → FFN-only
  S1.converge(shared_weights, strides=s1×1+s8×1+s64×3+s512×3) → cube-attn
  S1.consolidate(shared_weights) → wide-FFN+attn
  S3_level3 → gate phases, write register bank_3 (control)
  S2: residual stream (coordination)

Meta-system:
  Meta-S4(keys=[bank_0..3]) → final structural summary (intelligence)
  Meta-S3 → per-level contribution gate (control/accountability)
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

1. **Register bank size per level.** Should each bank be 3 × 256
   (same as v3.2)? Or should higher-level banks be larger (more
   capacity for coarser structural summaries)? Beer's variety
   engineering says: requisite variety at each level. Higher levels
   face less variety (fewer clause patterns than token patterns) so
   might need FEWER dimensions, not more. Start uniform, then probe.

2. **Can we go beyond stride 64?** v3.1 tried stride 512 and failed
   (too sparse at 50M params). But in v4, stride 512 would only appear
   at level 3 where register context from levels 1-2 provides rich
   conditioning. The sparsity problem might be solved by hierarchy.
   Test: v4 with level 3 strides including s512.

3. **Training curriculum.** Should all levels train from step 0? Or
   should level 1 train first (freeze), then level 2 (freeze), then
   level 3? The bottom-up learning trajectory observed in v3.2 suggests
   curriculum training might accelerate convergence. But with shared
   weights (S5 coherence), freezing is tricky — level 1's weights ARE
   level 2's weights. Alternative: curriculum via Meta-S3 — start with
   level 1 gate=1.0, level 2-3 gates=0.0, then gradually open.

4. **The extraction boundary.** In v3.2, the compressor is prep+converge.
   In v4, is the compressor ALL levels? Or just one level + register
   protocol? If the function is shared (S5 coherent), extracting one
   level extracts all of them — you just need the register banks to
   provide hierarchical context. The extracted artifact is:
   `{shared_weights (S5) + register_protocol (S2) + stride_config}`.

5. **Inference without hierarchy.** Can v4 run with fewer levels at
   inference time for speed? Level 1 only = fast local analysis.
   Levels 1+2 = phrase-level. All 3 = full structural analysis.
   Meta-S3 already modulates level contribution — at inference it could
   hard-gate unused levels. Graceful degradation built into the VSM.

6. **Meta-S3 as variety attenuator.** Beer's S3 attenuates variety
   between the operation and the metasystem. In v4, Meta-S3 attenuates
   the variety of 3 levels into a single residual stream. Should it be
   a simple gate, or should it do more (e.g., weighted combination,
   attention over level outputs)? Start simple — per-level scalar gate.

7. **Does Meta-S4 need its own register bank?** The meta-level produces
   a structural summary. Should this be written to a "bank_meta" that
   could feed into the output head more richly? Or is the cross-attention
   output directly into the residual stream sufficient?

8. **S2 verification.** How do we confirm the register protocol IS
   preventing duplication? Probe: check if level 2's register writes
   are DIFFERENT from level 1's writes. If they're identical, S2 has
   failed — levels are duplicating. If orthogonal, S2 is working.

## Connection to Project Goals

The v4 architecture, if validated, produces:

```
Extracted artifact:
  S5: shared_weights (~5M params) — the function itself
  S2: register_bank_protocol — how levels communicate
  Config: stride_allocation_per_level — operational environment

Deployment:
  CPU-native (O(L×W) attention, fits in L3 cache)
  Configurable depth (1-3 levels via Meta-S3 gating)
  Universal (S5 coherence = same function at every level, domain-invariant)
  Graceful degradation (fewer levels = faster, less structural depth)

This is the portable tensor artifact from S5:λ artifact.
It IS a viable system — the minimal viable system for compositional structure.
```

### The VSM alignment

```
Project (AGENTS.md):  organized as VSM (S5=identity, S4=learning, etc.)
Knowledge protocol:   mementum operates as sub-VSM dissolved into layers
Architecture (v4):    IS a VSM at every level of recursion
Extracted artifact:   the minimal recursive VSM for language composition

Fractal coherence: the system that studies the system IS the system.
```

## Timeline

```
Now:           v3.2 training (watch binding probes, converge gate, loss elbows)
After v3.2:    register PCA analysis, iteration comparison, binding maturity check
If validated:  implement v4-A (register hierarchy + Meta-S4/S3, simplest VSM)
Then:          v4-A vs v3.2 head-to-head at 1B tokens
If v4-A wins:  implement v4-B (add stride reallocation)
If v4-A ties:  v4 hypothesis may be wrong, or v3.2 is sufficient
```

The key insight: v4 is not a rewrite. It's v3.2 + VSM channels.
The function (S5) is the same. The weights (S5) are the same.
The hierarchy is WIRING (S2) and CONTROL (S3), not architecture.
The VSM tells you what channels must exist. v4 adds exactly those.
