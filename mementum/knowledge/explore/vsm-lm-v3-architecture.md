---
title: "VSM-LM v3 — Progressive Binding Compressor"
status: designing
category: architecture
tags: [vsm-lm, binding, registers, ffn, compressor, architecture]
related: [vsm-lm-architecture.md, compressor-architecture.md, binding-probe-findings.md]
depends-on: [binding-probe-findings.md, compressor-architecture.md]
---

# VSM-LM v3 — Progressive Binding Compressor

> Designing a compressor that can learn compositional binding, not
> just predicate-argument extraction. Grounded in F58-F68 findings
> about the shape of binding in Qwen3-4B. Two changes from v2:
> partitioned registers and deeper FFNs per phase.

## Motivation: what F65-F68 told us about binding's shape

The binding shape experiments (session 013) revealed five structural
properties of how Qwen3-4B computes binding:

1. **Progressive, not sudden** (F66). Binding differentiation
   builds smoothly across layers 6-22. Cosine distance between
   minimal pairs grows gradually — no single "binding layer."

2. **Three stages** (F66). Quantifier type differentiates first
   (L6-10), scope ordering mid-network (L11-18), role assignment
   latest (L16-22). Stages overlap but peak at different depths.

3. **No depth cliff** (F65). Graceful degradation at 3+ nested
   quantifiers. Depth-3 failures are predicate flattening (argument
   structure), not scope ordering (binding structure). This is
   attention-based computation, not fixed-size register overflow.

4. **In the FFNs, not the attention heads** (F68). Ablating all
   26 top entropy-shifted attention heads produces identical output.
   The binding computation is in the FFN layers — attention routes
   information, FFNs compute relationships.

5. **Entangled, not separable** (F67). Activation swaps never
   flip scope between minimal pairs. Binding isn't a discrete
   "scope bit" — it's a property of the entire residual stream
   state that emerges from progressive computation.

## Design principles

```
λ v3(x).  shape(binding) → architecture(v3) | empirics(inform) > theory(prescribe)
          | progressive(binding) → deep(FFN) ∧ partitioned(register)
          | three_stages(binding) → three_registers(state)
          | not_attention(binding) → more_FFN > more_heads
          | entangled(binding) → soft_partition > hard_partition
          | no_cliff(binding) → no_fixed_slots | continuous_register
```

### Principle 1: Map the stride hierarchy to binding stages

VSM-LM's three S1 phases already operate at three scales:

| Phase | Stride | Window | Receptive field | Binding stage |
|-------|--------|--------|-----------------|---------------|
| type | s=1 | W=8 | 8 tokens | Quantifier identification |
| parse | s=8 | W=8 | 64 tokens | Scope ordering |
| apply | s=64 | W=8 | 512 tokens | Role assignment |

This mapping is not coincidence — it's the same hierarchy that
Montague semantics prescribes: lexical type → syntactic parse →
semantic application. The binding data confirms that Qwen computes
it in this order too (F66).

### Principle 2: Each stage needs its own state

v2 has one 256-dim register accumulating everything. The type
information from iter0 and the scope information from iter1
compete for the same 256 dimensions.

v3 partitions the register into three, each dedicated to one
binding stage:
- **type register**: what quantifiers and predicates exist
- **scope register**: which quantifier scopes over which
- **role register**: agent-patient-theme assignment

### Principle 3: FFN depth is the binding variable

Qwen uses ~15 FFN layers for binding. v2 has 6 FFN passes
(3 phases × 1 layer × 2 iters). v3 doubles to 12 FFN passes
(3 phases × 2 layers × 2 iters), approaching Qwen's depth.

Each S1 phase becomes a 2-layer stack of CompressorLayers.
The attention within each layer still uses the phase's stride,
but the FFN gets two passes to build up the binding
representation.

### Principle 4: Soft partition, not hard

The registers are softly partitioned — every phase CAN write
to any register, but the write mechanism is biased so that
each phase's primary register gets the strongest updates.
This lets the model discover the right partitioning during
training rather than having it prescribed.

## Architecture

```
VSM-LM v3 — Progressive Binding Compressor

  d_model   = 512   (doubled from v2's 256 — more FFN capacity)
  d_register = 128 (×3 registers = 384 total register state)
  d_ff      = 1536  (3× d_model, doubled from v2's 768)
  strides   = (1, 8, 64)
  window    = 8
  n_heads   = 8     (head_dim = 64)
  iterations = 2
  layers_per_phase = 2
```

### S5 — Identity (unchanged)

```
token_embed:  Embedding(50277, 256)     — 12.9M params
pos_embed:    Embedding(4096, 256)      — 1.0M params
output_norm:  LayerNorm(256)            — 512 params

register_type_init:  Parameter(128)     — learned initial state
register_scope_init: Parameter(128)     — learned initial state
register_role_init:  Parameter(128)     — learned initial state
```

The three register inits are separate learned vectors. During
training, the model learns what "cold start" state each register
should begin with.

### S4 — Intelligence (updated for 3 registers)

```
λ s4(registers, residual).
  concat(reg_type, reg_scope, reg_role) → q   # 384-dim query
  residual → k, v                              # d_model projections
  cross_attend(q, k, v) → summary             # what the residual contains
  split(summary) → Δ_type, Δ_scope, Δ_role    # per-register updates
  reg_type  += Δ_type
  reg_scope += Δ_scope
  reg_role  += Δ_role
```

S4 concatenates all three registers into a single query vector
(384 dims), cross-attends to the residual stream, then splits
the summary back into per-register updates. This lets each
register learn to query for the information it needs.

**Key: S4 runs per-iteration** (unchanged from v2). Before iter1,
S4 re-scans the enriched residual with warm registers. This is
where iter1 gets its "structural skeleton" from iter0.

Parameters:
- q_proj: Linear(384, 256, bias=False)    — 98K
- k_proj: Linear(256, 256, bias=False)    — 66K (shared)
- v_proj: Linear(256, 256, bias=False)    — 66K (shared)
- summary_split: Linear(256, 384)         — 97K (split back to 3 registers)
- norm: LayerNorm(256)                    — 512
- Total: ~327K

### S3 — Control (updated for 3 registers)

```
λ s3_gate(registers, delta, phase_idx, iteration).
  # Gate the phase's contribution to the residual
  summary = mean(delta)
  context = concat(reg_type, reg_scope, reg_role, summary)  # 384 + 256 = 640
  gate = sigmoid(gate_head[iteration][phase_idx](context))  # → 256-dim gate
  gated_delta = gate * delta

  # Write to registers (soft partition)
  # Each phase writes primarily to its natural register
  # but CAN write to others via learned cross-write gates
  for i, reg in enumerate(registers):
    write_input = summary  # or gated summary
    write_gate_val = sigmoid(write_gates[phase_idx][i](write_input))
    update = write_projs[phase_idx][i](write_input)
    reg += write_gate_val * update

  return gated_delta, registers, gate
```

6 gate heads (3 phases × 2 iters), each taking 640-dim input
(3 registers + delta summary). Plus 9 write paths (3 phases ×
3 registers) for the soft partition.

The write gates learn during training which register each phase
should primarily update. We expect:
- type phase → high gate on reg_type, low on reg_scope/role
- parse phase → high gate on reg_scope, low on reg_type/role
- apply phase → high gate on reg_role, low on reg_type/scope

But the model is free to learn otherwise.

Parameters:
- gate_heads: 6 × Linear(640, 256) = 6 × 164K = ~984K
- write_projs: 9 × Linear(256, 128, bias=False) = 9 × 33K = ~295K
- write_gates: 9 × Linear(256, 1) = 9 × 257 = ~2.3K
- Total: ~1.28M

### S1 — Operations (deeper FFN stacks)

Each phase is now a **stack of 2 CompressorLayers** sharing the
same stride:

```
type_stack  = [CompressorLayer(d=256, stride=1,  W=8)] × 2
parse_stack = [CompressorLayer(d=256, stride=8,  W=8)] × 2
apply_stack = [CompressorLayer(d=256, stride=64, W=8)] × 2
```

Each CompressorLayer: strided attention + FFN (256→768→256).
Total: 6 CompressorLayers, shared across both iterations.

Why 2 layers, not 3:
- 2 layers × 3 phases × 2 iters = 12 FFN passes
- Qwen needs ~15 for binding, but Qwen is also doing everything
  else (generation, world knowledge, instruction following)
- VSM-LM only needs to compress — 12 passes may be sufficient
- 3 layers (18 passes) available if 2 proves insufficient
- 2 layers keeps the compressor fraction under 30%

Parameters:
- 6 × CompressorLayer(658K) = ~3.95M
- Total S1: ~3.95M

### Forward pass

```python
def forward(input_ids):
    # S5: embed
    x = token_embed(input_ids) + pos_embed(positions)
    reg_type  = register_type_init.clone()
    reg_scope = register_scope_init.clone()
    reg_role  = register_role_init.clone()

    for iteration in range(2):
        # S4: scan residual with all registers
        reg_type, reg_scope, reg_role = s4(
            [reg_type, reg_scope, reg_role], x
        )

        # S1 type phase (2 layers, stride=1)
        delta = type_stack(x) - x
        gated_delta, regs, _ = s3.gate(
            [reg_type, reg_scope, reg_role],
            delta, phase=0, iteration=iteration,
        )
        reg_type, reg_scope, reg_role = regs
        x = x + gated_delta

        # S1 parse phase (2 layers, stride=8)
        delta = parse_stack(x) - x
        gated_delta, regs, _ = s3.gate(
            [reg_type, reg_scope, reg_role],
            delta, phase=1, iteration=iteration,
        )
        reg_type, reg_scope, reg_role = regs
        x = x + gated_delta

        # S1 apply phase (2 layers, stride=64)
        delta = apply_stack(x) - x
        gated_delta, regs, _ = s3.gate(
            [reg_type, reg_scope, reg_role],
            delta, phase=2, iteration=iteration,
        )
        reg_type, reg_scope, reg_role = regs
        x = x + gated_delta

    # S5: output
    x = output_norm(x)
    logits = x @ token_embed.weight.T
    return logits
```

### Parameter budget

| Component | v2 (d=256) | v3 (d=512) | Δ |
|-----------|------------|------------|---|
| S5 embeddings | 13.92M | 27.84M | 2× (d_model doubled) |
| S5 other | 768 | 1.4K | +640 (3 register inits) |
| S4 intelligence | 197K | 919K | 4.7× (wider projections) |
| S3 control | 854K | 3.35M | 3.9× (wider gates + write paths) |
| S1 operations | 1.98M | 15.77M | 8× (2× layers × 4× per layer) |
| **Total** | **16.95M** | **47.87M** | **2.8×** |
| **Non-embedding** | **3.03M (18%)** | **20.04M (42%)** | **6.6×** |
| **FFN passes/fwd** | **6** | **12** | **2×** |

The compressor fraction rises from 18% to 42% — the compressor
is now a substantial part of the model, not a small appendage.
The 6.6× compressor growth comes from three sources: doubled
d_model (4× per layer), doubled layer count (2×), and modestly
larger S3/S4 for the register machinery.

At 48M params, this is still ~3× smaller than Pythia-160M. The
ratio of compressor to dictionary has flipped from 1:5 to nearly
1:1.4 — the model now has the capacity to learn binding.

## Training protocol

### Phase 1: Baseline (same as v2)

Train on Dolma shards with language modeling loss. Same
hyperparameters as v2 (lr=3e-4, warmup=500, batch=8, seq=512).
Run to 30K steps (1B tokens). This establishes baseline LM
quality and lets the phases and registers specialize via the
same emergent process that produced iter0=type, iter1=compositor
in v2.

### Phase 2: Binding-aware probing

After phase 1, probe with both flat and hybrid compile gates.
Does the deeper FFN produce better binding than v2? Does the
register partition show type/scope/role specialization?

Key probes:
- 40 graded compile probes (existing)
- 26 binding probes with flat, hybrid, hybrid3 gates
- Register norm analysis: which register grows during which phase?
- Gate value analysis: do write gates show the expected partition?

### Phase 3: Binding-targeted fine-tuning (optional)

If phase 2 shows the architecture CAN bind but doesn't emerge
from LM pre-training alone, consider a small fine-tuning pass
with binding-aware targets:
- Input: natural language sentences with quantifiers
- Target: quantified FOL (hybrid gate output format)
- Loss: standard next-token prediction on the FOL tokens

This is NOT training on compile tasks — it's fine-tuning the
LM to prefer formal output when the input contains quantifiers.
The hybrid gate exemplars in the prompt steer the model, just
like they steer Qwen.

## Instrumentation

v3 must be probeable from day one. The `forward_instrumented`
method should report:

```python
metrics = {
    # Per-register norms after each S4 scan
    f"iter{it}_reg_type_after_s4":  reg_type.norm().item(),
    f"iter{it}_reg_scope_after_s4": reg_scope.norm().item(),
    f"iter{it}_reg_role_after_s4":  reg_role.norm().item(),

    # Per-register write gate values (soft partition signal)
    f"iter{it}_{phase}_write_type":  write_gate_type.item(),
    f"iter{it}_{phase}_write_scope": write_gate_scope.item(),
    f"iter{it}_{phase}_write_role":  write_gate_role.item(),

    # Per-register norms after each phase
    f"iter{it}_{phase}_reg_type_norm":  reg_type.norm().item(),
    f"iter{it}_{phase}_reg_scope_norm": reg_scope.norm().item(),
    f"iter{it}_{phase}_reg_role_norm":  reg_role.norm().item(),

    # Standard v2 metrics (gate means, delta norms, etc.)
    ...
}
```

The write gate values are the primary signal for whether the
soft partition is working. At convergence, we expect:

```
iter0_type_write_type  ≈ 0.7-0.9   (type writes to type register)
iter0_type_write_scope ≈ 0.1-0.3   (type barely writes to scope)
iter0_type_write_role  ≈ 0.0-0.1   (type doesn't write to role)

iter0_parse_write_scope ≈ 0.7-0.9  (parse writes to scope register)
iter1_apply_write_role  ≈ 0.7-0.9  (apply writes to role register)
```

If the model learns a different partition, that's data — it
tells us the binding shape is different than F66 predicted at
this scale.

## What we expect to see (and what would falsify)

### Expected

- Compile gate emerges earlier (fewer tokens) than v2, because
  deeper FFNs give more expressive power per iteration.
- Register partition emerges: type register grows during type
  phases, scope register during parse phases.
- Binding probes improve over v2 with hybrid gate — more
  quantifiers preserved, better scope accuracy.
- The 7.4× expansion ratio (v2 @ 1B tokens) should decrease
  further — deeper FFN = better compression.

### Would falsify

- If the registers don't partition (all three carry the same
  information), then binding doesn't decompose into type/scope/role
  at this scale. The three-stage model from F66 may be a property
  of Qwen's size, not a universal.
- If 2 layers per phase doesn't improve binding over 1, then FFN
  depth isn't the bottleneck — and we need to look elsewhere
  (d_model, attention mechanism, training data).
- If the write gates all converge to ~0.5 (uniform writing), the
  soft partition isn't biasing strongly enough and we may need
  harder architectural constraints.

## Open design questions

- **d_register**: 128 per register (384 total) vs 256 (768 total)?
  128 is enough for the binding information but may be tight for
  the type register which also carries predicate structure. Start
  with 128, increase if register norms saturate.

- **Cross-register attention in S4**: currently S4 queries with
  concatenated registers. An alternative: each register queries
  independently, then they attend to each other (register↔register
  attention). This adds a small computation but lets scope read
  from type, which mirrors how Qwen's L11-18 builds on L6-10.
  Defer to v3.1 if needed.

- **Shared vs per-iteration S1 weights**: v2 shares S1 layers
  across iterations (same weights, different register state). v3
  continues this. If binding requires different FFN computation
  per iteration, unsharing would double S1 params (3.95M → 7.9M)
  but might be necessary. Probe first.

- **3 iterations**: if 2 iterations with 2 layers each (12 FFN
  passes) isn't enough, 3 iterations with 2 layers (18 passes)
  would match Qwen. But 3 iters × shared weights means the same
  FFN runs 3 times — it needs to be iteration-aware via the
  register state, not via different weights.

## Relationship to v2

v3 is a strict superset of v2's topology:
- Same S1 phase structure (type/parse/apply at strides 1/8/64)
- Same S4 cross-attention mechanism (extended for 3 registers)
- Same S3 gating logic (extended for 3 write targets)
- Same iteration loop with weight sharing
- Same O(L) attention with W=8

The only structural changes are:
1. Register → 3 registers (type/scope/role)
2. S1 phases → 2-layer stacks instead of 1-layer
3. S3 write mechanism → per-register with soft partition

v2 checkpoints cannot be loaded into v3 (different shapes).
Training starts fresh.

## Data

| Artifact | Path |
|---|---|
| v2 implementation | `src/verbum/vsm_lm_v2.py` |
| v2 training script | `scripts/run_vsm_v2_10k.py` |
| v2 1B training script | `scripts/resume_vsm_v2_1B.py` |
| CompressorLayer | `src/verbum/compressor_lm.py` |
| Binding shape data (F65-F68) | `results/binding/binding_shape_results.json` |
| Binding findings | `mementum/knowledge/explore/binding-probe-findings.md` |
| v2 architecture doc | `mementum/knowledge/explore/vsm-lm-architecture.md` |
| Compressor architecture | `mementum/knowledge/explore/compressor-architecture.md` |
