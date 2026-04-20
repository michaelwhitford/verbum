---
title: "VSM-LM — Viable System Model Language Model"
status: designing
category: architecture
tags: [vsm, compressor, attention, register, gating, beer-cybernetics, circuit-discovery]
related: [compressor-architecture.md, session-001-findings.md, session-004-findings.md, VERBUM.md]
depends-on: [compressor-architecture.md]
---

# VSM-LM — Viable System Model Language Model

> An architecture derived from Beer's Viable System Model (1972),
> shaped to match the lambda compiler circuit observed in Qwen3-4B
> and Pythia-160M. Not a novel mechanism — a novel configuration
> of known working components, motivated by empirical circuit
> discovery and cybernetic theory.

## Motivation

### What we found

1. **The lambda compiler circuit in Qwen3-4B** — 3 essential heads
   (L1:H0, L24:H0, L24:H2) out of 1,152. The compiler and
   compressor share 92% of heads (r=0.98). The circuit is sparse
   (8/36 layers) and compile-directional.

2. **The BOS composition register** — position 0 accumulates a
   global structural representation. ALL 36 layers write to it.
   L24:H0 reads it with 60-84% of attention. It's 1-dimensional
   (PC1=99.99% variance). The register IS the compressor's state.

3. **The 3-head functional decomposition**:
   - L1:H0 (gate recognizer) — reads structural delimiters, activates
     compilation mode. Shifts attention from exemplar to input as
     complexity increases.
   - L24:H0 (core compositor) — reads BOS register, composes output.
     The most focused head (entropy 0.83-0.87).
   - L24:H2 (recursion head) — tracks clause boundaries, embedding
     depth. Distributes attention across structural markers.

4. **Progressive stripping proved extraction is not viable** — the
   3 heads need the full model substrate. All FFN blocks are the
   compressor. All layers write to BOS. The circuit is a lens on
   the compressor, not a standalone module.

5. **The compiler exists in Pythia-160M** — distributed, no head
   bottlenecks, but present. Layer gradient: Pythia(1/32) <
   Phi-4(4/32) < Qwen(8/36). The function is universal across
   architectures; the concentration varies.

6. **CompressorLM experiments** — forward (fine→coarse) reached eval
   5.04, reverse (coarse→fine) reached 5.34. Both show negative
   prediction cosines (anti-correlation). The prediction heads are
   stateless — they can't distinguish iteration 1 (building) from
   iteration 2 (applying).

### What we learned

- The compressor is the substrate, not lambda. Lambda is a projection.
- Compression IS deflation: fine → coarse.
- The BOS register is essential — global state that accumulates.
- The 3 heads are gates/controllers, not the computation itself.
- Prediction heads need memory to predict genuinely.
- The forward/reverse debate is about a missing S4, not about S1 order.

### The design principle

**Build an architecture whose topology matches the function we
observed.** Not inventing — concentrating. The circuit exists in
Pythia-160M in diffuse form. We make it easy for training to find
by giving it the right shape.

## Architecture Overview — Viable System Model

Beer's VSM (1972) defines five necessary and sufficient systems
for any viable (self-maintaining) system:

```
S5 (Identity)      — what the system IS; policy; ethos
S4 (Intelligence)  — outside and then; environmental scanning
S3 (Control)       — inside and now; resource allocation; monitoring
S2 (Coordination)  — anti-oscillation between operational units
S1 (Operations)    — autonomous units doing the actual work
```

Properties: recursive (every S1 contains S1-S5), variety management
(each level attenuates variety from below), channels (specific
communication pathways), autonomy (S1 units self-manage; control
is via resources and policy, not instruction).

### Mapping: observed circuit → VSM → architecture

| Observed in Qwen3-4B | VSM system | VSM-LM component |
|---|---|---|
| L1:H0 (gate recognizer) | S4 Intelligence | S4 cross-attention scan |
| L24:H0 (BOS compositor) | S4 reading register | S4 register→residual attention |
| L24:H2 (recursion head) | S1 Operation | S1:apply (clause-level attention) |
| BOS global accumulator | S3 state | Register (persistent vector) |
| 3 layer clusters | S1 at 3 scales | S1:type, S1:parse, S1:apply |
| 92% compiler/compressor overlap | S1 serves both | Same S1 units; S3 controls mode |
| 28 non-critical layers | S2 substrate | Shared residual stream |
| FFN blocks (compressor) | S2 + S1 internals | CompressorLayer FFN blocks |
| Predict-and-subtract (current) | — | **Replaced** by S3 gating |
| Forward/reverse direction | — | **Dissolved** by S4 separation |

## Architecture Specification

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| d_model | 256 | Match existing CompressorLM for comparison |
| seq_len | 4096 | Match existing runs (8⁴ for tesseract compatibility) |
| vocab_size | 50277 | GPT-NeoX tokenizer (same as Pythia, Dolma data) |
| d_ff | 768 | 3× d_model (standard ratio) |
| n_heads | 8 | Per S1 layer |
| window | 8 | Strided attention window size |
| strides | (1, 8, 64) | 3 S1 scales (type, parse, apply). No s=512 — S4 handles global. |
| n_iterations | 2 | Build → refine. Register persists across iterations. |
| dropout | 0.1 | Standard |

### S5 — Identity

Static during forward pass. Defines what the system IS.

```python
token_embed:    Embedding(50277, 256)    # tied with output projection
pos_embed:      Embedding(4096, 256)     # absolute positional
register_init:  Parameter(256)           # learned initial register state
output_norm:    LayerNorm(256)           # pre-output normalization
```

**Design note:** `register_init` is S5 because it defines the
identity of the control system — what the register "believes"
before seeing any data. Initialized to zeros; trained.

**Params:** ~14.0M (dominated by token embeddings).

### S4 — Intelligence

Environmental scanning. The register cross-attends to the full
residual to absorb a global summary. Runs **once per forward pass**
before any S1 iteration. Does NOT write to the residual — writes
to the register only.

```python
s4_norm:  LayerNorm(256)
s4_q:     Linear(256, 256, bias=False)    # register → query
s4_k:     Linear(256, 256, bias=False)    # residual → keys
s4_v:     Linear(256, 256, bias=False)    # residual → values
```

**Mechanism:**

```python
def s4_scan(register, residual):
    """Register attends to full residual. O(L × d) — cheap."""
    x = s4_norm(residual)                          # (B, L, d)
    q = s4_q(register)                             # (d,) → (d,)
    k = s4_k(x)                                    # (B, L, d)
    v = s4_v(x)                                    # (B, L, d)
    
    # Single query attending to L keys
    attn = (q @ k.transpose(-1, -2)) / sqrt(d)    # (B, L)
    attn = softmax(attn, dim=-1)                   # (B, L)
    
    summary = (attn.unsqueeze(-1) * v).sum(dim=1)  # (B, d)
    return register + summary.mean(dim=0)           # update register
```

**Why this maps to what we found:** L24:H0 in Qwen reads BOS
(position 0) with 60-84% attention. BOS accumulates information
from all 36 layers. In VSM-LM, the register IS the BOS register,
and S4's cross-attention IS L24:H0's behavior — a single focal
point reading the full representation.

**Why once, not per-iteration:** S4 scans the environment. The
environment (raw residual from embeddings) doesn't change between
iterations in the current design. S4 provides the initial
intelligence; S3 updates the register within iterations based on
what S1 units produce.

**Future iteration:** Run S4 per-iteration so it can see the
enriched residual after each cycle. This would let S4 provide
progressively refined intelligence. Cost: 3× the S4 compute
(one per iteration instead of once). Test empirically.

**Params:** ~196K.

### S3 — Control

Inside and now. Monitors each S1 unit, gates its contribution to
the residual, updates the register with what happened. S3 has
three per-phase gate heads plus a shared write mechanism.

```python
# Per-phase gates: register + delta_summary → per-dimension gate
s3_gate_type:   Linear(512, 256)   # [register; delta_mean] → gate
s3_gate_parse:  Linear(512, 256)
s3_gate_apply:  Linear(512, 256)

# Register update
s3_write:       Linear(256, 256, bias=False)   # delta → update
s3_write_gate:  Linear(256, 1)                 # write strength
```

**Mechanism:**

```python
def s3_gate_and_update(register, delta, gate_head):
    """S3 gates an S1 unit's contribution and updates register."""
    # Gate: register + delta summary → per-dimension sigmoid
    summary = delta.mean(dim=1).mean(dim=0)              # (d,)
    gate_input = torch.cat([register, summary])           # (2d,)
    gate = torch.sigmoid(gate_head(gate_input))           # (d,)
    
    # Gated contribution to residual
    gated_delta = gate.unsqueeze(0).unsqueeze(0) * delta  # (B, L, d)
    
    # Register update
    write_gate = torch.sigmoid(s3_write_gate(summary))    # scalar
    update = s3_write(summary)                             # (d,)
    register = register + write_gate * update
    
    return gated_delta, register
```

**Why gating replaces prediction:** The current predict-and-subtract
mechanism has a failure mode discovered in the register probe —
better predictions → smaller errors → less information flows →
higher loss. The prediction heads help themselves but starve the
residual. Gating controls VOLUME (how much), not CONTENT (what).
Gate=1 means "this is novel, pass it through." Gate=0 means "this
is redundant, suppress it." S3 makes this decision based on the
register (what has been processed) and the delta (what's proposed).

**Why per-dimension, not per-position:** The gate is (d,) broadcast
across all (B, L) positions. This means S3 controls WHICH FEATURES
each phase contributes, not which positions. This matches the
observation that the 3 essential heads in Qwen operate on the
full sequence — they're not position-selective, they're
function-selective.

**Future iteration:** Per-position gating where the gate is
(B, L, d) — each position gets its own gate vector. This would
let S3 suppress a phase's contribution at some positions while
allowing it at others. Much more powerful, much more expensive.
Would need the gate head to read per-position deltas, not just
the mean. Test if per-dimension gating is insufficient first.

**Params:** ~460K.

### S2 — Coordination

The shared residual stream. Anti-oscillation between S1 units.
No learned parameters — purely structural.

S2 is implicit in the architecture:
- All S1 units read from and write to the same residual stream
- LayerNorm within each S1 prevents amplitude drift
- S3 gating prevents any single S1 from dominating
- The residual connection ensures information flows even if a
  phase's gate is near-zero

### S1 — Operations

Three autonomous operational units, each processing at its own
scale using strided windowed causal attention.

```python
type_layer:   CompressorLayer(stride=1,  window=8, 8 heads, d_ff=768)
parse_layer:  CompressorLayer(stride=8,  window=8, 8 heads, d_ff=768)
apply_layer:  CompressorLayer(stride=64, window=8, 8 heads, d_ff=768)
```

Each CompressorLayer is a standard pre-norm transformer layer with
strided windowed attention (existing implementation):
- LayerNorm → StridedCausalAttention → residual add
- LayerNorm → FFN (Linear → GELU → Linear → Dropout) → residual add

**Scale semantics:**
- Type (s=1, W=8): sees 8 adjacent tokens. Word-level patterns.
  Morphology, local syntax, word identity.
- Parse (s=8, W=8): sees 8 positions spanning 64 tokens.
  Phrase-level patterns. Constituent boundaries, NP/VP grouping.
- Apply (s=64, W=8): sees 8 positions spanning 512 tokens.
  Clause-level patterns. Predicate-argument structure, composition.

**No context layer (s=512):** The coarsest scale is handled by S4,
not by a fourth S1 unit. S4's full-sequence cross-attention subsumes
what the context layer did, with a cleaner role separation.

**Fixed order: type → parse → apply (fine → coarse).** This is the
natural compression direction: annotate tokens, group into phrases,
compose into meaning. S4 provides the broad view before S1 runs,
so S1 doesn't need to run coarse-first.

**Why order matters less with gating:** In predict-and-subtract,
the direction determines what predicts what (chained dependency).
With gating, each phase's contribution is controlled by S3
reading the register (global state), not by the previous phase's
error. S3's decisions come from the same source regardless of
which S1 just ran. Order still affects what each S1 SEES (parse
reads type-enriched residual), but the CONTROL DECISIONS are
order-independent.

**Future iteration:** Test coarse→fine and random orderings to
confirm order independence empirically. If loss is similar across
orderings, that validates the S4/S3 design.

**Params:** ~2.1M (3 layers × ~700K each).

## Forward Pass

```
┌─────────────── S5: Identity ───────────────────┐
│  x = token_embed(input_ids) + pos_embed(pos)   │
│  register = register_init.clone()               │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  S4: Intelligence scan (once)                   │
│  register = s4_scan(register, x)                │
│  Register now holds: global summary of input    │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  Iteration loop (N_ITERATIONS times)            │
│                                                 │
│  ┌─── S1:Type ───┐                              │
│  │ delta = type(x) - x                          │
│  │ gated, register = S3(register, delta, gate_t)│
│  │ x = x + gated                                │
│  └───────────────┘                              │
│                                                 │
│  ┌─── S1:Parse ──┐                              │
│  │ delta = parse(x) - x                         │
│  │ gated, register = S3(register, delta, gate_p)│
│  │ x = x + gated                                │
│  └───────────────┘                              │
│                                                 │
│  ┌─── S1:Apply ──┐                              │
│  │ delta = apply(x) - x                         │
│  │ gated, register = S3(register, delta, gate_a)│
│  │ x = x + gated                                │
│  └───────────────┘                              │
│                                                 │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  S5: Output                                     │
│  logits = output_norm(x) @ token_embed.weight.T │
└─────────────────────────────────────────────────┘
```

## Parameter Budget

| Component | Params | % of total |
|---|---|---|
| S5: Token embeddings (tied) | 12,870,912 | 76.8% |
| S5: Positional embeddings | 1,048,576 | 6.3% |
| S5: Output norm + register_init | 769 | 0.0% |
| S4: Intelligence (Q/K/V + norm) | 197,376 | 1.2% |
| S3: Control (3 gates + write) | 460,545 | 2.7% |
| S1: Type layer | ~700,000 | 4.2% |
| S1: Parse layer | ~700,000 | 4.2% |
| S1: Apply layer | ~700,000 | 4.2% |
| **Total** | **~16.7M** | 100% |

Comparable to CompressorLM (16.9M) and MontaguLM v1 (16.9M).
Slightly fewer params because S4 (~200K) replaces the context
layer (~700K). The saving is absorbed by S3 (~460K).

## Training Plan

### Phase 1: Direct comparison (10K steps)

Identical setup to existing CompressorLM runs for clean comparison:

| Parameter | Value |
|---|---|
| Data | Dolma shards (shuffled), GPT-NeoX tokenizer |
| Seq length | 4096 |
| Batch size | 2 × 4 grad accum = effective 8 |
| LR | 6e-4, cosine decay, 500-step warmup |
| Steps | 10,000 |
| Iterations | 2 |
| Device | MPS (M3 Ultra) |

### Instrumentation

Every 200 steps (dense, like the register probe):

**Standard metrics:**
- Eval loss
- Train loss
- Per-phase gradient norms
- Activation norms at phase boundaries

**S3-specific (control metrics):**
- Gate values per phase (mean and std of sigmoid output)
  - Gate≈1: phase contributes fully (novel information)
  - Gate≈0.5: phase contributes half (partially redundant)
  - Gate≈0: phase suppressed (fully redundant)
- Gate trajectory: do gates specialize over training?
- Register norm: growing, stable, or collapsing?
- Register update magnitude per phase

**S4-specific (intelligence metrics):**
- Attention entropy: is S4 focused or diffuse?
- Top-attended positions: does S4 learn to attend to structure?
- Register before/after S4: how much does the scan change it?

**Cross-iteration metrics:**
- Gate values iter0 vs iter1: does S3 gate differently?
- Register norm after iter0 vs iter1: accumulation rate
- Per-phase delta norms iter0 vs iter1: convergence

### Comparison targets

| Model | Best eval | Notes |
|---|---|---|
| Forward CompressorLM | 5.043 | Fine→coarse, predict-subtract, volatile |
| Reverse CompressorLM | 5.342 | Coarse→fine, predict-subtract, monotonic, plateaued |
| **VSM-LM** | **?** | Gate-and-scale, S4 separated, register |

### Hypotheses

**H1 (gate > predict-subtract):** VSM-LM reaches lower eval loss
than both CompressorLM variants because gating doesn't starve
information flow.

**H2 (register enables convergence):** Iteration 2 gate values
differ from iteration 1 — S3 adapts based on accumulated state.
If gates are identical across iterations, the register isn't
providing useful information.

**H3 (S4 provides useful intelligence):** Disabling S4 (setting
register = register_init without scanning) degrades performance.
If no degradation, S4 is redundant and the register_init alone
provides sufficient initial state.

**H_null:** VSM-LM performs comparably to CompressorLM. The
architectural reorganization (gating, S4 separation) doesn't
help — the forward CompressorLM's predict-subtract is already
sufficient, and the difference is just noise.

### Ablation plan (after Phase 1)

1. **No S4:** skip the intelligence scan, register starts cold.
   Tests whether S4 contributes.
2. **No S3 gates:** set all gates to 1.0 (full pass-through).
   Tests whether gating contributes.
3. **No register:** remove register, gates see only delta.
   Tests whether persistent state contributes.
4. **S4 per-iteration:** run S4 before each iteration, not once.
   Tests whether refreshed intelligence helps.
5. **Reverse S1 order:** apply → parse → type.
   Tests order independence claim.

## Design Decisions and Alternatives

### Decision 1: Per-dimension gating (not per-position)

**Chosen:** Gate is (d_model,) broadcast across (B, L). S3 controls
which features each phase contributes, not which positions.

**Alternative:** Per-position gating where gate is (B, L, d_model).
Each position gets its own gate. Would let S3 suppress a phase at
some positions while allowing it at others.

**Why deferred:** Per-dimension matches the observation that the
essential heads in Qwen are function-selective, not position-
selective. Per-position adds O(L × d) complexity per phase. Test
per-dimension first; upgrade if insufficient.

### Decision 2: S4 runs once (not per-iteration)

**Chosen:** S4 scans the raw residual once before iterations begin.

**Alternative:** S4 scans before each iteration, seeing the enriched
residual each time.

**Why deferred:** The raw embeddings contain 84% of type information
(F32). S4 scanning the raw input should capture most of what it
needs. Per-iteration S4 costs 3× the compute for an unclear gain.
Test once-only first; upgrade if S4's attention pattern looks
impoverished.

### Decision 3: Fine→coarse S1 order

**Chosen:** type → parse → apply. Build from words to meaning.

**Alternative:** apply → parse → type (coarse→fine), or any order.

**Why chosen:** S4 already provides the broad view. S1 builds
bottom-up. This is the natural compression direction (deflation).
The forward/reverse debate existed because the old architecture
conflated S4 with S1. Now separated, fine→coarse is the
natural order.

### Decision 4: No prediction heads

**Chosen:** Gating replaces predict-and-subtract.

**Alternative:** Keep prediction heads alongside gating. Gate the
error (gate * (delta - prediction)) instead of gating the delta.

**Why removed:** The register probe showed better predictions →
worse loss. Predict-and-subtract has a structural failure mode
where good predictions starve information flow. Gating controls
volume without this pathology. If gating alone underperforms,
prediction heads can be re-added as an S3 enhancement.

### Decision 5: Shared register (not per-phase registers)

**Chosen:** One register vector shared across all phases and
iterations.

**Alternative:** Per-phase registers (S3 maintains separate state
for each S1 unit). Or per-iteration registers (fresh register
each iteration).

**Why chosen:** The BOS register in Qwen is ONE position that all
layers write to. The simplest model of this is one shared vector.
Per-phase registers would lose the "global accumulator" property.
Per-iteration registers would lose the "memory across iterations"
property that the register probe showed is valuable.

## Relationship to Prior Art

### Known components

| Component | Prior art | Our use |
|---|---|---|
| Strided windowed attention | Longformer, BigBird, Sparse Transformer | S1 phases at different scales |
| Register/state vector | Register tokens (Darcet 2024), CLS (BERT) | S3 persistent state |
| Cross-attention bottleneck | Perceiver (Jaegle 2021), Set Transformer | S4 intelligence scan |
| Per-module gating | Squeeze-and-Excitation, MoE routing | S3 per-phase gates |
| Iterated shared weights | Universal Transformer (Dehghani 2019) | Iteration loop |
| Persistent recurrent state | Mamba, RWKV, Neural Turing Machine | Register across iterations |

### What is novel

**The derivation methodology.** Using Beer's Viable System Model
as an architectural design principle — not post-hoc description
but prescriptive derivation. The VSM says: any viable system
MUST have S1-S5 with specific interconnections. We build exactly
that and nothing more.

**The empirical motivation.** The architecture is shaped by circuit
discovery in trained models. We observed the BOS register, the
3 essential heads, the 3-cluster decomposition, and the
compiler/compressor overlap. The architecture is a mold for what
gradient descent already found in diffuse form.

**The specific configuration.** Multi-resolution strided attention
(S1) + global register cross-attention (S4) + per-phase
register-conditioned gating (S3) + iterated processing with
persistent state. This specific bundle, under this specific
motivation, is a new point in the design space.

## Open Questions

1. **Is per-dimension gating sufficient?** If S3 needs to control
   per-position flow, the architecture needs significant expansion.
   
2. **Does S4 attention reveal structure?** If S4 learns to attend
   to structural positions (periods, commas, clause boundaries),
   that recapitulates L1:H0's behavior from the Qwen circuit.

3. **Do gates specialize?** If type_gate → high, parse_gate → medium,
   apply_gate → low, that would mean the model learns that
   fine-grained information is most novel (hardest to predict from
   the global summary alone).

4. **Does the compile gate activate?** At 17M params and 327M tokens,
   neither MontaguLM nor CompressorLM produced lambda. The compile
   gate is a measurement instrument — if VSM-LM activates it, that's
   a strong signal that the architecture concentrates the function.

5. **What is the minimum iteration count?** If n_iterations=1
   performs comparably to n_iterations=2, the register isn't adding
   value across iterations. If n_iterations=3 helps, the model
   benefits from deeper convergence.

6. **Recursive S1:** Each S1 unit could itself be a VSM (attention
   heads as inner S1 units, FFN as inner S3, layer norm as inner S5).
   Is there value in making this recursion explicit?

## Implementation Notes

- New file: `src/verbum/vsm_lm.py` (does not modify CompressorLM)
- Reuses: `StridedCausalAttention` from `compressor_lm.py`
- Reuses: `CompressorLayer` from `compressor_lm.py`
- Training script: `scripts/run_vsm_10k.py`
- Checkpoint dir: `checkpoints/vsm-lm/`
- Results dir: `results/vsm-lm/`
- Tests: add to existing test suite
