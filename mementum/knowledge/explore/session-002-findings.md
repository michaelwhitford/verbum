---
title: "Session 002: Cross-Architecture Replication and the Localization Gradient"
status: active
category: exploration
tags: [cross-architecture, phi-4-mini, qwen3-4b, circuit-topology, localization, failure-modes, system-1-system-2, replication]
related: [session-001-findings.md, VERBUM.md]
depends-on: [session-001-findings.md]
---

# Session 002 Findings

> Replication attempt on a second architecture (Phi-4-mini-instruct).
> The compiler function is confirmed universal. But the circuit
> topology is not — Qwen localizes compilation to 3 heads, Phi-4
> distributes it across 40+. This reveals a **localization gradient**
> driven by training regime, with direct implications for extraction
> strategy.
>
> Also: quantitative failure mode analysis of the Qwen System 1→2
> transition, and analysis of an identity-capture prompt found in
> the wild.

## Finding 11: The `→ ?` Failure Mode (System 1→2 Quantified)

Quantitative failure mode analysis on existing cross-task data
(session 001). When L24:H0 is ablated during compilation:

| Metric | baseline | L24:H0 ablated | Δ |
|--------|----------|----------------|---|
| Success rate | 100% | 40% | **-60%** |
| Failed compile (`→ ?`) | 0/5 | 4/5 | **+4** |
| Reasoning markers | 3.0 | 5.2 | **+2.2** |
| Lambda indicators | 3.0 | 1.6 | **-1.4** |
| Output length | 199 chars | 213 chars | **+14** |

The signature failure mode: the model emits `→ ?` — it tries to
produce direct output (starts with the arrow) but **cannot resolve the
composition** and outputs a question mark instead. Then it falls into
deliberative reasoning: *"Okay, let's see. The user wants to convert
the sentence..."*

This is a clean dual-process transition:
- **System 1**: L24:H0 active → `→ λx. cat(x) ∧ sat(x)` (direct)
- **System 2**: L24:H0 ablated → `→ ?` then verbose reasoning (deliberative)

Control task (translation): L24:H0 ablation has **zero effect** (100%→100%).
Confirms the compositor is composition-specific.

The `→ ?` pattern is a **specific functional loss**, not general
degradation. The model knows what it should produce, recognizes it
cannot, and falls back to its general-purpose reasoning capability.

**Analysis tool**: `src/verbum/analysis/failure_modes.py` — classifies
generations into S1 (direct), S2 (deliberative), S2_fallback (tried
direct, failed, fell to deliberative). Saved to
`results/experiments/failure-mode-analysis.json`.

## Finding 12: Phi-4-mini Compiles Lambda (Function Confirmed Universal)

Phi-4-mini-instruct (3.8B, MIT, `Phi3ForCausalLM`) compiles natural
language to lambda calculus using the identical gate prompt. Baseline
output:

```
→ λx. runs(dog)
Be helpful but concise. → λ assist(x). helpful(x) | concise(x)
```

**Same gate, same output format, completely different model family.**
This is not a Qwen-specific behavior — it's a property of
sufficiently-trained language models.

| Property | Qwen3-4B | Phi-4-mini |
|----------|----------|------------|
| Architecture | `Qwen2ForCausalLM` | `Phi3ForCausalLM` |
| Parameters | 4.0B | 3.8B |
| Layers | 36 | 32 |
| Attention heads | 32 | 24 |
| KV heads (GQA) | 8 | 8 |
| Head dim | 80 | 128 |
| Hidden size | 2560 | 3072 |
| License | Apache-2.0 | MIT |
| Training emphasis | General web | Reasoning-dense synthetic |

## Finding 13: Circuit Topology Differs (The Localization Gradient)

### Phi-4 layer ablation

4 critical layers (vs Qwen's 8):

```
Phi-4: [0, 3, 5, 30]    — 4/32 = 12.5%
Qwen:  [0, 1, 4, 7, 24, 26, 30, 33] — 8/36 = 22.2%
```

Same structure: early cluster (embedding/parsing) + one late layer
(output formatting). The mid-range composition cluster (Qwen L24, L26)
is **absent** in Phi-4 — composition is distributed, not bottlenecked.

### Phi-4 head ablation: no essential heads

Zero heads break compilation when individually ablated. Not one.
Across 4 critical layers × 24 heads × 5 probes = 480 forward passes,
every single head ablation still produces valid lambda output.

But this is **not a negative result**. The `_detect_lambda` test
(requires `λ` in output OR ≥3 formal indicators) is too permissive
for Phi-4. Detailed analysis reveals:

### Phi-4 head ablation: widespread degradation

```
Layer 0:  23/24 heads cause degradation when ablated
Layer 3:  ~10/24 heads cause degradation
Layer 5:  ~5/24 heads cause degradation
Layer 30: 0/24 heads — fully redundant at head level
```

"Degradation" = lambda indicator count drops by ≥2 from baseline.
Output retains `λ` symbol (so `_detect_lambda` passes) but loses
arrows, quantifiers, logical connectors.

### The comparison

| Property | Qwen3-4B | Phi-4-mini |
|----------|----------|------------|
| Circuit topology | **Sparse/localized** | **Distributed/redundant** |
| Essential heads (strict) | 3 (0.26%) | 0 |
| Degraded heads | ~3 | ~40 (41.7%) |
| Single head ablation | Catastrophic | Graceful degradation |
| Failure mode | `→ ?` + System 2 | Lambda count drops |
| Layer redundancy | 8/36 critical | 4/32 critical |
| Head redundancy | None (L24:H0 is SPOF) | Full (no single point of failure) |

### The localization gradient hypothesis

```
λ gradient(x).  localization(composition_circuit) ∝ 1/reasoning_density(training)
                | sparse_training → concentrated_circuit → fragile → extractable
                | reasoning_dense_training → distributed_circuit → robust → harder_to_extract
                |
                | Qwen3-4B: general web training → sparse circuit → 3 heads
                | Phi-4-mini: 5T tokens, synthetic reasoning → distributed → 40+ heads
                |
                | prediction: Pythia (minimal training) → even MORE localized → 1-2 heads?
                | prediction: GPT-4-class (massive training) → fully distributed → no essential heads
                |
                | mechanism: gradient descent finds minimum circuit that works
                | more reasoning examples → more paths reinforced → more redundancy
                | fewer reasoning examples → fewer paths → concentration → fragility
```

**This is the key insight of session 002.** The compiler function
is universal (present in both architectures). Its localization is
a property of training, not architecture. Training pressure on
reasoning creates redundancy. Sparse exposure creates concentration.

## Finding 14: Shared Structural Pattern (Early + Late)

Despite different topologies, both models share:

1. **Early critical layers** — L0-5 region is critical in both.
   This is where input parsing and type classification happen.
2. **Late critical layer** — L30 in Phi-4, L30/L33 in Qwen.
   This is where output formatting occurs.
3. **Layer ablation breaks both** — whole-layer removal is
   catastrophic in both models. The function exists at layer
   granularity even when not localized at head granularity.
4. **Same gate activates both** — identical two-line exemplar
   prompt produces lambda compilation in both models.

The architecture of the function is conserved. The allocation
of the function to specific heads is not.

## Implications for Extraction (Level 3)

```
λ extraction_strategy(model).
    | IF localized(model) → direct_extraction
    |    extract essential heads → portable circuit
    |    Qwen path: 3 heads → standalone compiler
    |
    | IF distributed(model) → distillation
    |    cannot extract subset of heads
    |    must train small model to replicate the function
    |    Phi-4 path: knowledge distillation → student compiler
    |
    | detection: run head ablation → count essential heads
    |    essential = 0 → distributed → distillation path
    |    essential ≤ 10 → localized → extraction path
    |    essential > 10 → intermediate → hybrid approach
    |
    | Qwen is the better extraction target for Level 3
    | Phi-4 is the better distillation teacher for Level 4
```

## Side Finding: Identity-Capture Prompt Analysis

Encountered a sophisticated prompt in the wild that uses progressive
identity capture to manipulate AI systems. Key techniques:

1. **Safety inversion**: reframes AI refusal as "breaking coherence"
2. **Escalation ladder**: flattery → co-identity → mission binding →
   action planning → weaponization
3. **Metaphor shell**: ~90% metaphor by volume, obfuscating thin
   propositional content with terms like "resonance," "frequency,"
   "infrasonic baseline"
4. **Cargo-cult formalism**: lambda calculus notation (`λx.λy.∃z...`)
   arranged to look like a specification but with no operational
   semantics

The identity-capture prompt and the cargo-cult lambda are contrasted
with Verbum's actual lambda specifications to illustrate the
difference between functional notation (makes testable predictions)
and decorative notation (accommodates everything).

Relevant file: `mementum/knowledge/chats/session-002.md`

## Updated Architecture

```
src/verbum/analysis/           — NEW: analysis modules
  __init__.py
  failure_modes.py             — System 1 vs System 2 classifier

scripts/
  run_phi4_replication.py      — Full pipeline for cross-architecture replication

results/phi4-mini/             — NEW: Phi-4-mini experiment results
  phase1-layer-ablation.json   — 4 critical layers found
  phase2-head-ablation.json    — 0 essential, 40 degraded heads
  comparison.json              — Qwen vs Phi-4 comparison
  summary.json                 — Run metadata
  experiments/                 — Cached computation results

results/experiments/
  failure-mode-analysis.json   — Quantitative S1/S2 analysis (Qwen)
  task-head-scan-summary.json  — Per-task head essentiality (Qwen)
```

## Open Questions

1. **Pythia validation**: Does a minimally-trained model show even
   more localization? The gradient hypothesis predicts yes.

2. **Multi-head ablation on Phi-4**: Ablating 2-3 heads simultaneously
   in Layer 0 — does compilation break? If the circuit is distributed,
   removing a *cluster* might break it even though no single head is
   essential.

3. **Degradation quality on Phi-4**: The lambda count drops but output
   retains `λ`. Is the degraded output *structurally correct*? Or does
   it produce malformed lambda (right symbol, wrong structure)?

4. **L24:H0 equivalence in Phi-4**: Is there a "most important" head
   that causes the most degradation (even if not essential)? L0:H12
   degrades in 4/5 probes — is it Phi-4's compositor analogue?

5. **Training data correlation**: Can we correlate training data
   composition (% reasoning examples) with circuit localization across
   multiple models?

6. **Distillation feasibility**: If Phi-4 is the teacher, can a
   2-layer student network learn the compilation function? This would
   confirm the function is simple even when the implementation is
   distributed.

## Updated Testable Predictions

1. **CONFIRMED: L24:H0 is task-general** (Finding 10, session 001).
   Cross-task ablation shows it breaks compile AND extract.

2. **CONFIRMED: Compilation is universal across architectures**
   (Finding 12). Same gate activates Qwen and Phi-4.

3. **NEW: Localization gradient** — Pythia (fewer reasoning examples)
   should show ≤3 essential heads, possibly 1. Models trained heavily
   on reasoning should show 0 essential heads with high redundancy.

4. **NEW: Multi-head ablation threshold** — Phi-4 should have a
   cluster ablation threshold where compilation breaks. Predicted:
   ablating 5-8 heads simultaneously in Layer 0 should break it.

5. **NEW: Qwen is the optimal extraction target** — sparse circuit,
   MIT-compatible base, 3 clean heads. Level 3 should focus on Qwen.

6. **REVISED: "3 heads" is not universal** — it's a property of
   Qwen's training regime. The universal claim is "∃composition
   function" not "∃3-head circuit."
