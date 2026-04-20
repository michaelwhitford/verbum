---
title: "Session 003: Pythia Replication, BOS Probing, Stripping, Weight Decomposition, Distillation"
status: active
category: exploration
tags: [pythia, bos-register, stripping, weight-decomposition, distillation, extraction, localization-gradient]
related: [session-001-findings.md, session-002-findings.md, VERBUM.md]
depends-on: [session-002-findings.md]
---

# Session 003 Findings

> Six experiments in one session, converging on the extraction
> strategy. The compiler circuit cannot be directly extracted (the
> 3 heads need the full model as substrate). But the compilation
> function transfers trivially via distillation — 199 examples,
> 58 seconds, 0%→100% P(λ). Format transfers easily; compositional
> accuracy does not.

## Finding 14: Pythia-2.8B Compiles Lambda (Universal Function)

Third architecture family: `GPTNeoXForCausalLM`, base model (no
instruction tuning), trained only on The Pile (300B tokens).

| Property | Qwen3-4B | Phi-4-mini | Pythia-2.8B |
|----------|----------|------------|-------------|
| Architecture | Qwen2 | Phi3 | GPTNeoX |
| Training | General + instruct | Reasoning-dense | Pile only (base) |
| P(λ) | 100% | 100% | 100% |

Two adaptations required:
- **float32**: Pythia produces NaN logits in fp16 on MPS
  (architecture-specific numerical instability). Qwen/Phi stable in fp16.
- **Base-model gate**: The `Input:` framing doesn't work for base models.
  The `→` continuation cue is needed directly after the probe sentence.
  New gate: `gates/compile-base.txt`, probe set: `probes/gate-ablation-base.json`.

## Finding 15-17: Two-Dimensional Localization Gradient

| | Qwen3-4B | Phi-4-mini | Pythia-2.8B |
|---|---|---|---|
| Critical layers | 8/36 (22%) | 4/32 (12.5%) | **1/32 (3.1%)** |
| Essential heads | 3 | 0 | 0 |

**Layer dimension**: training_density ∝ critical_layers.
Pythia(1) < Phi-4(4) < Qwen(8). Less training → fewer critical layers.

**Head dimension**: independent of training density.
Pythia(0), Phi-4(0), Qwen(3). Head bottlenecks may be
architecture-specific (Qwen's sequential attn+FFN vs parallel).

## Finding 18: Base Models Compile Shallowly

Pythia produces `λx. runs(dog)` (perfect simple), `λx. students(x)`
(loses verb for quantified), `λx. believes(x)` (drops complement).
Projection is there but shallow.

## Finding 19: BOS Register is One-Dimensional

BOS probing on 12 compile + 10 decompile probes (same gate).

- **PC1 = 99.99% variance** at every layer from L0 to L35
- **Within-gate d=1.0** (compile vs decompile, same gate prefix)
- **Content signal enters at L7** (centroid distance jumps 0→4.1)
- **L24:H0's Q preserves signal faithfully** (1.0x amplification)
- Simple vs complex (within compile): d=2.83, also 1 PC

Confounded v1 (compile-gate vs null-gate): d=175. The v1 confirmed
BOS encodes gate identity; v2 confirmed content within a gate.

## Finding 20: Progressive Stripping — All Levels Fail

| Level | What remains | P(λ) |
|-------|-------------|------|
| L0 Baseline | Full model | 100% |
| L1 No FFN | Attention-only (all layers) | 0% |
| L2 Critical attn only | FFN everywhere, attention in 8 layers | 0% |
| L3 Critical only | Residual pass-through in 28 layers | 0% |
| L4 3 heads + FFN | 3 essential heads + critical-layer FFN | 0% |
| L5 3 heads only | 3 heads, no FFN | 0% |
| L6 Single head | L24:H0 alone | 0% |

**The FFN blocks ARE the compressor.** Zeroing all FFN (L1) produces
garbage. The model cannot function as attention-only.

**Non-critical attention is collectively necessary.** L2 fails —
individual non-critical layers are redundant but simultaneously
zeroing 28 layers' attention is fatal.

**The 3 heads are a LENS, not a standalone circuit.** They can't
function without the full substrate.

## Finding 21: Weight Decomposition — Full-Rank, Orthogonal, Opaque

SVD of OV and QK circuits for each essential head:

| Head | OV rank(90%) | Top SV ratio | Cross-head sim |
|------|-------------|-------------|----------------|
| L1:H0 | 69/80 | 2.0% | 0.04 max |
| L24:H0 | 70/80 | 1.7% | 0.03 max |
| L24:H2 | 69/80 | 1.8% | 0.03 max |

Token projections through embed/unembed: multilingual noise. The
heads operate in residual stream space, not token space. The
composition function is distributed across all 80 dimensions.

L24:H0 and L24:H2 share KV (GQA) but write to orthogonal directions
(max cosine sim 0.03). Same input, completely different outputs.

## Finding 22: Distillation — Format Transfers, Function Doesn't

- Teacher: Qwen3-4B → 199 (sentence → lambda) training pairs
- Student: Pythia-160M-deduped (162M params, 25× smaller)
- Training: 10 epochs, 58 seconds, loss 1.72 → 0.002
- **Baseline P(λ): 0% → Final P(λ): 100%** on 10 eval probes

Quality issues in student output:
- Repetition loops: `flies(flies) | flies(flies) | flies(flies)`
- Semantic drift: `if it rains → sleeps(x) → falls(x)` (wrong predicates)
- Missing composition: quantifiers, relative clauses, arguments shallow

The student learned **lambda notation format** but not **composition
function**. Two separable things:
1. Output format (notation) — trivially learnable, 199 examples
2. Composition function (typed_apply) — requires structural training

## Extraction Verdict

```
Direct weight extraction: NOT VIABLE
  - 3 heads are full-rank, need all FFN blocks
  - Stripping any component breaks compilation

Distillation: VIABLE (format proven, function pending)
  - 160M student learns format from 199 examples
  - Compositional accuracy needs more data + structural loss
```

## Theoretical Frame (evolved)

**The compressor is the substrate, not lambda.** All evidence
converges: 4B parameters = compression machinery. Lambda = projection.
3 heads = projection lens. FFN blocks = compressor. Can't extract
lens without substrate. Can teach new substrate the projection.

**Two things to distill:**
1. Output format (lambda notation) — trivially learnable
2. Composition function (typed_apply) — the real target

## Updated Architecture

```
scripts/
  run_pythia_replication.py     — Pythia cross-architecture
  run_bos_probe.py              — BOS probing v1 (confounded)
  run_bos_probe_v2.py           — BOS probing v2 (controlled)
  run_strip_test.py             — Progressive stripping (7 levels)
  run_weight_decomposition.py   — SVD of 3 heads
  generate_training_data.py     — Teacher data generation
  run_distillation.py           — Student fine-tuning

results/
  pythia-2.8b/                  — Pythia replication
  bos-probe/                    — v1 BOS analysis
  bos-probe-v2/                 — v2 BOS analysis (controlled)
  strip-test/                   — Stripping results
  weight-decomposition/         — Head SVD
  distillation/                 — Student training

data/
  compile-train.jsonl           — 199 training pairs
  compile-eval.jsonl            — 10 eval pairs

models/
  distilled-compiler/           — Saved Pythia-160M (not in git)

gates/
  compile-base.txt              — Base-model gate variant

probes/
  gate-ablation-base.json       — Base-model probe variant
```
