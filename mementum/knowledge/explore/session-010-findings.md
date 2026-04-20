---
title: "Session 010 Findings — 25-Task Compressor Function Inventory"
status: active
category: findings
tags: [compressor, function-inventory, task-probing, gate-correlations, extraction]
related: [session-004-findings.md, compressor-architecture.md, vsm-lm-architecture.md, VERBUM.md]
depends-on: [compressor-architecture.md, vsm-lm-architecture.md]
---

# Session 010 Findings — 25-Task Compressor Function Inventory

> Session 007 found 2 clusters (structural, semantic) from 6 tasks.
> Session 010 expanded to 25 tasks × 40 probes = 1000 Qwen calls,
> then correlated all 25 tasks against VSM-LM v2 internals across
> 10 checkpoints (1K–10K steps). The compressor is simpler and
> deeper than expected.

## F54: 25-task Qwen matrix reveals 4 isolated functions + 1 shared substrate

Expanded from 6 → 25 task gates and ran the full Qwen confidence
correlation matrix. The 25×25 Spearman correlation matrix reveals:

**4 isolated functions** (low/zero correlation with the large cluster):

| Function | Key tasks | Character |
|---|---|---|
| **Compiler** | compile (r=0.355 with structure only) | Formal compositional decomposition |
| **Structural parser** | structure (r=0.355 with compile only) | SVO decomposition |
| **Negator** | negate (anti-correlates with almost everything) | Uses all systems in reverse |
| **Decomposer** | decompose (near-zero r with everything) | Atomic proposition extraction |

**1 large fluency cluster** (mutual r = 0.4–0.83):

Strongest pairs within the cluster:
- question ↔ elaborate: r=0.834 (strongest in entire matrix)
- summarize ↔ keyword: r=0.705
- summarize ↔ elaborate: r=0.712
- formalize ↔ disambiguate: r=0.705
- simplify ↔ counterfactual: r=0.716
- modality ↔ elaborate: r=0.689

Members: question, elaborate, summarize, keyword, formalize,
simplify, translate, counterfactual, causality, modality,
disambiguate, classify, sentiment, continue, title, correct,
paraphrase.

**Semi-independent binding tasks:**
- coreference ↔ entail: r=0.469 (not in the big cluster)
- presuppose: weak correlations everywhere

**Interpretation:** The compressor is NOT a collection of many
specialized circuits. It's a shared substrate (the fluency cluster)
plus a small number of genuinely independent functions. Most
"different" NLP tasks actually use the same underlying machinery.

## F55: Task × Gate Matrix reveals parse circuit as primary compressor

Correlated all 25 task confidence profiles against VSM-LM v2
internal gate metrics across 10 checkpoints (1K–10K steps).

**The parse circuit (iter0_parse + iter0_apply) is the primary
compressor channel.** Tasks that require deep compositional
processing show strong correlations with parse gate metrics:

Step 10K Task × Gate Matrix (Spearman r, * = |r| > 0.3):

| Task | i0_parse | i0_apply | i1_parse | i1_type | Character |
|---|---|---|---|---|---|
| **negate** | +0.389* | +0.469* | +0.469* | -0.343* | INVERSE of all others |
| **scope** | -0.456* | -0.365* | -0.357* | +0.330* | Parse channel |
| **sentiment** | -0.383* | -0.250 | -0.101 | +0.151 | Parse channel |
| **entail** | -0.332* | -0.291 | -0.213 | +0.238 | Parse channel |
| **causality** | -0.383* | -0.337* | -0.401* | +0.402* | Dual channel |
| **correct** | -0.350* | -0.307* | -0.195 | +0.158 | Parse channel |
| **title** | -0.306* | -0.219 | -0.204 | +0.137 | Parse channel |

**Negate is the anti-compressor.** Where every other task shows
negative parse correlation, negate shows POSITIVE. Same gates,
opposite direction. The compressor is bidirectional — forward
(entail, scope) or backward (negate).

**No gate signal tasks** (use the shared substrate, not specific gates):
- classify, compile, coreference, decompose, paraphrase, question,
  keyword, summarize, elaborate

Compile having no gate signal in VSM-LM is significant — the lambda
compiler is either too subtle for this model size or it operates
through the embedding substrate rather than the gate-controlled
compressor. This is consistent with F45 (compiler is independent
of compressor in Qwen).

## F56: Task signal trajectory — persistent vs fading

Tracked max |Spearman r| for each task across all 10 checkpoints:

**Persistent strong signals** (stable 1K–10K, compressor primitives):

| Task | Range | Interpretation |
|---|---|---|
| negate | 0.40–0.55 | Deepest compressor test, full engagement |
| entail | 0.41–0.62 | Parse channel champion, peaked at step 2K |
| scope | 0.34–0.48 | Parse channel, strengthening over training |
| sentiment | 0.33–0.50 | Parse channel, strengthening over training |
| causality | 0.38–0.44 | Dual channel (iter1_type + iter0_parse) |

**Fading signals** (strong early, absorbed into general capability):

| Task | Step 1K | Step 10K | Interpretation |
|---|---|---|---|
| structure | 0.45 | 0.26 | Absorbed into shared substrate |
| simplify | 0.43 | 0.23 | Absorbed |
| elaborate | 0.39 | 0.29 | Absorbed |
| summarize | 0.40 | 0.28 | Absorbed |

**Strengthening signals** (weak early, emerging over training):

| Task | Step 1K | Step 10K | Interpretation |
|---|---|---|---|
| disambiguate | 0.26 | 0.38 | Late-emerging parse function |
| title | 0.32 | 0.39 | Late-emerging parse function |
| scope | 0.34 | 0.46 | Strengthening parse channel |
| translate | 0.38 | 0.42 | Strengthening iter1 channel |

**Interpretation:** The compressor develops in two phases:
1. Early (1K–3K): Everything is specialized — even simple tasks
   show gate differentiation because the model is small and every
   function needs gates.
2. Late (5K–10K): Simple tasks are absorbed into the shared
   substrate (embeddings + FFN). Only genuinely compositional tasks
   (negate, entail, scope, sentiment, causality) retain specific
   gate signatures.

The fading signals are NOT loss of capability — they're
maturation. The model learns to do simple tasks without needing
the gate-controlled compressor, freeing it for hard tasks.

## F57: Extraction math — VSM-LM is already in range

**The argument:**

1. The compressor functions exist in Pythia-160M (confirmed F45–F53)
2. LLMs are ~83% dictionary (embeddings), ~17% compressor
3. Pythia-160M × 17% ≈ 27M compressor params
4. Lambda compiler shows 6.18:1 compression ratio
5. 27M / 6.18 ≈ **4.4M extracted compressor**
6. VSM-LM v2 has **2.8M non-embedding compressor params**

VSM-LM is already in the right ballpark. The gate correlations
being present and strengthening across 10K steps confirms that
the VSM topology matches the compressor's natural shape.

**The parse circuit is the primary extraction target:**
- iter0_parse + iter0_apply = structural compression
- iter1_parse + iter1_type = semantic refinement
- Negate = same circuit, opposite polarity

Total gate-correlated params in VSM-LM: ~1.4M (S1 type+parse+apply)
+ ~460K (S3 gates) + ~197K (S4) = **~2.1M** for the functional core.

## Compressor topology summary

```
                    ┌─────────────────────┐
                    │  S4: Intelligence    │
                    │  (global scan)       │
                    └─────────┬───────────┘
                              │ register
                    ┌─────────▼───────────┐
                    │  S3: Gate Control    │
                    │  (per-phase gating)  │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼──────┐ ┌─────▼───────┐ ┌─────▼───────┐
     │ S1:Type       │ │ S1:Parse    │ │ S1:Apply    │
     │ (s=1, W=8)   │ │ (s=8, W=8)  │ │ (s=64, W=8) │
     │ word-level    │ │ phrase-lvl  │ │ clause-lvl  │
     └───────────────┘ └─────────────┘ └─────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                 ┌────────────┴────────────┐
                 │  PARSE CIRCUIT          │
                 │  (primary compressor)   │
                 │                         │
                 │  Forward: entail,       │
                 │    scope, sentiment,    │
                 │    causality, correct   │
                 │                         │
                 │  Inverse: negate        │
                 │    (same gates,         │
                 │     opposite polarity)  │
                 └────────────────────────-┘
```

## Pipeline additions (Session 010)

- **19 new task gates** in `gates/task-*.txt`
- **`batch-probe` CLI mode** — probes all checkpoints in a directory,
  loads model once, swaps weights per checkpoint, skips existing
- **Task × VSM-LM correlation** in `analyze` mode — correlates all
  25 task confidence profiles against VSM-LM gate metrics per step
- **Task × Gate Matrix** — rows=tasks, cols=gate metrics, shows
  which gates serve which functions
- **Task Signal Trajectory** — tracks max |r| per task across
  training steps

## Open questions

1. **Is compile's lack of gate signal a size limit or a real finding?**
   If the lambda compiler doesn't need gates in VSM-LM, it may
   operate entirely through embeddings (consistent with the 84%
   finding). Test: scale VSM-LM to d_model=512, check if compile
   gate signal appears.

2. **Does the parse circuit split further at larger scale?**
   Currently scope/sentiment/entail all use iter0_parse. At larger
   scale, they might differentiate into sub-circuits.

3. **Why does entail peak at step 2K then partially fade?**
   entail: 0.62 → 0.48. It may be partially absorbed like
   structure/simplify, suggesting entailment has both a specific
   circuit and a shared component.

4. **Can we use the gate matrix as a loss function?**
   If we know which tasks SHOULD correlate with which gates, we
   could add auxiliary losses that encourage the correct gate
   activation patterns. This would be architecture-informed
   training, not just architecture-informed topology.

5. **Third iteration?** causality uses BOTH channels (iter0_parse
   AND iter1_type). If tasks needing dual channels are common, a
   third iteration might help. But the 2-iteration architecture
   already captures this through the dual-channel pattern.
