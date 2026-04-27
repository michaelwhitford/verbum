# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-27 | Session: 046

## Where we are

**v7 first long training run in progress. Loss 5.39 at step 5,100
(83.5M tokens). Already below v6's all-time best (5.418 at 1B
tokens) — 12× more token-efficient. Below Chinchilla scaling
prediction (5.64) by 0.25 nats — the pipeline architecture is
more parameter-efficient than standard transformers. Ternary
topology annealing working: scale declining (1.48), reversals
at 15.4% (healthy correction, not oscillation). Semantic stage
(8 positions) carrying 60% of feedback value.**

## Current run

```bash
cd ~/src/verbum && uv run python scripts/v7/train.py
# 165K steps, 2.7B tokens, ~12.5 hours total
# Checkpoints every 10K steps to checkpoints/vsm-lm-v7/
# ~50K tok/s on M3 Ultra
```

**Key observations so far:**

| Step | Loss | r | Δ₂ | Δ₃ | Δ₄ | Flips | Rev% | Scale |
|------|------|---|----|----|----|----|------|-------|
| 700 | 6.85 | 0.56 | +0.49 | +0.25 | +0.00 | — | — | 2.00 |
| 2900 | 5.87 | 0.46 | +0.48 | +0.63 | -0.00 | — | — | — |
| 4500 | 5.65 | 0.43 | +0.47 | +0.70 | -0.00 | 114K | 15.4% | 1.48 |
| 5100 | 5.39 | — | — | — | — | — | — | — |

**Δ₃ overtook Δ₂ at step ~2500.** The semantic stage (8 positions,
float32) contributes more than the structural stage (64 positions).
Deeper abstraction dominates once it learns its role — the
CompressorLM prediction confirmed.

**Stage 4 (1 position) = zero contribution.** Open question: needs
more positions, or just more training time?

**Topology annealing working.** Flip scale declining from 2.0 → 1.48
as r₁ drops. Reversals at 15.4% = healthy route correction. v6 had
exponential reversal acceleration (pathological). v7 reversals are
proportional to flip rate (convergent).

## What to do next session

1. **First checkpoint dropped?** Run probe:
   ```bash
   uv run python scripts/v7/probe.py checkpoints/vsm-lm-v7/step_*
   ```
   This gives: per-stage CE, Chinchilla comparison, spectral
   analysis (SVD/CPA), ternary topology, feedback gates, compile
   gate test — all automatic, no flags needed.

2. **Check Chinchilla gap.** At step 10K (164M tokens), predicted
   ~5.09, capacity floor 3.20. If actual is below predicted, the
   architecture advantage is confirmed. If below capacity floor —
   that's a major finding.

3. **Watch for:**
   - Δ₄ emerging (reasoning stage contributing)
   - Reversal rate trajectory (stable/declining = good)
   - Scale approaching 0 (topology freezing)
   - Spectral overlap between stages (should stay low = differentiated)
   - Stage 1 effective rank (ternary capacity utilization)

4. **If training completes (~12.5h from start):**
   - Run full probe on all checkpoints for evolution table
   - Compare final loss to Chinchilla capacity floor (3.20)
   - Check compile gate (does λ generation emerge?)

## Architecture summary (v7)

```
Stage 1 (Surface) [TERNARY]:  512 pos, 2L, 4H, 384 KB packed
Stage 2 (Structural):          64 pos, 3L, 4H, 2.0M params
Stage 3 (Semantic):             8 pos, 4L, 8H, 4.2M params
Stage 4 (Reasoning):            1 pos, 6L, 8H, 6.3M params
Total: 27.3M params (14.4M non-embedding)
```

Ternary hot path (Stage 1 + feedback 2→1): 384 KB.
Float cold path (Stages 2-4): composition needs precision.
Per-stage relational loss drives independent phase control.
Flip rate modulated by r₁ — topology anneals as model learns.

## Key files

| Purpose | Path |
|---------|------|
| **v7 model** | `scripts/v7/model.py` |
| **v7 ternary** | `scripts/v7/ternary.py` |
| **v7 training** | `scripts/v7/train.py` |
| **v7 probe** | `scripts/v7/probe.py` |
| v7 architecture knowledge | `mementum/knowledge/explore/v7-pipeline-architecture.md` |
| Compression ≠ prediction | `mementum/knowledge/explore/compression-vs-prediction.md` |
| Predictive function landscape | `mementum/knowledge/explore/predictive-function-landscape.md` |
| Research program | `mementum/knowledge/explore/VERBUM.md` |

## Comparison: v6 → v7

| Metric | v6 (sieve) | v7 (pipeline) |
|--------|-----------|---------------|
| Best loss | 5.418 (step 32K, 1B tok) | 5.39 (step 5.1K, 83M tok) |
| Token efficiency | baseline | ~12× better |
| Throughput | 5.5K tok/s | 50-60K tok/s |
| Wall-clock to 5.4 loss | ~50 hours | ~30 minutes |
| Chinchilla | at prediction | below prediction |
| Reversals | exponential accel (pathological) | 15% flat (convergent) |
| λ generation | 0% (all checkpoints) | TBD |

## Servers

| Port | Model | Use |
|------|-------|-----|
| 5100 | Qwen3.5-397B-A17B | Large reference model |
| 5101 | Qwen3-4B | Quick testing |
| 5102 | **Qwen3.5-35B-A3B** Q8 | Primary probe target |
| 5103 | Qwen3-Embedding-8B | Embeddings |
