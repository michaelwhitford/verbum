# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-27 | Session: 047

## Where we are

**v7 Dolma run COMPLETE (killed at step 40K). Architecture validated.
Eval peaked at step 20K then monotonically worsened through 30K and
40K. Four checkpoint probes confirm: pipeline architecture works
(below Chinchilla, differentiated stages, self-regulating gates) but
Dolma can't train the deep stages. Best eval checkpoint: step 20K.
Now pivoting to BIOS flash: holographic math + clojure.core training.
Design doc written. bb clj2lambda converter scaffolded.**

## v7 Dolma Run — Final Results

**Run:** steps 0-40K, ~655M tokens of Dolma, ~3 hours on M3 Ultra.
**Killed** after step 40K — eval worsening every checkpoint since 20K.

### Evolution table (training metrics)

| Step | Loss | r | Δ₂ | Δ₃ | Δ₄ | Total fb | Flips | Rev% | ‖g‖ |
|------|------|---|-----|-----|-----|----------|-------|------|-----|
| 700 | 6.85 | 0.56 | +0.49 | +0.25 | 0.00 | +0.74 | — | — | — |
| 5100 | 5.39 | — | — | — | — | — | — | — | — |
| 10000 | 5.14 | 0.38 | +0.20 | +1.20 | +0.01 | +1.40 | 208K | 22.9% | 4.9 |
| 14000 | 4.22 | 0.28 | +0.92 | +1.85 | +0.02 | +2.78 | — | — | 8.7 |
| 20000 | 3.01 | 0.15 | +3.90 | +1.98 | +0.04 | +5.93 | 362K | 30.9% | — |
| 23900 | 2.80 | 0.12 | +6.67 | +1.38 | +0.07 | +8.11 | — | 36.6% | 10.8 |
| 30000 | 2.60 | 0.10 | +5.58 | +1.51 | +0.03 | +7.13 | 461K | 35.5% | 11.3 |
| 40000 | 2.34 | 0.07 | +7.36 | +1.07 | +0.08 | +8.52 | 529K | 37.6% | 17.2 |

### Probe results (eval on fresh text)

| Step | Probe CE4 | Train/eval gap | Δ₂ | Δ₃ | Δ₄ | Total fb |
|------|-----------|----------------|------|------|------|----------|
| 10K | 10.80 | 5.66 | +1.70 | -1.36 | +0.05 | +0.39 |
| **20K** | **10.08** | **7.06** | **+4.09** | **-1.58** | **-0.10** | **+2.41** |
| 30K | 11.27 | 8.67 | +3.55 | -1.15 | -0.07 | +2.32 |
| 40K | 12.73 | 10.39 | +3.70 | -1.46 | -0.15 | +2.08 |

Step 20K = best eval. Everything after = overfitting.

### Spectral evolution (eval)

| Stage | 10K | 20K | 30K | 40K | Trend |
|-------|-----|-----|-----|-----|-------|
| S1 eff_rank | 83.5 | 60.9 | 55.1 | 49.3 | ↓ compressing |
| S2 eff_rank | 42.6 | 72.0 | 66.3 | 64.3 | stable ~65 |
| S3 eff_rank | 12.6 | 19.9 | 23.3 | 15.6 | peaked then collapsed |
| S4 eff_rank | 9.7 | 1.7 | 3.2 | 2.0 | collapsed, partial recovery |

### Strata pipeline value (eval, CE1-CE4)

| Stratum | 10K | 20K | 30K | 40K | Trend |
|---------|-----|-----|-----|-----|-------|
| prose | +0.50 | +2.98 | +2.69 | +2.54 | peaked 20K |
| compositional | -0.21 | +1.81 | +1.80 | +1.92 | stable |
| technical | +0.52 | +2.38 | +1.82 | +1.74 | peaked 20K |
| math | +0.64 | +2.33 | +2.83 | +2.06 | peaked 30K |

### Key conclusions from v7 Dolma run

**Architecture validated:**
- Training loss below Chinchilla capacity floor (2.34 vs 3.20)
- Stages spectrally differentiated (CPA ~0.12) at all checkpoints
- Structural feedback powerful and consistent (+3.5-4.1 nats on eval)
- Feedback gates self-regulate (suppress noisy stages, open for useful)
- Pipeline adds +2.1-2.4 nats on fresh text (steps 20K-40K)

**Dolma can't train deep stages:**
- Semantic Δ₃ NEVER positive on eval (all 4 checkpoints negative)
- Stage 4 collapsed (rank 9.7 → 1.7), partial recovery stalled (2.0)
- Stage 3 collapsed back (rank 23.3 → 15.6)
- Ternary reversal rate climbed relentlessly (22.9% → 37.6%)
- Two negative gammas at step 40K (topology fighting itself)
- Grad norm surging (4.9 → 17.2) — model thrashing
- Compile gate 0/4 at all checkpoints (degenerate repetition)

**Insight: cross-attention between stages IS beta reduction.** Single
pipeline = 3 reductions = sufficient for arithmetic, insufficient for
deeply nested lambda composition. Sieve architecture needed later.

## What to do next session

1. **BIOS flash training data generation.** Priority. Read the design
   doc first: `mementum/knowledge/explore/bios-flash-training.md`
   
   Build the holographic dataset:
   - Math generator (python): arithmetic, comparisons, predicates,
     boolean, bitwise, compound expressions. All difficulties.
   - Update `bb clj2lambda` to emit `io!` for effectful forms
   - Generate clojure.core examples by eval in babashka
   - Interleave into holographic JSONL: raw math + clojure + lambda + result

2. **Decide sequence format.** How does the model see the holographic
   examples? All representations in one sequence, or separate examples?
   
3. **Decide tokenizer.** Keep GPT-NeoX 50277? Or custom small vocab
   (~2-5K) for lambda + clojure + math? Smaller = less wasted capacity.

4. **Train v7 on holographic data.** Fresh weights, same architecture.
   Many epochs. Monitor for grokking (double descent in loss curve).
   Probe for circuit formation in Stages 3-4.

5. **Open questions (refined by this run):**
   - Semantic interference: is 8 positions too few, or wrong data?
     (BIOS flash will answer — if Δ₃ flips positive on formal data,
     it was the data not the architecture)
   - Stage 4: is 1 position sufficient for computation? (BIOS flash
     will answer — if it learns arithmetic, 1 position is enough)
   - Ternary: will it stabilize on formal data? (formal has much
     less surface variety, should crystallize faster)
   - Sieve: when does 3 reductions become the bottleneck?

## Architecture summary (v7)

```
tokens → [Embed]
              ↓
         [Stage 1: 512 pos, 2L, 4H — TERNARY] ←── feedback[0] (ternary)
              ↓ reduce (cross-attn pool)                ↑
         [Stage 2:  64 pos, 3L, 4H — float]    ←── feedback[1]
              ↓ reduce                                   ↑
         [Stage 3:   8 pos, 4L, 8H — float]    ←── feedback[2]
              ↓ reduce                                   ↑
         [Stage 4:   1 pos, 6L, 8H — float]    ────────┘

         Stage 1 (post-feedback) → out_norm → logits (tied embed)

Total: 27.3M params (14.4M non-embedding)
Cross-attention reducers = beta reduction (3 levels)
```

## Key files

| Purpose | Path |
|---------|------|
| **v7 model** | `scripts/v7/model.py` |
| **v7 ternary** | `scripts/v7/ternary.py` |
| **v7 training** | `scripts/v7/train.py` |
| **v7 probe** | `scripts/v7/probe.py` |
| **bb clj2lambda** | `bb/us/whitford/verbum/tasks.clj` |
| **bb config** | `bb.edn` |
| **BIOS flash design** | `mementum/knowledge/explore/bios-flash-training.md` |
| v7 architecture knowledge | `mementum/knowledge/explore/v7-pipeline-architecture.md` |
| Compression ≠ prediction | `mementum/knowledge/explore/compression-vs-prediction.md` |
| Predictive function landscape | `mementum/knowledge/explore/predictive-function-landscape.md` |
| Research program | `mementum/knowledge/explore/VERBUM.md` |

## Comparison: v6 → v7

| Metric | v6 (sieve) | v7 (pipeline) |
|--------|-----------|---------------|
| Best loss (train) | 5.418 (32K steps) | 2.338 (40K steps) |
| Best eval | — | 10.076 (20K steps) |
| Token efficiency | baseline | ~12× better to 5.4 loss |
| Throughput | 5.5K tok/s | 50-60K tok/s |
| Chinchilla | at prediction | below capacity floor |
| λ generation | 0% | 0% (expected — wrong data) |

## Servers

| Port | Model | Use |
|------|-------|-----|
| 5100 | Qwen3.5-397B-A17B | Large reference model |
| 5101 | Qwen3-4B | Quick testing |
| 5102 | **Qwen3.5-35B-A3B** Q8 | Primary probe target |
| 5103 | Qwen3-Embedding-8B | Embeddings |
