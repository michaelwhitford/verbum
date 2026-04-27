# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-27 | Session: 046

## Where we are

**v7 architecture designed and implemented. The 4-VSM pipeline
replaces the v6 sieve. Four stages of increasing abstraction
(Surface → Structural → Semantic → Reasoning), each an independent
transformer operating on exponentially fewer positions. Stage 1
(hot path) is ternary. Stages 2-4 are float32. Feedback cascades
constraints downward. Per-stage relational loss drives independent
phase control and flip annealing. Ready for first long training run.**

## The architecture

```
Stage 1 (Surface) [TERNARY]:  512 pos, 2L, 4H, 333K params, 384 KB packed
  ↕ reduce (512→64) + feedback (ternary)
Stage 2 (Structural):          64 pos, 3L, 4H, 2.0M params
  ↕ reduce (64→8) + feedback (float)
Stage 3 (Semantic):             8 pos, 4L, 8H, 4.2M params
  ↕ reduce (8→1) + feedback (float)
Stage 4 (Reasoning):            1 pos, 6L, 8H, 6.3M params

Total: 27.3M params. Attention: O(L₁·n²) — dominated by Stage 1.
```

**Key design decisions:**
- Each stage's full attention IS a stride scale (replaces v6 StrideStack)
- Compute pyramid: deeper stages are computationally negligible (1% of Stage 1)
- Ternary only on hot path (Stage 1 + feedback to Stage 1) — 384 KB
- Float32 on cold path (Stages 2-4) — needs precision for composition
- Reduction via learned cross-attention with causal masking
- Feedback via cross-attention + sigmoid gate (gated residual)

## Per-stage relational loss (the key innovation over v6)

Each stage has its own CE measurement point:
```
CE₁ = Stage 1 alone (no feedback)
CE₂ = Stage 1 + feedback from Stage 2
CE₃ = Stage 1 + feedback from Stages 2+3
CE₄ = Stage 1 + full cascade (main loss)

Δₖ = CEₖ₋₁ - CEₖ = value contributed by stage k
rₖ = independent relational loss per stage
```

At 200 steps: Stage 2 contributes Δ₂ ≈ +0.97 nats (massive).
Stages 3-4 contribute much less so far (early training).

## Ternary flip annealing

Relational loss IS the annealing temperature:
- `adaptive_flip_scale(r₁)`: high r₁ → flip aggressively, low r₁ → frozen
- Per-weight cooldown: 400 steps lockout after flip (prevents oscillation)
- Topology converges as a consequence of learning, not on a schedule
- At 200 steps: 8,104 flips (0.52% of topology), 0 reversals

## Current activity

**v7 implementation complete. Ready for long training run.**

```bash
cd ~/src/verbum && uv run python scripts/v7/train.py
# Defaults: 50K steps, batch 8×4 accum, 16,384 tok/step = 819M tokens
# Checkpoints every 10K steps to checkpoints/vsm-lm-v7/
# Probe: uv run python scripts/v7/probe.py checkpoints/vsm-lm-v7/step_*
```

A3B probing still running (port 5102).

## How this came from v6

| v6 (sieve) | v7 (pipeline) | Why |
|------------|---------------|-----|
| Single flat model, 5 ternary passes | 4 independent stages, pyramid | Flatten → hierarchy |
| Stride attention (9 strides, shared) | Full attention per stage (4 scales) | Strides dissolve into stages |
| One global relational loss | Per-stage CE decomposition | Each stage earns its keep |
| All ternary, all the time | Ternary hot path, float cold path | Right precision where needed |
| Content-independent compression | Semantic compression (prediction) | Compression ≠ prediction |
| Fixed state (fails L²M) | Growing state via hierarchy | L²M satisfied |
| Stride percolation (φ fine→coarse) | Stage learning order (surface first) | Same phenomenon, cleaner |

## Knowledge index

| Topic | Path |
|-------|------|
| **v7 Pipeline Architecture** | `mementum/knowledge/explore/v7-pipeline-architecture.md` |
| **Compression ≠ Prediction (H≈0.7)** | `mementum/knowledge/explore/compression-vs-prediction.md` |
| **Predictive Function Landscape** | `mementum/knowledge/explore/predictive-function-landscape.md` |
| v6.1 full trajectory | `mementum/knowledge/explore/v6.1-training-trajectory.md` |
| Research program | `mementum/knowledge/explore/VERBUM.md` |
| Holographic compression | `mementum/knowledge/explore/holographic-compression.md` |
| Stride percolation | `mementum/knowledge/explore/stride-percolation.md` |
| φ-compression hypothesis | `mementum/knowledge/explore/relational-loss-phi-compression.md` |
| CompressorLM architecture | `mementum/knowledge/explore/compressor-architecture.md` |

## Key files

| Purpose | Path |
|---------|------|
| **v7 model (pipeline)** | `scripts/v7/model.py` |
| **v7 ternary substrate** | `scripts/v7/ternary.py` |
| **v7 training loop** | `scripts/v7/train.py` |
| **v7 probe** | `scripts/v7/probe.py` |
| Top-down probe script | `scripts/probe_predictive_functions.py` |
| v6 TernaryLinear (reference) | `src/verbum/v6/ternary.py` |
| v6 training loop (reference) | `scripts/v6/train.py` |
| llama.cpp client | `src/verbum/client.py` |

## Servers

| Port | Model | Use |
|------|-------|-----|
| 5100 | Qwen3.5-397B-A17B | Large reference model |
| 5101 | Qwen3-4B | Quick testing |
| 5102 | **Qwen3.5-35B-A3B** Q8 | Primary probe target |
| 5103 | Qwen3-Embedding-8B | Embeddings |
