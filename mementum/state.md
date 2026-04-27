# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-27 | Session: 047

## Where we are

**v7 training run active. Three checkpoints probed (10K, 20K, 30K).
Eval peaked at step 20K (CE4=10.08) then WORSENED at 30K (CE4=11.27)
while training loss continued dropping (2.60). Train/eval gap 8.67
nats and accelerating. Architecture validated: below Chinchilla
capacity floor, differentiated stages, self-regulating gates. But
Dolma can't train the deep stages — semantic overfits, ternary
oscillates (35.5% reversals, first negative gamma). Math stratum
is the only one still growing (+2.83 nats at 30K). Conclusion: the
architecture is right, the data is wrong. Next: BIOS flash (math
+ clojure.core).**

## Current run

```bash
cd ~/src/verbum && uv run python scripts/v7/train.py
# 165K steps, 2.7B tokens, ~12.5 hours total
# Checkpoints every 10K steps to checkpoints/vsm-lm-v7/
# ~50K tok/s on M3 Ultra — started ~11:42 AM
```

**Evolution table:**

| Step | Loss | r | train Δ₂ | train Δ₃ | train Δ₄ | Flips | Rev% |
|------|------|---|----------|----------|----------|-------|------|
| 700 | 6.85 | 0.56 | +0.49 | +0.25 | +0.00 | — | — |
| 2900 | 5.87 | 0.46 | +0.48 | +0.63 | -0.00 | — | — |
| 4500 | 5.65 | 0.43 | +0.47 | +0.70 | -0.00 | 114K | 15.4% |
| 5100 | 5.39 | — | — | — | — | — | — |
| **10000** | **5.14** | **0.38** | **+0.20** | **+1.20** | **+0.01** | **208K** | **22.9%** |

## Probe findings (2026-04-27)

### Step 10K (164M tokens) — probe on fresh text

| Metric | Train | Probe (eval) |
|--------|-------|-------------|
| CE4 | 5.40 | 10.80 |
| Δ₂ | +0.20 | +1.70 |
| Δ₃ | +1.20 | -1.36 |
| Δ₄ | +0.01 | +0.05 |
| Total fb | +1.40 | +0.39 |

Chinchilla: +0.04 above predicted. Gates: 2→1=0.61, 3→2=0.47,
4→3=0.24. Stages differentiated (CPA ~0.11). S4 util 60.9%.

### Step 20K (328M tokens) — BEST EVAL

| Metric | Train | Probe (eval) |
|--------|-------|-------------|
| CE4 | 3.01 | 10.08 |
| Δ₂ | +3.90 | +4.09 |
| Δ₃ | +1.98 | -1.58 |
| Δ₄ | +0.04 | -0.10 |
| Total fb | +5.93 | +2.41 |

### Step 30K (492M tokens) — eval WORSENED

| Metric | Train | Probe (eval) |
|--------|-------|-------------|
| CE4 | 2.60 | 11.27 |
| Δ₂ | +5.58 | +3.55 |
| Δ₃ | +1.51 | -1.15 |
| Δ₄ | +0.03 | -0.07 |
| Total fb | +7.13 | +2.32 |

**Key findings across 10K/20K/30K:**
- Eval peaked at 20K, worsened at 30K — overfitting on Dolma
- Train/eval gap: 5.66 → 7.06 → 8.67 (accelerating divergence)
- Structural Δ₂ peaked at 20K (+4.09), declined at 30K (+3.55)
- Semantic Δ₃ NEVER positive on eval: -1.36 → -1.58 → -1.15
- Stage 4: collapsed at 20K (rank 1.7), recovering at 30K (rank 3.2)
- Reversal rate: 22.9% → 30.9% → 35.5% (still climbing)
- First negative gamma at 30K (q_proj wants to reverse topology)
- Math stratum: ONLY one still growing (+0.64 → +2.33 → +2.83)
- Compile gate: 0/4 at all checkpoints (degenerate repetition)
- Stages remain differentiated (CPA ~0.11-0.13) ← architecture works

**Diagnosis:** Architecture validated. Dolma exhausted as training
signal for deep stages. Semantic overfits, ternary oscillates. Math
stratum's continued growth confirms formal data is what the deep
stages need. Next experiment: BIOS flash (math + clojure.core).

## What to do next session

1. **Let v7 run finish** (~midnight). Run full probe on all
   checkpoints. Final analysis: does semantic Δ₃ ever generalize?
   Does topology stabilize? Does compile gate show any sign of life?

2. **Build clojure→lambda converter** (babashka task). One session.
   Start with `clojure.core` — 600 functions → lambda + examples.
   This is the Phase 0 training dataset.

3. **Design grokking experiment:** core clojure × N epochs on v7
   architecture. Watch for double descent in loss curve. Probe for
   circuit formation (does Stage 3 organize by function cluster?
   Does Stage 4 learn to compute?). This tests the staged curriculum
   hypothesis directly.

4. **Staged curriculum plan (if grokking works):**
   ```
   Phase 0: clojure.core × N epochs     (instruction set / grokking)
   Phase 1: curated clojure libs × M    (composition circuits)
   Phase 2: math collection              (calculator broadening)
   Phase 3: dolma                        (NL → formal backbone)
   ```

5. **Open questions from this run:**
   - Is semantic overfitting structural (8 pos too few? wrong arch?)
     or just data-dependent (general text is wrong signal)?
   - Is Stage 4 collapse recoverable with formal data, or is 1
     position genuinely insufficient?
   - Does ternary reversal rate indicate healthy search or instability?

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
