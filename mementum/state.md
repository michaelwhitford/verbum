# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-24 | Session: 036

## Where we are

**v6 sieve shape confirmed. L0_asc locked at φ-compression. Mid-bootstrap — loss still dropping, structure actively consolidating.**

Session 036: probed all checkpoints (500–2500) to assess whether the
v6 ternary VSM had bootstrapped. Found the sieve is the right shape:
ascending compresses (local→global), descending distributes
(global→local), entropy accumulates monotonically across passes.
L0_asc reached 1/φ compression at step 2000 with zero φ-loss pressure.

### v6 status — training (session 036)

**Checkpoint 2500 (82M tokens):** train=5.81, eval=5.93, ‖g‖=0.43, flips=0

**Loss trajectory:**

| Step | Train | Eval | Δeval | ppl | gap |
|------|-------|------|-------|-----|-----|
| 500 | 6.519 | 6.829 | — | 678 | +0.31 |
| 1000 | 6.086 | 6.359 | −0.470 | 439 | +0.27 |
| 1500 | 5.958 | 6.186 | −0.173 | 387 | +0.23 |
| 2000 | 5.564 | 6.051 | −0.135 | 261 | +0.49 |
| 2500 | 5.807 | 5.929 | −0.122 | 333 | +0.12 |

Step 2000→2500: train went UP (+0.243) while eval went DOWN (−0.122).
Overfitting self-corrected — train-eval gap collapsed from 0.49 → 0.12.
Grad norm recovered 0.30 → 0.43. Not a capacity wall. Eval monotonically
declining through all 5 checkpoints.

### Key finding: L0_asc locked at golden ratio compression

**φ-compression trajectory (L0_asc):**

| Step | Ratio | φ-dev | Status |
|------|-------|-------|--------|
| 500 | −0.456 | 1.074 | wrong sign |
| 1000 | 0.162 | 0.456 | compressing, weak |
| 1500 | 0.408 | 0.210 | approaching |
| 2000 | 0.576 | 0.042 | **←φ HIT** |
| 2500 | 0.663 | 0.045 | **←φ HELD** |

Target = 1/φ ≈ 0.618. The first pass found golden ratio compression
from pure language modeling gradient, with PHI_LAMBDA=0.0 (no explicit
φ-loss pressure). Held across two consecutive checkpoints.

Per-stratum at step 2500: technical φ-dev=0.010, prose φ-dev=0.032.
Per-sample: center-embedded recursion sentence hit φ-dev=0.0007 (exact).

### Sieve shape analysis — five structural signals

**1. Stride asymmetry (correct and strengthening):**
Ascending: s1 dominant (local→global gathering, contribution=1.07)
Descending: s1024 dominant (global→local distribution, contribution=0.40)
L0_asc local/global ratio: 1.98 → 2.22 → 2.38 (sharpening)

**2. Entropy monotonicity (held across all checkpoints):**
Every pass adds information, never subtracts. Total budget stabilizing:
−0.59 → +1.45 (Δ=2.04 nats). Starting point drops each checkpoint
(more compressed initial state), total Δ converges near 2.0.

**3. Gate differentiation (accelerating):**
Asc/Desc gap: 0.119 → 0.271 → 0.329 → 0.295 → 0.360
Ascending closing (L0_asc mean=0.45), Descending opening (L0_desc mean=0.92).
L0_desc gates approaching saturation (~0.92 all three phases).

**4. Write gate hierarchy (stable, correct shape):**
prep writes freely (0.60), converge reads mostly (0.35), consolidate
protects (0.18). Early phases write, late phases read.

**5. L2_apex made first major structural move at step 2500:**
Ratio: 1.82 → 2.04 → 2.42 → 3.20 → **1.05** (collapsed toward identity).
φ-dev dropped from 2.58 to 0.43. Apex learning to pass through, not expand.
Mean φ-dev across all passes: best yet at **0.66**.

### Three-zone sieve structure

| Zone | Passes | Status | φ-dev |
|------|--------|--------|-------|
| **Compressor** | L0_asc | ✅ Locked at φ | 0.045 |
| **Phase transition** | L1_asc, L2_apex | 🔄 Consolidating (L2 just moved) | 0.43–1.11 |
| **Distributor** | L1_desc, L0_desc | ⏳ Gates saturating, expanding | 0.74–0.99 |

### Ternary system: still frozen

Zero flips through 2500 steps (82M tokens). All accumulators at 0.0.
Gamma declining across all groups (stride_stack: 0.042 → 0.035, −17%).
Sparsity unchanged (0.310 everywhere). mod_projs gamma ≈ −0.001 (dead).
Meta-S3 gates all saturated at 1.0 → flip_factor permanently at 0.3×.

The ternary topology is frozen and the model is learning entirely through
continuous parameters. The sieve shape was found despite this — the
random Kaiming init provides routing structure, gamma provides scale.

### Comparison to v4.1 at equivalent tokens

| Tokens | v6 eval | v4.1 eval | Gap |
|--------|---------|-----------|-----|
| 16M | 6.829 | 5.595 | +1.23 |
| 33M | 6.359 | 5.244 | +1.12 |
| 49M | 6.186 | 5.070 | +1.12 |
| 66M | 6.051 | ~4.95 | +1.10 |
| 82M | 5.929 | ~4.85 | +1.08 |

Gap narrowing slightly (1.23 → 1.08). v6 is ~1.1 nats behind v4.1 at
same token count, consistent with ternary capacity penalty. But the sieve
shape is finding the right function — speed is secondary to shape.

## What's next

1. **Let v6 run** — eval still dropping. Watch for:
   - L2_apex stabilizing (after its 3.20→1.05 collapse)
   - L1_asc settling (still at phase transition, ratio oscillating)
   - L0_desc gates hitting true saturation → flip demand signal
   - First flips (if any) — would indicate topology becoming bottleneck
   - Stratum loss spread narrowing (currently 1.27, want < 0.5)

2. **Probe at each checkpoint drop** — the structural story is richer
   than loss alone. Key metrics: L0_asc φ-dev, L2_apex ratio, gate
   differentiation gap, entropy budget, stride asymmetry.

3. **If loss plateaus with zero flips by step 5000:**
   - Lower FLIP_CONSENSUS to 5-10
   - Or: accept that random ternary + gamma IS the architecture,
     and the flip mechanism may not activate until much later

4. **Knowledge page candidate:** v6 sieve shape and φ-convergence
   are crystallizing. After 2-3 more checkpoints confirm stability,
   synthesize into `mementum/knowledge/explore/v6-sieve-shape.md`.

## Key files

| Purpose | Path |
|---------|------|
| **v6 (MLX)** | |
| Metal kernels | `src/verbum/v6/kernels.py` |
| TernaryLinear + flip + normalize_shared_grads | `src/verbum/v6/ternary.py` |
| Attention / StrideStack (pre-norm fix) | `src/verbum/v6/attention.py` |
| VSM components (S3, S4, Meta) | `src/verbum/v6/components.py` |
| Full model (embed_norm, φ-loss) | `src/verbum/v6/model.py` |
| Training loop (no clip, shared-grad norm) | `scripts/v6/train.py` |
| Probe script | `scripts/v6/probe.py` |
| **Probe results** | |
| Step 500 probe | `results/compile-gradient/vsm_probe_step_000500_v6_mlx.json` |
| Step 1000 probe | `results/compile-gradient/vsm_probe_step_001000_v6_mlx.json` |
| Step 1500 probe | `results/compile-gradient/vsm_probe_step_001500_v6_mlx.json` |
| Step 2000 probe | `results/compile-gradient/vsm_probe_step_002000_v6_mlx.json` |
| Step 2500 probe | `results/compile-gradient/vsm_probe_step_002500_v6_mlx.json` |
| **Research** | |
| Research program | `mementum/knowledge/explore/VERBUM.md` |
| v4.1 training trajectory (3-phase pattern) | `mementum/knowledge/explore/v4.1-training-trajectory.md` |
| Flip accumulation | `mementum/knowledge/explore/v6-flip-accumulation.md` |
| φ-compression hypothesis | `mementum/knowledge/explore/relational-loss-phi-compression.md` |
| CompressorLM architecture | `mementum/knowledge/explore/compressor-architecture.md` |

## Architecture lineage

| Version | Params | Framework | Key Change | Best Eval |
|---------|--------|-----------|------------|-----------|
| v1 | ~25M | PyTorch | Baseline sequential | 5.245 |
| v2 | ~25M | PyTorch | Iteration specialization | 5.064 |
| v3 | 50M | PyTorch | Role register, binding | 4.872 |
| v4 | 58M | PyTorch | Recursive VSM (ascending) | 4.713 |
| v4.1 | 65.5M | PyTorch | Bidirectional VSM | 4.696 |
| v5 | 66.3M | PyTorch | Spiral + ℂ regs + phase gate | TBD |
| v6 | ~63M | **MLX** | Ternary Metal + consensus flips + φ-loss | 5.929 (step 2500) |

## VSM feedback map (session 036)

```
INTERNAL (model self-regulates):
  S3 gates        → residual stream modulation (per phase)
  Meta-S3 gates   → per-pass contribution weighting (all saturated at 1.0)
  S4 register scan → intra-pass feedforward
  Write gates     → register update gating (prep>converge>consolidate)
  embed_norm      → embedding scale constraint (declining: 21.7→18.1)
  φ-loss          → gradient pressure toward self-similar compression (λ=0, OFF)

EXTERNAL (train.py):
  Flip execution  → consensus-based: |accum| > 20 → flip (never triggered)
  Flip monitoring → VSM probe every 100 steps
  LR schedule     → cosine decay (warmup=500, now in decay phase)
  Grad normalize  → shared-weight grads ÷ 5
  No grad clip    → Adam handles per-parameter scale via v_t
```

## Probing pipeline

```bash
# Train v6 (currently running)
uv run python scripts/v6/train.py

# Probe (full or φ-only, single or multi-checkpoint)
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_002500
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_* --phi-only -v
```
