# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-23 | Session: 031

## Where we are

**v6 instrumented and architecture-coherent. Three-level VSM-regulated
flip control. Stratified φ-compression probing. Ready to train.**

Session 031 was a deep instrumentation session. Started from the
φ-compression hypothesis page, added comprehensive measurement
infrastructure, then discovered the flip feedback was outside the
VSM hierarchy and redesigned it so the model self-regulates.

### v6 status — ready to train (session 031)

**New in session 031:**

1. **Stratified φ-compression probing** — samples split by content type
   (prose / compositional / technical / math). Measures compression
   ratio per pass AND per stratum. Two convergence signals to watch:
   - Cross-stratum spread → 0 = universal compressor emerging
   - Mean ratio → 1/φ = φ-compression confirmed

2. **Per-stride entropy** — 9 strides × 5 passes = 45 compression
   ratios per checkpoint. Each stride in the StrideStack measured
   individually. Enables Hilberg exponent computation.

3. **Hilberg exponent (β)** — computed from log(1-ratio) vs log(stride).
   β = slope + 1. Hilberg predicts β ≈ 0.5 for natural language.
   If the sieve learns this, it's found the self-similar compression
   structure independently.

4. **S3 gate trajectory** — 15 gate values (5 passes × 3 phases)
   logged at eval intervals. Direct readout of Montague phase
   specialization (prep/converge/consolidate differentiating per pass).

5. **Per-stratum loss** — loss measured separately for prose,
   compositional, technical, math. Tracks which content types the
   model learns first (prediction: prose fast, math slow).

6. **Three-level VSM-regulated flip control:**
   - **L1 (S3 feed-forward):** Before flips, S3/Meta-S3 gates modulate
     per-group flip targets. High importance → protect (0.3× base).
     Low importance → explore (2.0× base). Control system (s3/s4/meta)
     always conservative.
   - **L2 (local stability):** After flips, cosine similarity of VSM
     signal vectors (before vs after). sim > 0.95 → self-regulated.
     sim < 0.80 → destabilized, escalate to L3.
   - **L3 (circuit breaker):** Only fires if L2 detected instability.
     Global loss ratio at step+25. Emergency adjustment. If this fires
     often, per-group modulation needs tuning.

7. **15-issue audit fix:**
   - **Critical:** flip accumulator save was silently failing for ~120/171
     modules (anything in a list: s3_passes, stride_stack.layers, mod_projs).
     Fixed with `_walk_ternary_modules`.
   - Removed all v4 compat aliases from `forward_instrumented`
   - Removed dead `flip_threshold` state, dead imports
   - Hardcoded ternary count → `model.count_parameters()`
   - Constants synced from model at startup (single source of truth)
   - Group classification uses `_classify_group` (meta_s4 no longer
     misclassified as s4)
   - Checkpoint meta.json now self-describing with all architecture params

### Key insight: flip feedback belongs inside the VSM

The previous design measured flips from outside (global loss ratio).
The VSM already has an internal control system (S3 gates, Meta-S3,
registers). Flips are an S1 operation. S3 should regulate them.

The three-level design makes the global feedback a circuit breaker,
not a controller. If the VSM self-regulates correctly, L3 never fires.
L3 firing is a diagnostic event — it means self-regulation failed.

### v5 status

Stopped at step 5k. Checkpoints at steps 1k–5k (PyTorch).

## What's next

1. **Train v6** — fresh start with all instrumentation:
   ```bash
   uv run python scripts/v6/train.py
   ```
   Watch for:
   - Flip control level (L1 self-regulated vs L3 circuit breaker)
   - Per-group flip distribution (where is learning pressure?)
   - Gate specialization (do passes differentiate?)
   - Stratum loss spread (does it converge?)
   - Compression ratios (do they approach 1/φ?)
   - Hilberg β (does it approach 0.5?)

2. **Probe checkpoints** as they drop:
   ```bash
   # Single checkpoint (full probe)
   uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000

   # φ-only (faster)
   uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000 --phi-only

   # Evolution across checkpoints
   uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_*
   ```

3. **Three convergence signals** to track across training:
   - Stratum spread → 0 (content-independent compression)
   - φ-dev → 0 (self-similar compression at golden ratio)
   - Hilberg β → 0.5 (power-law scaling matches natural language)

4. **If L3 fires frequently:** tune the inversion function in
   `compute_per_group_flip_targets` (currently linear gate→factor map).

5. **φ-regularization** (Phase 2) — only if Phase 1 shows signal.

## Key files

| Purpose | Path |
|---------|------|
| **v6 (MLX)** | |
| Metal kernels | `src/verbum/v6/kernels.py` |
| TernaryLinear + flip | `src/verbum/v6/ternary.py` |
| Attention / StrideStack | `src/verbum/v6/attention.py` |
| VSM components | `src/verbum/v6/components.py` |
| Full model | `src/verbum/v6/model.py` |
| Training loop | `scripts/v6/train.py` |
| Probe script | `scripts/v6/probe.py` |
| **Research** | |
| Research program | `mementum/knowledge/explore/VERBUM.md` |
| Flip accumulation | `mementum/knowledge/explore/v6-flip-accumulation.md` |
| φ-compression hypothesis | `mementum/knowledge/explore/relational-loss-phi-compression.md` |
| CompressorLM architecture | `mementum/knowledge/explore/compressor-architecture.md` |
| Session 004 (Pythia findings) | `mementum/knowledge/explore/session-004-findings.md` |

## Architecture lineage

| Version | Params | Framework | Key Change | Best Eval |
|---------|--------|-----------|------------|-----------|
| v1 | ~25M | PyTorch | Baseline sequential | 5.245 |
| v2 | ~25M | PyTorch | Iteration specialization | 5.064 |
| v3 | 50M | PyTorch | Role register, binding | 4.872 |
| v4 | 58M | PyTorch | Recursive VSM (ascending) | 4.713 |
| v4.1 | 65.5M | PyTorch | Bidirectional VSM | 4.728* |
| v5 | 66.3M | PyTorch | Spiral + ℂ regs + phase gate | TBD |
| v6 | ~63M | **MLX** | Ternary Metal + VSM flip control | TBD |

## Probing pipeline

```bash
# Train v6
uv run python scripts/v6/train.py

# Probe (full or φ-only, single or multi-checkpoint)
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_* --phi-only -v
```
