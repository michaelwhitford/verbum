# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-23 | Session: 032

## Where we are

**v6 design evolved. Feedback internalized into VSM. Ready to train.**

Session 032 was a design evolution session. Deep architectural audit
of all feedback/feedforward loops, then systematic internalization of
external mechanisms into the model. No training run yet — all changes
are pre-training design improvements.

### v6 status — ready to train (session 032)

**New in session 032:**

1. **FlipS3 — learned flip policy component:**
   - Reads all 6 register banks (same input as MetaS3)
   - Outputs per-group flip rate factors in [0.3, 2.0]
   - nn.Linear (fp16, tiny) — trained by AdamW through main loss
   - Replaces hand-coded `compute_per_group_flip_targets` inversion
   - Zero-init → sigmoid=0.5 → factor=1.15 (neutral at startup)
   - The model LEARNS which groups need protection vs exploration
   - Stratum spread and Hilberg β still modulate on top (additive)

2. **Int8 flip accumulators — 60% memory savings:**
   - `_flip_accum`: fp32 → int8 with saturating clip at ±127
   - Training memory per ternary weight: 5 bytes → 2 bytes
   - At full scale (35M weights): ~105MB saved
   - NaN guards removed (int8 can't be NaN)

3. **φ-deviation loss term (opt-in via phi_lambda):**
   - `model.__call__` returns `(logits, ce_loss, phi_loss)`
   - Differentiable per-pass compression ratios via `_activation_entropy_differentiable`
   - Phase 1 (now): `PHI_LAMBDA=0.0` — observe only
   - Phase 2 (later): tune to 0.01–0.1 for gradient pressure toward φ

4. **φ-deviation replaces L3 circuit breaker:**
   - Old: 25-step delayed loss-ratio comparison (external Python scalar)
   - New: immediate φ-deviation before/after flips (same step)
   - Information-theoretic signal instead of loss-delta heuristic
   - Emergency brake when L2 destabilization AND φ regression coincide

5. **Stratum-aware + Hilberg β flip routing:**
   - `compute_per_group_flip_targets` accepts `stratum_spread` and `hilberg_beta_dev`
   - High compositional-prose spread → more stride_stack exploration
   - |β - 0.5| > 0.2 → strides need more topological freedom

6. **embed_norm (RMSNorm after embedding):**
   - Breaks tied-embedding amplification loop internally
   - `MAX_GRAD_NORM` relaxed from 1.0 to 2.0 (root cause contained)

7. **Write gate bias init -2.0:**
   - sigmoid(-2) ≈ 0.12 → registers start mostly protected
   - Matches mod_projs zero-init philosophy
   - Smoke test showed gates already diverging by step 150:
     consolidate ≈ 0.93, converge ≈ 0.32 (learning to differentiate)

8. **Per-stride contribution metrics:**
   - `delta_norm`: ||stride_out - stride_in|| per stride
   - `rel_contrib`: delta_norm / ||x|| — relative influence
   - Probe displays contribution table with ★ on dominant stride

### Key architectural insight: mx.eval inside value_and_grad = GPU hang

FlipS3 initially called `mx.eval()` inside the forward pass (via
`factors_dict()`). When `nn.value_and_grad` is tracing the computation
graph, forcing synchronous Metal evaluation deadlocks the GPU. Fix:
store raw tensor, eval after `loss_and_grad_fn` returns.

**Rule: never call `mx.eval()` inside a forward pass that
`nn.value_and_grad` is tracing.**

### Smoke test results (150 steps, random data)

- Loss: 15.97 → 11.32 (learning)
- Flips: 407K across 3 intervals
- FlipS3: all neutral at 1.15 (expected — needs real training to learn)
- Write gates: diverged from 0.12 init to 0.32–0.93 (healthy)
- Int8 accumulators: working correctly, dtype verified after flips
- Full probe pipeline: all 386 metrics captured

### What was NOT changed

- **Flip execution** stays in train.py (discrete weight mutation can't
  be in the computation graph)
- **LR schedule** stays external (cosine, no model signal)
- **Write gate coherence constraint** deferred (observe first)
- **Stability-conditioned flip trigger** deferred (low priority)

### v5 status

Stopped at step 5k. Checkpoints at steps 1k–5k (PyTorch).

## What's next

1. **Train v6** — fresh start with all design improvements:
   ```bash
   uv run python scripts/v6/train.py
   ```
   Watch for:
   - FlipS3 factor differentiation (are groups getting different rates?)
   - Write gate evolution (do they specialize per phase?)
   - Per-stride contribution (which strides dominate?)
   - Gradient norms (smoke test showed huge norms on random data)
   - φ-compression convergence toward 1/φ ≈ 0.618
   - Hilberg β convergence toward 0.5
   - Stratum spread convergence toward 0

2. **If gradient norms explode:** tighten `MAX_GRAD_NORM` back to 1.0.
   The embed_norm handles the root cause but the 5-pass depth can still
   produce large gradients.

3. **Phase 2 φ-loss** — once initial training shows signal:
   - Set `PHI_LAMBDA = 0.01` and observe effect on convergence
   - If compression ratios move toward φ without hurting CE loss, increase

4. **Probe checkpoints** as they drop:
   ```bash
   uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000
   uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_* --phi-only
   ```

## Key files

| Purpose | Path |
|---------|------|
| **v6 (MLX)** | |
| Metal kernels | `src/verbum/v6/kernels.py` |
| TernaryLinear + flip (int8 accum) | `src/verbum/v6/ternary.py` |
| Attention / StrideStack | `src/verbum/v6/attention.py` |
| VSM components (S3, S4, Meta, FlipS3) | `src/verbum/v6/components.py` |
| Full model (embed_norm, φ-loss, FlipS3) | `src/verbum/v6/model.py` |
| Training loop (FlipS3 policy, φ-feedback) | `scripts/v6/train.py` |
| Probe script (stride contrib, FlipS3 display) | `scripts/v6/probe.py` |
| **Research** | |
| Research program | `mementum/knowledge/explore/VERBUM.md` |
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
| v4.1 | 65.5M | PyTorch | Bidirectional VSM | 4.728* |
| v5 | 66.3M | PyTorch | Spiral + ℂ regs + phase gate | TBD |
| v6 | ~63M | **MLX** | Ternary Metal + FlipS3 + φ-loss | TBD |

## VSM feedback map (session 032)

What's internal vs external after this session:

```
INTERNAL (model self-regulates):
  S3 gates        → residual stream modulation (per phase)
  Meta-S3 gates   → per-pass contribution weighting
  S4 register scan → intra-pass feedforward
  Write gates     → register update gating (init bias -2.0)
  FlipS3          → learned per-group flip rate factors [NEW]
  embed_norm      → embedding scale constraint [NEW]
  φ-loss          → gradient pressure toward self-similar compression [NEW, opt-in]

EXTERNAL (train.py, informed by model signals):
  Flip execution  → apply_flips_per_group (discrete mutation)
  φ-feedback      → immediate φ-dev before/after → flip_target_pct [NEW]
  Stratum routing → compositional-prose spread → stride_stack [NEW]
  Hilberg routing → |β-0.5| → stride_stack [NEW]
  LR schedule     → cosine decay (no model signal)
  Grad clipping   → MAX_GRAD_NORM=2.0 (relaxed, embed_norm handles root cause)
```

## Probing pipeline

```bash
# Train v6
uv run python scripts/v6/train.py

# Probe (full or φ-only, single or multi-checkpoint)
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_* --phi-only -v
```
