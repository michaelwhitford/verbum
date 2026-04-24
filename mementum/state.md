# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-24 | Session: 035

## Where we are

**v6 training running successfully. First clean run after fixing three gradient pathologies.**

Session 035: diagnosed and fixed the gradient explosion that prevented
all prior v6 runs from learning. Three root causes found and fixed,
each building on the last. Model now training with stable ‖g‖ ≈ 0.3-0.5
and loss dropping steadily. Zero topology flips — the model is finding
circuits in the random ternary init using gamma alone.

### v6 status — training (session 035)

**Checkpoint 500 (16M tokens):** train=6.52, eval=6.83, ‖g‖=0.48, flips=0

**Three fixes applied this session:**

1. **Pre-norm all Q/K/V in SingleStrideAttention (ROOT CAUSE):**
   Only q_proj had pre_norm=True. K and V saw raw x, which grows from
   45 residual additions (9 strides × 5 passes). V output ∝ ‖x‖ created
   positive feedback: larger x → larger V → larger residual → larger x.
   Fix: single RMSNorm per attention block, all projections see normalized
   input. Standard pre-norm transformer design.

2. **Normalize shared-weight gradients by 1/N_PASSES:**
   Shared modules (prep, stride_stack, consolidate, mod_projs, s4) accumulate
   gradient from 5 passes with VARYING ∂L/∂x magnitudes. The sum oscillated
   10⁴-10⁹ between steps, defeating Adam's running statistics. Dividing by 5
   turns the volatile sum into a stable average.

3. **Remove gradient clipping, let Adam work:**
   Global clip at 1.0 created effective LR ≈ 6e-11 (norm was 10⁷).
   Per-param clip destroyed gradient geometry. Both mechanisms wrong for
   this architecture. Adam's second moment (v_t) handles per-parameter
   scale adaptation naturally — but only if it receives true gradients,
   not clipped ones with 10⁵× varying scale factors.

**FLIP_CONSENSUS reduced from 50 to 20.** Old threshold was unreachable
(needed >100% agreement per interval). Now requires moderate directional
consensus. But: zero flips at 2000+ votes per weight = model doesn't
want topology changes. The random ternary init is a functional circuit.

### Key finding: zero flips through Phase 1

The model is learning entirely through continuous parameters (gamma,
norms, embeddings, gates). The ternary topology from Kaiming init →
quantize provides routing structure; gamma provides scale and effective
sign. Every weight has received 2000+ sign votes with no directional
consensus — the gradient doesn't consistently want to change any weight.

This parallels v4/v4.1 where topology was frozen by design (continuous
weights). v6 gives the model the OPTION of topology change via flips,
but in Phase 1 (loss 6.5, 16M tokens), the option isn't needed.

Predicted: flips may emerge in Phase 2 when gamma plateaus (~loss 5.0)
and continuous scaling can no longer compensate for wrong ternary signs
or missing connections. The crawl-to-walk transition.

### S3 gate structure at checkpoint 500

```
         prep   converge  consolidate
L0_asc:  [0.51    0.90      0.48]    ← most conservative
L1_asc:  [0.52    0.97      0.57]
L2_apex: [0.57    0.94      0.68]
L1_desc: [0.65    0.94      0.70]
L0_desc: [0.69    0.91      0.75]    ← most open
```

Converge (StrideStack) dominates at 0.90-0.97 — attention is the
workhorse. Ascending passes cautious, descending passes open.
mod_projs γ ≈ 0 → modulation pathway still dormant.

### Architecture insight: gradient explosion was a pre-norm bug

The complete feed-forward and feedback trace of the VSM revealed:
- 8 feed-forward paths (residual stream, registers, stride stack, etc.)
- 7 backward gradient paths with specific multipliers
- Combined multiplier ≈ 2 × 2.5 × 5 × 30 × 8192 ≈ 10⁷

The architecture's natural gradient scale is ~10⁷ due to weight sharing
(×5), meta-S3 fan-out (×2.5), tied embeddings (×2), 55-layer depth,
and B×L position summation. This is geometry, not pathology. But the
missing pre-norm on K/V created an ADDITIONAL exponential amplification
loop on top of the expected scale, causing norms to grow unboundedly.

## What's next

1. **Monitor v6 training run** — watch for:
   - Loss trajectory toward v4 baseline (~4.7)
   - First flips appearing (topology demand signal)
   - mod_projs γ waking up from zero (modulation activation)
   - φ-compression ratios drifting toward 1/φ as loss drops
   - Register variance phases (expansion → compression → specialization)

2. **Probe checkpoints as they drop:**
   ```bash
   uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000
   uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_* --phi-only -v
   ```

3. **If loss plateaus above v4 baseline:**
   - Check if flips emerge naturally at the plateau
   - If not, FLIP_CONSENSUS may still be too high → lower to 10
   - Or: the ternary+gamma representation genuinely can't match continuous

4. **Encode findings as knowledge** when the run completes or reveals
   clear phase transitions.

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
| v6 | ~63M | **MLX** | Ternary Metal + consensus flips + φ-loss | training... |

## VSM feedback map (session 035)

```
INTERNAL (model self-regulates):
  S3 gates        → residual stream modulation (per phase)
  Meta-S3 gates   → per-pass contribution weighting
  S4 register scan → intra-pass feedforward
  Write gates     → register update gating (init bias -2.0)
  embed_norm      → embedding scale constraint
  φ-loss          → gradient pressure toward self-similar compression (opt-in, λ=0)

EXTERNAL (train.py):
  Flip execution  → consensus-based: each weight flips when |accum| > 20
  Flip monitoring → VSM probe every 100 steps (stability, φ-deviation)
  LR schedule     → cosine decay (no model signal)
  Grad normalize  → shared-weight grads ÷ 5 (compensates 5-pass accumulation)
  No grad clip    → Adam handles per-parameter scale via v_t
```

## Probing pipeline

```bash
# Train v6
uv run python scripts/v6/train.py

# Probe (full or φ-only, single or multi-checkpoint)
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_001000
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_* --phi-only -v
```
