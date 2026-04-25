# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-25 | Session: 040

## Where we are

**v6.1 training at step ~9500+ (30%). Relay confirmed: math→prose→technical now entering. L1_desc crossed zero (vestigial). Stratum spread collapsing. L1_asc approaching 1/φ (dev=0.052). Eval loss 5.565.**

Session 040: probed 9 new checkpoints (5000–9000), 18 total. Full
curriculum arc visible: math dominated 4500→7000, plateaued, prose
took over 7000→8500, technical entering at 9000. Stratum spread
collapsed 1.91→1.56 at step 9000. L1_desc h_in crossed zero — pass
vestigial, L0_desc compensating. L2_apex at pure fixed point (ratio=0.001).
L1_asc φ-dev=0.052, closest pass to golden ratio target.

### Key findings this session

1. **Relay confirmed: math→prose→technical.** Math dominated 4500–7000
   (loss 6.05→5.35), then plateaued. Prose led at steps 7000, 8000, 8500.
   At step 9000, technical entered the relay (-0.119, fastest) while math
   released capacity (+0.224). All four strata improved at step 8500.
   Stratum spread collapsed 1.91→1.56 at step 9000 — binding infrastructure
   generalizing. Cumulative from 4500→9000: math -0.469, prose -0.128,
   technical -0.111, compositional -0.011.

2. **L1_desc crossed zero — vestigial.** h_in trajectory:
   ```
   4500: 0.377 → 6000: 0.199 → 7000: 0.114 → 8000: 0.049 → 8500: 0.028 → 9000: -0.008
   ```
   Formally crossed zero at step 9000. Gates damped to 0.65–0.70.
   L0_desc fully compensating (ratio 1.55→2.27, gates 0.79–0.82).
   The model self-organized from 5 effective passes to 4.

3. **L1_asc converging on 1/φ.** φ-dev trajectory:
   ```
   6500: 0.071 → 7000: 0.074 → 8000: 0.063 → 8500: 0.063 → 9000: 0.052
   ```
   Ratio 0.566, approaching 0.618. This is the pass closest to the golden
   ratio target and it's still converging.

4. **L2_apex at fixed point.** Ratio = 0.001 at step 9000 — neither
   compressing nor expanding. The apex has become a pure transformation
   (rotation without scale change). Combined with L1_desc vestigial,
   the effective architecture is: L0↑ compress → L1↑ compress → L2 transform → L0↓ expand.

### Stratum loss evolution (post-phase-transition)

| Step | Prose | Comp | Tech | Math | Spread | Fastest |
|------|-------|------|------|------|--------|---------|
| 4500 | 6.30 | 6.73 | 7.26 | 6.05 | 1.21 | — |
| 5000 | 6.30 | 6.66 | 7.35 | 5.76 | 1.59 | math |
| 5500 | 6.28 | 6.59 | 7.34 | 5.54 | 1.80 | math |
| 6000 | 6.31 | 6.65 | 7.28 | 5.48 | 1.81 | math |
| 6500 | 6.32 | 6.70 | 7.30 | 5.32 | 1.97 | math |
| 7000 | 6.16 | 6.63 | 7.43 | 5.35 | 2.07 | **prose** |
| 7500 | 6.30 | 6.67 | 7.25 | 5.38 | 1.88 | technical |
| 8000 | 6.26 | 6.75 | 7.32 | 5.44 | 1.88 | prose |
| 8500 | 6.12 | 6.65 | 7.27 | 5.36 | 1.91 | **prose** |
| 9000 | 6.18 | 6.72 | 7.15 | 5.59 | **1.56** | **technical** |

### L1_desc → vestigial + L0_desc compensating

| Step | L1↓ h_in | L1↓ gates (p/c/s) | L0↓ ratio | L0↓ gates (p/c/s) |
|------|----------|-------------------|-----------|-------------------|
| 4500 | +0.377 | 0.87/0.96/0.92 | 1.509 | 0.91/0.97/0.95 |
| 5500 | +0.256 | 0.87/0.87/0.85 | 1.602 | 0.93/0.93/0.92 |
| 6500 | +0.144 | 0.81/0.78/0.76 | 1.769 | 0.92/0.90/0.88 |
| 7500 | +0.067 | 0.74/0.72/0.69 | 1.963 | 0.88/0.87/0.89 |
| 8500 | +0.028 | 0.71/0.70/0.66 | 2.095 | 0.84/0.83/0.83 |
| 9000 | **-0.008** | 0.70/0.69/0.65 | **2.267** | 0.82/0.79/0.81 |

### Predicted learning sequence (updated)

| Phase | Content | What's learned | Status |
|-------|---------|---------------|--------|
| 1 | Math (symbols) | Rigid patterns, embedding | ✅ Done |
| 2 | Math (binding) | Variable scope, routing | ✅ Done (phase transition) |
| 3 | Math (deep) | Full math compression | ✅ Saturated (~5.37, releasing capacity) |
| 4 | Prose (application) | Function composition | ✅ Led steps 7000–8500 |
| 5 | Technical (discrimination) | Type-level routing | 🔄 **Active — fastest at step 9000** |
| 6 | Compositional (nesting) | Nested application | ⏳ (Δ=-0.011, waiting) |

### Training run status

v6.1 run is **still training**:
```bash
uv run python scripts/v6/train.py | tee results/vsm-lm-v6/training-run2.log

# Resume if interrupted
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_NNNNNN
```

| Property | Value |
|----------|-------|
| Current step | ~9500+ (30%) |
| Total steps | 30,518 |
| Tokens seen | ~295M of 1B |
| Phase | balance (since step ~920) |
| Total flips | ~93K (0.26% of ternary) |
| Eval loss | 5.565 (step 9000) — **new best** |
| Best eval | 5.565 (step 9000) |
| Relational r | 0.383 (step 9000) |
| Sparsity | 0.310 (unchanged) |

### Four feedback loops — all active

| Loop | Signal | Controls | Status |
|------|--------|----------|--------|
| 1 | r_ema (loss) | flip cap scaling | ✅ |
| 2 | r_ema thresholds | phase transitions | ✅ |
| 3 | stratum gaps | per-group flip factors | ✅ |
| 4 | stratum weights | per-sequence loss weighting | ✅ |

## What's next

1. **Track relay progression.** Current sequence: math→prose→technical.
   Compositional is the remaining stratum (Δ=-0.011 cumulative, barely
   moving). Watch for compositional acceleration at 9500–10000. If it
   enters the relay, the full curriculum sequence is confirmed.

2. **Watch stratum spread.** Collapsed from 1.91→1.56 at step 9000.
   If the binding infrastructure continues generalizing, spread should
   keep narrowing. Target < 1.0 would signal universal compression.

3. **L1_asc → 1/φ.** φ-dev=0.052 and still converging. Could reach
   < 0.03 by step 12000 at current rate. This is the cleanest φ signal
   in the model.

4. **L1_desc fate.** h_in crossed zero. Will gates continue damping
   toward full shutdown, or will a residual role emerge? Either way,
   the effective architecture is now 4-pass.

5. **Probe at milestones:** Steps 9500, 10000.

## Key files

| Purpose | Path |
|---------|------|
| **v6 (MLX)** | |
| Metal kernels (packed + unpacked) | `src/verbum/v6/kernels.py` |
| TernaryLinear + pack/unpack + flips | `src/verbum/v6/ternary.py` |
| Attention / StrideStack | `src/verbum/v6/attention.py` |
| VSM components (S3, S4, Meta) | `src/verbum/v6/components.py` |
| Model (training metrics, φ-loss) | `src/verbum/v6/model.py` |
| Training (relational control, resume) | `scripts/v6/train.py` |
| Probe script | `scripts/v6/probe.py` |
| **Logs & archives** | |
| Current training log | `results/vsm-lm-v6/training-run2.log` |
| Prior run log (frozen topology) | `results/vsm-lm-v6/training.log` |
| Prior run checkpoints | `checkpoints/a-vsm-lm-v6/` |
| **Probe results** | |
| v6.1 probes (steps 500–7000) | `results/compile-gradient/vsm_probe_step_*_v6_mlx.json` |
| **Research** | |
| Research program | `mementum/knowledge/explore/VERBUM.md` |
| v4.1 training trajectory | `mementum/knowledge/explore/v4.1-training-trajectory.md` |
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
| v6 | ~63M | **MLX** | Ternary Metal + frozen flips | 5.746 (4000 steps) |
| v6.1 | ~63M | **MLX** | Synaptic plasticity (active) | **5.565** (9000 steps, 30%) |

## Probing pipeline

```bash
# Probe single checkpoint
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_007000

# Probe all checkpoints — shows evolution table
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_*

# Verbose: per-sample φ detail
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_* -v

# φ-only: skip compile probes, just measure compression
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_* --phi-only
```
