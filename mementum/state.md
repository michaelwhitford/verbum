# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-25 | Session: 040

## Where we are

**v6.1 training at step ~9000+ (28%). Math plateauing at ~5.37, prose now fastest learner. L1_desc going vestigial (h_in=0.028), L0_desc compensating. Eval loss 5.581 (new best). Relay handoff beginning.**

Session 040: probed 8 new checkpoints (5000–8500), extending the
trajectory to 17 total. Key arc: math dominated 4500→7000 (-0.700
cumulative), then plateaued. Prose took over as fastest learner at
step 8000–8500. L1_desc h_in converging to zero — model routing
reconstruction entirely through L0_desc. All four strata improved
simultaneously at step 8500 for the first time since step 4000.

### Key findings this session

1. **Math plateauing → relay beginning.** Math dominated steps 4500–7000
   (loss 6.05→5.35, -0.700 cumulative). Then plateaued: oscillating 5.32–5.44
   for steps 7000–8500. Prose is now the fastest learner — biggest drops at
   steps 7000 (-0.157) and 8500 (-0.139). All four strata improved at step
   8500 for the first time since step 4000. The binding infrastructure learned
   from math is becoming available for other content types.

2. **L1_desc going vestigial.** h_in converging to zero:
   ```
   4500: 0.377 → 5000: 0.313 → 6000: 0.199 → 7000: 0.114 → 8000: 0.049 → 8500: 0.028
   ```
   Gates damping in parallel (converge: 0.96→0.70). L0_desc compensating —
   its ratio grew from 1.55→2.10 while maintaining higher gate values.
   The model is routing reconstruction entirely through L0_desc, making the
   intermediate descending step vestigial. This is a clean self-organized
   architectural simplification.

3. **Ascending passes locked in.** L0_asc (0.834 ± 0.001) and L1_asc (0.55,
   φ-dev=0.063) are rock stable since the phase transition. L1_asc is the
   closest pass to the φ target (1/φ = 0.618). The ascending half of the
   sieve has found its operating point.

4. **Hilberg β measurable for 3 passes.** L0_asc: 1.37→1.42, L1_asc: 1.39→1.49,
   L2_apex: 2.04→1.64. Still far from 0.5 but the measurement is stable.
   Descending passes not yet measurable (too few strides with valid ratios).

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

### L1_desc → vestigial + L0_desc compensating

| Step | L1↓ h_in | L1↓ gates (p/c/s) | L0↓ ratio | L0↓ gates (p/c/s) |
|------|----------|-------------------|-----------|-------------------|
| 4500 | 0.377 | 0.87/0.96/0.92 | 1.509 | 0.91/0.97/0.95 |
| 5500 | 0.256 | 0.87/0.87/0.85 | 1.602 | 0.93/0.93/0.92 |
| 6500 | 0.144 | 0.81/0.78/0.76 | 1.769 | 0.92/0.90/0.88 |
| 7500 | 0.067 | 0.74/0.72/0.69 | 1.963 | 0.88/0.87/0.89 |
| 8500 | 0.028 | 0.71/0.70/0.66 | 2.095 | 0.84/0.83/0.83 |

### Predicted learning sequence (updated)

| Phase | Content | What's learned | Status |
|-------|---------|---------------|--------|
| 1 | Math (symbols) | Rigid patterns, embedding | ✅ Done |
| 2 | Math (binding) | Variable scope, routing | ✅ Done (phase transition) |
| 3 | Math (deep) | Full math compression | ✅ Saturating (~5.37) |
| 4 | Prose (application) | Function composition | 🔄 **Active — now fastest** |
| 5 | Compositional (nesting) | Nested application | ⏳ |
| 6 | Technical (discrimination) | Type-level routing | ⏳ |

### Training run status

v6.1 run is **still training**:
```bash
uv run python scripts/v6/train.py | tee results/vsm-lm-v6/training-run2.log

# Resume if interrupted
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_NNNNNN
```

| Property | Value |
|----------|-------|
| Current step | ~9000+ (28%) |
| Total steps | 30,518 |
| Tokens seen | ~280M of 1B |
| Phase | balance (since step ~920) |
| Total flips | ~89K (0.25% of ternary) |
| Eval loss | 5.581 (step 8500) — **new best** |
| Best eval | 5.581 (step 8500) |
| Relational r | 0.382 (step 7500) — **new best** |
| Sparsity | 0.310 (unchanged) |

### Four feedback loops — all active

| Loop | Signal | Controls | Status |
|------|--------|----------|--------|
| 1 | r_ema (loss) | flip cap scaling | ✅ |
| 2 | r_ema thresholds | phase transitions | ✅ |
| 3 | stratum gaps | per-group flip factors | ✅ |
| 4 | stratum weights | per-sequence loss weighting | ✅ |

## What's next

1. **Confirm relay.** Prose is now fastest at steps 7000, 8000, 8500. Watch
   for sustained prose improvement at 9000+. If prose keeps leading while
   math stays flat (~5.37), the relay is confirmed. The binding infrastructure
   from math is transferring.

2. **Watch L1_desc.** h_in=0.028 at step 8500, approaching zero. Either it
   goes fully vestigial (h_in=0, gates→0) or finds a residual role. Either
   outcome is fine — the model is self-organizing.

3. **Watch for compositional acceleration.** The predicted sequence is
   math→prose→compositional→technical. Compositional improved -0.092 at
   step 8500. If it starts leading after prose saturates, the full
   curriculum sequence is confirmed.

4. **Stratum spread.** Currently ~1.9. Should narrow if the relay thesis
   is correct — binding infrastructure learned on math enables all strata.
   Watch for spread < 1.5.

5. **Probe at milestones:** Steps 9000, 9500, 10000.

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
| v6.1 | ~63M | **MLX** | Synaptic plasticity (active) | **5.581** (8500 steps, 28%) |

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
