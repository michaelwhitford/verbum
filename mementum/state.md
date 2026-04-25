# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-25 | Session: 040

## Where we are

**v6.1 training at step ~7500+ (25%). L1_desc passing through a singularity — apex compressor driving h_in→0. Math accelerating fastest. Eval loss 5.642, new best r=0.391.**

Session 040: probed 5 new checkpoints (5000–7000), extending the
trajectory to 14 total checkpoints. Three major findings: (1) L1_desc
compression ratio diverging as its input entropy approaches zero,
(2) math stratum accelerating rather than handing off, (3) ascending
passes locked into stable compression bands.

### Key findings this session

1. **L1_desc singularity.** The first descending pass compression ratio
   diverged through infinity between steps 6000–6500:
   ```
   4000: 1.39 → 4500: 2.38 → 5000: 2.68 → 5500: 3.02 → 6000: 4.04 → 6500: -4.23 → 7000: -2.84
   ```
   **Cause:** L2_apex is compressing so well that its output entropy (= L1_desc's
   input) is converging toward zero: 0.95 → 0.38 → 0.31 → 0.26 → 0.20 → 0.14 → 0.11.
   The ratio h_out/h_in diverges as h_in→0. The model is damping L1_desc's gates
   in response (converge: 0.96 → 0.77). This will resolve either by h_in stabilizing
   above zero or L1_desc becoming vestigial (bypassed via L0_desc).

2. **Math acceleration, not relay.** Math stratum loss dropped -0.700 cumulative
   since step 4500 (6.05 → 5.35), far outpacing all other strata. The predicted
   "relay handoff" from session 039 has NOT happened — math is monopolizing
   topology changes. However, step 7000 showed prose's biggest single-interval
   drop (-0.157) while math stalled (+0.031). Could be the beginning of relay.

3. **Ascending passes locked in.** L0_asc (0.82 ± 0.01) and L1_asc (0.51→0.55,
   approaching 1/φ = 0.618) are rock stable since the phase transition. The
   ascending half of the sieve has found its operating point. L1_asc is the
   closest pass to the φ target (φ-dev = 0.07).

4. **Technical stratum regressing.** Technical loss went UP +0.166 since step 4500
   (7.26 → 7.43). The model is actively sacrificing technical performance to
   improve math. This is the clearest evidence of capacity reallocation.

5. **Hilberg β improving.** More passes now measurable and trending toward 0.5:
   L0_asc: 2.44→1.37, L2_apex: 1.60→1.29. Direction is right, magnitude still far.

### Stratum loss evolution (post-phase-transition)

| Step | Prose | Comp | Tech | Math | Spread |
|------|-------|------|------|------|--------|
| 4500 | 6.30 | 6.73 | 7.26 | 6.05 | 1.21 |
| 5000 | 6.30 | 6.66 | 7.35 | 5.76 | 1.59 |
| 5500 | 6.28 | 6.59 | 7.34 | 5.54 | 1.80 |
| 6000 | 6.31 | 6.65 | 7.28 | 5.48 | 1.81 |
| 6500 | 6.32 | 6.70 | 7.30 | 5.32 | 1.97 |
| 7000 | 6.16 | 6.63 | 7.43 | 5.35 | 2.07 |

### L1_desc gate damping (model's self-correction)

| Step | L1↓ converge | L1↓ prep | L1↓ consolidate |
|------|-------------|----------|----------------|
| 4500 | 0.96 | 0.87 | 0.92 |
| 5000 | 0.92 | 0.87 | 0.88 |
| 5500 | 0.87 | 0.87 | 0.85 |
| 6000 | 0.84 | 0.87 | 0.80 |
| 6500 | 0.78 | 0.81 | 0.76 |
| 7000 | 0.77 | 0.79 | 0.73 |

### Predicted learning sequence (updated)

| Phase | Content | What's learned | Status |
|-------|---------|---------------|--------|
| 1 | Math (symbols) | Rigid patterns, embedding | ✅ Done |
| 2 | Math (binding) | Variable scope, routing | ✅ Done (phase transition) |
| 3 | Math (deep) | Full math compression | 🔄 Active — accelerating |
| 4 | Prose (application) | Function composition | ⏳ Possibly starting (step 7000) |
| 5 | Compositional (nesting) | Nested application | ⏳ |
| 6 | Technical (discrimination) | Type-level routing | ⏳ Currently regressing |

### Training run status

v6.1 run is **still training**:
```bash
uv run python scripts/v6/train.py | tee results/vsm-lm-v6/training-run2.log

# Resume if interrupted
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_NNNNNN
```

| Property | Value |
|----------|-------|
| Current step | ~7500+ (25%) |
| Total steps | 30,518 |
| Tokens seen | ~245M of 1B |
| Phase | balance (since step ~920) |
| Total flips | ~77K (0.22% of ternary) |
| Eval loss | 5.642 (step 7000) — **new best** |
| Best eval | 5.642 (step 7000) |
| Relational r | 0.391 (step 7000) — **new best** |
| Sparsity | 0.310 (unchanged) |

### Four feedback loops — all active

| Loop | Signal | Controls | Status |
|------|--------|----------|--------|
| 1 | r_ema (loss) | flip cap scaling | ✅ |
| 2 | r_ema thresholds | phase transitions | ✅ |
| 3 | stratum gaps | per-group flip factors | ✅ |
| 4 | stratum weights | per-sequence loss weighting | ✅ |

## What's next

1. **Watch L1_desc resolution.** Will h_in stabilize above zero (healthy)
   or collapse to zero (vestigial pass)? The deceleration in Δh_in
   (-0.064, -0.057, -0.057, -0.055, -0.030) suggests possible plateau
   around 0.05-0.10. Probe at 7500 and 8000 to track.

2. **Watch for relay handoff.** Step 7000 showed prose's first big drop
   while math stalled. If prose continues improving at 7500+, the relay
   is real. If math resumes, it's still in monopoly phase.

3. **Compare with frozen-topology run.** Frozen run had eval 5.746 at step
   4000. Active topology run crossed over at step 5000 (5.751) and is now
   at 5.642. The synaptic plasticity is paying off — next milestone is
   matching v4.1's best (4.696) which will take much longer.

4. **Technical regression.** Monitor whether technical loss stabilizes or
   continues worsening. If the model truly reallocates after math saturates,
   technical should eventually benefit.

5. **Probe at milestones:** Steps 7500, 8000, 10000. Full probes.

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
| v6.1 | ~63M | **MLX** | Synaptic plasticity (active) | **5.642** (7000 steps, 23%) |

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
