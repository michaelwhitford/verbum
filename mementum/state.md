# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-25 | Session: 038

## Where we are

**Flip system rebuilt from scratch. Five bugs/design flaws fixed. Ready to train v6.1 with proper synaptic plasticity.**

Session 038: discovered the v6.1 training run had 6M+ flips by step 50
despite a 0.1% cap. Traced through five interlocking issues in the flip
accumulation system and fixed them all. Also closed feedback Loop 3
(stratum-based per-group flip factors were computed but never wired to
actual flips).

### Changes this session (9 commits)

1. **Cap bypass fix** — `apply_flips` binary search over `[threshold, 127]`
   can't exceed int8 max. When weights saturate at 127, all flip uncapped.
   Fix: random subsample with `keep_prob = max_flips / n_qualifying`.

2. **Rate reduction 100×** — `FLIP_MAX_PCT` 0.001 → 0.00001 (0.1% → 0.001%).
   Explore: ~1400 flips/interval = ~8/module. Balance: ~500. Refine: ~90.
   Full 30K run explores ~11% of topology → 1.7% with interval=25.

3. **Interval 4 → 25** — 25 steps = 3.5 Adam β1 half-lives between checks.
   Gradient signal now reflects consequences of prior flips, not stale momentum.
   100 votes per interval (25 × 4 micro-batches). Clean consensus signal.

4. **Accumulator reset** — previously only flipped weights reset, creating an
   infinite backlog. Millions of weights saturate at ±127 and block reversals.
   Fix: reset ALL accumulators after each flip check. Each interval is a fresh
   question: "which weights want to flip NOW?"

5. **Consensus 40 → 50** — 75% agreement required (50 net votes out of 100).
   Higher bar → fewer flips, stronger evidence before committing.

6. **Flip warmup** — no flips before step 500 (LR warmup). Adam needs stable
   moments before topology changes are meaningful. Also removed consensus
   scaling — 75% is the bar in all phases. r modulates only the cap.

7. **Loop 3 closed** — `apply_flips_per_group` now uses `cached_group_factors`
   from stratum gap analysis. stride_stack gets more flips when compositional
   lags prose, prep gets more when abstraction lags.

8. **generate() unpack fix** — model returns 4 values, generate expected 3.

9. **Gate accumulation during warmup** — `accumulate_flips` was running every
   micro-batch ungated. By step 500, 2000 votes saturated at ±127. First flip
   check would see warmup noise. Now gated by `step >= WARMUP_STEPS`.

### Design principles crystallized

The flip system now embodies **synaptic plasticity**: flip a few routes,
let continuous params adapt around them for many steps, then flip a few
more based on what the gradient says *now*.

| Property | Value | Why |
|----------|-------|-----|
| First flip | Step 500 | After LR warmup, Adam moments initialized |
| Interval | 25 steps | 3.5 Adam β1 half-lives between checks |
| Votes | 100/interval | 25 steps × 4 micro-batches |
| Consensus | 75% fixed | 50 net votes, all phases |
| Cap | 0.001% base | r × phase scales only the cap |
| Accum reset | Every check | No backlog, flips reversible |

### Four feedback loops — all wired

| Loop | Signal | Controls | Status |
|------|--------|----------|--------|
| 1 | r_ema (loss) | flip cap scaling | ✅ |
| 2 | r_ema thresholds | phase transitions (explore→balance→refine) | ✅ |
| 3 | stratum gaps | per-group flip factors | ✅ now closed |
| 4 | stratum weights | per-sequence loss weighting | ✅ |

### Prior run analysis (archived as a-vsm-lm-v6)

4000 steps, frozen topology (zero flips due to bug):
- Eval: 6.829 → 5.746 (7 consecutive drops, decelerating)
- Stratum rotation: math/prose/technical take turns, compositional stuck
- φ-compression: L0_asc found 1/φ, drifted; L2_apex oscillating wildly
- Sieve shape correct despite frozen topology
- Stratum spread widening: 0.57 → 1.51 (compositional can't route through
  frozen ternary — strongest demand signal for flips)

## What's next

1. **Start training v6.1:**
   ```bash
   uv run python scripts/v6/train.py | tee results/vsm-lm-v6/training-run3.log
   ```
2. **Watch for:** first flips at step 500+, which groups flip first,
   whether compositional loss starts improving with active topology,
   stratum spread narrowing, phase transition timing
3. **Compare with prior run** — does active topology beat frozen?
4. **Key question:** does the stratum rotation pattern change once
   flips are active? Compositional has never led improvement.

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
| Prior run log (frozen topology) | `results/vsm-lm-v6/training.log` |
| Prior run checkpoints | `checkpoints/a-vsm-lm-v6/` |
| Prior run probes | `results/compile-gradient/vsm_probe_step_*_v6_mlx.json` |
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
| v6.1 | ~63M | **MLX** | Synaptic plasticity (rebuilt) | ready to train |

## Probing pipeline

```bash
# Train v6.1 (from scratch, rebuilt flip system)
uv run python scripts/v6/train.py | tee results/vsm-lm-v6/training-run3.log

# Resume from checkpoint
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_NNNNNN

# Probe
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_NNNNNN
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_* --phi-only -v
```
