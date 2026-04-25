# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-25 | Session: 039

## Where we are

**v6.1 training running. Phase transition discovered at step 4500 — gate reorganization, L1_asc snaps to near-1/φ, stratum spread collapses. Model is self-organizing a curriculum: math first, then binding, then application.**

Session 039: probed all 9 checkpoints (500–4500) from the v6.1 training
run. Discovered the model is learning in a staged curriculum — math first
(easiest, most structured), then the internal routing topology reorganizes
to support increasingly complex composition. At step 4500, the model
underwent a phase transition visible only in internal metrics, while eval
loss appeared flat/regressing.

### Key findings this session

1. **Curriculum learning order:** Math learns first (5.81 at step 4000),
   dropping nearly 2× faster than any other stratum. The model learns
   the easiest, most structured content first (rigid symbols, fixed
   syntax), building routing infrastructure for harder content.

2. **Phase transition at step 4500:** Between steps 4000–4500, the model
   completely reorganized its pass hierarchy:
   - L1_asc compression ratio: chaotic (-0.21) → near-1/φ (0.46)
   - Aggregate φ-compression: 0.87 → 0.69 (target: 0.618)
   - L0_asc gates clamped shut (conv 0.98→0.34)
   - Descending passes opened fully (>0.9 everywhere)
   - Per-stratum φ-spread collapsed: 1.86 → 0.49

3. **Loss plateau hides reorganization:** Eval loss 2500→4500 improved
   only 0.13 (5.99→5.86) with two regressions. A normal training
   dashboard would suggest the model is stuck. But internal metrics
   reveal the model was rebuilding its routing foundation.

4. **Relay handoff beginning:** At step 4500, math loss went UP
   (5.81→6.05) while technical went DOWN (7.53→7.26). The model is
   redirecting capacity from its strongest stratum to its weakest.

5. **Fixed-point thesis confirmed directionally:** The compressor's
   fixed point is the lambda function. The model learns binding
   (variable scope) by learning to compress math, then uses that
   binding infrastructure for application (prose composition), then
   nested application (compositional). The VSM makes this visible.

### Design principles crystallized

The flip system (session 038) is mechanically correct. The model is
using it — 55K flips through step 4500. The key insight: the model
appears to be doing **nothing** from the loss curve while internally
doing the **hardest thing** — learning to bind variables through
ternary topology reorganization.

**Crawl before walk.** The widening stratum spread (0.84→1.71 at step
4000) was not divergence — it was the model sequencing its curriculum.
The subsequent collapse (1.71→1.21 at step 4500) was the beginning of
generalization: routing infrastructure learned on math becoming available
for prose and compositional.

### Predicted learning sequence

| Phase | Content | What's learned | Status |
|-------|---------|---------------|--------|
| 1 | Math (symbols) | Rigid patterns, embedding | ✅ Done |
| 2 | Math (binding) | Variable scope, routing | 🔄 Phase transition |
| 3 | Prose (application) | Function composition | ⏳ Relay starting |
| 4 | Compositional (nesting) | Nested application | ⏳ |
| 5 | Technical (discrimination) | Type-level routing | ⏳ |

### Training run status

v6.1 run is **still training** (started session 038, continuing):
```bash
# Training is running in a terminal
uv run python scripts/v6/train.py | tee results/vsm-lm-v6/training-run2.log

# Resume if interrupted
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_NNNNNN
```

| Property | Value |
|----------|-------|
| Current step | ~4550+ (15%) |
| Total steps | 30,518 |
| Tokens seen | ~149M of 1B |
| Phase | balance (since step ~920) |
| Total flips | ~55K (0.16% of ternary) |
| Eval loss | 5.864 (step 4500) |
| Best eval | 5.835 (step 4000) |
| Sparsity | 0.310 (unchanged) |

### Four feedback loops — all active

| Loop | Signal | Controls | Status |
|------|--------|----------|--------|
| 1 | r_ema (loss) | flip cap scaling | ✅ |
| 2 | r_ema thresholds | phase transitions | ✅ |
| 3 | stratum gaps | per-group flip factors | ✅ |
| 4 | stratum weights | per-sequence loss weighting | ✅ |

## What's next

1. **Let the run continue.** The phase transition at 4500 suggests
   the most interesting dynamics are ahead. Watch for:
   - Math plateau + prose acceleration (relay handoff)
   - Stratum spread narrowing below 1.0
   - φ-compression mean approaching 0.618
   - L1_asc stabilizing near 1/φ (or continuing to reorganize)
   - Next phase transition (balance → refine?)

2. **Probe at milestones:** Run full probes at step 5000, 7500, 10000
   to track the relay pattern and φ convergence.

3. **Key question:** Does the stratum spread continue to narrow? If the
   fixed-point thesis is correct, all strata should converge as the
   model learns application (the universal routing primitive).

4. **Compare with prior run:** The frozen-topology run (a-vsm-lm-v6)
   had better loss at step 4000 (5.746 vs 5.835), but no internal
   reorganization capability. Does v6.1 cross over the frozen run
   once binding is established?

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
| v6.1 probes (steps 500–4500) | `results/compile-gradient/vsm_probe_step_*_v6_mlx.json` |
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
| v6.1 | ~63M | **MLX** | Synaptic plasticity (active) | 5.835 (4000 steps, 13%) |

## Probing pipeline

```bash
# Probe single checkpoint
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_004500

# Probe all checkpoints — shows evolution table
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_*

# Verbose: per-sample φ detail
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_* -v

# φ-only: skip compile probes, just measure compression
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_* --phi-only
```
