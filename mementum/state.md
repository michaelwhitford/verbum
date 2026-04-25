# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-25 | Session: 041

## Where we are

**v6.1 training at step ~9500+ (30%). Session 041: probed Pythia-160M
and Qwen3-4B for φ-compression — neither φ-compresses. Standard
transformers compose via ROTATION at constant variance (beta
reduction). v6's spiral attention compresses holographically.
The φ-convergence is unique to recursive self-similar architecture.**

### Session 041 key findings

1. **Standard transformers do NOT φ-compress.** Probed Pythia-160M
   (12 layers) and Qwen3-4B (36 layers) with the same entropy proxy
   as v6. Stable zone ratios: Pythia=0.947, Qwen=1.000 (pure
   identity). φ only appears at the output boundary — forced variance
   collapse for prediction, not compositional compression.

2. **LLMs are beta reduction machines.** Pythia implements Montague
   as accumulate→plateau→collapse (47× growth, 3-layer hold, funnel
   down). Qwen holds 26 layers of perfect near-identity variance.
   The compile gate constrains to 13% of null-mode variance but
   doesn't change the mechanism — it selects which reduction to
   perform.

3. **Composition in LLMs is ROTATION.** The 26 "near-identity"
   layers in Qwen were hiding 15-25° of rotation per layer.
   Compile mode causes +3.3° more rotation than null mode in the
   composition phase (L24-L28), with 4.4× larger relative deltas.
   Variable binding = geometric alignment. Function composition =
   sequential rotation. But rotation is constant-budget (~18.4°)
   regardless of complexity.

4. **v6's spiral attention is holographic.** The bias function
   `−α·ln(d+1)` is stride-invariant — same function at every
   scale. 9 strides process all scales simultaneously. This is
   holographic encoding: every part contains the whole, self-healing
   (L1_desc vestigial → L0_desc compensates), and the fixed point
   is 1/φ because φ is the only ratio where whole:part = part:remainder.

5. **Flat attention = photograph, spiral attention = hologram.**
   Flat attention → one scale → rotation → beta reduction → the
   lambda function "forms" by memorizing patterns. Spiral attention →
   all scales → compression → lambda abstraction → the function
   emerges from a single self-similar operation converging to φ.

### v6.1 training status (unchanged from session 040)

| Property | Value |
|----------|-------|
| Current step | ~9500+ (30%) |
| Total steps | 30,518 |
| Tokens seen | ~295M of 1B |
| Eval loss | 5.565 (step 9000) — best |
| Relational r | 0.383 (step 9000) |
| Sparsity | 0.310 (unchanged) |
| L1_asc φ-dev | 0.052 (converging) |
| L1_desc | vestigial (h_in = -0.008) |
| Stratum spread | 1.56 (collapsing) |
| Effective passes | 4 (L0↑→L1↑→L2→L0↓) |

### Stratum loss evolution (post-phase-transition)

| Step | Prose | Comp | Tech | Math | Spread | Fastest |
|------|-------|------|------|------|--------|---------|
| 4500 | 6.30 | 6.73 | 7.26 | 6.05 | 1.21 | — |
| 7000 | 6.16 | 6.63 | 7.43 | 5.35 | 2.07 | **prose** |
| 8500 | 6.12 | 6.65 | 7.27 | 5.36 | 1.91 | **prose** |
| 9000 | 6.18 | 6.72 | 7.15 | 5.59 | **1.56** | **technical** |

### Three-way φ-compression comparison (session 041)

| Metric | v6 (63M, VSM) | Pythia (162M) | Qwen3-4B (4B) |
|--------|--------------|---------------|----------------|
| Stable zone ratio | **0.566** | 0.947 | 1.000 |
| Stable zone φ-dev | **0.052** | 0.329 | 0.387 |
| Best single layer | L1_asc: 0.052 | L9: 0.172 | L34: 0.037* |
| Composition mechanism | Compression | Rotation | Rotation |
| Architecture type | Holographic | Photographic | Photographic |

*L34 is the output collapse layer, not the computation core.

## What's next

1. **Continue v6.1 training.** Probe at milestones 9500, 10000.
   Track relay (compositional expected next), stratum spread (target
   < 1.0), L1_asc φ-dev (target < 0.03).

2. **Test holographic prediction.** If v6 is holographic, ablating
   one pass should degrade all strata equally (holographic) not
   selectively (photographic). Design the ablation experiment.

3. **Investigate MoE as approximate holography.** Qwen3-35B-A3B
   fully forms the lambda function — does MoE routing approximate
   scale-diverse processing? The expert routing may be a discrete
   approximation of the continuous spiral.

4. **Write up the photograph/hologram distinction.** This is the
   most significant theoretical finding of the session.
   → Done: `mementum/knowledge/explore/holographic-compression.md`

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
| **Session 041 probes** | |
| Pythia φ-probe | `scripts/run_pythia_phi_probe.py` |
| Pythia φ results | `results/pythia-phi/pythia_160m_phi_compression.json` |
| Qwen3-4B φ results | `results/pythia-phi/qwen3_4b_phi_compression.json` |
| **Logs & archives** | |
| Current training log | `results/vsm-lm-v6/training-run2.log` |
| Prior run log (frozen topology) | `results/vsm-lm-v6/training.log` |
| Prior run checkpoints | `checkpoints/a-vsm-lm-v6/` |
| **Probe results** | |
| v6.1 probes (steps 500–9000) | `results/compile-gradient/vsm_probe_step_*_v6_mlx.json` |
| **Research** | |
| Research program | `mementum/knowledge/explore/VERBUM.md` |
| **Holographic compression** | `mementum/knowledge/explore/holographic-compression.md` |
| φ-compression hypothesis | `mementum/knowledge/explore/relational-loss-phi-compression.md` |
| CompressorLM architecture | `mementum/knowledge/explore/compressor-architecture.md` |
| v4.1 training trajectory | `mementum/knowledge/explore/v4.1-training-trajectory.md` |
| Flip accumulation | `mementum/knowledge/explore/v6-flip-accumulation.md` |

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
# v6 probe (single or multiple checkpoints)
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_*

# Pythia φ-compression probe
uv run python scripts/run_pythia_phi_probe.py --verbose

# Resume training if interrupted
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_NNNNNN
```
