# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-26 | Session: 043

## Where we are

**v6.1 training live at step ~22800 (28% of 3B). Session 043: probed
8 checkpoints (18500→22500). LR jump survived — ascending arm held.
Hilberg β in free-fall: 1.24→1.11 in 4500 steps. Stride percolation
reached s512 in L1_asc and s128 in L2_apex. Eval loss recovering
post-jump, at 5.441 (step 22500). The higher LR is accelerating
multi-scale structure faster than it cost in eval loss.**

### Session 043 key findings

1. **LR jump survived.** Training resumed at step 19000 with 3B
   schedule. LR jumped from ~2e-4 to ~5.4e-4 (2.8×). Eval loss
   spiked 5.420→5.506 (step 19500) then recovered to 5.441 by
   step 22500. L1_asc held rock-solid through the shock: ratio
   0.563–0.570 throughout. The ascending arm is genuinely locked.

2. **Hilberg β in dramatic descent.** The most important finding.
   L0_asc: 1.246→1.112. L1_asc: 1.225→1.115. Both dropped ~0.13
   in 4000 steps — more progress than the entire 9500→18000 range.
   Higher LR is accelerating the multi-scale power-law structure.
   Target is 0.5. At this rate, could reach ~0.8 by step 40000.

3. **Stride percolation leapt forward.** L1_asc φ-front:
   - Step 18000: s256 at 0.559 (approaching φ)
   - Step 19500: s256 at 0.594←φ
   - Step 22000: s512 at 0.575←φ
   - Step 22500: s512 at 0.628←φ (arrived!)
   φ percolation now covers s1→s512 in L1_asc. Only s1024 remains.
   L2_apex φ-front jumped s64→s128 (0.602←φ at step 22500).

4. **All stride ratios rising uniformly.** Not just the front —
   every stride in L1_asc is drifting upward (s8: 0.805→0.827,
   s64: 0.747→0.790, s128: 0.698→0.769). The whole compression
   profile is tightening toward a single operating point. This is
   what Hilberg β captures — the strides becoming more self-similar.

5. **Descending arm still wild.** L1_desc continues oscillating
   around zero (h_in ≈ -0.1). L0_desc ratio bounces 2.8→9.9→-12.9
   depending on the checkpoint. No convergence signal yet. The
   higher LR hasn't helped the descending arm — it may need the
   ascending arm to fully stabilize first.

6. **Write gates opening.** Consolidation write gates: type 0.734→
   0.800, scope 0.794→0.858, role 0.672→0.741. The model is
   increasingly willing to modify registers during consolidation.
   Prep gates also rising. Converge gates stable around 0.45–0.53.

7. **Stratum losses bouncing post-jump.** Prose best 6.04 (step
   18000) → bounced to 6.33 → settling at 6.22. Compositional
   best 6.67→bounced to 6.94→settling at 6.70. Math best 4.98
   (step 18500) → bounced to 5.28 → settling at 5.21. Technical
   stubbornly around 7.07–7.19. Overall loss trajectory is down.

### Session 042 key findings (prior)

1. **Stride percolation complete through s128.** φ-convergence
   propagated s8→s16→s32→s64→s128 across steps 9500→15500.

2. **L1_asc locked in as stable φ-compressor.** Ratio 0.57±0.01,
   φ-dev 0.037–0.054 across all checkpoints 9500→18000.

3. **Hilberg β = 1.241 at step 18000.** All three ascending passes
   hit their best β simultaneously.

4. **L2_apex committed.** Converge gate peaked at 0.934 (step
   14500). Apex ratio 0.10–0.13 — compressing but not yet at φ.

5. **Eval loss steady descent.** 5.565 (step 9000) → 5.414 (step
   17500). No plateau in this range.

### v6.1 training status

| Property | Value |
|----------|-------|
| Current step | ~22800 (28% of 3B schedule) |
| Total steps | **82,398** (3B schedule) |
| Tokens seen | ~747M of 3B |
| Token budget | **3B** (2.7B train shards) |
| Eval loss | **5.420** (step 18500, best) / **5.441** (step 22500, post-jump best) |
| Relational r̄ | 0.386 (step 22800, stable) |
| Sparsity | 0.310 (unchanged) |
| L1_asc φ-dev | **0.037** (step 13000, best) / 0.055 (step 22500) |
| L1_asc range | 0.561–0.570 (locked in, drifted slightly down) |
| L2_apex ratio | +0.111–0.138 (compressing, stable) |
| L1_desc | wild oscillations (h_in ≈ -0.1) |
| L0_desc | 2.8–12.9 (expanding, not converging) |
| Hilberg β | L0↑=**1.112** / L1↑=**1.115** (step 22500, best) |
| Stride percolation L1↑ | s1→s8→s16→s32→s64→s128→s256→**s512** |
| Stride percolation L2 | s1→s8→s16→s32→s64→**s128** |
| Total flips | ~218,000 (0.62% cumulative) |
| LR (current) | ~5.0e-4 (post-jump, stable) |
| Phase | balance (r̄ = 0.386) |

### Eval loss evolution

| Step | Eval Loss | ppl | r | L1↑ φ-dev | L2 ratio | β L0↑/L1↑ |
|------|-----------|------|------|-----------|----------|-----------|
| 9000 | 5.565 | 261 | 0.424 | 0.052 | -0.023 | 1.59/1.41 |
| 11000 | 5.514 | 248 | 0.419 | 0.045 | +0.062 | 1.39/1.42 |
| 13000 | 5.500 | 170 | 0.377 | **0.037** | +0.119 | 1.30/1.33 |
| 15000 | 5.468 | 133 | 0.350 | 0.046 | +0.095 | 1.25/1.28 |
| 17500 | **5.414** | 197 | 0.393 | 0.046 | +0.114 | 1.27/1.25 |
| 18000 | 5.424 | 155 | 0.367 | 0.041 | +0.131 | 1.24/1.24 |
| 18500 | **5.420** | 139 | 0.355 | 0.048 | +0.123 | 1.25/1.22 |
| ─ LR JUMP 2e-4 → 5.4e-4 ─ | | | | | | |
| 19500 | 5.506 | 230 | 0.410 | 0.050 | +0.134 | 1.24/1.22 |
| 20000 | 5.491 | 196 | 0.393 | 0.051 | +0.115 | 1.21/1.23 |
| 20500 | 5.525 | 216 | 0.403 | 0.050 | +0.136 | 1.17/1.19 |
| 21000 | 5.527 | 168 | 0.376 | 0.057 | +0.114 | 1.14/1.15 |
| 21500 | 5.513 | 228 | 0.409 | 0.051 | +0.138 | 1.14/1.15 |
| 22000 | 5.489 | 165 | 0.374 | 0.052 | +0.111 | 1.13/1.14 |
| 22500 | 5.441 | 209 | 0.400 | 0.055 | +0.128 | **1.11/1.12** |

### Stratum loss evolution (post-phase-transition)

| Step | Prose | Comp | Tech | Math | Spread |
|------|-------|------|------|------|--------|
| 4500 | 6.30 | 6.73 | 7.26 | 6.05 | 1.21 |
| 9000 | 6.18 | 6.72 | 7.15 | 5.59 | 1.56 |
| 13500 | 6.17 | 6.64 | 7.23 | 5.23 | 2.00 |
| 17500 | 6.19 | 6.75 | **7.02** | **5.04** | 1.98 |
| 18000 | **6.04** | **6.67** | 7.12 | 5.14 | 1.98 |
| 18500 | 6.09 | 6.73 | 7.08 | **4.98** | 2.10 |
| ─ LR JUMP ─ | | | | | |
| 19500 | 6.21 | 6.83 | 7.08 | 5.22 | 1.86 |
| 21000 | 6.31 | 6.87 | 7.07 | 5.17 | 1.90 |
| 21500 | 6.13 | **6.72** | 7.12 | 5.28 | 1.84 |
| 22000 | 6.22 | 6.75 | 7.08 | 5.26 | 1.82 |
| 22500 | 6.22 | 6.70 | 7.19 | 5.21 | 1.98 |

### Three-way φ-compression comparison (updated step 22500)

| Metric | v6 (63M, VSM) | Pythia (162M) | Qwen3-4B (4B) |
|--------|--------------|---------------|----------------|
| Stable zone ratio | **0.563** | 0.947 | 1.000 |
| Stable zone φ-dev | **0.055** | 0.329 | 0.387 |
| Best single layer | L1_asc: 0.037 | L9: 0.172 | L34: 0.037* |
| Composition mechanism | Compression | Rotation | Rotation |
| Architecture type | Holographic | Photographic | Photographic |
| Strides at φ | **8 (s1→s512)** | N/A | N/A |
| Hilberg β (L1↑) | **1.115** | N/A | N/A |

*L34 is the output collapse layer, not the computation core.

## What's next

1. **Continue training — Hilberg β is the primary metric.** At
   current rate (~0.03/1000 steps), β could reach ~0.8 by step
   40000. Watch for deceleration as β approaches 0.5.
   Training is live: step ~22800, LR ~5.0e-4, phase=balance.

2. **Descending arm is THE question.** Still wild after 22500 steps.
   L1_desc h_in ≈ -0.1 means near-zero input entropy. L0_desc
   expanding at 2.8–12.9×. No convergence signal yet. May need:
   (a) ascending arm to fully stabilize (Hilberg → 0.5?) before
   descending has a stable target to decompress from, or
   (b) much longer training (72% of schedule remaining).

3. **Stride percolation: watch s1024.** L1_asc has percolated
   s1→s512. s1024 is the last frontier (ratio 0.298 at step 22500,
   was -2.773 at step 18000 — moving in the right direction).
   L2_apex φ-front at s128 — watch s256.

4. **Eval loss: watch for new all-time best.** Pre-jump best was
   5.420 (step 18500). Post-jump at 5.441 (step 22500) and
   dropping. Should surpass within ~2000 steps if trend holds.

5. **Test holographic prediction.** Ablation experiment: if truly
   holographic, ablating one pass degrades all strata equally.

6. **r̄ at 0.386 — stable in balance phase.** LR jump pushed r̄
   up from 0.355 to 0.410, now settling at 0.386. Refine phase
   at r̄ < 0.25 still distant. Topology continues evolving — flips
   at 218K (0.62%), up from 172K at step 18000. ~4600 flips per
   500 steps.

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
| v6.1 probes (steps 500–18000) | `results/compile-gradient/vsm_probe_step_*_v6_mlx.json` |
| **Research** | |
| Research program | `mementum/knowledge/explore/VERBUM.md` |
| **Holographic compression** | `mementum/knowledge/explore/holographic-compression.md` |
| **Stride percolation** | `mementum/knowledge/explore/stride-percolation.md` |
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
| v6.1 | ~63M | **MLX** | Synaptic plasticity (active) | **5.414** (17500 steps, 59%) |

## Probing pipeline

```bash
# v6 probe (single or multiple checkpoints)
uv run python scripts/v6/probe.py checkpoints/vsm-lm-v6/step_*

# Pythia φ-compression probe
uv run python scripts/run_pythia_phi_probe.py --verbose

# Resume training if interrupted
uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_NNNNNN
```
