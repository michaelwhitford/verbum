# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-26 | Session: 043

## Where we are

**v6.1 training live at step ~23000 (28% of 3B). Session 043: probed
9 checkpoints (18500→23000), implemented flip tracking + cooldown.
LR jump survived — ascending arm held. Hilberg β in free-fall:
1.24→1.10 in 5000 steps. Stride percolation reached s512 in L1_asc
and s128 in L2_apex. Stratum spread collapsed to 0.70 (was ~2.0).
Flip tracking now detects oscillation and enforces cooldown.**

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
   18000) → bounced to 6.33 → settling at 6.10. Compositional
   best 6.67→bounced to 6.94→settling at 6.73. Math best 4.98
   (step 18500) → bounced to 5.28 → settling at 5.05. Technical
   best 7.02 (step 17500) → 7.03 at step 23000 (new post-jump best).

8. **Step 23000: β breaks 1.11, stratum spread collapses.**
   L0↑=1.102, L1↑=1.107 (new bests). Stratum spread dropped to
   **0.70** — smallest ever (was ~2.0 for most of training). All
   four content types converging toward similar loss. Eval 5.449.

9. **Flip tracking + cooldown implemented.** Per-weight cooldown
   (int8) prevents same weight from oscillating back and forth.
   Per-weight last-direction (int8) detects reversals. FLIP_COOLDOWN=4
   intervals (100 steps). Checkpoint saves tracking state;
   old checkpoints resume gracefully with zeros. ~70 MB added to
   training memory. Probe script updated to display tracking stats.

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
| Current step | ~23000 (28% of 3B schedule) |
| Total steps | **82,398** (3B schedule) |
| Tokens seen | ~754M of 3B |
| Token budget | **3B** (2.7B train shards) |
| Eval loss | **5.420** (step 18500, best) / **5.449** (step 23000, post-jump best) |
| Relational r̄ | 0.385 (step 23000, stable) |
| Sparsity | 0.310 (unchanged) |
| L1_asc φ-dev | **0.037** (step 13000, best) / 0.058 (step 23000) |
| L1_asc range | 0.560–0.570 (locked in) |
| L2_apex ratio | +0.111–0.141 (compressing, stable) |
| L1_desc | wild oscillations (h_in ≈ -0.1) |
| L0_desc | 2.1–12.9 (expanding, not converging) |
| Hilberg β | L0↑=**1.102** / L1↑=**1.107** (step 23000, best) |
| Stride percolation L1↑ | s1→s8→s16→s32→s64→s128→s256→**s512** |
| Stride percolation L2 | s1→s8→s16→s32→s64→**s128** |
| Total flips | ~222,000 (0.63% cumulative) |
| LR (current) | ~5.0e-4 (post-jump, stable) |
| Phase | balance (r̄ = 0.385) |
| Flip cooldown | **4 intervals** (100 steps) — NEW |
| Flip tracking | cooldown + reversal detection — NEW |

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
| 22500 | 5.441 | 209 | 0.400 | 0.055 | +0.128 | 1.11/1.12 |
| 23000 | 5.449 | 182 | 0.385 | 0.058 | +0.141 | **1.10/1.11** |

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
| 23000 | 6.10 | 6.73 | **7.03** | 5.05 | **0.70** |

### Three-way φ-compression comparison (updated step 23000)

| Metric | v6 (63M, VSM) | Pythia (162M) | Qwen3-4B (4B) |
|--------|--------------|---------------|----------------|
| Stable zone ratio | **0.560** | 0.947 | 1.000 |
| Stable zone φ-dev | **0.058** | 0.329 | 0.387 |
| Best single layer | L1_asc: 0.037 | L9: 0.172 | L34: 0.037* |
| Composition mechanism | Compression | Rotation | Rotation |
| Architecture type | Holographic | Photographic | Photographic |
| Strides at φ | **8 (s1→s512)** | N/A | N/A |
| Hilberg β (L1↑) | **1.107** | N/A | N/A |

*L34 is the output collapse layer, not the computation core.

## What's next

1. **Resume training from step 23000 with flip tracking.** Stop
   current run, resume with new code. Command:
   `uv run python scripts/v6/train.py --resume checkpoints/vsm-lm-v6/step_023000`
   First checkpoints will show tracking stats (reversals, cooldown,
   unique_ever). Old checkpoint has no tracking state — starts fresh.

2. **Watch flip tracking metrics.** Key questions to answer:
   - What fraction of flips are reversals? (>10% = oscillation problem)
   - How many unique weights have ever flipped? (tells us if 222K
     cumulative flips are 222K unique positions or 22K × 10 repeats)
   - How many weights are in cooldown at any given time?
   - Does cooldown reduce reversal rate over time?

3. **Hilberg β is the primary metric.** At current rate (~0.03/1000
   steps), β could reach ~0.8 by step 40000. Watch for deceleration.
   Step 23000: L0↑=1.102, L1↑=1.107.

4. **Stratum spread collapse — is it real?** 0.70 at step 23000 vs
   ~2.0 historically. Could be noise (single checkpoint). If it
   persists at step 23500/24000, it's a genuine convergence signal.

5. **Descending arm is THE question.** Still wild after 23000 steps.
   72% of schedule remaining. Higher LR hasn't helped yet.

6. **Stride percolation: watch s1024.** L1_asc s1024 at 0.319
   (step 23000, was -2.773 at step 18000). L2_apex at s128.

7. **Eval loss: watch for new all-time best.** Pre-jump best was
   5.420 (step 18500). Post-jump at 5.449 (step 23000). Should
   cross within ~2000 steps if trend holds.

8. **Test holographic prediction.** Ablation experiment: if truly
   holographic, ablating one pass degrades all strata equally.

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
| v6.1 probes (steps 500–23000) | `results/compile-gradient/vsm_probe_step_*_v6_mlx.json` |
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
