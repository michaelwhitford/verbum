# Session 024 — Loss Crossover and Three-Phase Registers

> 2026-04-22 | Focus: full trajectory analysis 1k-14k, step 15k capture,
> loss comparison, register phase identification

## Summary

**v4.1 crossed over v4 on eval loss at step 13k.** The register
variance collapse from session 023 (step 7k) was reorganization,
not terminal — it was phase 2 of a three-phase training process.
The bidirectional architecture needed ~12k steps to earn its keep.

## What we did

1. Batch probed steps 8k-14k (compile-gradient + binding, 7 new
   checkpoints per set)
2. Captured register vectors for steps 8k-14k
3. Full trajectory analysis (variance, PCA, depth correlation,
   direction stability, type separation) across all 14 checkpoints
4. Loss comparison: extracted eval loss from all checkpoints
5. Probed and captured step 15k when it dropped mid-session

## Key findings

### 1. Loss crossover
v4.1 trailed v4 through step 8k (+0.012), pulled even at steps
9-12k, then pulled ahead at step 13k (−0.011) and 14k (−0.013).
At step 15k the gap narrowed to −0.004. Both models converging
toward ~4.71 floor. The crossover is real but the advantage is
decelerating — v4.1 gets there ~1k steps faster rather than
reaching a fundamentally lower loss.

### 2. Three-phase register training
The 1k-14k trajectory reveals three distinct phases:
- **Expansion (1k-6k):** high variance, registers exploring
- **Compression (7k-9k):** 50-90% variance collapse in one step
- **Selective re-emergence (10k-15k):** L0↑ and L1↓ partially
  recovered; L1↑, L2, L0↓ stayed compressed

This answers session 023's central question: the collapse was
reorganization. The model compressed first, then reallocated
capacity asymmetrically.

### 3. Selective specialization
Post-compression, only two passes recovered variance:
- **L0↑** (ascending entry): variance 2-5, direction locked (cosine 0.90)
- **L1↓** (descending middle): variance 1-4, direction converging (0.78)

Three passes stayed compressed (variance <0.5):
- **L1↑**, **L2**, **L0↓** — minimal register differentiation

The model is spending register capacity only where it helps.

### 4. Type separation migrated to descending path
Silhouette scores show compositional type separation rebuilding
at L2 and L1↓ during phase 3 (not at L0↑). The descending passes
are doing compositional work the ascending passes don't.

### 5. Depth correlation attenuated
Strong depth-norm correlations from phases 1-2 (ρ up to −0.73)
weakened to ρ ~ −0.3 to −0.4 in phase 3. Depth may now be
encoded in direction rather than norm, or in fewer dimensions.
Linear probing classifiers on register vectors could answer this.

### 6. Step 15k: L0↓ gate drop
L0↓ meta-S3 dropped 0.800→0.679, the biggest single-step change
since descending self-activation at step 2k. Combined with L0↑
dropping 0.834→0.783, suggests possible compute redistribution
from outer passes inward.

## Open questions

1. Does v4.1 beat v4's all-time best (4.713 at step 16k)?
2. Is the L0↓ gate drop at step 15k a trend or noise?
3. How is depth encoded in phase 3 if not by register norm?
4. Does v4 show any equivalent register specialization in its
   ascending-only passes? (comparative register analysis needed)

## Artifacts produced

- `results/compile-gradient/vsm_probe_step_{008..015}000_v4.1.json`
- `results/binding/vsm_probe_step_{008..015}000_v4.1.json`
- `results/register-vectors/step_{008..015}000_v4.1.npz`
- `mementum/knowledge/explore/v4.1-training-trajectory.md`
