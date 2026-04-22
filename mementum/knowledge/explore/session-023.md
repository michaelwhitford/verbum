# Session 023 — Register Trajectory: Compression vs Structure

> 2026-04-22 | Focus: register analysis 1k-7k, variance collapse,
> depth correlation, loss comparison, binding routing shifts

## Summary

**The compressor is compressing its own representations.** Extended
register analysis from 3 checkpoints to 7 (steps 1k-7k). Registers
peaked in differentiation around steps 4k-6k then collapsed at 7k
— variance dropped 50-90% while depth correlation strengthened.
The model found a more efficient encoding that sacrifices register
diversity for compression quality. Loss tracks v4 within noise.

## What we did

1. Batch probed steps 4k-7k (compile-gradient + binding, 4 new
   checkpoints each)
2. Captured register vectors for steps 4k-7k
3. Full trajectory analysis: silhouette, variance, PCA, depth
   correlation, direction stability, register differentiation
4. Added PCA and depth correlation to `register_analysis.py`
   trajectory mode (was missing — now all metrics in one command)
5. Loss trajectory comparison v4.1 vs v4 through step 7k

## Key findings

### F1: Register variance collapse at step 7k

All registers across all passes saw dramatic variance reduction
from step 6k→7k. Selected examples:

| Register | Pass | Step 6k | Step 7k | Change |
|----------|------|---------|---------|--------|
| type | L1↑ | 19.7 | 2.1 | −89% |
| scope | L1↑ | 15.5 | 1.1 | −93% |
| type | L0↓ | 11.2 | 1.4 | −87% |
| role | L1↓ | 17.5 | 4.8 | −73% |
| role | L0↑ | 16.8 | 11.9 | −29% |

L0↑ role was the most resistant to collapse. The role register at
L1↓ that was spiking in session 022 (5.73→7.58→12.20 through step
3k) continued to 17.5 at step 6k then fell sharply.

### F2: Depth correlation strengthened despite collapse

Pearson r between register norm and FA composition depth:

| Register | Pass | Step 3k | Step 6k | Step 7k |
|----------|------|---------|---------|---------|
| type | L0↑ | −0.65 | −0.73★ | −0.71★ |
| scope | L0↑ | −0.68 | −0.70★ | −0.70★ |
| role | L0↑ | −0.61 | −0.71★ | −0.70★ |
| scope | L0↓ | −0.64 | +0.02 | −0.65★ |

L0↑ depth encoding is the strongest and most stable signal —
monotonically strengthening from step 2k onward. Descending passes
(L1↓, L0↓) show volatile depth correlation, losing it at steps
5k-6k then partially recovering at 7k.

### F3: Loss tracks v4 within noise

| Step | v4.1 | v4 | Δ |
|------|------|----|---|
| 4k | 5.244 | 5.241 | +0.003 |
| 5k | 5.136 | 5.132 | +0.004 |
| 6k | 5.070 | 5.042 | +0.028 |
| 6.5k | 5.020 | 5.026 | −0.006 |
| 7k | 5.027 | 5.016 | +0.011 |

v4.1 briefly led at step 6.5k but the gap is noise-level. v4
reached 4.732 by step 15k. The bidirectional architecture has not
yet demonstrated a loss advantage.

### F4: Meta-S3 declining across the board

| Pass | Step 1k | Peak | Step 7k | Trend |
|------|---------|------|---------|-------|
| L0↑ | 0.898 | 0.951 (3k) | 0.808 | declining |
| L1↑ | 0.896 | 0.896 (1k) | 0.505 | halved |
| L2 | 0.502 | 0.755 (2k) | 0.546 | stable |
| L1↓ | 0.047 | 0.871 (2k) | 0.609 | declining |
| L0↓ | 0.037 | 0.963 (4k) | 0.866 | declining |

Every pass peaked and is now declining or stable. The system is
reducing how much it uses the multi-pass architecture.

### F5: Binding routing shifts

Variable binding at L0↓ collapsed from 0.884 (4k) to 0.559 (7k).
Control structures at L1↓ strengthened from 0.596 (4k) to 0.913 (7k).
Relative clause at L2 declined from 0.482 to 0.278. The descending
path is specializing narrowly for control structures while
deprioritizing everything else.

### F6: Direction stability — ascending stable, descending searching

Cosine similarity of mean register vectors between consecutive steps:

| Pass | 3→4k | 4→5k | 5→6k | 6→7k |
|------|------|------|------|------|
| L0↑ | 0.78 | 0.75 | 0.77 | 0.80 |
| L1↑ | 0.50 | 0.64 | 0.71 | 0.45 |
| L2 | 0.25 | 0.42 | 0.48 | 0.53 |
| L1↓ | 0.40 | 0.43 | 0.57 | 0.61 |
| L0↓ | 0.24 | 0.36 | 0.29 | 0.37 |

(Averaged across registers.) L0↑ is locked in. L0↓ is still
wandering. L1↓ is converging.

## Interpretation

The register variance collapse is a compression phase transition.
The model explored high-variance register representations during
steps 2k-6k (the "expansion" phase after descending activation),
then discovered that most of that variance was wasteful for
prediction. It compressed the register space while preserving
(and strengthening) the depth encoding.

This is consistent with the framing: **the compressor optimizes
for prediction, not for interpretability**. Expansion declining +
loss declining = the compressor finding the function through a
more efficient path. The organization doesn't matter as long as
both numbers keep going down.

## Open questions (updated from session 021)

1. ~~Will descending passes self-activate?~~ → YES (session 021)
2. ~~Do registers encode depth?~~ → YES, strengthening (session 022-023)
3. **Is the register collapse permanent or a reorganization phase?**
   Next 2-3 checkpoints (8k-10k) are decisive.
4. **Will v4.1 separate from v4?** v4 went to 4.732. If v4.1
   can't match that, the extra architecture is overhead.
5. **Is LM loss the right metric for compositional structure?**
   Depth correlation strengthens while registers compress — the
   structure is there but increasingly invisible to variance-based
   measures. Linear probes might be needed.

## Artifacts produced

- Probes: `results/compile-gradient/vsm_probe_step_00{4,5,6,7}000_v4.1.json`
- Probes: `results/binding/vsm_probe_step_00{4,5,6,7}000_v4.1.json`
- Register vectors: `results/register-vectors/step_00{4,5,6,7}000_v4.1.npz`
- Script: `register_analysis.py` trajectory mode now includes PCA + depth correlation
