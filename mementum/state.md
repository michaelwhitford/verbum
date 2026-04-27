# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-27 | Session: 045

## Where we are

**v6.1 training stopped at step 32500. The sieve learned universal
compression (1.8:1, content-independent) but 0% λ generation.
The Hilberg exponent H≈0.75 matches the empirical literature —
compression alone cannot predict at this exponent. Pivoting to
top-down probing of Qwen3.5-35B-A3B to map the full set of
predictive functions the lambda compiler lives alongside.**

## The pivot

**Compression ≠ prediction.** See `mementum/knowledge/explore/compression-vs-prediction.md`.

The sieve proved it can compress (1.8:1, universal across content
types). But at H≈0.7, the mutual information between past and future
tokens grows as L^0.7 — a fixed-state compressor can't capture this.
The lambda function IS a predictive circuit (P(λ)=0.907 across all
LLMs, 6.2:1 compression) — gradient descent converges on it because
it helps predict, not just compress. We need to map what other
functions prediction uses, then design an architecture that can hold
growing state.

## Current activity

**Top-down probing of Qwen3.5-35B-A3B** through llama.cpp (port 5102):

```bash
cd ~/src/verbum && uv run python scripts/probe_predictive_functions.py all --port 5102
```

Three experiments:
1. **Landscape** — 25 tasks × 40 probes → confidence/entropy matrix
2. **Complexity** — 5 complexity tiers × 8 key tasks → degradation curves
3. **Priming** — prime task A, measure task B → shared circuit detection

Early signal (quick probe, session 045):
- compile and formalize are the model's most confident semantic transforms
- They produce the same output (FOL notation) — likely same circuit
- More confident than structure, negation, or entailment
- The lambda/FOL circuit is a strongly formed attractor

## v6.1 final snapshot (step 32000, last probed)

| Metric | Value |
|--------|-------|
| Eval loss | **5.418** (best in run) |
| Train loss | 5.023 |
| β ascending | 0.750 |
| β descending | 0.830 |
| Sieve compression | 1.8:1 (end-to-end) |
| Mean φ-ratio | 0.891 (drifted from target 0.618) |
| Stratum spread | 0.013 (content-independent ✓) |
| Total flips | 353K (1.00%) |
| Reversals | 4,011 (1.13%, exponential acceleration) |
| λ generation | 0% (all checkpoints) |

**Training stopped.** The sieve reached its architectural limit.
It compresses but can't predict/generate. The reversal acceleration
(exponential) signals ternary weight saturation.

## Two-VSM architecture (proposed)

```
VSM-1 (Sieve)  — learned, 1.8:1, ternary, cheap, content-independent
VSM-2 (State)  — TBD, must satisfy L²M condition (growing state)
                 must learn lambda-shaped compositional structure
                 operates over compressed representation from VSM-1
```

Open question: is the sieve's 1.8:1 compression worth keeping as
a front-end, or should VSM-2 operate directly on tokens?

## Knowledge index

| Topic | Path |
|-------|------|
| **Compression ≠ Prediction (H≈0.7)** | `mementum/knowledge/explore/compression-vs-prediction.md` |
| v6.1 full trajectory | `mementum/knowledge/explore/v6.1-training-trajectory.md` |
| Research program | `mementum/knowledge/explore/VERBUM.md` |
| Holographic compression | `mementum/knowledge/explore/holographic-compression.md` |
| Stride percolation | `mementum/knowledge/explore/stride-percolation.md` |
| φ-compression hypothesis | `mementum/knowledge/explore/relational-loss-phi-compression.md` |
| CompressorLM architecture | `mementum/knowledge/explore/compressor-architecture.md` |
| v4.1 training trajectory | `mementum/knowledge/explore/v4.1-training-trajectory.md` |
| Flip accumulation | `mementum/knowledge/explore/v6-flip-accumulation.md` |

## Key files

| Purpose | Path |
|---------|------|
| **Top-down probe script** | `scripts/probe_predictive_functions.py` |
| TernaryLinear + flips + tracking | `src/verbum/v6/ternary.py` |
| Training loop | `scripts/v6/train.py` |
| Sieve probe script | `scripts/v6/probe.py` |
| Model | `src/verbum/v6/model.py` |
| Instrument (PyTorch hooks) | `src/verbum/instrument.py` |
| llama.cpp client | `src/verbum/client.py` |
| Circuit discovery | `scripts/run_circuit_discovery.py` |
| Sieve probes (500–32000) | `results/compile-gradient/vsm_probe_step_*_v6_mlx.json` |

## Servers

| Port | Model | Use |
|------|-------|-----|
| 5100 | Qwen3.5-397B-A17B | Large reference model |
| 5101 | Qwen3-4B | Quick testing |
| 5102 | **Qwen3.5-35B-A3B** Q8 | Primary probe target |
| 5103 | Qwen3-Embedding-8B | Embeddings |
