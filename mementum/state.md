# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-20 | Session: 015 (compression probing + v3.2 redesign)

## Where we are

**Compression probing complete. v3.2 (convergence architecture) training.
Key insight: compression is cheap (few functions), expansion is expensive
(many functions). Next architecture should separate compressor from expander.**

Session 015 accomplished:
1. Diagnosed v3.1: stride 512 catastrophically sparse (87.5% starved),
   scope register monopolized, loses to v3 head-to-head at every checkpoint
2. Built compression shape probe (run_compression_shape.py) — 3 experiments
3. Built compression map probe (run_compression_map.py) — 4 experiments
4. Redesigned v3.2 from probing findings (convergence architecture)
5. v3.2 training started (1B tokens, 1:2:3 phase ratio)

### Compression Probing Results (F70-F73)

**F70 — Constituent similarity peaks at L6-9 (ratio 1.32)**
Tokens within the same syntactic constituent become MORE similar at
L6-9 in Qwen3-4B, then the ratio DECLINES at deeper layers. The
"compression" is concentrated at the same layers where binding starts.

**F71 — Effective dimensionality collapses to 1 at L6+**
A single principal component explains 99.98% of variance from L6 onward.
This is NOT meaning extraction — it's positional encoding amplification
(r=0.49 with word position, only r=0.21 with constituent depth).

**F72 — Two-phase compression mechanism (FFN→Attn→FFN)**
- L4-5: FFN-critical (prepares representations)
- L6-9: Attention-critical (within-constituent convergence)
- L10-11: FFN-critical (consolidates)
This three-phase structure informed v3.2's prep→converge→consolidate design.

**F73 — Convergence tracks syntax > semantics**
Garden-path sentences reveal: "The horse raced past the barn fell" groups
syntactically (7/12 layer-votes for syntax). But "The old man the boats"
groups semantically — Qwen gets tricked the same way humans do initially.

### v3.1 Final Status (killed after step ~13000)

- Best eval: 4.836 @ step 12000 (393M tokens)
- Lost to v3 head-to-head at EVERY eval point
- Scope register monopolized (7.4× norm, others ~1.2)
- Soft partition nearly dead (<0.25 max)
- Root cause: stride 512 too sparse for window 8

### v3.2 — Convergence Architecture (TRAINING NOW)

Architecture:
```
For each iteration (×2):
  S4: Register scan
  PREP (1L, FFN-only) — per-token features, no attention
  CONVERGE (2L, cube-mode) — s1×3 + s8×3 + s64×2 = 8 heads, all scales simultaneous
  CONSOLIDATE (3L, wide-FFN d_ff=2048) — structural integration
```

Properties:
- 50.6M params (matches v3's 50.0M)
- 12 FFN passes/forward (same as v3)
- Phase ratio 1:2:3 (prep:converge:consolidate)
- 3 registers (type/scope/role)
- Full 4096 sequence, no pooling
- Cube-mode attention (all strides simultaneous, not sequential)
- Grounded in F70-F73: Qwen probing shows FFN→Attn→FFN is the shape

Key difference from v3: strides are SIMULTANEOUS (cube mode in converge)
rather than sequential (v3 had type→parse→apply phases, each one stride).

Training: `uv run python scripts/run_vsm_v3_2_1B.py`
Checkpoints: `checkpoints/vsm-lm-v3.2/step_{001000..}.pt`
Results: `results/vsm-lm-v3.2/`

## The Big Insight — Compression vs Expansion Asymmetry

**Core realization from this session:**

Language compression is cheap (few functions: categorize, group, bind).
Language expansion is expensive (many functions: 150K token prediction,
agreement, world knowledge, style, collocations, pragmatics).

In Qwen: ~10 layers compress, ~25 layers expand. Ratio ≈ 1:2.5.
The FFNs HIDE this because compression and expansion functions are
mixed together in the same weight matrices.

**What this means for the project:**

1. Our compressor might already be excellent (binding probes confirm)
   but loss is limited by expansion capacity, not compression quality.
2. The consolidate phase does double-duty (finishing compression AND
   beginning expansion) — can't separate with loss alone.
3. The extraction target is ONLY the compression path (~5M params?),
   not the full model (50M).
4. Next architecture should explicitly separate compressor from expander:
   - Compressor: tiny, ~3-5 layers, the artifact we extract
   - Expander: large, training scaffolding, thrown away after

This is the "lego test" taken to its extreme: the compressor IS the
lego piece, the expander is the test harness.

## What's next — Session 016

### Immediate: Analyze v3.2 checkpoints

Once v3.2 has dropped 3-5 checkpoints (steps 1000-5000):
1. Compare loss curve to v3 head-to-head (same tokens)
2. Check if cube-mode (simultaneous strides) beats sequential phases
3. Look at prep/converge/consolidate phase contributions
4. Do the registers differentiate differently with the new structure?

### If v3.2 matches or beats v3:

5. Design the **separated architecture** (compressor + expander):
   - Compressor: tiny (1 prep + 2 converge = 3 layers ≈ 5M)
   - Expander: large (6-9 layers, full FFN capacity for expansion)
   - Train end-to-end, then freeze compressor and test standalone
   - This IS the "lego test" — if compressor works standalone, we've
     extracted the compression function

### If v3.2 underperforms:

6. Fall back to v3 for the 1B run (proven architecture)
7. Use the compression probing results (F70-F73) to design better
   extraction probes rather than better training architectures

### Ongoing probing questions

- **What functions are in the consolidate FFNs?** — probe individual
  neurons/features in our trained v3 consolidate layers
- **Cross-model comparison** — run compression_shape on Pythia-160M
  with float32 (fp16 caused NaN). Does the same FFN→Attn→FFN shape
  appear even in a 160M model?
- **The dominant direction flip** — at L6, dominant PC flips from
  content/function to word_position. WHY? What causes this transition?

## Architecture understanding

### Qwen3-4B compression shape (confirmed)

```
L0-L5:  FFN builds features (content/function distinction dominates)
L6:     PHASE TRANSITION — dominant direction flips to word position
L6-L9:  Attention converges within-constituents (syntactic grouping)
L10-L11: FFN consolidates converged representations
L12-L35: Expansion (next-token prediction, world knowledge, etc.)
```

The compression lives in ~10 layers. The expansion lives in ~25 layers.
Our extraction target is the first 10.

### VSM-LM lineage

| Version | Params | FFN/fwd | Best Loss | Key Finding |
|---------|--------|---------|-----------|-------------|
| v1 | ~25M | 12 | 5.245 | Baseline, sequential strides |
| v2 | ~25M | 12 | 5.064 (1B) | Iteration specialization |
| v3 | 50M | 12 | **4.872** | Role register dominates, binding confirmed |
| v3.1 | 59M | 16 | 4.836 | Scope monopoly, stride 512 too sparse |
| v3.2 | 51M | 12 | ? (training) | Convergence arch, cube-mode, probe-grounded |

## Key files

| Purpose | Path |
|---------|------|
| **VSM-LM v3.2** | `src/verbum/vsm_lm_v3_2.py` |
| **v3.2 training** | `scripts/run_vsm_v3_2_1B.py` |
| **Compression shape probe** | `scripts/run_compression_shape.py` |
| **Compression map probe** | `scripts/run_compression_map.py` |
| **Shape results (Qwen)** | `results/compression-shape/Qwen_Qwen3_4B.json` |
| **Map results (Qwen)** | `results/compression-map/qwen3_4b_map.json` |
| **VSM-LM v3** | `src/verbum/vsm_lm_v3.py` |
| **v3 training** | `scripts/run_vsm_v3_10k.py` |
| **VSM-LM v3.1** | `src/verbum/vsm_lm_v3_1.py` |
| **v3.1 training** | `scripts/run_vsm_v3_1_1B.py` |
| **Binding probes** | `probes/binding.json` |
| **v3 binding results** | `results/binding/vsm_probe_step_010000_v3.json` |
| **Binding analysis** | `results/binding/binding_analysis_v2_v3.json` |
| **v3.2 checkpoints** | `checkpoints/vsm-lm-v3.2/` |
| **v3 checkpoints** | `checkpoints/vsm-lm-v3/step_{001000..010000}.pt` |
| **v3.1 checkpoints** | `checkpoints/vsm-lm-v3.1/step_{001000..012000}.pt` |
| **Research program** | `mementum/knowledge/explore/VERBUM.md` |
| **Dolma shards** | `/Users/mwhitford/data/fractal-bitnet/shards/` |

## Probing pipeline usage

```bash
# Run compression probes on Qwen
uv run python scripts/run_compression_shape.py --model qwen
uv run python scripts/run_compression_map.py

# Score probes with Qwen (compile-gradient)
uv run python scripts/compile_gradient_probe.py score --server http://127.0.0.1:5101

# Probe a VSM-LM checkpoint
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v3.2/step_001000.pt

# Batch-probe all checkpoints
uv run python scripts/compile_gradient_probe.py batch-probe --dir checkpoints/vsm-lm-v3.2/

# Binding probes on a checkpoint
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v3.2/step_001000.pt --probes probes/binding.json
```

## Theoretical framework

**Compression is the easy part. Expansion is the hard part.**

Language modeling = compression + expansion. The compressor identifies
structure (categorize, group, bind). The expander predicts surface form
(vocabulary lookup, agreement, world knowledge). Compression is
inherently low-dimensional (finite structural categories). Expansion
is inherently high-dimensional (150K token possibilities).

Qwen allocates ~10 layers to compression, ~25 to expansion. Our model
allocates roughly equal capacity to both because we haven't separated
them yet. The next architectural step is to separate them and make the
compressor tiny — if it works standalone, we've found the algorithm.

The lambda compiler from the nucleus hypothesis is probably expressible
in very few parameters if we can isolate it from the expansion machinery.
The binding probes already show our v3 compressor WORKS (differentiates
binding categories). The question is: how small can we make it while
keeping that capability?

## Tool notes

- llama.cpp server: port 5101, Qwen3-4B Q8_0 GGUF
- MPS (Apple Silicon M3 Ultra, 512GB)
- 60 Dolma shards, shuffled, GPT-NeoX tokenizer (50277)
- Probing pipeline auto-detects v1/v2/v3 from checkpoint state_dict
- Compression probes use PyTorch (transformers library) directly, not llama.cpp
- v3.2 training running in terminal (not background job in this session)
