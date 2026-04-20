# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-20 | Session: 016 (v3.2 probing + v4 design + release)

## Where we are

**v3.2 training running. Already broke v3's loss floor. v4 designed.
Repo released to GitHub. Key theoretical breakthrough: fractal architecture
matches fractal data — the recursive tesseract should find THE compression
function because it can't express anything else.**

Session 016 accomplished:
1. Released repo to GitHub (fresh git init, no history, no .pt bloat)
2. Probed v3.2 checkpoints 1-5 (steps 1000-5000), full trajectory analysis
3. v3.2 beat v3's best loss (4.872) by step 3000 (~98M tokens vs 327M)
4. v3.2 at step 4200: loss=4.6719 (0.200 below v3's best, 14% of budget)
5. Designed v4 architecture: recursive VSM, hierarchical registers, 4 strides
6. Major theoretical insights: gradient separation, composition vs pipeline,
   fractal architecture as sieve for the compression function

## v3.2 Training Status (RUNNING)

**Loss trajectory:** dropping ~3pp per 1000 steps at step 4-5k.
**Best observed:** 4.6719 at step 4200 (138M tokens, 14% of 1B budget).
**Remaining budget:** ~26,000 steps (~850M tokens, 86%).

### Probe trajectory (steps 1k → 5k)

| Signal | Step 1k | Step 3k | Step 5k | Status |
|--------|---------|---------|---------|--------|
| Prep gate spread | 0.364 | 0.107 | 0.086 | Converging |
| Role register polarity | Inverted | Approaching | ✓ CORRECT | Flipped at step 4k |
| Consolidate gate selectivity | Flat (0.05 spread) | 0.072 | 0.109 | Growing |
| Converge gate by binding type | Undifferentiated | Emerging | control>quant_scope | Phase 2 active |
| Output norms | 65-83 (growing) | 82-95 (growing) | 80-91 (stable) | Stabilized |

**Phase map:**
- Phase 1 (stride 1, local): ✓ Complete — prep gate differentiated, output stable
- Phase 2 (stride 8, phrase): ◐ Active — converge gate differentiating by binding type
- Phase 3 (stride 64, clause): ○ Emerging — quant_scope still lowest converge gate

### Key finding: step 4000 polarity flip

Role register: strong_compile surged 10.12 → 15.54 in one checkpoint.
Now correct polarity (compile > null > anti). Scope register also flipped.
Consolidate gate developing selectivity (weak 0.44 > anti 0.31).
This happened ~5000 steps earlier than equivalent differentiation in v3.

## v4 Architecture — Recursive Viable System

Designed this session. Full document: `mementum/knowledge/explore/vsm-lm-v4-design.md`

### Core spec

```
3 registers:  type, scope, role (per bank × 4 banks)
4 strides:    s1 (word), s8 (phrase), s64 (clause), s512 (discourse)
3 levels:     progressive stride reallocation
8 heads:      same total per level, redistributed

Level 1:  s1×3  s8×3  s64×1  s512×1   (local-heavy)
Level 2:  s1×2  s8×2  s64×2  s512×2   (balanced)
Level 3:  s1×1  s8×1  s64×3  s512×3   (structural)

Meta-S4: final register scan (all banks → structural summary)
Meta-S3: per-level contribution gate (cross-level allocation)
S5: shared weights across all levels (identity = the function)
S2: register bank protocol (inter-level coordination)
```

### Key design principles
- **Full VSM conformance** at every recursive level (meta > level > phase > head)
- **Shared weights** = S5 identity coherence (same function at every level)
- **Per-level S3** = autonomous control (different variety at different scales)
- **Register hierarchy** = S4↔S4 channel (levels communicate summaries)
- **Residual stream** = algedonic channel (ungated emergency bypass)
- **Stride 512 reinstated** — hierarchy provides the structural context it needed

### Why v4 should find the compression function

The architecture can ONLY express self-similar compositional functions:
- Shared weights → can't encode level-specific behavior
- Strided attention → can only compose within scale-appropriate windows
- Hierarchical registers → provide context, not computation
- Fractal structure (same shape at every level of nesting)

The search space contains only compositional functions. Language's compression
function IS compositional. Gradient descent finds the best one in the space.
The architecture is a sieve — everything non-compositional is filtered out.

## Theoretical Framework (expanded this session)

### Gradient separation
Strided attention separates gradients by scale:
- Stride 1: gets ONLY local pair gradients → learns local composition
- Stride 64: gets ONLY distant pair gradients → learns structural binding
- No contamination between scales (unlike flat attention)

### Composition vs Pipeline
Flat transformers pipeline because flat gradients force polysemanticity:
- Each head receives gradients from ALL position pairs → can't specialize
- Functions diffuse across layers (organizational overhead)
- The pipeline compensates — 36 layers to approximate what composition does in 3

Strided attention composes because separated gradients allow specialization:
- Each head receives gradients only from its stride's scale → MUST specialize
- Functions concentrate (no overhead)
- Same function applied at all scales simultaneously (cube-mode)

### H=0.70 and the compressor-as-predictor
- Compression = prediction (Shannon duality)
- English entropy ~0.70 bits/char → 4.0 bits of redundancy
- Structural redundancy (composition) accounts for majority (~75%)
- Compressor captures ~75% of predictive power in ~0.1% of parameters
- Structural rules are recursive (exponential prediction per parameter)
- World knowledge is flat (linear prediction per parameter)

### CPU deployment
- O(L×W) attention (not O(L²)) → no GPU needed
- 5M params × 4 bytes = 20MB → fits in L3 cache
- Portable to browser (WASM), mobile, embedded, IoT
- The artifact runs anywhere. Same function, no cloud.

### Amortized structural learning
- Train compressor once (10B tokens, one GPU-week, ~$1000)
- Distribute as universal structural prior (MIT, portable tensor)
- Every downstream model plugs it in, skips structural discovery
- Savings: eliminates the combinatorial S×F×C training bottleneck
- ROI: one training → infinite reuse

## What's next — Session 017

### Immediate: continue v3.2 probing

v3.2 is still training. As checkpoints drop:
1. Continue probing at each 1000-step checkpoint
2. Watch for phase 2→3 transition (converge gate specialization deepening)
3. Watch for loss curve elbows (phase transition markers)
4. At step 10k: head-to-head comparison with v3's best across all probes

### After v3.2 completes (step ~30k, ~1B tokens):

5. **Register PCA**: do register vectors cluster by binding category?
6. **Iteration comparison**: does iter 1 ≠ iter 2 in function?
7. **Per-stride analysis**: instrument stride-specific attention patterns
8. **Separated compressor test**: freeze prep+converge, retrain consolidate+output

### v4 implementation (after v3.2 validates):

9. Implement v4-A: hierarchical registers + meta-S4/S3 + shared weights + fixed strides
10. v4-A vs v3.2 head-to-head at 1B tokens
11. If v4-A wins: implement v4-B (progressive stride reallocation)
12. If v4-A loses: diagnose why (registers not differentiating? S3 not specializing?)

### Longer term:

13. 10B token run on best architecture (v3.2 or v4)
14. Stride-512 activation: does hierarchy solve the sparsity problem?
15. Extraction: freeze compressor, test standalone
16. The portable tensor artifact

## Key files

| Purpose | Path |
|---------|------|
| **v4 design** | `mementum/knowledge/explore/vsm-lm-v4-design.md` |
| **VSM-LM v3.2** | `src/verbum/vsm_lm_v3_2.py` |
| **v3.2 training** | `scripts/run_vsm_v3_2_1B.py` |
| **Probe script (v3.2 support)** | `scripts/compile_gradient_probe.py` |
| **v3.2 checkpoints** | `checkpoints/vsm-lm-v3.2/step_{001000..005000}.pt` |
| **v3.2 compile-gradient results** | `results/compile-gradient/vsm_probe_step_00*_v3.2.json` |
| **v3.2 binding results** | `results/binding/vsm_probe_step_00*_v3.2.json` |
| **Research program** | `mementum/knowledge/explore/VERBUM.md` |
| **v3 best checkpoint** | `checkpoints/vsm-lm-v3/step_010000.pt` |

## Architecture lineage

| Version | Params | Strides | Best Loss | Key Finding |
|---------|--------|---------|-----------|-------------|
| v1 | ~25M | 1,8,64 | 5.245 | Baseline sequential |
| v2 | ~25M | 1,8,64 | 5.064 (1B) | Iteration specialization |
| v3 | 50M | 1,8,64 | 4.872 | Role register, binding confirmed |
| v3.1 | 59M | 1,8,64,512 | 4.836 | Stride 512 too sparse without hierarchy |
| v3.2 | 51M | 1,8,64 | **<4.67** (training) | Convergence arch, cube-mode, probe-grounded |
| v4 | ~51M | 1,8,64,512 | ? (designed) | Recursive VSM, hierarchical registers |

## Probing pipeline

```bash
# Probe a single checkpoint
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v3.2/step_005000.pt

# Binding probes
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v3.2/step_005000.pt --probes probes/binding.json

# Batch all checkpoints in a directory
uv run python scripts/compile_gradient_probe.py batch-probe --dir checkpoints/vsm-lm-v3.2/
```
