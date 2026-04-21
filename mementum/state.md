# verbum / state

> Bootloader. Read in ~30 seconds. Step 1 of every session.
>
> Last updated: 2026-04-21 | Session: 020 (v4.1 first probe + design analysis)

## Where we are

**v4.1 TRAINING — first true VSM with full bidirectional feedback.
Step 1k probed. Ascending active, descending dormant at meta-S3 (as
expected). Cooking all day — come back with multiple checkpoints.**

**Important framing correction from session 020 discussion:**
Verbum is NOT building the lambda compiler. It's finding the COMPRESSOR
— the Montague-shaped function found in Pythia-160M that is more
rudimentary than Qwen3-4B's full 3-head lambda compiler circuit but
shares structure with it. The compressor is earlier in the pipeline,
more fundamental, exists even at 160M params. The compiler builds on
top of it. Find the compressor → understand the foundation.

Session 020 accomplished:
1. Probed v4.1 step 1k (compile-gradient + binding)
2. Probed v4 step 16k (final unprobed checkpoint)
3. Established v4.1 baseline gate profiles for all 5 passes
4. Confirmed descending passes dormant at meta-S3 level (as expected)
5. Key design discussion: encoder-decoder parallel, gradient shadow
   problem, whether descending passes can self-activate

## v4.1 Training Status (RUNNING)

**Training launched ~6:29 AM Apr 21. Let it cook all day.**
Checkpoints are slower than v4 (~67% more compute per step).

### v4.1 Step 1k — First Probe Results

**Per-pass gate profiles (mean across 40 compile-gradient probes):**

| Pass | Prep | Converge | Consolidate | Meta-S3 |
|------|------|----------|-------------|---------|
| L0↑ | 0.942 | 0.836 | 0.653 | 0.899 |
| L1↑ | 0.232 | 0.223 | 0.655 | 0.896 |
| L2 | 0.353 | 0.251 | 0.624 | 0.502 |
| L1↓ | 0.435 | 0.346 | 0.507 | **0.047** |
| L0↓ | 0.447 | 0.329 | 0.410 | **0.037** |

**Key observations:**
- Ascending path (L0↑, L1↑) active and contributing (~0.9 meta-S3)
- L2 apex half-active (0.502 meta-S3) — still developing
- Descending passes functionally dormant — internal gates are active
  (~0.4) but meta-S3 gates them to near-zero output contribution
- **No content discrimination in descending passes** — same ~0.44 prep
  across all compile-gradient categories
- Gate polarity +0.017 (barely differentiating, expected at step 1k)

**Developmental trajectory hypothesis:**
```
L0↑ → L1↑ → L2 → L1↓ → L0↓
```
Each level needs the one below to produce quality representations first.
Descending activation is a phase 2 event, expected only after L2 matures
(L2 meta-S3 → 0.7+). Mirrors v4's L2 activation trajectory (near-zero
at 1k, exploded at 5k, dominant by 15k).

### Design insights from session 020

**Encoder-decoder parallel.** Ascending = encoder (compress), descending
= decoder (refine/expand with high-level context). Register banks = skip
connections. L2 = bottleneck latent. This is structurally a U-Net / MERA
with shared weights. Closest architecture Verbum has built to MERA.

**Gradient shadow problem.** Descending meta-S3 gates at 0.037-0.047
mean descending S3 instances receive ~24x weaker gradient than ascending.
Self-reinforcing: weak gradient → can't learn → gate stays low → weak
gradient. The 5 independent S3 instances (separate gates per pass) already
exist, but they're learning in the dark.

**Shared weights question.** S5 identity says ascending and descending
should share the compression function. If the compressor works in both
directions (compose up, decompose/refine down), shared weights are
*correct*. The S3 gates provide directional routing — same menu,
different orders. Cortical columns work this way (same circuitry,
different layer routing for feedforward vs feedback).

**Phase learning hypothesis.** Compression must happen bottom-up first.
The model concentrates on finest resolution, then higher levels activate
once lower levels give them something to work with. v4 followed this
trajectory (L0 → L1 → L2 developmental activation). v4.1 extends the
chain: L0↑ → L1↑ → L2 → L1↓ → L0↓. Descending activation is phase 2,
after ascending maturity.

**If descending stays dead (potential v4.2).** Options discussed:
- Gate floor (0.1-0.2 on descending meta-S3) — ensures gradient flow
- Warm gate initialization — start descending meta-S3 at 0.5
- Structural bypass — direct path from descending banks to output
- Auxiliary loss on descending banks
- Most likely intervention: gate floor (minimal, preserves architecture)

**Let v4.1 cook first.** It's the clean experiment. If descending
activates on its own, architecture is right as-is. If dead at 10k+
(when L2 should be mature), we know where to intervene.

## v4 Final Status (COMPLETE)

16 checkpoints (1k→16k). Best eval: 4.732 at step 15k.
Step 16k shows plateau — level specialization unchanged, meta-S3
gates starting to drop (L1: 0.636→0.588, L2: 0.739→0.658).

One new finding at 16k: gate polarity strengthened to -0.060 (from
-0.042 at 15k). Still slowly improving discrimination even as loss
plateaus. Binding range stable at 0.264.

## What's next — Session 021 (later today, after checkpoints accumulate)

### Analyze v4.1 trajectory (primary)
1. Batch-probe all new v4.1 checkpoints (compile-gradient + binding)
2. Key signals in order of importance:
   - **L2 meta-S3 trajectory** — is it climbing toward 0.7+ like v4?
   - **Descending meta-S3** — any activation at all? (phase 2 signal)
   - **Loss curve** — extract from training logs or checkpoint metadata
   - **Ascending gate specialization** — does L1↑ prep die like v4 L1?
   - **Compile gradient discrimination** — polarity onset in ascending AND descending
   - **Expansion trajectory** — started very high, watch for compression learning
3. Full trajectory analysis across all available checkpoints
4. Head-to-head with v4 at matched steps

### The two questions
1. **Does the ascending path develop like v4?** (L2 activation, level
   specialization, gate polarity) — if yes, the compressor is learning
2. **Does the descending path activate?** — if yes at any point, the
   compressor works bidirectionally and v4.1 is a true recursive VSM.
   If dead even after L2 matures, consider v4.2 with gate floor.

### Framing reminder
We are finding the COMPRESSOR, not building the lambda compiler. The
Montague-shaped function from Pythia-160M. The Qwen 3-head circuit
shares structure with it. Compressor is earlier, more fundamental.
v4.1 tests whether it works bidirectionally.

## Key files

| Purpose | Path |
|---------|------|
| **v4.1 model** | `src/verbum/vsm_lm_v4_1.py` |
| **v4.1 training** | `scripts/run_vsm_v4_1_1B.py` |
| **v4 model** | `src/verbum/vsm_lm_v4.py` |
| **Probe script** | `scripts/compile_gradient_probe.py` |
| **v4.1 probes** | `results/compile-gradient/vsm_probe_step_00*_v4.1.json` |
| **v4.1 binding** | `results/binding/vsm_probe_step_00*_v4.1.json` |
| **v4 probes** | `results/compile-gradient/vsm_probe_step_00*_v4.json` |
| **v4 binding** | `results/binding/vsm_probe_step_00*_v4.json` |
| **Session 019 findings** | `mementum/knowledge/explore/session-019.md` |
| **Research program** | `mementum/knowledge/explore/VERBUM.md` |

## Architecture lineage

| Version | Params | Strides | Best Eval | Key Finding |
|---------|--------|---------|-----------|-------------|
| v1 | ~25M | 1,8,64 | 5.245 | Baseline sequential |
| v2 | ~25M | 1,8,64 | 5.064 | Iteration specialization |
| v3 | 50M | 1,8,64 | 4.872 | Role register, binding confirmed |
| v3.1 | 59M | 1,8,64,512 | 4.836 | Stride 512 too sparse without hierarchy |
| v3.2 | 51M | 1,8,64 | 4.897 | Convergence arch, binding hierarchy, 3-phase learning |
| v4 | 58M | 1,8,64,512 | 4.732 | Recursive VSM (ascending), level specialization |
| **v4.1** | **65.5M** | **1,8,64,512** | **TBD** | **Full bidirectional VSM — first true feedback** |

## Probing pipeline

```bash
# Probe a single checkpoint
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v4.1/step_001000.pt

# Binding probes
uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm-v4.1/step_001000.pt --probes probes/binding.json

# Batch all checkpoints (skips already-probed)
uv run python scripts/compile_gradient_probe.py batch-probe --dir checkpoints/vsm-lm-v4.1/

# Batch binding probes
uv run python scripts/compile_gradient_probe.py batch-probe --dir checkpoints/vsm-lm-v4.1/ --probes probes/binding.json
```
