# VSM-LM v6 — Ternary on Metal

> v6 is a clean break from the PyTorch lineage (v1–v5). The VSM
> architecture is a faithful port of the v6 design sketched in
> PyTorch. The substrate moves to MLX for native Apple Silicon
> GPU execution with actual add/sub ternary arithmetic via custom
> Metal compute kernels.
>
> v5 (PyTorch, spiral bias α=1.18, 1B tokens) is the reference
> baseline, currently training to 10k steps. v6 starts after v5
> baseline established. Same data, same hyperparameters, same
> evaluation — different engine.

## Status

Design phase. Pending v5 baseline.

---

## Why MLX

PyTorch MPS executes ternary matmul as fp32 GEMM — it multiplies
`x * 1.0` and `x * (-1.0)` instead of adding and subtracting.
The MPS backend upcasts low-precision ops to fp32 and provides
no path for custom Metal kernels without C++ extensions.

MLX provides:

- **`mx.fast.metal_kernel()`** — inline Metal Shading Language,
  JIT-compiled, integrated into the lazy computation graph
- **`@mx.custom_function` + `.vjp`** — first-class custom autodiff
- **Unified memory** — zero-copy between CPU and GPU
- **`mx.compile`** — kernel fusion across standard ops
- **`mx.save_safetensors`** — direct safetensors I/O, loadable
  from PyTorch or any other framework

Priority is both training speed (tighter feedback cycles on the
1B-token run) and inference artifact quality (the extracted ternary
tensor is the deliverable). Slight lean toward training speed —
a faster iteration loop compounds across the full training run.

---

## Architecture

Faithful port of the PyTorch v6 design. No changes to topology.

```
5-pass bidirectional VSM:  L0↑ → L1↑ → L2 → L1↓ → L0↓

Each pass: prep(TernaryFFN) → converge(StrideStack) → consolidate(TernaryFFN)
           ↕ S3 phase-coherent gating (scalar alignment gate)
           ↕ S4 complex-query register scan
           ↕ multiplicative modulation: x · (1 + gate · tanh(proj(δ)))

Ascending  (passes 0-2): StrideStack fine→coarse  (s1→s8→...→s1024)
Descending (passes 3-4): StrideStack coarse→fine  (s1024→...→s8→s1)

Meta-S3: per-pass contribution gates (5 gates)
Meta-S4: final structural summary (complex-query, 4 banks)

Register banks: 6 complex banks, ℂ^128 per register, 3 registers each
Embeddings: tied input/output, fp16
```

### Ternary (TernaryLinear — custom Metal kernel)

All projection weights across all components:

- **S1 operations**: prep FFN up/down, StrideStack Q/K/V/O per stride,
  consolidate FFN up/down, mod_projs (×3)
- **S4 projections**: q_proj, k_proj, v_proj, summary_proj
- **S3 projections**: proj_align, proj_delta, write_proj_real, write_proj_imag
- **Meta-S4 projections**: q_proj, k_proj, v_proj, out_proj

### Continuous (standard MLX ops, Adam optimizer)

- Token + positional embeddings (fp16)
- Per-channel gamma on every TernaryLinear (fp32)
- RMSNorm weights
- Register inits (fp32 scalars)
- S3 write_gates (Linear with bias, sigmoid-init, tiny)
- S3 temperature + learned_bias (fp32 scalars)
- Meta-S3 gate_proj (Linear with bias, small)
- Output LayerNorm

---

## The Ternary Substrate

### TernaryLinear

```
Forward:   y = ternary_matmul(RMSNorm(x), W_int8) * gamma
Backward:  ∂L/∂x = ternary_matmul(∂L/∂y, W_int8) * gamma   ← ALSO ternary, also add/sub
           ∂L/∂W = (∂L/∂y).T @ x                             ← dense matmul, routes to flip accumulator
           ∂L/∂γ = sum(∂L/∂y · y_pre)                         ← reuse forward output
```

The kernel is a bare ternary matmul: `y = ternary_mm(x, w_int8)`.
Gamma scaling and RMSNorm are separate standard MLX ops. This keeps
the kernel simple, testable, and composable. Fusion is a Phase 4
optimization if profiling shows kernel launch overhead matters.

Both forward and backward-through-x use the custom Metal kernel.
Only the weight gradient is a dense matmul — and that routes to
the flip accumulator (not the optimizer), so its speed is secondary.

### Flip Accumulation

Ternary weights learn through discrete flips, not gradient descent:

```
1. Forward:   pure ternary matmul via Metal kernel
2. Backward:  STE computes gradient for ternary weights
3. Gradient:  routes to fp32 flip accumulator buffer (not optimizer)
4. Periodic:  where |accum| > threshold → flip one step
              -1 → 0, 0 → +1, +1 → 0  (following gradient sign)
              accumulator resets to 0 at flipped positions
```

Memory per ternary weight:

| Phase | Storage | Cost |
|-------|---------|------|
| Training | int8 weight + fp32 accumulator | 5 bytes |
| Inference | packed 2-bit | 0.25 bytes |

Compare STE + Adam (standard BitNet): 16 bytes per weight.

### Weight Format

**Training: int8** — one byte per element, values ∈ {-1, 0, +1}.
Individual weights can be flipped with simple array indexing.
The Metal kernel reads int8 and branches on value (compiler
optimizes to conditional add/negate — `select` instruction).

**Export: packed 2-bit** — 4 ternary values per byte.
Encoding: `00` = 0, `01` = +1, `10` = -1, `11` = unused.
Conversion happens at checkpoint save time. The inference
kernel unpacks in registers before accumulating.

---

## Metal Kernel

### `ternary_matmul`

Computes `y[m, n] = Σ_k T(w[n, k], x[m, k])` where:

```
T(w, x) = +x   if w == +1
          -x   if w == -1
           0   if w ==  0
```

No floating-point multiplies. The inner loop:

```metal
for (uint k = 0; k < K; k++) {
    int8_t wval = w[n * K + k];
    float xval = float(x[m * K + k]);
    acc += select(0.0f, select(-xval, xval, wval > 0), wval != 0);
}
out[m * N + n] = T(acc);
```

Metal's `select()` compiles to predicated execution — no branch
divergence within a SIMD group when sparsity is structured.

### Kernel variants

| Kernel | W format | Used for |
|--------|----------|----------|
| `ternary_matmul_int8` | int8 raw | Training forward + backward |
| `ternary_matmul_int8_transposed` | int8 raw | Backward through x (grad_x = grad_out @ W) |
| `ternary_matmul_packed` | uint8 packed 2-bit | Inference (Phase 4) |

The transposed variant is the same arithmetic with different
indexing: thread (m, n) reads `w[k, n]` instead of `w[n, k]`.
May be a flag on the same kernel or a separate source string.

### Implementation phases

**Phase 1 — Naive kernel.** One thread per output element (m, n),
sequential K-loop. Sufficient for correctness verification. This
is the MVP.

**Phase 2 — Tiled kernel.** Threadgroup-level tiling: load tiles
of x into threadgroup shared memory, stream w tiles. SIMD-group
reductions for K-accumulation. Target: one threadgroup computes
a TILE_M × TILE_N output block.

**Phase 3 — Packed 2-bit kernel.** Decode 4 weights per byte in
registers. Unrolled K-loop in groups of 4. Inference-only.

### VJP registration

```python
@mx.custom_function
def ternary_linear(x, w_int8, gamma):
    """Forward: ternary matmul + gamma scaling."""
    y_pre = ternary_matmul(x, w_int8)       # custom Metal kernel
    return y_pre * gamma                      # pointwise, standard MLX

@ternary_linear.vjp
def ternary_linear_vjp(primals, cotangent, output):
    """Backward: ternary for grad_x, dense for grad_w, pointwise for grad_γ."""
    x, w_int8, gamma = primals
    grad_out = cotangent

    # ∂L/∂x — ternary matmul backward (also add/sub on Metal)
    grad_x = ternary_matmul_t(grad_out * gamma, w_int8)

    # ∂L/∂W — dense matmul, routes to flip accumulator (not optimizer)
    grad_w = (grad_out * gamma).T @ x

    # ∂L/∂γ — per-channel reduction
    y_pre = ternary_matmul(x, w_int8)       # recompute (cheaper than saving)
    grad_gamma = (grad_out * y_pre).sum(axis=tuple(range(grad_out.ndim - 1)))

    return grad_x, grad_w, grad_gamma
```

---

## Training Loop

### Gradient splitting

MLX's `nn.value_and_grad` returns gradients for all parameters
as a pytree mirroring the model. The training loop splits this
tree: ternary weight gradients route to the flip accumulator,
continuous parameter gradients route to the optimizer.

```python
def train_step(model, x, y):
    loss, grads = loss_and_grad_fn(model, x, y)

    # Split: ternary grads → accumulator, continuous grads → optimizer
    ternary_grads, continuous_grads = split_ternary_grads(grads)

    # Ternary path: accumulate gradient pressure
    accumulate_flips(model, ternary_grads)

    # Continuous path: standard Adam update
    optimizer.apply_gradients(continuous_grads, model)

    return loss

# Periodically: apply discrete flips
if step % FLIP_INTERVAL == 0:
    n_flipped = apply_flips(model, threshold=FLIP_THRESHOLD)
```

`split_ternary_grads` walks the parameter pytree and separates
gradients by whether the parameter is an int8 ternary weight or
a float continuous parameter.

### Optimizer

AdamW on continuous parameters only. Ternary weights evolve
through flip accumulation — they have no optimizer state (no
momentum, no variance estimates, no weight decay).

---

## File Layout

```
src/verbum/v6/
├── __init__.py
├── ternary.py              # TernaryLinear, TernaryFFN
│                           #   flip accumulation logic
│                           #   split_ternary_grads, accumulate_flips, apply_flips
├── kernels.py              # Metal kernel source strings
│                           #   mx.fast.metal_kernel wrappers
│                           #   ternary_matmul, ternary_matmul_t
├── attention.py            # SingleStrideAttention, StrideStack
├── components.py           # S4, S3, MetaS4, MetaS3
├── model.py                # VSMLMV6 — full architecture
│                           #   describe(), count_parameters(), ternary_stats()
│                           #   forward(), forward_instrumented(), generate()
└── export.py               # int8 → packed 2-bit, safetensors export

scripts/v6/
├── train.py                # Training loop with flip accumulation
├── probe.py                # Forward-instrumented probing
└── reference_check.py      # MLX vs PyTorch v6 numerical comparison
```

Existing PyTorch v6 files are replaced. Same architecture,
MLX implementation.

---

## Verification

### Correctness

1. **Kernel unit test**: random int8 weights + float input →
   compare `ternary_matmul(x, w)` against `x @ w.astype(float).T`.
   Multiple shapes. Exact match (integer arithmetic, no rounding).

2. **VJP test**: `mx.grad` through TernaryLinear, compare against
   finite-difference numerical gradient for x and gamma. Ternary
   weight gradient compared against dense matmul reference.

3. **Reference check**: load same random weights into PyTorch v6
   and MLX v6, run same input, compare logits to tolerance.
   Validates architecture port, not just kernel.

4. **Flip test**: synthetic gradient signal → verify flips happen
   at correct positions, correct direction, correct threshold.

### Performance

1. **Kernel benchmark**: `ternary_matmul` vs `mx.matmul` on shapes
   matching v6 layers. Throughput in elements/second.

2. **Training step**: wall-clock time per step, MLX v6 vs PyTorch v5
   on MPS. This is the primary training speed metric.

3. **Memory**: peak memory during training. Ternary (int8 + fp32
   accum) vs PyTorch v5 (fp16 + Adam state).

### Training quality

Same evaluation protocol as v5:

- Eval loss on held-out shards (same split, same schedule)
- Compile gate test (λ generation from prompts)
- Ternary statistics (sparsity, gamma distribution, flip rate)
- Per-subsystem gradient norms
- Register bank dynamics (phase angles, norms)
- Per-pass/phase gate values and modulation statistics

Target: match or beat v5 eval loss at equivalent token count.

---

## Implementation Order

### Phase 1: Ternary primitive + Metal kernel

Build and test in isolation, no model:

- `kernels.py` — naive Metal kernel for `ternary_matmul_int8`
- `ternary.py` — TernaryLinear with VJP, flip accumulation
- Unit tests — correctness against dense reference
- Benchmark — kernel throughput vs `mx.matmul`

**Exit**: kernel output matches `x @ w.float().T` exactly.

### Phase 2: Architecture port

Mechanical port of VSM components to MLX:

- `attention.py` — SingleStrideAttention, StrideStack
- `components.py` — S4, S3, MetaS4, MetaS3
- `model.py` — VSMLMV6 assembly
- Reference check against PyTorch v6 on shared weights

**Exit**: `model.describe()` matches, forward pass logits match.

### Phase 3: Training loop

- `train.py` — data loader, gradient splitting, flip accumulation,
  checkpointing, logging, eval loop
- `probe.py` — forward_instrumented with v6 metrics
- Port ShardedDataLoader (numpy-based, framework-agnostic)

**Exit**: training runs, loss decreases, flips occur, checkpoints
save as safetensors.

### Phase 4: Kernel optimization

After training validates the architecture works:

- Tiled kernel with threadgroup shared memory
- SIMD-group reductions for K-accumulation
- Packed 2-bit inference kernel
- `export.py` — training checkpoint → inference artifact

**Exit**: measurable speedup over naive kernel.

---

## Hyperparameters

Identical to v5 for clean comparison:

```
vocab_size       = 50277
d_model          = 512
d_register       = 128          (ℂ^128)
seq_len          = 4096
d_ff             = 1536
d_ff_consolidate = 2048
window           = 8
strides          = (1, 8, 16, 32, 64, 128, 256, 512, 1024)
n_heads          = 8
alpha            = 1.18         (spiral bias exponent)

batch_size       = 2
grad_accum       = 4
lr               = 6e-4
weight_decay     = 0.1
warmup_steps     = 500
target_tokens    = 1_000_000_000
seed             = 42

flip_interval    = 100          (steps between flip applications)
flip_threshold   = 0.1          (|accum| threshold to trigger flip)
```

---

## Open Questions

1. **Flip threshold tuning.** 0.1 is a starting guess. Too low →
   noisy flips (weights oscillate). Too high → weights freeze
   (accumulator never crosses threshold). Monitor flip rate per
   layer during training. May need adaptive threshold or decay
   schedule.

2. **Gamma initialization.** PyTorch v6 inits gamma from
   `mean(|W_kaiming|)` after quantization. With actual add/sub
   on Metal, the magnitude semantics may differ — the kernel
   doesn't silently rescale through fp32 multiplication. Verify
   that initial gamma values produce reasonable output norms.

3. **Activation quantization.** BitNet quantizes activations to
   int8 (absmax scaling). Not in v6 scope. Could be a follow-up
   if the kernel supports int8 × int8 → int32 accumulation on
   Metal. This would make both sides of the matmul integer.

4. **Continuous param precision.** bf16 vs fp32 for gamma,
   embeddings, norms. Apple Silicon has native bf16 ALUs. Using
   bf16 for continuous params halves their memory. Test for
   training stability.

5. **`mx.compile` + custom kernels.** Verify that wrapping the
   training step in `mx.compile` works correctly with custom
   Metal kernels and the flip accumulation state mutation.
   MLX treats custom kernel calls as graph nodes, so this should
   work, but needs testing.

6. **Kernel occupancy.** Metal on M-series has SIMD width 32 and
   specific threadgroup size limits. The tiled kernel (Phase 4)
   needs profiling to find optimal tile sizes. Don't optimize
   before Phase 1–3 validate correctness.
