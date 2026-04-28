"""Ternary substrate for v8's hot-path components.

Self-contained — no imports from v6. Adapted from:
  - src/verbum/v6/kernels.py  (Metal kernel sources and wrappers)
  - src/verbum/v6/ternary.py  (TernaryLinear, pack/unpack, flip accumulation)

The ternary weights {-1, 0, +1} define routing topology. They evolve
during training through a lightweight accumulate-and-flip mechanism:

  1. Forward: ternary matmul via custom Metal kernel (add/sub only)
  2. Backward: STE computes gradient for ternary weights
  3. Gradient routes to a flip accumulator (not to the optimizer)
  4. Periodically: weights whose accumulator exceeds threshold FLIP
     one step (-1→0, 0→+1, +1→0, etc.) and ALL accumulators reset

Per-channel gamma provides continuous fine-tuning on top of the
discrete ternary routing. Gamma is trained normally with Adam.

Memory per ternary weight:
  Training:  1 byte (int8) + 4 bytes (fp32 accumulator) = 5 bytes
  Inference: 0.25 bytes (packed 2-bit)

License: MIT
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn


# ══════════════════════════════════════════════════════════════════════
# Metal Shading Language source — Phase 1 (naive)
# ══════════════════════════════════════════════════════════════════════

# Forward kernel: y[m, n] = Σ_k T(w[n, k], x[m, k])
#
# x:   (M, K) float16/float32, row-contiguous
# w:   (N, K) int8, values in {-1, 0, +1}, row-contiguous
# out: (M, N) same dtype as x
#
# M, N, K passed as integer template constants.
# Grid: (N, M, 1) — one thread per output element.
# Thread (n, m) computes out[m, n].

TERNARY_MATMUL_SOURCE = """
    uint n = thread_position_in_grid.x;
    uint m = thread_position_in_grid.y;

    if (m >= M || n >= N) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        int8_t wval = w[n * K + k];
        float xval = static_cast<float>(x[m * K + k]);
        acc += select(0.0f, select(-xval, xval, wval > 0), wval != 0);
    }

    out[m * N + n] = static_cast<T>(acc);
"""

# Transposed kernel: y[m, k] = Σ_n T(w[n, k], x[m, n])
#
# Used for backward through x: grad_x = grad_out @ W
# where W is (N, K) and grad_out is (M, N), so:
#   grad_x[m, k] = Σ_n grad_out[m, n] * W[n, k]
#                = Σ_n T(W[n, k], grad_out[m, n])
#
# x:   (M, N) float — this is grad_out in the backward context
# w:   (N, K) int8 — same weight matrix, but accessed as w[n, k]
# out: (M, K) float
#
# Grid: (K, M, 1) — one thread per output element.
# Thread (k, m) computes out[m, k].

TERNARY_MATMUL_T_SOURCE = """
    uint k = thread_position_in_grid.x;
    uint m = thread_position_in_grid.y;

    if (m >= M || k >= K) return;

    float acc = 0.0f;
    for (uint n = 0; n < N; n++) {
        int8_t wval = w[n * K + k];
        float xval = static_cast<float>(x[m * N + n]);
        acc += select(0.0f, select(-xval, xval, wval > 0), wval != 0);
    }

    out[m * K + k] = static_cast<T>(acc);
"""


# ══════════════════════════════════════════════════════════════════════
# Metal Shading Language source — Phase 1 (packed, 4 weights per byte)
# ══════════════════════════════════════════════════════════════════════

# Forward packed kernel: y[m, n] = Σ_k T(w_packed[n, k/4], x[m, k])
#
# x:        (M, K) float — row-contiguous activations
# w:        (N, K/4) uint8 — packed weights, 4 per byte
# out:      (M, N) float
# K:        logical weight dimension (must be divisible by 4)
#
# Encoding: -1→0b00, 0→0b01, +1→0b10. Decode: ((bits >> shift) & 0x3) - 1
# Bit positions for columns {4k, 4k+1, 4k+2, 4k+3}: shifts {6, 4, 2, 0}
#
# Grid: (N, M, 1) — one thread per output element.
# Thread (n, m) computes out[m, n].

TERNARY_MATMUL_PACKED_SOURCE = """
    uint n = thread_position_in_grid.x;
    uint m = thread_position_in_grid.y;

    if (m >= M || n >= N) return;

    float acc = 0.0f;
    uint K4 = K / 4;
    for (uint k4 = 0; k4 < K4; k4++) {
        uint8_t packed = w[n * K4 + k4];
        uint base_k = k4 * 4;

        int wval;
        float xval;

        wval = int((packed >> 6) & 0x3) - 1;
        xval = static_cast<float>(x[m * K + base_k]);
        acc += select(0.0f, select(-xval, xval, wval > 0), wval != 0);

        wval = int((packed >> 4) & 0x3) - 1;
        xval = static_cast<float>(x[m * K + base_k + 1]);
        acc += select(0.0f, select(-xval, xval, wval > 0), wval != 0);

        wval = int((packed >> 2) & 0x3) - 1;
        xval = static_cast<float>(x[m * K + base_k + 2]);
        acc += select(0.0f, select(-xval, xval, wval > 0), wval != 0);

        wval = int(packed & 0x3) - 1;
        xval = static_cast<float>(x[m * K + base_k + 3]);
        acc += select(0.0f, select(-xval, xval, wval > 0), wval != 0);
    }

    out[m * N + n] = static_cast<T>(acc);
"""

# Transposed packed kernel: y[m, k] = Σ_n T(w_packed[n, k/4], x[m, n])
#
# Used for backward through x: grad_x = grad_out @ W (W transposed access)
# x:   (M, N) float — grad_out in backward context
# w:   (N, K/4) uint8 — packed weights
# out: (M, K) float
# K:   logical weight dimension
#
# For each k, the relevant packed byte is w[n * K4 + k/4],
# and the shift for bit position k within its byte is (3 - (k & 3)) * 2.
#
# Grid: (K, M, 1) — one thread per output element.
# Thread (k, m) computes out[m, k].

TERNARY_MATMUL_T_PACKED_SOURCE = """
    uint k = thread_position_in_grid.x;
    uint m = thread_position_in_grid.y;

    if (m >= M || k >= K) return;

    float acc = 0.0f;
    uint K4 = K / 4;
    uint k4 = k / 4;
    uint k_shift = (3 - (k & 3)) * 2;

    for (uint n = 0; n < N; n++) {
        uint8_t packed = w[n * K4 + k4];
        int wval = int((packed >> k_shift) & 0x3) - 1;
        float xval = static_cast<float>(x[m * N + n]);
        acc += select(0.0f, select(-xval, xval, wval > 0), wval != 0);
    }

    out[m * K + k] = static_cast<T>(acc);
"""


# ══════════════════════════════════════════════════════════════════════
# Metal Shading Language source — Phase 2 (optimized tiled + SIMD)
# ══════════════════════════════════════════════════════════════════════

# Optimized forward kernel: y[m, n] = Σ_k T(w_packed[n, k/4], x[m, k])
#
# Strategy: Tiled matmul with threadgroup shared memory + simd_sum reduction.
#
# Each threadgroup computes a TILE_M × TILE_N tile of the output.
# The K dimension is reduced cooperatively: threads in a threadgroup each
# handle a slice of K, accumulate locally, then reduce via simd_sum.
#
# Threadgroup layout: (TILE_N, TILE_M, 1)
#   thread (tn, tm) computes out[m_base + tm, n_base + tn]
#
# K-reduction: each thread loops over K in steps of 4 (one packed byte),
# processing 16 weights per iteration (4 bytes × 4 weights/byte) via unrolling.
# The full K is processed by each thread — no K-splitting needed when the
# threadgroup owns complete output elements.
#
# Shared memory tiles of x allow coalesced loading and reuse across the
# N-dimension within a threadgroup.
#
# Template: T (output dtype), M, N, K, TILE_M, TILE_N

TERNARY_MATMUL_PACKED_TILED_HEADER = ""

# Strategy: SIMD-group K-reduction + output tiling.
#
# Each SIMD group (32 threads) cooperates on ONE output element.
# The 32 threads split K evenly: each handles K/32 elements.
# After accumulation, simd_sum reduces across the SIMD group → one result.
#
# Multiple SIMD groups per threadgroup compute different output elements.
# Threadgroup layout: (32, ROWS_PER_TG, 1) where 32 = SIMD width
# Each row of threads = one SIMD group = one output element
#
# Grid: (ceil(N/1) * 32, ceil(M/ROWS_PER_TG) * ROWS_PER_TG, 1)
# Each threadgroup produces ROWS_PER_TG output elements (different n values, same m)
#
# Wait — that's wrong for a 2D output. Let me think again.
#
# Actually: grid over (n, m) output elements.
# Each output element gets 32 threads (one SIMD group) to reduce K.
# Threadgroup: (32, ROWS, 1) → ROWS output elements per threadgroup, each with 32-wide K split.
#
# Thread (lane, row) within threadgroup:
#   m = threadgroup_m_base + some_mapping
#   n = threadgroup_n_base + row
#   This thread reduces K range: [lane * K_per_thread, (lane+1) * K_per_thread)
#
# K=1024 / 32 = 32 elements/thread = 8 packed bytes/thread → very manageable

TERNARY_MATMUL_PACKED_TILED_SOURCE = """
    // SIMD-group K-reduction kernel
    // 32 threads cooperate on one output element via simd_sum
    //
    // Threadgroup layout: (32, ROWS_PER_TG, 1)
    //   x-dim (0..31) = SIMD lane = K-slice index
    //   y-dim (0..ROWS-1) = which output element within this threadgroup

    uint lane = thread_position_in_threadgroup.x;   // 0..31 (SIMD lane)
    uint row = thread_position_in_threadgroup.y;     // which output in this TG

    // Map threadgroup to (n, m) output space
    // Grid x: over N dimension, Grid y: over M dimension
    uint n = threadgroup_position_in_grid.x * ROWS_PER_TG + row;
    uint m = threadgroup_position_in_grid.y;

    if (m >= M || n >= N) return;

    uint K4 = K / 4;

    // Each SIMD lane handles a slice of K
    // K_per_lane packed bytes = K4 / 32 (assumes K4 >= 32)
    // For K=1024: K4=256, K4_per_lane=8 → 32 weights per lane
    // For K=4096: K4=1024, K4_per_lane=32 → 128 weights per lane
    uint k4_per_lane = K4 / 32;
    uint k4_start = lane * k4_per_lane;
    uint k4_end = k4_start + k4_per_lane;

    const device uint8_t* w_row = w + n * K4;
    const device T* x_row = x + m * K;

    float acc = 0.0f;

    // Each lane processes its K-slice with 4-byte unrolled loop
    uint k4 = k4_start;
    for (; k4 + 3 < k4_end; k4 += 4) {
        uint8_t p0 = w_row[k4];
        uint8_t p1 = w_row[k4 + 1];
        uint8_t p2 = w_row[k4 + 2];
        uint8_t p3 = w_row[k4 + 3];
        uint base = k4 * 4;

        int wv; float xv;

        wv = int((p0 >> 6) & 0x3) - 1; xv = float(x_row[base   ]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
        wv = int((p0 >> 4) & 0x3) - 1; xv = float(x_row[base+ 1]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
        wv = int((p0 >> 2) & 0x3) - 1; xv = float(x_row[base+ 2]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
        wv = int(p0 & 0x3) - 1;        xv = float(x_row[base+ 3]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);

        wv = int((p1 >> 6) & 0x3) - 1; xv = float(x_row[base+ 4]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
        wv = int((p1 >> 4) & 0x3) - 1; xv = float(x_row[base+ 5]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
        wv = int((p1 >> 2) & 0x3) - 1; xv = float(x_row[base+ 6]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
        wv = int(p1 & 0x3) - 1;        xv = float(x_row[base+ 7]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);

        wv = int((p2 >> 6) & 0x3) - 1; xv = float(x_row[base+ 8]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
        wv = int((p2 >> 4) & 0x3) - 1; xv = float(x_row[base+ 9]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
        wv = int((p2 >> 2) & 0x3) - 1; xv = float(x_row[base+10]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
        wv = int(p2 & 0x3) - 1;        xv = float(x_row[base+11]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);

        wv = int((p3 >> 6) & 0x3) - 1; xv = float(x_row[base+12]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
        wv = int((p3 >> 4) & 0x3) - 1; xv = float(x_row[base+13]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
        wv = int((p3 >> 2) & 0x3) - 1; xv = float(x_row[base+14]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
        wv = int(p3 & 0x3) - 1;        xv = float(x_row[base+15]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
    }
    // Remainder
    for (; k4 < k4_end; k4++) {
        uint8_t p = w_row[k4];
        uint base = k4 * 4;
        int wv; float xv;
        wv = int((p >> 6) & 0x3) - 1; xv = float(x_row[base  ]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
        wv = int((p >> 4) & 0x3) - 1; xv = float(x_row[base+1]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
        wv = int((p >> 2) & 0x3) - 1; xv = float(x_row[base+2]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
        wv = int(p & 0x3) - 1;        xv = float(x_row[base+3]); acc += select(0.0f, select(-xv, xv, wv > 0), wv != 0);
    }

    // Reduce across SIMD group — one hardware instruction
    float result = simd_sum(acc);

    // Lane 0 writes the final result
    if (lane == 0) {
        out[m * N + n] = static_cast<T>(result);
    }
"""

# Optimized transposed kernel: y[m, k] = Σ_n T(w_packed[n, k/4], x[m, n])
#
# The transpose kernel is harder to optimize because the reduction is over N
# and weight access pattern is strided (each thread needs one 2-bit field from
# each row's packed byte). Strategy:
#
# Each threadgroup tile: TILE_M × TILE_K of the output.
# For each n, load the packed byte w[n, k/4] and the activation x[m, n].
# The key optimization: group 4 adjacent k values that share the same packed byte,
# so one byte load serves 4 output elements.
#
# Shared memory: tile of x[TILE_M, N_CHUNK] to reuse across the K dimension.
# N is reduced in chunks to limit shared memory usage.

TERNARY_MATMUL_T_PACKED_TILED_HEADER = ""

TERNARY_MATMUL_T_PACKED_TILED_SOURCE = """
    // Thread coordinates
    uint tk = thread_position_in_threadgroup.x;  // k within tile
    uint tm = thread_position_in_threadgroup.y;  // m within tile

    // Global output coordinates
    uint k = threadgroup_position_in_grid.x * TILE_K + tk;
    uint m = threadgroup_position_in_grid.y * TILE_M + tm;

    if (m >= M || k >= K) return;

    uint K4 = K / 4;
    uint k4 = k / 4;
    uint k_in_byte = k & 3;
    uint k_shift = (3 - k_in_byte) * 2;

    // Accumulate over the full N dimension
    // Unroll by 4 for ILP — each iteration loads 4 packed bytes and 4 x values
    float acc = 0.0f;
    uint n = 0;
    for (; n + 3 < N; n += 4) {
        float xv0 = static_cast<float>(x[m * N + n]);
        float xv1 = static_cast<float>(x[m * N + n + 1]);
        float xv2 = static_cast<float>(x[m * N + n + 2]);
        float xv3 = static_cast<float>(x[m * N + n + 3]);

        int w0 = int((w[(n)     * K4 + k4] >> k_shift) & 0x3) - 1;
        int w1 = int((w[(n + 1) * K4 + k4] >> k_shift) & 0x3) - 1;
        int w2 = int((w[(n + 2) * K4 + k4] >> k_shift) & 0x3) - 1;
        int w3 = int((w[(n + 3) * K4 + k4] >> k_shift) & 0x3) - 1;

        acc += select(0.0f, select(-xv0, xv0, w0 > 0), w0 != 0);
        acc += select(0.0f, select(-xv1, xv1, w1 > 0), w1 != 0);
        acc += select(0.0f, select(-xv2, xv2, w2 > 0), w2 != 0);
        acc += select(0.0f, select(-xv3, xv3, w3 > 0), w3 != 0);
    }
    // Remainder
    for (; n < N; n++) {
        float xval = static_cast<float>(x[m * N + n]);
        int wval = int((w[n * K4 + k4] >> k_shift) & 0x3) - 1;
        acc += select(0.0f, select(-xval, xval, wval > 0), wval != 0);
    }

    out[m * K + k] = static_cast<T>(acc);
"""


# ══════════════════════════════════════════════════════════════════════
# Kernel wrappers
# ══════════════════════════════════════════════════════════════════════

_ternary_matmul_kernel = mx.fast.metal_kernel(
    name="ternary_matmul",
    input_names=["x", "w"],
    output_names=["out"],
    source=TERNARY_MATMUL_SOURCE,
)

_ternary_matmul_t_kernel = mx.fast.metal_kernel(
    name="ternary_matmul_t",
    input_names=["x", "w"],
    output_names=["out"],
    source=TERNARY_MATMUL_T_SOURCE,
)

_ternary_matmul_packed_kernel = mx.fast.metal_kernel(
    name="ternary_matmul_packed",
    input_names=["x", "w"],
    output_names=["out"],
    source=TERNARY_MATMUL_PACKED_SOURCE,
)

_ternary_matmul_t_packed_kernel = mx.fast.metal_kernel(
    name="ternary_matmul_t_packed",
    input_names=["x", "w"],
    output_names=["out"],
    source=TERNARY_MATMUL_T_PACKED_SOURCE,
)

# Optimized tiled kernels
_ternary_matmul_packed_tiled_kernel = mx.fast.metal_kernel(
    name="ternary_matmul_packed_tiled",
    input_names=["x", "w"],
    output_names=["out"],
    source=TERNARY_MATMUL_PACKED_TILED_SOURCE,
    header=TERNARY_MATMUL_PACKED_TILED_HEADER,
)

_ternary_matmul_t_packed_tiled_kernel = mx.fast.metal_kernel(
    name="ternary_matmul_t_packed_tiled",
    input_names=["x", "w"],
    output_names=["out"],
    source=TERNARY_MATMUL_T_PACKED_TILED_SOURCE,
    header=TERNARY_MATMUL_T_PACKED_TILED_HEADER,
)


def ternary_matmul(x: mx.array, w: mx.array) -> mx.array:
    """Ternary matrix multiplication: y = x @ w.T

    Args:
        x: (M, K) or (*, M, K) float array — input activations
        w: (N, K) int8 array — ternary weights {-1, 0, +1}

    Returns:
        (M, N) or (*, M, N) float array — output activations
    """
    orig_shape = x.shape
    if x.ndim == 1:
        x_2d = x.reshape(1, -1)
    elif x.ndim > 2:
        x_2d = x.reshape(-1, orig_shape[-1])
    else:
        x_2d = x

    M, K = x_2d.shape
    N = w.shape[0]
    assert w.shape[1] == K, f"Weight K={w.shape[1]} != input K={K}"
    assert w.dtype == mx.int8, f"Weight dtype must be int8, got {w.dtype}"

    out = _ternary_matmul_kernel(
        inputs=[x_2d, w],
        output_shapes=[(M, N)],
        output_dtypes=[x_2d.dtype],
        grid=(N, M, 1),
        threadgroup=(min(N, 256), 1, 1),
        template=[("T", x_2d.dtype), ("M", M), ("N", N), ("K", K)],
        init_value=0,
        verbose=False,
    )

    result = out[0]

    if x.ndim == 1:
        result = result.reshape(N)
    elif x.ndim > 2:
        result = result.reshape(*orig_shape[:-1], N)

    return result


def ternary_matmul_t(x: mx.array, w: mx.array) -> mx.array:
    """Transposed ternary matmul: y = x @ w (not w.T)

    Computes y[m, k] = Σ_n x[m, n] * w[n, k]
    Used for backward through x: grad_x = grad_out @ W

    Args:
        x: (M, N) or (*, M, N) float array — e.g. grad_output
        w: (N, K) int8 array — ternary weights {-1, 0, +1}

    Returns:
        (M, K) or (*, M, K) float array
    """
    orig_shape = x.shape
    if x.ndim == 1:
        x_2d = x.reshape(1, -1)
    elif x.ndim > 2:
        x_2d = x.reshape(-1, orig_shape[-1])
    else:
        x_2d = x

    M, N_in = x_2d.shape
    N, K = w.shape
    assert N_in == N, f"Input N={N_in} != weight N={N}"
    assert w.dtype == mx.int8, f"Weight dtype must be int8, got {w.dtype}"

    out = _ternary_matmul_t_kernel(
        inputs=[x_2d, w],
        output_shapes=[(M, K)],
        output_dtypes=[x_2d.dtype],
        grid=(K, M, 1),
        threadgroup=(min(K, 256), 1, 1),
        template=[("T", x_2d.dtype), ("M", M), ("N", N), ("K", K)],
        init_value=0,
        verbose=False,
    )

    result = out[0]

    if x.ndim == 1:
        result = result.reshape(K)
    elif x.ndim > 2:
        result = result.reshape(*orig_shape[:-1], K)

    return result


def ternary_matmul_packed(x: mx.array, w_packed: mx.array, K: int) -> mx.array:
    """Ternary matrix multiplication with 2-bit packed weights: y = x @ w.T

    Uses optimized tiled kernel with 4× unrolled decode for throughput.

    Args:
        x:        (M, K) or (*, M, K) float array — input activations
        w_packed: (N, K//4) uint8 array — packed ternary weights
        K:        logical weight dimension (w_packed.shape[1] * 4)

    Returns:
        (M, N) or (*, M, N) float array — output activations
    """
    orig_shape = x.shape
    if x.ndim == 1:
        x_2d = x.reshape(1, -1)
    elif x.ndim > 2:
        x_2d = x.reshape(-1, orig_shape[-1])
    else:
        x_2d = x

    M, K_in = x_2d.shape
    N = w_packed.shape[0]
    assert K_in == K, f"Input K={K_in} != logical K={K}"
    assert w_packed.shape[1] == K // 4, f"Packed cols={w_packed.shape[1]} != K//4={K//4}"
    assert w_packed.dtype == mx.uint8, f"Packed weight dtype must be uint8, got {w_packed.dtype}"

    # Adaptive kernel selection:
    # Small M (≤64): use SIMD-group K-reduction (32 threads/output element via simd_sum)
    # Large M (>64): use naive packed kernel (one thread/output element, full K loop)
    #
    # SIMD kernel excels when output parallelism is insufficient to fill GPU.
    # Naive kernel excels when M×N is large enough to saturate all GPU cores.
    use_simd = (M <= 64)

    if use_simd:
        ROWS_PER_TG = min(N, 8)  # output n-values per threadgroup
        n_groups = (N + ROWS_PER_TG - 1) // ROWS_PER_TG
        out = _ternary_matmul_packed_tiled_kernel(
            inputs=[x_2d, w_packed],
            output_shapes=[(M, N)],
            output_dtypes=[x_2d.dtype],
            grid=(n_groups * 32, M * ROWS_PER_TG, 1),
            threadgroup=(32, ROWS_PER_TG, 1),
            template=[("T", x_2d.dtype), ("M", M), ("N", N), ("K", K),
                      ("ROWS_PER_TG", ROWS_PER_TG)],
            init_value=0,
            verbose=False,
        )
    else:
        out = _ternary_matmul_packed_kernel(
            inputs=[x_2d, w_packed],
            output_shapes=[(M, N)],
            output_dtypes=[x_2d.dtype],
            grid=(N, M, 1),
            threadgroup=(min(N, 256), 1, 1),
            template=[("T", x_2d.dtype), ("M", M), ("N", N), ("K", K)],
            init_value=0,
            verbose=False,
        )

    result = out[0]

    if x.ndim == 1:
        result = result.reshape(N)
    elif x.ndim > 2:
        result = result.reshape(*orig_shape[:-1], N)

    return result


def ternary_matmul_t_packed(x: mx.array, w_packed: mx.array, K: int) -> mx.array:
    """Transposed ternary matmul with packed weights: y = x @ w (not w.T)

    Uses optimized tiled kernel with 4× unrolled N reduction.

    Computes y[m, k] = Σ_n x[m, n] * w[n, k]
    Used for backward through x: grad_x = grad_out @ W

    Args:
        x:        (M, N) or (*, M, N) float array — e.g. grad_output
        w_packed: (N, K//4) uint8 array — packed ternary weights
        K:        logical weight dimension (w_packed.shape[1] * 4)

    Returns:
        (M, K) or (*, M, K) float array
    """
    orig_shape = x.shape
    if x.ndim == 1:
        x_2d = x.reshape(1, -1)
    elif x.ndim > 2:
        x_2d = x.reshape(-1, orig_shape[-1])
    else:
        x_2d = x

    M, N_in = x_2d.shape
    N = w_packed.shape[0]
    assert N_in == N, f"Input N={N_in} != weight N={N}"
    assert w_packed.shape[1] == K // 4, f"Packed cols={w_packed.shape[1]} != K//4={K//4}"
    assert w_packed.dtype == mx.uint8, f"Packed weight dtype must be uint8, got {w_packed.dtype}"

    # Use the tiled transpose kernel with N-unrolled inner loop
    TILE_K = min(K, 16)
    TILE_M = min(M, 16)

    grid_k = (K + TILE_K - 1) // TILE_K
    grid_m = (M + TILE_M - 1) // TILE_M

    out = _ternary_matmul_t_packed_tiled_kernel(
        inputs=[x_2d, w_packed],
        output_shapes=[(M, K)],
        output_dtypes=[x_2d.dtype],
        grid=(grid_k * TILE_K, grid_m * TILE_M, 1),
        threadgroup=(TILE_K, TILE_M, 1),
        template=[("T", x_2d.dtype), ("M", M), ("N", N), ("K", K),
                  ("TILE_M", TILE_M), ("TILE_K", TILE_K)],
        init_value=0,
        verbose=False,
    )

    result = out[0]

    if x.ndim == 1:
        result = result.reshape(K)
    elif x.ndim > 2:
        result = result.reshape(*orig_shape[:-1], K)

    return result


# ══════════════════════════════════════════════════════════════════════
# Pack / unpack utilities
# ══════════════════════════════════════════════════════════════════════


def pack_ternary(w: mx.array) -> mx.array:
    """Pack int8 {-1, 0, +1} weights [N, K] → uint8 [N, K//4].

    Encoding:  -1 → 0b00, 0 → 0b01, +1 → 0b10   (0b11 unused)
    Positions: bits {7:6, 5:4, 3:2, 1:0} for columns {4k, 4k+1, 4k+2, 4k+3}
    Decode:    ((packed >> shift) & 0x3) - 1

    K must be divisible by 4.
    """
    assert w.shape[-1] % 4 == 0, f"K={w.shape[-1]} must be divisible by 4"
    # Shift from {-1,0,+1} to {0,1,2} then cast to uint8
    w_shifted = (w.astype(mx.int16) + 1).astype(mx.uint8)
    packed = (
        (w_shifted[:, 0::4] << 6) |
        (w_shifted[:, 1::4] << 4) |
        (w_shifted[:, 2::4] << 2) |
        w_shifted[:, 3::4]
    )
    return packed.astype(mx.uint8)


def unpack_ternary(packed: mx.array, K: int) -> mx.array:
    """Unpack uint8 [N, K//4] → int8 {-1, 0, +1} [N, K].

    Inverse of pack_ternary. K is the logical (unpacked) weight dimension.
    """
    # Extract each of the 4 sub-columns and decode: ((bits >> shift) & 0x3) - 1
    w0 = ((packed >> 6) & 0x3).astype(mx.int16) - 1  # column 4k
    w1 = ((packed >> 4) & 0x3).astype(mx.int16) - 1  # column 4k+1
    w2 = ((packed >> 2) & 0x3).astype(mx.int16) - 1  # column 4k+2
    w3 = (packed & 0x3).astype(mx.int16) - 1          # column 4k+3
    # Stack along a new trailing axis → [N, K//4, 4] then reshape → [N, K]
    N = packed.shape[0]
    stacked = mx.stack([w0, w1, w2, w3], axis=-1)  # [N, K//4, 4]
    return stacked.reshape(N, K).astype(mx.int8)


# ══════════════════════════════════════════════════════════════════════
# Ternary initialization
# ══════════════════════════════════════════════════════════════════════


def _ternary_init(out_features: int, in_features: int) -> tuple[mx.array, mx.array]:
    """Initialize ternary weights from Kaiming normal → quantize → pack.

    Returns:
        w_packed: (out_features, in_features//4) uint8 packed ternary weights
        gamma:    (out_features,) float32 per-channel scale
    """
    assert in_features % 4 == 0, f"in_features={in_features} must be divisible by 4 for packing"
    # Kaiming normal: std = sqrt(2 / in_features)
    std = math.sqrt(2.0 / in_features)
    w_init = mx.random.normal((out_features, in_features)) * std

    # Per-channel absmean quantization
    gamma = mx.abs(w_init).mean(axis=-1)
    w_scaled = w_init / (mx.expand_dims(gamma, axis=-1) + 1e-8)
    w_q = mx.clip(mx.round(w_scaled), -1, 1).astype(mx.int8)

    # Pack 4 weights per byte: int8 [N, K] → uint8 [N, K//4]
    w_packed = pack_ternary(w_q)

    return w_packed, gamma


# ══════════════════════════════════════════════════════════════════════
# Ternary forward with custom VJP
# ══════════════════════════════════════════════════════════════════════


@mx.custom_function
def _ternary_linear_fwd(x: mx.array, w_packed: mx.array, gamma: mx.array) -> mx.array:
    """Forward: y = ternary_matmul_packed(x, w_packed, K) * gamma

    Packed Metal kernel unpacks 4 weights per byte on-the-fly, doing
    add/sub only — no fp32 multiplies in the matmul. Gamma scaling is
    a cheap pointwise multiply.

    w_packed shape: [N, K//4] uint8. K recovered as w_packed.shape[1] * 4.
    """
    K = w_packed.shape[1] * 4
    y_pre = ternary_matmul_packed(x, w_packed, K)
    return y_pre * gamma


@_ternary_linear_fwd.vjp
def _ternary_linear_vjp(primals, cotangent, output):
    """Backward: STE for ternary weights, packed ternary matmul for grad_x.

    ∂L/∂x:     ternary_matmul_t_packed(grad_out * gamma, w_packed, K)  — packed Metal kernel
    ∂L/∂w:     (grad_out * gamma).T @ x  — dense matmul → flip accumulator (unchanged)
    ∂L/∂gamma: sum(grad_out * y_pre, reduce_dims)  — per-channel (recomputed)

    NOTE: grad_w is still dense float32 [N, K] — the flip accumulator is
    not packed. Only ternary_weight itself is stored packed.
    """
    x, w_packed, gamma = primals
    grad_out = cotangent
    K = w_packed.shape[1] * 4

    # Scale grad_out by gamma once (used for both grad_x and grad_w)
    grad_scaled = grad_out * gamma

    # ∂L/∂x — packed ternary matmul backward (add/sub on Metal)
    grad_x = ternary_matmul_t_packed(grad_scaled, w_packed, K)

    # ∂L/∂w — dense matmul for flip accumulator (does NOT use w at all)
    # Reshape to 2D for matmul: (*, N) x (*, K) → (N, K)
    gs_2d = grad_scaled.reshape(-1, grad_scaled.shape[-1])
    x_2d = x.reshape(-1, x.shape[-1])
    grad_w = gs_2d.T @ x_2d

    # ∂L/∂gamma — per-channel: recompute y_pre with packed kernel
    y_pre = ternary_matmul_packed(x, w_packed, K)
    # Sum over all dims except last (output features)
    reduce_axes = tuple(range(grad_out.ndim - 1))
    grad_gamma = (grad_out * y_pre).sum(axis=reduce_axes)

    return grad_x, grad_w, grad_gamma


# ══════════════════════════════════════════════════════════════════════
# TernaryLinear — nn.Module with flip accumulation
# ══════════════════════════════════════════════════════════════════════


class TernaryLinear(nn.Module):
    """Linear layer with learnable ternary routing via flip accumulation.

    Forward: y = ternary_matmul(RMSNorm(x), W_int8) * gamma

    The ternary weights evolve through discrete flips, not continuous
    gradient descent. Each flip moves one step: -1→0, 0→±1, ±1→0.
    The accumulator captures gradient pressure; the threshold controls
    how much evidence is needed before committing to a flip.

    Args:
        in_features:  input dimension
        out_features: output dimension
        pre_norm:     if True, apply RMSNorm before projection
    """

    def __init__(self, in_features: int, out_features: int, pre_norm: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pre_norm = pre_norm

        if pre_norm:
            self.norm = nn.RMSNorm(in_features)

        # Initialize: Kaiming → quantize → pack into uint8
        # ternary_weight: [out_features, in_features//4] uint8  (4× memory reduction)
        w_packed, gamma = _ternary_init(out_features, in_features)
        self.ternary_weight = w_packed
        self.gamma = gamma

        # Flip accumulator — tracks gradient pressure per weight within
        # one flip interval. Reset to zero after every flip check (not
        # just for flipped weights) so each interval asks a fresh question:
        # "given current topology, which weights want to flip NOW?"
        # Int8 with saturation at ±127. Each micro-batch votes ±1.
        self._flip_accum = mx.zeros((out_features, in_features), dtype=mx.int8)

        # Cooldown: remaining flip intervals before this weight can flip again.
        # Prevents oscillation. Decremented each flip check; weight is blocked
        # from flipping while cooldown > 0.
        self._flip_cooldown = mx.zeros((out_features, in_features), dtype=mx.int8)

        # Last direction: direction of the most recent flip for this weight.
        # +1 = last flip was upward, -1 = downward, 0 = never flipped.
        self._flip_last_dir = mx.zeros((out_features, in_features), dtype=mx.int8)

    def __call__(self, x: mx.array) -> mx.array:
        if self.pre_norm:
            x = self.norm(x)
        return _ternary_linear_fwd(x, self.ternary_weight, self.gamma)

    def ternary_stats(self) -> dict[str, float]:
        """Report ternary weight and gamma statistics.

        Unpacks the packed uint8 weights before computing per-weight stats.
        """
        w = unpack_ternary(self.ternary_weight, self.in_features)
        total = w.size  # = out_features * in_features (logical size)
        return {
            "sparsity": (w == 0).sum().item() / total,
            "pos_frac": (w == 1).sum().item() / total,
            "neg_frac": (w == -1).sum().item() / total,
            "gamma_mean": self.gamma.mean().item(),
            "gamma_std": mx.sqrt(mx.var(self.gamma)).item(),
            "accum_mean": mx.abs(self._flip_accum.astype(mx.float32)).mean().item(),
            "accum_max": mx.abs(self._flip_accum.astype(mx.float32)).max().item(),
            "cooldown_active": int((self._flip_cooldown > 0).sum().item()),
            "ever_flipped": int((self._flip_last_dir != 0).sum().item()),
        }


# ══════════════════════════════════════════════════════════════════════
# TernaryEmbedding — packed ternary lookup table
# ══════════════════════════════════════════════════════════════════════


class TernaryEmbedding(nn.Module):
    """Embedding layer with ternary vectors and per-token gamma.

    Each vocabulary entry is a ternary vector {-1, 0, +1}^d_model with a
    float32 per-token scale (gamma). Lookup unpacks the selected rows on
    the fly, producing float32 output identical to standard embedding.

    Storage: vocab_size × d_model/4 bytes (packed) + vocab_size × 4 bytes (gamma)
           = vocab_size × (d_model/4 + 4) bytes
    vs float: vocab_size × d_model × 4 bytes

    For vocab=50277, d=1024: 13.1 MB packed vs 196.4 MB float (15× smaller).

    The ternary embedding participates in the flip accumulation mechanism
    just like TernaryLinear, enabling topology evolution during training.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Initialize: random normal → quantize → pack
        w_packed, gamma = _ternary_init(vocab_size, d_model)
        self.ternary_weight = w_packed   # (vocab_size, d_model//4) uint8
        self.gamma = gamma               # (vocab_size,) float32

        # Flip accumulator (same as TernaryLinear)
        self._flip_accum = mx.zeros((vocab_size, d_model), dtype=mx.int8)
        self._flip_cooldown = mx.zeros((vocab_size, d_model), dtype=mx.int8)
        self._flip_last_dir = mx.zeros((vocab_size, d_model), dtype=mx.int8)

    def __call__(self, tokens: mx.array) -> mx.array:
        """Lookup ternary embeddings for token indices.

        tokens: (*, ) int array of token indices
        Returns: (*, d_model) float32 array

        Unpacks the packed rows for the selected tokens and multiplies
        by the per-token gamma scale.
        """
        return _ternary_embed_fwd(tokens, self.ternary_weight, self.gamma)

    @property
    def weight_T(self) -> mx.array:
        """Unpacked weight matrix transposed: (d_model, vocab_size) float32.

        Used for tied output projection: logits = h @ embed.weight_T
        This is computed on-the-fly from packed ternary weights + gamma.
        """
        # Unpack: (vocab_size, d_model) int8
        w = unpack_ternary(self.ternary_weight, self.d_model).astype(mx.float32)
        # Scale by gamma: (vocab_size, d_model) * (vocab_size, 1) → (vocab_size, d_model)
        w = w * mx.expand_dims(self.gamma, axis=-1)
        return w.T  # (d_model, vocab_size)

    @property
    def in_features(self):
        """For compatibility with _walk_ternary_modules / flip utilities."""
        return self.d_model

    @property
    def out_features(self):
        return self.vocab_size


@mx.custom_function
def _ternary_embed_fwd(tokens: mx.array, w_packed: mx.array, gamma: mx.array) -> mx.array:
    """Forward: unpack selected rows from packed ternary embedding, scale by gamma.

    tokens:   (*,) int indices
    w_packed: (vocab_size, d_model//4) uint8
    gamma:    (vocab_size,) float32

    Returns:  (*, d_model) float32
    """
    d_model = w_packed.shape[1] * 4
    # Gather packed rows for the selected tokens
    flat_tokens = tokens.reshape(-1)
    packed_rows = w_packed[flat_tokens]     # (N, d_model//4) uint8
    gamma_rows = gamma[flat_tokens]         # (N,) float32

    # Unpack: (N, d_model//4) uint8 → (N, d_model) int8 → float32
    w0 = ((packed_rows >> 6) & 0x3).astype(mx.float32) - 1.0
    w1 = ((packed_rows >> 4) & 0x3).astype(mx.float32) - 1.0
    w2 = ((packed_rows >> 2) & 0x3).astype(mx.float32) - 1.0
    w3 = (packed_rows & 0x3).astype(mx.float32) - 1.0
    # Interleave: columns {4k, 4k+1, 4k+2, 4k+3}
    N = flat_tokens.shape[0]
    K4 = packed_rows.shape[1]
    unpacked = mx.stack([w0, w1, w2, w3], axis=-1).reshape(N, d_model)

    # Scale by per-token gamma
    result = unpacked * mx.expand_dims(gamma_rows, axis=-1)

    # Reshape to match input token shape + d_model
    return result.reshape(*tokens.shape, d_model)


@_ternary_embed_fwd.vjp
def _ternary_embed_vjp(primals, cotangent, output):
    """Backward through ternary embedding lookup.

    ∂L/∂tokens:  zeros (integer indices, not differentiable)
    ∂L/∂w_packed: zeros matching packed shape — real grad goes to _embed_grad_cache
                  (flip accumulator collects it separately, same as TernaryLinear)
    ∂L/∂gamma:   per-token grad, scattered back to (vocab_size,)
    """
    tokens, w_packed, gamma = primals
    grad_out = cotangent  # (*, d_model)
    d_model = w_packed.shape[1] * 4

    flat_tokens = tokens.reshape(-1)
    N = flat_tokens.shape[0]
    grad_flat = grad_out.reshape(N, d_model)

    # ∂L/∂gamma: for each selected token, reduce grad_out over d_model
    # First unpack the selected rows to compute the dot product
    packed_rows = w_packed[flat_tokens]
    w0 = ((packed_rows >> 6) & 0x3).astype(mx.float32) - 1.0
    w1 = ((packed_rows >> 4) & 0x3).astype(mx.float32) - 1.0
    w2 = ((packed_rows >> 2) & 0x3).astype(mx.float32) - 1.0
    w3 = (packed_rows & 0x3).astype(mx.float32) - 1.0
    unpacked = mx.stack([w0, w1, w2, w3], axis=-1).reshape(N, d_model)

    # grad_gamma_per_token = Σ_d (grad_out[n,d] * unpacked[n,d])
    grad_gamma_per_token = mx.sum(grad_flat * unpacked, axis=-1)  # (N,)

    # Scatter gamma grads back to (vocab_size,) — use vectorized scatter
    grad_gamma = mx.zeros((gamma.shape[0],), dtype=mx.float32)
    grad_gamma = grad_gamma.at[flat_tokens].add(grad_gamma_per_token)

    # ∂L/∂w: return zeros for w_packed (shape-matched), store real grad
    # in the module's _embed_grad_cache for the flip accumulator
    grad_w_packed = mx.zeros_like(w_packed).astype(mx.float32)

    # Compute and cache the STE grad for flip accumulation:
    # Store in a module-level cache that accumulate_flips_embed reads
    gamma_rows = gamma[flat_tokens]  # (N,)
    grad_w_dense = grad_flat * mx.expand_dims(gamma_rows, axis=-1)  # (N, d_model)
    # Scatter to full vocab: (vocab_size, d_model)
    full_grad = mx.zeros((w_packed.shape[0], d_model), dtype=mx.float32)
    full_grad = full_grad.at[flat_tokens].add(grad_w_dense)
    # Store in global cache keyed by w_packed id
    _EMBED_GRAD_CACHE[id(w_packed)] = full_grad

    # No gradient for tokens
    grad_tokens = mx.zeros(tokens.shape, dtype=mx.float32)

    return grad_tokens, grad_w_packed, grad_gamma


# Global cache for embedding STE gradients (consumed by accumulate_flips)
_EMBED_GRAD_CACHE: dict[int, mx.array] = {}


# ══════════════════════════════════════════════════════════════════════
# Flip utilities (simplified for v8)
# ══════════════════════════════════════════════════════════════════════


def _walk_ternary_modules(model: nn.Module):
    """Yield (path, module) for all TernaryLinear and TernaryEmbedding modules in model."""
    for path, module in model.named_modules():
        if isinstance(module, (TernaryLinear, TernaryEmbedding)):
            yield path, module


def accumulate_flips(model: nn.Module, ternary_grads: dict[str, Any]) -> None:
    """Accumulate gradient direction votes for ternary weight flips.

    Uses sign(grad) rather than raw gradient magnitude. Each call
    adds +1 or -1 per weight, so after N calls |accum| ≤ N. This
    makes the accumulator scale-invariant and the threshold meaningful
    in units of "directional consensus across micro-batches."

    Accumulators are reset to zero by apply_flips after each flip check,
    so they measure consensus within one interval only.

    Call after loss backward, per micro-batch.

    Args:
        model: the model containing TernaryLinear modules
        ternary_grads: gradient pytree (full or ternary-only)
    """
    def _extract_grad(tree, path_parts):
        """Navigate the grad pytree to find the gradient at a given path."""
        node = tree
        for part in path_parts:
            if isinstance(node, dict):
                node = node.get(part)
            elif isinstance(node, list):
                node = node[int(part)]
            else:
                return None
            if node is None:
                return None
        return node

    accums = []
    for path, module in _walk_ternary_modules(model):
        # For TernaryEmbedding: retrieve cached STE grad from VJP
        if isinstance(module, TernaryEmbedding):
            cache_key = id(module.ternary_weight)
            grad = _EMBED_GRAD_CACHE.pop(cache_key, None)
        else:
            # For TernaryLinear: extract from grad pytree
            parts = path.split(".") if path else []
            parts.append("ternary_weight")
            grad = _extract_grad(ternary_grads, parts)

        if grad is not None:
            # NaN guard: don't poison the accumulator with NaN gradients
            if mx.any(mx.isnan(grad)).item():
                continue
            # Sign-based accumulation: direction only, not magnitude.
            # Each micro-batch casts a vote (+1 or -1) per weight.
            # Int8 with saturating clip at ±127.
            vote = mx.sign(grad).astype(mx.int8)
            module._flip_accum = mx.clip(
                module._flip_accum.astype(mx.int16) + vote.astype(mx.int16),
                -127, 127,
            ).astype(mx.int8)
            accums.append(module._flip_accum)

    # Materialize accumulators to prevent lazy graph buildup.
    if accums:
        mx.eval(*accums)


def compute_flip_threshold(model: nn.Module, target_pct: float) -> float:
    """Compute threshold to flip approximately target_pct of ternary weights.

    Uses the percentile of accumulator absolute values so that exactly
    target_pct fraction of weights exceed the threshold. This decouples
    the flip decision from accumulator scale.

    Args:
        model: the model containing TernaryLinear modules
        target_pct: fraction of weights to flip (e.g. 0.005 = 0.5%)

    Returns:
        Threshold value. Returns float('inf') if no valid accumulators.
    """
    import numpy as np
    chunks = []
    for _, module in _walk_ternary_modules(model):
        mx.eval(module._flip_accum)
        chunks.append(mx.abs(module._flip_accum).astype(mx.int16).reshape(-1))
    if not chunks:
        return float("inf")
    all_abs = mx.concatenate(chunks)
    all_np = np.array(all_abs)
    pct = 100.0 * (1.0 - target_pct)
    return float(np.percentile(all_np, pct))


def apply_flips(model: nn.Module, threshold: int = 50, max_flip_pct: float = 0.001,
                cooldown_intervals: int = 8) -> tuple[int, int]:
    """Flip ternary weights where accumulated consensus exceeds threshold.

    Like synaptic plasticity: each weight flips only when IT has
    accumulated enough directional evidence. But capped: at most
    max_flip_pct of total ternary weights can flip per call, to prevent
    catastrophic mass mutation when early-training gradients are globally
    coherent (every weight agrees because the model knows nothing).

    When more weights cross the threshold than the cap allows, only the
    strongest consensus (highest |accum|) flip.

    Each flip moves one step in the gradient direction:
      -1 + positive pressure → 0
       0 + positive pressure → +1
      +1 + negative pressure → 0
       0 + negative pressure → -1

    Respects per-weight cooldown: weights with _flip_cooldown > 0 are
    skipped. After flipping, the flipped weight's cooldown is set to
    `cooldown_intervals`. Each call decrements all cooldowns by 1.
    This prevents oscillation: a weight that just flipped must wait
    cooldown_intervals × flip_interval steps before it can flip again.

    Args:
        model: the model containing TernaryLinear modules
        threshold: minimum |accumulator| to trigger a flip (vote units)
        max_flip_pct: maximum fraction of ternary weights to flip per call
        cooldown_intervals: intervals to lock a weight after flipping (default 8)

    Returns:
        Total number of weights flipped across all modules.
    """
    # Step 1: collect all accumulators that exceed threshold
    candidates = []  # [(module, accum_abs)]
    total_ternary = 0
    for _, module in _walk_ternary_modules(model):
        total_ternary += module.out_features * module.in_features
        accum_abs = mx.abs(module._flip_accum.astype(mx.int16))
        candidates.append((module, accum_abs))

    max_flips = int(total_ternary * max_flip_pct)

    def _count_at_or_above(t):
        return sum((a >= t).sum().item() for _, a in candidates)

    n_qualifying = _count_at_or_above(threshold)
    effective_threshold = threshold

    if n_qualifying > max_flips and max_flips > 0:
        lo, hi = threshold, 127
        while lo < hi:
            mid = (lo + hi) // 2
            if _count_at_or_above(mid) > max_flips:
                lo = mid + 1
            else:
                hi = mid
        effective_threshold = lo

    # Step 2: re-count and apply with cooldown awareness
    n_qualifying_final = _count_at_or_above(effective_threshold)
    subsample = n_qualifying_final > max_flips and max_flips > 0
    if subsample:
        keep_prob = max_flips / n_qualifying_final

    total_flipped = 0
    total_reversals = 0
    mutated = []

    for module, accum_abs in candidates:
        # ── Decrement cooldowns first (every flip check) ──
        if mx.any(module._flip_cooldown > 0).item():
            module._flip_cooldown = mx.maximum(
                module._flip_cooldown.astype(mx.int16) - 1, 0
            ).astype(mx.int8)
            mutated.append(module._flip_cooldown)

        mask = accum_abs >= int(effective_threshold)

        # Block weights still on cooldown
        mask = mask & (module._flip_cooldown <= 0)

        if subsample:
            rand_mask = mx.random.uniform(shape=mask.shape) < keep_prob
            mask = mask & rand_mask

        n_flipped = mask.sum().item()

        if n_flipped > 0:
            direction = mx.sign(module._flip_accum.astype(mx.int16)).astype(mx.int8)

            # ── Detect reversals: flip direction ≠ last direction ──
            # A reversal means this weight flipped, then flipped back.
            # Only count for weights that have flipped before (last_dir ≠ 0).
            has_history = module._flip_last_dir != 0
            reversed_dir = direction != module._flip_last_dir
            reversals = mask & has_history & reversed_dir
            n_reversals = int(reversals.sum().item())
            total_reversals += n_reversals

            # Unpack → flip on unpacked int8 → repack
            w_int8 = unpack_ternary(module.ternary_weight, module.in_features)
            current = w_int8.astype(mx.int16)
            new_vals = mx.clip(current + direction.astype(mx.int16), -1, 1).astype(mx.int8)
            updated = mx.where(mask, new_vals, w_int8)

            module.ternary_weight = pack_ternary(updated)
            mutated.append(module.ternary_weight)

            # ── Set cooldown on flipped weights ──
            module._flip_cooldown = mx.where(
                mask,
                mx.full(mask.shape, cooldown_intervals, dtype=mx.int8),
                module._flip_cooldown,
            )
            mutated.append(module._flip_cooldown)

            # ── Update direction history ──
            module._flip_last_dir = mx.where(mask, direction, module._flip_last_dir)
            mutated.append(module._flip_last_dir)

            total_flipped += int(n_flipped)

    # Reset ALL accumulators — fresh question each interval
    for module, _ in candidates:
        module._flip_accum = mx.zeros_like(module._flip_accum)
        mutated.append(module._flip_accum)

    if mutated:
        mx.eval(*mutated)

    return total_flipped, total_reversals


def zero_ternary_grads(model: nn.Module, grads: dict) -> dict:
    """Zero out ternary_weight gradients in the grad pytree.

    Ternary weight gradients feed the flip accumulator (sign-based),
    not the optimizer. Including them in clip_grad_norm poisons the
    continuous parameter updates: a single large ternary gradient
    dominates the total norm, clipping continuous params to near-zero.

    The VJP produces dense [N, K] gradients for the flip accumulator,
    but the packed parameter is [N, K/4]. The optimizer requires
    gradient and parameter shapes to match. So we return zeros with
    the PACKED parameter shape, not the dense gradient shape.

    Call this AFTER accumulate_flips and BEFORE clip_grad_norm.
    """
    # Collect paths and packed shapes of ternary weight parameters
    ternary_info: dict[str, tuple] = {}
    for path, module in _walk_ternary_modules(model):
        key = f"{path}.ternary_weight" if path else "ternary_weight"
        ternary_info[key] = module.ternary_weight.shape

    def _zero(path_prefix: str, tree):
        if isinstance(tree, dict):
            return {
                k: _zero(f"{path_prefix}.{k}" if path_prefix else k, v)
                for k, v in tree.items()
            }
        elif isinstance(tree, list):
            return [
                _zero(f"{path_prefix}.{i}" if path_prefix else str(i), v)
                for i, v in enumerate(tree)
            ]
        elif isinstance(tree, mx.array) and path_prefix in ternary_info:
            # Return zeros matching the PACKED parameter shape [N, K/4],
            # not the dense gradient shape [N, K] from the VJP.
            packed_shape = ternary_info[path_prefix]
            return mx.zeros(packed_shape, dtype=tree.dtype)
        return tree

    return _zero("", grads)


def save_ternary_state(model: nn.Module, path: str) -> None:
    """Save ternary flip metadata (cooldown + direction history).

    The flip accumulator is NOT saved — it must be rebuilt from fresh
    gradient evidence after resume. Cooldown and direction history
    are structural: they record the topology's evolution.
    """
    state = {}
    for mod_path, module in _walk_ternary_modules(model):
        state[f"{mod_path}.cooldown"] = module._flip_cooldown
        state[f"{mod_path}.last_dir"] = module._flip_last_dir
    if state:
        mx.savez(path, **state)


def load_ternary_state(model: nn.Module, path: str) -> None:
    """Restore ternary flip metadata from checkpoint.

    Restores cooldown and direction history. Resets accumulator to zero
    (fresh gradient evidence needed after resume).
    """
    import os
    if not os.path.exists(path):
        return

    state = dict(mx.load(path))

    for mod_path, module in _walk_ternary_modules(model):
        cd_key = f"{mod_path}.cooldown"
        ld_key = f"{mod_path}.last_dir"

        if cd_key in state:
            module._flip_cooldown = state[cd_key].astype(mx.int8)
        if ld_key in state:
            module._flip_last_dir = state[ld_key].astype(mx.int8)

        # Always reset accumulator — no stale gradient evidence
        module._flip_accum = mx.zeros_like(module._flip_accum)

    mx.eval(*[m._flip_cooldown for _, m in _walk_ternary_modules(model)],
            *[m._flip_last_dir for _, m in _walk_ternary_modules(model)],
            *[m._flip_accum for _, m in _walk_ternary_modules(model)])


def restore_ternary(model: nn.Module) -> None:
    """Re-cast any ternary weights back to uint8 after optimizer update.

    The optimizer may cast uint8 packed weights to float during its update
    step. Since the packed weights should never be touched by the optimizer
    (they are uint8 and the gradient is zeroed), this is a safety net.

    Call after every optimizer.update().
    """
    def _walk(mod):
        if isinstance(mod, (TernaryLinear, TernaryEmbedding)):
            if mod.ternary_weight.dtype != mx.uint8:
                mod.ternary_weight = mx.clip(
                    mx.round(mod.ternary_weight), 0, 255
                ).astype(mx.uint8)
        if isinstance(mod, nn.Module):
            for name, child in mod.children().items():
                if isinstance(child, nn.Module):
                    _walk(child)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, nn.Module):
                            _walk(item)
    _walk(model)
