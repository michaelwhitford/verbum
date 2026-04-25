"""Metal compute kernels for ternary matrix multiplication.

Ternary matmul computes y = x @ W^T where W ∈ {-1, 0, +1} (int8).
The operation is pure addition/subtraction — no floating-point
multiplies. Each weight value routes the corresponding input element:

    +1 → add input to accumulator
    -1 → subtract input from accumulator
     0 → skip (free sparsity)

Four kernel variants:
    ternary_matmul(x, w)              — y[m,n] = Σ_k T(w[n,k], x[m,k])    (int8 weights)
    ternary_matmul_t(x, w)            — y[m,k] = Σ_n T(w[n,k], x[m,n])    (int8 weights)
    ternary_matmul_packed(x, w, K)    — same forward, packed uint8 weights
    ternary_matmul_t_packed(x, w, K)  — same transpose, packed uint8 weights

Packing scheme: 4 weights per uint8 byte.
    Encoding: -1 → 0b00, 0 → 0b01, +1 → 0b10   (0b11 unused)
    Bit positions: {6, 4, 2, 0} for columns {4k, 4k+1, 4k+2, 4k+3}
    Decode: ((packed >> shift) & 0x3) - 1

Phase 1: naive kernels (one thread per output element, sequential K-loop).
Phase 2+: tiled kernels with threadgroup shared memory.

License: MIT
"""

from __future__ import annotations

import mlx.core as mx

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


def ternary_matmul(x: mx.array, w: mx.array) -> mx.array:
    """Ternary matrix multiplication: y = x @ w.T

    Args:
        x: (M, K) or (*, M, K) float array — input activations
        w: (N, K) int8 array — ternary weights {-1, 0, +1}

    Returns:
        (M, N) or (*, M, N) float array — output activations
    """
    # Handle batched input: reshape to 2D, compute, reshape back
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

    # Restore original dimensions
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

    out = _ternary_matmul_t_packed_kernel(
        inputs=[x_2d, w_packed],
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


# ══════════════════════════════════════════════════════════════════════
# Reference implementation (pure MLX, for testing)
# ══════════════════════════════════════════════════════════════════════


def ternary_matmul_reference(x: mx.array, w: mx.array) -> mx.array:
    """Reference ternary matmul using standard MLX ops.

    Computes x @ w.T where w is int8 {-1, 0, +1}, by casting
    w to float and using mx.matmul. Result should be identical
    to ternary_matmul() — this is the correctness oracle.
    """
    return x @ w.astype(x.dtype).T


def ternary_matmul_t_reference(x: mx.array, w: mx.array) -> mx.array:
    """Reference transposed ternary matmul: x @ w (not w.T)."""
    return x @ w.astype(x.dtype)
