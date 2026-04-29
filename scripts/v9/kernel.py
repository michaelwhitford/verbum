"""
v9 — Lambda Kernel: Exact Computation Primitives

The kernel hypothesis: instead of learning arithmetic through
beta-reduction (expand-reduce over many layers), provide exact
primitives that the model routes to via ternary topology.

This module provides:
  1. Exact arithmetic dispatch (add, sub, mul, integer div)
  2. Decode: continuous vector → (op_code, arg1, arg2) via argmax/round
  3. Encode: integer result → d-dimensional vector via learned embedding

The routing layer (ternary) learns WHERE to send. The kernel does
WHAT to compute. Evolution finds the wiring. The kernel is exact.

Phase 1: arithmetic only. Lambda calculus primitives come later
if the routing concept proves viable.

License: MIT
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


# ══════════════════════════════════════════════════════════════════════
# Kernel operations — exact, no gradient, no approximation
# ══════════════════════════════════════════════════════════════════════

# Op codes: indices into the dispatch table
OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
N_OPS = 3

OP_NAMES = {OP_ADD: "+", OP_SUB: "-", OP_MUL: "*"}


def kernel_dispatch(op: mx.array, arg1: mx.array, arg2: mx.array) -> mx.array:
    """Execute exact arithmetic given discrete op/arg tensors.

    All inputs are integer tensors (same shape). Output is integer.
    This is the non-differentiable exact computation core.

    Args:
        op:   int tensor, values in [0, N_OPS). Op code.
        arg1: int tensor. First operand.
        arg2: int tensor. Second operand.

    Returns:
        int tensor of results. Same shape as inputs.
    """
    # Compute all operations, select by op code
    r_add = arg1 + arg2
    r_sub = arg1 - arg2
    r_mul = arg1 * arg2

    # Select: op==0 → add, op==1 → sub, op==2 → mul
    result = mx.where(op == OP_ADD, r_add,
             mx.where(op == OP_SUB, r_sub,
                       r_mul))  # default to mul for op==2

    return result


# ══════════════════════════════════════════════════════════════════════
# Decode: continuous routing vector → discrete kernel inputs
# ══════════════════════════════════════════════════════════════════════


def decode_routing(
    routing_logits: mx.array,
    n_ops: int = N_OPS,
    max_val: int = 100,
) -> tuple[mx.array, mx.array, mx.array]:
    """Decode continuous routing vector into discrete kernel inputs.

    The routing layer outputs a vector of shape (..., n_ops + 2*max_val).
    First n_ops dimensions are op-code logits (argmax selects op).
    Next max_val dimensions are arg1 logits (argmax selects value).
    Last max_val dimensions are arg2 logits (argmax selects value).

    Args:
        routing_logits: (..., n_ops + 2*max_val) float tensor
        n_ops:          number of operations
        max_val:        number of possible integer values [0, max_val)

    Returns:
        (op, arg1, arg2) — each int tensor of shape (...)
    """
    op_logits = routing_logits[..., :n_ops]
    arg1_logits = routing_logits[..., n_ops:n_ops + max_val]
    arg2_logits = routing_logits[..., n_ops + max_val:n_ops + 2 * max_val]

    op = mx.argmax(op_logits, axis=-1).astype(mx.int32)
    arg1 = mx.argmax(arg1_logits, axis=-1).astype(mx.int32)
    arg2 = mx.argmax(arg2_logits, axis=-1).astype(mx.int32)

    return op, arg1, arg2


# ══════════════════════════════════════════════════════════════════════
# Encode: integer result → vector for downstream processing
# ══════════════════════════════════════════════════════════════════════


class ResultEncoder(nn.Module):
    """Encode kernel output (integer) back into a d-dimensional vector.

    Simple approach: learned embedding table for result values.
    The result range is bounded by the input range and operations.
    For max_val=100 with +/-/*: results range roughly [-100, 9801].
    We bucket into n_buckets values via clamping.

    This is the re-entry point from exact computation back into
    the neural network's continuous representation space.
    """

    def __init__(self, n_buckets: int = 512, d_model: int = 64):
        super().__init__()
        self.n_buckets = n_buckets
        self.offset = n_buckets // 2  # center at 0
        self.embed = nn.Embedding(n_buckets, d_model)

    def __call__(self, result: mx.array) -> mx.array:
        """Encode integer result to d-dimensional vector.

        Args:
            result: int tensor of any shape

        Returns:
            float tensor of shape (*result.shape, d_model)
        """
        # Shift result into [0, n_buckets) range
        idx = mx.clip(result + self.offset, 0, self.n_buckets - 1).astype(mx.int32)
        return self.embed(idx)


# ══════════════════════════════════════════════════════════════════════
# Full kernel forward: route → decode → dispatch → encode
# ══════════════════════════════════════════════════════════════════════


def kernel_forward(
    routing_logits: mx.array,
    encoder: ResultEncoder,
    max_val: int = 100,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Full kernel pipeline: decode routing → exact dispatch → encode result.

    Args:
        routing_logits: (..., N_OPS + 2*max_val) from the ternary routing layer
        encoder:        ResultEncoder module
        max_val:        integer value range [0, max_val)

    Returns:
        (encoded_result, op, arg1, arg2, result)
        - encoded_result: (..., d_model) float tensor for downstream use
        - op, arg1, arg2: int tensors — what the kernel decoded
        - result: int tensor — what the kernel computed
    """
    op, arg1, arg2 = decode_routing(routing_logits, N_OPS, max_val)
    result = kernel_dispatch(op, arg1, arg2)
    encoded = encoder(result)
    return encoded, op, arg1, arg2, result


# ══════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  v9 — Lambda Kernel Smoke Test")
    print("=" * 60)

    # Test dispatch
    op = mx.array([OP_ADD, OP_SUB, OP_MUL])
    a = mx.array([3, 10, 4])
    b = mx.array([4, 3, 5])
    r = kernel_dispatch(op, a, b)
    mx.eval(r)
    print(f"\nDispatch test:")
    print(f"  3 + 4 = {r[0].item()}")
    print(f"  10 - 3 = {r[1].item()}")
    print(f"  4 * 5 = {r[2].item()}")
    assert r[0].item() == 7
    assert r[1].item() == 7
    assert r[2].item() == 20

    # Test decode
    max_val = 100
    logit_dim = N_OPS + 2 * max_val
    fake_logits = mx.zeros((2, logit_dim))
    # Encode: op=ADD(0), arg1=3, arg2=4
    fake_logits = fake_logits.at[0, OP_ADD].add(10.0)  # op = add
    fake_logits = fake_logits.at[0, N_OPS + 3].add(10.0)  # arg1 = 3
    fake_logits = fake_logits.at[0, N_OPS + max_val + 4].add(10.0)  # arg2 = 4
    # Encode: op=MUL(2), arg1=7, arg2=8
    fake_logits = fake_logits.at[1, OP_MUL].add(10.0)
    fake_logits = fake_logits.at[1, N_OPS + 7].add(10.0)
    fake_logits = fake_logits.at[1, N_OPS + max_val + 8].add(10.0)

    op, a1, a2 = decode_routing(fake_logits, N_OPS, max_val)
    mx.eval(op, a1, a2)
    print(f"\nDecode test:")
    print(f"  Decoded: op={op[0].item()}, arg1={a1[0].item()}, arg2={a2[0].item()}")
    print(f"  Decoded: op={op[1].item()}, arg1={a1[1].item()}, arg2={a2[1].item()}")
    assert op[0].item() == OP_ADD and a1[0].item() == 3 and a2[0].item() == 4
    assert op[1].item() == OP_MUL and a1[1].item() == 7 and a2[1].item() == 8

    # Test full pipeline
    encoder = ResultEncoder(n_buckets=512, d_model=64)
    enc, op, a1, a2, res = kernel_forward(fake_logits, encoder, max_val=100)
    mx.eval(enc, res)
    print(f"\nFull pipeline test:")
    print(f"  3 + 4 = {res[0].item()}, encoded shape: {enc[0].shape}")
    print(f"  7 * 8 = {res[1].item()}, encoded shape: {enc[1].shape}")
    assert res[0].item() == 7
    assert res[1].item() == 56

    print(f"\n{'=' * 60}")
    print(f"  ✓ All kernel tests passed")
    print(f"{'=' * 60}")
