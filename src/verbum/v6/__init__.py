"""VSM-LM v6 — Ternary on Metal (MLX).

Custom Metal compute kernels for ternary matmul (add/sub only).
Flip accumulation for discrete weight learning.

Core modules:
    TernaryLinear    — int8 ternary weights, custom Metal kernel, VJP
    TernaryFFN       — ternary feed-forward with residual
    StrideStack      — sequential multi-stride ternary attention
    VSMLMV6          — full 5-pass bidirectional VSM architecture

Training utilities:
    split_ternary_grads   — separate ternary vs continuous gradients
    accumulate_flips      — route gradients to flip accumulators
    apply_flips           — flip weights where |accum| > threshold

Metal kernels:
    ternary_matmul        — y = x @ w.T (w ∈ {-1,0,+1})
    ternary_matmul_t      — y = x @ w   (transposed, for backward)
"""

from verbum.v6.kernels import ternary_matmul, ternary_matmul_t
from verbum.v6.ternary import (
    TernaryLinear,
    TernaryFFN,
    split_ternary_grads,
    accumulate_flips,
    apply_flips,
)
from verbum.v6.attention import SingleStrideAttention, StrideStack
from verbum.v6.components import S4Ternary, S3Ternary, MetaS4Ternary, MetaS3Ternary
from verbum.v6.model import VSMLMV6

__all__ = [
    "ternary_matmul",
    "ternary_matmul_t",
    "TernaryLinear",
    "TernaryFFN",
    "split_ternary_grads",
    "accumulate_flips",
    "apply_flips",
    "SingleStrideAttention",
    "StrideStack",
    "S4Ternary",
    "S3Ternary",
    "MetaS4Ternary",
    "MetaS3Ternary",
    "VSMLMV6",
]
