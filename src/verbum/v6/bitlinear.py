"""BitLinear — Ternary weight quantization with Straight-Through Estimator.

Implements the core 1.58-bit linear layer from BitNet b1.58 (Ma et al., 2024).

Weight quantization (forward only — master weights stay fp32):
  γ = mean(|W|)
  W_q = RoundClip(W / γ, -1, 1)  →  {-1, 0, +1}
  y = RMSNorm(x) @ W_q^T · γ

Backward: STE — gradients pass through the quantization function as if
it were identity. The optimizer updates the full-precision master weights.

No bias (BitNet convention — RMSNorm + ternary weights make bias redundant).

Activation quantization (8-bit absmax, optional) is deferred — start with
fp32 activations and add activation quantization as a refinement if needed.

References:
  - "The Era of 1-bit LLMs" (Ma et al., 2024)
  - "When Are 1.58 Bits Enough?" (Nielsen et al., 2024)

License: MIT
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# RMSNorm (BitNet standard — no bias, no centering)
# ══════════════════════════════════════════════════════════════════════


class BitRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not center (subtract mean) or add bias.
    Just scales by 1/RMS and a learnable gain. Cheaper and works better
    with ternary weights (no mean shift to fight the quantization).

    norm(x) = x / RMS(x) · gain
    RMS(x) = √(mean(x²) + ε)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ══════════════════════════════════════════════════════════════════════
# Ternary quantization functions
# ══════════════════════════════════════════════════════════════════════


def _ternary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to {-1, 0, +1} using absmean scaling.

    Args:
        w: full-precision weight tensor

    Returns:
        w_q: ternary weight tensor {-1, 0, +1}
        gamma: scale factor (mean of absolute values)
    """
    gamma = w.abs().mean()
    w_scaled = w / (gamma + 1e-8)
    # Round to nearest integer and clip to [-1, 1] → {-1, 0, +1}
    w_q = w_scaled.round().clamp(-1, 1)
    return w_q, gamma


class _TernaryQuantizeSTE(torch.autograd.Function):
    """Ternary quantization with Straight-Through Estimator.

    Forward: quantize to {-1, 0, +1}, scale by gamma.
    Backward: pass gradients through as if quantization is identity.

    The STE is the key training trick — it lets gradients flow to the
    full-precision master weights even though the forward pass uses
    ternary weights.
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w_q, gamma = _ternary_quantize(w)
        ctx.save_for_backward(w)
        return w_q, gamma

    @staticmethod
    def backward(ctx, grad_w_q, grad_gamma):
        # STE: pass gradient through to the full-precision weights
        return grad_w_q


def ternary_quantize_ste(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to ternary with STE gradient.

    Returns (w_q, gamma) where:
      w_q: {-1, 0, +1} tensor (same shape as w)
      gamma: scalar scale factor
    """
    return _TernaryQuantizeSTE.apply(w)


# ══════════════════════════════════════════════════════════════════════
# BitLinear — the ternary linear layer
# ══════════════════════════════════════════════════════════════════════


class BitLinear(nn.Module):
    """Linear layer with ternary weight quantization.

    Stores master weights in full precision (fp32). During forward:
      1. RMSNorm the input (stabilize activations before ternary matmul)
      2. Quantize weights to {-1, 0, +1} via absmean
      3. Matmul (becomes additions/subtractions with ternary weights)
      4. Scale output by gamma (the absmean of the full-precision weights)

    During backward: STE passes gradients through to master weights.

    The matmul with ternary weights is mathematically:
      y_j = Σ_i x_i · w_q_{ij}  where w_q ∈ {-1, 0, +1}
          = Σ_{w=+1} x_i - Σ_{w=-1} x_i  (additions only!)

    No bias (BitNet convention). The RMSNorm gain provides per-feature
    scaling, making explicit bias unnecessary.

    Parameters:
        in_features: input dimension
        out_features: output dimension
        pre_norm: whether to RMSNorm the input (default True)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pre_norm = pre_norm

        # Master weights — full precision, updated by optimizer
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Pre-norm (RMSNorm on input before ternary matmul)
        if pre_norm:
            self.norm = BitRMSNorm(in_features)
        else:
            self.norm = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Kaiming uniform, same as nn.Linear default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Pre-norm input
        if self.norm is not None:
            x = self.norm(x)

        # 2. Quantize weights to ternary with STE
        w_q, gamma = ternary_quantize_ste(self.weight)

        # 3. Matmul with ternary weights (additions/subtractions)
        # 4. Scale by gamma
        return F.linear(x, w_q) * gamma

    @torch.no_grad()
    def ternary_stats(self) -> dict[str, float]:
        """Report quantization statistics (for instrumentation).

        Returns:
            sparsity: fraction of weights quantized to 0
            pos_frac: fraction quantized to +1
            neg_frac: fraction quantized to -1
            gamma: current absmean scale factor
        """
        w_q, gamma = _ternary_quantize(self.weight)
        total = w_q.numel()
        return {
            "sparsity": (w_q == 0).sum().item() / total,
            "pos_frac": (w_q == 1).sum().item() / total,
            "neg_frac": (w_q == -1).sum().item() / total,
            "gamma": gamma.item(),
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"pre_norm={self.pre_norm}, "
            f"bits=1.58"
        )


# ══════════════════════════════════════════════════════════════════════
# BitFFN — Ternary feed-forward network
# ══════════════════════════════════════════════════════════════════════


class BitFFN(nn.Module):
    """Feed-forward network with ternary weights.

    Pre-norm → BitLinear(up) → GELU → BitLinear(down)

    The up-projection and down-projection are both ternary.
    GELU activation stays fp32 (activations are not quantized).
    The pre-norm in BitLinear handles input normalization.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Up-projection: pre_norm=True (RMSNorm before ternary matmul)
        self.up = BitLinear(d_model, d_ff, pre_norm=True)
        self.act = nn.GELU()
        # Down-projection: pre_norm=False (GELU output is already scaled)
        self.down = BitLinear(d_ff, d_model, pre_norm=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.down(self.act(self.up(x))))
