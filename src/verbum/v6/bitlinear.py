"""BitLinear — Ternary weights trained with STE + per-channel learned scales.

Weights are trained via Straight-Through Estimator: the forward pass
quantizes to {-1, 0, +1}, the backward pass passes gradients through
as if the quantization were identity. The optimizer updates full-precision
master weights; the ternary quantization happens fresh each forward.

Per-channel gamma provides an additional learned scale per output dimension.
This gives the model more expressiveness than a single scalar scale —
individual output channels can be amplified or silenced.

At inference, the master weights are quantized once, gamma is folded in,
and the model runs with native ternary matmuls (additions/subtractions).

Training memory: same as fp16 (master weights + Adam state in fp32).
Inference: ternary weights at 2 bits + gamma at fp16 → ~61 MB total.
The training cost is one-time. The deployment benefit is permanent.

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
# Ternary quantization with STE
# ══════════════════════════════════════════════════════════════════════


def _ternary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to {-1, 0, +1} using absmean scaling.

    Args:
        w: full-precision weight tensor (out_features, in_features)

    Returns:
        w_q: ternary weight tensor {-1, 0, +1}
        gamma: scalar scale factor (mean of absolute values)
    """
    gamma = w.abs().mean()
    w_scaled = w / (gamma + 1e-8)
    w_q = w_scaled.round().clamp(-1, 1)
    return w_q, gamma


class _TernaryQuantizeSTE(torch.autograd.Function):
    """Ternary quantization with Straight-Through Estimator.

    Forward: quantize to {-1, 0, +1}.
    Backward: pass gradients through as if quantization is identity.
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor) -> torch.Tensor:
        gamma = w.abs().mean()
        w_scaled = w / (gamma + 1e-8)
        w_q = w_scaled.round().clamp(-1, 1)
        ctx.save_for_backward(w)
        # Return quantized weights (unscaled — gamma applied separately)
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient straight through
        return grad_output


# ══════════════════════════════════════════════════════════════════════
# BitLinear — ternary linear with STE training + per-channel scale
# ══════════════════════════════════════════════════════════════════════


class BitLinear(nn.Module):
    """Linear layer with ternary weights (STE-trained) + per-channel gamma.

    Stores master weights in full precision (fp32). During forward:
      1. RMSNorm the input (stabilize activations)
      2. Quantize weights to {-1, 0, +1} via STE (gradient passes through)
      3. Compute absmean scale factor
      4. Matmul with ternary weights
      5. Scale by absmean × per-channel gamma

    The per-channel gamma (out_features,) gives extra expressiveness:
    each output dimension learns its own scale on top of the absmean.
    This lets the model amplify useful routing channels and silence
    others, beyond what the ternary pattern alone can express.

    During training: master weights + gamma both receive gradients via Adam.
    At inference: quantize once, fold gamma into scale, deploy at 2 bits.

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

        # Master weights — full precision, trained with STE
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Per-channel scale — learned alongside the ternary pattern
        self.gamma = nn.Parameter(torch.ones(out_features))

        # Pre-norm (RMSNorm on input before ternary matmul)
        if pre_norm:
            self.norm = BitRMSNorm(in_features)
        else:
            self.norm = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.ones_(self.gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Pre-norm input
        if self.norm is not None:
            x = self.norm(x)

        # 2. Quantize weights with STE (gradient passes through)
        w_q = _TernaryQuantizeSTE.apply(self.weight)

        # 3. Absmean scale
        absmean = self.weight.abs().mean()

        # 4. Ternary matmul + scale by absmean × per-channel gamma
        return F.linear(x, w_q) * (absmean * self.gamma)

    @torch.no_grad()
    def ternary_stats(self) -> dict[str, float]:
        """Report quantization and gamma statistics.

        Returns:
            sparsity: fraction of weights quantized to 0
            pos_frac: fraction quantized to +1
            neg_frac: fraction quantized to -1
            absmean: current absmean of master weights
            gamma_mean/std/min/max: per-channel scale statistics
        """
        w_q, absmean = _ternary_quantize(self.weight)
        total = w_q.numel()
        return {
            "sparsity": (w_q == 0).sum().item() / total,
            "pos_frac": (w_q == 1).sum().item() / total,
            "neg_frac": (w_q == -1).sum().item() / total,
            "absmean": absmean.item(),
            "gamma_mean": self.gamma.mean().item(),
            "gamma_std": self.gamma.std().item(),
            "gamma_min": self.gamma.min().item(),
            "gamma_max": self.gamma.max().item(),
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"pre_norm={self.pre_norm}, "
            f"bits=1.58+gamma"
        )


# ══════════════════════════════════════════════════════════════════════
# BitFFN — Ternary feed-forward network
# ══════════════════════════════════════════════════════════════════════


class BitFFN(nn.Module):
    """Feed-forward network with ternary weights.

    Pre-norm → BitLinear(up) → GELU → BitLinear(down)

    Both projections use STE-trained ternary weights + per-channel gamma.
    GELU activation stays fp32.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.up = BitLinear(d_model, d_ff, pre_norm=True)
        self.act = nn.GELU()
        self.down = BitLinear(d_ff, d_model, pre_norm=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.down(self.act(self.up(x))))
