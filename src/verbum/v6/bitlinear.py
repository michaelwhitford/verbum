"""BitLinear — Native ternary routing with learned continuous control.

The ternary weights {-1, 0, +1} define a fixed circuit topology:
which neurons connect positively, negatively, or not at all. They are
initialized once (Kaiming → absmean quantize) and frozen.

The learning happens in the CONTINUOUS CONTROL HIERARCHY above them:
  - per-channel gamma: amplify/silence individual routing channels
  - RMSNorm gains: scale activations entering each layer
  - S3 alignment gates: control phase-level information flow
  - S3 temperature/bias: sharpen/soften gating decisions
  - Meta-S3 gates: control pass-level contribution
  - Embeddings, registers, norms: the semantic substrate

This is NOT quantization. There are no master weights, no STE, no
quantization step in the forward pass. The matmul is pure ternary:
  y_j = Σ_{w=+1} x_i - Σ_{w=-1} x_i   (additions/subtractions only)

The per-channel gamma then scales each output dimension:
  y = (x @ W_ternary^T) * gamma

The VSM's 5-pass recursive structure means each ternary circuit is
used 5 times with different gating each time. 28M continuous params
learn to ROUTE THROUGH 35.3M fixed ternary connections.

Training: 561 MB (ternary buffers + continuous trained with Adam)
Inference: 61 MB (ternary at 2 bits + fp16)

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
# Ternary initialization
# ══════════════════════════════════════════════════════════════════════


def _ternary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to {-1, 0, +1} using absmean scaling.

    Returns:
        w_q: ternary weight tensor {-1, 0, +1}
        gamma: per-channel scale factors (out_features,)
    """
    gamma = w.abs().mean(dim=-1)  # (out_features,)
    w_scaled = w / (gamma.unsqueeze(-1) + 1e-8)
    w_q = w_scaled.round().clamp(-1, 1)
    return w_q, gamma


# ══════════════════════════════════════════════════════════════════════
# BitLinear — native ternary routing + learned per-channel scale
# ══════════════════════════════════════════════════════════════════════


class BitLinear(nn.Module):
    """Linear layer with native ternary routing and learned per-channel scale.

    The ternary weight defines a fixed routing topology. The per-channel
    gamma controls how much each output dimension contributes. Together
    with the VSM's continuous control hierarchy (S3 gates, meta-S3,
    temperature/bias, register system), the model learns to use the
    fixed topology effectively.

    Initialization:
      1. Generate fp32 weights with Kaiming uniform
      2. Quantize to {-1, 0, +1} via per-channel absmean
      3. Store ternary pattern as buffer (frozen, no grad)
      4. Store per-channel gamma as parameter (trained)

    Forward:
      y = RMSNorm(x) @ W_ternary^T * gamma
      (no quantization step — pure ternary matmul + scale)

    Trainable: gamma (out_features) + norm.weight (in_features)
    Frozen: ternary_weight (out_features × in_features)
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

        if pre_norm:
            self.norm = BitRMSNorm(in_features)
        else:
            self.norm = None

        # Initialize: Kaiming → quantize → freeze ternary, learn gamma
        w_init = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w_init, a=math.sqrt(5))
        w_q, gamma = _ternary_quantize(w_init)

        # Frozen ternary routing (no grad, no optimizer)
        self.register_buffer("ternary_weight", w_q)

        # Learned per-channel scale
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            x = self.norm(x)
        return F.linear(x, self.ternary_weight) * self.gamma

    @torch.no_grad()
    def ternary_stats(self) -> dict[str, float]:
        """Report ternary weight and gamma statistics."""
        w = self.ternary_weight
        total = w.numel()
        return {
            "sparsity": (w == 0).sum().item() / total,
            "pos_frac": (w == 1).sum().item() / total,
            "neg_frac": (w == -1).sum().item() / total,
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
            f"ternary={self.ternary_weight.numel()} frozen, "
            f"gamma={self.gamma.numel()} trained"
        )


# ══════════════════════════════════════════════════════════════════════
# BitFFN — Ternary feed-forward network
# ══════════════════════════════════════════════════════════════════════


class BitFFN(nn.Module):
    """Feed-forward network with native ternary routing.

    Pre-norm → BitLinear(up) → GELU → BitLinear(down) + residual
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
