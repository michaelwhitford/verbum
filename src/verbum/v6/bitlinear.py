"""BitLinear — Native ternary weights with learned per-channel scales.

Ternary weights {-1, 0, +1} are initialized once (from Kaiming → quantize)
and FROZEN. No fp32 master weights, no STE, no quantization overhead in the
forward pass. The matmul is just additions and subtractions.

The only trainable parameters per layer are:
  - gamma: per-channel scale factor (out_features,) — controls how much
    each output dimension of the ternary routing contributes
  - norm.weight: RMSNorm gain (in_features,) — controls input magnitude

This is NOT quantization-aware training. The ternary weights define a fixed
random routing topology. Training adjusts the gain knobs (gamma, norms) and
the non-ternary components (embeddings, gates, registers) to exploit the
fixed routes. Like a randomly wired circuit with learnable amplifiers.

Training implications:
  - 35.3M ternary weights → 8.4 MB buffer (no optimizer state, no gradients)
  - ~88K gamma params + ~88K norm params → fully trained with Adam
  - ~28M fp16 params (embeddings, gates, etc.) → fully trained
  - Total training memory: ~590 MB (vs 964 MB with QAT, vs 1012 MB for v5)
  - Forward pass: FASTER than QAT (no quantize step)

The per-channel gamma allows the model to learn which output dimensions
of each routing layer matter. A gamma near zero effectively silences that
dimension's routing. This is richer than a single per-layer scalar.

References:
  - Lottery ticket hypothesis (Frankle & Carlin, 2019)
  - Random projections / reservoir computing
  - BitNet b1.58 (Ma et al., 2024) for the ternary init distribution

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
# Ternary initialization
# ══════════════════════════════════════════════════════════════════════


def _ternary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to {-1, 0, +1} using absmean scaling.

    Args:
        w: full-precision weight tensor (out_features, in_features)

    Returns:
        w_q: ternary weight tensor {-1, 0, +1}
        gamma: per-channel scale factors (out_features,)
    """
    # Per-channel absmean (one gamma per output neuron)
    gamma = w.abs().mean(dim=-1)  # (out_features,)
    w_scaled = w / (gamma.unsqueeze(-1) + 1e-8)
    w_q = w_scaled.round().clamp(-1, 1)
    return w_q, gamma


# ══════════════════════════════════════════════════════════════════════
# BitLinear — native ternary with learned per-channel scales
# ══════════════════════════════════════════════════════════════════════


class BitLinear(nn.Module):
    """Linear layer with frozen ternary weights and learned per-channel scales.

    Initialization:
      1. Generate fp32 weights with Kaiming uniform
      2. Quantize to {-1, 0, +1} via per-channel absmean
      3. Freeze the ternary pattern as a buffer (no grad, no optimizer)
      4. Store the per-channel gamma as a learnable parameter

    Forward:
      y = RMSNorm(x) @ W_q^T * gamma

    The matmul with W_q is mathematically additions/subtractions only.
    Gamma broadcasts per output channel, scaling each routing dimension.

    Trainable parameters: gamma (out_features) + norm.weight (in_features).
    Frozen: ternary_weight buffer (out_features × in_features).

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

        # Pre-norm (RMSNorm on input before ternary matmul)
        if pre_norm:
            self.norm = BitRMSNorm(in_features)
        else:
            self.norm = None

        # Initialize: Kaiming → quantize → freeze ternary, learn gamma
        w_init = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w_init, a=math.sqrt(5))
        w_q, gamma = _ternary_quantize(w_init)

        # Frozen ternary routing pattern (no grad, no optimizer state)
        self.register_buffer("ternary_weight", w_q)

        # Learnable per-channel scale (the only trainable weight in this layer)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Pre-norm input
        if self.norm is not None:
            x = self.norm(x)

        # 2. Ternary matmul (additions/subtractions) + per-channel scale
        return F.linear(x, self.ternary_weight) * self.gamma

    @torch.no_grad()
    def ternary_stats(self) -> dict[str, float]:
        """Report ternary weight and gamma statistics.

        Returns:
            sparsity: fraction of ternary weights that are 0 (fixed)
            pos_frac: fraction that are +1 (fixed)
            neg_frac: fraction that are -1 (fixed)
            gamma_mean: mean of per-channel scales (evolves during training)
            gamma_std: std of per-channel scales
            gamma_min: minimum scale
            gamma_max: maximum scale
        """
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
            f"frozen_ternary={self.ternary_weight.numel()}, "
            f"trainable_gamma={self.gamma.numel()}"
        )


# ══════════════════════════════════════════════════════════════════════
# BitFFN — Ternary feed-forward network
# ══════════════════════════════════════════════════════════════════════


class BitFFN(nn.Module):
    """Feed-forward network with frozen ternary weights.

    Pre-norm → BitLinear(up) → GELU → BitLinear(down)

    Both projections use frozen ternary routing + learned per-channel
    scales. The GELU activation stays fp32.
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
