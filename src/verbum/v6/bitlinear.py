"""BitLinear — Ternary routing that learns through flip accumulation.

The ternary weights {-1, 0, +1} define routing topology. They evolve
during training through a lightweight accumulate-and-flip mechanism:

  1. Forward: pure ternary matmul (x @ W_ternary) * gamma
  2. Backward: STE computes gradient for ternary weights
  3. Gradient routes to a flip accumulator (not to the optimizer)
  4. Periodically: weights whose accumulator exceeds threshold FLIP
     one step (-1→0, 0→+1, +1→0, etc.) and the accumulator resets

This gives ternary weights that LEARN useful routing patterns, without
maintaining fp32 master weights or Adam optimizer state for them.
The flip accumulator is the only overhead: 4 bytes per ternary weight.

Per ternary weight: 4 bytes (fp32 value) + 4 bytes (accumulator) = 8 bytes
vs STE + Adam:      4 bytes (master) + 4+4 (Adam m,v) + 4 (grad) = 16 bytes
vs frozen:          4 bytes (buffer) + 0 = 4 bytes (but doesn't learn!)

The per-channel gamma (out_features,) provides continuous fine-tuning
on top of the discrete ternary routing. Gamma is trained normally with
Adam via the optimizer.

License: MIT
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# RMSNorm
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
    """Quantize weights to {-1, 0, +1} using per-channel absmean.

    Returns:
        w_q: ternary weight tensor {-1, 0, +1}
        gamma: per-channel scale factors (out_features,)
    """
    gamma = w.abs().mean(dim=-1)
    w_scaled = w / (gamma.unsqueeze(-1) + 1e-8)
    w_q = w_scaled.round().clamp(-1, 1)
    return w_q, gamma


# ══════════════════════════════════════════════════════════════════════
# BitLinear — ternary routing with flip accumulation
# ══════════════════════════════════════════════════════════════════════


class BitLinear(nn.Module):
    """Linear layer with learnable ternary routing via flip accumulation.

    Initialization:
      1. Generate fp32 weights with Kaiming uniform
      2. Quantize to {-1, 0, +1} via per-channel absmean
      3. Store as nn.Parameter (autograd computes gradient via STE)
      4. Store per-channel gamma as separate nn.Parameter
      5. Create flip accumulator buffer (same shape as weights)

    Forward:
      y = RMSNorm(x) @ W_ternary^T * gamma

    Training loop (managed by model, not optimizer):
      - After backward: ternary gradient → flip_accum, then zero grad
      - Periodically: where |accum| > threshold → flip weight, reset
      - Optimizer only sees gamma + norm (via model.continuous_parameters())

    The ternary weights evolve through discrete flips, not continuous
    gradient descent. Each flip moves one step: -1→0, 0→±1, ±1→0.
    The accumulator captures gradient pressure; the threshold controls
    how much evidence is needed before committing to a flip.
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

        # Initialize: Kaiming → quantize → ternary param + gamma param
        w_init = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w_init, a=math.sqrt(5))
        w_q, gamma = _ternary_quantize(w_init)

        # Ternary routing — Parameter so autograd computes gradient,
        # but NOT passed to optimizer. Gradient routes to flip_accum.
        self.ternary_weight = nn.Parameter(w_q)

        # Flip accumulator — tracks gradient pressure for each weight
        self.register_buffer("flip_accum", torch.zeros_like(w_q))

        # Per-channel scale — trained normally via optimizer
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            x = self.norm(x)
        return F.linear(x, self.ternary_weight) * self.gamma

    def accumulate(self) -> None:
        """Route ternary gradient to flip accumulator, then zero grad.

        Call after loss.backward(), before optimizer.step().
        """
        if self.ternary_weight.grad is not None:
            self.flip_accum.add_(self.ternary_weight.grad)
            self.ternary_weight.grad = None

    @torch.no_grad()
    def flip_step(self, threshold: float) -> int:
        """Flip ternary weights where accumulated gradient exceeds threshold.

        Each flip moves one step in the gradient direction:
          -1 + positive pressure → 0
           0 + positive pressure → +1
          +1 + negative pressure → 0
           0 + negative pressure → -1

        Returns number of weights flipped.
        """
        mask = self.flip_accum.abs() > threshold
        n_flipped = mask.sum().item()

        if n_flipped > 0:
            direction = self.flip_accum[mask].sign()
            current = self.ternary_weight.data[mask]
            new_vals = (current + direction).clamp(-1, 1).round()
            self.ternary_weight.data[mask] = new_vals
            self.flip_accum[mask] = 0.0

        return int(n_flipped)

    @torch.no_grad()
    def ternary_stats(self) -> dict[str, float]:
        """Report ternary weight, gamma, and accumulator statistics."""
        w = self.ternary_weight.data
        total = w.numel()
        return {
            "sparsity": (w == 0).sum().item() / total,
            "pos_frac": (w == 1).sum().item() / total,
            "neg_frac": (w == -1).sum().item() / total,
            "gamma_mean": self.gamma.mean().item(),
            "gamma_std": self.gamma.std().item(),
            "accum_mean": self.flip_accum.abs().mean().item(),
            "accum_max": self.flip_accum.abs().max().item(),
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"pre_norm={self.pre_norm}, "
            f"ternary={self.ternary_weight.numel()} (flip-learnable), "
            f"gamma={self.gamma.numel()}"
        )


# ══════════════════════════════════════════════════════════════════════
# BitFFN — Ternary feed-forward network
# ══════════════════════════════════════════════════════════════════════


class BitFFN(nn.Module):
    """Feed-forward network with learnable ternary routing.

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
