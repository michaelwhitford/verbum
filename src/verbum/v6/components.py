"""Ternary VSM components — S4, S3, Meta-S4, Meta-S3 with BitLinear projections.

Surgical ternary port of the four VSM control components from vsm_lm_v5.py.
All projection weights (bias=False in v5) are replaced with BitLinear(pre_norm=False).
Normalization layers (nn.LayerNorm) are replaced with BitRMSNorm.

Preserved exactly:
  - Complex register arithmetic and interleaving
  - Phase-sensitive attention: Re(q·conj(k)) = q_r·k_r + q_i·k_i
  - Phase-coherent S3 alignment gating
  - write_gates (nn.Linear with bias — kept fp16, tiny, sigmoid-init)
  - temperature and learned_bias (nn.Parameter scalars — kept fp32)
  - MetaS3 gate_proj (nn.Linear with bias — kept fp16, small, sigmoid-init)
  - All forward signatures and return types

License: MIT
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from verbum.v6.bitlinear import BitLinear, BitRMSNorm


# ══════════════════════════════════════════════════════════════════════
# Helper functions (copied from vsm_lm_v5)
# ══════════════════════════════════════════════════════════════════════


def _interleave_complex(registers: list[torch.Tensor]) -> torch.Tensor:
    """Convert complex registers to interleaved real representation.

    Each complex register (d_register,) → (2*d_register,) real via
    view_as_real: [r0, i0, r1, i1, ...].
    """
    parts = []
    for reg in registers:
        parts.append(torch.view_as_real(reg).flatten())
    return torch.cat(parts, dim=-1)


def _interleave_banks(banks: list[list[torch.Tensor]]) -> torch.Tensor:
    """Flatten all banks' complex registers to interleaved real."""
    parts = []
    for bank in banks:
        parts.append(_interleave_complex(bank))
    return torch.cat(parts, dim=-1)


# ══════════════════════════════════════════════════════════════════════
# S4Ternary — Intelligence (complex-query register scan)
# ══════════════════════════════════════════════════════════════════════


class S4Ternary(nn.Module):
    """Register cross-attention with complex-valued queries — ternary weights.

    Registers are ℂ^d_register. Residual stream is ℝ^d_model.
    Phase-sensitive attention: Re(q·conj(k)) = q_r·k_r + q_i·k_i.

    Complex arithmetic decomposed into real ops for device compat:
      q_proj output → split even/odd → q_r, q_i (d_model//2 each)
      k_proj output → split even/odd → k_r, k_i (d_model//2 each)
      attn = (q_r·k_r + q_i·k_i) / √d_model

    Ternary changes from S4IntelligenceComplex:
      q_proj, k_proj, v_proj, summary_proj: nn.Linear → BitLinear(pre_norm=False)
      norm: nn.LayerNorm → BitRMSNorm
    """

    def __init__(
        self,
        d_model: int,
        d_register: int,
        n_registers: int = 3,
        max_banks: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_register = d_register
        self.n_registers = n_registers
        self.max_banks = max_banks
        self.scale = d_model ** -0.5

        # Input: interleaved real/imag of complex registers
        max_q_dim = max_banks * n_registers * d_register * 2
        self.q_proj = BitLinear(max_q_dim, d_model, pre_norm=False)
        self.k_proj = BitLinear(d_model, d_model, pre_norm=False)
        self.v_proj = BitLinear(d_model, d_model, pre_norm=False)
        # Output: interleaved real/imag for complex register updates
        self.summary_proj = BitLinear(d_model, n_registers * d_register * 2, pre_norm=False)
        self.norm = BitRMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        banks: list[list[torch.Tensor]],
        residual: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Scan residual conditioned on complex register banks."""
        B, L, D = residual.shape

        # Interleave all registers to real, zero-pad to max
        all_regs_real = []
        for bank in banks:
            for reg in bank:
                all_regs_real.append(torch.view_as_real(reg).flatten())

        max_regs = self.max_banks * self.n_registers
        pad_dim = self.d_register * 2
        while len(all_regs_real) < max_regs:
            all_regs_real.append(torch.zeros(pad_dim, device=residual.device))

        q_input = torch.cat(all_regs_real, dim=-1)

        # Complex query: split even/odd for Re(q·conj(k))
        q_flat = self.q_proj(q_input)        # (d_model,)
        q_r = q_flat[0::2]                   # (d_model//2,)
        q_i = q_flat[1::2]                   # (d_model//2,)

        x = self.norm(residual)
        k_flat = self.k_proj(x)              # (B, L, d_model)
        k_r = k_flat[..., 0::2]             # (B, L, d_model//2)
        k_i = k_flat[..., 1::2]             # (B, L, d_model//2)
        v = self.v_proj(x)                   # (B, L, d_model) — real

        # Phase-sensitive attention: Re(q · conj(k)) = q_r·k_r + q_i·k_i
        attn = (
            torch.einsum("d,bld->bl", q_r, k_r)
            + torch.einsum("d,bld->bl", q_i, k_i)
        ) * self.scale

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of real values
        summary = torch.einsum("bl,bld->bd", attn_weights, v)
        summary = summary.mean(dim=0)  # (d_model,)

        # Complex register updates
        updates_flat = self.summary_proj(summary)  # (n_reg * d_reg * 2,)
        update_splits = updates_flat.split(self.d_register * 2, dim=-1)
        updates = [
            torch.view_as_complex(u.reshape(-1, 2))
            for u in update_splits
        ]

        return updates, attn_weights.detach()


# ══════════════════════════════════════════════════════════════════════
# S3Ternary — Phase-Coherent Gating (alignment-based scalar gate)
# ══════════════════════════════════════════════════════════════════════


class S3Ternary(nn.Module):
    """Phase-coherent control for a single level-pass — ternary weights.

    Scalar alignment gate: gate opens when register direction matches
    delta direction. Replaces v4.1's per-dimension gating.

      reg_dir = normalize(concat(real, imag))
      delta_dir = normalize(mean(delta))
      alignment = proj_align(reg_dir) · proj_delta(delta_dir)
      gate = σ(alignment · temperature + learned_bias)

    Temperature (init=1.0): sharpen or soften gating.
    Learned bias (init=0.0): fallback when registers immature.
    At init: alignment ≈ 0, gate ≈ σ(0) = 0.5 (pass-through).

    Register writes produce complex updates: separate real/imag projections.

    Ternary changes from S3PhaseCoherent:
      proj_align, proj_delta: nn.Linear → BitLinear(pre_norm=False)
      write_proj_real, write_proj_imag: nn.Linear → BitLinear(pre_norm=False)
      write_gates: KEPT as nn.Linear (has bias, 513 params each, fp16)
      temperature, learned_bias: KEPT as nn.Parameter (scalar, fp32)
    """

    def __init__(
        self,
        d_model: int,
        d_register: int,
        n_phases: int = 3,
        n_registers: int = 3,
        d_align: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_register = d_register
        self.n_phases = n_phases
        self.n_registers = n_registers

        reg_real_dim = 2 * d_register * n_registers  # interleaved real dim

        # Alignment projections (per phase) — ternary
        self.proj_align = nn.ModuleList([
            BitLinear(reg_real_dim, d_align, pre_norm=False)
            for _ in range(n_phases)
        ])
        self.proj_delta = nn.ModuleList([
            BitLinear(d_model, d_align, pre_norm=False)
            for _ in range(n_phases)
        ])

        # Learnable temperature and bias (per phase) — fp32 scalars, unchanged
        self.temperature = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(n_phases)
        ])
        self.learned_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(n_phases)
        ])

        # Complex register write (per phase × per register) — ternary
        self.write_proj_real = nn.ModuleList([
            BitLinear(d_model, d_register, pre_norm=False)
            for _ in range(n_phases * n_registers)
        ])
        self.write_proj_imag = nn.ModuleList([
            BitLinear(d_model, d_register, pre_norm=False)
            for _ in range(n_phases * n_registers)
        ])
        # write_gates: KEPT as nn.Linear — has bias, tiny (513 params each),
        # and the sigmoid default relies on the bias being near zero.
        self.write_gates = nn.ModuleList([
            nn.Linear(d_model, 1)
            for _ in range(n_phases * n_registers)
        ])

    def gate_phase(
        self,
        registers: list[torch.Tensor],
        delta: torch.Tensor,
        phase_idx: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, list[float]]:
        """Gate a phase's output using alignment-based scalar gate.

        Args:
            registers: list of n_registers complex tensors, each (d_register,)
            delta: (B, L, d_model) real tensor
            phase_idx: which phase (0=prep, 1=converge, 2=consolidate)

        Returns:
            gated_delta: (B, L, d_model)
            updated_registers: list of n_registers complex tensors
            gate_value: scalar gate (detached)
            write_gate_values: list of floats
        """
        eps = 1e-8

        # Flatten complex registers to real
        reg_flat = _interleave_complex(registers)  # (2 * n_reg * d_reg,)
        reg_dir = reg_flat / (reg_flat.norm() + eps)

        # Delta summary and direction
        summary = delta.mean(dim=(0, 1))  # (d_model,)
        delta_dir = summary / (summary.norm() + eps)

        # Alignment score → scalar gate
        reg_proj = self.proj_align[phase_idx](reg_dir)      # (d_align,)
        delta_proj = self.proj_delta[phase_idx](delta_dir)   # (d_align,)
        alignment = (reg_proj * delta_proj).sum()            # scalar

        gate = torch.sigmoid(
            alignment * self.temperature[phase_idx]
            + self.learned_bias[phase_idx]
        )
        gated_delta = gate * delta  # scalar broadcasts to (B, L, d_model)

        # Complex register updates
        updated_registers = []
        write_gate_values = []
        for reg_idx in range(self.n_registers):
            write_idx = phase_idx * self.n_registers + reg_idx
            wg = torch.sigmoid(self.write_gates[write_idx](summary))
            update_r = self.write_proj_real[write_idx](summary)  # (d_register,)
            update_i = self.write_proj_imag[write_idx](summary)  # (d_register,)
            update = torch.complex(update_r, update_i)           # (d_register,) complex
            updated_registers.append(registers[reg_idx] + wg * update)
            write_gate_values.append(wg.item())

        return gated_delta, updated_registers, gate, write_gate_values


# ══════════════════════════════════════════════════════════════════════
# MetaS4Ternary — Final structural summary (complex-query)
# ══════════════════════════════════════════════════════════════════════


class MetaS4Ternary(nn.Module):
    """Final intelligence scan with complex-query attention — ternary weights.

    Same phase-sensitive mechanism as S4Ternary: Re(q·conj(k)).
    Reads most-refined register banks, produces real residual update.

    Ternary changes from MetaS4Complex:
      q_proj, k_proj, v_proj, out_proj: nn.Linear → BitLinear(pre_norm=False)
      norm: nn.LayerNorm → BitRMSNorm
    """

    def __init__(
        self,
        d_model: int,
        d_register: int,
        n_registers: int = 3,
        n_banks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_register = d_register
        self.n_registers = n_registers
        self.n_banks = n_banks
        self.scale = d_model ** -0.5

        total_reg_dim = n_banks * n_registers * d_register * 2  # interleaved
        self.q_proj = BitLinear(total_reg_dim, d_model, pre_norm=False)
        self.k_proj = BitLinear(d_model, d_model, pre_norm=False)
        self.v_proj = BitLinear(d_model, d_model, pre_norm=False)
        self.out_proj = BitLinear(d_model, d_model, pre_norm=False)
        self.norm = BitRMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        meta_banks: list[list[torch.Tensor]],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = residual.shape

        q_input = _interleave_banks(meta_banks)

        # Complex query decomposed into real ops
        q_flat = self.q_proj(q_input)        # (d_model,)
        q_r = q_flat[0::2]                   # (d_model//2,)
        q_i = q_flat[1::2]                   # (d_model//2,)

        x = self.norm(residual)
        k_flat = self.k_proj(x)              # (B, L, d_model)
        k_r = k_flat[..., 0::2]
        k_i = k_flat[..., 1::2]
        v = self.v_proj(x)                   # (B, L, d_model) real

        # Phase-sensitive attention
        attn = (
            torch.einsum("d,bld->bl", q_r, k_r)
            + torch.einsum("d,bld->bl", q_i, k_i)
        ) * self.scale

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        summary = torch.einsum("bl,bld->bd", attn_weights, v)
        out = self.out_proj(summary).unsqueeze(1).expand_as(residual)
        return residual + out


# ══════════════════════════════════════════════════════════════════════
# MetaS3Ternary — Cross-level contribution gate (complex register banks)
# ══════════════════════════════════════════════════════════════════════


class MetaS3Ternary(nn.Module):
    """Top-level resource allocation reading complex register banks.

    Identical to MetaS3Complex: gate_proj is kept as nn.Linear because:
      - It has bias (needed for sigmoid default — gates should start at ~0.5)
      - It is only ~23K params
      - Not worth adding a separate bias parameter for BitLinear

    Created for naming consistency with the other ternary components.
    """

    def __init__(self, d_register: int, n_registers: int, n_banks: int, n_passes: int):
        super().__init__()
        input_dim = n_banks * n_registers * d_register * 2  # interleaved real
        self.gate_proj = nn.Linear(input_dim, n_passes)

    def forward(self, all_banks: list[list[torch.Tensor]]) -> torch.Tensor:
        flat = _interleave_banks(all_banks)
        return torch.sigmoid(self.gate_proj(flat))
