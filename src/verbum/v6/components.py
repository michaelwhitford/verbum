"""VSM control components — S4, S3, MetaS4, MetaS3 with TernaryLinear — MLX.

Faithful port of the PyTorch v6 components. All projection weights
use TernaryLinear. Complex register arithmetic preserved exactly.

Kept as fp16/fp32 (not ternary):
  - write_gates (nn.Linear with bias, tiny, sigmoid-init)
  - temperature and learned_bias (scalar parameters)
  - MetaS3 gate_proj (nn.Linear with bias, small)

License: MIT
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from verbum.v6.ternary import TernaryLinear


# ══════════════════════════════════════════════════════════════════════
# Helpers — complex register interleaving
# ══════════════════════════════════════════════════════════════════════


def _interleave_complex(registers: list[mx.array]) -> mx.array:
    """Convert complex registers to interleaved real: [r0,i0,r1,i1,...]."""
    parts = []
    for reg in registers:
        real = mx.real(reg)
        imag = mx.imag(reg)
        interleaved = mx.stack([real, imag], axis=-1).reshape(-1)
        parts.append(interleaved)
    return mx.concatenate(parts, axis=-1)


def _interleave_banks(banks: list[list[mx.array]]) -> mx.array:
    """Flatten all banks' complex registers to interleaved real."""
    parts = []
    for bank in banks:
        parts.append(_interleave_complex(bank))
    return mx.concatenate(parts, axis=-1)


# ══════════════════════════════════════════════════════════════════════
# S4 — Intelligence (complex-query register scan)
# ══════════════════════════════════════════════════════════════════════


class S4Ternary(nn.Module):
    """Register cross-attention with complex-valued queries.

    Phase-sensitive attention: Re(q·conj(k)) = q_r·k_r + q_i·k_i
    Decomposed into real ops for device compat.
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

        max_q_dim = max_banks * n_registers * d_register * 2
        self.q_proj = TernaryLinear(max_q_dim, d_model, pre_norm=False)
        self.k_proj = TernaryLinear(d_model, d_model, pre_norm=False)
        self.v_proj = TernaryLinear(d_model, d_model, pre_norm=False)
        self.summary_proj = TernaryLinear(d_model, n_registers * d_register * 2, pre_norm=False)
        self.norm = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        banks: list[list[mx.array]],
        residual: mx.array,
    ) -> tuple[list[mx.array], mx.array]:
        B, L, D = residual.shape

        # Interleave all registers, zero-pad to max
        all_regs_real = []
        for bank in banks:
            for reg in bank:
                real = mx.real(reg)
                imag = mx.imag(reg)
                all_regs_real.append(mx.stack([real, imag], axis=-1).reshape(-1))

        max_regs = self.max_banks * self.n_registers
        pad_dim = self.d_register * 2
        while len(all_regs_real) < max_regs:
            all_regs_real.append(mx.zeros((pad_dim,)))

        q_input = mx.concatenate(all_regs_real, axis=-1)

        # Complex query decomposed
        q_flat = self.q_proj(q_input)
        q_r = q_flat[0::2]
        q_i = q_flat[1::2]

        x = self.norm(residual)
        k_flat = self.k_proj(x)
        k_r = k_flat[..., 0::2]
        k_i = k_flat[..., 1::2]
        v = self.v_proj(x)

        # Phase-sensitive attention: Re(q·conj(k))
        attn = (q_r[None, None, :] * k_r + q_i[None, None, :] * k_i).sum(axis=-1) * self.scale
        attn_weights = mx.softmax(attn, axis=-1)
        attn_weights = self.dropout(attn_weights)

        summary = (attn_weights[:, :, None] * v).sum(axis=1)
        summary = summary.mean(axis=0)

        # Complex register updates
        updates_flat = self.summary_proj(summary)
        updates = []
        for i in range(self.n_registers):
            start = i * self.d_register * 2
            end = start + self.d_register * 2
            u_flat = updates_flat[start:end]
            u_real = u_flat[0::2]
            u_imag = u_flat[1::2]
            updates.append(u_real + 1j * u_imag)

        return updates, mx.stop_gradient(attn_weights)


# ══════════════════════════════════════════════════════════════════════
# S3 — Phase-Coherent Gating
# ══════════════════════════════════════════════════════════════════════


class S3Ternary(nn.Module):
    """Phase-coherent control for a single level-pass.

    Scalar alignment gate based on register-delta direction match.
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

        reg_real_dim = 2 * d_register * n_registers

        # Alignment projections — ternary
        self.proj_align = [
            TernaryLinear(reg_real_dim, d_align, pre_norm=False)
            for _ in range(n_phases)
        ]
        self.proj_delta = [
            TernaryLinear(d_model, d_align, pre_norm=False)
            for _ in range(n_phases)
        ]

        # Temperature and bias — fp32 scalars
        self.temperature = [mx.ones((1,)) for _ in range(n_phases)]
        self.learned_bias = [mx.zeros((1,)) for _ in range(n_phases)]

        # Complex register write projections — ternary
        self.write_proj_real = [
            TernaryLinear(d_model, d_register, pre_norm=False)
            for _ in range(n_phases * n_registers)
        ]
        self.write_proj_imag = [
            TernaryLinear(d_model, d_register, pre_norm=False)
            for _ in range(n_phases * n_registers)
        ]
        # write_gates: kept as nn.Linear (has bias, tiny)
        self.write_gates = [
            nn.Linear(d_model, 1)
            for _ in range(n_phases * n_registers)
        ]

    def gate_phase(
        self,
        registers: list[mx.array],
        delta: mx.array,
        phase_idx: int,
    ) -> tuple[mx.array, list[mx.array], mx.array, list[float]]:
        """Gate a phase's output using alignment-based scalar gate."""
        eps = 1e-8

        reg_flat = _interleave_complex(registers)
        reg_dir = reg_flat / (mx.sqrt((reg_flat * reg_flat).sum()) + eps)

        summary = delta.mean(axis=(0, 1))
        delta_dir = summary / (mx.sqrt((summary * summary).sum()) + eps)

        reg_proj = self.proj_align[phase_idx](reg_dir)
        delta_proj = self.proj_delta[phase_idx](delta_dir)
        alignment = (reg_proj * delta_proj).sum()

        gate = mx.sigmoid(
            alignment * self.temperature[phase_idx]
            + self.learned_bias[phase_idx]
        )
        gated_delta = gate * delta

        # Complex register updates
        updated_registers = []
        write_gate_values = []
        for reg_idx in range(self.n_registers):
            write_idx = phase_idx * self.n_registers + reg_idx
            wg = mx.sigmoid(self.write_gates[write_idx](summary))
            update_r = self.write_proj_real[write_idx](summary)
            update_i = self.write_proj_imag[write_idx](summary)
            update = update_r + 1j * update_i
            updated_registers.append(registers[reg_idx] + wg * update)
            write_gate_values.append(wg.item())

        return gated_delta, updated_registers, gate, write_gate_values


# ══════════════════════════════════════════════════════════════════════
# MetaS4 — Final structural summary
# ══════════════════════════════════════════════════════════════════════


class MetaS4Ternary(nn.Module):
    """Final intelligence scan with complex-query attention."""

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

        total_reg_dim = n_banks * n_registers * d_register * 2
        self.q_proj = TernaryLinear(total_reg_dim, d_model, pre_norm=False)
        self.k_proj = TernaryLinear(d_model, d_model, pre_norm=False)
        self.v_proj = TernaryLinear(d_model, d_model, pre_norm=False)
        self.out_proj = TernaryLinear(d_model, d_model, pre_norm=False)
        self.norm = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        meta_banks: list[list[mx.array]],
        residual: mx.array,
    ) -> mx.array:
        B, L, D = residual.shape

        q_input = _interleave_banks(meta_banks)

        q_flat = self.q_proj(q_input)
        q_r = q_flat[0::2]
        q_i = q_flat[1::2]

        x = self.norm(residual)
        k_flat = self.k_proj(x)
        k_r = k_flat[..., 0::2]
        k_i = k_flat[..., 1::2]
        v = self.v_proj(x)

        attn = (q_r[None, None, :] * k_r + q_i[None, None, :] * k_i).sum(axis=-1) * self.scale
        attn_weights = mx.softmax(attn, axis=-1)
        attn_weights = self.dropout(attn_weights)

        summary = (attn_weights[:, :, None] * v).sum(axis=1)
        out = self.out_proj(summary)
        out = mx.broadcast_to(out[:, None, :], residual.shape)
        return residual + out


# ══════════════════════════════════════════════════════════════════════
# MetaS3 — Cross-level contribution gates
# ══════════════════════════════════════════════════════════════════════


class MetaS3Ternary(nn.Module):
    """Top-level per-pass contribution gates from complex register banks.

    gate_proj kept as nn.Linear (has bias, needed for sigmoid default).
    """

    def __init__(self, d_register: int, n_registers: int, n_banks: int, n_passes: int):
        super().__init__()
        input_dim = n_banks * n_registers * d_register * 2
        self.gate_proj = nn.Linear(input_dim, n_passes)

    def __call__(self, all_banks: list[list[mx.array]]) -> mx.array:
        flat = _interleave_banks(all_banks)
        return mx.sigmoid(self.gate_proj(flat))
