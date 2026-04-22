"""VSM-LM v5 — Spiral Attention Bias + Complex Registers.

v5 introduces four topology changes over v4.1:

  1. Spiral attention bias: bias(w) = -α·ln(stride·w + 1)
     Power-law distance decay. weight ∝ 1/(distance+1)^α.
     Smooth attention landscape across stride boundaries.
     α=1.18 (R²=0.997 empirical fit). Zero new parameters.

  2. Complex-valued registers: ℂ^128 replaces ℝ^256.
     Phase angle encodes a new degree of freedom beyond magnitude.
     S4 uses phase-sensitive attention: Re(q·conj(k)) = q_r·k_r + q_i·k_i.
     Residual stream stays real. Only registers carry phase.

  3. Phase-coherent S3 gating: scalar alignment gate.
     gate = σ(alignment · temperature + bias)
     alignment = proj_reg(reg_dir) · proj_delta(delta_dir)
     Geometric: gate opens when register direction matches delta direction.
     Learnable temperature (init=1.0) and bias (init=0.0).

  4. Multiplicative modulation: replaces additive composition.
     modulation = 1 + gate · tanh(proj(delta))
     x_new = x · modulation
     Zero-init proj → identity at start. S5 coherent (3 shared projs).
     The chain x · m₁ · m₂ · ... · mₙ produces power-law magnitude.

Same architecture otherwise:
  ASCENDING:   L0↑ → L1↑ → L2    (build structural summaries)
  DESCENDING:  L1↓ → L0↓          (refine with high-level context)
  5 level-passes, 6 register banks, shared weights (S5 coherence).
  ~65.5M params (within 0.01% of v4.1).

All complex arithmetic decomposed into real operations for MPS compat.

License: MIT
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verbum.compressor_lm import CompressorLayer


# ══════════════════════════════════════════════════════════════════════
# Utilities
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
# FFN-only layer (same as v4.1, copied for isolation)
# ══════════════════════════════════════════════════════════════════════


class FFNLayer(nn.Module):
    """Pre-norm FFN layer without attention."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


# ══════════════════════════════════════════════════════════════════════
# S4 — Intelligence (complex-query register scan)
# ══════════════════════════════════════════════════════════════════════


class S4IntelligenceComplex(nn.Module):
    """Register cross-attention with complex-valued queries.

    Registers are ℂ^d_register. Residual stream is ℝ^d_model.
    Phase-sensitive attention: Re(q·conj(k)) = q_r·k_r + q_i·k_i.

    Complex arithmetic decomposed into real ops for device compat:
      q_proj output → split even/odd → q_r, q_i (d_model//2 each)
      k_proj output → split even/odd → k_r, k_i (d_model//2 each)
      attn = (q_r·k_r + q_i·k_i) / √d_model
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
        self.q_proj = nn.Linear(max_q_dim, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # Output: interleaved real/imag for complex register updates
        self.summary_proj = nn.Linear(d_model, n_registers * d_register * 2, bias=False)
        self.norm = nn.LayerNorm(d_model)
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
# S3 — Phase-Coherent Gating (alignment-based scalar gate)
# ══════════════════════════════════════════════════════════════════════


class S3PhaseCoherent(nn.Module):
    """Phase-coherent control for a single level-pass.

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

        # Alignment projections (per phase)
        self.proj_align = nn.ModuleList([
            nn.Linear(reg_real_dim, d_align, bias=False)
            for _ in range(n_phases)
        ])
        self.proj_delta = nn.ModuleList([
            nn.Linear(d_model, d_align, bias=False)
            for _ in range(n_phases)
        ])

        # Learnable temperature and bias (per phase)
        self.temperature = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(n_phases)
        ])
        self.learned_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(n_phases)
        ])

        # Complex register write (per phase × per register)
        self.write_proj_real = nn.ModuleList([
            nn.Linear(d_model, d_register, bias=False)
            for _ in range(n_phases * n_registers)
        ])
        self.write_proj_imag = nn.ModuleList([
            nn.Linear(d_model, d_register, bias=False)
            for _ in range(n_phases * n_registers)
        ])
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
# Meta-S3 — Cross-level contribution gate (complex register banks)
# ══════════════════════════════════════════════════════════════════════


class MetaS3Complex(nn.Module):
    """Top-level resource allocation reading complex register banks."""

    def __init__(self, d_register: int, n_registers: int, n_banks: int, n_passes: int):
        super().__init__()
        input_dim = n_banks * n_registers * d_register * 2  # interleaved real
        self.gate_proj = nn.Linear(input_dim, n_passes)

    def forward(self, all_banks: list[list[torch.Tensor]]) -> torch.Tensor:
        flat = _interleave_banks(all_banks)
        return torch.sigmoid(self.gate_proj(flat))


# ══════════════════════════════════════════════════════════════════════
# Meta-S4 — Final structural summary (complex-query)
# ══════════════════════════════════════════════════════════════════════


class MetaS4Complex(nn.Module):
    """Final intelligence scan with complex-query attention.

    Same phase-sensitive mechanism as S4: Re(q·conj(k)).
    Reads most-refined register banks, produces real residual update.
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
        self.q_proj = nn.Linear(total_reg_dim, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
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
# VSM-LM v5 — Spiral + Complex Registers + Phase-Coherent Gating
# ══════════════════════════════════════════════════════════════════════


class VSMLMV5(nn.Module):
    """Viable System Model Language Model — v5 topology.

    v4.1 architecture with three topology changes:
    1. Spiral attention bias (α=1.18) on all strided attention.
    2. Complex-valued registers (ℂ^128 = ℝ^256 equivalent).
    3. Phase-coherent S3 gating (scalar alignment gate).
    """

    REGISTER_NAMES = ("type", "scope", "role")
    PHASE_NAMES = ("prep", "converge", "consolidate")
    N_LEVELS = 3
    N_PASSES = 5
    PASS_NAMES = ("L0_asc", "L1_asc", "L2_apex", "L1_desc", "L0_desc")

    def __init__(
        self,
        vocab_size: int = 50277,
        d_model: int = 512,
        d_register: int = 128,       # Complex dim (ℂ^128 = ℝ^256 equivalent)
        max_len: int = 4096,
        n_heads: int = 8,
        d_ff: int = 1536,
        d_ff_consolidate: int = 2048,
        window: int = 8,
        strides: tuple[int, ...] = (1, 8, 64, 512),
        n_prep_layers: int = 1,
        n_converge_layers: int = 2,
        n_consolidate_layers: int = 3,
        dropout: float = 0.1,
        alpha: float = 1.18,          # Spiral attention bias exponent
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_register = d_register
        self.max_len = max_len
        self.n_heads = n_heads
        self.window = window
        self.strides = strides
        self.alpha = alpha
        self.n_registers = len(self.REGISTER_NAMES)
        self.n_phases = len(self.PHASE_NAMES)
        self.n_levels = self.N_LEVELS
        self.n_passes = self.N_PASSES

        # Bank layout (same as v4.1):
        #   0=bank_0, 1=bank_1↑, 2=bank_2↑, 3=bank_3, 4=bank_2↓, 5=bank_1↓
        self.n_banks = 6

        self.n_prep_layers = n_prep_layers
        self.n_converge_layers = n_converge_layers
        self.n_consolidate_layers = n_consolidate_layers

        # ── Progressive stride allocation per level ───────────────
        s1, s8, s64, s512 = strides[0], strides[1], strides[2], strides[3]
        self.level_configs = [
            [(s1, window)] * 3 + [(s8, window)] * 3 + [(s64, window)] * 1 + [(s512, window)] * 1,
            [(s1, window)] * 2 + [(s8, window)] * 2 + [(s64, window)] * 2 + [(s512, window)] * 2,
            [(s1, window)] * 1 + [(s8, window)] * 1 + [(s64, window)] * 3 + [(s512, window)] * 3,
        ]

        # ── S5: Identity (shared weights + embeddings) ────────────
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.output_norm = nn.LayerNorm(d_model)

        # Register bank 0: learnable init (real part; imag starts at 0)
        self.register_inits = nn.ParameterDict({
            f"reg_{name}": nn.Parameter(torch.zeros(d_register))
            for name in self.REGISTER_NAMES
        })

        # Shared S1 operations (S5 coherence)
        self.prep_layers = nn.ModuleList([
            FFNLayer(d_model, d_ff, dropout)
            for _ in range(n_prep_layers)
        ])

        # Converge: per-level stride configs, shared weights, spiral bias
        self.converge_layers_base = nn.ModuleList([
            CompressorLayer(d_model, self.level_configs[0], d_ff, dropout, alpha=alpha)
            for _ in range(n_converge_layers)
        ])
        self.converge_layers_l2 = nn.ModuleList([
            CompressorLayer(d_model, self.level_configs[1], d_ff, dropout, alpha=alpha)
            for _ in range(n_converge_layers)
        ])
        self.converge_layers_l3 = nn.ModuleList([
            CompressorLayer(d_model, self.level_configs[2], d_ff, dropout, alpha=alpha)
            for _ in range(n_converge_layers)
        ])
        for i in range(n_converge_layers):
            self._tie_compressor_weights(self.converge_layers_base[i], self.converge_layers_l2[i])
            self._tie_compressor_weights(self.converge_layers_base[i], self.converge_layers_l3[i])

        # Consolidate: shared across levels, spiral bias
        self.consolidate_layers = nn.ModuleList([
            CompressorLayer(d_model, self.level_configs[1], d_ff_consolidate, dropout, alpha=alpha)
            for _ in range(n_consolidate_layers)
        ])

        # ── S4: Complex-query intelligence ────────────────────────
        self.s4 = S4IntelligenceComplex(
            d_model, d_register, self.n_registers,
            max_banks=self.n_banks,
            dropout=dropout,
        )

        # ── S3: Phase-coherent gating (5 instances) ──────────────
        self.s3_passes = nn.ModuleList([
            S3PhaseCoherent(
                d_model, d_register, self.n_phases, self.n_registers,
                d_align=d_model,
            )
            for _ in range(self.n_passes)
        ])

        # ── Multiplicative modulation (S5 coherent: 3 shared projs) ──
        # modulation = 1 + gate · tanh(proj(delta))
        # Zero-init → identity at start (applied after self.apply)
        self.mod_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(self.n_phases)
        ])

        # ── Meta-S4: Complex-query final summary (4 best banks) ──
        self.meta_s4 = MetaS4Complex(
            d_model, d_register, self.n_registers,
            n_banks=4,
            dropout=dropout,
        )

        # ── Meta-S3: Per-pass contribution gates ─────────────────
        self.meta_s3 = MetaS3Complex(
            d_register, self.n_registers,
            n_banks=self.n_banks,
            n_passes=self.n_passes,
        )

        # ── Initialize ────────────────────────────────────────────
        self.apply(self._init_weights)
        # Zero-init modulation projs → modulation = 1 → identity at start
        for proj in self.mod_projs:
            nn.init.zeros_(proj.weight)

    @staticmethod
    def _tie_compressor_weights(source: CompressorLayer, target: CompressorLayer):
        """Tie all learnable weights of target to source (S5 coherence)."""
        target.attn.q_proj.weight = source.attn.q_proj.weight
        target.attn.k_proj.weight = source.attn.k_proj.weight
        target.attn.v_proj.weight = source.attn.v_proj.weight
        target.attn.out_proj.weight = source.attn.out_proj.weight
        if target.attn.q_proj.bias is not None:
            target.attn.q_proj.bias = source.attn.q_proj.bias
        if target.attn.k_proj.bias is not None:
            target.attn.k_proj.bias = source.attn.k_proj.bias
        if target.attn.v_proj.bias is not None:
            target.attn.v_proj.bias = source.attn.v_proj.bias
        if target.attn.out_proj.bias is not None:
            target.attn.out_proj.bias = source.attn.out_proj.bias

        target.norm1.weight = source.norm1.weight
        target.norm1.bias = source.norm1.bias
        target.norm2.weight = source.norm2.weight
        target.norm2.bias = source.norm2.bias

        for i in range(len(source.ff)):
            src_mod = source.ff[i]
            tgt_mod = target.ff[i]
            if hasattr(src_mod, 'weight'):
                tgt_mod.weight = src_mod.weight
            if hasattr(src_mod, 'bias') and src_mod.bias is not None:
                tgt_mod.bias = src_mod.bias

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _init_bank0(self) -> list[torch.Tensor]:
        """Initialize register bank 0: complex(learned_real, zeros)."""
        return [
            torch.complex(
                self.register_inits[f"reg_{name}"].clone(),
                torch.zeros_like(self.register_inits[f"reg_{name}"]),
            )
            for name in self.REGISTER_NAMES
        ]

    def _fresh_bank(self) -> list[torch.Tensor]:
        """Create a zero-initialized complex register bank."""
        device = self.register_inits["reg_type"].device
        return [
            torch.zeros(self.d_register, device=device, dtype=torch.cfloat)
            for _ in self.REGISTER_NAMES
        ]

    def _get_converge_layers(self, level: int) -> nn.ModuleList:
        if level == 0:
            return self.converge_layers_base
        elif level == 1:
            return self.converge_layers_l2
        elif level == 2:
            return self.converge_layers_l3
        else:
            raise ValueError(f"Invalid level: {level}")

    def _run_prep(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.prep_layers:
            x = layer(x)
        return x

    def _run_converge(self, x: torch.Tensor, level: int) -> torch.Tensor:
        for layer in self._get_converge_layers(level):
            x = layer(x)
        return x

    def _run_consolidate(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.consolidate_layers:
            x = layer(x)
        return x

    def _modulate(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        gate: torch.Tensor,
        phase_idx: int,
    ) -> torch.Tensor:
        """Multiplicative modulation: x_new = x · (1 + gate · tanh(proj(delta))).

        Zero-init proj → modulation = 1 → identity at start.
        Chain x · m₁ · m₂ · ... · mₙ produces power-law magnitude.
        """
        modulation = 1.0 + gate * torch.tanh(self.mod_projs[phase_idx](delta))
        return x * modulation

    def _run_level_pass(
        self,
        x: torch.Tensor,
        level: int,
        pass_idx: int,
        readable_banks: list[list[torch.Tensor]],
        target_bank: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Run one level-pass (ascending or descending)."""
        x_before = x

        # S4: complex-query scan
        s4_updates, _ = self.s4(readable_banks, x)
        target_bank = [
            target_bank[i] + s4_updates[i]
            for i in range(self.n_registers)
        ]

        # Phase 1: PREP — multiplicative modulation
        prep_out = self._run_prep(x)
        delta = prep_out - x
        _, target_bank, gate, _ = self.s3_passes[pass_idx].gate_phase(
            target_bank, delta, 0)
        x = self._modulate(x, delta, gate, 0)

        # Phase 2: CONVERGE — multiplicative modulation
        converge_out = self._run_converge(x, level)
        delta = converge_out - x
        _, target_bank, gate, _ = self.s3_passes[pass_idx].gate_phase(
            target_bank, delta, 1)
        x = self._modulate(x, delta, gate, 1)

        # Phase 3: CONSOLIDATE — multiplicative modulation
        consolidate_out = self._run_consolidate(x)
        delta = consolidate_out - x
        _, target_bank, gate, _ = self.s3_passes[pass_idx].gate_phase(
            target_bank, delta, 2)
        x = self._modulate(x, delta, gate, 2)

        return x, target_bank, x - x_before

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L = input_ids.shape
        device = input_ids.device

        positions = torch.arange(L, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        # ── Complex register banks ────────────────────────────────
        bank_0 = self._init_bank0()
        bank_1_asc = self._fresh_bank()
        bank_2_asc = self._fresh_bank()
        bank_3 = self._fresh_bank()
        bank_2_desc = self._fresh_bank()
        bank_1_desc = self._fresh_bank()

        pass_deltas = []

        # ── ASCENDING: L0↑ → L1↑ → L2 ───────────────────────────
        x, bank_1_asc, delta = self._run_level_pass(
            x, level=0, pass_idx=0,
            readable_banks=[bank_0],
            target_bank=bank_1_asc,
        )
        pass_deltas.append(delta)

        x, bank_2_asc, delta = self._run_level_pass(
            x, level=1, pass_idx=1,
            readable_banks=[bank_0, bank_1_asc],
            target_bank=bank_2_asc,
        )
        pass_deltas.append(delta)

        x, bank_3, delta = self._run_level_pass(
            x, level=2, pass_idx=2,
            readable_banks=[bank_0, bank_1_asc, bank_2_asc],
            target_bank=bank_3,
        )
        pass_deltas.append(delta)

        # ── DESCENDING: L1↓ → L0↓ ────────────────────────────────
        x, bank_2_desc, delta = self._run_level_pass(
            x, level=1, pass_idx=3,
            readable_banks=[bank_0, bank_1_asc, bank_2_asc, bank_3],
            target_bank=bank_2_desc,
        )
        pass_deltas.append(delta)

        x, bank_1_desc, delta = self._run_level_pass(
            x, level=0, pass_idx=4,
            readable_banks=[bank_0, bank_1_asc, bank_2_desc, bank_3],
            target_bank=bank_1_desc,
        )
        pass_deltas.append(delta)

        # ── Meta-S3: per-pass contribution gates ──────────────────
        all_banks = [bank_0, bank_1_asc, bank_2_asc, bank_3, bank_2_desc, bank_1_desc]
        meta_gates = self.meta_s3(all_banks)

        total_ungated = sum(pass_deltas)
        total_gated = sum(
            meta_gates[i] * pass_deltas[i]
            for i in range(self.n_passes)
        )
        x = x - total_ungated + total_gated

        # ── Meta-S4: final structural summary ─────────────────────
        meta_banks = [bank_0, bank_1_desc, bank_2_desc, bank_3]
        x = self.meta_s4(meta_banks, x)

        x = self.output_norm(x)
        logits = F.linear(x, self.token_embed.weight)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def forward_instrumented(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """Forward pass with full instrumentation for probing."""
        B, L = input_ids.shape
        device = input_ids.device
        metrics: dict = {}
        reg_names = list(self.REGISTER_NAMES)

        positions = torch.arange(L, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        metrics["embed_norm"] = x.norm(dim=-1).mean().item()

        # Complex register banks
        bank_0 = self._init_bank0()
        bank_1_asc = self._fresh_bank()
        bank_2_asc = self._fresh_bank()
        bank_3 = self._fresh_bank()
        bank_2_desc = self._fresh_bank()
        bank_1_desc = self._fresh_bank()

        for i, name in enumerate(reg_names):
            metrics[f"register_{name}_init_norm"] = torch.view_as_real(bank_0[i]).norm().item()

        pass_deltas = []

        pass_schedule = [
            (0, 0, "L0_asc"),
            (1, 1, "L1_asc"),
            (2, 2, "L2_apex"),
            (3, 1, "L1_desc"),
            (4, 0, "L0_desc"),
        ]

        for pass_idx, level, pass_name in pass_schedule:
            pfx = pass_name

            if pass_idx == 0:
                readable = [bank_0]
                target_bank = bank_1_asc
            elif pass_idx == 1:
                readable = [bank_0, bank_1_asc]
                target_bank = bank_2_asc
            elif pass_idx == 2:
                readable = [bank_0, bank_1_asc, bank_2_asc]
                target_bank = bank_3
            elif pass_idx == 3:
                readable = [bank_0, bank_1_asc, bank_2_asc, bank_3]
                target_bank = bank_2_desc
            elif pass_idx == 4:
                readable = [bank_0, bank_1_asc, bank_2_desc, bank_3]
                target_bank = bank_1_desc

            x_before = x

            # S4
            s4_updates, s4_attn = self.s4(readable, x)
            target_bank = [
                target_bank[i] + s4_updates[i]
                for i in range(self.n_registers)
            ]

            for i, name in enumerate(reg_names):
                metrics[f"{pfx}_reg_{name}_after_s4"] = torch.view_as_real(target_bank[i]).norm().item()
                metrics[f"{pfx}_reg_{name}_phase_mean"] = torch.angle(target_bank[i]).mean().item()
            s4_entropy = -(s4_attn * (s4_attn + 1e-10).log()).sum(dim=-1).mean()
            metrics[f"{pfx}_s4_attn_entropy"] = s4_entropy.item()

            # Three phases — multiplicative modulation
            for phase_idx, phase_name in enumerate(self.PHASE_NAMES):
                if phase_name == "prep":
                    phase_out = self._run_prep(x)
                elif phase_name == "converge":
                    phase_out = self._run_converge(x, level)
                elif phase_name == "consolidate":
                    phase_out = self._run_consolidate(x)

                delta = phase_out - x
                gated_delta, target_bank, gate, write_gates = (
                    self.s3_passes[pass_idx].gate_phase(target_bank, delta, phase_idx))

                # Multiplicative modulation (replaces x = x + gated_delta)
                modulation = 1.0 + gate * torch.tanh(self.mod_projs[phase_idx](delta))
                x = x * modulation

                metrics[f"{pfx}_{phase_name}_delta_norm"] = delta.norm(dim=-1).mean().item()
                metrics[f"{pfx}_{phase_name}_gated_norm"] = gated_delta.norm(dim=-1).mean().item()
                metrics[f"{pfx}_{phase_name}_gate_mean"] = gate.detach().item()
                metrics[f"{pfx}_{phase_name}_gate_std"] = 0.0  # scalar gate, compat
                metrics[f"{pfx}_{phase_name}_mod_mean"] = modulation.detach().mean().item()
                metrics[f"{pfx}_{phase_name}_mod_std"] = modulation.detach().std().item()
                metrics[f"{pfx}_after_{phase_name}"] = x.norm(dim=-1).mean().item()
                for i, rn in enumerate(reg_names):
                    metrics[f"{pfx}_{phase_name}_write_{rn}"] = write_gates[i]

            # Register norms after pass (complex magnitude)
            for i, name in enumerate(reg_names):
                metrics[f"{pfx}_register_{name}_norm"] = torch.view_as_real(target_bank[i]).norm().item()
                metrics[f"{pfx}_register_{name}_phase_final"] = torch.angle(target_bank[i]).mean().item()

            # Write back
            if pass_idx == 0:
                bank_1_asc = target_bank
            elif pass_idx == 1:
                bank_2_asc = target_bank
            elif pass_idx == 2:
                bank_3 = target_bank
            elif pass_idx == 3:
                bank_2_desc = target_bank
            elif pass_idx == 4:
                bank_1_desc = target_bank

            pass_deltas.append(x - x_before)

        # ── Level-indexed metrics for v4 compatibility ────────────
        level_map = {
            "L0_asc": "level0", "L1_asc": "level1", "L2_apex": "level2",
            "L1_desc": "level1_desc", "L0_desc": "level0_desc",
        }
        for pass_name, level_pfx in level_map.items():
            for key in list(metrics.keys()):
                if key.startswith(pass_name + "_"):
                    suffix = key[len(pass_name) + 1:]
                    metrics[f"{level_pfx}_{suffix}"] = metrics[key]

        # Backward-compat iter aliases
        for level in range(min(self.N_LEVELS, 2)):
            src_pfx = f"level{level}"
            dst_pfx = f"iter{level}"
            for phase in self.PHASE_NAMES:
                for suffix in ["delta_norm", "gated_norm", "gate_mean", "gate_std"]:
                    k = f"{src_pfx}_{phase}_{suffix}"
                    if k in metrics:
                        metrics[f"{dst_pfx}_{phase}_{suffix}"] = metrics[k]
                for rn in reg_names:
                    k = f"{src_pfx}_{phase}_write_{rn}"
                    if k in metrics:
                        metrics[f"{dst_pfx}_{phase}_write_{rn}"] = metrics[k]
            for rn in reg_names:
                for key_suffix in [f"reg_{rn}_after_s4", f"register_{rn}_norm"]:
                    k = f"{src_pfx}_{key_suffix}"
                    if k in metrics:
                        metrics[f"{dst_pfx}_{key_suffix}"] = metrics[k]
            k = f"{src_pfx}_s4_attn_entropy"
            if k in metrics:
                metrics[f"{dst_pfx}_s4_attn_entropy"] = metrics[k]
            for phase in self.PHASE_NAMES:
                k = f"{src_pfx}_after_{phase}"
                if k in metrics:
                    metrics[f"{dst_pfx}_after_{phase}"] = metrics[k]

        # Meta-S3
        all_banks = [bank_0, bank_1_asc, bank_2_asc, bank_3, bank_2_desc, bank_1_desc]
        meta_gates = self.meta_s3(all_banks)
        for i, pname in enumerate(self.PASS_NAMES):
            metrics[f"meta_s3_gate_{pname}"] = meta_gates[i].item()
        metrics["meta_s3_gate_level0"] = meta_gates[0].item()
        metrics["meta_s3_gate_level1"] = meta_gates[1].item()
        metrics["meta_s3_gate_level2"] = meta_gates[2].item()

        total_ungated = sum(pass_deltas)
        total_gated = sum(
            meta_gates[i] * pass_deltas[i]
            for i in range(self.n_passes)
        )
        x = x - total_ungated + total_gated

        # Meta-S4
        meta_banks = [bank_0, bank_1_desc, bank_2_desc, bank_3]
        x = self.meta_s4(meta_banks, x)

        # Global compat aliases
        metrics["s4_attn_entropy"] = metrics["L0_asc_s4_attn_entropy"]
        metrics["register_after_s4"] = sum(
            metrics[f"L0_asc_reg_{n}_after_s4"] for n in reg_names
        )

        metrics["output_norm"] = x.norm(dim=-1).mean().item()
        metrics["overall_expansion"] = metrics["output_norm"] / max(metrics["embed_norm"], 1e-8)

        x = self.output_norm(x)
        logits = F.linear(x, self.token_embed.weight)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, metrics

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by VSM subsystem."""
        seen_ids = set()

        def _count_unique(module):
            total = 0
            for p in module.parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    total += p.numel()
            return total

        seen_ids.clear()

        s5_embed = _count_unique(self.token_embed)
        s5_pos = _count_unique(self.pos_embed)
        s5_other = (
            sum(p.numel() for p in self.register_inits.parameters())
            + _count_unique(self.output_norm)
        )
        s4 = _count_unique(self.s4)
        s3 = sum(_count_unique(s3p) for s3p in self.s3_passes)
        s3 += _count_unique(self.mod_projs)  # S5 coherent modulation projs
        meta_s4 = _count_unique(self.meta_s4)
        meta_s3 = _count_unique(self.meta_s3)

        s1_prep = _count_unique(self.prep_layers)
        s1_converge = _count_unique(self.converge_layers_base)
        _count_unique(self.converge_layers_l2)
        _count_unique(self.converge_layers_l3)
        s1_consolidate = _count_unique(self.consolidate_layers)
        s1 = s1_prep + s1_converge + s1_consolidate

        seen_ids.clear()
        total = 0
        for p in self.parameters():
            if id(p) not in seen_ids:
                seen_ids.add(id(p))
                total += p.numel()

        return {
            "S5_token_embeddings": s5_embed,
            "S5_positional": s5_pos,
            "S5_other": s5_other,
            "S4_intelligence": s4,
            "S3_passes": s3,
            "Meta_S4": meta_s4,
            "Meta_S3": meta_s3,
            "S1_prep": s1_prep,
            "S1_converge": s1_converge,
            "S1_consolidate": s1_consolidate,
            "S1_total": s1,
            "total": total,
        }

    def describe(self) -> str:
        def _stride_desc(config):
            from collections import Counter
            counts = Counter(s for s, _ in config)
            return "+".join(f"s{s}×{n}" for s, n in sorted(counts.items()))

        ffn_per_level = self.n_prep_layers + self.n_converge_layers + self.n_consolidate_layers
        ffn_total = ffn_per_level * self.n_passes

        lines = [
            f"VSM-LM v5 — Spiral + Complex Registers + Phase-Coherent Gating",
            f"  d_model={self.d_model}, d_register=ℂ^{self.d_register} (={self.d_register*2}ℝ), "
            f"seq_len={self.max_len}",
            f"  Passes: {self.n_passes} (L0↑, L1↑, L2, L1↓, L0↓)",
            f"  Phase structure: prep({self.n_prep_layers}L, FFN) → "
            f"converge({self.n_converge_layers}L, attn) → "
            f"consolidate({self.n_consolidate_layers}L, wide-FFN)",
            f"  Strides: {self.strides} (spiral bias α={self.alpha})",
            f"  Spiral: bias(w) = -{self.alpha}·ln(stride·w + 1)",
            f"  Registers: ℂ^{self.d_register} (phase-sensitive S4 attention)",
            f"  S3: phase-coherent alignment gating (scalar gate, temperature+bias)",
            f"  Composition: multiplicative modulation x·(1 + gate·tanh(proj(δ)))",
        ]
        for i, config in enumerate(self.level_configs):
            lines.append(f"    Level {i}: {_stride_desc(config)}")
        lines.extend([
            f"  S5: Shared weights across all passes (identity coherence)",
            f"  S4: Complex-query register scan: Re(q·conj(k))",
            f"  Register banks: {self.n_banks} (1 init + 3 ascending + 2 descending)",
            f"  Meta-S4: Complex-query structural summary (4 most-refined banks)",
            f"  Meta-S3: Per-pass contribution gates ({self.n_passes} gates)",
            f"  FFN passes/forward: {ffn_total} ({ffn_per_level}/pass × {self.n_passes})",
        ])
        return "\n".join(lines)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            x = input_ids[:, -self.max_len:]
            logits, _ = self(x)
            logits = logits[:, -1, :] / temperature
            next_token = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
