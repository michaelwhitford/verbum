"""VSM-LM v4 — Recursive Viable System Architecture.

The full recursive VSM: same compositional function (S5) applied at 3
hierarchical levels with growing register context (S4↔S4 channel),
per-level autonomous control (S3), register bank protocol (S2), and
residual stream as algedonic bypass.

Built on v3.2's validated prep→converge→consolidate phases. v4 adds:
  - 3 levels (not 2 iterations), each a nested VSM
  - 4 register banks: bank_0 (learnable init) + bank_1-3 (per-level writes)
  - S4 reads grow per level: level N reads banks 0..N-1
  - Shared S1 weights across levels (S5 identity coherence)
  - Per-level S3 instances (autonomous control, NOT shared)
  - Progressive stride reallocation (local-heavy → clause-heavy)
  - 4 strides: s1, s8, s64, s512 (stride 512 reinstated with hierarchy)
  - Meta-S4: final register scan over all banks
  - Meta-S3: per-level contribution gate

Design rationale: the hierarchy is in the WIRING (S2) and CONTROL (S3),
not the weights (S5). Same function at every level. Only the context
(register banks) changes. This is Beer's recursive viability realized
as a neural architecture.

Parameter budget: ~51M (same as v3.2). The hierarchy is free — shared
weights mean 3 levels cost the same as 1.

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
# FFN-only layer (reused from v3.2)
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
# S4 — Intelligence (hierarchical register scan)
# ══════════════════════════════════════════════════════════════════════


class S4Intelligence(nn.Module):
    """Register cross-attention that reads from a variable number of banks.

    Each bank has n_registers registers (type, scope, role). The query is
    formed from the concatenation of ALL readable banks. Keys and values
    come from the residual stream. The summary is projected back into
    per-register updates for the CURRENT level's bank.

    This is the S4↔S4 channel: each level reads structural summaries
    from all previous levels before scanning the residual.
    """

    def __init__(
        self,
        d_model: int,
        d_register: int,
        n_registers: int = 3,
        max_banks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_register = d_register
        self.n_registers = n_registers
        self.max_banks = max_banks
        self.scale = d_model ** -0.5

        # Query projection: takes concatenated registers from all readable banks
        # Max input size = max_banks * n_registers * d_register
        max_q_dim = max_banks * n_registers * d_register
        self.q_proj = nn.Linear(max_q_dim, d_model, bias=False)

        # K, V from residual
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Summary back to register space (always writes n_registers)
        self.summary_proj = nn.Linear(d_model, n_registers * d_register, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        banks: list[list[torch.Tensor]],  # list of banks, each bank = [type, scope, role]
        residual: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Scan residual conditioned on all readable register banks.

        Args:
            banks: list of register banks to read. Each bank is a list
                   of n_registers tensors of shape (d_register,).
            residual: (B, L, D) residual stream.

        Returns:
            updated_registers: list of n_registers updated tensors (for current level)
            attn_weights: (B, L) attention weights (detached)
        """
        B, L, D = residual.shape

        # Concatenate all registers from all readable banks
        all_regs = []
        for bank in banks:
            all_regs.extend(bank)
        n_active = len(all_regs)

        # Pad to max size (so the projection weight is fixed-size)
        max_regs = self.max_banks * self.n_registers
        while len(all_regs) < max_regs:
            all_regs.append(torch.zeros_like(all_regs[0]))

        q_input = torch.cat(all_regs, dim=-1)  # (max_banks * n_registers * d_register,)
        q = self.q_proj(q_input)  # (d_model,)

        x = self.norm(residual)
        k = self.k_proj(x)  # (B, L, D)
        v = self.v_proj(x)  # (B, L, D)

        # Attention: q (D,) against k (B, L, D)
        attn = torch.einsum("d,bld->bl", q, k) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        summary = torch.einsum("bl,bld->bd", attn_weights, v)
        summary = summary.mean(dim=0)  # (D,)

        # Project to register updates
        updates = self.summary_proj(summary)  # (n_registers * d_register,)
        update_splits = updates.split(self.d_register, dim=-1)

        # The caller provides the "base" registers to update (current level's bank)
        # We return the updates; the caller adds them
        updated = list(update_splits)
        return updated, attn_weights.detach()


# ══════════════════════════════════════════════════════════════════════
# S3 — Per-level Control (autonomous resource allocation)
# ══════════════════════════════════════════════════════════════════════


class S3LevelControl(nn.Module):
    """Control for a single level: gates 3 phases, writes 1 register bank.

    Each level has its OWN S3 instance (Beer: nested viable systems have
    autonomous control). The gate reads the current level's register bank
    plus a residual summary to decide how much of each phase's output to
    keep. Register writes update this level's bank only.
    """

    def __init__(
        self,
        d_model: int,
        d_register: int,
        n_phases: int = 3,
        n_registers: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_register = d_register
        self.n_phases = n_phases
        self.n_registers = n_registers

        gate_input_dim = d_register * n_registers + d_model

        # One gate head per phase
        self.gate_heads = nn.ModuleList([
            nn.Linear(gate_input_dim, d_model)
            for _ in range(n_phases)
        ])

        # Register write projections + gates per phase
        self.write_projs = nn.ModuleList([
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
        """Gate a phase's output and update this level's register bank.

        Args:
            registers: current level's register bank [type, scope, role]
            delta: (B, L, D) phase output minus input
            phase_idx: 0=prep, 1=converge, 2=consolidate

        Returns:
            gated_delta: (B, L, D) phase output after gating
            updated_registers: updated register bank
            gate_vals: (D,) gate values (detached, for logging)
            write_gate_values: list of scalar write gate values
        """
        summary = delta.mean(dim=(0, 1))  # (D,)
        reg_concat = torch.cat(registers, dim=-1)  # (n_registers * d_register,)
        gate_input = torch.cat([reg_concat, summary])  # (gate_input_dim,)

        gate = torch.sigmoid(self.gate_heads[phase_idx](gate_input))
        gated_delta = gate.unsqueeze(0).unsqueeze(0) * delta

        updated_registers = []
        write_gate_values = []
        for reg_idx in range(self.n_registers):
            write_idx = phase_idx * self.n_registers + reg_idx
            wg = torch.sigmoid(self.write_gates[write_idx](summary))
            update = self.write_projs[write_idx](summary)
            updated_registers.append(registers[reg_idx] + wg * update)
            write_gate_values.append(wg.item())

        return gated_delta, updated_registers, gate.detach(), write_gate_values


# ══════════════════════════════════════════════════════════════════════
# Meta-S3 — Cross-level contribution gate
# ══════════════════════════════════════════════════════════════════════


class MetaS3(nn.Module):
    """Top-level resource allocation: per-level contribution gates.

    Modulates how much each level's output contributes to the final
    residual stream. Some inputs need mostly level 1 (simple local
    prediction). Others need deep level 3 (complex binding). Meta-S3
    learns to allocate. This is Beer's S3 "inside and now" at the
    top recursive level.
    """

    def __init__(self, d_register: int, n_registers: int, n_levels: int):
        super().__init__()
        # Input: all register banks concatenated
        input_dim = (n_levels + 1) * n_registers * d_register  # +1 for bank_0
        self.gate_proj = nn.Linear(input_dim, n_levels)

    def forward(self, all_banks: list[list[torch.Tensor]]) -> torch.Tensor:
        """Compute per-level contribution weights.

        Args:
            all_banks: list of all register banks (including bank_0)

        Returns:
            gates: (n_levels,) sigmoid values
        """
        flat = torch.cat([reg for bank in all_banks for reg in bank], dim=-1)
        return torch.sigmoid(self.gate_proj(flat))


# ══════════════════════════════════════════════════════════════════════
# Meta-S4 — Final structural summary
# ══════════════════════════════════════════════════════════════════════


class MetaS4(nn.Module):
    """Final intelligence scan: reads ALL register banks.

    After all levels complete, Meta-S4 produces the full structural
    summary — what was found at every level of abstraction. This
    feeds into the output head via the residual stream.
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

        total_reg_dim = n_banks * n_registers * d_register
        self.q_proj = nn.Linear(total_reg_dim, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        all_banks: list[list[torch.Tensor]],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Produce structural summary and add to residual.

        Args:
            all_banks: all 4 register banks
            residual: (B, L, D)

        Returns:
            updated residual: (B, L, D)
        """
        B, L, D = residual.shape

        # Query from all registers
        all_regs = [reg for bank in all_banks for reg in bank]
        q_input = torch.cat(all_regs, dim=-1)
        q = self.q_proj(q_input)  # (D,)

        x = self.norm(residual)
        k = self.k_proj(x)  # (B, L, D)
        v = self.v_proj(x)  # (B, L, D)

        attn = torch.einsum("d,bld->bl", q, k) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        summary = torch.einsum("bl,bld->bd", attn_weights, v)  # (B, D)

        # Broadcast structural summary to all positions
        out = self.out_proj(summary).unsqueeze(1).expand_as(residual)  # (B, L, D)
        return residual + out


# ══════════════════════════════════════════════════════════════════════
# VSM-LM v4 — Recursive Viable System
# ══════════════════════════════════════════════════════════════════════


class VSMLMV4(nn.Module):
    """Viable System Model Language Model — v4 recursive architecture.

    Three hierarchical levels, each a nested VSM. Same function (S5),
    different register context (S4↔S4), autonomous control (S3).
    Progressive stride reallocation across levels.
    """

    REGISTER_NAMES = ("type", "scope", "role")
    PHASE_NAMES = ("prep", "converge", "consolidate")
    N_LEVELS = 3

    def __init__(
        self,
        vocab_size: int = 50277,
        d_model: int = 512,
        d_register: int = 256,
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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_register = d_register
        self.max_len = max_len
        self.n_heads = n_heads
        self.window = window
        self.strides = strides
        self.n_registers = len(self.REGISTER_NAMES)
        self.n_phases = len(self.PHASE_NAMES)
        self.n_levels = self.N_LEVELS
        self.n_banks = self.n_levels + 1  # bank_0 (init) + 3 level banks

        self.n_prep_layers = n_prep_layers
        self.n_converge_layers = n_converge_layers
        self.n_consolidate_layers = n_consolidate_layers

        # ── Progressive stride allocation per level ───────────────
        # Level 1 (local-heavy):   s1×3, s8×3, s64×1, s512×1
        # Level 2 (balanced):      s1×2, s8×2, s64×2, s512×2
        # Level 3 (clause-heavy):  s1×1, s8×1, s64×3, s512×3
        s1, s8, s64, s512 = strides[0], strides[1], strides[2], strides[3]
        self.level_configs = [
            # Level 1: local-heavy
            [(s1, window)] * 3 + [(s8, window)] * 3 + [(s64, window)] * 1 + [(s512, window)] * 1,
            # Level 2: balanced
            [(s1, window)] * 2 + [(s8, window)] * 2 + [(s64, window)] * 2 + [(s512, window)] * 2,
            # Level 3: clause/discourse-heavy
            [(s1, window)] * 1 + [(s8, window)] * 1 + [(s64, window)] * 3 + [(s512, window)] * 3,
        ]

        # ── S5: Identity (shared weights + embeddings) ────────────
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.output_norm = nn.LayerNorm(d_model)

        # Register bank 0: learnable init (S5 — model identity)
        self.register_inits = nn.ParameterDict({
            f"reg_{name}": nn.Parameter(torch.zeros(d_register))
            for name in self.REGISTER_NAMES
        })

        # Shared S1 operations (S5 coherence: same function at every level)
        # Prep: FFN-only
        self.prep_layers = nn.ModuleList([
            FFNLayer(d_model, d_ff, dropout)
            for _ in range(n_prep_layers)
        ])

        # Converge: one set per level (different stride configs but same layer type)
        # NOTE: we share the underlying weights but need separate CompressorLayer
        # instances because stride configs differ. The Q/K/V/FFN weights are what
        # matter for S5 coherence. We achieve this by having each level's converge
        # layers share parameters with level 0's converge layers.
        self.converge_layers_base = nn.ModuleList([
            CompressorLayer(d_model, self.level_configs[0], d_ff, dropout)
            for _ in range(n_converge_layers)
        ])
        # Level 1 and 2 get their own converge layers with different stride configs
        # but we tie the FFN and projection weights to the base.
        # DESIGN DECISION: For true S5 coherence, the CompressorLayer's attention
        # Q/K/V/O projections and FFN are the "function identity". The stride config
        # is the "operational environment". Since StridedCausalAttention shares Q/K/V
        # projections across all heads (regardless of stride), we CAN share weights
        # across levels — the same Q/K/V produce the same head features, just gathered
        # at different positions. The FFN is already stride-independent.
        #
        # Implementation: create separate layer instances per level for the converge
        # phase (needed for different stride configs in the attention index cache),
        # then tie their weights to the base.
        self.converge_layers_l2 = nn.ModuleList([
            CompressorLayer(d_model, self.level_configs[1], d_ff, dropout)
            for _ in range(n_converge_layers)
        ])
        self.converge_layers_l3 = nn.ModuleList([
            CompressorLayer(d_model, self.level_configs[2], d_ff, dropout)
            for _ in range(n_converge_layers)
        ])
        # Tie weights: l2 and l3 share parameters with base (l1)
        for i in range(n_converge_layers):
            self._tie_compressor_weights(self.converge_layers_base[i], self.converge_layers_l2[i])
            self._tie_compressor_weights(self.converge_layers_base[i], self.converge_layers_l3[i])

        # Consolidate: shared across levels (same stride config — uses base config)
        self.consolidate_layers = nn.ModuleList([
            CompressorLayer(d_model, self.level_configs[1], d_ff_consolidate, dropout)
            for _ in range(n_consolidate_layers)
        ])

        # ── S4: Intelligence (one shared instance, variable bank reads) ──
        self.s4 = S4Intelligence(
            d_model, d_register, self.n_registers,
            max_banks=self.n_banks, dropout=dropout,
        )

        # ── S3: Per-level control (3 independent instances) ──────
        self.s3_levels = nn.ModuleList([
            S3LevelControl(d_model, d_register, self.n_phases, self.n_registers)
            for _ in range(self.n_levels)
        ])

        # ── Meta-S4: Final structural summary ────────────────────
        self.meta_s4 = MetaS4(
            d_model, d_register, self.n_registers,
            n_banks=self.n_banks, dropout=dropout,
        )

        # ── Meta-S3: Cross-level contribution gates ──────────────
        self.meta_s3 = MetaS3(d_register, self.n_registers, self.n_levels)

        # ── Initialize ────────────────────────────────────────────
        self.apply(self._init_weights)

    @staticmethod
    def _tie_compressor_weights(source: CompressorLayer, target: CompressorLayer):
        """Tie all learnable weights of target to source (S5 coherence).

        The stride-index cache in the attention module is instance-specific
        (it's a dict, not a parameter), so tying weights is safe — each
        instance caches its own stride patterns but shares the projections.
        """
        # Attention: Q, K, V, out projections
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

        # LayerNorms
        target.norm1.weight = source.norm1.weight
        target.norm1.bias = source.norm1.bias
        target.norm2.weight = source.norm2.weight
        target.norm2.bias = source.norm2.bias

        # FFN
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
        """Initialize register bank 0 (S5 identity)."""
        return [
            self.register_inits[f"reg_{name}"].clone()
            for name in self.REGISTER_NAMES
        ]

    def _fresh_bank(self) -> list[torch.Tensor]:
        """Create a zero-initialized register bank (for levels to write into)."""
        device = self.register_inits["reg_type"].device
        return [
            torch.zeros(self.d_register, device=device)
            for _ in self.REGISTER_NAMES
        ]

    def _get_converge_layers(self, level: int) -> nn.ModuleList:
        """Get the converge layers for a given level (0-indexed)."""
        if level == 0:
            return self.converge_layers_base
        elif level == 1:
            return self.converge_layers_l2
        elif level == 2:
            return self.converge_layers_l3
        else:
            raise ValueError(f"Invalid level: {level}")

    def _run_prep(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 1: FFN-only (shared across levels)."""
        for layer in self.prep_layers:
            x = layer(x)
        return x

    def _run_converge(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """Phase 2: Multi-scale attention (stride config varies by level)."""
        for layer in self._get_converge_layers(level):
            x = layer(x)
        return x

    def _run_consolidate(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 3: FFN-heavy (shared across levels)."""
        for layer in self.consolidate_layers:
            x = layer(x)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L = input_ids.shape
        device = input_ids.device

        positions = torch.arange(L, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        # Register banks: bank_0 = learnable init, bank_1-3 = per-level
        bank_0 = self._init_bank0()
        banks = [bank_0]  # banks[0] = bank_0

        # Pre-allocate level banks (will be written by each level's S3)
        for _ in range(self.n_levels):
            banks.append(self._fresh_bank())

        # Track per-level deltas for Meta-S3 gating
        level_deltas = []

        for level in range(self.n_levels):
            x_before_level = x

            # S4: read all banks up to current level (banks[0..level])
            readable_banks = banks[:level + 1]
            current_bank = banks[level + 1]  # this level's writable bank
            s4_updates, _ = self.s4(readable_banks, x)

            # Apply S4 updates to current level's bank
            current_bank = [
                current_bank[i] + s4_updates[i]
                for i in range(self.n_registers)
            ]

            # Phase 1: PREP (shared S1)
            prep_out = self._run_prep(x)
            delta = prep_out - x
            gated_delta, current_bank, _, _ = self.s3_levels[level].gate_phase(
                current_bank, delta, 0)
            x = x + gated_delta

            # Phase 2: CONVERGE (level-specific stride config, shared weights)
            converge_out = self._run_converge(x, level)
            delta = converge_out - x
            gated_delta, current_bank, _, _ = self.s3_levels[level].gate_phase(
                current_bank, delta, 1)
            x = x + gated_delta

            # Phase 3: CONSOLIDATE (shared S1)
            consolidate_out = self._run_consolidate(x)
            delta = consolidate_out - x
            gated_delta, current_bank, _, _ = self.s3_levels[level].gate_phase(
                current_bank, delta, 2)
            x = x + gated_delta

            # Write back the updated bank
            banks[level + 1] = current_bank

            # Track level delta for Meta-S3
            level_deltas.append(x - x_before_level)

        # Meta-S3: per-level contribution gates
        meta_gates = self.meta_s3(banks)  # (n_levels,)

        # Apply Meta-S3: re-weight the level contributions
        # x currently = original + sum(all level deltas) due to residual additions
        # We want: x = original + sum(gate_i * level_delta_i)
        # So subtract ungated deltas and add gated ones
        total_ungated = sum(level_deltas)
        total_gated = sum(
            meta_gates[i] * level_deltas[i]
            for i in range(self.n_levels)
        )
        x = x - total_ungated + total_gated

        # Meta-S4: final structural summary
        x = self.meta_s4(banks, x)

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

        # Register banks
        bank_0 = self._init_bank0()
        banks = [bank_0]
        for _ in range(self.n_levels):
            banks.append(self._fresh_bank())

        for i, name in enumerate(reg_names):
            metrics[f"register_{name}_init_norm"] = bank_0[i].norm().item()

        level_deltas = []

        for level in range(self.n_levels):
            pfx = f"level{level}"
            x_before_level = x

            # S4
            readable_banks = banks[:level + 1]
            current_bank = banks[level + 1]
            s4_updates, s4_attn = self.s4(readable_banks, x)
            current_bank = [
                current_bank[i] + s4_updates[i]
                for i in range(self.n_registers)
            ]

            for i, name in enumerate(reg_names):
                metrics[f"{pfx}_reg_{name}_after_s4"] = current_bank[i].norm().item()
            s4_entropy = -(s4_attn * (s4_attn + 1e-10).log()).sum(dim=-1).mean()
            metrics[f"{pfx}_s4_attn_entropy"] = s4_entropy.item()

            # Phase 1: PREP
            prep_out = self._run_prep(x)
            delta = prep_out - x
            gated_delta, current_bank, gate_vals, write_gates = (
                self.s3_levels[level].gate_phase(current_bank, delta, 0))
            x = x + gated_delta

            metrics[f"{pfx}_prep_delta_norm"] = delta.norm(dim=-1).mean().item()
            metrics[f"{pfx}_prep_gated_norm"] = gated_delta.norm(dim=-1).mean().item()
            metrics[f"{pfx}_prep_gate_mean"] = gate_vals.mean().item()
            metrics[f"{pfx}_prep_gate_std"] = gate_vals.std().item()
            metrics[f"{pfx}_after_prep"] = x.norm(dim=-1).mean().item()
            for i, rn in enumerate(reg_names):
                metrics[f"{pfx}_prep_write_{rn}"] = write_gates[i]

            # Phase 2: CONVERGE
            converge_out = self._run_converge(x, level)
            delta = converge_out - x
            gated_delta, current_bank, gate_vals, write_gates = (
                self.s3_levels[level].gate_phase(current_bank, delta, 1))
            x = x + gated_delta

            metrics[f"{pfx}_converge_delta_norm"] = delta.norm(dim=-1).mean().item()
            metrics[f"{pfx}_converge_gated_norm"] = gated_delta.norm(dim=-1).mean().item()
            metrics[f"{pfx}_converge_gate_mean"] = gate_vals.mean().item()
            metrics[f"{pfx}_converge_gate_std"] = gate_vals.std().item()
            metrics[f"{pfx}_after_converge"] = x.norm(dim=-1).mean().item()
            for i, rn in enumerate(reg_names):
                metrics[f"{pfx}_converge_write_{rn}"] = write_gates[i]

            # Phase 3: CONSOLIDATE
            consolidate_out = self._run_consolidate(x)
            delta = consolidate_out - x
            gated_delta, current_bank, gate_vals, write_gates = (
                self.s3_levels[level].gate_phase(current_bank, delta, 2))
            x = x + gated_delta

            metrics[f"{pfx}_consolidate_delta_norm"] = delta.norm(dim=-1).mean().item()
            metrics[f"{pfx}_consolidate_gated_norm"] = gated_delta.norm(dim=-1).mean().item()
            metrics[f"{pfx}_consolidate_gate_mean"] = gate_vals.mean().item()
            metrics[f"{pfx}_consolidate_gate_std"] = gate_vals.std().item()
            metrics[f"{pfx}_after_consolidate"] = x.norm(dim=-1).mean().item()
            for i, rn in enumerate(reg_names):
                metrics[f"{pfx}_consolidate_write_{rn}"] = write_gates[i]

            # Register norms after level
            for i, name in enumerate(reg_names):
                metrics[f"{pfx}_register_{name}_norm"] = current_bank[i].norm().item()

            banks[level + 1] = current_bank
            level_deltas.append(x - x_before_level)

        # Meta-S3
        meta_gates = self.meta_s3(banks)
        for i in range(self.n_levels):
            metrics[f"meta_s3_gate_level{i}"] = meta_gates[i].item()

        total_ungated = sum(level_deltas)
        total_gated = sum(
            meta_gates[i] * level_deltas[i]
            for i in range(self.n_levels)
        )
        x = x - total_ungated + total_gated

        # Meta-S4
        x = self.meta_s4(banks, x)

        # Backward-compat aliases for probing pipeline
        metrics["s4_attn_entropy"] = metrics["level0_s4_attn_entropy"]
        metrics["register_after_s4"] = sum(
            metrics[f"level0_reg_{n}_after_s4"] for n in reg_names
        )

        # Iter-compatible aliases (probe script expects iter0/iter1 prefix)
        # Map level0 → iter0, level1 → iter1 for backward compat
        for level in range(min(self.n_levels, 2)):
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
            # S4, register, after_ aliases
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
        """Count parameters by VSM subsystem. Accounts for weight tying."""
        # Use a set to avoid double-counting tied parameters
        seen_ids = set()

        def _count_unique(module):
            total = 0
            for p in module.parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    total += p.numel()
            return total

        # Reset for each category
        seen_ids.clear()

        s5_embed = _count_unique(self.token_embed)
        s5_pos = _count_unique(self.pos_embed)
        s5_other = (
            sum(p.numel() for p in self.register_inits.parameters())
            + _count_unique(self.output_norm)
        )
        seen_ids_before_s4 = seen_ids.copy()
        s4 = _count_unique(self.s4)
        s3 = sum(_count_unique(s3l) for s3l in self.s3_levels)
        meta_s4 = _count_unique(self.meta_s4)
        meta_s3 = _count_unique(self.meta_s3)

        s1_prep = _count_unique(self.prep_layers)
        # Converge: base + l2 + l3 but l2/l3 have tied weights
        s1_converge = _count_unique(self.converge_layers_base)
        _count_unique(self.converge_layers_l2)  # adds nothing due to tying
        _count_unique(self.converge_layers_l3)  # adds nothing due to tying
        s1_consolidate = _count_unique(self.consolidate_layers)
        s1 = s1_prep + s1_converge + s1_consolidate

        total = sum(p.numel() for p in self.parameters() if id(p) in seen_ids or True)
        # Recount total properly
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
            "S3_levels": s3,
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
        ffn_total = ffn_per_level * self.n_levels

        lines = [
            "VSM-LM v4 — Recursive Viable System Architecture",
            f"  d_model={self.d_model}, d_register={self.d_register}×{self.n_registers}, "
            f"seq_len={self.max_len}, levels={self.n_levels}",
            f"  Phase structure: prep({self.n_prep_layers}L, FFN) → "
            f"converge({self.n_converge_layers}L, attn) → "
            f"consolidate({self.n_consolidate_layers}L, wide-FFN)",
            f"  Strides: {self.strides} (4 scales, progressive reallocation)",
        ]
        for i, config in enumerate(self.level_configs):
            lines.append(f"    Level {i+1}: {_stride_desc(config)}")
        lines.extend([
            f"  S5: Shared weights across all levels (identity coherence)",
            f"  S4: Hierarchical register scan (level N reads banks 0..N)",
            f"  S3: 3 independent instances (per-level autonomous control)",
            f"  S2: Register bank protocol + residual stream (coordination)",
            f"  Meta-S4: Final structural summary (all {self.n_banks} banks)",
            f"  Meta-S3: Per-level contribution gates ({self.n_levels} gates)",
            f"  Register banks: {self.n_banks} (1 init + {self.n_levels} level banks)",
            f"  FFN passes/forward: {ffn_total} ({ffn_per_level}/level × {self.n_levels})",
            f"  Sequence: {self.max_len} positions throughout (no pooling)",
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
