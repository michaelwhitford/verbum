"""VSM-LM v4.1 — Full Recursive Viable System Architecture.

v4.1 completes the VSM recursion that v4 left half-built. v4 implemented
only the ascending (bottom-up) half of Beer's bidirectional S4↔S4
intelligence channel. v4.1 adds the descending (top-down) pass:

  ASCENDING:   L0↑ → L1↑ → L2    (build structural summaries)
  DESCENDING:  L1↓ → L0↓          (refine with high-level context)

Same shared weights (S5 coherent). Same function at every level, in
both directions. Only the register context changes — descending levels
read ALL banks including bank_3 (L2's clause-level findings).

L2 is the apex (Beer's metasystem) — it runs once. L1 and L0 run twice:
once ascending (bottom-up observation) and once descending (top-down
refinement). This implements the cortical feedback loop.

Register bank protocol:
  bank_0:   learnable init (S5 identity)
  bank_1↑:  L0 ascending output (bottom-up local features)
  bank_2↑:  L1 ascending output (bottom-up phrase structure)
  bank_3:   L2 output (clause/discourse structure — apex)
  bank_2↓:  L1 descending output (refined with clause context)
  bank_1↓:  L0 descending output (refined with full hierarchy)

Meta-S4 reads descending banks (most refined) + bank_3.
Meta-S3 gates 5 level-passes (L0↑, L1↑, L2, L1↓, L0↓).

5 level-passes total vs v4's 3. ~67% more compute. Zero additional
parameters for the shared function (S5). Small overhead for extra S3
instances and wider S4/Meta inputs.

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
# FFN-only layer (shared with v3.2/v4)
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
    """Register cross-attention reading from a variable number of banks.

    v4.1: max_banks increased to accommodate ascending + descending banks.
    The query projection is sized for the maximum possible input; unused
    bank slots are zero-padded.
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

        max_q_dim = max_banks * n_registers * d_register
        self.q_proj = nn.Linear(max_q_dim, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.summary_proj = nn.Linear(d_model, n_registers * d_register, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        banks: list[list[torch.Tensor]],
        residual: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Scan residual conditioned on all readable register banks."""
        B, L, D = residual.shape

        all_regs = []
        for bank in banks:
            all_regs.extend(bank)

        max_regs = self.max_banks * self.n_registers
        while len(all_regs) < max_regs:
            all_regs.append(torch.zeros_like(all_regs[0]))

        q_input = torch.cat(all_regs, dim=-1)
        q = self.q_proj(q_input)

        x = self.norm(residual)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn = torch.einsum("d,bld->bl", q, k) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        summary = torch.einsum("bl,bld->bd", attn_weights, v)
        summary = summary.mean(dim=0)

        updates = self.summary_proj(summary)
        update_splits = updates.split(self.d_register, dim=-1)

        return list(update_splits), attn_weights.detach()


# ══════════════════════════════════════════════════════════════════════
# S3 — Per-level Control (autonomous resource allocation)
# ══════════════════════════════════════════════════════════════════════


class S3LevelControl(nn.Module):
    """Control for a single level-pass: gates 3 phases, writes 1 register bank.

    v4.1: one S3 instance per level-pass (5 total: L0↑, L1↑, L2, L1↓, L0↓).
    Each pass has autonomous control — ascending and descending may gate
    differently because they face different variety.
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

        self.gate_heads = nn.ModuleList([
            nn.Linear(gate_input_dim, d_model)
            for _ in range(n_phases)
        ])

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
        """Gate a phase's output and update register bank."""
        summary = delta.mean(dim=(0, 1))
        reg_concat = torch.cat(registers, dim=-1)
        gate_input = torch.cat([reg_concat, summary])

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
    """Top-level resource allocation: per-pass contribution gates.

    v4.1: gates 5 level-passes (L0↑, L1↑, L2, L1↓, L0↓) instead of
    v4's 3 (L0, L1, L2). Reads all register banks to decide allocation.
    """

    def __init__(self, d_register: int, n_registers: int, n_banks: int, n_passes: int):
        super().__init__()
        input_dim = n_banks * n_registers * d_register
        self.gate_proj = nn.Linear(input_dim, n_passes)

    def forward(self, all_banks: list[list[torch.Tensor]]) -> torch.Tensor:
        """Compute per-pass contribution weights."""
        flat = torch.cat([reg for bank in all_banks for reg in bank], dim=-1)
        return torch.sigmoid(self.gate_proj(flat))


# ══════════════════════════════════════════════════════════════════════
# Meta-S4 — Final structural summary
# ══════════════════════════════════════════════════════════════════════


class MetaS4(nn.Module):
    """Final intelligence scan: reads descending (most refined) banks.

    v4.1: reads bank_0, bank_1↓, bank_2↓, bank_3 — the most refined
    version of each level's output.
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
        meta_banks: list[list[torch.Tensor]],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Produce structural summary from most-refined banks."""
        B, L, D = residual.shape

        all_regs = [reg for bank in meta_banks for reg in bank]
        q_input = torch.cat(all_regs, dim=-1)
        q = self.q_proj(q_input)

        x = self.norm(residual)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn = torch.einsum("d,bld->bl", q, k) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        summary = torch.einsum("bl,bld->bd", attn_weights, v)
        out = self.out_proj(summary).unsqueeze(1).expand_as(residual)
        return residual + out


# ══════════════════════════════════════════════════════════════════════
# VSM-LM v4.1 — Full Recursive Viable System
# ══════════════════════════════════════════════════════════════════════


class VSMLMV4_1(nn.Module):
    """Viable System Model Language Model — v4.1 recursive architecture.

    Full bidirectional VSM: ascending (bottom-up) + descending (top-down).
    5 level-passes: L0↑, L1↑, L2, L1↓, L0↓.
    """

    REGISTER_NAMES = ("type", "scope", "role")
    PHASE_NAMES = ("prep", "converge", "consolidate")
    N_LEVELS = 3
    N_PASSES = 5  # L0↑, L1↑, L2, L1↓, L0↓

    # Named passes for clarity
    PASS_NAMES = ("L0_asc", "L1_asc", "L2_apex", "L1_desc", "L0_desc")

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
        self.n_passes = self.N_PASSES

        # Bank layout:
        #   0: bank_0 (init)
        #   1: bank_1↑ (L0 ascending)
        #   2: bank_2↑ (L1 ascending)
        #   3: bank_3  (L2 apex)
        #   4: bank_2↓ (L1 descending)
        #   5: bank_1↓ (L0 descending)
        self.n_banks = 6  # bank_0 + 3 ascending + 2 descending

        self.n_prep_layers = n_prep_layers
        self.n_converge_layers = n_converge_layers
        self.n_consolidate_layers = n_consolidate_layers

        # ── Progressive stride allocation per level ───────────────
        s1, s8, s64, s512 = strides[0], strides[1], strides[2], strides[3]
        self.level_configs = [
            # Level 0 (local-heavy): s1×3, s8×3, s64×1, s512×1
            [(s1, window)] * 3 + [(s8, window)] * 3 + [(s64, window)] * 1 + [(s512, window)] * 1,
            # Level 1 (balanced): s1×2, s8×2, s64×2, s512×2
            [(s1, window)] * 2 + [(s8, window)] * 2 + [(s64, window)] * 2 + [(s512, window)] * 2,
            # Level 2 (clause/discourse-heavy): s1×1, s8×1, s64×3, s512×3
            [(s1, window)] * 1 + [(s8, window)] * 1 + [(s64, window)] * 3 + [(s512, window)] * 3,
        ]

        # ── S5: Identity (shared weights + embeddings) ────────────
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.output_norm = nn.LayerNorm(d_model)

        # Register bank 0: learnable init (S5 identity)
        self.register_inits = nn.ParameterDict({
            f"reg_{name}": nn.Parameter(torch.zeros(d_register))
            for name in self.REGISTER_NAMES
        })

        # Shared S1 operations (S5 coherence)
        self.prep_layers = nn.ModuleList([
            FFNLayer(d_model, d_ff, dropout)
            for _ in range(n_prep_layers)
        ])

        # Converge: per-level stride configs, shared weights
        self.converge_layers_base = nn.ModuleList([
            CompressorLayer(d_model, self.level_configs[0], d_ff, dropout)
            for _ in range(n_converge_layers)
        ])
        self.converge_layers_l2 = nn.ModuleList([
            CompressorLayer(d_model, self.level_configs[1], d_ff, dropout)
            for _ in range(n_converge_layers)
        ])
        self.converge_layers_l3 = nn.ModuleList([
            CompressorLayer(d_model, self.level_configs[2], d_ff, dropout)
            for _ in range(n_converge_layers)
        ])
        for i in range(n_converge_layers):
            self._tie_compressor_weights(self.converge_layers_base[i], self.converge_layers_l2[i])
            self._tie_compressor_weights(self.converge_layers_base[i], self.converge_layers_l3[i])

        # Consolidate: shared across levels
        self.consolidate_layers = nn.ModuleList([
            CompressorLayer(d_model, self.level_configs[1], d_ff_consolidate, dropout)
            for _ in range(n_consolidate_layers)
        ])

        # ── S4: Intelligence (one shared instance, wider bank capacity) ──
        self.s4 = S4Intelligence(
            d_model, d_register, self.n_registers,
            max_banks=self.n_banks,  # 6 banks max for descending pass
            dropout=dropout,
        )

        # ── S3: Per-pass control (5 independent instances) ───────
        # L0↑, L1↑, L2, L1↓, L0↓ — each has autonomous control
        self.s3_passes = nn.ModuleList([
            S3LevelControl(d_model, d_register, self.n_phases, self.n_registers)
            for _ in range(self.n_passes)
        ])

        # ── Meta-S4: Final structural summary (reads 4 best banks) ──
        # Reads: bank_0, bank_1↓, bank_2↓, bank_3
        self.meta_s4 = MetaS4(
            d_model, d_register, self.n_registers,
            n_banks=4,  # 4 most-refined banks
            dropout=dropout,
        )

        # ── Meta-S3: Per-pass contribution gates (5 passes) ─────
        self.meta_s3 = MetaS3(
            d_register, self.n_registers,
            n_banks=self.n_banks,
            n_passes=self.n_passes,
        )

        # ── Initialize ────────────────────────────────────────────
        self.apply(self._init_weights)

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
        """Initialize register bank 0 (S5 identity)."""
        return [
            self.register_inits[f"reg_{name}"].clone()
            for name in self.REGISTER_NAMES
        ]

    def _fresh_bank(self) -> list[torch.Tensor]:
        """Create a zero-initialized register bank."""
        device = self.register_inits["reg_type"].device
        return [
            torch.zeros(self.d_register, device=device)
            for _ in self.REGISTER_NAMES
        ]

    def _get_converge_layers(self, level: int) -> nn.ModuleList:
        """Get converge layers for a given level (0-indexed)."""
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

    def _run_level_pass(
        self,
        x: torch.Tensor,
        level: int,
        pass_idx: int,
        readable_banks: list[list[torch.Tensor]],
        target_bank: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Run one level-pass (ascending or descending).

        Args:
            x: residual stream (B, L, D)
            level: which level's converge config to use (0, 1, 2)
            pass_idx: index into s3_passes (0-4)
            readable_banks: banks this pass's S4 can read
            target_bank: bank this pass writes to

        Returns:
            x: updated residual
            target_bank: updated bank
            level_delta: (B, L, D) this pass's contribution
        """
        x_before = x

        # S4: scan residual conditioned on readable banks
        s4_updates, _ = self.s4(readable_banks, x)
        target_bank = [
            target_bank[i] + s4_updates[i]
            for i in range(self.n_registers)
        ]

        # Phase 1: PREP
        prep_out = self._run_prep(x)
        delta = prep_out - x
        gated_delta, target_bank, _, _ = self.s3_passes[pass_idx].gate_phase(
            target_bank, delta, 0)
        x = x + gated_delta

        # Phase 2: CONVERGE
        converge_out = self._run_converge(x, level)
        delta = converge_out - x
        gated_delta, target_bank, _, _ = self.s3_passes[pass_idx].gate_phase(
            target_bank, delta, 1)
        x = x + gated_delta

        # Phase 3: CONSOLIDATE
        consolidate_out = self._run_consolidate(x)
        delta = consolidate_out - x
        gated_delta, target_bank, _, _ = self.s3_passes[pass_idx].gate_phase(
            target_bank, delta, 2)
        x = x + gated_delta

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

        # ── Register banks ────────────────────────────────────────
        # Index: 0=bank_0, 1=bank_1↑, 2=bank_2↑, 3=bank_3, 4=bank_2↓, 5=bank_1↓
        bank_0 = self._init_bank0()
        bank_1_asc = self._fresh_bank()
        bank_2_asc = self._fresh_bank()
        bank_3 = self._fresh_bank()
        bank_2_desc = self._fresh_bank()
        bank_1_desc = self._fresh_bank()

        pass_deltas = []  # 5 deltas, one per pass

        # ── ASCENDING: L0↑ → L1↑ → L2 ───────────────────────────

        # Pass 0: L0↑ — reads bank_0
        x, bank_1_asc, delta = self._run_level_pass(
            x, level=0, pass_idx=0,
            readable_banks=[bank_0],
            target_bank=bank_1_asc,
        )
        pass_deltas.append(delta)

        # Pass 1: L1↑ — reads bank_0, bank_1↑
        x, bank_2_asc, delta = self._run_level_pass(
            x, level=1, pass_idx=1,
            readable_banks=[bank_0, bank_1_asc],
            target_bank=bank_2_asc,
        )
        pass_deltas.append(delta)

        # Pass 2: L2 (apex) — reads bank_0, bank_1↑, bank_2↑
        x, bank_3, delta = self._run_level_pass(
            x, level=2, pass_idx=2,
            readable_banks=[bank_0, bank_1_asc, bank_2_asc],
            target_bank=bank_3,
        )
        pass_deltas.append(delta)

        # ── DESCENDING: L1↓ → L0↓ ────────────────────────────────

        # Pass 3: L1↓ — reads bank_0, bank_1↑, bank_2↑, bank_3
        x, bank_2_desc, delta = self._run_level_pass(
            x, level=1, pass_idx=3,
            readable_banks=[bank_0, bank_1_asc, bank_2_asc, bank_3],
            target_bank=bank_2_desc,
        )
        pass_deltas.append(delta)

        # Pass 4: L0↓ — reads bank_0, bank_1↑, bank_2↓, bank_3
        x, bank_1_desc, delta = self._run_level_pass(
            x, level=0, pass_idx=4,
            readable_banks=[bank_0, bank_1_asc, bank_2_desc, bank_3],
            target_bank=bank_1_desc,
        )
        pass_deltas.append(delta)

        # ── Meta-S3: per-pass contribution gates ──────────────────
        all_banks = [bank_0, bank_1_asc, bank_2_asc, bank_3, bank_2_desc, bank_1_desc]
        meta_gates = self.meta_s3(all_banks)  # (5,)

        total_ungated = sum(pass_deltas)
        total_gated = sum(
            meta_gates[i] * pass_deltas[i]
            for i in range(self.n_passes)
        )
        x = x - total_ungated + total_gated

        # ── Meta-S4: final structural summary (most refined banks) ──
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

        # Register banks
        bank_0 = self._init_bank0()
        bank_1_asc = self._fresh_bank()
        bank_2_asc = self._fresh_bank()
        bank_3 = self._fresh_bank()
        bank_2_desc = self._fresh_bank()
        bank_1_desc = self._fresh_bank()

        for i, name in enumerate(reg_names):
            metrics[f"register_{name}_init_norm"] = bank_0[i].norm().item()

        pass_deltas = []

        # ── Define pass schedule ──────────────────────────────────
        pass_schedule = [
            # (pass_idx, level, pass_name, readable_banks_fn, target_bank_name)
            (0, 0, "L0_asc"),
            (1, 1, "L1_asc"),
            (2, 2, "L2_apex"),
            (3, 1, "L1_desc"),
            (4, 0, "L0_desc"),
        ]

        for pass_idx, level, pass_name in pass_schedule:
            pfx = pass_name

            # Determine readable banks and target bank for this pass
            if pass_idx == 0:  # L0↑
                readable = [bank_0]
                target_bank = bank_1_asc
            elif pass_idx == 1:  # L1↑
                readable = [bank_0, bank_1_asc]
                target_bank = bank_2_asc
            elif pass_idx == 2:  # L2
                readable = [bank_0, bank_1_asc, bank_2_asc]
                target_bank = bank_3
            elif pass_idx == 3:  # L1↓
                readable = [bank_0, bank_1_asc, bank_2_asc, bank_3]
                target_bank = bank_2_desc
            elif pass_idx == 4:  # L0↓
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
                metrics[f"{pfx}_reg_{name}_after_s4"] = target_bank[i].norm().item()
            s4_entropy = -(s4_attn * (s4_attn + 1e-10).log()).sum(dim=-1).mean()
            metrics[f"{pfx}_s4_attn_entropy"] = s4_entropy.item()

            # Phase 1: PREP
            prep_out = self._run_prep(x)
            delta = prep_out - x
            gated_delta, target_bank, gate_vals, write_gates = (
                self.s3_passes[pass_idx].gate_phase(target_bank, delta, 0))
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
            gated_delta, target_bank, gate_vals, write_gates = (
                self.s3_passes[pass_idx].gate_phase(target_bank, delta, 1))
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
            gated_delta, target_bank, gate_vals, write_gates = (
                self.s3_passes[pass_idx].gate_phase(target_bank, delta, 2))
            x = x + gated_delta

            metrics[f"{pfx}_consolidate_delta_norm"] = delta.norm(dim=-1).mean().item()
            metrics[f"{pfx}_consolidate_gated_norm"] = gated_delta.norm(dim=-1).mean().item()
            metrics[f"{pfx}_consolidate_gate_mean"] = gate_vals.mean().item()
            metrics[f"{pfx}_consolidate_gate_std"] = gate_vals.std().item()
            metrics[f"{pfx}_after_consolidate"] = x.norm(dim=-1).mean().item()
            for i, rn in enumerate(reg_names):
                metrics[f"{pfx}_consolidate_write_{rn}"] = write_gates[i]

            # Register norms after pass
            for i, name in enumerate(reg_names):
                metrics[f"{pfx}_register_{name}_norm"] = target_bank[i].norm().item()

            # Write back the target bank
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

        # ── Also emit level-indexed metrics for v4 compatibility ──
        # Map: L0_asc→level0, L1_asc→level1, L2_apex→level2
        level_map = {
            "L0_asc": "level0", "L1_asc": "level1", "L2_apex": "level2",
            "L1_desc": "level1_desc", "L0_desc": "level0_desc",
        }
        for pass_name, level_pfx in level_map.items():
            for key in list(metrics.keys()):
                if key.startswith(pass_name + "_"):
                    suffix = key[len(pass_name) + 1:]
                    metrics[f"{level_pfx}_{suffix}"] = metrics[key]

        # Backward-compat iter aliases (level0→iter0, level1→iter1)
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
        # Also emit v4-compat meta_s3_gate_level{i} (ascending passes only)
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
            "VSM-LM v4.1 — Full Recursive Viable System Architecture",
            f"  d_model={self.d_model}, d_register={self.d_register}×{self.n_registers}, "
            f"seq_len={self.max_len}",
            f"  Passes: {self.n_passes} (L0↑, L1↑, L2, L1↓, L0↓)",
            f"  Phase structure: prep({self.n_prep_layers}L, FFN) → "
            f"converge({self.n_converge_layers}L, attn) → "
            f"consolidate({self.n_consolidate_layers}L, wide-FFN)",
            f"  Strides: {self.strides} (4 scales, progressive reallocation)",
        ]
        for i, config in enumerate(self.level_configs):
            lines.append(f"    Level {i}: {_stride_desc(config)}")
        lines.extend([
            f"  S5: Shared weights across all passes (identity coherence)",
            f"  S4: Bidirectional register scan (ascending + descending banks)",
            f"  S3: 5 independent instances (per-pass autonomous control)",
            f"  S2: Register bank protocol + residual stream (coordination)",
            f"  Register banks: {self.n_banks} (1 init + 3 ascending + 2 descending)",
            f"  Meta-S4: Final structural summary (4 most-refined banks)",
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
