"""VSM-LM v3.2 — Convergence Architecture (probe-informed redesign).

Redesigned from probing results (run_compression_map.py):

  Q1: Dominant direction at L6+ is WORD POSITION (r=0.49)
  Q2: FFN preps (L4-5) → Attention converges (L6-9) → FFN consolidates (L10-11)
  Q3: 95% of representation is in the dominant PC; residual carries position
  Q4: Convergence is primarily syntactic (7/12 layer-votes)

Key architectural insights:
  - NO POOLING. Qwen keeps all positions throughout. So do we.
  - Multi-scale compression via STRIDED ATTENTION (like v3), not spatial reduction
  - The compression mechanism is within-span attention convergence:
    tokens in the same constituent attend to each other and become similar
  - Three-phase structure per iteration: prep → converge → consolidate

Architecture:
  Phase 1 "prep" — FFN-only (no attention)
    Builds per-token features. Analogous to Qwen L0-5.
    Each token independently determines its type/role/position features.
    The probes show FFN is critical at L4-5 for preparing representations.

  Phase 2 "converge" — Multi-scale attention (cube mode: all strides active)
    Tokens attend to others at multiple scales SIMULTANEOUSLY:
      stride 1  (3 heads): local 8-token convergence (word/morpheme)
      stride 8  (3 heads): phrase 64-token convergence (NP, VP, PP)
      stride 64 (2 heads): clause 512-token convergence (binding, scope)
    This is the actual compression: within-constituent similarity increases.
    Analogous to Qwen L6-9 where attention is critical.

  Phase 3 "consolidate" — FFN-heavy with light attention
    Solidifies the converged representation. Wider FFN for more capacity.
    Analogous to Qwen L10-11 where FFN is critical again.

Each iteration cycles through all three phases. Two iterations total.
Registers (type/scope/role) track convergence state across iterations.
S3 gates each phase. S4 scans the residual between iterations.

This gives: 3 phases × 2 iterations = 6 gated phase applications per forward.
With 2 layers in converge and 1 layer each in prep/consolidate = 8 layer
evaluations per iteration, 16 total. Same depth as v3 (16 FFN passes/forward).

Changes from v3:
  - Explicit prep/converge/consolidate phases (informed by probing)
  - Cube-mode multi-scale in converge (all strides fire together)
  - Wider FFN in consolidate phase (2048 vs 1536)
  - Prep phase is FFN-only (no attention — probing shows FFN critical L4-5)
  - No pooling, no 4th stride. Sequence stays at 4096.
  - Back to 3 registers (type/scope/role) — confirmed by v3 binding probes.

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
# FFN-only layer (prep and consolidate phases)
# ══════════════════════════════════════════════════════════════════════


class FFNLayer(nn.Module):
    """Pre-norm FFN layer without attention.

    For the prep phase: per-token feature building without cross-position
    communication. Each token independently builds its feature vector.
    For consolidate: wider FFN to solidify converged representations.
    """

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
# S4 — Intelligence (3-register, same as v3)
# ══════════════════════════════════════════════════════════════════════


class S4Intelligence(nn.Module):
    """Register cross-attention for 3 partitioned registers."""

    def __init__(self, d_model: int, d_register: int, n_registers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_register = d_register
        self.n_registers = n_registers
        self.d_query = d_register * n_registers

        self.scale = d_model ** -0.5

        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(self.d_query, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.summary_proj = nn.Linear(d_model, self.d_query, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        registers: list[torch.Tensor],
        residual: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        B, L, D = residual.shape

        q_input = torch.cat(registers, dim=-1)
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

        updated = [reg + upd for reg, upd in zip(registers, update_splits)]
        return updated, attn_weights.detach()


# ══════════════════════════════════════════════════════════════════════
# S3 — Control (3-phase × 3-register soft-partitioned writes)
# ══════════════════════════════════════════════════════════════════════


class S3Control(nn.Module):
    """Per-phase, per-iteration gating with soft-partitioned register writes."""

    def __init__(self, d_model: int, d_register: int, n_phases: int,
                 n_iterations: int, n_registers: int):
        super().__init__()
        self.d_model = d_model
        self.d_register = d_register
        self.n_phases = n_phases
        self.n_iterations = n_iterations
        self.n_registers = n_registers

        gate_input_dim = d_register * n_registers + d_model

        self.gate_heads = nn.ModuleList([
            nn.Linear(gate_input_dim, d_model)
            for _ in range(n_phases * n_iterations)
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
        iteration: int = 0,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, list[float]]:
        summary = delta.mean(dim=(0, 1))

        reg_concat = torch.cat(registers, dim=-1)
        gate_input = torch.cat([reg_concat, summary])

        head_idx = iteration * self.n_phases + phase_idx
        gate = torch.sigmoid(self.gate_heads[head_idx](gate_input))
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
# VSM-LM v3.2 — Convergence Architecture
# ══════════════════════════════════════════════════════════════════════


class VSMLMV3_2(nn.Module):
    """Viable System Model Language Model — v3.2 convergence architecture.

    Probe-informed design: prep → converge → consolidate.
    Full 4096 sequence throughout. Multi-scale via strides, not pooling.
    """

    REGISTER_NAMES = ("type", "scope", "role")
    PHASE_NAMES = ("prep", "converge", "consolidate")

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
        strides: tuple[int, ...] = (1, 8, 64),
        n_iterations: int = 2,
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
        self.n_iterations = n_iterations
        self.window = window
        self.strides = strides
        self.n_registers = len(self.REGISTER_NAMES)
        self.n_phases = len(self.PHASE_NAMES)

        self.n_prep_layers = n_prep_layers
        self.n_converge_layers = n_converge_layers
        self.n_consolidate_layers = n_consolidate_layers

        # Head distribution for cube-mode converge phase
        # 3+3+2 = 8 heads across three strides (all active simultaneously)
        n_s1 = 3  # local heads (stride 1)
        n_s8 = 3  # phrase heads (stride 8)
        n_s64 = n_heads - n_s1 - n_s8  # clause heads (stride 64)
        self.cube_config = (
            [(strides[0], window)] * n_s1 +
            [(strides[1], window)] * n_s8 +
            [(strides[2], window)] * n_s64
        )

        # ── S5: Identity ──────────────────────────────────────────
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.output_norm = nn.LayerNorm(d_model)

        self.register_inits = nn.ParameterDict({
            f"reg_{name}": nn.Parameter(torch.zeros(d_register))
            for name in self.REGISTER_NAMES
        })

        # ── S4: Intelligence ──────────────────────────────────────
        self.s4 = S4Intelligence(d_model, d_register, self.n_registers, dropout)

        # ── S3: Control ───────────────────────────────────────────
        self.s3 = S3Control(
            d_model, d_register,
            n_phases=self.n_phases,
            n_iterations=n_iterations,
            n_registers=self.n_registers,
        )

        # ── S1: Operations ────────────────────────────────────────

        # Phase 1: PREP — FFN-only, no attention
        # Per-token feature building (analogous to Qwen L0-5, FFN critical)
        self.prep_layers = nn.ModuleList([
            FFNLayer(d_model, d_ff, dropout)
            for _ in range(n_prep_layers)
        ])

        # Phase 2: CONVERGE — Multi-scale attention (cube mode)
        # All strides active simultaneously — tokens converge within spans
        # Analogous to Qwen L6-9 where attention is critical
        self.converge_layers = nn.ModuleList([
            CompressorLayer(d_model, self.cube_config, d_ff, dropout)
            for _ in range(n_converge_layers)
        ])

        # Phase 3: CONSOLIDATE — FFN-heavy with light attention
        # Solidifies converged representation (analogous to Qwen L10-11)
        # Wider FFN for more consolidation capacity
        self.consolidate_layers = nn.ModuleList([
            CompressorLayer(
                d_model,
                self.cube_config,  # keep multi-scale attention
                d_ff_consolidate,  # wider FFN
                dropout,
            )
            for _ in range(n_consolidate_layers)
        ])

        # ── Initialize ────────────────────────────────────────────
        self.apply(self._init_weights)

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

    def _init_registers(self) -> list[torch.Tensor]:
        return [
            self.register_inits[f"reg_{name}"].clone()
            for name in self.REGISTER_NAMES
        ]

    def _run_prep(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 1: FFN-only per-token feature building."""
        for layer in self.prep_layers:
            x = layer(x)
        return x

    def _run_converge(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 2: Multi-scale attention convergence."""
        for layer in self.converge_layers:
            x = layer(x)
        return x

    def _run_consolidate(self, x: torch.Tensor) -> torch.Tensor:
        """Phase 3: FFN-heavy consolidation."""
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
        registers = self._init_registers()

        for iteration in range(self.n_iterations):
            # S4: register scans residual
            registers, _ = self.s4(registers, x)

            # Phase 1: PREP (FFN-only)
            prep_out = self._run_prep(x)
            delta = prep_out - x
            gated_delta, registers, _, _ = self.s3.gate_phase(
                registers, delta, 0, iteration)
            x = x + gated_delta

            # Phase 2: CONVERGE (multi-scale attention)
            converge_out = self._run_converge(x)
            delta = converge_out - x
            gated_delta, registers, _, _ = self.s3.gate_phase(
                registers, delta, 1, iteration)
            x = x + gated_delta

            # Phase 3: CONSOLIDATE (FFN-heavy)
            consolidate_out = self._run_consolidate(x)
            delta = consolidate_out - x
            gated_delta, registers, _, _ = self.s3.gate_phase(
                registers, delta, 2, iteration)
            x = x + gated_delta

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
        """Forward pass with full instrumentation."""
        B, L = input_ids.shape
        device = input_ids.device
        metrics: dict = {}
        reg_names = list(self.REGISTER_NAMES)

        positions = torch.arange(L, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        registers = self._init_registers()

        metrics["embed_norm"] = x.norm(dim=-1).mean().item()
        for i, name in enumerate(reg_names):
            metrics[f"register_{name}_init_norm"] = registers[i].norm().item()

        for it in range(self.n_iterations):
            pfx = f"iter{it}"

            # S4
            registers, s4_attn = self.s4(registers, x)
            for i, name in enumerate(reg_names):
                metrics[f"{pfx}_reg_{name}_after_s4"] = registers[i].norm().item()
            s4_entropy = -(s4_attn * (s4_attn + 1e-10).log()).sum(dim=-1).mean()
            metrics[f"{pfx}_s4_attn_entropy"] = s4_entropy.item()

            # Phase 1: PREP
            prep_out = self._run_prep(x)
            delta = prep_out - x
            gated_delta, registers, gate_vals, write_gates = self.s3.gate_phase(
                registers, delta, 0, it)
            x = x + gated_delta

            metrics[f"{pfx}_prep_delta_norm"] = delta.norm(dim=-1).mean().item()
            metrics[f"{pfx}_prep_gated_norm"] = gated_delta.norm(dim=-1).mean().item()
            metrics[f"{pfx}_prep_gate_mean"] = gate_vals.mean().item()
            metrics[f"{pfx}_prep_gate_std"] = gate_vals.std().item()
            metrics[f"{pfx}_after_prep"] = x.norm(dim=-1).mean().item()
            for i, rn in enumerate(reg_names):
                metrics[f"{pfx}_prep_write_{rn}"] = write_gates[i]

            # Phase 2: CONVERGE
            converge_out = self._run_converge(x)
            delta = converge_out - x
            gated_delta, registers, gate_vals, write_gates = self.s3.gate_phase(
                registers, delta, 1, it)
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
            gated_delta, registers, gate_vals, write_gates = self.s3.gate_phase(
                registers, delta, 2, it)
            x = x + gated_delta

            metrics[f"{pfx}_consolidate_delta_norm"] = delta.norm(dim=-1).mean().item()
            metrics[f"{pfx}_consolidate_gated_norm"] = gated_delta.norm(dim=-1).mean().item()
            metrics[f"{pfx}_consolidate_gate_mean"] = gate_vals.mean().item()
            metrics[f"{pfx}_consolidate_gate_std"] = gate_vals.std().item()
            metrics[f"{pfx}_after_consolidate"] = x.norm(dim=-1).mean().item()
            for i, rn in enumerate(reg_names):
                metrics[f"{pfx}_consolidate_write_{rn}"] = write_gates[i]

            # Per-iteration register norms
            for i, name in enumerate(reg_names):
                metrics[f"{pfx}_register_{name}_norm"] = registers[i].norm().item()

        # Backward-compat aliases
        metrics["s4_attn_entropy"] = metrics["iter0_s4_attn_entropy"]
        metrics["register_after_s4"] = sum(
            metrics[f"iter0_reg_{n}_after_s4"] for n in reg_names
        )
        metrics["output_norm"] = x.norm(dim=-1).mean().item()
        metrics["overall_expansion"] = metrics["output_norm"] / metrics["embed_norm"]

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
        s5_embed = sum(p.numel() for p in self.token_embed.parameters())
        s5_pos = sum(p.numel() for p in self.pos_embed.parameters())
        s5_other = (
            sum(p.numel() for p in self.register_inits.parameters())
            + sum(p.numel() for p in self.output_norm.parameters())
        )
        s4 = sum(p.numel() for p in self.s4.parameters())
        s3 = sum(p.numel() for p in self.s3.parameters())
        s1_prep = sum(p.numel() for p in self.prep_layers.parameters())
        s1_converge = sum(p.numel() for p in self.converge_layers.parameters())
        s1_consolidate = sum(p.numel() for p in self.consolidate_layers.parameters())
        s1 = s1_prep + s1_converge + s1_consolidate
        total = s5_embed + s5_pos + s5_other + s4 + s3 + s1
        return {
            "S5_token_embeddings": s5_embed,
            "S5_positional": s5_pos,
            "S5_other": s5_other,
            "S4_intelligence": s4,
            "S3_control": s3,
            "S1_prep": s1_prep,
            "S1_converge": s1_converge,
            "S1_consolidate": s1_consolidate,
            "S1_total": s1,
            "total": total,
        }

    def describe(self) -> str:
        strides_str = "+".join(f"s{s}×{n}" for s, n in
                               sorted(set((s, self.cube_config.count((s, self.window)))
                                          for s, _ in self.cube_config)))
        ffn_per_iter = (self.n_prep_layers + self.n_converge_layers + self.n_consolidate_layers)
        ffn_total = ffn_per_iter * self.n_iterations

        return "\n".join([
            "VSM-LM v3.2 — Convergence Architecture (probe-informed)",
            f"  d_model={self.d_model}, d_register={self.d_register}×{self.n_registers}, "
            f"seq_len={self.max_len}, iterations={self.n_iterations}",
            f"  Phase structure: prep({self.n_prep_layers}L, FFN-only) → "
            f"converge({self.n_converge_layers}L, cube-attn) → "
            f"consolidate({self.n_consolidate_layers}L, wide-FFN)",
            f"  Converge heads: {strides_str} (cube mode, all scales simultaneous)",
            f"  S4: 3-register cross-attention (per-iteration)",
            f"  S3: 3 phases × 2 iters = 6 gates + 9 soft-partition writes",
            f"  Registers: type × scope × role",
            f"  FFN passes/forward: {ffn_total} ({ffn_per_iter}/iter × {self.n_iterations})",
            f"  Sequence: {self.max_len} positions throughout (no pooling)",
            f"  Grounding: Qwen probe shows FFN→Attn→FFN is the compression shape",
        ])

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
