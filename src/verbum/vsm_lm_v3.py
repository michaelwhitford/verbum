"""VSM-LM v3 — Progressive Binding Compressor.

Two changes from v2, grounded in F65-F68 binding shape findings:

1. **Three partitioned registers** (type/scope/role, 128 dims each).
   Each S1 phase writes primarily to its natural register via learned
   soft-partition gates. The partition mirrors F66's three progressive
   binding stages and the existing stride hierarchy.

2. **Deeper FFN per phase** (2 CompressorLayers per phase instead of 1).
   Doubles FFN passes from 6 to 12. F68 showed binding is in the FFNs,
   not attention heads — depth is the binding variable.

Everything else unchanged: same strides (1, 8, 64), same W=8, same
2-iteration loop with weight sharing, same O(L) attention.

See: mementum/knowledge/explore/vsm-lm-v3-architecture.md

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
# S4 — Intelligence (3-register variant)
# ══════════════════════════════════════════════════════════════════════


class S4Intelligence3R(nn.Module):
    """Register cross-attention for three partitioned registers.

    Concatenates all registers into a single query, cross-attends to
    the residual stream, then splits the summary back into per-register
    updates.

    Runs per-iteration (same as v2).
    """

    def __init__(self, d_model: int, d_register: int, n_registers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_register = d_register
        self.n_registers = n_registers
        self.d_query = d_register * n_registers  # 384

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
        """Cross-attend and update all registers.

        Args:
            registers: list of 3 tensors, each (d_register,)
            residual: (B, L, d_model)

        Returns:
            updated_registers: list of 3 tensors
            attn_weights: (B, L) — detached for instrumentation
        """
        B, L, D = residual.shape

        # Concatenate registers → single query
        q_input = torch.cat(registers, dim=-1)  # (d_query,)
        q = self.q_proj(q_input)                # (d_model,)

        x = self.norm(residual)
        k = self.k_proj(x)  # (B, L, d_model)
        v = self.v_proj(x)  # (B, L, d_model)

        # Cross-attention: register queries the residual
        attn = torch.einsum("d,bld->bl", q, k) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        summary = torch.einsum("bl,bld->bd", attn_weights, v)  # (B, d_model)
        summary = summary.mean(dim=0)  # (d_model,) — batch-mean

        # Project back to register space and split
        updates = self.summary_proj(summary)  # (d_query,)
        update_splits = updates.split(self.d_register, dim=-1)

        # Residual update per register
        updated = [
            reg + upd for reg, upd in zip(registers, update_splits)
        ]

        return updated, attn_weights.detach()


# ══════════════════════════════════════════════════════════════════════
# S3 — Control (3-register soft-partitioned writes)
# ══════════════════════════════════════════════════════════════════════


class S3ControlV3(nn.Module):
    """Per-phase, per-iteration gating with soft-partitioned register writes.

    6 gate heads (3 phases × 2 iterations) gate the residual stream delta.
    9 write paths (3 phases × 3 registers) update registers with learned
    soft partition — each phase CAN write to any register, but the write
    gates learn to bias toward the natural mapping:
      type phase  → reg_type
      parse phase → reg_scope
      apply phase → reg_role
    """

    def __init__(self, d_model: int, d_register: int, n_phases: int = 3,
                 n_iterations: int = 2, n_registers: int = 3):
        super().__init__()
        self.d_model = d_model
        self.d_register = d_register
        self.n_phases = n_phases
        self.n_iterations = n_iterations
        self.n_registers = n_registers

        # Gate input: all registers concatenated + delta summary
        gate_input_dim = d_register * n_registers + d_model  # 384 + 256 = 640

        # Per-phase, per-iteration gate heads
        self.gate_heads = nn.ModuleList([
            nn.Linear(gate_input_dim, d_model)
            for _ in range(n_phases * n_iterations)
        ])

        # Per-phase, per-register write paths (soft partition)
        # 3 phases × 3 registers = 9 write projections
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
        """Gate one S1 phase's contribution and update registers.

        Args:
            registers: list of 3 tensors, each (d_register,)
            delta: (B, L, d_model)
            phase_idx: which phase (0=type, 1=parse, 2=apply)
            iteration: which iteration (0 or 1)

        Returns:
            gated_delta: (B, L, d_model)
            updated_registers: list of 3 tensors
            gate_values: (d_model,) — detached
            write_gate_values: list of 3 floats — per-register write gates
        """
        summary = delta.mean(dim=(0, 1))  # (d_model,)

        # Gate input: all registers + delta summary
        reg_concat = torch.cat(registers, dim=-1)  # (d_query,)
        gate_input = torch.cat([reg_concat, summary])  # (gate_input_dim,)

        # Select iteration-specific gate head
        head_idx = iteration * self.n_phases + phase_idx
        gate = torch.sigmoid(self.gate_heads[head_idx](gate_input))
        gated_delta = gate.unsqueeze(0).unsqueeze(0) * delta

        # Soft-partitioned register writes
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
# VSM-LM v3
# ══════════════════════════════════════════════════════════════════════


class VSMLMV3(nn.Module):
    """Viable System Model Language Model — v3 progressive binding compressor.

    Two changes from v2:
    - Three partitioned registers (type/scope/role) instead of one
    - 2 CompressorLayers per S1 phase instead of 1 (deeper FFN)
    """

    def __init__(
        self,
        vocab_size: int = 50277,
        d_model: int = 512,
        d_register: int = 256,
        max_len: int = 4096,
        n_heads: int = 8,
        d_ff: int = 1536,
        window: int = 8,
        strides: tuple[int, ...] = (1, 8, 64),
        n_iterations: int = 2,
        n_layers_per_phase: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_register = d_register
        self.max_len = max_len
        self.n_iterations = n_iterations
        self.n_layers_per_phase = n_layers_per_phase
        self.window = window
        self.strides = strides
        self.n_registers = len(strides)  # 3: one per phase/binding stage

        # ── S5: Identity ──────────────────────────────────────────
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.output_norm = nn.LayerNorm(d_model)

        # Three register inits (type/scope/role)
        self.register_type_init = nn.Parameter(torch.zeros(d_register))
        self.register_scope_init = nn.Parameter(torch.zeros(d_register))
        self.register_role_init = nn.Parameter(torch.zeros(d_register))

        # ── S4: Intelligence (3-register) ─────────────────────────
        self.s4 = S4Intelligence3R(d_model, d_register, self.n_registers, dropout)

        # ── S3: Control (soft-partitioned writes) ─────────────────
        self.s3 = S3ControlV3(
            d_model, d_register,
            n_phases=len(strides),
            n_iterations=n_iterations,
            n_registers=self.n_registers,
        )

        # ── S1: Operations (2-layer stacks per phase) ─────────────
        self.s1_stacks = nn.ModuleList([
            nn.ModuleList([
                CompressorLayer(
                    d_model,
                    [(stride, window)] * n_heads,
                    d_ff,
                    dropout,
                )
                for _ in range(n_layers_per_phase)
            ])
            for stride in strides
        ])
        self.phase_names = ["type", "parse", "apply"]

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
        """Clone initial register states for a forward pass."""
        return [
            self.register_type_init.clone(),
            self.register_scope_init.clone(),
            self.register_role_init.clone(),
        ]

    def _run_phase_stack(self, stack: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        """Run a phase's layer stack and return the output."""
        h = x
        for layer in stack:
            h = layer(h)
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L = input_ids.shape
        device = input_ids.device

        # ── S5: Identity ──────────────────────────────────────────
        positions = torch.arange(L, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        registers = self._init_registers()

        # ── Iteration loop ────────────────────────────────────────
        for iteration in range(self.n_iterations):
            # S4: scan residual with all registers
            registers, _ = self.s4(registers, x)

            # S1 phases with S3 control
            for phase_idx, stack in enumerate(self.s1_stacks):
                phase_out = self._run_phase_stack(stack, x)
                delta = phase_out - x
                gated_delta, registers, _, _ = self.s3.gate_phase(
                    registers, delta, phase_idx, iteration,
                )
                x = x + gated_delta

        # ── S5: Output ────────────────────────────────────────────
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
        reg_names = ["type", "scope", "role"]

        # ── S5: Identity ──────────────────────────────────────────
        positions = torch.arange(L, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        registers = self._init_registers()

        metrics["embed_norm"] = x.norm(dim=-1).mean().item()
        for i, name in enumerate(reg_names):
            metrics[f"register_{name}_init_norm"] = registers[i].norm().item()

        # ── Iteration loop ────────────────────────────────────────
        for it in range(self.n_iterations):
            pfx = f"iter{it}"

            # S4: per-iteration scan
            registers, s4_attn = self.s4(registers, x)
            for i, name in enumerate(reg_names):
                metrics[f"{pfx}_reg_{name}_after_s4"] = registers[i].norm().item()

            # S4 attention entropy
            s4_entropy = -(s4_attn * (s4_attn + 1e-10).log()).sum(dim=-1).mean()
            metrics[f"{pfx}_s4_attn_entropy"] = s4_entropy.item()

            # S1 phases with S3 control
            for phase_idx, (stack, phase_name) in enumerate(
                zip(self.s1_stacks, self.phase_names)
            ):
                phase_out = self._run_phase_stack(stack, x)
                delta = phase_out - x
                gated_delta, registers, gate_vals, write_gates = self.s3.gate_phase(
                    registers, delta, phase_idx, it,
                )
                x = x + gated_delta

                # Standard metrics
                metrics[f"{pfx}_{phase_name}_delta_norm"] = (
                    delta.norm(dim=-1).mean().item()
                )
                metrics[f"{pfx}_{phase_name}_gated_norm"] = (
                    gated_delta.norm(dim=-1).mean().item()
                )
                metrics[f"{pfx}_{phase_name}_gate_mean"] = gate_vals.mean().item()
                metrics[f"{pfx}_{phase_name}_gate_std"] = gate_vals.std().item()
                metrics[f"{pfx}_{phase_name}_gate_min"] = gate_vals.min().item()
                metrics[f"{pfx}_{phase_name}_gate_max"] = gate_vals.max().item()
                metrics[f"{pfx}_after_{phase_name}"] = (
                    x.norm(dim=-1).mean().item()
                )

                # Per-register write gate values (soft partition signal)
                for i, reg_name in enumerate(reg_names):
                    metrics[f"{pfx}_{phase_name}_write_{reg_name}"] = write_gates[i]

                # Per-register norms after this phase
                for i, reg_name in enumerate(reg_names):
                    metrics[f"{pfx}_{phase_name}_reg_{reg_name}_norm"] = (
                        registers[i].norm().item()
                    )

            # Per-iteration register norms
            for i, name in enumerate(reg_names):
                metrics[f"{pfx}_register_{name}_norm"] = registers[i].norm().item()

        # Backward-compat aliases for probing pipeline
        metrics["s4_attn_entropy"] = metrics["iter0_s4_attn_entropy"]
        metrics["register_after_s4"] = sum(
            metrics[f"iter0_reg_{n}_after_s4"] for n in reg_names
        )

        metrics["output_norm"] = x.norm(dim=-1).mean().item()
        metrics["overall_expansion"] = (
            metrics["output_norm"] / metrics["embed_norm"]
        )

        # ── S5: Output ────────────────────────────────────────────
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
            self.register_type_init.numel()
            + self.register_scope_init.numel()
            + self.register_role_init.numel()
            + sum(p.numel() for p in self.output_norm.parameters())
        )
        s4 = sum(p.numel() for p in self.s4.parameters())
        s3 = sum(p.numel() for p in self.s3.parameters())
        s1 = sum(p.numel() for p in self.s1_stacks.parameters())
        total = s5_embed + s5_pos + s5_other + s4 + s3 + s1
        return {
            "S5_token_embeddings": s5_embed,
            "S5_positional": s5_pos,
            "S5_other": s5_other,
            "S4_intelligence": s4,
            "S3_control": s3,
            "S1_operations": s1,
            "total": total,
        }

    def describe(self) -> str:
        n_layers = self.n_layers_per_phase
        phases = " → ".join(
            f"{n}(s={s}, {n_layers}L)"
            for n, s in zip(self.phase_names, self.strides)
        )
        return "\n".join([
            "VSM-LM v3 — Progressive Binding Compressor",
            f"  d_model={self.d_model}, d_register={self.d_register}×3, "
            f"seq_len={self.max_len}, iterations={self.n_iterations}",
            f"  S1: {phases}",
            f"  S4: 3-register cross-attention (per-iteration)",
            f"  S3: per-phase per-iteration gating "
            f"({len(self.strides)} phases × {self.n_iterations} iters "
            f"= {len(self.strides) * self.n_iterations} gates) "
            f"+ {len(self.strides) * self.n_registers} soft-partition writes",
            f"  Window: {self.window}",
            f"  FFN passes/forward: {len(self.strides) * n_layers * self.n_iterations}",
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
