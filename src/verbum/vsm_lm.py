"""VSM-LM — Viable System Model Language Model.

Architecture derived from Beer's Viable System Model (1972), shaped
to match the lambda compiler circuit observed in Qwen3-4B and
Pythia-160M.

Five systems:
  S5 (Identity)      — embeddings, register_init, output norm
  S4 (Intelligence)  — register cross-attends to full residual
  S3 (Control)       — per-phase gating conditioned on register
  S2 (Coordination)  — shared residual stream (structural)
  S1 (Operations)    — type(s=1), parse(s=8), apply(s=64)

Key design choices:
  - S4 runs once (not per-iteration) — scans raw input
  - S3 gates are per-dimension (not per-position)
  - S1 order is fine→coarse (type → parse → apply)
  - No prediction heads — gating replaces predict-and-subtract
  - Register persists across iterations

See: mementum/knowledge/explore/vsm-lm-architecture.md

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
# S4 — Intelligence
# ══════════════════════════════════════════════════════════════════════


class S4Intelligence(nn.Module):
    """Register cross-attends to the full residual.

    The register is a single vector (d_model,) that queries all L
    positions in the residual. It absorbs a weighted summary —
    the "intelligence" about the current input.

    This is what L24:H0 does in Qwen3-4B: reads position 0 (BOS)
    with 60-84% attention, where BOS has accumulated global state
    from all layers. Here, the register IS the BOS register, and
    this cross-attention IS L24:H0's behavior.

    Cost: O(L × d) per forward — one query, L keys. Cheap.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.scale = d_model ** -0.5

        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, register: torch.Tensor, residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scan the full residual, update register.

        Args:
            register: (d_model,) — current register state
            residual: (B, L, d_model) — full residual stream

        Returns:
            updated_register: (d_model,) — register + intelligence
            attn_weights: (B, L) — attention distribution (for instrumentation)
        """
        B, L, D = residual.shape

        x = self.norm(residual)                             # (B, L, D)
        q = self.q_proj(register)                           # (D,)
        k = self.k_proj(x)                                  # (B, L, D)
        v = self.v_proj(x)                                  # (B, L, D)

        # Single query attending to L keys: (B, L)
        attn = torch.einsum("d,bld->bl", q, k) * self.scale
        attn_weights = F.softmax(attn, dim=-1)              # (B, L)
        attn_weights = self.dropout(attn_weights)

        # Weighted summary: (B, D)
        summary = torch.einsum("bl,bld->bd", attn_weights, v)

        # Update register with mean across batch
        updated = register + summary.mean(dim=0)            # (D,)

        return updated, attn_weights.detach()


# ══════════════════════════════════════════════════════════════════════
# S3 — Control
# ══════════════════════════════════════════════════════════════════════


class S3Control(nn.Module):
    """Per-phase gating conditioned on register state.

    For each S1 unit, S3:
      1. Reads the register (global state) and the phase delta (what S1 proposes)
      2. Computes a per-dimension gate (sigmoid)
      3. Gates the delta: residual += gate * delta
      4. Updates the register with what happened

    The gate replaces predict-and-subtract. Instead of subtracting a
    prediction (which can starve information), S3 scales the contribution.
    Gate≈1: "this is novel, let it through."
    Gate≈0: "this is redundant, suppress it."
    """

    def __init__(self, d_model: int, n_phases: int = 3):
        super().__init__()
        self.d_model = d_model
        self.n_phases = n_phases

        # Per-phase gate heads: [register; delta_summary] → gate
        self.gate_heads = nn.ModuleList([
            nn.Linear(2 * d_model, d_model)
            for _ in range(n_phases)
        ])

        # Shared register write mechanism
        self.write_proj = nn.Linear(d_model, d_model, bias=False)
        self.write_gate = nn.Linear(d_model, 1)

    def gate_phase(
        self,
        register: torch.Tensor,
        delta: torch.Tensor,
        phase_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gate one S1 unit's contribution and update register.

        Args:
            register: (d_model,) — current register state
            delta: (B, L, d_model) — what S1 proposes to add
            phase_idx: which phase (0=type, 1=parse, 2=apply)

        Returns:
            gated_delta: (B, L, d_model) — gated contribution
            updated_register: (d_model,) — register after absorbing
            gate_values: (d_model,) — the gate vector (for instrumentation)
        """
        # Pool delta across batch and sequence → (d_model,)
        summary = delta.mean(dim=(0, 1))

        # Gate: register + summary → per-dimension sigmoid
        gate_input = torch.cat([register, summary])         # (2 * d_model,)
        gate = torch.sigmoid(self.gate_heads[phase_idx](gate_input))  # (d_model,)

        # Apply gate (broadcast across B, L)
        gated_delta = gate.unsqueeze(0).unsqueeze(0) * delta  # (B, L, d_model)

        # Update register
        wg = torch.sigmoid(self.write_gate(summary))        # scalar
        update = self.write_proj(summary)                    # (d_model,)
        updated_register = register + wg * update

        return gated_delta, updated_register, gate.detach()


# ══════════════════════════════════════════════════════════════════════
# VSM-LM
# ══════════════════════════════════════════════════════════════════════


class VSMLM(nn.Module):
    """Viable System Model Language Model.

    S5 seeds identity → S4 scans environment → S1 operates with
    S3 control → S2 carries coordination → iterate.
    """

    def __init__(
        self,
        vocab_size: int = 50277,
        d_model: int = 256,
        max_len: int = 4096,
        n_heads: int = 8,
        d_ff: int = 768,
        window: int = 8,
        strides: tuple[int, ...] = (1, 8, 64),
        n_iterations: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.n_iterations = n_iterations
        self.window = window
        self.strides = strides

        # ── S5: Identity ──────────────────────────────────────────
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.register_init = nn.Parameter(torch.zeros(d_model))
        self.output_norm = nn.LayerNorm(d_model)

        # ── S4: Intelligence ──────────────────────────────────────
        self.s4 = S4Intelligence(d_model, dropout)

        # ── S3: Control ───────────────────────────────────────────
        self.s3 = S3Control(d_model, n_phases=len(strides))

        # ── S1: Operations ────────────────────────────────────────
        # Fine → coarse: type(s=1) → parse(s=8) → apply(s=64)
        self.s1_layers = nn.ModuleList([
            CompressorLayer(
                d_model,
                [(stride, window)] * n_heads,
                d_ff,
                dropout,
            )
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

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L = input_ids.shape
        device = input_ids.device

        # ── S5: Identity → initial representations ────────────────
        positions = torch.arange(L, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        register = self.register_init.clone()

        # ── S4: Intelligence scan (once) ──────────────────────────
        register, _ = self.s4(register, x)

        # ── Iteration loop ────────────────────────────────────────
        for _iteration in range(self.n_iterations):
            # S1 operations with S3 control (fine → coarse)
            for phase_idx, s1_layer in enumerate(self.s1_layers):
                delta = s1_layer(x) - x
                gated_delta, register, _ = self.s3.gate_phase(
                    register, delta, phase_idx,
                )
                x = x + gated_delta                         # S2: residual

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
        """Forward pass with full instrumentation for checkpoints.

        Returns (logits, loss, metrics) where metrics contains:
          - register_norms: after S4 and after each phase/iteration
          - gate_values: per-phase per-iteration gate vectors
          - s4_attn_entropy: entropy of S4 attention distribution
          - phase_delta_norms: per-phase per-iteration
          - activation_norms: at each phase boundary
        """
        B, L = input_ids.shape
        device = input_ids.device
        metrics: dict = {}

        # ── S5: Identity ──────────────────────────────────────────
        positions = torch.arange(L, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        register = self.register_init.clone()

        metrics["embed_norm"] = x.norm(dim=-1).mean().item()
        metrics["register_init_norm"] = register.norm().item()

        # ── S4: Intelligence scan ─────────────────────────────────
        register, s4_attn = self.s4(register, x)
        metrics["register_after_s4"] = register.norm().item()

        # S4 attention entropy: -sum(p * log(p))
        s4_entropy = -(s4_attn * (s4_attn + 1e-10).log()).sum(dim=-1).mean()
        metrics["s4_attn_entropy"] = s4_entropy.item()

        # ── Iteration loop ────────────────────────────────────────
        for it in range(self.n_iterations):
            pfx = f"iter{it}"

            for phase_idx, (s1_layer, name) in enumerate(
                zip(self.s1_layers, self.phase_names)
            ):
                delta = s1_layer(x) - x
                gated_delta, register, gate_vals = self.s3.gate_phase(
                    register, delta, phase_idx,
                )
                x = x + gated_delta

                # Instrumentation
                metrics[f"{pfx}_{name}_delta_norm"] = (
                    delta.norm(dim=-1).mean().item()
                )
                metrics[f"{pfx}_{name}_gated_norm"] = (
                    gated_delta.norm(dim=-1).mean().item()
                )
                metrics[f"{pfx}_{name}_gate_mean"] = gate_vals.mean().item()
                metrics[f"{pfx}_{name}_gate_std"] = gate_vals.std().item()
                metrics[f"{pfx}_{name}_gate_min"] = gate_vals.min().item()
                metrics[f"{pfx}_{name}_gate_max"] = gate_vals.max().item()
                metrics[f"{pfx}_after_{name}"] = (
                    x.norm(dim=-1).mean().item()
                )

            metrics[f"{pfx}_register_norm"] = register.norm().item()

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
        s5_other = self.register_init.numel() + sum(
            p.numel() for p in self.output_norm.parameters()
        )
        s4 = sum(p.numel() for p in self.s4.parameters())
        s3 = sum(p.numel() for p in self.s3.parameters())
        s1 = sum(p.numel() for p in self.s1_layers.parameters())
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
        lines = [
            "VSM-LM — Viable System Model Language Model",
            f"  d_model={self.d_model}, seq_len={self.max_len}, "
            f"iterations={self.n_iterations}",
            f"  S1: {' → '.join(f'{n}(s={s})' for n, s in zip(self.phase_names, self.strides))}",
            f"  S4: register cross-attention (once)",
            f"  S3: per-phase gating ({len(self.strides)} gates)",
            f"  Window: {self.window}",
        ]
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
