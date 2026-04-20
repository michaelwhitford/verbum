"""VSM-LM v2 — Two-channel compressor.

Changes from v1 (informed by cross-model probing at steps 1K-8K):

1. Per-iteration gate heads (3 phases × 2 iterations = 6 gates)
   Data showed iter0 serves structural decomposition, iter1 serves
   semantic refinement. Shared gates forced both channels through
   the same weights, differentiated only by register state. v2 gives
   each iteration its own gate weights so the structural and semantic
   channels can specialize explicitly.

2. S4 per-iteration (scans before each iteration, not once)
   S4 at -0.19 was stable but weak across all checkpoints. It only
   saw raw embeddings, missing structural information built by iter0.
   v2 re-scans before iter1 so the semantic channel gets intelligence
   about the structurally-enriched residual.

Everything else unchanged: same S1 phases (type/parse/apply), same
CompressorLayer, same register, same residual stream.

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
# S4 — Intelligence (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════


class S4Intelligence(nn.Module):
    """Register cross-attends to the full residual.

    Runs per-iteration in v2. Cost: O(L × d) per call — cheap.
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
        B, L, D = residual.shape

        x = self.norm(residual)
        q = self.q_proj(register)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn = torch.einsum("d,bld->bl", q, k) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        summary = torch.einsum("bl,bld->bd", attn_weights, v)
        updated = register + summary.mean(dim=0)

        return updated, attn_weights.detach()


# ══════════════════════════════════════════════════════════════════════
# S3 — Control (v2: per-iteration gate heads)
# ══════════════════════════════════════════════════════════════════════


class S3ControlV2(nn.Module):
    """Per-phase, per-iteration gating conditioned on register state.

    v1: 3 gate heads (one per phase), shared across iterations.
    v2: 6 gate heads (one per phase per iteration).

    iter0 gates learn structural gating — which features each phase
    contributes to the structural skeleton.
    iter1 gates learn semantic gating — which features each phase
    contributes to meaning refinement, conditioned on the warm register.
    """

    def __init__(self, d_model: int, n_phases: int = 3, n_iterations: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_phases = n_phases
        self.n_iterations = n_iterations

        # Per-phase, per-iteration gate heads
        # Index: iteration * n_phases + phase_idx
        self.gate_heads = nn.ModuleList([
            nn.Linear(2 * d_model, d_model)
            for _ in range(n_phases * n_iterations)
        ])

        # Shared register write mechanism (shared across iterations —
        # the register update logic doesn't need to specialize)
        self.write_proj = nn.Linear(d_model, d_model, bias=False)
        self.write_gate = nn.Linear(d_model, 1)

    def gate_phase(
        self,
        register: torch.Tensor,
        delta: torch.Tensor,
        phase_idx: int,
        iteration: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gate one S1 unit's contribution and update register.

        Args:
            register: (d_model,)
            delta: (B, L, d_model)
            phase_idx: which phase (0=type, 1=parse, 2=apply)
            iteration: which iteration (0=structural, 1=semantic)

        Returns:
            gated_delta, updated_register, gate_values
        """
        summary = delta.mean(dim=(0, 1))

        # Select iteration-specific gate head
        head_idx = iteration * self.n_phases + phase_idx
        gate_input = torch.cat([register, summary])
        gate = torch.sigmoid(self.gate_heads[head_idx](gate_input))

        gated_delta = gate.unsqueeze(0).unsqueeze(0) * delta

        wg = torch.sigmoid(self.write_gate(summary))
        update = self.write_proj(summary)
        updated_register = register + wg * update

        return gated_delta, updated_register, gate.detach()


# ══════════════════════════════════════════════════════════════════════
# VSM-LM v2
# ══════════════════════════════════════════════════════════════════════


class VSMLMV2(nn.Module):
    """Viable System Model Language Model — v2 two-channel compressor.

    Changes from v1:
    - S3 gate heads are per-iteration (6 instead of 3)
    - S4 runs per-iteration (scans enriched residual before iter1)
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

        # ── S4: Intelligence (per-iteration) ──────────────────────
        self.s4 = S4Intelligence(d_model, dropout)

        # ── S3: Control (per-iteration gates) ─────────────────────
        self.s3 = S3ControlV2(d_model, n_phases=len(strides),
                              n_iterations=n_iterations)

        # ── S1: Operations ────────────────────────────────────────
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

        # ── S5: Identity ──────────────────────────────────────────
        positions = torch.arange(L, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        register = self.register_init.clone()

        # ── Iteration loop (S4 per-iteration) ─────────────────────
        for iteration in range(self.n_iterations):
            # S4: scan residual (raw for iter0, enriched for iter1+)
            register, _ = self.s4(register, x)

            # S1 operations with S3 control (per-iteration gates)
            for phase_idx, s1_layer in enumerate(self.s1_layers):
                delta = s1_layer(x) - x
                gated_delta, register, _ = self.s3.gate_phase(
                    register, delta, phase_idx, iteration,
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
        """Forward pass with full instrumentation."""
        B, L = input_ids.shape
        device = input_ids.device
        metrics: dict = {}

        # ── S5: Identity ──────────────────────────────────────────
        positions = torch.arange(L, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        register = self.register_init.clone()

        metrics["embed_norm"] = x.norm(dim=-1).mean().item()
        metrics["register_init_norm"] = register.norm().item()

        # ── Iteration loop ────────────────────────────────────────
        for it in range(self.n_iterations):
            pfx = f"iter{it}"

            # S4: per-iteration scan
            register, s4_attn = self.s4(register, x)
            metrics[f"{pfx}_register_after_s4"] = register.norm().item()

            # S4 attention entropy
            s4_entropy = -(s4_attn * (s4_attn + 1e-10).log()).sum(dim=-1).mean()
            metrics[f"{pfx}_s4_attn_entropy"] = s4_entropy.item()

            for phase_idx, (s1_layer, name) in enumerate(
                zip(self.s1_layers, self.phase_names)
            ):
                delta = s1_layer(x) - x
                gated_delta, register, gate_vals = self.s3.gate_phase(
                    register, delta, phase_idx, it,
                )
                x = x + gated_delta

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

        # Backward-compat aliases for probing pipeline
        metrics["s4_attn_entropy"] = metrics["iter0_s4_attn_entropy"]
        metrics["register_after_s4"] = metrics["iter0_register_after_s4"]

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
            "VSM-LM v2 — Two-channel compressor",
            f"  d_model={self.d_model}, seq_len={self.max_len}, "
            f"iterations={self.n_iterations}",
            f"  S1: {' → '.join(f'{n}(s={s})' for n, s in zip(self.phase_names, self.strides))}",
            f"  S4: register cross-attention (per-iteration)",
            f"  S3: per-phase per-iteration gating "
            f"({len(self.strides)} phases × {self.n_iterations} iters "
            f"= {len(self.strides) * self.n_iterations} gates)",
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
