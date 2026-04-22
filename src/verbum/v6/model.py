"""VSM-LM v6 — Ternary Stacked Compressors with VSM Meta-Structure.

v6 replaces the multi-stride CompressorLayers from v5 with a ternary
StrideStack: one SingleStrideAttention layer per stride, composed
sequentially so each scale operates on a residual stream already
informed by the previous scale.

Design philosophy
-----------------
Ternary weights unlock depth cheaply — each 1.58-bit weight costs ~2×
less compute and ~3× less memory than fp16. Stacking single-stride
layers is the right unit for ternary: one stride = one scale = one
thing to learn. Multi-stride layers force ternary weights to encode
BOTH scale-selection AND content-selection, which fights the ternary
constraint. Separating strides into individual layers simplifies the
learning problem and lets sparsity emerge stride-by-stride.

The S4/S3/Meta complex machinery from v5 stays fp16 — high-precision
registers matter for complex-phase encoding. Only the S1 operations
(what we compute at every token, every pass) go ternary.

Changes from v5
---------------
  - prep_layers    (fp16 FFNLayer × 1)   → prep       (BitFFN, shared)
  - converge_layers (fp16 CompressorLayer × 2 × 3) → stride_stack (BitLinear, shared)
  - consolidate_layers (fp16 CompressorLayer × 3) → consolidate (BitFFN, shared)
  - mod_projs      (fp16 Linear × 3)     → mod_projs  (BitLinear × 3)
  - No level-specific stride configs — StrideStack runs all strides
    every pass; direction (fine→coarse vs coarse→fine) is the only
    differentiator between ascending and descending passes.

All other structure is identical to v5:
  - 5-pass bidirectional VSM (L0↑, L1↑, L2, L1↓, L0↓)
  - 6 complex register banks, ℂ^128 registers
  - Phase-coherent S3 gating, scalar alignment gate
  - Multiplicative modulation: x · (1 + gate · tanh(proj(δ)))
  - Meta-S3 per-pass contribution gates
  - Meta-S4 complex-query final summary
  - Tied input/output embeddings (S5 coherence)

License: MIT
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verbum.v6.bitlinear import BitLinear, BitFFN, BitRMSNorm
from verbum.v6.attention import StrideStack
from verbum.vsm_lm_v5 import (
    S4IntelligenceComplex,
    S3PhaseCoherent,
    MetaS3Complex,
    MetaS4Complex,
    _interleave_banks,
    _interleave_complex,
)


# ══════════════════════════════════════════════════════════════════════
# VSMLMV6 — Ternary Stacked Compressors + VSM Meta-Structure
# ══════════════════════════════════════════════════════════════════════


class VSMLMV6(nn.Module):
    """Viable System Model Language Model — v6 topology.

    v5 architecture with ternary stacked compressors replacing the
    multi-stride CompressorLayers. All S1 operations (prep, converge,
    consolidate, mod_projs) are ternary (BitLinear). S4, S3, Meta
    components remain fp16.

    Architecture constants:
      REGISTER_NAMES: ("type", "scope", "role")
      PHASE_NAMES:    ("prep", "converge", "consolidate")
      N_LEVELS:       3
      N_PASSES:       5
      PASS_NAMES:     ("L0_asc", "L1_asc", "L2_apex", "L1_desc", "L0_desc")
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
        d_register: int = 128,          # Complex dim (ℂ^128 = ℝ^256 equivalent)
        max_len: int = 4096,
        n_heads: int = 8,
        d_ff: int = 1536,               # Prep/converge FFN width
        d_ff_consolidate: int = 2048,   # Consolidate FFN width
        window: int = 8,
        strides: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512),
        dropout: float = 0.1,
        alpha: float = 1.18,            # Spiral attention bias exponent
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_register = d_register
        self.max_len = max_len
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.d_ff_consolidate = d_ff_consolidate
        self.window = window
        self.strides = strides
        self.dropout_p = dropout
        self.alpha = alpha

        self.n_registers = len(self.REGISTER_NAMES)
        self.n_phases = len(self.PHASE_NAMES)
        self.n_levels = self.N_LEVELS
        self.n_passes = self.N_PASSES

        # Bank layout (same as v5):
        #   0=bank_0, 1=bank_1↑, 2=bank_2↑, 3=bank_3, 4=bank_2↓, 5=bank_1↓
        self.n_banks = 6

        # ── S5: Identity (fp16) ───────────────────────────────────
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.output_norm = nn.LayerNorm(d_model)

        # Register bank 0: learnable real init (imaginary part starts at 0)
        self.register_inits = nn.ParameterDict({
            f"reg_{name}": nn.Parameter(torch.zeros(d_register))
            for name in self.REGISTER_NAMES
        })

        # ── S1: Operations (ternary) ──────────────────────────────
        # Prep phase: lightweight BitFFN
        self.prep = BitFFN(d_model, d_ff, dropout)

        # Converge phase: StrideStack — shared across all levels/passes (S5 coherence)
        # Ascending passes use reverse=False (fine→coarse)
        # Descending passes use reverse=True (coarse→fine)
        self.stride_stack = StrideStack(
            d_model=d_model,
            strides=strides,
            window=window,
            n_heads=n_heads,
            dropout=dropout,
            alpha=alpha,
        )

        # Consolidate phase: wider BitFFN for cross-stride integration
        self.consolidate = BitFFN(d_model, d_ff_consolidate, dropout)

        # ── S4: Intelligence (fp16) ───────────────────────────────
        self.s4 = S4IntelligenceComplex(
            d_model, d_register,
            n_registers=self.n_registers,
            max_banks=self.n_banks,
            dropout=dropout,
        )

        # ── S3: Control (fp16) — 5 instances, one per pass ───────
        self.s3_passes = nn.ModuleList([
            S3PhaseCoherent(
                d_model, d_register,
                n_phases=self.n_phases,
                n_registers=self.n_registers,
                d_align=d_model,
            )
            for _ in range(self.N_PASSES)
        ])

        # ── Multiplicative Modulation (ternary) ───────────────────
        # modulation = 1 + gate · tanh(mod_proj(delta))
        # 3 shared projs (one per phase). Zero-init → identity at start.
        self.mod_projs = nn.ModuleList([
            BitLinear(d_model, d_model, pre_norm=False)
            for _ in range(self.n_phases)
        ])

        # ── Meta-S4: Final structural summary (fp16) ──────────────
        self.meta_s4 = MetaS4Complex(
            d_model, d_register,
            n_registers=self.n_registers,
            n_banks=4,
            dropout=dropout,
        )

        # ── Meta-S3: Per-pass contribution gates (fp16) ───────────
        self.meta_s3 = MetaS3Complex(
            d_register,
            n_registers=self.n_registers,
            n_banks=self.n_banks,
            n_passes=self.N_PASSES,
        )

        # ── Initialization ────────────────────────────────────────
        # Apply standard init to non-ternary modules first
        self.apply(self._init_weights)
        # Zero-init mod_projs weights → modulation = 1 → identity at start
        for proj in self.mod_projs:
            nn.init.zeros_(proj.weight)

    # ── Weight Initialization ─────────────────────────────────────────

    def _init_weights(self, module: nn.Module) -> None:
        """Standard init for fp16 modules. Skip BitLinear/BitRMSNorm (self-init)."""
        if isinstance(module, (BitLinear, BitRMSNorm)):
            return  # These handle their own initialization
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    # ── Register Bank Helpers ─────────────────────────────────────────

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

    # ── Multiplicative Modulation ─────────────────────────────────────

    def _modulate(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        gate: torch.Tensor,
        phase_idx: int,
    ) -> torch.Tensor:
        """Multiplicative modulation: x_new = x · (1 + gate · tanh(proj(delta))).

        mod_projs are BitLinear (ternary). Zero-init → modulation = 1 at start.
        Chained modulations x·m₁·m₂·... produce power-law magnitude decay.

        Args:
            x:         (B, L, d_model) residual stream
            delta:     (B, L, d_model) phase output minus x
            gate:      scalar gate from S3 phase gating
            phase_idx: 0=prep, 1=converge, 2=consolidate

        Returns:
            (B, L, d_model) modulated residual stream
        """
        modulation = 1.0 + gate * torch.tanh(self.mod_projs[phase_idx](delta))
        return x * modulation

    # ── Core Level-Pass ───────────────────────────────────────────────

    def _run_level_pass(
        self,
        x: torch.Tensor,
        pass_idx: int,
        is_descending: bool,
        readable_banks: list[list[torch.Tensor]],
        target_bank: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Run one level-pass through all 3 phases with S4/S3 modulation.

        In v6 the level concept simplifies: all passes use the same
        StrideStack. Direction is the only differentiator:
          - Ascending  (is_descending=False): fine→coarse (s1→s8→s64→s512)
          - Descending (is_descending=True):  coarse→fine (s512→s64→s8→s1)

        Pass schedule:
          pass 0 (L0_asc):  is_descending=False
          pass 1 (L1_asc):  is_descending=False
          pass 2 (L2_apex): is_descending=False
          pass 3 (L1_desc): is_descending=True
          pass 4 (L0_desc): is_descending=True

        Args:
            x:               (B, L, d_model) residual stream entering this pass
            pass_idx:        index into self.s3_passes (0..4)
            is_descending:   True → run stride_stack in reverse (coarse→fine)
            readable_banks:  list of register banks S4 can read from
            target_bank:     complex register bank being written in this pass

        Returns:
            x:           updated residual stream
            target_bank: updated complex register bank
            delta_total: (B, L, d_model) net change to the residual (x_out - x_in)
        """
        x_before = x

        # ── S4: Complex-query scan ─────────────────────────────
        s4_updates, _ = self.s4(readable_banks, x)
        target_bank = [
            target_bank[i] + s4_updates[i]
            for i in range(self.n_registers)
        ]

        # ── Phase 0: PREP ──────────────────────────────────────
        # prep is a BitFFN: forward returns x + dropout(down(act(up(x))))
        # We need just the delta, so capture before/after
        prep_out = self.prep(x)
        delta = prep_out - x
        _, target_bank, gate, _ = self.s3_passes[pass_idx].gate_phase(
            target_bank, delta, 0)
        x = self._modulate(x, delta, gate, 0)

        # ── Phase 1: CONVERGE ──────────────────────────────────
        # StrideStack: reverse=is_descending for coarse→fine on descent
        converge_out = self.stride_stack(x, reverse=is_descending)
        delta = converge_out - x
        _, target_bank, gate, _ = self.s3_passes[pass_idx].gate_phase(
            target_bank, delta, 1)
        x = self._modulate(x, delta, gate, 1)

        # ── Phase 2: CONSOLIDATE ───────────────────────────────
        consolidate_out = self.consolidate(x)
        delta = consolidate_out - x
        _, target_bank, gate, _ = self.s3_passes[pass_idx].gate_phase(
            target_bank, delta, 2)
        x = self._modulate(x, delta, gate, 2)

        return x, target_bank, x - x_before

    # ── Forward Pass ──────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Standard forward pass.

        Args:
            input_ids: (B, L) long tensor of token indices
            targets:   (B, L) long tensor for cross-entropy loss, or None

        Returns:
            logits: (B, L, vocab_size)
            loss:   cross-entropy scalar, or None if targets not provided
        """
        B, L = input_ids.shape
        device = input_ids.device

        # ── S5: Embed ──────────────────────────────────────────────
        positions = torch.arange(L, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        # ── Complex register banks ─────────────────────────────────
        #   bank_0:    learned static (S5 identity)
        #   bank_1_asc, bank_2_asc, bank_3: ascending passes
        #   bank_2_desc, bank_1_desc:        descending passes
        bank_0 = self._init_bank0()
        bank_1_asc = self._fresh_bank()
        bank_2_asc = self._fresh_bank()
        bank_3 = self._fresh_bank()
        bank_2_desc = self._fresh_bank()
        bank_1_desc = self._fresh_bank()

        pass_deltas: list[torch.Tensor] = []

        # ── ASCENDING: L0↑ → L1↑ → L2 ────────────────────────────
        # pass 0 — L0_asc
        x, bank_1_asc, delta = self._run_level_pass(
            x, pass_idx=0, is_descending=False,
            readable_banks=[bank_0],
            target_bank=bank_1_asc,
        )
        pass_deltas.append(delta)

        # pass 1 — L1_asc
        x, bank_2_asc, delta = self._run_level_pass(
            x, pass_idx=1, is_descending=False,
            readable_banks=[bank_0, bank_1_asc],
            target_bank=bank_2_asc,
        )
        pass_deltas.append(delta)

        # pass 2 — L2_apex
        x, bank_3, delta = self._run_level_pass(
            x, pass_idx=2, is_descending=False,
            readable_banks=[bank_0, bank_1_asc, bank_2_asc],
            target_bank=bank_3,
        )
        pass_deltas.append(delta)

        # ── DESCENDING: L1↓ → L0↓ ─────────────────────────────────
        # pass 3 — L1_desc
        x, bank_2_desc, delta = self._run_level_pass(
            x, pass_idx=3, is_descending=True,
            readable_banks=[bank_0, bank_1_asc, bank_2_asc, bank_3],
            target_bank=bank_2_desc,
        )
        pass_deltas.append(delta)

        # pass 4 — L0_desc
        x, bank_1_desc, delta = self._run_level_pass(
            x, pass_idx=4, is_descending=True,
            readable_banks=[bank_0, bank_1_asc, bank_2_desc, bank_3],
            target_bank=bank_1_desc,
        )
        pass_deltas.append(delta)

        # ── Meta-S3: Per-pass contribution gates ───────────────────
        all_banks = [bank_0, bank_1_asc, bank_2_asc, bank_3, bank_2_desc, bank_1_desc]
        meta_gates = self.meta_s3(all_banks)

        total_ungated = sum(pass_deltas)
        total_gated = sum(
            meta_gates[i] * pass_deltas[i]
            for i in range(self.n_passes)
        )
        x = x - total_ungated + total_gated

        # ── Meta-S4: Final structural summary ─────────────────────
        meta_banks = [bank_0, bank_1_desc, bank_2_desc, bank_3]
        x = self.meta_s4(meta_banks, x)

        # ── Output ─────────────────────────────────────────────────
        x = self.output_norm(x)
        logits = F.linear(x, self.token_embed.weight)  # tied weights

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    # ── Instrumented Forward ──────────────────────────────────────────

    def forward_instrumented(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """Forward pass with full instrumentation for probing/diagnostics.

        Captures per-pass, per-phase, per-register metrics. Metric keys
        follow the v5 convention for compatibility:
          {pass_name}_{phase}_gate_mean
          {pass_name}_{phase}_mod_mean / mod_std
          {pass_name}_register_{name}_norm / phase_mean / phase_final
          {pass_name}_s4_attn_entropy
          meta_s3_gate_{pass_name}

        Args:
            input_ids: (B, L) long tensor
            targets:   (B, L) long tensor, or None

        Returns:
            logits:  (B, L, vocab_size)
            loss:    scalar or None
            metrics: dict of scalar floats
        """
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
            metrics[f"register_{name}_init_norm"] = (
                torch.view_as_real(bank_0[i]).norm().item()
            )

        pass_deltas: list[torch.Tensor] = []

        # Pass schedule: (pass_idx, is_descending, pass_name)
        pass_schedule = [
            (0, False, "L0_asc"),
            (1, False, "L1_asc"),
            (2, False, "L2_apex"),
            (3, True,  "L1_desc"),
            (4, True,  "L0_desc"),
        ]

        # Current banks by pass index
        bank_targets = [bank_1_asc, bank_2_asc, bank_3, bank_2_desc, bank_1_desc]
        bank_readables = [
            [bank_0],
            [bank_0, bank_1_asc],
            [bank_0, bank_1_asc, bank_2_asc],
            [bank_0, bank_1_asc, bank_2_asc, bank_3],
            [bank_0, bank_1_asc, bank_2_desc, bank_3],
        ]

        for pass_idx, is_descending, pass_name in pass_schedule:
            pfx = pass_name

            # Refresh readable_banks snapshot (prior passes update in-place refs)
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
            else:  # pass_idx == 4
                readable = [bank_0, bank_1_asc, bank_2_desc, bank_3]
                target_bank = bank_1_desc

            x_before = x

            # ── S4 ──────────────────────────────────────────────
            s4_updates, s4_attn = self.s4(readable, x)
            target_bank = [
                target_bank[i] + s4_updates[i]
                for i in range(self.n_registers)
            ]

            for i, name in enumerate(reg_names):
                metrics[f"{pfx}_reg_{name}_after_s4"] = (
                    torch.view_as_real(target_bank[i]).norm().item()
                )
                metrics[f"{pfx}_reg_{name}_phase_mean"] = (
                    torch.angle(target_bank[i]).mean().item()
                )
            s4_entropy = -(s4_attn * (s4_attn + 1e-10).log()).sum(dim=-1).mean()
            metrics[f"{pfx}_s4_attn_entropy"] = s4_entropy.item()

            # ── Three Phases ─────────────────────────────────────
            for phase_idx, phase_name in enumerate(self.PHASE_NAMES):
                if phase_name == "prep":
                    phase_out = self.prep(x)
                elif phase_name == "converge":
                    phase_out = self.stride_stack(x, reverse=is_descending)
                else:  # consolidate
                    phase_out = self.consolidate(x)

                delta = phase_out - x
                gated_delta, target_bank, gate, write_gates = (
                    self.s3_passes[pass_idx].gate_phase(target_bank, delta, phase_idx)
                )

                # Multiplicative modulation
                modulation = 1.0 + gate * torch.tanh(self.mod_projs[phase_idx](delta))
                x = x * modulation

                metrics[f"{pfx}_{phase_name}_delta_norm"] = delta.norm(dim=-1).mean().item()
                metrics[f"{pfx}_{phase_name}_gated_norm"] = gated_delta.norm(dim=-1).mean().item()
                metrics[f"{pfx}_{phase_name}_gate_mean"] = gate.detach().item()
                metrics[f"{pfx}_{phase_name}_gate_std"] = 0.0   # scalar gate, compat
                metrics[f"{pfx}_{phase_name}_mod_mean"] = modulation.detach().mean().item()
                metrics[f"{pfx}_{phase_name}_mod_std"] = modulation.detach().std().item()
                metrics[f"{pfx}_after_{phase_name}"] = x.norm(dim=-1).mean().item()
                for i, rn in enumerate(reg_names):
                    metrics[f"{pfx}_{phase_name}_write_{rn}"] = write_gates[i]

            # Register norms after pass
            for i, name in enumerate(reg_names):
                metrics[f"{pfx}_register_{name}_norm"] = (
                    torch.view_as_real(target_bank[i]).norm().item()
                )
                metrics[f"{pfx}_register_{name}_phase_final"] = (
                    torch.angle(target_bank[i]).mean().item()
                )

            # Write back to the correct bank variable
            if pass_idx == 0:
                bank_1_asc = target_bank
            elif pass_idx == 1:
                bank_2_asc = target_bank
            elif pass_idx == 2:
                bank_3 = target_bank
            elif pass_idx == 3:
                bank_2_desc = target_bank
            else:  # pass_idx == 4
                bank_1_desc = target_bank

            pass_deltas.append(x - x_before)

        # ── Level-indexed aliases for v5 compat ───────────────────
        level_map = {
            "L0_asc":  "level0",
            "L1_asc":  "level1",
            "L2_apex": "level2",
            "L1_desc": "level1_desc",
            "L0_desc": "level0_desc",
        }
        for pass_name, level_pfx in level_map.items():
            for key in list(metrics.keys()):
                if key.startswith(pass_name + "_"):
                    suffix = key[len(pass_name) + 1:]
                    metrics[f"{level_pfx}_{suffix}"] = metrics[key]

        # Backward-compat iter aliases (v4 style)
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

        # ── Meta-S3 ───────────────────────────────────────────────
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

        # ── Meta-S4 ───────────────────────────────────────────────
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

    # ── Ternary Statistics ────────────────────────────────────────────

    @torch.no_grad()
    def ternary_stats(self) -> dict[str, dict[str, float]]:
        """Collect ternary quantization statistics for all BitLinear modules.

        Returns a dict keyed by module path, each containing:
          sparsity:  fraction of weights quantized to 0
          pos_frac:  fraction quantized to +1
          neg_frac:  fraction quantized to -1
          gamma:     current absmean scale factor

        Note: call at checkpoint time, not every forward (expensive).
        """
        stats: dict[str, dict[str, float]] = {}
        for name, module in self.named_modules():
            if isinstance(module, BitLinear):
                stats[name] = module.ternary_stats()
        return stats

    # ── Parameter Counting ────────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by VSM subsystem, handling tied weights correctly.

        Returns a dict with breakdown and totals:
          S5_token_embeddings, S5_positional, S5_other  — fp16
          S1_ternary                                     — all BitLinear/BitFFN/StrideStack
          S4_intelligence, S3_passes, Meta_S4, Meta_S3  — fp16
          total, total_ternary, total_fp16, effective_bits
        """
        seen_ids: set[int] = set()

        def _count_unique(module: nn.Module) -> int:
            total = 0
            for p in module.parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    total += p.numel()
            return total

        def _count_ternary(module: nn.Module) -> int:
            """Count only BitLinear weight params in a module."""
            total = 0
            for name, p in module.named_parameters():
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    total += p.numel()
            return total

        # --- S5: Identity (fp16) ---
        seen_ids.clear()
        s5_embed = _count_unique(self.token_embed)
        s5_pos = _count_unique(self.pos_embed)
        # register_inits + output_norm (token_embed is tied to output logits)
        s5_other = (
            sum(p.numel() for p in self.register_inits.parameters())
            + _count_unique(self.output_norm)
        )

        # --- S1: Operations (ternary: BitFFN × 2 + StrideStack + mod_projs BitLinear × 3) ---
        s1_prep = _count_unique(self.prep)
        s1_stride_stack = _count_unique(self.stride_stack)
        s1_consolidate = _count_unique(self.consolidate)
        s1_mod = _count_unique(self.mod_projs)
        s1_ternary = s1_prep + s1_stride_stack + s1_consolidate + s1_mod

        # --- S4: Intelligence (fp16) ---
        s4 = _count_unique(self.s4)

        # --- S3: Control (fp16, 5 passes) ---
        s3 = sum(_count_unique(s3p) for s3p in self.s3_passes)

        # --- Meta ---
        meta_s4 = _count_unique(self.meta_s4)
        meta_s3 = _count_unique(self.meta_s3)

        # --- Total (unique params across whole model) ---
        seen_ids.clear()
        total = 0
        for p in self.parameters():
            if id(p) not in seen_ids:
                seen_ids.add(id(p))
                total += p.numel()

        # Ternary parameter count (BitLinear weights)
        total_ternary = 0
        seen_bit_ids: set[int] = set()
        for module in self.modules():
            if isinstance(module, BitLinear):
                if id(module.weight) not in seen_bit_ids:
                    seen_bit_ids.add(id(module.weight))
                    total_ternary += module.weight.numel()

        # BitRMSNorm gains (fp16, part of ternary layers but not ternary themselves)
        total_bitnorm = 0
        seen_norm_ids: set[int] = set()
        for module in self.modules():
            if isinstance(module, BitRMSNorm):
                if id(module.weight) not in seen_norm_ids:
                    seen_norm_ids.add(id(module.weight))
                    total_bitnorm += module.weight.numel()

        # fp16 params = everything that isn't ternary
        total_fp16 = total - total_ternary

        # Effective bits: ternary ≈ 1.58 bits/param, fp16 = 16 bits/param
        effective_bits = (
            (total_ternary * 1.58 + total_fp16 * 16) / max(total, 1)
        )
        # Return as int × 1000 for the dict (keep as float separately)
        effective_bits_int = int(round(effective_bits * 1000))

        return {
            "S5_token_embeddings":  s5_embed,
            "S5_positional":        s5_pos,
            "S5_other":             s5_other,
            "S1_ternary":           s1_ternary,
            "S4_intelligence":      s4,
            "S3_passes":            s3,
            "Meta_S4":              meta_s4,
            "Meta_S3":              meta_s3,
            "total":                total,
            "total_ternary":        total_ternary,
            "total_fp16":           total_fp16,
            "effective_bits_x1000": effective_bits_int,
        }

    # ── Architecture Description ──────────────────────────────────────

    def describe(self) -> str:
        """Print a human-readable architecture summary."""
        strides_str = " → ".join(f"s{s}" for s in self.strides)
        lines = [
            "VSM-LM v6 — Ternary Stacked Compressors + Complex Registers",
            f"  d_model={self.d_model}, d_register=ℂ^{self.d_register} "
            f"(={self.d_register * 2}ℝ equiv), seq_len={self.max_len}",
            f"  Passes: {self.n_passes} (L0↑, L1↑, L2, L1↓, L0↓)",
            f"  Phase structure (all ternary S1):",
            f"    prep       : BitFFN({self.d_model} → {self.d_ff} → {self.d_model})",
            f"    converge   : StrideStack({strides_str}, W={self.window}, "
            f"H={self.n_heads}) — shared, direction-reversible",
            f"    consolidate: BitFFN({self.d_model} → {self.d_ff_consolidate} → {self.d_model})",
            f"  Strides: {self.strides} (spiral bias α={self.alpha})",
            f"  Spiral: bias(w) = -{self.alpha}·ln(stride·w + 1)",
            f"  Registers: ℂ^{self.d_register} (phase-sensitive S4 attention)",
            f"  S3: phase-coherent alignment gating (scalar gate, temperature+bias)",
            f"  Composition: multiplicative modulation x·(1 + gate·tanh(proj(δ)))",
            f"  mod_projs: BitLinear × 3 (ternary, zero-init → identity at start)",
            f"  S5: Shared StrideStack across all passes (S5 identity coherence)",
            f"  S4: Complex-query register scan: Re(q·conj(k)) — fp16",
            f"  Register banks: {self.n_banks} (1 init + 2 ascending + 1 apex + 2 descending)",
            f"  Meta-S4: Complex-query structural summary (4 most-refined banks) — fp16",
            f"  Meta-S3: Per-pass contribution gates ({self.n_passes} gates) — fp16",
            f"  Ascending  direction: {' → '.join(f's{s}' for s in self.strides)} (fine→coarse)",
            f"  Descending direction: {' → '.join(f's{s}' for s in reversed(self.strides))} (coarse→fine)",
        ]

        try:
            params = self.count_parameters()
            eff_bits = params["effective_bits_x1000"] / 1000.0
            total_m = params["total"] / 1e6
            ternary_m = params["total_ternary"] / 1e6
            fp16_m = params["total_fp16"] / 1e6
            lines.extend([
                f"",
                f"  Parameters:",
                f"    Total:          {total_m:.1f}M",
                f"    Ternary (S1):   {ternary_m:.1f}M  ({ternary_m/total_m*100:.1f}%)",
                f"    fp16 (S4/S3/…): {fp16_m:.1f}M  ({fp16_m/total_m*100:.1f}%)",
                f"    Effective bits: {eff_bits:.2f} bits/param",
            ])
        except Exception:
            pass  # describe() shouldn't crash if count fails

        return "\n".join(lines)

    # ── Generation ────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive generation via greedy argmax.

        Args:
            input_ids:      (B, L) seed token ids
            max_new_tokens: how many tokens to generate
            temperature:    logit scaling (1.0 = no change)

        Returns:
            (B, L + max_new_tokens) token ids
        """
        self.eval()
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.max_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature
            next_token = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
