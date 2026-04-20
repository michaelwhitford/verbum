"""CompressorLM — Strided windowed attention at three scales.

The semantic compressor deflates fine-grained token sequences into
coarse compositional meanings. The three Montague primitives (type,
parse, apply) operate at three natural scales of language:

  Type:    stride=1,  W=8  → 8 neighboring tokens (word-level)
  Parse:   stride=8,  W=8  → 8 neighboring phrases (phrase-level)
  Apply:   stride=64, W=8  → 8 neighboring clauses (clause-level)

Same window everywhere (W=8 ≈ 6 words ≈ one clause atom). Different
stride per scale. Total cube: 8×8×8 = 512. At seq=4096 = 8⁴, three
levels bottom out at 8 positions.

Two modes:
  cube:     every layer has heads at all three strides (4+4+4)
  pipeline: strides concentrate in specific layers (early→late)

Shared residual stream throughout. No pooling (no future leak).
Strictly causal — each position attends only to past positions
at its stride.

License: MIT
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# Strided Windowed Causal Attention
# ══════════════════════════════════════════════════════════════════════


class StridedCausalAttention(nn.Module):
    """Multi-head attention where each head has a stride and window.

    Each head attends to W past positions at its stride:
      stride=1:  positions [i, i-1, i-2, ..., i-W+1]
      stride=8:  positions [i, i-8, i-16, ..., i-8*(W-1)]
      stride=64: positions [i, i-64, i-128, ..., i-64*(W-1)]

    Sparse implementation: gather K,V at strided indices, compute
    small (L, W) attention per stride group. O(L×W) not O(L²).
    No L×L matrix ever materialized.

    At seq=4096 with W=8: 32K entries per head vs 16.7M dense.
    """

    def __init__(
        self,
        d_model: int,
        head_configs: list[tuple[int, int]],  # [(stride, window), ...] per head
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.head_configs = head_configs
        self.n_heads = len(head_configs)
        self.d_head = d_model // self.n_heads
        assert d_model % self.n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.d_head ** -0.5

        # Group heads by (stride, window) for batched processing
        self._stride_groups: dict[tuple[int, int], list[int]] = {}
        for i, (stride, window) in enumerate(head_configs):
            key = (stride, window)
            if key not in self._stride_groups:
                self._stride_groups[key] = []
            self._stride_groups[key].append(i)

        self._index_cache: dict[tuple[int, int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_indices(
        self, seq_len: int, stride: int, window: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute gather indices for a stride/window combination.

        Returns:
            indices: (L, W) int64 — clamped to [0, L-1]
            valid:   (L, W) bool — True where original index >= 0
        """
        cache_key = (seq_len, stride, window, str(device))
        if cache_key not in self._index_cache:
            query_pos = torch.arange(seq_len, device=device).unsqueeze(1)  # (L, 1)
            offsets = torch.arange(window, device=device).unsqueeze(0) * stride  # (1, W)
            raw = query_pos - offsets  # (L, W)
            valid = raw >= 0
            indices = raw.clamp(min=0)
            self._index_cache[cache_key] = (indices, valid)
        return self._index_cache[cache_key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Project all heads at once: (B, L, n_heads, d_head)
        Q_all = self.q_proj(x).view(B, L, self.n_heads, self.d_head)
        K_all = self.k_proj(x).view(B, L, self.n_heads, self.d_head)
        V_all = self.v_proj(x).view(B, L, self.n_heads, self.d_head)

        # Output buffer
        out = torch.zeros_like(Q_all)  # (B, L, n_heads, d_head)

        for (stride, window), head_ids in self._stride_groups.items():
            n_g = len(head_ids)
            h_idx = torch.tensor(head_ids, device=x.device, dtype=torch.long)

            # Select this stride group's heads: (B, L, n_g, d_head)
            Q = Q_all[:, :, h_idx]
            K = K_all[:, :, h_idx]
            V = V_all[:, :, h_idx]

            # Precomputed gather indices: (L, W)
            indices, valid = self._get_indices(L, stride, window, x.device)

            # Gather K,V at strided positions
            # K: (B, L, n_g*d_head) → gather along dim=1 → (B, L*W, n_g*d_head)
            GD = n_g * self.d_head
            K_flat = K.reshape(B, L, GD)
            V_flat = V.reshape(B, L, GD)

            idx = indices.reshape(1, L * window, 1).expand(B, -1, GD)
            K_gathered = K_flat.gather(1, idx).reshape(B, L, window, n_g, self.d_head)
            V_gathered = V_flat.gather(1, idx).reshape(B, L, window, n_g, self.d_head)

            # Attention: Q·K → (B, n_g, L, W)
            Q_r = Q.permute(0, 2, 1, 3)                   # (B, n_g, L, d_head)
            K_r = K_gathered.permute(0, 3, 1, 2, 4)        # (B, n_g, L, W, d_head)
            attn = torch.einsum("bgld,bglwd->bglw", Q_r, K_r) * self.scale

            # Mask invalid (pre-sequence) positions
            attn = attn.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float("-inf"))

            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            # Apply to V → (B, n_g, L, d_head)
            V_r = V_gathered.permute(0, 3, 1, 2, 4)        # (B, n_g, L, W, d_head)
            head_out = torch.einsum("bglw,bglwd->bgld", attn, V_r)
            head_out = head_out.permute(0, 2, 1, 3)         # (B, L, n_g, d_head)

            # Place into output
            out[:, :, h_idx] = head_out

        # Merge heads and project
        out = out.reshape(B, L, D)
        return self.out_proj(out)


# ══════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════


class CompressorLayer(nn.Module):
    """Pre-norm transformer layer with strided windowed attention."""

    def __init__(
        self,
        d_model: int,
        head_configs: list[tuple[int, int]],
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = StridedCausalAttention(d_model, head_configs, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.ff(self.norm2(x))
        return x


class PredictiveCodingBlock(nn.Module):
    """One iteration of the multi-scale pipeline with prediction error.

    Two directions:
      forward (fine→coarse):  type → parse → apply → context
      reverse (coarse→fine):  context → apply → parse → type

    Forward: each finer scale predicts what the next coarser scale adds.
    Reverse: each coarser scale predicts what the next finer scale adds.

    Reverse matches cortical predictive coding (Rao & Ballard 1999):
    higher areas predict lower-level activity, only errors propagate up.
    Coarse context CAN predict fine detail; fine detail CANNOT predict
    coarse context. Compression should emerge naturally in reverse mode.

    Pass 1: predictions are cold, errors ≈ full outputs.
    Pass 2+: predictions improve, errors shrink → convergence.

    Register (opt-in via use_register=True):
      A persistent state vector that prediction heads can read.
      Updated after each phase, persists across iterations.
      Gives the prediction heads memory — they can distinguish
      "building" (iteration 1, cold register) from "applying"
      (iteration 2, warm register). Inspired by BOS composition
      register found in Qwen3-4B (L24:H0 reads position 0 with
      60-84% attention; all 36 layers write to it).

      The register is sequence-wide (broadcast to all positions)
      and added to the delta before each prediction head reads it.
      Each phase's error updates the register via a learned gate.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        window: int,
        strides: tuple[int, ...],
        dropout: float = 0.1,
        reverse: bool = False,
        use_register: bool = False,
    ):
        super().__init__()
        self.n_scales = len(strides)
        self.reverse = reverse
        self.use_register = use_register

        # Scale layers — one per stride
        self.type_layer = CompressorLayer(
            d_model, [(strides[0], window)] * n_heads, d_ff, dropout
        )
        self.parse_layer = CompressorLayer(
            d_model, [(strides[1], window)] * n_heads, d_ff, dropout
        )
        self.apply_layer = CompressorLayer(
            d_model, [(strides[2], window)] * n_heads, d_ff, dropout
        )

        # Prediction heads: cheap linear projections
        self.predict_parse = nn.Linear(d_model, d_model, bias=False)
        self.predict_apply = nn.Linear(d_model, d_model, bias=False)

        # Tesseract: 4th scale (context) at stride=8³=512
        if self.n_scales >= 4:
            self.context_layer = CompressorLayer(
                d_model, [(strides[3], window)] * n_heads, d_ff, dropout
            )
            self.predict_context = nn.Linear(d_model, d_model, bias=False)
        else:
            self.context_layer = None
            self.predict_context = None

        # Reverse mode adds predict_type (coarsest predicts finest)
        if reverse:
            self.predict_type = nn.Linear(d_model, d_model, bias=False)
        else:
            self.predict_type = None

        # Register: persistent state for prediction heads
        if use_register:
            # Initial register value (learned)
            self.register_init = nn.Parameter(torch.zeros(d_model))
            # Write gate: project phase error → register update
            self.register_write = nn.Linear(d_model, d_model, bias=False)
            # Gate scalar: sigmoid controls how much to update
            self.register_gate = nn.Linear(d_model, 1, bias=True)

    def _predict_with_register(
        self,
        predict_head: nn.Linear,
        delta: torch.Tensor,
        register: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Prediction conditioned on register state.

        Without register: predict_head(delta)
        With register: predict_head(delta + register)

        The register is broadcast across all sequence positions.
        Adding it to the delta gives the prediction head access to
        global iteration state — what has been compressed so far.
        """
        if register is not None:
            return predict_head(delta + register.unsqueeze(0).unsqueeze(0))
        return predict_head(delta)

    def _update_register(
        self,
        register: Optional[torch.Tensor],
        error: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Update register with phase error (mean-pooled across sequence).

        Gate controls how much of the error to absorb. Early in training
        the gate can learn to stay near-zero (preserve init); later it
        opens as the phases produce meaningful errors.
        """
        if register is None:
            return None
        # Mean-pool error across batch and sequence → (d_model,)
        error_summary = error.mean(dim=(0, 1))
        # Gated update
        gate = torch.sigmoid(self.register_gate(error_summary))  # scalar
        update = self.register_write(error_summary)
        return register + gate * update

    def _forward_fine_to_coarse(
        self, x: torch.Tensor, register: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Original: type → parse → apply → context."""
        # Type: full output (finest scale, no prediction to subtract)
        x_type = self.type_layer(x)
        type_delta = x_type - x
        register = self._update_register(register, type_delta)

        # Parse: predicted by type, only error propagates
        parse_predicted = self._predict_with_register(
            self.predict_parse, type_delta, register,
        )
        x_parse = self.parse_layer(x_type)
        parse_delta = x_parse - x_type
        parse_error = parse_delta - parse_predicted
        register = self._update_register(register, parse_error)

        # Apply: predicted by parse error, only error propagates
        x_with_parse = x_type + parse_error
        apply_predicted = self._predict_with_register(
            self.predict_apply, parse_error, register,
        )
        x_apply = self.apply_layer(x_with_parse)
        apply_delta = x_apply - x_with_parse
        apply_error = apply_delta - apply_predicted
        register = self._update_register(register, apply_error)

        if self.context_layer is not None:
            # Context: predicted by apply error, only error propagates
            x_with_apply = x_type + parse_error + apply_error
            context_predicted = self._predict_with_register(
                self.predict_context, apply_error, register,
            )
            x_context = self.context_layer(x_with_apply)
            context_delta = x_context - x_with_apply
            context_error = context_delta - context_predicted
            register = self._update_register(register, context_error)

            return x + type_delta + parse_error + apply_error + context_error, register
        else:
            return x + type_delta + parse_error + apply_error, register

    def _forward_coarse_to_fine(
        self, x: torch.Tensor, register: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Reversed: context → apply → parse → type.

        Coarse scales run first, predict what finer scales will add.
        Only prediction error propagates — the genuinely novel detail
        that the broader view couldn't anticipate.

        Matches cortical predictive coding: high-level predictions
        flow down, only surprises flow up.
        """
        if self.context_layer is not None:
            # Context: full output (coarsest scale, document-level frame)
            x_ctx = self.context_layer(x)
            ctx_delta = x_ctx - x
            register = self._update_register(register, ctx_delta)

            # Apply: predicted by context, only error propagates
            apply_predicted = self._predict_with_register(
                self.predict_apply, ctx_delta, register,
            )
            x_apply = self.apply_layer(x_ctx)
            apply_delta = x_apply - x_ctx
            apply_error = apply_delta - apply_predicted
            register = self._update_register(register, apply_error)

            # Parse: predicted by apply error, only error propagates
            x_with_apply = x_ctx + apply_error
            parse_predicted = self._predict_with_register(
                self.predict_parse, apply_error, register,
            )
            x_parse = self.parse_layer(x_with_apply)
            parse_delta = x_parse - x_with_apply
            parse_error = parse_delta - parse_predicted
            register = self._update_register(register, parse_error)

            # Type: predicted by parse error, only error propagates
            x_with_parse = x_ctx + apply_error + parse_error
            type_predicted = self._predict_with_register(
                self.predict_type, parse_error, register,
            )
            x_type = self.type_layer(x_with_parse)
            type_delta = x_type - x_with_parse
            type_error = type_delta - type_predicted
            register = self._update_register(register, type_error)

            return x + ctx_delta + apply_error + parse_error + type_error, register
        else:
            # 3-scale: apply → parse → type
            x_apply = self.apply_layer(x)
            apply_delta = x_apply - x
            register = self._update_register(register, apply_delta)

            parse_predicted = self._predict_with_register(
                self.predict_parse, apply_delta, register,
            )
            x_parse = self.parse_layer(x_apply)
            parse_delta = x_parse - x_apply
            parse_error = parse_delta - parse_predicted
            register = self._update_register(register, parse_error)

            x_with_parse = x_apply + parse_error
            type_predicted = self._predict_with_register(
                self.predict_type, parse_error, register,
            )
            x_type = self.type_layer(x_with_parse)
            type_delta = x_type - x_with_parse
            type_error = type_delta - type_predicted
            register = self._update_register(register, type_error)

            return x + apply_delta + parse_error + type_error, register

    def _init_register(self) -> Optional[torch.Tensor]:
        """Initialize register for a new forward pass."""
        if self.use_register:
            return self.register_init.clone()
        return None

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns (output, final_register).

        Register is None when use_register=False (backward compatible).
        """
        register = self._init_register()
        if self.reverse:
            return self._forward_coarse_to_fine(x, register)
        else:
            return self._forward_fine_to_coarse(x, register)


def _make_head_configs(
    mode: str,
    n_layers: int = 6,
    n_heads: int = 8,
    window: int = 8,
    strides: tuple[int, ...] = (1, 8, 64),
) -> list[list[tuple[int, int]]]:
    """Generate per-layer head configs for cube or pipeline mode.

    With n_heads=8 and 3 strides: 3+3+2 distribution (type and parse
    get 3, apply gets 2 — apply heads see the most context per head
    so fewer heads is acceptable).

    Returns: list of n_layers lists, each containing n_heads (stride, window) tuples.
    """
    if mode == "cube":
        # 3+3+2: type×3, parse×3, apply×2 per layer
        layer_config = (
            [(strides[0], window)] * 3 +
            [(strides[1], window)] * 3 +
            [(strides[2], window)] * 2
        )
        return [layer_config for _ in range(n_layers)]

    elif mode == "pipeline":
        # Concentrate strides by layer position, same totals:
        # 6 layers × 8 heads = 48 total
        # Cube gives: 18×s1, 18×s8, 12×s64 = 48
        # Pipeline distributes the same counts across layers:
        assignments = [
            # (s1, s8, s64) heads per layer — totals: 18, 18, 12
            (6, 2, 0),   # L0: mostly type
            (6, 2, 0),   # L1: mostly type
            (3, 4, 1),   # L2: transition
            (3, 4, 1),   # L3: transition
            (0, 3, 5),   # L4: mostly apply
            (0, 3, 5),   # L5: mostly apply
        ]
        configs = []
        for n_s1, n_s8, n_s64 in assignments:
            layer = ([(strides[0], window)] * n_s1 +
                     [(strides[1], window)] * n_s8 +
                     [(strides[2], window)] * n_s64)
            configs.append(layer)
        return configs

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'cube' or 'pipeline'.")


class CompressorLM(nn.Module):
    """Strided windowed attention language model.

    Three modes:
      cube:      every layer has heads at all three strides
      pipeline:  strides concentrate by layer (type→parse→apply)
      iterative: 3-layer block with predictive coding, iterated N times

    All use W=8 windows at strides 1, 8, 64.
    Shared residual stream. Tied input/output embeddings.

    reverse=True flips iterative mode to coarse→fine predictive coding:
      context → apply → parse → type (cortical hierarchy).
    """

    def __init__(
        self,
        vocab_size: int = 50277,
        d_model: int = 256,
        max_len: int = 4096,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 768,
        window: int = 8,
        strides: tuple[int, ...] = (1, 8, 64),
        mode: str = "cube",
        n_iterations: int = 2,
        dropout: float = 0.1,
        reverse: bool = False,
        use_register: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.mode = mode
        self.window = window
        self.strides = strides
        self.n_iterations = n_iterations

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        if mode == "iterative":
            # Single predictive coding block, iterated
            self.block = PredictiveCodingBlock(
                d_model, d_ff, n_heads, window, strides, dropout,
                reverse=reverse, use_register=use_register,
            )
            self.layers = None
            self._head_configs = None
        else:
            # Standard layered model
            all_configs = _make_head_configs(mode, n_layers, n_heads, window, strides)
            self.layers = nn.ModuleList([
                CompressorLayer(d_model, all_configs[i], d_ff, dropout)
                for i in range(n_layers)
            ])
            self.block = None
            self._head_configs = all_configs

        # Output
        self.output_norm = nn.LayerNorm(d_model)

        # Initialize
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

        positions = torch.arange(L, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        if self.mode == "iterative":
            register = self.block._init_register()
            for _ in range(self.n_iterations):
                if self.block.use_register:
                    if self.block.reverse:
                        x, register = self.block._forward_coarse_to_fine(x, register)
                    else:
                        x, register = self.block._forward_fine_to_coarse(x, register)
                else:
                    x, _ = self.block(x)
        else:
            for layer in self.layers:
                x = layer(x)

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

    def count_parameters(self) -> dict[str, int]:
        embed = sum(p.numel() for p in self.token_embed.parameters())
        pos = sum(p.numel() for p in self.pos_embed.parameters())
        if self.mode == "iterative":
            layer_p = sum(p.numel() for p in self.block.parameters())
        else:
            layer_p = sum(p.numel() for p in self.layers.parameters())
        head_p = sum(p.numel() for p in self.output_norm.parameters())
        total = embed + pos + layer_p + head_p
        return {
            "token_embeddings": embed,
            "positional_embeddings": pos,
            "layers": layer_p,
            "output_head": head_p,
            "total": total,
        }

    def describe_heads(self) -> str:
        """Human-readable head assignment summary."""
        lines = [f"Mode: {self.mode}, W={self.window}, strides={self.strides}"]
        if self.mode == "iterative":
            n_scales = len(self.strides)
            shape = "tesseract (8⁴)" if n_scales >= 4 else "cube (8³)"
            lines.append(f"  Shape: {shape}, iterations: {self.n_iterations}")
            scale_names = ["type", "parse", "apply", "context"]
            chain = " → ".join(
                f"{scale_names[i]}(s{self.strides[i]})"
                for i in range(n_scales)
            )
            lines.append(f"  Block: {chain}")
            pc_chain = " → ".join(
                f"{scale_names[i]}→{scale_names[i+1]}"
                for i in range(n_scales - 1)
            )
            lines.append(f"  Predictive coding: {pc_chain}")
        else:
            for i, cfg in enumerate(self._head_configs):
                counts = {}
                for s, w in cfg:
                    counts[s] = counts.get(s, 0) + 1
                desc = "  ".join(f"s{s}×{n}" for s, n in sorted(counts.items()))
                lines.append(f"  Layer {i}: {desc}")
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
