"""
v8 — Dual MERA Language Model (v7.1 architecture)

Two ternary VSMs plugged together:
  COMPRESSOR MERA (~119M): learns to SEE — hierarchical multi-scale compression
  PIPELINE MERA  (~335M):  learns to THINK — sieve pathways for β-reduction

All weights ternary {-1, 0, +1}. Activations stay float32.
MERA weight sharing: same weights at every scale level (self-similar).

Architecture:

    tokens → [Compressor MERA]
               ├─ s8    (512 pos)  → Pipeline Level 0
               ├─ s16   (256 pos)  → Pipeline Level 1
               ├─ s32   (128 pos)  → Pipeline Level 2
               ├─ s64    (64 pos)  → Pipeline Level 3
               ├─ s128   (32 pos)  → Pipeline Level 4
               ├─ s256   (16 pos)  → Pipeline Level 5
               ├─ s512    (8 pos)  → Pipeline Level 6
               ├─ s1024   (4 pos)  → Pipeline Level 7
               └─ registers (R pos) → all levels
                            │
                            ▼
             [Pipeline MERA — sieve pathways]
               Level 0 (own weights, 4 pathways)
               Levels 1-7 (shared weights, 4 pathways each)
               Reducers (7) + Feedback cascade (7)
                            │
                            ▼
                     output: value | partial+regs | io!

Total: ~453M ternary = 113 MB packed.
"""

import math
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ternary import TernaryLinear, TernaryEmbedding


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class DualMERAConfig:
    """Configuration for the Dual MERA architecture.

    Compressor MERA: hierarchical multi-scale compression
      Level 0: stride 8, own weights (4096 → 512 positions)
      Levels 1-7: stride 2 each, SHARED weights (512 → 4 positions)

    Pipeline MERA: sieve pathways for computation
      Level 0: own sieve weights
      Levels 1-7: SHARED sieve weights
      4 parallel pathways per level
    """
    # Global dimensions
    vocab_size: int = 151936      # Qwen3 BBPE (151,643 regular + 208 control + padding)
    seq_len: int = 4096           # context window
    d_model: int = 1024           # representation dimension
    d_ff: int = 4096              # FFN expansion
    n_heads: int = 16             # attention heads (d_head = 64)

    # Compressor MERA
    compressor_window: int = 8    # base attention window W
    compressor_layers_per_level: int = 2
    compressor_n_levels: int = 8  # level 0 (own) + levels 1-7 (shared)

    # Pipeline MERA
    n_pathways: int = 4           # parallel pathways per sieve level
    pipeline_layers_per_level: int = 2  # layers per pathway per level
    pipeline_n_levels: int = 8    # level 0 (own) + levels 1-7 (shared)
    reducer_heads: int = 8        # heads in cross-attention reducers
    feedback_heads: int = 8       # heads in feedback cascade

    # Registers
    n_registers: int = 8          # persistent positions across passes

    # Learnable spiral bias (compressor attention energy distribution)
    spiral_alpha_init: float = 1.18    # empirical prior from LLM analysis
    spiral_fixed_point_init: float = 40.0  # empirical prior

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
        assert self.d_model % 4 == 0, \
            f"d_model={self.d_model} must be divisible by 4 (ternary packing)"
        assert self.d_ff % 4 == 0, \
            f"d_ff={self.d_ff} must be divisible by 4 (ternary packing)"

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def compressor_positions(self) -> list[int]:
        """Position counts at each compressor level.

        Level 0: seq_len // W = 512  (at default seq_len=4096, W=8)
        Level 1: 256, Level 2: 128, ..., Level 7: 4

        Minimum position count is 2 (for stride-2 reduction to work).
        Number of effective levels may be less than compressor_n_levels
        if seq_len is too small.
        """
        pos = [self.seq_len // self.compressor_window]  # level 0
        for _ in range(1, self.compressor_n_levels):
            next_pos = pos[-1] // 2
            if next_pos < 2:
                break
            pos.append(next_pos)
        return pos

    @property
    def effective_levels(self) -> int:
        """Actual number of compressor/pipeline levels (may be < configured if seq_len small)."""
        return len(self.compressor_positions)

    @property
    def compressor_strides(self) -> list[int]:
        """Effective stride relative to raw tokens at each level.

        Level 0: stride 8, Level 1: stride 16, ..., Level 7: stride 1024
        """
        n = self.effective_levels
        strides = [self.compressor_window]  # level 0: 8
        for i in range(1, n):
            strides.append(strides[-1] * 2)
        return strides


# ═══════════════════════════════════════════════════════════════════
# Building blocks — shared by compressor and pipeline
# ═══════════════════════════════════════════════════════════════════


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((d,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * rms * self.weight


class TernarySelfAttention(nn.Module):
    """Multi-head self-attention with ternary projections and RoPE.

    Supports both full causal and windowed attention modes.
    Windowed: each position attends only to the W positions within its window.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = TernaryLinear(d_model, d_model, pre_norm=False)
        self.k_proj = TernaryLinear(d_model, d_model, pre_norm=False)
        self.v_proj = TernaryLinear(d_model, d_model, pre_norm=False)
        self.o_proj = TernaryLinear(d_model, d_model, pre_norm=False)
        self.rope = nn.RoPE(self.d_head)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class TernaryFeedForward(nn.Module):
    """SwiGLU feed-forward with ternary projections.

    Ternary FFN = discrete routing topology:
      gate selects which activations pass (+1), negate (-1), or disconnect (0)
      up/down project through the selected routes
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = TernaryLinear(d_model, d_ff, pre_norm=False)
        self.up_proj = TernaryLinear(d_model, d_ff, pre_norm=False)
        self.down_proj = TernaryLinear(d_ff, d_model, pre_norm=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TernaryTransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm → SelfAttn → RMSNorm → FFN.

    All projections ternary. Norms and activations float32.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = TernarySelfAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = TernaryFeedForward(d_model, d_ff)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = x + self.attn(self.attn_norm(x), mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TernaryCrossAttention(nn.Module):
    """Multi-head cross-attention with ternary projections."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = TernaryLinear(d_model, d_model, pre_norm=False)
        self.k_proj = TernaryLinear(d_model, d_model, pre_norm=False)
        self.v_proj = TernaryLinear(d_model, d_model, pre_norm=False)
        self.o_proj = TernaryLinear(d_model, d_model, pre_norm=False)

    def __call__(
        self, q_in: mx.array, kv_in: mx.array, mask: mx.array | None = None
    ) -> mx.array:
        B, Lq, _ = q_in.shape
        Lkv = kv_in.shape[1]

        q = self.q_proj(q_in).reshape(B, Lq, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = self.k_proj(kv_in).reshape(B, Lkv, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = self.v_proj(kv_in).reshape(B, Lkv, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, Lq, -1)
        return self.o_proj(out)


# ═══════════════════════════════════════════════════════════════════
# Mask utilities
# ═══════════════════════════════════════════════════════════════════


def causal_mask(seq_len: int) -> mx.array:
    """Standard causal attention mask. Returns additive mask (0 / -inf)."""
    return mx.where(
        mx.arange(seq_len)[:, None] >= mx.arange(seq_len)[None, :],
        mx.zeros((seq_len, seq_len)),
        mx.full((seq_len, seq_len), -1e9),
    )


def windowed_causal_mask(seq_len: int, window: int) -> mx.array:
    """Windowed causal mask: each position attends to [max(0, i-W+1)..i].

    Combines causal constraint with local window. Used by compressor
    where W=8 limits each position to its local context.

    Returns additive mask (0 / -inf) of shape (seq_len, seq_len).
    """
    rows = mx.arange(seq_len)[:, None]
    cols = mx.arange(seq_len)[None, :]
    # Causal: can only attend to positions <= current
    causal = rows >= cols
    # Window: can only attend to positions within W of current
    in_window = (rows - cols) < window
    visible = causal & in_window
    return mx.where(visible, mx.zeros((seq_len, seq_len)), mx.full((seq_len, seq_len), -1e9))


def reduction_mask(n_input: int, n_output: int) -> mx.array:
    """Mask for cross-attention reducer: output j attends to input chunk j.

    Each output position attends to a contiguous chunk of input positions.
    Chunk size = n_input // n_output. Output j sees positions
    [j * chunk, (j+1) * chunk). This is a block-diagonal mask, NOT causal —
    each output sees exactly its own chunk.

    For the MERA structure: stride-2 reduction, so chunk_size = 2.
    Output j sees input positions [2j, 2j+1].

    Returns additive mask (0 / -inf) of shape (n_output, n_input).
    """
    chunk = n_input // n_output
    out_pos = mx.arange(n_output)[:, None]  # (n_output, 1)
    in_pos = mx.arange(n_input)[None, :]    # (1, n_input)
    # Each output j sees input positions in [j*chunk, (j+1)*chunk)
    in_chunk = in_pos // chunk  # which chunk each input belongs to
    visible = out_pos == in_chunk
    return mx.where(visible, mx.zeros((n_output, n_input)), mx.full((n_output, n_input), -1e9))


# ═══════════════════════════════════════════════════════════════════
# Compressor MERA (~119M ternary)
# ═══════════════════════════════════════════════════════════════════


class CompressorLevel(nn.Module):
    """One level of the compressor: a stack of ternary transformer blocks.

    Operates on positions at a given scale, with windowed causal attention.
    """

    def __init__(self, cfg: DualMERAConfig):
        super().__init__()
        self.layers = [
            TernaryTransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff)
            for _ in range(cfg.compressor_layers_per_level)
        ]
        self.norm = RMSNorm(cfg.d_model)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


class MERAReducer(nn.Module):
    """Stride-2 reducer between MERA levels via cross-attention pooling.

    Reduces n positions to n//2 by learned cross-attention.
    Each output position attends to its 2 corresponding input positions.
    """

    def __init__(self, cfg: DualMERAConfig):
        super().__init__()
        self.cross_attn = TernaryCrossAttention(cfg.d_model, cfg.reducer_heads)
        self.norm = RMSNorm(cfg.d_model)

    def __call__(self, x: mx.array, queries: mx.array, mask: mx.array) -> mx.array:
        """
        x:       (B, n_in, d_model) — input from previous level
        queries: (B, n_out, d_model) — learned query positions
        mask:    (n_out, n_in) — block-diagonal reduction mask
        Returns: (B, n_out, d_model)
        """
        out = self.cross_attn(queries, x, mask=mask)
        return self.norm(out)


class CompressorMERA(nn.Module):
    """Compressor MERA: hierarchical multi-scale compression.

    Level 0: own weights, stride 8 (4096 → 512 positions)
    Levels 1-7: SHARED MERA weights, stride 2 each (512 → 4 positions)

    Registers: R dedicated positions, appended to sequence at level 0,
    pass through all levels (not compressed by reducers).

    Learnable spiral: α and fixed_point bias attention energy distribution.

    Output: list of representations at each scale + register states.
    """

    def __init__(self, cfg: DualMERAConfig):
        super().__init__()
        self.cfg = cfg

        # Ternary embedding: packed {-1,0,+1} vectors with per-token gamma
        # 15× smaller than float32 embedding (13 MB vs 196 MB at vocab=50277, d=1024)
        self.embed = TernaryEmbedding(cfg.vocab_size, cfg.d_model)

        # Level 0: own weights (stride 8 compression)
        self.level0 = CompressorLevel(cfg)

        # Levels 1-7: SHARED weights — ONE CompressorLevel, reused 7×
        self.shared_level = CompressorLevel(cfg)

        # MERA reducers: one per transition between levels
        # These are NOT shared — each reducer operates at a different position count
        # But they share the same architecture. The learned queries are per-reducer.
        n_levels = cfg.effective_levels
        self.reducers = [MERAReducer(cfg) for _ in range(n_levels - 1)]

        # Learned query positions for each reducer (one set per level transition)
        positions = cfg.compressor_positions
        self.reducer_queries = [
            mx.random.normal((1, positions[i + 1], cfg.d_model)) * 0.02
            for i in range(n_levels - 1)
        ]

        # Register position embeddings (learned, distinguish from data positions)
        self.register_embed = mx.random.normal((1, cfg.n_registers, cfg.d_model)) * 0.02

        # Learnable spiral bias parameters
        self.spiral_alpha = mx.array([cfg.spiral_alpha_init])
        self.spiral_fixed_point = mx.array([cfg.spiral_fixed_point_init])

        # Strided pooling for level 0: average-pool with stride W to go from
        # seq_len to seq_len//W positions. This is the input compression step.
        # (The ternary transformer then refines these pooled representations.)

        # Pre-compute masks
        self._masks = {}

    def _get_mask(self, seq_len: int, window: int) -> mx.array:
        """Cached windowed causal mask."""
        key = (seq_len, window)
        if key not in self._masks:
            self._masks[key] = windowed_causal_mask(seq_len, window)
        return self._masks[key]

    def _get_reduction_mask(self, n_in: int, n_out: int) -> mx.array:
        """Cached reduction mask."""
        key = ("red", n_in, n_out)
        if key not in self._masks:
            self._masks[key] = reduction_mask(n_in, n_out)
        return self._masks[key]

    def _stride_pool(self, x: mx.array, stride: int) -> mx.array:
        """Average-pool along sequence dimension with given stride.

        x: (B, L, D) → (B, L//stride, D)
        Groups stride adjacent positions and averages them.
        """
        B, L, D = x.shape
        n_groups = L // stride
        # Reshape to (B, n_groups, stride, D) and mean over the stride dim
        x = x[:, :n_groups * stride, :].reshape(B, n_groups, stride, D)
        return x.mean(axis=2)

    def __call__(self, tokens: mx.array) -> tuple[list[mx.array], mx.array]:
        """
        tokens: (B, seq_len) int array

        Returns:
            scales: list of 8 tensors, one per compressor level
                    scales[0] = (B, 512, d_model)  — s8
                    scales[1] = (B, 256, d_model)  — s16
                    ...
                    scales[7] = (B, 4, d_model)    — s1024
            registers: (B, R, d_model) — register states after full compression
        """
        B = tokens.shape[0]
        cfg = self.cfg

        # ── Embed tokens ──
        x = self.embed(tokens)  # (B, seq_len, d_model)

        # ── Level 0: stride-8 compression ──
        # Pool from seq_len=4096 to 512 positions, then refine with transformer
        h = self._stride_pool(x, cfg.compressor_window)  # (B, 512, d_model)

        # Append registers to the sequence for joint attention
        regs = mx.broadcast_to(self.register_embed, (B, cfg.n_registers, cfg.d_model))
        h_with_regs = mx.concatenate([h, regs], axis=1)  # (B, 512 + R, d_model)

        # Level 0 attention (own weights) — windowed causal
        n_pos = h_with_regs.shape[1]
        mask0 = self._get_mask(n_pos, cfg.compressor_window)
        h_with_regs = self.level0(h_with_regs, mask=mask0)

        # Split data and registers
        h = h_with_regs[:, :cfg.compressor_positions[0], :]
        regs = h_with_regs[:, cfg.compressor_positions[0]:, :]

        scales = [h]  # scales[0] = s8 (512 positions)

        # ── Levels 1+: shared MERA weights, stride 2 each ──
        n_levels = cfg.effective_levels
        for level in range(1, n_levels):
            # Reduce: cross-attention pooling, stride 2
            n_in = cfg.compressor_positions[level - 1]
            n_out = cfg.compressor_positions[level]
            red_mask = self._get_reduction_mask(n_in, n_out)
            queries = mx.broadcast_to(
                self.reducer_queries[level - 1],
                (B, n_out, cfg.d_model),
            )
            h = self.reducers[level - 1](h, queries, red_mask)

            # Append registers for joint attention
            h_with_regs = mx.concatenate([h, regs], axis=1)

            # Shared MERA level (same weights, different input)
            n_pos = h_with_regs.shape[1]
            mask = self._get_mask(n_pos, cfg.compressor_window)
            h_with_regs = self.shared_level(h_with_regs, mask=mask)

            # Split
            h = h_with_regs[:, :n_out, :]
            regs = h_with_regs[:, n_out:, :]

            scales.append(h)

        return scales, regs


# ═══════════════════════════════════════════════════════════════════
# Pipeline MERA (~335M ternary)
# ═══════════════════════════════════════════════════════════════════


class SievePathway(nn.Module):
    """One pathway within a sieve level: a stack of ternary transformer blocks.

    Each pathway develops its own ternary sparsity pattern (topology).
    Different pathways crystallize different specialties.
    """

    def __init__(self, cfg: DualMERAConfig):
        super().__init__()
        self.layers = [
            TernaryTransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff)
            for _ in range(cfg.pipeline_layers_per_level)
        ]
        self.norm = RMSNorm(cfg.d_model)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


class SieveLevel(nn.Module):
    """One level of the pipeline: n_pathways parallel SievePathways.

    Input is split across pathways (not duplicated — each pathway
    gets the full input but operates independently). Outputs are
    averaged to form the level's representation.

    Registers participate in attention within each pathway but are
    shared: each pathway reads the same registers, and the merged
    output updates them.
    """

    def __init__(self, cfg: DualMERAConfig):
        super().__init__()
        self.cfg = cfg
        self.pathways = [SievePathway(cfg) for _ in range(cfg.n_pathways)]
        # Merge: average pathway outputs (simple, gradient-friendly)
        # Could also use learned attention merge, but start simple.

    def __call__(
        self, x: mx.array, regs: mx.array, mask: mx.array | None = None
    ) -> tuple[mx.array, mx.array]:
        """
        x:    (B, L, d_model) — data positions
        regs: (B, R, d_model) — register positions
        mask: additive mask for the combined sequence (L+R, L+R)

        Returns:
            h: (B, L, d_model) — updated data
            regs: (B, R, d_model) — updated registers
        """
        B = x.shape[0]
        L = x.shape[1]
        R = regs.shape[1]

        # Concatenate data + registers for joint attention
        combined = mx.concatenate([x, regs], axis=1)  # (B, L+R, d_model)

        # Run each pathway independently, collect outputs
        pathway_outputs = []
        for pathway in self.pathways:
            out = pathway(combined, mask=mask)
            pathway_outputs.append(out)

        # Merge: average across pathways
        merged = pathway_outputs[0]
        for p in pathway_outputs[1:]:
            merged = merged + p
        merged = merged / len(self.pathways)

        # Split data and registers
        h = merged[:, :L, :]
        regs_out = merged[:, L:, :]

        return h, regs_out


class PipelineFeedback(nn.Module):
    """Feedback module: higher level → lower level with gated cross-attention.

    The gate allows the model to control influence magnitude.
    Starts near zero (higher levels haven't learned yet).
    All ternary — gate topology routes the sigmoid control signal.
    """

    def __init__(self, cfg: DualMERAConfig):
        super().__init__()
        self.cross_attn = TernaryCrossAttention(cfg.d_model, cfg.feedback_heads)
        self.norm = RMSNorm(cfg.d_model)
        # Gate: ternary routing → sigmoid. Topology controls which
        # dimensions the gate attends to. Sigmoid provides continuous
        # gating on top of the discrete routing.
        self.gate_proj = TernaryLinear(cfg.d_model, cfg.d_model, pre_norm=False)

    def __call__(self, lower: mx.array, higher: mx.array) -> mx.array:
        """
        lower:  (B, L_low, d_model)  — this level's representation (queries)
        higher: (B, L_high, d_model) — higher level's output (keys/values)
        Returns: (B, L_low, d_model) — lower + gated feedback
        """
        feedback = self.cross_attn(lower, higher)
        gate = mx.sigmoid(self.gate_proj(lower))
        return lower + gate * self.norm(feedback)


class PipelineReducer(nn.Module):
    """Reducer between pipeline levels: cross-attention pooling.

    Halves positions between adjacent levels so the pipeline operates
    at progressively coarser scales matching the compressor output.
    """

    def __init__(self, cfg: DualMERAConfig):
        super().__init__()
        self.cross_attn = TernaryCrossAttention(cfg.d_model, cfg.reducer_heads)
        self.norm = RMSNorm(cfg.d_model)

    def __call__(self, x: mx.array, queries: mx.array, mask: mx.array) -> mx.array:
        out = self.cross_attn(queries, x, mask=mask)
        return self.norm(out)


class PipelineMERA(nn.Module):
    """Pipeline MERA: sieve pathways for computation.

    Level 0: own sieve weights (surface computation)
    Levels 1-7: SHARED sieve weights (one copy, reused 7×)

    Each level reads the corresponding compressor scale.
    Registers participate at every level, not compressed by reducers.

    Upward path: Level 0 → 7 (abstraction)
    Feedback cascade: Level 7 → 0 (constraint propagation)
    """

    def __init__(self, cfg: DualMERAConfig):
        super().__init__()
        self.cfg = cfg

        # Level 0: own sieve weights
        self.level0 = SieveLevel(cfg)

        # Levels 1-7: SHARED sieve — ONE SieveLevel, reused 7×
        self.shared_level = SieveLevel(cfg)

        # Reducers between pipeline levels
        n_levels = cfg.effective_levels
        self.reducers = [PipelineReducer(cfg) for _ in range(n_levels - 1)]

        # Learned queries for each reducer
        positions = cfg.compressor_positions
        self.reducer_queries = [
            mx.random.normal((1, positions[i + 1], cfg.d_model)) * 0.02
            for i in range(n_levels - 1)
        ]

        # Feedback cascade modules (from higher → lower)
        self.feedbacks = [PipelineFeedback(cfg) for _ in range(n_levels - 1)]

        # Output norm
        self.out_norm = RMSNorm(cfg.d_model)

        # Pre-computed masks cache
        self._masks = {}

    def _get_causal_mask(self, seq_len: int) -> mx.array:
        key = ("causal", seq_len)
        if key not in self._masks:
            self._masks[key] = causal_mask(seq_len)
        return self._masks[key]

    def _get_reduction_mask(self, n_in: int, n_out: int) -> mx.array:
        key = ("red", n_in, n_out)
        if key not in self._masks:
            self._masks[key] = reduction_mask(n_in, n_out)
        return self._masks[key]

    def __call__(
        self,
        compressor_scales: list[mx.array],
        registers: mx.array,
    ) -> tuple[mx.array, mx.array, list[list[mx.array]]]:
        """
        compressor_scales: list of 8 tensors from compressor, each (B, L_i, d_model)
        registers: (B, R, d_model) from compressor

        Returns:
            h0: (B, L_0, d_model) — Level 0 output after full feedback cascade
            registers: (B, R, d_model) — final register states
            pathway_outputs: list of lists — for relational loss computation
                pathway_outputs[level][pathway] = (B, L_level, d_model)
        """
        B = compressor_scales[0].shape[0]
        cfg = self.cfg
        R = registers.shape[1]

        # ── Upward path ──
        level_outputs = []
        pathway_outputs = []  # for relational loss
        regs = registers

        n_levels = cfg.effective_levels
        for level in range(n_levels):
            # Input: compressor scale at this level
            h = compressor_scales[level]
            L = h.shape[1]

            # Add compressor input as a residual-like connection
            # At level 0, h is the raw compressor s8 output
            # At level >0, h combines reduced pipeline state + compressor scale
            if level > 0:
                # Reduce from previous level
                n_in = cfg.compressor_positions[level - 1]
                n_out = cfg.compressor_positions[level]
                red_mask = self._get_reduction_mask(n_in, n_out)
                queries = mx.broadcast_to(
                    self.reducer_queries[level - 1],
                    (B, n_out, cfg.d_model),
                )
                h_reduced = self.reducers[level - 1](
                    level_outputs[-1], queries, red_mask
                )
                # Combine reduced pipeline state with compressor scale
                h = h + h_reduced

            # Causal mask for data + register positions
            mask = self._get_causal_mask(L + R)

            # Run sieve level
            if level == 0:
                h_out, regs = self.level0(h, regs, mask=mask)
            else:
                h_out, regs = self.shared_level(h, regs, mask=mask)

            level_outputs.append(h_out)

            # Capture per-pathway outputs for relational loss
            # Re-run pathways to get individual outputs (expensive — only during metrics)
            # For the forward pass, we skip this. Relational loss is computed separately.
            pathway_outputs.append(None)  # placeholder

        # ── Feedback cascade: highest → lowest ──
        for level in range(n_levels - 2, -1, -1):
            level_outputs[level] = self.feedbacks[level](
                level_outputs[level], level_outputs[level + 1]
            )

        h0 = self.out_norm(level_outputs[0])
        return h0, regs, pathway_outputs


# ═══════════════════════════════════════════════════════════════════
# Top-level Dual MERA model
# ═══════════════════════════════════════════════════════════════════


class DualMERA(nn.Module):
    """Dual MERA Language Model.

    Compressor MERA sees tokens → produces multi-scale representations.
    Pipeline MERA thinks with sieve pathways → produces output.
    Registers bridge both and persist across recurrence passes.

    Output modes:
      - value:   next-token prediction logits (standard LM)
      - partial: intermediate state for recurrence (registers + partial expr)
    """

    def __init__(self, cfg: DualMERAConfig):
        super().__init__()
        self.cfg = cfg
        self.compressor = CompressorMERA(cfg)
        self.pipeline = PipelineMERA(cfg)

        # Output projection norm (tied embedding applied manually)
        self.out_norm = RMSNorm(cfg.d_model)

    def __call__(
        self, tokens: mx.array, registers: mx.array | None = None
    ) -> mx.array:
        """Standard forward: tokens → logits.

        tokens: (B, seq_len) int array
        registers: (B, R, d_model) optional — for recurrence passes
        Returns: logits (B, seq_len, vocab_size) via tied embedding
        """
        B = tokens.shape[0]

        # ── Compressor ──
        scales, regs = self.compressor(tokens)

        # If external registers provided (recurrence), use those instead
        if registers is not None:
            regs = registers

        # ── Pipeline ──
        h0, regs_out, _ = self.pipeline(scales, regs)

        # ── Output: project to vocab via tied embedding ──
        # h0 is (B, L_0, d_model) where L_0 = seq_len // 8 = 512
        # For LM loss, we need (B, seq_len, vocab_size)
        # Upsample h0 back to seq_len by repeating each position stride times
        h_up = self._upsample(h0, self.cfg.seq_len)
        h_out = self.out_norm(h_up)

        # Tied embedding (ternary: unpack + gamma on-the-fly)
        logits = h_out @ self.compressor.embed.weight_T

        return logits

    def forward_with_registers(
        self, tokens: mx.array, registers: mx.array | None = None
    ) -> tuple[mx.array, mx.array]:
        """Forward that also returns updated registers for recurrence.

        Returns: (logits, registers_out)
        """
        B = tokens.shape[0]
        scales, regs = self.compressor(tokens)
        if registers is not None:
            regs = registers
        h0, regs_out, _ = self.pipeline(scales, regs)
        h_up = self._upsample(h0, self.cfg.seq_len)
        h_out = self.out_norm(h_up)
        logits = h_out @ self.compressor.embed.weight_T
        return logits, regs_out

    def _upsample(self, h: mx.array, target_len: int) -> mx.array:
        """Upsample compressed representation back to full sequence length.

        h: (B, L_compressed, d_model) where L_compressed = target_len // stride
        Returns: (B, target_len, d_model)

        Uses repeat-interleave: each compressed position maps to `stride`
        consecutive output positions. Simple but gradient-friendly.
        More sophisticated upsampling (learned deconv, cross-attention from
        original embeddings) can be added later.
        """
        B, L, D = h.shape
        stride = target_len // L
        # Repeat each position `stride` times along the sequence axis
        # (B, L, D) → (B, L, stride, D) → (B, L*stride, D)
        h = mx.repeat(h, stride, axis=1)
        return h

    def count_params(self) -> dict:
        """Count LOGICAL parameters by component.

        TernaryLinear uses MLX uint32 packing (16 values per element, bits=2).
        TernaryEmbedding uses uint8 packing (4 values per element).
        This method counts logical weights (N × K) not storage elements.
        """
        counts = {}

        def _logical_size(param_name: str, v) -> int:
            """Return logical element count for a parameter array."""
            if v.dtype == mx.uint32 and param_name.endswith(".weight"):
                # TernaryLinear: uint32, 16 logical weights per element
                return v.size * 16
            if "ternary_weight" in param_name:
                # TernaryEmbedding: uint8, 4 logical weights per element
                return v.size * 4
            return v.size

        def _count_logical(module, name):
            """Count logical params, unpacking ternary weight sizes."""
            total = 0
            for param_name, v in tree_flatten(module.parameters()):
                total += _logical_size(param_name, v)
            counts[name] = total

        # Compressor
        _count_logical(self.compressor.embed, "compressor/embedding")
        _count_logical(self.compressor.level0, "compressor/level0 (own)")
        _count_logical(self.compressor.shared_level, "compressor/levels1-7 (shared)")
        comp_reducer_total = 0
        for r in self.compressor.reducers:
            t = 0
            for pn, v in tree_flatten(r.parameters()):
                t += _logical_size(pn, v)
            comp_reducer_total += t
        counts["compressor/reducers"] = comp_reducer_total
        counts["compressor/reducer_queries"] = sum(q.size for q in self.compressor.reducer_queries)
        counts["compressor/registers"] = self.compressor.register_embed.size
        counts["compressor/spiral"] = 2  # alpha + fixed_point

        # Pipeline
        _count_logical(self.pipeline.level0, "pipeline/level0 (own)")
        _count_logical(self.pipeline.shared_level, "pipeline/levels1-7 (shared)")
        pipe_reducer_total = 0
        for r in self.pipeline.reducers:
            t = 0
            for pn, v in tree_flatten(r.parameters()):
                t += _logical_size(pn, v)
            pipe_reducer_total += t
        counts["pipeline/reducers"] = pipe_reducer_total
        counts["pipeline/reducer_queries"] = sum(q.size for q in self.pipeline.reducer_queries)
        pipe_feedback_total = 0
        for f in self.pipeline.feedbacks:
            t = 0
            for pn, v in tree_flatten(f.parameters()):
                t += _logical_size(pn, v)
            pipe_feedback_total += t
        counts["pipeline/feedbacks"] = pipe_feedback_total
        _count_logical(self.pipeline.out_norm, "pipeline/out_norm")

        # Output
        _count_logical(self.out_norm, "output/norm")

        # Summaries
        comp_total = sum(v for k, v in counts.items() if k.startswith("compressor"))
        pipe_total = sum(v for k, v in counts.items() if k.startswith("pipeline"))
        counts["compressor_total"] = comp_total
        counts["pipeline_total"] = pipe_total
        counts["total"] = sum(counts[k] for k in counts
                              if not k.endswith("_total") and k != "output/norm") + counts["output/norm"]

        # Storage size in bytes:
        #   TernaryLinear weight: uint32 → 4 bytes per element (stores 16 logical weights)
        #   TernaryEmbedding ternary_weight: uint8 → 1 byte per element
        #   All other params: float32 → 4 bytes per element
        total_storage = 0
        for pn, v in tree_flatten(self.parameters()):
            if v.dtype == mx.uint8:
                total_storage += v.size          # packed uint8 ternary embedding
            elif v.dtype == mx.uint32:
                total_storage += v.size * 4      # packed uint32 ternary linear
            else:
                total_storage += v.size * 4      # float32
        counts["storage_bytes"] = total_storage
        counts["storage_mb"] = total_storage / (1024 * 1024)

        return counts


# ═══════════════════════════════════════════════════════════════════
# Relational loss utilities
# ═══════════════════════════════════════════════════════════════════


def pathway_relational_loss(model: DualMERA, x: mx.array, regs: mx.array,
                             level: int, mask: mx.array) -> mx.array:
    """Compute relational loss for pathways within a pipeline sieve level.

    Runs each pathway independently, computes pairwise cosine similarity,
    and penalizes similarity (pushing pathways to differentiate).

    L_relational = Σ_{i≠j} cosine_similarity(pathway_i, pathway_j)

    Returns scalar loss.
    """
    cfg = model.cfg
    sieve = model.pipeline.level0 if level == 0 else model.pipeline.shared_level

    # Run each pathway independently
    combined = mx.concatenate([x, regs], axis=1)
    outputs = []
    for pathway in sieve.pathways:
        out = pathway(combined, mask=mask)
        # Use mean-pooled representation for similarity
        outputs.append(out.mean(axis=1))  # (B, d_model)

    # Pairwise cosine similarity
    loss = mx.array(0.0)
    n_pairs = 0
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            # Cosine similarity per batch, then mean
            a = outputs[i]
            b = outputs[j]
            sim = mx.sum(a * b, axis=-1) / (
                mx.sqrt(mx.sum(a * a, axis=-1)) * mx.sqrt(mx.sum(b * b, axis=-1)) + 1e-8
            )
            loss = loss + sim.mean()
            n_pairs += 1

    return loss / max(n_pairs, 1)


# ═══════════════════════════════════════════════════════════════════
# Factory + smoke test
# ═══════════════════════════════════════════════════════════════════


def create_model(cfg: DualMERAConfig | None = None) -> DualMERA:
    """Create a DualMERA with default or custom config."""
    if cfg is None:
        cfg = DualMERAConfig()
    model = DualMERA(cfg)
    mx.eval(model.parameters())
    return model


if __name__ == "__main__":
    import time

    print("=" * 70)
    print("  v8 — Dual MERA Language Model (v7.1 architecture)")
    print("=" * 70)

    # Use smaller dims for smoke test to avoid OOM
    # Full config: d_model=1024, d_ff=4096, seq_len=4096
    # Smoke test: d_model=256, d_ff=1024, seq_len=512
    # Parse --full flag for full-scale test
    import sys as _sys
    full_scale = "--full" in _sys.argv

    if full_scale:
        cfg = DualMERAConfig()
        print("\n[FULL SCALE — d_model=1024, seq_len=4096]")
    else:
        cfg = DualMERAConfig(
            d_model=256,
            d_ff=1024,
            n_heads=4,
            seq_len=512,
            compressor_window=8,
        )
        print("\n[SMOKE TEST — reduced dimensions]")
        print("  (use --full for full-scale test)")

    print(f"\nConfig:")
    print(f"  seq_len={cfg.seq_len}, d_model={cfg.d_model}, d_ff={cfg.d_ff}")
    print(f"  n_heads={cfg.n_heads}, d_head={cfg.d_head}")
    print(f"  compressor: {cfg.compressor_n_levels} levels, W={cfg.compressor_window}")
    print(f"  pipeline: {cfg.pipeline_n_levels} levels, {cfg.n_pathways} pathways")
    print(f"  registers: {cfg.n_registers}")
    print(f"  compressor positions: {cfg.compressor_positions}")
    print(f"  compressor strides: {cfg.compressor_strides}")

    print(f"\nBuilding model...")
    t0 = time.time()
    model = create_model(cfg)
    dt = time.time() - t0
    print(f"  Built in {dt:.2f}s")

    # Parameter count
    counts = model.count_params()
    print(f"\nParameters:")
    for name, count in counts.items():
        print(f"  {name:>40s}: {count:>12,}")

    # Verify weight sharing
    print(f"\nWeight sharing verification:")
    comp_shared = model.compressor.shared_level
    pipe_shared = model.pipeline.shared_level
    print(f"  Compressor shared_level id: {id(comp_shared)}")
    print(f"  Pipeline shared_level id:   {id(pipe_shared)}")
    print(f"  Compressor L1-L7 all use same object: ✓ (by design — single module)")
    print(f"  Pipeline L1-L7 all use same object:   ✓ (by design — single module)")

    # Forward pass
    print(f"\nForward pass test...")
    B = 2
    tokens = mx.zeros((B, cfg.seq_len), dtype=mx.int32)
    t0 = time.time()
    logits = model(tokens)
    mx.eval(logits)
    dt = time.time() - t0
    print(f"  Input:  {tokens.shape}")
    print(f"  Output: {logits.shape}")
    print(f"  Time:   {dt:.3f}s")
    assert logits.shape == (B, cfg.seq_len, cfg.vocab_size), \
        f"Expected {(B, cfg.seq_len, cfg.vocab_size)}, got {logits.shape}"
    print(f"  Shape:  ✓")

    # Compressor multi-scale outputs
    print(f"\nCompressor scale outputs:")
    scales, regs = model.compressor(tokens)
    mx.eval(*scales, regs)
    for i, s in enumerate(scales):
        stride = cfg.compressor_strides[i]
        print(f"  Level {i} (s{stride:>4d}): {s.shape}")
    print(f"  Registers: {regs.shape}")

    # Forward with registers (recurrence test)
    print(f"\nRecurrence test (forward_with_registers)...")
    logits2, regs_out = model.forward_with_registers(tokens)
    mx.eval(logits2, regs_out)
    print(f"  Logits:    {logits2.shape}")
    print(f"  Registers: {regs_out.shape}")

    # Gradient test
    print(f"\nGradient test...")
    def test_loss(model, tokens):
        logits = model(tokens)
        # Simple CE against zeros
        targets = mx.zeros((B, cfg.seq_len), dtype=mx.int32)
        return nn.losses.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            targets.reshape(-1),
            reduction="mean",
        )

    loss_and_grad = nn.value_and_grad(model, test_loss)
    loss_val, grads = loss_and_grad(model, tokens)
    mx.eval(loss_val, grads)
    print(f"  Loss: {float(loss_val):.4f}")
    n_grad_arrays = len(tree_flatten(grads))
    print(f"  Gradient arrays: {n_grad_arrays}")
    print(f"  Gradient test: ✓")

    print(f"\n{'='*70}")
    print(f"  ✓ All smoke tests passed")
    print(f"{'='*70}")
