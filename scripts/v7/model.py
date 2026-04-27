"""
v7 — 4-VSM Pipeline Language Model

Four stages of increasing abstraction, each an independent transformer.
Upward path: abstraction (tokens → surface → structural → semantic → reasoning).
Downward path: constraint propagation (reasoning → semantic → structural → surface).
Prediction emerges from Stage 1 after feedback from all higher stages.

Each stage operates on fewer positions than the previous one (the compute
pyramid). Reduction between stages via learned cross-attention pooling.
Feedback via cross-attention with learned gating.

Attention complexity: O(L₁·n²) — dominated by Stage 1 (shallowest).
Deeper stages are computationally negligible due to position reduction.

Architecture:

    tokens → [Embed] → [Stage1: n pos] → [Reduce] → [Stage2: n/r pos]
                 ↑          ↓ feedback        ↓
              logits    [Stage3: n/r² pos] ← [Reduce]
                             ↓ feedback
                        [Stage4: n/r³ pos] ← [Reduce]

Forward: up through 4 stages. Feedback: down through 4 stages.
Output: Stage 1 representation → logits.
"""

import math
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ternary import TernaryLinear


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class StageConfig:
    """Configuration for a single VSM stage."""

    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    vocab_size: int = 50277  # GPT-NeoX
    seq_len: int = 512
    d_model: int = 256  # shared representation dimension

    # Per-stage configs (surface → structural → semantic → reasoning)
    stages: list[StageConfig] = field(default_factory=lambda: [
        StageConfig(n_layers=2, n_heads=4, d_model=256, d_ff=512),     # Stage 1: Surface
        StageConfig(n_layers=3, n_heads=4, d_model=256, d_ff=512),     # Stage 2: Structural
        StageConfig(n_layers=4, n_heads=8, d_model=256, d_ff=1024),    # Stage 3: Semantic
        StageConfig(n_layers=6, n_heads=8, d_model=256, d_ff=1024),    # Stage 4: Reasoning
    ])

    # Position counts per stage. Stage 0 = seq_len, rest = reduced.
    # Default: 512 → 64 → 8 → 1  (three 8× reductions)
    stage_positions: list[int] = field(default_factory=lambda: [512, 64, 8, 1])

    # Feedback / reducer heads
    reducer_heads: int = 4
    feedback_heads: int = 4

    # Ternary control: which stages and components use ternary weights
    # Stage 1 (surface) = hot path → ternary. Stages 2-4 = cold path → float.
    ternary_stages: list[bool] = field(default_factory=lambda: [True, False, False, False])
    ternary_feedback: bool = True  # feedback modules are also hot path

    def __post_init__(self):
        assert len(self.stages) == len(self.stage_positions)
        assert len(self.ternary_stages) == len(self.stages)
        assert self.stage_positions[0] == self.seq_len
        # Ternary requires d_model divisible by 4 (packing constraint)
        for i, is_ternary in enumerate(self.ternary_stages):
            if is_ternary:
                assert self.stages[i].d_model % 4 == 0, \
                    f"Stage {i} d_model={self.stages[i].d_model} must be divisible by 4 for ternary"


# ═══════════════════════════════════════════════════════════════════
# Building blocks
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


class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE and causal masking."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
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


class CrossAttention(nn.Module):
    """Multi-head cross-attention. Queries from one stage, keys/values from another."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

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


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm → SelfAttn → RMSNorm → FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = x + self.attn(self.attn_norm(x), mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ═══════════════════════════════════════════════════════════════════
# Ternary building blocks (hot-path: Stage 1 + Feedback)
# ═══════════════════════════════════════════════════════════════════


class TernarySelfAttention(nn.Module):
    """Multi-head self-attention with ternary Q,K,V,O projections.

    RoPE and causal masking are identical to float version.
    Projections use TernaryLinear (packed uint8, add/sub only on Metal).
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Ternary projections: no bias, pre_norm handled externally
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
    """SwiGLU feed-forward with ternary projections."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = TernaryLinear(d_model, d_ff, pre_norm=False)
        self.up_proj = TernaryLinear(d_model, d_ff, pre_norm=False)
        self.down_proj = TernaryLinear(d_ff, d_model, pre_norm=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TernaryTransformerBlock(nn.Module):
    """Pre-norm transformer block with ternary attention + FFN."""

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
# Stage components
# ═══════════════════════════════════════════════════════════════════


class TransformerStage(nn.Module):
    """A stack of transformer blocks — one VSM stage.

    Operates over a fixed number of positions with causal self-attention.
    Each stage is an independent transformer with its own parameters.
    Supports ternary or float blocks based on the `ternary` flag.
    """

    def __init__(self, cfg: StageConfig, ternary: bool = False):
        super().__init__()
        Block = TernaryTransformerBlock if ternary else TransformerBlock
        self.layers = [
            Block(cfg.d_model, cfg.n_heads, cfg.d_ff)
            for _ in range(cfg.n_layers)
        ]
        self.norm = RMSNorm(cfg.d_model)
        self.is_ternary = ternary

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


class StageReducer(nn.Module):
    """Reduce positions between stages via learned cross-attention pooling.

    Uses a set of learned query vectors that cross-attend to the previous
    stage's output. Causality: output position j attends only to input
    positions in chunks 0..j (each chunk = input_positions / output_positions).

    This is where the 10× search-space reduction happens — each output
    position learns to summarize its chunk of the input into a denser
    representation at the next level of abstraction.
    """

    def __init__(self, d_model: int, n_output_positions: int, n_heads: int):
        super().__init__()
        self.n_output = n_output_positions
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.queries = mx.random.normal((1, n_output_positions, d_model)) * 0.02
        self.norm = RMSNorm(d_model)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        """
        x:    (B, n_input, d_model) — previous stage output
        mask: (n_output, n_input) — causal reduction mask
        Returns: (B, n_output, d_model)
        """
        B = x.shape[0]
        q = mx.broadcast_to(self.queries, (B, self.n_output, x.shape[-1]))
        out = self.cross_attn(q, x, mask=mask)
        return self.norm(out)


class StageFeedback(nn.Module):
    """Incorporate higher stage's output into lower stage's representation.

    Cross-attention (lower queries, higher keys/values) with a learned
    sigmoid gate on the residual. The gate lets the model control how
    much influence the higher stage has — starting near zero and
    increasing as the higher stage learns meaningful representations.

    This is the downward constraint propagation path.
    Supports ternary cross-attention for the hot path (feedback to Stage 1).
    """

    def __init__(self, d_model: int, n_heads: int, ternary: bool = False):
        super().__init__()
        Attn = TernaryCrossAttention if ternary else CrossAttention
        self.cross_attn = Attn(d_model, n_heads)
        self.norm = RMSNorm(d_model)
        # Gate: always float (cheap, needs precision for sigmoid)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.is_ternary = ternary

    def __call__(self, lower: mx.array, higher: mx.array) -> mx.array:
        """
        lower:  (B, n_lower, d_model) — this stage's representation (queries)
        higher: (B, n_higher, d_model) — higher stage's output (keys/values)
        Returns: (B, n_lower, d_model) — lower + gated feedback
        """
        feedback = self.cross_attn(lower, higher)
        gate = mx.sigmoid(self.gate_proj(lower))
        return lower + gate * self.norm(feedback)


# ═══════════════════════════════════════════════════════════════════
# Mask utilities
# ═══════════════════════════════════════════════════════════════════


def causal_mask(seq_len: int) -> mx.array:
    """Standard causal attention mask. Returns additive mask (0 / -inf)."""
    mask = mx.full((seq_len, seq_len), -1e9)
    mask = mx.triu(mask, k=1)  # zero on and below diagonal
    # Invert: we want causal (lower-triangular allowed)
    return mx.where(
        mx.arange(seq_len)[:, None] >= mx.arange(seq_len)[None, :],
        mx.zeros((seq_len, seq_len)),
        mx.full((seq_len, seq_len), -1e9),
    )


def reduction_causal_mask(n_input: int, n_output: int) -> mx.array:
    """Causal mask for the StageReducer cross-attention.

    Output position j can attend to input positions in chunks 0..j.
    Chunk size = n_input / n_output (integer division).

    If n_output == 1 (Stage 4), the single output position sees all inputs.
    """
    chunk_size = n_input // n_output
    # Last input position visible to each output position
    # output j sees input positions 0..((j+1)*chunk_size - 1)
    boundaries = mx.arange(1, n_output + 1) * chunk_size  # (n_output,)
    input_positions = mx.arange(n_input)  # (n_input,)

    # mask[j, i] = 0.0 if input_positions[i] < boundaries[j], else -1e9
    visible = input_positions[None, :] < boundaries[:, None]  # (n_output, n_input)
    return mx.where(visible, mx.zeros((n_output, n_input)), mx.full((n_output, n_input), -1e9))


# ═══════════════════════════════════════════════════════════════════
# The full pipeline
# ═══════════════════════════════════════════════════════════════════


class VSMPipeline(nn.Module):
    """4-VSM Pipeline Language Model.

    Forward pass:
      1. Embed tokens
      2. Stage 1 (Surface): full-resolution causal self-attention
      3. Reduce → Stage 2 (Structural): reduced positions
      4. Reduce → Stage 3 (Semantic): further reduced
      5. Reduce → Stage 4 (Reasoning): minimal positions
      6. Feedback: Stage 4 → 3 → 2 → 1 (constraint propagation)
      7. Project Stage 1 output → logits (tied embeddings)

    The compute pyramid: each stage is deeper but over exponentially
    fewer positions. Total attention cost ≈ O(L₁ · n²).
    """

    def __init__(self, cfg: PipelineConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding (tied with output projection)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # 4 transformer stages (ternary or float per config)
        self.stages = [
            TransformerStage(s, ternary=cfg.ternary_stages[i])
            for i, s in enumerate(cfg.stages)
        ]

        # 3 reducers (between stages 1→2, 2→3, 3→4) — always float
        # Reducers are cold path (run rarely), precision matters for learned queries
        self.reducers = [
            StageReducer(cfg.d_model, cfg.stage_positions[i + 1], cfg.reducer_heads)
            for i in range(len(cfg.stages) - 1)
        ]

        # 3 feedback modules (from stages 4→3, 3→2, 2→1)
        # feedback[0] = 2→1 (hot: runs every token) → ternary if configured
        # feedback[1] = 3→2, feedback[2] = 4→3 → float (cold path)
        self.feedbacks = [
            StageFeedback(
                cfg.d_model, cfg.feedback_heads,
                ternary=(cfg.ternary_feedback and i == 0),  # only feedback to Stage 1
            )
            for i in range(len(cfg.stages) - 1)
        ]

        # Output projection (tied with embeddings — applied manually)
        self.out_norm = RMSNorm(cfg.d_model)

        # Pre-compute masks (static for a given config)
        self._causal_masks = [causal_mask(p) for p in cfg.stage_positions]
        self._reduction_masks = [
            reduction_causal_mask(cfg.stage_positions[i], cfg.stage_positions[i + 1])
            for i in range(len(cfg.stages) - 1)
        ]

    def __call__(self, tokens: mx.array) -> mx.array:
        """
        tokens: (B, seq_len) int array
        Returns: logits (B, seq_len, vocab_size)
        """
        B, L = tokens.shape

        # ── Embed ──
        x = self.embed(tokens)  # (B, L, d_model)

        # ── Upward path: abstraction ──
        stage_outputs = []
        h = x
        for i, stage in enumerate(self.stages):
            h = stage(h, mask=self._causal_masks[i])
            stage_outputs.append(h)
            # Reduce for next stage (except last)
            if i < len(self.stages) - 1:
                h = self.reducers[i](h, mask=self._reduction_masks[i])

        # ── Downward path: constraint propagation ──
        # Walk backwards: stage 4→3, 3→2, 2→1
        # Each feedback uses the ALREADY-REFINED higher stage output,
        # so constraints cascade: 4's reasoning refines 3, refined-3
        # then refines 2, refined-2 then refines 1.
        for i in range(len(self.stages) - 2, -1, -1):
            stage_outputs[i] = self.feedbacks[i](stage_outputs[i], stage_outputs[i + 1])

        # ── Output from Stage 1 (full token resolution) ──
        h_out = self.out_norm(stage_outputs[0])
        # Tied embedding: logits = h_out @ embed.weight.T
        logits = h_out @ self.embed.weight.T

        return logits

    def _stage1_ce(self, h1: mx.array, targets: mx.array) -> mx.array:
        """Project Stage 1 representation to logits and compute CE.

        Returns an mx.array scalar — caller is responsible for mx.eval().
        Do NOT call float() here; batch evaluations externally.
        """
        h_out = self.out_norm(h1)
        logits = h_out @ self.embed.weight.T
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        )

    def forward_with_metrics(
        self, tokens: mx.array, targets: mx.array | None = None
    ) -> tuple[mx.array, dict]:
        """Forward pass with per-stage metrics. Use outside grad computation.

        When targets are provided, computes cross-entropy at each step
        of the feedback cascade to measure each stage's contribution:

          ce_stage1: Stage 1 alone (no feedback)
          ce_stage2: Stage 1 + feedback from raw Stage 2
          ce_stage3: Stage 1 + feedback from Stage 2 refined by Stage 3
          ce_stage4: Stage 1 + full cascade (2 refined by 3 refined by 4)

        CE₁ ≥ CE₂ ≥ CE₃ ≥ CE₄ when each stage adds value.
        Δₖ = CEₖ₋₁ - CEₖ = value contributed by stage k's feedback.
        """
        B, L = tokens.shape
        metrics = {}

        x = self.embed(tokens)

        # ── Upward path: abstraction ──
        stage_outputs = []
        h_norms = []
        h = x
        for i, stage in enumerate(self.stages):
            h = stage(h, mask=self._causal_masks[i])
            stage_outputs.append(h)
            h_norms.append(mx.mean(mx.sqrt(mx.sum(h * h, axis=-1))))
            if i < len(self.stages) - 1:
                h = self.reducers[i](h, mask=self._reduction_masks[i])

        # Single eval for all norms
        mx.eval(*h_norms)
        for i, hn in enumerate(h_norms):
            metrics[f"stage{i+1}_h_norm"] = float(hn)

        # ── Per-stage CE measurement (incremental feedback) ──
        if targets is not None:
            # Save raw stage outputs (before any feedback modifies them)
            raw = [s for s in stage_outputs]

            # Build all 4 CE computations lazily, then eval once
            # CE₁: Stage 1 alone — surface-only prediction
            ce1 = self._stage1_ce(raw[0], targets)

            # CE₂: Stage 1 + feedback from raw Stage 2
            h1_fb2 = self.feedbacks[0](raw[0], raw[1])
            ce2 = self._stage1_ce(h1_fb2, targets)

            # CE₃: Stage 1 + feedback from Stage 2 refined by raw Stage 3
            s2_with_s3 = self.feedbacks[1](raw[1], raw[2])
            h1_fb23 = self.feedbacks[0](raw[0], s2_with_s3)
            ce3 = self._stage1_ce(h1_fb23, targets)

            # CE₄: Full cascade — Stage 3 refined by 4, Stage 2 by refined-3,
            # Stage 1 by refined-2. This equals the main training loss.
            s3_with_s4 = self.feedbacks[2](raw[2], raw[3])
            s2_with_s34 = self.feedbacks[1](raw[1], s3_with_s4)
            h1_fb234 = self.feedbacks[0](raw[0], s2_with_s34)
            ce4 = self._stage1_ce(h1_fb234, targets)

            # Single eval for all 4 CEs — one sync point, not four
            mx.eval(ce1, ce2, ce3, ce4)
            metrics["ce_stage1"] = float(ce1)
            metrics["ce_stage2"] = float(ce2)
            metrics["ce_stage3"] = float(ce3)
            metrics["ce_stage4"] = float(ce4)

        # ── Full cascade for logits (same as grad path) ──
        for i in range(len(self.stages) - 2, -1, -1):
            stage_outputs[i] = self.feedbacks[i](
                stage_outputs[i], stage_outputs[i + 1]
            )

        h_out = self.out_norm(stage_outputs[0])
        logits = h_out @ self.embed.weight.T

        return logits, metrics

    def count_params(self) -> dict:
        """Count parameters by component, distinguishing ternary vs float."""
        counts = {}
        ternary_bytes = 0  # track ternary memory savings

        def _count(module, name):
            total = sum(v.size for _, v in tree_flatten(module.parameters()))
            counts[name] = total

        _count(self.embed, "embedding")
        for i, stage in enumerate(self.stages):
            label = f"stage{i+1}"
            if stage.is_ternary:
                label += " (ternary)"
            _count(stage, label)
        for i, reducer in enumerate(self.reducers):
            _count(reducer, f"reducer{i+1}→{i+2}")
        for i, fb in enumerate(self.feedbacks):
            label = f"feedback{i+2}→{i+1}"
            if fb.is_ternary:
                label += " (ternary)"
            _count(fb, label)
        _count(self.out_norm, "out_norm")

        counts["total"] = sum(counts.values())

        # Compute hot-path memory in bytes (ternary = 0.25 bytes/weight, float = 4)
        hot_ternary = 0  # ternary weight count
        hot_float = 0    # float weight count on hot path
        for i, stage in enumerate(self.stages):
            if stage.is_ternary:
                from ternary import _walk_ternary_modules
                for _, mod in _walk_ternary_modules(stage):
                    hot_ternary += mod.out_features * mod.in_features
            elif i == 0:  # Stage 1 is hot path even if float
                stage_params = sum(v.size for _, v in tree_flatten(stage.parameters()))
                hot_float += stage_params
        for fb in self.feedbacks:
            if fb.is_ternary:
                from ternary import _walk_ternary_modules
                for _, mod in _walk_ternary_modules(fb):
                    hot_ternary += mod.out_features * mod.in_features

        counts["hot_ternary_weights"] = hot_ternary
        counts["hot_ternary_bytes"] = hot_ternary // 4  # packed 2-bit
        counts["hot_float_bytes"] = hot_float * 4
        counts["hot_total_bytes"] = counts["hot_ternary_bytes"] + counts["hot_float_bytes"]

        return counts


# ═══════════════════════════════════════════════════════════════════
# Factory + smoke test
# ═══════════════════════════════════════════════════════════════════


def create_model(cfg: PipelineConfig | None = None) -> VSMPipeline:
    """Create a VSMPipeline with default or custom config."""
    if cfg is None:
        cfg = PipelineConfig()
    model = VSMPipeline(cfg)
    mx.eval(model.parameters())
    return model


if __name__ == "__main__":
    print("Building VSM Pipeline...")
    cfg = PipelineConfig()
    model = create_model(cfg)

    # Print architecture
    print(f"\nConfig: seq_len={cfg.seq_len}, stages={len(cfg.stages)}")
    print(f"Positions per stage: {cfg.stage_positions}")
    for i, s in enumerate(cfg.stages):
        print(f"  Stage {i+1}: {s.n_layers}L, {s.n_heads}H, d={s.d_model}, ff={s.d_ff}, pos={cfg.stage_positions[i]}")

    # Parameter count
    counts = model.count_params()
    print(f"\nParameters:")
    for name, count in counts.items():
        print(f"  {name:>20s}: {count:>10,}")

    # Forward pass test (grad-safe path)
    print(f"\nForward pass test (grad path)...")
    tokens = mx.zeros((2, cfg.seq_len), dtype=mx.int32)
    logits = model(tokens)
    mx.eval(logits)
    print(f"  Input:  {tokens.shape}")
    print(f"  Output: {logits.shape}")

    # Forward pass test (metrics path)
    print(f"\nForward pass test (metrics path)...")
    logits, metrics = model.forward_with_metrics(tokens)
    mx.eval(logits)
    print(f"  Metrics: {metrics}")
    print("\n✓ Forward pass successful")
