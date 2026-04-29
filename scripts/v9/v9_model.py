"""
v9 — Integrated Prototype: Ascending Arm + Type/Parse/Apply Kernel

The full pipeline:
  tokens → REDUCE (ascending arm, self-similar strided attention)
         → TYPE   (classify each reduced unit's semantic type)
         → PARSE  (query-based routing: extract op, arg1, arg2)
         → APPLY  (type-checked kernel dispatch → exact computation)

Three separate concerns, cleanly separated:
  - Ascending arm: builds representation (proven by v7)
  - Type/Parse heads: routing mechanism (proven by query prototype)
  - Kernel: exact computation (proven by arithmetic prototype)

Type system (arithmetic, expandable):
  INT   — integer value (3, 42, -7)
  OP    — binary operator (+ - *)
  EXPR  — unevaluated expression → needs reduction
  ERROR — type mismatch or invalid

Apply rules:
  apply(OP, INT, INT) → dispatch to arithmetic → INT result
  apply(_, _, _)      → ERROR (type mismatch)

Self-similar: the ascending arm uses SHARED weights at every
stride level. Same ternary attention processes level 0 (raw tokens)
and level N (reduced summaries). This is the wavelet.

License: MIT
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "v8"))
from ternary import TernaryLinear


# ══════════════════════════════════════════════════════════════════════
# Type system
# ══════════════════════════════════════════════════════════════════════

TYPE_INT = 0    # integer value
TYPE_OP = 1     # binary operator
TYPE_EXPR = 2   # unevaluated expression
TYPE_ERROR = 3  # type error / invalid
N_TYPES = 4

TYPE_NAMES = {TYPE_INT: "Int", TYPE_OP: "Op", TYPE_EXPR: "Expr", TYPE_ERROR: "Err"}

# Arithmetic ops
OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
N_OPS = 3
OP_NAMES = {OP_ADD: "+", OP_SUB: "-", OP_MUL: "*"}


# ══════════════════════════════════════════════════════════════════════
# Kernel — type-checked exact computation
# ══════════════════════════════════════════════════════════════════════


def kernel_type_check(op_type: mx.array, a1_type: mx.array, a2_type: mx.array) -> mx.array:
    """Check if types are valid for application.

    Valid: op_type == OP, a1_type == INT, a2_type == INT
    Returns: (B,) bool tensor — True if types check.
    """
    valid = (
        (op_type == TYPE_OP) &
        (a1_type == TYPE_INT) &
        (a2_type == TYPE_INT)
    )
    return valid


def kernel_apply(op: mx.array, arg1: mx.array, arg2: mx.array) -> mx.array:
    """Exact arithmetic dispatch. Same as before but now conceptually
    this is the APPLY primitive — β-reduction for arithmetic.

    apply(+, 3, 4) ≡ β-reduce((λx.λy.x+y) 3 4) → 7
    """
    r_add = arg1 + arg2
    r_sub = arg1 - arg2
    r_mul = arg1 * arg2
    return mx.where(op == OP_ADD, r_add,
           mx.where(op == OP_SUB, r_sub, r_mul))


def kernel_dispatch(
    op: mx.array, arg1: mx.array, arg2: mx.array,
    op_type: mx.array, a1_type: mx.array, a2_type: mx.array,
) -> tuple[mx.array, mx.array]:
    """Type-checked kernel dispatch.

    Returns:
        result:      (B,) int — computation result (0 if type error)
        result_type: (B,) int — TYPE_INT if valid, TYPE_ERROR if mismatch
    """
    valid = kernel_type_check(op_type, a1_type, a2_type)
    result = kernel_apply(op, arg1, arg2)

    # Mask invalid results
    result = mx.where(valid, result, mx.zeros_like(result))
    result_type = mx.where(valid,
                           mx.full(valid.shape, TYPE_INT, dtype=mx.int32),
                           mx.full(valid.shape, TYPE_ERROR, dtype=mx.int32))

    return result, result_type


# ══════════════════════════════════════════════════════════════════════
# Character tokenizer
# ══════════════════════════════════════════════════════════════════════

CHAR_VOCAB = {
    "<pad>": 0, "(": 1, ")": 2, "+": 3, "-": 4, "*": 5, " ": 6,
    "0": 7, "1": 8, "2": 9, "3": 10, "4": 11, "5": 12,
    "6": 13, "7": 14, "8": 15, "9": 16,
}
CHAR_VOCAB_SIZE = len(CHAR_VOCAB)


def tokenize_expr(expr: str, max_len: int = 24) -> list[int]:
    ids = [CHAR_VOCAB.get(c, 0) for c in expr][:max_len]
    return ids + [0] * (max_len - len(ids))


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════


@dataclass
class V9Config:
    d_model: int = 64
    n_heads: int = 4           # attention heads in ascending arm
    n_ascending_levels: int = 3  # stride levels in ascending arm
    stride: int = 4            # tokens per window
    n_mix_layers: int = 1      # ternary mix layers per ascending level
    n_ops: int = N_OPS
    n_types: int = N_TYPES
    max_val: int = 100         # routing logit range for args
    max_len: int = 24          # max tokenized expression length

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


# ══════════════════════════════════════════════════════════════════════
# Ascending Arm — self-similar strided ternary attention
# ══════════════════════════════════════════════════════════════════════


class TernaryAttention(nn.Module):
    """Multi-head self-attention with ternary Q/K/V/O projections.

    This is the core operation of the ascending arm. Applied within
    each stride window. The same instance is reused at every level
    (self-similar / wavelet).
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Q/K/V projections — ternary routing topology
        self.q_proj = TernaryLinear(d_model, d_model, pre_norm=True)
        self.k_proj = TernaryLinear(d_model, d_model, pre_norm=False)
        self.v_proj = TernaryLinear(d_model, d_model, pre_norm=False)
        self.o_proj = TernaryLinear(d_model, d_model, pre_norm=False)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        """Self-attention within a sequence.

        Args:
            x:    (B, T, d_model)
            mask: (B, T) float — 1.0 for real tokens, 0.0 for padding

        Returns:
            (B, T, d_model) — attended output
        """
        B, T, D = x.shape
        H = self.n_heads
        dh = self.d_head

        q = self.q_proj(x).reshape(B, T, H, dh).transpose(0, 2, 1, 3)  # (B, H, T, dh)
        k = self.k_proj(x).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, H, dh).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, H, T, T)

        if mask is not None:
            # mask: (B, T) → (B, 1, 1, T) for key masking
            mask_4d = mask[:, None, None, :]
            scores = mx.where(mask_4d > 0, scores, mx.array(-1e9))

        attn = mx.softmax(scores, axis=-1)  # (B, H, T, T)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)  # (B, T, D)

        return self.o_proj(out)


class AscendingArm(nn.Module):
    """Multi-level strided reduction with self-similar ternary attention.

    Each level:
      1. Split sequence into stride-sized windows
      2. Add within-window positional encoding
      3. Self-attend within each window (shared TernaryAttention)
      4. Pool each window to one vector (attention-weighted)
      5. Output becomes the sequence for the next level

    SHARED weights across all levels — the wavelet principle.
    Level 0 processes raw token embeddings. Level N processes
    level N-1 summaries. Same operation at every scale.
    """

    def __init__(self, config: V9Config):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(CHAR_VOCAB_SIZE, config.d_model)
        self.pos_embed = nn.Embedding(config.max_len, config.d_model)

        # SHARED attention — self-similar across all levels
        self.shared_attn = TernaryAttention(config.d_model, config.n_heads)

        # SHARED mix layer — additional ternary processing after attention
        self.shared_mix = TernaryLinear(config.d_model, config.d_model, pre_norm=True)

        # Within-window position encoding (reused at every level)
        self.window_pos = nn.Embedding(config.stride, config.d_model)

        # Pool query — learned vector that attends over window to produce summary
        self.pool_query = mx.random.normal((1, 1, config.d_model)) * 0.02

    def _reduce_one_level(self, x: mx.array) -> mx.array:
        """One level of strided reduction.

        Args:
            x: (B, T, d_model)

        Returns:
            (B, ceil(T/stride), d_model) — reduced sequence
        """
        B, T, D = x.shape
        stride = self.config.stride

        if T <= 1:
            return x

        # Pad to multiple of stride
        pad_len = (stride - T % stride) % stride
        if pad_len > 0:
            x = mx.concatenate([x, mx.zeros((B, pad_len, D))], axis=1)
            T = T + pad_len

        n_windows = T // stride

        # Reshape into windows: (B * n_windows, stride, D)
        windows = x.reshape(B, n_windows, stride, D)
        win_pos = self.window_pos(mx.arange(stride))  # (stride, D)
        windows = windows + win_pos  # add within-window position
        flat = windows.reshape(B * n_windows, stride, D)

        # Self-attend within each window (shared weights)
        attended = flat + self.shared_attn(flat)  # residual

        # Mix
        flat_2d = attended.reshape(B * n_windows * stride, D)
        mixed = flat_2d + self.shared_mix(flat_2d)
        attended = mixed.reshape(B * n_windows, stride, D)

        # Attention-weighted pooling: pool_query attends over window
        pool_q = mx.broadcast_to(self.pool_query, (B * n_windows, 1, D))
        pool_scores = (pool_q @ attended.transpose(0, 2, 1)) * (D ** -0.5)
        pool_attn = mx.softmax(pool_scores, axis=-1)  # (B*nw, 1, stride)
        pooled = (pool_attn @ attended).squeeze(1)     # (B*nw, D)

        return pooled.reshape(B, n_windows, D)

    def __call__(self, tokens: mx.array) -> mx.array:
        """Full ascending arm: tokens → multi-level reduction → multi-scale output.

        Returns the CONCATENATION of all level outputs, giving the
        parse queries a rich multi-scale sequence to attend over.
        Level 0 outputs capture local patterns (digits, operators).
        Higher levels capture broader structure (sub-expressions).

        Args:
            tokens: (B, max_len) int

        Returns:
            (B, T_multi_scale, d_model) — multi-scale representation
        """
        B, T = tokens.shape

        # Embed
        pos_ids = mx.arange(T)
        x = self.embed(tokens) + self.pos_embed(pos_ids)  # (B, T, D)

        # Collect outputs from each level for multi-scale representation
        level_outputs = []

        for level in range(self.config.n_ascending_levels):
            if x.shape[1] <= 1:
                level_outputs.append(x)
                break
            x = self._reduce_one_level(x)
            level_outputs.append(x)  # (B, T_level, D)

        # Concatenate all levels: gives queries a rich multi-scale view
        # Level 0: ~6 windows (local patterns)
        # Level 1: ~2 windows (medium structure)
        # Level 2: 1 window  (global summary)
        # Total: ~9 positions for queries to attend over
        multi_scale = mx.concatenate(level_outputs, axis=1)  # (B, sum_T, D)

        return multi_scale


# ══════════════════════════════════════════════════════════════════════
# Type / Parse / Apply Heads
# ══════════════════════════════════════════════════════════════════════


class TypeParseApply(nn.Module):
    """The three Montague heads: type, parse, apply.

    TYPE:  Classifies the semantic type of the full expression.
           Output: type logits (INT, OP, EXPR, ERROR)

    PARSE: Query-based routing that extracts (op, arg1, arg2) from
           the reduced representation. Three learned queries attend
           over the ascending arm output.

    APPLY: Type-checked dispatch to the exact kernel.
           If types match → kernel computes exactly.
           If types don't match → ERROR.
    """

    def __init__(self, config: V9Config):
        super().__init__()
        self.config = config

        # ── TYPE head ──
        # Classifies expression type from pooled representation
        type_dim_padded = ((config.n_types + 15) // 16) * 16
        self.type_proj = TernaryLinear(config.d_model, type_dim_padded, pre_norm=True)
        self._type_dim = config.n_types

        # ── PARSE head ── (query-based routing, proven to work)
        # Three learned queries for op, arg1, arg2
        self.op_query = mx.random.normal((1, 1, config.d_model)) * 0.02
        self.arg1_query = mx.random.normal((1, 1, config.d_model)) * 0.02
        self.arg2_query = mx.random.normal((1, 1, config.d_model)) * 0.02

        # Per-head mix layers
        self.op_mix = TernaryLinear(config.d_model, config.d_model, pre_norm=True)
        self.arg1_mix = TernaryLinear(config.d_model, config.d_model, pre_norm=True)
        self.arg2_mix = TernaryLinear(config.d_model, config.d_model, pre_norm=True)

        # Routing projections
        op_dim = ((config.n_ops + 15) // 16) * 16
        arg_dim = ((config.max_val + 15) // 16) * 16
        self.op_proj = TernaryLinear(config.d_model, op_dim, pre_norm=True)
        self.arg1_proj = TernaryLinear(config.d_model, arg_dim, pre_norm=True)
        self.arg2_proj = TernaryLinear(config.d_model, arg_dim, pre_norm=True)
        self._op_dim = op_dim
        self._arg_dim = arg_dim

        # Type projections for arg1 and arg2 (what type did the parse head find?)
        type_dim_padded2 = ((config.n_types + 15) // 16) * 16
        self.arg1_type_proj = TernaryLinear(config.d_model, type_dim_padded2, pre_norm=True)
        self.arg2_type_proj = TernaryLinear(config.d_model, type_dim_padded2, pre_norm=True)

    def forward(
        self, reduced: mx.array,
    ) -> dict[str, mx.array]:
        """Full type/parse/apply pipeline.

        Args:
            reduced: (B, T_reduced, d_model) — ascending arm output

        Returns dict with:
            expr_type_logits: (B, n_types)
            op_logits:        (B, n_ops)
            arg1_logits:      (B, max_val)
            arg2_logits:      (B, max_val)
            arg1_type_logits: (B, n_types)
            arg2_type_logits: (B, n_types)
            pred_type:        (B,) int
            pred_op:          (B,) int
            pred_arg1:        (B,) int
            pred_arg2:        (B,) int
            pred_a1_type:     (B,) int
            pred_a2_type:     (B,) int
            pred_result:      (B,) int
            pred_result_type: (B,) int
        """
        B = reduced.shape[0]
        T = reduced.shape[1]
        D = self.config.d_model
        scale = D ** -0.5

        # ── TYPE: classify expression ──
        # Pool the reduced representation
        expr_pooled = reduced.mean(axis=1)  # (B, D)
        type_logits = self.type_proj(expr_pooled)[:, :self._type_dim]  # (B, n_types)
        pred_type = mx.argmax(type_logits, axis=-1).astype(mx.int32)

        # ── PARSE: extract op, arg1, arg2 via query attention ──
        reduced_T = reduced.transpose(0, 2, 1)  # (B, D, T)

        def _query_attend(query, mix_layer):
            q = mx.broadcast_to(query, (B, 1, D))
            scores = (q @ reduced_T) * scale  # (B, 1, T)
            attn = mx.softmax(scores, axis=-1)
            attended = (attn @ reduced).squeeze(1)  # (B, D)
            return attended + mix_layer(attended)  # (B, D)

        op_repr = _query_attend(self.op_query, self.op_mix)
        arg1_repr = _query_attend(self.arg1_query, self.arg1_mix)
        arg2_repr = _query_attend(self.arg2_query, self.arg2_mix)

        # Project to routing logits
        op_logits = self.op_proj(op_repr)[:, :self.config.n_ops]
        arg1_logits = self.arg1_proj(arg1_repr)[:, :self.config.max_val]
        arg2_logits = self.arg2_proj(arg2_repr)[:, :self.config.max_val]

        pred_op = mx.argmax(op_logits, axis=-1).astype(mx.int32)
        pred_arg1 = mx.argmax(arg1_logits, axis=-1).astype(mx.int32)
        pred_arg2 = mx.argmax(arg2_logits, axis=-1).astype(mx.int32)

        # Type-classify each argument
        a1_type_logits = self.arg1_type_proj(arg1_repr)[:, :self.config.n_types]
        a2_type_logits = self.arg2_type_proj(arg2_repr)[:, :self.config.n_types]
        pred_a1_type = mx.argmax(a1_type_logits, axis=-1).astype(mx.int32)
        pred_a2_type = mx.argmax(a2_type_logits, axis=-1).astype(mx.int32)

        # ── APPLY: type-checked kernel dispatch ──
        # Op is always TYPE_OP for our expressions
        op_type = mx.full((B,), TYPE_OP, dtype=mx.int32)
        pred_result, pred_result_type = kernel_dispatch(
            pred_op, pred_arg1, pred_arg2,
            op_type, pred_a1_type, pred_a2_type,
        )

        return {
            "expr_type_logits": type_logits,
            "op_logits": op_logits,
            "arg1_logits": arg1_logits,
            "arg2_logits": arg2_logits,
            "arg1_type_logits": a1_type_logits,
            "arg2_type_logits": a2_type_logits,
            "pred_type": pred_type,
            "pred_op": pred_op,
            "pred_arg1": pred_arg1,
            "pred_arg2": pred_arg2,
            "pred_a1_type": pred_a1_type,
            "pred_a2_type": pred_a2_type,
            "pred_result": pred_result,
            "pred_result_type": pred_result_type,
        }


# ══════════════════════════════════════════════════════════════════════
# Full V9 Model
# ══════════════════════════════════════════════════════════════════════


class V9Model(nn.Module):
    """tokens → REDUCE → TYPE → PARSE → APPLY → result"""

    def __init__(self, config: V9Config | None = None):
        super().__init__()
        if config is None:
            config = V9Config()
        self.config = config
        self.ascending = AscendingArm(config)
        self.tpa = TypeParseApply(config)

    def __call__(self, tokens: mx.array) -> dict[str, mx.array]:
        B, T = tokens.shape

        # Get raw token embeddings (skip connection for gradient flow)
        pos_ids = mx.arange(T)
        raw_embed = self.ascending.embed(tokens) + self.ascending.pos_embed(pos_ids)

        # Get multi-scale ascending arm output
        reduced = self.ascending(tokens)

        # Concatenate: raw embeddings + multi-scale reduction
        # This gives queries BOTH the raw positional token info AND
        # the hierarchically reduced structure. The gradient flows
        # through the raw path even if the ascending arm's ternary
        # topology isn't useful yet.
        combined = mx.concatenate([raw_embed, reduced], axis=1)

        return self.tpa.forward(combined)

    def count_params(self) -> dict[str, int]:
        from mlx.utils import tree_flatten as tf
        total = ternary = continuous = 0
        for name, p in tf(self.parameters()):
            n = p.size
            total += n
            if p.dtype == mx.uint32:
                ternary += n * 16
            elif p.dtype == mx.uint8:
                ternary += n * 4
            else:
                continuous += n
        return {"total": total, "ternary_logical": ternary, "continuous": continuous}


# ══════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  v9 — Integrated Model Smoke Test")
    print("=" * 60)

    config = V9Config(d_model=64, max_val=10, max_len=24)
    model = V9Model(config)

    params = model.count_params()
    print(f"\nParameters:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    exprs = ["(+ 3 4)", "(* 7 2)", "(+ 3 (* 4 5))", "(- (* 3 4) 7)"]
    tokens = mx.array([tokenize_expr(e, 24) for e in exprs])

    # Test ascending arm
    reduced = model.ascending(tokens)
    mx.eval(reduced)
    print(f"\nAscending arm:")
    print(f"  Input:  {tokens.shape}")
    print(f"  Output: {reduced.shape}")

    # Test full forward
    out = model(tokens)
    for k, v in out.items():
        mx.eval(v)

    print(f"\nFull forward:")
    for i, expr in enumerate(exprs):
        print(f"  {expr:25s} → type={TYPE_NAMES.get(out['pred_type'][i].item(), '?')}"
              f"  op={OP_NAMES.get(out['pred_op'][i].item(), '?')}"
              f"  a1={out['pred_arg1'][i].item()}"
              f"(t={TYPE_NAMES.get(out['pred_a1_type'][i].item(), '?')})"
              f"  a2={out['pred_arg2'][i].item()}"
              f"(t={TYPE_NAMES.get(out['pred_a2_type'][i].item(), '?')})"
              f"  result={out['pred_result'][i].item()}"
              f"(t={TYPE_NAMES.get(out['pred_result_type'][i].item(), '?')})")

    print(f"\n{'=' * 60}")
    print(f"  ✓ Integrated model smoke test passed")
    print(f"{'=' * 60}")
