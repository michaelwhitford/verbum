"""
v9 — Strided Kernel Router: Expression-Tree Levels

The real v9 architecture test: strided attention that follows
expression structure, not fixed spatial windows.

Key ideas:
  1. Parse S-expression → tree of (op, arg1, arg2) nodes
  2. One shared ternary level processes ANY node (self-similar)
  3. Bottom-up: resolve leaves first, substitute results into parents
  4. Each node routes to the exact kernel independently
  5. Same weights at every tree depth (v7 wavelet principle)

This tests the central v9 thesis: can a single self-similar level,
applied recursively, learn to route expression nodes to exact
computation primitives?

For `(+ 3 (* 4 5))`:
  Level 0: node (* 4 5)  → kernel(mul, 4, 5) → 20
  Level 1: node (+ 3 20) → kernel(add, 3, 20) → 23

The model sees each node as a fixed-format triple: (op, arg1, arg2)
where args are either literal integers or kernel results from below.

License: MIT
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "v8"))
from ternary import TernaryLinear

from kernel import (
    N_OPS, OP_ADD, OP_SUB, OP_MUL, OP_NAMES,
    kernel_dispatch, decode_routing, ResultEncoder,
)


# ══════════════════════════════════════════════════════════════════════
# Expression tree representation
# ══════════════════════════════════════════════════════════════════════

# A node is (op_str, arg1, arg2) where each arg is int or another node.
ExprNode = tuple  # (op: str, arg1: int|ExprNode, arg2: int|ExprNode)

OP_STR_TO_CODE = {"+": OP_ADD, "-": OP_SUB, "*": OP_MUL}
OP_CODE_TO_STR = {v: k for k, v in OP_STR_TO_CODE.items()}
OPS = ["+", "-", "*"]


def parse_sexpr(s: str) -> Union[int, ExprNode]:
    """Parse an S-expression string into a tree.

    Examples:
        '(+ 3 4)'           → ('+', 3, 4)
        '(+ 3 (* 4 5))'     → ('+', 3, ('*', 4, 5))
        '7'                  → 7
    """
    s = s.strip()
    if not s.startswith("("):
        return int(s)

    # Strip outer parens
    assert s.endswith(")"), f"Malformed: {s}"
    inner = s[1:-1].strip()

    # Extract operator (first token)
    space_idx = inner.index(" ")
    op = inner[:space_idx]

    # Parse the two arguments
    rest = inner[space_idx + 1:].strip()
    arg1, rest = _parse_one_arg(rest)
    rest = rest.strip()
    arg2, rest = _parse_one_arg(rest)

    return (op, arg1, arg2)


def _parse_one_arg(s: str) -> tuple[Union[int, ExprNode], str]:
    """Parse one argument from the front of s, return (parsed, remaining)."""
    s = s.strip()
    if s.startswith("("):
        # Find matching close paren
        depth = 0
        for i, c in enumerate(s):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return parse_sexpr(s[:i + 1]), s[i + 1:]
        raise ValueError(f"Unmatched paren: {s}")
    else:
        # Integer — read until space or end
        end = len(s)
        for i, c in enumerate(s):
            if c == " " or c == ")":
                end = i
                break
        return int(s[:end]), s[end:]


def eval_tree(node: Union[int, ExprNode]) -> int:
    """Evaluate an expression tree to get the ground truth result."""
    if isinstance(node, int):
        return node
    op, a1, a2 = node
    v1 = eval_tree(a1)
    v2 = eval_tree(a2)
    if op == "+":
        return v1 + v2
    elif op == "-":
        return v1 - v2
    elif op == "*":
        return v1 * v2
    else:
        raise ValueError(f"Unknown op: {op}")


def tree_depth(node: Union[int, ExprNode]) -> int:
    """Depth of the expression tree (0 for literals, 1 for flat, etc.)."""
    if isinstance(node, int):
        return 0
    _, a1, a2 = node
    return 1 + max(tree_depth(a1), tree_depth(a2))


def tree_to_str(node: Union[int, ExprNode]) -> str:
    """Convert tree back to S-expression string."""
    if isinstance(node, int):
        return str(node)
    op, a1, a2 = node
    return f"({op} {tree_to_str(a1)} {tree_to_str(a2)})"


def linearize_bottomup(node: Union[int, ExprNode]) -> list[tuple[str, int, int, int]]:
    """Linearize tree into bottom-up evaluation order.

    Returns list of (op_str, arg1_val, arg2_val, result_val) tuples,
    ordered so that inner expressions come first.

    Each tuple represents one kernel dispatch. For training, each
    is a routing supervision target.
    """
    if isinstance(node, int):
        return []

    op, a1, a2 = node
    steps = []

    # Recurse into sub-expressions first (bottom-up)
    steps.extend(linearize_bottomup(a1))
    steps.extend(linearize_bottomup(a2))

    # Evaluate this node
    v1 = eval_tree(a1)
    v2 = eval_tree(a2)
    result = eval_tree(node)
    steps.append((op, v1, v2, result))

    return steps


# ══════════════════════════════════════════════════════════════════════
# Node encoding: (op, arg1, arg2) → fixed-size tensor
# ══════════════════════════════════════════════════════════════════════

# Each node is encoded as a fixed-size vector representing:
#   - The operator (one-hot or embedded)
#   - arg1 value (integer, possibly from kernel result below)
#   - arg2 value (integer, possibly from kernel result below)
#
# The encoding is simple: we embed the triple into d_model space
# using separate embeddings for op, and a shared value embedding
# for the two arguments.


@dataclass
class StridedConfig:
    """Config for the strided kernel router."""
    d_model: int = 64              # representation dimension
    n_ops: int = N_OPS             # 3: add, sub, mul
    max_val: int = 100             # operand range [0, max_val) for routing
    val_embed_range: int = 200     # embedding range for values [-100, 100)
    n_mix_layers: int = 2          # ternary mixing layers per node
    result_buckets: int = 512      # result encoder buckets
    stride: int = 4                # tokens per window at level 0
    max_len: int = 24              # max tokenized expression length


# ══════════════════════════════════════════════════════════════════════
# Character tokenizer (shared with kernel_model.py)
# ══════════════════════════════════════════════════════════════════════

CHAR_VOCAB = {
    "<pad>": 0, "(": 1, ")": 2, "+": 3, "-": 4, "*": 5, " ": 6,
    "0": 7, "1": 8, "2": 9, "3": 10, "4": 11, "5": 12,
    "6": 13, "7": 14, "8": 15, "9": 16,
}
CHAR_VOCAB_SIZE = len(CHAR_VOCAB)  # 17
ID_TO_CHAR = {v: k for k, v in CHAR_VOCAB.items()}


def tokenize_expr(expr: str, max_len: int = 24) -> list[int]:
    """Tokenize expression to char IDs, pad to max_len."""
    ids = [CHAR_VOCAB.get(c, 0) for c in expr][:max_len]
    return ids + [0] * (max_len - len(ids))


class StridedKernelRouter(nn.Module):
    """Token-based strided kernel router with self-similar levels.

    Architecture:
      1. Tokenize expression → char embeddings + positional
      2. Split into stride-sized windows
      3. Per-window: ternary self-attention (mix within window)
      4. Pool each window → one vector per window
      5. Next level: treat window summaries as the new sequence
      6. Repeat until one vector remains
      7. Route to kernel

    Self-similar: same ternary weights at every level. The stride
    window operates the same way whether processing raw tokens at
    level 0 or window summaries at level 1+.

    For flat `(+ 3 4)`: ~7 tokens, stride=4 → 2 windows → 1 level
    For nested `(+ 3 (* 4 5))`: ~13 tokens, stride=4 → 4 windows → 2 levels
    """

    def __init__(self, config: StridedConfig | None = None):
        super().__init__()
        if config is None:
            config = StridedConfig()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(CHAR_VOCAB_SIZE, config.d_model)
        self.pos_embed = nn.Embedding(config.max_len, config.d_model)

        # SHARED ternary mixing layers — same at every stride level
        # This is the wavelet: one function applied at every scale
        self.mix_layers = []
        for _ in range(config.n_mix_layers):
            self.mix_layers.append(TernaryLinear(config.d_model, config.d_model, pre_norm=True))

        # Within-window position encoding (stride positions)
        self.window_pos = nn.Embedding(config.stride, config.d_model)

        # Routing projection: d_model → n_ops + 2*max_val
        route_dim = config.n_ops + 2 * config.max_val
        route_dim_padded = ((route_dim + 15) // 16) * 16
        self._route_dim = route_dim
        self._route_dim_padded = route_dim_padded
        self.route_proj = TernaryLinear(config.d_model, route_dim_padded, pre_norm=True)

        # Result encoder
        self.result_encoder = ResultEncoder(
            n_buckets=config.result_buckets,
            d_model=config.d_model,
        )

    def _stride_reduce(self, x: mx.array) -> mx.array:
        """Apply one stride level: split into windows, attend, pool.

        Uses learned within-window attention to preserve positional info
        before pooling. The attention lets the model weight which tokens
        in each window matter for routing.

        Args:
            x: (B, T, d_model) — sequence of vectors

        Returns:
            (B, T//stride, d_model) — one summary vector per window
            If T <= stride, returns (B, 1, d_model).
        """
        B, T, D = x.shape
        stride = self.config.stride

        if T <= 1:
            return x  # already a single vector

        if T <= stride:
            # Final level: attend over all remaining positions then pool
            # Add window positional encoding (truncated to T)
            win_pos_ids = mx.arange(T)
            win_pos_emb = self.window_pos(mx.minimum(win_pos_ids, stride - 1))
            x = x + win_pos_emb

            # Apply mix layers to each position, then attention-pool
            for layer in self.mix_layers:
                # Apply to each position independently
                x_flat = x.reshape(B * T, D)
                mixed = x_flat + layer(x_flat)
                x = mixed.reshape(B, T, D)

            # Attention pooling: learned query attends over positions
            # Use the mix layers' output to compute attention weights
            # Simple: sum of activations as score, softmax, weighted sum
            scores = x.sum(axis=-1, keepdims=True)  # (B, T, 1)
            attn = mx.softmax(scores, axis=1)  # (B, T, 1)
            pooled = (x * attn).sum(axis=1, keepdims=True)  # (B, 1, D)
            return pooled

        # Pad T to multiple of stride
        pad_len = (stride - T % stride) % stride
        if pad_len > 0:
            padding = mx.zeros((B, pad_len, D))
            x = mx.concatenate([x, padding], axis=1)
            T = T + pad_len

        n_windows = T // stride

        # Reshape into windows: (B, n_windows, stride, D)
        windows = x.reshape(B, n_windows, stride, D)

        # Add within-window positional encoding
        win_pos_ids = mx.arange(stride)
        win_pos_emb = self.window_pos(win_pos_ids)  # (stride, D)
        windows = windows + win_pos_emb

        # Flatten: (B*n_windows, stride, D)
        flat = windows.reshape(B * n_windows, stride, D)

        # Apply shared ternary mix layers to each position in each window
        for layer in self.mix_layers:
            flat_2d = flat.reshape(B * n_windows * stride, D)
            mixed = flat_2d + layer(flat_2d)
            flat = mixed.reshape(B * n_windows, stride, D)

        # Attention pooling within each window
        scores = flat.sum(axis=-1, keepdims=True)  # (B*nw, stride, 1)
        attn = mx.softmax(scores, axis=1)           # (B*nw, stride, 1)
        pooled = (flat * attn).sum(axis=1)           # (B*nw, D)

        return pooled.reshape(B, n_windows, D)

    def forward_routing(self, tokens: mx.array) -> mx.array:
        """Full forward: tokens → multi-level stride reduction → routing logits.

        Args:
            tokens: (B, max_len) int

        Returns:
            routing_logits: (B, n_ops + 2*max_val)
        """
        B, T = tokens.shape
        config = self.config

        # Embed tokens + positions
        pos_ids = mx.arange(T)
        x = self.embed(tokens) + self.pos_embed(pos_ids)  # (B, T, d_model)

        # Multi-level stride reduction until we have 1 vector
        max_levels = 5  # safety limit
        for _ in range(max_levels):
            x = self._stride_reduce(x)
            if x.shape[1] <= 1:
                break

        # x is now (B, 1, d_model) — squeeze
        x = x.squeeze(1)  # (B, d_model)

        # Route to kernel logits
        route_logits = self.route_proj(x)[:, :self._route_dim]
        return route_logits

    def __call__(self, tokens: mx.array):
        """Full forward: tokens → routing → kernel → result."""
        route_logits = self.forward_routing(tokens)
        pred_op, pred_a1, pred_a2 = decode_routing(
            route_logits, self.config.n_ops, self.config.max_val,
        )
        pred_result = kernel_dispatch(pred_op, pred_a1, pred_a2)
        return route_logits, pred_op, pred_a1, pred_a2, pred_result

    def count_params(self) -> dict[str, int]:
        """Count parameters by type."""
        from mlx.utils import tree_flatten as tf
        total = 0
        ternary = 0
        continuous = 0
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
    print("  v9 — Strided Kernel Router Smoke Test")
    print("=" * 60)

    # Test expression parsing
    test_exprs = [
        "(+ 3 4)",
        "(* 12 5)",
        "(+ 3 (* 4 5))",
        "(- (* 3 4) 7)",
        "(+ 1 (* 2 (- 7 3)))",
    ]

    print("\nParsing and evaluation:")
    for s in test_exprs:
        tree = parse_sexpr(s)
        result = eval_tree(tree)
        depth = tree_depth(tree)
        roundtrip = tree_to_str(tree)
        steps = linearize_bottomup(tree)
        print(f"  {s:30s} = {result:5d}  depth={depth}  "
              f"steps={len(steps)}  roundtrip={roundtrip}")
        for step in steps:
            op, a1, a2, r = step
            print(f"    → ({op} {a1} {a2}) = {r}")

    # Test token-based model
    config = StridedConfig(d_model=64, max_val=100, max_len=24)
    model = StridedKernelRouter(config)

    print(f"\nModel parameters:")
    params = model.count_params()
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    # Test token-based forward
    print(f"\nToken-based forward test:")
    for s in test_exprs[:3]:
        tree = parse_sexpr(s)
        gt = eval_tree(tree)
        toks = mx.array([tokenize_expr(s, max_len=24)])
        route, pred_op, pred_a1, pred_a2, pred_result = model(toks)
        mx.eval(pred_op, pred_a1, pred_a2, pred_result)
        print(f"  {s:30s} gt={gt:5d}  "
              f"pred_op={pred_op[0].item()} a1={pred_a1[0].item()} "
              f"a2={pred_a2[0].item()} result={pred_result[0].item()}")

    # Test stride reduction
    print(f"\nStride reduction test:")
    for s in test_exprs:
        toks = tokenize_expr(s, max_len=24)
        n_real = sum(1 for t in toks if t != 0)
        stride = config.stride
        n_windows_l0 = (n_real + stride - 1) // stride
        n_windows_l1 = (n_windows_l0 + stride - 1) // stride if n_windows_l0 > 1 else 1
        print(f"  {s:30s}  tokens={n_real}  "
              f"L0_windows={n_windows_l0}  L1_windows={n_windows_l1}")

    print(f"\n{'=' * 60}")
    print(f"  ✓ Strided kernel smoke test passed")
    print(f"{'=' * 60}")
