"""
v9 — Kernel Router Model

Minimal model to test whether ternary evolution can find routing
from token embeddings to exact kernel primitives.

Architecture:
    tokens → TernaryEmbedding → positional → TernaryLinear (mix) →
    pool across sequence → TernaryLinear (route) → kernel decode →
    kernel dispatch (exact) → ResultEncoder → output projection

The model is deliberately tiny. We're testing a concept, not
training a language model. The question is:

    Can ternary evolution discover routing topology that maps
    "(+ 3 4)" → kernel(add, 3, 4) → 7?

Vocab is character-level: digits 0-9, operators +-*, parens, space.
Expressions are fixed-format: (op arg1 arg2).

License: MIT
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "v8"))
from ternary import TernaryLinear, TernaryEmbedding

from kernel import (
    N_OPS,
    kernel_forward,
    ResultEncoder,
)


# ══════════════════════════════════════════════════════════════════════
# Character-level tokenizer for arithmetic expressions
# ══════════════════════════════════════════════════════════════════════

# Vocab: PAD=0, then characters. Keep it tiny.
CHAR_VOCAB = {
    "<pad>": 0,
    "(": 1,
    ")": 2,
    "+": 3,
    "-": 4,
    "*": 5,
    " ": 6,
    "0": 7,
    "1": 8,
    "2": 9,
    "3": 10,
    "4": 11,
    "5": 12,
    "6": 13,
    "7": 14,
    "8": 15,
    "9": 16,
}
VOCAB_SIZE = len(CHAR_VOCAB)  # 17
ID_TO_CHAR = {v: k for k, v in CHAR_VOCAB.items()}


def tokenize_expr(expr: str, max_len: int = 16) -> list[int]:
    """Tokenize an arithmetic expression to character IDs.

    Pads to max_len with 0s. Unknown chars map to PAD.
    """
    ids = [CHAR_VOCAB.get(c, 0) for c in expr]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


def detokenize(ids: list[int]) -> str:
    """Convert token IDs back to string."""
    return "".join(ID_TO_CHAR.get(i, "?") for i in ids if i != 0)


# ══════════════════════════════════════════════════════════════════════
# Model configuration
# ══════════════════════════════════════════════════════════════════════


@dataclass
class KernelRouterConfig:
    """Config for the minimal kernel routing model."""
    vocab_size: int = VOCAB_SIZE   # 17 characters
    max_len: int = 16              # max expression length
    d_model: int = 64              # embedding dimension (must be % 16 == 0)
    d_route: int = 64              # routing projection dim (must be % 16 == 0)
    n_ops: int = N_OPS             # 3: add, sub, mul
    max_val: int = 100             # operands in [0, 100)
    result_buckets: int = 512      # result embedding size
    n_mix_layers: int = 2          # ternary mixing layers before routing
    use_positional_routing: bool = True  # use per-position routing heads


# ══════════════════════════════════════════════════════════════════════
# Kernel Router Model
# ══════════════════════════════════════════════════════════════════════


class KernelRouter(nn.Module):
    """Minimal model: embed expression → ternary route → exact kernel.

    The ternary layers learn to:
    1. Mix information across token positions (which token is the op?
       which tokens form arg1? arg2?)
    2. Project the mixed representation to routing logits that the
       kernel can decode into (op, arg1, arg2)

    The kernel then executes the operation exactly.

    Training: ternary topology evolves via tournament selection.
    Continuous params (gamma, embeddings, norms) train via Adam.
    """

    def __init__(self, config: KernelRouterConfig | None = None):
        super().__init__()
        if config is None:
            config = KernelRouterConfig()
        self.config = config

        # Token embedding (standard float — small vocab, not worth ternary)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embedding (learned, small)
        self.pos_embed = nn.Embedding(config.max_len, config.d_model)

        # Ternary mixing layers: combine information across the pooled representation
        # These are the layers whose topology evolves to find the routing
        self.mix_layers = []
        for i in range(config.n_mix_layers):
            self.mix_layers.append(TernaryLinear(config.d_model, config.d_model, pre_norm=True))

        # Routing projection: d_model → (n_ops + 2*max_val)
        route_dim = config.n_ops + 2 * config.max_val
        # Route dim needs to be multiple of 16 for TernaryLinear
        # Pad if necessary
        self._route_dim = route_dim
        self._route_dim_padded = ((route_dim + 15) // 16) * 16

        if config.use_positional_routing:
            # Three separate routing heads — one for each component:
            #   op_head:   reads from position 1 (the operator)
            #   arg1_head: reads from concat of positions 3+ (first number)
            #   arg2_head: reads from later positions (second number)
            # Each head is a ternary linear that projects from the token
            # representation at specific positions.
            #
            # But we don't hardcode positions — instead we use 3 learned
            # "query" vectors that attend over the sequence to find what
            # they need. Like a 3-head cross-attention with learned queries.
            self.op_query = mx.random.normal((1, config.d_model)) * 0.02
            self.arg1_query = mx.random.normal((1, config.d_model)) * 0.02
            self.arg2_query = mx.random.normal((1, config.d_model)) * 0.02

            # Projection heads: each takes d_model → its logit space
            n_op_logits = ((config.n_ops + 15) // 16) * 16
            n_arg_logits = ((config.max_val + 15) // 16) * 16
            self.op_proj = TernaryLinear(config.d_model, n_op_logits, pre_norm=True)
            self.arg1_proj = TernaryLinear(config.d_model, n_arg_logits, pre_norm=True)
            self.arg2_proj = TernaryLinear(config.d_model, n_arg_logits, pre_norm=True)
            self._n_op_logits = n_op_logits
            self._n_arg_logits = n_arg_logits
        else:
            self.route_proj = TernaryLinear(config.d_model, self._route_dim_padded, pre_norm=True)

        # Result encoder: kernel output → d_model vector
        self.result_encoder = ResultEncoder(
            n_buckets=config.result_buckets,
            d_model=config.d_model,
        )

        # Output projection: d_model → vocab (for next-token prediction if needed)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

    def forward_routing(self, tokens: mx.array) -> mx.array:
        """Forward pass through embedding and routing layers.

        Args:
            tokens: (batch, max_len) int tensor

        Returns:
            routing_logits: (batch, n_ops + 2*max_val) float tensor
        """
        B, T = tokens.shape
        config = self.config

        # Embed tokens + positions
        pos_ids = mx.arange(T)
        x = self.embed(tokens) + self.pos_embed(pos_ids)  # (B, T, d_model)

        # Mask for non-pad positions
        mask = (tokens != 0).astype(mx.float32)  # (B, T)

        if config.use_positional_routing:
            # Three learned queries attend over the sequence to extract
            # op, arg1, arg2 representations independently.
            # This preserves positional information — each query can learn
            # to attend to the right positions.

            # Attention: query @ keys^T / sqrt(d), masked
            scale = config.d_model ** -0.5

            # Expand queries for batch: (1, d_model) → (B, 1, d_model)
            op_q = mx.broadcast_to(self.op_query, (B, 1, config.d_model))
            a1_q = mx.broadcast_to(self.arg1_query, (B, 1, config.d_model))
            a2_q = mx.broadcast_to(self.arg2_query, (B, 1, config.d_model))

            # Attention scores: (B, 1, d) @ (B, d, T) → (B, 1, T)
            x_T = mx.transpose(x, axes=(0, 2, 1))  # (B, d, T)
            op_scores = (op_q @ x_T) * scale   # (B, 1, T)
            a1_scores = (a1_q @ x_T) * scale
            a2_scores = (a2_q @ x_T) * scale

            # Mask padding
            mask_3d = mask[:, None, :]  # (B, 1, T)
            big_neg = mx.array(-1e9)
            op_scores = mx.where(mask_3d > 0, op_scores, big_neg)
            a1_scores = mx.where(mask_3d > 0, a1_scores, big_neg)
            a2_scores = mx.where(mask_3d > 0, a2_scores, big_neg)

            # Softmax → weighted sum
            op_attn = mx.softmax(op_scores, axis=-1)   # (B, 1, T)
            a1_attn = mx.softmax(a1_scores, axis=-1)
            a2_attn = mx.softmax(a2_scores, axis=-1)

            op_repr = (op_attn @ x).squeeze(1)   # (B, d_model)
            a1_repr = (a1_attn @ x).squeeze(1)
            a2_repr = (a2_attn @ x).squeeze(1)

            # Mix layers on each representation independently
            for layer in self.mix_layers:
                op_repr = op_repr + layer(op_repr)
                a1_repr = a1_repr + layer(a1_repr)
                a2_repr = a2_repr + layer(a2_repr)

            # Project each to its logit space
            op_logits = self.op_proj(op_repr)[:, :config.n_ops]      # (B, n_ops)
            a1_logits = self.arg1_proj(a1_repr)[:, :config.max_val]  # (B, max_val)
            a2_logits = self.arg2_proj(a2_repr)[:, :config.max_val]  # (B, max_val)

            # Concatenate into the standard routing logits format
            route_logits = mx.concatenate([op_logits, a1_logits, a2_logits], axis=-1)
            return route_logits

        else:
            # Original mean-pool path
            mask_sum = mx.maximum(mask.sum(axis=-1, keepdims=True), 1.0)
            x_pooled = (x * mask[..., None]).sum(axis=1) / mask_sum  # (B, d_model)

            for layer in self.mix_layers:
                x_pooled = x_pooled + layer(x_pooled)

            route_logits = self.route_proj(x_pooled)
            route_logits = route_logits[..., :self._route_dim]
            return route_logits

    def __call__(
        self, tokens: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Full forward: tokens → routing → kernel → result.

        Args:
            tokens: (batch, max_len) int tensor

        Returns:
            (encoded_result, op, arg1, arg2, result)
        """
        route_logits = self.forward_routing(tokens)
        encoded, op, arg1, arg2, result = kernel_forward(
            route_logits, self.result_encoder, max_val=self.config.max_val,
        )
        return encoded, op, arg1, arg2, result

    def count_params(self) -> dict[str, int]:
        """Count parameters by type."""
        from mlx.utils import tree_flatten
        total = 0
        ternary = 0
        continuous = 0
        for name, p in tree_flatten(self.parameters()):
            n = p.size
            total += n
            if p.dtype == mx.uint32:
                ternary += n * 16  # each uint32 holds 16 ternary weights
            elif p.dtype == mx.uint8:
                ternary += n * 4   # each uint8 holds 4 ternary weights
            else:
                continuous += n
        return {"total": total, "ternary_logical": ternary, "continuous": continuous}


# ══════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  v9 — Kernel Router Model Smoke Test")
    print("=" * 60)

    config = KernelRouterConfig()
    model = KernelRouter(config)

    # Test tokenization
    expr1 = "(+ 3 4)"
    expr2 = "(* 12 5)"
    t1 = tokenize_expr(expr1)
    t2 = tokenize_expr(expr2)
    print(f"\nTokenization:")
    print(f"  '{expr1}' → {t1[:len(expr1)]}")
    print(f"  '{expr2}' → {t2[:len(expr2)]}")
    print(f"  Roundtrip: '{detokenize(t1)}'")

    # Test forward pass
    tokens = mx.array([t1, t2])
    encoded, op, arg1, arg2, result = model(tokens)
    mx.eval(encoded, op, arg1, arg2, result)

    print(f"\nForward pass:")
    print(f"  Input: '{expr1}' → decoded op={op[0].item()}, "
          f"arg1={arg1[0].item()}, arg2={arg2[0].item()}, "
          f"result={result[0].item()}")
    print(f"  Input: '{expr2}' → decoded op={op[1].item()}, "
          f"arg1={arg1[1].item()}, arg2={arg2[1].item()}, "
          f"result={result[1].item()}")
    print(f"  Encoded shape: {encoded.shape}")

    # Test routing logits shape
    route = model.forward_routing(tokens)
    mx.eval(route)
    expected_dim = config.n_ops + 2 * config.max_val
    print(f"\nRouting logits shape: {route.shape} (expected: (2, {expected_dim}))")
    assert route.shape == (2, expected_dim), f"Shape mismatch!"

    # Parameter count
    params = model.count_params()
    print(f"\nParameters:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    print(f"\n{'=' * 60}")
    print(f"  ✓ Model smoke test passed")
    print(f"{'=' * 60}")
