"""Three-phase Montague Compiler — architecture shaped by empirical circuit discovery.

Empirical findings from Pythia-160M circuit analysis:
  Phase 1 (Embedding+L0): Type assignment — 84% in embeddings, L0 refines to 93%
  Phase 2 (L3):           Structure parse — patching L3 transfers composition order
  Phase 3 (L8-L11):       Typed application — compile-selective, consumes structure

Superposition theory (Elhage et al. 2022) predicts these three computations
are packed at 120° in the residual stream. This architecture separates them
into dedicated modules — eliminating superposition interference.

Architecture:
  ┌─────────────────┐
  │ Type Embedding   │  Phase 1: token → (type + meaning) vectors
  │ (from Pythia)    │  Frozen or lightly fine-tuned
  └────────┬────────┘
           │
  ┌────────▼────────┐
  │ Structure Parser │  Phase 2: self-attention encoder (2 layers)
  │                  │  Computes composition order from type+meaning
  └────────┬────────┘
           │
  ┌────────▼────────┐
  │ Typed Apply      │  Phase 3: cross-attention decoder (3 layers)
  │ (Decoder)        │  Generates lambda tokens using cross-attention
  │                  │  to parser output — natural copy mechanism
  └────────┬────────┘
           │
         output (lambda notation tokens)

The cross-attention in the decoder is the key insight: it gives the model
a natural mechanism to COPY tokens from the input to the output — exactly
what the Pythia-160M fine-tuning lacked (Finding 29: content mapping gap
is architectural — decoder-only models can't point back at input).

License: MIT
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# Lambda output vocabulary
# ══════════════════════════════════════════════════════════════════════
# Small dedicated vocabulary for lambda notation output.
# Much smaller than a full LM vocabulary — enables tiny output embedding.

# Special tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# Lambda notation tokens
LAMBDA_VOCAB = [
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
    # Binders
    "λ", "∀", "∃", "ι",
    # Connectives
    "∧", "∨", "→", "¬",
    # Structural
    "(", ")", ".", ",", " ",
    # Variables
    "u", "v", "w", "x", "y", "z",
]

# Predicates and entities are added dynamically from training data


class LambdaTokenizer:
    """Simple character/token-level tokenizer for lambda notation output."""

    def __init__(self, extra_tokens: list[str] | None = None):
        self.tokens = list(LAMBDA_VOCAB)
        if extra_tokens:
            for t in extra_tokens:
                if t not in self.tokens:
                    self.tokens.append(t)
        self.token2id = {t: i for i, t in enumerate(self.tokens)}
        self.id2token = {i: t for i, t in enumerate(self.tokens)}
        self.pad_id = self.token2id[PAD_TOKEN]
        self.bos_id = self.token2id[BOS_TOKEN]
        self.eos_id = self.token2id[EOS_TOKEN]
        self.unk_id = self.token2id[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        """Greedy longest-match tokenization."""
        ids = [self.bos_id]
        i = 0
        while i < len(text):
            # Try longest match first
            matched = False
            for length in range(min(20, len(text) - i), 0, -1):
                substr = text[i : i + length]
                if substr in self.token2id:
                    ids.append(self.token2id[substr])
                    i += length
                    matched = True
                    break
            if not matched:
                # Single character fallback
                char = text[i]
                if char in self.token2id:
                    ids.append(self.token2id[char])
                else:
                    ids.append(self.unk_id)
                i += 1
        ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for id in ids:
            if id in (self.pad_id, self.bos_id, self.eos_id):
                continue
            tokens.append(self.id2token.get(id, UNK_TOKEN))
        return "".join(tokens)

    @classmethod
    def from_training_data(cls, lambda_expressions: list[str]) -> "LambdaTokenizer":
        """Build tokenizer from training data, extracting all predicates/entities."""
        import re

        extra = set()
        for expr in lambda_expressions:
            # Extract identifiers (2+ lowercase chars, possibly with underscores)
            identifiers = re.findall(r"[a-z][a-z_]+", expr)
            extra.update(identifiers)

        return cls(extra_tokens=sorted(extra))


# ══════════════════════════════════════════════════════════════════════
# Phase 1: Type Embedding
# ══════════════════════════════════════════════════════════════════════


class TypeEmbedding(nn.Module):
    """Project input token embeddings into type + meaning subspaces.

    Optionally initialized from a pretrained model's embedding table.
    The type projection and meaning projection operate on the same
    input embedding but produce vectors in separate subspaces.
    """

    def __init__(
        self,
        input_vocab_size: int,
        d_input: int,
        d_model: int,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = True,
    ):
        super().__init__()

        self.embedding = nn.Embedding(input_vocab_size, d_input)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # Project from pretrained embedding dim to model dim
        self.projection = nn.Linear(d_input, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (batch, seq_len) → (batch, seq_len, d_model)"""
        x = self.embedding(input_ids)
        x = self.projection(x)
        x = self.norm(x)
        return x


# ══════════════════════════════════════════════════════════════════════
# Phase 2: Structure Parser (Encoder)
# ══════════════════════════════════════════════════════════════════════


class ParserLayer(nn.Module):
    """One self-attention encoder layer for structural parsing."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention + residual
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN + residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class StructureParser(nn.Module):
    """Phase 2: encode input with structural information.

    Self-attention layers that compute composition relationships
    between tokens. The output encodes both the original meaning
    AND the structural parse — which tokens compose with which.
    """

    def __init__(
        self, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [ParserLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x


# ══════════════════════════════════════════════════════════════════════
# Phase 3: Typed Application (Decoder)
# ══════════════════════════════════════════════════════════════════════


class ApplyLayer(nn.Module):
    """One decoder layer: self-attention + cross-attention + FFN.

    The cross-attention is the KEY mechanism: it allows the decoder
    to look back at the parser output (which encodes structure + types)
    and copy/transform information from input to output.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Causal self-attention (for autoregressive generation)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention to encoder (parser) output
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Causal self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Cross-attention to encoder
        cross_out, _ = self.cross_attn(
            x, encoder_out, encoder_out, key_padding_mask=encoder_padding_mask
        )
        x = self.norm2(x + self.dropout(cross_out))
        # FFN
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class TypedApplyDecoder(nn.Module):
    """Phase 3: generate lambda tokens using typed application.

    Autoregressive decoder with cross-attention to parser output.
    The cross-attention naturally implements the copy mechanism
    that decoder-only models lack.
    """

    def __init__(
        self,
        output_vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_embed = nn.Embedding(output_vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList(
            [ApplyLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(d_model, output_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_ids: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        tgt_ids: (batch, tgt_len) — lambda token IDs (teacher-forced)
        encoder_out: (batch, src_len, d_model) — parser output
        Returns: (batch, tgt_len, output_vocab_size) — logits
        """
        batch, tgt_len = tgt_ids.shape

        # Embed output tokens + positional
        positions = torch.arange(tgt_len, device=tgt_ids.device)
        x = self.output_embed(tgt_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embed(positions)
        x = self.dropout(x)

        # Causal mask (upper triangular)
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt_ids.device), diagonal=1
        ).bool()

        for layer in self.layers:
            x = layer(x, encoder_out, tgt_mask=causal_mask,
                      encoder_padding_mask=encoder_padding_mask)

        logits = self.output_proj(x)
        return logits


# ══════════════════════════════════════════════════════════════════════
# Full Model: MontaguCompiler
# ══════════════════════════════════════════════════════════════════════


class MontaguCompiler(nn.Module):
    """Three-phase encoder-decoder for natural language → lambda compilation.

    Phase 1 (Type Embedding):   token → typed meaning vectors
    Phase 2 (Structure Parser): self-attention encoder → composition structure
    Phase 3 (Typed Apply):      cross-attention decoder → lambda token sequence

    The separation of phases eliminates superposition interference between
    the three Montague primitives (type assignment, structural parsing,
    typed application) that are packed at ~120° in a standard transformer.
    """

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        d_input: int = 768,  # pretrained embedding dim
        d_model: int = 256,  # internal model dim
        n_parser_layers: int = 2,
        n_parser_heads: int = 4,
        n_apply_layers: int = 3,
        n_apply_heads: int = 4,
        d_ff: int = 512,
        max_len: int = 256,
        dropout: float = 0.1,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = True,
    ):
        super().__init__()

        # Phase 1: Type Embedding
        self.type_embed = TypeEmbedding(
            input_vocab_size, d_input, d_model,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
        )

        # Phase 2: Structure Parser (encoder)
        self.parser = StructureParser(
            d_model, n_parser_layers, n_parser_heads, d_ff, dropout
        )

        # Phase 3: Typed Application (decoder)
        self.decoder = TypedApplyDecoder(
            output_vocab_size, d_model, n_apply_layers, n_apply_heads,
            d_ff, max_len, dropout,
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run Phase 1 + Phase 2: input tokens → parsed representations."""
        x = self.type_embed(input_ids)
        x = self.parser(x, padding_mask)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full forward: input sentence → lambda logits (teacher-forced).

        input_ids:  (batch, src_len)  — English sentence tokens
        target_ids: (batch, tgt_len)  — lambda notation tokens (shifted right)
        Returns:    (batch, tgt_len, output_vocab_size) — logits
        """
        encoder_out = self.encode(input_ids, input_padding_mask)
        logits = self.decoder(target_ids, encoder_out, input_padding_mask)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        lambda_tokenizer: "LambdaTokenizer",
        max_len: int = 100,
        input_padding_mask: Optional[torch.Tensor] = None,
    ) -> list[str]:
        """Autoregressive generation of lambda expressions.

        Returns list of decoded strings (one per batch element).
        """
        self.eval()
        batch = input_ids.shape[0]
        device = input_ids.device

        encoder_out = self.encode(input_ids, input_padding_mask)

        # Start with BOS
        generated = torch.full(
            (batch, 1), lambda_tokenizer.bos_id, dtype=torch.long, device=device
        )

        for _ in range(max_len):
            logits = self.decoder(generated, encoder_out, input_padding_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have produced EOS
            if (next_token == lambda_tokenizer.eos_id).all():
                break

        results = []
        for i in range(batch):
            ids = generated[i].tolist()
            results.append(lambda_tokenizer.decode(ids))
        return results

    def count_parameters(self) -> dict[str, int]:
        """Count parameters per phase."""
        phase1 = sum(
            p.numel() for p in self.type_embed.parameters() if p.requires_grad
        )
        phase2 = sum(p.numel() for p in self.parser.parameters())
        phase3 = sum(p.numel() for p in self.decoder.parameters())
        total = phase1 + phase2 + phase3
        return {
            "phase1_type_embed": phase1,
            "phase2_parser": phase2,
            "phase3_decoder": phase3,
            "total_trainable": total,
        }
