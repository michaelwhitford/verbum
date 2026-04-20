"""Three-phase causal language model — Montague primitives as architecture.

The MontaguCompiler (encoder-decoder) proved that separating the three
primitives improves compilation. This module extends the idea to a
causal language model: if the compressor IS the function that emerges
from next-token prediction, then an architecture shaped for the three
primitives should learn it more efficiently from raw text.

Architecture (causal, all left-to-right):

  Phase 1: Type Embedding (1 layer)
    Token embeddings + 1 self-attention layer
    Assigns typed representations to each token position
    Initialized from Pythia-160M embeddings (optional)

  Phase 2: Structure Parser (2 layers)
    Causal self-attention
    Determines composition relationships (what composes with what)
    Own residual stream, receives Phase 1 output via projection

  Phase 3: Typed Application (3 layers)
    Causal self-attention
    Executes composition, routes to next-token prediction
    Own residual stream, receives Phase 2 output via projection

Each phase has its own residual stream dimension. Information flows
between phases via learned linear projections — not shared residual.
This eliminates the superposition interference that forces the three
computations to pack at 120° in a standard transformer.

Total: 6 self-attention layers (same depth as Pythia-14M)
but organized into the three Montague primitives.

License: MIT
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttentionLayer(nn.Module):
    """Standard causal self-attention + FFN layer."""

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

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class PhaseBlock(nn.Module):
    """A phase = N causal self-attention layers with own residual stream.

    Each phase operates in its own d_model dimension. Input from the
    previous phase is projected into this phase's space. This prevents
    superposition interference between phases.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Project from previous phase's dimension
        self.input_proj = nn.Linear(d_input, d_model) if d_input != d_model else nn.Identity()
        self.layers = nn.ModuleList([
            CausalSelfAttentionLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, causal_mask)
        return x


class MontaguLM(nn.Module):
    """Three-phase causal language model.

    Phase 1 (Type):      1 layer,  d=256, 4 heads
    Phase 2 (Structure): 2 layers, d=256, 4 heads
    Phase 3 (Apply):     3 layers, d=256, 8 heads
    Total:               6 layers (same depth as Pythia-14M)

    Each phase has its own residual stream with projections between
    phases — eliminating superposition of the three Montague primitives.
    """

    def __init__(
        self,
        vocab_size: int,
        d_embed: int = 768,      # pretrained embedding dim
        d_type: int = 256,       # phase 1 dim
        d_parse: int = 256,      # phase 2 dim
        d_apply: int = 256,      # phase 3 dim
        n_type_layers: int = 1,
        n_type_heads: int = 4,
        n_parse_layers: int = 2,
        n_parse_heads: int = 4,
        n_apply_layers: int = 3,
        n_apply_heads: int = 8,
        d_ff_type: int = 512,
        d_ff_parse: int = 512,
        d_ff_apply: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_apply = d_apply

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_embed)
        if pretrained_embeddings is not None:
            self.token_embed.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.token_embed.weight.requires_grad = False

        # Positional embedding
        self.pos_embed = nn.Embedding(max_len, d_embed)

        # Phase 1: Type assignment (embed → typed representations)
        self.type_phase = PhaseBlock(
            d_input=d_embed, d_model=d_type,
            n_layers=n_type_layers, n_heads=n_type_heads,
            d_ff=d_ff_type, dropout=dropout,
        )

        # Phase 2: Structure parsing (type → structural relationships)
        self.parse_phase = PhaseBlock(
            d_input=d_type, d_model=d_parse,
            n_layers=n_parse_layers, n_heads=n_parse_heads,
            d_ff=d_ff_parse, dropout=dropout,
        )

        # Phase 3: Typed application (structure → composed meaning)
        self.apply_phase = PhaseBlock(
            d_input=d_parse, d_model=d_apply,
            n_layers=n_apply_layers, n_heads=n_apply_heads,
            d_ff=d_ff_apply, dropout=dropout,
        )

        # Output head: project to vocabulary
        self.output_norm = nn.LayerNorm(d_apply)
        # If d_apply == d_embed, tie output weights to input embeddings
        # (standard practice — halves embedding parameter count)
        if d_apply == d_embed and not freeze_embeddings:
            self.output_proj = None  # will use token_embed.weight
        else:
            self.output_proj = nn.Linear(d_apply, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        input_ids: (batch, seq_len) — token IDs
        targets:   (batch, seq_len) — next-token targets (optional)
        Returns:   (logits, loss) where loss is None if targets not given
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Embed
        positions = torch.arange(seq_len, device=device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        # Causal mask (same for all phases)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

        # Phase 1: Type assignment
        x = self.type_phase(x, causal_mask)

        # Phase 2: Structure parsing
        x = self.parse_phase(x, causal_mask)

        # Phase 3: Typed application
        x = self.apply_phase(x, causal_mask)

        # Output
        x = self.output_norm(x)
        if self.output_proj is not None:
            logits = self.output_proj(x)
        else:
            logits = F.linear(x, self.token_embed.weight)  # tied weights

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def count_parameters(self) -> dict[str, int]:
        embed_params = sum(
            p.numel() for p in self.token_embed.parameters() if p.requires_grad
        )
        embed_params += sum(p.numel() for p in self.pos_embed.parameters())
        type_params = sum(p.numel() for p in self.type_phase.parameters())
        parse_params = sum(p.numel() for p in self.parse_phase.parameters())
        apply_params = sum(p.numel() for p in self.apply_phase.parameters())
        head_params = sum(p.numel() for p in self.output_norm.parameters())
        if self.output_proj is not None:
            head_params += sum(p.numel() for p in self.output_proj.parameters())
        total = embed_params + type_params + parse_params + apply_params + head_params
        return {
            "embeddings": embed_params,
            "phase1_type": type_params,
            "phase2_parse": parse_params,
            "phase3_apply": apply_params,
            "output_head": head_params,
            "total": total,
        }

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50) -> torch.Tensor:
        """Simple greedy generation."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_len
            x = input_ids[:, -self.max_len:]
            logits, _ = self(x)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
