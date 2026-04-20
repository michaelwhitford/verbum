"""Decompilation circuit — are the same heads necessary for lambda-to-English?

Tests whether L1:H0, L24:H0, L24:H2 are also necessary for
decompilation (reversing the compile direction). Uses a decompile
gate (lambda -> English exemplars) and decompile probes.

If the same heads break decompilation, the circuit is bidirectional.
If different heads break, compilation and decompilation are distinct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from verbum.experiment import Computation, Context, Graph
from verbum.instrument import (
    _detect_lambda,
    _generate,
    zero_heads_generate,
)
from verbum.probes import load_probe_set, resolve_probes

__all__ = [
    "DecompileAblationConfig",
    "DecompileAblationNode",
    "build_decompile_ablation",
]


# For decompilation, "success" = English output (not lambda)
ENGLISH_INDICATORS = [" the ", " a ", " is ", " are ", " was ", " has "]


def _detect_english(text: str) -> bool:
    """Heuristic: does this text contain natural English?"""
    lower = text.lower()
    return sum(lower.count(s) for s in ENGLISH_INDICATORS) >= 2


class DecompileAblationConfig(BaseModel):
    """Config for testing one head's effect on decompilation."""

    model_config = ConfigDict(frozen=True)

    kind: str = "decompile_ablation"
    model: str
    ablated_head: list[int] | None  # [layer, head] or None for baseline
    prompt_hash: str
    prompt_preview: str
    max_new_tokens: int = 50


class DecompileAblationNode(Computation):
    """Ablate one head and check if decompilation survives."""

    def __init__(self, config: DecompileAblationConfig, prompt: str) -> None:
        self._config = config
        self._prompt = prompt

    @property
    def config(self) -> DecompileAblationConfig:
        return self._config

    def execute(self, ctx: Context) -> dict[str, Any]:
        model = ctx.resources["model"]
        tokenizer = ctx.resources["tokenizer"]
        prompt = self._prompt
        max_new = self._config.max_new_tokens

        if self._config.ablated_head is None:
            # Baseline: no ablation
            gen = _generate(model, tokenizer, prompt, max_new)
        else:
            layer, head = self._config.ablated_head
            gen, _, _ = zero_heads_generate(
                model,
                tokenizer,
                prompt,
                ctx.resources["info"],
                heads=[(layer, head)],
                max_new_tokens=max_new,
            )

        has_english = _detect_english(gen)
        has_lambda = _detect_lambda(gen)

        return {
            "generation": gen,
            "has_english": has_english,
            "has_lambda": has_lambda,
            "ablated_head": self._config.ablated_head,
            "is_baseline": self._config.ablated_head is None,
        }


def build_decompile_ablation(
    *,
    probe_set_path: str | Path = "probes/decompile.json",
    gates_dir: str | Path = "gates",
    essential_heads: list[tuple[int, int]] | None = None,
    model_name: str = "Qwen/Qwen3-4B",
    max_new_tokens: int = 50,
) -> Graph:
    """Build decompilation ablation experiment.

    Tests each essential head + baseline on decompile probes.
    """
    if essential_heads is None:
        essential_heads = [(1, 0), (24, 0), (24, 2)]

    probe_set_path = Path(probe_set_path)
    gates_dir = Path(gates_dir)
    probe_set = load_probe_set(probe_set_path)
    resolved = resolve_probes(probe_set, gates_dir)

    probe_graphs: dict[str, Computation] = {}

    for rp in resolved:
        nodes: dict[str, Computation] = {}

        # Baseline (no ablation)
        nodes["baseline"] = DecompileAblationNode(
            config=DecompileAblationConfig(
                model=model_name,
                ablated_head=None,
                prompt_hash=rp.prompt_hash,
                prompt_preview=rp.full_prompt[:60],
                max_new_tokens=max_new_tokens,
            ),
            prompt=rp.full_prompt,
        )

        # Each essential head
        for layer, head in essential_heads:
            nodes[f"L{layer}-H{head}"] = DecompileAblationNode(
                config=DecompileAblationConfig(
                    model=model_name,
                    ablated_head=[layer, head],
                    prompt_hash=rp.prompt_hash,
                    prompt_preview=rp.full_prompt[:60],
                    max_new_tokens=max_new_tokens,
                ),
                prompt=rp.full_prompt,
            )

        probe_graphs[f"probe-{rp.probe_id}"] = Graph(
            id=f"probe-{rp.probe_id}",
            children=nodes,
        )

    return Graph(id="decompile", children=probe_graphs)
