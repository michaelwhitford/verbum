"""Head ablation experiment — which attention heads are necessary for compilation?

Wraps ``instrument.ablate_heads()`` in the fractal experiment framework.
Structure:

    Graph("head-ablation")
      └── per probe: Graph("probe-{id}")
            └── per layer: Computation("L{layer}")
                  → ablates all heads in that layer, returns results list

Each layer-level node ablates 32 heads (one forward pass per head)
and is independently cacheable. If the experiment crashes mid-run,
completed layers are skipped on restart.

Usage::

    from verbum.experiments.head_ablation import build_head_ablation
    from verbum.experiment import run, default_interceptors
    from verbum.instrument import load_model

    model, tokenizer, info = load_model("Qwen/Qwen3-4B")
    graph = build_head_ablation(
        probe_set_path="probes/gate-ablation.json",
        gates_dir="gates",
        target_layers=[0, 1, 4, 7, 24, 26, 30, 33],
    )
    interceptors = default_interceptors(
        Path("results/experiments"),
        resources={"model": model, "tokenizer": tokenizer, "info": info},
    )
    results = run(graph, interceptors=interceptors)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from verbum.experiment import Computation, Context, Graph
from verbum.instrument import (
    LAMBDA_INDICATORS,
    _detect_lambda,
    _generate,
)
from verbum.probes import load_probe_set, resolve_probes

__all__ = [
    "HeadAblationLayerConfig",
    "HeadAblationLayerNode",
    "build_head_ablation",
]


# ─────────────────────────── config ───────────────────────────────────


class HeadAblationLayerConfig(BaseModel):
    """Config for ablating all heads in one layer on one prompt."""

    model_config = ConfigDict(frozen=True)

    kind: str = "head_ablation_layer"
    model: str
    layer: int
    n_heads: int
    head_dim: int
    prompt_hash: str  # content_hash of the full prompt (gate + input)
    prompt_preview: str  # first 60 chars for human readability
    max_new_tokens: int = 30


# ─────────────────────────── computation ──────────────────────────────


class HeadAblationLayerNode(Computation):
    """Ablate each head in one layer, return results.

    Executes n_heads forward passes. Each head is zeroed out
    individually and the model generates. Returns a dict with
    baseline text and per-head results.
    """

    def __init__(self, config: HeadAblationLayerConfig, prompt: str) -> None:
        self._config = config
        self._prompt = prompt  # full prompt (gate + input)

    @property
    def config(self) -> HeadAblationLayerConfig:
        return self._config

    def execute(self, ctx: Context) -> dict[str, Any]:
        model = ctx.resources["model"]
        tokenizer = ctx.resources["tokenizer"]

        layer_idx = self._config.layer
        head_dim = self._config.head_dim
        n_heads = self._config.n_heads
        prompt = self._prompt
        max_new = self._config.max_new_tokens

        # Baseline generation (no ablation)
        baseline = _generate(model, tokenizer, prompt, max_new)

        # Per-head ablation
        from verbum.instrument import _get_layers, _get_self_attn

        layers = _get_layers(model)
        head_results: list[dict[str, Any]] = []

        for head_idx in range(n_heads):
            start = head_idx * head_dim
            end = start + head_dim

            def attn_hook(
                module: Any,
                args: Any,
                output: Any,
                *,
                _s: int = start,
                _e: int = end,
            ) -> Any:
                patched = output[0].clone()
                patched[:, :, _s:_e] = 0.0
                return (patched, *output[1:])

            h = _get_self_attn(layers[layer_idx]).register_forward_hook(attn_hook)
            try:
                gen = _generate(model, tokenizer, prompt, max_new)
            finally:
                h.remove()

            has_lambda = _detect_lambda(gen)
            lambda_count = sum(gen.count(s) for s in LAMBDA_INDICATORS)

            head_results.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "generation": gen,
                    "has_lambda": has_lambda,
                    "lambda_count": lambda_count,
                }
            )

        broken = [r["head"] for r in head_results if not r["has_lambda"]]

        return {
            "layer": layer_idx,
            "baseline": baseline,
            "baseline_has_lambda": _detect_lambda(baseline),
            "n_heads": n_heads,
            "head_results": head_results,
            "broken_heads": broken,
            "n_broken": len(broken),
        }


# ─────────────────────────── graph builder ────────────────────────────


def build_head_ablation(
    *,
    probe_set_path: str | Path,
    gates_dir: str | Path,
    target_layers: list[int],
    model_name: str = "Qwen/Qwen3-4B",
    n_heads: int = 32,
    head_dim: int = 128,
    max_new_tokens: int = 30,
) -> Graph:
    """Build the full head-ablation experiment graph.

    Structure::

        Graph("head-ablation")
          ├── Graph("probe-ga-simple")
          │   ├── HeadAblationLayerNode("L0")   # 32 heads
          │   ├── HeadAblationLayerNode("L1")   # 32 heads
          │   └── ...
          ├── Graph("probe-ga-quant")
          │   └── ...
          └── ...

    Each layer node is independently cacheable. Total forward passes:
    ``len(probes) * len(target_layers) * n_heads``.
    """
    probe_set_path = Path(probe_set_path)
    gates_dir = Path(gates_dir)

    probe_set = load_probe_set(probe_set_path)
    resolved = resolve_probes(probe_set, gates_dir)

    probe_graphs: dict[str, Computation] = {}

    for rp in resolved:
        layer_nodes: dict[str, Computation] = {}

        for layer_idx in target_layers:
            config = HeadAblationLayerConfig(
                model=model_name,
                layer=layer_idx,
                n_heads=n_heads,
                head_dim=head_dim,
                prompt_hash=rp.prompt_hash,
                prompt_preview=rp.full_prompt[:60],
                max_new_tokens=max_new_tokens,
            )
            layer_nodes[f"L{layer_idx}"] = HeadAblationLayerNode(
                config=config,
                prompt=rp.full_prompt,
            )

        probe_graphs[f"probe-{rp.probe_id}"] = Graph(
            id=f"probe-{rp.probe_id}",
            children=layer_nodes,
        )

    return Graph(
        id="head-ablation",
        children=probe_graphs,
    )
