"""Multi-head experiment — sufficiency and threshold tests.

Tests whether a small set of essential heads is sufficient for
compilation and at what head-count lambda generation starts to degrade.

Structure::

    Graph("multi-head")
      ├── Graph("sufficiency")
      │   └── per probe: SufficiencyNode
      │         → zeros all heads in critical layers EXCEPT essential_heads
      ├── Graph("threshold-5")
      │   └── per probe: ThresholdNode (5 non-essential heads zeroed)
      ├── Graph("threshold-10")
      │   └── ...
      ...

Usage::

    from verbum.experiments.multi_head import build_multi_head_experiment
    from verbum.experiment import run, default_interceptors
    from verbum.instrument import load_model

    model, tokenizer, info = load_model("Qwen/Qwen3-4B")
    graph = build_multi_head_experiment(
        probe_set_path="probes/gate-ablation.json",
        gates_dir="gates",
        essential_heads=[(1, 0), (24, 0), (24, 2)],
        critical_layers=[1, 24],
    )
    interceptors = default_interceptors(
        Path("results/experiments"),
        resources={"model": model, "tokenizer": tokenizer, "info": info},
    )
    results = run(graph, interceptors=interceptors)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from verbum.experiment import Computation, Context, Graph
from verbum.instrument import (
    _generate,
    zero_heads_generate,
)
from verbum.probes import load_probe_set, resolve_probes

__all__ = [
    "SufficiencyConfig",
    "SufficiencyNode",
    "ThresholdConfig",
    "ThresholdNode",
    "build_multi_head_experiment",
]


# ─────────────────────────── configs ──────────────────────────────────


class SufficiencyConfig(BaseModel):
    """Config for a sufficiency test on one prompt.

    Tests whether ``essential_heads`` alone (with all other heads in
    ``all_heads`` zeroed) are sufficient to produce lambda output.
    """

    model_config = ConfigDict(frozen=True)

    kind: str = "sufficiency_test"
    model: str
    essential_heads: list[list[int]]  # [(layer, head), ...]
    all_heads: list[list[int]]  # all heads in critical layers
    prompt_hash: str
    prompt_preview: str
    max_new_tokens: int = 30


class ThresholdConfig(BaseModel):
    """Config for a threshold test on one prompt.

    Zeros ``heads_to_zero`` simultaneously and checks whether lambda
    generation survives.
    """

    model_config = ConfigDict(frozen=True)

    kind: str = "threshold_test"
    model: str
    heads_to_zero: list[list[int]]  # [(layer, head), ...]
    prompt_hash: str
    prompt_preview: str
    max_new_tokens: int = 30


# ─────────────────────────── computations ─────────────────────────────


class SufficiencyNode(Computation):
    """Test whether essential heads alone can drive lambda compilation.

    Zeros every head in ``all_heads`` that is NOT in ``essential_heads``,
    leaving only the essential heads active in the critical layers. A
    single forward pass checks whether lambda output is preserved.
    """

    def __init__(self, config: SufficiencyConfig, prompt: str) -> None:
        self._config = config
        self._prompt = prompt

    @property
    def config(self) -> SufficiencyConfig:
        return self._config

    def execute(self, ctx: Context) -> dict[str, Any]:
        model = ctx.resources["model"]
        tokenizer = ctx.resources["tokenizer"]
        info = ctx.resources["info"]

        essential = {tuple(h) for h in self._config.essential_heads}
        all_heads = self._config.all_heads
        max_new = self._config.max_new_tokens
        prompt = self._prompt

        # Baseline: unablated generation
        baseline = _generate(model, tokenizer, prompt, max_new)

        # Zero every head in all_heads that is not essential
        heads_to_zero = [h for h in all_heads if tuple(h) not in essential]
        heads_as_tuples = [tuple(h) for h in heads_to_zero]

        gen, has_lambda, lambda_count = zero_heads_generate(
            model,
            tokenizer,
            prompt,
            info,
            heads_as_tuples,  # type: ignore[arg-type]
            max_new_tokens=max_new,
        )

        return {
            "baseline": baseline,
            "generation": gen,
            "has_lambda": has_lambda,
            "lambda_count": lambda_count,
            "n_zeroed": len(heads_to_zero),
            "essential_heads_only": True,
        }


class ThresholdNode(Computation):
    """Zero a fixed set of heads simultaneously and check lambda survival.

    Each instance zeros exactly the heads in ``heads_to_zero`` in one
    forward pass. Used across a range of threshold sizes (5, 10, ..., 25)
    to find where lambda generation begins to fail.
    """

    def __init__(self, config: ThresholdConfig, prompt: str) -> None:
        self._config = config
        self._prompt = prompt

    @property
    def config(self) -> ThresholdConfig:
        return self._config

    def execute(self, ctx: Context) -> dict[str, Any]:
        model = ctx.resources["model"]
        tokenizer = ctx.resources["tokenizer"]
        info = ctx.resources["info"]

        heads_as_tuples = [tuple(h) for h in self._config.heads_to_zero]
        max_new = self._config.max_new_tokens

        gen, has_lambda, lambda_count = zero_heads_generate(
            model,
            tokenizer,
            self._prompt,
            info,
            heads_as_tuples,  # type: ignore[arg-type]
            max_new_tokens=max_new,
        )

        return {
            "generation": gen,
            "has_lambda": has_lambda,
            "lambda_count": lambda_count,
            "n_zeroed": len(heads_as_tuples),
        }


# ─────────────────────────── graph builder ────────────────────────────


def build_multi_head_experiment(
    *,
    probe_set_path: str | Path,
    gates_dir: str | Path,
    essential_heads: list[tuple[int, int]],
    critical_layers: list[int],
    n_heads: int = 32,
    model_name: str = "Qwen/Qwen3-4B",
    max_new_tokens: int = 30,
) -> Graph:
    """Build the multi-head experiment graph.

    Constructs a ``Graph("multi-head")`` containing:

    - ``Graph("sufficiency")`` — one ``SufficiencyNode`` per probe,
      zeroing all non-essential heads in critical layers.
    - ``Graph("threshold-N")`` for N in {5, 10, 15, 20, 25} — one
      ``ThresholdNode`` per probe, zeroing N randomly-chosen
      non-essential heads (seed=42 for reproducibility).

    Args:
        probe_set_path: Path to the probe-set JSON file.
        gates_dir: Directory containing gate ``.txt`` files.
        essential_heads: Heads that must NOT be zeroed in sufficiency test.
        critical_layers: Layers whose full head set forms ``all_heads``.
        n_heads: Number of attention heads per layer (default 32).
        model_name: HuggingFace model identifier.
        max_new_tokens: Generation budget per forward pass.
    """
    probe_set_path = Path(probe_set_path)
    gates_dir = Path(gates_dir)

    probe_set = load_probe_set(probe_set_path)
    resolved = resolve_probes(probe_set, gates_dir)

    # Build the full list of (layer, head) pairs for critical layers
    all_heads: list[list[int]] = [
        [layer, head] for layer in sorted(critical_layers) for head in range(n_heads)
    ]
    essential_set = {(layer_idx, head_idx) for layer_idx, head_idx in essential_heads}
    non_essential: list[list[int]] = [
        h for h in all_heads if (h[0], h[1]) not in essential_set
    ]

    # Deterministic RNG for threshold sampling
    rng = random.Random(42)
    threshold_sizes = [5, 10, 15, 20, 25]

    # Pre-sample head sets per threshold (same across probes for comparison)
    threshold_head_sets: dict[int, list[list[int]]] = {}
    for n in threshold_sizes:
        sample_size = min(n, len(non_essential))
        threshold_head_sets[n] = rng.sample(non_essential, sample_size)

    # ── sufficiency sub-graph ──────────────────────────────────────────
    sufficiency_nodes: dict[str, Computation] = {}
    for rp in resolved:
        cfg = SufficiencyConfig(
            model=model_name,
            essential_heads=[list(h) for h in essential_heads],
            all_heads=all_heads,
            prompt_hash=rp.prompt_hash,
            prompt_preview=rp.full_prompt[:60],
            max_new_tokens=max_new_tokens,
        )
        sufficiency_nodes[f"probe-{rp.probe_id}"] = SufficiencyNode(
            config=cfg,
            prompt=rp.full_prompt,
        )

    # ── threshold sub-graphs ──────────────────────────────────────────
    threshold_graphs: dict[str, Computation] = {
        "sufficiency": Graph(id="sufficiency", children=sufficiency_nodes),
    }

    for n in threshold_sizes:
        heads_to_zero = threshold_head_sets[n]
        threshold_nodes: dict[str, Computation] = {}
        for rp in resolved:
            cfg = ThresholdConfig(
                model=model_name,
                heads_to_zero=heads_to_zero,
                prompt_hash=rp.prompt_hash,
                prompt_preview=rp.full_prompt[:60],
                max_new_tokens=max_new_tokens,
            )
            threshold_nodes[f"probe-{rp.probe_id}"] = ThresholdNode(
                config=cfg,
                prompt=rp.full_prompt,
            )
        threshold_graphs[f"threshold-{n}"] = Graph(
            id=f"threshold-{n}",
            children=threshold_nodes,
        )

    return Graph(
        id="multi-head",
        children=threshold_graphs,
    )
