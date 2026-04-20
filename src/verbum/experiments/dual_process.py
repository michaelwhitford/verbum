"""Dual-process experiment — reasoning vs. compilation pathways.

Tests whether specific attention heads mediate the switch between
extended chain-of-thought reasoning (System 2) and direct lambda
compilation (System 1). Ablating a head that is necessary for the
compilation pathway should force the model into a slower, reasoning-
heavy output mode without lambda syntax.

Structure::

    Graph("dual-process")
      └── per probe: Graph("probe-{id}")
            ├── DualProcessNode("baseline")  — no ablation, long generation
            ├── DualProcessNode("L1-H0")     — head (1, 0) zeroed
            ├── DualProcessNode("L24-H0")    — head (24, 0) zeroed
            └── DualProcessNode("L24-H2")    — head (24, 2) zeroed

Usage::

    from verbum.experiments.dual_process import build_dual_process
    from verbum.experiment import run, default_interceptors
    from verbum.instrument import load_model

    model, tokenizer, info = load_model("Qwen/Qwen3-4B")
    graph = build_dual_process(
        probe_set_path="probes/gate-ablation.json",
        gates_dir="gates",
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
    zero_heads_generate,
)
from verbum.probes import load_probe_set, resolve_probes

__all__ = [
    "DualProcessConfig",
    "DualProcessNode",
    "build_dual_process",
]

# Verbal markers associated with explicit chain-of-thought reasoning
_REASONING_INDICATORS = ["I need to", "Let me", "Okay", "so", "figure out"]


# ─────────────────────────── config ───────────────────────────────────


class DualProcessConfig(BaseModel):
    """Config for one dual-process trial on one prompt.

    ``ablated_head`` is ``[layer, head]`` or ``None`` for the baseline
    (no ablation). The baseline uses the same long generation budget so
    that reasoning-mode vs. compilation-mode length differences are
    directly comparable.
    """

    model_config = ConfigDict(frozen=True)

    kind: str = "dual_process"
    model: str
    ablated_head: list[int] | None  # [layer, head] or None
    prompt_hash: str
    prompt_preview: str
    max_new_tokens: int = 150


# ─────────────────────────── computation ──────────────────────────────


class DualProcessNode(Computation):
    """Generate with or without one head zeroed, checking process mode.

    Baseline (``ablated_head=None``): generates with 150 tokens, no
    ablation — captures the natural compilation / reasoning balance.

    Ablated (``ablated_head=[L, H]``): zeros head H in layer L for the
    full generation. If the head is part of the compilation pathway,
    the model should switch to extended reasoning mode.

    Returns:
        generation: The generated text (new tokens only).
        has_lambda: Whether lambda indicators appear in the output.
        lambda_count: Total lambda indicator count.
        has_reasoning: Whether reasoning-mode phrases appear.
        ablated_head: ``[layer, head]`` or ``None``.
        is_baseline: ``True`` when no head is ablated.
    """

    def __init__(self, config: DualProcessConfig, prompt: str) -> None:
        self._config = config
        self._prompt = prompt

    @property
    def config(self) -> DualProcessConfig:
        return self._config

    def execute(self, ctx: Context) -> dict[str, Any]:
        model = ctx.resources["model"]
        tokenizer = ctx.resources["tokenizer"]
        info = ctx.resources["info"]

        ablated_head = self._config.ablated_head
        max_new = self._config.max_new_tokens
        prompt = self._prompt
        is_baseline = ablated_head is None

        if is_baseline:
            gen = _generate(model, tokenizer, prompt, max_new)
        else:
            head_tuple = (ablated_head[0], ablated_head[1])
            gen, _, _ = zero_heads_generate(
                model,
                tokenizer,
                prompt,
                info,
                [head_tuple],
                max_new_tokens=max_new,
            )

        has_lambda = _detect_lambda(gen)
        lambda_count = sum(gen.count(s) for s in LAMBDA_INDICATORS)
        has_reasoning = any(indicator in gen for indicator in _REASONING_INDICATORS)

        return {
            "generation": gen,
            "has_lambda": has_lambda,
            "lambda_count": lambda_count,
            "has_reasoning": has_reasoning,
            "ablated_head": ablated_head,
            "is_baseline": is_baseline,
        }


# ─────────────────────────── graph builder ────────────────────────────


def build_dual_process(
    *,
    probe_set_path: str | Path,
    gates_dir: str | Path,
    essential_heads: list[tuple[int, int]] = ((1, 0), (24, 0), (24, 2)),  # type: ignore[assignment]
    model_name: str = "Qwen/Qwen3-4B",
    max_new_tokens: int = 150,
) -> Graph:
    """Build the dual-process experiment graph.

    Constructs ``Graph("dual-process")`` with one sub-graph per probe.
    Each probe sub-graph contains a baseline node and one ablation node
    per head in ``essential_heads``.

    Node names follow the pattern ``L{layer}-H{head}`` for ablation
    nodes and ``baseline`` for the unablated trial.

    Args:
        probe_set_path: Path to the probe-set JSON file.
        gates_dir: Directory containing gate ``.txt`` files.
        essential_heads: Heads to ablate individually. Defaults to the
            three essential heads identified in prior experiments.
        model_name: HuggingFace model identifier.
        max_new_tokens: Generation budget (default 150 — long enough to
            distinguish reasoning vs. direct compilation).
    """
    probe_set_path = Path(probe_set_path)
    gates_dir = Path(gates_dir)

    probe_set = load_probe_set(probe_set_path)
    resolved = resolve_probes(probe_set, gates_dir)

    essential_heads_list: list[tuple[int, int]] = list(essential_heads)

    probe_graphs: dict[str, Computation] = {}

    for rp in resolved:
        trial_nodes: dict[str, Computation] = {}

        # Baseline — no ablation
        baseline_cfg = DualProcessConfig(
            model=model_name,
            ablated_head=None,
            prompt_hash=rp.prompt_hash,
            prompt_preview=rp.full_prompt[:60],
            max_new_tokens=max_new_tokens,
        )
        trial_nodes["baseline"] = DualProcessNode(
            config=baseline_cfg,
            prompt=rp.full_prompt,
        )

        # One ablation node per essential head
        for layer, head in essential_heads_list:
            cfg = DualProcessConfig(
                model=model_name,
                ablated_head=[layer, head],
                prompt_hash=rp.prompt_hash,
                prompt_preview=rp.full_prompt[:60],
                max_new_tokens=max_new_tokens,
            )
            trial_nodes[f"L{layer}-H{head}"] = DualProcessNode(
                config=cfg,
                prompt=rp.full_prompt,
            )

        probe_graphs[f"probe-{rp.probe_id}"] = Graph(
            id=f"probe-{rp.probe_id}",
            children=trial_nodes,
        )

    return Graph(
        id="dual-process",
        children=probe_graphs,
    )
