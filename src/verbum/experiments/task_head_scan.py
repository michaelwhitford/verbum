"""Full head scan per task — find task-specific essential heads.

Generalizes head_ablation.py to work with any task and success detector.
Runs 8 critical layers x 32 heads per probe, checking task-specific
success criteria. Finds which heads are specialized preprocessors
that configure the universal compositor (L24:H0) for each task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from verbum.experiment import Computation, Context, Graph
from verbum.experiments.cross_task import DETECTORS
from verbum.instrument import (
    LAMBDA_INDICATORS,
    _detect_lambda,
    _generate,
)
from verbum.probes import load_probe_set, resolve_probes

__all__ = [
    "TaskHeadScanConfig",
    "TaskHeadScanNode",
    "build_task_head_scan",
]


class TaskHeadScanConfig(BaseModel):
    """Config for scanning all heads in one layer for one task."""

    model_config = ConfigDict(frozen=True)

    kind: str = "task_head_scan"
    task: str
    model: str
    layer: int
    n_heads: int
    head_dim: int
    prompt_hash: str
    prompt_preview: str
    max_new_tokens: int = 50


class TaskHeadScanNode(Computation):
    """Ablate each head in one layer, check task-specific success.

    Same structure as HeadAblationLayerNode but uses a task-specific
    detector instead of lambda detection.
    """

    def __init__(self, config: TaskHeadScanConfig, prompt: str) -> None:
        self._config = config
        self._prompt = prompt

    @property
    def config(self) -> TaskHeadScanConfig:
        return self._config

    def execute(self, ctx: Context) -> dict[str, Any]:
        model = ctx.resources["model"]
        tokenizer = ctx.resources["tokenizer"]

        layer_idx = self._config.layer
        head_dim = self._config.head_dim
        n_heads = self._config.n_heads
        prompt = self._prompt
        max_new = self._config.max_new_tokens
        task = self._config.task

        detector = DETECTORS[task]

        # Baseline (no ablation)
        baseline = _generate(model, tokenizer, prompt, max_new)
        baseline_success = detector(baseline)

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

            success = detector(gen)
            # Also check lambda (for cross-reference with compile)
            has_lambda = _detect_lambda(gen)
            lambda_count = sum(gen.count(s) for s in LAMBDA_INDICATORS)

            head_results.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "generation": gen,
                    "success": success,
                    "has_lambda": has_lambda,
                    "lambda_count": lambda_count,
                }
            )

        broken = [r["head"] for r in head_results if not r["success"]]

        return {
            "task": task,
            "layer": layer_idx,
            "baseline": baseline,
            "baseline_success": baseline_success,
            "n_heads": n_heads,
            "head_results": head_results,
            "broken_heads": broken,
            "n_broken": len(broken),
        }


def build_task_head_scan(
    *,
    tasks: dict[str, str] | None = None,
    gates_dir: str | Path = "gates",
    target_layers: list[int] | None = None,
    model_name: str = "Qwen/Qwen3-4B",
    n_heads: int = 32,
    head_dim: int = 80,
    max_new_tokens: int = 50,
) -> Graph:
    """Build full head scan for multiple tasks.

    Structure::

        Graph("task-head-scan")
          +-- Graph("extract")
          |     +-- Graph("probe-ext-simple")
          |     |     +-- TaskHeadScanNode("L0")  # 32 heads
          |     |     +-- TaskHeadScanNode("L1")  # 32 heads
          |     |     +-- ...
          |     +-- ...
          +-- Graph("translate")
          |     +-- ...
          +-- ...

    Total: len(tasks) * len(probes) * len(layers) leaf nodes.
    Each leaf does n_heads forward passes (32).
    """
    if target_layers is None:
        target_layers = [0, 1, 4, 7, 24, 26, 30, 33]

    if tasks is None:
        tasks = {
            "extract": "probes/extract.json",
            "translate": "probes/translate.json",
            "classify": "probes/classify.json",
            "summarize": "probes/summarize.json",
        }

    gates_dir = Path(gates_dir)
    task_graphs: dict[str, Computation] = {}

    for task_name, probe_path in tasks.items():
        probe_set = load_probe_set(probe_path)
        resolved = resolve_probes(probe_set, gates_dir)

        probe_graphs: dict[str, Computation] = {}

        for rp in resolved:
            layer_nodes: dict[str, Computation] = {}

            for layer_idx in target_layers:
                cfg = TaskHeadScanConfig(
                    task=task_name,
                    model=model_name,
                    layer=layer_idx,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    prompt_hash=rp.prompt_hash,
                    prompt_preview=rp.full_prompt[:60],
                    max_new_tokens=max_new_tokens,
                )
                layer_nodes[f"L{layer_idx}"] = TaskHeadScanNode(
                    config=cfg,
                    prompt=rp.full_prompt,
                )

            probe_graphs[f"probe-{rp.probe_id}"] = Graph(
                id=f"probe-{rp.probe_id}",
                children=layer_nodes,
            )

        task_graphs[task_name] = Graph(id=task_name, children=probe_graphs)

    return Graph(id="task-head-scan", children=task_graphs)
