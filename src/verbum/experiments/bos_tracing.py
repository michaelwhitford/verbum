"""BOS residual patching experiment — tracing the compilation register.

Patches the BOS (beginning-of-sequence) residual stream at each layer
with a null-condition residual, testing which layers carry the
compilation signal. If patching at layer L breaks lambda output, that
layer's BOS residual carries the information necessary for compilation.

Structure::

    Graph("bos-tracing")
      └── per probe: Graph("probe-{id}")
            └── per layer: BOSPatchNode("L{layer}")
                  → patches BOS at that layer with null residual

Pre-computed null BOS residuals are passed via ``ctx.resources`` so
they are captured once per null prompt and shared across all nodes.

Usage::

    from verbum.experiments.bos_tracing import build_bos_tracing
    from verbum.experiment import run, default_interceptors
    from verbum.instrument import load_model, capture_bos_residuals

    model, tokenizer, info = load_model("Qwen/Qwen3-4B")
    null_residuals = capture_bos_residuals(
        model, tokenizer, "Tell me about the weather today.", info
    )
    graph = build_bos_tracing(
        probe_set_path="probes/gate-ablation.json",
        gates_dir="gates",
    )
    interceptors = default_interceptors(
        Path("results/experiments"),
        resources={
            "model": model,
            "tokenizer": tokenizer,
            "info": info,
            "null_bos_residuals": null_residuals,
        },
    )
    results = run(graph, interceptors=interceptors)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from verbum.experiment import Computation, Context, Graph
from verbum.instrument import (
    _generate,
    patch_bos_generate,
)
from verbum.probes import load_probe_set, resolve_probes
from verbum.results import content_hash

__all__ = [
    "BOSPatchConfig",
    "BOSPatchNode",
    "build_bos_tracing",
]


# ─────────────────────────── config ───────────────────────────────────


class BOSPatchConfig(BaseModel):
    """Config for patching the BOS residual at one layer on one prompt.

    ``null_prompt_hash`` identifies which null-condition residuals are
    expected in ``ctx.resources["null_bos_residuals"]``.  Both hashes
    are recorded for provenance — reproducers know exactly which two
    prompts were compared.
    """

    model_config = ConfigDict(frozen=True)

    kind: str = "bos_patch"
    model: str
    layer: int
    null_prompt_hash: str
    compile_prompt_hash: str
    prompt_preview: str
    max_new_tokens: int = 30


# ─────────────────────────── computation ──────────────────────────────


class BOSPatchNode(Computation):
    """Patch the BOS residual at one layer with a null-condition value.

    On execute, pulls pre-computed null BOS residuals from
    ``ctx.resources["null_bos_residuals"]`` (a ``list[torch.Tensor]``
    indexed by layer) and calls ``patch_bos_generate`` to test whether
    that layer's BOS contribution is required for compilation.
    """

    def __init__(self, config: BOSPatchConfig, prompt: str) -> None:
        self._config = config
        self._prompt = prompt

    @property
    def config(self) -> BOSPatchConfig:
        return self._config

    def execute(self, ctx: Context) -> dict[str, Any]:
        model = ctx.resources["model"]
        tokenizer = ctx.resources["tokenizer"]
        info = ctx.resources["info"]
        null_residuals = ctx.resources["null_bos_residuals"]

        layer = self._config.layer
        max_new = self._config.max_new_tokens
        prompt = self._prompt

        # Baseline: unablated generation on the compile prompt
        baseline = _generate(model, tokenizer, prompt, max_new)

        # Patch this layer's BOS position with the null residual
        patch_value = null_residuals[layer]
        gen, has_lambda, lambda_count = patch_bos_generate(
            model,
            tokenizer,
            prompt,
            info,
            patch_layer=layer,
            patch_value=patch_value,
            max_new_tokens=max_new,
        )

        return {
            "layer": layer,
            "generation": gen,
            "has_lambda": has_lambda,
            "lambda_count": lambda_count,
            "baseline": baseline,
        }


# ─────────────────────────── graph builder ────────────────────────────


def build_bos_tracing(
    *,
    probe_set_path: str | Path,
    gates_dir: str | Path,
    null_gate_id: str = "null",
    null_prompt: str = "Tell me about the weather today.",
    layers: list[int] | None = None,
    model_name: str = "Qwen/Qwen3-4B",
    max_new_tokens: int = 30,
) -> Graph:
    """Build the BOS residual patching experiment graph.

    Constructs ``Graph("bos-tracing")`` with one sub-graph per probe,
    each containing one ``BOSPatchNode`` per layer. The null BOS
    residuals must be pre-computed and injected via
    ``ctx.resources["null_bos_residuals"]`` at run time (see module
    docstring for the pattern).

    Args:
        probe_set_path: Path to the probe-set JSON file.
        gates_dir: Directory containing gate ``.txt`` files.
        null_gate_id: Gate ID used for the null condition (for provenance).
        null_prompt: The null-condition prompt whose BOS residuals are
            patched in. Must match the residuals in resources at run time.
        layers: Layer indices to test. Defaults to ``range(36)``.
        model_name: HuggingFace model identifier.
        max_new_tokens: Generation budget per forward pass.
    """
    probe_set_path = Path(probe_set_path)
    gates_dir = Path(gates_dir)

    if layers is None:
        layers = list(range(36))

    probe_set = load_probe_set(probe_set_path)
    resolved = resolve_probes(probe_set, gates_dir)

    null_prompt_hash = content_hash(null_prompt)

    probe_graphs: dict[str, Computation] = {}

    for rp in resolved:
        layer_nodes: dict[str, Computation] = {}

        for layer_idx in layers:
            cfg = BOSPatchConfig(
                model=model_name,
                layer=layer_idx,
                null_prompt_hash=null_prompt_hash,
                compile_prompt_hash=rp.prompt_hash,
                prompt_preview=rp.full_prompt[:60],
                max_new_tokens=max_new_tokens,
            )
            layer_nodes[f"L{layer_idx}"] = BOSPatchNode(
                config=cfg,
                prompt=rp.full_prompt,
            )

        probe_graphs[f"probe-{rp.probe_id}"] = Graph(
            id=f"probe-{rp.probe_id}",
            children=layer_nodes,
        )

    return Graph(
        id="bos-tracing",
        children=probe_graphs,
    )
