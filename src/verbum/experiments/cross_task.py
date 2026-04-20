"""Cross-task ablation — do the same 3 heads control different tasks?

Tests whether L1:H0, L24:H0, L24:H2 are essential for tasks beyond
lambda compilation: summarization, translation, classification,
relation extraction. If the same heads control multiple tasks,
typed_apply is the universal composition primitive.
"""

from __future__ import annotations

import re
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
    "CrossTaskConfig",
    "CrossTaskNode",
    "build_cross_task",
]


# ─────────────────────────── success detectors ────────────────────────

FRENCH_WORDS = [
    "le ",
    "la ",
    "les ",
    "un ",
    "une ",
    "des ",
    "du ",
    "de ",
    "est ",
    "sont ",
    "qui ",
    "que ",
    "et ",
    "ou ",
    "dans ",
    "chien",
    "chat",
    "court",
    "noir",
    "livre",
    "maison",
]

PREDICATE_PATTERN = re.compile(r"\w+\([^)]+\)")


def detect_compile(text: str) -> bool:
    """Lambda compilation success."""
    return _detect_lambda(text)


def detect_summarize(text: str) -> bool:
    """Summarization success — short output, not a question or reasoning."""
    clean = text.strip()
    if not clean:
        return False
    # Should be shorter than ~100 chars and not start with reasoning
    reasoning = ["okay", "let me", "i need", "so,", "well,"]
    lower = clean.lower()
    if any(lower.startswith(r) for r in reasoning):
        return False
    # Should produce actual content (not empty or just punctuation)
    return len(clean) > 5 and len(clean) < 200


def detect_translate(text: str) -> bool:
    """Translation success — contains French words."""
    lower = text.lower()
    return sum(lower.count(w) for w in FRENCH_WORDS) >= 2


def detect_classify(text: str) -> bool:
    """Classification success — contains positive or negative."""
    lower = text.lower().strip()
    return "positive" in lower or "negative" in lower


def detect_extract(text: str) -> bool:
    """Extraction success — contains predicate notation."""
    return bool(PREDICATE_PATTERN.search(text))


DETECTORS = {
    "compile": detect_compile,
    "summarize": detect_summarize,
    "translate": detect_translate,
    "classify": detect_classify,
    "extract": detect_extract,
}


# ─────────────────────────── computation ──────────────────────────────


class CrossTaskConfig(BaseModel):
    """Config for testing one head on one task."""

    model_config = ConfigDict(frozen=True)

    kind: str = "cross_task"
    task: str  # compile, summarize, translate, classify, extract
    model: str
    ablated_head: list[int] | None  # [layer, head] or None for baseline
    prompt_hash: str
    prompt_preview: str
    max_new_tokens: int = 50


class CrossTaskNode(Computation):
    """Ablate one head, run one task, check task-specific success."""

    def __init__(self, config: CrossTaskConfig, prompt: str) -> None:
        self._config = config
        self._prompt = prompt

    @property
    def config(self) -> CrossTaskConfig:
        return self._config

    def execute(self, ctx: Context) -> dict[str, Any]:
        model = ctx.resources["model"]
        tokenizer = ctx.resources["tokenizer"]
        prompt = self._prompt
        max_new = self._config.max_new_tokens

        if self._config.ablated_head is None:
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

        detector = DETECTORS[self._config.task]
        success = detector(gen)

        return {
            "task": self._config.task,
            "generation": gen,
            "success": success,
            "ablated_head": self._config.ablated_head,
            "is_baseline": self._config.ablated_head is None,
        }


# ─────────────────────────── builder ──────────────────────────────────


def build_cross_task(
    *,
    tasks: dict[str, str] | None = None,
    gates_dir: str | Path = "gates",
    essential_heads: list[tuple[int, int]] | None = None,
    model_name: str = "Qwen/Qwen3-4B",
    max_new_tokens: int = 50,
) -> Graph:
    """Build cross-task ablation experiment.

    ``tasks`` maps task name to probe set path. Defaults to all 5 tasks.
    Tests each essential head + baseline on each task's probes.

    Graph structure::

        Graph("cross-task")
          +-- Graph("compile")
          |     +-- Graph("probe-ga-simple")
          |     |     +-- baseline
          |     |     +-- L1-H0
          |     |     +-- L24-H0
          |     |     +-- L24-H2
          |     +-- ...
          +-- Graph("summarize")
          |     +-- ...
          +-- ...
    """
    if essential_heads is None:
        essential_heads = [(1, 0), (24, 0), (24, 2)]

    if tasks is None:
        tasks = {
            "compile": "probes/gate-ablation.json",
            "summarize": "probes/summarize.json",
            "translate": "probes/translate.json",
            "classify": "probes/classify.json",
            "extract": "probes/extract.json",
        }

    gates_dir = Path(gates_dir)
    task_graphs: dict[str, Computation] = {}

    for task_name, probe_path in tasks.items():
        probe_set = load_probe_set(probe_path)
        resolved = resolve_probes(probe_set, gates_dir)

        probe_graphs: dict[str, Computation] = {}

        for rp in resolved:
            nodes: dict[str, Computation] = {}

            # Baseline
            nodes["baseline"] = CrossTaskNode(
                config=CrossTaskConfig(
                    task=task_name,
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
                nodes[f"L{layer}-H{head}"] = CrossTaskNode(
                    config=CrossTaskConfig(
                        task=task_name,
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

        task_graphs[task_name] = Graph(id=task_name, children=probe_graphs)

    return Graph(id="cross-task", children=task_graphs)
