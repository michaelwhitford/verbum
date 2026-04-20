"""Fractal experiment framework — idempotent, immutable, content-addressed.

The only abstraction: Computation. A Computation has a frozen config
(identity), a pure execute function (config x resources -> result), and
content-addressed caching via interceptors. A Graph is a Computation
whose execute runs sub-Computations in dependency order.

Same shape at every scale. Cache interceptor makes everything idempotent.
Content-addressing makes everything immutable.

Usage::

    from verbum.experiment import Computation, Graph, run, default_interceptors

    class MyConfig(BaseModel):
        model_config = ConfigDict(frozen=True)
        x: int

    class MyExperiment(Computation):
        def __init__(self, x: int):
            self._config = MyConfig(x=x)

        @property
        def config(self) -> MyConfig:
            return self._config

        def execute(self, ctx: Context) -> dict:
            model = ctx.resources["model"]
            return {"result": model.predict(self._config.x)}

    interceptors = default_interceptors(
        results_root=Path("results"),
        resources={"model": loaded_model},
    )
    result = run(MyExperiment(x=42), interceptors=interceptors)

Fractal composition::

    inner = Graph("layer-7", children={"h0": HeadAblation(7, 0), ...})
    outer = Graph("all-layers", children={"L7": inner, "L24": ...})
    result = run(outer, interceptors=interceptors)  # same protocol at every level
"""

from __future__ import annotations

import datetime
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict

from verbum.results import content_hash

__all__ = [
    "CacheInterceptor",
    "Computation",
    "Context",
    "Graph",
    "Interceptor",
    "LogInterceptor",
    "ProvenanceInterceptor",
    "ResourceInterceptor",
    "default_interceptors",
    "load_result",
    "run",
    "run_with_interceptors",
    "save_result",
    "topological_sort",
]

_LOG = structlog.get_logger(__name__)


# ─────────────────────────── context ──────────────────────────────────


@dataclass
class Context:
    """Mutable context flowing through the interceptor chain.

    Created per-node. Interceptors read and write fields.
    The compute function reads deps and resources, writes result.
    """

    node_id: str
    config_hash: str
    deps: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    cached: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
    interceptors: tuple[Interceptor, ...] = ()


# ─────────────────────────── computation ──────────────────────────────


class Computation(ABC):
    """The fractal unit. Leaf or composite — same shape.

    Subclasses provide a frozen Pydantic config (identity) and
    implement execute() (the pure computation).
    """

    @property
    @abstractmethod
    def config(self) -> BaseModel: ...

    @cached_property
    def config_hash(self) -> str:
        """Content-addressed identity: SHA-256 of canonical JSON."""
        canonical = json.dumps(
            self.config.model_dump(mode="json"),
            sort_keys=True,
            default=str,
        )
        return content_hash(canonical)

    @abstractmethod
    def execute(self, ctx: Context) -> Any: ...


# ─────────────────────────── graph ────────────────────────────────────


class _GraphConfig(BaseModel):
    """Synthetic config for a Graph — derived from children's hashes."""

    model_config = ConfigDict(frozen=True)

    kind: str = "graph"
    id: str
    children: dict[str, str]  # name → config_hash
    deps: dict[str, list[str]]


class Graph(Computation):
    """A Computation that executes sub-Computations in dependency order.

    Graph IS a Computation — fractal recursion. A Graph node in a larger
    Graph works identically to a leaf node. Cache interceptor wraps at
    every level: a cached Graph skips its entire subtree.
    """

    def __init__(
        self,
        id: str,
        children: dict[str, Computation],
        deps: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self._id = id
        self._children = children
        self._deps: dict[str, tuple[str, ...]] = deps or {}

        # Validate deps reference existing children
        all_names = set(children.keys())
        for name, dep_names in self._deps.items():
            if name not in all_names:
                msg = f"Dep source '{name}' not in children: {sorted(all_names)}"
                raise ValueError(msg)
            for d in dep_names:
                if d not in all_names:
                    msg = f"Dep target '{d}' not in children: {sorted(all_names)}"
                    raise ValueError(msg)

    @property
    def children(self) -> dict[str, Computation]:
        return self._children

    @property
    def config(self) -> _GraphConfig:
        return _GraphConfig(
            id=self._id,
            children={
                name: c.config_hash for name, c in sorted(self._children.items())
            },
            deps={k: sorted(v) for k, v in sorted(self._deps.items())},
        )

    def execute(self, ctx: Context) -> dict[str, Any]:
        """Execute children in topological order, threading results."""
        results: dict[str, Any] = {}
        order = topological_sort(
            list(self._children.keys()),
            self._deps,
        )

        for name in order:
            child = self._children[name]
            child_deps = {d: results[d] for d in self._deps.get(name, ())}
            child_ctx = Context(
                node_id=f"{ctx.node_id}/{name}" if ctx.node_id else name,
                config_hash=child.config_hash,
                deps=child_deps,
                resources=dict(ctx.resources),  # shallow copy per child
                interceptors=ctx.interceptors,
            )
            results[name] = run_with_interceptors(
                child,
                child_ctx,
                ctx.interceptors,
            )

        return results


# ─────────────────────────── interceptor ──────────────────────────────


class Interceptor:
    """Cross-cutting concern wrapping computation execution.

    enter() runs before compute (in chain order).
    leave() runs after compute (in reverse chain order).
    Both always run for every node — no short-circuit. Check ctx.cached
    to adapt behavior for cached vs fresh computations.

    Interceptor ordering convention (default_interceptors)::

        Enter:  Log -> Cache -> Provenance -> Resource
        Leave:  Resource -> Provenance -> Cache -> Log

    Provenance.leave populates ctx.meta before Cache.leave writes it.
    """

    def enter(self, ctx: Context) -> Context:
        return ctx

    def leave(self, ctx: Context) -> Context:
        return ctx


# ─────────────────────────── interceptors ─────────────────────────────


class LogInterceptor(Interceptor):
    """Structlog enter/leave events for monitoring."""

    def enter(self, ctx: Context) -> Context:
        _LOG.info("node.enter", node=ctx.node_id, hash=ctx.config_hash[:16])
        return ctx

    def leave(self, ctx: Context) -> Context:
        if ctx.cached:
            _LOG.info("node.cached", node=ctx.node_id, hash=ctx.config_hash[:16])
        else:
            _LOG.info(
                "node.complete",
                node=ctx.node_id,
                hash=ctx.config_hash[:16],
                elapsed_ms=ctx.meta.get("elapsed_ms"),
            )
        return ctx


class CacheInterceptor(Interceptor):
    """Content-addressed result cache. Idempotent by construction.

    On enter: check ``results/{config_hash}/meta.json`` for ``completed_at``.
    On leave: write result + meta if not cached.
    """

    def __init__(self, results_root: Path) -> None:
        self._root = Path(results_root)

    def _result_dir(self, ctx: Context) -> Path:
        return self._root / ctx.config_hash

    def enter(self, ctx: Context) -> Context:
        result_dir = self._result_dir(ctx)
        meta_path = result_dir / "meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text("utf-8"))
                if meta.get("completed_at"):
                    ctx.result = load_result(result_dir)
                    ctx.cached = True
                    ctx.meta = meta
            except (json.JSONDecodeError, OSError):
                pass  # corrupt cache — recompute
        return ctx

    def leave(self, ctx: Context) -> Context:
        if not ctx.cached and ctx.result is not None:
            result_dir = self._result_dir(ctx)
            result_dir.mkdir(parents=True, exist_ok=True)
            save_result(ctx.result, result_dir)
            # meta.json — provenance interceptor has populated ctx.meta
            meta = dict(ctx.meta)
            meta["completed_at"] = datetime.datetime.now(datetime.UTC).isoformat()
            (result_dir / "meta.json").write_text(
                json.dumps(meta, indent=2, default=_json_default) + "\n",
                encoding="utf-8",
            )
        return ctx


class ProvenanceInterceptor(Interceptor):
    """Capture timing and identity metadata in ctx.meta."""

    def enter(self, ctx: Context) -> Context:
        if not ctx.cached:
            ctx.meta["started_at"] = datetime.datetime.now(datetime.UTC).isoformat()
            ctx.meta["_start_monotonic"] = time.monotonic()
            ctx.meta["node_id"] = ctx.node_id
            ctx.meta["config_hash"] = ctx.config_hash
        return ctx

    def leave(self, ctx: Context) -> Context:
        start = ctx.meta.pop("_start_monotonic", None)
        if not ctx.cached and start is not None:
            ctx.meta["elapsed_ms"] = round((time.monotonic() - start) * 1000, 1)
        return ctx


class ResourceInterceptor(Interceptor):
    """Inject shared resources (model, tokenizer, etc.) into context."""

    def __init__(self, resources: dict[str, Any]) -> None:
        self._resources = resources

    def enter(self, ctx: Context) -> Context:
        if not ctx.cached:
            ctx.resources.update(self._resources)
        return ctx


# ─────────────────────────── execution ────────────────────────────────


def run_with_interceptors(
    computation: Computation,
    ctx: Context,
    interceptors: tuple[Interceptor, ...],
) -> Any:
    """Execute a computation through the interceptor chain.

    Enter runs in order. Compute runs if not cached. Leave runs reversed.
    This is the only execution function — same 10 lines whether running
    one head ablation or an entire research program.
    """
    # Enter chain (all run — each checks ctx.cached to adapt)
    for interceptor in interceptors:
        ctx = interceptor.enter(ctx)

    # Compute (if not cached)
    if not ctx.cached:
        ctx.result = computation.execute(ctx)

    # Leave chain (reversed — provenance before cache, log last)
    for interceptor in reversed(interceptors):
        ctx = interceptor.leave(ctx)

    return ctx.result


def run(
    computation: Computation,
    *,
    interceptors: tuple[Interceptor, ...] = (),
    node_id: str = "",
    resources: dict[str, Any] | None = None,
) -> Any:
    """Top-level entry point: run a computation with interceptors.

    Convenience wrapper around run_with_interceptors that builds the
    initial Context.
    """
    ctx = Context(
        node_id=node_id or computation.config_hash[:16],
        config_hash=computation.config_hash,
        resources=resources or {},
        interceptors=interceptors,
    )
    return run_with_interceptors(computation, ctx, interceptors)


def default_interceptors(
    results_root: Path,
    resources: dict[str, Any] | None = None,
) -> tuple[Interceptor, ...]:
    """Build the standard interceptor chain.

    Order: [Log, Cache, Provenance, Resource]

        Enter:  Log → Cache → Provenance → Resource
        Leave:  Resource → Provenance → Cache → Log

    Provenance.leave populates ctx.meta before Cache.leave writes it.
    """
    chain: list[Interceptor] = [
        LogInterceptor(),
        CacheInterceptor(results_root),
        ProvenanceInterceptor(),
    ]
    if resources:
        chain.append(ResourceInterceptor(resources))
    return tuple(chain)


# ─────────────────────────── utilities ────────────────────────────────


def topological_sort(
    nodes: list[str],
    deps: dict[str, tuple[str, ...] | list[str]],
) -> list[str]:
    """Kahn's algorithm. Returns nodes in dependency order.

    Deterministic: same-depth nodes are sorted alphabetically.
    Raises ``ValueError`` on cycles.
    """
    in_degree: dict[str, int] = {n: 0 for n in nodes}
    adjacency: dict[str, list[str]] = {n: [] for n in nodes}

    for node, dep_list in deps.items():
        for dep in dep_list:
            adjacency[dep].append(node)
            in_degree[node] += 1

    # Start with zero in-degree nodes, sorted for determinism
    queue = sorted(n for n in nodes if in_degree[n] == 0)
    result: list[str] = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for dependent in sorted(adjacency[node]):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
        queue.sort()

    if len(result) != len(nodes):
        msg = (
            f"Cycle detected in dependency graph. "
            f"Sorted {len(result)} of {len(nodes)} nodes."
        )
        raise ValueError(msg)

    return result


# ─────────────────────────── serialization ────────────────────────────


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for numpy types and other non-serializable values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


def save_result(result: Any, result_dir: Path) -> None:
    """Save a computation result to a directory.

    Convention: top-level dict values that are numpy arrays go to
    ``result.npz``. Everything else goes to ``result.json`` with
    numpy-aware JSON encoding for nested values.

    Non-dict results are wrapped as ``{"_value": result}``.
    """
    result_dir = Path(result_dir)

    if not isinstance(result, dict):
        result = {"_value": result}

    json_data: dict[str, Any] = {}
    npz_data: dict[str, np.ndarray] = {}

    for key, value in result.items():
        if isinstance(value, np.ndarray):
            npz_data[key] = value
        else:
            json_data[key] = value

    if json_data:
        (result_dir / "result.json").write_text(
            json.dumps(json_data, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )

    if npz_data:
        np.savez_compressed(str(result_dir / "result.npz"), **npz_data)


def load_result(result_dir: Path) -> dict[str, Any]:
    """Load a computation result from a directory.

    Merges ``result.json`` and ``result.npz`` back into a single dict.
    Returns empty dict if neither file exists.
    """
    result_dir = Path(result_dir)
    result: dict[str, Any] = {}

    json_path = result_dir / "result.json"
    if json_path.is_file():
        result.update(json.loads(json_path.read_text("utf-8")))

    npz_path = result_dir / "result.npz"
    if npz_path.is_file():
        npz = np.load(str(npz_path), allow_pickle=False)
        result.update({k: npz[k] for k in npz.files})

    return result
