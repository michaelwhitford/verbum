"""Fractal experiment framework tests.

Verifies:
  - Computation: config_hash determinism, variance, execute
  - Graph: child execution, dependency ordering, config_hash composition,
    dep validation, fractal nesting (Graph-in-Graph)
  - topological_sort: no deps, linear chain, diamond, cycle detection,
    deterministic ordering
  - Serialization: JSON roundtrip, numpy roundtrip, non-dict wrapping,
    empty dir
  - CacheInterceptor: miss-then-hit, idempotent (no recompute on hit),
    different configs get different caches
  - ProvenanceInterceptor: records timing and node metadata
  - default_interceptors: factory, full pipeline integration
  - Fractal caching: Graph with CacheInterceptor caches at every level
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from verbum.experiment import (
    CacheInterceptor,
    Computation,
    Context,
    Graph,
    ProvenanceInterceptor,
    ResourceInterceptor,
    default_interceptors,
    load_result,
    run,
    save_result,
    topological_sort,
)

# ─────────────────────────── test computations ────────────────────────


class AddConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    a: int
    b: int


class Add(Computation):
    """Leaf: returns sum of two numbers."""

    def __init__(self, a: int, b: int) -> None:
        self._cfg = AddConfig(a=a, b=b)

    @property
    def config(self) -> AddConfig:
        return self._cfg

    def execute(self, ctx: Context) -> dict[str, Any]:
        return {"sum": self._cfg.a + self._cfg.b}


class MultiplyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    factor: int


class Multiply(Computation):
    """Leaf: multiplies a dep result by a factor."""

    def __init__(self, factor: int) -> None:
        self._cfg = MultiplyConfig(factor=factor)

    @property
    def config(self) -> MultiplyConfig:
        return self._cfg

    def execute(self, ctx: Context) -> dict[str, Any]:
        input_sum = ctx.deps["add"]["sum"]
        return {"product": input_sum * self._cfg.factor}


class ResourceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    key: str


class ResourceReader(Computation):
    """Leaf: reads a value from resources."""

    def __init__(self, key: str) -> None:
        self._cfg = ResourceConfig(key=key)

    @property
    def config(self) -> ResourceConfig:
        return self._cfg

    def execute(self, ctx: Context) -> dict[str, Any]:
        return {"value": ctx.resources[self._cfg.key]}


class CounterConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str


class Counter(Computation):
    """Leaf that increments a shared counter (for testing cache bypass)."""

    def __init__(self, id: str) -> None:
        self._cfg = CounterConfig(id=id)

    @property
    def config(self) -> CounterConfig:
        return self._cfg

    def execute(self, ctx: Context) -> dict[str, Any]:
        counter = ctx.resources.get("call_counter", {})
        count = counter.get(self._cfg.id, 0) + 1
        counter[self._cfg.id] = count
        return {"call_count": count, "id": self._cfg.id}


# ─────────────────────────── Computation ──────────────────────────────


class TestComputation:
    def test_config_hash_deterministic(self) -> None:
        a1 = Add(3, 5)
        a2 = Add(3, 5)
        assert a1.config_hash == a2.config_hash

    def test_config_hash_varies(self) -> None:
        assert Add(3, 5).config_hash != Add(3, 6).config_hash

    def test_config_hash_prefix(self) -> None:
        assert Add(1, 2).config_hash.startswith("sha256:")

    def test_execute_returns_result(self) -> None:
        ctx = Context(node_id="test", config_hash="x")
        result = Add(3, 5).execute(ctx)
        assert result == {"sum": 8}

    def test_run_without_interceptors(self) -> None:
        result = run(Add(10, 20))
        assert result == {"sum": 30}


# ─────────────────────────── Graph ────────────────────────────────────


class TestGraph:
    def test_graph_executes_children(self) -> None:
        graph = Graph(
            id="simple",
            children={"a": Add(1, 2), "b": Add(3, 4)},
        )
        result = run(graph)
        assert result["a"] == {"sum": 3}
        assert result["b"] == {"sum": 7}

    def test_graph_respects_deps(self) -> None:
        graph = Graph(
            id="dep-test",
            children={
                "add": Add(2, 3),
                "mul": Multiply(factor=10),
            },
            deps={"mul": ("add",)},
        )
        result = run(graph)
        assert result["add"] == {"sum": 5}
        assert result["mul"] == {"product": 50}

    def test_graph_config_hash_changes_with_children(self) -> None:
        g1 = Graph(id="g", children={"a": Add(1, 2)})
        g2 = Graph(id="g", children={"a": Add(1, 3)})
        assert g1.config_hash != g2.config_hash

    def test_graph_config_hash_deterministic(self) -> None:
        g1 = Graph(id="g", children={"a": Add(1, 2), "b": Add(3, 4)})
        g2 = Graph(id="g", children={"a": Add(1, 2), "b": Add(3, 4)})
        assert g1.config_hash == g2.config_hash

    def test_graph_config_hash_changes_with_id(self) -> None:
        g1 = Graph(id="alpha", children={"a": Add(1, 2)})
        g2 = Graph(id="beta", children={"a": Add(1, 2)})
        assert g1.config_hash != g2.config_hash

    def test_graph_rejects_invalid_dep_source(self) -> None:
        with pytest.raises(ValueError, match="Dep source"):
            Graph(
                id="bad",
                children={"a": Add(1, 2)},
                deps={"nonexistent": ("a",)},
            )

    def test_graph_rejects_invalid_dep_target(self) -> None:
        with pytest.raises(ValueError, match="Dep target"):
            Graph(
                id="bad",
                children={"a": Add(1, 2)},
                deps={"a": ("nonexistent",)},
            )

    def test_fractal_nested_graphs(self) -> None:
        """A Graph containing a Graph — same shape at every level."""
        inner = Graph(
            id="inner",
            children={"a": Add(1, 2), "b": Add(3, 4)},
        )
        outer = Graph(
            id="outer",
            children={"g1": inner, "c": Add(5, 6)},
        )
        result = run(outer)
        assert result["g1"]["a"] == {"sum": 3}
        assert result["g1"]["b"] == {"sum": 7}
        assert result["c"] == {"sum": 11}

    def test_fractal_three_levels(self) -> None:
        """Three levels deep — the shape holds."""
        leaf = Add(1, 1)
        level1 = Graph(id="L1", children={"x": leaf})
        level2 = Graph(id="L2", children={"inner": level1})
        level3 = Graph(id="L3", children={"mid": level2})

        result = run(level3)
        assert result["mid"]["inner"]["x"] == {"sum": 2}

    def test_graph_child_node_ids_are_namespaced(self) -> None:
        """Child node_ids should be parent/child path."""
        captured_ids: list[str] = []

        class SpyConfig(BaseModel):
            model_config = ConfigDict(frozen=True)
            label: str

        class Spy(Computation):
            def __init__(self, label: str) -> None:
                self._cfg = SpyConfig(label=label)

            @property
            def config(self) -> SpyConfig:
                return self._cfg

            def execute(self, ctx: Context) -> dict[str, Any]:
                captured_ids.append(ctx.node_id)
                return {"id": ctx.node_id}

        graph = Graph(id="parent", children={"child_a": Spy("a"), "child_b": Spy("b")})
        run(graph, node_id="root")

        assert any("child_a" in nid for nid in captured_ids)
        assert any("child_b" in nid for nid in captured_ids)


# ─────────────────────────── topological_sort ─────────────────────────


class TestTopologicalSort:
    def test_no_deps(self) -> None:
        result = topological_sort(["a", "b", "c"], {})
        assert set(result) == {"a", "b", "c"}

    def test_linear_chain(self) -> None:
        result = topological_sort(
            ["a", "b", "c"],
            {"b": ("a",), "c": ("b",)},
        )
        assert result.index("a") < result.index("b") < result.index("c")

    def test_diamond(self) -> None:
        result = topological_sort(
            ["a", "b", "c", "d"],
            {"b": ("a",), "c": ("a",), "d": ("b", "c")},
        )
        assert result.index("a") < result.index("b")
        assert result.index("a") < result.index("c")
        assert result.index("b") < result.index("d")
        assert result.index("c") < result.index("d")

    def test_cycle_raises(self) -> None:
        with pytest.raises(ValueError, match="Cycle"):
            topological_sort(["a", "b"], {"a": ("b",), "b": ("a",)})

    def test_deterministic_order(self) -> None:
        """Same-depth nodes are sorted alphabetically."""
        r1 = topological_sort(["c", "a", "b"], {})
        r2 = topological_sort(["b", "c", "a"], {})
        assert r1 == r2
        assert r1 == ["a", "b", "c"]

    def test_single_node(self) -> None:
        assert topological_sort(["only"], {}) == ["only"]

    def test_self_cycle(self) -> None:
        with pytest.raises(ValueError, match="Cycle"):
            topological_sort(["a"], {"a": ("a",)})


# ─────────────────────────── serialization ────────────────────────────


class TestSerialization:
    def test_json_roundtrip(self, tmp_path: Path) -> None:
        data = {"a": 1, "b": "hello", "c": [1, 2, 3]}
        save_result(data, tmp_path)
        loaded = load_result(tmp_path)
        assert loaded == data

    def test_numpy_roundtrip(self, tmp_path: Path) -> None:
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data: dict[str, Any] = {"values": arr, "label": "test"}
        save_result(data, tmp_path)
        loaded = load_result(tmp_path)
        assert loaded["label"] == "test"
        np.testing.assert_allclose(loaded["values"], arr)

    def test_mixed_json_and_numpy(self, tmp_path: Path) -> None:
        arr = np.zeros((3, 3), dtype=np.float64)
        data: dict[str, Any] = {"matrix": arr, "name": "identity", "size": 3}
        save_result(data, tmp_path)

        # Both files created
        assert (tmp_path / "result.json").is_file()
        assert (tmp_path / "result.npz").is_file()

        loaded = load_result(tmp_path)
        assert loaded["name"] == "identity"
        assert loaded["size"] == 3
        np.testing.assert_allclose(loaded["matrix"], arr)

    def test_non_dict_wrapped(self, tmp_path: Path) -> None:
        save_result(42, tmp_path)
        loaded = load_result(tmp_path)
        assert loaded["_value"] == 42

    def test_empty_dir_returns_empty_dict(self, tmp_path: Path) -> None:
        assert load_result(tmp_path) == {}

    def test_nested_dict_roundtrip(self, tmp_path: Path) -> None:
        """Nested structures survive JSON roundtrip."""
        data = {
            "results": [
                {"layer": 0, "head": 3, "has_lambda": True},
                {"layer": 0, "head": 7, "has_lambda": False},
            ],
            "summary": {"total": 2, "broken": 1},
        }
        save_result(data, tmp_path)
        loaded = load_result(tmp_path)
        assert loaded == data


# ─────────────────────────── CacheInterceptor ─────────────────────────


class TestCacheInterceptor:
    def test_miss_then_hit(self, tmp_path: Path) -> None:
        cache = CacheInterceptor(tmp_path)
        prov = ProvenanceInterceptor()
        interceptors = (cache, prov)

        add = Add(1, 2)

        # First run: cache miss
        r1 = run(add, interceptors=interceptors)
        assert r1 == {"sum": 3}

        # Result dir exists with meta.json and result.json
        result_dir = tmp_path / add.config_hash
        assert (result_dir / "meta.json").is_file()
        assert (result_dir / "result.json").is_file()

        # Second run: cache hit — same result
        r2 = run(add, interceptors=interceptors)
        assert r2 == {"sum": 3}

    def test_idempotent_no_recompute(self, tmp_path: Path) -> None:
        """Computation does not re-execute on cache hit."""
        call_counter: dict[str, int] = {}

        cache = CacheInterceptor(tmp_path)
        prov = ProvenanceInterceptor()
        resource = ResourceInterceptor({"call_counter": call_counter})
        interceptors = (cache, prov, resource)

        counter = Counter(id="test")

        # First run — executes
        r1 = run(counter, interceptors=interceptors)
        assert r1["call_count"] == 1
        assert call_counter["test"] == 1

        # Second run — cached, counter NOT incremented
        r2 = run(counter, interceptors=interceptors)
        assert r2["call_count"] == 1  # cached result
        assert call_counter["test"] == 1  # not called again

    def test_different_config_different_cache(self, tmp_path: Path) -> None:
        cache = CacheInterceptor(tmp_path)
        prov = ProvenanceInterceptor()
        interceptors = (cache, prov)

        r1 = run(Add(1, 2), interceptors=interceptors)
        r2 = run(Add(3, 4), interceptors=interceptors)

        assert r1 == {"sum": 3}
        assert r2 == {"sum": 7}

        # Both cached separately
        assert (tmp_path / Add(1, 2).config_hash / "meta.json").is_file()
        assert (tmp_path / Add(3, 4).config_hash / "meta.json").is_file()

    def test_meta_json_has_completed_at(self, tmp_path: Path) -> None:
        cache = CacheInterceptor(tmp_path)
        prov = ProvenanceInterceptor()
        interceptors = (cache, prov)

        add = Add(7, 8)
        run(add, interceptors=interceptors)

        meta = json.loads((tmp_path / add.config_hash / "meta.json").read_text("utf-8"))
        assert "completed_at" in meta

    def test_corrupt_cache_triggers_recompute(self, tmp_path: Path) -> None:
        """Corrupt meta.json should not prevent recomputation."""
        add = Add(99, 1)
        result_dir = tmp_path / add.config_hash
        result_dir.mkdir(parents=True)
        (result_dir / "meta.json").write_text("not valid json", encoding="utf-8")

        cache = CacheInterceptor(tmp_path)
        prov = ProvenanceInterceptor()
        interceptors = (cache, prov)

        result = run(add, interceptors=interceptors)
        assert result == {"sum": 100}


# ─────────────────────────── ProvenanceInterceptor ────────────────────


class TestProvenanceInterceptor:
    def test_records_timing(self, tmp_path: Path) -> None:
        cache = CacheInterceptor(tmp_path)
        prov = ProvenanceInterceptor()
        interceptors = (cache, prov)

        add = Add(1, 2)
        run(add, interceptors=interceptors)

        meta = json.loads((tmp_path / add.config_hash / "meta.json").read_text("utf-8"))
        assert "started_at" in meta
        assert "elapsed_ms" in meta
        assert "completed_at" in meta
        assert meta["elapsed_ms"] >= 0
        # Internal monotonic timer should NOT leak to meta.json
        assert "_start_monotonic" not in meta

    def test_records_node_identity(self, tmp_path: Path) -> None:
        cache = CacheInterceptor(tmp_path)
        prov = ProvenanceInterceptor()
        interceptors = (cache, prov)

        add = Add(2, 3)
        run(add, interceptors=interceptors, node_id="my-node")

        meta = json.loads((tmp_path / add.config_hash / "meta.json").read_text("utf-8"))
        assert meta["node_id"] == "my-node"
        assert meta["config_hash"] == add.config_hash


# ─────────────────────────── ResourceInterceptor ──────────────────────


class TestResourceInterceptor:
    def test_injects_resources(self) -> None:
        resource = ResourceInterceptor({"model": "test-model"})
        interceptors = (resource,)

        result = run(ResourceReader(key="model"), interceptors=interceptors)
        assert result == {"value": "test-model"}

    def test_multiple_resources(self) -> None:
        resource = ResourceInterceptor({"a": 1, "b": 2})
        interceptors = (resource,)

        result = run(ResourceReader(key="a"), interceptors=interceptors)
        assert result == {"value": 1}


# ─────────────────────────── default_interceptors ─────────────────────


class TestDefaultInterceptors:
    def test_factory_without_resources(self, tmp_path: Path) -> None:
        chain = default_interceptors(tmp_path)
        assert isinstance(chain, tuple)
        assert len(chain) == 3  # Log, Cache, Provenance

    def test_factory_with_resources(self, tmp_path: Path) -> None:
        chain = default_interceptors(tmp_path, resources={"model": "test"})
        assert len(chain) == 4  # Log, Cache, Provenance, Resource

    def test_full_pipeline(self, tmp_path: Path) -> None:
        chain = default_interceptors(tmp_path)
        result = run(Add(10, 20), interceptors=chain)
        assert result == {"sum": 30}

    def test_full_pipeline_with_caching(self, tmp_path: Path) -> None:
        """Full chain: compute, cache, then hit on second call."""
        call_counter: dict[str, int] = {}
        chain = default_interceptors(tmp_path, resources={"call_counter": call_counter})

        counter = Counter(id="full")
        r1 = run(counter, interceptors=chain)
        assert r1["call_count"] == 1

        r2 = run(counter, interceptors=chain)
        assert r2["call_count"] == 1  # cached
        assert call_counter["full"] == 1  # not re-executed


# ─────────────────────────── fractal caching ──────────────────────────


class TestFractalCaching:
    def test_graph_children_cached_individually(self, tmp_path: Path) -> None:
        """Each child in a graph gets its own cache entry."""
        cache = CacheInterceptor(tmp_path)
        prov = ProvenanceInterceptor()
        interceptors = (cache, prov)

        a = Add(1, 1)
        b = Add(2, 2)
        graph = Graph(id="pair", children={"a": a, "b": b})

        run(graph, interceptors=interceptors)

        # Children cached individually
        assert (tmp_path / a.config_hash / "meta.json").is_file()
        assert (tmp_path / b.config_hash / "meta.json").is_file()
        # Graph also cached
        assert (tmp_path / graph.config_hash / "meta.json").is_file()

    def test_nested_graph_all_levels_cached(self, tmp_path: Path) -> None:
        """Fractal: every level of nesting gets cached."""
        cache = CacheInterceptor(tmp_path)
        prov = ProvenanceInterceptor()
        interceptors = (cache, prov)

        leaf = Add(5, 5)
        inner = Graph(id="inner", children={"x": leaf})
        outer = Graph(id="outer", children={"g": inner})

        run(outer, interceptors=interceptors)

        # All three levels cached
        assert (tmp_path / leaf.config_hash / "meta.json").is_file()
        assert (tmp_path / inner.config_hash / "meta.json").is_file()
        assert (tmp_path / outer.config_hash / "meta.json").is_file()

    def test_cached_graph_skips_children(self, tmp_path: Path) -> None:
        """On cache hit, a Graph's children are NOT re-executed."""
        call_counter: dict[str, int] = {}

        cache = CacheInterceptor(tmp_path)
        prov = ProvenanceInterceptor()
        resource = ResourceInterceptor({"call_counter": call_counter})
        interceptors = (cache, prov, resource)

        graph = Graph(
            id="counting",
            children={"c1": Counter(id="a"), "c2": Counter(id="b")},
        )

        # First run: both children execute
        run(graph, interceptors=interceptors)
        assert call_counter == {"a": 1, "b": 1}

        # Second run: graph cached, children NOT re-executed
        run(graph, interceptors=interceptors)
        assert call_counter == {"a": 1, "b": 1}  # unchanged
