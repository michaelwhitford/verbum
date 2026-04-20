"""Probe-set model, loading, and resolution tests.

Uses ``tmp_path`` for gate files and probe-set JSON. Verifies:
  - Probe and ProbeSet model validation (happy + error paths)
  - Gate loading and hashing (happy + missing file)
  - ProbeSet loading from JSON (happy + invalid)
  - Hash determinism
  - Resolved probe construction (gate content injection)
  - Default gate fallback (probe without gate uses set default)
  - Gate caching (same gate loaded once per resolve call)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from verbum.probes import (
    Gate,
    Probe,
    ProbeSet,
    ResolvedProbe,
    gate_hash,
    load_gate,
    load_probe_set,
    probe_set_hash,
    resolve_probes,
)
from verbum.results import content_hash

# ─────────────────────────── fixtures ─────────────────────────────────


@pytest.fixture
def gates_dir(tmp_path: Path) -> Path:
    """Create a gates directory with two gate files."""
    d = tmp_path / "gates"
    d.mkdir()
    (d / "compile.txt").write_text("You are a lambda compiler.\n", encoding="utf-8")
    (d / "null.txt").write_text("You are a helpful assistant.\n", encoding="utf-8")
    return d


def _make_probe_set_dict(
    *,
    n_probes: int = 3,
    default_gate: str = "compile",
    override_gate: str | None = None,
) -> dict:
    """Build a valid probe-set dict for JSON serialization."""
    probes = []
    for i in range(n_probes):
        p: dict = {
            "id": f"p{i:02d}",
            "category": "compile" if i % 2 == 0 else "null",
            "prompt": f"Translate: sentence {i}",
            "ground_truth": f"λx. x{i}",
        }
        if override_gate is not None and i == 0:
            p["gate"] = override_gate
        probes.append(p)
    return {
        "id": "v0-test",
        "version": 1,
        "description": "Test probe set",
        "created": "2026-04-16T00:00:00Z",
        "author": "test",
        "default_gate": default_gate,
        "probes": probes,
    }


def _write_probe_set(tmp_path: Path, data: dict, name: str = "test.json") -> Path:
    """Write a probe-set dict to a JSON file and return its path."""
    p = tmp_path / name
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


# ─────────────────────────── Probe model ──────────────────────────────


class TestProbeModel:
    def test_valid_probe(self) -> None:
        p = Probe(
            id="p01",
            category="compile",
            prompt="hello",
            ground_truth="λx. x",
        )
        assert p.id == "p01"
        assert p.category == "compile"
        assert p.gate is None
        assert p.metadata == {}

    def test_probe_with_gate_override(self) -> None:
        p = Probe(
            id="p01",
            category="compile",
            gate="null",
            prompt="hello",
            ground_truth="λx. x",
        )
        assert p.gate == "null"

    def test_probe_with_metadata(self) -> None:
        p = Probe(
            id="p01",
            category="compile",
            prompt="hello",
            ground_truth="λx. x",
            metadata={"difficulty": "easy", "source": "manual"},
        )
        assert p.metadata["difficulty"] == "easy"

    def test_probe_extensible_category(self) -> None:
        """Category is any string, not just compile/decompile/null."""
        p = Probe(
            id="p01",
            category="custom-category",
            prompt="hello",
            ground_truth="something",
        )
        assert p.category == "custom-category"

    def test_probe_extra_fields_allowed(self) -> None:
        """extra='allow' preserves unknown fields."""
        p = Probe(
            id="p01",
            category="compile",
            prompt="hello",
            ground_truth="λx. x",
            notes="extra field",  # type: ignore[call-arg]  # ty: ignore[unknown-argument]
        )
        assert p.model_dump()["notes"] == "extra field"

    def test_probe_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            Probe(id="p01", category="compile")  # type: ignore[call-arg]  # ty: ignore[missing-argument]


# ─────────────────────────── ProbeSet model ───────────────────────────


class TestProbeSetModel:
    def test_valid_probe_set(self) -> None:
        ps = ProbeSet(
            id="v0",
            default_gate="compile",
            probes=[
                Probe(
                    id="p01",
                    category="compile",
                    prompt="hello",
                    ground_truth="world",
                )
            ],
        )
        assert ps.id == "v0"
        assert ps.default_gate == "compile"
        assert len(ps.probes) == 1
        assert ps.version == 1
        assert ps.description == ""

    def test_probe_set_missing_default_gate(self) -> None:
        with pytest.raises(ValidationError):
            ProbeSet(id="v0")  # type: ignore[call-arg]  # ty: ignore[missing-argument]

    def test_probe_set_empty_probes_ok(self) -> None:
        ps = ProbeSet(id="v0", default_gate="compile")
        assert ps.probes == []


# ─────────────────────────── gate loading ─────────────────────────────


class TestGateLoading:
    def test_load_gate_happy(self, gates_dir: Path) -> None:
        gate = load_gate("compile", gates_dir)
        assert isinstance(gate, Gate)
        assert gate.id == "compile"
        assert gate.content == "You are a lambda compiler.\n"
        assert gate.hash.startswith("sha256:")

    def test_load_gate_hash_matches_content_hash(self, gates_dir: Path) -> None:
        gate = load_gate("compile", gates_dir)
        expected = content_hash("You are a lambda compiler.\n")
        assert gate.hash == expected

    def test_load_gate_missing_file(self, gates_dir: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_gate("nonexistent", gates_dir)

    def test_gate_hash_function(self, gates_dir: Path) -> None:
        h = gate_hash("compile", gates_dir)
        assert h == content_hash("You are a lambda compiler.\n")

    def test_gate_is_frozen(self, gates_dir: Path) -> None:
        gate = load_gate("compile", gates_dir)
        with pytest.raises(ValidationError):
            gate.id = "changed"  # type: ignore[misc]


# ─────────────────────────── probe-set loading ────────────────────────


class TestProbeSetLoading:
    def test_load_probe_set_happy(self, tmp_path: Path) -> None:
        data = _make_probe_set_dict()
        path = _write_probe_set(tmp_path, data)
        ps = load_probe_set(path)
        assert ps.id == "v0-test"
        assert len(ps.probes) == 3
        assert ps.probes[0].id == "p00"

    def test_load_probe_set_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_probe_set(tmp_path / "nope.json")

    def test_load_probe_set_invalid_json(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_probe_set(bad)

    def test_load_probe_set_missing_required(self, tmp_path: Path) -> None:
        """A probe set JSON without 'id' or 'default_gate' should fail."""
        bad = tmp_path / "bad.json"
        bad.write_text('{"description": "no id or gate"}', encoding="utf-8")
        with pytest.raises(ValidationError):
            load_probe_set(bad)


# ─────────────────────────── probe-set hash ───────────────────────────


class TestProbeSetHash:
    def test_hash_deterministic(self, tmp_path: Path) -> None:
        data = _make_probe_set_dict()
        path = _write_probe_set(tmp_path, data)
        h1 = probe_set_hash(path)
        h2 = probe_set_hash(path)
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_hash_varies_with_content(self, tmp_path: Path) -> None:
        p1 = _write_probe_set(tmp_path, _make_probe_set_dict(n_probes=1), "a.json")
        p2 = _write_probe_set(tmp_path, _make_probe_set_dict(n_probes=2), "b.json")
        assert probe_set_hash(p1) != probe_set_hash(p2)


# ─────────────────────────── resolve_probes ───────────────────────────


class TestResolveProbes:
    def test_resolve_basic(self, tmp_path: Path, gates_dir: Path) -> None:
        data = _make_probe_set_dict()
        path = _write_probe_set(tmp_path, data)
        ps = load_probe_set(path)
        resolved = resolve_probes(ps, gates_dir)

        assert len(resolved) == 3
        rp = resolved[0]
        assert isinstance(rp, ResolvedProbe)
        assert rp.probe_id == "p00"
        assert rp.gate_id == "compile"
        assert rp.gate_content == "You are a lambda compiler.\n"
        expected = "You are a lambda compiler.\nTranslate: sentence 0"
        assert rp.full_prompt == expected
        assert rp.prompt_hash == content_hash(rp.full_prompt)
        assert rp.ground_truth == "λx. x0"

    def test_resolve_gate_override(self, tmp_path: Path, gates_dir: Path) -> None:
        """Probe with its own gate overrides the set default."""
        data = _make_probe_set_dict(override_gate="null")
        path = _write_probe_set(tmp_path, data)
        ps = load_probe_set(path)
        resolved = resolve_probes(ps, gates_dir)

        # First probe should use the "null" gate
        assert resolved[0].gate_id == "null"
        assert resolved[0].gate_content == "You are a helpful assistant.\n"

        # Others should use default "compile" gate
        assert resolved[1].gate_id == "compile"
        assert resolved[2].gate_id == "compile"

    def test_resolve_missing_gate_raises(self, tmp_path: Path, gates_dir: Path) -> None:
        data = _make_probe_set_dict(default_gate="nonexistent")
        path = _write_probe_set(tmp_path, data)
        ps = load_probe_set(path)
        with pytest.raises(FileNotFoundError):
            resolve_probes(ps, gates_dir)

    def test_resolve_preserves_metadata(self, tmp_path: Path, gates_dir: Path) -> None:
        data = _make_probe_set_dict(n_probes=1)
        data["probes"][0]["metadata"] = {"difficulty": "hard"}
        path = _write_probe_set(tmp_path, data)
        ps = load_probe_set(path)
        resolved = resolve_probes(ps, gates_dir)
        assert resolved[0].metadata == {"difficulty": "hard"}

    def test_resolve_empty_probe_set(self, tmp_path: Path, gates_dir: Path) -> None:
        data = _make_probe_set_dict(n_probes=0)
        path = _write_probe_set(tmp_path, data)
        ps = load_probe_set(path)
        resolved = resolve_probes(ps, gates_dir)
        assert resolved == []

    def test_resolved_probe_is_frozen(self, tmp_path: Path, gates_dir: Path) -> None:
        data = _make_probe_set_dict(n_probes=1)
        path = _write_probe_set(tmp_path, data)
        ps = load_probe_set(path)
        resolved = resolve_probes(ps, gates_dir)
        with pytest.raises(ValidationError):
            resolved[0].probe_id = "changed"  # type: ignore[misc]
