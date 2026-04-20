"""Runner tests — mocked HTTP transport, no real server.

Verifies:
  - Successful run with multiple probes produces correct records
  - Error on one probe doesn't abort the run; error field is populated
  - RunMeta provenance is populated (run_id, probe_set_id, sampling)
  - Results directory contains meta.json + results.jsonl after run
  - ProbeRecord fields match resolved probe provenance fields
  - fire_probe catches exceptions and returns error records
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from verbum.client import Client
from verbum.probes import load_probe_set, probe_set_hash
from verbum.results import content_hash, load_run
from verbum.runner import RunSummary, fire_probe, run_probe_set

# ─────────────────────────── helpers ──────────────────────────────────


def _setup_probe_env(tmp_path: Path, *, n_probes: int = 3) -> tuple[Path, Path, Path]:
    """Create gates, probe-set JSON, and results dirs under tmp_path."""
    gates = tmp_path / "gates"
    gates.mkdir()
    (gates / "compile.txt").write_text(
        "You are a lambda compiler.\n\nInput: ", encoding="utf-8"
    )
    (gates / "null.txt").write_text(
        "You are a helpful assistant.\n\nInput: ", encoding="utf-8"
    )

    probes_dir = tmp_path / "probes"
    probes_dir.mkdir()
    probe_list = []
    for i in range(n_probes):
        p = {
            "id": f"p{i:02d}",
            "category": "compile" if i < n_probes - 1 else "null",
            "prompt": f"Sentence {i}",
            "ground_truth": f"λx. x{i}",
        }
        if p["category"] == "null":
            p["gate"] = "null"
        probe_list.append(p)

    ps_data = {
        "id": "test-set",
        "version": 1,
        "description": "test",
        "created": "2026-01-01T00:00:00Z",
        "author": "test",
        "default_gate": "compile",
        "probes": probe_list,
    }
    ps_path = probes_dir / "test.json"
    ps_path.write_text(json.dumps(ps_data, indent=2), encoding="utf-8")

    results = tmp_path / "results"
    results.mkdir()

    return ps_path, gates, results


def _mock_transport(
    *, completion_content: str = "λx. x", fail_on_probe: str | None = None
) -> httpx.MockTransport:
    """Build a MockTransport that handles /props and /completion."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "model_path": "/models/test.gguf",
                    "n_ctx": 8192,
                    "default_generation_settings": {},
                },
            )
        if request.url.path == "/completion":
            call_count["n"] += 1
            body = json.loads(request.content)
            # Check if this probe should fail
            if fail_on_probe and fail_on_probe in body.get("prompt", ""):
                return httpx.Response(500, json={"error": "server error"})
            return httpx.Response(
                200,
                json={
                    "content": completion_content,
                    "stop": True,
                    "tokens_predicted": 4,
                    "tokens_evaluated": 20,
                },
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


# ─────────────────────────── fire_probe ───────────────────────────────


class TestFireProbe:
    def test_successful_fire(self, tmp_path: Path) -> None:
        ps_path, gates, _results = _setup_probe_env(tmp_path, n_probes=2)
        ps = load_probe_set(ps_path)

        from verbum.probes import resolve_probes

        resolved = resolve_probes(ps, gates)
        rp = resolved[0]  # first probe is always "compile" category

        transport = _mock_transport(completion_content="λx. x")
        with Client(base_url="http://srv", transport=transport) as c:
            record = fire_probe(rp, c, n_predict=64)

        assert record.probe_id == "p00"
        assert record.gate_id == "compile"
        assert record.gate_hash == content_hash("You are a lambda compiler.\n\nInput: ")
        assert record.generation == "λx. x"
        assert record.error is None
        assert record.elapsed_ms > 0

    def test_fire_catches_http_error(self, tmp_path: Path) -> None:
        ps_path, gates, _results = _setup_probe_env(tmp_path, n_probes=2)
        ps = load_probe_set(ps_path)

        from verbum.probes import resolve_probes

        resolved = resolve_probes(ps, gates)
        rp = resolved[0]  # compile category

        transport = httpx.MockTransport(
            lambda _: httpx.Response(500, json={"error": "boom"})
        )
        with Client(base_url="http://srv", transport=transport) as c:
            record = fire_probe(rp, c, n_predict=64)

        assert record.probe_id == "p00"
        assert record.error is not None
        assert record.generation == ""
        assert record.elapsed_ms > 0


# ─────────────────────────── run_probe_set ────────────────────────────


class TestRunProbeSet:
    def test_successful_run(self, tmp_path: Path) -> None:
        ps_path, gates, results = _setup_probe_env(tmp_path, n_probes=3)
        transport = _mock_transport(completion_content="λy. y")

        with Client(base_url="http://srv", transport=transport) as c:
            summary = run_probe_set(
                ps_path,
                gates_dir=gates,
                results_dir=results,
                client=c,
                n_predict=64,
            )

        assert isinstance(summary, RunSummary)
        assert summary.total == 3
        assert summary.failed == 0
        assert summary.succeeded == 3
        assert summary.elapsed_s > 0
        assert len(summary.records) == 3

        # Check records have correct fields
        for rec in summary.records:
            assert rec.generation == "λy. y"
            assert rec.error is None
            assert rec.gate_hash.startswith("sha256:")
            assert rec.prompt_hash.startswith("sha256:")

    def test_run_creates_result_directory(self, tmp_path: Path) -> None:
        ps_path, gates, results = _setup_probe_env(tmp_path, n_probes=2)
        transport = _mock_transport()

        with Client(base_url="http://srv", transport=transport) as c:
            summary = run_probe_set(
                ps_path,
                gates_dir=gates,
                results_dir=results,
                client=c,
            )

        run_dir = Path(summary.run_dir)
        assert run_dir.is_dir()
        assert (run_dir / "meta.json").is_file()
        assert (run_dir / "results.jsonl").is_file()

    def test_run_meta_has_provenance(self, tmp_path: Path) -> None:
        ps_path, gates, results = _setup_probe_env(tmp_path, n_probes=1)
        transport = _mock_transport()

        with Client(base_url="http://srv", transport=transport) as c:
            summary = run_probe_set(
                ps_path,
                gates_dir=gates,
                results_dir=results,
                client=c,
                project_root=tmp_path,
            )

        loaded = load_run(summary.run_dir)
        meta = loaded.meta
        assert meta.run_id == summary.run_id
        assert meta.probe_set_id == "test-set"
        assert meta.probe_set_hash == probe_set_hash(ps_path)
        assert meta.model == "/models/test.gguf"
        assert meta.sampling.temperature == 0.0
        assert meta.total_probes == 1
        assert meta.failed_probes == 0
        assert meta.completed_at is not None

    def test_run_records_roundtrip(self, tmp_path: Path) -> None:
        ps_path, gates, results = _setup_probe_env(tmp_path, n_probes=3)
        transport = _mock_transport(completion_content="result text")

        with Client(base_url="http://srv", transport=transport) as c:
            summary = run_probe_set(
                ps_path,
                gates_dir=gates,
                results_dir=results,
                client=c,
            )

        loaded = load_run(summary.run_dir)
        assert len(loaded.records) == 3
        for rec in loaded.records:
            assert rec.generation == "result text"

    def test_run_with_error_continues(self, tmp_path: Path) -> None:
        """One probe fails; rest still fire and are recorded."""
        ps_path, gates, results = _setup_probe_env(tmp_path, n_probes=3)
        # Probe p01 has "Sentence 1" in its prompt
        transport = _mock_transport(fail_on_probe="Sentence 1")

        with Client(base_url="http://srv", transport=transport) as c:
            summary = run_probe_set(
                ps_path,
                gates_dir=gates,
                results_dir=results,
                client=c,
            )

        assert summary.total == 3
        assert summary.failed == 1
        assert summary.succeeded == 2

        # The failed record has error
        failed = [r for r in summary.records if r.error is not None]
        assert len(failed) == 1
        assert failed[0].probe_id == "p01"
        assert failed[0].generation == ""

        # Successful records are fine
        ok = [r for r in summary.records if r.error is None]
        assert len(ok) == 2

    def test_run_with_custom_sampling(self, tmp_path: Path) -> None:
        ps_path, gates, results = _setup_probe_env(tmp_path, n_probes=1)
        transport = _mock_transport()

        with Client(base_url="http://srv", transport=transport) as c:
            summary = run_probe_set(
                ps_path,
                gates_dir=gates,
                results_dir=results,
                client=c,
                temperature=0.7,
                seed=42,
            )

        loaded = load_run(summary.run_dir)
        assert loaded.meta.sampling.temperature == 0.7
        assert loaded.meta.sampling.seed == 42

    def test_run_probe_ids_unique_in_records(self, tmp_path: Path) -> None:
        ps_path, gates, results = _setup_probe_env(tmp_path, n_probes=5)
        transport = _mock_transport()

        with Client(base_url="http://srv", transport=transport) as c:
            summary = run_probe_set(
                ps_path,
                gates_dir=gates,
                results_dir=results,
                client=c,
            )

        probe_ids = [r.probe_id for r in summary.records]
        assert len(probe_ids) == len(set(probe_ids))
