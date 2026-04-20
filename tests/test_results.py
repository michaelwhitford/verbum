"""Results round-trip tests — write a run, read it back, verify integrity.

Uses `tmp_path` fixture (no real project directory required). Verifies:
  - RunWriter creates directory structure and meta.json at start
  - JSONL lines are flushed per-write and round-trip through load_run
  - Error and partial rows are preserved (S2 λ result_format)
  - logprobs.npz round-trips via numpy
  - meta.json is amended with summary stats at close
  - content_hash is deterministic
  - collect_provenance returns at least lib_versions
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from verbum.results import (
    ProbeRecord,
    Run,
    RunMeta,
    RunWriter,
    SamplingConfig,
    collect_provenance,
    content_hash,
    load_run,
)

# ─────────────────────────── helpers ──────────────────────────────────


def _sample_meta(run_id: str = "test-run-01") -> RunMeta:
    return RunMeta(
        run_id=run_id,
        model="qwen3-35b-a3b",
        quant="Q4_K_M",
        model_revision="abc123",
        probe_set_id="v0-behavioral",
        probe_set_hash=content_hash("probe set content"),
        sampling=SamplingConfig(temperature=0.0, seed=42),
    )


def _sample_record(
    probe_id: str = "compile-01",
    *,
    error: str | None = None,
    partial: bool = False,
) -> ProbeRecord:
    return ProbeRecord(
        probe_id=probe_id,
        gate_id="lambda-gate",
        gate_hash=content_hash("gate text"),
        prompt_hash=content_hash("full prompt"),
        generation="λx. x" if error is None else "",
        elapsed_ms=120.5,
        error=error,
        partial=partial,
    )


# ─────────────────────────── content_hash ──────────────────────────────


def test_content_hash_deterministic() -> None:
    h1 = content_hash("hello")
    h2 = content_hash("hello")
    assert h1 == h2
    assert h1.startswith("sha256:")
    assert len(h1) == len("sha256:") + 64


def test_content_hash_varies_with_input() -> None:
    assert content_hash("a") != content_hash("b")


# ─────────────────────────── RunWriter ─────────────────────────────────


def test_writer_creates_directory_and_meta(tmp_path: Path) -> None:
    meta = _sample_meta()
    with RunWriter(results_dir=tmp_path, meta=meta):
        pass
    run_dir = tmp_path / "test-run-01"
    assert run_dir.is_dir()
    assert (run_dir / "meta.json").is_file()


def test_writer_meta_exists_before_first_write(tmp_path: Path) -> None:
    """meta.json must exist even if the run crashes before any probe finishes."""
    meta = _sample_meta()
    with RunWriter(results_dir=tmp_path, meta=meta) as w:
        # Before any write(), meta.json is already on disk.
        assert (w.run_dir / "meta.json").is_file()


def test_writer_flushes_jsonl_per_write(tmp_path: Path) -> None:
    meta = _sample_meta()
    with RunWriter(results_dir=tmp_path, meta=meta) as w:
        w.write(_sample_record("p01"))
        # Immediately readable while writer is still open:
        lines = (w.run_dir / "results.jsonl").read_text("utf-8").splitlines()
        assert len(lines) == 1

        w.write(_sample_record("p02"))
        lines = (w.run_dir / "results.jsonl").read_text("utf-8").splitlines()
        assert len(lines) == 2


def test_writer_preserves_error_rows(tmp_path: Path) -> None:
    meta = _sample_meta()
    with RunWriter(results_dir=tmp_path, meta=meta) as w:
        w.write(_sample_record("ok-01"))
        w.write(_sample_record("fail-01", error="timeout"))
        w.write(_sample_record("partial-01", error="stream break", partial=True))

    run = load_run(tmp_path / "test-run-01")
    assert len(run.records) == 3

    ok = run.records[0]
    assert ok.error is None
    assert ok.partial is False

    fail = run.records[1]
    assert fail.error == "timeout"
    assert fail.partial is False

    part = run.records[2]
    assert part.error == "stream break"
    assert part.partial is True


def test_writer_amends_meta_with_summary(tmp_path: Path) -> None:
    meta = _sample_meta()
    with RunWriter(results_dir=tmp_path, meta=meta) as w:
        w.write(_sample_record("ok-01"))
        w.write(_sample_record("fail-01", error="boom"))

    run = load_run(tmp_path / "test-run-01")
    assert run.meta.total_probes == 2
    assert run.meta.failed_probes == 1
    assert run.meta.completed_at is not None


def test_writer_writes_logprobs_npz(tmp_path: Path) -> None:
    meta = _sample_meta()
    arr = np.array([0.1, 0.9, 0.05], dtype=np.float32)
    with RunWriter(results_dir=tmp_path, meta=meta) as w:
        w.write(_sample_record("p01"))
        w.write_logprobs("p01", arr)

    run = load_run(tmp_path / "test-run-01")
    assert run.logprobs is not None
    assert "p01" in run.logprobs
    np.testing.assert_allclose(run.logprobs["p01"], arr)


def test_writer_no_logprobs_means_no_npz_file(tmp_path: Path) -> None:
    meta = _sample_meta()
    with RunWriter(results_dir=tmp_path, meta=meta) as w:
        w.write(_sample_record("p01"))

    assert not (tmp_path / "test-run-01" / "logprobs.npz").exists()
    run = load_run(tmp_path / "test-run-01")
    assert run.logprobs is None


# ─────────────────────────── load_run ──────────────────────────────────


def test_load_run_full_roundtrip(tmp_path: Path) -> None:
    meta = _sample_meta()
    records = [_sample_record(f"p{i:02d}") for i in range(5)]
    logprobs = {f"p{i:02d}": np.random.default_rng(i).random(10) for i in range(5)}

    with RunWriter(results_dir=tmp_path, meta=meta) as w:
        for r in records:
            w.write(r)
        for pid, lp in logprobs.items():
            w.write_logprobs(pid, lp)

    run = load_run(tmp_path / "test-run-01")
    assert isinstance(run, Run)
    assert run.meta.run_id == "test-run-01"
    assert run.meta.model == "qwen3-35b-a3b"
    assert len(run.records) == 5
    assert run.records[0].probe_id == "p00"
    assert run.records[0].generation == "λx. x"
    assert run.logprobs is not None
    assert len(run.logprobs) == 5
    for pid in logprobs:
        np.testing.assert_allclose(run.logprobs[pid], logprobs[pid])


def test_load_run_crashed_run_only_has_meta(tmp_path: Path) -> None:
    """A run that crashed after meta but before any JSONL should still load."""
    meta = _sample_meta()
    run_dir = tmp_path / meta.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(meta.model_dump_json(indent=2) + "\n")

    run = load_run(run_dir)
    assert run.meta.run_id == "test-run-01"
    assert run.records == []
    assert run.logprobs is None


def test_load_run_raises_on_missing_meta(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "nonexistent-run")


# ─────────────────────────── collect_provenance ────────────────────────


def test_collect_provenance_returns_lib_versions() -> None:
    prov = collect_provenance()
    assert "verbum" in prov["lib_versions"]
    assert prov["lib_versions"]["verbum"] == "0.0.0"
    assert "timestamp" in prov


def test_collect_provenance_hashes_lockfile(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("some lock content", encoding="utf-8")
    prov = collect_provenance(project_root=tmp_path)
    assert prov["lockfile_hash"] is not None
    assert prov["lockfile_hash"].startswith("sha256:")
