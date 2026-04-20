"""Result writing and reading — the S2 membrane.

Every measurement crosses this boundary. Canonical form per AGENTS.md:

    results/<run_id>/
    ├── meta.json          — self-sufficient provenance (S2 λ run_provenance)
    ├── results.jsonl       — one line per probe, streamable (S2 λ result_format)
    └── logprobs.npz        — np.savez_compressed, keyed by probe_id

Design principles:
- `meta.json` is written at run-start so it exists even on crash.
  Amended at close with summary stats (counts, completed_at).
- JSONL is flushed after every line — each written row is durable.
- `error ≠ null` partitions failed rows; `partial: true` flags
  broken-stream rows. Never skip a line; visible failure > missing data.
- `logprobs.npz` is written only at close. If the run crashes,
  logprobs for that run are lost — JSONL is the record of truth.
- `collect_provenance()` auto-gathers lib versions, lockfile hash,
  git SHA, and timestamp at call time (¬inferred_later).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ProbeRecord",
    "Run",
    "RunMeta",
    "RunWriter",
    "SamplingConfig",
    "collect_provenance",
    "content_hash",
    "load_run",
]

_LOG = structlog.get_logger(__name__)


# ─────────────────────────── models ───────────────────────────────────


class SamplingConfig(BaseModel):
    """Sampling parameters recorded per run for reproducibility."""

    model_config = ConfigDict(extra="allow")

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int | None = None
    grammar: str | None = None


class RunMeta(BaseModel):
    """Self-sufficient provenance sidecar (S2 λ run_provenance).

    Every field flagged as 'must_record' in the AGENTS.md spec is present.
    Written at run-start; amended at close with summary stats.
    """

    model_config = ConfigDict(extra="allow")

    # identity
    run_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )

    # model
    model: str = ""
    quant: str | None = None
    model_revision: str | None = None  # HF revision hash or GGUF SHA

    # environment
    lib_versions: dict[str, str] = Field(default_factory=dict)
    lockfile_hash: str | None = None
    git_sha: str | None = None

    # probe set
    probe_set_id: str = ""
    probe_set_hash: str | None = None

    # sampling
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)

    # summary (populated at close)
    completed_at: str | None = None
    total_probes: int | None = None
    failed_probes: int | None = None


class ProbeRecord(BaseModel):
    """One JSONL line — one probe's result.

    Schema from AGENTS.md S2 λ result_format:
    `{probe_id, gate_id, gate_hash, prompt_hash, generation, elapsed_ms, error}`

    Plus verbum extension `partial` for broken-stream rows.
    """

    model_config = ConfigDict(extra="allow")

    probe_id: str
    gate_id: str
    gate_hash: str
    prompt_hash: str
    generation: str
    elapsed_ms: float
    error: str | None = None  # null ≡ success
    partial: bool = False  # verbum extension: broken-stream row


# ─────────────────────────── helpers ──────────────────────────────────


def content_hash(text: str) -> str:
    """SHA-256 of UTF-8 bytes, prefixed ``sha256:``.

    Canonical hash for gate content, prompt content, and probe set files.
    """
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def collect_provenance(*, project_root: Path | None = None) -> dict[str, Any]:
    """Auto-gather reproducibility metadata at call time.

    Returns a dict suitable for unpacking into `RunMeta(**provenance)`.
    Fields that can't be determined are omitted (caller overrides).

    Per S2 λ run_provenance: ``recorded_at_write_time ¬inferred_later``.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    # lib versions
    lib_versions: dict[str, str] = {}
    for pkg in (
        "verbum",
        "httpx",
        "httpx-sse",
        "pydantic",
        "numpy",
        "structlog",
        "polars",
    ):
        try:
            lib_versions[pkg] = pkg_version(pkg)
        except PackageNotFoundError:
            pass

    # lockfile hash
    lockfile_hash: str | None = None
    if project_root is not None:
        lock = project_root / "uv.lock"
        if lock.is_file():
            lockfile_hash = content_hash(lock.read_text("utf-8"))

    # git SHA
    git_sha: str | None = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(project_root) if project_root else None,
        )
        if result.returncode == 0:
            git_sha = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "lib_versions": lib_versions,
        "lockfile_hash": lockfile_hash,
        "git_sha": git_sha,
    }


# ─────────────────────────── writer ───────────────────────────────────


class RunWriter:
    """Context-managed writer for a single run's result directory.

    Usage::

        meta = RunMeta(run_id="...", model="...", probe_set_id="...", ...)
        with RunWriter(results_dir=Path("results"), meta=meta) as w:
            w.write(ProbeRecord(probe_id="p01", ...))
            w.write_logprobs("p01", np.array([...]))
        # meta.json amended with summary; logprobs.npz written; JSONL flushed.
    """

    def __init__(self, results_dir: Path, meta: RunMeta) -> None:
        self._results_dir = Path(results_dir)
        self._meta = meta
        self._run_dir = self._results_dir / meta.run_id
        self._jsonl_path = self._run_dir / "results.jsonl"
        self._meta_path = self._run_dir / "meta.json"
        self._npz_path = self._run_dir / "logprobs.npz"

        self._logprobs: dict[str, np.ndarray] = {}
        self._jsonl_file = None
        self._count = 0
        self._errors = 0

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    # lifecycle ---------------------------------------------------------

    def __enter__(self) -> RunWriter:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        # Write meta.json immediately — exists even on crash.
        self._write_meta()
        self._jsonl_file = self._jsonl_path.open("a", encoding="utf-8")
        _LOG.info(
            "run.started",
            run_id=self._meta.run_id,
            run_dir=str(self._run_dir),
        )
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        # Flush JSONL
        if self._jsonl_file is not None and not self._jsonl_file.closed:
            self._jsonl_file.close()

        # Write logprobs if any accumulated
        if self._logprobs:
            # ty false-positive: probe-ID keys can't collide with allow_pickle
            np.savez_compressed(str(self._npz_path), **self._logprobs)  # ty: ignore[invalid-argument-type]
            _LOG.info(
                "logprobs.written", path=str(self._npz_path), keys=len(self._logprobs)
            )

        # Amend meta.json with summary
        self._meta.completed_at = datetime.datetime.now(datetime.UTC).isoformat()
        self._meta.total_probes = self._count
        self._meta.failed_probes = self._errors
        self._write_meta()

        _LOG.info(
            "run.completed",
            run_id=self._meta.run_id,
            total=self._count,
            failed=self._errors,
        )

    # writing -----------------------------------------------------------

    def write(self, record: ProbeRecord) -> None:
        """Append one probe record to results.jsonl. Flushed immediately."""
        if self._jsonl_file is None or self._jsonl_file.closed:
            raise RuntimeError("RunWriter is not open; use as context manager.")
        line = record.model_dump_json()
        self._jsonl_file.write(line + "\n")
        self._jsonl_file.flush()
        self._count += 1
        if record.error is not None:
            self._errors += 1

    def write_logprobs(self, probe_id: str, logprobs: np.ndarray) -> None:
        """Buffer logprobs for a probe. Written to npz at close."""
        self._logprobs[probe_id] = logprobs

    # internal ----------------------------------------------------------

    def _write_meta(self) -> None:
        self._meta_path.write_text(
            self._meta.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )


# ─────────────────────────── reader ───────────────────────────────────


class Run(NamedTuple):
    """Loaded result directory — meta, JSONL records, optional logprobs."""

    meta: RunMeta
    records: list[ProbeRecord]
    logprobs: dict[str, np.ndarray] | None


def load_run(run_dir: Path | str) -> Run:
    """Read a result directory back into memory.

    Raises `FileNotFoundError` if `meta.json` is missing.
    JSONL and logprobs are optional (a crashed run may only have meta).
    """
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    jsonl_path = run_dir / "results.jsonl"
    npz_path = run_dir / "logprobs.npz"

    meta = RunMeta.model_validate_json(meta_path.read_text("utf-8"))

    records: list[ProbeRecord] = []
    if jsonl_path.is_file():
        for line in jsonl_path.read_text("utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(ProbeRecord.model_validate(json.loads(line)))

    logprobs: dict[str, np.ndarray] | None = None
    if npz_path.is_file():
        npz = np.load(str(npz_path))
        logprobs = {k: npz[k] for k in npz.files}

    return Run(meta=meta, records=records, logprobs=logprobs)
