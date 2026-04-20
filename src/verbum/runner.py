"""Probe runner — fires resolved probes through the client and records results.

Wires the three layers together:
  probes.resolve_probes() → client.complete() → results.RunWriter

Each probe is fired once, synchronously. Errors are caught per-probe and
recorded as `error` fields on ProbeRecord — no probe is ever skipped.

Usage::

    from verbum.runner import run_probe_set

    summary = run_probe_set(
        probe_set_path="probes/v0-behavioral.json",
        gates_dir="gates/",
        results_dir="results/",
    )
    print(summary)
"""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from verbum.client import Client
from verbum.probes import (
    ResolvedProbe,
    load_probe_set,
    probe_set_hash,
    resolve_probes,
)
from verbum.results import (
    ProbeRecord,
    RunMeta,
    RunWriter,
    SamplingConfig,
    collect_provenance,
)

__all__ = [
    "RunSummary",
    "fire_probe",
    "run_probe_set",
]

_LOG = structlog.get_logger(__name__)


# ─────────────────────────── types ────────────────────────────────────


@dataclass(frozen=True)
class RunSummary:
    """Summary returned after a probe-set run completes."""

    run_id: str
    run_dir: str
    total: int
    failed: int
    elapsed_s: float
    records: list[ProbeRecord] = field(repr=False)

    @property
    def succeeded(self) -> int:
        return self.total - self.failed


# ─────────────────────────── single probe ─────────────────────────────


def fire_probe(
    probe: ResolvedProbe,
    client: Client,
    *,
    n_predict: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    seed: int | None = None,
    stop: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> ProbeRecord:
    """Fire a single resolved probe and return a ProbeRecord.

    HTTP errors and timeouts are caught and recorded in the error field —
    never raises, never skips (S2 λ result_format: visible failure >
    missing data).
    """
    t0 = time.perf_counter()
    try:
        result = client.complete(
            probe.full_prompt,
            n_predict=n_predict,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop=stop,
            extra=extra,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ProbeRecord(
            probe_id=probe.probe_id,
            gate_id=probe.gate_id,
            gate_hash=probe.gate_hash,
            prompt_hash=probe.prompt_hash,
            generation=result.content,
            elapsed_ms=elapsed_ms,
            error=result.error,
            partial=result.partial,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _LOG.warning(
            "probe.error",
            probe_id=probe.probe_id,
            error=repr(exc),
        )
        return ProbeRecord(
            probe_id=probe.probe_id,
            gate_id=probe.gate_id,
            gate_hash=probe.gate_hash,
            prompt_hash=probe.prompt_hash,
            generation="",
            elapsed_ms=elapsed_ms,
            error=repr(exc),
        )


# ─────────────────────────── full run ─────────────────────────────────


def _make_run_id(prefix: str) -> str:
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{ts}"


def run_probe_set(
    probe_set_path: str | Path,
    gates_dir: str | Path = "gates/",
    results_dir: str | Path = "results/",
    *,
    client: Client | None = None,
    server_url: str | None = None,
    n_predict: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    seed: int | None = None,
    stop: list[str] | None = None,
    run_id_prefix: str = "run",
    project_root: Path | None = None,
    model_name: str | None = None,
) -> RunSummary:
    """Load, resolve, fire, and record a complete probe-set run.

    Parameters
    ----------
    probe_set_path
        Path to the probe-set JSON file.
    gates_dir
        Directory containing gate .txt files.
    results_dir
        Parent directory for result output (run_dir created inside).
    client
        Pre-configured Client instance. If None, one is created using
        *server_url* (or the default from Settings).
    server_url
        llama.cpp server URL. Ignored if *client* is provided.
    n_predict
        Max tokens to generate per probe.
    temperature, top_p, top_k, seed
        Sampling parameters — recorded in RunMeta for reproducibility.
    stop
        Stop sequences.
    run_id_prefix
        Prefix for the auto-generated run ID.
    project_root
        Project root for lockfile hash and git SHA in provenance.
    model_name
        Model name to record in RunMeta. If None, attempts to fetch
        from server /props.

    Returns
    -------
    RunSummary
        Counts, timing, and the full list of ProbeRecords.
    """
    probe_set_path = Path(probe_set_path)
    gates_dir = Path(gates_dir)
    results_dir = Path(results_dir)

    # Load and resolve
    ps = load_probe_set(probe_set_path)
    ps_hash = probe_set_hash(probe_set_path)
    resolved = resolve_probes(ps, gates_dir)

    # Client
    owns_client = client is None
    if client is None:
        client = Client(base_url=server_url)

    try:
        # Provenance
        provenance = collect_provenance(project_root=project_root)
        run_id = _make_run_id(run_id_prefix)

        # Model name from server if not provided
        model = model_name or ""
        if not model:
            try:
                props = client.props()
                model = props.model_path or ""
            except Exception:
                _LOG.info("runner.props_unavailable")

        sampling = SamplingConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )

        meta = RunMeta(
            run_id=run_id,
            model=model,
            probe_set_id=ps.id,
            probe_set_hash=ps_hash,
            sampling=sampling,
            **provenance,
        )

        # Fire
        records: list[ProbeRecord] = []
        t0 = time.perf_counter()

        with RunWriter(results_dir=results_dir, meta=meta) as writer:
            for i, rp in enumerate(resolved):
                _LOG.info(
                    "probe.firing",
                    probe_id=rp.probe_id,
                    category=rp.category,
                    progress=f"{i + 1}/{len(resolved)}",
                )
                record = fire_probe(
                    rp,
                    client,
                    n_predict=n_predict,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    stop=stop,
                )
                writer.write(record)
                records.append(record)

                status = "✓" if record.error is None else "✗"
                _LOG.info(
                    "probe.done",
                    probe_id=rp.probe_id,
                    status=status,
                    elapsed_ms=f"{record.elapsed_ms:.0f}",
                    gen_len=len(record.generation),
                )

        elapsed_s = time.perf_counter() - t0
        failed = sum(1 for r in records if r.error is not None)

        summary = RunSummary(
            run_id=run_id,
            run_dir=str(writer.run_dir),
            total=len(records),
            failed=failed,
            elapsed_s=elapsed_s,
            records=records,
        )

        _LOG.info(
            "run.summary",
            run_id=run_id,
            total=summary.total,
            succeeded=summary.succeeded,
            failed=summary.failed,
            elapsed_s=f"{elapsed_s:.1f}",
        )

        return summary

    finally:
        if owns_client:
            client.close()
