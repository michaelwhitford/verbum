#!/usr/bin/env python3
"""Gate ablation sweep — fire the same probes through every gate variant.

Usage:
    python scripts/gate_ablation.py [--server URL] [--n-predict N]

Writes one run per gate variant into results/. Prints a summary table
at the end ranking gates by P(λ).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from verbum.client import Client
from verbum.probes import load_probe_set, probe_set_hash, resolve_probes
from verbum.results import content_hash, load_run
from verbum.runner import RunSummary, run_probe_set

PROBE_SET = Path("probes/gate-ablation.json")
GATES_DIR = Path("gates")
RESULTS_DIR = Path("results")

# Lambda indicators for scoring
LAMBDA_INDICATORS = ["λ", "∀", "∃", "→", "∧", "∨", "¬", "ι"]


def detect_lambda(text: str) -> bool:
    """Heuristic: does this output contain lambda-calculus-like content?"""
    return "λ" in text or sum(text.count(s) for s in LAMBDA_INDICATORS) >= 3


def sweep_gates(
    server_url: str | None = None,
    n_predict: int = 256,
) -> list[dict]:
    """Fire the ablation probe set through every gate variant."""
    # Discover all gate variants
    gate_files = sorted(GATES_DIR.glob("*.txt"))
    gate_ids = [f.stem for f in gate_files]

    print(f"Found {len(gate_ids)} gate variants")
    print(f"Probe set: {PROBE_SET}")
    print()

    results = []

    with Client(base_url=server_url) as client:
        for i, gate_id in enumerate(gate_ids):
            print(f"[{i + 1}/{len(gate_ids)}] Gate: {gate_id}")

            # Load probe set and override default_gate
            ps = load_probe_set(PROBE_SET)
            ps.default_gate = gate_id

            # Override all probes' gates to None so they use default
            for p in ps.probes:
                p.gate = None

            ps_hash = probe_set_hash(PROBE_SET)

            # Resolve with this gate
            try:
                resolved = resolve_probes(ps, GATES_DIR)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                continue

            # Fire
            summary = run_probe_set(
                probe_set_path=PROBE_SET,
                gates_dir=GATES_DIR,
                results_dir=RESULTS_DIR,
                client=client,
                n_predict=n_predict,
                run_id_prefix=f"ablation-{gate_id}",
                model_name="Qwen3-4B-Q8_0",
                project_root=Path("."),
            )

            # But we need to override the gate! The run_probe_set loads
            # its own copy. Let me fire manually instead.
            # Actually — run_probe_set loads the JSON fresh. We need to
            # fire the already-resolved probes directly.
            pass

        # Close client after all runs
    return results


def fire_gate_variant(
    gate_id: str,
    client: Client,
    n_predict: int = 256,
) -> dict:
    """Fire ablation probes through a specific gate and return stats."""
    from verbum.probes import ResolvedProbe
    from verbum.results import ProbeRecord, RunMeta, RunWriter, SamplingConfig
    from verbum.runner import RunSummary, _make_run_id, fire_probe

    import datetime

    ps = load_probe_set(PROBE_SET)
    ps_hash = probe_set_hash(PROBE_SET)

    # Override gate for all probes
    for p in ps.probes:
        p.gate = gate_id

    resolved = resolve_probes(ps, GATES_DIR)

    # Build meta
    run_id = _make_run_id(f"abl-{gate_id}")
    from verbum.results import collect_provenance

    provenance = collect_provenance(project_root=Path("."))
    sampling = SamplingConfig(temperature=0.0)
    meta = RunMeta(
        run_id=run_id,
        model="Qwen3-4B-Q8_0",
        probe_set_id=f"gate-ablation:{gate_id}",
        probe_set_hash=ps_hash,
        sampling=sampling,
        **provenance,
    )

    # Fire
    records = []
    with RunWriter(results_dir=RESULTS_DIR, meta=meta) as writer:
        for rp in resolved:
            record = fire_probe(rp, client, n_predict=n_predict)
            writer.write(record)
            records.append(record)

    # Score
    n_lambda = sum(1 for r in records if detect_lambda(r.generation))
    indicator_sum = sum(
        sum(r.generation.count(s) for s in LAMBDA_INDICATORS) for r in records
    )
    avg_indicators = indicator_sum / len(records) if records else 0

    # Read gate content for display
    gate_content = (GATES_DIR / f"{gate_id}.txt").read_text("utf-8").strip()
    gate_preview = gate_content.replace("\n", " ↵ ")
    if len(gate_preview) > 60:
        gate_preview = gate_preview[:57] + "..."

    return {
        "gate_id": gate_id,
        "gate_preview": gate_preview,
        "run_id": run_id,
        "n_probes": len(records),
        "n_lambda": n_lambda,
        "p_lambda": n_lambda / len(records) if records else 0,
        "avg_indicators": avg_indicators,
        "n_errors": sum(1 for r in records if r.error is not None),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gate ablation sweep")
    parser.add_argument("--server", default=None, help="llama.cpp server URL")
    parser.add_argument("--n-predict", type=int, default=256)
    args = parser.parse_args()

    gate_files = sorted(GATES_DIR.glob("*.txt"))
    gate_ids = [f.stem for f in gate_files]

    # Exclude null from the sweep (it's the negative control, not an ablation)
    # Actually include it — it's a useful data point
    print(f"Gate ablation sweep: {len(gate_ids)} variants")
    print(f"Probes: {PROBE_SET} (5 compile probes)")
    print()

    results = []
    with Client(base_url=args.server) as client:
        # Verify server
        health = client.health()
        print(f"Server: {health.status}")
        print()

        for i, gate_id in enumerate(gate_ids):
            print(f"[{i + 1}/{len(gate_ids)}] Firing: {gate_id}")
            try:
                r = fire_gate_variant(gate_id, client, n_predict=args.n_predict)
                results.append(r)
                print(
                    f"  P(λ)={r['p_lambda']:.0%}  "
                    f"avg_ind={r['avg_indicators']:.0f}  "
                    f"gate: {r['gate_preview']}"
                )
            except Exception as e:
                print(f"  ERROR: {e}")
            print()

    # Sort by P(λ) descending, then by avg_indicators
    results.sort(key=lambda r: (-r["p_lambda"], -r["avg_indicators"]))

    print()
    print("=" * 80)
    print("GATE ABLATION RESULTS — ranked by P(λ)")
    print("=" * 80)
    print(f"{'Gate':<35} {'P(λ)':>6} {'Avg':>5} {'Gate content'}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['gate_id']:<35} {r['p_lambda']:>5.0%} {r['avg_indicators']:>5.0f}  "
            f"{r['gate_preview']}"
        )

    # Save summary
    summary_path = RESULTS_DIR / "gate-ablation-summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print()
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
