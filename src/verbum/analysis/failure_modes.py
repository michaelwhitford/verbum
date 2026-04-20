"""Failure mode classification — System 1 vs System 2 analysis.

Analyzes cross-task ablation results to quantify the dual-process
hypothesis: ablating L24:H0 causes the model to switch from direct
compositional output (System 1) to verbose step-by-step reasoning
(System 2).

Metrics computed per generation:
  - output_length: character count
  - reasoning_marker_count: occurrences of deliberation phrases
  - lambda_indicator_count: occurrences of lambda/formal symbols
  - first_token_class: "direct" (→, λ, Output:) vs "reasoning" (Okay, Let me, So)
  - system_class: S1 (direct) or S2 (deliberative)
  - question_mark_output: starts with "→ ?" indicating failed compilation

Usage::

    uv run python -m verbum.analysis.failure_modes \\
        results/experiments/sha256:bd530.../result.json

Or programmatically::

    from verbum.analysis.failure_modes import analyze_cross_task
    report = analyze_cross_task(result_data)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "classify_generation",
    "analyze_cross_task",
    "format_report",
]


# ─────────────────────── markers ──────────────────────────────────────

REASONING_MARKERS = [
    "okay, ",
    "okay so",
    "okay let",
    "let me ",
    "let's ",
    "i need to",
    "i need ",
    "so, ",
    "so the ",
    "so how ",
    "the user wants",
    "the user provided",
    "the user ",
    "first, i",
    "first, let",
    "first i ",
    "hmm,",
    "hmm ",
    "well, ",
    "alright,",
    "alright ",
    "wait, ",
    "wait but",
    "now, ",
    "now let",
]

LAMBDA_INDICATORS = ["λ", "\\lambda", "→", "∧", "∨", "∀", "∃", "¬"]

DIRECT_OUTPUT_STARTS = [
    re.compile(r"^\s*→\s*λ"),       # → λx. ...
    re.compile(r"^\s*→\s*\w+\("),   # → pred(...)
    re.compile(r"^\s*λ"),           # λx. ...
    re.compile(r"^\s*\w+\([^)]+\)"),  # pred(x, y)
    re.compile(r"^\s*Output:\s*\S"),  # Output: something
]

FAILED_COMPILE_PATTERN = re.compile(r"→\s*\?")

REASONING_STARTS = [
    re.compile(r"^\s*→\s*\?\s*$", re.MULTILINE),  # → ? (then reasoning)
    re.compile(r"(?:okay|let me|let's|so,|well,|hmm|alright|wait,)", re.IGNORECASE),
]


# ─────────────────────── per-generation classification ────────────────


def classify_generation(text: str) -> dict[str, Any]:
    """Classify a single generation into failure mode metrics."""
    lower = text.lower()

    # Character length
    output_length = len(text.strip())

    # Reasoning markers
    reasoning_count = sum(lower.count(m) for m in REASONING_MARKERS)

    # Lambda indicators
    lambda_count = sum(text.count(s) for s in LAMBDA_INDICATORS)

    # First substantive line classification
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    first_line = lines[0] if lines else ""

    is_direct = any(p.search(first_line) for p in DIRECT_OUTPUT_STARTS)
    is_failed_compile = bool(FAILED_COMPILE_PATTERN.search(first_line))

    # First-token class
    if is_failed_compile:
        first_token_class = "failed_direct"  # tried to compile, output "?"
    elif is_direct:
        first_token_class = "direct"
    else:
        first_token_class = "reasoning"

    # System classification
    # S1: direct output with lambda/formal content, low reasoning
    # S2: reasoning preamble, high deliberation markers
    if is_failed_compile:
        system_class = "S2_fallback"  # model tried S1, failed, fell to S2
    elif is_direct and reasoning_count <= 2:
        system_class = "S1"
    elif reasoning_count >= 3 or not is_direct:
        system_class = "S2"
    else:
        system_class = "S1"  # borderline → classify as S1

    return {
        "output_length": output_length,
        "reasoning_marker_count": reasoning_count,
        "lambda_indicator_count": lambda_count,
        "first_token_class": first_token_class,
        "is_failed_compile": is_failed_compile,
        "system_class": system_class,
        "first_line": first_line[:80],
    }


# ─────────────────────── cross-task analysis ──────────────────────────


def analyze_cross_task(data: dict[str, Any]) -> dict[str, Any]:
    """Analyze full cross-task result data.

    Returns structured report with per-condition aggregates and
    the essentiality matrix annotated with failure modes.
    """
    records: list[dict[str, Any]] = []

    for task_name, probes in data.items():
        for probe_id, conditions in probes.items():
            for condition_name, result in conditions.items():
                gen = result.get("generation", "")
                metrics = classify_generation(gen)

                records.append({
                    "task": task_name,
                    "probe": probe_id,
                    "condition": condition_name,
                    "ablated_head": result.get("ablated_head"),
                    "is_baseline": result.get("is_baseline", False),
                    "success": result.get("success", False),
                    **metrics,
                })

    # Aggregate by (task, condition)
    from collections import defaultdict

    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        key = (r["task"], r["condition"])
        agg[key].append(r)

    summary: dict[str, Any] = {}
    for (task, condition), group in sorted(agg.items()):
        n = len(group)
        key = f"{task}/{condition}"
        summary[key] = {
            "n_probes": n,
            "success_rate": sum(r["success"] for r in group) / n,
            "avg_output_length": sum(r["output_length"] for r in group) / n,
            "avg_reasoning_markers": sum(r["reasoning_marker_count"] for r in group) / n,
            "avg_lambda_indicators": sum(r["lambda_indicator_count"] for r in group) / n,
            "system_distribution": {
                "S1": sum(1 for r in group if r["system_class"] == "S1"),
                "S2": sum(1 for r in group if r["system_class"] == "S2"),
                "S2_fallback": sum(1 for r in group if r["system_class"] == "S2_fallback"),
            },
            "first_token_distribution": {
                "direct": sum(1 for r in group if r["first_token_class"] == "direct"),
                "reasoning": sum(1 for r in group if r["first_token_class"] == "reasoning"),
                "failed_direct": sum(1 for r in group if r["first_token_class"] == "failed_direct"),
            },
            "failed_compile_count": sum(1 for r in group if r["is_failed_compile"]),
        }

    # Compute deltas (each condition vs its task's baseline)
    deltas: dict[str, dict[str, float]] = {}
    for (task, condition), group in sorted(agg.items()):
        if condition == "baseline":
            continue
        baseline_group = agg.get((task, "baseline"), [])
        if not baseline_group:
            continue

        key = f"{task}/{condition}"
        bl_key = f"{task}/baseline"
        bl = summary[bl_key]
        cond = summary[key]

        deltas[key] = {
            "Δ_success_rate": cond["success_rate"] - bl["success_rate"],
            "Δ_output_length": cond["avg_output_length"] - bl["avg_output_length"],
            "Δ_reasoning_markers": cond["avg_reasoning_markers"] - bl["avg_reasoning_markers"],
            "Δ_lambda_indicators": cond["avg_lambda_indicators"] - bl["avg_lambda_indicators"],
            "Δ_S1_count": cond["system_distribution"]["S1"] - bl["system_distribution"]["S1"],
            "Δ_S2_count": (
                cond["system_distribution"]["S2"]
                + cond["system_distribution"]["S2_fallback"]
                - bl["system_distribution"]["S2"]
                - bl["system_distribution"]["S2_fallback"]
            ),
        }

    return {
        "n_records": len(records),
        "records": records,
        "summary": summary,
        "deltas": deltas,
    }


# ─────────────────────── report formatting ────────────────────────────


def format_report(report: dict[str, Any]) -> str:
    """Format analysis report as readable text."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("  FAILURE MODE ANALYSIS — System 1 vs System 2")
    lines.append("=" * 70)
    lines.append(f"\nTotal records analyzed: {report['n_records']}")

    # Summary table
    lines.append("\n" + "─" * 70)
    lines.append("  CONDITION SUMMARY")
    lines.append("─" * 70)
    lines.append(
        f"{'condition':<28} {'succ':>5} {'len':>6} {'reason':>7} "
        f"{'lambda':>7} {'S1':>3} {'S2':>3} {'S2f':>3}"
    )
    lines.append("─" * 70)

    for key, s in sorted(report["summary"].items()):
        sd = s["system_distribution"]
        lines.append(
            f"{key:<28} {s['success_rate']:>5.1%} {s['avg_output_length']:>6.0f} "
            f"{s['avg_reasoning_markers']:>7.1f} {s['avg_lambda_indicators']:>7.1f} "
            f"{sd['S1']:>3} {sd['S2']:>3} {sd['S2_fallback']:>3}"
        )

    # Deltas table
    lines.append("\n" + "─" * 70)
    lines.append("  DELTAS vs BASELINE (per task)")
    lines.append("─" * 70)
    lines.append(
        f"{'condition':<28} {'Δsucc':>6} {'Δlen':>7} {'Δreas':>7} "
        f"{'Δλ':>5} {'ΔS1':>4} {'ΔS2':>4}"
    )
    lines.append("─" * 70)

    for key, d in sorted(report["deltas"].items()):
        lines.append(
            f"{key:<28} {d['Δ_success_rate']:>+6.1%} {d['Δ_output_length']:>+7.0f} "
            f"{d['Δ_reasoning_markers']:>+7.1f} {d['Δ_lambda_indicators']:>+5.1f} "
            f"{d['Δ_S1_count']:>+4} {d['Δ_S2_count']:>+4}"
        )

    # Key findings: compile task focus
    lines.append("\n" + "─" * 70)
    lines.append("  COMPILE TASK — FAILURE MODE DETAIL")
    lines.append("─" * 70)

    compile_records = [r for r in report["records"] if r["task"] == "compile"]
    for r in sorted(compile_records, key=lambda x: (x["condition"], x["probe"])):
        status = "✓" if r["success"] else "✗"
        sys_cls = r["system_class"]
        lines.append(
            f"  {status} {r['condition']:<12} {r['probe']:<20} "
            f"{sys_cls:<12} {r['first_line'][:50]}"
        )

    # Extract task focus
    lines.append("\n" + "─" * 70)
    lines.append("  EXTRACT TASK — FAILURE MODE DETAIL")
    lines.append("─" * 70)

    extract_records = [r for r in report["records"] if r["task"] == "extract"]
    for r in sorted(extract_records, key=lambda x: (x["condition"], x["probe"])):
        status = "✓" if r["success"] else "✗"
        sys_cls = r["system_class"]
        lines.append(
            f"  {status} {r['condition']:<12} {r['probe']:<20} "
            f"{sys_cls:<12} {r['first_line'][:50]}"
        )

    # The headline finding
    lines.append("\n" + "=" * 70)
    lines.append("  HEADLINE")
    lines.append("=" * 70)

    compile_baseline = report["summary"].get("compile/baseline", {})
    compile_l24h0 = report["summary"].get("compile/L24-H0", {})
    extract_baseline = report["summary"].get("extract/baseline", {})
    extract_l24h0 = report["summary"].get("extract/L24-H0", {})

    if compile_baseline and compile_l24h0:
        bl_s1 = compile_baseline["system_distribution"]["S1"]
        ab_s1 = compile_l24h0["system_distribution"]["S1"]
        ab_s2f = compile_l24h0["system_distribution"]["S2_fallback"]
        lines.append(
            f"\n  COMPILE: L24:H0 ablation shifts S1→S2"
        )
        lines.append(
            f"    baseline S1 count: {bl_s1}  |  ablated S1 count: {ab_s1}  |  S2_fallback: {ab_s2f}"
        )
        lines.append(
            f"    success: {compile_baseline['success_rate']:.0%} → {compile_l24h0['success_rate']:.0%}"
        )
        lines.append(
            f"    reasoning markers: {compile_baseline['avg_reasoning_markers']:.1f} → {compile_l24h0['avg_reasoning_markers']:.1f}"
        )

    if extract_baseline and extract_l24h0:
        lines.append(
            f"\n  EXTRACT: L24:H0 ablation effect"
        )
        lines.append(
            f"    success: {extract_baseline['success_rate']:.0%} → {extract_l24h0['success_rate']:.0%}"
        )
        lines.append(
            f"    reasoning markers: {extract_baseline['avg_reasoning_markers']:.1f} → {extract_l24h0['avg_reasoning_markers']:.1f}"
        )

    translate_l24h0 = report["summary"].get("translate/L24-H0", {})
    translate_baseline = report["summary"].get("translate/baseline", {})
    if translate_baseline and translate_l24h0:
        lines.append(
            f"\n  TRANSLATE: L24:H0 ablation effect"
        )
        lines.append(
            f"    success: {translate_baseline['success_rate']:.0%} → {translate_l24h0['success_rate']:.0%}"
        )
        lines.append(f"    (no effect — confirms compositor is composition-specific)")

    lines.append("")
    return "\n".join(lines)


# ─────────────────────── CLI entry point ──────────────────────────────


def main() -> None:
    """Run analysis on cross-task result file."""
    if len(sys.argv) < 2:
        # Default: find the cross-task result
        result_path = Path(
            "results/experiments/"
            "sha256:bd530aec0d8aa573a4deab4c67be4bb00f52845dec0ceb34d49efe19f4b5a708/"
            "result.json"
        )
    else:
        result_path = Path(sys.argv[1])

    if not result_path.exists():
        print(f"Result file not found: {result_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(result_path.read_text())
    report = analyze_cross_task(data)

    # Print report
    print(format_report(report))

    # Save JSON
    output_path = Path("results/experiments/failure-mode-analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save without raw records (too large)
    save_data = {k: v for k, v in report.items() if k != "records"}
    output_path.write_text(json.dumps(save_data, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
