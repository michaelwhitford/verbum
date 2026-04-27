#!/usr/bin/env python3
"""Top-down probe of predictive functions in Qwen3.5-35B-A3B.

Three experiments mapping which internal functions the model uses
for prediction, probed through llama.cpp behavioral measurement.

Experiments:
  1. landscape   — 25 tasks × 40 probes → confidence/entropy matrix
  2. complexity  — 5 complexity tiers × key tasks → degradation curves
  3. priming     — prime with task A, measure task B → shared circuits

Usage:
    # All experiments
    uv run python scripts/probe_predictive_functions.py all

    # Individual experiments
    uv run python scripts/probe_predictive_functions.py landscape
    uv run python scripts/probe_predictive_functions.py complexity
    uv run python scripts/probe_predictive_functions.py priming

    # Custom server
    uv run python scripts/probe_predictive_functions.py landscape --port 5102
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBES_PATH = PROJECT_ROOT / "probes" / "compile-gradient.json"
GATES_DIR = PROJECT_ROOT / "gates"
RESULTS_DIR = PROJECT_ROOT / "results" / "predictive-functions"

# ══════════════════════════════════════════════════════════════════════
# Task definitions
# ══════════════════════════════════════════════════════════════════════

# Task gates — one-line instructions that activate different functions
TASK_GATES = {}
for gate_path in sorted(GATES_DIR.glob("task-*.txt")):
    name = gate_path.stem.replace("task-", "")
    TASK_GATES[name] = gate_path.read_text("utf-8").strip()

# Add compile from main gate (not task-compile)
COMPILE_GATE = "Convert to lambda calculus:"
TASK_GATES["compile"] = COMPILE_GATE

# Ensure compile is first for readability
TASK_ORDER = ["compile"] + sorted(k for k in TASK_GATES if k != "compile")

# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


def load_probes() -> list[dict]:
    data = json.loads(PROBES_PATH.read_text())
    return data["probes"]


def complete(
    base_url: str,
    prompt: str,
    *,
    n_predict: int = 60,
    temperature: float = 0.0,
    n_probs: int = 10,
) -> dict:
    """Call llama.cpp /completion and return parsed response."""
    r = httpx.post(
        f"{base_url}/completion",
        json={
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "n_probs": n_probs,
            "cache_prompt": False,
        },
        timeout=60.0,
    )
    r.raise_for_status()
    return r.json()


def measure_response(response: dict) -> dict:
    """Extract entropy, confidence, and lambda indicators from response."""
    content = response.get("content", "")
    probs = response.get("completion_probabilities", [])

    top1_logprobs = []
    entropies = []
    for p in probs:
        top = p.get("top_logprobs", [])
        if top:
            top1_logprobs.append(top[0].get("logprob", 0))
            # Entropy from top-k logprobs
            ps = [math.exp(t["logprob"]) for t in top if t.get("logprob") is not None]
            total = sum(ps)
            if total > 0:
                ps_norm = [pi / total for pi in ps]
                ent = -sum(pi * math.log(pi) for pi in ps_norm if pi > 0)
                entropies.append(ent)

    avg_logprob = sum(top1_logprobs) / len(top1_logprobs) if top1_logprobs else 0
    avg_entropy = sum(entropies) / len(entropies) if entropies else 0

    # First-token confidence (most diagnostic)
    first_logprob = top1_logprobs[0] if top1_logprobs else 0
    first_entropy = entropies[0] if entropies else 0

    # Lambda/formal indicators
    LAMBDA_MARKERS = {"λ", "∀", "∃", "∧", "∨", "¬", "→"}
    has_lambda = any(m in content for m in LAMBDA_MARKERS)
    has_think = "<think>" in content

    return {
        "content": content[:200],
        "tokens_predicted": response.get("tokens_predicted", 0),
        "avg_logprob": round(avg_logprob, 4),
        "avg_entropy": round(avg_entropy, 4),
        "first_logprob": round(first_logprob, 4),
        "first_entropy": round(first_entropy, 4),
        "has_lambda": has_lambda,
        "has_think": has_think,
    }


# ══════════════════════════════════════════════════════════════════════
# Experiment 1: Confidence Landscape
# ══════════════════════════════════════════════════════════════════════


def run_landscape(base_url: str) -> dict:
    """25 tasks × 40 probes → confidence/entropy matrix."""
    probes = load_probes()
    n_tasks = len(TASK_ORDER)
    n_probes = len(probes)
    total = n_tasks * n_probes

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 1: Confidence Landscape")
    print(f"  {n_tasks} tasks × {n_probes} probes = {total} measurements")
    print(f"{'='*60}\n")

    results = []
    done = 0

    for task_name in TASK_ORDER:
        gate = TASK_GATES[task_name]
        task_results = []

        for probe in probes:
            prompt = f"{gate}\n{probe['prompt']}\n→"

            try:
                t0 = time.time()
                resp = complete(base_url, prompt, n_predict=60, n_probs=10)
                elapsed = time.time() - t0
                measurement = measure_response(resp)
                measurement["elapsed_ms"] = round(elapsed * 1000)
            except Exception as e:
                measurement = {"error": str(e)}

            measurement["task"] = task_name
            measurement["probe_id"] = probe["id"]
            measurement["category"] = probe.get("category", "")
            measurement["gradient"] = probe.get("metadata", {}).get("gradient", 0)
            task_results.append(measurement)

            done += 1
            if done % 25 == 0 or done == total:
                pct = done / total * 100
                print(f"  [{done:>4}/{total}] {pct:>5.1f}%  {task_name:15s}  {probe['id']}")

        results.extend(task_results)

    # Summary table
    print(f"\n  {'Task':20s} {'AvgLogprob':>11} {'AvgEntropy':>11} {'1stLogprob':>11} {'1stEntropy':>11} {'λ%':>5}")
    print(f"  {'─'*20} {'─'*11} {'─'*11} {'─'*11} {'─'*11} {'─'*5}")

    task_summaries = {}
    for task_name in TASK_ORDER:
        task_rows = [r for r in results if r["task"] == task_name and "error" not in r]
        if not task_rows:
            continue
        avg_lp = sum(r["avg_logprob"] for r in task_rows) / len(task_rows)
        avg_ent = sum(r["avg_entropy"] for r in task_rows) / len(task_rows)
        avg_1lp = sum(r["first_logprob"] for r in task_rows) / len(task_rows)
        avg_1ent = sum(r["first_entropy"] for r in task_rows) / len(task_rows)
        lam_pct = sum(1 for r in task_rows if r.get("has_lambda")) / len(task_rows) * 100

        task_summaries[task_name] = {
            "avg_logprob": round(avg_lp, 4),
            "avg_entropy": round(avg_ent, 4),
            "first_logprob": round(avg_1lp, 4),
            "first_entropy": round(avg_1ent, 4),
            "lambda_pct": round(lam_pct, 1),
            "n": len(task_rows),
        }
        print(
            f"  {task_name:20s} {avg_lp:>11.4f} {avg_ent:>11.4f} "
            f"{avg_1lp:>11.4f} {avg_1ent:>11.4f} {lam_pct:>4.0f}%"
        )

    return {
        "experiment": "landscape",
        "timestamp": datetime.now(UTC).isoformat(),
        "model": "Qwen3.5-35B-A3B",
        "n_tasks": n_tasks,
        "n_probes": n_probes,
        "task_summaries": task_summaries,
        "results": results,
    }


# ══════════════════════════════════════════════════════════════════════
# Experiment 2: Compositional Complexity Scaling
# ══════════════════════════════════════════════════════════════════════

COMPLEXITY_TIERS = {
    "trivial": [
        "The dog runs.",
        "Birds fly.",
        "Snow is white.",
    ],
    "simple": [
        "The cat sat on the mat.",
        "Every student reads a book.",
        "If it rains, the ground is wet.",
    ],
    "moderate": [
        "The cat that sat on the mat is black.",
        "The teacher gave every student a grade.",
        "Someone believes that the earth is flat.",
    ],
    "complex": [
        "The old man who lived next door told every child that the park would close before sunset.",
        "Most politicians who promised reform during the campaign failed to deliver on their commitments.",
        "What the witness saw contradicted the official report that the committee had published.",
    ],
    "nested": [
        "Every professor who supervises a student that published a paper which cited a theorem that Gödel proved received a commendation.",
        "The fact that the scientist who discovered the compound that the company manufactured improperly resigned surprised nobody who understood the situation.",
        "No critic who reviewed the film that the director who won the award produced believed that it deserved the prize that the jury selected it for.",
    ],
}

COMPLEXITY_TASKS = ["compile", "formalize", "structure", "negate", "paraphrase", "entail", "decompose", "scope"]


def run_complexity(base_url: str) -> dict:
    """Complexity scaling: how each function degrades with input complexity."""
    n_tiers = len(COMPLEXITY_TIERS)
    n_tasks = len(COMPLEXITY_TASKS)
    n_inputs = sum(len(v) for v in COMPLEXITY_TIERS.values())
    total = n_tasks * n_inputs

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 2: Compositional Complexity Scaling")
    print(f"  {n_tasks} tasks × {n_tiers} tiers × 3 inputs = {total} measurements")
    print(f"{'='*60}\n")

    results = []
    done = 0

    for task_name in COMPLEXITY_TASKS:
        gate = TASK_GATES[task_name]

        for tier_name, inputs in COMPLEXITY_TIERS.items():
            for input_text in inputs:
                prompt = f"{gate}\n{input_text}\n→"

                try:
                    t0 = time.time()
                    resp = complete(base_url, prompt, n_predict=80, n_probs=10)
                    elapsed = time.time() - t0
                    measurement = measure_response(resp)
                    measurement["elapsed_ms"] = round(elapsed * 1000)
                except Exception as e:
                    measurement = {"error": str(e)}

                measurement["task"] = task_name
                measurement["tier"] = tier_name
                measurement["input"] = input_text
                results.append(measurement)

                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  [{done:>3}/{total}]  {task_name:12s}  {tier_name:10s}  {input_text[:40]}")

    # Summary: task × tier matrix
    print(f"\n  Avg entropy by task × complexity tier:")
    print(f"  {'task':15s}", end="")
    for tier in COMPLEXITY_TIERS:
        print(f"  {tier:>10s}", end="")
    print()
    print(f"  {'─'*15}", end="")
    for _ in COMPLEXITY_TIERS:
        print(f"  {'─'*10}", end="")
    print()

    tier_summaries = {}
    for task_name in COMPLEXITY_TASKS:
        print(f"  {task_name:15s}", end="")
        tier_summaries[task_name] = {}
        for tier_name in COMPLEXITY_TIERS:
            rows = [
                r for r in results
                if r["task"] == task_name and r.get("tier") == tier_name and "error" not in r
            ]
            if rows:
                avg_ent = sum(r["avg_entropy"] for r in rows) / len(rows)
                avg_lp = sum(r["avg_logprob"] for r in rows) / len(rows)
                tier_summaries[task_name][tier_name] = {
                    "avg_entropy": round(avg_ent, 4),
                    "avg_logprob": round(avg_lp, 4),
                }
                print(f"  {avg_ent:>10.3f}", end="")
            else:
                print(f"  {'?':>10}", end="")
        print()

    return {
        "experiment": "complexity",
        "timestamp": datetime.now(UTC).isoformat(),
        "model": "Qwen3.5-35B-A3B",
        "n_tasks": n_tasks,
        "n_tiers": n_tiers,
        "tier_summaries": tier_summaries,
        "results": results,
    }


# ══════════════════════════════════════════════════════════════════════
# Experiment 3: Cross-Priming Interference
# ══════════════════════════════════════════════════════════════════════

# Prime with one task's exemplar, then ask for a different task
PRIME_TASKS = ["compile", "formalize", "structure", "negate", "paraphrase"]
MEASURE_TASKS = ["compile", "formalize", "structure", "negate", "paraphrase", "entail", "decompose"]

# Fixed exemplar pairs for priming (input → output)
PRIME_EXEMPLARS = {
    "compile": ("The dog runs.", "λx. runs(dog)"),
    "formalize": ("The dog runs.", "∃x (Dog(x) ∧ Runs(x))"),
    "structure": ("The dog runs.", "Subject: The dog, Verb: runs"),
    "negate": ("The dog runs.", "The dog does not run."),
    "paraphrase": ("The dog runs.", "A canine is running."),
}

PRIME_INPUTS = [
    "Every student reads a book.",
    "The cat sat on the mat.",
    "If it rains, the ground is wet.",
]


def run_priming(base_url: str) -> dict:
    """Cross-priming: does activating one function affect another?"""
    n_primes = len(PRIME_TASKS)
    n_measures = len(MEASURE_TASKS)
    n_inputs = len(PRIME_INPUTS)
    # +1 for no-prime baseline
    total = (n_primes + 1) * n_measures * n_inputs

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 3: Cross-Priming Interference")
    print(f"  ({n_primes} primes + baseline) × {n_measures} tasks × {n_inputs} inputs = {total}")
    print(f"{'='*60}\n")

    results = []
    done = 0

    # Baseline: no priming
    for task_name in MEASURE_TASKS:
        gate = TASK_GATES[task_name]
        for input_text in PRIME_INPUTS:
            prompt = f"{gate}\n{input_text}\n→"

            try:
                t0 = time.time()
                resp = complete(base_url, prompt, n_predict=60, n_probs=10)
                elapsed = time.time() - t0
                measurement = measure_response(resp)
                measurement["elapsed_ms"] = round(elapsed * 1000)
            except Exception as e:
                measurement = {"error": str(e)}

            measurement["prime"] = "none"
            measurement["task"] = task_name
            measurement["input"] = input_text
            results.append(measurement)
            done += 1

    print(f"  Baseline done ({done}/{total})")

    # Primed: show exemplar of prime task, then ask for measure task
    for prime_name in PRIME_TASKS:
        ex_in, ex_out = PRIME_EXEMPLARS[prime_name]
        prime_gate = TASK_GATES[prime_name]

        for task_name in MEASURE_TASKS:
            measure_gate = TASK_GATES[task_name]

            for input_text in PRIME_INPUTS:
                # Prime prefix: gate + exemplar
                prompt = (
                    f"{prime_gate}\n{ex_in}\n→ {ex_out}\n\n"
                    f"{measure_gate}\n{input_text}\n→"
                )

                try:
                    t0 = time.time()
                    resp = complete(base_url, prompt, n_predict=60, n_probs=10)
                    elapsed = time.time() - t0
                    measurement = measure_response(resp)
                    measurement["elapsed_ms"] = round(elapsed * 1000)
                except Exception as e:
                    measurement = {"error": str(e)}

                measurement["prime"] = prime_name
                measurement["task"] = task_name
                measurement["input"] = input_text
                results.append(measurement)
                done += 1

        pct = done / total * 100
        print(f"  [{done:>4}/{total}] {pct:>5.1f}%  prime={prime_name}")

    # Summary: priming effect matrix (avg entropy delta vs baseline)
    print(f"\n  Priming effect (Δ entropy vs baseline, negative = helped):")
    print(f"  {'prime→task':15s}", end="")
    for task in MEASURE_TASKS:
        print(f"  {task:>10s}", end="")
    print()
    print(f"  {'─'*15}", end="")
    for _ in MEASURE_TASKS:
        print(f"  {'─'*10}", end="")
    print()

    # Compute baselines
    baselines = {}
    for task in MEASURE_TASKS:
        rows = [r for r in results if r["prime"] == "none" and r["task"] == task and "error" not in r]
        if rows:
            baselines[task] = sum(r["avg_entropy"] for r in rows) / len(rows)

    priming_effects = {}
    for prime in PRIME_TASKS:
        print(f"  {prime:15s}", end="")
        priming_effects[prime] = {}
        for task in MEASURE_TASKS:
            rows = [r for r in results if r["prime"] == prime and r["task"] == task and "error" not in r]
            if rows and task in baselines:
                avg_ent = sum(r["avg_entropy"] for r in rows) / len(rows)
                delta = avg_ent - baselines[task]
                priming_effects[prime][task] = round(delta, 4)
                marker = "↓" if delta < -0.02 else "↑" if delta > 0.02 else "≈"
                print(f"  {delta:>+9.3f}{marker}", end="")
            else:
                print(f"  {'?':>10}", end="")
        print()

    return {
        "experiment": "priming",
        "timestamp": datetime.now(UTC).isoformat(),
        "model": "Qwen3.5-35B-A3B",
        "baselines": {k: round(v, 4) for k, v in baselines.items()},
        "priming_effects": priming_effects,
        "results": results,
    }


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Top-down predictive function probing")
    parser.add_argument("experiment", choices=["landscape", "complexity", "priming", "all"])
    parser.add_argument("--port", type=int, default=5102)
    parser.add_argument("--host", type=str, default="localhost")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Health check
    try:
        r = httpx.get(f"{base_url}/health", timeout=5)
        health = r.json()
        print(f"Server: {base_url} — {health.get('status', '?')}")
    except Exception as e:
        print(f"ERROR: Cannot reach {base_url}: {e}")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = (
        ["landscape", "complexity", "priming"]
        if args.experiment == "all"
        else [args.experiment]
    )

    for exp_name in experiments:
        if exp_name == "landscape":
            result = run_landscape(base_url)
        elif exp_name == "complexity":
            result = run_complexity(base_url)
        elif exp_name == "priming":
            result = run_priming(base_url)
        else:
            continue

        out_path = RESULTS_DIR / f"{exp_name}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"\n  Saved: {out_path}\n")


if __name__ == "__main__":
    main()
