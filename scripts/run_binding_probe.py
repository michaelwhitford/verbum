#!/usr/bin/env python3
"""Binding probe — test Qwen3-4B's ability to produce correct binding structures.

Fires the binding probe set through multiple gate variants and assesses
whether the model can produce correct quantifier scope, variable binding,
anaphora resolution, and control verb structures.

Usage:
    # Run all gates against all probes
    uv run python scripts/run_binding_probe.py --server http://127.0.0.1:5101

    # Run a specific gate only
    uv run python scripts/run_binding_probe.py --server http://127.0.0.1:5101 --gate compile-binding-typed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROBES_PATH = Path("probes/binding.json")
GATES_DIR = Path("gates/")
RESULTS_DIR = Path("results/binding")

# Gates to test — from existing flat gate to binding-aware gates
BINDING_GATES = [
    "compile",                    # existing flat gate (baseline)
    "compile-binding-montague",   # ∀/∃ exemplars
    "compile-binding-scope",      # scope + definite description
    "compile-binding-typed",      # 3-shot with ι, ∀/∃, ¬∃
]

# ══════════════════════════════════════════════════════════════════════
# Binding quality assessment
# ══════════════════════════════════════════════════════════════════════

QUANTIFIER_MARKERS = {"∀", "∃", "¬∃", "MOST"}
SCOPE_MARKERS = {"→", "∧", "∨"}
BINDING_MARKERS = {"ι", "ιx", "ιy", "ιz"}
VARIABLE_PATTERN_CHARS = set("xyz")


def assess_binding(generation: str, ground_truth: str, probe: dict) -> dict:
    """Assess binding quality of a generation against ground truth.

    Returns structured quality metrics, not a single score.
    """
    gen = generation.strip()
    gt = ground_truth.strip()
    meta = probe.get("metadata", {})
    phenomena = meta.get("phenomena", [])

    # Basic lambda presence
    has_lambda = "λ" in gen or "\\" in gen
    has_formal = any(m in gen for m in ["→", "∀", "∃", "∧", "∨", "¬"])

    # Quantifier binding
    has_universal = "∀" in gen
    has_existential = "∃" in gen
    has_negation = "¬" in gen
    has_definite = "ι" in gen

    gt_has_universal = "∀" in gt
    gt_has_existential = "∃" in gt
    gt_has_negation = "¬" in gt
    gt_has_definite = "ι" in gt

    # Count variable bindings (x, y, z appearing after quantifiers)
    import re
    gen_bound_vars = set(re.findall(r'[∀∃]([xyz])', gen))
    gt_bound_vars = set(re.findall(r'[∀∃]([xyz])', gt))

    # Check if quantifier structure matches
    quantifier_match = (
        (has_universal == gt_has_universal) and
        (has_existential == gt_has_existential) and
        (has_negation == gt_has_negation)
    )

    # Check scope order — does the first quantifier in gen match gt?
    def first_quantifier(text):
        for i, c in enumerate(text):
            if c in "∀∃":
                return c
        return None

    gen_first_q = first_quantifier(gen)
    gt_first_q = first_quantifier(gt)
    scope_order_match = gen_first_q == gt_first_q

    # Check predicate presence
    # Extract predicate names from ground truth
    gt_predicates = set(re.findall(r'([a-z_]+)\(', gt))
    gen_predicates = set(re.findall(r'([a-z_]+)\(', gen))
    predicate_overlap = len(gt_predicates & gen_predicates) / max(len(gt_predicates), 1)

    # Check argument order for agent/patient binding
    # Simple heuristic: look for predicate(X, Y) patterns
    def extract_args(text):
        """Extract first predicate's arguments."""
        m = re.search(r'([a-z_]+)\(([^)]+)\)', text)
        if m:
            return m.group(1), [a.strip() for a in m.group(2).split(",")]
        return None, []

    gen_pred, gen_args = extract_args(gen)
    gt_pred, gt_args = extract_args(gt)

    # Nesting depth — count parentheses depth
    def max_depth(text):
        d, mx = 0, 0
        for c in text:
            if c == '(':
                d += 1
                mx = max(mx, d)
            elif c == ')':
                d -= 1
        return mx

    gen_depth = max_depth(gen)
    gt_depth = max_depth(gt)

    # Check for flat conjunction vs proper nesting
    # Flat: P(x) ∧ Q(x) ∧ R(x) — all at same depth
    # Nested: ∀x. P(x) → ∃y. Q(y) ∧ R(x, y) — quantifiers create scope
    is_flat = gen_depth <= 2 and gen.count("∧") >= 2 and "∀" not in gen and "∃" not in gen

    return {
        "has_formal": has_formal,
        "has_lambda": has_lambda,
        "quantifier_present": {
            "universal": has_universal,
            "existential": has_existential,
            "negation": has_negation,
            "definite": has_definite,
        },
        "quantifier_expected": {
            "universal": gt_has_universal,
            "existential": gt_has_existential,
            "negation": gt_has_negation,
            "definite": gt_has_definite,
        },
        "quantifier_match": quantifier_match,
        "scope_order_match": scope_order_match,
        "bound_vars_gen": sorted(gen_bound_vars),
        "bound_vars_gt": sorted(gt_bound_vars),
        "predicate_overlap": round(predicate_overlap, 3),
        "gen_depth": gen_depth,
        "gt_depth": gt_depth,
        "is_flat": is_flat,
    }


# ══════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════

def run_binding_probes(
    server_url: str = "http://127.0.0.1:5101",
    gates: list[str] | None = None,
    n_predict: int = 80,
    temperature: float = 0.0,
) -> dict:
    """Run binding probes through Qwen with each gate variant."""
    from verbum.client import Client

    if gates is None:
        gates = BINDING_GATES

    # Load probes
    data = json.loads(PROBES_PATH.read_text())
    probes = data["probes"]

    # Load gates
    gate_contents = {}
    for gate_id in gates:
        gate_path = GATES_DIR / f"{gate_id}.txt"
        assert gate_path.exists(), f"Gate not found: {gate_path}"
        gate_contents[gate_id] = gate_path.read_text()

    total_calls = len(probes) * len(gates)
    print(f"Binding probe: {len(probes)} probes × {len(gates)} gates = {total_calls} calls")
    print(f"  Server: {server_url}")
    print(f"  Gates: {gates}")
    print(f"  n_predict: {n_predict}")
    print()

    results = []

    with Client(base_url=server_url) as client:
        health = client.health()
        print(f"  Server status: {health.status}")
        try:
            props = client.props()
            model_path = props.model_path or "unknown"
            print(f"  Model: {model_path}")
        except Exception:
            model_path = "unknown"
        print()

        for i, probe in enumerate(probes):
            probe_results = {"probe_id": probe["id"], "category": probe["category"],
                             "prompt": probe["prompt"], "ground_truth": probe["ground_truth"],
                             "metadata": probe["metadata"], "gates": {}}

            for gate_id in gates:
                gate_text = gate_contents[gate_id]
                full_prompt = gate_text + probe["prompt"]

                t0 = time.perf_counter()
                try:
                    result = client.complete(
                        full_prompt,
                        n_predict=n_predict,
                        temperature=temperature,
                    )
                    elapsed = time.perf_counter() - t0
                    generation = result.content.strip()
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    generation = ""
                    print(f"    ⚠ 500 on {probe['id']} × {gate_id}: {e!s:.80s}")

                # Take only the first line of actual output (before thinking)
                first_line = generation.split("\n")[0].strip()

                quality = assess_binding(first_line, probe["ground_truth"], probe)

                probe_results["gates"][gate_id] = {
                    "generation": generation,
                    "first_line": first_line,
                    "elapsed_ms": round(elapsed * 1000, 1),
                    "quality": quality,
                }

            results.append(probe_results)

            # Progress
            marker = ""
            for gate_id in gates:
                q = probe_results["gates"][gate_id]["quality"]
                qm = "✓" if q["quantifier_match"] else "✗"
                sm = "✓" if q["scope_order_match"] else "✗"
                marker += f"  {gate_id.split('-')[-1][:5]}:q={qm},s={sm}"
            print(f"  [{i+1}/{len(probes)}] {probe['id']:20s} {marker}")

    return {
        "model": model_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_predict": n_predict,
        "temperature": temperature,
        "gates": gates,
        "probes": results,
    }


def print_summary(data: dict) -> None:
    """Print summary of binding probe results."""
    print()
    print("=" * 70)
    print("  BINDING PROBE SUMMARY")
    print("=" * 70)

    gates = data["gates"]
    probes = data["probes"]
    categories = sorted(set(p["category"] for p in probes))

    # Per-gate summary
    for gate_id in gates:
        print(f"\n  Gate: {gate_id}")
        print(f"  {'─' * 50}")

        total_q_match = 0
        total_s_match = 0
        total_formal = 0
        total_flat = 0

        for cat in categories:
            cat_probes = [p for p in probes if p["category"] == cat]
            cat_q = sum(1 for p in cat_probes if p["gates"][gate_id]["quality"]["quantifier_match"])
            cat_s = sum(1 for p in cat_probes if p["gates"][gate_id]["quality"]["scope_order_match"])
            cat_f = sum(1 for p in cat_probes if p["gates"][gate_id]["quality"]["has_formal"])
            cat_flat = sum(1 for p in cat_probes if p["gates"][gate_id]["quality"]["is_flat"])

            total_q_match += cat_q
            total_s_match += cat_s
            total_formal += cat_f
            total_flat += cat_flat

            print(f"    {cat:20s}  quant={cat_q}/{len(cat_probes)}  scope={cat_s}/{len(cat_probes)}  "
                  f"formal={cat_f}/{len(cat_probes)}  flat={cat_flat}/{len(cat_probes)}")

        n = len(probes)
        print(f"    {'TOTAL':20s}  quant={total_q_match}/{n}  scope={total_s_match}/{n}  "
              f"formal={total_formal}/{n}  flat={total_flat}/{n}")

    # Minimal pairs analysis
    print(f"\n  MINIMAL PAIRS")
    print(f"  {'─' * 50}")

    for probe in probes:
        pair_id = probe["metadata"].get("pair")
        if pair_id and probe["probe_id"] < pair_id:
            # Find the pair
            pair = next((p for p in probes if p["probe_id"] == pair_id), None)
            if pair:
                print(f"\n    Pair: {probe['probe_id']} ↔ {pair['probe_id']}")
                print(f"      A: {probe['prompt']}")
                print(f"      B: {pair['prompt']}")
                for gate_id in gates:
                    a_line = probe["gates"][gate_id]["first_line"]
                    b_line = pair["gates"][gate_id]["first_line"]
                    same = a_line == b_line
                    print(f"      {gate_id.split('-')[-1][:8]:8s}  A={a_line[:60]}")
                    print(f"      {'':8s}  B={b_line[:60]}  {'⚠ SAME' if same else '✓ DIFFER'}")


def main():
    parser = argparse.ArgumentParser(description="Binding probe runner")
    parser.add_argument("--server", default="http://127.0.0.1:5101")
    parser.add_argument("--gate", help="Run a single gate only")
    parser.add_argument("--n-predict", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gates = [args.gate] if args.gate else None

    results = run_binding_probes(
        server_url=args.server,
        gates=gates,
        n_predict=args.n_predict,
        temperature=args.temperature,
    )

    # Save
    out_path = RESULTS_DIR / "binding_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n  Saved: {out_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
