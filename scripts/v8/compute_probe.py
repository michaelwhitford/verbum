"""Computation probe for v8 BIOS circuit detection.

Generates fresh math/clojure examples the model has never seen,
feeds the expression prefix, greedy-decodes the answer, and checks
exact match. Accuracy jumping from ~0% to >0% = circuit formation
(grokking signal).

Tiers:
  1: Single arithmetic on novel numbers
  2: Compound expressions (2 operations)
  3: Clojure HOF (map, filter, reduce)

Usage:
    # Standalone
    uv run python scripts/v8/compute_probe.py checkpoints/v8-bios/step_005000

    # From train.py (imported)
    from compute_probe import run_computation_probe
    results = run_computation_probe(model, seq_len=512, seed=step)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tokenizer import encode, decode, PAD_ID, EOD_ID, VOCAB_SIZE


# ═══════════════════════════════════════════════════════════════
# Example generators — fresh examples with ground truth
# ═══════════════════════════════════════════════════════════════

def _gen_tier1(rng: random.Random, n: int = 50) -> list[tuple[str, str]]:
    """Tier 1: single arithmetic ops on novel numbers.

    Same operations and notations as BIOS training data,
    but with fresh random numbers.
    """
    examples = []

    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    ]

    for _ in range(n):
        op_sym, op_fn = rng.choice(ops)
        a = rng.randint(0, 999)
        b = rng.randint(0, 999)
        result = op_fn(a, b)

        notation = rng.choice(["sexpr", "raw", "lambda"])
        if notation == "sexpr":
            prompt = f"({op_sym} {a} {b}) → "
        elif notation == "raw":
            prompt = f"{a} {op_sym} {b} = "
        else:
            prompt = f"(λx. λy. ({op_sym} x y) {a} {b}) → "

        examples.append((prompt, str(result), "tier1", op_sym))

    # Predicates
    for _ in range(n // 5):
        v = rng.randint(0, 999)
        pred = rng.choice(["even?", "odd?", "zero?", "pos?", "neg?"])
        if pred == "even?":
            answer = "true" if v % 2 == 0 else "false"
        elif pred == "odd?":
            answer = "true" if v % 2 == 1 else "false"
        elif pred == "zero?":
            answer = "true" if v == 0 else "false"
        elif pred == "pos?":
            answer = "true" if v > 0 else "false"
        else:  # neg?
            answer = "false"  # v is always >= 0
        prompt = f"({pred} {v}) → "
        examples.append((prompt, answer, "tier1", pred))

    # Unary
    for _ in range(n // 5):
        v = rng.randint(0, 999)
        op = rng.choice(["inc", "dec"])
        result = v + 1 if op == "inc" else v - 1
        prompt = f"({op} {v}) → "
        examples.append((prompt, str(result), "tier1", op))

    return examples


def _gen_tier2(rng: random.Random, n: int = 30) -> list[tuple[str, str]]:
    """Tier 2: compound expressions (2 operations)."""
    examples = []

    for _ in range(n):
        a, b, c = rng.randint(1, 99), rng.randint(1, 99), rng.randint(1, 99)
        pattern = rng.choice(["add_mul", "mul_add", "sub_mul", "nested_add"])

        if pattern == "add_mul":
            result = (a + b) * c
            sexpr = f"(* (+ {a} {b}) {c})"
            raw = f"({a} + {b}) * {c}"
        elif pattern == "mul_add":
            result = a * b + c
            sexpr = f"(+ (* {a} {b}) {c})"
            raw = f"{a} * {b} + {c}"
        elif pattern == "sub_mul":
            result = (a - b) * c
            sexpr = f"(* (- {a} {b}) {c})"
            raw = f"({a} - {b}) * {c}"
        else:  # nested_add
            result = (a + b) + (c + a)
            sexpr = f"(+ (+ {a} {b}) (+ {c} {a}))"
            raw = f"({a} + {b}) + ({c} + {a})"

        notation = rng.choice(["sexpr", "raw"])
        if notation == "sexpr":
            prompt = f"{sexpr} → "
        else:
            prompt = f"{raw} = "

        examples.append((prompt, str(result), "tier2", pattern))

    return examples


def _gen_tier3(rng: random.Random, n: int = 20) -> list[tuple[str, str]]:
    """Tier 3: clojure HOF on novel inputs."""
    examples = []

    for _ in range(n):
        # Random short list
        length = rng.randint(2, 5)
        nums = [rng.randint(0, 20) for _ in range(length)]
        list_str = "[" + " ".join(str(x) for x in nums) + "]"

        hof = rng.choice(["map_inc", "map_dec", "filter_even", "reduce_add",
                           "first", "last", "count", "reverse", "sort"])

        if hof == "map_inc":
            prompt = f"(map inc {list_str}) → "
            answer = "[" + " ".join(str(x + 1) for x in nums) + "]"
        elif hof == "map_dec":
            prompt = f"(map dec {list_str}) → "
            answer = "[" + " ".join(str(x - 1) for x in nums) + "]"
        elif hof == "filter_even":
            prompt = f"(filter even? {list_str}) → "
            evens = [x for x in nums if x % 2 == 0]
            answer = "(" + " ".join(str(x) for x in evens) + ")" if evens else "()"
        elif hof == "reduce_add":
            prompt = f"(reduce + {list_str}) → "
            answer = str(sum(nums))
        elif hof == "first":
            prompt = f"(first {list_str}) → "
            answer = str(nums[0])
        elif hof == "last":
            prompt = f"(last {list_str}) → "
            answer = str(nums[-1])
        elif hof == "count":
            prompt = f"(count {list_str}) → "
            answer = str(len(nums))
        elif hof == "reverse":
            prompt = f"(reverse {list_str}) → "
            answer = "(" + " ".join(str(x) for x in reversed(nums)) + ")"
        else:  # sort
            prompt = f"(sort {list_str}) → "
            answer = "(" + " ".join(str(x) for x in sorted(nums)) + ")"

        examples.append((prompt, answer, "tier3", hof))

    return examples


# ═══════════════════════════════════════════════════════════════
# Generation — greedy decode from model
# ═══════════════════════════════════════════════════════════════

def _greedy_generate(
    model,
    prompt_ids: list[int],
    seq_len: int,
    max_tokens: int = 20,
) -> list[int]:
    """Generate tokens greedily from prompt.

    Pads prompt to seq_len, generates one token at a time.
    Stops at EOD or newline or max_tokens.
    """
    ids = list(prompt_ids)

    for _ in range(max_tokens):
        # Prepare input: take last seq_len tokens (or pad if shorter)
        if len(ids) >= seq_len:
            input_ids = ids[-seq_len:]
        else:
            # Right-align: pad on left with PAD tokens
            pad_len = seq_len - len(ids)
            input_ids = [PAD_ID] * pad_len + ids

        tokens = mx.array([input_ids], dtype=mx.int32)
        logits = model(tokens)
        mx.eval(logits)

        # Take logits at last position
        next_logits = logits[0, -1, :]
        next_id = int(mx.argmax(next_logits))

        if next_id == EOD_ID:
            break
        ids.append(next_id)

        # Stop at newline (end of example in BIOS format)
        decoded_char = decode([next_id])
        if "\n" in decoded_char:
            break

    # Return only the generated tokens (after prompt)
    return ids[len(prompt_ids):]


# ═══════════════════════════════════════════════════════════════
# Main probe function
# ═══════════════════════════════════════════════════════════════

def run_computation_probe(
    model,
    seq_len: int = 512,
    n_tier1: int = 50,
    n_tier2: int = 30,
    n_tier3: int = 20,
    seed: int = 12345,
) -> dict:
    """Run computation probe. Returns accuracy per tier.

    Args:
        model: DualMERA model (in eval mode)
        seq_len: model sequence length
        n_tier1: number of tier 1 examples
        n_tier2: number of tier 2 examples
        n_tier3: number of tier 3 examples
        seed: random seed (use step number for reproducibility across runs)

    Returns:
        dict with per-tier accuracy and example details
    """
    rng = random.Random(seed)

    examples = []
    examples.extend(_gen_tier1(rng, n_tier1))
    examples.extend(_gen_tier2(rng, n_tier2))
    examples.extend(_gen_tier3(rng, n_tier3))

    # Shuffle to avoid ordering effects
    rng.shuffle(examples)

    results_by_tier = {"tier1": [], "tier2": [], "tier3": []}

    for prompt, expected, tier, op in examples:
        prompt_ids = encode(prompt)
        gen_ids = _greedy_generate(model, prompt_ids, seq_len, max_tokens=20)
        gen_text = decode(gen_ids).strip()

        # Check: does generation start with expected answer?
        # Strip any trailing characters after the answer
        is_correct = gen_text.startswith(expected)

        results_by_tier[tier].append({
            "prompt": prompt,
            "expected": expected,
            "generated": gen_text[:60],
            "correct": is_correct,
            "op": op,
        })

    # Aggregate
    summary = {}
    for tier, results in results_by_tier.items():
        n = len(results)
        correct = sum(1 for r in results if r["correct"])
        summary[tier] = {
            "accuracy": correct / n if n > 0 else 0,
            "correct": correct,
            "total": n,
        }

    total_correct = sum(s["correct"] for s in summary.values())
    total_n = sum(s["total"] for s in summary.values())
    summary["overall"] = {
        "accuracy": total_correct / total_n if total_n > 0 else 0,
        "correct": total_correct,
        "total": total_n,
    }

    return {
        "summary": summary,
        "details": results_by_tier,
    }


def print_probe_results(results: dict, step: int = 0) -> None:
    """Print formatted probe results."""
    s = results["summary"]
    print(f"\n  ── COMPUTE PROBE step {step} ──")
    for tier in ["tier1", "tier2", "tier3", "overall"]:
        m = s[tier]
        bar = "█" * int(m["accuracy"] * 20) + "░" * (20 - int(m["accuracy"] * 20))
        print(f"    {tier:>7s}: {m['correct']:>3d}/{m['total']:<3d} "
              f"({m['accuracy']*100:5.1f}%) {bar}")

    # Show a few examples (2 correct, 2 wrong if available)
    all_results = []
    for tier_results in results["details"].values():
        all_results.extend(tier_results)

    correct_ex = [r for r in all_results if r["correct"]][:2]
    wrong_ex = [r for r in all_results if not r["correct"]][:2]

    if correct_ex:
        print(f"    ✓ examples:")
        for r in correct_ex:
            print(f"      {r['prompt']}{r['generated'][:30]}")
    if wrong_ex:
        print(f"    ✗ examples:")
        for r in wrong_ex:
            print(f"      {r['prompt']}expected={r['expected']}  got={r['generated'][:30]}")
    print()


# ═══════════════════════════════════════════════════════════════
# Standalone CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="v8 Computation Probe")
    parser.add_argument("checkpoint", type=Path, help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-tier1", type=int, default=50)
    parser.add_argument("--n-tier2", type=int, default=30)
    parser.add_argument("--n-tier3", type=int, default=20)
    args = parser.parse_args()

    from model import DualMERA, DualMERAConfig, create_model

    # Load checkpoint
    ckpt = args.checkpoint
    state = json.loads((ckpt / "state.json").read_text()) if (ckpt / "state.json").exists() else {}
    step = state.get("step", 0)

    cfg = DualMERAConfig(seq_len=512)
    model = create_model(cfg)

    weights_path = ckpt / "model.npz"
    if weights_path.exists():
        weights = dict(mx.load(str(weights_path)))
        model.load_weights(list(weights.items()))
        print(f"  Loaded: {ckpt} (step {step})")

    results = run_computation_probe(
        model, seq_len=512,
        n_tier1=args.n_tier1, n_tier2=args.n_tier2, n_tier3=args.n_tier3,
        seed=args.seed,
    )
    print_probe_results(results, step)

    # Save results alongside checkpoint
    out_path = ckpt / "compute_probe.json"
    if ckpt.exists():
        out_path.write_text(json.dumps({
            "step": step,
            "seed": args.seed,
            "summary": results["summary"],
        }, indent=2))
        print(f"  Saved: {out_path}")
