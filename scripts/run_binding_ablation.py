#!/usr/bin/env python3
"""Binding ablation — find the shape of the binding function in Qwen3-4B.

Compares head ablation under flat gate vs hybrid binding gate.
Saves results after each experiment so it can resume on failure.

Usage:
    uv run python scripts/run_binding_ablation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

RESULTS_DIR = Path("results/binding")
RESULTS_PATH = RESULTS_DIR / "binding_ablation_results.json"


def _save(results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))


def has_binding(text: str) -> bool:
    return ("∀" in text or "∃" in text) and ("→" in text or "∧" in text)


def binding_score(text: str) -> dict:
    import re
    has_q = "∀" in text or "∃" in text
    has_formal = any(m in text for m in ["→", "∧", "∨", "¬"])
    n_quantifiers = text.count("∀") + text.count("∃")
    n_bound_vars = len(re.findall(r"[∀∃]([xyz])", text))
    depth = 0
    max_depth = 0
    for c in text:
        if c == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif c == ")":
            depth -= 1
    return {
        "has_quantifiers": has_q, "has_formal": has_formal,
        "n_quantifiers": n_quantifiers, "n_bound_vars": n_bound_vars,
        "max_depth": max_depth, "has_binding": has_binding(text),
    }


def _generate(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    from verbum.instrument import (
        ablate_heads, load_model, record_attention, zero_heads_generate,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load prior results if resuming
    results = {}
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())
        print(f"Resuming — have: {list(results.keys())}")

    print("Loading Qwen3-4B...")
    model, tokenizer, info = load_model("Qwen/Qwen3-4B")
    print(f"  {info.n_layers} layers, {info.n_heads} heads, {info.head_dim} head_dim")

    flat_gate = open("gates/compile.txt").read()
    hybrid_gate = open("gates/compile-binding-hybrid.txt").read()
    essential_heads = [(1, 0), (24, 0), (24, 2)]

    test_probes = [
        ("Everyone loves someone.", "∀", "∃"),
        ("Someone loves everyone.", "∃", "∀"),
        ("Every student read a book.", "∀", None),
        ("No student passed every exam.", "¬", None),
        ("The dog runs.", None, None),
        ("Birds fly.", None, None),
        ("She told him to leave.", None, None),
        ("The cat that chased the dog is black.", None, None),
    ]

    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["model"] = "Qwen/Qwen3-4B"
    results["essential_heads"] = essential_heads

    # ══════════════════════════════════════════════════════════════
    # Experiment 1: Baselines
    # ══════════════════════════════════════════════════════════════
    if "baselines" not in results:
        print("\n" + "=" * 60)
        print("  EXPERIMENT 1: Baseline (flat vs hybrid gate)")
        print("=" * 60)

        baselines = []
        for prompt, _, _ in test_probes:
            flat_gen = _generate(model, tokenizer, flat_gate + prompt + " → ", 60)
            hybrid_gen = _generate(model, tokenizer, hybrid_gate + prompt + " → ", 60)
            flat_fl = flat_gen.split("\n")[0].strip()
            hybrid_fl = hybrid_gen.split("\n")[0].strip()

            baselines.append({
                "prompt": prompt,
                "flat_first_line": flat_fl,
                "hybrid_first_line": hybrid_fl,
                "flat_score": binding_score(flat_fl),
                "hybrid_score": binding_score(hybrid_fl),
            })
            print(f"  {prompt}")
            print(f"    flat:   {flat_fl[:70]}")
            print(f"    hybrid: {hybrid_fl[:70]}")

        results["baselines"] = baselines
        _save(results)
    else:
        print("\n  Experiment 1: cached ✓")

    # ══════════════════════════════════════════════════════════════
    # Experiment 2: Single-head ablation
    # ══════════════════════════════════════════════════════════════
    if "single_head_ablation" not in results:
        print("\n" + "=" * 60)
        print("  EXPERIMENT 2: Single-head ablation (essential heads)")
        print("=" * 60)

        ablation_results = []
        for layer_idx, head_idx in essential_heads:
            print(f"\n  Ablating L{layer_idx}:H{head_idx}")
            head_results = {"head": f"L{layer_idx}:H{head_idx}", "probes": []}

            for prompt, _, _ in test_probes:
                flat_gen, flat_has_l, _ = zero_heads_generate(
                    model, tokenizer, flat_gate + prompt + " → ", info,
                    [(layer_idx, head_idx)], max_new_tokens=60)
                hybrid_gen, hybrid_has_l, _ = zero_heads_generate(
                    model, tokenizer, hybrid_gate + prompt + " → ", info,
                    [(layer_idx, head_idx)], max_new_tokens=60)

                flat_fl = flat_gen.split("\n")[0].strip()
                hybrid_fl = hybrid_gen.split("\n")[0].strip()

                head_results["probes"].append({
                    "prompt": prompt,
                    "flat_first_line": flat_fl, "flat_has_lambda": flat_has_l,
                    "flat_binding": binding_score(flat_fl),
                    "hybrid_first_line": hybrid_fl, "hybrid_has_lambda": hybrid_has_l,
                    "hybrid_binding": binding_score(hybrid_fl),
                })
                print(f"    {prompt}")
                print(f"      flat:   {flat_fl[:70]}  λ={'✓' if flat_has_l else '✗'}")
                print(f"      hybrid: {hybrid_fl[:70]}  bind={'✓' if has_binding(hybrid_fl) else '✗'}")

            ablation_results.append(head_results)

        results["single_head_ablation"] = ablation_results
        _save(results)
    else:
        print("\n  Experiment 2: cached ✓")

    # ══════════════════════════════════════════════════════════════
    # Experiment 3: All-3 ablation
    # ══════════════════════════════════════════════════════════════
    if "all3_ablation" not in results:
        print("\n" + "=" * 60)
        print("  EXPERIMENT 3: All 3 essential heads ablated")
        print("=" * 60)

        all3 = []
        for prompt, _, _ in test_probes:
            flat_gen, flat_has_l, _ = zero_heads_generate(
                model, tokenizer, flat_gate + prompt + " → ", info,
                essential_heads, max_new_tokens=60)
            hybrid_gen, hybrid_has_l, _ = zero_heads_generate(
                model, tokenizer, hybrid_gate + prompt + " → ", info,
                essential_heads, max_new_tokens=60)

            flat_fl = flat_gen.split("\n")[0].strip()
            hybrid_fl = hybrid_gen.split("\n")[0].strip()

            all3.append({
                "prompt": prompt,
                "flat_first_line": flat_fl, "flat_has_lambda": flat_has_l,
                "hybrid_first_line": hybrid_fl, "hybrid_has_lambda": hybrid_has_l,
                "hybrid_binding": binding_score(hybrid_fl),
            })
            print(f"  {prompt}")
            print(f"    flat:   {flat_fl[:70]}  λ={'✓' if flat_has_l else '✗'}")
            print(f"    hybrid: {hybrid_fl[:70]}  bind={'✓' if has_binding(hybrid_fl) else '✗'}")

        results["all3_ablation"] = all3
        _save(results)
    else:
        print("\n  Experiment 3: cached ✓")

    # ══════════════════════════════════════════════════════════════
    # Experiment 4: Full layer scan
    # ══════════════════════════════════════════════════════════════
    if "binding_scan" not in results:
        print("\n" + "=" * 60)
        print("  EXPERIMENT 4: Full layer scan — which heads break binding?")
        print("=" * 60)

        binding_prompt = hybrid_gate + "Everyone loves someone. → "
        baseline_gen = _generate(model, tokenizer, binding_prompt, 60)
        baseline_fl = baseline_gen.split("\n")[0].strip()
        print(f"  Baseline: {baseline_fl}")

        _, head_results_full = ablate_heads(
            model, tokenizer, binding_prompt, info,
            target_layers=list(range(info.n_layers)),
            max_new_tokens=60,
        )

        binding_scan = []
        for r in head_results_full:
            fl = r.generation.split("\n")[0].strip()
            bs = binding_score(fl)
            binding_scan.append({
                "layer": r.layer, "head": r.head, "first_line": fl,
                "has_lambda": r.has_lambda, "has_binding": bs["has_binding"],
                "n_quantifiers": bs["n_quantifiers"],
            })

        results["binding_scan"] = binding_scan
        _save(results)

        breaks = [(s["layer"], s["head"]) for s in binding_scan if not s["has_binding"]]
        print(f"\n  BREAK binding: {len(breaks)}/{len(binding_scan)}")
        for l, h in sorted(breaks):
            fl = next(s["first_line"] for s in binding_scan if s["layer"]==l and s["head"]==h)
            print(f"    L{l}:H{h}  → {fl[:60]}")
    else:
        print("\n  Experiment 4: cached ✓")

    # ══════════════════════════════════════════════════════════════
    # Experiment 5: Attention diff
    # ══════════════════════════════════════════════════════════════
    if "attention_diffs" not in results:
        print("\n" + "=" * 60)
        print("  EXPERIMENT 5: Attention patterns (flat vs hybrid)")
        print("=" * 60)

        prompt = "Everyone loves someone."
        flat_prompt = flat_gate + prompt + " → "
        hybrid_prompt = hybrid_gate + prompt + " → "

        flat_cap = record_attention(model, tokenizer, [flat_prompt])
        hybrid_cap = record_attention(model, tokenizer, [hybrid_prompt])

        flat_mat = flat_cap[flat_prompt].patterns
        hybrid_mat = hybrid_cap[hybrid_prompt].patterns

        flat_entropy = np.zeros((info.n_layers, info.n_heads))
        hybrid_entropy = np.zeros((info.n_layers, info.n_heads))

        for li in range(info.n_layers):
            for hi in range(info.n_heads):
                fa = flat_mat[li, hi, -1, :]
                ha = hybrid_mat[li, hi, -1, :]
                fa_clean = fa[fa > 1e-10]
                ha_clean = ha[ha > 1e-10]
                flat_entropy[li, hi] = -np.sum(fa_clean * np.log2(fa_clean))
                hybrid_entropy[li, hi] = -np.sum(ha_clean * np.log2(ha_clean))

        entropy_diff = hybrid_entropy - flat_entropy

        diffs = []
        for li in range(info.n_layers):
            for hi in range(info.n_heads):
                diffs.append((li, hi, entropy_diff[li, hi],
                              flat_entropy[li, hi], hybrid_entropy[li, hi]))
        diffs.sort(key=lambda x: abs(x[2]), reverse=True)

        print(f"  Top 15 heads with largest entropy change (hybrid - flat):")
        for li, hi, diff, fe, he in diffs[:15]:
            marker = " ★" if (li, hi) in essential_heads else ""
            print(f"    L{li:2d}:H{hi:2d}  Δ={diff:+.3f}  flat={fe:.3f}  hybrid={he:.3f}{marker}")

        results["attention_diffs"] = [{
            "prompt": prompt,
            "top_changed_heads": [
                {"layer": li, "head": hi, "entropy_diff": round(float(diff), 4),
                 "flat_entropy": round(float(fe), 4), "hybrid_entropy": round(float(he), 4)}
                for li, hi, diff, fe, he in diffs[:30]
            ],
        }]

        np.savez_compressed(
            RESULTS_DIR / "attention_entropy.npz",
            flat_entropy=flat_entropy, hybrid_entropy=hybrid_entropy,
            entropy_diff=entropy_diff,
        )

        _save(results)
    else:
        print("\n  Experiment 5: cached ✓")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
