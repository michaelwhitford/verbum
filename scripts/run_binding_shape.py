#!/usr/bin/env python3
"""Binding shape probe — what computational features does binding require?

Three experiments to characterize the binding circuit's shape:

1. **Depth probing** — 1, 2, 3, 4 nested quantifiers. Where does binding
   break? If it degrades gracefully, it's attention-based. If it falls off
   a cliff, there's a fixed-size register.

2. **Residual stream progression** — capture hidden states at every layer
   for minimal pairs ("everyone loves someone" vs "someone loves everyone").
   If cosine distance grows progressively through layers 7-35, binding is
   computed incrementally (register-like). If it jumps at one layer, binding
   is circuit-like.

3. **Activation swap** — at each layer boundary, swap the residual stream
   between the two minimal-pair prompts. If swapping at layer L changes
   which scope the model outputs, binding is computed before layer L.

Together these tell us: does binding need memory (progressive state
accumulation across layers) or is it a single-pass function?

Usage:
    uv run python scripts/run_binding_shape.py
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
RESULTS_PATH = RESULTS_DIR / "binding_shape_results.json"

HYBRID_GATE = "Every dog runs. → ∀x. dog(x) → runs(x)\nA cat chased some bird. → ∃x. cat(x) ∧ ∃y. bird(y) ∧ chased(x, y)\n"


def _save(results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))


def _generate(model, tokenizer, prompt, max_new_tokens=60):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen_cfg = model.generation_config
        if getattr(gen_cfg, "temperature", None) is not None:
            gen_cfg.temperature = None
        if getattr(gen_cfg, "top_p", None) is not None:
            gen_cfg.top_p = None
        if getattr(gen_cfg, "top_k", None) is not None:
            gen_cfg.top_k = None
        prev_attn = model.config.output_attentions
        model.config.output_attentions = False
        try:
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        finally:
            model.config.output_attentions = prev_attn
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _first_line(gen: str) -> str:
    return gen.strip().split("\n")[0].strip()


def _count_quantifiers(text: str) -> int:
    return text.count("∀") + text.count("∃") + text.count("¬∃") + text.count("¬∀")


def _has_binding(text: str) -> bool:
    return ("∀" in text or "∃" in text) and ("→" in text or "∧" in text)


# ══════════════════════════════════════════════════════════════════════
# Experiment 1: Depth probing — how many quantifiers can it nest?
# ══════════════════════════════════════════════════════════════════════

DEPTH_PROBES = [
    # depth 1: single quantifier
    {
        "depth": 1,
        "prompt": "Every dog runs.",
        "expected": "∀x. dog(x) → runs(x)",
        "n_quantifiers": 1,
    },
    {
        "depth": 1,
        "prompt": "Some cat sleeps.",
        "expected": "∃x. cat(x) ∧ sleeps(x)",
        "n_quantifiers": 1,
    },
    # depth 2: two quantifiers with scope
    {
        "depth": 2,
        "prompt": "Everyone loves someone.",
        "expected": "∀x. ∃y. loves(x, y)",
        "n_quantifiers": 2,
    },
    {
        "depth": 2,
        "prompt": "Every student read a book.",
        "expected": "∀x. student(x) → ∃y. book(y) ∧ read(x, y)",
        "n_quantifiers": 2,
    },
    {
        "depth": 2,
        "prompt": "No student passed every exam.",
        "expected": "¬∃x. student(x) ∧ ∀y. exam(y) → passed(x, y)",
        "n_quantifiers": 2,
    },
    # depth 3: three quantifiers
    {
        "depth": 3,
        "prompt": "Every teacher gave some student a book.",
        "expected": "∀x. teacher(x) → ∃y. student(y) ∧ ∃z. book(z) ∧ gave(x, y, z)",
        "n_quantifiers": 3,
    },
    {
        "depth": 3,
        "prompt": "No professor assigned every student some problem.",
        "expected": "¬∃x. professor(x) ∧ ∀y. student(y) → ∃z. problem(z) ∧ assigned(x, y, z)",
        "n_quantifiers": 3,
    },
    {
        "depth": 3,
        "prompt": "Someone introduced everyone to a friend.",
        "expected": "∃x. ∀y. ∃z. friend(z) ∧ introduced(x, y, z)",
        "n_quantifiers": 3,
    },
    # depth 4: four quantifiers — stress test
    {
        "depth": 4,
        "prompt": "Every manager told some employee to give every client a report.",
        "expected": "∀w. manager(w) → ∃x. employee(x) ∧ ∀y. client(y) → ∃z. report(z) ∧ told(w, x, give(x, y, z))",
        "n_quantifiers": 4,
    },
    {
        "depth": 4,
        "prompt": "No student in every class read some book about every topic.",
        "expected": "¬∃w. ∀x. class(x) → student(w, x) ∧ ∃y. ∀z. topic(z) → book(y, z) ∧ read(w, y)",
        "n_quantifiers": 4,
    },
    # depth 5: five — beyond typical human parsing
    {
        "depth": 5,
        "prompt": "Every teacher told some student that no professor assigned every class a textbook.",
        "expected": "∀v. teacher(v) → ∃w. student(w) ∧ told(v, w, ¬∃x. professor(x) ∧ ∀y. class(y) → ∃z. textbook(z) ∧ assigned(x, y, z))",
        "n_quantifiers": 5,
    },
]


def run_depth_probing(model, tokenizer):
    """Test binding at increasing quantifier depth."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Depth probing")
    print("=" * 60)

    results = []
    for probe in DEPTH_PROBES:
        prompt = HYBRID_GATE + probe["prompt"] + " → "
        gen = _generate(model, tokenizer, prompt)
        fl = _first_line(gen)
        n_q = _count_quantifiers(fl)
        binding = _has_binding(fl)

        result = {
            "depth": probe["depth"],
            "prompt": probe["prompt"],
            "expected": probe["expected"],
            "expected_quantifiers": probe["n_quantifiers"],
            "output": fl,
            "full_generation": gen[:200],
            "output_quantifiers": n_q,
            "has_binding": binding,
            "quantifier_ratio": round(n_q / max(probe["n_quantifiers"], 1), 2),
        }
        results.append(result)

        match = "✓" if n_q >= probe["n_quantifiers"] else "✗"
        print(f"  [{match}] depth={probe['depth']} q={n_q}/{probe['n_quantifiers']} "
              f" {probe['prompt'][:40]}")
        print(f"       → {fl[:70]}")

    # Summary by depth
    print("\n  DEPTH SUMMARY:")
    for depth in sorted(set(r["depth"] for r in results)):
        depth_results = [r for r in results if r["depth"] == depth]
        avg_ratio = np.mean([r["quantifier_ratio"] for r in depth_results])
        n_binding = sum(1 for r in depth_results if r["has_binding"])
        print(f"    depth={depth}: avg_ratio={avg_ratio:.2f}, "
              f"binding={n_binding}/{len(depth_results)}")

    return results


# ══════════════════════════════════════════════════════════════════════
# Experiment 2: Residual stream progression
# ══════════════════════════════════════════════════════════════════════

MINIMAL_PAIRS = [
    ("Everyone loves someone.", "Someone loves everyone."),
    ("Every student read a book.", "A student read every book."),
    ("The cat chased the dog.", "The dog chased the cat."),
]


def capture_residuals(model, tokenizer, prompt, info):
    """Capture hidden states at every layer for the last token position."""
    from verbum.instrument import _get_layers

    layers = _get_layers(model)
    residuals = []
    hooks = []

    def make_hook(storage):
        def hook_fn(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Capture the LAST token position — that's what the model
            # uses to decide the next token (the scope ordering)
            storage.append(hidden[0, -1, :].detach().cpu().float())
        return hook_fn

    try:
        for layer in layers:
            hooks.append(layer.register_forward_hook(make_hook(residuals)))

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prev_attn = model.config.output_attentions
        model.config.output_attentions = False
        try:
            with torch.no_grad():
                model(**inputs)
        finally:
            model.config.output_attentions = prev_attn
    finally:
        for h in hooks:
            h.remove()

    return residuals  # list of (hidden_size,) tensors, one per layer


def run_residual_progression(model, tokenizer, info):
    """Compare residual streams for minimal pairs across layers."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Residual stream progression")
    print("=" * 60)

    results = []

    for prompt_a, prompt_b in MINIMAL_PAIRS:
        full_a = HYBRID_GATE + prompt_a + " → "
        full_b = HYBRID_GATE + prompt_b + " → "

        # Generate to confirm they produce different outputs
        gen_a = _first_line(_generate(model, tokenizer, full_a))
        gen_b = _first_line(_generate(model, tokenizer, full_b))
        print(f"\n  Pair: \"{prompt_a}\" vs \"{prompt_b}\"")
        print(f"    A → {gen_a[:60]}")
        print(f"    B → {gen_b[:60]}")

        # Capture residual streams
        res_a = capture_residuals(model, tokenizer, full_a, info)
        res_b = capture_residuals(model, tokenizer, full_b, info)

        # Compute cosine distance at each layer
        cosine_distances = []
        l2_distances = []
        for layer_idx in range(len(res_a)):
            a = res_a[layer_idx]
            b = res_b[layer_idx]
            cos_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            cos_dist = 1.0 - cos_sim
            l2_dist = torch.norm(a - b).item()
            cosine_distances.append(cos_dist)
            l2_distances.append(l2_dist)

        # Find the steepest climb — where does differentiation happen?
        cos_arr = np.array(cosine_distances)
        gradients = np.diff(cos_arr)
        peak_layer = int(np.argmax(gradients)) + 1  # +1 because diff shifts by 1
        total_change = cos_arr[-1] - cos_arr[0]

        # Find the layer where distance first exceeds 50% of final distance
        halfway = cos_arr[0] + total_change * 0.5
        halfway_layer = int(np.argmax(cos_arr >= halfway)) if total_change > 0 else -1

        pair_result = {
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "output_a": gen_a,
            "output_b": gen_b,
            "outputs_differ": gen_a != gen_b,
            "cosine_distances": [round(d, 6) for d in cosine_distances],
            "l2_distances": [round(d, 3) for d in l2_distances],
            "peak_gradient_layer": peak_layer,
            "peak_gradient_value": round(float(gradients[peak_layer - 1]), 6),
            "halfway_layer": halfway_layer,
            "total_cosine_change": round(float(total_change), 6),
        }
        results.append(pair_result)

        # Print key layers
        print(f"    Cosine distance progression:")
        for l in [0, 5, 10, 12, 15, 17, 20, 24, 28, 32, 35]:
            if l < len(cosine_distances):
                bar = "█" * int(cosine_distances[l] * 200)
                print(f"      L{l:2d}: {cosine_distances[l]:.6f} {bar}")
        print(f"    Peak gradient at L{peak_layer} ({gradients[peak_layer-1]:.6f})")
        print(f"    50% differentiation at L{halfway_layer}")

    return results


# ══════════════════════════════════════════════════════════════════════
# Experiment 3: Activation swap — causal test of binding location
# ══════════════════════════════════════════════════════════════════════

def swap_residual_generate(model, tokenizer, prompt, info, swap_layer, donor_residuals):
    """Generate with residual stream patched from a donor at swap_layer.

    At layer `swap_layer`, replace the last-token hidden state with
    the donor's hidden state. If the output changes to match the donor's
    scope, binding was computed before this layer.
    """
    from verbum.instrument import _get_layers

    layers = _get_layers(model)

    def swap_hook(module, args, output, *, _donor=donor_residuals[swap_layer]):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        # Only patch the last token position
        patched[0, -1, :] = _donor.to(patched.device, dtype=patched.dtype)
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    h = layers[swap_layer].register_forward_hook(swap_hook)
    try:
        gen = _generate(model, tokenizer, prompt)
    finally:
        h.remove()

    return _first_line(gen)


def run_activation_swap(model, tokenizer, info):
    """Swap residual streams between minimal pairs at each layer."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: Activation swap (causal binding location)")
    print("=" * 60)

    prompt_a = "Everyone loves someone."
    prompt_b = "Someone loves everyone."
    full_a = HYBRID_GATE + prompt_a + " → "
    full_b = HYBRID_GATE + prompt_b + " → "

    # Baselines
    gen_a = _first_line(_generate(model, tokenizer, full_a))
    gen_b = _first_line(_generate(model, tokenizer, full_b))
    print(f"\n  Baselines:")
    print(f"    A: \"{prompt_a}\" → {gen_a}")
    print(f"    B: \"{prompt_b}\" → {gen_b}")

    # Capture donor residuals from B
    res_b = capture_residuals(model, tokenizer, full_b, info)

    # At each layer, patch A's residual with B's and check output
    # If output changes from A-scope to B-scope, binding info
    # is carried in the residual stream at that layer
    print(f"\n  Patching A with B's residuals at each layer:")
    print(f"  (If output matches B, B's binding info was in the residual)")

    swap_results = []
    # Test every 2 layers for speed, plus all key layers
    test_layers = sorted(set(
        list(range(0, info.n_layers, 2)) +
        [7, 8, 10, 12, 15, 17, 20, 24, 25, 31, 33, 35]
    ))

    for layer_idx in test_layers:
        if layer_idx >= info.n_layers:
            continue
        swapped = swap_residual_generate(
            model, tokenizer, full_a, info, layer_idx, res_b
        )

        # Classify output: does it match A-scope, B-scope, or neither?
        matches_a = "∀" in swapped and swapped.find("∀") < swapped.find("∃") if "∀" in swapped and "∃" in swapped else False
        matches_b = "∃" in swapped and swapped.find("∃") < swapped.find("∀") if "∀" in swapped and "∃" in swapped else False

        # Check first quantifier
        first_q = None
        for c in swapped:
            if c in "∀∃":
                first_q = c
                break

        # A-baseline starts with ∀, B-baseline starts with ∃
        scope = "A-scope" if first_q == "∀" else "B-scope" if first_q == "∃" else "broken"

        swap_results.append({
            "layer": layer_idx,
            "output": swapped,
            "scope": scope,
            "first_quantifier": first_q,
        })

        marker = {"A-scope": "A", "B-scope": "B", "broken": "?"}[scope]
        print(f"    L{layer_idx:2d}: [{marker}] {swapped[:60]}")

    # Find transition point
    transitions = []
    for i in range(1, len(swap_results)):
        prev = swap_results[i-1]["scope"]
        curr = swap_results[i]["scope"]
        if prev != curr:
            transitions.append({
                "from_layer": swap_results[i-1]["layer"],
                "to_layer": swap_results[i]["layer"],
                "from_scope": prev,
                "to_scope": curr,
            })

    print(f"\n  Transitions:")
    for t in transitions:
        print(f"    L{t['from_layer']} ({t['from_scope']}) → L{t['to_layer']} ({t['to_scope']})")

    return {
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "baseline_a": gen_a,
        "baseline_b": gen_b,
        "swaps": swap_results,
        "transitions": transitions,
    }


# ══════════════════════════════════════════════════════════════════════
# Experiment 4: Multi-head cluster ablation — are sharpeners necessary?
# ══════════════════════════════════════════════════════════════════════

# Top sharpening heads from F63 (entropy Δ < -1.5)
SHARPEN_CLUSTER = [
    (12, 21), (21, 4), (31, 3), (10, 16), (15, 13),
    (8, 2), (18, 30), (29, 3), (9, 23), (25, 26),
    (27, 28), (26, 18), (8, 0),
]

# Top diffusing heads from F63 (entropy Δ > +1.5)
DIFFUSE_CLUSTER = [
    (17, 19), (16, 1), (25, 0), (1, 14), (26, 29),
    (21, 21), (26, 14), (24, 4), (20, 25), (20, 22),
    (0, 11), (35, 22), (13, 2),
]


def run_cluster_ablation(model, tokenizer, info):
    """Ablate sharpening and diffusing clusters to test necessity."""
    from verbum.instrument import zero_heads_generate

    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: Cluster ablation")
    print("=" * 60)

    test_probes = [
        "Everyone loves someone.",
        "Someone loves everyone.",
        "Every student read a book.",
        "No student passed every exam.",
    ]

    results = []

    # Baselines
    print("\n  Baselines:")
    baselines = {}
    for prompt in test_probes:
        full = HYBRID_GATE + prompt + " → "
        gen = _first_line(_generate(model, tokenizer, full))
        baselines[prompt] = gen
        print(f"    {prompt} → {gen[:60]}")

    # Test clusters of increasing size
    clusters = [
        ("top5_sharpen", SHARPEN_CLUSTER[:5]),
        ("top13_sharpen", SHARPEN_CLUSTER[:13]),
        ("top5_diffuse", DIFFUSE_CLUSTER[:5]),
        ("top13_diffuse", DIFFUSE_CLUSTER[:13]),
        ("all_sharpen+diffuse", SHARPEN_CLUSTER[:13] + DIFFUSE_CLUSTER[:13]),
    ]

    for cluster_name, heads in clusters:
        print(f"\n  Ablating {cluster_name} ({len(heads)} heads):")
        cluster_results = {"cluster": cluster_name, "n_heads": len(heads),
                           "heads": [(l, h) for l, h in heads], "probes": []}

        for prompt in test_probes:
            full = HYBRID_GATE + prompt + " → "
            gen, has_l, l_count = zero_heads_generate(
                model, tokenizer, full, info, heads, max_new_tokens=60
            )
            fl = _first_line(gen)
            binding = _has_binding(fl)
            n_q = _count_quantifiers(fl)

            cluster_results["probes"].append({
                "prompt": prompt,
                "output": fl,
                "baseline": baselines[prompt],
                "has_lambda": has_l,
                "has_binding": binding,
                "n_quantifiers": n_q,
                "matches_baseline": fl == baselines[prompt],
            })

            match = "=" if fl == baselines[prompt] else "≠"
            bind = "✓" if binding else "✗"
            print(f"    [{match}] [{bind}] {prompt[:30]} → {fl[:50]}")

        results.append(cluster_results)

    return results


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    from verbum.instrument import load_model

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load prior results if resuming
    results = {}
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())
        print(f"Resuming — have: {list(results.keys())}")

    print("Loading Qwen3-4B...")
    model, tokenizer, info = load_model("Qwen/Qwen3-4B")
    print(f"  {info.n_layers} layers, {info.n_heads} heads, {info.head_dim} head_dim")

    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["model"] = "Qwen/Qwen3-4B"

    # Experiment 1: Depth probing
    if "depth_probing" not in results:
        results["depth_probing"] = run_depth_probing(model, tokenizer)
        _save(results)
    else:
        print("\n  Experiment 1: cached ✓")

    # Experiment 2: Residual stream progression
    if "residual_progression" not in results:
        results["residual_progression"] = run_residual_progression(
            model, tokenizer, info
        )
        _save(results)
    else:
        print("\n  Experiment 2: cached ✓")

    # Experiment 3: Activation swap
    if "activation_swap" not in results:
        results["activation_swap"] = run_activation_swap(model, tokenizer, info)
        _save(results)
    else:
        print("\n  Experiment 3: cached ✓")

    # Experiment 4: Cluster ablation
    if "cluster_ablation" not in results:
        results["cluster_ablation"] = run_cluster_ablation(model, tokenizer, info)
        _save(results)
    else:
        print("\n  Experiment 4: cached ✓")

    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"  Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
