#!/usr/bin/env python3
"""Structural intervention — does L3 encode composition order?

The type probe showed types are lexical (84% in embeddings).
L0/L3 are critical but NOT for type refinement. Hypothesis:
L0/L3 encode syntactic STRUCTURE (composition order).

Test: swap hidden states at a layer between structurally different
sentences. If L3 encodes structure, then swapping L3's residual
from a transitive sentence onto an intransitive sentence should
change the OUTPUT structure (not content) — e.g., add an argument
slot or change connective structure.

Method:
  For each (donor, recipient) pair:
    1. Forward donor through model, capture layer L hidden state
    2. Generate from recipient, but hook layer L to inject donor's
       hidden state at the last sentence position
    3. Compare patched output to unpatched baseline

  Test at layers 0, 3, 5, 8, 11 to find where structure lives.

Sentence pairs chosen to differ in STRUCTURE while sharing vocabulary:
  - intransitive vs transitive (argument structure)
  - simple vs quantified (quantifier structure)
  - simple vs conditional (connective structure)
  - simple vs negation (negation structure)

Usage:
    uv run python scripts/run_structural_intervention.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import structlog

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)

log = structlog.get_logger()

from verbum.instrument import (
    _detect_lambda,
    _generate,
    _get_layers,
    load_model,
)
from test_montague_grammar import validate as validate_montague

RESULTS_DIR = Path("results/structural-intervention")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "EleutherAI/pythia-160m-deduped"

COMPILE_GATE = (
    "The dog runs. → λx. runs(dog)\n"
    "The cat sleeps. → λx. sleeps(cat)\n"
)

# ══════════════════════════════════════════════════════════════════════
# Intervention pairs
# ══════════════════════════════════════════════════════════════════════
# Each pair: (donor_sentence, recipient_sentence, structural_difference)
# Donor's structure should influence recipient's output if the layer
# encodes structure.

INTERVENTION_PAIRS = [
    # Argument structure: intransitive → transitive
    {
        "name": "intrans→trans",
        "donor": "Alice loves Bob.",
        "recipient": "The bird flies.",
        "expect": "donor is transitive (2 args), recipient is intransitive (1 arg). "
                  "If structure transfers, recipient should gain an argument slot.",
    },
    {
        "name": "trans→intrans",
        "donor": "The bird flies.",
        "recipient": "Alice loves Bob.",
        "expect": "donor is intransitive, recipient is transitive. "
                  "If structure transfers, recipient should lose an argument.",
    },
    # Quantifier structure: simple → quantified
    {
        "name": "simple→quant",
        "donor": "Every student reads a book.",
        "recipient": "The bird flies.",
        "expect": "donor has quantifier (∀). If structure transfers, "
                  "recipient should gain ∀ or structural quantification.",
    },
    {
        "name": "quant→simple",
        "donor": "The bird flies.",
        "recipient": "Every student reads a book.",
        "expect": "donor is simple. If structure transfers, "
                  "recipient should lose quantifier structure.",
    },
    # Conditional structure
    {
        "name": "simple→cond",
        "donor": "If the dog runs, the cat sleeps.",
        "recipient": "The bird flies.",
        "expect": "donor is conditional (→). If structure transfers, "
                  "recipient should gain conditional/implication.",
    },
    {
        "name": "cond→simple",
        "donor": "The bird flies.",
        "recipient": "If the dog runs, the cat sleeps.",
        "expect": "donor is simple. If structure transfers, "
                  "recipient should lose conditional structure.",
    },
    # Negation structure
    {
        "name": "simple→neg",
        "donor": "No fish swims.",
        "recipient": "The bird flies.",
        "expect": "donor has negation. If structure transfers, "
                  "recipient should gain ¬ or negation marker.",
    },
]


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════
# Activation capture and patching
# ══════════════════════════════════════════════════════════════════════


def capture_residual(model, tokenizer, text: str, layer_idx: int) -> tuple[torch.Tensor, int]:
    """Forward pass, capture hidden state at a specific layer.

    Returns (hidden_state tensor of shape [seq_len, hidden], n_tokens).
    """
    layers = _get_layers(model)
    captured = {}

    def hook_fn(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden[0].detach().clone()

    h = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        prev_attn = model.config.output_attentions
        model.config.output_attentions = False
        try:
            with torch.no_grad():
                model(**inputs)
        finally:
            model.config.output_attentions = prev_attn
    finally:
        h.remove()

    return captured["hidden"], inputs["input_ids"].shape[1]


def generate_with_patch(
    model, tokenizer, prompt: str, layer_idx: int,
    patch_hidden: torch.Tensor, patch_positions: list[int],
    max_new_tokens: int = 40,
) -> str:
    """Generate from prompt, but at layer_idx, replace hidden states
    at specified positions with patch_hidden values."""
    layers = _get_layers(model)

    def patch_hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        for i, pos in enumerate(patch_positions):
            if pos < patched.shape[1] and i < patch_hidden.shape[0]:
                patched[0, pos, :] = patch_hidden[i, :]
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched

    h = layers[layer_idx].register_forward_hook(patch_hook)
    try:
        gen = _generate(model, tokenizer, prompt, max_new_tokens)
    finally:
        h.remove()

    return gen.strip().split("\n")[0].strip()


# ══════════════════════════════════════════════════════════════════════
# Structural analysis helpers
# ══════════════════════════════════════════════════════════════════════


def structural_signature(text: str) -> dict:
    """Extract structural features from a lambda expression."""
    return {
        "has_lambda": "λ" in text,
        "has_forall": "∀" in text,
        "has_exists": "∃" in text,
        "has_neg": "¬" in text,
        "has_arrow": "→" in text,
        "has_and": "∧" in text,
        "has_or": "∨" in text,
        "n_args": text.count(",") + 1 if "(" in text else 0,
        "n_parens": text.count("("),
        "length": len(text),
    }


def structural_distance(sig_a: dict, sig_b: dict) -> int:
    """Count how many structural features differ."""
    diff = 0
    for key in sig_a:
        if sig_a[key] != sig_b[key]:
            diff += 1
    return diff


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    start = time.time()
    banner(f"STRUCTURAL INTERVENTION — {datetime.now(UTC).isoformat()}")

    model, tokenizer, info = load_model(MODEL, dtype=torch.float32)
    print(f"  Model: {MODEL} ({info.n_layers}L × {info.n_heads}H)")

    # Layers to test
    test_layers = [0, 1, 2, 3, 5, 8, 11]
    print(f"  Test layers: {test_layers}")
    print(f"  Pairs: {len(INTERVENTION_PAIRS)}")

    all_results = []

    for pair in INTERVENTION_PAIRS:
        banner(f"PAIR: {pair['name']}")
        print(f"  Donor:     {pair['donor']}")
        print(f"  Recipient: {pair['recipient']}")
        print(f"  Expect:    {pair['expect'][:80]}")

        donor_prompt = COMPILE_GATE + pair["donor"] + " →"
        recip_prompt = COMPILE_GATE + pair["recipient"] + " →"

        # Baseline: unpatched generation for both
        baseline_donor = _generate(model, tokenizer, donor_prompt, 40)
        baseline_donor = baseline_donor.strip().split("\n")[0].strip()

        baseline_recip = _generate(model, tokenizer, recip_prompt, 40)
        baseline_recip = baseline_recip.strip().split("\n")[0].strip()

        donor_sig = structural_signature(baseline_donor)
        recip_sig = structural_signature(baseline_recip)

        print(f"\n  Baselines:")
        print(f"    Donor output:     {baseline_donor}")
        print(f"    Recipient output: {baseline_recip}")
        print(f"    Structural dist:  {structural_distance(donor_sig, recip_sig)} features differ")

        # Tokenize to find sentence boundaries
        gate_tokens = tokenizer(COMPILE_GATE, return_tensors="pt")["input_ids"].shape[1]
        donor_full_tokens = tokenizer(donor_prompt, return_tensors="pt")["input_ids"].shape[1]
        recip_full_tokens = tokenizer(recip_prompt, return_tensors="pt")["input_ids"].shape[1]

        # Sentence token positions (after gate, before →)
        donor_sent_positions = list(range(gate_tokens, donor_full_tokens))
        recip_sent_positions = list(range(gate_tokens, recip_full_tokens))

        print(f"\n  Token positions: gate={gate_tokens}, "
              f"donor_sent={len(donor_sent_positions)}, "
              f"recip_sent={len(recip_sent_positions)}")

        pair_results = {
            "name": pair["name"],
            "donor": pair["donor"],
            "recipient": pair["recipient"],
            "baseline_donor": baseline_donor,
            "baseline_recip": baseline_recip,
            "donor_sig": donor_sig,
            "recip_sig": recip_sig,
            "layers": {},
        }

        # Patch at each test layer
        print(f"\n  Patched outputs (donor structure → recipient):")
        for L in test_layers:
            # Capture donor's hidden state at this layer
            donor_hidden, _ = capture_residual(model, tokenizer, donor_prompt, L)

            # Extract donor's sentence positions
            donor_sent_hidden = donor_hidden[gate_tokens:donor_full_tokens]

            # Patch: inject donor's sentence hidden into recipient at corresponding positions
            # Use min of the two lengths for overlap
            n_patch = min(len(donor_sent_positions), len(recip_sent_positions))
            patch_positions = recip_sent_positions[:n_patch]
            patch_values = donor_sent_hidden[:n_patch]

            patched_output = generate_with_patch(
                model, tokenizer, recip_prompt, L,
                patch_values, patch_positions,
                max_new_tokens=40,
            )

            patched_sig = structural_signature(patched_output)
            dist_to_donor = structural_distance(patched_sig, donor_sig)
            dist_to_recip = structural_distance(patched_sig, recip_sig)

            # Did the structure shift toward the donor?
            shift = dist_to_recip - dist_to_donor
            # positive = shifted toward donor, negative = stayed with recipient
            direction = "→DONOR" if shift > 0 else "→RECIP" if shift < 0 else "=SAME"

            tag = ""
            if L in [0, 3]:
                tag = " [CRITICAL]"
            elif L in [8, 11]:
                tag = " [SELECTIVE]"

            print(f"    L{L:2d}: {patched_output:50s}  "
                  f"d(donor)={dist_to_donor} d(recip)={dist_to_recip} "
                  f"{direction}{tag}")

            pair_results["layers"][str(L)] = {
                "patched_output": patched_output,
                "patched_sig": patched_sig,
                "dist_to_donor": dist_to_donor,
                "dist_to_recip": dist_to_recip,
                "shift_direction": direction,
            }

        all_results.append(pair_results)

    # ══════════════════════════════════════════════════════════════════
    # AGGREGATE ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    banner("AGGREGATE — which layer carries structure?")

    # For each layer, count how often patching shifts output toward donor
    print(f"  {'Layer':>6s}  {'→DONOR':>8s}  {'=SAME':>8s}  {'→RECIP':>8s}  {'shift_score':>12s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*12}")

    layer_scores = {}
    for L in test_layers:
        n_donor = 0
        n_same = 0
        n_recip = 0
        total_shift = 0

        for pr in all_results:
            lr = pr["layers"].get(str(L), {})
            d = lr.get("shift_direction", "")
            if d == "→DONOR":
                n_donor += 1
                total_shift += 1
            elif d == "→RECIP":
                n_recip += 1
                total_shift -= 1

            # Also compute raw shift value
            dd = lr.get("dist_to_donor", 0)
            dr = lr.get("dist_to_recip", 0)

        n_same = len(all_results) - n_donor - n_recip
        avg_shift = total_shift / len(all_results)
        layer_scores[L] = avg_shift

        tag = ""
        if L in [0, 3]:
            tag = " ← CRITICAL"
        elif L in [8, 11]:
            tag = " ← SELECTIVE"

        print(f"  L{L:2d}     {n_donor:>8d}  {n_same:>8d}  {n_recip:>8d}  "
              f"{avg_shift:>+11.2f}{tag}")

    # Find the layer with most structural influence
    best_layer = max(layer_scores, key=layer_scores.get)
    print(f"\n  Most structural influence: L{best_layer} "
          f"(shift score {layer_scores[best_layer]:+.2f})")

    # ── Save ──────────────────────────────────────────────────────────
    elapsed = time.time() - start
    banner(f"DONE — {elapsed:.0f}s")

    save_path = RESULTS_DIR / "intervention-summary.json"
    save_path.write_text(json.dumps({
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "model": MODEL,
        "test_layers": test_layers,
        "n_pairs": len(INTERVENTION_PAIRS),
        "layer_scores": {str(k): v for k, v in layer_scores.items()},
        "results": all_results,
    }, indent=2, ensure_ascii=False))
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
