#!/usr/bin/env python3
"""Probe the Pythia model family for latent compilation ability.

Tests whether smaller Pythia models have the language compressor
function, even without fine-tuning. Uses the same few-shot gate
that activates compilation in Pythia-2.8B.

This directly addresses VERBUM open question #7:
"What is the smallest model that exhibits the compiler?"

And the session 004 hypothesis: does Pythia-160M already have the
compressor, but we're failing to activate it?

Usage:
    uv run python scripts/run_pythia_scaling.py

Tests: Pythia-14M, 70M, 160M, 410M, 1B, 1.4B, 2.8B
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import structlog

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)

log = structlog.get_logger()

RESULTS_DIR = Path("results/pythia-scaling")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


# ── Pythia models to test (ascending size) ────────────────────────────

PYTHIA_MODELS = [
    "EleutherAI/pythia-14m-deduped",
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
]

# ── Gate prompts to test ──────────────────────────────────────────────
# Multiple gate strengths to find the activation threshold.

GATES = {
    # Minimal: just the arrow continuation pattern
    "minimal": "{sent} →",

    # Two-shot: two exemplars then the sentence
    "two_shot": (
        "The dog runs. → λx. runs(dog)\n"
        "The cat sleeps. → λx. sleeps(cat)\n"
        "{sent} →"
    ),

    # Five-shot: richer exemplar set covering more patterns
    "five_shot": (
        "The dog runs. → λx. runs(dog)\n"
        "Alice loves Bob. → λx. loves(alice, bob)\n"
        "Every cat sleeps. → ∀x. cat(x) → sleeps(x)\n"
        "No bird flies. → ¬∃x. bird(x) ∧ flies(x)\n"
        "The fish is small. → λx. small(fish)\n"
        "{sent} →"
    ),
}

# ── Test sentences ────────────────────────────────────────────────────

TEST_SENTENCES = [
    ("The dog runs.", "simple"),
    ("The bird flies.", "simple"),
    ("Alice helps Bob.", "transitive"),
    ("Every student reads a book.", "quantified"),
    ("No fish swims.", "negation"),
    ("If the dog runs, the cat sleeps.", "conditional"),
    ("The teacher laughs.", "simple"),
    ("Tom runs quickly.", "adverb"),
]


# ── Lambda detection ──────────────────────────────────────────────────

LAMBDA_INDICATORS = ["λ", "∀", "∃", "→", "∧", "∨", "¬", "ι"]


def detect_lambda(text: str) -> tuple[bool, int]:
    """Check if text contains lambda-like content."""
    count = sum(text.count(s) for s in LAMBDA_INDICATORS)
    has = "λ" in text or count >= 3
    return has, count


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 60) -> str:
    """Generate from a prompt, return only new tokens."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_prompt = inputs["input_ids"].shape[1]

    # Clear any sampling params that conflict with greedy
    gen_cfg = model.generation_config
    for attr in ("temperature", "top_p", "top_k"):
        if getattr(gen_cfg, attr, None) is not None:
            setattr(gen_cfg, attr, None)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output[0, n_prompt:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    # Take first line only
    return text.strip().split("\n")[0].strip()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from test_montague_grammar import validate as validate_montague

    start = time.time()
    banner(f"PYTHIA SCALING PROBE — {datetime.now(UTC).isoformat()}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Models: {len(PYTHIA_MODELS)}")
    print(f"  Gates: {list(GATES.keys())}")
    print(f"  Test sentences: {len(TEST_SENTENCES)}")

    all_results = {}

    for model_name in PYTHIA_MODELS:
        banner(f"LOADING {model_name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Pythia needs fp32 on MPS
            ).to(device)
            model.eval()
        except Exception as e:
            print(f"  SKIP — failed to load: {e}")
            continue

        n_params = sum(p.numel() for p in model.parameters())
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        print(f"  Params: {n_params:,}")
        print(f"  Layers: {n_layers}  Heads: {n_heads}")

        model_results = {
            "model": model_name,
            "n_params": n_params,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "gates": {},
        }

        for gate_name, gate_template in GATES.items():
            print(f"\n  Gate: {gate_name}")
            gate_results = []

            for sent, category in TEST_SENTENCES:
                prompt = gate_template.format(sent=sent)
                gen = generate(model, tokenizer, prompt)
                has_lambda, lcount = detect_lambda(gen)
                parses, _ = validate_montague(gen)

                gate_results.append({
                    "input": sent,
                    "category": category,
                    "generated": gen,
                    "has_lambda": has_lambda,
                    "lambda_count": lcount,
                    "parses": parses,
                })

                sym = "λ" if has_lambda else "·"
                parse_sym = "P" if parses else "·"
                print(f"    {sym}{parse_sym} {sent:40s} → {gen[:60]}")

            n_lambda = sum(1 for r in gate_results if r["has_lambda"])
            n_parse = sum(1 for r in gate_results if r["parses"])
            n = len(gate_results)
            rate = n_lambda / n
            parse_rate = n_parse / n

            print(f"    P(λ)={rate:.0%}  Parse={parse_rate:.0%}  ({n_lambda}/{n})")

            model_results["gates"][gate_name] = {
                "p_lambda": rate,
                "parse_rate": parse_rate,
                "results": gate_results,
            }

        all_results[model_name] = model_results

        # Free memory
        del model
        del tokenizer
        if device == "mps":
            torch.mps.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start
    banner(f"SUMMARY — {elapsed:.0f}s")

    print(f"  {'Model':40s} {'Params':>10s}  {'minimal':>8s}  {'2-shot':>8s}  {'5-shot':>8s}")
    print(f"  {'─'*40} {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")

    for model_name, mr in all_results.items():
        short = model_name.split("/")[-1]
        params = f"{mr['n_params']/1e6:.0f}M"
        rates = []
        for gate in ["minimal", "two_shot", "five_shot"]:
            if gate in mr["gates"]:
                rates.append(f"{mr['gates'][gate]['p_lambda']:.0%}")
            else:
                rates.append("—")
        print(f"  {short:40s} {params:>10s}  {rates[0]:>8s}  {rates[1]:>8s}  {rates[2]:>8s}")

    # Save
    save_path = RESULTS_DIR / "scaling-summary.json"
    save_path.write_text(json.dumps({
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "device": device,
        "results": all_results,
    }, indent=2, ensure_ascii=False))
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
