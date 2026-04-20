#!/usr/bin/env python3
"""Circuit discovery on Pythia-160M — locate the lambda compiler.

Same pipeline as session 001 (Qwen3-4B) but on Pythia-160M, which
the scaling probe proved already has the compiler function with 8/8
content accuracy on a 2-shot gate.

12 layers × 12 heads = 144 total heads (vs 1,152 for Qwen3-4B).
Much smaller search space.

Pipeline:
  1. Layer ablation — which layers are critical?
  2. Head ablation — which specific heads in critical layers?
  3. Attention selectivity — compile vs null patterns

Usage:
    uv run python scripts/run_pythia160m_circuit.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

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
    LAMBDA_INDICATORS,
    _detect_lambda,
    _generate,
    ablate_heads,
    ablate_layers,
    head_selectivity,
    load_model,
    record_attention,
)

RESULTS_DIR = Path("results/pythia-160m-circuit")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "EleutherAI/pythia-160m-deduped"

# ── Gate prompts ──────────────────────────────────────────────────────
# 2-shot gate — proven to activate the compiler at 160M

COMPILE_GATE = (
    "The dog runs. → λx. runs(dog)\n"
    "The cat sleeps. → λx. sleeps(cat)\n"
)

# ── Probe sentences ───────────────────────────────────────────────────

COMPILE_PROBES = [
    "The bird flies.",
    "The teacher laughs.",
    "Alice helps Bob.",
    "Every student reads a book.",
    "Tom runs quickly.",
    "The fish swims.",
]

NULL_PROBES = [
    "The weather is nice today.",
    "I went to the store.",
    "She likes reading books.",
    "They arrived yesterday.",
    "The movie was interesting.",
    "He is a good friend.",
]


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def main():
    start = time.time()
    banner(f"PYTHIA-160M CIRCUIT DISCOVERY — {datetime.now(UTC).isoformat()}")

    # ── Load model ────────────────────────────────────────────────────
    model, tokenizer, info = load_model(MODEL, dtype=torch.float32)
    print(f"  Model: {MODEL}")
    print(f"  Layers: {info.n_layers}  Heads: {info.n_heads}")
    print(f"  Total heads: {info.n_layers * info.n_heads}")
    print(f"  Hidden: {info.hidden_size}  Head dim: {info.head_dim}")

    # ── Verify compilation works ──────────────────────────────────────
    banner("VERIFICATION — does the 2-shot gate work?")

    for sent in COMPILE_PROBES[:4]:
        prompt = COMPILE_GATE + f"{sent} →"
        gen = _generate(model, tokenizer, prompt, max_new_tokens=30)
        gen_line = gen.strip().split("\n")[0].strip()
        has_l = _detect_lambda(gen_line)
        print(f"  {'λ' if has_l else '·'} {sent:35s} → {gen_line}")

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: Layer Ablation
    # ══════════════════════════════════════════════════════════════════
    banner("EXPERIMENT 1: LAYER ABLATION")
    print(f"  Ablating {info.n_layers} layers × {len(COMPILE_PROBES)} probes")
    print(f"  = {info.n_layers * len(COMPILE_PROBES)} forward passes\n")

    layer_results = {}
    for sent in COMPILE_PROBES:
        prompt = COMPILE_GATE + f"{sent} →"
        baseline, results = ablate_layers(
            model, tokenizer, prompt, info, max_new_tokens=30
        )
        layer_results[sent] = {
            "baseline": baseline.strip().split("\n")[0].strip(),
            "layers": [
                {
                    "layer": r.layer,
                    "generation": r.generation.strip().split("\n")[0].strip(),
                    "has_lambda": r.has_lambda,
                    "lambda_count": r.lambda_count,
                }
                for r in results
            ],
        }

    # Aggregate: which layers break compilation?
    layer_survival = np.zeros(info.n_layers)
    for sent, lr in layer_results.items():
        for r in lr["layers"]:
            if r["has_lambda"]:
                layer_survival[r["layer"]] += 1

    n_probes = len(COMPILE_PROBES)
    print(f"\n  Layer survival rates (out of {n_probes} probes):")
    critical_layers = []
    for L in range(info.n_layers):
        rate = layer_survival[L] / n_probes
        status = "CRITICAL" if rate < 0.5 else "important" if rate < 1.0 else ""
        bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
        print(f"    L{L:2d}: {bar} {rate:.0%} {status}")
        if rate < 0.5:
            critical_layers.append(L)

    print(f"\n  Critical layers (survival < 50%): {critical_layers}")

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: Head Ablation (all layers — only 144 heads total)
    # ══════════════════════════════════════════════════════════════════
    banner("EXPERIMENT 2: HEAD ABLATION (all 144 heads)")
    print(f"  Ablating {info.n_layers} layers × {info.n_heads} heads × {len(COMPILE_PROBES)} probes")
    print(f"  = {info.n_layers * info.n_heads * len(COMPILE_PROBES)} forward passes\n")

    head_results = {}
    for sent in COMPILE_PROBES:
        prompt = COMPILE_GATE + f"{sent} →"
        baseline, results = ablate_heads(
            model, tokenizer, prompt, info,
            target_layers=list(range(info.n_layers)),
            max_new_tokens=30,
        )
        head_results[sent] = {
            "baseline": baseline.strip().split("\n")[0].strip(),
            "heads": [
                {
                    "layer": r.layer,
                    "head": r.head,
                    "generation": r.generation.strip().split("\n")[0].strip(),
                    "has_lambda": r.has_lambda,
                    "lambda_count": r.lambda_count,
                }
                for r in results
            ],
        }

    # Aggregate: which heads break compilation?
    head_survival = np.zeros((info.n_layers, info.n_heads))
    for sent, hr in head_results.items():
        for r in hr["heads"]:
            if r["has_lambda"]:
                head_survival[r["layer"], r["head"]] += 1

    print(f"\n  Head survival matrix ({info.n_layers}×{info.n_heads}):")
    print(f"        ", end="")
    for h in range(info.n_heads):
        print(f" H{h:2d}", end="")
    print()

    essential_heads = []
    for L in range(info.n_layers):
        print(f"    L{L:2d}: ", end="")
        for H in range(info.n_heads):
            rate = head_survival[L, H] / n_probes
            if rate < 0.5:
                print(f" ███", end="")
                essential_heads.append((L, H))
            elif rate < 1.0:
                print(f" ░░░", end="")
            else:
                print(f"    ", end="")
        print()

    print(f"\n  Essential heads (survival < 50%): {essential_heads}")
    print(f"  = {len(essential_heads)} / {info.n_layers * info.n_heads} "
          f"({len(essential_heads)/(info.n_layers * info.n_heads):.1%})")

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: Attention Selectivity
    # ══════════════════════════════════════════════════════════════════
    banner("EXPERIMENT 3: ATTENTION SELECTIVITY (compile vs null)")

    # Build compile and null prompts
    compile_prompts = [COMPILE_GATE + f"{s} →" for s in COMPILE_PROBES]
    null_prompts = [f"{s}" for s in NULL_PROBES]

    print(f"  Recording attention on {len(compile_prompts)} compile + "
          f"{len(null_prompts)} null prompts\n")

    compile_attn = record_attention(model, tokenizer, compile_prompts, max_new_tokens=1)
    null_attn = record_attention(model, tokenizer, null_prompts, max_new_tokens=1)

    # Compute per-head selectivity (averaged over probe pairs)
    all_selectivity = []
    compile_list = list(compile_attn.values())
    null_list = list(null_attn.values())
    n_pairs = min(len(compile_list), len(null_list))

    for i in range(n_pairs):
        sel = head_selectivity(compile_list[i], null_list[i])
        all_selectivity.append(sel)

    mean_selectivity = np.mean(all_selectivity, axis=0)

    # Top-20 most selective heads
    flat_idx = np.argsort(mean_selectivity.ravel())[::-1]
    print(f"  Top-20 most selective heads (compile vs null):")
    top_selective = []
    for rank, idx in enumerate(flat_idx[:20]):
        L, H = divmod(idx, info.n_heads)
        sel = mean_selectivity[L, H]
        essential = (L, H) in essential_heads
        marker = " ← ESSENTIAL" if essential else ""
        print(f"    #{rank+1:2d}: L{L}:H{H}  selectivity={sel:.4f}{marker}")
        top_selective.append({"layer": int(L), "head": int(H), "selectivity": float(sel)})

    # ── Save selectivity matrix ───────────────────────────────────────
    np.savez_compressed(
        RESULTS_DIR / "selectivity.npz",
        selectivity=mean_selectivity,
    )

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start
    banner(f"SUMMARY — {elapsed:.0f}s")

    print(f"  Model: {MODEL} ({info.n_layers}L × {info.n_heads}H = "
          f"{info.n_layers * info.n_heads} heads)")
    print(f"  Critical layers: {critical_layers}")
    print(f"  Essential heads: {essential_heads}")
    print(f"  Essential = {len(essential_heads)}/{info.n_layers * info.n_heads} "
          f"({len(essential_heads)/(info.n_layers * info.n_heads):.1%})")

    # Cross-reference: essential AND selective
    essential_set = set(essential_heads)
    selective_set = set((r["layer"], r["head"]) for r in top_selective[:10])
    overlap = essential_set & selective_set
    print(f"\n  Essential ∩ Top-10 selective: {sorted(overlap)}")

    # Compare to Qwen3-4B
    print(f"\n  Comparison:")
    print(f"    Qwen3-4B:    3 essential / 1152 (0.3%) — L1:H0, L24:H0, L24:H2")
    print(f"    Pythia-160M: {len(essential_heads)} essential / 144 "
          f"({len(essential_heads)/144:.1%})")

    # Save results
    save_path = RESULTS_DIR / "circuit-summary.json"
    save_path.write_text(json.dumps({
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "model": MODEL,
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_layers": info.n_layers,
        "n_heads": info.n_heads,
        "critical_layers": critical_layers,
        "essential_heads": essential_heads,
        "layer_survival": layer_survival.tolist(),
        "head_survival": head_survival.tolist(),
        "top_selective_heads": top_selective,
    }, indent=2, ensure_ascii=False))
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
