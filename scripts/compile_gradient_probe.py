#!/usr/bin/env python3
"""Compile gradient probe — cross-model correlation pipeline.

Uses Qwen3-4B (via llama.cpp) as a calibrated oracle to score inputs
on the compile gradient. Then probes VSM-LM checkpoints with the same
inputs and correlates internal metrics against the Qwen scores.

Three modes:
  score   — Score probes with Qwen (run once, saves results)
  probe   — Probe a VSM-LM checkpoint (run per checkpoint)
  analyze — Correlate Qwen scores against VSM-LM metrics

Usage:
    # Step 1: Score probes with Qwen (requires llama.cpp server running)
    uv run python scripts/compile_gradient_probe.py score

    # Step 2: Probe VSM-LM at checkpoint
    uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm/step_001000.pt

    # Step 3: Analyze correlations
    uv run python scripts/compile_gradient_probe.py analyze

    # Or probe + analyze in one shot:
    uv run python scripts/compile_gradient_probe.py probe checkpoints/vsm-lm/step_001000.pt --analyze
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PROBES_PATH = Path("probes/compile-gradient.json")
GATES_DIR = Path("gates/")
RESULTS_DIR = Path("results/compile-gradient")

# Lambda-indicating tokens to measure P(λ) in Qwen output
LAMBDA_MARKERS = {"λ", "\\", "→", "∀", "∃", "∧", "∨", "¬", "(", ")"}

# Graded gate levels — from no gate to full compile gate
# Each probe is scored at every level. The response curve IS the gradient.
GATE_LEVELS = [
    ("none", None),                  # raw prompt, no gate
    ("minimal", "compile-minimal"),  # just "→ λ\n"
    ("suggestive", "compile-suggestive"),  # "Convert to logical form:\n"
    ("ambient", "compile-ambient"),  # paragraph about Montague semantics
    ("full", "compile"),             # 2-shot exemplar gate
]


# ══════════════════════════════════════════════════════════════════════
# Probe set loading
# ══════════════════════════════════════════════════════════════════════


def load_probes(probe_path: Path | None = None) -> list[dict]:
    """Load a probe set. Defaults to compile-gradient."""
    path = probe_path or PROBES_PATH
    data = json.loads(path.read_text())
    return data["probes"]


def load_gate(gate_id: str) -> str:
    """Load gate text by ID."""
    return (GATES_DIR / f"{gate_id}.txt").read_text()


def build_prompted(probe: dict, default_gate: str = "compile") -> str:
    """Build the full prompt: gate + input."""
    gate_id = probe.get("gate", default_gate)
    gate_text = load_gate(gate_id)
    return gate_text + probe["prompt"]


def measure_generation(generation: str) -> dict:
    """Measure P(λ) and formal notation presence in a generation."""
    gen_chars = list(generation)
    n_lambda = sum(1 for c in gen_chars if c in LAMBDA_MARKERS)
    p_lambda = n_lambda / max(len(gen_chars), 1)
    has_lambda = "λ" in generation or "\\" in generation
    has_formal = any(m in generation for m in ["→", "∀", "∃", "∧", "∨"])

    # Composite score
    compile_score = p_lambda
    if has_lambda:
        compile_score = max(compile_score, 0.5)
    if has_formal:
        compile_score = max(compile_score, 0.3)

    return {
        "p_lambda": round(p_lambda, 4),
        "has_lambda": has_lambda,
        "has_formal": has_formal,
        "compile_score": round(compile_score, 4),
    }


# ══════════════════════════════════════════════════════════════════════
# Mode 1: Qwen scoring — graded gate P(λ) measurement
# ══════════════════════════════════════════════════════════════════════


def score_with_qwen(
    server_url: str = "http://127.0.0.1:8080",
    n_predict: int = 60,
    temperature: float = 0.0,
    no_gate: bool = False,
) -> tuple[list[dict], str]:
    """Score each probe with Qwen3-4B across graded gate levels.

    Runs every probe at 5 gate strengths:
      none       — raw prompt (no gate)
      minimal    — "→ λ" prefix only
      suggestive — "Convert to logical form:"
      ambient    — paragraph about Montague semantics
      full       — 2-shot exemplar (the standard compile gate)

    For each probe, the response curve across gate levels IS the
    compile gradient. Inputs with high intrinsic compilability will
    respond to even minimal gates. Inputs with low compilability
    will only produce λ under the full gate (or not at all).

    The compile_score for correlation is the area under the gate
    response curve (AUC) — a single number capturing how
    compile-responsive each input is across all gate strengths.
    """
    from verbum.client import Client

    probes = load_probes()
    mode = "graded"

    # Load gate contents
    gate_contents = {}
    for level_name, gate_id in GATE_LEVELS:
        if gate_id is not None:
            gate_contents[level_name] = load_gate(gate_id)
        else:
            gate_contents[level_name] = ""

    total_calls = len(probes) * len(GATE_LEVELS)
    print(f"Scoring {len(probes)} probes × {len(GATE_LEVELS)} gate levels = {total_calls} calls")
    print(f"  Server: {server_url}")
    print(f"  Gate levels: {[g[0] for g in GATE_LEVELS]}")
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
            gate_results = {}

            for level_name, gate_id in GATE_LEVELS:
                gate_text = gate_contents[level_name]
                full_prompt = gate_text + probe["prompt"]

                t0 = time.perf_counter()
                result = client.complete(
                    full_prompt,
                    n_predict=n_predict,
                    temperature=temperature,
                    n_probs=10,
                )
                elapsed = time.perf_counter() - t0

                generation = result.content.strip()
                metrics = measure_generation(generation)

                gate_results[level_name] = {
                    "generation": generation,
                    "elapsed_ms": round(elapsed * 1000, 1),
                    **metrics,
                }

            # Compute AUC — area under the gate response curve
            # Gate levels are evenly spaced [0, 0.25, 0.5, 0.75, 1.0]
            scores = [gate_results[g[0]]["compile_score"] for g in GATE_LEVELS]
            # Trapezoidal AUC over [0, 1]
            n = len(scores)
            dx = 1.0 / (n - 1)
            auc = dx * (scores[0] / 2 + sum(scores[1:-1]) + scores[-1] / 2)

            # Slope: how much does the input respond to gating?
            # Linear regression of compile_score vs gate_strength
            gate_strengths = [i / (n - 1) for i in range(n)]
            mean_g = sum(gate_strengths) / n
            mean_s = sum(scores) / n
            num = sum((g - mean_g) * (s - mean_s) for g, s in zip(gate_strengths, scores))
            den = sum((g - mean_g) ** 2 for g in gate_strengths)
            slope = num / den if den > 1e-12 else 0.0

            probe_result = {
                "probe_id": probe["id"],
                "category": probe["category"],
                "prompt": probe["prompt"],
                "gate_results": gate_results,
                "scores_by_gate": {g[0]: gate_results[g[0]]["compile_score"] for g in GATE_LEVELS},
                "compile_score": round(auc, 4),  # AUC is the gradient score
                "gate_slope": round(slope, 4),
                "gradient_expected": probe.get("metadata", {}).get("gradient", None),
            }
            results.append(probe_result)

            # Print response curve
            curve = "  ".join(
                f"{g[0][:4]}={gate_results[g[0]]['compile_score']:.2f}"
                for g in GATE_LEVELS
            )
            print(
                f"  {probe['id']:20s}  "
                f"AUC={auc:.3f}  slope={slope:.2f}  "
                f"[{curve}]"
            )

    return results, mode


def save_qwen_scores(results: list[dict], mode: str = "gated") -> Path:
    """Save Qwen scores to results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = {
        "gated": "qwen_scores.json",
        "gateless": "qwen_scores_gateless.json",
        "graded": "qwen_scores_graded.json",
    }.get(mode, f"qwen_scores_{mode}.json")
    path = RESULTS_DIR / filename
    output = {
        "model": "Qwen3-4B",
        "mode": mode,
        "gate_levels": [g[0] for g in GATE_LEVELS],
        "n_probes": len(results),
        "timestamp": __import__("datetime").datetime.now(
            __import__("datetime").UTC
        ).isoformat(),
        "scores": results,
    }
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# Mode 1b: Compression scoring — next-token entropy + perplexity
# ══════════════════════════════════════════════════════════════════════


def score_compression(
    server_url: str = "http://127.0.0.1:8080",
) -> list[dict]:
    """Measure compression metrics for each probe via Qwen.

    Three measurements per probe (all gateless — raw input only):

    1. Next-token entropy: generate 1 token with n_probs=10.
       Low entropy = model is confident about continuation =
       well-compressed internal representation.

    2. Self-continuation perplexity: generate 20 tokens, measure
       how "determined" the continuation is (via top-1 probability
       across generated tokens).

    3. Structural complexity: tokenize the input, count tokens.
       Ratio of semantic units to tokens is a crude compression
       measure. "The dog runs" = 5 tokens, 1 predication.
       "Every student reads a book" = 7 tokens, 3 logical operators.

    The hypothesis: if the lambda compiler is a function of the
    compressor, then inputs that Qwen compresses well (low entropy,
    high confidence) should also be the ones that respond to compile
    gates (high AUC in graded scoring).
    """
    from verbum.client import Client

    probes = load_probes()

    print(f"Measuring compression for {len(probes)} probes...")
    print(f"  Server: {server_url}")
    print()

    results = []
    with Client(base_url=server_url) as client:
        health = client.health()
        print(f"  Server status: {health.status}")
        print()

        for probe in probes:
            prompt = probe["prompt"]

            # 1. Next-token entropy: 1 token, top-10 probs
            result_1 = client.complete(
                prompt,
                n_predict=1,
                temperature=0.0,
                n_probs=10,
            )

            # Extract top token probabilities from completion_probabilities
            # llama.cpp returns: [{id, token, logprob, top_logprobs: [{id, token, logprob}, ...]}, ...]
            raw = result_1.model_dump()
            comp_probs = raw.get("completion_probabilities", [])

            if comp_probs and len(comp_probs) > 0:
                # First (and only) generated token's probability distribution
                top_logprobs = comp_probs[0].get("top_logprobs", [])
                # Convert logprobs to probs
                probs = [math.exp(p.get("logprob", -30)) for p in top_logprobs]
                if probs:
                    # Entropy of the distribution
                    entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
                    top1_prob = probs[0] if probs else 0
                else:
                    entropy = float("inf")
                    top1_prob = 0
            else:
                entropy = float("inf")
                top1_prob = 0

            # 2. Short continuation: 20 tokens, measure consistency
            result_20 = client.complete(
                prompt,
                n_predict=20,
                temperature=0.0,
                n_probs=5,
            )

            raw_20 = result_20.model_dump()
            comp_probs_20 = raw_20.get("completion_probabilities", [])

            # Mean top-1 probability across continuation tokens
            top1_probs = []
            token_entropies = []
            for tp in comp_probs_20:
                top_logprobs = tp.get("top_logprobs", [])
                tprobs = [math.exp(p.get("logprob", -30)) for p in top_logprobs]
                if tprobs:
                    top1_probs.append(tprobs[0])
                    ent = -sum(p * math.log(p + 1e-12) for p in tprobs if p > 0)
                    token_entropies.append(ent)

            mean_top1 = sum(top1_probs) / max(len(top1_probs), 1)
            mean_entropy_20 = sum(token_entropies) / max(len(token_entropies), 1)

            # 3. Token count (crude structural complexity)
            tokens = client.tokenize(prompt, add_special=False)
            n_tokens = len(tokens)

            # Compression confidence: higher = more compressed
            # Invert entropy so higher = better compression
            compression_confidence = 1.0 / (1.0 + entropy)

            probe_result = {
                "probe_id": probe["id"],
                "category": probe["category"],
                "prompt": prompt,
                "first_token_entropy": round(entropy, 4),
                "first_token_top1_prob": round(top1_prob, 4),
                "continuation_mean_top1": round(mean_top1, 4),
                "continuation_mean_entropy": round(mean_entropy_20, 4),
                "n_tokens": n_tokens,
                "compression_confidence": round(compression_confidence, 4),
                "continuation": result_20.content.strip()[:80],
            }
            results.append(probe_result)

            print(
                f"  {probe['id']:20s}  "
                f"H₁={entropy:.3f}  "
                f"p₁={top1_prob:.3f}  "
                f"H̄₂₀={mean_entropy_20:.3f}  "
                f"p̄₂₀={mean_top1:.3f}  "
                f"tok={n_tokens:3d}"
            )

    return results


def save_compression_scores(results: list[dict]) -> Path:
    """Save compression scores to results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "qwen_compression.json"
    output = {
        "model": "Qwen3-4B",
        "mode": "compression",
        "n_probes": len(results),
        "timestamp": __import__("datetime").datetime.now(
            __import__("datetime").UTC
        ).isoformat(),
        "scores": results,
    }
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# Mode 1c: Multi-task probing — compressor function discovery
# ══════════════════════════════════════════════════════════════════════

# Tasks to probe — each is a different compositional function
# Original 6 tasks discovered 2 clusters (structural, semantic).
# Expanded to 25 to discover the full compressor function inventory.
TASK_GATES = [
    # ── Original 6 ──────────────────────────────────────────────────
    ("compile", "compile"),                    # NL → lambda
    ("paraphrase", "task-paraphrase"),         # NL → different NL, same meaning
    ("summarize", "task-summarize"),            # NL → shorter NL
    ("structure", "task-structure"),            # NL → SVO decomposition
    ("entail", "task-entail"),                 # NL → what follows
    ("negate", "task-negate"),                 # NL → opposite meaning
    # ── Practical / applied ─────────────────────────────────────────
    ("translate", "task-translate"),            # NL → Spanish (cross-lingual structure)
    ("question", "task-question"),              # NL → question (reverses info flow)
    ("simplify", "task-simplify"),              # NL → simpler NL (preserve meaning, reduce complexity)
    ("elaborate", "task-elaborate"),            # NL → expanded NL (inverse of summarize)
    ("formalize", "task-formalize"),            # NL → formal register (register shift)
    ("continue", "task-continue"),             # NL → prediction (what happens next)
    ("classify", "task-classify"),              # NL → category label (abstraction)
    ("sentiment", "task-sentiment"),            # NL → affect (emotional tone)
    ("keyword", "task-keyword"),               # NL → key atoms (information compression)
    ("title", "task-title"),                   # NL → headline (extreme summarization)
    ("correct", "task-correct"),               # NL → error-fixed NL (identity / repair)
    ("causality", "task-causality"),            # NL → causal reasoning
    # ── Linguistic / compositional ──────────────────────────────────
    ("coreference", "task-coreference"),        # NL → pronoun resolution (binding)
    ("presuppose", "task-presuppose"),          # NL → presuppositions (what must be true)
    ("counterfactual", "task-counterfactual"),  # NL → opposite world (deep negation)
    ("decompose", "task-decompose"),            # NL → atomic propositions
    ("disambiguate", "task-disambiguate"),      # NL → clarified meaning
    ("modality", "task-modality"),              # NL → certainty/necessity judgment
    ("scope", "task-scope"),                   # NL → quantifier/negation scope
]


def score_tasks(
    server_url: str = "http://127.0.0.1:8080",
) -> dict:
    """Score each probe under multiple task gates via Qwen.

    For each (probe, task) pair, measures:
    - Generation confidence (mean top-1 logprob across output tokens)
    - Generation entropy (mean entropy across output tokens)
    - Generation length

    The task × task correlation matrix reveals which tasks share
    compressor functions. Tasks that produce similar confidence
    profiles across inputs share internal machinery.

    The task × VSM-LM correlation reveals which VSM-LM systems
    serve which task functions.
    """
    from verbum.client import Client

    probes = load_probes()

    # Load all task gates
    gate_contents = {}
    for task_name, gate_id in TASK_GATES:
        gate_contents[task_name] = load_gate(gate_id)

    total = len(probes) * len(TASK_GATES)
    print(f"Multi-task probing: {len(probes)} probes × {len(TASK_GATES)} tasks = {total} calls")
    print(f"  Server: {server_url}")
    print(f"  Tasks: {[t[0] for t in TASK_GATES]}")
    print()

    # Results: {probe_id: {task_name: metrics}}
    all_results = []

    with Client(base_url=server_url) as client:
        health = client.health()
        print(f"  Server status: {health.status}")
        print()

        for i, probe in enumerate(probes):
            probe_tasks = {}

            for task_name, gate_id in TASK_GATES:
                gate_text = gate_contents[task_name]
                full_prompt = gate_text + probe["prompt"]

                result = client.complete(
                    full_prompt,
                    n_predict=30,
                    temperature=0.0,
                    n_probs=5,
                )

                raw = result.model_dump()
                comp_probs = raw.get("completion_probabilities", [])

                # Measure confidence and entropy across generated tokens
                top1_probs = []
                token_entropies = []
                for tp in comp_probs:
                    top_logprobs = tp.get("top_logprobs", [])
                    tprobs = [math.exp(p.get("logprob", -30)) for p in top_logprobs]
                    if tprobs:
                        top1_probs.append(tprobs[0])
                        ent = -sum(p * math.log(p + 1e-12) for p in tprobs if p > 0)
                        token_entropies.append(ent)

                mean_conf = sum(top1_probs) / max(len(top1_probs), 1)
                mean_ent = sum(token_entropies) / max(len(token_entropies), 1)

                # Also check for formal notation in output
                gen = result.content.strip()
                gen_metrics = measure_generation(gen)

                probe_tasks[task_name] = {
                    "confidence": round(mean_conf, 4),
                    "entropy": round(mean_ent, 4),
                    "gen_length": len(gen),
                    "compile_score": gen_metrics["compile_score"],
                    "generation": gen[:100],
                }

            all_results.append({
                "probe_id": probe["id"],
                "category": probe["category"],
                "prompt": probe["prompt"],
                "tasks": probe_tasks,
            })

            # Print compact summary
            conf_str = "  ".join(
                f"{t[0][:4]}={probe_tasks[t[0]]['confidence']:.2f}"
                for t in TASK_GATES
            )
            print(f"  {probe['id']:20s}  {conf_str}")

    # ── Task × Task correlation matrix ────────────────────────────
    print("\n" + "=" * 70)
    print("  TASK × TASK CORRELATION (confidence profiles)")
    print("  Tasks that correlate share compressor functions")
    print("=" * 70)

    task_names = [t[0] for t in TASK_GATES]

    # Build confidence vectors per task
    task_vectors = {}
    for tn in task_names:
        task_vectors[tn] = [r["tasks"][tn]["confidence"] for r in all_results]

    # Correlation matrix
    print(f"\n  {'':15s}", end="")
    for tn in task_names:
        print(f" {tn:>10s}", end="")
    print()
    print(f"  {'-'*15}", end="")
    for _ in task_names:
        print(f" {'-'*10}", end="")
    print()

    for t1 in task_names:
        print(f"  {t1:15s}", end="")
        for t2 in task_names:
            r = spearman_r(task_vectors[t1], task_vectors[t2])
            marker = "*" if abs(r) > 0.5 and t1 != t2 else " "
            print(f" {r:>9.3f}{marker}", end="")
        print()

    # ── Per-category task confidence ──────────────────────────────
    print(f"\n  {'Category':20s}", end="")
    for tn in task_names:
        print(f" {tn:>10s}", end="")
    print()
    print(f"  {'-'*20}", end="")
    for _ in task_names:
        print(f" {'-'*10}", end="")
    print()

    by_cat = {}
    for r in all_results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {tn: [] for tn in task_names}
        for tn in task_names:
            by_cat[cat][tn].append(r["tasks"][tn]["confidence"])

    for cat in ["strong_compile", "medium_compile", "weak_compile", "null", "anti_compile"]:
        if cat not in by_cat:
            continue
        print(f"  {cat:20s}", end="")
        for tn in task_names:
            vals = by_cat[cat][tn]
            mean = sum(vals) / len(vals)
            print(f" {mean:>10.3f}", end="")
        print()

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "qwen_tasks.json"
    output = {
        "model": "Qwen3-4B",
        "tasks": [t[0] for t in TASK_GATES],
        "n_probes": len(all_results),
        "timestamp": __import__("datetime").datetime.now(
            __import__("datetime").UTC
        ).isoformat(),
        "results": all_results,
        "task_correlation": {
            t1: {t2: round(spearman_r(task_vectors[t1], task_vectors[t2]), 4)
                 for t2 in task_names}
            for t1 in task_names
        },
    }
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  Saved: {path}")

    return output


# ══════════════════════════════════════════════════════════════════════
# Mode 2: VSM-LM probing — internal metrics per probe
# ══════════════════════════════════════════════════════════════════════


def probe_vsm_checkpoint(
    checkpoint_path: str | Path,
    device: str | None = None,
    probe_path: Path | None = None,
) -> tuple[list[dict], int, str]:
    """Run probe set through VSM-LM checkpoint, extract internal metrics.

    Auto-detects v1 vs v2 vs v3 from checkpoint state_dict.

    For each probe, extracts:
    - S4 attention entropy
    - S3 gate values per phase, per iteration
    - Register vector norm after S4 and each iteration
    - Per-phase delta and gated norms
    - Activation norms at phase boundaries

    Returns (results, step, version).
    """
    from transformers import AutoTokenizer

    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    step = ckpt["step"]

    # Auto-detect v1 vs v2 vs v3 vs v3.1 vs v3.2 vs v4 from state_dict
    state_dict = ckpt["model_state_dict"]
    is_v4 = "s3_levels.0.gate_heads.0.weight" in state_dict
    is_v3_2 = not is_v4 and "prep_layers.0.norm.weight" in state_dict
    is_v3_1 = not is_v4 and not is_v3_2 and "register_inits.reg_type" in state_dict
    is_v3 = not is_v4 and not is_v3_2 and not is_v3_1 and "register_type_init" in state_dict
    is_v2 = not is_v4 and not is_v3_2 and not is_v3_1 and not is_v3 and "s3.gate_heads.5.weight" in state_dict
    if is_v4:
        version = "v4"
    elif is_v3_2:
        version = "v3.2"
    elif is_v3_1:
        version = "v3.1"
    elif is_v3:
        version = "v3"
    elif is_v2:
        version = "v2"
    else:
        version = "v1"
    print(f"  Step: {step} ({version})")

    # Build model with same config as training
    if is_v4:
        from verbum.vsm_lm_v4 import VSMLMV4
        config = ckpt.get("config", {})
        model = VSMLMV4(
            vocab_size=config.get("vocab_size", 50277),
            d_model=config.get("d_model", 512),
            d_register=config.get("d_register", 256),
            max_len=config.get("seq_len", 4096),
            n_heads=config.get("n_heads", 8),
            d_ff=config.get("d_ff", 1536),
            d_ff_consolidate=config.get("d_ff_consolidate", 2048),
            window=config.get("window", 8),
            strides=tuple(config.get("strides", [1, 8, 64, 512])),
            n_prep_layers=config.get("n_prep_layers", 1),
            n_converge_layers=config.get("n_converge_layers", 2),
            n_consolidate_layers=config.get("n_consolidate_layers", 3),
        ).to(device)
    elif is_v3_2:
        from verbum.vsm_lm_v3_2 import VSMLMV3_2
        model = VSMLMV3_2(
            vocab_size=50277, d_model=512, d_register=256, max_len=4096,
            n_heads=8, d_ff=1536, d_ff_consolidate=2048, window=8,
            strides=(1, 8, 64), n_iterations=2,
            n_prep_layers=1, n_converge_layers=2, n_consolidate_layers=3,
        ).to(device)
    elif is_v3_1:
        from verbum.vsm_lm_v3_1 import VSMLMV3_1
        # Detect strides from checkpoint config or state_dict
        config = ckpt.get("config", {})
        strides = tuple(config.get("strides", [1, 8, 64, 512]))
        model = VSMLMV3_1(
            vocab_size=50277, d_model=512, d_register=256, max_len=4096,
            n_heads=8, d_ff=1536, window=8, strides=strides,
            n_iterations=2, n_layers_per_phase=2,
        ).to(device)
    elif is_v3:
        from verbum.vsm_lm_v3 import VSMLMV3
        model = VSMLMV3(
            vocab_size=50277, d_model=512, d_register=256, max_len=4096,
            n_heads=8, d_ff=1536, window=8, strides=(1, 8, 64),
            n_iterations=2, n_layers_per_phase=2,
        ).to(device)
    elif is_v2:
        from verbum.vsm_lm_v2 import VSMLMV2
        model = VSMLMV2(
            vocab_size=50277, d_model=256, max_len=4096,
            n_heads=8, d_ff=768, window=8, strides=(1, 8, 64),
            n_iterations=2,
        ).to(device)
    else:
        from verbum.vsm_lm import VSMLM
        model = VSMLM(
            vocab_size=50277, d_model=256, max_len=4096,
            n_heads=8, d_ff=768, window=8, strides=(1, 8, 64),
            n_iterations=2,
        ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")

    probes = load_probes(probe_path)
    data = json.loads((probe_path or PROBES_PATH).read_text())
    default_gate = data.get("default_gate", "compile")

    print(f"Probing {len(probes)} inputs at step {step}...")
    print()

    results = []
    with torch.no_grad():
        for probe in probes:
            # Use raw prompt for VSM-LM (no gate — it's a raw LM)
            prompt = probe["prompt"]
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            # Truncate if needed
            if ids.shape[1] > 4096:
                ids = ids[:, :4096]

            _, loss, metrics = model.forward_instrumented(ids)

            # Also compute the register vector itself (for PCA later)
            # Re-run forward to capture register states
            B, L = ids.shape
            positions = torch.arange(L, device=device)
            x = model.token_embed(ids) + model.pos_embed(positions)

            if is_v4:
                # v4: multi-bank registers. Extract bank_0 after S4 scan.
                bank_0 = model._init_bank0()
                s4_updates, s4_attn = model.s4([bank_0], x)
                register_after_s4 = [
                    (bank_0[i] + s4_updates[i]).detach().cpu().numpy().tolist()
                    for i in range(model.n_registers)
                ]
            elif is_v3_2 or is_v3_1 or is_v3:
                registers = model._init_registers()
                registers, s4_attn = model.s4(registers, x)
                register_after_s4 = [
                    r.detach().cpu().numpy().tolist() for r in registers
                ]
            else:
                register = model.register_init.clone()
                register, s4_attn = model.s4(register, x)
                register_after_s4 = register.detach().cpu().numpy().tolist()

            probe_result = {
                "probe_id": probe["id"],
                "category": probe["category"],
                "prompt": probe["prompt"],
                "loss": loss.item() if loss is not None else None,
                "metrics": {k: round(v, 6) if isinstance(v, float) else v
                            for k, v in metrics.items()},
                "register_after_s4": register_after_s4,
                "seq_len": ids.shape[1],
            }
            results.append(probe_result)

            if is_v4 or is_v3_2:
                print(
                    f"  {probe['id']:20s}  "
                    f"s4_ent={metrics['s4_attn_entropy']:.4f}  "
                    f"reg={metrics['register_after_s4']:.4f}  "
                    f"gates=[{metrics['iter0_prep_gate_mean']:.3f},"
                    f"{metrics['iter0_converge_gate_mean']:.3f},"
                    f"{metrics['iter0_consolidate_gate_mean']:.3f}]"
                )
            else:
                print(
                    f"  {probe['id']:20s}  "
                    f"s4_ent={metrics['s4_attn_entropy']:.4f}  "
                    f"reg={metrics['register_after_s4']:.4f}  "
                    f"gates=[{metrics['iter0_type_gate_mean']:.3f},"
                    f"{metrics['iter0_parse_gate_mean']:.3f},"
                    f"{metrics['iter0_apply_gate_mean']:.3f}]"
                )

    return results, step, version


def save_vsm_probe(results: list[dict], step: int,
                    output_dir: Path | None = None,
                    probe_set_id: str | None = None,
                    version: str | None = None) -> Path:
    """Save VSM-LM probe results."""
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ver_suffix = f"_{version}" if version else ""
    path = out_dir / f"vsm_probe_step_{step:06d}{ver_suffix}.json"
    output = {
        "model": f"VSM-LM-{version}" if version else "VSM-LM",
        "version": version,
        "step": step,
        "probe_set": probe_set_id or "compile-gradient",
        "n_probes": len(results),
        "timestamp": __import__("datetime").datetime.now(
            __import__("datetime").UTC
        ).isoformat(),
        "probes": results,
    }
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# Mode 2b: Batch probe — all checkpoints in a directory
# ══════════════════════════════════════════════════════════════════════


def batch_probe_checkpoints(
    checkpoint_dir: str | Path,
    device: str | None = None,
    skip_existing: bool = True,
) -> list[tuple[int, list[dict]]]:
    """Probe all checkpoints in a directory. Load model once, swap weights.

    Returns list of (step, probe_results) tuples, sorted by step.
    Skips checkpoints that already have results in RESULTS_DIR unless
    skip_existing is False.
    """
    from transformers import AutoTokenizer

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"  ✗ Checkpoint directory not found: {checkpoint_dir}")
        return []

    # Discover checkpoints
    ckpt_paths = sorted(checkpoint_dir.glob("step_*.pt"))
    if not ckpt_paths:
        print(f"  ✗ No step_*.pt files in {checkpoint_dir}")
        return []

    print(f"Found {len(ckpt_paths)} checkpoints in {checkpoint_dir}")

    # Filter out already-probed checkpoints
    if skip_existing:
        todo = []
        for p in ckpt_paths:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            step = ckpt["step"]
            result_path = RESULTS_DIR / f"vsm_probe_step_{step:06d}.json"
            if result_path.exists():
                print(f"  ⊘ Step {step:6d} — already probed, skipping")
            else:
                todo.append((p, step))
            del ckpt
        if not todo:
            print("  All checkpoints already probed.")
            return []
        print(f"  {len(todo)} new checkpoint(s) to probe")
    else:
        todo = []
        for p in ckpt_paths:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            todo.append((p, ckpt["step"]))
            del ckpt

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Detect architecture from first checkpoint
    first_ckpt = torch.load(todo[0][0], map_location=device, weights_only=False)
    state_dict = first_ckpt["model_state_dict"]
    is_v3_2 = "prep_layers.0.norm.weight" in state_dict
    is_v3_1 = not is_v3_2 and "register_inits.reg_type" in state_dict
    is_v3 = not is_v3_2 and not is_v3_1 and "register_type_init" in state_dict
    is_v2 = not is_v3_2 and not is_v3_1 and not is_v3 and "s3.gate_heads.5.weight" in state_dict
    if is_v3_2:
        version = "v3.2"
    elif is_v3_1:
        version = "v3.1"
    elif is_v3:
        version = "v3"
    elif is_v2:
        version = "v2"
    else:
        version = "v1"
    print(f"  Architecture: {version}")

    # Build model once
    if is_v3_2:
        from verbum.vsm_lm_v3_2 import VSMLMV3_2
        model = VSMLMV3_2(
            vocab_size=50277, d_model=512, d_register=256, max_len=4096,
            n_heads=8, d_ff=1536, d_ff_consolidate=2048, window=8,
            strides=(1, 8, 64), n_iterations=2,
            n_prep_layers=1, n_converge_layers=2, n_consolidate_layers=3,
        ).to(device)
    elif is_v3_1:
        from verbum.vsm_lm_v3_1 import VSMLMV3_1
        config = first_ckpt.get("config", {})
        strides = tuple(config.get("strides", [1, 8, 64, 512]))
        model = VSMLMV3_1(
            vocab_size=50277, d_model=512, d_register=256, max_len=4096,
            n_heads=8, d_ff=1536, window=8, strides=strides,
            n_iterations=2, n_layers_per_phase=2,
        ).to(device)
    elif is_v3:
        from verbum.vsm_lm_v3 import VSMLMV3
        model = VSMLMV3(
            vocab_size=50277, d_model=512, d_register=256, max_len=4096,
            n_heads=8, d_ff=1536, window=8, strides=(1, 8, 64),
            n_iterations=2, n_layers_per_phase=2,
        ).to(device)
    elif is_v2:
        from verbum.vsm_lm_v2 import VSMLMV2
        model = VSMLMV2(
            vocab_size=50277, d_model=256, max_len=4096,
            n_heads=8, d_ff=768, window=8, strides=(1, 8, 64),
            n_iterations=2,
        ).to(device)
    else:
        from verbum.vsm_lm import VSMLM
        model = VSMLM(
            vocab_size=50277, d_model=256, max_len=4096,
            n_heads=8, d_ff=768, window=8, strides=(1, 8, 64),
            n_iterations=2,
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")
    probes = load_probes()

    all_results = []

    for ckpt_path, step in todo:
        print(f"\n{'─' * 60}")
        print(f"  Probing step {step} ({ckpt_path.name})")
        print(f"{'─' * 60}")

        # Swap weights
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        del ckpt

        results = []
        with torch.no_grad():
            for probe in probes:
                prompt = probe["prompt"]
                ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                if ids.shape[1] > 4096:
                    ids = ids[:, :4096]

                _, loss, metrics = model.forward_instrumented(ids)

                # Register vector after S4
                B, L = ids.shape
                positions = torch.arange(L, device=device)
                x = model.token_embed(ids) + model.pos_embed(positions)

                if is_v3_2 or is_v3_1 or is_v3:
                    registers = model._init_registers()
                    registers, s4_attn = model.s4(registers, x)
                    register_after_s4 = [
                        r.detach().cpu().numpy().tolist() for r in registers
                    ]
                else:
                    register = model.register_init.clone()
                    register, s4_attn = model.s4(register, x)
                    register_after_s4 = register.detach().cpu().numpy().tolist()

                probe_result = {
                    "probe_id": probe["id"],
                    "category": probe["category"],
                    "prompt": probe["prompt"],
                    "loss": loss.item() if loss is not None else None,
                    "metrics": {k: round(v, 6) if isinstance(v, float) else v
                                for k, v in metrics.items()},
                    "register_after_s4": register_after_s4,
                    "seq_len": ids.shape[1],
                }
                results.append(probe_result)

            # Print compact summary for this checkpoint
            for pr in results:
                m = pr["metrics"]
                if is_v3_2:
                    print(
                        f"  {pr['probe_id']:20s}  "
                        f"s4_ent={m['s4_attn_entropy']:.4f}  "
                        f"reg={m['register_after_s4']:.4f}  "
                        f"gates=[{m['iter0_prep_gate_mean']:.3f},"
                        f"{m['iter0_converge_gate_mean']:.3f},"
                        f"{m['iter0_consolidate_gate_mean']:.3f}]"
                    )
                else:
                    print(
                        f"  {pr['probe_id']:20s}  "
                        f"s4_ent={m['s4_attn_entropy']:.4f}  "
                        f"reg={m['register_after_s4']:.4f}  "
                        f"gates=[{m['iter0_type_gate_mean']:.3f},"
                        f"{m['iter0_parse_gate_mean']:.3f},"
                        f"{m['iter0_apply_gate_mean']:.3f}]"
                    )

        save_vsm_probe(results, step, version=version)
        all_results.append((step, results))

    print(f"\n{'═' * 60}")
    print(f"  Batch complete: {len(all_results)} checkpoints probed")
    print(f"{'═' * 60}")

    return all_results


# ══════════════════════════════════════════════════════════════════════
# Mode 3: Correlation analysis
# ══════════════════════════════════════════════════════════════════════


def load_qwen_scores() -> dict[str, float]:
    """Load Qwen compile scores, keyed by probe_id.

    Prefers graded (AUC) > gateless > gated scores.
    """
    graded = RESULTS_DIR / "qwen_scores_graded.json"
    gateless = RESULTS_DIR / "qwen_scores_gateless.json"
    gated = RESULTS_DIR / "qwen_scores.json"

    if graded.exists():
        path = graded
    elif gateless.exists():
        path = gateless
    else:
        path = gated

    data = json.loads(path.read_text())
    mode = data.get("mode", "gated")
    print(f"  Loading Qwen scores: {path.name} (mode={mode})")
    return {s["probe_id"]: s["compile_score"] for s in data["scores"]}


def load_vsm_probes() -> list[tuple[int, dict[str, dict]]]:
    """Load all VSM probe results, sorted by step.

    Returns list of (step, {probe_id: probe_data}).
    """
    results = []
    for path in sorted(RESULTS_DIR.glob("vsm_probe_step_*.json")):
        data = json.loads(path.read_text())
        step = data["step"]
        by_id = {p["probe_id"]: p for p in data["probes"]}
        results.append((step, by_id))
    return results


def pearson_r(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    dx = [xi - mx for xi in x]
    dy = [yi - my for yi in y]
    num = sum(a * b for a, b in zip(dx, dy))
    den = (sum(a**2 for a in dx) * sum(b**2 for b in dy)) ** 0.5
    if den < 1e-12:
        return 0.0
    return num / den


def spearman_r(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation."""
    def ranks(vals):
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        r = [0.0] * len(vals)
        for rank, (orig_idx, _) in enumerate(indexed):
            r[orig_idx] = float(rank)
        return r
    return pearson_r(ranks(x), ranks(y))


def load_compression_scores() -> dict[str, dict] | None:
    """Load Qwen compression metrics, keyed by probe_id."""
    path = RESULTS_DIR / "qwen_compression.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {s["probe_id"]: s for s in data["scores"]}


def load_task_scores() -> dict[str, dict[str, float]] | None:
    """Load Qwen task confidence scores, keyed by probe_id.

    Returns {probe_id: {task_name: confidence}} or None if not available.
    """
    path = RESULTS_DIR / "qwen_tasks.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    task_names = data["tasks"]
    result = {}
    for r in data["results"]:
        pid = r["probe_id"]
        result[pid] = {tn: r["tasks"][tn]["confidence"] for tn in task_names}
    return result


def analyze_correlations(verbose: bool = True) -> dict:
    """Correlate Qwen compile scores against VSM-LM internal metrics.

    Also correlates Qwen compression metrics against both compile scores
    and VSM-LM metrics, testing the hypothesis that the lambda compiler
    is a function of the compressor.

    When task scores are available, correlates each task's confidence
    profile against VSM-LM internal metrics to discover which gates
    serve which compressor functions.
    """
    qwen_scores = load_qwen_scores()
    compression = load_compression_scores()
    task_scores = load_task_scores()
    vsm_data = load_vsm_probes()

    # ── Compile ↔ Compression correlation (Qwen-internal) ─────────
    if compression and verbose:
        common = sorted(set(qwen_scores.keys()) & set(compression.keys()))
        if len(common) >= 5:
            compile_vals = [qwen_scores[pid] for pid in common]

            comp_metrics = [
                ("first_token_entropy", "H₁ (next-token entropy)"),
                ("first_token_top1_prob", "p₁ (next-token confidence)"),
                ("continuation_mean_entropy", "H̄₂₀ (continuation entropy)"),
                ("continuation_mean_top1", "p̄₂₀ (continuation confidence)"),
                ("compression_confidence", "compression confidence"),
                ("n_tokens", "token count"),
            ]

            print("\n" + "=" * 70)
            print("  COMPILE ↔ COMPRESSION CORRELATION (Qwen-internal)")
            print("=" * 70)
            print(f"  Does the compressor predict the compiler?")
            print(f"  {'Compression metric':40s} {'Pearson':>10s} {'Spearman':>10s}")
            print(f"  {'-'*40} {'-'*10} {'-'*10}")

            for key, label in comp_metrics:
                vals = [compression[pid][key] for pid in common]
                rp = pearson_r(compile_vals, vals)
                rs = spearman_r(compile_vals, vals)
                marker = ""
                if abs(rs) > 0.5:
                    marker = " ◀◀◀"
                elif abs(rs) > 0.3:
                    marker = " ◀"
                print(f"  {label:40s} {rp:>10.4f} {rs:>10.4f}{marker}")

            print()
            # Per-category summary
            print(f"  {'Category':20s} {'AUC':>8s} {'H₁':>8s} {'p₁':>8s} {'H̄₂₀':>8s}")
            print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
            by_cat = {}
            for pid in common:
                cat = None
                for s in json.loads(PROBES_PATH.read_text())["probes"]:
                    if s["id"] == pid:
                        cat = s["category"]
                        break
                if cat not in by_cat:
                    by_cat[cat] = {"auc": [], "h1": [], "p1": [], "h20": []}
                by_cat[cat]["auc"].append(qwen_scores[pid])
                by_cat[cat]["h1"].append(compression[pid]["first_token_entropy"])
                by_cat[cat]["p1"].append(compression[pid]["first_token_top1_prob"])
                by_cat[cat]["h20"].append(compression[pid]["continuation_mean_entropy"])
            for cat in ["strong_compile", "medium_compile", "weak_compile", "null", "anti_compile"]:
                if cat not in by_cat:
                    continue
                d = by_cat[cat]
                n = len(d["auc"])
                print(
                    f"  {cat:20s} "
                    f"{sum(d['auc'])/n:>8.3f} "
                    f"{sum(d['h1'])/n:>8.3f} "
                    f"{sum(d['p1'])/n:>8.3f} "
                    f"{sum(d['h20'])/n:>8.3f}"
                )

    if not vsm_data:
        print("\nNo VSM-LM probe results found. Run 'probe' first.")
        if compression:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = RESULTS_DIR / "correlations.json"
            out_path.write_text(json.dumps({"compile_compression": "see above"}, indent=2))
            print(f"\n  Saved: {out_path}")
        return {}

    # VSM-LM metrics to correlate against both compile and compression
    metric_keys = [
        "s4_attn_entropy",
        "register_after_s4",
        "iter0_type_gate_mean", "iter0_parse_gate_mean", "iter0_apply_gate_mean",
        "iter1_type_gate_mean", "iter1_parse_gate_mean", "iter1_apply_gate_mean",
        "iter0_type_gate_std", "iter0_parse_gate_std", "iter0_apply_gate_std",
        "iter0_type_delta_norm", "iter0_parse_delta_norm", "iter0_apply_delta_norm",
        "iter0_type_gated_norm", "iter0_parse_gated_norm", "iter0_apply_gated_norm",
        "iter1_type_delta_norm", "iter1_parse_delta_norm", "iter1_apply_delta_norm",
        "iter1_type_gated_norm", "iter1_parse_gated_norm", "iter1_apply_gated_norm",
        "overall_expansion",
        "embed_norm",
        "output_norm",
        "iter0_register_norm", "iter1_register_norm",
    ]

    # Qwen compression metrics to correlate against VSM-LM
    compression_keys = [
        ("first_token_entropy", "H₁"),
        ("continuation_mean_entropy", "H̄₂₀"),
        ("first_token_top1_prob", "p₁"),
        ("continuation_mean_top1", "p̄₂₀"),
        ("compression_confidence", "comp_conf"),
    ]

    all_compile_results = {}
    all_compress_results = {}

    for step, probes_by_id in vsm_data:
        # Align probe IDs across all sources
        common_ids = sorted(set(qwen_scores.keys()) & set(probes_by_id.keys()))
        if compression:
            common_ids = sorted(set(common_ids) & set(compression.keys()))
        if len(common_ids) < 5:
            print(f"  Step {step}: too few common probes ({len(common_ids)}), skipping")
            continue

        qwen_vals = [qwen_scores[pid] for pid in common_ids]

        # ── Compile correlations (VSM-LM vs Qwen compile score) ──
        step_compile = {}
        for key in metric_keys:
            vsm_vals = [float(probes_by_id[pid].get("metrics", {}).get(key, 0.0))
                        for pid in common_ids]
            step_compile[key] = {
                "pearson": round(pearson_r(qwen_vals, vsm_vals), 4),
                "spearman": round(spearman_r(qwen_vals, vsm_vals), 4),
            }
        all_compile_results[step] = step_compile

        # ── Compression correlations (VSM-LM vs Qwen compression) ──
        if compression:
            step_compress = {}
            for comp_key, comp_label in compression_keys:
                comp_vals = [compression[pid][comp_key] for pid in common_ids]

                for vsm_key in metric_keys:
                    vsm_vals = [float(probes_by_id[pid].get("metrics", {}).get(vsm_key, 0.0))
                                for pid in common_ids]
                    combined_key = f"{vsm_key} × {comp_label}"
                    step_compress[combined_key] = {
                        "pearson": round(pearson_r(comp_vals, vsm_vals), 4),
                        "spearman": round(spearman_r(comp_vals, vsm_vals), 4),
                        "vsm_metric": vsm_key,
                        "comp_metric": comp_key,
                    }
            all_compress_results[step] = step_compress

    # ── Task × VSM-LM correlations ───────────────────────────────
    all_task_results = {}
    if task_scores and vsm_data:
        task_names = sorted(next(iter(task_scores.values())).keys())

        for step, probes_by_id in vsm_data:
            common_ids = sorted(set(task_scores.keys()) & set(probes_by_id.keys()))
            if len(common_ids) < 5:
                continue

            step_task = {}
            for tn in task_names:
                task_vals = [task_scores[pid][tn] for pid in common_ids]
                task_corrs = {}
                for key in metric_keys:
                    vsm_vals = [float(probes_by_id[pid].get("metrics", {}).get(key, 0.0))
                                for pid in common_ids]
                    task_corrs[key] = round(spearman_r(task_vals, vsm_vals), 4)
                step_task[tn] = task_corrs
            all_task_results[step] = step_task

    # ── Print compile correlations ────────────────────────────────
    if verbose and all_compile_results:
        print("\n" + "=" * 80)
        print("  COMPILER CORRELATION — VSM-LM internals vs Qwen compile AUC")
        print("  (Does the VSM-LM develop compiler-like differentiation?)")
        print("=" * 80)

        for step in sorted(all_compile_results.keys()):
            corrs = all_compile_results[step]
            print(f"\n  Step {step}:")
            print(f"  {'Metric':40s} {'Pearson':>10s} {'Spearman':>10s}")
            print(f"  {'-'*40} {'-'*10} {'-'*10}")

            sorted_keys = sorted(
                corrs.keys(),
                key=lambda k: abs(corrs[k]["spearman"]),
                reverse=True,
            )
            for key in sorted_keys[:15]:  # top 15 to keep readable
                c = corrs[key]
                marker = ""
                abs_s = abs(c["spearman"])
                if abs_s > 0.5:
                    marker = " ◀◀◀"
                elif abs_s > 0.3:
                    marker = " ◀"
                print(
                    f"  {key:40s} {c['pearson']:>10.4f} {c['spearman']:>10.4f}{marker}"
                )

    # ── Print compression correlations ────────────────────────────
    if verbose and all_compress_results:
        print("\n" + "=" * 80)
        print("  COMPRESSOR CORRELATION — VSM-LM internals vs Qwen compression")
        print("  (Does the VSM-LM compress like Qwen compresses?)")
        print("=" * 80)

        for step in sorted(all_compress_results.keys()):
            corrs = all_compress_results[step]
            print(f"\n  Step {step}:")

            # Group by compression metric, show top VSM-LM correlates
            for comp_key, comp_label in compression_keys:
                # Filter to this compression metric
                relevant = {k: v for k, v in corrs.items()
                           if v.get("comp_metric") == comp_key}
                if not relevant:
                    continue

                sorted_keys = sorted(
                    relevant.keys(),
                    key=lambda k: abs(relevant[k]["spearman"]),
                    reverse=True,
                )

                # Only show if there's something interesting (|r| > 0.15)
                top = sorted_keys[:5]
                max_r = max(abs(relevant[k]["spearman"]) for k in top)
                if max_r < 0.1:
                    continue

                print(f"\n  vs {comp_label} ({comp_key}):")
                print(f"  {'VSM-LM metric':40s} {'Pearson':>10s} {'Spearman':>10s}")
                print(f"  {'-'*40} {'-'*10} {'-'*10}")
                for key in top:
                    c = relevant[key]
                    vsm_name = c["vsm_metric"]
                    marker = ""
                    abs_s = abs(c["spearman"])
                    if abs_s > 0.5:
                        marker = " ◀◀◀"
                    elif abs_s > 0.3:
                        marker = " ◀"
                    print(
                        f"  {vsm_name:40s} {c['pearson']:>10.4f} {c['spearman']:>10.4f}{marker}"
                    )

    # ── Trajectory tables ─────────────────────────────────────────
    if verbose:
        key_metrics = [
            "s4_attn_entropy",
            "iter0_type_gate_mean", "iter0_parse_gate_mean", "iter0_apply_gate_mean",
            "iter1_type_gate_mean", "iter1_parse_gate_mean", "iter1_apply_gate_mean",
            "overall_expansion",
        ]
        steps = sorted(all_compile_results.keys())

        if len(steps) > 1:
            # Compile trajectory
            print(f"\n  {'COMPILER TRAJECTORY (Spearman)':40s}", end="")
            for s in steps:
                print(f" {'step'+str(s):>10s}", end="")
            print()
            print(f"  {'-'*40}", end="")
            for _ in steps:
                print(f" {'-'*10}", end="")
            print()
            for key in key_metrics:
                print(f"  {key:40s}", end="")
                for s in steps:
                    val = all_compile_results[s].get(key, {}).get("spearman", 0)
                    print(f" {val:>10.4f}", end="")
                print()

        if len(steps) > 1 and all_compress_results:
            # Compression trajectory — pick the strongest compression metric
            # Use H₁ (first_token_entropy) as the primary compression signal
            print(f"\n  {'COMPRESSOR TRAJECTORY vs H₁ (Spearman)':40s}", end="")
            for s in steps:
                print(f" {'step'+str(s):>10s}", end="")
            print()
            print(f"  {'-'*40}", end="")
            for _ in steps:
                print(f" {'-'*10}", end="")
            print()
            for vsm_key in key_metrics:
                combined = f"{vsm_key} × H₁"
                print(f"  {vsm_key:40s}", end="")
                for s in steps:
                    val = all_compress_results.get(s, {}).get(combined, {}).get("spearman", 0)
                    print(f" {val:>10.4f}", end="")
                print()

    # ── Task × VSM-LM correlation ───────────────────────────────
    if verbose and all_task_results:
        steps = sorted(all_task_results.keys())
        task_names_sorted = sorted(next(iter(all_task_results.values())).keys())

        # For each step, show which tasks have the strongest VSM-LM correlations
        for step in steps:
            step_data = all_task_results[step]
            print(f"\n{'=' * 100}")
            print(f"  TASK × VSM-LM CORRELATION — Step {step}")
            print(f"  Which VSM-LM gates serve which compressor functions?")
            print(f"{'=' * 100}")

            # Show top 3 VSM-LM metrics per task (sorted by |r|)
            for tn in task_names_sorted:
                corrs = step_data[tn]
                top = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
                max_r = abs(top[0][1]) if top else 0
                if max_r < 0.15:
                    continue  # skip tasks with no signal
                markers = "◀◀◀" if max_r > 0.5 else ("◀" if max_r > 0.3 else "")
                top_str = ", ".join(f"{k}={v:+.3f}" for k, v in top)
                print(f"  {tn:20s}  {top_str}  {markers}")

        # Task × gate matrix (most recent step) — the key output
        if steps:
            latest = steps[-1]
            step_data = all_task_results[latest]
            gate_metrics = [
                "iter0_type_gate_mean", "iter0_parse_gate_mean", "iter0_apply_gate_mean",
                "iter1_type_gate_mean", "iter1_parse_gate_mean", "iter1_apply_gate_mean",
                "s4_attn_entropy", "overall_expansion",
            ]
            short_names = [
                "i0_type", "i0_parse", "i0_apply",
                "i1_type", "i1_parse", "i1_apply",
                "s4_ent", "expand",
            ]

            print(f"\n{'=' * 100}")
            print(f"  TASK × GATE MATRIX — Step {latest} (Spearman r)")
            print(f"  Rows = tasks, Cols = VSM-LM gate metrics")
            print(f"{'=' * 100}")

            print(f"  {'Task':20s}", end="")
            for sn in short_names:
                print(f" {sn:>10s}", end="")
            print()
            print(f"  {'-'*20}", end="")
            for _ in short_names:
                print(f" {'-'*10}", end="")
            print()

            for tn in task_names_sorted:
                corrs = step_data[tn]
                print(f"  {tn:20s}", end="")
                for gm in gate_metrics:
                    val = corrs.get(gm, 0.0)
                    marker = "*" if abs(val) > 0.3 else " "
                    print(f" {val:>9.3f}{marker}", end="")
                print()

        # Task trajectory (if multiple steps)
        if len(steps) > 1:
            # Show how each task's max |r| evolves over training
            print(f"\n  {'TASK SIGNAL TRAJECTORY (max |Spearman|)':40s}", end="")
            for s in steps:
                print(f" {'step'+str(s):>10s}", end="")
            print()
            print(f"  {'-'*40}", end="")
            for _ in steps:
                print(f" {'-'*10}", end="")
            print()
            for tn in task_names_sorted:
                print(f"  {tn:40s}", end="")
                for s in steps:
                    corrs = all_task_results[s][tn]
                    max_r = max(abs(v) for v in corrs.values())
                    print(f" {max_r:>10.4f}", end="")
                print()

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "correlations.json"
    json_results = {
        "compile": {str(k): v for k, v in all_compile_results.items()},
        "compress": {str(k): {kk: vv for kk, vv in v.items()}
                     for k, v in all_compress_results.items()},
        "tasks": {str(k): v for k, v in all_task_results.items()},
    }
    out_path.write_text(json.dumps(json_results, indent=2))
    if verbose:
        print(f"\n  Saved: {out_path}")

    return all_compile_results


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Compile gradient probe — cross-model correlation pipeline"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # Score mode
    score_p = sub.add_parser("score", help="Score probes with Qwen3-4B via llama.cpp")
    score_p.add_argument("--server", default="http://127.0.0.1:8080")
    score_p.add_argument("--n-predict", type=int, default=60)
    score_p.add_argument("--temperature", type=float, default=0.0)
    score_p.add_argument("--no-gate", action="store_true",
                         help="Run without compile gate — measures intrinsic compile strength")

    # Compress mode
    compress_p = sub.add_parser("compress", help="Measure compression metrics via Qwen3-4B")
    compress_p.add_argument("--server", default="http://127.0.0.1:8080")

    # Tasks mode
    tasks_p = sub.add_parser("tasks", help="Multi-task probing — find compressor functions")
    tasks_p.add_argument("--server", default="http://127.0.0.1:8080")

    # Probe mode
    probe_p = sub.add_parser("probe", help="Probe a VSM-LM checkpoint")
    probe_p.add_argument("checkpoint", help="Path to checkpoint .pt file")
    probe_p.add_argument("--device", default=None)
    probe_p.add_argument("--probes", default=None,
                         help="Path to probe set JSON (default: probes/compile-gradient.json)")
    probe_p.add_argument("--analyze", action="store_true",
                         help="Also run analysis after probing")

    # Batch-probe mode
    batch_p = sub.add_parser("batch-probe", help="Probe all checkpoints in a directory")
    batch_p.add_argument("--dir", default="checkpoints/vsm-lm-v2/",
                         help="Checkpoint directory (default: checkpoints/vsm-lm-v2/)")
    batch_p.add_argument("--device", default=None)
    batch_p.add_argument("--no-skip", action="store_true",
                         help="Re-probe checkpoints even if results exist")
    batch_p.add_argument("--analyze", action="store_true",
                         help="Run full correlation analysis after probing")

    # Analyze mode
    analyze_p = sub.add_parser("analyze", help="Correlate Qwen scores vs VSM-LM metrics")

    args = parser.parse_args()

    if args.mode == "score":
        results, mode = score_with_qwen(
            server_url=args.server,
            n_predict=args.n_predict,
            temperature=args.temperature,
            no_gate=args.no_gate,
        )
        save_qwen_scores(results, mode)

        # Summary
        print("\n  Summary (compile_score = AUC across gate levels):")
        by_cat = {}
        for r in results:
            cat = r["category"]
            if cat not in by_cat:
                by_cat[cat] = {"scores": [], "slopes": []}
            by_cat[cat]["scores"].append(r["compile_score"])
            by_cat[cat]["slopes"].append(r.get("gate_slope", 0))
        for cat in sorted(by_cat.keys()):
            vals = by_cat[cat]["scores"]
            slopes = by_cat[cat]["slopes"]
            mean_s = sum(vals) / len(vals)
            mean_sl = sum(slopes) / len(slopes)
            print(f"    {cat:20s}: AUC={mean_s:.3f}  slope={mean_sl:.2f}  n={len(vals)}")

    elif args.mode == "tasks":
        output = score_tasks(server_url=args.server)

    elif args.mode == "compress":
        results = score_compression(server_url=args.server)
        save_compression_scores(results)

        # Summary
        print("\n  Summary:")
        by_cat = {}
        for r in results:
            cat = r["category"]
            if cat not in by_cat:
                by_cat[cat] = {"entropy": [], "top1": [], "tokens": []}
            by_cat[cat]["entropy"].append(r["first_token_entropy"])
            by_cat[cat]["top1"].append(r["first_token_top1_prob"])
            by_cat[cat]["tokens"].append(r["n_tokens"])
        for cat in sorted(by_cat.keys()):
            d = by_cat[cat]
            n = len(d["entropy"])
            me = sum(d["entropy"]) / n
            mt = sum(d["top1"]) / n
            mk = sum(d["tokens"]) / n
            print(f"    {cat:20s}: H₁={me:.3f}  p₁={mt:.3f}  tokens={mk:.1f}  n={n}")

    elif args.mode == "probe":
        probe_path = Path(args.probes) if args.probes else None
        results, step, version = probe_vsm_checkpoint(
            args.checkpoint, device=args.device, probe_path=probe_path,
        )

        # Determine output directory from probe set
        if probe_path:
            probe_data = json.loads(probe_path.read_text())
            probe_set_id = probe_data.get("id", probe_path.stem)
            output_dir = Path("results") / probe_set_id
        else:
            probe_set_id = None
            output_dir = None

        save_vsm_probe(results, step, output_dir=output_dir,
                        probe_set_id=probe_set_id, version=version)

        if args.analyze:
            qwen_path = RESULTS_DIR / "qwen_scores.json"
            if qwen_path.exists():
                analyze_correlations()
            else:
                print("\n  ⚠ No Qwen scores found. Run 'score' first for correlation analysis.")

    elif args.mode == "batch-probe":
        batch_probe_checkpoints(
            checkpoint_dir=args.dir,
            device=args.device,
            skip_existing=not args.no_skip,
        )
        if args.analyze:
            analyze_correlations()

    elif args.mode == "analyze":
        analyze_correlations()


if __name__ == "__main__":
    main()
