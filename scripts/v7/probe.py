#!/usr/bin/env python3
"""v7 probe — 4-VSM Pipeline diagnostic.

Probes a VSMPipeline checkpoint measuring:
  1. Per-stage CE decomposition (CE₁-CE₄, deltas)
  2. Ternary topology stats (sparsity, gamma, cooldown, reversals)
  3. Feedback gate analysis (are gates open/closed?)
  4. Representation geometry per stage (norms, variance)
  5. Stratified evaluation (prose, compositional, technical, math)
  6. Compile gate test (can it produce lambda expressions?)

Usage:
    cd ~/src/verbum

    # Single checkpoint
    uv run python scripts/v7/probe.py checkpoints/vsm-lm-v7/step_000200

    # Multiple (evolution table)
    uv run python scripts/v7/probe.py checkpoints/vsm-lm-v7/step_*

    # Quick mode (skip generation, strata only)
    uv run python scripts/v7/probe.py checkpoints/vsm-lm-v7/step_000200 --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).parent))
from model import PipelineConfig, StageConfig, VSMPipeline, create_model
from ternary import TernaryLinear, _walk_ternary_modules

# Information-theoretic constants (must match train.py)
E_IRREDUCIBLE = 1.69
LOG_V = float(np.log(50277))
LEARNABLE_RANGE = LOG_V - E_IRREDUCIBLE

STAGE_NAMES = ["Surface", "Structural", "Semantic", "Reasoning"]

# ═══════════════════════════════════════════════════════════════════
# Stratified evaluation samples
# ═══════════════════════════════════════════════════════════════════

STRATA = {
    "prose": [
        "The cat sat on the mat and looked out the window at the birds flying south for the winter.",
        "Every student who passed the final exam received a certificate of achievement from the dean.",
        "In a quiet village nestled between rolling hills, the old baker opened his shop at dawn.",
        "She walked through the garden, pausing to admire the roses that bloomed along the fence.",
    ],
    "compositional": [
        "The man who the dog that the cat chased bit ran away quickly.",
        "If every student reads a book then some teacher who knows the author is happy.",
        "No politician who endorsed the candidate that lost the election won their own race.",
        "Every lawyer who represents a client that a judge dismissed the case against appealed.",
    ],
    "technical": [
        "The gradient of the loss with respect to the weights is computed via backpropagation.",
        "Attention scores are computed as the softmax of the scaled dot product of queries and keys.",
        "The learning rate schedule uses cosine annealing with linear warmup over 500 steps.",
        "Each layer applies layer normalization before the self-attention and feed-forward blocks.",
    ],
    "math": [
        "∀x ∈ ℝ: x² ≥ 0 ∧ x² = 0 ↔ x = 0",
        "λx. λy. apply(x, y) → result",
        "P(A|B) = P(B|A) × P(A) / P(B)",
        "∫₀¹ x² dx = 1/3",
    ],
}


# ═══════════════════════════════════════════════════════════════════
# Checkpoint loading
# ═══════════════════════════════════════════════════════════════════


def load_checkpoint(path: Path) -> tuple[VSMPipeline, int, dict]:
    """Load a VSMPipeline checkpoint.

    Returns (model, step, state_dict).
    """
    state_path = path / "state.json"
    weights_path = path / "model.npz"

    if not state_path.exists():
        print(f"  ⚠ No state.json in {path}")
        state = {}
    else:
        state = json.loads(state_path.read_text())

    config_data = state.get("config", {})
    step = state.get("step", 0)

    # Reconstruct config
    stage_configs = [
        StageConfig(**s) for s in config_data.get("stages", [
            {"n_layers": 2, "n_heads": 4, "d_model": 256, "d_ff": 512},
            {"n_layers": 3, "n_heads": 4, "d_model": 256, "d_ff": 512},
            {"n_layers": 4, "n_heads": 8, "d_model": 256, "d_ff": 1024},
            {"n_layers": 6, "n_heads": 8, "d_model": 256, "d_ff": 1024},
        ])
    ]

    cfg = PipelineConfig(
        vocab_size=config_data.get("vocab_size", 50277),
        seq_len=config_data.get("seq_len", 512),
        d_model=config_data.get("d_model", 256),
        stages=stage_configs,
        stage_positions=config_data.get("stage_positions", [512, 64, 8, 1]),
    )

    model = create_model(cfg)

    if weights_path.exists():
        weights = dict(mx.load(str(weights_path)))
        model.load_weights(list(weights.items()))
        print(f"  Loaded weights from {weights_path}")

    return model, step, state


# ═══════════════════════════════════════════════════════════════════
# Per-stage CE decomposition
# ═══════════════════════════════════════════════════════════════════


def relational_loss(loss: float) -> float:
    return min(1.0, max(0.0, (loss - E_IRREDUCIBLE) / LEARNABLE_RANGE))


def measure_stage_ce(model: VSMPipeline, tokenizer, texts: list[str]) -> dict:
    """Measure per-stage CE on a set of texts.

    Returns dict with ce_stage1..4, deltas, relational losses.
    """
    total_ce = {f"ce_stage{i}": 0.0 for i in range(1, 5)}
    total_tokens = 0

    for text in texts:
        ids = mx.array(tokenizer.encode(text), dtype=mx.int32).reshape(1, -1)
        if ids.shape[1] < 2:
            continue

        inputs = ids[:, :-1]
        targets = ids[:, 1:]
        T = targets.shape[1]

        # Pad to seq_len if needed
        if inputs.shape[1] < model.cfg.seq_len:
            pad_len = model.cfg.seq_len - inputs.shape[1]
            inputs = mx.concatenate([inputs, mx.zeros((1, pad_len), dtype=mx.int32)], axis=1)
            targets = mx.concatenate([targets, mx.zeros((1, pad_len), dtype=mx.int32)], axis=1)

        _, metrics = model.forward_with_metrics(inputs, targets=targets)

        for k in total_ce:
            total_ce[k] += metrics.get(k, 0.0) * T
        total_tokens += T

    if total_tokens == 0:
        return {}

    result = {}
    for k in total_ce:
        result[k] = total_ce[k] / total_tokens
        result[k.replace("ce_", "r_")] = relational_loss(result[k])

    # Deltas
    for i in range(2, 5):
        result[f"delta_{i}"] = result[f"ce_stage{i-1}"] - result[f"ce_stage{i}"]

    return result


# ═══════════════════════════════════════════════════════════════════
# Ternary topology analysis
# ═══════════════════════════════════════════════════════════════════


def analyze_ternary(model: VSMPipeline) -> dict:
    """Analyze ternary weight topology."""
    modules = list(_walk_ternary_modules(model))
    if not modules:
        return {"has_ternary": False}

    total_weights = 0
    total_zero = 0
    total_pos = 0
    total_neg = 0
    total_cooldown_active = 0
    total_ever_flipped = 0
    gamma_values = []
    accum_values = []

    per_module = []

    for path, mod in modules:
        stats = mod.ternary_stats()
        n = mod.out_features * mod.in_features
        total_weights += n
        total_zero += int(stats["sparsity"] * n)
        total_pos += int(stats["pos_frac"] * n)
        total_neg += int(stats["neg_frac"] * n)
        total_cooldown_active += stats["cooldown_active"]
        total_ever_flipped += stats["ever_flipped"]
        gamma_values.append(stats["gamma_mean"])
        accum_values.append(stats["accum_mean"])

        per_module.append({
            "path": path,
            "shape": f"{mod.out_features}×{mod.in_features}",
            "sparsity": stats["sparsity"],
            "gamma_mean": stats["gamma_mean"],
            "cooldown_active": stats["cooldown_active"],
            "ever_flipped": stats["ever_flipped"],
        })

    return {
        "has_ternary": True,
        "total_weights": total_weights,
        "sparsity": total_zero / total_weights if total_weights else 0,
        "pos_frac": total_pos / total_weights if total_weights else 0,
        "neg_frac": total_neg / total_weights if total_weights else 0,
        "cooldown_active": total_cooldown_active,
        "ever_flipped": total_ever_flipped,
        "ever_flipped_pct": total_ever_flipped / total_weights * 100 if total_weights else 0,
        "gamma_mean": np.mean(gamma_values) if gamma_values else 0,
        "accum_pressure": np.mean(accum_values) if accum_values else 0,
        "per_module": per_module,
    }


# ═══════════════════════════════════════════════════════════════════
# Feedback gate analysis
# ═══════════════════════════════════════════════════════════════════


def analyze_feedback_gates(model: VSMPipeline, tokenizer, texts: list[str]) -> list[dict]:
    """Measure average sigmoid gate values for each feedback module.

    High gate value = feedback is active (stage contributes).
    Low gate value = feedback is suppressed.
    """
    gate_sums = [0.0] * len(model.feedbacks)
    gate_counts = [0] * len(model.feedbacks)

    for text in texts[:4]:  # small sample
        ids = mx.array(tokenizer.encode(text), dtype=mx.int32).reshape(1, -1)
        if ids.shape[1] < 2:
            continue

        inputs = ids[:, :-1]
        if inputs.shape[1] < model.cfg.seq_len:
            pad_len = model.cfg.seq_len - inputs.shape[1]
            inputs = mx.concatenate([inputs, mx.zeros((1, pad_len), dtype=mx.int32)], axis=1)

        # Run forward to get stage outputs
        x = model.embed(inputs)
        stage_outputs = []
        h = x
        for i, stage in enumerate(model.stages):
            h = stage(h, mask=model._causal_masks[i])
            stage_outputs.append(h)
            if i < len(model.stages) - 1:
                h = model.reducers[i](h, mask=model._reduction_masks[i])

        # Measure gate values at each feedback point
        for i in range(len(model.stages) - 2, -1, -1):
            fb = model.feedbacks[i]
            lower = stage_outputs[i]
            higher = stage_outputs[i + 1]
            gate_val = mx.sigmoid(fb.gate_proj(lower))
            mean_gate = float(mx.mean(gate_val))
            gate_sums[i] += mean_gate
            gate_counts[i] += 1
            # Apply feedback for next iteration
            stage_outputs[i] = fb(lower, higher)

    results = []
    for i in range(len(model.feedbacks)):
        src = i + 2  # feedback from stage src to stage src-1
        dst = i + 1
        avg = gate_sums[i] / gate_counts[i] if gate_counts[i] > 0 else 0
        results.append({
            "feedback": f"Stage {src} → {dst}",
            "mean_gate": avg,
            "status": "active" if avg > 0.6 else "partial" if avg > 0.4 else "suppressed",
            "is_ternary": model.feedbacks[i].is_ternary,
        })
    return results


# ═══════════════════════════════════════════════════════════════════
# Representation geometry
# ═══════════════════════════════════════════════════════════════════


def analyze_representations(model: VSMPipeline, tokenizer, texts: list[str]) -> list[dict]:
    """Measure per-stage representation statistics."""
    stage_norms = [[] for _ in range(len(model.stages))]
    stage_vars = [[] for _ in range(len(model.stages))]

    for text in texts[:4]:
        ids = mx.array(tokenizer.encode(text), dtype=mx.int32).reshape(1, -1)
        if ids.shape[1] < 2:
            continue

        inputs = ids[:, :-1]
        if inputs.shape[1] < model.cfg.seq_len:
            pad_len = model.cfg.seq_len - inputs.shape[1]
            inputs = mx.concatenate([inputs, mx.zeros((1, pad_len), dtype=mx.int32)], axis=1)

        _, metrics = model.forward_with_metrics(inputs)
        for i in range(len(model.stages)):
            stage_norms[i].append(metrics.get(f"stage{i+1}_h_norm", 0))

    results = []
    for i in range(len(model.stages)):
        results.append({
            "stage": i + 1,
            "name": STAGE_NAMES[i],
            "positions": model.cfg.stage_positions[i],
            "mean_norm": np.mean(stage_norms[i]) if stage_norms[i] else 0,
            "is_ternary": model.stages[i].is_ternary,
        })
    return results


# ═══════════════════════════════════════════════════════════════════
# Compile gate test
# ═══════════════════════════════════════════════════════════════════

COMPILE_GATE = """You are a semantic compiler. Convert natural language to lambda calculus.

Example: "the cat sits" → λx.(sit x) ∧ (cat x)
Example: "every dog runs" → ∀x.(dog x) → (run x)

Convert: """

COMPILE_PROMPTS = [
    "the bird flies",
    "every student reads",
    "the man who runs",
    "no cat sleeps",
]


def compile_gate_test(model: VSMPipeline, tokenizer) -> list[dict]:
    """Test if the model can produce lambda expressions."""
    results = []

    for prompt in COMPILE_PROMPTS:
        full = COMPILE_GATE + f'"{prompt}" → '
        ids = mx.array(tokenizer.encode(full), dtype=mx.int32).reshape(1, -1)

        # Truncate if needed
        if ids.shape[1] >= model.cfg.seq_len:
            ids = ids[:, -model.cfg.seq_len + 20:]

        # Pad to seq_len
        if ids.shape[1] < model.cfg.seq_len:
            pad_len = model.cfg.seq_len - ids.shape[1]
            ids = mx.concatenate([mx.zeros((1, pad_len), dtype=mx.int32), ids], axis=1)

        # Generate 30 tokens
        prompt_len = ids.shape[1]
        generated = []
        for _ in range(30):
            logits = model(ids)
            next_logits = logits[0, -1, :]
            # Greedy
            next_id = mx.argmax(next_logits, axis=-1)
            mx.eval(next_id)
            generated.append(int(next_id))
            ids = mx.concatenate([ids[:, 1:], next_id.reshape(1, 1)], axis=1)

        gen_text = tokenizer.decode(generated)
        has_lambda = any(c in gen_text for c in "λ∀∃¬∧∨→\\")

        results.append({
            "prompt": prompt,
            "generation": gen_text[:80],
            "has_lambda": has_lambda,
        })

    return results


# ═══════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════


def print_probe_results(
    step: int,
    state: dict,
    stage_ce: dict,
    strata_ce: dict[str, dict],
    ternary_stats: dict,
    gate_analysis: list[dict],
    repr_analysis: list[dict],
    compile_results: list[dict] | None = None,
):
    """Print formatted probe results."""
    print(f"\n{'='*70}")
    print(f"  v7 Pipeline Probe — Step {step:,}")
    print(f"{'='*70}")

    # ── Training state ──
    metrics = state.get("metrics", {})
    print(f"\n  Training: loss={metrics.get('train_loss', '?'):.4f}  "
          f"r={metrics.get('relational', '?'):.3f}")

    # ── Per-stage CE ──
    print(f"\n  ── Per-Stage CE Decomposition ──")
    print(f"  {'Stage':<12} {'CE':>8} {'r':>8} {'Δ':>8}  Description")
    print(f"  {'─'*60}")
    for i in range(1, 5):
        ce = stage_ce.get(f"ce_stage{i}", 0)
        r = stage_ce.get(f"r_stage{i}", 0)
        delta = stage_ce.get(f"delta_{i}", 0) if i > 1 else 0
        delta_str = f"{delta:+.3f}" if i > 1 else "   —  "
        desc = ["surface only", "+ structural fb", "+ semantic fb", "+ reasoning fb"][i - 1]
        print(f"  CE{i:<9} {ce:8.3f} {r:8.3f} {delta_str:>8}  {desc}")

    total_delta = stage_ce.get("ce_stage1", 0) - stage_ce.get("ce_stage4", 0)
    print(f"  {'─'*60}")
    print(f"  Total feedback value: {total_delta:+.3f} nats")

    # ── Strata ──
    if strata_ce:
        print(f"\n  ── Stratified CE ──")
        print(f"  {'Stratum':<15} {'CE₁':>8} {'CE₄':>8} {'Δtotal':>8}")
        print(f"  {'─'*45}")
        for stratum, ce_data in strata_ce.items():
            ce1 = ce_data.get("ce_stage1", 0)
            ce4 = ce_data.get("ce_stage4", 0)
            dt = ce1 - ce4
            print(f"  {stratum:<15} {ce1:8.3f} {ce4:8.3f} {dt:+8.3f}")

    # ── Ternary topology ──
    if ternary_stats.get("has_ternary"):
        # Pull aggregate flip counters from checkpoint state
        total_flips = state.get("total_flips", ternary_stats.get("ever_flipped", 0))
        total_reversals = state.get("total_reversals", 0)
        flip_pct = total_flips / ternary_stats['total_weights'] * 100 if ternary_stats['total_weights'] else 0
        rev_rate = total_reversals / total_flips * 100 if total_flips > 0 else 0

        print(f"\n  ── Ternary Topology ──")
        print(f"  Weights:        {ternary_stats['total_weights']:>10,}")
        print(f"  Sparsity:       {ternary_stats['sparsity']:>10.1%}  (zero weights)")
        print(f"  Distribution:   +1={ternary_stats['pos_frac']:.1%}  "
              f"0={ternary_stats['sparsity']:.1%}  "
              f"-1={ternary_stats['neg_frac']:.1%}")
        print(f"  Gamma mean:     {ternary_stats['gamma_mean']:>10.4f}")
        print(f"  Total flips:    {total_flips:>10,}  ({flip_pct:.2f}% of topology)")
        print(f"  Reversals:      {total_reversals:>10,}  ({rev_rate:.1f}% reversal rate)")
        print(f"  Cooldown active:{ternary_stats['cooldown_active']:>10,}")
        print(f"  Accum pressure: {ternary_stats['accum_pressure']:>10.2f}")

        if ternary_stats.get("per_module"):
            print(f"\n  Per-module:")
            for mod in ternary_stats["per_module"]:
                print(f"    {mod['path']:<40s} {mod['shape']:>10s}  "
                      f"sparse={mod['sparsity']:.1%}  γ={mod['gamma_mean']:.4f}")

    # ── Feedback gates ──
    if gate_analysis:
        print(f"\n  ── Feedback Gates ──")
        for g in gate_analysis:
            t_mark = " [T]" if g["is_ternary"] else ""
            print(f"  {g['feedback']}{t_mark}:  gate={g['mean_gate']:.3f}  ({g['status']})")

    # ── Representation geometry ──
    if repr_analysis:
        print(f"\n  ── Representation Geometry ──")
        for r in repr_analysis:
            t_mark = " [T]" if r["is_ternary"] else ""
            print(f"  Stage {r['stage']} ({r['name']}){t_mark}:  "
                  f"‖h‖={r['mean_norm']:.2f}  pos={r['positions']}")

    # ── Compile gate ──
    if compile_results:
        n_lambda = sum(1 for r in compile_results if r["has_lambda"])
        print(f"\n  ── Compile Gate ({n_lambda}/{len(compile_results)} λ) ──")
        for r in compile_results:
            mark = "✓λ" if r["has_lambda"] else "  "
            print(f"  {mark} \"{r['prompt']}\"")
            print(f"     → {r['generation'][:70]}")

    print(f"\n{'='*70}")


def print_evolution(all_results: list[dict]):
    """Print evolution table across multiple checkpoints."""
    if len(all_results) < 2:
        return

    print(f"\n{'='*70}")
    print(f"  Evolution ({len(all_results)} checkpoints)")
    print(f"{'='*70}")
    print(f"  {'Step':>8} {'Loss':>8} {'CE₁':>8} {'CE₄':>8} "
          f"{'Δ₂':>7} {'Δ₃':>7} {'Δ₄':>7} {'Flipped':>8} {'Sparse':>7}")
    print(f"  {'─'*75}")

    for r in all_results:
        ce = r.get("stage_ce", {})
        ts = r.get("ternary", {})
        print(f"  {r['step']:>8,} "
              f"{r.get('loss', 0):>8.3f} "
              f"{ce.get('ce_stage1', 0):>8.3f} "
              f"{ce.get('ce_stage4', 0):>8.3f} "
              f"{ce.get('delta_2', 0):>+7.3f} "
              f"{ce.get('delta_3', 0):>+7.3f} "
              f"{ce.get('delta_4', 0):>+7.3f} "
              f"{ts.get('ever_flipped', 0):>8,} "
              f"{ts.get('sparsity', 0):>6.1%}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="v7 Pipeline Probe")
    parser.add_argument("checkpoints", type=Path, nargs="+",
                        help="Checkpoint directory/directories")
    parser.add_argument("--quick", action="store_true",
                        help="Skip compile gate test")
    parser.add_argument("--no-strata", action="store_true",
                        help="Skip stratified evaluation")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")

    # Sort checkpoints by step
    ckpts = sorted(
        [p for p in args.checkpoints if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]) if p.name.startswith("step_") else 0,
    )

    if not ckpts:
        print("  No checkpoint directories found.")
        return

    # Sample texts for evaluation
    all_texts = []
    for samples in STRATA.values():
        all_texts.extend(samples)

    all_results = []

    for ckpt_path in ckpts:
        print(f"\n  Loading: {ckpt_path}")
        model, step, state = load_checkpoint(ckpt_path)
        print(f"  Step {step:,}, seq_len={model.cfg.seq_len}")

        # ── Per-stage CE ──
        print(f"  Measuring per-stage CE...")
        stage_ce = measure_stage_ce(model, tokenizer, all_texts)

        # ── Stratified CE ──
        strata_ce = {}
        if not args.no_strata:
            print(f"  Measuring strata...")
            for stratum, samples in STRATA.items():
                strata_ce[stratum] = measure_stage_ce(model, tokenizer, samples)

        # ── Ternary analysis ──
        ternary_stats = analyze_ternary(model)

        # ── Feedback gates ──
        print(f"  Analyzing feedback gates...")
        gate_analysis = analyze_feedback_gates(model, tokenizer, all_texts[:4])

        # ── Representation geometry ──
        repr_analysis = analyze_representations(model, tokenizer, all_texts[:4])

        # ── Compile gate test ──
        compile_results = None
        if not args.quick:
            print(f"  Running compile gate test...")
            compile_results = compile_gate_test(model, tokenizer)

        # ── Display ──
        print_probe_results(
            step, state, stage_ce, strata_ce,
            ternary_stats, gate_analysis, repr_analysis,
            compile_results,
        )

        # ── Save results ──
        results_dir = Path("results/vsm-lm-v7")
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"probe_step_{step:06d}.json"
        output = {
            "timestamp": datetime.now(UTC).isoformat(),
            "architecture": "vsm-lm-v7",
            "step": step,
            "state_metrics": state.get("metrics", {}),
            "stage_ce": stage_ce,
            "strata_ce": strata_ce,
            "ternary": ternary_stats if ternary_stats.get("has_ternary") else None,
            "feedback_gates": gate_analysis,
            "representations": repr_analysis,
            "compile_results": compile_results,
            "phase_controllers": state.get("phase_controllers", []),
        }
        # Clean for JSON serialization
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_clean(v) for v in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj

        out_path.write_text(json.dumps(_clean(output), indent=2))
        print(f"\n  Saved: {out_path}")

        all_results.append({
            "step": step,
            "loss": state.get("metrics", {}).get("train_loss", 0),
            "stage_ce": stage_ce,
            "ternary": ternary_stats,
        })

    # ── Evolution table ──
    print_evolution(all_results)


if __name__ == "__main__":
    main()
