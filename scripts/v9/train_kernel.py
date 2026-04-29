"""
v9 — Kernel Router Training: Evolutionary + Gradient Hybrid

Tests the core viability question: can ternary evolution discover
routing from token embeddings to exact kernel primitives?

Two training signals:
  1. Gradient (Adam): trains continuous params toward correct routing
     logits via cross-entropy supervision on (op, arg1, arg2) targets.
  2. Evolution (tournament): mutates ternary topology, keeps mutations
     that improve routing accuracy (exact match of kernel output).

Data: random arithmetic expressions (+ a b), (- a b), (* a b)
      where a, b ∈ [0, 99]. Infinite fresh data, no memorization.

Usage:
    cd ~/src/verbum
    uv run python scripts/v9/train_kernel.py
    uv run python scripts/v9/train_kernel.py --generations 5000 --batch-size 64

License: MIT
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).parent.parent / "v8"))
from ternary import (
    TernaryLinear,
    _walk_ternary_modules,
    save_topology,
    load_topology,
    zero_ternary_grads,
    restore_ternary,
    count_ternary_weights,
    pack_ternary_mlx,
    unpack_ternary_mlx,
)

from kernel import N_OPS, OP_ADD, OP_SUB, OP_MUL, OP_NAMES, kernel_dispatch
from kernel_model import (
    KernelRouter,
    KernelRouterConfig,
    tokenize_expr,
    VOCAB_SIZE,
)


# ══════════════════════════════════════════════════════════════════════
# Data generation
# ══════════════════════════════════════════════════════════════════════

OPS = ["+", "-", "*"]
OP_TO_CODE = {"+": OP_ADD, "-": OP_SUB, "*": OP_MUL}


def generate_batch(
    rng: np.random.RandomState,
    batch_size: int,
    max_val: int = 100,
    max_len: int = 16,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Generate a batch of random arithmetic expressions with ground truth.

    Returns:
        tokens:      (B, max_len) int — tokenized expressions
        gt_ops:      (B,) int         — ground truth op codes
        gt_arg1:     (B,) int         — ground truth first operands
        gt_arg2:     (B,) int         — ground truth second operands
        gt_results:  (B,) int         — ground truth results
    """
    tokens_list = []
    ops = []
    arg1s = []
    arg2s = []
    results = []

    for _ in range(batch_size):
        op_str = OPS[rng.randint(0, len(OPS))]
        a = rng.randint(0, max_val)
        b = rng.randint(0, max_val)

        expr = f"({op_str} {a} {b})"
        toks = tokenize_expr(expr, max_len=max_len)

        if op_str == "+":
            result = a + b
        elif op_str == "-":
            result = a - b
        else:
            result = a * b

        tokens_list.append(toks)
        ops.append(OP_TO_CODE[op_str])
        arg1s.append(a)
        arg2s.append(b)
        results.append(result)

    return (
        mx.array(tokens_list),
        mx.array(ops),
        mx.array(arg1s),
        mx.array(arg2s),
        mx.array(results),
    )


# ══════════════════════════════════════════════════════════════════════
# Loss function — routing supervision
# ══════════════════════════════════════════════════════════════════════


def routing_loss(
    model: KernelRouter,
    tokens: mx.array,
    gt_ops: mx.array,
    gt_arg1: mx.array,
    gt_arg2: mx.array,
) -> mx.array:
    """Cross-entropy loss on routing logits vs ground truth (op, arg1, arg2).

    This is the GRADIENT signal for Adam — it trains continuous params
    (embeddings, gamma, norms) to produce routing logits that peak
    at the correct op/arg positions.

    The ternary topology gets zero gradient (evolutionary only).
    """
    config = model.config
    route_logits = model.forward_routing(tokens)  # (B, n_ops + 2*max_val)

    # Split into op, arg1, arg2 logit sections
    op_logits = route_logits[:, :config.n_ops]                              # (B, 3)
    arg1_logits = route_logits[:, config.n_ops:config.n_ops + config.max_val]  # (B, 100)
    arg2_logits = route_logits[:, config.n_ops + config.max_val:]              # (B, 100)

    # Cross-entropy for each component
    loss_op = nn.losses.cross_entropy(op_logits, gt_ops, reduction="mean")
    loss_a1 = nn.losses.cross_entropy(arg1_logits, gt_arg1, reduction="mean")
    loss_a2 = nn.losses.cross_entropy(arg2_logits, gt_arg2, reduction="mean")

    return loss_op + loss_a1 + loss_a2


# ══════════════════════════════════════════════════════════════════════
# Evaluation — kernel accuracy
# ══════════════════════════════════════════════════════════════════════


def evaluate_accuracy(
    model: KernelRouter,
    rng: np.random.RandomState,
    n_samples: int = 256,
    max_val: int = 100,
) -> dict:
    """Evaluate kernel routing accuracy on fresh random expressions.

    Returns dict with overall accuracy and per-op breakdown.
    """
    tokens, gt_ops, gt_arg1, gt_arg2, gt_results = generate_batch(
        rng, n_samples, max_val=max_val,
    )

    _, pred_op, pred_a1, pred_a2, pred_result = model(tokens)
    mx.eval(pred_op, pred_a1, pred_a2, pred_result)

    # Convert to numpy for analysis
    pred_op_np = np.array(pred_op)
    pred_a1_np = np.array(pred_a1)
    pred_a2_np = np.array(pred_a2)
    pred_result_np = np.array(pred_result)
    gt_ops_np = np.array(gt_ops)
    gt_a1_np = np.array(gt_arg1)
    gt_a2_np = np.array(gt_arg2)
    gt_results_np = np.array(gt_results)

    # Exact match: kernel got the right answer
    result_correct = (pred_result_np == gt_results_np)
    # Component matches
    op_correct = (pred_op_np == gt_ops_np)
    a1_correct = (pred_a1_np == gt_a1_np)
    a2_correct = (pred_a2_np == gt_a2_np)
    # Full routing correct (op AND arg1 AND arg2 all right)
    route_correct = op_correct & a1_correct & a2_correct

    # Per-op breakdown
    per_op = {}
    for op_code, op_name in OP_NAMES.items():
        mask = gt_ops_np == op_code
        if mask.sum() > 0:
            per_op[op_name] = {
                "count": int(mask.sum()),
                "result_acc": float(result_correct[mask].mean()),
                "op_acc": float(op_correct[mask].mean()),
                "arg1_acc": float(a1_correct[mask].mean()),
                "arg2_acc": float(a2_correct[mask].mean()),
                "route_acc": float(route_correct[mask].mean()),
            }

    return {
        "result_accuracy": float(result_correct.mean()),
        "route_accuracy": float(route_correct.mean()),
        "op_accuracy": float(op_correct.mean()),
        "arg1_accuracy": float(a1_correct.mean()),
        "arg2_accuracy": float(a2_correct.mean()),
        "per_op": per_op,
    }


# ══════════════════════════════════════════════════════════════════════
# Simple mutation for the tiny model
# ══════════════════════════════════════════════════════════════════════


def mutate_model(model: KernelRouter, budget: int, rng: np.random.RandomState) -> int:
    """Simple uniform mutation for the tiny kernel router.

    No importance weighting or depth priorities — model is too small
    to benefit from those. Just flip `budget` random ternary weights.
    """
    from ternary import mutate_topology
    return mutate_topology(model, budget, rng, sign_flip_rate=0.2)


# ══════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════


def train(
    generations: int = 2000,
    batch_size: int = 32,
    adam_steps_per_gen: int = 5,
    lr: float = 1e-3,
    mutation_pct: float = 0.02,
    eval_interval: int = 50,
    eval_samples: int = 512,
    seed: int = 42,
    max_val: int = 100,
    d_model: int = 64,
    no_evolution: bool = False,
):
    """Main training loop: interleave Adam steps with evolutionary mutation.

    Each generation:
      1. Adam updates continuous params for `adam_steps_per_gen` steps
         (gradient on routing supervision loss)
      2. Mutate ternary topology
      3. Evaluate: if mutation improved routing accuracy, keep it;
         otherwise revert (champion never degrades)

    This interleaving lets both signals contribute:
    - Adam shapes the representation so routing logits are meaningful
    - Evolution wires the ternary topology to route correctly
    """
    print("=" * 70)
    print("  v9 — Kernel Router Training")
    print("=" * 70)

    rng = np.random.RandomState(seed)
    config = KernelRouterConfig(max_val=max_val, d_model=d_model)
    model = KernelRouter(config)

    # Count ternary weights
    n_ternary = count_ternary_weights(model)
    mutation_budget = max(1, int(n_ternary * mutation_pct))

    print(f"\nConfig:")
    print(f"  d_model:          {config.d_model}")
    print(f"  n_mix_layers:     {config.n_mix_layers}")
    print(f"  max_val:          {max_val}")
    print(f"  route_dim:        {config.n_ops + 2 * config.max_val}")
    print(f"  ternary weights:  {n_ternary:,}")
    print(f"  mutation budget:  {mutation_budget:,} ({mutation_pct*100:.1f}%)")
    print(f"  generations:      {generations}")
    print(f"  adam steps/gen:   {adam_steps_per_gen}")
    print(f"  batch size:       {batch_size}")
    print(f"  learning rate:    {lr}")
    print(f"  evolution:        {'OFF (Adam only)' if no_evolution else 'ON'}")

    # Optimizer for continuous params
    optimizer = optim.Adam(learning_rate=lr)

    # Loss + grad function
    loss_and_grad = nn.value_and_grad(model, routing_loss)

    # Champion tracking for evolutionary selection
    best_accuracy = -1.0
    champion_topology = save_topology(model)

    # Initial eval
    eval_rng = np.random.RandomState(seed + 1000)
    metrics = evaluate_accuracy(model, eval_rng, n_samples=eval_samples, max_val=max_val)
    best_accuracy = metrics["route_accuracy"]
    print(f"\nInitial accuracy:")
    print(f"  Route: {metrics['route_accuracy']*100:.1f}%  "
          f"Op: {metrics['op_accuracy']*100:.1f}%  "
          f"Arg1: {metrics['arg1_accuracy']*100:.1f}%  "
          f"Arg2: {metrics['arg2_accuracy']*100:.1f}%  "
          f"Result: {metrics['result_accuracy']*100:.1f}%")

    print(f"\n{'Gen':>6}  {'Loss':>8}  {'Route%':>7}  {'Op%':>5}  "
          f"{'A1%':>5}  {'A2%':>5}  {'Res%':>5}  {'Mut':>5}  {'dt':>6}")
    print("-" * 70)

    t_start = time.time()
    total_adam_steps = 0
    mutations_accepted = 0
    mutations_total = 0

    for gen in range(generations):
        gen_start = time.time()

        # ── Phase 1: Adam steps on continuous params ──
        avg_loss = 0.0
        for _ in range(adam_steps_per_gen):
            tokens, gt_ops, gt_a1, gt_a2, gt_res = generate_batch(
                rng, batch_size, max_val=max_val,
            )
            loss, grads = loss_and_grad(model, tokens, gt_ops, gt_a1, gt_a2)
            # Zero out ternary weight gradients — they evolve, not gradient descend
            grads = zero_ternary_grads(model, grads)
            optimizer.update(model, grads)
            restore_ternary(model)
            mx.eval(model.parameters(), optimizer.state)
            avg_loss += loss.item()
            total_adam_steps += 1

        avg_loss /= adam_steps_per_gen

        # ── Phase 2: Evolutionary mutation ──
        if no_evolution:
            pass  # skip mutation entirely
        else:
            pre_mutation_topology = save_topology(model)
            mutate_model(model, mutation_budget, rng)
            mx.eval(model.parameters())

        # ── Phase 3: Evaluate and select ──
        if no_evolution:
            # No tournament — just log at eval intervals
            if gen % eval_interval == 0 or gen == generations - 1:
                eval_rng_local = np.random.RandomState(seed + gen + 2000)
                metrics = evaluate_accuracy(
                    model, eval_rng_local, n_samples=eval_samples, max_val=max_val,
                )
                dt = time.time() - gen_start
                print(f"  {gen:5d}  {avg_loss:8.4f}  "
                      f"{metrics['route_accuracy']*100:6.1f}%  "
                      f"{metrics['op_accuracy']*100:4.1f}%  "
                      f"{metrics['arg1_accuracy']*100:4.1f}%  "
                      f"{metrics['arg2_accuracy']*100:4.1f}%  "
                      f"{metrics['result_accuracy']*100:4.1f}%  "
                      f"    —  {dt:5.2f}s")
                if metrics["route_accuracy"] >= 0.99:
                    print(f"\n  🎯 Routing converged at generation {gen}!")
                    break
            continue

        if gen % eval_interval == 0 or gen == generations - 1:
            eval_rng_local = np.random.RandomState(seed + gen + 2000)
            metrics = evaluate_accuracy(
                model, eval_rng_local, n_samples=eval_samples, max_val=max_val,
            )
            current_accuracy = metrics["route_accuracy"]

            # Champion selection: keep if improved or equal
            if current_accuracy >= best_accuracy:
                best_accuracy = current_accuracy
                champion_topology = save_topology(model)
                mutations_accepted += 1
                accepted = "✓"
            else:
                # Revert to champion
                load_topology(model, champion_topology)
                accepted = "✗"
            mutations_total += 1

            dt = time.time() - gen_start
            print(f"  {gen:5d}  {avg_loss:8.4f}  "
                  f"{metrics['route_accuracy']*100:6.1f}%  "
                  f"{metrics['op_accuracy']*100:4.1f}%  "
                  f"{metrics['arg1_accuracy']*100:4.1f}%  "
                  f"{metrics['arg2_accuracy']*100:4.1f}%  "
                  f"{metrics['result_accuracy']*100:4.1f}%  "
                  f"  {accepted:>3}  {dt:5.2f}s")

            # Check for full convergence
            if metrics["route_accuracy"] >= 0.99:
                print(f"\n  🎯 Routing converged at generation {gen}!")
                break
        else:
            # Quick fitness check on small batch for mutation acceptance
            quick_tokens, quick_ops, quick_a1, quick_a2, quick_res = generate_batch(
                rng, 64, max_val=max_val,
            )
            _, pred_op, pred_a1, pred_a2, pred_result = model(quick_tokens)
            mx.eval(pred_op, pred_a1, pred_a2, pred_result)

            # Quick accuracy
            quick_correct = (
                (np.array(pred_op) == np.array(quick_ops)) &
                (np.array(pred_a1) == np.array(quick_a1)) &
                (np.array(pred_a2) == np.array(quick_a2))
            ).mean()

            if quick_correct >= best_accuracy:
                champion_topology = save_topology(model)
                best_accuracy = max(best_accuracy, quick_correct)
                mutations_accepted += 1
            else:
                load_topology(model, champion_topology)
            mutations_total += 1

    # ── Final evaluation ──
    print(f"\n{'=' * 70}")
    t_total = time.time() - t_start
    print(f"  Training complete: {generations} generations, "
          f"{total_adam_steps} Adam steps, {t_total:.1f}s")
    print(f"  Mutations: {mutations_accepted}/{mutations_total} accepted "
          f"({mutations_accepted/max(1,mutations_total)*100:.0f}%)")

    # Comprehensive final eval
    final_rng = np.random.RandomState(seed + 9999)
    final = evaluate_accuracy(model, final_rng, n_samples=1024, max_val=max_val)
    print(f"\n  Final accuracy (1024 samples):")
    print(f"    Route:  {final['route_accuracy']*100:.1f}%")
    print(f"    Op:     {final['op_accuracy']*100:.1f}%")
    print(f"    Arg1:   {final['arg1_accuracy']*100:.1f}%")
    print(f"    Arg2:   {final['arg2_accuracy']*100:.1f}%")
    print(f"    Result: {final['result_accuracy']*100:.1f}%")

    print(f"\n  Per-op breakdown:")
    for op_name, stats in final["per_op"].items():
        print(f"    {op_name}: route={stats['route_acc']*100:.1f}% "
              f"result={stats['result_acc']*100:.1f}% "
              f"(n={stats['count']})")

    # ── Viability assessment ──
    print(f"\n{'=' * 70}")
    if final["route_accuracy"] > 0.5:
        print("  ✅ VIABLE: Ternary evolution found kernel routing.")
        print("     The representation boundary CAN be crossed.")
    elif final["route_accuracy"] > 0.05:
        print("  🔄 PARTIAL: Some routing learned. Needs investigation.")
        print("     Check which components work and which don't.")
    elif final["op_accuracy"] > 0.5:
        print("  💡 INSIGHT: Op routing works but arg routing doesn't.")
        print("     The discrete structure is learnable but value extraction is hard.")
    else:
        print("  ❌ NOT VIABLE (at this scale/config): Evolution didn't find routing.")
        print("     Possible fixes: larger model, different architecture,")
        print("     softer decode, or the concept doesn't work.")
    print(f"{'=' * 70}")

    return model, final


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v9 Kernel Router Training")
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--adam-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mutation-pct", type=float, default=0.02)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--max-val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--no-evolution", action="store_true",
                        help="Disable ternary mutation — Adam only")
    args = parser.parse_args()

    train(
        generations=args.generations,
        batch_size=args.batch_size,
        adam_steps_per_gen=args.adam_steps,
        lr=args.lr,
        mutation_pct=args.mutation_pct,
        eval_interval=args.eval_interval,
        eval_samples=args.eval_samples,
        max_val=args.max_val,
        seed=args.seed,
        d_model=args.d_model,
        no_evolution=args.no_evolution,
    )
