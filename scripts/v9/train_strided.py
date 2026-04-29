"""
v9 — Strided Kernel Router Training

Tests the strided architecture: self-similar shared weights applied
bottom-up through expression trees, routing each node to exact
kernel primitives.

Compares against the query-based router from train_kernel.py.
Supports mixed-depth expressions (flat through nested).

Usage:
    cd ~/src/verbum
    uv run python scripts/v9/train_strided.py
    uv run python scripts/v9/train_strided.py --max-depth 2 --max-val 10

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
    mutate_topology,
)

from kernel import N_OPS, OP_ADD, OP_SUB, OP_MUL, OP_NAMES
from strided_kernel import (
    StridedKernelRouter,
    StridedConfig,
    parse_sexpr,
    eval_tree,
    tree_depth,
    tree_to_str,
    linearize_bottomup,
    tokenize_expr,
    OP_STR_TO_CODE,
    OPS,
)


# ══════════════════════════════════════════════════════════════════════
# Data generation — mixed-depth arithmetic expressions
# ══════════════════════════════════════════════════════════════════════


def random_expr(rng: np.random.RandomState, max_val: int, max_depth: int, depth: int = 0):
    """Generate a random arithmetic expression tree.

    At each position that could be a number, there's a probability of
    recursing into a sub-expression (decreasing with depth).

    Args:
        rng:       random state
        max_val:   integer range [0, max_val)
        max_depth: maximum nesting depth
        depth:     current depth

    Returns:
        ExprNode or int
    """
    op = OPS[rng.randint(0, len(OPS))]

    def make_arg():
        # Probability of nesting decreases with depth
        if depth < max_depth - 1 and rng.random() < 0.4:
            return random_expr(rng, max_val, max_depth, depth + 1)
        return int(rng.randint(0, max_val))

    a1 = make_arg()
    a2 = make_arg()
    return (op, a1, a2)


def generate_batch_tokens(
    rng: np.random.RandomState,
    batch_size: int,
    max_val: int = 10,
    max_depth: int = 1,
    max_len: int = 24,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Generate a batch of tokenized expressions with ground truth.

    Returns:
        tokens:     (B, max_len) int — tokenized expressions
        gt_ops:     (B,) int         — op code of the ROOT expression
        gt_arg1:    (B,) int         — root arg1 value (evaluated)
        gt_arg2:    (B,) int         — root arg2 value (evaluated)
        gt_results: (B,) int         — full expression result
    """
    all_tokens = []
    all_ops = []
    all_a1 = []
    all_a2 = []
    all_res = []

    for _ in range(batch_size):
        tree = random_expr(rng, max_val, max_depth)
        expr_str = tree_to_str(tree)
        toks = tokenize_expr(expr_str, max_len=max_len)

        # Root-level ground truth
        op_str, a1, a2 = tree
        v1 = eval_tree(a1)
        v2 = eval_tree(a2)
        result = eval_tree(tree)

        all_tokens.append(toks)
        all_ops.append(OP_STR_TO_CODE[op_str])
        all_a1.append(v1)
        all_a2.append(v2)
        all_res.append(result)

    return (
        mx.array(all_tokens),
        mx.array(all_ops),
        mx.array(all_a1),
        mx.array(all_a2),
        mx.array(all_res),
    )


# ══════════════════════════════════════════════════════════════════════
# Loss — per-node routing supervision
# ══════════════════════════════════════════════════════════════════════


def routing_loss(
    model: StridedKernelRouter,
    tokens: mx.array,
    gt_ops: mx.array,
    gt_arg1: mx.array,
    gt_arg2: mx.array,
) -> mx.array:
    """Cross-entropy on routing logits for token-based strided model.

    Supervises the ROOT expression's routing: the model reads the full
    tokenized expression and must produce routing logits for (op, arg1, arg2)
    of the outermost operation.

    For nested expressions, arg1/arg2 may be the evaluated result of
    sub-expressions (e.g., for `(+ 3 (* 4 5))`, arg2 target is 20).
    """
    config = model.config
    route_logits = model.forward_routing(tokens)

    op_logits = route_logits[:, :config.n_ops]
    arg1_logits = route_logits[:, config.n_ops:config.n_ops + config.max_val]
    arg2_logits = route_logits[:, config.n_ops + config.max_val:]

    # Clamp targets to routing range
    gt_a1_clamped = mx.clip(gt_arg1, 0, config.max_val - 1).astype(mx.int32)
    gt_a2_clamped = mx.clip(gt_arg2, 0, config.max_val - 1).astype(mx.int32)

    loss_op = nn.losses.cross_entropy(op_logits, gt_ops, reduction="mean")
    loss_a1 = nn.losses.cross_entropy(arg1_logits, gt_a1_clamped, reduction="mean")
    loss_a2 = nn.losses.cross_entropy(arg2_logits, gt_a2_clamped, reduction="mean")

    return loss_op + loss_a1 + loss_a2


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════


def evaluate_accuracy(
    model: StridedKernelRouter,
    rng: np.random.RandomState,
    n_exprs: int = 256,
    max_val: int = 10,
    max_depth: int = 2,
    max_len: int = 24,
) -> dict:
    """Evaluate the token-based strided model on expression routing.

    For each expression, checks if the model routes the ROOT operation
    correctly: does it identify the op, arg1 (evaluated), and arg2 (evaluated)?

    Per-depth breakdown shows whether nesting degrades routing.
    """
    route_correct = 0
    result_correct = 0
    op_correct = 0
    arg1_correct = 0
    arg2_correct = 0
    total = 0
    depth_stats = {}

    for _ in range(n_exprs):
        tree = random_expr(rng, max_val, max_depth)
        expr_str = tree_to_str(tree)
        gt_result = eval_tree(tree)
        depth = tree_depth(tree)

        # Root-level ground truth
        op_str, a1, a2 = tree
        gt_op = OP_STR_TO_CODE[op_str]
        gt_a1 = eval_tree(a1)
        gt_a2 = eval_tree(a2)

        toks = mx.array([tokenize_expr(expr_str, max_len=max_len)])
        _, pred_op, pred_a1, pred_a2, pred_result = model(toks)
        mx.eval(pred_op, pred_a1, pred_a2, pred_result)

        po = pred_op[0].item()
        pa1 = pred_a1[0].item()
        pa2 = pred_a2[0].item()
        pr = pred_result[0].item()

        total += 1
        if po == gt_op:
            op_correct += 1
        if pa1 == gt_a1:
            arg1_correct += 1
        if pa2 == gt_a2:
            arg2_correct += 1
        if po == gt_op and pa1 == gt_a1 and pa2 == gt_a2:
            route_correct += 1
        if pr == gt_result:
            result_correct += 1

        if depth not in depth_stats:
            depth_stats[depth] = {"route": 0, "result": 0, "op": 0, "total": 0}
        depth_stats[depth]["total"] += 1
        if po == gt_op:
            depth_stats[depth]["op"] += 1
        if po == gt_op and pa1 == gt_a1 and pa2 == gt_a2:
            depth_stats[depth]["route"] += 1
        if pr == gt_result:
            depth_stats[depth]["result"] += 1

    return {
        "node_route_accuracy": route_correct / max(1, total),
        "tree_accuracy": result_correct / max(1, total),
        "op_accuracy": op_correct / max(1, total),
        "arg1_accuracy": arg1_correct / max(1, total),
        "arg2_accuracy": arg2_correct / max(1, total),
        "node_total": total,
        "tree_total": total,
        "depth_stats": {
            d: {
                "tree_acc": s["result"] / max(1, s["total"]),
                "node_acc": s["route"] / max(1, s["total"]),
                "op_acc": s["op"] / max(1, s["total"]),
                "n_trees": s["total"],
                "n_nodes": s["total"],
            }
            for d, s in sorted(depth_stats.items())
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════


def train(
    generations: int = 2000,
    batch_size: int = 64,
    adam_steps_per_gen: int = 10,
    lr: float = 1e-3,
    mutation_pct: float = 0.02,
    eval_interval: int = 50,
    eval_exprs: int = 512,
    seed: int = 42,
    max_val: int = 10,
    max_depth: int = 2,
    d_model: int = 64,
    no_evolution: bool = False,
):
    """Train the strided kernel router."""
    print("=" * 70)
    print("  v9 — Strided Kernel Router Training")
    print("=" * 70)

    rng = np.random.RandomState(seed)
    config = StridedConfig(d_model=d_model, max_val=max_val, max_len=24)
    model = StridedKernelRouter(config)

    n_ternary = count_ternary_weights(model)
    mutation_budget = max(1, int(n_ternary * mutation_pct))

    print(f"\nConfig:")
    print(f"  d_model:          {config.d_model}")
    print(f"  n_mix_layers:     {config.n_mix_layers}")
    print(f"  max_val:          {max_val}")
    print(f"  max_depth:        {max_depth}")
    print(f"  route_dim:        {config.n_ops + 2 * config.max_val}")
    print(f"  ternary weights:  {n_ternary:,}")
    print(f"  mutation budget:  {mutation_budget:,} ({mutation_pct*100:.1f}%)")
    print(f"  generations:      {generations}")
    print(f"  adam steps/gen:   {adam_steps_per_gen}")
    print(f"  batch size:       {batch_size} (expressions, nodes vary)")
    print(f"  learning rate:    {lr}")
    print(f"  evolution:        {'OFF' if no_evolution else 'ON'}")

    params = model.count_params()
    print(f"\n  Parameters: {params['total']:,} total, "
          f"{params['ternary_logical']:,} ternary, "
          f"{params['continuous']:,} continuous")

    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, routing_loss)

    best_accuracy = -1.0
    champion_topology = save_topology(model)

    # Initial eval
    eval_rng = np.random.RandomState(seed + 1000)
    metrics = evaluate_accuracy(model, eval_rng, n_exprs=eval_exprs,
                                max_val=max_val, max_depth=max_depth,
                                max_len=config.max_len)
    best_accuracy = metrics["node_route_accuracy"]
    print(f"\nInitial: node_route={metrics['node_route_accuracy']*100:.1f}%  "
          f"tree={metrics['tree_accuracy']*100:.1f}%  "
          f"op={metrics['op_accuracy']*100:.1f}%  "
          f"a1={metrics['arg1_accuracy']*100:.1f}%  "
          f"a2={metrics['arg2_accuracy']*100:.1f}%")

    print(f"\n{'Gen':>6}  {'Loss':>8}  {'Node%':>6}  {'Tree%':>6}  "
          f"{'Op%':>5}  {'A1%':>5}  {'A2%':>5}  {'Mut':>3}  {'dt':>5}")
    print("-" * 65)

    t_start = time.time()
    total_adam = 0
    mut_accepted = 0
    mut_total = 0

    for gen in range(generations):
        gen_start = time.time()

        # ── Adam steps ──
        avg_loss = 0.0
        for _ in range(adam_steps_per_gen):
            tokens, gt_ops, gt_a1, gt_a2, gt_res = generate_batch_tokens(
                rng, batch_size, max_val=max_val, max_depth=max_depth,
                max_len=config.max_len,
            )
            loss, grads = loss_and_grad(model, tokens, gt_ops, gt_a1, gt_a2)
            grads = zero_ternary_grads(model, grads)
            optimizer.update(model, grads)
            restore_ternary(model)
            mx.eval(model.parameters(), optimizer.state)
            avg_loss += loss.item()
            total_adam += 1
        avg_loss /= adam_steps_per_gen

        # ── Evolution ──
        if not no_evolution:
            mutate_topology(model, mutation_budget, rng, sign_flip_rate=0.2)
            mx.eval(model.parameters())

        # ── Evaluate ──
        if gen % eval_interval == 0 or gen == generations - 1:
            eval_rng_local = np.random.RandomState(seed + gen + 2000)
            metrics = evaluate_accuracy(
                model, eval_rng_local, n_exprs=eval_exprs,
                max_val=max_val, max_depth=max_depth,
                max_len=config.max_len,
            )
            current = metrics["node_route_accuracy"]

            if not no_evolution:
                if current >= best_accuracy:
                    best_accuracy = current
                    champion_topology = save_topology(model)
                    mut_accepted += 1
                    status = "✓"
                else:
                    load_topology(model, champion_topology)
                    status = "✗"
                mut_total += 1
            else:
                status = "—"

            dt = time.time() - gen_start
            print(f"  {gen:5d}  {avg_loss:8.4f}  "
                  f"{metrics['node_route_accuracy']*100:5.1f}%  "
                  f"{metrics['tree_accuracy']*100:5.1f}%  "
                  f"{metrics['op_accuracy']*100:4.1f}%  "
                  f"{metrics['arg1_accuracy']*100:4.1f}%  "
                  f"{metrics['arg2_accuracy']*100:4.1f}%  "
                  f"  {status}  {dt:4.1f}s")

            if metrics["node_route_accuracy"] >= 0.99:
                print(f"\n  🎯 Converged at generation {gen}!")
                break
        else:
            # Quick check for evolution
            if not no_evolution:
                q_tok, q_ops, q_a1, q_a2, q_res = generate_batch_tokens(
                    rng, 32, max_val=max_val, max_depth=max_depth,
                    max_len=config.max_len,
                )
                _, p_op, p_a1, p_a2, p_res = model(q_tok)
                mx.eval(p_op, p_a1, p_a2, p_res)
                quick = (
                    (np.array(p_op) == np.array(q_ops)) &
                    (np.array(p_a1) == np.array(q_a1)) &
                    (np.array(p_a2) == np.array(q_a2))
                ).mean()
                if quick >= best_accuracy:
                    champion_topology = save_topology(model)
                    best_accuracy = max(best_accuracy, quick)
                    mut_accepted += 1
                else:
                    load_topology(model, champion_topology)
                mut_total += 1

    # ── Final report ──
    t_total = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"  Done: {generations} gens, {total_adam} Adam steps, {t_total:.1f}s")
    if mut_total > 0:
        print(f"  Mutations: {mut_accepted}/{mut_total} accepted "
              f"({mut_accepted/max(1,mut_total)*100:.0f}%)")

    final_rng = np.random.RandomState(seed + 9999)
    final = evaluate_accuracy(model, final_rng, n_exprs=1024,
                              max_val=max_val, max_depth=max_depth,
                              max_len=config.max_len)

    print(f"\n  Final (1024 trees):")
    print(f"    Node route: {final['node_route_accuracy']*100:.1f}%")
    print(f"    Tree exact: {final['tree_accuracy']*100:.1f}%")
    print(f"    Op:         {final['op_accuracy']*100:.1f}%")
    print(f"    Arg1:       {final['arg1_accuracy']*100:.1f}%")
    print(f"    Arg2:       {final['arg2_accuracy']*100:.1f}%")

    print(f"\n  Per-depth breakdown:")
    for d, stats in final["depth_stats"].items():
        print(f"    depth {d}: tree_acc={stats['tree_acc']*100:.1f}%  "
              f"node_acc={stats['node_acc']*100:.1f}%  "
              f"(trees={stats['n_trees']}, nodes={stats['n_nodes']})")

    # Viability
    print(f"\n{'=' * 65}")
    if final["node_route_accuracy"] > 0.5:
        print("  ✅ VIABLE: Strided self-similar routing works.")
    elif final["node_route_accuracy"] > 0.1:
        print("  🔄 PARTIAL: Some routing learned. Check per-depth.")
    elif final["op_accuracy"] > 0.5:
        print("  💡 Op routing works, arg routing doesn't.")
    else:
        print("  ❌ Not viable at this scale.")
    print(f"{'=' * 65}")

    return model, final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v9 Strided Kernel Training")
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--adam-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mutation-pct", type=float, default=0.02)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-exprs", type=int, default=512)
    parser.add_argument("--max-val", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-evolution", action="store_true")
    args = parser.parse_args()

    train(
        generations=args.generations,
        batch_size=args.batch_size,
        adam_steps_per_gen=args.adam_steps,
        lr=args.lr,
        mutation_pct=args.mutation_pct,
        eval_interval=args.eval_interval,
        eval_exprs=args.eval_exprs,
        max_val=args.max_val,
        max_depth=args.max_depth,
        d_model=args.d_model,
        seed=args.seed,
        no_evolution=args.no_evolution,
    )
