"""
v9 — Integrated Training: Ascending Arm + Type/Parse/Apply

Tests the full pipeline: reduce → type → parse → apply
on S-expressions with mixed nesting depth.

Loss components:
  1. Type CE: expression type + arg types
  2. Parse CE: op + arg1 value + arg2 value
  3. Apply is exact (kernel) — no loss, just accuracy metric

Usage:
    cd ~/src/verbum
    uv run python scripts/v9/train_v9.py
    uv run python scripts/v9/train_v9.py --generations 3000 --max-depth 2

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

sys.path.insert(0, str(Path(__file__).parent.parent / "v8"))
from ternary import (
    save_topology, load_topology, zero_ternary_grads,
    restore_ternary, count_ternary_weights, mutate_topology,
)

from v9_model import (
    V9Model, V9Config, tokenize_expr,
    TYPE_INT, TYPE_OP, TYPE_EXPR, TYPE_ERROR, N_TYPES, TYPE_NAMES,
    OP_ADD, OP_SUB, OP_MUL, N_OPS, OP_NAMES,
)

# Reuse expression generation from strided_kernel
from strided_kernel import (
    parse_sexpr, eval_tree, tree_depth, tree_to_str,
    linearize_bottomup, OPS, OP_STR_TO_CODE,
)


# ══════════════════════════════════════════════════════════════════════
# Data generation with type labels
# ══════════════════════════════════════════════════════════════════════


def random_expr(rng, max_val, max_depth, depth=0):
    op = OPS[rng.randint(0, len(OPS))]
    def make_arg():
        if depth < max_depth - 1 and rng.random() < 0.4:
            return random_expr(rng, max_val, max_depth, depth + 1)
        return int(rng.randint(0, max_val))
    return (op, make_arg(), make_arg())


def generate_batch(
    rng: np.random.RandomState,
    batch_size: int,
    max_val: int = 10,
    max_depth: int = 2,
    max_len: int = 24,
) -> dict[str, mx.array]:
    """Generate batch with full ground truth for type/parse/apply.

    Ground truth for the ROOT expression:
      - expr_type: TYPE_EXPR (always, since it's an expression)
      - op: the root operator code
      - arg1: evaluated value of first argument
      - arg2: evaluated value of second argument
      - arg1_type: TYPE_INT (since evaluated args are always ints)
      - arg2_type: TYPE_INT
      - result: fully evaluated result
    """
    tokens_list = []
    gt_expr_type = []
    gt_ops = []
    gt_a1 = []
    gt_a2 = []
    gt_a1_type = []
    gt_a2_type = []
    gt_results = []

    for _ in range(batch_size):
        tree = random_expr(rng, max_val, max_depth)
        expr_str = tree_to_str(tree)
        toks = tokenize_expr(expr_str, max_len=max_len)

        op_str, a1, a2 = tree
        v1 = eval_tree(a1)
        v2 = eval_tree(a2)
        result = eval_tree(tree)

        tokens_list.append(toks)
        gt_expr_type.append(TYPE_EXPR)
        gt_ops.append(OP_STR_TO_CODE[op_str])
        gt_a1.append(v1)
        gt_a2.append(v2)
        gt_a1_type.append(TYPE_INT)
        gt_a2_type.append(TYPE_INT)
        gt_results.append(result)

    return {
        "tokens": mx.array(tokens_list),
        "gt_expr_type": mx.array(gt_expr_type),
        "gt_ops": mx.array(gt_ops),
        "gt_arg1": mx.array(gt_a1),
        "gt_arg2": mx.array(gt_a2),
        "gt_arg1_type": mx.array(gt_a1_type),
        "gt_arg2_type": mx.array(gt_a2_type),
        "gt_results": mx.array(gt_results),
    }


# ══════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════


def v9_loss(
    model: V9Model,
    tokens: mx.array,
    gt_expr_type: mx.array,
    gt_ops: mx.array,
    gt_arg1: mx.array,
    gt_arg2: mx.array,
    gt_arg1_type: mx.array,
    gt_arg2_type: mx.array,
) -> mx.array:
    """Combined type + parse loss.

    Components:
      1. Expression type classification
      2. Arg1/Arg2 type classification
      3. Op routing
      4. Arg1/Arg2 value routing
    """
    config = model.config
    out = model(tokens)

    # Type losses
    loss_expr_type = nn.losses.cross_entropy(
        out["expr_type_logits"], gt_expr_type, reduction="mean")
    loss_a1_type = nn.losses.cross_entropy(
        out["arg1_type_logits"], gt_arg1_type, reduction="mean")
    loss_a2_type = nn.losses.cross_entropy(
        out["arg2_type_logits"], gt_arg2_type, reduction="mean")

    # Parse losses
    loss_op = nn.losses.cross_entropy(
        out["op_logits"], gt_ops, reduction="mean")

    gt_a1_clamped = mx.clip(gt_arg1, 0, config.max_val - 1).astype(mx.int32)
    gt_a2_clamped = mx.clip(gt_arg2, 0, config.max_val - 1).astype(mx.int32)
    loss_a1 = nn.losses.cross_entropy(
        out["arg1_logits"], gt_a1_clamped, reduction="mean")
    loss_a2 = nn.losses.cross_entropy(
        out["arg2_logits"], gt_a2_clamped, reduction="mean")

    # Combined: type weight 0.5, parse weight 1.0
    type_loss = 0.5 * (loss_expr_type + loss_a1_type + loss_a2_type)
    parse_loss = loss_op + loss_a1 + loss_a2

    return type_loss + parse_loss


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════


def evaluate(
    model: V9Model,
    rng: np.random.RandomState,
    n_exprs: int = 512,
    max_val: int = 10,
    max_depth: int = 2,
    max_len: int = 24,
) -> dict:
    """Evaluate type, parse, apply accuracy."""
    batch = generate_batch(rng, n_exprs, max_val, max_depth, max_len)
    out = model(batch["tokens"])
    for v in out.values():
        mx.eval(v)

    # Convert
    pred_type = np.array(out["pred_type"])
    pred_op = np.array(out["pred_op"])
    pred_a1 = np.array(out["pred_arg1"])
    pred_a2 = np.array(out["pred_arg2"])
    pred_a1t = np.array(out["pred_a1_type"])
    pred_a2t = np.array(out["pred_a2_type"])
    pred_res = np.array(out["pred_result"])
    pred_res_t = np.array(out["pred_result_type"])

    gt_type = np.array(batch["gt_expr_type"])
    gt_op = np.array(batch["gt_ops"])
    gt_a1 = np.array(batch["gt_arg1"])
    gt_a2 = np.array(batch["gt_arg2"])
    gt_a1t = np.array(batch["gt_arg1_type"])
    gt_a2t = np.array(batch["gt_arg2_type"])
    gt_res = np.array(batch["gt_results"])

    # Accuracies
    type_acc = (pred_type == gt_type).mean()
    op_acc = (pred_op == gt_op).mean()
    a1_acc = (pred_a1 == gt_a1).mean()
    a2_acc = (pred_a2 == gt_a2).mean()
    a1t_acc = (pred_a1t == gt_a1t).mean()
    a2t_acc = (pred_a2t == gt_a2t).mean()

    # Route = op AND arg1 AND arg2 all correct
    route_acc = ((pred_op == gt_op) & (pred_a1 == gt_a1) & (pred_a2 == gt_a2)).mean()

    # Apply = route AND types correct → kernel result correct
    result_acc = (pred_res == gt_res).mean()

    # Type gate: how often does the type system allow dispatch?
    dispatch_rate = (pred_res_t != TYPE_ERROR).mean()

    # Per-depth breakdown
    depth_stats = {}
    for i in range(n_exprs):
        tree = random_expr(np.random.RandomState(rng.randint(0, 2**31)),
                           max_val, max_depth)
        # We can't reconstruct depth from batch — approximate from expression
    # Skip per-depth for now, use aggregate

    return {
        "type_acc": float(type_acc),
        "op_acc": float(op_acc),
        "arg1_acc": float(a1_acc),
        "arg2_acc": float(a2_acc),
        "arg1_type_acc": float(a1t_acc),
        "arg2_type_acc": float(a2t_acc),
        "route_acc": float(route_acc),
        "result_acc": float(result_acc),
        "dispatch_rate": float(dispatch_rate),
    }


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════


def train(
    generations: int = 2000,
    batch_size: int = 128,
    adam_steps: int = 10,
    lr: float = 1e-3,
    mutation_pct: float = 0.02,
    eval_interval: int = 100,
    eval_exprs: int = 512,
    seed: int = 42,
    max_val: int = 10,
    max_depth: int = 2,
    d_model: int = 64,
):
    print("=" * 75)
    print("  v9 — Integrated Training: Ascending + Type/Parse/Apply")
    print("=" * 75)

    rng = np.random.RandomState(seed)
    config = V9Config(d_model=d_model, max_val=max_val, max_len=24)
    model = V9Model(config)

    n_ternary = count_ternary_weights(model)
    mut_budget = max(1, int(n_ternary * mutation_pct))

    params = model.count_params()
    print(f"\nConfig:")
    print(f"  d_model:       {d_model}")
    print(f"  ascending:     {config.n_ascending_levels} levels, stride={config.stride}")
    print(f"  max_val:       {max_val}")
    print(f"  max_depth:     {max_depth}")
    print(f"  ternary:       {n_ternary:,} weights")
    print(f"  continuous:    {params['continuous']:,}")
    print(f"  mut budget:    {mut_budget:,} ({mutation_pct*100:.1f}%)")
    print(f"  generations:   {generations}")
    print(f"  adam steps:    {adam_steps}")
    print(f"  batch size:    {batch_size}")
    print(f"  lr:            {lr}")

    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, v9_loss)

    best_route = -1.0
    champion = save_topology(model)

    # Header
    print(f"\n{'Gen':>5}  {'Loss':>7}  {'Type':>5}  {'Op':>4}  "
          f"{'A1':>4}  {'A2':>4}  {'A1t':>4}  {'A2t':>4}  "
          f"{'Route':>5}  {'Res':>4}  {'Disp':>4}  {'M':>2}  {'dt':>4}")
    print("-" * 75)

    t0 = time.time()
    total_adam = 0
    mut_acc = 0
    mut_tot = 0

    for gen in range(generations):
        g0 = time.time()

        # Adam
        avg_loss = 0.0
        for _ in range(adam_steps):
            batch = generate_batch(rng, batch_size, max_val, max_depth, 24)
            loss, grads = loss_and_grad(
                model, batch["tokens"], batch["gt_expr_type"],
                batch["gt_ops"], batch["gt_arg1"], batch["gt_arg2"],
                batch["gt_arg1_type"], batch["gt_arg2_type"],
            )
            grads = zero_ternary_grads(model, grads)
            optimizer.update(model, grads)
            restore_ternary(model)
            mx.eval(model.parameters(), optimizer.state)
            avg_loss += loss.item()
            total_adam += 1
        avg_loss /= adam_steps

        # Evolution
        mutate_topology(model, mut_budget, rng, sign_flip_rate=0.2)
        mx.eval(model.parameters())

        # Eval
        if gen % eval_interval == 0 or gen == generations - 1:
            erng = np.random.RandomState(seed + gen + 5000)
            m = evaluate(model, erng, eval_exprs, max_val, max_depth, 24)

            if m["route_acc"] >= best_route:
                best_route = m["route_acc"]
                champion = save_topology(model)
                mut_acc += 1
                status = "✓"
            else:
                load_topology(model, champion)
                status = "✗"
            mut_tot += 1

            dt = time.time() - g0
            print(f"  {gen:4d}  {avg_loss:7.3f}  "
                  f"{m['type_acc']*100:4.0f}%  "
                  f"{m['op_acc']*100:3.0f}%  "
                  f"{m['arg1_acc']*100:3.0f}%  "
                  f"{m['arg2_acc']*100:3.0f}%  "
                  f"{m['arg1_type_acc']*100:3.0f}%  "
                  f"{m['arg2_type_acc']*100:3.0f}%  "
                  f"{m['route_acc']*100:4.0f}%  "
                  f"{m['result_acc']*100:3.0f}%  "
                  f"{m['dispatch_rate']*100:3.0f}%  "
                  f" {status}  {dt:3.1f}")

            if m["route_acc"] >= 0.95:
                print(f"\n  🎯 Converged at gen {gen}!")
                break
        else:
            # Quick tournament
            qb = generate_batch(rng, 32, max_val, max_depth, 24)
            qo = model(qb["tokens"])
            mx.eval(qo["pred_op"], qo["pred_arg1"], qo["pred_arg2"])
            qa = (
                (np.array(qo["pred_op"]) == np.array(qb["gt_ops"])) &
                (np.array(qo["pred_arg1"]) == np.array(qb["gt_arg1"])) &
                (np.array(qo["pred_arg2"]) == np.array(qb["gt_arg2"]))
            ).mean()
            if qa >= best_route:
                champion = save_topology(model)
                best_route = max(best_route, qa)
                mut_acc += 1
            else:
                load_topology(model, champion)
            mut_tot += 1

    # Final
    t_total = time.time() - t0
    print(f"\n{'=' * 75}")
    print(f"  Done: {generations} gens, {total_adam} Adam steps, {t_total:.1f}s")
    if mut_tot > 0:
        print(f"  Mutations: {mut_acc}/{mut_tot} ({mut_acc/mut_tot*100:.0f}%)")

    frng = np.random.RandomState(seed + 99999)
    f = evaluate(model, frng, 1024, max_val, max_depth, 24)
    print(f"\n  Final (1024 expressions):")
    print(f"    Type:     {f['type_acc']*100:.1f}%")
    print(f"    Op:       {f['op_acc']*100:.1f}%")
    print(f"    Arg1:     {f['arg1_acc']*100:.1f}%  (type: {f['arg1_type_acc']*100:.1f}%)")
    print(f"    Arg2:     {f['arg2_acc']*100:.1f}%  (type: {f['arg2_type_acc']*100:.1f}%)")
    print(f"    Route:    {f['route_acc']*100:.1f}%")
    print(f"    Result:   {f['result_acc']*100:.1f}%")
    print(f"    Dispatch: {f['dispatch_rate']*100:.1f}%")

    print(f"\n{'=' * 75}")
    if f["route_acc"] > 0.5:
        print("  ✅ VIABLE: Ascending arm + type/parse/apply works.")
    elif f["route_acc"] > 0.1:
        print("  🔄 PARTIAL: Learning, needs more capacity or training.")
    elif f["op_acc"] > 0.5:
        print("  💡 Op routing works but value extraction needs work.")
    else:
        print("  ❌ Not converging at this scale.")
    print(f"{'=' * 75}")

    return model, f


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--adam-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mutation-pct", type=float, default=0.02)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--max-val", type=int, default=10)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    train(
        generations=a.generations, batch_size=a.batch_size,
        adam_steps=a.adam_steps, lr=a.lr, mutation_pct=a.mutation_pct,
        eval_interval=a.eval_interval, max_val=a.max_val,
        max_depth=a.max_depth, d_model=a.d_model, seed=a.seed,
    )
