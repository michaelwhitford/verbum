"""
v9 — VSM Tree v3: Pass-through values, op-only routing

Key insight from v2 experiments: the VSM node's arg classification heads
were solving the wrong problem. With 100% accuracy on in-range leaf
values and 0% on out-of-range sub-expression results, the bottleneck
was representation, not learning.

The tree structure already routes values — each node receives its
children's computed values. The VSM node only needs to:
  1. Classify the operation (3 classes — trivially learnable, 100%)
  2. Pass child values to the kernel directly
  3. Return the exact result to the parent

This gives 100% result accuracy at any depth, with any value range.

The remaining challenge: for PROSE (not S-expressions), the tree
structure isn't given — it must be discovered. That's the ascending
arm's job (future work). But for S-expressions, this architecture
is complete and exact.

This file implements the full VSM tree with pass-through values and
tests it on deeper expressions and larger value ranges.

License: MIT
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "v8"))
from ternary import (
    TernaryLinear,
    save_topology,
    load_topology,
    zero_ternary_grads,
    restore_ternary,
    count_ternary_weights,
    mutate_topology,
)

# ══════════════════════════════════════════════════════════════════════
# Types and ops
# ══════════════════════════════════════════════════════════════════════

TYPE_INT = 0
N_TYPES = 4
OP_ADD, OP_SUB, OP_MUL = 0, 1, 2
N_OPS = 3
OPS = ["+", "-", "*"]
OP_STR_TO_CODE = {"+": 0, "-": 1, "*": 2}


def kernel_apply(op, a1, a2):
    return mx.where(op == 0, a1 + a2, mx.where(op == 1, a1 - a2, a1 * a2))


# ══════════════════════════════════════════════════════════════════════
# Expression tree utilities
# ══════════════════════════════════════════════════════════════════════

def random_expr(rng, max_val, max_depth, depth=0):
    op = OPS[rng.randint(0, len(OPS))]
    def arg():
        if depth < max_depth - 1 and rng.random() < 0.4:
            return random_expr(rng, max_val, max_depth, depth + 1)
        return int(rng.randint(0, max_val))
    return (op, arg(), arg())


def eval_tree(node):
    if isinstance(node, int):
        return node
    op, a1, a2 = node
    v1, v2 = eval_tree(a1), eval_tree(a2)
    return v1 + v2 if op == "+" else v1 - v2 if op == "-" else v1 * v2


def tree_to_str(node):
    if isinstance(node, int):
        return str(node)
    op, a1, a2 = node
    return f"({op} {tree_to_str(a1)} {tree_to_str(a2)})"


# ══════════════════════════════════════════════════════════════════════
# VSM Node v3 — op-only routing + value pass-through
# ══════════════════════════════════════════════════════════════════════

@dataclass
class VSMConfig:
    d_model: int = 32          # Smaller! Only need to classify 3 ops
    n_ops: int = N_OPS
    n_types: int = N_TYPES
    val_embed_range: int = 200  # Still need value embedding for context
    n_mix_layers: int = 2


class VSMNodeV3(nn.Module):
    """VSM node: classifies operation, passes values through.

    The node receives:
      - op_code: the operator at this tree position
      - c1_type, c1_val: child 1 type and value
      - c2_type, c2_val: child 2 type and value

    It outputs:
      - pred_op: classified operation (for kernel dispatch)
      - result: kernel_apply(pred_op, c1_val, c2_val) — exact computation

    The value embedding provides contextual information (the op
    classification might depend on the magnitude/type of children),
    but values pass through to the kernel without classification.
    """

    def __init__(self, config: VSMConfig | None = None):
        super().__init__()
        if config is None:
            config = VSMConfig()
        self.config = config
        d = config.d_model

        # Embeddings
        self.op_embed = nn.Embedding(config.n_ops, d)
        self.type_embed = nn.Embedding(config.n_types, d)
        self.val_embed = nn.Embedding(config.val_embed_range, d)
        self._val_offset = config.val_embed_range // 2

        # Input projection: concat [op; c1; c2] → d
        self.input_proj = nn.Linear(3 * d, d)

        # Mix layers
        self.mix_layers = [TernaryLinear(d, d, pre_norm=True)
                           for _ in range(config.n_mix_layers)]

        # Type head
        type_dim = ((config.n_types + 15) // 16) * 16
        self.type_proj = TernaryLinear(d, type_dim, pre_norm=True)
        self._type_dim = config.n_types

        # Op head — the ONLY routing decision
        op_dim = ((config.n_ops + 15) // 16) * 16
        self.op_proj = TernaryLinear(d, op_dim, pre_norm=True)
        self._op_dim = config.n_ops

    def _val_idx(self, val):
        return mx.clip(val + self._val_offset, 0, self.config.val_embed_range - 1).astype(mx.int32)

    def forward(self, op_codes, c1_types, c1_vals, c2_types, c2_vals):
        d = self.config.d_model

        # Embed
        op_repr = self.op_embed(op_codes)
        c1_repr = self.type_embed(c1_types) + self.val_embed(self._val_idx(c1_vals))
        c2_repr = self.type_embed(c2_types) + self.val_embed(self._val_idx(c2_vals))

        # Fuse
        x = self.input_proj(mx.concatenate([op_repr, c1_repr, c2_repr], axis=-1))

        # Mix
        for mix in self.mix_layers:
            x = x + mix(x)

        # Type
        type_logits = self.type_proj(x)[:, :self._type_dim]
        pred_type = mx.argmax(type_logits, axis=-1).astype(mx.int32)

        # Op — the only routing decision
        op_logits = self.op_proj(x)[:, :self._op_dim]
        pred_op = mx.argmax(op_logits, axis=-1).astype(mx.int32)

        # Kernel dispatch with ACTUAL child values (pass-through)
        pred_result = kernel_apply(pred_op, c1_vals, c2_vals)
        pred_result = mx.where(pred_type == TYPE_INT, pred_result, mx.zeros_like(pred_result))

        return {
            "type_logits": type_logits,
            "op_logits": op_logits,
            "pred_type": pred_type,
            "pred_op": pred_op,
            "pred_result": pred_result,
        }


# ══════════════════════════════════════════════════════════════════════
# Loss — op-only (much simpler)
# ══════════════════════════════════════════════════════════════════════

def vsm_loss(model, op_codes, c1_types, c1_vals, c2_types, c2_vals,
             gt_ops, gt_results):
    out = model.forward(op_codes, c1_types, c1_vals, c2_types, c2_vals)

    gt_type = mx.full(op_codes.shape, TYPE_INT, dtype=mx.int32)
    loss_type = nn.losses.cross_entropy(out["type_logits"], gt_type, reduction="mean")
    loss_op = nn.losses.cross_entropy(out["op_logits"], gt_ops, reduction="mean")

    return 0.5 * loss_type + loss_op


# ══════════════════════════════════════════════════════════════════════
# Data generation — tree execution with pass-through
# ══════════════════════════════════════════════════════════════════════

def _collect_nodes_v3(node, out):
    """Collect nodes bottom-up with actual computed child values."""
    if isinstance(node, int):
        return
    op, a1, a2 = node
    # Recurse first (bottom-up)
    _collect_nodes_v3(a1, out)
    _collect_nodes_v3(a2, out)

    op_code = OP_STR_TO_CODE[op]
    c1_val = eval_tree(a1)  # actual computed value (may be out of range)
    c2_val = eval_tree(a2)
    gt_result = eval_tree(node)
    out.append((op_code, TYPE_INT, c1_val, TYPE_INT, c2_val, gt_result))


def generate_node_batch(rng, batch_size, max_val, max_depth):
    all_ops, all_c1t, all_c1v, all_c2t, all_c2v = [], [], [], [], []
    all_gt_ops, all_gt_res = [], []

    for _ in range(batch_size):
        tree = random_expr(rng, max_val, max_depth)
        nodes = []
        _collect_nodes_v3(tree, nodes)
        for op_code, c1_type, c1_val, c2_type, c2_val, gt_result in nodes:
            all_ops.append(op_code)
            all_c1t.append(c1_type)
            all_c1v.append(c1_val)
            all_c2t.append(c2_type)
            all_c2v.append(c2_val)
            all_gt_ops.append(op_code)
            all_gt_res.append(gt_result)

    return {
        "op_codes": mx.array(all_ops, dtype=mx.int32),
        "c1_types": mx.array(all_c1t, dtype=mx.int32),
        "c1_vals": mx.array(all_c1v, dtype=mx.int64),
        "c2_types": mx.array(all_c2t, dtype=mx.int32),
        "c2_vals": mx.array(all_c2v, dtype=mx.int64),
        "gt_ops": mx.array(all_gt_ops, dtype=mx.int32),
        "gt_res": mx.array(all_gt_res, dtype=mx.int64),
    }


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate(model, rng, n_exprs, max_val, max_depth):
    batch = generate_node_batch(rng, n_exprs, max_val, max_depth)
    out = model.forward(batch["op_codes"], batch["c1_types"], batch["c1_vals"],
                        batch["c2_types"], batch["c2_vals"])
    for v in out.values():
        mx.eval(v)

    po = np.array(out["pred_op"])
    pt = np.array(out["pred_type"])
    pr = np.array(out["pred_result"])
    go = np.array(batch["gt_ops"])
    gr = np.array(batch["gt_res"])

    return {
        "type_acc": float((pt == TYPE_INT).mean()),
        "op_acc": float((po == go).mean()),
        "result_acc": float((pr == gr).mean()),
        "n_nodes": len(go),
    }


def evaluate_trees(model, rng, n_trees, max_val, max_depth):
    """Evaluate on whole trees: execute bottom-up, check final result."""
    correct = 0
    total = 0
    node_results = {"correct": 0, "total": 0}

    for _ in range(n_trees):
        tree = random_expr(rng, max_val, max_depth)
        gt = eval_tree(tree)

        # Execute tree bottom-up through the model
        pred = _execute_tree(model, tree, node_results)
        if pred == gt:
            correct += 1
        total += 1

    return {
        "tree_acc": correct / total if total > 0 else 0.0,
        "node_op_acc": node_results["correct"] / node_results["total"] if node_results["total"] > 0 else 0.0,
        "n_trees": total,
        "n_nodes": node_results["total"],
    }


def _execute_tree(model, node, stats):
    """Execute a tree bottom-up through the VSM model."""
    if isinstance(node, int):
        return node

    op, a1, a2 = node
    # Recurse first
    v1 = _execute_tree(model, a1, stats)
    v2 = _execute_tree(model, a2, stats)

    # Run this node through the model
    op_code = mx.array([OP_STR_TO_CODE[op]])
    c1_type = mx.array([TYPE_INT])
    c1_val = mx.array([v1])
    c2_type = mx.array([TYPE_INT])
    c2_val = mx.array([v2])

    out = model.forward(op_code, c1_type, c1_val, c2_type, c2_val)
    mx.eval(out["pred_result"])

    result = int(out["pred_result"].item())

    stats["total"] += 1
    if out["pred_op"].item() == OP_STR_TO_CODE[op]:
        stats["correct"] += 1

    return result


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════

def train(
    generations=2000,
    batch_size=128,
    adam_steps=10,
    lr=1e-3,
    mutation_pct=0.02,
    eval_interval=200,
    max_val=10,
    max_depth=2,
    d_model=32,
    n_mix=2,
    seed=42,
):
    print("=" * 70)
    print("  v9 — VSM Tree v3: Op-Only Routing + Value Pass-Through")
    print("=" * 70)

    rng = np.random.RandomState(seed)
    config = VSMConfig(d_model=d_model, n_mix_layers=n_mix)
    model = VSMNodeV3(config)

    n_ternary = count_ternary_weights(model)
    mut_budget = max(1, int(n_ternary * mutation_pct))

    print(f"\n  d={d_model}  mix={n_mix}  max_val={max_val}  max_depth={max_depth}")
    print(f"  ternary={n_ternary:,}  mut_budget={mut_budget}")
    print(f"  gens={generations}  adam_steps={adam_steps}  lr={lr}")

    optimizer = optim.Adam(learning_rate=lr)
    loss_fn = nn.value_and_grad(model, vsm_loss)

    best_route = -1.0
    champion = save_topology(model)

    print(f"\n{'Gen':>5}  {'Loss':>7}  {'Typ':>4}  {'Op':>4}  {'Res':>4}  "
          f"{'Tree':>5}  {'N':>4}  {'M':>2}  {'dt':>4}")
    print("-" * 55)

    t0 = time.time()

    for gen in range(generations):
        g0 = time.time()

        avg_loss = 0.0
        for _ in range(adam_steps):
            b = generate_node_batch(rng, batch_size, max_val, max_depth)
            loss, grads = loss_fn(model, b["op_codes"], b["c1_types"], b["c1_vals"],
                                  b["c2_types"], b["c2_vals"], b["gt_ops"], b["gt_res"])
            grads = zero_ternary_grads(model, grads)
            optimizer.update(model, grads)
            restore_ternary(model)
            mx.eval(model.parameters(), optimizer.state)
            avg_loss += loss.item()
        avg_loss /= adam_steps

        # Evolve
        mutate_topology(model, mut_budget, rng, sign_flip_rate=0.2)
        mx.eval(model.parameters())

        if gen % eval_interval == 0 or gen == generations - 1:
            erng = np.random.RandomState(seed + gen + 5000)
            m = evaluate(model, erng, 512, max_val, max_depth)

            # Tree-level evaluation (smaller sample — slower due to serial execution)
            trng = np.random.RandomState(seed + gen + 9000)
            tm = evaluate_trees(model, trng, 200, max_val, max_depth)

            if m["op_acc"] >= best_route:
                best_route = m["op_acc"]
                champion = save_topology(model)
                status = "✓"
            else:
                load_topology(model, champion)
                status = "✗"

            dt = time.time() - g0
            print(f"  {gen:4d}  {avg_loss:7.4f}  {m['type_acc']*100:3.0f}%  "
                  f"{m['op_acc']*100:3.0f}%  {m['result_acc']*100:3.0f}%  "
                  f"{tm['tree_acc']*100:4.1f}%  {m['n_nodes']:3d}   {status}  {dt:3.1f}")

            if m["op_acc"] >= 0.999 and tm["tree_acc"] >= 0.999:
                print(f"\n  🎯 Converged at gen {gen}!")
                break
        else:
            qb = generate_node_batch(rng, 32, max_val, max_depth)
            qo = model.forward(qb["op_codes"], qb["c1_types"], qb["c1_vals"],
                               qb["c2_types"], qb["c2_vals"])
            mx.eval(qo["pred_op"])
            qa = (np.array(qo["pred_op"]) == np.array(qb["gt_ops"])).mean()
            if qa >= best_route:
                champion = save_topology(model)
                best_route = max(best_route, qa)
            else:
                load_topology(model, champion)

    t_total = time.time() - t0
    load_topology(model, champion)

    print(f"\n{'=' * 55}")
    print(f"  Training: {generations} gens, {t_total:.1f}s")

    # Final comprehensive evaluation
    print(f"\n  === Final Evaluation ===")
    for mv in [10, 50, 100]:
        for md in [2, 3, 4, 5, 6, 8]:
            frng = np.random.RandomState(seed + mv * 100 + md)
            fm = evaluate(model, frng, 1024, mv, md)
            trng = np.random.RandomState(seed + mv * 100 + md + 50000)
            tm = evaluate_trees(model, trng, 500, mv, md)
            print(f"    max_val={mv:4d}  depth={md}  "
                  f"op={fm['op_acc']*100:5.1f}%  "
                  f"result={fm['result_acc']*100:5.1f}%  "
                  f"tree={tm['tree_acc']*100:5.1f}%  "
                  f"nodes={fm['n_nodes']}")

    print(f"\n{'=' * 55}")
    return model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--adam-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mutation-pct", type=float, default=0.02)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--max-val", type=int, default=10)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-mix", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    train(**vars(a))
