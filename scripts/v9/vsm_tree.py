"""
v9 — VSM Tree: A tree of Viable System Models

Each expression tree node is a VSM with the same shared weights.
No pipeline bottleneck — each node only sees its children's outputs.

VSM Node (same weights everywhere):
  S5 (identity):     embed my operator → what am I?
  S4 (intelligence): read children's types → are they ready?
  S3 (control):      type check → should I dispatch?
  S2 (coordination): output (type, value) → signal to parent
  S1 (operations):   kernel dispatch → exact computation

For S-expressions: mechanical parse → VSM tree → exact results.
No ascending arm needed. The tree IS the architecture.

License: MIT
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "v8"))
from ternary import TernaryLinear

# ══════════════════════════════════════════════════════════════════════
# Types and ops (same as v9_model)
# ══════════════════════════════════════════════════════════════════════

TYPE_INT = 0
TYPE_OP = 1
TYPE_EXPR = 2
TYPE_ERROR = 3
N_TYPES = 4
TYPE_NAMES = {0: "Int", 1: "Op", 2: "Expr", 3: "Err"}

OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
N_OPS = 3
OP_NAMES = {0: "+", 1: "-", 2: "*"}
OP_STR_TO_CODE = {"+": OP_ADD, "-": OP_SUB, "*": OP_MUL}
OPS = ["+", "-", "*"]


def kernel_apply(op, a1, a2):
    """Exact arithmetic dispatch."""
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
    if isinstance(node, int): return node
    op, a1, a2 = node
    v1, v2 = eval_tree(a1), eval_tree(a2)
    return v1 + v2 if op == "+" else v1 - v2 if op == "-" else v1 * v2

def tree_to_str(node):
    if isinstance(node, int): return str(node)
    op, a1, a2 = node
    return f"({op} {tree_to_str(a1)} {tree_to_str(a2)})"

def linearize_bottomup(node):
    """Flatten tree into bottom-up order: list of (op_code, a1_val, a2_val, result)."""
    if isinstance(node, int): return []
    op, a1, a2 = node
    steps = linearize_bottomup(a1) + linearize_bottomup(a2)
    steps.append((OP_STR_TO_CODE[op], eval_tree(a1), eval_tree(a2), eval_tree(node)))
    return steps


# ══════════════════════════════════════════════════════════════════════
# VSM Node — the shared viable system
# ══════════════════════════════════════════════════════════════════════

@dataclass
class VSMConfig:
    d_model: int = 64
    n_ops: int = N_OPS
    n_types: int = N_TYPES
    max_val: int = 100      # routing range for value logits
    val_embed_range: int = 200  # embedding range [-100, 100)


class VSMNode(nn.Module):
    """One Viable System — shared weights applied at every tree position.

    Input encoding:
      - For leaf (literal int): value embedding
      - For inner node: op embedding + child1 (type, value) + child2 (type, value)

    The node produces routing logits for (op, arg1, arg2) and dispatches
    to the kernel exactly like the query-based router — but operating on
    LOCAL information (just this node's children), not a global sequence.

    S5: op_embed + type_embeds → identity
    S4: child representations → intelligence (mix layers assess children)
    S3: type logits → control (type check)
    S1: kernel dispatch → operations (exact computation)
    S2: output (type, value) → coordination (to parent)
    """

    def __init__(self, config: VSMConfig | None = None):
        super().__init__()
        if config is None:
            config = VSMConfig()
        self.config = config
        d = config.d_model

        # Embeddings for the node's local context
        self.op_embed = nn.Embedding(config.n_ops, d)
        self.type_embed = nn.Embedding(config.n_types, d)
        self.val_embed = nn.Embedding(config.val_embed_range, d)
        self._val_offset = config.val_embed_range // 2

        # Position embeddings: 0=op, 1=child1, 2=child2
        self.pos_embed = nn.Embedding(3, d)

        # S4: ternary mixing — assess the combined (op, child1, child2)
        self.mix1 = TernaryLinear(d, d, pre_norm=True)
        self.mix2 = TernaryLinear(d, d, pre_norm=True)

        # S3+S5: type head — what type is my output?
        type_dim = ((config.n_types + 15) // 16) * 16
        self.type_proj = TernaryLinear(d, type_dim, pre_norm=True)
        self._type_dim = config.n_types

        # S1: routing projections — what op and args to dispatch?
        op_dim = ((config.n_ops + 15) // 16) * 16
        arg_dim = ((config.max_val + 15) // 16) * 16
        self.op_proj = TernaryLinear(d, op_dim, pre_norm=True)
        self.arg1_proj = TernaryLinear(d, arg_dim, pre_norm=True)
        self.arg2_proj = TernaryLinear(d, arg_dim, pre_norm=True)
        self._op_dim = config.n_ops
        self._arg_dim = config.max_val

    def _val_idx(self, val):
        return mx.clip(val + self._val_offset, 0, self.config.val_embed_range - 1).astype(mx.int32)

    def forward(
        self,
        op_codes: mx.array,     # (B,) int — operator code
        c1_types: mx.array,     # (B,) int — child 1 type
        c1_vals: mx.array,      # (B,) int — child 1 value
        c2_types: mx.array,     # (B,) int — child 2 type
        c2_vals: mx.array,      # (B,) int — child 2 value
    ) -> dict[str, mx.array]:
        """Process one batch of tree nodes through the shared VSM.

        Each node sees ONLY its operator and its two children's (type, value).
        Same weights regardless of tree position or depth.
        """
        d = self.config.d_model
        pos = self.pos_embed(mx.arange(3))  # (3, d)

        # S5: Encode identity — who am I?
        op_repr = self.op_embed(op_codes) + pos[0]  # (B, d)

        # S4: Encode children — what do I see?
        # Each child is represented as type_embed + val_embed
        c1_repr = self.type_embed(c1_types) + self.val_embed(self._val_idx(c1_vals)) + pos[1]
        c2_repr = self.type_embed(c2_types) + self.val_embed(self._val_idx(c2_vals)) + pos[2]

        # Combine: op + child1 + child2
        x = op_repr + c1_repr + c2_repr  # (B, d)

        # S4: Mix — assess the combined information
        x = x + self.mix1(x)
        x = x + self.mix2(x)

        # S3: Type check — what type is my output?
        type_logits = self.type_proj(x)[:, :self._type_dim]
        pred_type = mx.argmax(type_logits, axis=-1).astype(mx.int32)

        # S1: Route — what op and args should I dispatch?
        op_logits = self.op_proj(x)[:, :self._op_dim]
        a1_logits = self.arg1_proj(x)[:, :self._arg_dim]
        a2_logits = self.arg2_proj(x)[:, :self._arg_dim]

        pred_op = mx.argmax(op_logits, axis=-1).astype(mx.int32)
        pred_a1 = mx.argmax(a1_logits, axis=-1).astype(mx.int32)
        pred_a2 = mx.argmax(a2_logits, axis=-1).astype(mx.int32)

        # S1: Dispatch — exact kernel computation
        # Type gate: only dispatch if output type is INT (not ERROR)
        pred_result = kernel_apply(pred_op, pred_a1, pred_a2)
        pred_result = mx.where(pred_type == TYPE_INT, pred_result, mx.zeros_like(pred_result))

        return {
            "type_logits": type_logits,
            "op_logits": op_logits,
            "arg1_logits": a1_logits,
            "arg2_logits": a2_logits,
            "pred_type": pred_type,
            "pred_op": pred_op,
            "pred_a1": pred_a1,
            "pred_a2": pred_a2,
            "pred_result": pred_result,
        }

    def count_params(self):
        from mlx.utils import tree_flatten as tf
        total = ternary = continuous = 0
        for _, p in tf(self.parameters()):
            n = p.size
            total += n
            if p.dtype == mx.uint32: ternary += n * 16
            elif p.dtype == mx.uint8: ternary += n * 4
            else: continuous += n
        return {"total": total, "ternary": ternary, "continuous": continuous}


# ══════════════════════════════════════════════════════════════════════
# Batched tree execution
# ══════════════════════════════════════════════════════════════════════

def batch_trees_bottomup(
    trees: list,
    max_val: int,
) -> list[dict[str, list]]:
    """Convert a list of expression trees into batched bottom-up levels.

    Returns a list of levels, each level is a dict with:
      - op_codes: operator codes for nodes at this level
      - c1_types, c1_vals: child 1 info
      - c2_types, c2_vals: child 2 info
      - gt_op, gt_a1, gt_a2, gt_result: ground truth
      - tree_idx, node_idx: for mapping results back

    Leaves are level 0 (both children are literals).
    Inner nodes are level 1+ (at least one child is a sub-expression).
    """
    # Linearize all trees into bottom-up ordered nodes
    all_nodes = []  # (level, op_code, c1_type, c1_val, c2_type, c2_val, gt_result, tree_i)

    for i, tree in enumerate(trees):
        _collect_nodes(tree, all_nodes, i, max_val)

    # Group by level
    max_level = max(n[0] for n in all_nodes) if all_nodes else 0
    levels = []
    for lv in range(max_level + 1):
        nodes = [n for n in all_nodes if n[0] == lv]
        if not nodes:
            continue
        levels.append({
            "op_codes": [n[1] for n in nodes],
            "c1_types": [n[2] for n in nodes],
            "c1_vals": [n[3] for n in nodes],
            "c2_types": [n[4] for n in nodes],
            "c2_vals": [n[5] for n in nodes],
            "gt_ops": [n[1] for n in nodes],
            "gt_a1": [n[3] for n in nodes],
            "gt_a2": [n[5] for n in nodes],
            "gt_results": [n[6] for n in nodes],
        })
    return levels


def _node_depth(node) -> int:
    if isinstance(node, int): return 0
    _, a1, a2 = node
    return 1 + max(_node_depth(a1), _node_depth(a2))


def _collect_nodes(node, out, tree_i, max_val, depth=0):
    """Recursively collect tree nodes with their level and children info."""
    if isinstance(node, int):
        return  # leaves aren't nodes — they're children of nodes

    op, a1, a2 = node
    op_code = OP_STR_TO_CODE[op]

    # Determine level: max depth of children
    level = max(_node_depth(a1), _node_depth(a2))

    # Child 1 info
    if isinstance(a1, int):
        c1_type, c1_val = TYPE_INT, a1
    else:
        c1_type, c1_val = TYPE_INT, eval_tree(a1)  # sub-expr evaluates to INT

    # Child 2 info
    if isinstance(a2, int):
        c2_type, c2_val = TYPE_INT, a2
    else:
        c2_type, c2_val = TYPE_INT, eval_tree(a2)

    gt_result = eval_tree(node)

    out.append((level, op_code, c1_type, c1_val, c2_type, c2_val, gt_result, tree_i))

    # Recurse into children
    _collect_nodes(a1, out, tree_i, max_val, depth + 1)
    _collect_nodes(a2, out, tree_i, max_val, depth + 1)


# ══════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════

def vsm_loss(
    model: VSMNode,
    op_codes: mx.array,
    c1_types: mx.array,
    c1_vals: mx.array,
    c2_types: mx.array,
    c2_vals: mx.array,
    gt_ops: mx.array,
    gt_a1: mx.array,
    gt_a2: mx.array,
) -> mx.array:
    """Per-node routing loss for a batch of VSM nodes."""
    config = model.config
    out = model.forward(op_codes, c1_types, c1_vals, c2_types, c2_vals)

    # Type: output should be INT (all our expressions produce ints)
    gt_type = mx.full(op_codes.shape, TYPE_INT, dtype=mx.int32)
    loss_type = nn.losses.cross_entropy(out["type_logits"], gt_type, reduction="mean")

    # Parse: op, arg1, arg2
    loss_op = nn.losses.cross_entropy(out["op_logits"], gt_ops, reduction="mean")
    gt_a1c = mx.clip(gt_a1, 0, config.max_val - 1).astype(mx.int32)
    gt_a2c = mx.clip(gt_a2, 0, config.max_val - 1).astype(mx.int32)
    loss_a1 = nn.losses.cross_entropy(out["arg1_logits"], gt_a1c, reduction="mean")
    loss_a2 = nn.losses.cross_entropy(out["arg2_logits"], gt_a2c, reduction="mean")

    return 0.5 * loss_type + loss_op + loss_a1 + loss_a2


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════

def generate_node_batch(rng, batch_size, max_val, max_depth):
    """Generate a flat batch of VSM nodes from random trees.

    Every node from every tree goes into one batch — the VSM doesn't
    care which tree a node came from. Same weights, same processing.
    """
    all_ops, all_c1t, all_c1v, all_c2t, all_c2v = [], [], [], [], []
    all_gt_ops, all_gt_a1, all_gt_a2, all_gt_res = [], [], [], []

    for _ in range(batch_size):
        tree = random_expr(rng, max_val, max_depth)
        levels = batch_trees_bottomup([tree], max_val)
        for lv in levels:
            all_ops.extend(lv["op_codes"])
            all_c1t.extend(lv["c1_types"])
            all_c1v.extend(lv["c1_vals"])
            all_c2t.extend(lv["c2_types"])
            all_c2v.extend(lv["c2_vals"])
            all_gt_ops.extend(lv["gt_ops"])
            all_gt_a1.extend(lv["gt_a1"])
            all_gt_a2.extend(lv["gt_a2"])
            all_gt_res.extend(lv["gt_results"])

    return {
        "op_codes": mx.array(all_ops),
        "c1_types": mx.array(all_c1t),
        "c1_vals": mx.array(all_c1v),
        "c2_types": mx.array(all_c2t),
        "c2_vals": mx.array(all_c2v),
        "gt_ops": mx.array(all_gt_ops),
        "gt_a1": mx.array(all_gt_a1),
        "gt_a2": mx.array(all_gt_a2),
        "gt_res": mx.array(all_gt_res),
    }


def evaluate(model, rng, n_exprs, max_val, max_depth):
    batch = generate_node_batch(rng, n_exprs, max_val, max_depth)
    out = model.forward(batch["op_codes"], batch["c1_types"], batch["c1_vals"],
                        batch["c2_types"], batch["c2_vals"])
    for v in out.values(): mx.eval(v)

    po = np.array(out["pred_op"])
    pa1 = np.array(out["pred_a1"])
    pa2 = np.array(out["pred_a2"])
    pt = np.array(out["pred_type"])
    pr = np.array(out["pred_result"])
    go = np.array(batch["gt_ops"])
    ga1 = np.array(batch["gt_a1"])
    ga2 = np.array(batch["gt_a2"])
    gr = np.array(batch["gt_res"])

    return {
        "type_acc": float((pt == TYPE_INT).mean()),
        "op_acc": float((po == go).mean()),
        "a1_acc": float((pa1 == ga1).mean()),
        "a2_acc": float((pa2 == ga2).mean()),
        "route_acc": float(((po == go) & (pa1 == ga1) & (pa2 == ga2)).mean()),
        "result_acc": float((pr == gr).mean()),
        "n_nodes": len(go),
    }


def train(
    generations=2000, batch_size=128, adam_steps=10, lr=1e-3,
    mutation_pct=0.02, eval_interval=100, max_val=10, max_depth=2,
    d_model=64, seed=42,
):
    import time
    from ternary import (save_topology, load_topology, zero_ternary_grads,
                         restore_ternary, count_ternary_weights, mutate_topology)
    import mlx.optimizers as optim

    print("=" * 70)
    print("  v9 — VSM Tree Training")
    print("=" * 70)

    rng = np.random.RandomState(seed)
    config = VSMConfig(d_model=d_model, max_val=max_val)
    model = VSMNode(config)

    n_ternary = count_ternary_weights(model)
    mut_budget = max(1, int(n_ternary * mutation_pct))
    params = model.count_params()

    print(f"\n  d_model={d_model}  max_val={max_val}  max_depth={max_depth}")
    print(f"  ternary={n_ternary:,}  continuous={params['continuous']:,}  mut={mut_budget}")
    print(f"  gens={generations}  adam_steps={adam_steps}  batch={batch_size}  lr={lr}")

    optimizer = optim.Adam(learning_rate=lr)
    loss_fn = nn.value_and_grad(model, vsm_loss)

    best_route = -1.0
    champion = save_topology(model)

    # Initial
    erng = np.random.RandomState(seed + 1000)
    m = evaluate(model, erng, 512, max_val, max_depth)
    best_route = m["route_acc"]
    print(f"\n  Initial: op={m['op_acc']*100:.0f}%  a1={m['a1_acc']*100:.0f}%  "
          f"a2={m['a2_acc']*100:.0f}%  route={m['route_acc']*100:.0f}%  "
          f"result={m['result_acc']*100:.0f}%  nodes={m['n_nodes']}")

    print(f"\n{'Gen':>5}  {'Loss':>7}  {'Typ':>4}  {'Op':>4}  {'A1':>4}  "
          f"{'A2':>4}  {'Rte':>4}  {'Res':>4}  {'N':>4}  {'M':>2}  {'dt':>4}")
    print("-" * 65)

    t0 = time.time()
    ma, mt = 0, 0

    for gen in range(generations):
        g0 = time.time()

        avg_loss = 0.0
        for _ in range(adam_steps):
            b = generate_node_batch(rng, batch_size, max_val, max_depth)
            loss, grads = loss_fn(model, b["op_codes"], b["c1_types"], b["c1_vals"],
                                  b["c2_types"], b["c2_vals"], b["gt_ops"], b["gt_a1"], b["gt_a2"])
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
            if m["route_acc"] >= best_route:
                best_route = m["route_acc"]
                champion = save_topology(model)
                ma += 1; status = "✓"
            else:
                load_topology(model, champion)
                status = "✗"
            mt += 1
            dt = time.time() - g0
            print(f"  {gen:4d}  {avg_loss:7.3f}  {m['type_acc']*100:3.0f}%  "
                  f"{m['op_acc']*100:3.0f}%  {m['a1_acc']*100:3.0f}%  "
                  f"{m['a2_acc']*100:3.0f}%  {m['route_acc']*100:3.0f}%  "
                  f"{m['result_acc']*100:3.0f}%  {m['n_nodes']:3d}  "
                  f" {status}  {dt:3.1f}")
            if m["route_acc"] >= 0.95:
                print(f"\n  🎯 Converged at gen {gen}!")
                break
        else:
            qb = generate_node_batch(rng, 32, max_val, max_depth)
            qo = model.forward(qb["op_codes"], qb["c1_types"], qb["c1_vals"],
                               qb["c2_types"], qb["c2_vals"])
            mx.eval(qo["pred_op"], qo["pred_a1"], qo["pred_a2"])
            qa = ((np.array(qo["pred_op"]) == np.array(qb["gt_ops"])) &
                  (np.array(qo["pred_a1"]) == np.array(qb["gt_a1"])) &
                  (np.array(qo["pred_a2"]) == np.array(qb["gt_a2"]))).mean()
            if qa >= best_route:
                champion = save_topology(model)
                best_route = max(best_route, qa)
                ma += 1
            else:
                load_topology(model, champion)
            mt += 1

    t_total = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"  Done: {generations} gens, {t_total:.1f}s, mutations {ma}/{mt}")

    frng = np.random.RandomState(seed + 99999)
    f = evaluate(model, frng, 1024, max_val, max_depth)
    print(f"\n  Final (1024 trees, {f['n_nodes']} nodes):")
    print(f"    Type:   {f['type_acc']*100:.1f}%")
    print(f"    Op:     {f['op_acc']*100:.1f}%")
    print(f"    Arg1:   {f['a1_acc']*100:.1f}%")
    print(f"    Arg2:   {f['a2_acc']*100:.1f}%")
    print(f"    Route:  {f['route_acc']*100:.1f}%")
    print(f"    Result: {f['result_acc']*100:.1f}%")

    print(f"\n{'=' * 65}")
    if f["route_acc"] > 0.5:
        print("  ✅ VIABLE: VSM tree routing works.")
    elif f["route_acc"] > 0.1:
        print("  🔄 PARTIAL: Learning. Check components.")
    elif f["op_acc"] > 0.5:
        print("  💡 Op works, arg routing needs work.")
    else:
        print("  ❌ Not converging at this scale.")
    print(f"{'=' * 65}")
    return model, f


if __name__ == "__main__":
    import argparse
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
    train(**vars(a))
