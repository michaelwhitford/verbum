"""
v9 — VSM Tree v4: Expanded Kernel

Extends v3 from 3 arithmetic ops to 18 ops across 5 categories:
  - Arithmetic binary:  +, -, *, //, mod, min, max   (7 ops, INT×INT→INT)
  - Comparison:         =, <, >, <=, >=              (5 ops, INT×INT→BOOL)
  - Boolean binary:     and, or                      (2 ops, BOOL×BOOL→BOOL)
  - Boolean unary:      not                          (1 op,  BOOL→BOOL)
  - Arithmetic unary:   abs, neg                     (2 ops, INT→INT)
  - Conditional:        if                           (1 op,  BOOL×T×T→T, ternary)

Tests three architectural properties beyond v3:
  1. Mixed types (INT and BOOL flowing through the tree)
  2. Variable arity (unary, binary, ternary nodes)
  3. Type-directed dispatch (type determines kernel behavior)

The VSM node still only classifies the operation. Values pass through.
But now it must handle variable-arity children and predict result type.

License: MIT
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from enum import IntEnum
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
# Type system
# ══════════════════════════════════════════════════════════════════════

class Type(IntEnum):
    INT = 0
    BOOL = 1
    ERROR = 2

N_TYPES = 3


# ══════════════════════════════════════════════════════════════════════
# Operations — the expanded kernel
# ══════════════════════════════════════════════════════════════════════

class Op(IntEnum):
    # Arithmetic binary (INT × INT → INT)
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3      # integer division, 0 on div-by-zero
    MOD = 4      # modulo, 0 on div-by-zero
    MIN = 5
    MAX = 6
    # Comparison (INT × INT → BOOL)
    EQ = 7
    LT = 8
    GT = 9
    LE = 10
    GE = 11
    # Boolean binary (BOOL × BOOL → BOOL)
    AND = 12
    OR = 13
    # Boolean unary (BOOL → BOOL)
    NOT = 14
    # Arithmetic unary (INT → INT)
    ABS = 15
    NEG = 16
    # Conditional (BOOL × T × T → T)
    IF = 17

N_OPS = 18

# Operation metadata: arity, input types, output type
# For IF: output type matches children (always INT in our generation)
OP_META = {
    # (arity, input_types, output_type)
    Op.ADD: (2, (Type.INT, Type.INT), Type.INT),
    Op.SUB: (2, (Type.INT, Type.INT), Type.INT),
    Op.MUL: (2, (Type.INT, Type.INT), Type.INT),
    Op.DIV: (2, (Type.INT, Type.INT), Type.INT),
    Op.MOD: (2, (Type.INT, Type.INT), Type.INT),
    Op.MIN: (2, (Type.INT, Type.INT), Type.INT),
    Op.MAX: (2, (Type.INT, Type.INT), Type.INT),
    Op.EQ:  (2, (Type.INT, Type.INT), Type.BOOL),
    Op.LT:  (2, (Type.INT, Type.INT), Type.BOOL),
    Op.GT:  (2, (Type.INT, Type.INT), Type.BOOL),
    Op.LE:  (2, (Type.INT, Type.INT), Type.BOOL),
    Op.GE:  (2, (Type.INT, Type.INT), Type.BOOL),
    Op.AND: (2, (Type.BOOL, Type.BOOL), Type.BOOL),
    Op.OR:  (2, (Type.BOOL, Type.BOOL), Type.BOOL),
    Op.NOT: (1, (Type.BOOL,), Type.BOOL),
    Op.ABS: (1, (Type.INT,), Type.INT),
    Op.NEG: (1, (Type.INT,), Type.INT),
    Op.IF:  (3, (Type.BOOL, None, None), None),  # polymorphic
}

# Group ops by arity for generation
BINARY_INT_OPS = [Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD, Op.MIN, Op.MAX]
COMPARISON_OPS = [Op.EQ, Op.LT, Op.GT, Op.LE, Op.GE]
BINARY_BOOL_OPS = [Op.AND, Op.OR]
UNARY_BOOL_OPS = [Op.NOT]
UNARY_INT_OPS = [Op.ABS, Op.NEG]

OP_NAMES = {
    Op.ADD: "+", Op.SUB: "-", Op.MUL: "*", Op.DIV: "//", Op.MOD: "%",
    Op.MIN: "min", Op.MAX: "max",
    Op.EQ: "=", Op.LT: "<", Op.GT: ">", Op.LE: "<=", Op.GE: ">=",
    Op.AND: "and", Op.OR: "or", Op.NOT: "not",
    Op.ABS: "abs", Op.NEG: "neg",
    Op.IF: "if",
}


# ══════════════════════════════════════════════════════════════════════
# Kernel dispatch — exact computation
# ══════════════════════════════════════════════════════════════════════

def kernel_eval(op: int, args: list[int]) -> tuple[int, int]:
    """Evaluate one kernel operation. Returns (result_value, result_type).

    Values: integers (BOOL encoded as 0/1).
    """
    if op == Op.ADD:
        return args[0] + args[1], Type.INT
    elif op == Op.SUB:
        return args[0] - args[1], Type.INT
    elif op == Op.MUL:
        return args[0] * args[1], Type.INT
    elif op == Op.DIV:
        return (args[0] // args[1] if args[1] != 0 else 0), Type.INT
    elif op == Op.MOD:
        return (args[0] % args[1] if args[1] != 0 else 0), Type.INT
    elif op == Op.MIN:
        return min(args[0], args[1]), Type.INT
    elif op == Op.MAX:
        return max(args[0], args[1]), Type.INT
    elif op == Op.EQ:
        return int(args[0] == args[1]), Type.BOOL
    elif op == Op.LT:
        return int(args[0] < args[1]), Type.BOOL
    elif op == Op.GT:
        return int(args[0] > args[1]), Type.BOOL
    elif op == Op.LE:
        return int(args[0] <= args[1]), Type.BOOL
    elif op == Op.GE:
        return int(args[0] >= args[1]), Type.BOOL
    elif op == Op.AND:
        return int(bool(args[0]) and bool(args[1])), Type.BOOL
    elif op == Op.OR:
        return int(bool(args[0]) or bool(args[1])), Type.BOOL
    elif op == Op.NOT:
        return int(not bool(args[0])), Type.BOOL
    elif op == Op.ABS:
        return abs(args[0]), Type.INT
    elif op == Op.NEG:
        return -args[0], Type.INT
    elif op == Op.IF:
        return (args[1] if bool(args[0]) else args[2]), Type.INT
    else:
        return 0, Type.ERROR


def kernel_apply_batch(ops, c1_vals, c2_vals, c3_vals, arities):
    """Vectorized kernel dispatch for a batch of operations.

    All values are int64. BOOL encoded as 0/1.
    c2_vals and c3_vals are 0 for ops that don't use them.
    """
    result = mx.zeros_like(c1_vals)

    # Arithmetic binary
    result = mx.where(ops == Op.ADD, c1_vals + c2_vals, result)
    result = mx.where(ops == Op.SUB, c1_vals - c2_vals, result)
    result = mx.where(ops == Op.MUL, c1_vals * c2_vals, result)
    # Safe division
    safe_c2 = mx.where(c2_vals == 0, mx.ones_like(c2_vals), c2_vals)
    result = mx.where(ops == Op.DIV, c1_vals // safe_c2, result)
    result = mx.where((ops == Op.DIV) & (c2_vals == 0), mx.zeros_like(result), result)
    result = mx.where(ops == Op.MOD, c1_vals % safe_c2, result)
    result = mx.where((ops == Op.MOD) & (c2_vals == 0), mx.zeros_like(result), result)
    result = mx.where(ops == Op.MIN, mx.minimum(c1_vals, c2_vals), result)
    result = mx.where(ops == Op.MAX, mx.maximum(c1_vals, c2_vals), result)

    # Comparison (result is 0 or 1)
    result = mx.where(ops == Op.EQ, (c1_vals == c2_vals).astype(mx.int64), result)
    result = mx.where(ops == Op.LT, (c1_vals < c2_vals).astype(mx.int64), result)
    result = mx.where(ops == Op.GT, (c1_vals > c2_vals).astype(mx.int64), result)
    result = mx.where(ops == Op.LE, (c1_vals <= c2_vals).astype(mx.int64), result)
    result = mx.where(ops == Op.GE, (c1_vals >= c2_vals).astype(mx.int64), result)

    # Boolean binary
    result = mx.where(ops == Op.AND, (c1_vals & c2_vals), result)
    result = mx.where(ops == Op.OR, (c1_vals | c2_vals), result)

    # Boolean unary
    result = mx.where(ops == Op.NOT, 1 - c1_vals, result)

    # Arithmetic unary
    result = mx.where(ops == Op.ABS, mx.abs(c1_vals), result)
    result = mx.where(ops == Op.NEG, -c1_vals, result)

    # Conditional: if(test, then, else)
    result = mx.where(ops == Op.IF,
                      mx.where(c1_vals != 0, c2_vals, c3_vals),
                      result)

    return result


# ══════════════════════════════════════════════════════════════════════
# Expression tree generation — well-typed random expressions
# ══════════════════════════════════════════════════════════════════════

def random_expr(rng, max_val: int, max_depth: int, target_type: int = Type.INT, depth: int = 0):
    """Generate a random well-typed expression tree.

    Returns: (op, children...) or a leaf value.
    Leaf INT: random integer in [0, max_val)
    Leaf BOOL: 0 or 1
    """
    # Leaf if at max depth or randomly
    if depth >= max_depth or (depth > 0 and rng.random() < 0.3):
        if target_type == Type.INT:
            return int(rng.randint(0, max_val))
        else:
            return int(rng.randint(0, 2))  # 0 or 1

    if target_type == Type.INT:
        # Choose among: arithmetic binary, arithmetic unary, IF
        choice = rng.random()
        if choice < 0.55:
            # Arithmetic binary
            op = rng.choice(BINARY_INT_OPS)
            a1 = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            a2 = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            return (int(op), a1, a2)
        elif choice < 0.70:
            # Arithmetic unary
            op = rng.choice(UNARY_INT_OPS)
            a1 = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            return (int(op), a1)
        else:
            # IF expression (returns INT)
            test = random_expr(rng, max_val, max_depth, Type.BOOL, depth + 1)
            then = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            els = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            return (int(Op.IF), test, then, els)
    else:
        # target_type == Type.BOOL
        choice = rng.random()
        if choice < 0.45:
            # Comparison
            op = rng.choice(COMPARISON_OPS)
            a1 = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            a2 = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            return (int(op), a1, a2)
        elif choice < 0.75:
            # Boolean binary
            op = rng.choice(BINARY_BOOL_OPS)
            a1 = random_expr(rng, max_val, max_depth, Type.BOOL, depth + 1)
            a2 = random_expr(rng, max_val, max_depth, Type.BOOL, depth + 1)
            return (int(op), a1, a2)
        else:
            # Boolean unary
            a1 = random_expr(rng, max_val, max_depth, Type.BOOL, depth + 1)
            return (int(Op.NOT), a1)


def eval_tree(node) -> tuple[int, int]:
    """Evaluate a tree, returning (value, type)."""
    if isinstance(node, int):
        # Leaf — type inferred from value (0/1 could be either, but
        # we trust the generator made it correctly typed)
        return node, Type.INT  # will be overridden by context
    op = node[0]
    children = node[1:]
    child_vals = [eval_tree(c)[0] for c in children]
    return kernel_eval(op, child_vals)


def eval_tree_typed(node, expected_type=Type.INT) -> tuple[int, int]:
    """Evaluate with type tracking."""
    if isinstance(node, int):
        return node, expected_type

    op = node[0]
    children = node[1:]
    meta = OP_META[op]
    arity = meta[0]
    input_types = meta[1]
    output_type = meta[2] if meta[2] is not None else expected_type

    # Evaluate children with correct types
    child_vals = []
    for i, child in enumerate(children):
        if op == Op.IF:
            if i == 0:
                ct = Type.BOOL
            else:
                ct = expected_type  # then/else match output type
        else:
            ct = input_types[i] if i < len(input_types) else Type.INT
        val, _ = eval_tree_typed(child, ct)
        child_vals.append(val)

    result, rtype = kernel_eval(op, child_vals)
    return result, rtype


def tree_to_str(node) -> str:
    if isinstance(node, int):
        return str(node)
    op = node[0]
    children = node[1:]
    name = OP_NAMES[op]
    args = " ".join(tree_to_str(c) for c in children)
    return f"({name} {args})"


# ══════════════════════════════════════════════════════════════════════
# Data generation — collect tree nodes for batched training
# ══════════════════════════════════════════════════════════════════════

def _collect_nodes(node, out, expected_type=Type.INT):
    """Collect tree nodes bottom-up with computed child values and types.

    Each node record: (op, arity, c1_val, c1_type, c2_val, c2_type,
                       c3_val, c3_type, result_val, result_type)
    """
    if isinstance(node, int):
        return node, expected_type

    op = node[0]
    children = node[1:]
    meta = OP_META[op]
    arity = meta[0]
    input_types = meta[1]
    output_type = meta[2] if meta[2] is not None else expected_type

    # Determine child types
    child_types = []
    for i in range(arity):
        if op == Op.IF:
            ct = Type.BOOL if i == 0 else expected_type
        else:
            ct = input_types[i] if i < len(input_types) else Type.INT
        child_types.append(ct)

    # Recurse into children (bottom-up)
    child_vals = []
    child_actual_types = []
    for i, child in enumerate(children):
        val, ctype = _collect_nodes(child, out, child_types[i])
        child_vals.append(val)
        child_actual_types.append(ctype)

    # Compute this node's result
    result, rtype = kernel_eval(op, child_vals)

    # Pad to 3 children
    c_vals = child_vals + [0] * (3 - len(child_vals))
    c_types = child_actual_types + [Type.INT] * (3 - len(child_actual_types))

    out.append({
        "op": op,
        "arity": arity,
        "c1_val": c_vals[0], "c1_type": c_types[0],
        "c2_val": c_vals[1], "c2_type": c_types[1],
        "c3_val": c_vals[2], "c3_type": c_types[2],
        "result_val": result, "result_type": rtype,
    })

    return result, rtype


def generate_batch(rng, batch_size, max_val, max_depth):
    """Generate a batch of nodes from random well-typed trees."""
    all_nodes = []
    for _ in range(batch_size):
        # Mix INT and BOOL target trees
        target = Type.INT if rng.random() < 0.7 else Type.BOOL
        tree = random_expr(rng, max_val, max_depth, target)
        _collect_nodes(tree, all_nodes, target)

    if not all_nodes:
        # Edge case: all trees are leaves
        return None

    return {
        "ops": mx.array([n["op"] for n in all_nodes], dtype=mx.int32),
        "arities": mx.array([n["arity"] for n in all_nodes], dtype=mx.int32),
        "c1_vals": mx.array([n["c1_val"] for n in all_nodes], dtype=mx.int64),
        "c1_types": mx.array([n["c1_type"] for n in all_nodes], dtype=mx.int32),
        "c2_vals": mx.array([n["c2_val"] for n in all_nodes], dtype=mx.int64),
        "c2_types": mx.array([n["c2_type"] for n in all_nodes], dtype=mx.int32),
        "c3_vals": mx.array([n["c3_val"] for n in all_nodes], dtype=mx.int64),
        "c3_types": mx.array([n["c3_type"] for n in all_nodes], dtype=mx.int32),
        "gt_results": mx.array([n["result_val"] for n in all_nodes], dtype=mx.int64),
        "gt_types": mx.array([n["result_type"] for n in all_nodes], dtype=mx.int32),
    }


# ══════════════════════════════════════════════════════════════════════
# VSM Node v4 — handles variable arity + mixed types
# ══════════════════════════════════════════════════════════════════════

@dataclass
class VSMConfig:
    d_model: int = 64
    n_ops: int = N_OPS
    n_types: int = N_TYPES
    val_embed_range: int = 200
    n_mix_layers: int = 2


class VSMNodeV4(nn.Module):
    """VSM node for expanded kernel: classifies op + type, passes values.

    Input: op embedding + up to 3 children's (type_embed + val_embed).
    Output: predicted op, predicted result type, exact kernel result.

    The node handles variable arity by zero-padding unused children.
    Arity is implicit in the op classification — once the model knows
    the op, it knows how many children are meaningful.
    """

    def __init__(self, config: VSMConfig | None = None):
        super().__init__()
        if config is None:
            config = VSMConfig()
        self.config = config
        d = config.d_model

        # Op embedding (the identity of this node)
        self.op_embed = nn.Embedding(config.n_ops, d)
        # Type embedding (for children)
        self.type_embed = nn.Embedding(config.n_types, d)
        # Value embedding (for children's values — contextual, not routed)
        self.val_embed = nn.Embedding(config.val_embed_range, d)
        self._val_offset = config.val_embed_range // 2
        # Position embedding: 0=op, 1=child1, 2=child2, 3=child3
        self.pos_embed = nn.Embedding(4, d)
        # Arity embedding (1, 2, or 3)
        self.arity_embed = nn.Embedding(4, d)  # 0=unused, 1, 2, 3

        # Input: sum of all embeddings → d (additive is fine here since
        # we're NOT classifying values — only the op matters)
        # But use concat for op + children since op is the key signal
        self.input_proj = nn.Linear(4 * d, d)  # [op; c1; c2; c3]

        # Mix layers
        self.mix_layers = [TernaryLinear(d, d, pre_norm=True)
                           for _ in range(config.n_mix_layers)]

        # Op head — with RESIDUAL from op embedding (same insight as v2 value residual)
        # The op identity should not have to survive through the ternary bottleneck
        op_dim = ((config.n_ops + 15) // 16) * 16
        self.op_proj = nn.Linear(d + d, op_dim)  # [mixed; op_embed]
        self._op_dim = config.n_ops

        # Result type head — with op residual (type is determined by op)
        type_dim = ((config.n_types + 15) // 16) * 16
        self.type_proj = nn.Linear(d + d, type_dim)  # [mixed; op_embed]
        self._type_dim = config.n_types

    def _val_idx(self, val):
        return mx.clip(val + self._val_offset, 0, self.config.val_embed_range - 1).astype(mx.int32)

    def forward(self, ops, arities, c1_vals, c1_types, c2_vals, c2_types,
                c3_vals, c3_types):
        """Forward pass for a batch of tree nodes.

        All inputs are (B,) arrays. Unused children have val=0, type=0.
        """
        d = self.config.d_model

        # Embed this node's op identity
        op_repr = self.op_embed(ops) + self.pos_embed(mx.zeros_like(ops)) + self.arity_embed(arities)

        # Embed each child: type + value + position
        c1_repr = (self.type_embed(c1_types) +
                   self.val_embed(self._val_idx(c1_vals)) +
                   self.pos_embed(mx.ones_like(ops)))
        c2_repr = (self.type_embed(c2_types) +
                   self.val_embed(self._val_idx(c2_vals)) +
                   self.pos_embed(mx.full(ops.shape, 2, dtype=mx.int32)))
        c3_repr = (self.type_embed(c3_types) +
                   self.val_embed(self._val_idx(c3_vals)) +
                   self.pos_embed(mx.full(ops.shape, 3, dtype=mx.int32)))

        # Mask unused children (arity < 2 → zero c2, arity < 3 → zero c3)
        mask2 = (arities >= 2).astype(mx.float32).reshape(-1, 1)
        mask3 = (arities >= 3).astype(mx.float32).reshape(-1, 1)
        c2_repr = c2_repr * mask2
        c3_repr = c3_repr * mask3

        # Fuse: concat [op; c1; c2; c3] → project
        x = self.input_proj(mx.concatenate([op_repr, c1_repr, c2_repr, c3_repr], axis=-1))

        # Mix
        for mix in self.mix_layers:
            x = x + mix(x)

        # Op classification with op residual (18 classes)
        op_input = mx.concatenate([x, op_repr], axis=-1)
        op_logits = self.op_proj(op_input)[:, :self._op_dim]
        pred_op = mx.argmax(op_logits, axis=-1).astype(mx.int32)

        # Type classification with op residual (type is determined by op)
        type_input = mx.concatenate([x, op_repr], axis=-1)
        type_logits = self.type_proj(type_input)[:, :self._type_dim]
        pred_type = mx.argmax(type_logits, axis=-1).astype(mx.int32)

        # Kernel dispatch with actual values (pass-through)
        pred_result = kernel_apply_batch(pred_op, c1_vals, c2_vals, c3_vals, arities)

        return {
            "op_logits": op_logits,
            "type_logits": type_logits,
            "pred_op": pred_op,
            "pred_type": pred_type,
            "pred_result": pred_result,
        }


# ══════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════

def vsm_loss(model, ops, arities, c1_vals, c1_types, c2_vals, c2_types,
             c3_vals, c3_types, gt_results, gt_types):
    out = model.forward(ops, arities, c1_vals, c1_types, c2_vals, c2_types,
                        c3_vals, c3_types)

    loss_op = nn.losses.cross_entropy(out["op_logits"], ops, reduction="mean")
    loss_type = nn.losses.cross_entropy(out["type_logits"], gt_types, reduction="mean")

    return loss_op + 0.5 * loss_type


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_batch(model, rng, n_exprs, max_val, max_depth):
    """Evaluate on a flat batch of nodes."""
    batch = generate_batch(rng, n_exprs, max_val, max_depth)
    if batch is None:
        return {"op_acc": 0, "type_acc": 0, "result_acc": 0, "n_nodes": 0}

    out = model.forward(batch["ops"], batch["arities"],
                        batch["c1_vals"], batch["c1_types"],
                        batch["c2_vals"], batch["c2_types"],
                        batch["c3_vals"], batch["c3_types"])
    for v in out.values():
        mx.eval(v)

    po = np.array(out["pred_op"])
    pt = np.array(out["pred_type"])
    pr = np.array(out["pred_result"])
    go = np.array(batch["ops"])
    gt = np.array(batch["gt_types"])
    gr = np.array(batch["gt_results"])

    return {
        "op_acc": float((po == go).mean()),
        "type_acc": float((pt == gt).mean()),
        "result_acc": float((pr == gr).mean()),
        "n_nodes": len(go),
    }


def evaluate_trees(model, rng, n_trees, max_val, max_depth):
    """Evaluate by executing whole trees bottom-up through the model."""
    correct = 0
    total = 0
    node_stats = {"op_correct": 0, "type_correct": 0, "total": 0}

    for _ in range(n_trees):
        target = Type.INT if rng.random() < 0.7 else Type.BOOL
        tree = random_expr(rng, max_val, max_depth, target)
        gt_val, gt_type = eval_tree_typed(tree, target)

        pred_val = _execute_tree(model, tree, target, node_stats)
        if pred_val == gt_val:
            correct += 1
        total += 1

    return {
        "tree_acc": correct / total if total > 0 else 0.0,
        "node_op_acc": (node_stats["op_correct"] / node_stats["total"]
                        if node_stats["total"] > 0 else 0.0),
        "node_type_acc": (node_stats["type_correct"] / node_stats["total"]
                          if node_stats["total"] > 0 else 0.0),
        "n_trees": total,
        "n_nodes": node_stats["total"],
    }


def _execute_tree(model, node, expected_type, stats):
    """Execute a tree bottom-up through the model."""
    if isinstance(node, int):
        return node

    op = node[0]
    children = node[1:]
    meta = OP_META[op]
    arity = meta[0]
    input_types = meta[1]

    # Determine child types
    child_types = []
    for i in range(arity):
        if op == Op.IF:
            ct = Type.BOOL if i == 0 else expected_type
        else:
            ct = input_types[i] if i < len(input_types) else Type.INT
        child_types.append(ct)

    # Recurse
    child_vals = []
    for i, child in enumerate(children):
        val = _execute_tree(model, child, child_types[i], stats)
        child_vals.append(val)

    # Pad to 3
    while len(child_vals) < 3:
        child_vals.append(0)
    while len(child_types) < 3:
        child_types.append(int(Type.INT))

    # Forward through model
    op_arr = mx.array([op], dtype=mx.int32)
    arity_arr = mx.array([arity], dtype=mx.int32)
    c1v = mx.array([child_vals[0]], dtype=mx.int64)
    c1t = mx.array([int(child_types[0])], dtype=mx.int32)
    c2v = mx.array([child_vals[1]], dtype=mx.int64)
    c2t = mx.array([int(child_types[1])], dtype=mx.int32)
    c3v = mx.array([child_vals[2]], dtype=mx.int64)
    c3t = mx.array([int(child_types[2])], dtype=mx.int32)

    out = model.forward(op_arr, arity_arr, c1v, c1t, c2v, c2t, c3v, c3t)
    mx.eval(out["pred_result"], out["pred_op"], out["pred_type"])

    stats["total"] += 1
    if out["pred_op"].item() == op:
        stats["op_correct"] += 1
    gt_type = OP_META[op][2] if OP_META[op][2] is not None else expected_type
    if out["pred_type"].item() == gt_type:
        stats["type_correct"] += 1

    return int(out["pred_result"].item())


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
    max_depth=3,
    d_model=64,
    n_mix=2,
    seed=42,
):
    print("=" * 70)
    print("  v9 — VSM Tree v4: Expanded Kernel (18 ops, mixed types)")
    print("=" * 70)

    rng = np.random.RandomState(seed)
    config = VSMConfig(d_model=d_model, n_mix_layers=n_mix)
    model = VSMNodeV4(config)

    n_ternary = count_ternary_weights(model)
    mut_budget = max(1, int(n_ternary * mutation_pct))

    print(f"\n  d={d_model}  mix={n_mix}  max_val={max_val}  max_depth={max_depth}")
    print(f"  ops={N_OPS}  types={N_TYPES}  ternary={n_ternary:,}  mut={mut_budget}")
    print(f"  gens={generations}  adam_steps={adam_steps}  lr={lr}")

    optimizer = optim.Adam(learning_rate=lr)
    loss_fn = nn.value_and_grad(model, vsm_loss)

    best_op = -1.0
    champion = save_topology(model)

    print(f"\n{'Gen':>5}  {'Loss':>7}  {'Op':>4}  {'Typ':>4}  {'Res':>4}  "
          f"{'Tree':>5}  {'N':>5}  {'M':>2}")
    print("-" * 55)

    t0 = time.time()

    for gen in range(generations):
        avg_loss = 0.0
        for _ in range(adam_steps):
            b = generate_batch(rng, batch_size, max_val, max_depth)
            if b is None:
                continue
            loss, grads = loss_fn(model, b["ops"], b["arities"],
                                  b["c1_vals"], b["c1_types"],
                                  b["c2_vals"], b["c2_types"],
                                  b["c3_vals"], b["c3_types"],
                                  b["gt_results"], b["gt_types"])
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
            m = evaluate_batch(model, erng, 512, max_val, max_depth)

            # Tree-level (smaller sample — serial execution)
            trng = np.random.RandomState(seed + gen + 9000)
            tm = evaluate_trees(model, trng, 200, max_val, max_depth)

            if m["op_acc"] >= best_op:
                best_op = m["op_acc"]
                champion = save_topology(model)
                status = "✓"
            else:
                load_topology(model, champion)
                status = "✗"

            print(f"  {gen:4d}  {avg_loss:7.4f}  {m['op_acc']*100:3.0f}%  "
                  f"{m['type_acc']*100:3.0f}%  {m['result_acc']*100:3.0f}%  "
                  f"{tm['tree_acc']*100:4.1f}%  {m['n_nodes']:4d}   {status}")

            if m["op_acc"] >= 0.999 and tm["tree_acc"] >= 0.99:
                print(f"\n  🎯 Converged at gen {gen}!")
                break
        else:
            # Quick champion check
            qb = generate_batch(rng, 32, max_val, max_depth)
            if qb is not None:
                qo = model.forward(qb["ops"], qb["arities"],
                                   qb["c1_vals"], qb["c1_types"],
                                   qb["c2_vals"], qb["c2_types"],
                                   qb["c3_vals"], qb["c3_types"])
                mx.eval(qo["pred_op"])
                qa = (np.array(qo["pred_op"]) == np.array(qb["ops"])).mean()
                if qa >= best_op:
                    champion = save_topology(model)
                    best_op = max(best_op, qa)
                else:
                    load_topology(model, champion)

    t_total = time.time() - t0
    load_topology(model, champion)

    print(f"\n{'=' * 55}")
    print(f"  Training: {generations} gens, {t_total:.1f}s")

    # Final evaluation across scales
    print(f"\n  === Final Evaluation ===")
    for mv in [10, 50, 100]:
        for md in [2, 3, 4, 5]:
            frng = np.random.RandomState(seed + mv * 100 + md)
            fm = evaluate_batch(model, frng, 1024, mv, md)
            trng = np.random.RandomState(seed + mv * 100 + md + 50000)
            tm = evaluate_trees(model, trng, 500, mv, md)
            print(f"    val={mv:3d} d={md}  "
                  f"op={fm['op_acc']*100:5.1f}%  "
                  f"type={fm['type_acc']*100:5.1f}%  "
                  f"res={fm['result_acc']*100:5.1f}%  "
                  f"tree={tm['tree_acc']*100:5.1f}%  "
                  f"nodes={fm['n_nodes']}")

    # Per-op-category breakdown
    print(f"\n  === Per-Category Breakdown (val=10, depth=3) ===")
    frng = np.random.RandomState(seed + 77777)
    batch = generate_batch(frng, 2048, 10, 3)
    if batch is not None:
        out = model.forward(batch["ops"], batch["arities"],
                            batch["c1_vals"], batch["c1_types"],
                            batch["c2_vals"], batch["c2_types"],
                            batch["c3_vals"], batch["c3_types"])
        mx.eval(*out.values())
        po = np.array(out["pred_op"])
        go = np.array(batch["ops"])
        pr = np.array(out["pred_result"])
        gr = np.array(batch["gt_results"])

        categories = [
            ("Arith binary", BINARY_INT_OPS),
            ("Comparison", COMPARISON_OPS),
            ("Bool binary", BINARY_BOOL_OPS),
            ("Bool unary", UNARY_BOOL_OPS),
            ("Arith unary", UNARY_INT_OPS),
            ("Conditional", [Op.IF]),
        ]
        for name, ops in categories:
            mask = np.isin(go, [int(o) for o in ops])
            if mask.sum() == 0:
                continue
            op_acc = (po[mask] == go[mask]).mean()
            res_acc = (pr[mask] == gr[mask]).mean()
            print(f"    {name:<14s}: op={op_acc*100:5.1f}%  "
                  f"res={res_acc*100:5.1f}%  n={mask.sum()}")

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
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-mix", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    train(**vars(a))
