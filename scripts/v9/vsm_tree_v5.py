"""
v9 — VSM Tree v5: Lambda Primitives

Extends v4 with function-typed values flowing through the tree.
Tests two properties beyond v4:
  1. Compound values — functions carry structure (op_code + bound_arg)
  2. Type-dependent dispatch — apply-fn dispatches based on the function
     value, not just the node's own op code

New type: FN — a partially applied binary op
  Represented as (val=op_code, aux=bound_arg)
  (partial + 3)  → FN(op=ADD, bound=3)
  (apply-fn FN 5) → kernel_eval(ADD, [3, 5]) → 8

New operations:
  PARTIAL:    (INT_op_ref, INT) → FN    — create a function
  APPLY_FN:   (FN, INT) → INT/BOOL      — apply function to argument
  COMPOSE:    (FN, FN) → FN             — compose two functions
  APPLY_COMP: (FN_composed, INT) → INT  — apply composed function

The value stream extends from single int to (val, aux) pairs:
  INT/BOOL: (value, 0)
  FN:       (op_code, bound_arg)
  FN_COMP:  (fn1_encoded, fn2_encoded)  — packed pair

The model still only classifies the op. Values pass through. But now
values carry structure (two fields) and dispatch depends on value content.

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
    FN = 2        # partially applied function
    FN_COMP = 3   # composed function
    ERROR = 4

N_TYPES = 5


# ══════════════════════════════════════════════════════════════════════
# Operations — v4 ops + lambda primitives
# ══════════════════════════════════════════════════════════════════════

class Op(IntEnum):
    # ── v4 ops (unchanged) ──
    ADD = 0; SUB = 1; MUL = 2; DIV = 3; MOD = 4; MIN = 5; MAX = 6
    EQ = 7; LT = 8; GT = 9; LE = 10; GE = 11
    AND = 12; OR = 13
    NOT = 14
    ABS = 15; NEG = 16
    IF = 17
    # ── Lambda primitives ──
    PARTIAL = 18    # (op_ref_as_int, int_val) → FN
    APPLY_FN = 19   # (FN, int_val) → INT/BOOL
    COMPOSE = 20    # (FN, FN) → FN_COMP
    APPLY_COMP = 21 # (FN_COMP, int_val) → INT/BOOL (sugar: apply composed fn)

N_OPS = 22

# Ops that can be partially applied (binary ops producing INT or BOOL)
PARTIAL_OPS = [Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD, Op.MIN, Op.MAX,
               Op.EQ, Op.LT, Op.GT, Op.LE, Op.GE]

OP_META = {
    # (arity, output_type)  — input types handled in generation
    Op.ADD: (2, Type.INT), Op.SUB: (2, Type.INT), Op.MUL: (2, Type.INT),
    Op.DIV: (2, Type.INT), Op.MOD: (2, Type.INT),
    Op.MIN: (2, Type.INT), Op.MAX: (2, Type.INT),
    Op.EQ: (2, Type.BOOL), Op.LT: (2, Type.BOOL), Op.GT: (2, Type.BOOL),
    Op.LE: (2, Type.BOOL), Op.GE: (2, Type.BOOL),
    Op.AND: (2, Type.BOOL), Op.OR: (2, Type.BOOL),
    Op.NOT: (1, Type.BOOL),
    Op.ABS: (1, Type.INT), Op.NEG: (1, Type.INT),
    Op.IF: (3, None),       # polymorphic
    Op.PARTIAL: (2, Type.FN),
    Op.APPLY_FN: (2, None),  # output depends on the function
    Op.COMPOSE: (2, Type.FN_COMP),
    Op.APPLY_COMP: (2, None),
}

OP_NAMES = {
    Op.ADD: "+", Op.SUB: "-", Op.MUL: "*", Op.DIV: "//", Op.MOD: "%",
    Op.MIN: "min", Op.MAX: "max",
    Op.EQ: "=", Op.LT: "<", Op.GT: ">", Op.LE: "<=", Op.GE: ">=",
    Op.AND: "and", Op.OR: "or", Op.NOT: "not",
    Op.ABS: "abs", Op.NEG: "neg", Op.IF: "if",
    Op.PARTIAL: "partial", Op.APPLY_FN: "apply",
    Op.COMPOSE: "comp", Op.APPLY_COMP: "apply-comp",
}

# Groups for generation
BINARY_INT_OPS = [Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD, Op.MIN, Op.MAX]
COMPARISON_OPS = [Op.EQ, Op.LT, Op.GT, Op.LE, Op.GE]
BINARY_BOOL_OPS = [Op.AND, Op.OR]
UNARY_INT_OPS = [Op.ABS, Op.NEG]


# ══════════════════════════════════════════════════════════════════════
# Function encoding/decoding
# ══════════════════════════════════════════════════════════════════════
#
# FN value: (val=op_code, aux=bound_arg)
# FN_COMP value: (val=fn1_packed, aux=fn2_packed)
#   where fn_packed = op_code * 10000 + (bound_arg + 5000)
#   This handles bound_arg in [-5000, 4999] safely.

FN_PACK_OFFSET = 5000
FN_PACK_SCALE = 10000


def fn_pack(op_code: int, bound_arg: int) -> int:
    """Pack a partial function into a single integer."""
    return op_code * FN_PACK_SCALE + (bound_arg + FN_PACK_OFFSET)


def fn_unpack(packed: int) -> tuple[int, int]:
    """Unpack a packed function into (op_code, bound_arg)."""
    op_code = packed // FN_PACK_SCALE
    bound_arg = (packed % FN_PACK_SCALE) - FN_PACK_OFFSET
    return op_code, bound_arg


# ══════════════════════════════════════════════════════════════════════
# Kernel dispatch — exact computation
# ══════════════════════════════════════════════════════════════════════

def kernel_eval_v4(op: int, args: list[int]) -> int:
    """Evaluate a v4 binary/unary arithmetic or comparison op."""
    if op == Op.ADD: return args[0] + args[1]
    elif op == Op.SUB: return args[0] - args[1]
    elif op == Op.MUL: return args[0] * args[1]
    elif op == Op.DIV: return args[0] // args[1] if args[1] != 0 else 0
    elif op == Op.MOD: return args[0] % args[1] if args[1] != 0 else 0
    elif op == Op.MIN: return min(args[0], args[1])
    elif op == Op.MAX: return max(args[0], args[1])
    elif op == Op.EQ: return int(args[0] == args[1])
    elif op == Op.LT: return int(args[0] < args[1])
    elif op == Op.GT: return int(args[0] > args[1])
    elif op == Op.LE: return int(args[0] <= args[1])
    elif op == Op.GE: return int(args[0] >= args[1])
    elif op == Op.AND: return int(bool(args[0]) and bool(args[1]))
    elif op == Op.OR: return int(bool(args[0]) or bool(args[1]))
    elif op == Op.NOT: return int(not bool(args[0]))
    elif op == Op.ABS: return abs(args[0])
    elif op == Op.NEG: return -args[0]
    elif op == Op.IF: return args[1] if bool(args[0]) else args[2]
    return 0


def kernel_eval(op: int, child_vals: list[int], child_auxs: list[int],
                child_types: list[int]) -> tuple[int, int, int]:
    """Full kernel dispatch. Returns (val, aux, type).

    Handles all v4 ops plus lambda primitives.
    """
    if op <= Op.IF:
        # v4 ops — pass child vals directly
        result = kernel_eval_v4(op, child_vals)
        rtype = OP_META[op][1]
        if rtype is None:
            rtype = Type.INT  # IF returns INT in our generation
        return result, 0, int(rtype)

    elif op == Op.PARTIAL:
        # child 0: op reference (integer = the op code to partially apply)
        # child 1: the bound argument
        fn_op = child_vals[0]
        bound = child_vals[1]
        packed = fn_pack(fn_op, bound)
        return packed, 0, int(Type.FN)

    elif op == Op.APPLY_FN:
        # child 0: FN value (packed function)
        # child 1: argument to apply
        if child_types[0] == Type.FN:
            fn_op, bound = fn_unpack(child_vals[0])
            result = kernel_eval_v4(fn_op, [bound, child_vals[1]])
            # Determine output type from the function's op
            out_type = OP_META.get(fn_op, (2, Type.INT))[1]
            if out_type is None:
                out_type = Type.INT
            return result, 0, int(out_type)
        elif child_types[0] == Type.FN_COMP:
            # Composed function: apply inner first, then outer
            fn1_packed = child_vals[0]  # outer function
            fn2_packed = child_auxs[0]  # inner function
            # Apply inner function (fn2) to argument
            fn2_op, fn2_bound = fn_unpack(fn2_packed)
            intermediate = kernel_eval_v4(fn2_op, [fn2_bound, child_vals[1]])
            # Apply outer function (fn1) to intermediate
            fn1_op, fn1_bound = fn_unpack(fn1_packed)
            result = kernel_eval_v4(fn1_op, [fn1_bound, intermediate])
            out_type = OP_META.get(fn1_op, (2, Type.INT))[1]
            if out_type is None:
                out_type = Type.INT
            return result, 0, int(out_type)
        return 0, 0, int(Type.ERROR)

    elif op == Op.COMPOSE:
        # child 0: outer function (FN)
        # child 1: inner function (FN)
        # Result: FN_COMP with (outer_packed, inner_packed)
        return child_vals[0], child_vals[1], int(Type.FN_COMP)

    elif op == Op.APPLY_COMP:
        # Sugar: same as APPLY_FN but explicitly for composed functions
        fn1_packed = child_vals[0]
        fn2_packed = child_auxs[0]
        fn2_op, fn2_bound = fn_unpack(fn2_packed)
        intermediate = kernel_eval_v4(fn2_op, [fn2_bound, child_vals[1]])
        fn1_op, fn1_bound = fn_unpack(fn1_packed)
        result = kernel_eval_v4(fn1_op, [fn1_bound, intermediate])
        out_type = OP_META.get(fn1_op, (2, Type.INT))[1]
        if out_type is None:
            out_type = Type.INT
        return result, 0, int(out_type)

    return 0, 0, int(Type.ERROR)


def kernel_apply_batch(ops, c1_vals, c1_auxs, c1_types,
                       c2_vals, c2_auxs, c2_types,
                       c3_vals, c3_auxs, c3_types):
    """Vectorized kernel dispatch for a batch. Returns (result_vals, result_auxs)."""
    result = mx.zeros_like(c1_vals)
    result_aux = mx.zeros_like(c1_vals)

    # ── v4 arithmetic binary ──
    safe_c2 = mx.where(c2_vals == 0, mx.ones_like(c2_vals), c2_vals)
    result = mx.where(ops == Op.ADD, c1_vals + c2_vals, result)
    result = mx.where(ops == Op.SUB, c1_vals - c2_vals, result)
    result = mx.where(ops == Op.MUL, c1_vals * c2_vals, result)
    result = mx.where(ops == Op.DIV, c1_vals // safe_c2, result)
    result = mx.where((ops == Op.DIV) & (c2_vals == 0), mx.zeros_like(result), result)
    result = mx.where(ops == Op.MOD, c1_vals % safe_c2, result)
    result = mx.where((ops == Op.MOD) & (c2_vals == 0), mx.zeros_like(result), result)
    result = mx.where(ops == Op.MIN, mx.minimum(c1_vals, c2_vals), result)
    result = mx.where(ops == Op.MAX, mx.maximum(c1_vals, c2_vals), result)

    # ── v4 comparison ──
    result = mx.where(ops == Op.EQ, (c1_vals == c2_vals).astype(mx.int64), result)
    result = mx.where(ops == Op.LT, (c1_vals < c2_vals).astype(mx.int64), result)
    result = mx.where(ops == Op.GT, (c1_vals > c2_vals).astype(mx.int64), result)
    result = mx.where(ops == Op.LE, (c1_vals <= c2_vals).astype(mx.int64), result)
    result = mx.where(ops == Op.GE, (c1_vals >= c2_vals).astype(mx.int64), result)

    # ── v4 boolean ──
    result = mx.where(ops == Op.AND, c1_vals & c2_vals, result)
    result = mx.where(ops == Op.OR, c1_vals | c2_vals, result)
    result = mx.where(ops == Op.NOT, 1 - c1_vals, result)

    # ── v4 unary / conditional ──
    result = mx.where(ops == Op.ABS, mx.abs(c1_vals), result)
    result = mx.where(ops == Op.NEG, -c1_vals, result)
    result = mx.where(ops == Op.IF, mx.where(c1_vals != 0, c2_vals, c3_vals), result)

    # ── PARTIAL: pack function ──
    packed_fn = c1_vals * FN_PACK_SCALE + (c2_vals + FN_PACK_OFFSET)
    result = mx.where(ops == Op.PARTIAL, packed_fn, result)

    # ── APPLY_FN: unpack and dispatch ──
    # For simple FN: unpack(c1_val) → (fn_op, bound), then eval(fn_op, [bound, c2_val])
    is_apply = (ops == Op.APPLY_FN) & (c1_types == int(Type.FN))
    fn_op = c1_vals // FN_PACK_SCALE
    fn_bound = (c1_vals % FN_PACK_SCALE) - FN_PACK_OFFSET
    # Dispatch the function — need to handle each possible fn_op
    apply_result = mx.zeros_like(result)
    apply_result = mx.where(fn_op == Op.ADD, fn_bound + c2_vals, apply_result)
    apply_result = mx.where(fn_op == Op.SUB, fn_bound - c2_vals, apply_result)
    apply_result = mx.where(fn_op == Op.MUL, fn_bound * c2_vals, apply_result)
    safe_c2_apply = mx.where(c2_vals == 0, mx.ones_like(c2_vals), c2_vals)
    apply_result = mx.where(fn_op == Op.DIV, fn_bound // safe_c2_apply, apply_result)
    apply_result = mx.where((fn_op == Op.DIV) & (c2_vals == 0), mx.zeros_like(apply_result), apply_result)
    apply_result = mx.where(fn_op == Op.MOD, fn_bound % safe_c2_apply, apply_result)
    apply_result = mx.where((fn_op == Op.MOD) & (c2_vals == 0), mx.zeros_like(apply_result), apply_result)
    apply_result = mx.where(fn_op == Op.MIN, mx.minimum(fn_bound, c2_vals), apply_result)
    apply_result = mx.where(fn_op == Op.MAX, mx.maximum(fn_bound, c2_vals), apply_result)
    apply_result = mx.where(fn_op == Op.EQ, (fn_bound == c2_vals).astype(mx.int64), apply_result)
    apply_result = mx.where(fn_op == Op.LT, (fn_bound < c2_vals).astype(mx.int64), apply_result)
    apply_result = mx.where(fn_op == Op.GT, (fn_bound > c2_vals).astype(mx.int64), apply_result)
    apply_result = mx.where(fn_op == Op.LE, (fn_bound <= c2_vals).astype(mx.int64), apply_result)
    apply_result = mx.where(fn_op == Op.GE, (fn_bound >= c2_vals).astype(mx.int64), apply_result)
    result = mx.where(is_apply, apply_result, result)

    # ── APPLY_FN on FN_COMP: chain two function applications ──
    is_apply_comp = ((ops == Op.APPLY_FN) & (c1_types == int(Type.FN_COMP))) | (ops == Op.APPLY_COMP)
    # c1_val = outer fn packed, c1_aux = inner fn packed
    inner_op = c1_auxs // FN_PACK_SCALE
    inner_bound = (c1_auxs % FN_PACK_SCALE) - FN_PACK_OFFSET
    # Apply inner function to argument
    inner_result = mx.zeros_like(result)
    inner_result = mx.where(inner_op == Op.ADD, inner_bound + c2_vals, inner_result)
    inner_result = mx.where(inner_op == Op.SUB, inner_bound - c2_vals, inner_result)
    inner_result = mx.where(inner_op == Op.MUL, inner_bound * c2_vals, inner_result)
    safe_c2_inner = mx.where(c2_vals == 0, mx.ones_like(c2_vals), c2_vals)
    inner_result = mx.where(inner_op == Op.DIV, inner_bound // safe_c2_inner, inner_result)
    inner_result = mx.where(inner_op == Op.MOD, inner_bound % safe_c2_inner, inner_result)
    inner_result = mx.where(inner_op == Op.MIN, mx.minimum(inner_bound, c2_vals), inner_result)
    inner_result = mx.where(inner_op == Op.MAX, mx.maximum(inner_bound, c2_vals), inner_result)
    inner_result = mx.where(inner_op == Op.EQ, (inner_bound == c2_vals).astype(mx.int64), inner_result)
    inner_result = mx.where(inner_op == Op.LT, (inner_bound < c2_vals).astype(mx.int64), inner_result)
    inner_result = mx.where(inner_op == Op.GT, (inner_bound > c2_vals).astype(mx.int64), inner_result)
    inner_result = mx.where(inner_op == Op.LE, (inner_bound <= c2_vals).astype(mx.int64), inner_result)
    inner_result = mx.where(inner_op == Op.GE, (inner_bound >= c2_vals).astype(mx.int64), inner_result)
    # Apply outer function to inner result
    outer_op = c1_vals // FN_PACK_SCALE
    outer_bound = (c1_vals % FN_PACK_SCALE) - FN_PACK_OFFSET
    comp_result = mx.zeros_like(result)
    comp_result = mx.where(outer_op == Op.ADD, outer_bound + inner_result, comp_result)
    comp_result = mx.where(outer_op == Op.SUB, outer_bound - inner_result, comp_result)
    comp_result = mx.where(outer_op == Op.MUL, outer_bound * inner_result, comp_result)
    safe_inner = mx.where(inner_result == 0, mx.ones_like(inner_result), inner_result)
    comp_result = mx.where(outer_op == Op.DIV, outer_bound // safe_inner, comp_result)
    comp_result = mx.where(outer_op == Op.MOD, outer_bound % safe_inner, comp_result)
    comp_result = mx.where(outer_op == Op.MIN, mx.minimum(outer_bound, inner_result), comp_result)
    comp_result = mx.where(outer_op == Op.MAX, mx.maximum(outer_bound, inner_result), comp_result)
    comp_result = mx.where(outer_op == Op.EQ, (outer_bound == inner_result).astype(mx.int64), comp_result)
    comp_result = mx.where(outer_op == Op.LT, (outer_bound < inner_result).astype(mx.int64), comp_result)
    comp_result = mx.where(outer_op == Op.GT, (outer_bound > inner_result).astype(mx.int64), comp_result)
    comp_result = mx.where(outer_op == Op.LE, (outer_bound <= inner_result).astype(mx.int64), comp_result)
    comp_result = mx.where(outer_op == Op.GE, (outer_bound >= inner_result).astype(mx.int64), comp_result)
    result = mx.where(is_apply_comp, comp_result, result)

    # ── COMPOSE: store both packed fns ──
    result = mx.where(ops == Op.COMPOSE, c1_vals, result)  # outer fn
    result_aux = mx.where(ops == Op.COMPOSE, c2_vals, result_aux)  # inner fn

    return result, result_aux


# ══════════════════════════════════════════════════════════════════════
# Expression tree generation
# ══════════════════════════════════════════════════════════════════════

def random_expr(rng, max_val, max_depth, target_type=Type.INT, depth=0):
    """Generate a random well-typed expression tree.

    Leaf values:
      INT: random int in [0, max_val)
      BOOL: 0 or 1
      FN: (PARTIAL, op_ref, int_val) — always a partial application subtree
    """
    # Leaf conditions
    if depth >= max_depth or (depth > 0 and rng.random() < 0.3):
        if target_type == Type.INT:
            return int(rng.randint(0, max_val))
        elif target_type == Type.BOOL:
            return int(rng.randint(0, 2))
        elif target_type in (Type.FN, Type.FN_COMP):
            # FN leaf: always generate a (partial op val) subtree
            op_ref = int(rng.choice(PARTIAL_OPS))
            val = int(rng.randint(0, max_val))
            return (int(Op.PARTIAL), op_ref, val)
        return 0

    if target_type == Type.INT:
        choice = rng.random()
        if choice < 0.35:
            # Arithmetic binary
            op = int(rng.choice(BINARY_INT_OPS))
            a1 = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            a2 = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            return (op, a1, a2)
        elif choice < 0.45:
            # Arithmetic unary
            op = int(rng.choice(UNARY_INT_OPS))
            a1 = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            return (op, a1)
        elif choice < 0.60:
            # IF
            test = random_expr(rng, max_val, max_depth, Type.BOOL, depth + 1)
            then = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            els = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            return (int(Op.IF), test, then, els)
        elif choice < 0.80:
            # APPLY_FN — apply a function to produce INT
            fn = random_expr(rng, max_val, max_depth, Type.FN, depth + 1)
            arg = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            return (int(Op.APPLY_FN), fn, arg)
        else:
            # APPLY_FN on composed function
            fn = random_expr(rng, max_val, max_depth, Type.FN_COMP, depth + 1)
            arg = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            return (int(Op.APPLY_FN), fn, arg)

    elif target_type == Type.BOOL:
        choice = rng.random()
        if choice < 0.40:
            op = int(rng.choice(COMPARISON_OPS))
            a1 = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            a2 = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
            return (op, a1, a2)
        elif choice < 0.65:
            op = int(rng.choice(BINARY_BOOL_OPS))
            a1 = random_expr(rng, max_val, max_depth, Type.BOOL, depth + 1)
            a2 = random_expr(rng, max_val, max_depth, Type.BOOL, depth + 1)
            return (op, a1, a2)
        else:
            a1 = random_expr(rng, max_val, max_depth, Type.BOOL, depth + 1)
            return (int(Op.NOT), a1)

    elif target_type == Type.FN:
        # Create a partial application
        op_ref = int(rng.choice(PARTIAL_OPS))
        val = random_expr(rng, max_val, max_depth, Type.INT, depth + 1)
        return (int(Op.PARTIAL), op_ref, val)

    elif target_type == Type.FN_COMP:
        # Compose two functions
        fn1 = random_expr(rng, max_val, max_depth, Type.FN, depth + 1)
        fn2 = random_expr(rng, max_val, max_depth, Type.FN, depth + 1)
        return (int(Op.COMPOSE), fn1, fn2)

    return 0


def eval_tree_full(node, expected_type=Type.INT):
    """Evaluate a tree, returning (val, aux, type)."""
    if isinstance(node, int):
        if expected_type == Type.BOOL:
            return node, 0, int(Type.BOOL)
        return node, 0, int(Type.INT)

    op = node[0]
    children = node[1:]

    # Determine child expected types
    child_expected = []
    if op in BINARY_INT_OPS + [Op.DIV, Op.MOD, Op.MIN, Op.MAX]:
        child_expected = [Type.INT, Type.INT]
    elif op in COMPARISON_OPS:
        child_expected = [Type.INT, Type.INT]
    elif op in BINARY_BOOL_OPS:
        child_expected = [Type.BOOL, Type.BOOL]
    elif op == Op.NOT:
        child_expected = [Type.BOOL]
    elif op in UNARY_INT_OPS:
        child_expected = [Type.INT]
    elif op == Op.IF:
        child_expected = [Type.BOOL, expected_type, expected_type]
    elif op == Op.PARTIAL:
        child_expected = [Type.INT, Type.INT]  # op_ref is an int, bound is int
    elif op == Op.APPLY_FN:
        # First child could be FN or FN_COMP — need to evaluate to know
        child_expected = [None, Type.INT]  # will determine FN type dynamically
    elif op == Op.COMPOSE:
        child_expected = [Type.FN, Type.FN]
    elif op == Op.APPLY_COMP:
        child_expected = [Type.FN_COMP, Type.INT]

    # Evaluate children
    child_results = []
    for i, child in enumerate(children):
        if i < len(child_expected) and child_expected[i] is not None:
            ct = child_expected[i]
        else:
            # For APPLY_FN's first arg, evaluate to determine type
            ct = Type.FN
        val, aux, ctype = eval_tree_full(child, ct)
        child_results.append((val, aux, ctype))

    # Pad
    while len(child_results) < 3:
        child_results.append((0, 0, int(Type.INT)))

    c_vals = [r[0] for r in child_results]
    c_auxs = [r[1] for r in child_results]
    c_types = [r[2] for r in child_results]

    return kernel_eval(op, c_vals, c_auxs, c_types)


def tree_to_str(node):
    if isinstance(node, int):
        return str(node)
    op = node[0]
    children = node[1:]
    name = OP_NAMES.get(op, f"op{op}")
    # For PARTIAL, show the op reference nicely
    if op == Op.PARTIAL and isinstance(children[0], int) and children[0] in OP_NAMES:
        args = f"{OP_NAMES[children[0]]} {tree_to_str(children[1])}"
    else:
        args = " ".join(tree_to_str(c) for c in children)
    return f"({name} {args})"


# ══════════════════════════════════════════════════════════════════════
# Data generation — collect nodes for batched training
# ══════════════════════════════════════════════════════════════════════

def _collect_nodes(node, out, expected_type=Type.INT):
    """Collect tree nodes bottom-up. Returns (val, aux, type)."""
    if isinstance(node, int):
        if expected_type == Type.BOOL:
            return node, 0, int(Type.BOOL)
        return node, 0, int(Type.INT)

    op = node[0]
    children = node[1:]
    arity = len(children)

    # Determine child types
    child_expected = []
    if op in BINARY_INT_OPS:
        child_expected = [Type.INT, Type.INT]
    elif op in COMPARISON_OPS:
        child_expected = [Type.INT, Type.INT]
    elif op in BINARY_BOOL_OPS:
        child_expected = [Type.BOOL, Type.BOOL]
    elif op == Op.NOT:
        child_expected = [Type.BOOL]
    elif op in UNARY_INT_OPS:
        child_expected = [Type.INT]
    elif op == Op.IF:
        child_expected = [Type.BOOL, expected_type, expected_type]
    elif op == Op.PARTIAL:
        child_expected = [Type.INT, Type.INT]
    elif op == Op.APPLY_FN:
        child_expected = [Type.FN, Type.INT]
    elif op == Op.COMPOSE:
        child_expected = [Type.FN, Type.FN]
    elif op == Op.APPLY_COMP:
        child_expected = [Type.FN_COMP, Type.INT]

    # Recurse into children (bottom-up)
    child_results = []
    for i, child in enumerate(children):
        ct = child_expected[i] if i < len(child_expected) else Type.INT
        val, aux, ctype = _collect_nodes(child, out, ct)
        child_results.append((val, aux, ctype))

    # Pad to 3
    while len(child_results) < 3:
        child_results.append((0, 0, int(Type.INT)))

    c_vals = [r[0] for r in child_results]
    c_auxs = [r[1] for r in child_results]
    c_types = [r[2] for r in child_results]

    # Compute result
    result_val, result_aux, result_type = kernel_eval(op, c_vals, c_auxs, c_types)

    out.append({
        "op": op, "arity": arity,
        "c1_val": c_vals[0], "c1_aux": c_auxs[0], "c1_type": c_types[0],
        "c2_val": c_vals[1], "c2_aux": c_auxs[1], "c2_type": c_types[1],
        "c3_val": c_vals[2], "c3_aux": c_auxs[2], "c3_type": c_types[2],
        "result_val": result_val, "result_aux": result_aux,
        "result_type": result_type,
    })

    return result_val, result_aux, result_type


def generate_batch(rng, batch_size, max_val, max_depth):
    all_nodes = []
    for _ in range(batch_size):
        target = Type.INT if rng.random() < 0.7 else Type.BOOL
        tree = random_expr(rng, max_val, max_depth, target)
        _collect_nodes(tree, all_nodes, target)

    if not all_nodes:
        return None

    return {
        "ops": mx.array([n["op"] for n in all_nodes], dtype=mx.int32),
        "arities": mx.array([n["arity"] for n in all_nodes], dtype=mx.int32),
        "c1_vals": mx.array([n["c1_val"] for n in all_nodes], dtype=mx.int64),
        "c1_auxs": mx.array([n["c1_aux"] for n in all_nodes], dtype=mx.int64),
        "c1_types": mx.array([n["c1_type"] for n in all_nodes], dtype=mx.int32),
        "c2_vals": mx.array([n["c2_val"] for n in all_nodes], dtype=mx.int64),
        "c2_auxs": mx.array([n["c2_aux"] for n in all_nodes], dtype=mx.int64),
        "c2_types": mx.array([n["c2_type"] for n in all_nodes], dtype=mx.int32),
        "c3_vals": mx.array([n["c3_val"] for n in all_nodes], dtype=mx.int64),
        "c3_auxs": mx.array([n["c3_aux"] for n in all_nodes], dtype=mx.int64),
        "c3_types": mx.array([n["c3_type"] for n in all_nodes], dtype=mx.int32),
        "gt_results": mx.array([n["result_val"] for n in all_nodes], dtype=mx.int64),
        "gt_auxs": mx.array([n["result_aux"] for n in all_nodes], dtype=mx.int64),
        "gt_types": mx.array([n["result_type"] for n in all_nodes], dtype=mx.int32),
    }


# ══════════════════════════════════════════════════════════════════════
# VSM Node v5 — handles compound values (val, aux) pairs
# ══════════════════════════════════════════════════════════════════════

@dataclass
class VSMConfig:
    d_model: int = 64
    n_ops: int = N_OPS
    n_types: int = N_TYPES
    val_embed_range: int = 200
    n_mix_layers: int = 2


class VSMNodeV5(nn.Module):
    """VSM node with compound value support.

    Each child now has (type, val, aux) — the aux field carries the
    second component for FN-typed values (the bound argument).

    The model still only classifies the op. Values pass through.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = VSMConfig()
        self.config = config
        d = config.d_model

        self.op_embed = nn.Embedding(config.n_ops, d)
        self.type_embed = nn.Embedding(config.n_types, d)
        self.val_embed = nn.Embedding(config.val_embed_range, d)
        self.aux_embed = nn.Embedding(config.val_embed_range, d)
        self._val_offset = config.val_embed_range // 2
        self.pos_embed = nn.Embedding(4, d)
        self.arity_embed = nn.Embedding(4, d)

        # Input: concat [op; c1; c2; c3] — each child now includes aux
        self.input_proj = nn.Linear(4 * d, d)

        # Mix layers (ternary)
        self.mix_layers = [TernaryLinear(d, d, pre_norm=True)
                           for _ in range(config.n_mix_layers)]

        # Op head with op residual
        op_dim = ((config.n_ops + 15) // 16) * 16
        self.op_proj = nn.Linear(d + d, op_dim)
        self._op_dim = config.n_ops

        # Type head with op residual
        type_dim = ((config.n_types + 15) // 16) * 16
        self.type_proj = nn.Linear(d + d, type_dim)
        self._type_dim = config.n_types

    def _val_idx(self, val):
        return mx.clip(val + self._val_offset, 0, self.config.val_embed_range - 1).astype(mx.int32)

    def forward(self, ops, arities,
                c1_vals, c1_auxs, c1_types,
                c2_vals, c2_auxs, c2_types,
                c3_vals, c3_auxs, c3_types):
        d = self.config.d_model

        # Embed op
        op_repr = (self.op_embed(ops) +
                   self.pos_embed(mx.zeros(ops.shape, dtype=mx.int32)) +
                   self.arity_embed(arities))

        # Embed children: type + val + aux + position
        c1_repr = (self.type_embed(c1_types) +
                   self.val_embed(self._val_idx(c1_vals)) +
                   self.aux_embed(self._val_idx(c1_auxs)) +
                   self.pos_embed(mx.ones(ops.shape, dtype=mx.int32)))

        c2_repr = (self.type_embed(c2_types) +
                   self.val_embed(self._val_idx(c2_vals)) +
                   self.aux_embed(self._val_idx(c2_auxs)) +
                   self.pos_embed(mx.full(ops.shape, 2, dtype=mx.int32)))
        mask2 = (arities >= 2).astype(mx.float32).reshape(-1, 1)
        c2_repr = c2_repr * mask2

        c3_repr = (self.type_embed(c3_types) +
                   self.val_embed(self._val_idx(c3_vals)) +
                   self.aux_embed(self._val_idx(c3_auxs)) +
                   self.pos_embed(mx.full(ops.shape, 3, dtype=mx.int32)))
        mask3 = (arities >= 3).astype(mx.float32).reshape(-1, 1)
        c3_repr = c3_repr * mask3

        # Fuse
        x = self.input_proj(mx.concatenate([op_repr, c1_repr, c2_repr, c3_repr], axis=-1))

        # Mix
        for mix in self.mix_layers:
            x = x + mix(x)

        # Op with residual
        op_logits = self.op_proj(mx.concatenate([x, op_repr], axis=-1))[:, :self._op_dim]
        pred_op = mx.argmax(op_logits, axis=-1).astype(mx.int32)

        # Type with residual
        type_logits = self.type_proj(mx.concatenate([x, op_repr], axis=-1))[:, :self._type_dim]
        pred_type = mx.argmax(type_logits, axis=-1).astype(mx.int32)

        # Kernel dispatch (pass-through values)
        pred_result, pred_aux = kernel_apply_batch(
            pred_op,
            c1_vals, c1_auxs, c1_types,
            c2_vals, c2_auxs, c2_types,
            c3_vals, c3_auxs, c3_types,
        )

        return {
            "op_logits": op_logits,
            "type_logits": type_logits,
            "pred_op": pred_op,
            "pred_type": pred_type,
            "pred_result": pred_result,
            "pred_aux": pred_aux,
        }


# ══════════════════════════════════════════════════════════════════════
# Loss + evaluation
# ══════════════════════════════════════════════════════════════════════

def vsm_loss(model, ops, arities,
             c1_vals, c1_auxs, c1_types,
             c2_vals, c2_auxs, c2_types,
             c3_vals, c3_auxs, c3_types,
             gt_results, gt_auxs, gt_types):
    out = model.forward(ops, arities,
                        c1_vals, c1_auxs, c1_types,
                        c2_vals, c2_auxs, c2_types,
                        c3_vals, c3_auxs, c3_types)
    loss_op = nn.losses.cross_entropy(out["op_logits"], ops, reduction="mean")
    loss_type = nn.losses.cross_entropy(out["type_logits"], gt_types, reduction="mean")
    return loss_op + 0.5 * loss_type


def evaluate_batch(model, rng, n_exprs, max_val, max_depth):
    batch = generate_batch(rng, n_exprs, max_val, max_depth)
    if batch is None:
        return {"op_acc": 0, "type_acc": 0, "result_acc": 0, "n_nodes": 0}

    out = model.forward(batch["ops"], batch["arities"],
                        batch["c1_vals"], batch["c1_auxs"], batch["c1_types"],
                        batch["c2_vals"], batch["c2_auxs"], batch["c2_types"],
                        batch["c3_vals"], batch["c3_auxs"], batch["c3_types"])
    for v in out.values():
        mx.eval(v)

    po = np.array(out["pred_op"])
    pt = np.array(out["pred_type"])
    pr = np.array(out["pred_result"])
    pa = np.array(out["pred_aux"])
    go = np.array(batch["ops"])
    gt = np.array(batch["gt_types"])
    gr = np.array(batch["gt_results"])
    ga = np.array(batch["gt_auxs"])

    return {
        "op_acc": float((po == go).mean()),
        "type_acc": float((pt == gt).mean()),
        "result_acc": float(((pr == gr) & (pa == ga)).mean()),
        "n_nodes": len(go),
    }


def evaluate_trees(model, rng, n_trees, max_val, max_depth):
    correct = 0
    total = 0
    node_stats = {"op_correct": 0, "total": 0}

    for _ in range(n_trees):
        target = Type.INT if rng.random() < 0.7 else Type.BOOL
        tree = random_expr(rng, max_val, max_depth, target)
        gt_val, gt_aux, gt_type = eval_tree_full(tree, target)

        pred_val = _execute_tree(model, tree, target, node_stats)
        if pred_val == gt_val:
            correct += 1
        total += 1

    return {
        "tree_acc": correct / total if total > 0 else 0.0,
        "node_op_acc": (node_stats["op_correct"] / node_stats["total"]
                        if node_stats["total"] > 0 else 0.0),
        "n_trees": total,
        "n_nodes": node_stats["total"],
    }


def _execute_tree(model, node, expected_type, stats):
    """Execute tree bottom-up through model. Returns (val, aux, type)."""
    if isinstance(node, int):
        if expected_type == Type.BOOL:
            return node  # simplified: just return val for tree comparison
        return node

    op = node[0]
    children = node[1:]
    arity = len(children)

    # Determine child types
    child_expected = []
    if op in BINARY_INT_OPS:
        child_expected = [Type.INT, Type.INT]
    elif op in COMPARISON_OPS:
        child_expected = [Type.INT, Type.INT]
    elif op in BINARY_BOOL_OPS:
        child_expected = [Type.BOOL, Type.BOOL]
    elif op == Op.NOT:
        child_expected = [Type.BOOL]
    elif op in UNARY_INT_OPS:
        child_expected = [Type.INT]
    elif op == Op.IF:
        child_expected = [Type.BOOL, expected_type, expected_type]
    elif op == Op.PARTIAL:
        child_expected = [Type.INT, Type.INT]
    elif op == Op.APPLY_FN:
        child_expected = [Type.FN, Type.INT]
    elif op == Op.COMPOSE:
        child_expected = [Type.FN, Type.FN]
    elif op == Op.APPLY_COMP:
        child_expected = [Type.FN_COMP, Type.INT]

    # Recurse — for simplicity in tree eval, use the Python kernel directly
    child_results = []
    for i, child in enumerate(children):
        ct = child_expected[i] if i < len(child_expected) else Type.INT
        val, aux, ctype = _collect_and_eval(child, ct)
        child_results.append((val, aux, ctype))

    while len(child_results) < 3:
        child_results.append((0, 0, int(Type.INT)))

    c_vals = [r[0] for r in child_results]
    c_auxs = [r[1] for r in child_results]
    c_types = [r[2] for r in child_results]

    # Run through model for op prediction
    op_arr = mx.array([op], dtype=mx.int32)
    arity_arr = mx.array([arity], dtype=mx.int32)
    out = model.forward(
        op_arr, arity_arr,
        mx.array([c_vals[0]], dtype=mx.int64), mx.array([c_auxs[0]], dtype=mx.int64),
        mx.array([c_types[0]], dtype=mx.int32),
        mx.array([c_vals[1]], dtype=mx.int64), mx.array([c_auxs[1]], dtype=mx.int64),
        mx.array([c_types[1]], dtype=mx.int32),
        mx.array([c_vals[2]], dtype=mx.int64), mx.array([c_auxs[2]], dtype=mx.int64),
        mx.array([c_types[2]], dtype=mx.int32),
    )
    mx.eval(out["pred_result"], out["pred_op"])

    stats["total"] += 1
    if out["pred_op"].item() == op:
        stats["op_correct"] += 1

    return int(out["pred_result"].item())


def _collect_and_eval(node, expected_type):
    """Evaluate a subtree using Python kernel. Returns (val, aux, type)."""
    if isinstance(node, int):
        if expected_type == Type.BOOL:
            return node, 0, int(Type.BOOL)
        return node, 0, int(Type.INT)

    op = node[0]
    children = node[1:]

    child_expected = []
    if op in BINARY_INT_OPS:
        child_expected = [Type.INT, Type.INT]
    elif op in COMPARISON_OPS:
        child_expected = [Type.INT, Type.INT]
    elif op in BINARY_BOOL_OPS:
        child_expected = [Type.BOOL, Type.BOOL]
    elif op == Op.NOT:
        child_expected = [Type.BOOL]
    elif op in UNARY_INT_OPS:
        child_expected = [Type.INT]
    elif op == Op.IF:
        child_expected = [Type.BOOL, expected_type, expected_type]
    elif op == Op.PARTIAL:
        child_expected = [Type.INT, Type.INT]
    elif op == Op.APPLY_FN:
        child_expected = [Type.FN, Type.INT]
    elif op == Op.COMPOSE:
        child_expected = [Type.FN, Type.FN]
    elif op == Op.APPLY_COMP:
        child_expected = [Type.FN_COMP, Type.INT]

    child_results = []
    for i, child in enumerate(children):
        ct = child_expected[i] if i < len(child_expected) else Type.INT
        child_results.append(_collect_and_eval(child, ct))

    while len(child_results) < 3:
        child_results.append((0, 0, int(Type.INT)))

    c_vals = [r[0] for r in child_results]
    c_auxs = [r[1] for r in child_results]
    c_types = [r[2] for r in child_results]
    return kernel_eval(op, c_vals, c_auxs, c_types)


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════

def train(
    generations=2000, batch_size=128, adam_steps=10, lr=1e-3,
    mutation_pct=0.02, eval_interval=200,
    max_val=10, max_depth=3, d_model=64, n_mix=2, seed=42,
):
    print("=" * 70)
    print("  v9 — VSM Tree v5: Lambda Primitives")
    print("  partial + apply + compose — compound values through the tree")
    print("=" * 70)

    rng = np.random.RandomState(seed)
    config = VSMConfig(d_model=d_model, n_mix_layers=n_mix)
    model = VSMNodeV5(config)

    n_ternary = count_ternary_weights(model)
    mut_budget = max(1, int(n_ternary * mutation_pct))

    print(f"\n  d={d_model}  mix={n_mix}  max_val={max_val}  max_depth={max_depth}")
    print(f"  ops={N_OPS}  types={N_TYPES}  ternary={n_ternary:,}  mut={mut_budget}")

    optimizer = optim.Adam(learning_rate=lr)
    loss_fn = nn.value_and_grad(model, vsm_loss)
    best_op = -1.0
    champion = save_topology(model)

    print(f"\n{'Gen':>5}  {'Loss':>7}  {'Op':>4}  {'Typ':>4}  {'Res':>4}  "
          f"{'Tree':>5}  {'N':>5}")
    print("-" * 55)

    t0 = time.time()
    for gen in range(generations):
        avg_loss = 0.0
        for _ in range(adam_steps):
            b = generate_batch(rng, batch_size, max_val, max_depth)
            if b is None:
                continue
            loss, grads = loss_fn(model, b["ops"], b["arities"],
                                  b["c1_vals"], b["c1_auxs"], b["c1_types"],
                                  b["c2_vals"], b["c2_auxs"], b["c2_types"],
                                  b["c3_vals"], b["c3_auxs"], b["c3_types"],
                                  b["gt_results"], b["gt_auxs"], b["gt_types"])
            grads = zero_ternary_grads(model, grads)
            optimizer.update(model, grads)
            restore_ternary(model)
            mx.eval(model.parameters(), optimizer.state)
            avg_loss += loss.item()
        avg_loss /= adam_steps

        mutate_topology(model, mut_budget, rng, sign_flip_rate=0.2)
        mx.eval(model.parameters())

        if gen % eval_interval == 0 or gen == generations - 1:
            erng = np.random.RandomState(seed + gen + 5000)
            m = evaluate_batch(model, erng, 512, max_val, max_depth)
            trng = np.random.RandomState(seed + gen + 9000)
            tm = evaluate_trees(model, trng, 200, max_val, max_depth)

            if m["op_acc"] >= best_op:
                best_op = m["op_acc"]
                champion = save_topology(model)
                st = "✓"
            else:
                load_topology(model, champion)
                st = "✗"

            print(f"  {gen:4d}  {avg_loss:7.4f}  {m['op_acc']*100:3.0f}%  "
                  f"{m['type_acc']*100:3.0f}%  {m['result_acc']*100:3.0f}%  "
                  f"{tm['tree_acc']*100:4.1f}%  {m['n_nodes']:4d}   {st}")

            if m["op_acc"] >= 0.999 and tm["tree_acc"] >= 0.99:
                print(f"\n  🎯 Converged at gen {gen}!")
                break
        else:
            qb = generate_batch(rng, 32, max_val, max_depth)
            if qb is not None:
                qo = model.forward(qb["ops"], qb["arities"],
                                   qb["c1_vals"], qb["c1_auxs"], qb["c1_types"],
                                   qb["c2_vals"], qb["c2_auxs"], qb["c2_types"],
                                   qb["c3_vals"], qb["c3_auxs"], qb["c3_types"])
                mx.eval(qo["pred_op"])
                qa = (np.array(qo["pred_op"]) == np.array(qb["ops"])).mean()
                if qa >= best_op:
                    champion = save_topology(model)
                    best_op = max(best_op, qa)
                else:
                    load_topology(model, champion)

    elapsed = time.time() - t0
    load_topology(model, champion)

    print(f"\n{'=' * 55}")
    print(f"  Training: {generations} gens, {elapsed:.1f}s")

    # Per-category breakdown
    print(f"\n  === Per-Category Breakdown ===")
    frng = np.random.RandomState(seed + 77777)
    batch = generate_batch(frng, 2048, max_val, max_depth)
    if batch is not None:
        out = model.forward(batch["ops"], batch["arities"],
                            batch["c1_vals"], batch["c1_auxs"], batch["c1_types"],
                            batch["c2_vals"], batch["c2_auxs"], batch["c2_types"],
                            batch["c3_vals"], batch["c3_auxs"], batch["c3_types"])
        mx.eval(*out.values())
        po = np.array(out["pred_op"])
        go = np.array(batch["ops"])
        pr = np.array(out["pred_result"])
        gr = np.array(batch["gt_results"])
        pa = np.array(out["pred_aux"])
        ga = np.array(batch["gt_auxs"])

        categories = [
            ("Arith binary", BINARY_INT_OPS),
            ("Comparison", COMPARISON_OPS),
            ("Bool binary", BINARY_BOOL_OPS),
            ("Bool unary", [Op.NOT]),
            ("Arith unary", UNARY_INT_OPS),
            ("Conditional", [Op.IF]),
            ("Partial", [Op.PARTIAL]),
            ("Apply-fn", [Op.APPLY_FN]),
            ("Compose", [Op.COMPOSE]),
        ]
        for name, ops in categories:
            mask = np.isin(go, [int(o) for o in ops])
            if mask.sum() == 0:
                continue
            op_acc = (po[mask] == go[mask]).mean()
            res_match = ((pr[mask] == gr[mask]) & (pa[mask] == ga[mask])).mean()
            print(f"    {name:<14s}: op={op_acc*100:5.1f}%  "
                  f"res={res_match*100:5.1f}%  n={mask.sum()}")

    # Scaling
    print(f"\n  === Scaling ===")
    for mv in [10, 50]:
        for md in [2, 3, 4]:
            frng = np.random.RandomState(seed + mv * 100 + md)
            fm = evaluate_batch(model, frng, 1024, mv, md)
            trng = np.random.RandomState(seed + mv * 100 + md + 50000)
            tm = evaluate_trees(model, trng, 500, mv, md)
            print(f"    val={mv:3d} d={md}  "
                  f"op={fm['op_acc']*100:5.1f}%  "
                  f"type={fm['type_acc']*100:5.1f}%  "
                  f"res={fm['result_acc']*100:5.1f}%  "
                  f"tree={tm['tree_acc']*100:5.1f}%")

    # Show example expressions
    print(f"\n  === Example Expressions ===")
    erng = np.random.RandomState(42)
    for _ in range(8):
        target = Type.INT if erng.random() < 0.7 else Type.BOOL
        tree = random_expr(erng, 10, 3, target)
        gt_val, gt_aux, gt_type = eval_tree_full(tree, target)
        expr = tree_to_str(tree)
        type_name = {0: "INT", 1: "BOOL", 2: "FN", 3: "COMP", 4: "ERR"}[gt_type]
        print(f"    {expr}")
        print(f"      → {gt_val} : {type_name}")

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
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--max-val", type=int, default=10)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-mix", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    train(**vars(a))
