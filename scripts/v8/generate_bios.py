#!/usr/bin/env python3
"""Generate BIOS flash training data — math + clojure.core expressions.

Single-representation examples to force computation every time.
Burns arithmetic and functional programming circuits into the model's
deepest levels through extreme repetition on a small, curated dataset.

Each example is ONE randomly-chosen notation:
  - Raw math:  347 + 289 = 636
  - S-expr:    (+ 347 289) → 636
  - Lambda:    (λx. λy. (+ x y) 347 289) → 636

All results verified by Python eval. No hallucinated answers.

Usage:
    cd ~/src/verbum
    uv run python scripts/v8/generate_bios.py                    # generate + print stats
    uv run python scripts/v8/generate_bios.py --pack             # generate + pack into shards
    uv run python scripts/v8/generate_bios.py --count 100 --seed 42  # small test run
"""

from __future__ import annotations

import argparse
import json
import math
import operator
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# ═══════════════════════════════════════════════════════════════════
# Expression types
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Example:
    """A single training example."""
    text: str           # the formatted example string
    notation: str       # "raw", "sexpr", or "lambda"
    tier: int           # 1, 2, or 3 (math) or 0 (clojure)
    category: str       # e.g. "arithmetic", "comparison", "sequence"
    verified: bool = True


# ═══════════════════════════════════════════════════════════════════
# Math generators — Tier 1 (single operation)
# ═══════════════════════════════════════════════════════════════════

# Operand ranges by "difficulty"
RANGES = {
    1: (0, 9),         # single digit
    2: (0, 99),        # two digit
    3: (0, 999),       # three digit
    4: (0, 9999),      # four digit
}


def rand_int(rng: random.Random, digits: int = 0) -> int:
    """Random integer. If digits=0, pick a random digit count."""
    if digits == 0:
        digits = rng.choice([1, 1, 1, 2, 2, 3, 4])  # bias toward small
    lo, hi = RANGES[digits]
    return rng.randint(lo, hi)


def rand_positive(rng: random.Random, digits: int = 0) -> int:
    """Random positive integer (≥1)."""
    return max(1, rand_int(rng, digits))


def rand_bool(rng: random.Random) -> bool:
    return rng.choice([True, False])


# ── Arithmetic ────────────────────────────────────────────────────

ARITH_OPS = {
    "+": (operator.add, "add", 2),
    "-": (operator.sub, "sub", 2),
    "*": (operator.mul, "mul", 2),
    "/": (None, "div", 2),          # integer division, special handling
    "mod": (operator.mod, "mod", 2),
}

UNARY_OPS = {
    "inc": (lambda x: x + 1, 1),
    "dec": (lambda x: x - 1, 1),
    "abs": (abs, 1),
    "negate": (operator.neg, 1),
}

COMPARISON_OPS = {
    "<": (operator.lt, 2),
    ">": (operator.gt, 2),
    "<=": (operator.le, 2),
    ">=": (operator.ge, 2),
    "=": (operator.eq, 2),
    "!=": (operator.ne, 2),
}

PREDICATES = {
    "zero?": (lambda x: x == 0,),
    "pos?": (lambda x: x > 0,),
    "neg?": (lambda x: x < 0,),
    "even?": (lambda x: x % 2 == 0,),
    "odd?": (lambda x: x % 2 != 0,),
}

BOOLEAN_OPS = {
    "and": (lambda a, b: a and b,),
    "or": (lambda a, b: a or b,),
    "not": (lambda a: not a,),
}

BITWISE_OPS = {
    "bit-and": (operator.and_, 2),
    "bit-or": (operator.or_, 2),
    "bit-xor": (operator.xor, 2),
    "bit-shift-left": (None, 2),    # special: limit shift amount
    "bit-shift-right": (None, 2),
}


def _fmt_bool(v: bool) -> str:
    return "true" if v else "false"


def _fmt_result(v: Any) -> str:
    if isinstance(v, bool):
        return _fmt_bool(v)
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.6g}"
    return str(v)


def _fmt_list(v: list) -> str:
    return "[" + " ".join(_fmt_result(x) for x in v) + "]"


# ── Notation formatters ──────────────────────────────────────────

def fmt_raw_binary(op_sym: str, a: int, b: int, result: str) -> str:
    """Raw math: 347 + 289 = 636"""
    return f"{a} {op_sym} {b} = {result}"


def fmt_sexpr_binary(op_name: str, a: int, b: int, result: str) -> str:
    """S-expr: (+ 347 289) → 636"""
    return f"({op_name} {a} {b}) → {result}"


def fmt_lambda_binary(op_name: str, a: int, b: int, result: str) -> str:
    """Lambda: (λx. λy. (+ x y) a b) → result"""
    return f"(λx. λy. ({op_name} x y) {a} {b}) → {result}"


def fmt_raw_unary(op_sym: str, a: int, result: str) -> str:
    return f"{op_sym}({a}) = {result}"


def fmt_sexpr_unary(op_name: str, a: int, result: str) -> str:
    return f"({op_name} {a}) → {result}"


def fmt_lambda_unary(op_name: str, a: int, result: str) -> str:
    return f"(λx. ({op_name} x) {a}) → {result}"


def fmt_raw_predicate(pred: str, a: int, result: str) -> str:
    return f"{pred}({a}) = {result}"


def fmt_sexpr_predicate(pred: str, a: int, result: str) -> str:
    return f"({pred} {a}) → {result}"


def fmt_lambda_predicate(pred: str, a: int, result: str) -> str:
    return f"(λx. ({pred} x) {a}) → {result}"


# ── Tier 1 generators ────────────────────────────────────────────

def gen_arithmetic(rng: random.Random, notation: str) -> Example | None:
    """Generate a single arithmetic operation."""
    op_sym = rng.choice(list(ARITH_OPS.keys()))
    a = rand_int(rng)
    b = rand_int(rng)

    # Avoid division by zero and non-integer division
    if op_sym == "/":
        b = rand_positive(rng)
        # Make it divide evenly
        result_val = rand_int(rng, rng.choice([1, 1, 2, 2, 3]))
        a = result_val * b
        result = _fmt_result(result_val)
    elif op_sym == "mod":
        b = rand_positive(rng)
        result = _fmt_result(a % b)
    else:
        fn = ARITH_OPS[op_sym][0]
        result = _fmt_result(fn(a, b))

    if notation == "raw":
        text = fmt_raw_binary(op_sym, a, b, result)
    elif notation == "sexpr":
        text = fmt_sexpr_binary(op_sym, a, b, result)
    else:
        text = fmt_lambda_binary(op_sym, a, b, result)

    return Example(text=text, notation=notation, tier=1, category="arithmetic")


def gen_unary(rng: random.Random, notation: str) -> Example:
    """Generate a unary operation."""
    op_name = rng.choice(list(UNARY_OPS.keys()))
    fn = UNARY_OPS[op_name][0]

    if op_name == "negate":
        a = rand_int(rng)
    else:
        a = rand_int(rng)

    result = _fmt_result(fn(a))

    if notation == "raw":
        text = fmt_raw_unary(op_name, a, result)
    elif notation == "sexpr":
        text = fmt_sexpr_unary(op_name, a, result)
    else:
        text = fmt_lambda_unary(op_name, a, result)

    return Example(text=text, notation=notation, tier=1, category="unary")


def gen_comparison(rng: random.Random, notation: str) -> Example:
    """Generate a comparison operation."""
    op_sym = rng.choice(list(COMPARISON_OPS.keys()))
    fn = COMPARISON_OPS[op_sym][0]
    a = rand_int(rng)
    b = rand_int(rng)
    result = _fmt_result(fn(a, b))

    if notation == "raw":
        text = f"{a} {op_sym} {b} = {result}"
    elif notation == "sexpr":
        text = f"({op_sym} {a} {b}) → {result}"
    else:
        text = f"(λx. λy. ({op_sym} x y) {a} {b}) → {result}"

    return Example(text=text, notation=notation, tier=1, category="comparison")


def gen_predicate(rng: random.Random, notation: str) -> Example:
    """Generate a predicate check."""
    pred = rng.choice(list(PREDICATES.keys()))
    fn = PREDICATES[pred][0]

    # Bias inputs to make predicates sometimes true
    if pred == "zero?":
        a = rng.choice([0, 0, 0] + [rand_int(rng) for _ in range(7)])
    elif pred == "neg?":
        a = rng.choice([-rand_positive(rng)] * 3 + [rand_int(rng)] * 7)
    elif pred == "pos?":
        a = rng.choice([rand_positive(rng)] * 3 + [0, -rand_positive(rng)] * 2)
    else:
        a = rand_int(rng)

    result = _fmt_result(fn(a))

    if notation == "raw":
        text = fmt_raw_predicate(pred, a, result)
    elif notation == "sexpr":
        text = fmt_sexpr_predicate(pred, a, result)
    else:
        text = fmt_lambda_predicate(pred, a, result)

    return Example(text=text, notation=notation, tier=1, category="predicate")


def gen_boolean(rng: random.Random, notation: str) -> Example:
    """Generate a boolean operation."""
    op = rng.choice(["and", "or", "not"])

    if op == "not":
        a = rand_bool(rng)
        result = _fmt_bool(not a)
        a_s = _fmt_bool(a)
        if notation == "raw":
            text = f"not {a_s} = {result}"
        elif notation == "sexpr":
            text = f"(not {a_s}) → {result}"
        else:
            text = f"(λx. (not x) {a_s}) → {result}"
    else:
        a, b = rand_bool(rng), rand_bool(rng)
        fn = BOOLEAN_OPS[op][0]
        result = _fmt_bool(fn(a, b))
        a_s, b_s = _fmt_bool(a), _fmt_bool(b)
        if notation == "raw":
            text = f"{a_s} {op} {b_s} = {result}"
        elif notation == "sexpr":
            text = f"({op} {a_s} {b_s}) → {result}"
        else:
            text = f"(λx. λy. ({op} x y) {a_s} {b_s}) → {result}"

    return Example(text=text, notation=notation, tier=1, category="boolean")


def gen_bitwise(rng: random.Random, notation: str) -> Example:
    """Generate a bitwise operation."""
    op = rng.choice(list(BITWISE_OPS.keys()))

    if op == "bit-shift-left":
        a = rand_int(rng, rng.choice([1, 1, 2]))
        b = rng.randint(0, 8)
        result = _fmt_result(a << b)
    elif op == "bit-shift-right":
        a = rand_int(rng, rng.choice([2, 3, 4]))
        b = rng.randint(0, 8)
        result = _fmt_result(a >> b)
    else:
        a = rand_int(rng, rng.choice([1, 2, 3]))
        b = rand_int(rng, rng.choice([1, 2, 3]))
        fn = BITWISE_OPS[op][0]
        result = _fmt_result(fn(a, b))

    if notation == "raw":
        text = f"{a} {op} {b} = {result}"
    elif notation == "sexpr":
        text = f"({op} {a} {b}) → {result}"
    else:
        text = f"(λx. λy. ({op} x y) {a} {b}) → {result}"

    return Example(text=text, notation=notation, tier=1, category="bitwise")


# ═══════════════════════════════════════════════════════════════════
# Math generators — Tier 2 (compound: 2 operations)
# ═══════════════════════════════════════════════════════════════════

def gen_compound_arith(rng: random.Random, notation: str) -> Example | None:
    """Generate a compound arithmetic expression (2 operations)."""
    patterns = [
        "add_mul",      # (a + b) * c
        "mul_add",      # a * b + c * d
        "sub_mul",      # (a - b) * c
        "nested_pred",  # (even? (+ a b))
        "max_expr",     # (max (+ a b) (- c d))
        "min_expr",     # (min (* a b) (+ c d))
        "square",       # (* x x)
        "double",       # (+ x x)  or  (* 2 x)
    ]
    pat = rng.choice(patterns)

    try:
        if pat == "add_mul":
            a, b, c = rand_int(rng), rand_int(rng), rand_int(rng, rng.choice([1, 1, 2]))
            val = (a + b) * c
            if notation == "raw":
                text = f"({a} + {b}) * {c} = {val}"
            elif notation == "sexpr":
                text = f"(* (+ {a} {b}) {c}) → {val}"
            else:
                text = f"(λa. λb. λc. (* (+ a b) c) {a} {b} {c}) → {val}"

        elif pat == "mul_add":
            a, b = rand_int(rng, 1), rand_int(rng, 1)
            c, d = rand_int(rng, 1), rand_int(rng, 1)
            val = a * b + c * d
            if notation == "raw":
                text = f"{a} * {b} + {c} * {d} = {val}"
            elif notation == "sexpr":
                text = f"(+ (* {a} {b}) (* {c} {d})) → {val}"
            else:
                text = f"(λa. λb. λc. λd. (+ (* a b) (* c d)) {a} {b} {c} {d}) → {val}"

        elif pat == "sub_mul":
            a, b = rand_int(rng), rand_int(rng)
            c = rand_int(rng, rng.choice([1, 1, 2]))
            val = (a - b) * c
            if notation == "raw":
                text = f"({a} - {b}) * {c} = {val}"
            elif notation == "sexpr":
                text = f"(* (- {a} {b}) {c}) → {val}"
            else:
                text = f"(λa. λb. λc. (* (- a b) c) {a} {b} {c}) → {val}"

        elif pat == "nested_pred":
            pred = rng.choice(["even?", "odd?", "zero?", "pos?", "neg?"])
            op = rng.choice(["+", "-", "*"])
            a, b = rand_int(rng), rand_int(rng)
            inner = {"+": a + b, "-": a - b, "*": a * b}[op]
            fn = PREDICATES[pred][0]
            val = _fmt_bool(fn(inner))
            if notation == "raw":
                text = f"{pred}({a} {op} {b}) = {val}"
            elif notation == "sexpr":
                text = f"({pred} ({op} {a} {b})) → {val}"
            else:
                text = f"(λa. λb. ({pred} ({op} a b)) {a} {b}) → {val}"

        elif pat == "max_expr":
            a, b, c, d = [rand_int(rng) for _ in range(4)]
            val = max(a + b, c - d)
            if notation == "raw":
                text = f"max({a} + {b}, {c} - {d}) = {val}"
            elif notation == "sexpr":
                text = f"(max (+ {a} {b}) (- {c} {d})) → {val}"
            else:
                text = f"(λa. λb. λc. λd. (max (+ a b) (- c d)) {a} {b} {c} {d}) → {val}"

        elif pat == "min_expr":
            a, b, c, d = [rand_int(rng) for _ in range(4)]
            val = min(a * b, c + d)
            if notation == "raw":
                text = f"min({a} * {b}, {c} + {d}) = {val}"
            elif notation == "sexpr":
                text = f"(min (* {a} {b}) (+ {c} {d})) → {val}"
            else:
                text = f"(λa. λb. λc. λd. (min (* a b) (+ c d)) {a} {b} {c} {d}) → {val}"

        elif pat == "square":
            x = rand_int(rng, rng.choice([1, 1, 2, 3]))
            val = x * x
            if notation == "raw":
                text = f"{x}² = {val}"
            elif notation == "sexpr":
                text = f"(* {x} {x}) → {val}"
            else:
                text = f"(λx. (* x x) {x}) → {val}"

        elif pat == "double":
            x = rand_int(rng)
            val = x + x
            if notation == "raw":
                text = f"2 * {x} = {val}"
            elif notation == "sexpr":
                text = f"(+ {x} {x}) → {val}"
            else:
                text = f"(λx. (+ x x) {x}) → {val}"

        else:
            return None

    except (ZeroDivisionError, OverflowError):
        return None

    return Example(text=text, notation=notation, tier=2, category="compound")


# ═══════════════════════════════════════════════════════════════════
# Math generators — Tier 3 (nested: 3 operations)
# ═══════════════════════════════════════════════════════════════════

def gen_nested_arith(rng: random.Random, notation: str) -> Example | None:
    """Generate a nested arithmetic expression (3 operations)."""
    patterns = ["full_nest", "chain", "compare_compound"]
    pat = rng.choice(patterns)

    try:
        if pat == "full_nest":
            # ((a + b) * (c - d)) + e
            a, b, c, d, e = [rand_int(rng, rng.choice([1, 1, 2])) for _ in range(5)]
            val = (a + b) * (c - d) + e
            if notation == "raw":
                text = f"(({a} + {b}) * ({c} - {d})) + {e} = {val}"
            elif notation == "sexpr":
                text = f"(+ (* (+ {a} {b}) (- {c} {d})) {e}) → {val}"
            else:
                text = f"(λa. λb. λc. λd. λe. (+ (* (+ a b) (- c d)) e) {a} {b} {c} {d} {e}) → {val}"

        elif pat == "chain":
            # inc(abs(a - b)) or dec(a * b + c)
            a, b = rand_int(rng), rand_int(rng)
            c = rand_int(rng, rng.choice([1, 1, 2]))
            inner = a - b
            val = abs(inner) + c
            if notation == "raw":
                text = f"abs({a} - {b}) + {c} = {val}"
            elif notation == "sexpr":
                text = f"(+ (abs (- {a} {b})) {c}) → {val}"
            else:
                text = f"(λa. λb. λc. (+ (abs (- a b)) c) {a} {b} {c}) → {val}"

        elif pat == "compare_compound":
            # (> (+ a b) (* c d))
            a, b = rand_int(rng), rand_int(rng)
            c, d = rand_int(rng, 1), rand_int(rng, 1)
            cmp = rng.choice(["<", ">", "<=", ">=", "="])
            left, right = a + b, c * d
            fn = COMPARISON_OPS[cmp][0]
            val = _fmt_bool(fn(left, right))
            if notation == "raw":
                text = f"({a} + {b}) {cmp} ({c} * {d}) = {val}"
            elif notation == "sexpr":
                text = f"({cmp} (+ {a} {b}) (* {c} {d})) → {val}"
            else:
                text = f"(λa. λb. λc. λd. ({cmp} (+ a b) (* c d)) {a} {b} {c} {d}) → {val}"

        else:
            return None

    except (ZeroDivisionError, OverflowError):
        return None

    return Example(text=text, notation=notation, tier=3, category="nested")


# ═══════════════════════════════════════════════════════════════════
# Clojure.core generators
# ═══════════════════════════════════════════════════════════════════

def _rand_int_list(rng: random.Random, min_len: int = 2, max_len: int = 8) -> list[int]:
    """Random list of integers."""
    n = rng.randint(min_len, max_len)
    return [rand_int(rng, rng.choice([1, 1, 2])) for _ in range(n)]


def _rand_str_list(rng: random.Random) -> list[str]:
    """Random list of short strings."""
    words = ["apple", "banana", "cherry", "date", "fig", "grape",
             "kiwi", "lemon", "mango", "orange", "pear", "plum"]
    n = rng.randint(2, 6)
    return rng.sample(words, min(n, len(words)))


def gen_clojure_sequence(rng: random.Random, notation: str) -> Example | None:
    """Generate a clojure sequence operation example."""
    op = rng.choice([
        "map_inc", "map_dec", "map_double", "map_square", "map_abs", "map_negate",
        "filter_even", "filter_odd", "filter_pos", "filter_neg", "filter_zero",
        "reduce_add", "reduce_mul", "reduce_max", "reduce_min",
        "first", "last", "rest", "count", "reverse",
        "take", "drop", "nth",
        "sort", "sort_reverse",
        "range", "repeat",
        "concat", "flatten",
        "apply_add", "apply_mul", "apply_max", "apply_min",
    ])

    xs = _rand_int_list(rng)
    xs_s = "[" + " ".join(str(x) for x in xs) + "]"

    try:
        if op == "map_inc":
            result = [x + 1 for x in xs]
            if notation == "sexpr":
                text = f"(map inc {xs_s}) → {_fmt_list(result)}"
            else:
                text = f"(map (λx. (+ x 1)) {xs_s}) → {_fmt_list(result)}"

        elif op == "map_dec":
            result = [x - 1 for x in xs]
            if notation == "sexpr":
                text = f"(map dec {xs_s}) → {_fmt_list(result)}"
            else:
                text = f"(map (λx. (- x 1)) {xs_s}) → {_fmt_list(result)}"

        elif op == "map_double":
            result = [x * 2 for x in xs]
            if notation == "sexpr":
                text = f"(map #(* % 2) {xs_s}) → {_fmt_list(result)}"
            else:
                text = f"(map (λx. (* x 2)) {xs_s}) → {_fmt_list(result)}"

        elif op == "map_square":
            result = [x * x for x in xs]
            if notation == "sexpr":
                text = f"(map #(* % %) {xs_s}) → {_fmt_list(result)}"
            else:
                text = f"(map (λx. (* x x)) {xs_s}) → {_fmt_list(result)}"

        elif op == "map_abs":
            xs = [rng.randint(-50, 50) for _ in range(rng.randint(3, 7))]
            xs_s = "[" + " ".join(str(x) for x in xs) + "]"
            result = [abs(x) for x in xs]
            if notation == "sexpr":
                text = f"(map abs {xs_s}) → {_fmt_list(result)}"
            else:
                text = f"(map (λx. (abs x)) {xs_s}) → {_fmt_list(result)}"

        elif op == "map_negate":
            result = [-x for x in xs]
            if notation == "sexpr":
                text = f"(map - {xs_s}) → {_fmt_list(result)}"
            else:
                text = f"(map (λx. (- x)) {xs_s}) → {_fmt_list(result)}"

        elif op == "filter_even":
            result = [x for x in xs if x % 2 == 0]
            if notation == "sexpr":
                text = f"(filter even? {xs_s}) → {_fmt_list(result)}"
            else:
                text = f"(filter (λx. (even? x)) {xs_s}) → {_fmt_list(result)}"

        elif op == "filter_odd":
            result = [x for x in xs if x % 2 != 0]
            if notation == "sexpr":
                text = f"(filter odd? {xs_s}) → {_fmt_list(result)}"
            else:
                text = f"(filter (λx. (odd? x)) {xs_s}) → {_fmt_list(result)}"

        elif op == "filter_pos":
            xs = [rng.randint(-20, 20) for _ in range(rng.randint(4, 8))]
            xs_s = "[" + " ".join(str(x) for x in xs) + "]"
            result = [x for x in xs if x > 0]
            if notation == "sexpr":
                text = f"(filter pos? {xs_s}) → {_fmt_list(result)}"
            else:
                text = f"(filter (λx. (pos? x)) {xs_s}) → {_fmt_list(result)}"

        elif op == "filter_neg":
            xs = [rng.randint(-20, 20) for _ in range(rng.randint(4, 8))]
            xs_s = "[" + " ".join(str(x) for x in xs) + "]"
            result = [x for x in xs if x < 0]
            if notation == "sexpr":
                text = f"(filter neg? {xs_s}) → {_fmt_list(result)}"
            else:
                text = f"(filter (λx. (neg? x)) {xs_s}) → {_fmt_list(result)}"

        elif op == "filter_zero":
            xs = [rng.choice([0, 0, rng.randint(-10, 10)]) for _ in range(rng.randint(4, 8))]
            xs_s = "[" + " ".join(str(x) for x in xs) + "]"
            result = [x for x in xs if x == 0]
            if notation == "sexpr":
                text = f"(filter zero? {xs_s}) → {_fmt_list(result)}"
            else:
                text = f"(filter (λx. (zero? x)) {xs_s}) → {_fmt_list(result)}"

        elif op == "reduce_add":
            val = sum(xs)
            if notation == "sexpr":
                text = f"(reduce + {xs_s}) → {val}"
            else:
                text = f"(reduce (λacc. λx. (+ acc x)) {xs_s}) → {val}"

        elif op == "reduce_mul":
            xs = _rand_int_list(rng, 2, 5)  # keep small to avoid huge numbers
            xs = [max(1, min(x, 20)) for x in xs]  # cap values
            xs_s = "[" + " ".join(str(x) for x in xs) + "]"
            val = 1
            for x in xs:
                val *= x
            if notation == "sexpr":
                text = f"(reduce * {xs_s}) → {val}"
            else:
                text = f"(reduce (λacc. λx. (* acc x)) {xs_s}) → {val}"

        elif op == "reduce_max":
            val = max(xs)
            if notation == "sexpr":
                text = f"(reduce max {xs_s}) → {val}"
            else:
                text = f"(reduce (λacc. λx. (max acc x)) {xs_s}) → {val}"

        elif op == "reduce_min":
            val = min(xs)
            if notation == "sexpr":
                text = f"(reduce min {xs_s}) → {val}"
            else:
                text = f"(reduce (λacc. λx. (min acc x)) {xs_s}) → {val}"

        elif op == "first":
            val = xs[0]
            # Only sexpr for this — lambda notation doesn't add much
            text = f"(first {xs_s}) → {val}"

        elif op == "last":
            val = xs[-1]
            text = f"(last {xs_s}) → {val}"

        elif op == "rest":
            result = xs[1:]
            text = f"(rest {xs_s}) → {_fmt_list(result)}"

        elif op == "count":
            val = len(xs)
            text = f"(count {xs_s}) → {val}"

        elif op == "reverse":
            result = list(reversed(xs))
            text = f"(reverse {xs_s}) → {_fmt_list(result)}"

        elif op == "take":
            n = rng.randint(1, min(len(xs), 5))
            result = xs[:n]
            text = f"(take {n} {xs_s}) → {_fmt_list(result)}"

        elif op == "drop":
            n = rng.randint(1, min(len(xs), 5))
            result = xs[n:]
            text = f"(drop {n} {xs_s}) → {_fmt_list(result)}"

        elif op == "nth":
            n = rng.randint(0, len(xs) - 1)
            val = xs[n]
            text = f"(nth {xs_s} {n}) → {val}"

        elif op == "sort":
            result = sorted(xs)
            text = f"(sort {xs_s}) → {_fmt_list(result)}"

        elif op == "sort_reverse":
            result = sorted(xs, reverse=True)
            text = f"(sort > {xs_s}) → {_fmt_list(result)}"

        elif op == "range":
            n = rng.randint(2, 12)
            result = list(range(n))
            text = f"(range {n}) → {_fmt_list(result)}"

        elif op == "repeat":
            n = rng.randint(2, 6)
            v = rand_int(rng, 1)
            result = [v] * n
            text = f"(repeat {n} {v}) → {_fmt_list(result)}"

        elif op == "concat":
            ys = _rand_int_list(rng, 2, 5)
            ys_s = "[" + " ".join(str(y) for y in ys) + "]"
            result = xs + ys
            text = f"(concat {xs_s} {ys_s}) → {_fmt_list(result)}"

        elif op == "flatten":
            # Nested list
            a = _rand_int_list(rng, 1, 3)
            b = _rand_int_list(rng, 1, 3)
            a_s = "[" + " ".join(str(x) for x in a) + "]"
            b_s = "[" + " ".join(str(x) for x in b) + "]"
            result = a + b
            text = f"(flatten [{a_s} {b_s}]) → {_fmt_list(result)}"

        elif op == "apply_add":
            val = sum(xs)
            text = f"(apply + {xs_s}) → {val}"

        elif op == "apply_mul":
            xs = [max(1, min(x, 15)) for x in xs[:5]]
            xs_s = "[" + " ".join(str(x) for x in xs) + "]"
            val = 1
            for x in xs:
                val *= x
            text = f"(apply * {xs_s}) → {val}"

        elif op == "apply_max":
            val = max(xs)
            text = f"(apply max {xs_s}) → {val}"

        elif op == "apply_min":
            val = min(xs)
            text = f"(apply min {xs_s}) → {val}"

        else:
            return None

    except (ZeroDivisionError, IndexError, OverflowError, ValueError):
        return None

    return Example(text=text, notation=notation, tier=0, category="sequence")


def gen_clojure_collection(rng: random.Random, notation: str) -> Example | None:
    """Generate clojure collection operation examples."""
    op = rng.choice([
        "assoc", "dissoc", "get", "update_inc",
        "merge", "keys", "vals", "select_keys",
        "conj_vec", "conj_set", "into_vec", "empty",
        "contains", "count_map",
    ])

    try:
        if op == "assoc":
            k = rng.choice([":a", ":b", ":c", ":x", ":y", ":name", ":age", ":score"])
            v = rand_int(rng, 1)
            text = f'(assoc {{:a 1 :b 2}} {k} {v}) → {{:a 1 :b 2 {k} {v}}}'

        elif op == "dissoc":
            k = rng.choice([":a", ":b"])
            if k == ":a":
                text = f"(dissoc {{:a 1 :b 2 :c 3}} :a) → {{:b 2 :c 3}}"
            else:
                text = f"(dissoc {{:a 1 :b 2 :c 3}} :b) → {{:a 1 :c 3}}"

        elif op == "get":
            k = rng.choice([":a", ":b", ":c"])
            m = {":a": 10, ":b": 20, ":c": 30}
            val = m.get(k, "nil")
            text = f"(get {{:a 10 :b 20 :c 30}} {k}) → {val}"

        elif op == "update_inc":
            k = rng.choice([":a", ":b", ":count"])
            v = rand_int(rng, 1)
            if notation == "sexpr":
                text = f"(update {{{k} {v}}} {k} inc) → {{{k} {v + 1}}}"
            else:
                text = f"(update {{{k} {v}}} {k} (λx. (+ x 1))) → {{{k} {v + 1}}}"

        elif op == "merge":
            text = f"(merge {{:a 1 :b 2}} {{:b 3 :c 4}}) → {{:a 1 :b 3 :c 4}}"

        elif op == "keys":
            text = f"(keys {{:a 1 :b 2 :c 3}}) → [:a :b :c]"

        elif op == "vals":
            a, b, c = rand_int(rng, 1), rand_int(rng, 1), rand_int(rng, 1)
            text = f"(vals {{:a {a} :b {b} :c {c}}}) → [{a} {b} {c}]"

        elif op == "select_keys":
            text = f"(select-keys {{:a 1 :b 2 :c 3}} [:a :c]) → {{:a 1 :c 3}}"

        elif op == "conj_vec":
            xs = _rand_int_list(rng, 2, 5)
            v = rand_int(rng, 1)
            result = xs + [v]
            text = f"(conj {_fmt_list(xs)} {v}) → {_fmt_list(result)}"

        elif op == "conj_set":
            vals = sorted(set(rng.sample(range(1, 20), rng.randint(2, 5))))
            v = rand_int(rng, 1)
            result = sorted(set(vals) | {v})
            s_s = "#{" + " ".join(str(x) for x in vals) + "}"
            r_s = "#{" + " ".join(str(x) for x in result) + "}"
            text = f"(conj {s_s} {v}) → {r_s}"

        elif op == "into_vec":
            xs = _rand_int_list(rng, 2, 4)
            ys = _rand_int_list(rng, 2, 4)
            result = xs + ys
            text = f"(into {_fmt_list(xs)} {_fmt_list(ys)}) → {_fmt_list(result)}"

        elif op == "empty":
            coll = rng.choice(["[]", "{}", "#{}", "[1 2 3]", "{:a 1}"])
            is_empty = coll in ["[]", "{}", "#{}"]
            text = f"(empty? {coll}) → {_fmt_bool(is_empty)}"

        elif op == "contains":
            k = rng.choice([":a", ":b", ":d"])
            val = "true" if k in [":a", ":b"] else "false"
            text = f"(contains? {{:a 1 :b 2 :c 3}} {k}) → {val}"

        elif op == "count_map":
            n = rng.randint(1, 6)
            pairs = " ".join(f":{chr(97+i)} {i+1}" for i in range(n))
            text = f"(count {{{pairs}}}) → {n}"

        else:
            return None

    except Exception:
        return None

    return Example(text=text, notation=notation, tier=0, category="collection")


def gen_clojure_string(rng: random.Random, notation: str) -> Example | None:
    """Generate clojure string operation examples."""
    op = rng.choice([
        "str_concat", "count_str", "subs", "upper", "lower",
        "trim", "join", "split",
    ])

    words = ["hello", "world", "foo", "bar", "baz", "clojure", "lambda", "verbum"]

    try:
        if op == "str_concat":
            a, b = rng.sample(words, 2)
            text = f'(str "{a}" "{b}") → "{a}{b}"'

        elif op == "count_str":
            w = rng.choice(words)
            text = f'(count "{w}") → {len(w)}'

        elif op == "subs":
            w = rng.choice(words)
            start = rng.randint(0, max(0, len(w) - 2))
            end = rng.randint(start + 1, len(w))
            text = f'(subs "{w}" {start} {end}) → "{w[start:end]}"'

        elif op == "upper":
            w = rng.choice(words)
            text = f'(upper-case "{w}") → "{w.upper()}"'

        elif op == "lower":
            w = rng.choice(["Hello", "WORLD", "FooBar", "LAMBDA"])
            text = f'(lower-case "{w}") → "{w.lower()}"'

        elif op == "trim":
            w = rng.choice(words)
            text = f'(trim "  {w}  ") → "{w}"'

        elif op == "join":
            ws = rng.sample(words, rng.randint(2, 4))
            sep = rng.choice([" ", ", ", "-", "/"])
            items = "[" + " ".join(f'"{w}"' for w in ws) + "]"
            result = sep.join(ws)
            text = f'(join "{sep}" {items}) → "{result}"'

        elif op == "split":
            sep = rng.choice([" ", "-", "/"])
            ws = rng.sample(words, rng.randint(2, 4))
            s = sep.join(ws)
            result = "[" + " ".join(f'"{w}"' for w in ws) + "]"
            text = f'(split "{s}" #"{re_escape(sep)}") → {result}'

        else:
            return None

    except Exception:
        return None

    return Example(text=text, notation=notation, tier=0, category="string")


def re_escape(s: str) -> str:
    """Minimal regex escape for split patterns."""
    return s.replace("/", "\\/")


def gen_clojure_compound(rng: random.Random, notation: str) -> Example | None:
    """Generate compound clojure expressions (composition of 2+ operations)."""
    op = rng.choice([
        "filter_map", "map_filter", "reduce_map", "count_filter",
        "first_filter", "last_sort", "take_sort", "sum_range",
        "comp_inc_double", "partial_add",
    ])

    try:
        if op == "filter_map":
            xs = _rand_int_list(rng, 4, 8)
            xs_s = "[" + " ".join(str(x) for x in xs) + "]"
            result = [x for x in [y + 1 for y in xs] if x % 2 == 0]
            if notation == "sexpr":
                text = f"(filter even? (map inc {xs_s})) → {_fmt_list(result)}"
            else:
                text = f"(filter (λx. (even? x)) (map (λx. (+ x 1)) {xs_s})) → {_fmt_list(result)}"

        elif op == "map_filter":
            xs = _rand_int_list(rng, 4, 8)
            xs_s = "[" + " ".join(str(x) for x in xs) + "]"
            evens = [x for x in xs if x % 2 == 0]
            result = [x * x for x in evens]
            if notation == "sexpr":
                text = f"(map #(* % %) (filter even? {xs_s})) → {_fmt_list(result)}"
            else:
                text = f"(map (λx. (* x x)) (filter (λx. (even? x)) {xs_s})) → {_fmt_list(result)}"

        elif op == "reduce_map":
            xs = _rand_int_list(rng, 3, 6)
            xs_s = "[" + " ".join(str(x) for x in xs) + "]"
            val = sum(x * x for x in xs)
            if notation == "sexpr":
                text = f"(reduce + (map #(* % %) {xs_s})) → {val}"
            else:
                text = f"(reduce (λacc. λx. (+ acc x)) (map (λx. (* x x)) {xs_s})) → {val}"

        elif op == "count_filter":
            xs = _rand_int_list(rng, 5, 10)
            xs_s = "[" + " ".join(str(x) for x in xs) + "]"
            val = len([x for x in xs if x % 2 == 0])
            if notation == "sexpr":
                text = f"(count (filter even? {xs_s})) → {val}"
            else:
                text = f"(count (filter (λx. (even? x)) {xs_s})) → {val}"

        elif op == "first_filter":
            xs = _rand_int_list(rng, 5, 10)
            xs_s = "[" + " ".join(str(x) for x in xs) + "]"
            evens = [x for x in xs if x % 2 == 0]
            if not evens:
                return None
            val = evens[0]
            text = f"(first (filter even? {xs_s})) → {val}"

        elif op == "last_sort":
            xs = _rand_int_list(rng, 3, 7)
            xs_s = "[" + " ".join(str(x) for x in xs) + "]"
            val = max(xs)
            text = f"(last (sort {xs_s})) → {val}"

        elif op == "take_sort":
            xs = _rand_int_list(rng, 5, 10)
            xs_s = "[" + " ".join(str(x) for x in xs) + "]"
            n = rng.randint(2, 4)
            result = sorted(xs)[:n]
            text = f"(take {n} (sort {xs_s})) → {_fmt_list(result)}"

        elif op == "sum_range":
            n = rng.randint(2, 15)
            val = sum(range(n))
            if notation == "sexpr":
                text = f"(reduce + (range {n})) → {val}"
            else:
                text = f"(reduce (λacc. λx. (+ acc x)) (range {n})) → {val}"

        elif op == "comp_inc_double":
            x = rand_int(rng, rng.choice([1, 2]))
            # (comp inc #(* % 2)) applied to x = (inc (* x 2)) = x*2 + 1
            val = x * 2 + 1
            if notation == "sexpr":
                text = f"((comp inc #(* % 2)) {x}) → {val}"
            else:
                text = f"((λx. (+ (* x 2) 1)) {x}) → {val}"

        elif op == "partial_add":
            a = rand_int(rng, rng.choice([1, 2]))
            b = rand_int(rng)
            val = a + b
            if notation == "sexpr":
                text = f"((partial + {a}) {b}) → {val}"
            else:
                text = f"((λx. (+ {a} x)) {b}) → {val}"

        else:
            return None

    except (ZeroDivisionError, IndexError, OverflowError, ValueError):
        return None

    return Example(text=text, notation=notation, tier=0, category="compound_clojure")


def gen_clojure_predicate_check(rng: random.Random, notation: str) -> Example | None:
    """Generate type/predicate check examples."""
    checks = [
        ("nil?", "nil", True),
        ("nil?", "42", False),
        ("nil?", ":foo", False),
        ("some?", "42", True),
        ("some?", "nil", False),
        ("some?", '":hello"', True),
        ("number?", "42", True),
        ("number?", '":foo"', False),
        ("number?", "true", False),
        ("string?", '"hello"', True),
        ("string?", "42", False),
        ("string?", ":foo", False),
        ("keyword?", ":foo", True),
        ("keyword?", '"hello"', False),
        ("keyword?", "42", False),
        ("vector?", "[1 2 3]", True),
        ("vector?", "{:a 1}", False),
        ("vector?", "42", False),
        ("map?", "{:a 1}", True),
        ("map?", "[1 2]", False),
        ("map?", "nil", False),
        ("set?", "#{1 2 3}", True),
        ("set?", "[1 2 3]", False),
        ("coll?", "[1 2 3]", True),
        ("coll?", "{:a 1}", True),
        ("coll?", "42", False),
        ("seq?", "(list 1 2 3)", True),
        ("seq?", "[1 2 3]", False),
        ("true?", "true", True),
        ("true?", "false", False),
        ("true?", "1", False),
        ("false?", "false", True),
        ("false?", "true", False),
        ("false?", "nil", False),
    ]

    pred, val, result = rng.choice(checks)
    text = f"({pred} {val}) → {_fmt_bool(result)}"
    return Example(text=text, notation=notation, tier=0, category="type_predicate")


def gen_clojure_let(rng: random.Random, notation: str) -> Example | None:
    """Generate let-binding examples."""
    patterns = [
        "simple_add", "simple_mul", "use_twice", "nested",
    ]
    pat = rng.choice(patterns)

    try:
        if pat == "simple_add":
            a, b = rand_int(rng, 1), rand_int(rng, 1)
            val = a + b
            if notation == "sexpr":
                text = f"(let [x {a} y {b}] (+ x y)) → {val}"
            else:
                text = f"((λx. (λy. (+ x y) {b}) {a})) → {val}"

        elif pat == "simple_mul":
            a, b = rand_int(rng, 1), rand_int(rng, 1)
            val = a * b
            if notation == "sexpr":
                text = f"(let [x {a} y {b}] (* x y)) → {val}"
            else:
                text = f"((λx. (λy. (* x y) {b}) {a})) → {val}"

        elif pat == "use_twice":
            a = rand_int(rng, rng.choice([1, 2]))
            val = a + a
            if notation == "sexpr":
                text = f"(let [x {a}] (+ x x)) → {val}"
            else:
                text = f"((λx. (+ x x)) {a}) → {val}"

        elif pat == "nested":
            a, b = rand_int(rng, 1), rand_int(rng, 1)
            inner = a + b
            val = inner * 2
            if notation == "sexpr":
                text = f"(let [x {a} y (+ x {b})] (* y 2)) → {val}"
            else:
                text = f"((λx. ((λy. (* y 2)) (+ x {b}))) {a}) → {val}"

        else:
            return None

    except Exception:
        return None

    return Example(text=text, notation=notation, tier=0, category="let_binding")


def gen_clojure_conditional(rng: random.Random, notation: str) -> Example | None:
    """Generate conditional (if/when/cond) examples."""
    patterns = ["if_simple", "if_compare", "when", "cond"]
    pat = rng.choice(patterns)

    try:
        if pat == "if_simple":
            cond_val = rand_bool(rng)
            a, b = rand_int(rng, 1), rand_int(rng, 1)
            val = a if cond_val else b
            cond_s = _fmt_bool(cond_val)
            text = f"(if {cond_s} {a} {b}) → {val}"

        elif pat == "if_compare":
            a, b = rand_int(rng), rand_int(rng)
            cmp = rng.choice(["<", ">", "="])
            fn = COMPARISON_OPS[cmp][0]
            cond_val = fn(a, b)
            then_v, else_v = rand_int(rng, 1), rand_int(rng, 1)
            val = then_v if cond_val else else_v
            text = f"(if ({cmp} {a} {b}) {then_v} {else_v}) → {val}"

        elif pat == "when":
            cond_val = rand_bool(rng)
            a = rand_int(rng, 1)
            val = a if cond_val else "nil"
            text = f"(when {_fmt_bool(cond_val)} {a}) → {val}"

        elif pat == "cond":
            a = rand_int(rng, rng.choice([1, 2]))
            if a < 0:
                val = "negative"
            elif a == 0:
                val = "zero"
            else:
                val = "positive"
            text = f'(cond (neg? {a}) "negative" (zero? {a}) "zero" :else "positive") → "{val}"'

        else:
            return None

    except Exception:
        return None

    return Example(text=text, notation=notation, tier=0, category="conditional")


def gen_clojure_fn_def(rng: random.Random, notation: str) -> Example | None:
    """Generate function definition + application examples."""
    patterns = ["defn_apply", "fn_apply", "higher_order"]
    pat = rng.choice(patterns)

    try:
        if pat == "defn_apply":
            a, b = rand_int(rng, 1), rand_int(rng, 1)
            op = rng.choice(["+", "-", "*"])
            fn_map = {"+": a + b, "-": a - b, "*": a * b}
            val = fn_map[op]
            name = rng.choice(["f", "g", "h", "my-fn", "calc"])
            if notation == "sexpr":
                text = f"(defn {name} [x y] ({op} x y)) ({name} {a} {b}) → {val}"
            else:
                text = f"(def {name} (λx. λy. ({op} x y))) ({name} {a} {b}) → {val}"

        elif pat == "fn_apply":
            a = rand_int(rng, rng.choice([1, 2]))
            val = a * a + 1
            if notation == "sexpr":
                text = f"((fn [x] (+ (* x x) 1)) {a}) → {val}"
            else:
                text = f"((λx. (+ (* x x) 1)) {a}) → {val}"

        elif pat == "higher_order":
            a, b = rand_int(rng, 1), rand_int(rng, 1)
            val = a + b
            if notation == "sexpr":
                text = f"((fn [f x y] (f x y)) + {a} {b}) → {val}"
            else:
                text = f"((λf. λx. λy. (f x y)) + {a} {b}) → {val}"

        else:
            return None

    except Exception:
        return None

    return Example(text=text, notation=notation, tier=0, category="fn_def")


# ═══════════════════════════════════════════════════════════════════
# Master generator
# ═══════════════════════════════════════════════════════════════════

# Weight distribution for different generator categories
MATH_GENERATORS = [
    (gen_arithmetic, 25),       # heavy on basic arithmetic
    (gen_unary, 8),
    (gen_comparison, 12),
    (gen_predicate, 10),
    (gen_boolean, 8),
    (gen_bitwise, 7),
    (gen_compound_arith, 15),   # tier 2
    (gen_nested_arith, 10),     # tier 3
]

CLOJURE_GENERATORS = [
    (gen_clojure_sequence, 25),
    (gen_clojure_collection, 10),
    (gen_clojure_string, 8),
    (gen_clojure_compound, 15),
    (gen_clojure_predicate_check, 8),
    (gen_clojure_let, 10),
    (gen_clojure_conditional, 8),
    (gen_clojure_fn_def, 10),
]

ALL_GENERATORS = MATH_GENERATORS + CLOJURE_GENERATORS


def _build_weighted(generators):
    """Build a flat list for weighted random selection."""
    pool = []
    for gen_fn, weight in generators:
        pool.extend([gen_fn] * weight)
    return pool


def generate_examples(
    count: int,
    seed: int = 42,
    math_ratio: float = 0.5,
) -> list[Example]:
    """Generate `count` training examples.

    math_ratio: fraction of examples that are math (vs clojure).
    """
    rng = random.Random(seed)

    math_pool = _build_weighted(MATH_GENERATORS)
    clojure_pool = _build_weighted(CLOJURE_GENERATORS)

    examples = []
    attempts = 0
    max_attempts = count * 3  # safety limit

    while len(examples) < count and attempts < max_attempts:
        attempts += 1

        # Pick math or clojure
        if rng.random() < math_ratio:
            gen_fn = rng.choice(math_pool)
            # Math gets all three notations
            notation = rng.choice(["raw", "sexpr", "lambda"])
        else:
            gen_fn = rng.choice(clojure_pool)
            # Clojure gets sexpr or lambda
            notation = rng.choice(["sexpr", "lambda"])

        ex = gen_fn(rng, notation)
        if ex is not None:
            examples.append(ex)

    return examples


# ═══════════════════════════════════════════════════════════════════
# Packing into Qwen3 shards
# ═══════════════════════════════════════════════════════════════════

def pack_to_shards(
    examples: list[Example],
    out_dir: Path,
    shard_size: int = 50_000_000,
) -> dict:
    """Tokenize examples and pack into numpy shards."""
    sys.path.insert(0, str(Path(__file__).parent))
    from tokenizer import EOD_ID, VOCAB_SIZE, encode_document, load_tokenizer

    tok = load_tokenizer()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenize all examples
    all_ids = []
    total_tokens = 0
    for ex in examples:
        ids = encode_document(ex.text)
        all_ids.extend(ids)
        total_tokens += len(ids)

    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens/example: {total_tokens / len(examples):.1f}")

    # Pack into shards
    all_ids = np.array(all_ids, dtype=np.int32)
    n_shards = max(1, len(all_ids) // shard_size)
    remainder = len(all_ids) % shard_size

    shard_idx = 0
    for i in range(0, len(all_ids) - remainder, shard_size):
        shard = all_ids[i : i + shard_size]
        path = out_dir / f"shard_{shard_idx:05d}.npy"
        np.save(path, shard)
        shard_idx += 1

    # Last partial shard (if any meaningful data)
    if remainder > 1000:
        shard = np.zeros(shard_size, dtype=np.int32)
        shard[:remainder] = all_ids[-remainder:]
        path = out_dir / f"shard_{shard_idx:05d}.npy"
        np.save(path, shard)
        shard_idx += 1

    # Verify first shard
    s0 = np.load(out_dir / "shard_00000.npy")
    n_eod = (s0 == EOD_ID).sum()

    status = {
        "type": "bios-flash",
        "tokenizer": "Qwen3-BBPE",
        "vocab_size": VOCAB_SIZE,
        "eod_id": EOD_ID,
        "total_examples": len(examples),
        "total_tokens": total_tokens,
        "unique_tokens": total_tokens,
        "shards_written": shard_idx,
        "shard_size": shard_size,
        "avg_tokens_per_example": round(total_tokens / len(examples), 1),
        "eod_in_shard_0": int(n_eod),
        "max_token_id": int(all_ids.max()),
        "timestamp": datetime.now(UTC).isoformat(),
    }

    status_path = out_dir / "prep_status.json"
    status_path.write_text(json.dumps(status, indent=2))

    print(f"  Shards: {shard_idx} × {shard_size:,} tokens")
    print(f"  Max token ID: {all_ids.max()} (vocab: {VOCAB_SIZE})")
    print(f"  EOD in shard_0: {n_eod:,}")
    print(f"  Status: {status_path}")

    return status


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate BIOS flash training data")
    parser.add_argument("--count", type=int, default=2_600_000,
                        help="Number of examples to generate (default: 2.6M, ~50M tokens)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--math-ratio", type=float, default=0.5,
                        help="Fraction of math examples (default: 0.5)")
    parser.add_argument("--pack", action="store_true",
                        help="Pack into Qwen3 shards after generating")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/Users/mwhitford/data/fractal-bitnet/shards-bios"),
                        help="Output directory for shards")
    parser.add_argument("--shard-size", type=int, default=50_000_000,
                        help="Tokens per shard")
    parser.add_argument("--dump", type=int, default=0,
                        help="Dump N examples to stdout (for inspection)")
    args = parser.parse_args()

    print("=" * 60)
    print("  BIOS Flash — Training Data Generator")
    print("=" * 60)
    print()

    t0 = time.time()
    examples = generate_examples(
        count=args.count,
        seed=args.seed,
        math_ratio=args.math_ratio,
    )
    elapsed = time.time() - t0

    # ── Stats ─────────────────────────────────────────────────────
    print(f"  Generated: {len(examples):,} examples in {elapsed:.1f}s")
    print()

    # By category
    cats: dict[str, int] = {}
    for ex in examples:
        cats[ex.category] = cats.get(ex.category, 0) + 1
    print("  By category:")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {cat:25s}: {n:>7,}")

    # By notation
    nots: dict[str, int] = {}
    for ex in examples:
        nots[ex.notation] = nots.get(ex.notation, 0) + 1
    print(f"\n  By notation:")
    for notation, n in sorted(nots.items(), key=lambda x: -x[1]):
        print(f"    {notation:10s}: {n:>7,}")

    # By tier
    tiers: dict[int, int] = {}
    for ex in examples:
        tiers[ex.tier] = tiers.get(ex.tier, 0) + 1
    print(f"\n  By tier:")
    for tier, n in sorted(tiers.items()):
        label = {0: "clojure", 1: "tier-1 (single)", 2: "tier-2 (compound)", 3: "tier-3 (nested)"}
        print(f"    {label.get(tier, f'tier-{tier}'):25s}: {n:>7,}")

    # Sample examples
    print(f"\n  Samples:")
    rng = random.Random(args.seed + 1)
    samples = rng.sample(examples, min(20, len(examples)))
    for ex in samples:
        print(f"    [{ex.notation:6s}|{ex.category:20s}] {ex.text}")

    # Dump if requested
    if args.dump > 0:
        print(f"\n  Dumping {args.dump} examples:")
        for ex in examples[:args.dump]:
            print(ex.text)

    # Pack if requested
    if args.pack:
        print(f"\n{'=' * 60}")
        print(f"  PACKING INTO SHARDS")
        print(f"{'=' * 60}")
        print()

        # Shuffle before packing
        rng2 = random.Random(args.seed)
        rng2.shuffle(examples)

        status = pack_to_shards(examples, args.out_dir, args.shard_size)

        print(f"\n  Done! {status['shards_written']} shard(s) in {args.out_dir}")

    print()


if __name__ == "__main__":
    main()
