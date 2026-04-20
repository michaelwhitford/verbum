"""
Binding probe analysis: VSM-LM v3 (3-register) vs v2 (1-register).

Loads results/binding/vsm_probe_step_010000_v3.json and v2.json,
computes per-category and minimal-pair comparisons, prints structured
tables to stdout, and writes a JSON summary to
results/binding/binding_analysis_v2_v3.json.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
V3_PATH = ROOT / "results/binding/vsm_probe_step_010000_v3.json"
V2_PATH = ROOT / "results/binding/vsm_probe_step_010000_v2.json"
OUT_PATH = ROOT / "results/binding/binding_analysis_v2_v3.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CATEGORY_ABBREV = {
    "quantifier_scope": "QScope",
    "variable_binding": "VarBind",
    "anaphora": "Anaph",
    "control": "Control",
    "relative_clause": "RelCl",
}

CATEGORIES = [
    "quantifier_scope",
    "variable_binding",
    "anaphora",
    "control",
    "relative_clause",
]


def _mean(vals: list[float]) -> float | None:
    return statistics.mean(vals) if vals else None


def _std(vals: list[float]) -> float | None:
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def _fmt(v: float | None, width: int = 7, prec: int = 4) -> str:
    if v is None:
        return " " * width
    return f"{v:{width}.{prec}f}"


def _bar(v: float, lo: float, hi: float, width: int = 10) -> str:
    """ASCII bar scaled to [lo, hi]."""
    if hi == lo:
        return "─" * width
    frac = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def _section(title: str) -> None:
    print()
    print("=" * 76)
    print(f"  {title}")
    print("=" * 76)


def _divider() -> None:
    print("-" * 76)


# ---------------------------------------------------------------------------
# Extract per-probe records
# ---------------------------------------------------------------------------

def extract_v3(probe: dict) -> dict:
    m = probe["metrics"]
    pid = probe["probe_id"]
    cat = probe["category"]
    return {
        "probe_id": pid,
        "category": cat,
        "prompt": probe.get("prompt", ""),
        "seq_len": probe.get("seq_len"),
        # iter0 gates
        "i0_type_gate": m["iter0_type_gate_mean"],
        "i0_parse_gate": m["iter0_parse_gate_mean"],
        "i0_apply_gate": m["iter0_apply_gate_mean"],
        # iter1 gates
        "i1_type_gate": m["iter1_type_gate_mean"],
        "i1_parse_gate": m["iter1_parse_gate_mean"],
        "i1_apply_gate": m["iter1_apply_gate_mean"],
        # iter0 register norms
        "i0_reg_type": m["iter0_register_type_norm"],
        "i0_reg_scope": m["iter0_register_scope_norm"],
        "i0_reg_role": m["iter0_register_role_norm"],
        # iter1 register norms
        "i1_reg_type": m["iter1_register_type_norm"],
        "i1_reg_scope": m["iter1_register_scope_norm"],
        "i1_reg_role": m["iter1_register_role_norm"],
        # iter0 type-write partition (type→{type,scope,role})
        "i0_tw_type": m["iter0_type_write_type"],
        "i0_tw_scope": m["iter0_type_write_scope"],
        "i0_tw_role": m["iter0_type_write_role"],
        # iter1 type-write partition  ← KEY SIGNAL
        "i1_tw_type": m["iter1_type_write_type"],
        "i1_tw_scope": m["iter1_type_write_scope"],
        "i1_tw_role": m["iter1_type_write_role"],
        # iter1 parse-write and apply-write partitions
        "i1_pw_type": m["iter1_parse_write_type"],
        "i1_pw_scope": m["iter1_parse_write_scope"],
        "i1_pw_role": m["iter1_parse_write_role"],
        "i1_aw_type": m["iter1_apply_write_type"],
        "i1_aw_scope": m["iter1_apply_write_scope"],
        "i1_aw_role": m["iter1_apply_write_role"],
        # entropy
        "i0_entropy": m["iter0_s4_attn_entropy"],
        "i1_entropy": m["iter1_s4_attn_entropy"],
        # expansion
        "expansion": m["overall_expansion"],
    }


def extract_v2(probe: dict) -> dict:
    m = probe["metrics"]
    pid = probe["probe_id"]
    cat = probe["category"]
    return {
        "probe_id": pid,
        "category": cat,
        "prompt": probe.get("prompt", ""),
        "seq_len": probe.get("seq_len"),
        # iter0 gates
        "i0_type_gate": m["iter0_type_gate_mean"],
        "i0_parse_gate": m["iter0_parse_gate_mean"],
        "i0_apply_gate": m["iter0_apply_gate_mean"],
        # iter1 gates
        "i1_type_gate": m["iter1_type_gate_mean"],
        "i1_parse_gate": m["iter1_parse_gate_mean"],
        "i1_apply_gate": m["iter1_apply_gate_mean"],
        # single register norms
        "i0_reg": m["iter0_register_norm"],
        "i1_reg": m["iter1_register_norm"],
        # entropy
        "i0_entropy": m["iter0_s4_attn_entropy"],
        "i1_entropy": m["iter1_s4_attn_entropy"],
        # expansion
        "expansion": m["overall_expansion"],
    }


# ---------------------------------------------------------------------------
# Category aggregation
# ---------------------------------------------------------------------------

def cat_stats_v3(records: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
    out = {}
    for cat in CATEGORIES:
        g = by_cat.get(cat, [])
        if not g:
            out[cat] = {}
            continue
        out[cat] = {
            "n": len(g),
            # gates iter1
            "i1_type_gate": _mean([x["i1_type_gate"] for x in g]),
            "i1_parse_gate": _mean([x["i1_parse_gate"] for x in g]),
            "i1_apply_gate": _mean([x["i1_apply_gate"] for x in g]),
            # register norms iter1
            "i1_reg_type": _mean([x["i1_reg_type"] for x in g]),
            "i1_reg_scope": _mean([x["i1_reg_scope"] for x in g]),
            "i1_reg_role": _mean([x["i1_reg_role"] for x in g]),
            # iter1 type-write partition
            "i1_tw_type": _mean([x["i1_tw_type"] for x in g]),
            "i1_tw_scope": _mean([x["i1_tw_scope"] for x in g]),
            "i1_tw_role": _mean([x["i1_tw_role"] for x in g]),
            # entropy
            "i1_entropy": _mean([x["i1_entropy"] for x in g]),
            "i0_entropy": _mean([x["i0_entropy"] for x in g]),
            # expansion
            "expansion": _mean([x["expansion"] for x in g]),
            # role dominance: does role norm exceed type+scope avg?
            "role_dominance": _mean(
                [x["i1_reg_role"] / ((x["i1_reg_type"] + x["i1_reg_scope"]) / 2 + 1e-9)
                 for x in g]
            ),
            # write role bias: i1_tw_role vs (i1_tw_type + i1_tw_scope) / 2
            "write_role_bias": _mean(
                [x["i1_tw_role"] / ((x["i1_tw_type"] + x["i1_tw_scope"]) / 2 + 1e-9)
                 for x in g]
            ),
        }
    return out


def cat_stats_v2(records: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
    out = {}
    for cat in CATEGORIES:
        g = by_cat.get(cat, [])
        if not g:
            out[cat] = {}
            continue
        out[cat] = {
            "n": len(g),
            "i1_type_gate": _mean([x["i1_type_gate"] for x in g]),
            "i1_parse_gate": _mean([x["i1_parse_gate"] for x in g]),
            "i1_apply_gate": _mean([x["i1_apply_gate"] for x in g]),
            "i1_reg": _mean([x["i1_reg"] for x in g]),
            "i1_entropy": _mean([x["i1_entropy"] for x in g]),
            "i0_entropy": _mean([x["i0_entropy"] for x in g]),
            "expansion": _mean([x["expansion"] for x in g]),
        }
    return out


# ---------------------------------------------------------------------------
# Minimal pair detection
# ---------------------------------------------------------------------------

def find_minimal_pairs(records: list[dict]) -> list[tuple[dict, dict]]:
    """Return pairs whose probe_ids share a numeric base and differ only by a/b suffix."""
    by_base: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        m = re.match(r"^(.+?)(a|b)$", r["probe_id"])
        if m:
            by_base[m.group(1)].append(r)
    pairs = []
    for base, group in sorted(by_base.items()):
        a = next((r for r in group if r["probe_id"].endswith("a")), None)
        b = next((r for r in group if r["probe_id"].endswith("b")), None)
        if a and b:
            pairs.append((a, b))
    return pairs


def pair_delta_v3(a: dict, b: dict) -> dict:
    """Absolute differences for key v3 signals between pair members."""
    return {
        "delta_i1_type_gate": abs(a["i1_type_gate"] - b["i1_type_gate"]),
        "delta_i1_parse_gate": abs(a["i1_parse_gate"] - b["i1_parse_gate"]),
        "delta_i1_apply_gate": abs(a["i1_apply_gate"] - b["i1_apply_gate"]),
        "delta_i1_reg_role": abs(a["i1_reg_role"] - b["i1_reg_role"]),
        "delta_i1_reg_scope": abs(a["i1_reg_scope"] - b["i1_reg_scope"]),
        "delta_i1_reg_type": abs(a["i1_reg_type"] - b["i1_reg_type"]),
        "delta_i1_tw_role": abs(a["i1_tw_role"] - b["i1_tw_role"]),
        "delta_i1_tw_scope": abs(a["i1_tw_scope"] - b["i1_tw_scope"]),
        "delta_i1_entropy": abs(a["i1_entropy"] - b["i1_entropy"]),
        "delta_expansion": abs(a["expansion"] - b["expansion"]),
        # aggregate internal state distance (sum of key deltas)
        "total_internal_delta": sum([
            abs(a["i1_type_gate"] - b["i1_type_gate"]),
            abs(a["i1_parse_gate"] - b["i1_parse_gate"]),
            abs(a["i1_apply_gate"] - b["i1_apply_gate"]),
            abs(a["i1_reg_role"] - b["i1_reg_role"]),
            abs(a["i1_reg_scope"] - b["i1_reg_scope"]),
            abs(a["i1_reg_type"] - b["i1_reg_type"]),
            abs(a["i1_tw_role"] - b["i1_tw_role"]),
            abs(a["i1_tw_scope"] - b["i1_tw_scope"]),
            abs(a["i1_entropy"] - b["i1_entropy"]),
        ]),
    }


def pair_delta_v2(a: dict, b: dict) -> dict:
    return {
        "delta_i1_type_gate": abs(a["i1_type_gate"] - b["i1_type_gate"]),
        "delta_i1_parse_gate": abs(a["i1_parse_gate"] - b["i1_parse_gate"]),
        "delta_i1_apply_gate": abs(a["i1_apply_gate"] - b["i1_apply_gate"]),
        "delta_i1_reg": abs(a["i1_reg"] - b["i1_reg"]),
        "delta_i1_entropy": abs(a["i1_entropy"] - b["i1_entropy"]),
        "delta_expansion": abs(a["expansion"] - b["expansion"]),
        "total_internal_delta": sum([
            abs(a["i1_type_gate"] - b["i1_type_gate"]),
            abs(a["i1_parse_gate"] - b["i1_parse_gate"]),
            abs(a["i1_apply_gate"] - b["i1_apply_gate"]),
            abs(a["i1_reg"] - b["i1_reg"]),
            abs(a["i1_entropy"] - b["i1_entropy"]),
        ]),
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_header(cols: list[str], widths: list[int]) -> None:
    row = "  ".join(f"{c:>{w}}" for c, w in zip(cols, widths))
    print(row)
    print("  ".join("─" * w for w in widths))


def print_row(vals: list[Any], widths: list[int]) -> None:
    parts = []
    for v, w in zip(vals, widths):
        if isinstance(v, float):
            parts.append(f"{v:{w}.4f}")
        else:
            parts.append(f"{str(v):>{w}}")
    print("  ".join(parts))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Load -----------------------------------------------------------------
    with open(V3_PATH) as f:
        raw_v3 = json.load(f)
    with open(V2_PATH) as f:
        raw_v2 = json.load(f)

    recs_v3 = [extract_v3(p) for p in raw_v3["probes"]]
    recs_v2 = [extract_v2(p) for p in raw_v2["probes"]]

    # Map probe_id → record for cross-version lookup
    v3_map = {r["probe_id"]: r for r in recs_v3}
    v2_map = {r["probe_id"]: r for r in recs_v2}

    cs3 = cat_stats_v3(recs_v3)
    cs2 = cat_stats_v2(recs_v2)

    pairs_v3 = find_minimal_pairs(recs_v3)
    pairs_v2 = [(v2_map[a["probe_id"]], v2_map[b["probe_id"]])
                for a, b in pairs_v3
                if a["probe_id"] in v2_map and b["probe_id"] in v2_map]

    # =========================================================================
    # TABLE 1 — V3 Register Norm Differentiation per Category
    # =========================================================================
    _section("TABLE 1 · V3 iter1 Register Norms per Category (type / scope / role)")
    print("  Hypothesis: role register should carry binding/argument-structure signal")
    print("  and show more variation across categories than type/scope.\n")
    cols = ["Category", "N", "type_norm", "scope_norm", "role_norm", "role_dom", "entropy"]
    widths = [12, 3, 9, 9, 9, 8, 8]
    print_header(cols, widths)
    for cat in CATEGORIES:
        s = cs3.get(cat, {})
        if not s:
            continue
        print_row(
            [
                CATEGORY_ABBREV[cat],
                s["n"],
                s["i1_reg_type"],
                s["i1_reg_scope"],
                s["i1_reg_role"],
                s["role_dominance"],
                s["i1_entropy"],
            ],
            widths,
        )

    # Compute global range for role norm
    all_role = [cs3[c]["i1_reg_role"] for c in CATEGORIES if cs3.get(c) and cs3[c]]
    if all_role:
        print(f"\n  Role norm range: {min(all_role):.4f} – {max(all_role):.4f}  "
              f"(spread = {max(all_role)-min(all_role):.4f})")
        all_scope = [cs3[c]["i1_reg_scope"] for c in CATEGORIES if cs3.get(c) and cs3[c]]
        all_type_n = [cs3[c]["i1_reg_type"] for c in CATEGORIES if cs3.get(c) and cs3[c]]
        print(f"  Scope norm range: {min(all_scope):.4f} – {max(all_scope):.4f}  "
              f"(spread = {max(all_scope)-min(all_scope):.4f})")
        print(f"  Type  norm range: {min(all_type_n):.4f} – {max(all_type_n):.4f}  "
              f"(spread = {max(all_type_n)-min(all_type_n):.4f})")

    # =========================================================================
    # TABLE 2 — V3 Gate Pattern per Category
    # =========================================================================
    _section("TABLE 2 · V3 iter1 Gate Means per Category (type / parse / apply)")
    print("  Lower type_gate = earlier commitment to composition.\n")
    cols = ["Category", "N", "type_g", "parse_g", "apply_g", "Δtype(i1-i0)", "entropy_drop"]
    widths = [12, 3, 7, 7, 7, 12, 11]
    print_header(cols, widths)
    cs3_i0 = {}
    for cat in CATEGORIES:
        g = [r for r in recs_v3 if r["category"] == cat]
        if g:
            cs3_i0[cat] = _mean([r["i0_type_gate"] for r in g])
    for cat in CATEGORIES:
        s = cs3.get(cat, {})
        if not s:
            continue
        delta_type = s["i1_type_gate"] - cs3_i0.get(cat, s["i1_type_gate"])
        entropy_drop = s["i0_entropy"] - s["i1_entropy"]
        print_row(
            [
                CATEGORY_ABBREV[cat],
                s["n"],
                s["i1_type_gate"],
                s["i1_parse_gate"],
                s["i1_apply_gate"],
                delta_type,
                entropy_drop,
            ],
            widths,
        )

    # =========================================================================
    # TABLE 3 — V3 Write-Gate Partition per Category (the key iter1 signal)
    # =========================================================================
    _section("TABLE 3 · V3 iter1 Type→{type,scope,role} Write Partition per Category")
    print("  Each row sums to ≈1 (soft partition over 3 register targets).")
    print("  High write_role = type sub-network is routing to role register.\n")
    cols = ["Category", "N", "→type", "→scope", "→role", "role_bias", "write_role_bar"]
    widths = [12, 3, 7, 7, 7, 9, 12]
    print_header(cols, widths)

    all_role_bias = []
    for cat in CATEGORIES:
        s = cs3.get(cat, {})
        if not s:
            continue
        rb = s.get("write_role_bias", 0.0)
        all_role_bias.append((cat, rb))

    rb_vals = [v for _, v in all_role_bias]
    rb_lo, rb_hi = (min(rb_vals), max(rb_vals)) if rb_vals else (0, 1)

    for cat in CATEGORIES:
        s = cs3.get(cat, {})
        if not s:
            continue
        rb = s.get("write_role_bias", 0.0)
        bar = _bar(rb, rb_lo, rb_hi, 10)
        print_row(
            [CATEGORY_ABBREV[cat], s["n"],
             s["i1_tw_type"], s["i1_tw_scope"], s["i1_tw_role"], rb, bar],
            widths,
        )

    # =========================================================================
    # TABLE 4 — V2 vs V3 Gate Range / Spread Comparison
    # =========================================================================
    _section("TABLE 4 · V2 vs V3 Gate Spread Across All Probes")
    print("  Spread = max - min across all probes. Higher spread = more discrimination.\n")

    def gate_spread(records: list[dict], key: str) -> tuple[float, float, float]:
        vals = [r[key] for r in records]
        return min(vals), max(vals), max(vals) - min(vals)

    comparisons = [
        ("iter1 type_gate", "i1_type_gate", "i1_type_gate"),
        ("iter1 parse_gate", "i1_parse_gate", "i1_parse_gate"),
        ("iter1 apply_gate", "i1_apply_gate", "i1_apply_gate"),
        ("iter1 entropy",    "i1_entropy",    "i1_entropy"),
        ("expansion",        "expansion",     "expansion"),
    ]

    cols = ["Metric", "V2 lo", "V2 hi", "V2 spread", "V3 lo", "V3 hi", "V3 spread", "ratio"]
    widths = [16, 7, 7, 9, 7, 7, 9, 6]
    print_header(cols, widths)
    for label, v2k, v3k in comparisons:
        v2_lo, v2_hi, v2_sp = gate_spread(recs_v2, v2k)
        v3_lo, v3_hi, v3_sp = gate_spread(recs_v3, v3k)
        ratio = v3_sp / v2_sp if v2_sp > 1e-9 else float("inf")
        print_row([label, v2_lo, v2_hi, v2_sp, v3_lo, v3_hi, v3_sp, ratio], widths)

    # Also compare register norms (v3 has 3, v2 has 1)
    print()
    print("  V3 register norms (per-register spread):")
    for reg_key, label in [("i1_reg_type", "type"), ("i1_reg_scope", "scope"), ("i1_reg_role", "role")]:
        lo, hi, sp = gate_spread(recs_v3, reg_key)
        print(f"    iter1 reg_{label:6s}  lo={lo:.4f}  hi={hi:.4f}  spread={sp:.4f}")
    lo2, hi2, sp2 = gate_spread(recs_v2, "i1_reg")
    print(f"  V2 single register:  lo={lo2:.4f}  hi={hi2:.4f}  spread={sp2:.4f}")

    # =========================================================================
    # TABLE 5 — Minimal Pair Analysis
    # =========================================================================
    _section("TABLE 5 · Minimal Pair Analysis  (a vs b probe, same base sentence)")
    print("  Does v3's richer internal state differentiate structurally related sentences")
    print("  more than v2?  Higher Δ = model represents the two as more different.\n")

    cols = ["Pair", "Cat", "v3 Δ_role_reg", "v3 Δ_tw_role", "v3 Δ_entropy", "v3 total", "v2 total", "v3>v2?"]
    widths = [14, 8, 13, 12, 11, 9, 9, 7]
    print_header(cols, widths)

    pair_summary = []
    for (a3, b3), (a2, b2) in zip(pairs_v3, pairs_v2):
        d3 = pair_delta_v3(a3, b3)
        d2 = pair_delta_v2(a2, b2)
        pair_id = a3["probe_id"][:-1]  # strip trailing 'a'
        cat_ab = CATEGORY_ABBREV.get(a3["category"], a3["category"])
        v3_wins = d3["total_internal_delta"] > d2["total_internal_delta"]
        print_row(
            [
                pair_id,
                cat_ab,
                d3["delta_i1_reg_role"],
                d3["delta_i1_tw_role"],
                d3["delta_i1_entropy"],
                d3["total_internal_delta"],
                d2["total_internal_delta"],
                "YES" if v3_wins else "no",
            ],
            widths,
        )
        pair_summary.append({
            "pair": pair_id,
            "category": a3["category"],
            "v3_total_delta": d3["total_internal_delta"],
            "v2_total_delta": d2["total_internal_delta"],
            "v3_wins": v3_wins,
            "v3_delta_role_reg": d3["delta_i1_reg_role"],
            "v3_delta_tw_role": d3["delta_i1_tw_role"],
            "v3_delta_entropy": d3["delta_i1_entropy"],
            "v2_delta_entropy": d2["delta_i1_entropy"],
            "probe_a": {
                "id": a3["probe_id"],
                "prompt": a3["prompt"],
            },
            "probe_b": {
                "id": b3["probe_id"],
                "prompt": b3["prompt"],
            },
        })

    v3_win_count = sum(1 for p in pair_summary if p["v3_wins"])
    print(f"\n  V3 differentiates pair better: {v3_win_count}/{len(pair_summary)} pairs")

    # =========================================================================
    # TABLE 6 — Full per-probe listing (v3)
    # =========================================================================
    _section("TABLE 6 · Full Per-Probe V3 Metrics")
    print()
    cols = ["Probe", "Cat", "i1_tg", "i1_pg", "i1_ag", "reg_T", "reg_S", "reg_R", "tw_R", "entropy1"]
    widths = [16, 8, 7, 7, 7, 7, 7, 7, 7, 9]
    print_header(cols, widths)
    for r in recs_v3:
        print_row(
            [
                r["probe_id"],
                CATEGORY_ABBREV.get(r["category"], r["category"]),
                r["i1_type_gate"],
                r["i1_parse_gate"],
                r["i1_apply_gate"],
                r["i1_reg_type"],
                r["i1_reg_scope"],
                r["i1_reg_role"],
                r["i1_tw_role"],
                r["i1_entropy"],
            ],
            widths,
        )

    # =========================================================================
    # TABLE 7 — Full per-probe listing (v2)
    # =========================================================================
    _section("TABLE 7 · Full Per-Probe V2 Metrics")
    print()
    cols = ["Probe", "Cat", "i1_tg", "i1_pg", "i1_ag", "reg_norm", "entropy1", "expansion"]
    widths = [16, 8, 7, 7, 7, 9, 9, 10]
    print_header(cols, widths)
    for r in recs_v2:
        print_row(
            [
                r["probe_id"],
                CATEGORY_ABBREV.get(r["category"], r["category"]),
                r["i1_type_gate"],
                r["i1_parse_gate"],
                r["i1_apply_gate"],
                r["i1_reg"],
                r["i1_entropy"],
                r["expansion"],
            ],
            widths,
        )

    # =========================================================================
    # Interpretation summary
    # =========================================================================
    _section("INTERPRETATION SUMMARY")

    # Find category with highest and lowest role norms
    sorted_role = sorted(
        [(cat, cs3[cat]["i1_reg_role"]) for cat in CATEGORIES if cs3.get(cat) and cs3[cat]],
        key=lambda x: x[1],
    )
    hi_role_cat, hi_role_val = sorted_role[-1]
    lo_role_cat, lo_role_val = sorted_role[0]

    sorted_rb = sorted(
        [(cat, cs3[cat].get("write_role_bias", 0)) for cat in CATEGORIES if cs3.get(cat) and cs3[cat]],
        key=lambda x: x[1],
    )
    hi_rb_cat, hi_rb_val = sorted_rb[-1]
    lo_rb_cat, lo_rb_val = sorted_rb[0]

    # V3 overall gate spread
    v3_type_spread = gate_spread(recs_v3, "i1_type_gate")[2]
    v2_type_spread = gate_spread(recs_v2, "i1_type_gate")[2]

    print(f"""
  1. REGISTER DIFFERENTIATION
     Highest role register activation: {CATEGORY_ABBREV[hi_role_cat]} ({hi_role_val:.4f})
     Lowest  role register activation: {CATEGORY_ABBREV[lo_role_cat]} ({lo_role_val:.4f})
     Role spread across categories:    {hi_role_val - lo_role_val:.4f}

  2. WRITE-GATE PARTITION SIGNAL
     Highest role write bias: {CATEGORY_ABBREV[hi_rb_cat]} (role_bias={hi_rb_val:.4f})
     Lowest  role write bias: {CATEGORY_ABBREV[lo_rb_cat]} (role_bias={lo_rb_val:.4f})
     → iter1 type→role write gate varies by {hi_rb_val - lo_rb_val:.4f} across categories.

  3. GATE SPREAD  V3 vs V2 (iter1 type gate)
     V3 spread={v3_type_spread:.4f}   V2 spread={v2_type_spread:.4f}
     → V3 is {"MORE" if v3_type_spread > v2_type_spread else "LESS"} discriminating on the type gate.

  4. MINIMAL PAIRS
     V3 differentiates {v3_win_count}/{len(pair_summary)} pairs more strongly than V2.
""")

    # =========================================================================
    # Write JSON summary
    # =========================================================================
    summary: dict[str, Any] = {
        "meta": {
            "v3_file": str(V3_PATH),
            "v2_file": str(V2_PATH),
            "v3_model": raw_v3.get("model"),
            "v3_step": raw_v3.get("step"),
            "v3_n_probes": len(recs_v3),
            "v2_n_probes": len(recs_v2),
        },
        "category_stats_v3": {
            cat: {k: round(v, 6) if isinstance(v, float) else v
                  for k, v in cs3[cat].items()}
            for cat in CATEGORIES if cs3.get(cat) and cs3[cat]
        },
        "category_stats_v2": {
            cat: {k: round(v, 6) if isinstance(v, float) else v
                  for k, v in cs2[cat].items()}
            for cat in CATEGORIES if cs2.get(cat) and cs2[cat]
        },
        "minimal_pair_analysis": pair_summary,
        "gate_spread": {
            "v3_i1_type_gate": round(gate_spread(recs_v3, "i1_type_gate")[2], 6),
            "v2_i1_type_gate": round(gate_spread(recs_v2, "i1_type_gate")[2], 6),
            "v3_i1_parse_gate": round(gate_spread(recs_v3, "i1_parse_gate")[2], 6),
            "v2_i1_parse_gate": round(gate_spread(recs_v2, "i1_parse_gate")[2], 6),
            "v3_i1_apply_gate": round(gate_spread(recs_v3, "i1_apply_gate")[2], 6),
            "v2_i1_apply_gate": round(gate_spread(recs_v2, "i1_apply_gate")[2], 6),
            "v3_i1_entropy": round(gate_spread(recs_v3, "i1_entropy")[2], 6),
            "v2_i1_entropy": round(gate_spread(recs_v2, "i1_entropy")[2], 6),
            "v3_reg_role": round(gate_spread(recs_v3, "i1_reg_role")[2], 6),
            "v3_reg_scope": round(gate_spread(recs_v3, "i1_reg_scope")[2], 6),
            "v3_reg_type": round(gate_spread(recs_v3, "i1_reg_type")[2], 6),
            "v2_reg": round(gate_spread(recs_v2, "i1_reg")[2], 6),
        },
        "per_probe_v3": [
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
            for r in recs_v3
        ],
        "per_probe_v2": [
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
            for r in recs_v2
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  JSON summary written → {OUT_PATH}\n")


if __name__ == "__main__":
    main()
