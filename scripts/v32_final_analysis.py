"""
v3.2 Final Analysis — Session 018
Full 10-checkpoint trajectory + head-to-head vs v3 at step 10k.

Comparable signals between v3 and v3.2:
  v3: type/parse/apply gates → v3.2: prep/converge/consolidate gates
  Both: s4_attn_entropy, register_after_s4, output_norm, overall_expansion
  Both: iter0/iter1 structure (v3 has 2 iters, v3.2 has 2 iters)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics

BASE = Path(__file__).resolve().parent.parent

# ── Helpers ──────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

def category_means(probes, metric_fn):
    """Group probes by category, return {category: mean_value}."""
    by_cat = defaultdict(list)
    for p in probes:
        cat = p.get("category", p["probe_id"].rsplit("-", 1)[0])
        val = metric_fn(p)
        if val is not None:
            by_cat[cat].append(val)
    return {c: statistics.mean(vs) for c, vs in by_cat.items()}

def binding_type(probe_id):
    """Extract binding type from probe_id like 'bind-scope-01a' → 'scope'."""
    parts = probe_id.split("-")
    if len(parts) >= 3:
        return parts[1]
    return "unknown"

def binding_means(probes, metric_fn):
    """Group binding probes by type, return {type: mean_value}."""
    by_type = defaultdict(list)
    for p in probes:
        bt = binding_type(p["probe_id"])
        val = metric_fn(p)
        if val is not None:
            by_type[bt].append(val)
    return {t: statistics.mean(vs) for t, vs in by_type.items()}


# ── v3.2 metric extractors ──────────────────────────────────────────

def v32_prep_gate(p):
    return p["metrics"].get("iter0_prep_gate_mean")

def v32_converge_gate(p):
    return p["metrics"].get("iter0_converge_gate_mean")

def v32_consolidate_gate(p):
    return p["metrics"].get("iter0_consolidate_gate_mean")

def v32_output_norm(p):
    return p["metrics"].get("output_norm")

def v32_role_register(p):
    return p["metrics"].get("iter0_register_role_norm")

def v32_overall_expansion(p):
    return p["metrics"].get("overall_expansion")

def v32_s4_entropy(p):
    return p["metrics"].get("s4_attn_entropy")

def v32_register_after_s4(p):
    return p["metrics"].get("register_after_s4")


# ── v3 metric extractors ────────────────────────────────────────────

def v3_type_gate(p):
    return p["metrics"].get("iter0_type_gate_mean")

def v3_parse_gate(p):
    return p["metrics"].get("iter0_parse_gate_mean")

def v3_apply_gate(p):
    return p["metrics"].get("iter0_apply_gate_mean")

def v3_output_norm(p):
    return p["metrics"].get("output_norm")

def v3_s4_entropy(p):
    return p["metrics"].get("s4_attn_entropy")

def v3_register_after_s4(p):
    return p["metrics"].get("register_after_s4")

def v3_overall_expansion(p):
    return p["metrics"].get("overall_expansion")


# ══════════════════════════════════════════════════════════════════════
# PART 1: v3.2 Full Trajectory (compile-gradient)
# ══════════════════════════════════════════════════════════════════════

print("=" * 80)
print("PART 1: v3.2 COMPILE-GRADIENT TRAJECTORY (steps 1k → 10k)")
print("=" * 80)

steps = list(range(1000, 11000, 1000))
cg_data = {}
for step in steps:
    path = BASE / f"results/compile-gradient/vsm_probe_step_{step:06d}_v3.2.json"
    if path.exists():
        cg_data[step] = load_json(path)

# Compile-gradient category means per step
print("\n── Gate Means by Category ──")
print(f"{'Step':>6} │ {'strong':>7} {'medium':>7} {'weak':>7} {'null':>7} {'anti':>7} │ {'s-a':>7} {'s-w':>7}")
print("─" * 80)

prep_trajectory = {}
converge_trajectory = {}
consolidate_trajectory = {}
role_trajectory = {}
output_trajectory = {}
expansion_trajectory = {}
entropy_trajectory = {}
register_trajectory = {}

for step in sorted(cg_data.keys()):
    d = cg_data[step]
    
    # Prep gate means by category
    prep_cats = category_means(d["probes"], v32_prep_gate)
    conv_cats = category_means(d["probes"], v32_converge_gate)
    cons_cats = category_means(d["probes"], v32_consolidate_gate)
    role_cats = category_means(d["probes"], v32_role_register)
    out_cats = category_means(d["probes"], v32_output_norm)
    exp_cats = category_means(d["probes"], v32_overall_expansion)
    ent_cats = category_means(d["probes"], v32_s4_entropy)
    reg_cats = category_means(d["probes"], v32_register_after_s4)
    
    prep_trajectory[step] = prep_cats
    converge_trajectory[step] = conv_cats
    consolidate_trajectory[step] = cons_cats
    role_trajectory[step] = role_cats
    output_trajectory[step] = out_cats
    expansion_trajectory[step] = exp_cats
    entropy_trajectory[step] = ent_cats
    register_trajectory[step] = reg_cats

# Print prep gate trajectory
print("\n  PREP GATE (iter0):")
print(f"  {'Step':>6} │ {'strong':>7} {'medium':>7} {'weak':>7} {'null':>7} {'anti':>7} │ {'s-a':>7}")
print("  " + "─" * 72)
for step in sorted(prep_trajectory.keys()):
    cats = prep_trajectory[step]
    s = cats.get("strong_compile", 0)
    m = cats.get("medium_compile", 0)
    w = cats.get("weak_compile", 0)
    n = cats.get("null", 0)
    a = cats.get("anti_compile", 0)
    print(f"  {step:>6} │ {s:>7.4f} {m:>7.4f} {w:>7.4f} {n:>7.4f} {a:>7.4f} │ {s-a:>+7.4f}")

# Print converge gate trajectory
print("\n  CONVERGE GATE (iter0):")
print(f"  {'Step':>6} │ {'strong':>7} {'medium':>7} {'weak':>7} {'null':>7} {'anti':>7} │ {'s-a':>7}")
print("  " + "─" * 72)
for step in sorted(converge_trajectory.keys()):
    cats = converge_trajectory[step]
    s = cats.get("strong_compile", 0)
    m = cats.get("medium_compile", 0)
    w = cats.get("weak_compile", 0)
    n = cats.get("null", 0)
    a = cats.get("anti_compile", 0)
    print(f"  {step:>6} │ {s:>7.4f} {m:>7.4f} {w:>7.4f} {n:>7.4f} {a:>7.4f} │ {s-a:>+7.4f}")

# Print consolidate gate trajectory
print("\n  CONSOLIDATE GATE (iter0):")
print(f"  {'Step':>6} │ {'strong':>7} {'medium':>7} {'weak':>7} {'null':>7} {'anti':>7} │ {'s-a':>7}")
print("  " + "─" * 72)
for step in sorted(consolidate_trajectory.keys()):
    cats = consolidate_trajectory[step]
    s = cats.get("strong_compile", 0)
    m = cats.get("medium_compile", 0)
    w = cats.get("weak_compile", 0)
    n = cats.get("null", 0)
    a = cats.get("anti_compile", 0)
    print(f"  {step:>6} │ {s:>7.4f} {m:>7.4f} {w:>7.4f} {n:>7.4f} {a:>7.4f} │ {s-a:>+7.4f}")

# Role register trajectory
print("\n  ROLE REGISTER NORM (iter0):")
print(f"  {'Step':>6} │ {'strong':>7} {'medium':>7} {'weak':>7} {'null':>7} {'anti':>7} │ {'s-a':>7} {'range':>7}")
print("  " + "─" * 80)
for step in sorted(role_trajectory.keys()):
    cats = role_trajectory[step]
    s = cats.get("strong_compile", 0)
    m = cats.get("medium_compile", 0)
    w = cats.get("weak_compile", 0)
    n = cats.get("null", 0)
    a = cats.get("anti_compile", 0)
    vals = [v for v in [s, m, w, n, a] if v != 0]
    rng = max(vals) - min(vals) if vals else 0
    print(f"  {step:>6} │ {s:>7.2f} {m:>7.2f} {w:>7.2f} {n:>7.2f} {a:>7.2f} │ {s-a:>+7.2f} {rng:>7.2f}")

# Output norm trajectory
print("\n  OUTPUT NORM:")
print(f"  {'Step':>6} │ {'strong':>7} {'medium':>7} {'weak':>7} {'null':>7} {'anti':>7} │ {'range':>7}")
print("  " + "─" * 72)
for step in sorted(output_trajectory.keys()):
    cats = output_trajectory[step]
    s = cats.get("strong_compile", 0)
    m = cats.get("medium_compile", 0)
    w = cats.get("weak_compile", 0)
    n = cats.get("null", 0)
    a = cats.get("anti_compile", 0)
    vals = [v for v in [s, m, w, n, a] if v != 0]
    rng = max(vals) - min(vals) if vals else 0
    print(f"  {step:>6} │ {s:>7.2f} {m:>7.2f} {w:>7.2f} {n:>7.2f} {a:>7.2f} │ {rng:>7.2f}")


# ══════════════════════════════════════════════════════════════════════
# PART 2: v3.2 Binding Trajectory
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("PART 2: v3.2 BINDING TRAJECTORY (steps 1k → 10k)")
print("=" * 80)

bind_data = {}
for step in steps:
    path = BASE / f"results/binding/vsm_probe_step_{step:06d}_v3.2.json"
    if path.exists():
        bind_data[step] = load_json(path)

# Converge gate by binding type
print("\n  CONVERGE GATE by binding type:")
bind_types_order = ["scope", "var", "ana", "ctrl", "rel", "neg", "embed"]
header = f"  {'Step':>6} │ " + " ".join(f"{t:>7}" for t in bind_types_order) + " │ {'range':>7}"
print(header)
print("  " + "─" * (len(header) - 2))

bind_conv_trajectory = {}
bind_cons_trajectory = {}
bind_role_trajectory = {}

for step in sorted(bind_data.keys()):
    d = bind_data[step]
    conv_types = binding_means(d["probes"], v32_converge_gate)
    cons_types = binding_means(d["probes"], v32_consolidate_gate)
    role_types = binding_means(d["probes"], v32_role_register)
    
    bind_conv_trajectory[step] = conv_types
    bind_cons_trajectory[step] = cons_types
    bind_role_trajectory[step] = role_types
    
    vals = [conv_types.get(t, 0) for t in bind_types_order]
    nonzero = [v for v in vals if v != 0]
    rng = max(nonzero) - min(nonzero) if len(nonzero) > 1 else 0
    row = " ".join(f"{v:>7.4f}" for v in vals)
    print(f"  {step:>6} │ {row} │ {rng:>7.4f}")

# Consolidate gate by binding type
print("\n  CONSOLIDATE GATE by binding type:")
header = f"  {'Step':>6} │ " + " ".join(f"{t:>7}" for t in bind_types_order) + " │ {'range':>7}"
print(header)
print("  " + "─" * (len(header) - 2))

for step in sorted(bind_cons_trajectory.keys()):
    cons_types = bind_cons_trajectory[step]
    vals = [cons_types.get(t, 0) for t in bind_types_order]
    nonzero = [v for v in vals if v != 0]
    rng = max(nonzero) - min(nonzero) if len(nonzero) > 1 else 0
    row = " ".join(f"{v:>7.4f}" for v in vals)
    print(f"  {step:>6} │ {row} │ {rng:>7.4f}")

# Role register by binding type
print("\n  ROLE REGISTER by binding type:")
header = f"  {'Step':>6} │ " + " ".join(f"{t:>7}" for t in bind_types_order) + " │ {'range':>7}"
print(header)
print("  " + "─" * (len(header) - 2))

for step in sorted(bind_role_trajectory.keys()):
    role_types = bind_role_trajectory[step]
    vals = [role_types.get(t, 0) for t in bind_types_order]
    nonzero = [v for v in vals if v != 0]
    rng = max(nonzero) - min(nonzero) if len(nonzero) > 1 else 0
    row = " ".join(f"{v:>7.2f}" for v in vals)
    print(f"  {step:>6} │ {row} │ {rng:>7.2f}")


# ══════════════════════════════════════════════════════════════════════
# PART 3: HEAD-TO-HEAD v3 vs v3.2 at step 10k
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("PART 3: HEAD-TO-HEAD — v3 vs v3.2 at step 10k")
print("=" * 80)

v3_cg = load_json(BASE / "results/compile-gradient/vsm_probe_step_010000.json")
v32_cg = load_json(BASE / "results/compile-gradient/vsm_probe_step_010000_v3.2.json")
v3_bind = load_json(BASE / "results/binding/vsm_probe_step_010000.json")
v32_bind = load_json(BASE / "results/binding/vsm_probe_step_010000_v3.2.json")

# Comparable signals: s4_entropy, output_norm, overall_expansion
print("\n  COMPILE-GRADIENT — Comparable Signals:")
print(f"  {'Signal':>25} │ {'v3':>10} {'v3.2':>10} {'Δ':>10} {'%Δ':>8}")
print("  " + "─" * 68)

for label, v3_fn, v32_fn in [
    ("S4 entropy (strong)", 
     lambda: statistics.mean([v3_s4_entropy(p) for p in v3_cg["probes"] if p["category"] == "strong_compile"]),
     lambda: statistics.mean([v32_s4_entropy(p) for p in v32_cg["probes"] if p["category"] == "strong_compile"])),
    ("S4 entropy (anti)",
     lambda: statistics.mean([v3_s4_entropy(p) for p in v3_cg["probes"] if p["category"] == "anti_compile"]),
     lambda: statistics.mean([v32_s4_entropy(p) for p in v32_cg["probes"] if p["category"] == "anti_compile"])),
    ("Output norm (strong)",
     lambda: statistics.mean([v3_output_norm(p) for p in v3_cg["probes"] if p["category"] == "strong_compile"]),
     lambda: statistics.mean([v32_output_norm(p) for p in v32_cg["probes"] if p["category"] == "strong_compile"])),
    ("Output norm (anti)",
     lambda: statistics.mean([v3_output_norm(p) for p in v3_cg["probes"] if p["category"] == "anti_compile"]),
     lambda: statistics.mean([v32_output_norm(p) for p in v32_cg["probes"] if p["category"] == "anti_compile"])),
    ("Output norm range",
     lambda: (statistics.mean([v3_output_norm(p) for p in v3_cg["probes"] if p["category"] == "strong_compile"]) -
              statistics.mean([v3_output_norm(p) for p in v3_cg["probes"] if p["category"] == "anti_compile"])),
     lambda: (statistics.mean([v32_output_norm(p) for p in v32_cg["probes"] if p["category"] == "strong_compile"]) -
              statistics.mean([v32_output_norm(p) for p in v32_cg["probes"] if p["category"] == "anti_compile"]))),
    ("Expansion (strong)",
     lambda: statistics.mean([v3_overall_expansion(p) for p in v3_cg["probes"] if p["category"] == "strong_compile"]),
     lambda: statistics.mean([v32_overall_expansion(p) for p in v32_cg["probes"] if p["category"] == "strong_compile"])),
    ("Expansion (anti)",
     lambda: statistics.mean([v3_overall_expansion(p) for p in v3_cg["probes"] if p["category"] == "anti_compile"]),
     lambda: statistics.mean([v32_overall_expansion(p) for p in v32_cg["probes"] if p["category"] == "anti_compile"])),
]:
    v3_val = v3_fn()
    v32_val = v32_fn()
    delta = v32_val - v3_val
    pct = (delta / abs(v3_val) * 100) if v3_val != 0 else 0
    print(f"  {label:>25} │ {v3_val:>10.4f} {v32_val:>10.4f} {delta:>+10.4f} {pct:>+7.1f}%")

# v3 gate means vs v3.2 gate means (architectural analog comparison)
print("\n  GATE ARCHITECTURE COMPARISON (iter0 means):")
print(f"  {'v3 Gate':>15} {'v3 val':>8} │ {'v3.2 Gate':>15} {'v3.2 val':>8} │ {'Signal':>20}")
print("  " + "─" * 78)

for cat in ["strong_compile", "anti_compile"]:
    cat_label = "strong" if "strong" in cat else "anti"
    v3_probes = [p for p in v3_cg["probes"] if p["category"] == cat]
    v32_probes = [p for p in v32_cg["probes"] if p["category"] == cat]
    
    v3_type = statistics.mean([v3_type_gate(p) for p in v3_probes])
    v3_parse = statistics.mean([v3_parse_gate(p) for p in v3_probes])
    v3_apply = statistics.mean([v3_apply_gate(p) for p in v3_probes])
    
    v32_prep = statistics.mean([v32_prep_gate(p) for p in v32_probes])
    v32_conv = statistics.mean([v32_converge_gate(p) for p in v32_probes])
    v32_cons = statistics.mean([v32_consolidate_gate(p) for p in v32_probes])
    
    print(f"  {'type('+cat_label+')':>15} {v3_type:>8.4f} │ {'prep('+cat_label+')':>15} {v32_prep:>8.4f} │ {'initial processing':>20}")
    print(f"  {'parse('+cat_label+')':>15} {v3_parse:>8.4f} │ {'converge('+cat_label+')':>15} {v32_conv:>8.4f} │ {'structural merge':>20}")
    print(f"  {'apply('+cat_label+')':>15} {v3_apply:>8.4f} │ {'consol('+cat_label+')':>15} {v32_cons:>8.4f} │ {'final gating':>20}")
    print()

# Binding comparison — v3 vs v3.2 at 10k
print("\n  BINDING — v3 vs v3.2 at step 10k:")
print(f"  {'Type':>8} │ v3 parse_gate  v3.2 conv_gate │ v3 apply_gate  v3.2 cons_gate │ v3 out_norm  v3.2 out_norm")
print("  " + "─" * 100)

v3_bind_parse = binding_means(v3_bind["probes"], v3_parse_gate)
v3_bind_apply = binding_means(v3_bind["probes"], v3_apply_gate)
v3_bind_out = binding_means(v3_bind["probes"], v3_output_norm)
v32_bind_conv = binding_means(v32_bind["probes"], v32_converge_gate)
v32_bind_cons = binding_means(v32_bind["probes"], v32_consolidate_gate)
v32_bind_out = binding_means(v32_bind["probes"], v32_output_norm)

for bt in bind_types_order:
    v3p = v3_bind_parse.get(bt, 0)
    v32c = v32_bind_conv.get(bt, 0)
    v3a = v3_bind_apply.get(bt, 0)
    v32cs = v32_bind_cons.get(bt, 0)
    v3o = v3_bind_out.get(bt, 0)
    v32o = v32_bind_out.get(bt, 0)
    print(f"  {bt:>8} │ {v3p:>13.4f}  {v32c:>13.4f} │ {v3a:>13.4f}  {v32cs:>13.4f} │ {v3o:>11.2f}  {v32o:>11.2f}")


# ══════════════════════════════════════════════════════════════════════
# PART 4: Summary Signals for Termination Assessment
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("PART 4: TERMINATION ASSESSMENT — KEY TRAJECTORY SIGNALS")
print("=" * 80)

print("\n  CONSOLIDATED TRAJECTORY TABLE:")
print(f"  {'Step':>6} │ {'prep s-a':>8} {'conv s-a':>8} {'cons s-a':>8} │ {'role rng':>8} {'out rng':>8} │ {'bind c rng':>10} {'bind k rng':>10}")
print("  " + "─" * 90)

for step in sorted(cg_data.keys()):
    pc = prep_trajectory[step]
    cc = converge_trajectory[step]
    kc = consolidate_trajectory[step]
    rc = role_trajectory[step]
    oc = output_trajectory[step]
    
    prep_sa = pc.get("strong_compile", 0) - pc.get("anti_compile", 0)
    conv_sa = cc.get("strong_compile", 0) - cc.get("anti_compile", 0)
    cons_sa = kc.get("strong_compile", 0) - kc.get("anti_compile", 0)
    
    role_vals = [rc.get(c, 0) for c in ["strong_compile", "medium_compile", "weak_compile", "null", "anti_compile"]]
    role_nz = [v for v in role_vals if v != 0]
    role_rng = max(role_nz) - min(role_nz) if role_nz else 0
    
    out_vals = [oc.get(c, 0) for c in ["strong_compile", "medium_compile", "weak_compile", "null", "anti_compile"]]
    out_nz = [v for v in out_vals if v != 0]
    out_rng = max(out_nz) - min(out_nz) if out_nz else 0
    
    # Binding ranges (if step has binding data)
    bc_rng = 0
    bk_rng = 0
    if step in bind_conv_trajectory:
        bc_vals = [bind_conv_trajectory[step].get(t, 0) for t in bind_types_order]
        bc_nz = [v for v in bc_vals if v != 0]
        bc_rng = max(bc_nz) - min(bc_nz) if len(bc_nz) > 1 else 0
        
        bk_vals = [bind_cons_trajectory[step].get(t, 0) for t in bind_types_order]
        bk_nz = [v for v in bk_vals if v != 0]
        bk_rng = max(bk_nz) - min(bk_nz) if len(bk_nz) > 1 else 0
    
    print(f"  {step:>6} │ {prep_sa:>+8.4f} {conv_sa:>+8.4f} {cons_sa:>+8.4f} │ {role_rng:>8.2f} {out_rng:>8.2f} │ {bc_rng:>10.4f} {bk_rng:>10.4f}")

# Binding type rankings at 10k
print("\n  BINDING TYPE RANKINGS at step 10k:")
if 10000 in bind_conv_trajectory:
    conv_10k = bind_conv_trajectory[10000]
    cons_10k = bind_cons_trajectory[10000]
    role_10k = bind_role_trajectory[10000]
    
    print(f"    Converge gate:     {' > '.join(f'{t}({v:.3f})' for t, v in sorted(conv_10k.items(), key=lambda x: -x[1]))}")
    print(f"    Consolidate gate:  {' > '.join(f'{t}({v:.3f})' for t, v in sorted(cons_10k.items(), key=lambda x: -x[1]))}")
    print(f"    Role register:     {' > '.join(f'{t}({v:.2f})' for t, v in sorted(role_10k.items(), key=lambda x: -x[1]))}")

print("\n  Done.")
