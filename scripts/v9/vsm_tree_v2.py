"""
v9 — VSM Tree v2: Bottleneck diagnosis experiments

Hypothesis: the v1 VSM tree's additive collapse (op + c1 + c2 → d_model)
destroys the separability needed to recover individual arg values. The
mix layers can't undo the superposition.

This script tests four interventions:
  A) Concatenate [op; c1; c2] instead of add, then project down
  B) Capacity sweep: d_model {64, 128, 256}
  C) More mix layers: 2 vs 4
  D) Value residual: direct shortcut from value embeddings to arg heads

All variants share the same training loop and evaluation.

License: MIT
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
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
# Types and ops (shared with v1)
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
# Expression tree utilities (from v1)
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


def _node_depth(node):
    if isinstance(node, int):
        return 0
    return 1 + max(_node_depth(node[1]), _node_depth(node[2]))


# ══════════════════════════════════════════════════════════════════════
# VSM Node v2 — configurable architecture
# ══════════════════════════════════════════════════════════════════════

@dataclass
class VSMConfig:
    d_model: int = 64
    n_ops: int = N_OPS
    n_types: int = N_TYPES
    max_val: int = 100
    val_embed_range: int = 200
    n_mix_layers: int = 2
    concat_inputs: bool = False      # True = concat [op;c1;c2], False = add
    value_residual: bool = False     # True = shortcut from value embeds to arg heads


class VSMNodeV2(nn.Module):
    """VSM node with configurable input fusion and residual paths."""

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

        if config.concat_inputs:
            # Concat [op; c1; c2] = 3*d → project to d
            # Use nn.Linear (float) for this projection — it's a bottleneck
            # that needs full gradient, not ternary routing
            self.input_proj = nn.Linear(3 * d, d)
        else:
            # Additive: use position embeddings to disambiguate
            self.pos_embed = nn.Embedding(3, d)

        # Mix layers (ternary) — variable count
        self.mix_layers = [TernaryLinear(d, d, pre_norm=True)
                           for _ in range(config.n_mix_layers)]

        # Type head
        type_dim = ((config.n_types + 15) // 16) * 16
        self.type_proj = TernaryLinear(d, type_dim, pre_norm=True)
        self._type_dim = config.n_types

        # Routing heads
        op_dim = ((config.n_ops + 15) // 16) * 16
        arg_dim = ((config.max_val + 15) // 16) * 16
        self.op_proj = TernaryLinear(d, op_dim, pre_norm=True)
        self._op_dim = config.n_ops
        self._arg_dim = config.max_val

        if config.value_residual:
            # Value residual: concat [mixed; val1_embed; val2_embed] → arg logits
            # Each arg head sees the mixed representation + raw value embeddings
            self.arg1_proj = nn.Linear(d + d, arg_dim)  # [mixed; c1_val_embed]
            self.arg2_proj = nn.Linear(d + d, arg_dim)  # [mixed; c2_val_embed]
        else:
            self.arg1_proj = TernaryLinear(d, arg_dim, pre_norm=True)
            self.arg2_proj = TernaryLinear(d, arg_dim, pre_norm=True)

    def _val_idx(self, val):
        return mx.clip(val + self._val_offset, 0, self.config.val_embed_range - 1).astype(mx.int32)

    def forward(self, op_codes, c1_types, c1_vals, c2_types, c2_vals):
        d = self.config.d_model
        config = self.config

        # Embed children
        c1_val_embed = self.val_embed(self._val_idx(c1_vals))
        c2_val_embed = self.val_embed(self._val_idx(c2_vals))
        c1_repr = self.type_embed(c1_types) + c1_val_embed
        c2_repr = self.type_embed(c2_types) + c2_val_embed
        op_repr = self.op_embed(op_codes)

        if config.concat_inputs:
            # Concatenate and project: preserves separability
            x = self.input_proj(mx.concatenate([op_repr, c1_repr, c2_repr], axis=-1))
        else:
            # Additive with position embeddings (v1 style)
            pos = self.pos_embed(mx.arange(3))
            x = (op_repr + pos[0]) + (c1_repr + pos[1]) + (c2_repr + pos[2])

        # Mix layers with residual connections
        for mix in self.mix_layers:
            x = x + mix(x)

        # Type
        type_logits = self.type_proj(x)[:, :self._type_dim]
        pred_type = mx.argmax(type_logits, axis=-1).astype(mx.int32)

        # Op routing
        op_logits = self.op_proj(x)[:, :self._op_dim]
        pred_op = mx.argmax(op_logits, axis=-1).astype(mx.int32)

        # Arg routing — with optional value residual
        if config.value_residual:
            a1_input = mx.concatenate([x, c1_val_embed], axis=-1)
            a2_input = mx.concatenate([x, c2_val_embed], axis=-1)
            a1_logits = self.arg1_proj(a1_input)[:, :self._arg_dim]
            a2_logits = self.arg2_proj(a2_input)[:, :self._arg_dim]
        else:
            a1_logits = self.arg1_proj(x)[:, :self._arg_dim]
            a2_logits = self.arg2_proj(x)[:, :self._arg_dim]

        pred_a1 = mx.argmax(a1_logits, axis=-1).astype(mx.int32)
        pred_a2 = mx.argmax(a2_logits, axis=-1).astype(mx.int32)

        # Kernel dispatch
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


# ══════════════════════════════════════════════════════════════════════
# Data generation (from v1)
# ══════════════════════════════════════════════════════════════════════

def _collect_nodes(node, out, max_val):
    if isinstance(node, int):
        return
    op, a1, a2 = node
    op_code = OP_STR_TO_CODE[op]
    c1_type, c1_val = TYPE_INT, eval_tree(a1) if not isinstance(a1, int) else a1
    c2_type, c2_val = TYPE_INT, eval_tree(a2) if not isinstance(a2, int) else a2
    gt_result = eval_tree(node)
    out.append((op_code, c1_type, c1_val, c2_type, c2_val, gt_result))
    _collect_nodes(a1, out, max_val)
    _collect_nodes(a2, out, max_val)


def generate_node_batch(rng, batch_size, max_val, max_depth):
    all_ops, all_c1t, all_c1v, all_c2t, all_c2v = [], [], [], [], []
    all_gt_ops, all_gt_a1, all_gt_a2, all_gt_res = [], [], [], []

    for _ in range(batch_size):
        tree = random_expr(rng, max_val, max_depth)
        nodes = []
        _collect_nodes(tree, nodes, max_val)
        for op_code, c1_type, c1_val, c2_type, c2_val, gt_result in nodes:
            all_ops.append(op_code)
            all_c1t.append(c1_type)
            all_c1v.append(c1_val)
            all_c2t.append(c2_type)
            all_c2v.append(c2_val)
            all_gt_ops.append(op_code)
            all_gt_a1.append(c1_val)
            all_gt_a2.append(c2_val)
            all_gt_res.append(gt_result)

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


# ══════════════════════════════════════════════════════════════════════
# Loss and evaluation
# ══════════════════════════════════════════════════════════════════════

def vsm_loss(model, op_codes, c1_types, c1_vals, c2_types, c2_vals,
             gt_ops, gt_a1, gt_a2):
    config = model.config
    out = model.forward(op_codes, c1_types, c1_vals, c2_types, c2_vals)

    gt_type = mx.full(op_codes.shape, TYPE_INT, dtype=mx.int32)
    loss_type = nn.losses.cross_entropy(out["type_logits"], gt_type, reduction="mean")
    loss_op = nn.losses.cross_entropy(out["op_logits"], gt_ops, reduction="mean")

    gt_a1c = mx.clip(gt_a1, 0, config.max_val - 1).astype(mx.int32)
    gt_a2c = mx.clip(gt_a2, 0, config.max_val - 1).astype(mx.int32)
    loss_a1 = nn.losses.cross_entropy(out["arg1_logits"], gt_a1c, reduction="mean")
    loss_a2 = nn.losses.cross_entropy(out["arg2_logits"], gt_a2c, reduction="mean")

    return 0.5 * loss_type + loss_op + loss_a1 + loss_a2


def evaluate(model, rng, n_exprs, max_val, max_depth):
    batch = generate_node_batch(rng, n_exprs, max_val, max_depth)
    out = model.forward(batch["op_codes"], batch["c1_types"], batch["c1_vals"],
                        batch["c2_types"], batch["c2_vals"])
    for v in out.values():
        mx.eval(v)

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


# ══════════════════════════════════════════════════════════════════════
# Training loop (shared across all variants)
# ══════════════════════════════════════════════════════════════════════

def train_variant(
    config: VSMConfig,
    label: str,
    generations: int = 5000,
    batch_size: int = 128,
    adam_steps: int = 10,
    lr: float = 1e-3,
    mutation_pct: float = 0.02,
    eval_interval: int = 500,
    max_depth: int = 2,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Train one VSM variant and return final metrics."""
    rng = np.random.RandomState(seed)
    model = VSMNodeV2(config)

    n_ternary = count_ternary_weights(model)
    mut_budget = max(1, int(n_ternary * mutation_pct))

    if verbose:
        # Count params
        total_params = 0
        for _, p in model.parameters().items() if hasattr(model.parameters(), 'items') else []:
            pass
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"  d={config.d_model}  mix={config.n_mix_layers}  "
              f"concat={config.concat_inputs}  val_res={config.value_residual}")
        print(f"  ternary={n_ternary:,}  mut_budget={mut_budget}")
        print(f"{'─' * 60}")

    optimizer = optim.Adam(learning_rate=lr)
    loss_fn = nn.value_and_grad(model, vsm_loss)

    best_route = -1.0
    champion = save_topology(model)

    history = []
    t0 = time.time()

    for gen in range(generations):
        # Adam steps
        avg_loss = 0.0
        for _ in range(adam_steps):
            b = generate_node_batch(rng, batch_size, config.max_val, max_depth)
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

        # Evaluate periodically
        if gen % eval_interval == 0 or gen == generations - 1:
            erng = np.random.RandomState(seed + gen + 5000)
            m = evaluate(model, erng, 512, config.max_val, max_depth)

            if m["route_acc"] >= best_route:
                best_route = m["route_acc"]
                champion = save_topology(model)
                status = "✓"
            else:
                load_topology(model, champion)
                status = "✗"

            history.append({**m, "gen": gen, "loss": avg_loss})

            if verbose:
                print(f"  {gen:4d}  L={avg_loss:.3f}  op={m['op_acc']*100:3.0f}%  "
                      f"a1={m['a1_acc']*100:3.0f}%  a2={m['a2_acc']*100:3.0f}%  "
                      f"rte={m['route_acc']*100:3.0f}%  res={m['result_acc']*100:3.0f}%  {status}")

            if m["route_acc"] >= 0.95:
                if verbose:
                    print(f"  🎯 Converged at gen {gen}!")
                break
        else:
            # Quick champion check
            qb = generate_node_batch(rng, 32, config.max_val, max_depth)
            qo = model.forward(qb["op_codes"], qb["c1_types"], qb["c1_vals"],
                               qb["c2_types"], qb["c2_vals"])
            mx.eval(qo["pred_op"], qo["pred_a1"], qo["pred_a2"])
            qa = ((np.array(qo["pred_op"]) == np.array(qb["gt_ops"])) &
                  (np.array(qo["pred_a1"]) == np.array(qb["gt_a1"])) &
                  (np.array(qo["pred_a2"]) == np.array(qb["gt_a2"]))).mean()
            if qa >= best_route:
                champion = save_topology(model)
                best_route = max(best_route, qa)
            else:
                load_topology(model, champion)

    elapsed = time.time() - t0

    # Final eval with fresh seed and more samples
    load_topology(model, champion)
    frng = np.random.RandomState(seed + 99999)
    final = evaluate(model, frng, 1024, config.max_val, max_depth)

    result = {
        "label": label,
        "config": {
            "d_model": config.d_model,
            "n_mix_layers": config.n_mix_layers,
            "concat_inputs": config.concat_inputs,
            "value_residual": config.value_residual,
            "max_val": config.max_val,
        },
        "ternary_weights": n_ternary,
        "elapsed_s": elapsed,
        "best_route": best_route,
        "final": final,
        "history": history,
    }

    if verbose:
        print(f"\n  Final (1024 trees, {final['n_nodes']} nodes):")
        print(f"    Op:     {final['op_acc']*100:.1f}%")
        print(f"    Arg1:   {final['a1_acc']*100:.1f}%")
        print(f"    Arg2:   {final['a2_acc']*100:.1f}%")
        print(f"    Route:  {final['route_acc']*100:.1f}%")
        print(f"    Result: {final['result_acc']*100:.1f}%")
        print(f"    Time:   {elapsed:.1f}s")

    return result


# ══════════════════════════════════════════════════════════════════════
# Experiment runner
# ══════════════════════════════════════════════════════════════════════

def run_experiments(
    generations: int = 5000,
    max_val: int = 10,
    max_depth: int = 2,
    seed: int = 42,
):
    """Run all diagnostic variants and print comparison table."""

    print("=" * 70)
    print("  VSM Tree v2 — Bottleneck Diagnosis")
    print(f"  max_val={max_val}  max_depth={max_depth}  gens={generations}")
    print("=" * 70)

    variants = [
        # Baseline: v1 design (add, d=64, 2 mix, no residual)
        ("A: v1 baseline (add, d=64)",
         VSMConfig(d_model=64, max_val=max_val, n_mix_layers=2,
                   concat_inputs=False, value_residual=False)),

        # B: Concat instead of add
        ("B: concat (d=64)",
         VSMConfig(d_model=64, max_val=max_val, n_mix_layers=2,
                   concat_inputs=True, value_residual=False)),

        # C: Value residual
        ("C: val residual (d=64)",
         VSMConfig(d_model=64, max_val=max_val, n_mix_layers=2,
                   concat_inputs=False, value_residual=True)),

        # D: Concat + value residual
        ("D: concat + val_res (d=64)",
         VSMConfig(d_model=64, max_val=max_val, n_mix_layers=2,
                   concat_inputs=True, value_residual=True)),

        # E: Concat + val_res + 4 mix layers
        ("E: concat + val_res + 4mix (d=64)",
         VSMConfig(d_model=64, max_val=max_val, n_mix_layers=4,
                   concat_inputs=True, value_residual=True)),

        # F: Concat + val_res at d=128
        ("F: concat + val_res (d=128)",
         VSMConfig(d_model=128, max_val=max_val, n_mix_layers=2,
                   concat_inputs=True, value_residual=True)),

        # G: Concat + val_res at d=256
        ("G: concat + val_res (d=256)",
         VSMConfig(d_model=256, max_val=max_val, n_mix_layers=2,
                   concat_inputs=True, value_residual=True)),
    ]

    results = []
    for label, config in variants:
        r = train_variant(
            config, label,
            generations=generations,
            max_depth=max_depth,
            seed=seed,
        )
        results.append(r)

    # Print comparison table
    print("\n" + "=" * 90)
    print("  COMPARISON TABLE")
    print("=" * 90)
    print(f"  {'Variant':<35s}  {'Op':>4s}  {'A1':>4s}  {'A2':>4s}  "
          f"{'Rte':>4s}  {'Res':>4s}  {'Wts':>7s}  {'Time':>5s}")
    print("-" * 90)

    for r in results:
        f = r["final"]
        print(f"  {r['label']:<35s}  {f['op_acc']*100:3.0f}%  "
              f"{f['a1_acc']*100:3.0f}%  {f['a2_acc']*100:3.0f}%  "
              f"{f['route_acc']*100:3.0f}%  {f['result_acc']*100:3.0f}%  "
              f"{r['ternary_weights']:>7,}  {r['elapsed_s']:>4.0f}s")

    print("=" * 90)

    # Identify winner
    best = max(results, key=lambda r: r["final"]["route_acc"])
    print(f"\n  Best: {best['label']} — route={best['final']['route_acc']*100:.1f}%")

    if best["final"]["route_acc"] > 0.80:
        print("  ✅ VIABLE: Architecture can route accurately.")
    elif best["final"]["route_acc"] > 0.50:
        print("  🔄 PARTIAL: Routing improving. May need more training or larger scale.")
    else:
        print("  ❌ BOTTLENECK: Routing still limited. Consider regression or attention.")

    return results


# ══════════════════════════════════════════════════════════════════════
# Scaling tests (run after bottleneck diagnosis)
# ══════════════════════════════════════════════════════════════════════

def run_scaling_tests(
    best_config_kwargs: dict,
    generations: int = 5000,
    seed: int = 42,
):
    """Test how the best variant scales with max_val and depth."""

    print("\n" + "=" * 70)
    print("  VSM Tree v2 — Scaling Tests")
    print("=" * 70)

    results = []

    # max_val scaling at fixed depth=2
    for mv in [10, 20, 50, 100]:
        cfg = VSMConfig(**{**best_config_kwargs, "max_val": mv, "val_embed_range": mv * 2 + 100})
        label = f"max_val={mv}, depth=2"
        r = train_variant(cfg, label, generations=generations, max_depth=2, seed=seed)
        results.append(r)

    # depth scaling at fixed max_val=10
    for md in [2, 3, 4]:
        cfg = VSMConfig(**{**best_config_kwargs, "max_val": 10})
        label = f"max_val=10, depth={md}"
        r = train_variant(cfg, label, generations=generations, max_depth=md, seed=seed)
        results.append(r)

    # Print comparison
    print("\n" + "=" * 90)
    print("  SCALING TABLE")
    print("=" * 90)
    print(f"  {'Variant':<30s}  {'Op':>4s}  {'A1':>4s}  {'A2':>4s}  "
          f"{'Rte':>4s}  {'Res':>4s}  {'Nodes':>6s}")
    print("-" * 90)

    for r in results:
        f = r["final"]
        print(f"  {r['label']:<30s}  {f['op_acc']*100:3.0f}%  "
              f"{f['a1_acc']*100:3.0f}%  {f['a2_acc']*100:3.0f}%  "
              f"{f['route_acc']*100:3.0f}%  {f['result_acc']*100:3.0f}%  "
              f"{f['n_nodes']:>6d}")

    print("=" * 90)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=5000)
    p.add_argument("--max-val", type=int, default=10)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scaling", action="store_true",
                   help="Run scaling tests after bottleneck diagnosis")
    a = p.parse_args()

    results = run_experiments(
        generations=a.generations,
        max_val=a.max_val,
        max_depth=a.max_depth,
        seed=a.seed,
    )

    if a.scaling:
        # Use best variant's config for scaling
        best = max(results, key=lambda r: r["final"]["route_acc"])
        run_scaling_tests(
            best["config"],
            generations=a.generations,
            seed=a.seed,
        )
