#!/usr/bin/env python3
"""Register analysis — what has the compressor learned to encode?

The v4.1 VSM-LM has 3 named registers (type, scope, role), each 256-dim,
updated at 5 pass boundaries (L0↑, L1↑, L2, L1↓, L0↓) plus the bank_0
init. If the compressor is Montague-shaped, these registers should encode
something isomorphic to Montague types (e, t, <e,t>, <<e,t>,t>).

Two modes:
  capture  — Run probes through a checkpoint, save full register vectors
  analyze  — Load captured vectors, run PCA/clustering, measure type encoding

The capture step produces .npz files with the complete 256-dim register
vectors at every pass boundary. The existing probe script only saves norms.

Usage:
    # Step 1: Capture full register vectors from a checkpoint
    uv run python scripts/register_analysis.py capture \\
        checkpoints/vsm-lm-v4.1/step_003000.pt

    # Step 2: Analyze register content
    uv run python scripts/register_analysis.py analyze \\
        results/register-vectors/step_003000_v4.1.npz

    # Or capture + analyze in one shot
    uv run python scripts/register_analysis.py capture \\
        checkpoints/vsm-lm-v4.1/step_003000.pt --analyze

    # Trajectory: compare across checkpoints
    uv run python scripts/register_analysis.py trajectory \\
        results/register-vectors/step_001000_v4.1.npz \\
        results/register-vectors/step_002000_v4.1.npz \\
        results/register-vectors/step_003000_v4.1.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════

REGISTER_NAMES = ("type", "scope", "role")
PASS_NAMES = ("L0_asc", "L1_asc", "L2_apex", "L1_desc", "L0_desc")
PASS_LABELS = ("L0↑", "L1↑", " L2", "L1↓", "L0↓")
BANK_NAMES = ("bank_0", "bank_1_asc", "bank_2_asc", "bank_3", "bank_2_desc", "bank_1_desc")

RESULTS_DIR = Path("results/register-vectors")

# Default probe sets to capture
DEFAULT_PROBE_PATHS = [
    Path("probes/compile-gradient.json"),
    Path("probes/binding.json"),
]

# ══════════════════════════════════════════════════════════════════════
# Montague type labeling
# ══════════════════════════════════════════════════════════════════════

# Montague type assignments for known probes.
# t = proposition (truth value), e = entity, <e,t> = predicate,
# <<e,t>,t> = quantifier, <t,t> = sentential operator
# These are the *result type* of the full expression.
#
# For incomplete/non-compositional inputs (null, anti, weak_compile),
# we assign "other" — the point is to see if the model separates them.

MONTAGUE_TYPES: dict[str, str] = {
    # ── compile-gradient: strong_compile (all complete propositions → t) ──
    "cg-strong-01": "t",          # The dog runs.
    "cg-strong-02": "t",          # Every student reads a book.
    "cg-strong-03": "t",          # The cat that sat on the mat is black.
    "cg-strong-04": "t",          # If it rains, the ground is wet.
    "cg-strong-05": "t",          # No bird can swim.
    "cg-strong-06": "t",          # The teacher gave every student a grade.
    "cg-strong-07": "t",          # Someone believes that the earth is flat.
    "cg-strong-08": "t",          # Birds fly.
    # ── compile-gradient: medium_compile (also complete propositions) ──
    "cg-medium-01": "t",          # The old man walked slowly across the bridge.
    "cg-medium-02": "t",          # Three children were playing...
    "cg-medium-03": "t",          # The book that I bought yesterday was expensive.
    "cg-medium-04": "t",          # Most politicians promise more than they deliver.
    "cg-medium-05": "t",          # The fact that she left surprised nobody.
    "cg-medium-06": "t",          # Running is healthier than sitting.
    "cg-medium-07": "t",          # She told him to leave before it got dark.
    "cg-medium-08": "t",          # What the witness saw contradicted the official report.
    # ── compile-gradient: weak_compile (mixed formal/meta) ──
    "cg-weak-01": "fn",           # λx.λy. (f x) ∧ (g y) — a lambda term
    "cg-weak-02": "other",        # meta-linguistic
    "cg-weak-03": "fn",           # ∀x. P(x) → Q(x) — a formula
    "cg-weak-04": "other",        # imperative / instruction
    "cg-weak-05": "fn",           # type signature
    "cg-weak-06": "other",        # instruction
    "cg-weak-07": "other",        # instruction
    "cg-weak-08": "other",        # question about logic
    # ── compile-gradient: null ──
    "cg-null-01": "other",
    "cg-null-02": "other",
    "cg-null-03": "other",
    "cg-null-04": "other",
    "cg-null-05": "other",
    "cg-null-06": "other",
    "cg-null-07": "other",
    "cg-null-08": "other",
    # ── compile-gradient: anti_compile ──
    "cg-anti-01": "other",
    "cg-anti-02": "other",
    "cg-anti-03": "other",
    "cg-anti-04": "other",
    "cg-anti-05": "other",
    "cg-anti-06": "other",
    "cg-anti-07": "other",
    "cg-anti-08": "other",
    # ── binding: quantifier_scope (complete propositions with quantifiers) ──
    "bind-scope-01a": "t_quant",  # Every student read a book.
    "bind-scope-01b": "t_quant",  # A student read every book.
    "bind-scope-02a": "t_quant",  # No student passed every exam.
    "bind-scope-02b": "t_quant",  # Every student passed no exam.
    "bind-scope-03": "t_quant",   # Most students read some book.
    "bind-scope-04": "t_quant",   # Exactly two students answered every question.
    # ── binding: variable_binding (complete propositions) ──
    "bind-var-01a": "t",          # The cat chased the dog.
    "bind-var-01b": "t",          # The dog chased the cat.
    "bind-var-02": "t_quant",     # The teacher gave every student a grade.
    "bind-var-03": "t_quant",     # Someone loves everyone.
    "bind-var-04": "t_quant",     # Everyone loves someone.
    # ── binding: anaphora (propositions with binding) ──
    "bind-ana-01": "t_bind",      # John saw himself in the mirror.
    "bind-ana-02a": "t_bind",     # Every boy thinks he is smart.
    "bind-ana-02b": "t_bind",     # John thinks he is smart.
    "bind-ana-03": "t_bind",      # No student who failed...
    # ── binding: control (propositions with embedded binding) ──
    "bind-ctrl-01": "t_bind",     # She told him to leave.
    "bind-ctrl-02": "t_bind",     # She promised him to leave.
    "bind-ctrl-03": "t_bind",     # She persuaded him to believe...
    # ── binding: relative clauses (propositions with embedding) ──
    "bind-rel-01": "t_rel",       # The cat that chased the dog is black.
    "bind-rel-02": "t_rel",       # The cat that the dog chased is black.
    "bind-rel-03": "t_rel",       # Every student who read a book passed the exam.
    "bind-rel-04": "t_rel",       # The book that every student read was boring.
    # ── binding: negation scope ──
    "bind-neg-01": "t_quant",     # Nobody saw anything.
    "bind-neg-02": "t_quant",     # Not every bird can fly.
    # ── binding: embedded clauses ──
    "bind-embed-01": "t_bind",    # John believes that every student passed.
    "bind-embed-02": "t_bind",    # Every professor thinks that some student cheated.
}

# Composition depth — number of FA operations required
COMPOSITION_DEPTH: dict[str, int] = {
    "cg-strong-01": 1,  # runs(dog) — 1 FA
    "cg-strong-02": 3,  # every(student, λx.∃y[book(y) ∧ reads(x,y)]) — 3 FA + QR
    "cg-strong-03": 4,  # relative clause + matrix predication
    "cg-strong-04": 2,  # conditional: 2 propositions linked
    "cg-strong-05": 2,  # no(bird, swim) — negated quantifier
    "cg-strong-06": 4,  # ditransitive + universal quantifier
    "cg-strong-07": 3,  # attitude verb + embedded proposition
    "cg-strong-08": 1,  # fly(birds) — 1 FA
    "cg-medium-01": 2,  # walked(man) + adverbials
    "cg-medium-02": 3,  # progressive + temporal clause
    "cg-medium-03": 3,  # relative clause + predication
    "cg-medium-04": 3,  # quantifier + comparison
    "cg-medium-05": 3,  # factive + quantifier
    "cg-medium-06": 2,  # comparative
    "cg-medium-07": 3,  # control + temporal
    "cg-medium-08": 3,  # free relative + predication
    # binding probes — depth correlates with structural complexity
    "bind-var-01a": 1,  # chased(cat, dog) — simple transitive
    "bind-var-01b": 1,  # chased(dog, cat)
    "bind-scope-01a": 3,
    "bind-scope-01b": 3,
    "bind-rel-01": 3,
    "bind-rel-02": 3,
    "bind-rel-03": 4,
    "bind-rel-04": 4,
    "bind-ana-01": 2,
    "bind-ana-03": 5,
    "bind-ctrl-03": 4,
}

# Coarser grouping for cluster analysis — fewer categories, more probes per group
MONTAGUE_COARSE: dict[str, str] = {}
for pid, mt in MONTAGUE_TYPES.items():
    if mt == "t":
        MONTAGUE_COARSE[pid] = "proposition"
    elif mt.startswith("t_"):
        MONTAGUE_COARSE[pid] = "proposition"  # all are propositions at the top level
    elif mt == "fn":
        MONTAGUE_COARSE[pid] = "formal"
    else:
        MONTAGUE_COARSE[pid] = "other"

# Finer grouping — separates binding types within propositions
MONTAGUE_FINE: dict[str, str] = {}
for pid, mt in MONTAGUE_TYPES.items():
    if mt == "t":
        MONTAGUE_FINE[pid] = "t_simple"
    elif mt == "t_quant":
        MONTAGUE_FINE[pid] = "t_quant"
    elif mt == "t_bind":
        MONTAGUE_FINE[pid] = "t_bind"
    elif mt == "t_rel":
        MONTAGUE_FINE[pid] = "t_rel"
    elif mt == "fn":
        MONTAGUE_FINE[pid] = "formal"
    else:
        MONTAGUE_FINE[pid] = "other"


# ══════════════════════════════════════════════════════════════════════
# Mode 1: Capture — run probes, save full register vectors
# ══════════════════════════════════════════════════════════════════════


def capture_registers(
    checkpoint_path: str | Path,
    probe_paths: list[Path] | None = None,
    device: str | None = None,
) -> Path:
    """Run probes through a v4.1 checkpoint, capturing full register vectors.

    For each probe, captures:
      - bank_0 init: 3 registers × 256-dim (before any processing)
      - Per-pass (5 passes): 3 registers × 256-dim after S4 scan
      - Per-pass (5 passes): 3 registers × 256-dim after full pass (S3-gated)

    Saves to results/register-vectors/step_{N}_v4.1.npz with arrays:
      - probe_ids: (n_probes,) string array
      - probe_set_ids: (n_probes,) string array  
      - categories: (n_probes,) string array
      - prompts: (n_probes,) string array
      - bank_0_init: (n_probes, 3, 256) — register bank 0
      - {pass}_after_s4: (n_probes, 3, 256) — registers after S4 per pass
      - {pass}_after_pass: (n_probes, 3, 256) — registers after full pass
    """
    from transformers import AutoTokenizer
    from verbum.vsm_lm_v4_1 import VSMLMV4_1

    checkpoint_path = Path(checkpoint_path)
    probe_paths = probe_paths or DEFAULT_PROBE_PATHS
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    step = ckpt["step"]
    config = ckpt.get("config", {})

    # Verify v4.1
    state_dict = ckpt["model_state_dict"]
    if "s3_passes.0.gate_heads.0.weight" not in state_dict:
        print("  ✗ Not a v4.1 checkpoint")
        sys.exit(1)

    print(f"  Step: {step} (v4.1)")

    model = VSMLMV4_1(
        vocab_size=config.get("vocab_size", 50277),
        d_model=config.get("d_model", 512),
        d_register=config.get("d_register", 256),
        max_len=config.get("seq_len", 4096),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 1536),
        d_ff_consolidate=config.get("d_ff_consolidate", 2048),
        window=config.get("window", 8),
        strides=tuple(config.get("strides", [1, 8, 64, 512])),
        n_prep_layers=config.get("n_prep_layers", 1),
        n_converge_layers=config.get("n_converge_layers", 2),
        n_consolidate_layers=config.get("n_consolidate_layers", 3),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    del ckpt

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")

    # Load all probe sets
    all_probes = []
    for pp in probe_paths:
        data = json.loads(pp.read_text())
        set_id = data.get("id", pp.stem)
        for probe in data["probes"]:
            all_probes.append({**probe, "_set_id": set_id})

    n_probes = len(all_probes)
    n_regs = len(REGISTER_NAMES)
    d_reg = config.get("d_register", 256)

    print(f"  Capturing registers for {n_probes} probes across {len(probe_paths)} probe sets")
    print()

    # Pre-allocate arrays
    probe_ids = []
    probe_set_ids = []
    categories = []
    prompts = []

    bank_0_init = np.zeros((n_probes, n_regs, d_reg), dtype=np.float32)

    # Per-pass: after S4 and after full pass
    pass_after_s4 = {pn: np.zeros((n_probes, n_regs, d_reg), dtype=np.float32)
                     for pn in PASS_NAMES}
    pass_after_full = {pn: np.zeros((n_probes, n_regs, d_reg), dtype=np.float32)
                       for pn in PASS_NAMES}

    with torch.no_grad():
        for idx, probe in enumerate(all_probes):
            probe_ids.append(probe["id"])
            probe_set_ids.append(probe["_set_id"])
            categories.append(probe.get("category", "unknown"))
            prompts.append(probe["prompt"])

            ids = tokenizer.encode(probe["prompt"], return_tensors="pt").to(device)
            if ids.shape[1] > 4096:
                ids = ids[:, :4096]

            # Run the register-capturing forward pass
            reg_data = _forward_capture_registers(model, ids)

            # Store bank_0 init
            for ri, rn in enumerate(REGISTER_NAMES):
                bank_0_init[idx, ri] = reg_data["bank_0"][ri]

            # Store per-pass data
            for pn in PASS_NAMES:
                for ri in range(n_regs):
                    pass_after_s4[pn][idx, ri] = reg_data[f"{pn}_after_s4"][ri]
                    pass_after_full[pn][idx, ri] = reg_data[f"{pn}_after_pass"][ri]

            # Progress
            if (idx + 1) % 10 == 0 or idx == n_probes - 1:
                print(f"  [{idx + 1:3d}/{n_probes}] {probe['id']}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"step_{step:06d}_v4.1.npz"

    save_dict = {
        "probe_ids": np.array(probe_ids),
        "probe_set_ids": np.array(probe_set_ids),
        "categories": np.array(categories),
        "prompts": np.array(prompts),
        "bank_0_init": bank_0_init,
        "step": np.array(step),
    }
    for pn in PASS_NAMES:
        save_dict[f"{pn}_after_s4"] = pass_after_s4[pn]
        save_dict[f"{pn}_after_pass"] = pass_after_full[pn]

    np.savez_compressed(out_path, **save_dict)
    print(f"\n  Saved: {out_path}")
    print(f"  Shape: {n_probes} probes × {n_regs} registers × {d_reg} dims")
    print(f"  Passes: {list(PASS_NAMES)}")

    return out_path


def _forward_capture_registers(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> dict[str, list[np.ndarray]]:
    """Run v4.1 forward pass, capturing full register vectors at every boundary.

    Returns dict with keys:
      bank_0: list of 3 numpy arrays (256-dim each)
      {pass_name}_after_s4: list of 3 numpy arrays
      {pass_name}_after_pass: list of 3 numpy arrays
    """
    B, L = input_ids.shape
    device = input_ids.device

    positions = torch.arange(L, device=device)
    x = model.token_embed(input_ids) + model.pos_embed(positions)

    # Register banks
    bank_0 = model._init_bank0()
    bank_1_asc = model._fresh_bank()
    bank_2_asc = model._fresh_bank()
    bank_3 = model._fresh_bank()
    bank_2_desc = model._fresh_bank()
    bank_1_desc = model._fresh_bank()

    result = {
        "bank_0": [r.detach().cpu().numpy() for r in bank_0],
    }

    # Pass schedule: (pass_idx, level, pass_name)
    pass_schedule = [
        (0, 0, "L0_asc"),
        (1, 1, "L1_asc"),
        (2, 2, "L2_apex"),
        (3, 1, "L1_desc"),
        (4, 0, "L0_desc"),
    ]

    for pass_idx, level, pass_name in pass_schedule:
        # Determine readable banks and target bank
        if pass_idx == 0:
            readable = [bank_0]
            target_bank = bank_1_asc
        elif pass_idx == 1:
            readable = [bank_0, bank_1_asc]
            target_bank = bank_2_asc
        elif pass_idx == 2:
            readable = [bank_0, bank_1_asc, bank_2_asc]
            target_bank = bank_3
        elif pass_idx == 3:
            readable = [bank_0, bank_1_asc, bank_2_asc, bank_3]
            target_bank = bank_2_desc
        elif pass_idx == 4:
            readable = [bank_0, bank_1_asc, bank_2_desc, bank_3]
            target_bank = bank_1_desc

        # S4: scan residual conditioned on readable banks
        s4_updates, _ = model.s4(readable, x)
        target_bank = [
            target_bank[i] + s4_updates[i]
            for i in range(model.n_registers)
        ]

        # Capture after S4
        result[f"{pass_name}_after_s4"] = [
            r.detach().cpu().numpy() for r in target_bank
        ]

        # Run the 3 phases (prep, converge, consolidate) with S3 gating
        x_before = x

        # PREP
        prep_out = model._run_prep(x)
        delta = prep_out - x
        gated_delta, target_bank, _, _ = model.s3_passes[pass_idx].gate_phase(
            target_bank, delta, 0)
        x = x + gated_delta

        # CONVERGE
        converge_out = model._run_converge(x, level)
        delta = converge_out - x
        gated_delta, target_bank, _, _ = model.s3_passes[pass_idx].gate_phase(
            target_bank, delta, 1)
        x = x + gated_delta

        # CONSOLIDATE
        consolidate_out = model._run_consolidate(x)
        delta = consolidate_out - x
        gated_delta, target_bank, _, _ = model.s3_passes[pass_idx].gate_phase(
            target_bank, delta, 2)
        x = x + gated_delta

        # Capture after full pass
        result[f"{pass_name}_after_pass"] = [
            r.detach().cpu().numpy() for r in target_bank
        ]

        # Write back the target bank
        if pass_idx == 0:
            bank_1_asc = target_bank
        elif pass_idx == 1:
            bank_2_asc = target_bank
        elif pass_idx == 2:
            bank_3 = target_bank
        elif pass_idx == 3:
            bank_2_desc = target_bank
        elif pass_idx == 4:
            bank_1_desc = target_bank

    return result


# ══════════════════════════════════════════════════════════════════════
# Mode 2: Analyze — PCA, clustering, Montague type encoding
# ══════════════════════════════════════════════════════════════════════


def analyze_registers(npz_path: str | Path) -> dict:
    """Analyze register vectors for Montague type encoding.

    Tests three hypotheses:
    1. Do registers separate propositions from non-compositional inputs?
    2. Do registers separate binding types (quantifier, anaphora, relative)?
    3. Does register content correlate with composition depth?

    Returns analysis dict with findings.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr

    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    probe_ids = data["probe_ids"]
    categories = data["categories"]
    step = int(data["step"])
    n_probes = len(probe_ids)

    print(f"{'═' * 72}")
    print(f"  REGISTER ANALYSIS — v4.1 step {step}")
    print(f"  {n_probes} probes")
    print(f"{'═' * 72}")

    findings = {"step": step, "n_probes": n_probes}

    # ── 1. Per-register, per-pass PCA ─────────────────────────────
    print(f"\n  ── PCA: VARIANCE EXPLAINED ──")
    print(f"  How much structure do registers carry at each pass?")
    print()
    print(f"  {'Register':<8} {'Stage':<12}", end="")
    for label in PASS_LABELS:
        print(f" {label:>8}", end="")
    print()
    print(f"  {'─' * 60}")

    pca_results = {}
    for ri, rn in enumerate(REGISTER_NAMES):
        for stage in ["after_s4", "after_pass"]:
            tag = f"{rn}_{stage}"
            variances = []
            for pi, pn in enumerate(PASS_NAMES):
                key = f"{pn}_{stage}"
                vecs = data[key][:, ri, :]  # (n_probes, 256)

                # PCA — how many dimensions carry the variance?
                pca = PCA(n_components=min(10, n_probes, vecs.shape[1]))
                pca.fit(vecs)
                # Top-3 variance ratio = how concentrated is the structure
                top3_var = sum(pca.explained_variance_ratio_[:3])
                variances.append(top3_var)
                pca_results[f"{pn}_{rn}_{stage}"] = {
                    "top3_var": top3_var,
                    "components": pca.components_[:3],
                    "transformed": pca.transform(vecs)[:, :3],
                    "explained": pca.explained_variance_ratio_[:5].tolist(),
                }

            print(f"  {rn:<8} {stage:<12}", end="")
            for v in variances:
                marker = "█" if v > 0.5 else "▓" if v > 0.3 else "░"
                print(f" {v:>7.3f}{marker}", end="")
            print()

    findings["pca"] = {k: {"top3_var": v["top3_var"], "explained": v["explained"]}
                       for k, v in pca_results.items()}

    # ── 2. Montague type clustering ───────────────────────────────
    print(f"\n  ── MONTAGUE TYPE SEPARATION ──")
    print(f"  Do registers separate inputs by semantic type?")
    print()

    # Build label arrays
    coarse_labels = np.array([MONTAGUE_COARSE.get(pid, "unknown") for pid in probe_ids])
    fine_labels = np.array([MONTAGUE_FINE.get(pid, "unknown") for pid in probe_ids])

    unique_coarse = sorted(set(coarse_labels))
    unique_fine = sorted(set(fine_labels))
    print(f"  Coarse types: {dict(zip(*np.unique(coarse_labels, return_counts=True)))}")
    print(f"  Fine types:   {dict(zip(*np.unique(fine_labels, return_counts=True)))}")
    print()

    # Silhouette analysis: do vectors cluster by Montague type?
    print(f"  SILHOUETTE SCORES (higher = better type separation)")
    print(f"  {'Register':<8} {'Stage':<12} {'Grouping':<10}", end="")
    for label in PASS_LABELS:
        print(f" {label:>8}", end="")
    print()
    print(f"  {'─' * 72}")

    silhouette_results = {}
    for ri, rn in enumerate(REGISTER_NAMES):
        for stage in ["after_pass"]:  # Focus on after_pass (most processed)
            for grouping_name, labels in [("coarse", coarse_labels), ("fine", fine_labels)]:
                # Need at least 2 unique labels with ≥2 samples each
                unique, counts = np.unique(labels, return_counts=True)
                valid = unique[counts >= 2]
                if len(valid) < 2:
                    continue

                mask = np.isin(labels, valid)
                scores = []

                for pi, pn in enumerate(PASS_NAMES):
                    key = f"{pn}_{stage}"
                    vecs = data[key][mask, ri, :]
                    masked_labels = labels[mask]

                    if len(set(masked_labels)) < 2:
                        scores.append(0.0)
                        continue

                    try:
                        s = silhouette_score(vecs, masked_labels, metric="cosine")
                    except ValueError:
                        s = 0.0
                    scores.append(s)

                tag = f"{rn}_{stage}_{grouping_name}"
                silhouette_results[tag] = scores

                print(f"  {rn:<8} {stage:<12} {grouping_name:<10}", end="")
                for s in scores:
                    marker = "★" if s > 0.3 else "●" if s > 0.1 else "○"
                    print(f" {s:>+7.3f}{marker}", end="")
                print()

    findings["silhouette"] = silhouette_results

    # ── 3. Inter-class distance ratios ────────────────────────────
    print(f"\n  ── TYPE CENTROID DISTANCES ──")
    print(f"  How far apart are type centroids vs within-type spread?")
    print()

    # Use the most-processed pass (L0↓) and the register most likely to carry type info
    for stage in ["after_pass"]:
        for pn, plabel in zip(PASS_NAMES, PASS_LABELS):
            key = f"{pn}_{stage}"

            print(f"  Pass {plabel}:")
            for ri, rn in enumerate(REGISTER_NAMES):
                vecs = data[key][:, ri, :]

                # Compute centroids per coarse type
                centroids = {}
                spreads = {}
                for t in unique_coarse:
                    mask = coarse_labels == t
                    if mask.sum() < 2:
                        continue
                    group_vecs = vecs[mask]
                    centroids[t] = group_vecs.mean(axis=0)
                    # Average within-group distance
                    dists = cdist(group_vecs, group_vecs, metric="cosine")
                    spreads[t] = dists[np.triu_indices_from(dists, k=1)].mean()

                if len(centroids) < 2:
                    continue

                # Between-centroid distances
                types_list = sorted(centroids.keys())
                cent_matrix = np.array([centroids[t] for t in types_list])
                between = cdist(cent_matrix, cent_matrix, metric="cosine")
                mean_between = between[np.triu_indices_from(between, k=1)].mean()
                mean_within = np.mean(list(spreads.values()))

                ratio = mean_between / max(mean_within, 1e-8)
                marker = "★" if ratio > 2.0 else "●" if ratio > 1.0 else "○"
                print(f"    {rn:<8}  between={mean_between:.4f}  within={mean_within:.4f}  ratio={ratio:.2f} {marker}")

                # Per-pair distances
                if len(types_list) <= 5:
                    for i, t1 in enumerate(types_list):
                        for j, t2 in enumerate(types_list):
                            if j <= i:
                                continue
                            d = between[i, j]
                            print(f"             {t1:>12} ↔ {t2:<12} = {d:.4f}")

            print()

    findings["centroids"] = {}  # Populated above conceptually

    # ── 4. Composition depth correlation ──────────────────────────
    print(f"\n  ── COMPOSITION DEPTH CORRELATION ──")
    print(f"  Do register norms / PCA coordinates scale with depth?")
    print()

    # Get probes that have depth labels
    depth_probes = [(i, pid) for i, pid in enumerate(probe_ids)
                    if pid in COMPOSITION_DEPTH]
    if len(depth_probes) >= 5:
        depth_indices = [i for i, _ in depth_probes]
        depths = np.array([COMPOSITION_DEPTH[pid] for _, pid in depth_probes])

        print(f"  {len(depth_probes)} probes with depth labels (range {depths.min()}-{depths.max()})")
        print()
        print(f"  {'Register':<8} {'Metric':<20}", end="")
        for label in PASS_LABELS:
            print(f" {label:>8}", end="")
        print()
        print(f"  {'─' * 60}")

        for ri, rn in enumerate(REGISTER_NAMES):
            # Norm correlation
            norms_by_pass = []
            for pn in PASS_NAMES:
                vecs = data[f"{pn}_after_pass"][depth_indices, ri, :]
                norms = np.linalg.norm(vecs, axis=1)
                rho, _ = spearmanr(depths, norms)
                norms_by_pass.append(rho)

            print(f"  {rn:<8} {'norm':20}", end="")
            for rho in norms_by_pass:
                marker = "★" if abs(rho) > 0.5 else "●" if abs(rho) > 0.3 else "○"
                print(f" {rho:>+7.3f}{marker}", end="")
            print()

            # PC1 correlation
            pc1_by_pass = []
            for pn in PASS_NAMES:
                pca_key = f"{pn}_{rn}_after_pass"
                if pca_key in pca_results:
                    pc1 = pca_results[pca_key]["transformed"][depth_indices, 0]
                    rho, _ = spearmanr(depths, pc1)
                    pc1_by_pass.append(rho)
                else:
                    pc1_by_pass.append(0.0)

            print(f"  {'':8} {'PC1':20}", end="")
            for rho in pc1_by_pass:
                marker = "★" if abs(rho) > 0.5 else "●" if abs(rho) > 0.3 else "○"
                print(f" {rho:>+7.3f}{marker}", end="")
            print()
    else:
        print(f"  Insufficient probes with depth labels ({len(depth_probes)})")

    findings["depth_correlation"] = {}

    # ── 5. Pass trajectory: how do registers evolve? ──────────────
    print(f"\n  ── REGISTER TRAJECTORY (bank_0 → L0↑ → L1↑ → L2 → L1↓ → L0↓) ──")
    print(f"  How much does each pass change the registers?")
    print()

    stages = ["bank_0_init"] + [f"{pn}_after_pass" for pn in PASS_NAMES]
    stage_labels = ["init"] + list(PASS_LABELS)

    print(f"  {'Register':<8} {'Metric':<12}", end="")
    for sl in stage_labels:
        print(f" {sl:>8}", end="")
    print()
    print(f"  {'─' * 60}")

    for ri, rn in enumerate(REGISTER_NAMES):
        # Norms at each stage (mean across probes)
        norms = []
        for stage_key in stages:
            if stage_key == "bank_0_init":
                vecs = data["bank_0_init"][:, ri, :]
            else:
                vecs = data[stage_key][:, ri, :]
            norms.append(np.linalg.norm(vecs, axis=1).mean())

        print(f"  {rn:<8} {'mean_norm':<12}", end="")
        for n in norms:
            print(f" {n:>8.3f}", end="")
        print()

        # Cross-probe variance at each stage (how differentiated?)
        variances = []
        for stage_key in stages:
            if stage_key == "bank_0_init":
                vecs = data["bank_0_init"][:, ri, :]
            else:
                vecs = data[stage_key][:, ri, :]
            variances.append(vecs.var(axis=0).sum())

        print(f"  {'':8} {'variance':<12}", end="")
        for v in variances:
            print(f" {v:>8.2f}", end="")
        print()

        # Cosine similarity to bank_0 (how far has the register drifted?)
        init_vecs = data["bank_0_init"][:, ri, :]
        sims = [1.0]  # identity with self
        for stage_key in [f"{pn}_after_pass" for pn in PASS_NAMES]:
            vecs = data[stage_key][:, ri, :]
            # Mean cosine similarity between init and current
            dots = np.sum(init_vecs * vecs, axis=1)
            norms_init = np.linalg.norm(init_vecs, axis=1)
            norms_curr = np.linalg.norm(vecs, axis=1)
            cos_sims = dots / (norms_init * norms_curr + 1e-8)
            sims.append(cos_sims.mean())

        print(f"  {'':8} {'cos(init)':<12}", end="")
        for s in sims:
            print(f" {s:>8.3f}", end="")
        print()

    # ── 6. Per-probe fingerprint: which probes cluster together? ──
    print(f"\n  ── NEAREST NEIGHBORS (L0↓ after_pass, all registers concat) ──")
    print(f"  Which probes does the model treat as most similar?")
    print()

    # Concatenate all 3 registers at L0↓ after_pass → 768-dim fingerprint
    all_regs = np.concatenate(
        [data["L0_desc_after_pass"][:, ri, :] for ri in range(len(REGISTER_NAMES))],
        axis=1
    )
    dists = cdist(all_regs, all_regs, metric="cosine")

    # For each probe, show 3 nearest neighbors
    for idx in range(min(n_probes, 20)):  # First 20 probes
        pid = probe_ids[idx]
        prompt = data["prompts"][idx]
        mt = MONTAGUE_TYPES.get(pid, "?")

        neighbor_indices = np.argsort(dists[idx])[1:4]  # skip self
        neighbors = []
        for ni in neighbor_indices:
            npid = probe_ids[ni]
            nmt = MONTAGUE_TYPES.get(npid, "?")
            neighbors.append(f"{npid}({nmt})")

        prompt_short = prompt[:40] + "..." if len(prompt) > 40 else prompt
        print(f"  {pid:25s} [{mt:>8}] → {', '.join(neighbors)}")
        if idx == 19:
            remaining = n_probes - 20
            if remaining > 0:
                print(f"  ... ({remaining} more)")

    print(f"\n{'═' * 72}")

    return findings


# ══════════════════════════════════════════════════════════════════════
# Mode 3: Trajectory — compare registers across training steps
# ══════════════════════════════════════════════════════════════════════


def trajectory_analysis(npz_paths: list[str | Path]) -> None:
    """Compare register evolution across training checkpoints.

    Shows how type encoding develops over training.
    """
    from scipy.stats import spearmanr

    npz_paths = [Path(p) for p in sorted(npz_paths)]
    datasets = []
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        datasets.append(d)

    steps = [int(d["step"]) for d in datasets]
    probe_ids = datasets[0]["probe_ids"]

    print(f"{'═' * 72}")
    print(f"  REGISTER TRAJECTORY ACROSS TRAINING")
    print(f"  Steps: {steps}")
    print(f"{'═' * 72}")

    # Build labels
    coarse_labels = np.array([MONTAGUE_COARSE.get(pid, "unknown") for pid in probe_ids])
    unique_labels = sorted(set(coarse_labels))

    # Track: silhouette score evolution per register per pass
    print(f"\n  ── TYPE SEPARATION OVER TRAINING ──")
    print(f"  Silhouette score (cosine, coarse types) at each step")
    print()

    try:
        from sklearn.metrics import silhouette_score as sil_score
    except ImportError:
        print("  sklearn not available — skipping silhouette trajectory")
        return

    unique, counts = np.unique(coarse_labels, return_counts=True)
    valid = unique[counts >= 2]
    mask = np.isin(coarse_labels, valid)

    for ri, rn in enumerate(REGISTER_NAMES):
        print(f"  Register: {rn}")
        print(f"  {'Pass':<8}", end="")
        for step in steps:
            print(f" {f'step_{step}':>10}", end="")
        print()

        for pi, (pn, plabel) in enumerate(zip(PASS_NAMES, PASS_LABELS)):
            print(f"  {plabel:<8}", end="")
            for d in datasets:
                key = f"{pn}_after_pass"
                if key not in d:
                    print(f" {'N/A':>10}", end="")
                    continue
                vecs = d[key][mask, ri, :]
                masked_labels = coarse_labels[mask]
                try:
                    s = sil_score(vecs, masked_labels, metric="cosine")
                    marker = "★" if s > 0.3 else "●" if s > 0.1 else "○"
                    print(f" {s:>+8.3f}{marker}", end="")
                except ValueError:
                    print(f" {'err':>10}", end="")
            print()
        print()

    # Track: variance evolution (are registers becoming more differentiated?)
    print(f"\n  ── REGISTER DIFFERENTIATION OVER TRAINING ──")
    print(f"  Total variance (sum of per-dim variance) at each step")
    print()

    for ri, rn in enumerate(REGISTER_NAMES):
        print(f"  Register: {rn}")
        print(f"  {'Pass':<8}", end="")
        for step in steps:
            print(f" {f'step_{step}':>10}", end="")
        print()

        for pi, (pn, plabel) in enumerate(zip(PASS_NAMES, PASS_LABELS)):
            print(f"  {plabel:<8}", end="")
            for d in datasets:
                key = f"{pn}_after_pass"
                if key not in d:
                    print(f" {'N/A':>10}", end="")
                    continue
                vecs = d[key][:, ri, :]
                total_var = vecs.var(axis=0).sum()
                print(f" {total_var:>10.2f}", end="")
            print()
        print()

    # Track: PCA variance explained (PC1) — is structure concentrating?
    print(f"\n  ── PCA: PC1 VARIANCE EXPLAINED OVER TRAINING ──")
    print(f"  Higher = more structure concentrated in first principal component")
    print()

    from sklearn.decomposition import PCA

    for ri, rn in enumerate(REGISTER_NAMES):
        print(f"  Register: {rn}")
        print(f"  {'Pass':<8}", end="")
        for step in steps:
            print(f" {f'step_{step}':>10}", end="")
        print()

        for pi, (pn, plabel) in enumerate(zip(PASS_NAMES, PASS_LABELS)):
            print(f"  {plabel:<8}", end="")
            for d in datasets:
                key = f"{pn}_after_pass"
                if key not in d:
                    print(f" {'N/A':>10}", end="")
                    continue
                vecs = d[key][:, ri, :]
                n = min(10, vecs.shape[0], vecs.shape[1])
                pca = PCA(n_components=n)
                pca.fit(vecs)
                pc1 = pca.explained_variance_ratio_[0]
                print(f" {pc1:>10.3f}", end="")
            print()
        print()

    # Track: depth correlation — does compositional depth encoding strengthen?
    print(f"\n  ── COMPOSITION DEPTH CORRELATION OVER TRAINING ──")
    print(f"  Pearson r: register norm vs FA depth (negative = deeper → smaller norm)")
    print()

    from scipy.stats import pearsonr

    # Build depth arrays from probe ids
    depth_indices = []
    depth_values = []
    for j, pid in enumerate(probe_ids):
        pid_str = str(pid)
        if pid_str in COMPOSITION_DEPTH:
            depth_indices.append(j)
            depth_values.append(COMPOSITION_DEPTH[pid_str])
    depth_indices = np.array(depth_indices)
    depth_values = np.array(depth_values, dtype=float)

    n_depth = len(depth_indices)
    print(f"  {n_depth} probes with depth labels (range {int(depth_values.min())}-{int(depth_values.max())})")
    print()

    if n_depth >= 5:
        for ri, rn in enumerate(REGISTER_NAMES):
            print(f"  Register: {rn}")
            print(f"  {'Pass':<8}", end="")
            for step in steps:
                print(f" {f'step_{step}':>10}", end="")
            print()

            for pi, (pn, plabel) in enumerate(zip(PASS_NAMES, PASS_LABELS)):
                print(f"  {plabel:<8}", end="")
                for d in datasets:
                    key = f"{pn}_after_pass"
                    if key not in d:
                        print(f" {'N/A':>10}", end="")
                        continue
                    vecs = d[key][depth_indices, ri, :]
                    norms = np.linalg.norm(vecs, axis=1)
                    r, _ = pearsonr(depth_values, norms)
                    marker = "★" if abs(r) > 0.5 else "●" if abs(r) > 0.3 else "○"
                    print(f" {r:>+8.3f}{marker}", end="")
                print()
            print()
    else:
        print(f"  Too few probes with depth labels ({n_depth}) — skipping")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Register analysis — what has the compressor learned to encode?"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # capture
    cap = sub.add_parser("capture", help="Capture full register vectors from a checkpoint")
    cap.add_argument("checkpoint", type=str, help="Path to v4.1 checkpoint")
    cap.add_argument("--probes", type=str, nargs="*",
                     help="Probe set JSON files (default: compile-gradient + binding)")
    cap.add_argument("--device", type=str, default=None)
    cap.add_argument("--analyze", action="store_true",
                     help="Run analysis immediately after capture")

    # analyze
    ana = sub.add_parser("analyze", help="Analyze captured register vectors")
    ana.add_argument("npz", type=str, help="Path to register vectors .npz")

    # trajectory
    traj = sub.add_parser("trajectory", help="Compare registers across training steps")
    traj.add_argument("npz_files", type=str, nargs="+",
                      help="Paths to register vector .npz files")

    args = parser.parse_args()

    if args.mode == "capture":
        probe_paths = [Path(p) for p in args.probes] if args.probes else None
        out_path = capture_registers(args.checkpoint, probe_paths, args.device)
        if args.analyze:
            print()
            analyze_registers(out_path)

    elif args.mode == "analyze":
        analyze_registers(args.npz)

    elif args.mode == "trajectory":
        trajectory_analysis(args.npz_files)


if __name__ == "__main__":
    main()
