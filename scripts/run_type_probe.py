#!/usr/bin/env python3
"""Type probe — does Pythia-160M encode Montague semantic types?

If Montague is right, the first primitive operation is type assignment:
each word gets a semantic type (e, <e,t>, <<e,t>,t>, etc.) that
directs all subsequent composition. If L0/L3 are the type assignment
circuit, then a linear probe on the residual stream after these layers
should be able to predict the semantic type of each token.

Method:
  1. Build a labeled dataset: token → semantic type
  2. Run sentences through Pythia-160M, capture residual at every layer
  3. Train a linear probe (logistic regression) per layer
  4. Measure accuracy: where does type information become decodable?

If types are linearly decodable after L3 but not L0, then L0→L3 is
the type assignment circuit. If decodable from L0, types are in the
embeddings. If not decodable until L8+, type assignment is late.

Usage:
    uv run python scripts/run_type_probe.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import structlog

structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)

log = structlog.get_logger()

RESULTS_DIR = Path("results/type-probe")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "EleutherAI/pythia-160m-deduped"

# ══════════════════════════════════════════════════════════════════════
# Semantic Type Labels (simplified Montague)
# ══════════════════════════════════════════════════════════════════════
#
# Full Montague types are recursive (<e,<e,t>>, <<e,t>,<<e,t>,t>>, etc.)
# but for a linear probe we need flat categories. These capture the
# major type distinctions that matter for composition:
#
#   ENTITY     e           proper nouns, bare nouns as constants
#   PRED       <e,t>       intransitive verbs
#   REL        <e,<e,t>>   transitive verbs
#   QUANT      <<e,t>,t>   quantifier words (every, some, no)
#   DET        <e,t>→e     determiners (the, a)
#   CONN       t→t→t       connectives (and, or, if, not)
#   MOD        various     adjectives, adverbs
#   FUNC       (structural) punctuation, particles, function words

# Labeled sentences: (sentence, [(token_substring, type), ...])
# We label at the WORD level; the probe maps each token's residual
# to the type of the word it belongs to.

LABELED_DATA = [
    # Simple predication
    ("The dog runs.", [
        ("The", "DET"), ("dog", "ENTITY"), ("runs", "PRED"), (".", "FUNC"),
    ]),
    ("The bird flies.", [
        ("The", "DET"), ("bird", "ENTITY"), ("flies", "PRED"), (".", "FUNC"),
    ]),
    ("The cat sleeps.", [
        ("The", "DET"), ("cat", "ENTITY"), ("sleeps", "PRED"), (".", "FUNC"),
    ]),
    ("The teacher laughs.", [
        ("The", "DET"), ("teacher", "ENTITY"), ("laughs", "PRED"), (".", "FUNC"),
    ]),
    ("The fish swims.", [
        ("The", "DET"), ("fish", "ENTITY"), ("swims", "PRED"), (".", "FUNC"),
    ]),
    ("The farmer walks.", [
        ("The", "DET"), ("farmer", "ENTITY"), ("walks", "PRED"), (".", "FUNC"),
    ]),
    ("The singer dances.", [
        ("The", "DET"), ("singer", "ENTITY"), ("dances", "PRED"), (".", "FUNC"),
    ]),
    ("The child cries.", [
        ("The", "DET"), ("child", "ENTITY"), ("cries", "PRED"), (".", "FUNC"),
    ]),
    # Proper nouns
    ("Alice runs.", [
        ("Alice", "ENTITY"), ("runs", "PRED"), (".", "FUNC"),
    ]),
    ("Bob sleeps.", [
        ("Bob", "ENTITY"), ("sleeps", "PRED"), (".", "FUNC"),
    ]),
    ("Tom walks.", [
        ("Tom", "ENTITY"), ("walks", "PRED"), (".", "FUNC"),
    ]),
    ("Mary sings.", [
        ("Mary", "ENTITY"), ("sings", "PRED"), (".", "FUNC"),
    ]),
    # Transitive
    ("Alice loves Bob.", [
        ("Alice", "ENTITY"), ("loves", "REL"), ("Bob", "ENTITY"), (".", "FUNC"),
    ]),
    ("The dog sees the cat.", [
        ("The", "DET"), ("dog", "ENTITY"), ("sees", "REL"),
        ("the", "DET"), ("cat", "ENTITY"), (".", "FUNC"),
    ]),
    ("Tom helps Mary.", [
        ("Tom", "ENTITY"), ("helps", "REL"), ("Mary", "ENTITY"), (".", "FUNC"),
    ]),
    ("The teacher reads the book.", [
        ("The", "DET"), ("teacher", "ENTITY"), ("reads", "REL"),
        ("the", "DET"), ("book", "ENTITY"), (".", "FUNC"),
    ]),
    ("The farmer finds the bird.", [
        ("The", "DET"), ("farmer", "ENTITY"), ("finds", "REL"),
        ("the", "DET"), ("bird", "ENTITY"), (".", "FUNC"),
    ]),
    ("Alice watches Bob.", [
        ("Alice", "ENTITY"), ("watches", "REL"), ("Bob", "ENTITY"), (".", "FUNC"),
    ]),
    # Quantified
    ("Every dog runs.", [
        ("Every", "QUANT"), ("dog", "ENTITY"), ("runs", "PRED"), (".", "FUNC"),
    ]),
    ("Some cat sleeps.", [
        ("Some", "QUANT"), ("cat", "ENTITY"), ("sleeps", "PRED"), (".", "FUNC"),
    ]),
    ("No bird flies.", [
        ("No", "QUANT"), ("bird", "ENTITY"), ("flies", "PRED"), (".", "FUNC"),
    ]),
    ("Every student reads a book.", [
        ("Every", "QUANT"), ("student", "ENTITY"), ("reads", "REL"),
        ("a", "DET"), ("book", "ENTITY"), (".", "FUNC"),
    ]),
    ("Some teacher laughs.", [
        ("Some", "QUANT"), ("teacher", "ENTITY"), ("laughs", "PRED"), (".", "FUNC"),
    ]),
    ("No fish swims.", [
        ("No", "QUANT"), ("fish", "ENTITY"), ("swims", "PRED"), (".", "FUNC"),
    ]),
    # Modifiers
    ("The tall dog runs.", [
        ("The", "DET"), ("tall", "MOD"), ("dog", "ENTITY"),
        ("runs", "PRED"), (".", "FUNC"),
    ]),
    ("The small cat sleeps.", [
        ("The", "DET"), ("small", "MOD"), ("cat", "ENTITY"),
        ("sleeps", "PRED"), (".", "FUNC"),
    ]),
    ("Tom runs quickly.", [
        ("Tom", "ENTITY"), ("runs", "PRED"), ("quickly", "MOD"), (".", "FUNC"),
    ]),
    ("The bird flies slowly.", [
        ("The", "DET"), ("bird", "ENTITY"), ("flies", "PRED"),
        ("slowly", "MOD"), (".", "FUNC"),
    ]),
    ("The brave farmer walks.", [
        ("The", "DET"), ("brave", "MOD"), ("farmer", "ENTITY"),
        ("walks", "PRED"), (".", "FUNC"),
    ]),
    # Connectives
    ("If the dog runs, the cat sleeps.", [
        ("If", "CONN"), ("the", "DET"), ("dog", "ENTITY"), ("runs", "PRED"),
        (",", "FUNC"), ("the", "DET"), ("cat", "ENTITY"),
        ("sleeps", "PRED"), (".", "FUNC"),
    ]),
    ("Alice runs and Bob sleeps.", [
        ("Alice", "ENTITY"), ("runs", "PRED"), ("and", "CONN"),
        ("Bob", "ENTITY"), ("sleeps", "PRED"), (".", "FUNC"),
    ]),
    ("The dog runs or the cat sleeps.", [
        ("The", "DET"), ("dog", "ENTITY"), ("runs", "PRED"), ("or", "CONN"),
        ("the", "DET"), ("cat", "ENTITY"), ("sleeps", "PRED"), (".", "FUNC"),
    ]),
    # Copular (adjective as predicate)
    ("The dog is tall.", [
        ("The", "DET"), ("dog", "ENTITY"), ("is", "FUNC"),
        ("tall", "PRED"), (".", "FUNC"),
    ]),
    ("Alice is brave.", [
        ("Alice", "ENTITY"), ("is", "FUNC"), ("brave", "PRED"), (".", "FUNC"),
    ]),
    # Negation
    ("The dog does not run.", [
        ("The", "DET"), ("dog", "ENTITY"), ("does", "FUNC"),
        ("not", "CONN"), ("run", "PRED"), (".", "FUNC"),
    ]),
]


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════
# Residual Stream Capture
# ══════════════════════════════════════════════════════════════════════


def capture_all_residuals(model, tokenizer, text, n_layers):
    """Capture the residual stream at every layer for every token.

    Returns dict: layer_idx → tensor of shape (seq_len, hidden_size)
    Also returns: layer -1 = embedding output (before any transformer layer)
    """
    from verbum.instrument import _get_layers

    layers_module = _get_layers(model)
    residuals = {}
    hooks = []

    # Capture embedding output (before L0)
    embed_output = {}

    def embed_hook(module, args, output):
        # For GPTNeoX, the embedding layer outputs hidden_states
        if isinstance(output, tuple):
            embed_output["hidden"] = output[0][0].detach().cpu().float()
        else:
            embed_output["hidden"] = output[0].detach().cpu().float()

    # Hook the embedding layer
    if hasattr(model, "gpt_neox"):
        h = model.gpt_neox.embed_in.register_forward_hook(embed_hook)
        hooks.append(h)

    # Hook each transformer layer
    for layer_idx in range(n_layers):
        storage = {}

        def make_hook(idx, store):
            def hook_fn(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                store["hidden"] = hidden[0].detach().cpu().float()
            return hook_fn

        h = layers_module[layer_idx].register_forward_hook(make_hook(layer_idx, storage))
        hooks.append(h)
        residuals[layer_idx] = storage

    try:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        prev_attn = model.config.output_attentions
        model.config.output_attentions = False
        try:
            with torch.no_grad():
                model(**inputs)
        finally:
            model.config.output_attentions = prev_attn
    finally:
        for h in hooks:
            h.remove()

    result = {}
    if "hidden" in embed_output:
        result[-1] = embed_output["hidden"].numpy()
    for layer_idx in range(n_layers):
        if "hidden" in residuals[layer_idx]:
            result[layer_idx] = residuals[layer_idx]["hidden"].numpy()

    return result, inputs["input_ids"][0].tolist()


# ══════════════════════════════════════════════════════════════════════
# Build Probing Dataset
# ══════════════════════════════════════════════════════════════════════


def build_probing_dataset(model, tokenizer, n_layers):
    """Build (residual_vector, type_label) pairs for every token at every layer."""
    # layer_idx → list of (vector, label)
    data_by_layer = {L: ([], []) for L in range(-1, n_layers)}

    n_labeled = 0
    n_skipped = 0

    for sent, word_labels in LABELED_DATA:
        residuals, token_ids = capture_all_residuals(model, tokenizer, sent, n_layers)

        # Decode each token to find which word it belongs to
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]

        # Match tokens to word labels
        # Strategy: walk through tokens and word labels simultaneously
        word_idx = 0
        char_pos = 0  # position in the sentence

        for tok_idx, tok_str in enumerate(token_strs):
            if word_idx >= len(word_labels):
                break

            word_text, word_type = word_labels[word_idx]

            # Check if this token is part of the current word
            # Strip leading space from token (GPTNeoX adds space prefix)
            tok_clean = tok_str.lstrip()

            # Find this token in the sentence starting from char_pos
            remaining = sent[char_pos:]

            if tok_clean and word_text.lower().startswith(tok_clean.lower()):
                # Token matches start of current word
                for L in range(-1, n_layers):
                    if L in residuals:
                        data_by_layer[L][0].append(residuals[L][tok_idx])
                        data_by_layer[L][1].append(word_type)
                n_labeled += 1

                # If token fully covers the word, advance word_idx
                if len(tok_clean) >= len(word_text):
                    word_idx += 1
                    char_pos += len(tok_str.lstrip())
                else:
                    char_pos += len(tok_clean)
            elif tok_clean and remaining.lstrip().startswith(tok_clean):
                # Token matches but we might have skipped whitespace
                # Still assign current word's type
                for L in range(-1, n_layers):
                    if L in residuals:
                        data_by_layer[L][0].append(residuals[L][tok_idx])
                        data_by_layer[L][1].append(word_type)
                n_labeled += 1
                char_pos = sent.index(tok_clean, char_pos) + len(tok_clean)

                if char_pos >= sent.index(word_text, max(0, char_pos - len(word_text) - 2)) + len(word_text):
                    word_idx += 1
            else:
                # Token doesn't clearly match — skip or assign FUNC
                n_skipped += 1

    # Convert to numpy
    result = {}
    for L in range(-1, n_layers):
        X_list, y_list = data_by_layer[L]
        if X_list:
            result[L] = (np.array(X_list), np.array(y_list))

    return result, n_labeled, n_skipped


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main():
    from verbum.instrument import load_model

    start = time.time()
    banner(f"TYPE PROBE — {datetime.now(UTC).isoformat()}")

    model, tokenizer, info = load_model(MODEL, dtype=torch.float32)
    print(f"  Model: {MODEL}")
    print(f"  Layers: {info.n_layers}  Hidden: {info.hidden_size}")
    print(f"  Sentences: {len(LABELED_DATA)}")

    # Count labels
    from collections import Counter
    all_labels = []
    for _, word_labels in LABELED_DATA:
        for _, wtype in word_labels:
            all_labels.append(wtype)
    label_counts = Counter(all_labels)
    print(f"  Token labels: {dict(label_counts)}")
    print(f"  Total labeled tokens: {len(all_labels)}")

    # Build dataset
    banner("BUILDING PROBING DATASET")
    data_by_layer, n_labeled, n_skipped = build_probing_dataset(
        model, tokenizer, info.n_layers
    )
    print(f"  Labeled: {n_labeled}  Skipped: {n_skipped}")

    # Check class distribution at layer 0
    if 0 in data_by_layer:
        X, y = data_by_layer[0]
        print(f"  Dataset shape: X={X.shape}  y={y.shape}")
        vc = Counter(y)
        for cls, cnt in sorted(vc.items()):
            print(f"    {cls:8s}: {cnt}")

    # ── Train linear probes ───────────────────────────────────────────
    banner("TRAINING LINEAR PROBES (per layer)")
    print(f"  Method: Logistic Regression, 5-fold cross-validation")
    print(f"  Baseline (most frequent class): {max(label_counts.values())/sum(label_counts.values()):.0%}\n")

    layer_accuracies = {}
    layer_names = sorted(data_by_layer.keys())

    for L in layer_names:
        X, y = data_by_layer[L]
        if len(set(y)) < 2:
            print(f"  L{L:2d}: SKIP (only 1 class)")
            continue

        # Encode labels
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        # 5-fold CV
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        try:
            scores = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy")
            mean_acc = scores.mean()
            std_acc = scores.std()
        except Exception as e:
            print(f"  L{L:2d}: ERROR — {e}")
            continue

        layer_accuracies[L] = {"mean": float(mean_acc), "std": float(std_acc)}

        # Visualize
        label = "embed" if L == -1 else f"L{L}"
        bar = "█" * int(mean_acc * 40) + "░" * (40 - int(mean_acc * 40))
        critical = " ← CRITICAL" if L in [0, 3] else ""
        selective = " ← SELECTIVE" if L in [8, 9, 11] else ""
        print(f"  {label:5s}: {bar} {mean_acc:.1%} ±{std_acc:.1%}{critical}{selective}")

    # ── Per-class accuracy at key layers ──────────────────────────────
    banner("PER-CLASS ACCURACY AT KEY LAYERS")

    for L in [-1, 0, 3, 5, 8, 11]:
        if L not in data_by_layer:
            continue
        X, y = data_by_layer[L]
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X, y_enc)
        preds = clf.predict(X)  # train accuracy (not CV, but shows per-class)

        label = "embed" if L == -1 else f"L{L}"
        print(f"\n  {label}:")
        classes = le.classes_
        for cls_idx, cls_name in enumerate(classes):
            mask = y == cls_name
            if mask.sum() == 0:
                continue
            cls_acc = (preds[mask] == cls_idx).mean()
            n = mask.sum()
            print(f"    {cls_name:8s}: {cls_acc:.0%} ({n} tokens)")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start
    banner(f"SUMMARY — {elapsed:.0f}s")

    # Find peak layer
    if layer_accuracies:
        peak_layer = max(layer_accuracies, key=lambda k: layer_accuracies[k]["mean"])
        peak_acc = layer_accuracies[peak_layer]["mean"]
        peak_label = "embed" if peak_layer == -1 else f"L{peak_layer}"
        print(f"  Peak type decodability: {peak_label} at {peak_acc:.1%}")

        # Key comparisons
        embed_acc = layer_accuracies.get(-1, {}).get("mean", 0)
        l0_acc = layer_accuracies.get(0, {}).get("mean", 0)
        l3_acc = layer_accuracies.get(3, {}).get("mean", 0)
        l8_acc = layer_accuracies.get(8, {}).get("mean", 0)
        l11_acc = layer_accuracies.get(11, {}).get("mean", 0)

        print(f"\n  Type decodability at key layers:")
        print(f"    Embedding:  {embed_acc:.1%}")
        print(f"    L0 (crit):  {l0_acc:.1%}  Δ from embed: {l0_acc-embed_acc:+.1%}")
        print(f"    L3 (crit):  {l3_acc:.1%}  Δ from L0:    {l3_acc-l0_acc:+.1%}")
        print(f"    L8 (sel):   {l8_acc:.1%}  Δ from L3:    {l8_acc-l3_acc:+.1%}")
        print(f"    L11 (sel):  {l11_acc:.1%}  Δ from L8:    {l11_acc-l8_acc:+.1%}")

        if l3_acc > embed_acc + 0.05:
            print(f"\n  ✓ Types become MORE decodable L0→L3: type assignment confirmed")
        elif l3_acc < embed_acc - 0.05:
            print(f"\n  ✗ Types become LESS decodable L0→L3: types are in embeddings, "
                  f"L0-L3 do something else")
        else:
            print(f"\n  ~ Types roughly stable embed→L3: type info may be in embeddings already")

    # Save
    save_path = RESULTS_DIR / "type-probe-summary.json"
    save_path.write_text(json.dumps({
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "model": MODEL,
        "n_sentences": len(LABELED_DATA),
        "n_labeled_tokens": n_labeled,
        "label_counts": dict(label_counts),
        "layer_accuracies": {str(k): v for k, v in layer_accuracies.items()},
    }, indent=2, ensure_ascii=False))
    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
