"""
Probe: Do kernel operations form semantic basins in Qwen3-32B?

Two-level probe:
  Level 1 — Operator word clustering:
    Multiple phrasings of each of the 22 kernel ops.
    Does "add/plus/sum/combine" cluster separately from "subtract/minus/reduce"?

  Level 2 — Expression clustering:
    Equivalent computations in different notation.
    Does "(+ 3 4)" cluster with "three plus four" at the composition point?
    Extracts activation at the LAST token (where result is composed).

If both levels work: the ascending arm IS kernel dispatch.
The type basin routes to the kernel op with no symbolic type system.

Uses the same Qwen3-32B GGUF as probe_clusters.py.
Focuses on the typing zone (layers 26-37) identified in the first probe.

License: MIT
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ══════════════════════════════════════════════════════════════════════
# Level 1: Operator word clustering
# Each kernel op gets multiple natural language phrasings.
# Target word extracted in context.
# ══════════════════════════════════════════════════════════════════════

OPERATOR_PROBES = [
    # ── Arithmetic binary (7 ops) ──
    ("add", "kernel:add", [
        ("add", "Please add the two numbers together."),
        ("plus", "The answer is three plus four."),
        ("sum", "Compute the sum of these values."),
        ("combine", "Combine the values into a total."),
        ("increase", "Increase the count by five."),
        ("addition", "Perform the addition operation now."),
    ]),
    ("subtract", "kernel:sub", [
        ("subtract", "Please subtract three from seven."),
        ("minus", "The answer is seven minus three."),
        ("difference", "Find the difference between them."),
        ("decrease", "Decrease the total by two."),
        ("reduce", "Reduce the amount by three."),
        ("deduct", "Deduct the cost from the balance."),
    ]),
    ("multiply", "kernel:mul", [
        ("multiply", "Please multiply four by five."),
        ("times", "The answer is four times five."),
        ("product", "Compute the product of these numbers."),
        ("double", "Double the current value now."),
        ("scale", "Scale the input by three."),
    ]),
    ("divide", "kernel:div", [
        ("divide", "Please divide ten by two."),
        ("divided", "Ten divided by two equals five."),
        ("quotient", "Find the quotient of the division."),
        ("split", "Split the total into equal parts."),
        ("halve", "Halve the remaining amount now."),
    ]),
    ("modulo", "kernel:mod", [
        ("remainder", "Find the remainder after division."),
        ("modulo", "Compute seven modulo three now."),
        ("leftover", "What is the leftover after dividing."),
        ("mod", "Calculate ten mod three for me."),
    ]),
    ("minimum", "kernel:min", [
        ("minimum", "Find the minimum of these values."),
        ("smallest", "Return the smallest number here."),
        ("least", "Which is the least of all."),
        ("lowest", "Select the lowest value available."),
        ("min", "Compute the min of the set."),
    ]),
    ("maximum", "kernel:max", [
        ("maximum", "Find the maximum of these values."),
        ("largest", "Return the largest number here."),
        ("greatest", "Which is the greatest of all."),
        ("highest", "Select the highest value available."),
        ("max", "Compute the max of the set."),
    ]),

    # ── Comparison (5 ops) ──
    ("equal", "kernel:eq", [
        ("equal", "Check if the values are equal."),
        ("equals", "Three plus four equals seven exactly."),
        ("same", "Are these two numbers the same."),
        ("identical", "The results are identical in value."),
        ("matches", "The output matches the expected result."),
    ]),
    ("less_than", "kernel:lt", [
        ("less", "Three is less than five always."),
        ("smaller", "Three is smaller than five here."),
        ("below", "The value is below the threshold."),
        ("under", "The count is under the limit."),
        ("fewer", "There are fewer items than expected."),
    ]),
    ("greater_than", "kernel:gt", [
        ("greater", "Five is greater than three always."),
        ("larger", "Five is larger than three here."),
        ("above", "The value is above the threshold."),
        ("exceeds", "The count exceeds the limit now."),
        ("more", "There are more items than expected."),
    ]),

    # ── Boolean (3 ops) ──
    ("and_op", "kernel:and", [
        ("and", "Both conditions must be true and valid."),
        ("both", "Both values must satisfy the constraint."),
        ("conjunction", "Form the conjunction of these propositions."),
        ("together", "Both conditions hold together here."),
    ]),
    ("or_op", "kernel:or", [
        ("or", "Either condition can be true or false."),
        ("either", "Either value satisfies the requirement here."),
        ("disjunction", "Form the disjunction of the propositions."),
        ("alternatively", "Alternatively the second condition holds."),
    ]),
    ("not_op", "kernel:not", [
        ("not", "The condition is not satisfied here."),
        ("negation", "Apply the negation to the result."),
        ("negate", "Negate the boolean value entirely."),
        ("opposite", "Return the opposite of the truth."),
        ("false", "The statement evaluates to false now."),
    ]),

    # ── Unary (2 ops) ──
    ("absolute", "kernel:abs", [
        ("absolute", "Find the absolute value of negative."),
        ("magnitude", "Compute the magnitude of this number."),
        ("abs", "Take the abs of negative five."),
        ("distance", "The distance from zero is five."),
    ]),
    ("negate_num", "kernel:neg", [
        ("negate", "Negate the positive number to negative."),
        ("negative", "Make the number negative now please."),
        ("invert", "Invert the sign of the value."),
        ("flip", "Flip the sign from positive here."),
        ("reverse", "Reverse the sign of the number."),
    ]),

    # ── Conditional ──
    ("conditional", "kernel:if", [
        ("if", "If the condition holds then proceed."),
        ("when", "When the value exceeds five stop."),
        ("condition", "The condition determines which branch runs."),
        ("conditional", "Apply the conditional logic to decide."),
        ("whether", "Check whether the test passes first."),
        ("choose", "Choose the result based on truth."),
    ]),

    # ── Higher-order (3 ops) ──
    ("partial_app", "kernel:partial", [
        ("partial", "Create a partial application of add."),
        ("bind", "Bind the first argument to three."),
        ("fix", "Fix the first parameter to five."),
        ("curry", "Curry the function with one argument."),
        ("preset", "Preset the initial value to ten."),
    ]),
    ("compose", "kernel:compose", [
        ("compose", "Compose the two functions into one."),
        ("chain", "Chain the operations together sequentially."),
        ("pipe", "Pipe the output into the next."),
        ("combine", "Combine the functions into a pipeline."),
        ("sequence", "Sequence the transformations in order."),
    ]),
    ("apply", "kernel:apply", [
        ("apply", "Apply the function to the argument."),
        ("call", "Call the function with this value."),
        ("invoke", "Invoke the operation on the input."),
        ("execute", "Execute the function on the data."),
        ("evaluate", "Evaluate the expression to get result."),
    ]),
]


# ══════════════════════════════════════════════════════════════════════
# Level 2: Expression clustering
# Same computation, different notation. Extract at last token.
# ══════════════════════════════════════════════════════════════════════

EXPRESSION_PROBES = [
    # ── Addition ──
    ("expr_add_7", "result:7", [
        ("(+ 3 4)", "S-expr"),
        ("3 + 4", "math"),
        ("three plus four", "prose"),
        ("the sum of three and four", "prose_verbose"),
        ("add(3, 4)", "function_call"),
        ("3 added to 4", "passive"),
    ]),
    ("expr_add_10", "result:10", [
        ("(+ 7 3)", "S-expr"),
        ("7 + 3", "math"),
        ("seven plus three", "prose"),
        ("the sum of seven and three", "prose_verbose"),
    ]),

    # ── Subtraction ──
    ("expr_sub_4", "result:4", [
        ("(- 7 3)", "S-expr"),
        ("7 - 3", "math"),
        ("seven minus three", "prose"),
        ("the difference between seven and three", "prose_verbose"),
        ("subtract 3 from 7", "imperative"),
    ]),

    # ── Multiplication ──
    ("expr_mul_20", "result:20", [
        ("(* 4 5)", "S-expr"),
        ("4 * 5", "math"),
        ("four times five", "prose"),
        ("the product of four and five", "prose_verbose"),
        ("multiply 4 by 5", "imperative"),
    ]),

    # ── Division ──
    ("expr_div_5", "result:5", [
        ("(/ 10 2)", "S-expr"),
        ("10 / 2", "math"),
        ("ten divided by two", "prose"),
        ("half of ten", "prose_short"),
    ]),

    # ── Comparison ──
    ("expr_gt_true", "result:true", [
        ("(> 5 3)", "S-expr"),
        ("5 > 3", "math"),
        ("five is greater than three", "prose"),
        ("five exceeds three", "prose_alt"),
    ]),
    ("expr_lt_true", "result:true", [
        ("(< 2 7)", "S-expr"),
        ("2 < 7", "math"),
        ("two is less than seven", "prose"),
        ("two is smaller than seven", "prose_alt"),
    ]),

    # ── Nested composition ──
    ("expr_nested_23", "result:23", [
        ("(+ 3 (* 4 5))", "S-expr"),
        ("3 + 4 * 5", "math"),
        ("three plus the product of four and five", "prose"),
        ("three plus four times five", "prose_short"),
    ]),
    ("expr_nested_14", "result:14", [
        ("(+ (* 2 3) (* 2 4))", "S-expr"),
        ("2*3 + 2*4", "math"),
        ("two times three plus two times four", "prose"),
    ]),

    # ── Conditional ──
    ("expr_if_yes", "result:10", [
        ("(if (> 5 3) 10 0)", "S-expr"),
        ("if 5 > 3 then 10 else 0", "pseudo"),
        ("ten if five exceeds three otherwise zero", "prose"),
    ]),

    # ── Cross-result: same op, different values ──
    ("expr_add_various", "op:add", [
        ("(+ 1 2)", "S-expr_3"),
        ("(+ 5 5)", "S-expr_10"),
        ("(+ 100 200)", "S-expr_300"),
        ("1 + 2", "math_3"),
        ("5 + 5", "math_10"),
        ("100 + 200", "math_300"),
    ]),
    ("expr_mul_various", "op:mul", [
        ("(* 2 3)", "S-expr_6"),
        ("(* 7 8)", "S-expr_56"),
        ("(* 10 10)", "S-expr_100"),
        ("2 * 3", "math_6"),
        ("7 * 8", "math_56"),
        ("10 * 10", "math_100"),
    ]),
]


def find_target_token_indices(tokenizer, input_ids, target_word):
    """Find token positions for target word. Same as probe_clusters.py."""
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    target_ids = tokenizer.encode(target_word, add_special_tokens=False)
    target_tokens = tokenizer.convert_ids_to_tokens(target_ids)
    space_target_ids = tokenizer.encode(" " + target_word, add_special_tokens=False)
    space_target_tokens = tokenizer.convert_ids_to_tokens(space_target_ids)

    for pattern in [space_target_tokens, target_tokens]:
        pat_len = len(pattern)
        for i in range(len(tokens) - pat_len + 1):
            if tokens[i : i + pat_len] == pattern:
                return list(range(i, i + pat_len))

    indices = []
    for i, tok in enumerate(tokens):
        clean = tok.replace("Ġ", "").replace("▁", "").replace("##", "").lower()
        if target_word.lower() in clean or clean in target_word.lower():
            indices.append(i)
    return indices


def load_model(gguf_path, device="mps"):
    """Load Qwen3-32B from GGUF."""
    gguf_dir = str(Path(gguf_path).parent)
    gguf_file = Path(gguf_path).name

    print(f"Loading model from {gguf_path}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    model = AutoModelForCausalLM.from_pretrained(
        gguf_dir, gguf_file=gguf_file,
        dtype=torch.float16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    t1 = time.time()
    print(f"Loaded in {t1-t0:.1f}s: {model.config.num_hidden_layers} layers, "
          f"d={model.config.hidden_size}")
    return model, tokenizer


def run_with_hooks(model, tokenizer, text, device="mps"):
    """Run text through model, return per-layer hidden states."""
    n_layers = model.config.num_hidden_layers
    layer_outputs = {}

    def make_hook(idx):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            layer_outputs[idx] = h.detach().cpu()
        return hook_fn

    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        _ = model(**inputs)

    for h in hooks:
        h.remove()

    return layer_outputs, inputs["input_ids"]


def extract_operator_activations(model, tokenizer, device="mps"):
    """Level 1: Extract operator word activations."""
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    activations = {}
    metadata = {}
    total = sum(len(items) for _, _, items in OPERATOR_PROBES)
    done = 0

    for group_name, kernel_op, items in OPERATOR_PROBES:
        activations[group_name] = {}
        metadata[group_name] = {"kernel_op": kernel_op, "items": {}}

        for target_word, sentence in items:
            layer_outputs, input_ids = run_with_hooks(
                model, tokenizer, sentence, device
            )
            target_indices = find_target_token_indices(
                tokenizer, input_ids, target_word
            )
            if not target_indices:
                print(f"  WARNING: '{target_word}' not found in '{sentence}'")
                continue

            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            word_acts = np.zeros((n_layers, d_model), dtype=np.float32)
            for li in range(n_layers):
                h = layer_outputs[li]
                target_vecs = h[0, target_indices, :]
                word_acts[li] = target_vecs.mean(dim=0).numpy()

            key = f"{target_word}_{hash(sentence) % 10000:04d}"
            activations[group_name][key] = word_acts
            metadata[group_name]["items"][key] = {
                "word": target_word, "sentence": sentence,
                "token_indices": target_indices,
                "tokens": [tokens[i] for i in target_indices],
            }

            done += 1
            if done % 20 == 0 or done == total:
                print(f"  [{done}/{total}] {group_name}: '{target_word}'")

    return activations, metadata


def extract_expression_activations(model, tokenizer, device="mps"):
    """Level 2: Extract expression-level activations at last token.

    For each expression, we wrap it in a frame: "Compute: {expr} ="
    and extract the activation at the "=" token (where the model
    has composed the computation and is about to produce the result).
    """
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    activations = {}
    metadata = {}
    total = sum(len(items) for _, _, items in EXPRESSION_PROBES)
    done = 0

    for group_name, result_label, items in EXPRESSION_PROBES:
        activations[group_name] = {}
        metadata[group_name] = {"result_label": result_label, "items": {}}

        for expr, notation in items:
            # Frame the expression so the model is primed to compute
            prompt = f"Compute: {expr} ="

            layer_outputs, input_ids = run_with_hooks(
                model, tokenizer, prompt, device
            )

            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            seq_len = input_ids.shape[1]

            # Extract at LAST token position (the "=" where result is composed)
            last_idx = seq_len - 1

            expr_acts = np.zeros((n_layers, d_model), dtype=np.float32)
            for li in range(n_layers):
                h = layer_outputs[li]
                expr_acts[li] = h[0, last_idx, :].numpy()

            key = f"{notation}_{hash(expr) % 10000:04d}"
            activations[group_name][key] = expr_acts
            metadata[group_name]["items"][key] = {
                "expression": expr, "notation": notation,
                "prompt": prompt,
                "last_token_idx": last_idx,
                "tokens": tokens,
            }

            done += 1
            if done % 10 == 0 or done == total:
                print(f"  [{done}/{total}] {group_name}: '{expr}' ({notation})")

    return activations, metadata


def compute_scores(activations, probe_groups, layer_range=None):
    """Compute within/between scores, optionally limited to a layer range."""
    first_group = next(iter(activations.values()))
    first_acts = next(iter(first_group.values()))
    n_layers = first_acts.shape[0]

    if layer_range is None:
        layer_range = range(n_layers)

    def cosine_sim(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    scores = {}
    for li in layer_range:
        within_sims = []
        between_sims = []
        per_group = {}

        all_vecs = []
        all_labels = []

        for group_name in activations:
            group_vecs = []
            for key, acts in activations[group_name].items():
                vec = acts[li]
                group_vecs.append(vec)
                all_vecs.append(vec)
                all_labels.append(group_name)

            gw = []
            for i in range(len(group_vecs)):
                for j in range(i + 1, len(group_vecs)):
                    s = cosine_sim(group_vecs[i], group_vecs[j])
                    gw.append(s)
                    within_sims.append(s)
            per_group[group_name] = {
                "within_mean": float(np.mean(gw)) if gw else 0.0,
                "n_pairs": len(gw),
            }

        group_names = list(set(all_labels))
        for gi in range(len(group_names)):
            for gj in range(gi + 1, len(group_names)):
                vi = [v for v, g in zip(all_vecs, all_labels) if g == group_names[gi]]
                vj = [v for v, g in zip(all_vecs, all_labels) if g == group_names[gj]]
                for a in vi:
                    for b in vj:
                        between_sims.append(cosine_sim(a, b))

        wm = float(np.mean(within_sims)) if within_sims else 0.0
        bm = float(np.mean(between_sims)) if between_sims else 0.0

        scores[li] = {
            "within_mean": wm, "between_mean": bm,
            "ratio": wm / bm if bm > 0 else 0.0,
            "separation": wm - bm,
            "per_group": per_group,
        }

        if li % 8 == 0 or li == max(layer_range):
            print(f"  Layer {li:2d}: within={wm:.4f} between={bm:.4f} "
                  f"ratio={wm/bm if bm > 0 else 0:.3f} sep={wm-bm:+.4f}")

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf",
        default="/Users/mwhitford/localai/models/Qwen3-32B-Q8_0.gguf")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", default="results/kernel-basins")
    parser.add_argument("--level", type=int, default=0,
        help="0=both, 1=operators only, 2=expressions only")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.gguf, device=args.device)

    # ── Level 1: Operator words ──
    if args.level in (0, 1):
        print("\n═══ Level 1: Operator Word Clustering ═══")
        op_acts, op_meta = extract_operator_activations(
            model, tokenizer, device=args.device
        )

        npz = {}
        for gn, words in op_acts.items():
            for key, acts in words.items():
                npz[f"op__{gn}__{key}"] = acts
        np.savez_compressed(out_dir / "operator_activations.npz", **npz)
        with open(out_dir / "operator_metadata.json", "w") as f:
            json.dump(op_meta, f, indent=2)

        print(f"\nSaved {len(npz)} operator vectors")

        print("\n── Operator scores (typing zone L24-40) ──")
        op_scores = compute_scores(op_acts, OPERATOR_PROBES, range(24, 41))
        with open(out_dir / "operator_layer_scores.json", "w") as f:
            json.dump(op_scores, f, indent=2)

        # Best layer
        best_l = max(op_scores, key=lambda l: op_scores[l]["separation"])
        print(f"\nBest operator layer: {best_l} "
              f"(ratio={op_scores[best_l]['ratio']:.3f})")
        print(f"\nPer-op within-similarity at L{best_l}:")
        for gn, gs in sorted(
            op_scores[best_l]["per_group"].items(),
            key=lambda x: -x[1]["within_mean"]
        ):
            print(f"  {gn:20s}: {gs['within_mean']:.4f} ({gs['n_pairs']} pairs)")

    # ── Level 2: Expression clustering ──
    if args.level in (0, 2):
        print("\n═══ Level 2: Expression Clustering ═══")
        expr_acts, expr_meta = extract_expression_activations(
            model, tokenizer, device=args.device
        )

        npz = {}
        for gn, items in expr_acts.items():
            for key, acts in items.items():
                npz[f"expr__{gn}__{key}"] = acts
        np.savez_compressed(out_dir / "expression_activations.npz", **npz)
        with open(out_dir / "expression_metadata.json", "w") as f:
            json.dump(expr_meta, f, indent=2)

        print(f"\nSaved {len(npz)} expression vectors")

        print("\n── Expression scores (typing zone L24-40) ──")
        expr_scores = compute_scores(expr_acts, EXPRESSION_PROBES, range(24, 41))
        with open(out_dir / "expression_layer_scores.json", "w") as f:
            json.dump(expr_scores, f, indent=2)

        best_l = max(expr_scores, key=lambda l: expr_scores[l]["separation"])
        print(f"\nBest expression layer: {best_l} "
              f"(ratio={expr_scores[best_l]['ratio']:.3f})")
        print(f"\nPer-expression within-similarity at L{best_l}:")
        for gn, gs in sorted(
            expr_scores[best_l]["per_group"].items(),
            key=lambda x: -x[1]["within_mean"]
        ):
            print(f"  {gn:25s}: {gs['within_mean']:.4f} ({gs['n_pairs']} pairs)")

        # ── Cross-notation analysis at best layer ──
        print(f"\n── Cross-notation convergence at L{best_l} ──")
        print("Do S-expr, math, and prose for the same computation cluster?")
        for gn, result_label, items in EXPRESSION_PROBES:
            if gn not in expr_acts or len(expr_acts[gn]) < 2:
                continue
            keys = list(expr_acts[gn].keys())
            vecs = [expr_acts[gn][k][best_l] for k in keys]
            notations = [expr_meta[gn]["items"][k]["notation"] for k in keys]
            exprs = [expr_meta[gn]["items"][k]["expression"] for k in keys]

            # Pairwise cosine similarity
            sims = []
            pairs = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    na = np.linalg.norm(vecs[i])
                    nb = np.linalg.norm(vecs[j])
                    s = float(np.dot(vecs[i], vecs[j]) / (na * nb)) if na > 0 and nb > 0 else 0.0
                    sims.append(s)
                    pairs.append((notations[i], notations[j]))

            mean_sim = float(np.mean(sims))
            min_sim = float(np.min(sims))
            print(f"\n  {gn} ({result_label}):")
            print(f"    mean={mean_sim:.4f} min={min_sim:.4f}")
            for (n1, n2), s in zip(pairs, sims):
                marker = "✓" if s > 0.5 else "✗"
                print(f"    {marker} {n1:15s} ↔ {n2:15s}: {s:.4f}")

    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
