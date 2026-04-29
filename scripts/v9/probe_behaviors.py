"""
Probe: Do behaviors form distinct basins in Qwen3-32B?

Kernel ops are atoms. Behaviors are programs — compositional patterns
that activate entire circuits. "Summarize" and "TL;DR" should land in
the same behavioral attractor basin.

Questions:
  1. Do different phrasings of the same behavior cluster?
  2. Where do behavior basins emerge? Same layers as type basins (L28-37)
     or later (compositional)?
  3. How many natural behavior basins exist?
  4. Do they map to the kernel's compositional structure?
  5. Is compute-behavior distinct from language-behavior?

Strategy:
  Extract activation at the BEHAVIOR WORD — the token that signals
  what the model should DO. "Summarize the text" → extract at "Summarize".
  Also extract at LAST token for full-context representation.

Uses same Qwen3-32B GGUF model.

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
# Behavior probes: different phrasings of the same behavioral intent
# ══════════════════════════════════════════════════════════════════════

BEHAVIOR_PROBES = [
    # ── Computation behaviors ──
    ("compute", "behavior:calculate", [
        ("Calculate", "Calculate the total cost of these items."),
        ("Compute", "Compute the average of these numbers."),
        ("Evaluate", "Evaluate this expression for x equals five."),
        ("Solve", "Solve this equation for the unknown variable."),
        ("Work", "Work out the answer to this problem."),
        ("Figure", "Figure out how much this costs."),
    ]),
    ("count", "behavior:enumerate", [
        ("Count", "Count the number of items in the list."),
        ("Tally", "Tally up all the votes cast today."),
        ("Number", "Number the items in the collection."),
        ("Enumerate", "Enumerate all possible outcomes here."),
        ("Total", "Total the entries in this column."),
    ]),

    # ── Compression behaviors ──
    ("summarize", "behavior:compress", [
        ("Summarize", "Summarize the main points of this article."),
        ("Condense", "Condense this report into key findings."),
        ("Brief", "Brief the team on the current situation."),
        ("Distill", "Distill the essence of this argument."),
        ("Recap", "Recap what happened in the meeting."),
        ("Abbreviate", "Abbreviate this lengthy description please."),
    ]),
    ("simplify", "behavior:reduce_complexity", [
        ("Simplify", "Simplify this explanation for a beginner."),
        ("Clarify", "Clarify what this paragraph means exactly."),
        ("Explain", "Explain this concept in simple terms."),
        ("Unpack", "Unpack this dense technical passage."),
        ("Break", "Break this down into simpler parts."),
    ]),

    # ── Generation behaviors ──
    ("create", "behavior:generate", [
        ("Create", "Create a new function that sorts data."),
        ("Generate", "Generate a list of test cases."),
        ("Write", "Write a function to parse this format."),
        ("Build", "Build a solution for this problem."),
        ("Produce", "Produce a report from this data."),
        ("Compose", "Compose a response to this inquiry."),
    ]),
    ("expand", "behavior:elaborate", [
        ("Expand", "Expand on this idea with more detail."),
        ("Elaborate", "Elaborate on the implications of this."),
        ("Detail", "Detail the steps required for this."),
        ("Develop", "Develop this concept further please."),
        ("Flesh", "Flesh out the outline with content."),
    ]),

    # ── Transformation behaviors ──
    ("translate", "behavior:transform_language", [
        ("Translate", "Translate this sentence into French."),
        ("Convert", "Convert this text to formal language."),
        ("Rephrase", "Rephrase this in more professional terms."),
        ("Rewrite", "Rewrite this paragraph more concisely."),
        ("Paraphrase", "Paraphrase the key argument here."),
        ("Reformulate", "Reformulate this as a question."),
    ]),
    ("transform_data", "behavior:transform_structure", [
        ("Sort", "Sort these items by their priority."),
        ("Filter", "Filter out the irrelevant entries."),
        ("Organize", "Organize this data by category."),
        ("Restructure", "Restructure the code for clarity."),
        ("Rearrange", "Rearrange the sections logically."),
        ("Format", "Format this output as a table."),
    ]),

    # ── Analysis behaviors ──
    ("compare", "behavior:contrast", [
        ("Compare", "Compare these two approaches carefully."),
        ("Contrast", "Contrast the advantages and disadvantages."),
        ("Differentiate", "Differentiate between these two methods."),
        ("Distinguish", "Distinguish the key differences here."),
        ("Weigh", "Weigh the pros and cons of each."),
    ]),
    ("analyze", "behavior:decompose", [
        ("Analyze", "Analyze the root cause of this failure."),
        ("Examine", "Examine the evidence for this claim."),
        ("Investigate", "Investigate why this test is failing."),
        ("Diagnose", "Diagnose the problem in this system."),
        ("Inspect", "Inspect the output for any errors."),
        ("Dissect", "Dissect the argument into its parts."),
    ]),

    # ── Evaluation behaviors ──
    ("judge", "behavior:evaluate_quality", [
        ("Judge", "Judge the quality of this solution."),
        ("Assess", "Assess the risk of this approach."),
        ("Rate", "Rate the effectiveness of this method."),
        ("Review", "Review the code for potential issues."),
        ("Critique", "Critique the design of this system."),
        ("Evaluate", "Evaluate the performance of the model."),
    ]),
    ("verify", "behavior:check_correctness", [
        ("Verify", "Verify that this answer is correct."),
        ("Check", "Check the output against expected results."),
        ("Validate", "Validate the input data before processing."),
        ("Confirm", "Confirm that the test passes correctly."),
        ("Test", "Test whether this function handles edge cases."),
        ("Prove", "Prove that this invariant always holds."),
    ]),

    # ── Search/retrieval behaviors ──
    ("find", "behavior:search", [
        ("Find", "Find the error in this code."),
        ("Locate", "Locate the source of this bug."),
        ("Search", "Search for patterns matching this criteria."),
        ("Identify", "Identify which component is failing."),
        ("Detect", "Detect any anomalies in this data."),
        ("Discover", "Discover the underlying cause of this."),
    ]),

    # ── Lambda/formal behaviors (the compiler circuit) ──
    ("compile", "behavior:formalize", [
        ("Formalize", "Formalize this natural language statement."),
        ("Encode", "Encode this meaning as a logical form."),
        ("Express", "Express this constraint mathematically."),
        ("Represent", "Represent this relationship formally."),
        ("Define", "Define this concept precisely and formally."),
        ("Specify", "Specify the requirements in formal notation."),
    ]),
    ("decompose_formal", "behavior:decompile", [
        ("Interpret", "Interpret this formula in plain language."),
        ("Decode", "Decode this notation into readable text."),
        ("Describe", "Describe what this function does."),
        ("Narrate", "Narrate the steps of this algorithm."),
        ("Verbalize", "Verbalize the meaning of this expression."),
    ]),

    # ── Control behaviors ──
    ("decide", "behavior:branch", [
        ("Decide", "Decide which approach to take here."),
        ("Choose", "Choose the best option available now."),
        ("Select", "Select the appropriate method for this."),
        ("Pick", "Pick the most efficient algorithm."),
        ("Determine", "Determine the correct course of action."),
    ]),
    ("plan", "behavior:sequence", [
        ("Plan", "Plan the steps to complete this task."),
        ("Outline", "Outline the approach for this project."),
        ("Design", "Design a strategy to solve this."),
        ("Architect", "Architect a solution for scalability."),
        ("Map", "Map out the dependencies between tasks."),
        ("Sequence", "Sequence the operations in the right order."),
    ]),
]


def find_target_token_indices(tokenizer, input_ids, target_word):
    """Find token positions for target word."""
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
    print(f"Loaded in {time.time()-t0:.1f}s: {model.config.num_hidden_layers} layers, "
          f"d={model.config.hidden_size}")
    return model, tokenizer


def run_with_hooks(model, tokenizer, text, device="mps"):
    """Forward pass with hooks on all layers."""
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


def extract_behavior_activations(model, tokenizer, device="mps"):
    """Extract activations at behavior word AND at last token."""
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    word_activations = {}  # at the behavior word
    last_activations = {}  # at last token (full context)
    metadata = {}
    total = sum(len(items) for _, _, items in BEHAVIOR_PROBES)
    done = 0

    for group_name, behavior_label, items in BEHAVIOR_PROBES:
        word_activations[group_name] = {}
        last_activations[group_name] = {}
        metadata[group_name] = {"behavior": behavior_label, "items": {}}

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
            seq_len = input_ids.shape[1]

            # At behavior word
            word_acts = np.zeros((n_layers, d_model), dtype=np.float32)
            # At last token
            last_acts = np.zeros((n_layers, d_model), dtype=np.float32)

            for li in range(n_layers):
                h = layer_outputs[li]
                word_acts[li] = h[0, target_indices, :].mean(dim=0).numpy()
                last_acts[li] = h[0, seq_len - 1, :].numpy()

            key = f"{target_word}_{hash(sentence) % 10000:04d}"
            word_activations[group_name][key] = word_acts
            last_activations[group_name][key] = last_acts
            metadata[group_name]["items"][key] = {
                "word": target_word, "sentence": sentence,
                "token_indices": target_indices,
                "tokens": [tokens[i] for i in target_indices],
            }

            done += 1
            if done % 20 == 0 or done == total:
                print(f"  [{done}/{total}] {group_name}: '{target_word}'")

    return word_activations, last_activations, metadata


def compute_scores(activations, layer_range):
    """Within/between scores across layers."""
    first_group = next(iter(activations.values()))
    first_acts = next(iter(first_group.values()))
    n_layers = first_acts.shape[0]

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

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf",
        default="/Users/mwhitford/localai/models/Qwen3-32B-Q8_0.gguf")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", default="results/behavior-basins")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.gguf, device=args.device)

    # Extract activations
    print("\n=== Extracting behavior activations ===")
    word_acts, last_acts, metadata = extract_behavior_activations(
        model, tokenizer, device=args.device
    )

    # Save
    word_npz = {}
    last_npz = {}
    for gn in word_acts:
        for key, acts in word_acts[gn].items():
            word_npz[f"{gn}__{key}"] = acts
        for key, acts in last_acts[gn].items():
            last_npz[f"{gn}__{key}"] = acts
    np.savez_compressed(out_dir / "behavior_word_activations.npz", **word_npz)
    np.savez_compressed(out_dir / "behavior_last_activations.npz", **last_npz)
    with open(out_dir / "behavior_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {len(word_npz)} behavior vectors (word + last token)")

    # Full layer sweep — where do behaviors cluster?
    print("\n=== Behavior word: full layer sweep ===")
    word_scores = compute_scores(word_acts, range(0, 64, 2))
    for li in sorted(word_scores.keys()):
        s = word_scores[li]
        if li % 8 == 0:
            print(f"  Layer {li:2d}: within={s['within_mean']:.4f} "
                  f"between={s['between_mean']:.4f} "
                  f"ratio={s['ratio']:.3f} sep={s['separation']:+.4f}")

    with open(out_dir / "behavior_word_layer_scores.json", "w") as f:
        json.dump(word_scores, f, indent=2)

    # Find peak
    peak_l = max(word_scores, key=lambda l: word_scores[l]["separation"])
    print(f"\nPeak behavior-word layer: {peak_l} "
          f"(ratio={word_scores[peak_l]['ratio']:.3f}, "
          f"sep={word_scores[peak_l]['separation']:+.4f})")

    print(f"\nPer-behavior within-similarity at L{peak_l}:")
    for gn, gs in sorted(
        word_scores[peak_l]["per_group"].items(),
        key=lambda x: -x[1]["within_mean"]
    ):
        print(f"  {gn:25s}: {gs['within_mean']:.4f} ({gs['n_pairs']} pairs)")

    # Last-token sweep
    print("\n=== Behavior last-token: full layer sweep ===")
    last_scores = compute_scores(last_acts, range(0, 64, 2))
    for li in sorted(last_scores.keys()):
        s = last_scores[li]
        if li % 8 == 0:
            print(f"  Layer {li:2d}: within={s['within_mean']:.4f} "
                  f"between={s['between_mean']:.4f} "
                  f"ratio={s['ratio']:.3f} sep={s['separation']:+.4f}")

    with open(out_dir / "behavior_last_layer_scores.json", "w") as f:
        json.dump(last_scores, f, indent=2)

    peak_last = max(last_scores, key=lambda l: last_scores[l]["separation"])
    print(f"\nPeak behavior-last layer: {peak_last} "
          f"(ratio={last_scores[peak_last]['ratio']:.3f}, "
          f"sep={last_scores[peak_last]['separation']:+.4f})")

    print(f"\nPer-behavior within-similarity at L{peak_last} (last token):")
    for gn, gs in sorted(
        last_scores[peak_last]["per_group"].items(),
        key=lambda x: -x[1]["within_mean"]
    ):
        print(f"  {gn:25s}: {gs['within_mean']:.4f} ({gs['n_pairs']} pairs)")

    # Compare: do behaviors peak at same layer as types or later?
    print("\n=== Comparison: Type vs Behavior peak layers ===")
    print(f"  Type basins (probe 1):     L28 (ratio 3.9x)")
    print(f"  Behavior word basins:      L{peak_l} (ratio {word_scores[peak_l]['ratio']:.1f}x)")
    print(f"  Behavior last-token basins: L{peak_last} (ratio {last_scores[peak_last]['ratio']:.1f}x)")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
