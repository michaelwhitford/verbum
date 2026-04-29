"""
Probe: Are behaviors surface or deep?

Hypothesis: Behavioral instruction (Calculate/Summarize/Translate) is
surface-level pattern from instruction tuning — burned into early layers,
not reshaping deep circuits. The type basins at L28-37 should be
INVARIANT to the behavioral frame.

Test: Same content words embedded in different behavioral frames.
Extract activation at the CONTENT word, not the behavior word.
If L28 activations are identical across frames → behaviors are surface.
If they shift → behaviors reach deep.

This directly informs training data design:
  - Invariant → train ascending arm on ANY context, generalizes
  - Dependent → need diverse behavioral contexts in training

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
# Same content words in different behavioral frames
# ══════════════════════════════════════════════════════════════════════

# Each group: (content_word, [(frame_label, sentence)])
# We extract at the content_word position across all frames.

INVARIANCE_PROBES = [
    # ── Nouns: should stay in entity basin regardless of frame ──
    ("numbers", [
        ("compute",    "Calculate the numbers in the equation."),
        ("summarize",  "Summarize the numbers in the report."),
        ("translate",  "Translate the numbers into percentages."),
        ("analyze",    "Analyze the numbers for any patterns."),
        ("verify",     "Verify the numbers are all correct."),
        ("sort",       "Sort the numbers from smallest to largest."),
        ("compare",    "Compare the numbers across both tables."),
        ("find",       "Find the numbers that exceed the threshold."),
    ]),
    ("results", [
        ("compute",    "Calculate the results of each experiment."),
        ("summarize",  "Summarize the results for the committee."),
        ("translate",  "Translate the results into plain language."),
        ("analyze",    "Analyze the results to find the cause."),
        ("verify",     "Verify the results match the prediction."),
        ("create",     "Create the results table for publication."),
        ("compare",    "Compare the results with the baseline."),
        ("plan",       "Plan the results presentation for Friday."),
    ]),
    ("data", [
        ("compute",    "Calculate the data average for each group."),
        ("summarize",  "Summarize the data into a brief overview."),
        ("transform",  "Sort the data by relevance and priority."),
        ("analyze",    "Analyze the data for statistical significance."),
        ("verify",     "Verify the data integrity before processing."),
        ("find",       "Find the data points that are outliers."),
        ("create",     "Generate the data visualization for review."),
        ("simplify",   "Simplify the data into three key metrics."),
    ]),

    # ── Verbs/predicates: should stay in predicate basin ──
    ("exceeds", [
        ("compute",    "Calculate whether the total exceeds the budget."),
        ("verify",     "Verify that performance exceeds the minimum."),
        ("analyze",    "Analyze why the cost exceeds the estimate."),
        ("summarize",  "Summarize how revenue exceeds projections."),
        ("find",       "Find every case where usage exceeds limits."),
        ("compare",    "Compare which metric exceeds the threshold."),
    ]),
    ("contains", [
        ("compute",    "Calculate how much the container contains."),
        ("verify",     "Verify the list contains all required items."),
        ("analyze",    "Analyze what the dataset contains exactly."),
        ("summarize",  "Summarize what this section contains."),
        ("find",       "Find which file contains the configuration."),
        ("translate",  "Translate what this package contains."),
    ]),

    # ── Adjectives: should stay in property basin ──
    ("largest", [
        ("compute",    "Calculate the largest value in the set."),
        ("find",       "Find the largest element in the array."),
        ("verify",     "Verify the largest number is correct."),
        ("summarize",  "Summarize the largest trends this quarter."),
        ("compare",    "Compare the largest values across groups."),
        ("sort",       "Sort by the largest contributing factor."),
    ]),
    ("incorrect", [
        ("compute",    "Calculate which entries are incorrect."),
        ("find",       "Find the incorrect values in this table."),
        ("verify",     "Verify nothing here is incorrect."),
        ("analyze",    "Analyze why these predictions are incorrect."),
        ("summarize",  "Summarize which assumptions were incorrect."),
        ("translate",  "Rephrase the incorrect statement properly."),
    ]),

    # ── Function words: determiners should be maximally invariant ──
    ("the", [
        ("compute",    "Calculate the total cost of operations."),
        ("summarize",  "Summarize the main findings of the study."),
        ("translate",  "Translate the original text into English."),
        ("analyze",    "Analyze the root cause of this failure."),
        ("verify",     "Verify the output matches the expected."),
        ("find",       "Find the source of this performance issue."),
        ("create",     "Create the documentation for this feature."),
        ("plan",       "Plan the deployment schedule for next week."),
    ]),
    ("each", [
        ("compute",    "Calculate the cost of each component."),
        ("summarize",  "Summarize each section of the report."),
        ("analyze",    "Analyze each variable for significance."),
        ("verify",     "Verify each step produces correct output."),
        ("compare",    "Compare each approach on all metrics."),
        ("find",       "Find each instance of this pattern."),
    ]),

    # ── Math operation words: most relevant for kernel dispatch ──
    ("sum", [
        ("compute",    "Calculate the sum of all the values."),
        ("summarize",  "Summarize the sum total of expenses."),
        ("verify",     "Verify the sum is calculated correctly."),
        ("analyze",    "Analyze the sum across all departments."),
        ("compare",    "Compare the sum against last quarter."),
        ("find",       "Find the sum of the remaining entries."),
    ]),
    ("difference", [
        ("compute",    "Calculate the difference between the groups."),
        ("summarize",  "Summarize the difference in their approaches."),
        ("analyze",    "Analyze the difference between predictions."),
        ("compare",    "Compare the difference across all trials."),
        ("find",       "Find the difference that caused this bug."),
        ("verify",     "Verify the difference is statistically valid."),
    ]),
    ("greater", [
        ("compute",    "Calculate which value is greater overall."),
        ("verify",     "Verify the result is greater than zero."),
        ("find",       "Find items with greater priority rating."),
        ("analyze",    "Analyze why the error is greater here."),
        ("compare",    "Compare which factor has greater impact."),
        ("summarize",  "Summarize the greater implications of this."),
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


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf",
        default="/Users/mwhitford/localai/models/Qwen3-32B-Q8_0.gguf")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", default="results/behavior-depth")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.gguf, device=args.device)
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    # ── Extract activations ──
    print("\n=== Extracting content word activations across behavioral frames ===")

    all_results = {}
    total = sum(len(frames) for _, frames in INVARIANCE_PROBES)
    done = 0

    for content_word, frames in INVARIANCE_PROBES:
        word_data = {"frames": {}}

        for frame_label, sentence in frames:
            layer_outputs, input_ids = run_with_hooks(
                model, tokenizer, sentence, device=args.device
            )
            target_indices = find_target_token_indices(
                tokenizer, input_ids, content_word
            )
            if not target_indices:
                # Try first occurrence for "the" which appears multiple times
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
                for i, tok in enumerate(tokens):
                    clean = tok.replace("Ġ", "").replace("▁", "").lower()
                    if content_word.lower() == clean:
                        target_indices = [i]
                        break
            if not target_indices:
                print(f"  WARNING: '{content_word}' not found in '{sentence}'")
                continue

            acts = np.zeros((n_layers, d_model), dtype=np.float32)
            for li in range(n_layers):
                h = layer_outputs[li]
                acts[li] = h[0, target_indices, :].mean(dim=0).numpy()

            word_data["frames"][frame_label] = {
                "acts": acts,
                "sentence": sentence,
                "target_indices": target_indices,
            }

            done += 1
            if done % 20 == 0 or done == total:
                print(f"  [{done}/{total}] '{content_word}' in {frame_label}")

        all_results[content_word] = word_data

    # ── Analyze: per-layer invariance ──
    print("\n=== Per-layer behavioral invariance ===")
    print("(higher = content word activation is SAME across all behavioral frames)")

    layer_invariance = {}  # layer → mean within-word cross-frame similarity

    for li in range(0, n_layers, 2):
        word_sims = []

        for content_word, word_data in all_results.items():
            frame_vecs = []
            for frame_label, fdata in word_data["frames"].items():
                frame_vecs.append(fdata["acts"][li])

            # Pairwise similarity across frames for this word
            for i in range(len(frame_vecs)):
                for j in range(i + 1, len(frame_vecs)):
                    word_sims.append(cosine_sim(frame_vecs[i], frame_vecs[j]))

        mean_inv = float(np.mean(word_sims)) if word_sims else 0.0
        min_inv = float(np.min(word_sims)) if word_sims else 0.0
        layer_invariance[li] = {"mean": mean_inv, "min": min_inv, "n": len(word_sims)}

        if li % 8 == 0 or li == n_layers - 2:
            print(f"  Layer {li:2d}: mean_invariance={mean_inv:.4f} "
                  f"min={min_inv:.4f} ({len(word_sims)} pairs)")

    # ── Per-word analysis at key layers ──
    key_layers = [0, 16, 24, 28, 32, 37, 48, 62]

    print("\n=== Per-word invariance at key layers ===")
    print("(cosine similarity of same word across all behavioral frames)")

    per_word_scores = {}
    for content_word, word_data in all_results.items():
        per_word_scores[content_word] = {}
        for li in key_layers:
            frame_vecs = []
            frame_labels = []
            for fl, fdata in word_data["frames"].items():
                frame_vecs.append(fdata["acts"][li])
                frame_labels.append(fl)

            sims = []
            for i in range(len(frame_vecs)):
                for j in range(i + 1, len(frame_vecs)):
                    sims.append(cosine_sim(frame_vecs[i], frame_vecs[j]))

            per_word_scores[content_word][li] = {
                "mean": float(np.mean(sims)) if sims else 0.0,
                "min": float(np.min(sims)) if sims else 0.0,
                "n_frames": len(frame_vecs),
            }

    # Print table
    print(f"\n{'Word':>15s}", end="")
    for li in key_layers:
        print(f"  L{li:2d}", end="")
    print()
    print("-" * (15 + 6 * len(key_layers)))

    for content_word in sorted(per_word_scores.keys()):
        print(f"{content_word:>15s}", end="")
        for li in key_layers:
            v = per_word_scores[content_word][li]["mean"]
            print(f" {v:.3f}", end="")
        print()

    # ── Find the most/least invariant words at L28 (typing zone) ──
    typing_layer = 28
    print(f"\n=== Invariance ranking at L{typing_layer} (typing zone) ===")
    ranked = sorted(per_word_scores.items(),
                   key=lambda x: -x[1][typing_layer]["mean"])
    for word, scores in ranked:
        s = scores[typing_layer]
        print(f"  {word:>15s}: mean={s['mean']:.4f} min={s['min']:.4f} "
              f"({s['n_frames']} frames)")

    # ── The key test: does the behavioral frame shift the basin? ──
    # For each word pair (same word, different frame), compute the
    # SHIFT magnitude relative to the word's self-similarity
    print(f"\n=== Frame-induced shift analysis at L{typing_layer} ===")
    print("If shift << self_sim → behaviors are surface (don't reach typing zone)")
    print("If shift ~ self_sim → behaviors are deep (reshape type basins)")

    for content_word, word_data in all_results.items():
        frame_vecs = []
        frame_labels = []
        for fl, fdata in word_data["frames"].items():
            frame_vecs.append(fdata["acts"][typing_layer])
            frame_labels.append(fl)

        if len(frame_vecs) < 2:
            continue

        # Mean vector (centroid of this word across all frames)
        centroid = np.mean(frame_vecs, axis=0)

        # Deviation of each frame from centroid
        deviations = [np.linalg.norm(v - centroid) for v in frame_vecs]
        mean_dev = float(np.mean(deviations))
        centroid_norm = float(np.linalg.norm(centroid))

        # Relative shift: how much does the frame move the word
        # relative to the word's overall magnitude?
        rel_shift = mean_dev / centroid_norm if centroid_norm > 0 else 0

        # Cross-frame similarity
        sims = []
        for i in range(len(frame_vecs)):
            for j in range(i + 1, len(frame_vecs)):
                sims.append(cosine_sim(frame_vecs[i], frame_vecs[j]))

        print(f"  {content_word:>15s}: cross_frame_sim={np.mean(sims):.4f} "
              f"rel_shift={rel_shift:.4f} "
              f"({'SURFACE' if rel_shift < 0.05 else 'DEEP' if rel_shift > 0.15 else 'MIXED'})")

    # ── Save everything ──
    save_data = {
        "layer_invariance": layer_invariance,
        "per_word_scores": {
            w: {str(li): s for li, s in scores.items()}
            for w, scores in per_word_scores.items()
        },
    }
    with open(out_dir / "invariance_scores.json", "w") as f:
        json.dump(save_data, f, indent=2)

    # Save activations for further analysis
    npz_dict = {}
    metadata = {}
    for content_word, word_data in all_results.items():
        metadata[content_word] = {}
        for frame_label, fdata in word_data["frames"].items():
            key = f"{content_word}__{frame_label}"
            npz_dict[key] = fdata["acts"]
            metadata[content_word][frame_label] = {
                "sentence": fdata["sentence"],
                "target_indices": fdata["target_indices"],
            }
    np.savez_compressed(out_dir / "invariance_activations.npz", **npz_dict)
    with open(out_dir / "invariance_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved {len(npz_dict)} activation vectors to {out_dir}/")


if __name__ == "__main__":
    main()
