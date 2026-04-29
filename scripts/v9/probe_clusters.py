"""
Probe: Where do semantic types cluster in Qwen3-32B's activation space?

Hypothesis: The model's hidden states organize into basins of attraction
that correspond to functional types. Synonyms (brief/short/concise) land
in the same basin. The basin, not a symbolic label, IS the type.

Strategy:
  1. Load Qwen3-32B from GGUF via transformers (Q8 → dequantized to fp16)
  2. Define probe groups: sets of words/phrases with known semantic equivalence
  3. Embed each in minimal context sentences
  4. Hook every layer's residual stream output
  5. For each layer: measure within-group vs between-group cosine similarity
  6. The layer(s) where within/between ratio peaks = the "typing layers"

Output: results/cluster-probe/activations.npz + layer_scores.json

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
# Probe groups: words/phrases that should cluster together by type
# ══════════════════════════════════════════════════════════════════════

# Each group: (group_name, type_label, [(word, context_sentence)])
# The context_sentence places the target word in a natural position.
# We mark the target token(s) we want activations for.

PROBE_GROUPS = [
    # ── Semantic equivalence: synonyms should cluster ──
    ("shorten_verbs", "action:reduce_length", [
        ("brief", "Please brief the team on the situation."),
        ("shorten", "Please shorten the report before submitting."),
        ("abbreviate", "Please abbreviate the long description."),
        ("condense", "Please condense the document into key points."),
        ("summarize", "Please summarize the findings for the board."),
        ("truncate", "Please truncate the output to fit the screen."),
    ]),
    ("size_adjectives", "property:large", [
        ("big", "The big problem is resource allocation."),
        ("large", "The large problem is resource allocation."),
        ("huge", "The huge problem is resource allocation."),
        ("enormous", "The enormous problem is resource allocation."),
        ("massive", "The massive problem is resource allocation."),
        ("vast", "The vast problem is resource allocation."),
    ]),
    ("speed_verbs", "action:move_fast", [
        ("run", "The dog will run across the field."),
        ("sprint", "The dog will sprint across the field."),
        ("dash", "The dog will dash across the field."),
        ("rush", "The dog will rush across the field."),
        ("race", "The dog will race across the field."),
        ("bolt", "The dog will bolt across the field."),
    ]),
    ("think_verbs", "action:cognition", [
        ("think", "I think about the implications carefully."),
        ("consider", "I consider the implications carefully."),
        ("ponder", "I ponder the implications carefully."),
        ("contemplate", "I contemplate the implications carefully."),
        ("reflect", "I reflect on the implications carefully."),
        ("deliberate", "I deliberate on the implications carefully."),
    ]),

    # ── Syntactic type: same Montague type, different semantics ──
    ("intransitive_verbs", "type:e→t", [
        ("sleeps", "The cat sleeps on the mat."),
        ("runs", "The cat runs on the mat."),
        ("sits", "The cat sits on the mat."),
        ("breathes", "The cat breathes on the mat."),
        ("waits", "The cat waits on the mat."),
        ("rests", "The cat rests on the mat."),
    ]),
    ("transitive_verbs", "type:e→(e→t)", [
        ("chased", "The dog chased the rabbit through the forest."),
        ("ate", "The dog ate the rabbit through the forest."),
        ("found", "The dog found the rabbit through the forest."),
        ("watched", "The dog watched the rabbit through the forest."),
        ("followed", "The dog followed the rabbit through the forest."),
        ("caught", "The dog caught the rabbit through the forest."),
    ]),
    ("common_nouns", "type:e→t_noun", [
        ("cat", "The cat is sitting quietly."),
        ("dog", "The dog is sitting quietly."),
        ("bird", "The bird is sitting quietly."),
        ("horse", "The horse is sitting quietly."),
        ("fish", "The fish is sitting quietly."),
        ("frog", "The frog is sitting quietly."),
    ]),
    ("determiners", "type:(e→t)→e", [
        ("the", "The cat sat on the mat."),
        ("a", "A cat sat on the mat."),
        ("every", "Every cat sat on the mat."),
        ("some", "Some cat sat on the mat."),
        ("no", "No cat sat on the mat."),
        ("each", "Each cat sat on the mat."),
    ]),
    ("prepositions", "type:e→(e→t)→(e→t)", [
        ("on", "The cat sat on the big mat."),
        ("under", "The cat sat under the big mat."),
        ("near", "The cat sat near the big mat."),
        ("beside", "The cat sat beside the big mat."),
        ("behind", "The cat sat behind the big mat."),
        ("above", "The cat sat above the big mat."),
    ]),

    # ── Polysemy: same word, different type (should NOT cluster) ──
    ("run_verb", "type:e→t_verb_usage", [
        ("run", "The children run in the park every morning."),
        ("run", "The athletes run the marathon together."),
        ("run", "The horses run around the paddock."),
    ]),
    ("run_noun", "type:e_noun_usage", [
        ("run", "That was an excellent run this morning."),
        ("run", "She completed her daily run before breakfast."),
        ("run", "The morning run was particularly refreshing."),
    ]),

    # ── Computation: kernel operation words ──
    ("addition_words", "kernel:add", [
        ("add", "Please add three and four together."),
        ("plus", "Three plus four equals seven."),
        ("sum", "The sum of three and four is seven."),
        ("combine", "Combine three and four to get seven."),
        ("total", "The total of three and four is seven."),
    ]),
    ("comparison_words", "kernel:compare", [
        ("greater", "Three is greater than two."),
        ("larger", "Three is larger than two."),
        ("exceeds", "Three exceeds two by one."),
        ("bigger", "Three is bigger than two."),
        ("more", "Three is more than two."),
    ]),

    # ── Entities: proper nouns (all type e) ──
    ("person_names", "type:e_person", [
        ("Alice", "Alice walked through the garden quietly."),
        ("Bob", "Bob walked through the garden quietly."),
        ("Charlie", "Charlie walked through the garden quietly."),
        ("Diana", "Diana walked through the garden quietly."),
        ("Eve", "Eve walked through the garden quietly."),
    ]),

    # ── Quantifiers: ((e→t)→t) ──
    ("quantifiers", "type:(e→t)→t", [
        ("every", "Every student passed the exam."),
        ("all", "All students passed the exam."),
        ("some", "Some students passed the exam."),
        ("most", "Most students passed the exam."),
        ("few", "Few students passed the exam."),
        ("many", "Many students passed the exam."),
    ]),
]


def find_target_token_indices(
    tokenizer, input_ids: torch.Tensor, target_word: str
) -> list[int]:
    """Find which token positions correspond to the target word.

    Returns indices of ALL tokens that compose the target word.
    Uses the tokenizer to find exact subword matches.
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # Tokenize the target word alone to find its subword pieces
    target_ids = tokenizer.encode(target_word, add_special_tokens=False)
    target_tokens = tokenizer.convert_ids_to_tokens(target_ids)

    # Also try with a space prefix (common in BPE)
    space_target_ids = tokenizer.encode(" " + target_word, add_special_tokens=False)
    space_target_tokens = tokenizer.convert_ids_to_tokens(space_target_ids)

    # Search for the target token sequence in the full token list
    for pattern in [space_target_tokens, target_tokens]:
        pat_len = len(pattern)
        for i in range(len(tokens) - pat_len + 1):
            if tokens[i : i + pat_len] == pattern:
                return list(range(i, i + pat_len))

    # Fallback: find any token containing the target word
    indices = []
    for i, tok in enumerate(tokens):
        # Strip BPE prefix markers
        clean = tok.replace("Ġ", "").replace("▁", "").replace("##", "").lower()
        if target_word.lower() in clean or clean in target_word.lower():
            indices.append(i)

    return indices


def load_model(gguf_path: str, device: str = "mps"):
    """Load Qwen3-32B from GGUF with transformers."""
    gguf_dir = str(Path(gguf_path).parent)
    gguf_file = Path(gguf_path).name

    print(f"Loading model from {gguf_path}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

    model = AutoModelForCausalLM.from_pretrained(
        gguf_dir,
        gguf_file=gguf_file,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    t1 = time.time()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"Loaded in {t1-t0:.1f}s: {n_layers} layers, d={d_model}, device={device}")

    return model, tokenizer


def extract_activations(
    model, tokenizer, probe_groups: list, device: str = "mps"
) -> dict:
    """Run all probe sentences through the model, collecting per-layer activations.

    Returns dict with:
      - activations: {group_name: {word: np.array(n_layers, d_model)}}
      - metadata: {group_name: {word: {sentence, token_indices, tokens}}}
    """
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    # Storage for hooked activations
    layer_outputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple; first element is the hidden state
            hidden = output[0] if isinstance(output, tuple) else output
            layer_outputs[layer_idx] = hidden.detach().cpu()
        return hook_fn

    # Register hooks on each transformer layer
    hooks = []
    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    activations = {}
    metadata = {}

    total_probes = sum(len(items) for _, _, items in probe_groups)
    done = 0

    with torch.no_grad():
        for group_name, type_label, items in probe_groups:
            activations[group_name] = {}
            metadata[group_name] = {"type_label": type_label, "items": {}}

            for target_word, sentence in items:
                # Tokenize
                inputs = tokenizer(sentence, return_tensors="pt").to(device)
                input_ids = inputs["input_ids"]

                # Find target token positions
                target_indices = find_target_token_indices(
                    tokenizer, input_ids, target_word
                )
                if not target_indices:
                    print(f"  WARNING: '{target_word}' not found in '{sentence}'")
                    continue

                tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

                # Forward pass (activations captured by hooks)
                layer_outputs.clear()
                _ = model(**inputs)

                # Extract activations at target positions, mean-pool if multi-token
                word_acts = np.zeros((n_layers, d_model), dtype=np.float32)
                for layer_idx in range(n_layers):
                    h = layer_outputs[layer_idx]  # (1, seq_len, d_model)
                    target_vecs = h[0, target_indices, :]  # (n_tokens, d_model)
                    word_acts[layer_idx] = target_vecs.mean(dim=0).numpy()

                # Key: word + sentence hash to handle duplicates (polysemy probes)
                key = f"{target_word}_{hash(sentence) % 10000:04d}"
                activations[group_name][key] = word_acts
                metadata[group_name]["items"][key] = {
                    "word": target_word,
                    "sentence": sentence,
                    "token_indices": target_indices,
                    "tokens": [tokens[i] for i in target_indices],
                }

                done += 1
                if done % 10 == 0 or done == total_probes:
                    print(f"  [{done}/{total_probes}] {group_name}: '{target_word}'")

    # Remove hooks
    for h in hooks:
        h.remove()

    return activations, metadata


def compute_layer_scores(activations: dict, probe_groups: list) -> dict:
    """For each layer, compute within-group vs between-group cosine similarity.

    Returns {layer_idx: {within_mean, between_mean, ratio, per_group: {...}}}
    """
    # Get number of layers from first available activation
    first_group = next(iter(activations.values()))
    first_acts = next(iter(first_group.values()))
    n_layers = first_acts.shape[0]

    def cosine_sim(a, b):
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    layer_scores = {}

    for layer_idx in range(n_layers):
        within_sims = []
        between_sims = []
        per_group = {}

        # Collect all activation vectors at this layer
        all_vecs = []
        all_group_labels = []

        for group_name, type_label, _ in probe_groups:
            if group_name not in activations:
                continue
            group_vecs = []
            for key, acts in activations[group_name].items():
                vec = acts[layer_idx]
                group_vecs.append(vec)
                all_vecs.append(vec)
                all_group_labels.append(group_name)

            # Within-group pairwise similarity
            group_within = []
            for i in range(len(group_vecs)):
                for j in range(i + 1, len(group_vecs)):
                    sim = cosine_sim(group_vecs[i], group_vecs[j])
                    group_within.append(sim)
                    within_sims.append(sim)

            per_group[group_name] = {
                "within_mean": float(np.mean(group_within)) if group_within else 0.0,
                "n_pairs": len(group_within),
            }

        # Between-group pairwise similarity (sample to keep tractable)
        group_names = list(set(all_group_labels))
        for gi in range(len(group_names)):
            for gj in range(gi + 1, len(group_names)):
                vecs_i = [
                    v
                    for v, g in zip(all_vecs, all_group_labels)
                    if g == group_names[gi]
                ]
                vecs_j = [
                    v
                    for v, g in zip(all_vecs, all_group_labels)
                    if g == group_names[gj]
                ]
                for vi in vecs_i:
                    for vj in vecs_j:
                        between_sims.append(cosine_sim(vi, vj))

        within_mean = float(np.mean(within_sims)) if within_sims else 0.0
        between_mean = float(np.mean(between_sims)) if between_sims else 0.0
        ratio = within_mean / between_mean if between_mean > 0 else 0.0

        layer_scores[layer_idx] = {
            "within_mean": within_mean,
            "between_mean": between_mean,
            "ratio": ratio,
            "separation": within_mean - between_mean,
            "per_group": per_group,
        }

        if layer_idx % 8 == 0 or layer_idx == n_layers - 1:
            print(
                f"  Layer {layer_idx:2d}: within={within_mean:.4f} "
                f"between={between_mean:.4f} ratio={ratio:.3f} "
                f"sep={within_mean - between_mean:+.4f}"
            )

    return layer_scores


def main():
    parser = argparse.ArgumentParser(
        description="Probe semantic type clusters in Qwen3-32B"
    )
    parser.add_argument(
        "--gguf",
        default="/Users/mwhitford/localai/models/Qwen3-32B-Q8_0.gguf",
        help="Path to Qwen3-32B GGUF file",
    )
    parser.add_argument(
        "--device", default="mps", help="Device (mps, cuda, cpu)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/cluster-probe",
        help="Output directory",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(args.gguf, device=args.device)

    # Extract activations
    print("\n═══ Extracting activations ═══")
    activations, metadata = extract_activations(
        model, tokenizer, PROBE_GROUPS, device=args.device
    )

    # Save activations as npz (one array per group+word)
    npz_dict = {}
    for group_name, words in activations.items():
        for key, acts in words.items():
            npz_key = f"{group_name}__{key}"
            npz_dict[npz_key] = acts
    np.savez_compressed(out_dir / "activations.npz", **npz_dict)
    print(f"\nSaved activations: {len(npz_dict)} vectors to {out_dir}/activations.npz")

    # Save metadata
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Compute layer scores
    print("\n═══ Computing layer-wise type clustering ═══")
    layer_scores = compute_layer_scores(activations, PROBE_GROUPS)

    # Save scores
    with open(out_dir / "layer_scores.json", "w") as f:
        json.dump(layer_scores, f, indent=2)

    # Summary: find peak layers
    print("\n═══ Summary ═══")
    sorted_layers = sorted(
        layer_scores.items(),
        key=lambda x: x[1]["separation"],
        reverse=True,
    )
    print("\nTop 10 layers by within-between separation:")
    for layer_idx, scores in sorted_layers[:10]:
        print(
            f"  Layer {layer_idx:2d}: ratio={scores['ratio']:.3f} "
            f"sep={scores['separation']:+.4f} "
            f"(within={scores['within_mean']:.4f} "
            f"between={scores['between_mean']:.4f})"
        )

    # Per-group analysis at best layer
    best_layer = sorted_layers[0][0]
    print(f"\nPer-group within-similarity at best layer ({best_layer}):")
    best_scores = layer_scores[best_layer]["per_group"]
    for group_name, gs in sorted(
        best_scores.items(), key=lambda x: -x[1]["within_mean"]
    ):
        print(f"  {group_name:25s}: {gs['within_mean']:.4f} ({gs['n_pairs']} pairs)")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
