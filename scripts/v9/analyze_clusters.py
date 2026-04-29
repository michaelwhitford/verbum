"""
Analyze type basin structure in Qwen3-32B activations.

Reads activations.npz + metadata.json from probe_clusters.py.
At the peak typing layers: UMAP projection + HDBSCAN clustering.
Produces plots and cluster assignments.

License: MIT
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

import umap
import hdbscan


def load_data(data_dir: str = "results/cluster-probe"):
    """Load activations and metadata."""
    data_dir = Path(data_dir)

    acts_file = np.load(data_dir / "activations.npz")
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(data_dir / "layer_scores.json") as f:
        layer_scores = json.load(f)

    return acts_file, metadata, layer_scores


def plot_layer_curve(layer_scores: dict, out_dir: Path):
    """Plot within/between similarity and ratio across layers."""
    layers = sorted(int(k) for k in layer_scores.keys())
    within = [layer_scores[str(l)]["within_mean"] for l in layers]
    between = [layer_scores[str(l)]["between_mean"] for l in layers]
    ratio = [layer_scores[str(l)]["ratio"] for l in layers]
    sep = [layer_scores[str(l)]["separation"] for l in layers]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: within vs between similarity
    ax = axes[0]
    ax.plot(layers, within, "b-", linewidth=2, label="Within-group (same type)")
    ax.plot(layers, between, "r-", linewidth=2, label="Between-group (diff type)")
    ax.fill_between(layers, between, within, alpha=0.15, color="green")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Qwen3-32B: Type Clustering by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom: separation (within - between) 
    ax = axes[1]
    ax.plot(layers, sep, "g-", linewidth=2, label="Separation (within − between)")
    ax.axhline(y=0, color="k", linewidth=0.5)
    
    # Mark peak
    peak_layer = layers[np.argmax(sep)]
    peak_val = max(sep)
    ax.annotate(
        f"Peak: L{peak_layer}\nsep={peak_val:.3f}",
        xy=(peak_layer, peak_val),
        xytext=(peak_layer + 5, peak_val - 0.05),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Separation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark the typing zone
    for ax in axes:
        ax.axvspan(26, 37, alpha=0.08, color="blue", label="_typing zone")

    plt.tight_layout()
    plt.savefig(out_dir / "layer_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved layer_curve.png")


def plot_umap_at_layer(
    acts_file, metadata: dict, layer_idx: int, out_dir: Path, suffix: str = ""
):
    """UMAP projection of all probe tokens at a specific layer."""
    # Collect vectors and labels
    vectors = []
    labels = []
    words = []
    type_labels = []
    
    for group_name, group_meta in metadata.items():
        type_label = group_meta["type_label"]
        for key, item_meta in group_meta["items"].items():
            npz_key = f"{group_name}__{key}"
            if npz_key in acts_file:
                vec = acts_file[npz_key][layer_idx]  # (d_model,)
                vectors.append(vec)
                labels.append(group_name)
                words.append(item_meta["word"])
                type_labels.append(type_label)

    X = np.array(vectors)
    print(f"\nUMAP at layer {layer_idx}: {X.shape[0]} vectors, d={X.shape[1]}")

    # Fit UMAP
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=10,
        min_dist=0.1,
        random_state=42,
    )
    embedding = reducer.fit_transform(X)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=2,
        metric="euclidean",
    )
    cluster_ids = clusterer.fit_predict(embedding)
    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    n_noise = list(cluster_ids).count(-1)
    print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

    # ── Plot 1: colored by probe group ──
    fig, ax = plt.subplots(figsize=(16, 12))
    
    unique_labels = sorted(set(labels))
    # Use a colormap with enough distinct colors
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    colors = {label: cmap(i) for i, label in enumerate(unique_labels)}

    for label in unique_labels:
        mask = [l == label for l in labels]
        idx = [i for i, m in enumerate(mask) if m]
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            c=[colors[label]],
            label=label,
            s=80,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
        )

    # Annotate each point with the word
    for i, (word, label) in enumerate(zip(words, labels)):
        ax.annotate(
            word,
            (embedding[i, 0], embedding[i, 1]),
            fontsize=7,
            alpha=0.7,
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    ax.set_title(
        f"Qwen3-32B Layer {layer_idx}: Semantic Type Basins\n"
        f"(UMAP cosine, {n_clusters} HDBSCAN clusters, {n_noise} noise)",
        fontsize=14,
    )
    ax.legend(
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=8,
        title="Probe Group",
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    
    plt.tight_layout()
    plt.savefig(
        out_dir / f"umap_layer{layer_idx}_groups{suffix}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved umap_layer{layer_idx}_groups{suffix}.png")

    # ── Plot 2: colored by HDBSCAN cluster ──
    fig, ax = plt.subplots(figsize=(16, 12))
    
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=cluster_ids,
        cmap="Spectral",
        s=80,
        alpha=0.8,
        edgecolors="white",
        linewidth=0.5,
    )

    for i, (word, label) in enumerate(zip(words, labels)):
        ax.annotate(
            word,
            (embedding[i, 0], embedding[i, 1]),
            fontsize=7,
            alpha=0.7,
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
        )

    ax.set_title(
        f"Qwen3-32B Layer {layer_idx}: HDBSCAN Clusters\n"
        f"({n_clusters} clusters, {n_noise} noise points)",
        fontsize=14,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster ID")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    
    plt.tight_layout()
    plt.savefig(
        out_dir / f"umap_layer{layer_idx}_hdbscan{suffix}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved umap_layer{layer_idx}_hdbscan{suffix}.png")

    # ── Plot 3: cosine similarity heatmap ──
    sim_matrix = cosine_similarity(X)
    
    # Sort by group for block structure
    sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
    sorted_sim = sim_matrix[np.ix_(sorted_indices, sorted_indices)]
    sorted_words = [f"{words[i]} ({labels[i][:12]})" for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(sorted_sim, cmap="RdBu_r", vmin=-0.5, vmax=1.0)
    
    ax.set_xticks(range(len(sorted_words)))
    ax.set_yticks(range(len(sorted_words)))
    ax.set_xticklabels(sorted_words, rotation=90, fontsize=6)
    ax.set_yticklabels(sorted_words, fontsize=6)
    
    plt.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)
    ax.set_title(f"Qwen3-32B Layer {layer_idx}: Pairwise Cosine Similarity", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(
        out_dir / f"similarity_layer{layer_idx}{suffix}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved similarity_layer{layer_idx}{suffix}.png")

    # Save cluster assignments
    cluster_data = {
        "layer": layer_idx,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "points": [
            {
                "word": words[i],
                "group": labels[i],
                "type_label": type_labels[i],
                "cluster_id": int(cluster_ids[i]),
                "umap_x": float(embedding[i, 0]),
                "umap_y": float(embedding[i, 1]),
            }
            for i in range(len(words))
        ],
    }
    
    # Cluster contents summary
    cluster_summary = {}
    for cid in sorted(set(cluster_ids)):
        members = [
            {"word": words[i], "group": labels[i]}
            for i in range(len(words))
            if cluster_ids[i] == cid
        ]
        cluster_summary[int(cid)] = {
            "size": len(members),
            "members": members,
            "dominant_group": max(
                set(m["group"] for m in members),
                key=lambda g: sum(1 for m in members if m["group"] == g),
            ),
        }
    cluster_data["cluster_summary"] = cluster_summary

    with open(out_dir / f"clusters_layer{layer_idx}{suffix}.json", "w") as f:
        json.dump(cluster_data, f, indent=2)
    print(f"Saved clusters_layer{layer_idx}{suffix}.json")

    return cluster_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="results/cluster-probe")
    parser.add_argument(
        "--layers", type=str, default="auto",
        help="Comma-sep layers to UMAP, or 'auto' for peak + neighbors"
    )
    args = parser.parse_args()

    out_dir = Path(args.data_dir)
    acts_file, metadata, layer_scores = load_data(args.data_dir)

    # Layer curve plot
    plot_layer_curve(layer_scores, out_dir)

    # Determine which layers to UMAP
    if args.layers == "auto":
        # Peak + early + late for comparison
        sorted_layers = sorted(
            layer_scores.items(),
            key=lambda x: x[1]["separation"],
            reverse=True,
        )
        peak = int(sorted_layers[0][0])
        target_layers = sorted(set([0, peak, 63]))  # early, peak, final
    else:
        target_layers = [int(x) for x in args.layers.split(",")]

    for layer_idx in target_layers:
        print(f"\n{'═'*60}")
        print(f"  Analyzing layer {layer_idx}")
        print(f"{'═'*60}")
        cluster_data = plot_umap_at_layer(acts_file, metadata, layer_idx, out_dir)

        # Print cluster contents
        print(f"\nCluster contents at layer {layer_idx}:")
        for cid, info in sorted(cluster_data["cluster_summary"].items()):
            label = "NOISE" if cid == -1 else f"Cluster {cid}"
            words = [m["word"] for m in info["members"]]
            groups = set(m["group"] for m in info["members"])
            print(f"  {label} ({info['size']} pts): {', '.join(words[:10])}")
            print(f"    groups: {', '.join(sorted(groups))}")


if __name__ == "__main__":
    main()
