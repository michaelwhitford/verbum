"""
Probe: Inter-op basin topology in Qwen3-32B.

Uses activations already extracted by probe_kernel_basins.py.
Combines operator words AND expressions into one analysis to map
the full topology of kernel operations in activation space.

Questions:
  1. Do the 22 kernel ops form distinct basins or merge into super-basins?
  2. What hierarchy emerges? (add near sub? mul near div?)
  3. Where do prose expressions land relative to formal operator words?
  4. What's the natural dispatch granularity?

Also: computes a confusion matrix — which ops are most easily confused?
This directly informs kernel dispatch design.

License: MIT
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

import umap
import hdbscan


def load_activations(data_dir: str):
    """Load both operator and expression activations."""
    data_dir = Path(data_dir)
    
    op_acts = np.load(data_dir / "operator_activations.npz")
    with open(data_dir / "operator_metadata.json") as f:
        op_meta = json.load(f)
    
    expr_acts = np.load(data_dir / "expression_activations.npz")
    with open(data_dir / "expression_metadata.json") as f:
        expr_meta = json.load(f)
    
    return op_acts, op_meta, expr_acts, expr_meta


def build_op_centroids(op_acts, op_meta, layer_idx):
    """Compute centroid vector for each kernel op at a given layer."""
    centroids = {}
    for group_name, group_meta in op_meta.items():
        kernel_op = group_meta["kernel_op"]
        vecs = []
        for key in group_meta["items"]:
            npz_key = f"op__{group_name}__{key}"
            if npz_key in op_acts:
                vecs.append(op_acts[npz_key][layer_idx])
        if vecs:
            centroids[group_name] = {
                "centroid": np.mean(vecs, axis=0),
                "vectors": vecs,
                "kernel_op": kernel_op,
                "n": len(vecs),
            }
    return centroids


def plot_op_similarity_matrix(centroids, layer_idx, out_dir):
    """Cosine similarity between all op centroids — the dispatch confusion map."""
    names = sorted(centroids.keys())
    n = len(names)
    sim = np.zeros((n, n))
    
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            ci = centroids[ni]["centroid"]
            cj = centroids[nj]["centroid"]
            na, nb = np.linalg.norm(ci), np.linalg.norm(cj)
            sim[i, j] = float(np.dot(ci, cj) / (na * nb)) if na > 0 and nb > 0 else 0
    
    # Hierarchical clustering to order the matrix
    dist = pdist(sim, metric="correlation")
    Z = linkage(dist, method="ward")
    
    # Dendrogram for ordering
    fig, (ax_dendro, ax_heat) = plt.subplots(
        1, 2, figsize=(20, 10),
        gridspec_kw={"width_ratios": [1, 3]}
    )
    
    # Dendrogram
    dendro = dendrogram(Z, labels=names, orientation="left", ax=ax_dendro,
                        leaf_font_size=9, color_threshold=0.7 * max(Z[:, 2]))
    ax_dendro.set_title("Hierarchical Clustering", fontsize=12)
    
    # Reorder similarity matrix by dendrogram
    order = dendro["leaves"]
    ordered_names = [names[i] for i in order]
    ordered_sim = sim[np.ix_(order, order)]
    
    # Heatmap
    im = ax_heat.imshow(ordered_sim, cmap="RdBu_r", vmin=-0.2, vmax=1.0, aspect="auto")
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(ordered_names, rotation=45, ha="right", fontsize=9)
    ax_heat.set_yticklabels(ordered_names, fontsize=9)
    
    # Annotate with values
    for i in range(n):
        for j in range(n):
            val = ordered_sim[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)
    
    plt.colorbar(im, ax=ax_heat, label="Cosine Similarity", shrink=0.8)
    ax_heat.set_title(f"Kernel Op Centroid Similarity — Layer {layer_idx}", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"op_similarity_L{layer_idx}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved op_similarity_L{layer_idx}.png")
    
    # Return the hierarchy info
    # Cut the dendrogram at different levels to see natural groupings
    results = {"layer": layer_idx, "op_order": ordered_names}
    for n_clusters in [3, 5, 7, 10]:
        clusters = fcluster(Z, n_clusters, criterion="maxclust")
        grouping = {}
        for i, c in enumerate(clusters):
            c = int(c)
            if c not in grouping:
                grouping[c] = []
            grouping[c].append(names[i])
        results[f"cut_{n_clusters}"] = grouping
    
    return results, sim, names


def plot_combined_umap(op_acts, op_meta, expr_acts, expr_meta, layer_idx, out_dir):
    """UMAP of all operator words + expression results at one layer."""
    vectors = []
    labels = []
    markers = []  # "op" or "expr"
    texts = []
    kernel_ops = []
    
    # Operator words
    for group_name, group_meta in op_meta.items():
        kernel_op = group_meta["kernel_op"]
        for key, item in group_meta["items"].items():
            npz_key = f"op__{group_name}__{key}"
            if npz_key in op_acts:
                vectors.append(op_acts[npz_key][layer_idx])
                labels.append(group_name)
                markers.append("op")
                texts.append(item["word"])
                kernel_ops.append(kernel_op)
    
    # Expressions
    for group_name, group_meta in expr_meta.items():
        result_label = group_meta["result_label"]
        for key, item in group_meta["items"].items():
            npz_key = f"expr__{group_name}__{key}"
            if npz_key in expr_acts:
                vectors.append(expr_acts[npz_key][layer_idx])
                labels.append(f"EXPR:{group_name}")
                markers.append("expr")
                texts.append(item.get("expression", item.get("word", key)))
                kernel_ops.append(result_label)
    
    X = np.array(vectors)
    print(f"\nCombined UMAP at L{layer_idx}: {X.shape[0]} vectors ({sum(1 for m in markers if m=='op')} ops, {sum(1 for m in markers if m=='expr')} exprs)")
    
    # UMAP
    reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=12,
                        min_dist=0.15, random_state=42)
    embedding = reducer.fit_transform(X)
    
    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
    cluster_ids = clusterer.fit_predict(embedding)
    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    print(f"HDBSCAN: {n_clusters} clusters")
    
    # ── Plot: colored by kernel op category ──
    fig, ax = plt.subplots(figsize=(20, 14))
    
    # Group kernel ops into categories for coloring
    op_categories = {
        "arith": ["kernel:add", "kernel:sub", "kernel:mul", "kernel:div", "kernel:mod",
                  "kernel:min", "kernel:max"],
        "compare": ["kernel:eq", "kernel:lt", "kernel:gt"],
        "bool": ["kernel:and", "kernel:or", "kernel:not"],
        "unary": ["kernel:abs", "kernel:neg"],
        "control": ["kernel:if"],
        "higher": ["kernel:partial", "kernel:compose", "kernel:apply"],
        "expr_add": ["result:7", "result:10", "op:add"],
        "expr_sub": ["result:4"],
        "expr_mul": ["result:20", "op:mul"],
        "expr_div": ["result:5"],
        "expr_cmp": ["result:true"],
        "expr_nest": ["result:23", "result:14"],
        "expr_cond": ["result:10"],
    }
    
    # Invert: kernel_op → category
    op_to_cat = {}
    for cat, ops in op_categories.items():
        for op in ops:
            op_to_cat[op] = cat
    
    categories = [op_to_cat.get(ko, "other") for ko in kernel_ops]
    unique_cats = sorted(set(categories))
    
    cat_colors = {
        "arith": "#e41a1c", "compare": "#377eb8", "bool": "#4daf4a",
        "unary": "#984ea3", "control": "#ff7f00", "higher": "#a65628",
        "expr_add": "#e41a1c", "expr_sub": "#f781bf", "expr_mul": "#999999",
        "expr_div": "#377eb8", "expr_cmp": "#4daf4a", "expr_nest": "#ff7f00",
        "expr_cond": "#a65628", "other": "#666666",
    }
    
    for cat in unique_cats:
        mask = [c == cat for c in categories]
        idx = [i for i, m in enumerate(mask) if m]
        if not idx:
            continue
        
        is_expr = [markers[i] == "expr" for i in idx]
        
        # Operator words: circles
        op_idx = [i for i, e in zip(idx, is_expr) if not e]
        if op_idx:
            ax.scatter(embedding[op_idx, 0], embedding[op_idx, 1],
                      c=cat_colors.get(cat, "#666666"), label=f"{cat} (word)",
                      s=60, alpha=0.8, edgecolors="white", linewidth=0.5,
                      marker="o")
        
        # Expressions: stars
        expr_idx = [i for i, e in zip(idx, is_expr) if e]
        if expr_idx:
            ax.scatter(embedding[expr_idx, 0], embedding[expr_idx, 1],
                      c=cat_colors.get(cat, "#666666"), label=f"{cat} (expr)",
                      s=120, alpha=0.9, edgecolors="black", linewidth=0.8,
                      marker="*")
    
    # Annotate
    for i in range(len(texts)):
        text = texts[i]
        if len(text) > 20:
            text = text[:18] + ".."
        ax.annotate(text, (embedding[i, 0], embedding[i, 1]),
                   fontsize=5, alpha=0.6, ha="center", va="bottom",
                   xytext=(0, 3), textcoords="offset points")
    
    ax.set_title(f"Qwen3-32B Layer {layer_idx}: Kernel Op Topology\n"
                f"(circles=words, stars=expressions, {n_clusters} HDBSCAN clusters)",
                fontsize=14)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7,
             title="Category", ncol=1)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"op_topology_L{layer_idx}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved op_topology_L{layer_idx}.png")
    
    # Save cluster data
    cluster_data = {
        "layer": layer_idx, "n_clusters": n_clusters,
        "points": [
            {"text": texts[i], "label": labels[i], "type": markers[i],
             "kernel_op": kernel_ops[i], "cluster": int(cluster_ids[i]),
             "umap_x": float(embedding[i, 0]), "umap_y": float(embedding[i, 1])}
            for i in range(len(texts))
        ]
    }
    
    # Cluster contents
    for cid in sorted(set(cluster_ids)):
        members = [p for p in cluster_data["points"] if p["cluster"] == cid]
        ops_in = set(m["label"] for m in members)
        types_in = set(m["type"] for m in members)
        print(f"  {'NOISE' if cid == -1 else f'Cluster {cid}'} ({len(members)} pts): "
              f"{', '.join(sorted(ops_in)[:5])}")
    
    with open(out_dir / f"op_topology_L{layer_idx}.json", "w") as f:
        json.dump(cluster_data, f, indent=2)
    
    return cluster_data


def analyze_dispatch_granularity(centroids, layer_idx, out_dir):
    """For each pair of ops, compute similarity — the confusion risk.
    
    If two ops have centroids closer than a threshold, they can't be
    reliably dispatched independently → they need to merge or use
    a different feature for disambiguation.
    """
    names = sorted(centroids.keys())
    n = len(names)
    
    # Compute all pairwise centroid similarities
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            ci = centroids[names[i]]["centroid"]
            cj = centroids[names[j]]["centroid"]
            na, nb = np.linalg.norm(ci), np.linalg.norm(cj)
            sim = float(np.dot(ci, cj) / (na * nb)) if na > 0 and nb > 0 else 0
            pairs.append({
                "op_a": names[i], "op_b": names[j],
                "kernel_a": centroids[names[i]]["kernel_op"],
                "kernel_b": centroids[names[j]]["kernel_op"],
                "similarity": sim,
            })
    
    # Sort by similarity (most confusable first)
    pairs.sort(key=lambda x: -x["similarity"])
    
    print(f"\n── Dispatch Confusion Risk at L{layer_idx} ──")
    print(f"Most similar op pairs (hardest to dispatch independently):")
    for p in pairs[:15]:
        risk = "HIGH" if p["similarity"] > 0.7 else "MED" if p["similarity"] > 0.5 else "low"
        print(f"  [{risk:4s}] {p['op_a']:15s} ↔ {p['op_b']:15s}: {p['similarity']:.4f}")
    
    print(f"\nMost distinct op pairs (easiest to dispatch):")
    for p in pairs[-10:]:
        print(f"  [safe] {p['op_a']:15s} ↔ {p['op_b']:15s}: {p['similarity']:.4f}")
    
    with open(out_dir / f"dispatch_confusion_L{layer_idx}.json", "w") as f:
        json.dump({"layer": layer_idx, "pairs": pairs}, f, indent=2)
    
    return pairs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="results/kernel-basins")
    parser.add_argument("--layers", type=str, default="28,32,37",
                       help="Comma-separated layers to analyze")
    args = parser.parse_args()
    
    out_dir = Path(args.data_dir)
    op_acts, op_meta, expr_acts, expr_meta = load_activations(args.data_dir)
    
    layers = [int(x) for x in args.layers.split(",")]
    
    for layer_idx in layers:
        print(f"\n{'═'*60}")
        print(f"  Layer {layer_idx}")
        print(f"{'═'*60}")
        
        # Build centroids
        centroids = build_op_centroids(op_acts, op_meta, layer_idx)
        
        # Op similarity matrix + hierarchy
        hierarchy, sim, names = plot_op_similarity_matrix(centroids, layer_idx, out_dir)
        
        # Print hierarchy at different cuts
        for n_cuts in [3, 5, 7]:
            print(f"\n  Hierarchy at {n_cuts} groups:")
            for gid, members in sorted(hierarchy[f"cut_{n_cuts}"].items()):
                print(f"    Group {gid}: {', '.join(members)}")
        
        # Dispatch confusion analysis
        analyze_dispatch_granularity(centroids, layer_idx, out_dir)
        
        # Combined UMAP
        plot_combined_umap(op_acts, op_meta, expr_acts, expr_meta, layer_idx, out_dir)
    
    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
