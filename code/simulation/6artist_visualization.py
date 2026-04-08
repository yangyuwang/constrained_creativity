#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# =========================================================
# 1. Paths
# =========================================================
base_dir = "/home/wangyd/Projects/macs_thesis/data"

nodelist_key_paris_path = os.path.join(base_dir, "nodelist_key_paris.csv")
nodelist_key_shanghai_path = os.path.join(base_dir, "nodelist_key_shanghai.csv")
edgelist_key_paris_path = os.path.join(base_dir, "edgelist_key_paris.csv")
edgelist_key_shanghai_path = os.path.join(base_dir, "edgelist_key_shanghai.csv")

output_dir = os.path.join(base_dir, "network_geo_plots")
os.makedirs(output_dir, exist_ok=True)


# =========================================================
# 2. Helpers
# =========================================================
def load_nodelist(path):
    df = pd.read_csv(path)
    needed = {"node", "latitude", "longitude"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def load_edgelist(path):
    df = pd.read_csv(path)
    needed = {"source", "target"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def build_graph(nodelist_df, edgelist_df):
    G = nx.Graph()

    for _, row in nodelist_df.iterrows():
        G.add_node(
            row["node"],
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
        )

    for _, row in edgelist_df.iterrows():
        s = row["source"]
        t = row["target"]

        # skip self-loops
        if s == t:
            continue

        if s in G and t in G:
            attrs = row.to_dict()
            attrs.pop("source", None)
            attrs.pop("target", None)
            G.add_edge(s, t, **attrs)

    return G


def make_geo_positions(G):
    pos = {}
    for n, d in G.nodes(data=True):
        lon = d.get("longitude", 0.0)
        lat = d.get("latitude", 0.0)
        pos[n] = np.array([lon, lat], dtype=float)
    return pos


def jitter_overlaps(pos, jitter_strength=0.15, seed=42):
    rng = np.random.default_rng(seed)
    seen = {}
    out = {}

    for node, xy in pos.items():
        key = (round(float(xy[0]), 6), round(float(xy[1]), 6))
        count = seen.get(key, 0)
        seen[key] = count + 1

        if count == 0:
            out[node] = xy.copy()
        else:
            angle = rng.uniform(0, 2 * np.pi)
            radius = jitter_strength * (1 + 0.35 * count)
            offset = np.array([np.cos(angle), np.sin(angle)]) * radius
            out[node] = xy + offset

    return out


def geo_spring_layout(
    G,
    iterations=300,
    seed=42,
    k=0.95,
    jitter_strength=0.08,
    spring_weight=0.18,
):
    """
    Start from lon/lat positions, add small jitter for overlaps,
    then apply a spring layout while keeping each node weakly anchored
    near its geographic location.
    """
    base_pos = make_geo_positions(G)
    base_pos = jitter_overlaps(base_pos, jitter_strength=jitter_strength, seed=seed)

    arr = np.array(list(base_pos.values()))
    if len(arr) > 0:
        x_span = max(arr[:, 0].max() - arr[:, 0].min(), 1e-6)
        y_span = max(arr[:, 1].max() - arr[:, 1].min(), 1e-6)
        scale = max(x_span, y_span)
        for n in base_pos:
            base_pos[n] = base_pos[n] / scale

    H = G.copy()
    fixed_nodes = []
    initial_pos = {}

    for n, xy in base_pos.items():
        anchor = f"__anchor__{n}"
        H.add_node(anchor)
        H.add_edge(n, anchor, weight=spring_weight)
        initial_pos[n] = xy
        initial_pos[anchor] = xy
        fixed_nodes.append(anchor)

    pos_all = nx.spring_layout(
        H,
        pos=initial_pos,
        fixed=fixed_nodes,
        seed=seed,
        iterations=iterations,
        k=k,
        weight="weight",
    )

    return {n: pos_all[n] for n in G.nodes()}


def draw_network_geo(
    G,
    title,
    outpath,
    figsize=(12, 10),
    node_size_base=900,
    node_size_mult=180,
    font_size=9,
    seed=42,
):
    if G.number_of_nodes() == 0:
        print(f"[WARN] Empty graph for {title}")
        return

    pos = geo_spring_layout(
        G,
        iterations=300,
        seed=seed,
        k=0.95,
        jitter_strength=0.08,
        spring_weight=0.18,
    )

    degrees = dict(G.degree())  # self-loops already removed
    node_sizes = [node_size_base + node_size_mult * degrees[n] for n in G.nodes()]

    plt.figure(figsize=figsize)

    nx.draw_networkx_edges(
        G,
        pos,
        width=1.2,
        alpha=0.45,
        edge_color="gray",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.8,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=font_size,
        font_family="sans-serif",
    )

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


def draw_combined_network_geo(
    G_paris,
    G_shanghai,
    outpath,
    figsize=(14, 11),
    font_size=8,
):
    G = nx.Graph()

    for n, d in G_paris.nodes(data=True):
        G.add_node(n, **d, community="Paris")
    for u, v, d in G_paris.edges(data=True):
        if u != v:
            G.add_edge(u, v, **d)

    for n, d in G_shanghai.nodes(data=True):
        if n in G.nodes():
            G.nodes[n]["community"] = "Both"
        else:
            G.add_node(n, **d, community="Shanghai")

    for u, v, d in G_shanghai.edges(data=True):
        if u != v:
            G.add_edge(u, v, **d)

    pos = geo_spring_layout(
        G,
        iterations=350,
        seed=42,
        k=1.0,
        jitter_strength=0.1,
        spring_weight=0.15,
    )

    degrees = dict(G.degree())
    node_sizes = [900 + 180 * degrees[n] for n in G.nodes()]

    node_colors = []
    for n in G.nodes():
        comm = G.nodes[n].get("community")
        if comm == "Paris":
            node_colors.append("skyblue")
        elif comm == "Shanghai":
            node_colors.append("tomato")
        else:
            node_colors.append("gold")

    plt.figure(figsize=figsize)

    nx.draw_networkx_edges(
        G,
        pos,
        width=1.1,
        alpha=0.4,
        edge_color="gray",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.8,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=font_size,
        font_family="sans-serif",
    )

    plt.title("Combined Paris + Shanghai Key-Actor Networks", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


# =========================================================
# 3. Load data
# =========================================================
nodelist_paris = load_nodelist(nodelist_key_paris_path)
nodelist_shanghai = load_nodelist(nodelist_key_shanghai_path)
edgelist_paris = load_edgelist(edgelist_key_paris_path)
edgelist_shanghai = load_edgelist(edgelist_key_shanghai_path)

G_paris = build_graph(nodelist_paris, edgelist_paris)
G_shanghai = build_graph(nodelist_shanghai, edgelist_shanghai)

print(f"Paris: {G_paris.number_of_nodes()} nodes, {G_paris.number_of_edges()} edges")
print(f"Shanghai: {G_shanghai.number_of_nodes()} nodes, {G_shanghai.number_of_edges()} edges")


# =========================================================
# 4. Draw
# =========================================================
draw_network_geo(
    G_paris,
    title="Paris Key Actors Network",
    outpath=os.path.join(output_dir, "network_paris_geo.png"),
    figsize=(14, 11),
    font_size=8,
)

draw_network_geo(
    G_shanghai,
    title="Shanghai Key Actors Network",
    outpath=os.path.join(output_dir, "network_shanghai_geo.png"),
    figsize=(10, 8),
    font_size=9,
)

draw_combined_network_geo(
    G_paris,
    G_shanghai,
    outpath=os.path.join(output_dir, "network_combined_geo.png"),
    figsize=(15, 12),
    font_size=8,
)