#!/usr/bin/env python
"""
Artist-network simulation using sd_v15 + LoRA, with per-artist LoRA that evolves over rounds.

What this revised script does
-----------------------------
Simulates an artist network where each node (artist) has its own LoRA on top of SD v1.5.
In each round every artist generates images from text prompts using their current LoRA.
Then, for rounds r >= 1, each artist's LoRA is further trained on PRIOR generated images
(from rounds < r) sampled from all artists according to a polynomial interaction score.

Important changes
-----------------
1. The network is FIXED. There is no tie updating anymore.
2. Training images are sampled only from previously generated simulation images.
3. Sampling weights are based on the polynomial interaction score.
4. geoDistance is computed with haversine distance using node lat/lon and then normalized.
5. Optional movement:
   - each round, each artist moves with probability move_prob
   - movement is Gaussian noise on latitude/longitude
   - updated coordinates are used in interaction scores
6. If score >= 0, sampled prior generated images get light augmentation.
7. If score < 0, sampled prior generated images are CLIP-encoded, the embedding is negated,
   and then an offline unCLIP pipeline is used to generate a reversed training sample.

Offline requirement
-------------------
For negative-score samples, you must provide LOCAL paths to:
- a CLIP vision model directory
- an unCLIP image-decoder model directory

No network access is used at runtime.
"""

import argparse
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionPipeline, StableUnCLIPImg2ImgPipeline
from lora_diffusion import patch_pipe, tune_lora_scale
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


# ---------------------------------------------------------------------
# Fixed coefficients from the polynomial interaction model
# ---------------------------------------------------------------------
B0 = -0.050117
B_TIE_FIRST = 0.331598
B_TIE_SECOND = 0.286966
B_TIE_SELF = 0.309166
B_GEO1 = -0.083890
B_GEO2 = 0.018459
B_GEO1_SELF = -0.049966
B_GEO1_FIRST = -0.000597
B_GEO1_SECOND = 0.027998
B_GEO2_SELF = 0.025862
B_GEO2_FIRST = 0.002917
B_GEO2_SECOND = -0.038889

GEO_MEAN = 3141.184845
GEO_STD = 3509.569333


# ---------------------------------------------------------------------
# Global lazy-loaded offline models
# ---------------------------------------------------------------------
_CLIP_PROCESSOR = None
_CLIP_MODEL = None
_UNCLIP_PIPE = None


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Artist network simulation with sd15+LoRA (fixed network, per-artist, per-round)."
    )

    # Model + data paths
    parser.add_argument("--sd15_dir", type=str, required=True,
                        help="Path to local Stable Diffusion v1.5 directory.")
    parser.add_argument("--baseline_lora", type=str, required=True,
                        help="Path to baseline LoRA weights (UNet), used when artist has no own LoRA yet.")
    parser.add_argument("--lora_root", type=str, required=True,
                        help="Root directory where per-artist, per-round LoRAs are stored.")
    parser.add_argument("--pre_lora_root", type=str, default=None,
                        help="Root with pre-trained per-artist LoRAs for the FIRST round.")
    parser.add_argument("--prompts_jsonl", type=str, required=True,
                        help="painting_content.jsonl with {'image': ..., 'caption': ...}.")
    parser.add_argument("--nodes_csv", type=str, required=True,
                        help="CSV nodelist for the artist network with columns node, latitude, longitude.")
    parser.add_argument("--edges_csv", type=str, default=None,
                        help="CSV edgelist for the artist network with columns source and target.")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory for simulation outputs (images + jsonl).")

    # Offline local CLIP / unCLIP model dirs
    parser.add_argument("--clip_model_dir", type=str, default=None,
                        help="Local directory of CLIP vision model (required for negative-score reverse/unCLIP).")
    parser.add_argument("--unclip_model_dir", type=str, default=None,
                        help="Local directory of unCLIP img2img pipeline (required for negative-score reverse/unCLIP).")

    # Simulation parameters
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of simulation rounds.")
    parser.add_argument("--images_per_round", type=int, default=10,
                        help="How many images each artist generates per round.")
    parser.add_argument("--train_samples_per_round", type=int, default=30,
                        help="Total processed prior-generated training images per artist per round.")
    parser.add_argument("--random_edges", action="store_true",
                        help="If set, ignore edges_csv and generate random edges.")
    parser.add_argument("--edge_prob", type=float, default=0.5,
                        help="Probability of an edge in random graph mode (Erdos-Renyi).")

    # Movement parameters
    parser.add_argument("--move_prob", type=float, default=0.0,
                        help="Probability that an artist moves in a given round.")
    parser.add_argument("--move_lat_sd", type=float, default=0.0,
                        help="Std. dev. of latitude movement per move event (degrees).")
    parser.add_argument("--move_lon_sd", type=float, default=0.0,
                        help="Std. dev. of longitude movement per move event (degrees).")
    parser.add_argument("--move_start_round", type=int, default=1,
                        help="Round index at which movement starts.")

    # Generation hyperparameters
    parser.add_argument("--num_steps", type=int, default=25,
                        help="Num inference steps per image.")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--height", type=int, default=512,
                        help="Image height.")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width.")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for model inference ('cuda' or 'cpu').")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    if any(x is None for x in [lat1, lon1, lat2, lon2]):
        return None
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(6371.0 * c)


def z_geo(dist_km: Optional[float]) -> float:
    if dist_km is None or not np.isfinite(dist_km):
        return 0.0
    return float((dist_km - GEO_MEAN) / GEO_STD)


def clamp_latitude(lat: float) -> float:
    return float(max(-89.9, min(89.9, lat)))


def wrap_longitude(lon: float) -> float:
    lon = ((lon + 180.0) % 360.0) - 180.0
    return float(lon)


def maybe_move_nodes(
    node_meta_by_node: Dict[str, Dict],
    move_prob: float,
    move_lat_sd: float,
    move_lon_sd: float,
    rng: np.random.Generator,
) -> List[Dict]:
    """
    With probability move_prob, each node gets a random shift in latitude/longitude.
    Movement is Gaussian noise:
        dlat ~ N(0, move_lat_sd)
        dlon ~ N(0, move_lon_sd)
    """
    move_logs = []

    if move_prob <= 0:
        return move_logs

    for node_id, meta in node_meta_by_node.items():
        lat = meta.get("latitude")
        lon = meta.get("longitude")

        if lat is None or lon is None:
            continue

        if rng.random() >= move_prob:
            continue

        dlat = float(rng.normal(0.0, move_lat_sd))
        dlon = float(rng.normal(0.0, move_lon_sd))

        new_lat = clamp_latitude(lat + dlat)
        new_lon = wrap_longitude(lon + dlon)

        meta["latitude"] = new_lat
        meta["longitude"] = new_lon

        move_logs.append({
            "node": node_id,
            "artist_code": meta.get("artist_code"),
            "old_latitude": lat,
            "old_longitude": lon,
            "delta_latitude": dlat,
            "delta_longitude": dlon,
            "new_latitude": new_lat,
            "new_longitude": new_lon,
        })

    return move_logs


def save_node_positions(
    node_meta_by_node: Dict[str, Dict],
    out_path: Path,
):
    rows = []
    for node_id, meta in node_meta_by_node.items():
        rows.append({
            "node": node_id,
            "artist_code": meta.get("artist_code"),
            "latitude": meta.get("latitude"),
            "longitude": meta.get("longitude"),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ---------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------
def load_prompts(prompts_jsonl: str) -> List[str]:
    prompts = []
    path = Path(prompts_jsonl)
    if not path.is_file():
        raise FileNotFoundError(f"prompts_jsonl does not exist: {prompts_jsonl}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            caption = (obj.get("caption") or "").strip()
            if caption:
                prompts.append("a painting of " + caption)

    if not prompts:
        raise RuntimeError(f"No valid captions found in {prompts_jsonl}")

    print(f"[DATA] Loaded {len(prompts):,} prompts from {prompts_jsonl}")
    return prompts


def detect_node_and_code_columns(nodes_df: pd.DataFrame) -> Tuple[str, str]:
    if "node" not in nodes_df.columns:
        raise ValueError(
            f"Expected a 'node' column in nodes CSV, got columns: {nodes_df.columns.tolist()}"
        )
    print("[NETWORK] Using 'node' as node_id and artist_code.")
    return "node", "node"


def detect_edge_columns(edges_df: pd.DataFrame) -> Tuple[str, str]:
    if "source" not in edges_df.columns or "target" not in edges_df.columns:
        raise ValueError(
            f"Expected 'source' and 'target' columns in edges CSV, got columns: {edges_df.columns.tolist()}"
        )
    print("[NETWORK] Using 'source' as source, 'target' as target.")
    return "source", "target"


def load_node_metadata(nodes_csv: str) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    nodes_df = pd.read_csv(nodes_csv)
    node_id_col, code_col = detect_node_and_code_columns(nodes_df)

    if "latitude" not in nodes_df.columns or "longitude" not in nodes_df.columns:
        raise ValueError("nodes_csv must include 'latitude' and 'longitude' columns.")

    node_meta_by_node = {}
    for _, row in nodes_df.iterrows():
        node_id = str(row[node_id_col])
        lat = None if pd.isna(row["latitude"]) else float(row["latitude"])
        lon = None if pd.isna(row["longitude"]) else float(row["longitude"])
        node_meta_by_node[node_id] = {
            "artist_code": str(row[code_col]),
            "latitude": lat,
            "longitude": lon,
        }

    return nodes_df, node_meta_by_node


def build_network(
    nodes_csv: str,
    edges_csv: Optional[str],
    random_edges: bool,
    edge_prob: float,
    rng: random.Random,
) -> Tuple[nx.Graph, Dict[str, str], Dict[str, Dict]]:
    nodes_df, node_meta_by_node = load_node_metadata(nodes_csv)
    node_ids = nodes_df["node"].astype(str).tolist()

    G = nx.Graph()
    artist_code_by_node: Dict[str, str] = {}

    for node_id in node_ids:
        G.add_node(node_id)
        artist_code_by_node[node_id] = node_meta_by_node[node_id]["artist_code"]

    if random_edges or not edges_csv:
        print(f"[NETWORK] Building random graph with p={edge_prob}")
        n = len(node_ids)
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < edge_prob:
                    G.add_edge(node_ids[i], node_ids[j])
    else:
        print(f"[NETWORK] Loading edges from {edges_csv}")
        edges_df = pd.read_csv(edges_csv)
        src_col, dst_col = detect_edge_columns(edges_df)

        e = 0
        for row in edges_df.itertuples(index=False):
            u = str(getattr(row, src_col))
            v = str(getattr(row, dst_col))
            if u in G.nodes and v in G.nodes and u != v:
                G.add_edge(u, v)
                e += 1
        print(f"[NETWORK] Added {e:,} edges from {edges_csv}")

    print(f"[NETWORK] Graph has {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges")
    return G, artist_code_by_node, node_meta_by_node


# ---------------------------------------------------------------------
# LoRA & pipeline helpers
# ---------------------------------------------------------------------
def find_initial_lora_for_artist(
    pre_lora_root: Optional[Path],
    artist_code: str,
    baseline_lora: Path,
) -> Path:
    if pre_lora_root is not None:
        artist_dir = pre_lora_root / artist_code
        cand_pt = artist_dir / "lora_weight.pt"
        cand_safe = artist_dir / "lora_weight.safetensors"

        if cand_pt.is_file():
            return cand_pt
        if cand_safe.is_file():
            return cand_safe

    return baseline_lora


def load_pipeline_for_artist(
    sd15_dir: str,
    lora_path: Path,
    device: str,
) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        sd15_dir,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(device)

    patch_pipe(
        pipe,
        str(lora_path),
        patch_unet=True,
        patch_text=False,
        patch_ti=False,
    )
    tune_lora_scale(pipe.unet, 1.0)
    pipe.enable_attention_slicing("max")
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ---------------------------------------------------------------------
# Offline CLIP / unCLIP helpers
# ---------------------------------------------------------------------
def get_clip_models(clip_model_dir: str, device: str):
    global _CLIP_PROCESSOR, _CLIP_MODEL
    if _CLIP_PROCESSOR is None or _CLIP_MODEL is None:
        if clip_model_dir is None:
            raise ValueError("clip_model_dir is required for negative-score reverse/unCLIP samples.")
        _CLIP_PROCESSOR = CLIPImageProcessor.from_pretrained(clip_model_dir, local_files_only=True)
        _CLIP_MODEL = CLIPVisionModelWithProjection.from_pretrained(
            clip_model_dir,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            local_files_only=True
        ).to(device)
        _CLIP_MODEL.eval()
    return _CLIP_PROCESSOR, _CLIP_MODEL


def get_unclip_pipe(unclip_model_dir: str, device: str):
    global _UNCLIP_PIPE
    if _UNCLIP_PIPE is None:
        if unclip_model_dir is None:
            raise ValueError("unclip_model_dir is required for negative-score reverse/unCLIP samples.")
        _UNCLIP_PIPE = StableUnCLIPImg2ImgPipeline.from_pretrained(
            unclip_model_dir,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            local_files_only=True,
        )
        _UNCLIP_PIPE = _UNCLIP_PIPE.to(device)
        _UNCLIP_PIPE.set_progress_bar_config(disable=True)
    return _UNCLIP_PIPE


def reverse_clip_unclip_image(
    img: Image.Image,
    clip_model_dir: str,
    unclip_model_dir: str,
    device: str,
    seed: int,
) -> Image.Image:
    processor, clip_model = get_clip_models(clip_model_dir, device)
    unclip_pipe = get_unclip_pipe(unclip_model_dir, device)

    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    if device != "cpu":
        pixel_values = pixel_values.half()

    with torch.no_grad():
        embeds = clip_model(pixel_values).image_embeds

    rev_embeds = -1.0 * embeds
    rev_embeds = rev_embeds / (rev_embeds.norm(dim=-1, keepdim=True) + 1e-8)

    gen = torch.Generator(device=device).manual_seed(seed)

    out = unclip_pipe(
        image_embeds=rev_embeds,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=gen,
    ).images[0]

    return out


def augment_positive_image(img: Image.Image, rng: random.Random) -> Image.Image:
    img = img.convert("RGB")

    if rng.random() < 0.5:
        angle = rng.uniform(-8, 8)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False)

    if rng.random() < 0.5:
        img = ImageOps.mirror(img)

    if rng.random() < 0.5:
        w, h = img.size
        crop_ratio = rng.uniform(0.90, 0.98)
        nw, nh = int(w * crop_ratio), int(h * crop_ratio)
        left = rng.randint(0, max(0, w - nw))
        top = rng.randint(0, max(0, h - nh))
        img = img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.BICUBIC)

    return img


# ---------------------------------------------------------------------
# Interaction score computation
# ---------------------------------------------------------------------
def get_relation_type(G: nx.Graph, focal: str, other: str) -> Tuple[int, int, int]:
    if focal == other:
        return 0, 0, 1

    if G.has_edge(focal, other):
        return 1, 0, 0

    try:
        nbr_f = set(G.neighbors(focal))
        nbr_o = set(G.neighbors(other))
        if len(nbr_f.intersection(nbr_o)) > 0:
            return 0, 1, 0
    except Exception:
        pass

    return 0, 0, 0  # environment baseline


def interaction_score(
    G: nx.Graph,
    focal: str,
    other: str,
    node_meta_by_node: Dict[str, Dict],
) -> float:
    tie_first, tie_second, tie_self = get_relation_type(G, focal, other)

    if tie_self == 1:
        return 0.0

    lat1 = node_meta_by_node[focal]["latitude"]
    lon1 = node_meta_by_node[focal]["longitude"]
    lat2 = node_meta_by_node[other]["latitude"]
    lon2 = node_meta_by_node[other]["longitude"]

    dist = haversine_distance(lat1, lon1, lat2, lon2)
    geo1 = z_geo(dist)
    geo2 = geo1 ** 2

    y_hat = (
        B0
        + B_TIE_FIRST * tie_first
        + B_TIE_SECOND * tie_second
        + B_TIE_SELF * tie_self
        + B_GEO1 * geo1
        + B_GEO2 * geo2
        + B_GEO1_SELF * (geo1 * tie_self)
        + B_GEO1_FIRST * (geo1 * tie_first)
        + B_GEO1_SECOND * (geo1 * tie_second)
        + B_GEO2_SELF * (geo2 * tie_self)
        + B_GEO2_FIRST * (geo2 * tie_first)
        + B_GEO2_SECOND * (geo2 * tie_second)
    )
    return float(y_hat)


# ---------------------------------------------------------------------
# Training dataset construction from prior generated images only
# ---------------------------------------------------------------------
def safe_open_image(path: str) -> Optional[Image.Image]:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


def weighted_candidate_images(
    focal_node: str,
    round_idx: int,
    G: nx.Graph,
    node_meta_by_node: Dict[str, Dict],
    images_by_artist: Dict[str, List[Dict]],
) -> List[Dict]:
    candidates = []

    for other_node in G.nodes:
        if other_node == focal_node:
            continue

        pool = [
            meta for meta in images_by_artist.get(other_node, [])
            if meta["round"] < round_idx
        ]
        if not pool:
            continue

        score = interaction_score(G, focal_node, other_node, node_meta_by_node)
        weight = abs(score)
        if weight <= 0:
            continue

        candidates.append({
            "other_node": other_node,
            "artist_code": node_meta_by_node[other_node]["artist_code"],
            "score": score,
            "weight": weight,
            "pool": pool,
        })

    return candidates


def build_training_data_for_artist(
    node_id: str,
    artist_code: str,
    round_idx: int,
    G: nx.Graph,
    node_meta_by_node: Dict[str, Dict],
    images_by_artist: Dict[str, List[Dict]],
    train_root: Path,
    train_samples_per_round: int,
    clip_model_dir: Optional[str],
    unclip_model_dir: Optional[str],
    device: str,
    base_seed: int,
) -> Optional[Tuple[Path, Path]]:
    candidates = weighted_candidate_images(
        focal_node=node_id,
        round_idx=round_idx,
        G=G,
        node_meta_by_node=node_meta_by_node,
        images_by_artist=images_by_artist,
    )

    if not candidates:
        return None

    weights = np.array([c["weight"] for c in candidates], dtype=float)
    weights = weights / weights.sum()

    counts = np.random.default_rng(base_seed + round_idx + hash(node_id) % 100000).multinomial(
        train_samples_per_round,
        weights
    )

    train_dir = train_root / f"round{round_idx}" / artist_code
    image_dir = train_dir / "images"
    train_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = train_dir / "train.jsonl"
    rows_written = 0
    rng_local = random.Random(base_seed + 10007 * round_idx + hash(node_id) % 1000003)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for cand, n_take in zip(candidates, counts):
            if n_take <= 0:
                continue

            pool = cand["pool"]
            score = cand["score"]
            source_artist = cand["artist_code"]
            sampled = [rng_local.choice(pool) for _ in range(int(n_take))]

            for j, meta in enumerate(sampled):
                src_img = safe_open_image(meta["path"])
                if src_img is None:
                    continue

                stem_src = Path(meta["path"]).stem

                if score >= 0:
                    out_img = augment_positive_image(src_img, rng_local)
                    aug_tag = "pos_aug"
                else:
                    out_img = reverse_clip_unclip_image(
                        src_img,
                        clip_model_dir=clip_model_dir,
                        unclip_model_dir=unclip_model_dir,
                        device=device,
                        seed=base_seed + round_idx * 10000 + j,
                    )
                    aug_tag = "neg_rev_unclip"

                out_name = f"{aug_tag}__from_{source_artist}__r{round_idx}__{j}__{stem_src}.jpg"
                out_path = image_dir / out_name
                out_img.save(out_path)

                f.write(json.dumps({
                    "image": out_path.stem,
                    "caption": meta.get("caption", f"a painting by {source_artist}")
                }, ensure_ascii=False) + "\n")
                rows_written += 1

    if rows_written == 0:
        return None

    print(
        f"[TRAIN-DATA] round={round_idx} artist={artist_code} "
        f"jsonl={jsonl_path} n_images={rows_written}"
    )
    return jsonl_path, image_dir


def run_lora_training_for_artist(
    sd15_dir: str,
    train_script: Path,
    jsonl_path: Path,
    images_dir: Path,
    metadata_csv: str,
    output_dir: Path,
    resume_unet: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "accelerate",
        "launch",
        str(train_script),
        "--sd15_dir", str(sd15_dir),
        "--jsonl_path", str(jsonl_path),
        "--images_dir", str(images_dir),
        "--metadata_csv", str(metadata_csv),
        "--output_dir", str(output_dir),
        "--train_batch_size", "4",
        "--num_train_epochs", "10",
        "--learning_rate", "1e-4",
        "--lr_scheduler", "cosine_with_restarts",
        "--lr_warmup_steps", "500",
        "--lora_rank", "8",
        "--mixed_precision", "fp16",
        "--save_steps", "0",
        "--resume_unet", str(resume_unet),
        "--no_metadata",
    ]

    print(f"[TRAIN] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    base_seed = args.seed if args.seed is not None else 42
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    lora_root = Path(args.lora_root)
    lora_root.mkdir(parents=True, exist_ok=True)

    baseline_lora = Path(args.baseline_lora)
    if not baseline_lora.is_file():
        raise FileNotFoundError(f"baseline_lora not found: {baseline_lora}")

    pre_lora_root = Path(args.pre_lora_root) if args.pre_lora_root is not None else None
    if pre_lora_root is not None and not pre_lora_root.is_dir():
        raise FileNotFoundError(f"pre_lora_root does not exist or is not a directory: {pre_lora_root}")

    train_root = output_root / "_train"
    train_root.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts_jsonl)

    rng_net = random.Random(base_seed)
    G, artist_code_by_node, node_meta_by_node = build_network(
        args.nodes_csv,
        args.edges_csv,
        args.random_edges,
        args.edge_prob,
        rng_net,
    )

    all_nodes = list(G.nodes)

    graph_stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
    }

    config = {
        "simulation_args": vars(args),
        "initial_graph_stats": graph_stats,
    }

    config_path = output_root / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    edge_rows = []
    for u, v, data in G.edges(data=True):
        edge_rows.append({"source": u, "target": v})

    edge_path = output_root / "initial_edgelist.csv"
    pd.DataFrame(edge_rows).to_csv(edge_path, index=False)
    print(f"[INFO] Saved {len(edge_rows)} edges to {edge_path}")

    initial_positions_path = output_root / "node_positions_round0_initial.csv"
    save_node_positions(node_meta_by_node, initial_positions_path)
    print(f"[INFO] Saved initial node positions to {initial_positions_path}")

    current_lora_path_by_node: Dict[str, Path] = {}

    for node in all_nodes:
        artist_code = artist_code_by_node[node]
        path = find_initial_lora_for_artist(pre_lora_root, artist_code, baseline_lora)
        current_lora_path_by_node[node] = path
        if path == baseline_lora:
            print(f"[LORA-INIT] node={node} artist={artist_code} -> baseline LoRA")
        else:
            print(f"[LORA-INIT] node={node} artist={artist_code} -> pre-trained LoRA {path}")

    # images_by_artist[node_id] = list of {round, path, caption}
    images_by_artist: Dict[str, List[Dict]] = defaultdict(list)

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    print(f"[DEVICE] {device}")

    this_file = Path(__file__).resolve()
    train_script = this_file.parent / "6train_lora_sd15_paintings.py"

    movement_rng = np.random.default_rng(base_seed + 999)

    for r in range(args.rounds):
        print(f"\n[ROUND] ===== {r} =====")

        # 0) movement step
        if r >= args.move_start_round and args.move_prob > 0:
            move_logs = maybe_move_nodes(
                node_meta_by_node=node_meta_by_node,
                move_prob=args.move_prob,
                move_lat_sd=args.move_lat_sd,
                move_lon_sd=args.move_lon_sd,
                rng=movement_rng,
            )

            if move_logs:
                move_log_path = output_root / f"movement_round{r}.csv"
                pd.DataFrame(move_logs).to_csv(move_log_path, index=False)
                print(f"[MOVE] round={r}: {len(move_logs)} artists moved. Log saved to {move_log_path}")
            else:
                print(f"[MOVE] round={r}: no artists moved.")

        pos_path = output_root / f"node_positions_round{r}.csv"
        save_node_positions(node_meta_by_node, pos_path)
        print(f"[MOVE] Saved node positions for round {r} to {pos_path}")

        # 1) generation step
        for node_id in all_nodes:
            artist_code = artist_code_by_node[node_id]
            artist_dir = output_root / f"{r}" / artist_code
            artist_dir.mkdir(parents=True, exist_ok=True)

            lora_path = current_lora_path_by_node[node_id]
            print(f"[GEN] round={r} node={node_id} artist={artist_code} LoRA={lora_path}")

            pipe = load_pipeline_for_artist(
                sd15_dir=args.sd15_dir,
                lora_path=lora_path,
                device=device,
            )

            round_prompts = random.choices(prompts, k=args.images_per_round)

            generated_jsonl_path = artist_dir / "generated.jsonl"
            with generated_jsonl_path.open("w", encoding="utf-8") as jf:
                for idx, prompt in enumerate(round_prompts):
                    filename = f"{r}_{artist_code}_{idx}.jpg"
                    out_path = artist_dir / filename

                    if device != "cpu":
                        with torch.autocast("cuda"):
                            image = pipe(
                                prompt,
                                num_inference_steps=args.num_steps,
                                guidance_scale=args.guidance_scale,
                                height=args.height,
                                width=args.width,
                            ).images[0]
                    else:
                        image = pipe(
                            prompt,
                            num_inference_steps=args.num_steps,
                            guidance_scale=args.guidance_scale,
                            height=args.height,
                            width=args.width,
                        ).images[0]

                    image.save(out_path)

                    stem = out_path.stem
                    jf.write(json.dumps({"image": stem, "caption": prompt}, ensure_ascii=False) + "\n")

                    images_by_artist[node_id].append(
                        {"round": r, "path": str(out_path), "caption": prompt}
                    )

            del pipe
            if device != "cpu":
                torch.cuda.empty_cache()

        # 2) training step using prior generated images from rounds < r
        for node_id in all_nodes:
            artist_code = artist_code_by_node[node_id]

            td = build_training_data_for_artist(
                node_id=node_id,
                artist_code=artist_code,
                round_idx=r,
                G=G,
                node_meta_by_node=node_meta_by_node,
                images_by_artist=images_by_artist,
                train_root=train_root,
                train_samples_per_round=args.train_samples_per_round,
                clip_model_dir=args.clip_model_dir,
                unclip_model_dir=args.unclip_model_dir,
                device=device,
                base_seed=base_seed,
            )

            if td is None:
                print(f"[TRAIN] round={r} artist={artist_code}: no prior generated samples, skipping training.")
                continue

            jsonl_path, images_dir = td
            resume_lora = current_lora_path_by_node[node_id]

            artist_lora_dir = lora_root / artist_code / f"round{r}"
            run_lora_training_for_artist(
                sd15_dir=args.sd15_dir,
                train_script=train_script,
                jsonl_path=jsonl_path,
                images_dir=images_dir,
                metadata_csv="",
                output_dir=artist_lora_dir,
                resume_unet=resume_lora,
            )

            new_lora_path = artist_lora_dir / "lora_weight.pt"
            if new_lora_path.is_file():
                current_lora_path_by_node[node_id] = new_lora_path
                print(f"[LORA] Updated artist={artist_code} to {new_lora_path}")
            else:
                print(
                    f"[WARN] Expected LoRA {new_lora_path} not found; "
                    f"keeping previous LoRA for artist={artist_code}."
                )

    print("\n[SIM] Simulation finished.")


if __name__ == "__main__":
    main()