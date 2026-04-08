#!/usr/bin/env python3
"""
Build per-artist JSONLs and train initial LoRA weights.

What this script does
---------------------
Reads a list of artist nodes, filters metadata to pre-max_year paintings for each artist, builds a per-artist painting_content-style JSONL, and then calls the SD1.5 LoRA training script (via accelerate) to produce a pre-trained LoRA checkpoint for each artist, stored in a dedicated folder.

Inputs
------
--nodes_csv            : CSV with a 'node' column (or first column) listing artist identifiers used as node names.
--jsonl_path           : global painting_content.jsonl → {"image": <image_id>, "caption": <text>} for all paintings.
--images_dir           : directory containing all painting images (e.g., artwork_images/ with <image_id>.jpg).
--metadata_csv         : metadata CSV (e.g., artwork_data_merged.csv) with at least artist, image, and year columns.
--pre_artist_jsonl_dir : directory to write per-artist JSONL files (one <artist_slug>.jsonl per artist).
--pre_artist_lora_dir  : directory to store per-artist LoRA output folders (<pre_artist_lora>/<artist_slug>/).
--artist_column        : column name in metadata_csv containing artist names (default: Artist_name).
--image_column         : column name in metadata_csv containing image IDs (default: image_n).
--year_column          : column name in metadata_csv containing painting years (default: Year).
--max_year             : upper bound for pre-training; only paintings with Year < max_year are used (default: 1500).
--train_script         : path to the LoRA training script to launch with accelerate (e.g., 6train_lora_sd15_paintings.py).
--sd15_dir             : path to local Stable Diffusion v1.5 directory used by the LoRA training script.
--resume_unet          : path to initial/global LoRA weights for continued training (optional; can be None).
--train_batch_size     : batch size passed through to the LoRA training script (default: 8).
--num_train_epochs     : number of epochs passed through to the LoRA training script (default: 10).
--learning_rate        : learning rate passed through to the LoRA training script (default: 1e-4).
--lora_rank            : LoRA rank passed through to the LoRA training script (default: 8).
--mixed_precision      : mixed precision mode for the LoRA training script (e.g., fp16).
--save_steps           : how often the LoRA training script saves intermediate weights (default: 2000).
--lr_scheduler         : LR scheduler name passed to the LoRA training script (default: cosine_with_restarts).
--lr_warmup_steps      : warmup steps passed to the LoRA training script (default: 500).
--max_artists          : optional cap on the number of artists to process (for testing; default: None = all).
--skip_existing        : if set, skip training for an artist when its LoRA output directory already exists and is non-empty.

Outputs
-------
Per-artist JSONL       : one JSONL per artist under pre_artist_jsonl_dir → {"image": <image_id>, "caption": <text>} for pre-max_year paintings.
Per-artist LoRA dirs   : one folder per artist under pre_artist_lora_dir containing LoRA weights produced by the training script.
Console logs           : progress messages for each artist (number of examples, skipped artists, training commands, and final counts).
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List, Set, Optional

import pandas as pd


# ----------------------------
# Helpers
# ----------------------------

def slugify(name: str) -> str:
    """Make a filesystem-safe slug from an artist name."""
    s = name.strip().lower()
    for ch in [" ", "/", "\\", ":", ";", ",", ".", "'", "\"", "(", ")", "[", "]"]:
        s = s.replace(ch, "-")
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def read_nodes(nodes_csv: str) -> List[str]:
    """
    Read node CSV. Assumes there is a 'node' column or uses the first column.
    """
    nodes: List[str] = []
    with open(nodes_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"No header in nodes CSV: {nodes_csv}")
        node_col = "node" if "node" in fieldnames else fieldnames[0]
        for row in reader:
            node = row[node_col].strip()
            if node:
                nodes.append(node)
    return nodes


def load_painting_content(jsonl_path: str) -> Dict[str, str]:
    """
    Load painting_content.jsonl into {image_id(str): caption}.
    """
    mapping: Dict[str, str] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            img = str(obj["image"])
            cap = obj["caption"]
            mapping[img] = cap
    return mapping


# ----------------------------
# Per-artist JSONL + LoRA
# ----------------------------

def build_artist_jsonl_and_train(
    artist: str,
    args: argparse.Namespace,
    content_map: Dict[str, str],
    meta_df: pd.DataFrame,
) -> Optional[str]:
    """
    For one artist:
    1. Filter metadata to pre-1500 paintings.
    2. Build an artist-specific JSONL in pre_artist directory.
    3. Call accelerate + train_lora script to train a LoRA model.
    Returns the LoRA output directory (or None if skipped).
    """
    slug = slugify(artist)

    # Filter metadata: Artist_name == artist AND Year < max_year
    year_series = pd.to_numeric(meta_df[args.year_column], errors="coerce")
    subset = meta_df[
        (meta_df[args.artist_column] == artist) &
        (year_series < args.max_year)
    ]
    if subset.empty:
        print(f"[WARN] Artist {artist} has no pre-{args.max_year} paintings; skipping.", file=sys.stderr)
        return None

    # Unique image IDs as strings
    image_ids = {
        str(int(x))
        for x in subset[args.image_column].dropna().unique()
    }

    # Build per-artist JSONL
    os.makedirs(args.pre_artist_jsonl_dir, exist_ok=True)
    artist_jsonl = os.path.join(args.pre_artist_jsonl_dir, f"{slug}.jsonl")

    n_written = 0
    with open(artist_jsonl, "w", encoding="utf-8") as out:
        for img_id in image_ids:
            cap = content_map.get(img_id)
            if cap is None:
                continue
            obj = {"image": img_id, "caption": cap}
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_written += 1

    if n_written == 0:
        print(f"[WARN] Artist {artist} had no matching captions; skipping training.", file=sys.stderr)
        return None

    print(f"[INFO] Artist {artist}: wrote {n_written} lines to {artist_jsonl}")

    # LoRA output directory
    lora_dir = os.path.join(args.pre_artist_lora_dir, slug)
    os.makedirs(lora_dir, exist_ok=True)

    # Skip if requested and directory already has content
    if args.skip_existing and os.listdir(lora_dir):
        print(f"[INFO] Skipping training for {artist} (non-empty existing {lora_dir})")
        return lora_dir

    # Build training command
    cmd = [
        "accelerate", "launch",
        args.train_script,
        "--sd15_dir", args.sd15_dir,
        "--jsonl_path", artist_jsonl,
        "--images_dir", args.images_dir,
        "--metadata_csv", args.metadata_csv,
        "--output_dir", lora_dir,
        "--train_batch_size", str(args.train_batch_size),
        "--num_train_epochs", str(args.num_train_epochs),
        "--learning_rate", str(args.learning_rate),
        "--lora_rank", str(args.lora_rank),
        "--mixed_precision", args.mixed_precision,
        "--save_steps", str(args.save_steps),
        "--lr_scheduler", args.lr_scheduler,
        "--lr_warmup_steps", str(args.lr_warmup_steps),
    ]
    if args.resume_unet:
        cmd.extend(["--resume_unet", args.resume_unet])

    print(f"[INFO] Training LoRA for {artist} in {lora_dir}")
    print("[CMD]", " ".join(cmd))

    # Run training (fail-fast if one artist fails)
    subprocess.run(cmd, check=True)

    return lora_dir


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build per-artist pre-1500 JSONLs and train LoRA weights."
    )

    # --- Nodes / lists ---
    parser.add_argument(
        "--nodes_csv",
        type=str,
        default="/home/wangyd/Projects/macs_thesis/macs-40123-yangyuwang/data/artist_network_start_nodelist_key.csv",
        help="CSV with a 'node' column listing artists.",
    )

    # --- Data paths ---
    parser.add_argument(
        "--jsonl_path",
        type=str,
        default="/home/wangyd/Projects/macs_thesis/yangyu/painting_content.jsonl",
        help="Global painting_content.jsonl.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/home/wangyd/Projects/macs_thesis/yangyu/artwork_images",
        help="Directory with all painting images.",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="/home/wangyd/Projects/macs_thesis/macs-40123-yangyuwang/data/artwork_data_merged.csv",
    )
    parser.add_argument(
        "--pre_artist_jsonl_dir",
        type=str,
        default="/home/wangyd/Projects/macs_thesis/yangyu/simulation_data/pre_artist",
        help="Where to write per-artist JSONL files.",
    )
    parser.add_argument(
        "--pre_artist_lora_dir",
        type=str,
        default="/home/wangyd/Projects/macs_thesis/yangyu/simulation_data/pre_artist_lora",
        help="Where to save per-artist LoRA weights.",
    )

    # --- Metadata column names ---
    parser.add_argument("--artist_column", type=str, default="Artist_name")
    parser.add_argument("--image_column", type=str, default="image_n")
    parser.add_argument("--year_column", type=str, default="Year")
    parser.add_argument(
        "--max_year",
        type=int,
        default=1500,
        help="Use paintings with Year < max_year for the pre-training.",
    )

    # --- LoRA / training script paths ---
    parser.add_argument(
        "--train_script",
        type=str,
        default="code/simulation/6train_lora_sd15_paintings.py",
        help="Path (relative to project root) to the LoRA training script.",
    )
    parser.add_argument(
        "--sd15_dir",
        type=str,
        default="/home/wangyd/Projects/macs_thesis/yangyu/diffusion_model/sd_v15",
    )
    parser.add_argument(
        "--resume_unet",
        type=str,
        default="/home/wangyd/Projects/macs_thesis/yangyu/diffusion_model/lora_sd15_pre1900/lora_weight.pt",
        help="Initial LoRA weights for continued training (optional).",
    )

    # --- LoRA hyperparams ---
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)

    # --- Control ---
    parser.add_argument(
        "--max_artists",
        type=int,
        default=None,
        help="Optional cap on number of artists (for testing).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip training if the target LoRA directory already exists and is non-empty.",
    )

    args = parser.parse_args()

    # --- Load nodes ---
    nodes = read_nodes(args.nodes_csv)
    print(f"[INFO] Loaded {len(nodes)} nodes from {args.nodes_csv}")

    # --- Load painting content and metadata ---
    print(f"[INFO] Loading painting content from {args.jsonl_path}")
    content_map = load_painting_content(args.jsonl_path)

    print(f"[INFO] Loading metadata from {args.metadata_csv}")
    meta_df = pd.read_csv(args.metadata_csv)

    # Column checks
    for col in [args.artist_column, args.image_column, args.year_column]:
        if col not in meta_df.columns:
            raise ValueError(f"Column '{col}' not found in metadata CSV.")

    # --- Loop over artists ---
    trained_count = 0
    for artist in nodes:
        if args.max_artists is not None and trained_count >= args.max_artists:
            print(f"[INFO] Reached max_artists={args.max_artists}; stopping.")
            break

        print(f"\n[ARTIST] ===== {artist} =====")
        lora_dir = build_artist_jsonl_and_train(artist, args, content_map, meta_df)
        if lora_dir is not None:
            trained_count += 1

    print(f"\n[INFO] Finished. Trained LoRA weights for {trained_count} artists.")


if __name__ == "__main__":
    main()
