#!/usr/bin/env python
"""
Post-hoc SCFlow embedding for ALL images (single SCFlow call), then 90% pruning.

Usage example:

python scripts/9embedding_extraction_sim.py \
  --parent_dir /home/wangyd/Projects/macs_thesis/yangyu/simulation_data \
  --scflow_inference_script yangyu/SCFlow/inference.py \
  --scflow_config SCFlow/configs/inference.yaml \
  --scflow_ckpt yangyu/ckpts/scflow_last.ckpt \
  --scflow_unclip_ckpt yangyu/ckpts/sd21-unclip-l.ckpt \
  --scflow_output_root yangyu/artwork_embeddings_simulations \
  --embedding_fraction 0.1
"""

import argparse
import random
import subprocess
from pathlib import Path
from typing import List


# ----------------------------- CLI -----------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Post-hoc SCFlow embeddings (all images via single call) + 90% pruning for simulation outputs."
    )

    p.add_argument(
        "--parent_dir",
        type=str,
        required=True,
        help="Directory containing sim_* folders.",
    )
    p.add_argument(
        "--sim_prefix",
        type=str,
        default="sim_",
        help="Only process subdirectories whose name starts with this prefix (default: sim_).",
    )

    # SCFlow paths
    p.add_argument(
        "--scflow_inference_script",
        type=str,
        required=True,
        help="Path to SCFlow/inference.py (can be relative).",
    )
    p.add_argument(
        "--scflow_config",
        type=str,
        required=True,
        help="Path to SCFlow config YAML (e.g., SCFlow/configs/inference.yaml).",
    )
    p.add_argument(
        "--scflow_ckpt",
        type=str,
        required=True,
        help="Path to SCFlow checkpoint (.ckpt).",
    )
    p.add_argument(
        "--scflow_unclip_ckpt",
        type=str,
        required=True,
        help="Path to SD2.1-unclip checkpoint (.ckpt) used by SCFlow.",
    )
    p.add_argument(
        "--scflow_output_root",
        type=str,
        required=True,
        help="Root directory for SCFlow embedding outputs.",
    )

    # Sampling
    p.add_argument(
        "--embedding_fraction",
        type=float,
        default=0.1,
        help="Fraction of images to KEEP on disk (e.g., 0.1 = 10%%). "
             "Embeddings are still computed for 100%% of images.",
    )
    p.add_argument(
        "--embedding_seed",
        type=int,
        default=123,
        help="Base random seed for sampling images to keep.",
    )

    # Misc
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, just print what would be done (no SCFlow, no deletions).",
    )

    return p.parse_args()


# ---------------------- helpers to find dirs --------------------------

def find_sim_dirs(parent_dir: Path, prefix: str) -> List[Path]:
    """
    Return paths to each simulation's `simulation_output` directory:
      parent_dir/sim_*/simulation_output
    """
    sims = []
    for child in sorted(parent_dir.iterdir()):
        if child.is_dir() and child.name.startswith(prefix):
            sim_output = child / "simulation_output"
            if sim_output.is_dir():
                sims.append(sim_output)
    return sims


def find_round_dirs(sim_output_dir: Path) -> List[Path]:
    rounds = []
    for child in sorted(sim_output_dir.iterdir()):
        if child.is_dir() and child.name.isdigit():
            rounds.append(child)
    return rounds


def find_artist_dirs(round_dir: Path) -> List[Path]:
    artists = []
    for child in sorted(round_dir.iterdir()):
        if child.is_dir() and not child.name.startswith("_"):
            artists.append(child)
    return artists


# ----------------------- main -----------------------------------------

def main():
    args = parse_args()
    parent = Path(args.parent_dir).resolve()
    if not parent.is_dir():
        raise FileNotFoundError(f"parent_dir not found: {parent}")

    sim_dirs = find_sim_dirs(parent, args.sim_prefix)
    print(f"[INFO] Found {len(sim_dirs)} simulation output dirs in {parent} with prefix '{args.sim_prefix}'.")

    # ------------------------------------------------------------------
    # 1) Run SCFlow ONCE on the entire parent_dir (recursive)
    # ------------------------------------------------------------------
    embeddings_root = Path(args.scflow_output_root)

    scflow_cmd = [
        "python",
        str(args.scflow_inference_script),
        "--config",
        str(args.scflow_config),
        "--resume_checkpoint",
        str(args.scflow_ckpt),
        "--name",
        str(embeddings_root),
        "--reverse_inference",
        "--image_mix_path",
        str(parent),
        "--unclip_ckpt",
        str(args.scflow_unclip_ckpt),
        "--seed",
        str(args.embedding_seed),
        "--recursive",  # new flag handled by modified inference.py
    ]

    print("\n[SCFLOW] Running SCFlow ONCE over the entire parent_dir (recursive).")
    print(f"[SCFLOW] Command: {' '.join(scflow_cmd)}")

    if not args.dry_run:
        subprocess.run(scflow_cmd, check=True)

    # ------------------------------------------------------------------
    # 2) Prune JPGs: per sim / round / artist
    # ------------------------------------------------------------------
    frac = max(0.0, min(1.0, float(args.embedding_fraction)))

    print("\n[PRUNE] Starting per-artist pruning after embeddings have been computed.")
    for sim_output_dir in sim_dirs:
        sim_name = sim_output_dir.parent.name
        print(f"\n[SIM] ===== {sim_name} ({sim_output_dir}) =====")

        round_dirs = find_round_dirs(sim_output_dir)
        if not round_dirs:
            print(f"[SIM] No round dirs in {sim_output_dir}, skipping.")
            continue

        for round_dir in round_dirs:
            r = int(round_dir.name)
            print(f"[ROUND] {sim_name} / round {r}")

            artist_dirs = find_artist_dirs(round_dir)
            if not artist_dirs:
                print(f"[ROUND] No artist dirs in {round_dir}, skipping.")
                continue

            for artist_dir in artist_dirs:
                artist_code = artist_dir.name

                # Collect all .jpg in this artist_dir
                image_files = sorted([p for p in artist_dir.glob("*.jpg") if p.is_file()])
                if not image_files:
                    print(f"[PRUNE] {sim_name} round={r} artist={artist_code}: no images, skip.")
                    continue

                # Determine how many to keep
                k = max(1, int(len(image_files) * frac))
                rng = random.Random(
                    args.embedding_seed
                    + r * 1000
                    + (hash(sim_name) % 1000)
                    + (hash(artist_code) % 1000)
                )
                selected = rng.sample(image_files, k)
                selected_set = set(selected)
                to_delete = [p for p in image_files if p not in selected_set]

                print(
                    f"[PRUNE] {sim_name} round={r} artist={artist_code}: "
                    f"KEEP {len(selected)} / DELETE {len(to_delete)} (fraction={frac:.3f})"
                )

                if not args.dry_run:
                    for p in to_delete:
                        try:
                            p.unlink()
                        except FileNotFoundError:
                            # In case the file was already removed by some other process
                            pass

                print(
                    f"[DONE] {sim_name} round={r} artist={artist_code}: "
                    f"embeddings for ALL images (via global SCFlow run), "
                    f"only {len(selected)} JPGs kept."
                )

    print("\n[DONE] Global SCFlow embedding (all images) + per-artist pruning complete.")


if __name__ == "__main__":
    main()
