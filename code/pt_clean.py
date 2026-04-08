#!/usr/bin/env python3
"""
pt_clean.py

Examples of supported sim dirs:
  sim_paris_validation1_16
  sim_paris_edge01_smallmove1_16
  sim_paris_edge05_largemove4_16
  sim_shanghai_validation3_16
  sim_shanghai_edge01_smallmove2_16
  sim_shanghai_edge05_largemove4_16
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

SIM_DIR_RE = re.compile(
    r"^sim_(?P<city>paris|shanghai)_(?P<condition>.+?)(?P<replicate>\d+)_(?P<run>\d+)$"
)


def extract_tensor(obj: Any) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, dict):
        for k in ["tensor", "pred", "output", "data", "x", "s", "c"]:
            v = obj.get(k, None)
            if torch.is_tensor(v):
                return v
        for v in obj.values():
            if torch.is_tensor(v):
                return v

    if isinstance(obj, (list, tuple)):
        for v in obj:
            if torch.is_tensor(v):
                return v

    raise ValueError(f"Could not extract a torch.Tensor from object of type {type(obj)}")


def build_suffix(tensor_kind: str) -> str:
    """
    tensor_kind:
      - s  -> _clip_pred_s_tensor.pt
      - c  -> _clip_pred_c_tensor.pt
      - full suffix also allowed
    """
    if tensor_kind in {"s", "c"}:
        return f"_clip_pred_{tensor_kind}_tensor.pt"
    if tensor_kind.endswith(".pt"):
        return tensor_kind
    raise ValueError(
        "--tensor_kind must be one of {'s','c'} or a full suffix ending with .pt"
    )


def build_id_regex(suffix: str) -> re.Pattern:
    return re.compile(rf"_(?P<id>\d+){re.escape(suffix)}$")


def parse_metadata_from_path(
    pt_path: Path,
    *,
    suffix: str,
    id_regex: re.Pattern,
) -> Optional[Dict[str, Any]]:
    parts = pt_path.parts

    try:
        sim_out_idx = parts.index("simulation_output")
    except ValueError:
        return None

    if sim_out_idx + 3 >= len(parts):
        return None

    round_str = parts[sim_out_idx + 1]
    artist = parts[sim_out_idx + 2]
    filename = parts[sim_out_idx + 3]

    if not round_str.isdigit():
        return None
    round_i = int(round_str)

    if not filename.endswith(suffix):
        return None

    m_id = id_regex.search(filename)
    if not m_id:
        return None
    id_i = int(m_id.group("id"))

    prefix = f"{round_i}_{artist}_"
    if not filename.startswith(prefix):
        return None

    m_sim = None
    sim_dir_name = None
    for j in range(sim_out_idx - 1, -1, -1):
        if parts[j].startswith("sim_"):
            m_sim = SIM_DIR_RE.match(parts[j])
            if m_sim:
                sim_dir_name = parts[j]
                break

    if not m_sim or sim_dir_name is None:
        return None

    city = m_sim.group("city")
    condition = m_sim.group("condition")
    replicate = int(m_sim.group("replicate"))
    run = int(m_sim.group("run"))

    return {
        "city": city,
        "condition": condition,
        "replicate": replicate,
        "run": run,
        "round": round_i,
        "artist": artist,
        "id": id_i,
        "path": str(pt_path),
        "sim_dir": sim_dir_name,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory to search")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument(
        "--tensor_kind",
        default="s",
        help="Tensor kind: 's', 'c', or full suffix like '_clip_pred_s_tensor.pt'",
    )
    ap.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten tensors to 1D before saving.",
    )
    args = ap.parse_args()

    suffix = build_suffix(args.tensor_kind)
    id_regex = build_id_regex(suffix)

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"--root not found: {root}")

    pt_files = sorted(root.rglob(f"*{suffix}"))
    if not pt_files:
        raise RuntimeError(f"No files found under {root} matching *{suffix}")

    embeddings_list: List[np.ndarray] = []
    meta_city: List[str] = []
    meta_condition: List[str] = []
    meta_replicate: List[int] = []
    meta_run: List[int] = []
    meta_round: List[int] = []
    meta_artist: List[str] = []
    meta_id: List[int] = []
    meta_path: List[str] = []
    meta_sim_dir: List[str] = []

    skipped_pattern = 0
    load_errors = 0

    for p in pt_files:
        md = parse_metadata_from_path(
            p,
            suffix=suffix,
            id_regex=id_regex,
        )
        if md is None:
            skipped_pattern += 1
            continue

        try:
            obj = torch.load(str(p), map_location="cpu")
            t = extract_tensor(obj).detach().cpu()
        except Exception:
            load_errors += 1
            continue

        if args.flatten:
            t = t.reshape(-1)

        arr = t.to(torch.float32).numpy()

        embeddings_list.append(arr)
        meta_city.append(md["city"])
        meta_condition.append(md["condition"])
        meta_replicate.append(md["replicate"])
        meta_run.append(md["run"])
        meta_round.append(md["round"])
        meta_artist.append(md["artist"])
        meta_id.append(md["id"])
        meta_path.append(md["path"])
        meta_sim_dir.append(md["sim_dir"])

    if not embeddings_list:
        raise RuntimeError(
            "No valid .pt files parsed after applying expected path pattern "
            f"(skipped_pattern={skipped_pattern}, load_errors={load_errors})."
        )

    shapes = {e.shape for e in embeddings_list}
    if len(shapes) != 1:
        shape_counts: Dict[Tuple[int, ...], int] = {}
        for e in embeddings_list:
            shape_counts[e.shape] = shape_counts.get(e.shape, 0) + 1
        raise ValueError(
            "Tensors have inconsistent shapes; cannot stack into one array.\n"
            f"Shape counts: {shape_counts}\n"
            "Re-run with --flatten if appropriate."
        )

    embeddings = np.stack(embeddings_list, axis=0).astype(np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        city=np.array(meta_city, dtype=object),
        condition=np.array(meta_condition, dtype=object),
        replicate=np.array(meta_replicate, dtype=np.int32),
        run=np.array(meta_run, dtype=np.int32),
        round=np.array(meta_round, dtype=np.int32),
        artist=np.array(meta_artist, dtype=object),
        id=np.array(meta_id, dtype=np.int32),
        path=np.array(meta_path, dtype=object),
        sim_dir=np.array(meta_sim_dir, dtype=object),
        source_root=str(root),
        tensor_kind=str(args.tensor_kind),
        suffix=str(suffix),
        flatten=bool(args.flatten),
        found_files=int(len(pt_files)),
        skipped_pattern=int(skipped_pattern),
        load_errors=int(load_errors),
        embeddings_shape=np.array(embeddings.shape, dtype=np.int64),
    )

    print(
        "Found files: {found} | Parsed+saved: {saved} | Skipped (pattern mismatch): {skipped} | "
        "Load errors: {load_errors}".format(
            found=len(pt_files),
            saved=embeddings.shape[0],
            skipped=skipped_pattern,
            load_errors=load_errors,
        )
    )
    print(f"Tensor kind: {args.tensor_kind}")
    print(f"Suffix: {suffix}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()