#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.metrics.pairwise import cosine_distances


# ----------------------------
# Basic utilities
# ----------------------------
def cosine_similarity_vec(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def cosine_distance_vec(a: np.ndarray, b: np.ndarray) -> float:
    cs = cosine_similarity_vec(a, b)
    if not np.isfinite(cs):
        return np.nan
    return float(1.0 - cs)


def l2_norm(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    return float(np.linalg.norm(a))


def centroid_from_rows(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return np.full((X.shape[1],), np.nan, dtype=np.float32)
    return X[rows].mean(axis=0).astype(np.float32)


def mean_ci_normal(values: np.ndarray) -> Tuple[float, float, float]:
    """
    Mean ± 95% CI using normal approximation: mean ± 1.96 * SE.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    m = float(v.mean())
    if v.size == 1:
        return (m, np.nan, np.nan)
    se = float(v.std(ddof=1) / np.sqrt(v.size))
    lo = m - 1.96 * se
    hi = m + 1.96 * se
    return (m, lo, hi)


def wide_vectors_df(meta_rows: List[Dict], vecs: np.ndarray, prefix: str) -> pd.DataFrame:
    meta = pd.DataFrame(meta_rows)
    D = vecs.shape[1]
    cols = [f"{prefix}{i}" for i in range(D)]
    wide = pd.DataFrame(vecs, columns=cols)
    return pd.concat([meta.reset_index(drop=True), wide.reset_index(drop=True)], axis=1)


# ----------------------------
# ID normalization helpers
# ----------------------------
def normalize_image_n(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        xf = float(s)
        if np.isfinite(xf) and xf.is_integer():
            return str(int(xf))
        return s
    except Exception:
        return s


_slug_re = re.compile(r"[^a-z0-9]+")


def slugify_artist(name: str) -> str:
    s = str(name).strip().lower()
    s = _slug_re.sub("-", s)
    return s.strip("-")


# ----------------------------
# Embedding shape handling
# ----------------------------
def coerce_embeddings(X: np.ndarray, mode: str = "squeeze_last") -> np.ndarray:
    """
    Convert embeddings array into shape (N, D).
    - squeeze_last: squeezes singleton dims (e.g., (N,1,1,768)->(N,768)).
    - flatten: flattens all dims after N.
    """
    X = np.asarray(X)
    if X.ndim < 2:
        raise ValueError(f"embeddings must have at least 2 dims (N, D). Got {X.shape}")

    if mode == "flatten":
        N = X.shape[0]
        return X.reshape(N, -1).astype(np.float32, copy=False)

    if mode != "squeeze_last":
        raise ValueError(f"Unknown embed mode: {mode}")

    if X.ndim == 2:
        return X.astype(np.float32, copy=False)

    squeezed = X
    for axis in range(squeezed.ndim - 1, 0, -1):
        if squeezed.shape[axis] == 1:
            squeezed = np.squeeze(squeezed, axis=axis)

    if squeezed.ndim != 2:
        raise ValueError(
            f"After squeezing singleton dims, embeddings not 2D: {squeezed.shape}. "
            f"Use --embed-mode flatten if this is expected."
        )

    return squeezed.astype(np.float32, copy=False)


# ----------------------------
# Load SIM NPZ
# ----------------------------
def load_sim_npz(npz_path: str, embed_mode: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Supports both schemas:
    1) legacy NPZ: setting / number / round / artist / id / path
    2) city-style NPZ: condition / replicate / run / round / artist / id / path / city

    Returned canonical columns always include:
    setting, number, round, artist, id, path, run, city
    """
    data = np.load(npz_path, allow_pickle=True)
    files = set(data.files)
    X = coerce_embeddings(data["embeddings"], mode=embed_mode)

    if {"setting", "number", "round", "artist", "id", "path"}.issubset(files):
        df = pd.DataFrame({
            "setting": pd.Series(data["setting"]).astype(str),
            "number": pd.Series(data["number"]).astype(int),
            "round":  pd.Series(data["round"]).astype(int),
            "artist": pd.Series(data["artist"]).astype(str),
            "id":     pd.Series(data["id"]).astype(int),
            "path":   pd.Series(data["path"]).astype(str),
        })
        df["run"] = pd.Series(data["run"]).astype(int) if "run" in files else -1
        df["city"] = pd.Series(data["city"]).astype(str) if "city" in files else "unknown"

    elif {"condition", "replicate", "round", "artist", "id", "path"}.issubset(files):
        df = pd.DataFrame({
            "setting": pd.Series(data["condition"]).astype(str),
            "number": pd.Series(data["replicate"]).astype(int),
            "round":  pd.Series(data["round"]).astype(int),
            "artist": pd.Series(data["artist"]).astype(str),
            "id":     pd.Series(data["id"]).astype(int),
            "path":   pd.Series(data["path"]).astype(str),
        })
        df["run"] = pd.Series(data["run"]).astype(int) if "run" in files else -1
        df["city"] = pd.Series(data["city"]).astype(str) if "city" in files else "unknown"
    else:
        raise ValueError(
            "Sim NPZ schema not recognized. Need either legacy keys "
            "{embeddings, setting, number, round, artist, id, path} or city-style keys "
            "{embeddings, condition, replicate, round, artist, id, path}. "
            f"Found: {sorted(data.files)}"
        )

    if len(df) != X.shape[0]:
        raise ValueError(f"len(sim_meta)={len(df)} != sim_embeddings_rows={X.shape[0]}")

    df["_orig_row"] = np.arange(len(df), dtype=np.int64)
    return df.reset_index(drop=True), X


# ----------------------------
# Load REAL meta + embeddings
# ----------------------------
def load_real_meta(
    meta_path: str,
    year_lo: int,
    year_hi: int,
    real_artist_field: str,
    slugify_real: bool,
) -> pd.DataFrame:
    meta = pd.read_csv(meta_path, low_memory=False)
    required = ["image_n", "Year", real_artist_field]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Real meta missing columns: {missing}")

    meta = meta.copy()
    meta["Year"] = pd.to_numeric(meta["Year"], errors="coerce")
    meta["image_n"] = meta["image_n"].map(normalize_image_n)
    meta[real_artist_field] = meta[real_artist_field].astype(str)

    meta = meta.dropna(subset=["image_n", "Year", real_artist_field])
    meta["Year"] = meta["Year"].astype(int)
    meta = meta[(meta["Year"] >= year_lo) & (meta["Year"] <= year_hi)].copy()

    if slugify_real:
        meta["artist"] = meta[real_artist_field].map(slugify_artist)
    else:
        meta["artist"] = meta[real_artist_field].astype(str)

    return meta.reset_index(drop=True)


def load_real_npz(npz_path: str) -> Tuple[np.ndarray, pd.Series]:
    data = np.load(npz_path, allow_pickle=True)
    if "ids" not in data.files or "embeddings" not in data.files:
        raise ValueError(f"Real NPZ must contain 'ids' and 'embeddings'. Found: {data.files}")
    ids = pd.Series(data["ids"]).map(normalize_image_n).astype(str)
    X = np.asarray(data["embeddings"], dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Real embeddings must be (N,D). Got {X.shape}")
    if len(ids) != X.shape[0]:
        raise ValueError(f"len(ids)={len(ids)} != rows(X)={X.shape[0]}")
    return X, ids


def merge_real(meta: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    emb_df = pd.DataFrame({"image_n": ids, "_emb_row": np.arange(len(ids), dtype=np.int64)})
    merged = meta.merge(emb_df, on="image_n", how="inner")
    if merged.empty:
        raise ValueError("No rows after merging real meta with embeddings by image_n.")
    return merged.reset_index(drop=True)


# ----------------------------
# Task 1: within vs between (normal CI)
# ----------------------------
def mean_within_distance(Xa: np.ndarray) -> Optional[float]:
    if Xa.shape[0] < 2:
        return None
    D = cosine_distances(Xa)
    tri = D[np.triu_indices(D.shape[0], k=1)]
    return float(tri.mean()) if tri.size else None


def mean_between_distance(Xa: np.ndarray, Xb: np.ndarray) -> Optional[float]:
    if Xa.size == 0 or Xb.size == 0:
        return None
    D = cosine_distances(Xa, Xb)
    return float(D.mean())


def sample_rows(X: np.ndarray, idx: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if idx.size <= k:
        return X[idx]
    pick = rng.choice(idx, size=k, replace=False)
    return X[pick]


def compute_within_between_all_numbers(
    df_sim: pd.DataFrame,
    X_sim: np.ndarray,
    setting: str,
    numbers: List[int],
    round_min: int,
    round_max: int,
    max_per_artist_round: int,
    max_other_per_round: int,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df_sim.copy()
    df = df[(df["setting"] == setting) & (df["number"].isin(numbers))]
    df = df[(df["round"] >= round_min) & (df["round"] <= round_max)]
    if df.empty:
        return pd.DataFrame()

    out = []
    for (num, r), dfg in df.groupby(["number", "round"]):
        for artist, ga in dfg.groupby("artist"):
            rows_a = ga["_orig_row"].to_numpy(dtype=np.int64)
            Xa = sample_rows(X_sim, rows_a, max_per_artist_round, rng)
            within = mean_within_distance(Xa)

            other_rows = dfg.loc[dfg["artist"] != artist, "_orig_row"].to_numpy(dtype=np.int64)
            Xb = sample_rows(X_sim, other_rows, max_other_per_round, rng) if other_rows.size else np.empty((0, X_sim.shape[1]), dtype=np.float32)
            between = mean_between_distance(Xa, Xb)

            if within is None or between is None:
                continue

            out.append({
                "setting": setting,
                "number": int(num),
                "round": int(r),
                "artist": artist,
                "n_artist_round": int(rows_a.size),
                "within_mean": float(within),
                "between_mean": float(between),
                "between_minus_within": float(between - within),
            })

    return pd.DataFrame(out)


def summarize_within_between_by_round(df_stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r, g in df_stats.groupby("round"):
        w = g["within_mean"].to_numpy(dtype=float)
        b = g["between_mean"].to_numpy(dtype=float)
        d = g["between_minus_within"].to_numpy(dtype=float)

        w_m, w_lo, w_hi = mean_ci_normal(w)
        b_m, b_lo, b_hi = mean_ci_normal(b)
        d_m, d_lo, d_hi = mean_ci_normal(d)

        rows.append({
            "round": int(r),
            "n_units": int(len(g)),  # units = (number, artist)
            "within_mean": w_m, "within_ci_lo": w_lo, "within_ci_hi": w_hi,
            "between_mean": b_m, "between_ci_lo": b_lo, "between_ci_hi": b_hi,
            "diff_mean": d_m, "diff_ci_lo": d_lo, "diff_ci_hi": d_hi,
        })
    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)


def plot_within_between_ci(summary: pd.DataFrame, outdir: Path) -> None:
    rounds = summary["round"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.errorbar(rounds, summary["within_mean"], yerr=[
        summary["within_mean"] - summary["within_ci_lo"],
        summary["within_ci_hi"] - summary["within_mean"],
    ], fmt="o-", capsize=4, label="within")

    ax.errorbar(rounds, summary["between_mean"], yerr=[
        summary["between_mean"] - summary["between_ci_lo"],
        summary["between_ci_hi"] - summary["between_mean"],
    ], fmt="o-", capsize=4, label="between")

    ax.set_xticks(rounds)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cosine distance")
    ax.tick_params(axis="y", labelrotation=0)
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig_within_between_mean_ci.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.errorbar(rounds, summary["diff_mean"], yerr=[
        summary["diff_mean"] - summary["diff_ci_lo"],
        summary["diff_ci_hi"] - summary["diff_mean"],
    ], fmt="o-", capsize=4)
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(rounds)
    ax.set_xlabel("Round")
    ax.set_ylabel("Between - Within")
    ax.tick_params(axis="y", labelrotation=0)
    ax.tick_params(axis="x", labelrotation=0)
    fig.tight_layout()
    fig.savefig(outdir / "fig_between_minus_within_mean_ci.png", dpi=180)
    plt.close(fig)


# ----------------------------
# Step 2: per-artist centroids (intersection-only)
# ----------------------------
def compute_real_centroids(
    dfR: pd.DataFrame,
    X_real: np.ndarray,
    years: List[int],
) -> Dict[Tuple[str, int], np.ndarray]:
    out: Dict[Tuple[str, int], np.ndarray] = {}
    for (artist, y), g in dfR.groupby(["artist", "Year"]):
        y = int(y)
        if y not in years:
            continue
        emb_rows = g["_emb_row"].to_numpy(dtype=np.int64)
        out[(str(artist), y)] = centroid_from_rows(X_real, emb_rows)
    return out


def compute_sim_centroids(
    dfS: pd.DataFrame,
    XS: np.ndarray,
    numbers: List[int],
    rounds: List[int],
) -> Dict[Tuple[int, str, int], np.ndarray]:
    out: Dict[Tuple[int, str, int], np.ndarray] = {}
    for (num, artist, r), g in dfS.groupby(["number", "artist", "round"]):
        num = int(num)
        r = int(r)
        if (num not in numbers) or (r not in rounds):
            continue
        idx = g["_i"].to_numpy(dtype=np.int64)
        out[(num, str(artist), r)] = centroid_from_rows(XS, idx)
    return out


def compute_artist_centroid_metrics(
    artists: List[str],
    years: List[int],
    year_to_round: Dict[int, int],
    realC: Dict[Tuple[str, int], np.ndarray],
    simC: Dict[Tuple[int, str, int], np.ndarray],
    numbers: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_num_rows = []
    pooled_rows = []
    diff_meta = []
    diff_vecs = []

    for artist in artists:
        for y in years:
            if y not in year_to_round:
                continue
            r = year_to_round[y]

            cR = realC.get((artist, y), None)
            if cR is None or not np.isfinite(cR).all():
                continue

            cS_list = []
            cosd_list = []
            l2_list = []

            for num in numbers:
                cS = simC.get((num, artist, r), None)
                if cS is None or not np.isfinite(cS).all():
                    continue

                cS_list.append(cS)
                cd = cosine_distance_vec(cS, cR)
                dv = cS - cR

                by_num_rows.append({
                    "number": num,
                    "artist": artist,
                    "year": y,
                    "round": r,
                    "cos_dist": cd,
                    "l2_norm": l2_norm(dv),
                })

                cosd_list.append(cd)
                l2_list.append(l2_norm(dv))

            if not cS_list:
                continue

            cS_pool = np.mean(np.stack(cS_list, axis=0), axis=0).astype(np.float32)
            cd_pool = cosine_distance_vec(cS_pool, cR)
            dv_pool = (cS_pool - cR).astype(np.float32)

            cosd_m, cosd_lo, cosd_hi = mean_ci_normal(np.array(cosd_list, dtype=float))
            l2_m, l2_lo, l2_hi = mean_ci_normal(np.array(l2_list, dtype=float))

            pooled_rows.append({
                "artist": artist,
                "year": y,
                "round": r,
                "cos_dist_pooled": cd_pool,
                "l2_norm_pooled": l2_norm(dv_pool),
                "cos_dist_mean_over_numbers": cosd_m,
                "cos_dist_ci_lo_over_numbers": cosd_lo,
                "cos_dist_ci_hi_over_numbers": cosd_hi,
                "l2_mean_over_numbers": l2_m,
                "l2_ci_lo_over_numbers": l2_lo,
                "l2_ci_hi_over_numbers": l2_hi,
                "n_numbers_used": int(len(cS_list)),
            })

            diff_meta.append({
                "artist": artist,
                "year": y,
                "round": r,
                "n_numbers_used": int(len(cS_list)),
            })
            diff_vecs.append(dv_pool)

    df_by = pd.DataFrame(by_num_rows).sort_values(["artist", "year", "number"]).reset_index(drop=True)
    df_pool = pd.DataFrame(pooled_rows).sort_values(["artist", "year"]).reset_index(drop=True)

    if diff_vecs:
        DV = np.stack(diff_vecs, axis=0).astype(np.float32)
        df_diff = wide_vectors_df(diff_meta, DV, prefix="diff_")
    else:
        df_diff = pd.DataFrame()

    return df_by, df_pool, df_diff


# ----------------------------
# Artist metric plots in GRID layout (centroids): shrink x labels
# ----------------------------
def plot_artist_metric_grid(
    df_by_number: pd.DataFrame,
    df_pooled: pd.DataFrame,
    artists: List[str],
    years: List[int],
    outpath: Path,
    metric: str,                 # "cos_dist" or "l2_norm"
    ncols: int,
    nrows: int,
    x_tick_fontsize: int = 8,
) -> None:
    """
    GRID of panels: nrows x ncols. Each panel = one artist.
    Individual runs get distinct colors and the overall trend is shown
    as a black line with mean ± 95% CI.
    """
    if not artists:
        return

    x = np.arange(len(years), dtype=int)
    xlabels = [str(y) for y in years]

    K = nrows * ncols
    artists_plot = artists[:K]

    numbers = sorted(df_by_number["number"].dropna().astype(int).unique().tolist()) if not df_by_number.empty else []
    cmap = plt.get_cmap("tab10" if len(numbers) <= 10 else "tab20")
    run_colors = {num: cmap(i % cmap.N) for i, num in enumerate(numbers)}

    per_artist_h = 2.0
    per_artist_w = 4.2
    fig_w = max(8.0, per_artist_w * ncols)
    fig_h = max(6.0, per_artist_h * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=True)
    axes = np.asarray(axes).reshape(nrows, ncols)

    legend_handles = [
        Line2D([0], [0], color=run_colors[num], linewidth=1.5, label=f"Run {num}")
        for num in numbers
    ]
    legend_handles.append(Line2D([0], [0], color="black", marker="o", linewidth=2.2, label="Overall trend (mean ± 95% CI)"))

    for i, artist in enumerate(artists_plot):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        gN = df_by_number[df_by_number["artist"] == artist]
        for num, gg in gN.groupby("number"):
            gg = gg.set_index("year").reindex(years)
            yv = gg[metric].to_numpy(dtype=float)
            ax.plot(x, yv, linewidth=1.2, alpha=0.35, color=run_colors.get(int(num)), label=f"Run {int(num)}")

        gP = df_pooled[df_pooled["artist"] == artist].set_index("year").reindex(years)
        if metric == "cos_dist":
            y_mean = gP["cos_dist_mean_over_numbers"].to_numpy(dtype=float)
            y_lo = gP["cos_dist_ci_lo_over_numbers"].to_numpy(dtype=float)
            y_hi = gP["cos_dist_ci_hi_over_numbers"].to_numpy(dtype=float)
        else:
            y_mean = gP["l2_mean_over_numbers"].to_numpy(dtype=float)
            y_lo = gP["l2_ci_lo_over_numbers"].to_numpy(dtype=float)
            y_hi = gP["l2_ci_hi_over_numbers"].to_numpy(dtype=float)

        ax.errorbar(
            x,
            y_mean,
            yerr=[y_mean - y_lo, y_hi - y_mean],
            fmt="o-",
            linewidth=2.2,
            color="black",
            capsize=2,
            alpha=0.95,
            label="Overall trend (mean ± 95% CI)",
        )

        if metric == "cos_dist":
            ax.set_ylim(0.0, 0.06)

        ax.text(0.02, 0.94, artist, transform=ax.transAxes, fontsize=9, va="top", ha="left")
        ax.tick_params(axis="y", labelrotation=0)
        ax.tick_params(axis="x", labelrotation=0)
        ax.grid(True, linewidth=0.3, alpha=0.35)

        if r != (nrows - 1):
            ax.set_xticklabels([])

    for j in range(len(artists_plot), nrows * ncols):
        rr = j // ncols
        cc = j % ncols
        axes[rr, cc].axis("off")

    for c in range(ncols):
        axb = axes[nrows - 1, c]
        if axb.axison:
            axb.set_xticks(x)
            axb.set_xticklabels(xlabels, fontsize=x_tick_fontsize)
            axb.set_xlabel("Year")

    if metric == "cos_dist":
        fig.text(0.01, 0.5, "Cosine distance", va="center", rotation=90)
    else:
        fig.text(0.01, 0.5, "L2 norm", va="center", rotation=90)

    fig.legend(handles=legend_handles, frameon=False, fontsize=8, ncol=max(1, len(legend_handles)), loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0.04, 0.02, 1.0, 0.95])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# ----------------------------
# Step 3: global centroid differences
# ----------------------------
def compute_global_centroids_real(dfR: pd.DataFrame, X_real: np.ndarray, years: List[int]) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for y in years:
        g = dfR[dfR["Year"] == y]
        if g.empty:
            continue
        emb_rows = g["_emb_row"].to_numpy(dtype=np.int64)
        out[int(y)] = centroid_from_rows(X_real, emb_rows)
    return out


def compute_global_centroids_sim(dfS: pd.DataFrame, XS: np.ndarray, numbers: List[int], rounds: List[int]) -> Dict[Tuple[int, int], np.ndarray]:
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for num in numbers:
        for r in rounds:
            g = dfS[(dfS["number"] == num) & (dfS["round"] == r)]
            if g.empty:
                continue
            idx = g["_i"].to_numpy(dtype=np.int64)
            out[(num, r)] = centroid_from_rows(XS, idx)
    return out


def compute_global_metrics(
    years: List[int],
    year_to_round: Dict[int, int],
    realG: Dict[int, np.ndarray],
    simG: Dict[Tuple[int, int], np.ndarray],
    numbers: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_num_rows = []
    pooled_rows = []

    for y in years:
        if y not in year_to_round:
            continue
        r = year_to_round[y]
        cR = realG.get(y, None)
        if cR is None or not np.isfinite(cR).all():
            continue

        cS_list = []
        cosd_list = []
        l2_list = []

        for num in numbers:
            cS = simG.get((num, r), None)
            if cS is None or not np.isfinite(cS).all():
                continue

            cd = cosine_distance_vec(cS, cR)
            dv = cS - cR

            by_num_rows.append({
                "number": num,
                "year": y,
                "round": r,
                "cos_dist": cd,
                "l2_norm": l2_norm(dv),
            })

            cS_list.append(cS)
            cosd_list.append(cd)
            l2_list.append(l2_norm(dv))

        if not cS_list:
            continue

        cS_pool = np.mean(np.stack(cS_list, axis=0), axis=0).astype(np.float32)
        cd_pool = cosine_distance_vec(cS_pool, cR)
        dv_pool = cS_pool - cR

        cosd_m, cosd_lo, cosd_hi = mean_ci_normal(np.array(cosd_list, dtype=float))
        l2_m, l2_lo, l2_hi = mean_ci_normal(np.array(l2_list, dtype=float))

        pooled_rows.append({
            "year": y,
            "round": r,
            "cos_dist_pooled": cd_pool,
            "l2_norm_pooled": l2_norm(dv_pool),
            "cos_dist_mean_over_numbers": cosd_m,
            "cos_dist_ci_lo_over_numbers": cosd_lo,
            "cos_dist_ci_hi_over_numbers": cosd_hi,
            "l2_mean_over_numbers": l2_m,
            "l2_ci_lo_over_numbers": l2_lo,
            "l2_ci_hi_over_numbers": l2_hi,
            "n_numbers_used": int(len(cS_list)),
        })

    df_by = pd.DataFrame(by_num_rows).sort_values(["year", "number"]).reset_index(drop=True)
    df_pool = pd.DataFrame(pooled_rows).sort_values(["year"]).reset_index(drop=True)
    return df_by, df_pool


def plot_global_metric(
    df_by_number: pd.DataFrame,
    df_pooled: pd.DataFrame,
    years: List[int],
    outpath: Path,
    metric: str,           # "cos_dist" or "l2_norm"
    ylabel: str,
) -> None:
    x = np.arange(len(years), dtype=int)
    numbers = sorted(df_by_number["number"].dropna().astype(int).unique().tolist()) if not df_by_number.empty else []
    cmap = plt.get_cmap("tab10" if len(numbers) <= 10 else "tab20")
    run_colors = {num: cmap(i % cmap.N) for i, num in enumerate(numbers)}

    fig, ax = plt.subplots(figsize=(11, 4))

    all_vals = []
    for num, g in df_by_number.groupby("number"):
        g = g.set_index("year").reindex(years)
        yv = g[metric].to_numpy(dtype=float)
        all_vals.append(yv)
        ax.plot(x, yv, linewidth=1.2, alpha=0.35, color=run_colors.get(int(num)), label=f"Run {int(num)}")

    gp = df_pooled.set_index("year").reindex(years)
    if metric == "cos_dist":
        y_mean = gp["cos_dist_mean_over_numbers"].to_numpy(dtype=float)
        y_lo = gp["cos_dist_ci_lo_over_numbers"].to_numpy(dtype=float)
        y_hi = gp["cos_dist_ci_hi_over_numbers"].to_numpy(dtype=float)
    else:
        y_mean = gp["l2_mean_over_numbers"].to_numpy(dtype=float)
        y_lo = gp["l2_ci_lo_over_numbers"].to_numpy(dtype=float)
        y_hi = gp["l2_ci_hi_over_numbers"].to_numpy(dtype=float)

    all_vals.extend([y_mean, y_lo, y_hi])
    ax.errorbar(
        x,
        y_mean,
        yerr=[y_mean - y_lo, y_hi - y_mean],
        fmt="o-",
        linewidth=2.4,
        color="black",
        capsize=3,
        alpha=0.95,
        label="Overall trend (mean ± 95% CI)",
    )

    ymin, ymax = compute_dynamic_ylim(all_vals, include_zero=False)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="y", labelrotation=0)
    ax.tick_params(axis="x", labelrotation=0)

    handles = [
        Line2D([0], [0], color=run_colors[num], linewidth=1.5, label=f"Run {num}")
        for num in numbers
    ]
    handles.append(Line2D([0], [0], color="black", marker="o", linewidth=2.4, label="Overall trend (mean ± 95% CI)"))
    ax.legend(handles=handles, frameon=False, ncol=max(1, len(handles)))

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# ----------------------------
# Task 4: artist-level shift vector comparison (GRID)
# ----------------------------
def compute_artist_shift_vectors_real(
    artists: List[str],
    years: List[int],
    realC: Dict[Tuple[str, int], np.ndarray],
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], np.ndarray]]:
    meta_rows = []
    vecs = []
    shift_map: Dict[Tuple[str, int], np.ndarray] = {}

    for artist in artists:
        for y0, y1 in zip(years[:-1], years[1:]):
            c0 = realC.get((artist, y0), None)
            c1 = realC.get((artist, y1), None)
            if c0 is None or c1 is None:
                continue
            if not (np.isfinite(c0).all() and np.isfinite(c1).all()):
                continue
            sv = (c1 - c0).astype(np.float32)
            meta_rows.append({"artist": artist, "year_from": y0, "year_to": y1})
            vecs.append(sv)
            shift_map[(artist, y1)] = sv

    df_vec = wide_vectors_df(meta_rows, np.stack(vecs, axis=0).astype(np.float32), prefix="shift_") if vecs else pd.DataFrame()
    return df_vec, shift_map


def compute_artist_shift_vectors_sim_by_number(
    artists: List[str],
    years: List[int],
    year_to_round: Dict[int, int],
    simC: Dict[Tuple[int, str, int], np.ndarray],
    numbers: List[int],
) -> Tuple[pd.DataFrame, Dict[Tuple[int, str, int], np.ndarray]]:
    meta_rows = []
    vecs = []
    shift_map: Dict[Tuple[int, str, int], np.ndarray] = {}

    for artist in artists:
        for y0, y1 in zip(years[:-1], years[1:]):
            if y0 not in year_to_round or y1 not in year_to_round:
                continue
            r0 = year_to_round[y0]
            r1 = year_to_round[y1]
            for num in numbers:
                c0 = simC.get((num, artist, r0), None)
                c1 = simC.get((num, artist, r1), None)
                if c0 is None or c1 is None:
                    continue
                if not (np.isfinite(c0).all() and np.isfinite(c1).all()):
                    continue
                sv = (c1 - c0).astype(np.float32)
                meta_rows.append({"number": num, "artist": artist, "round_from": r0, "round_to": r1, "year_from": y0, "year_to": y1})
                vecs.append(sv)
                shift_map[(num, artist, r1)] = sv

    df_vec = wide_vectors_df(meta_rows, np.stack(vecs, axis=0).astype(np.float32), prefix="shift_") if vecs else pd.DataFrame()
    return df_vec, shift_map


def compute_artist_shift_metrics(
    artists: List[str],
    years: List[int],
    year_to_round: Dict[int, int],
    real_shift: Dict[Tuple[str, int], np.ndarray],
    sim_shift: Dict[Tuple[int, str, int], np.ndarray],
    numbers: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    SHIFT comparison uses COSINE DISTANCE (1 - cosine similarity).
    """
    by_rows = []
    pooled_rows = []
    sim_pool_meta, sim_pool_vecs = [], []
    diff_pool_meta, diff_pool_vecs = [], []

    for artist in artists:
        for y0, y1 in zip(years[:-1], years[1:]):
            if y1 not in year_to_round:
                continue
            r1 = year_to_round[y1]
            r0 = year_to_round.get(y0, None)
            if r0 is None:
                continue

            rs = real_shift.get((artist, y1), None)
            if rs is None or not np.isfinite(rs).all():
                continue

            sims = []
            cosd_vals = []
            l2_vals = []

            for num in numbers:
                ss = sim_shift.get((num, artist, r1), None)
                if ss is None or not np.isfinite(ss).all():
                    continue

                sims.append(ss)

                cd = cosine_distance_vec(ss, rs)
                dv = ss - rs

                by_rows.append({
                    "number": num,
                    "artist": artist,
                    "year_from": y0, "year_to": y1,
                    "round_from": r0, "round_to": r1,
                    "cos_dist": cd,
                    "l2_norm": l2_norm(dv),
                })

                cosd_vals.append(cd)
                l2_vals.append(l2_norm(dv))

            if not sims:
                continue

            ss_pool = np.mean(np.stack(sims, axis=0), axis=0).astype(np.float32)
            cd_pool = cosine_distance_vec(ss_pool, rs)
            dv_pool = (ss_pool - rs).astype(np.float32)

            cosd_m, cosd_lo, cosd_hi = mean_ci_normal(np.array(cosd_vals, dtype=float))
            l2_m, l2_lo, l2_hi = mean_ci_normal(np.array(l2_vals, dtype=float))

            pooled_rows.append({
                "artist": artist,
                "year_from": y0, "year_to": y1,
                "round_from": r0, "round_to": r1,
                "cos_dist_pooled": cd_pool,
                "l2_norm_pooled": l2_norm(dv_pool),
                "cos_dist_mean_over_numbers": cosd_m,
                "cos_dist_ci_lo_over_numbers": cosd_lo,
                "cos_dist_ci_hi_over_numbers": cosd_hi,
                "l2_mean_over_numbers": l2_m,
                "l2_ci_lo_over_numbers": l2_lo,
                "l2_ci_hi_over_numbers": l2_hi,
                "n_numbers_used": int(len(sims)),
            })

            sim_pool_meta.append({"artist": artist, "year_from": y0, "year_to": y1, "round_from": r0, "round_to": r1, "n_numbers_used": int(len(sims))})
            sim_pool_vecs.append(ss_pool)
            diff_pool_meta.append({"artist": artist, "year_from": y0, "year_to": y1, "round_from": r0, "round_to": r1, "n_numbers_used": int(len(sims))})
            diff_pool_vecs.append(dv_pool)

    df_by = pd.DataFrame(by_rows).sort_values(["artist", "year_to", "number"]).reset_index(drop=True)
    df_pool = pd.DataFrame(pooled_rows).sort_values(["artist", "year_to"]).reset_index(drop=True)

    df_sim_pool = wide_vectors_df(sim_pool_meta, np.stack(sim_pool_vecs, axis=0).astype(np.float32), prefix="shift_") if sim_pool_vecs else pd.DataFrame()
    df_diff_pool = wide_vectors_df(diff_pool_meta, np.stack(diff_pool_vecs, axis=0).astype(np.float32), prefix="diff_") if diff_pool_vecs else pd.DataFrame()
    return df_by, df_pool, df_sim_pool, df_diff_pool


def plot_artist_shift_cossim_grid(
    df_by: pd.DataFrame,
    df_pool: pd.DataFrame,
    artists: List[str],
    steps: List[Tuple[int, int]],
    outpath: Path,
    ncols: int,
    nrows: int,
    x_tick_fontsize: int = 8,
) -> None:
    """
    GRID of panels: nrows x ncols. Each panel = one artist.
    Individual runs get distinct colors and the overall trend is shown
    as a black line with mean ± 95% CI.
    """
    if not artists:
        return

    x = np.arange(len(steps), dtype=int)
    xlabels = [f"{a}-{b}" for (a, b) in steps]

    K = nrows * ncols
    artists_plot = artists[:K]

    numbers = sorted(df_by["number"].dropna().astype(int).unique().tolist()) if not df_by.empty else []
    cmap = plt.get_cmap("tab10" if len(numbers) <= 10 else "tab20")
    run_colors = {num: cmap(i % cmap.N) for i, num in enumerate(numbers)}

    per_artist_h = 2.0
    per_artist_w = 4.2
    fig_w = max(8.0, per_artist_w * ncols)
    fig_h = max(6.0, per_artist_h * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(nrows, ncols)

    legend_handles = [
        Line2D([0], [0], color=run_colors[num], linewidth=1.5, label=f"Run {num}")
        for num in numbers
    ]
    legend_handles.append(Line2D([0], [0], color="black", marker="o", linewidth=2.2, label="Overall trend (mean ± 95% CI)"))

    for i, artist in enumerate(artists_plot):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        gN = df_by[df_by["artist"] == artist]
        panel_vals = []
        for num, gg in gN.groupby("number"):
            gg = gg.set_index(["year_from", "year_to"])
            yv = np.array([gg["cos_dist"].get((y0, y1), np.nan) for (y0, y1) in steps], dtype=float)
            panel_vals.append(yv)
            ax.plot(x, yv, linewidth=1.2, alpha=0.35, color=run_colors.get(int(num)), label=f"Run {int(num)}")

        gP = df_pool[df_pool["artist"] == artist].set_index(["year_from", "year_to"])
        y_mean = np.array([gP["cos_dist_mean_over_numbers"].get((y0, y1), np.nan) for (y0, y1) in steps], dtype=float)
        y_lo = np.array([gP["cos_dist_ci_lo_over_numbers"].get((y0, y1), np.nan) for (y0, y1) in steps], dtype=float)
        y_hi = np.array([gP["cos_dist_ci_hi_over_numbers"].get((y0, y1), np.nan) for (y0, y1) in steps], dtype=float)
        panel_vals.extend([y_mean, y_lo, y_hi])

        ax.errorbar(
            x,
            y_mean,
            yerr=[y_mean - y_lo, y_hi - y_mean],
            fmt="o-",
            linewidth=2.2,
            color="black",
            capsize=2,
            alpha=0.95,
            label="Overall trend (mean ± 95% CI)",
        )

        ax.set_ylim(0.6, 1.4)
        ax.text(0.02, 0.94, artist, transform=ax.transAxes, fontsize=9, va="top", ha="left")
        ax.tick_params(axis="y", labelrotation=0)
        ax.tick_params(axis="x", labelrotation=0)
        ax.grid(True, linewidth=0.3, alpha=0.35)

        if r != (nrows - 1):
            ax.set_xticklabels([])

    for j in range(len(artists_plot), nrows * ncols):
        rr = j // ncols
        cc = j % ncols
        axes[rr, cc].axis("off")

    for c in range(ncols):
        axb = axes[nrows - 1, c]
        if axb.axison:
            axb.set_xticks(x)
            axb.set_xticklabels(xlabels, fontsize=x_tick_fontsize)
            axb.set_xlabel("Year step")

    fig.text(0.01, 0.5, "Cosine distance", va="center", rotation=90)
    fig.legend(handles=legend_handles, frameon=False, fontsize=8, ncol=max(1, len(legend_handles)), loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0.04, 0.02, 1.0, 0.95])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# ----------------------------
# Task 5: global shift vector comparison
# ----------------------------
def compute_global_shift_vectors_real(
    years: List[int],
    realG: Dict[int, np.ndarray],
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    meta_rows = []
    vecs = []
    shift_map: Dict[int, np.ndarray] = {}

    for y0, y1 in zip(years[:-1], years[1:]):
        c0 = realG.get(y0, None)
        c1 = realG.get(y1, None)
        if c0 is None or c1 is None:
            continue
        if not (np.isfinite(c0).all() and np.isfinite(c1).all()):
            continue
        sv = (c1 - c0).astype(np.float32)
        meta_rows.append({"year_from": y0, "year_to": y1})
        vecs.append(sv)
        shift_map[y1] = sv

    df_vec = wide_vectors_df(meta_rows, np.stack(vecs, axis=0).astype(np.float32), prefix="shift_") if vecs else pd.DataFrame()
    return df_vec, shift_map


def compute_global_shift_vectors_sim_by_number(
    years: List[int],
    year_to_round: Dict[int, int],
    simG: Dict[Tuple[int, int], np.ndarray],
    numbers: List[int],
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], np.ndarray]]:
    meta_rows = []
    vecs = []
    shift_map: Dict[Tuple[int, int], np.ndarray] = {}

    for y0, y1 in zip(years[:-1], years[1:]):
        if y0 not in year_to_round or y1 not in year_to_round:
            continue
        r0 = year_to_round[y0]
        r1 = year_to_round[y1]
        for num in numbers:
            c0 = simG.get((num, r0), None)
            c1 = simG.get((num, r1), None)
            if c0 is None or c1 is None:
                continue
            if not (np.isfinite(c0).all() and np.isfinite(c1).all()):
                continue
            sv = (c1 - c0).astype(np.float32)
            meta_rows.append({"number": num, "year_from": y0, "year_to": y1, "round_from": r0, "round_to": r1})
            vecs.append(sv)
            shift_map[(num, r1)] = sv

    df_vec = wide_vectors_df(meta_rows, np.stack(vecs, axis=0).astype(np.float32), prefix="shift_") if vecs else pd.DataFrame()
    return df_vec, shift_map


def compute_global_shift_metrics(
    steps: List[Tuple[int, int]],
    year_to_round: Dict[int, int],
    real_shift: Dict[int, np.ndarray],
    sim_shift: Dict[Tuple[int, int], np.ndarray],
    numbers: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    SHIFT comparison uses COSINE DISTANCE (1 - cosine similarity).
    """
    by_rows = []
    pooled_rows = []
    sim_pool_meta, sim_pool_vecs = [], []
    diff_pool_meta, diff_pool_vecs = [], []

    for (y0, y1) in steps:
        if y1 not in year_to_round or y0 not in year_to_round:
            continue
        r0 = year_to_round[y0]
        r1 = year_to_round[y1]

        rs = real_shift.get(y1, None)
        if rs is None or not np.isfinite(rs).all():
            continue

        sims = []
        cosd_vals = []
        l2_vals = []

        for num in numbers:
            ss = sim_shift.get((num, r1), None)
            if ss is None or not np.isfinite(ss).all():
                continue

            sims.append(ss)

            cd = cosine_distance_vec(ss, rs)
            dv = ss - rs

            by_rows.append({
                "number": num,
                "year_from": y0, "year_to": y1,
                "round_from": r0, "round_to": r1,
                "cos_dist": cd,
                "l2_norm": l2_norm(dv),
            })

            cosd_vals.append(cd)
            l2_vals.append(l2_norm(dv))

        if not sims:
            continue

        ss_pool = np.mean(np.stack(sims, axis=0), axis=0).astype(np.float32)
        cd_pool = cosine_distance_vec(ss_pool, rs)
        dv_pool = (ss_pool - rs).astype(np.float32)

        cosd_m, cosd_lo, cosd_hi = mean_ci_normal(np.array(cosd_vals, dtype=float))
        l2_m, l2_lo, l2_hi = mean_ci_normal(np.array(l2_vals, dtype=float))

        pooled_rows.append({
            "year_from": y0, "year_to": y1,
            "round_from": r0, "round_to": r1,
            "cos_dist_pooled": cd_pool,
            "l2_norm_pooled": l2_norm(dv_pool),
            "cos_dist_mean_over_numbers": cosd_m,
            "cos_dist_ci_lo_over_numbers": cosd_lo,
            "cos_dist_ci_hi_over_numbers": cosd_hi,
            "l2_mean_over_numbers": l2_m,
            "l2_ci_lo_over_numbers": l2_lo,
            "l2_ci_hi_over_numbers": l2_hi,
            "n_numbers_used": int(len(sims)),
        })

        sim_pool_meta.append({"year_from": y0, "year_to": y1, "round_from": r0, "round_to": r1, "n_numbers_used": int(len(sims))})
        sim_pool_vecs.append(ss_pool)
        diff_pool_meta.append({"year_from": y0, "year_to": y1, "round_from": r0, "round_to": r1, "n_numbers_used": int(len(sims))})
        diff_pool_vecs.append(dv_pool)

    df_by = pd.DataFrame(by_rows).sort_values(["year_to", "number"]).reset_index(drop=True)
    df_pool = pd.DataFrame(pooled_rows).sort_values(["year_to"]).reset_index(drop=True)
    df_sim_pool = wide_vectors_df(sim_pool_meta, np.stack(sim_pool_vecs, axis=0).astype(np.float32), prefix="shift_") if sim_pool_vecs else pd.DataFrame()
    df_diff_pool = wide_vectors_df(diff_pool_meta, np.stack(diff_pool_vecs, axis=0).astype(np.float32), prefix="diff_") if diff_pool_vecs else pd.DataFrame()
    return df_by, df_pool, df_sim_pool, df_diff_pool


def plot_global_shift_cossim(
    df_by: pd.DataFrame,
    df_pool: pd.DataFrame,
    steps: List[Tuple[int, int]],
    outpath: Path,
    x_tick_fontsize: int = 9,
) -> None:
    """
    Global SHIFT plot uses COSINE DISTANCE (1 - cosine similarity).
    Individual runs get distinct colors and the overall trend is shown
    as a black line with mean ± 95% CI.
    """
    x = np.arange(len(steps), dtype=int)
    xlabels = [f"{a}-{b}" for (a, b) in steps]

    numbers = sorted(df_by["number"].dropna().astype(int).unique().tolist()) if not df_by.empty else []
    cmap = plt.get_cmap("tab10" if len(numbers) <= 10 else "tab20")
    run_colors = {num: cmap(i % cmap.N) for i, num in enumerate(numbers)}

    fig, ax = plt.subplots(figsize=(11, 4))

    all_vals = []
    for num, g in df_by.groupby("number"):
        g = g.set_index(["year_from", "year_to"])
        yv = np.array([g["cos_dist"].get((y0, y1), np.nan) for (y0, y1) in steps], dtype=float)
        all_vals.append(yv)
        ax.plot(x, yv, linewidth=1.2, alpha=0.35, color=run_colors.get(int(num)), label=f"Run {int(num)}")

    gp = df_pool.set_index(["year_from", "year_to"])
    y_mean = np.array([gp["cos_dist_mean_over_numbers"].get((y0, y1), np.nan) for (y0, y1) in steps], dtype=float)
    y_lo = np.array([gp["cos_dist_ci_lo_over_numbers"].get((y0, y1), np.nan) for (y0, y1) in steps], dtype=float)
    y_hi = np.array([gp["cos_dist_ci_hi_over_numbers"].get((y0, y1), np.nan) for (y0, y1) in steps], dtype=float)
    all_vals.extend([y_mean, y_lo, y_hi])

    ax.errorbar(
        x,
        y_mean,
        yerr=[y_mean - y_lo, y_hi - y_mean],
        fmt="o-",
        linewidth=2.4,
        color="black",
        capsize=3,
        alpha=0.95,
        label="Overall trend (mean ± 95% CI)",
    )

    ymin, ymax = compute_dynamic_ylim(all_vals, include_zero=False)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=x_tick_fontsize)
    ax.set_xlabel("Year step")
    ax.set_ylabel("Cosine distance")
    ax.tick_params(axis="y", labelrotation=0)
    ax.tick_params(axis="x", labelrotation=0)

    handles = [
        Line2D([0], [0], color=run_colors[num], linewidth=1.5, label=f"Run {num}")
        for num in numbers
    ]
    handles.append(Line2D([0], [0], color="black", marker="o", linewidth=2.4, label="Overall trend (mean ± 95% CI)"))
    ax.legend(handles=handles, frameon=False, ncol=max(1, len(handles)))

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# ----------------------------
# Multi-condition helpers
# ----------------------------
COMPARISON_CONDITIONS = [
    "edge01_smallmove",
    "edge01_largemove",
    "edge05_smallmove",
    "edge05_largemove",
]


def resolve_sim_path(args) -> str:
    return args.sim_file


def make_year_to_round(year_lo: int, year_hi: int, round_min: int, round_max: int) -> Dict[int, int]:
    year_to_round: Dict[int, int] = {}
    for r in range(round_min, round_max + 1):
        y = year_lo + (r - round_min)
        if year_lo <= y <= year_hi:
            year_to_round[y] = r
    return year_to_round




def get_all_numbers_for_setting(df: pd.DataFrame, setting: str) -> List[int]:
    vals = df.loc[df["setting"] == setting, "number"].dropna().astype(int).unique().tolist()
    return sorted(vals)


def get_available_settings(df: pd.DataFrame) -> List[str]:
    return sorted(df["setting"].dropna().astype(str).unique().tolist())


def compute_dynamic_ylim(arrays: List[np.ndarray], include_zero: bool = False, pad_frac: float = 0.08) -> Tuple[float, float]:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    if not vals:
        return (0.0, 1.0)
    merged = np.concatenate(vals)
    ymin = float(np.min(merged))
    ymax = float(np.max(merged))
    if include_zero:
        ymin = min(ymin, 0.0)
        ymax = max(ymax, 0.0)
    if np.isclose(ymin, ymax):
        pad = 0.05 * max(1.0, abs(ymin))
    else:
        pad = pad_frac * (ymax - ymin)
    return (ymin - pad, ymax + pad)




def pretty_condition_label(cond: str) -> str:
    mapping = {
        "edge01_smallmove": "Sparse Network, Small Moving Probability",
        "edge01_largemove": "Sparse Network, Large Moving Probability",
        "edge05_smallmove": "Dense Network, Small Moving Probability",
        "edge05_largemove": "Dense Network, Large Moving Probability",
    }
    return mapping.get(cond, cond)


def plot_global_metric_panel(
    pooled_map: Dict[str, pd.DataFrame],
    years: List[int],
    outpath: Path,
    metric: str,
    ylabel: str,
    conditions: Optional[List[str]] = None,
) -> None:
    conditions = conditions or [c for c in COMPARISON_CONDITIONS if c in pooled_map]
    if not conditions:
        return

    x = np.arange(len(years), dtype=int)
    xlabels = [str(y) for y in years]
    cmap = plt.get_cmap("tab10" if len(conditions) <= 10 else "tab20")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    all_vals = []
    handles = []

    for i, cond in enumerate(conditions):
        color = cmap(i % cmap.N)
        gp = pooled_map[cond].set_index("year").reindex(years)
        if metric == "cos_dist":
            y_mean = gp["cos_dist_mean_over_numbers"].to_numpy(dtype=float)
            y_lo = gp["cos_dist_ci_lo_over_numbers"].to_numpy(dtype=float)
            y_hi = gp["cos_dist_ci_hi_over_numbers"].to_numpy(dtype=float)
        else:
            y_mean = gp["l2_mean_over_numbers"].to_numpy(dtype=float)
            y_lo = gp["l2_ci_lo_over_numbers"].to_numpy(dtype=float)
            y_hi = gp["l2_ci_hi_over_numbers"].to_numpy(dtype=float)
        all_vals.extend([y_mean, y_lo, y_hi])
        ax.errorbar(x, y_mean, yerr=[y_mean - y_lo, y_hi - y_mean], fmt="o-", linewidth=2.0, color=color, capsize=3, alpha=0.95)
        handles.append(Line2D([0], [0], color=color, marker="o", linewidth=2.0, label=pretty_condition_label(cond)))

    ymin, ymax = compute_dynamic_ylim(all_vals, include_zero=False)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="y", labelrotation=0)
    ax.tick_params(axis="x", labelrotation=0)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.legend(handles=handles, frameon=False, ncol=max(1, len(handles)))
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_global_shift_panel(
    pooled_map: Dict[str, pd.DataFrame],
    steps: List[Tuple[int, int]],
    outpath: Path,
    conditions: Optional[List[str]] = None,
) -> None:
    conditions = conditions or [c for c in COMPARISON_CONDITIONS if c in pooled_map]
    if not conditions:
        return

    x = np.arange(len(steps), dtype=int)
    xlabels = [f"{a}-{b}" for (a, b) in steps]
    cmap = plt.get_cmap("tab10" if len(conditions) <= 10 else "tab20")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    all_vals = []
    handles = []

    for i, cond in enumerate(conditions):
        color = cmap(i % cmap.N)
        gp = pooled_map[cond].set_index(["year_from", "year_to"])
        y_mean = np.array([gp["cos_dist_mean_over_numbers"].get(step, np.nan) for step in steps], dtype=float)
        y_lo = np.array([gp["cos_dist_ci_lo_over_numbers"].get(step, np.nan) for step in steps], dtype=float)
        y_hi = np.array([gp["cos_dist_ci_hi_over_numbers"].get(step, np.nan) for step in steps], dtype=float)
        all_vals.extend([y_mean, y_lo, y_hi])
        ax.errorbar(x, y_mean, yerr=[y_mean - y_lo, y_hi - y_mean], fmt="o-", linewidth=2.0, color=color, capsize=3, alpha=0.95)
        handles.append(Line2D([0], [0], color=color, marker="o", linewidth=2.0, label=pretty_condition_label(cond)))

    ymin, ymax = compute_dynamic_ylim(all_vals, include_zero=False)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Year step")
    ax.set_ylabel("Cosine distance")
    ax.tick_params(axis="y", labelrotation=0)
    ax.tick_params(axis="x", labelrotation=0)
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.legend(handles=handles, frameon=False, ncol=max(1, len(handles)))
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_within_between_panel(
    summary_map: Dict[str, pd.DataFrame],
    outpath: Path,
    conditions: Optional[List[str]] = None,
) -> None:
    conditions = conditions or [c for c in COMPARISON_CONDITIONS if c in summary_map]
    if not conditions:
        return

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 8), sharex=True)
    cmap = plt.get_cmap("tab10" if len(conditions) <= 10 else "tab20")
    all_within = []
    all_between = []
    handles = []

    for i, cond in enumerate(conditions):
        color = cmap(i % cmap.N)
        s = summary_map[cond].sort_values("round")
        x = s["round"].to_numpy(dtype=int)
        y_within = s["within_mean"].to_numpy(dtype=float)
        y_between = s["between_mean"].to_numpy(dtype=float)
        within_lo = s["within_ci_lo"].to_numpy(dtype=float)
        within_hi = s["within_ci_hi"].to_numpy(dtype=float)
        between_lo = s["between_ci_lo"].to_numpy(dtype=float)
        between_hi = s["between_ci_hi"].to_numpy(dtype=float)

        all_within.extend([y_within, within_lo, within_hi])
        all_between.extend([y_between, between_lo, between_hi])

        axes[0].errorbar(x, y_within, yerr=[y_within - within_lo, within_hi - y_within], fmt="o-", linewidth=2.0, color=color, capsize=3, alpha=0.95)
        axes[1].errorbar(x, y_between, yerr=[y_between - between_lo, between_hi - y_between], fmt="o-", linewidth=2.0, color=color, capsize=3, alpha=0.95)
        handles.append(Line2D([0], [0], color=color, marker="o", linewidth=2.0, label=pretty_condition_label(cond)))

    ymin0, ymax0 = compute_dynamic_ylim(all_within, include_zero=False)
    ymin1, ymax1 = compute_dynamic_ylim(all_between, include_zero=False)
    axes[0].set_ylim(ymin0, ymax0)
    axes[1].set_ylim(ymin1, ymax1)

    axes[0].set_ylabel("Within cosine distance")
    axes[1].set_ylabel("Between cosine distance")
    axes[1].set_xlabel("Round")
    axes[0].tick_params(axis="y", labelrotation=0)
    axes[1].tick_params(axis="y", labelrotation=0)
    axes[1].tick_params(axis="x", labelrotation=0)
    axes[0].grid(True, linewidth=0.3, alpha=0.35)
    axes[1].grid(True, linewidth=0.3, alpha=0.35)
    axes[0].legend(handles=handles, frameon=False, ncol=max(1, len(handles)))
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def run_single_setting(
    *,
    df_sim: pd.DataFrame,
    X_sim: np.ndarray,
    df_real: pd.DataFrame,
    X_real: np.ndarray,
    setting: str,
    year_lo: int,
    year_hi: int,
    round_min: int,
    round_max: int,
    max_per_artist_round: int,
    max_other_per_round: int,
    panel_cols: int,
    panel_rows: int,
    outdir: Path,
) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    numbers = get_all_numbers_for_setting(df_sim, setting)
    if not numbers:
        raise ValueError(f"No replicate runs found for setting={setting}.")

    dfS = df_sim[(df_sim["setting"] == setting) & (df_sim["number"].isin(numbers))].copy()
    dfS = dfS[(dfS["round"] >= round_min) & (dfS["round"] <= round_max)].copy()
    if dfS.empty:
        raise ValueError(f"No sim rows after filtering for setting={setting}.")
    dfS = dfS.reset_index(drop=True)
    dfS["_i"] = np.arange(len(dfS), dtype=np.int64)
    XS = X_sim[dfS["_orig_row"].to_numpy(dtype=np.int64)]

    sim_artists = set(dfS["artist"].unique().tolist())
    dfR_all = df_real[(df_real["Year"] >= year_lo) & (df_real["Year"] <= year_hi)].copy()
    dfR_pref = dfR_all[dfR_all["artist"].isin(sim_artists)].copy()
    intersect_artists = sorted(sim_artists.intersection(set(dfR_pref["artist"].unique().tolist())))
    if not intersect_artists:
        raise ValueError(
            f"No intersecting artists between SIM and REAL for setting={setting}."
        )

    dfS = dfS[dfS["artist"].isin(intersect_artists)].copy().reset_index(drop=True)
    dfS["_i"] = np.arange(len(dfS), dtype=np.int64)
    XS = X_sim[dfS["_orig_row"].to_numpy(dtype=np.int64)]
    dfR = dfR_pref[dfR_pref["artist"].isin(intersect_artists)].copy().reset_index(drop=True)

    years = list(range(year_lo, year_hi + 1))
    rounds = list(range(round_min, round_max + 1))
    steps = list(zip(years[:-1], years[1:]))
    year_to_round = make_year_to_round(year_lo, year_hi, round_min, round_max)

    df_wb = compute_within_between_all_numbers(
        df_sim=df_sim,
        X_sim=X_sim,
        setting=setting,
        numbers=numbers,
        round_min=round_min,
        round_max=round_max,
        max_per_artist_round=max_per_artist_round,
        max_other_per_round=max_other_per_round,
        seed=0,
    )
    wb_summary = pd.DataFrame()
    if not df_wb.empty:
        df_wb = df_wb[df_wb["artist"].isin(intersect_artists)].copy()
        df_wb.to_csv(outdir / "within_between_all_numbers.csv", index=False)
        wb_summary = summarize_within_between_by_round(df_wb)
        wb_summary.to_csv(outdir / "within_between_summary_by_round.csv", index=False)
        plot_within_between_ci(wb_summary, outdir)

    realC = compute_real_centroids(dfR, X_real, years)
    simC = compute_sim_centroids(dfS, XS, numbers, rounds)
    df_artist_by, df_artist_pool, df_artist_diff = compute_artist_centroid_metrics(
        artists=intersect_artists,
        years=years,
        year_to_round=year_to_round,
        realC=realC,
        simC=simC,
        numbers=numbers,
    )
    df_artist_by.to_csv(outdir / "artist_centroid_metrics_by_number.csv", index=False)
    df_artist_pool.to_csv(outdir / "artist_centroid_metrics_pooled.csv", index=False)
    if not df_artist_diff.empty:
        df_artist_diff.to_csv(outdir / "artist_centroid_diff_vectors_pooled.csv", index=False)

    plot_artist_metric_grid(
        df_artist_by, df_artist_pool, intersect_artists, years,
        outdir / "fig_artist_centroid_cosdist_grid.png",
        metric="cos_dist",
        ncols=panel_cols,
        nrows=panel_rows,
        x_tick_fontsize=8,
    )
    plot_artist_metric_grid(
        df_artist_by, df_artist_pool, intersect_artists, years,
        outdir / "fig_artist_centroid_l2norm_grid.png",
        metric="l2_norm",
        ncols=panel_cols,
        nrows=panel_rows,
        x_tick_fontsize=8,
    )

    realG = compute_global_centroids_real(dfR, X_real, years)
    simG = compute_global_centroids_sim(dfS, XS, numbers, rounds)
    df_global_by, df_global_pool = compute_global_metrics(years, year_to_round, realG, simG, numbers)
    df_global_by.to_csv(outdir / "global_centroid_metrics_by_number.csv", index=False)
    df_global_pool.to_csv(outdir / "global_centroid_metrics_pooled.csv", index=False)
    plot_global_metric(df_global_by, df_global_pool, years, outdir / "fig_global_centroid_cosdist.png", "cos_dist", "Cosine distance")
    plot_global_metric(df_global_by, df_global_pool, years, outdir / "fig_global_centroid_l2norm.png", "l2_norm", "L2 norm(sim - real)")

    df_real_shift_vec, real_shift_map = compute_artist_shift_vectors_real(intersect_artists, years, realC)
    df_real_shift_vec.to_csv(outdir / "artist_shift_vectors_real.csv", index=False)
    df_sim_shift_vec_by_num, sim_shift_map = compute_artist_shift_vectors_sim_by_number(
        intersect_artists, years, year_to_round, simC, numbers
    )
    df_sim_shift_vec_by_num.to_csv(outdir / "artist_shift_vectors_sim_by_number.csv", index=False)
    df_artist_shift_by, df_artist_shift_pool, df_artist_shift_sim_pool_vec, df_artist_shift_diff_pool_vec = compute_artist_shift_metrics(
        artists=intersect_artists,
        years=years,
        year_to_round=year_to_round,
        real_shift=real_shift_map,
        sim_shift=sim_shift_map,
        numbers=numbers,
    )
    df_artist_shift_by.to_csv(outdir / "artist_shift_metrics_by_number.csv", index=False)
    df_artist_shift_pool.to_csv(outdir / "artist_shift_metrics_pooled.csv", index=False)
    if not df_artist_shift_sim_pool_vec.empty:
        df_artist_shift_sim_pool_vec.to_csv(outdir / "artist_shift_vectors_sim_pooled.csv", index=False)
    if not df_artist_shift_diff_pool_vec.empty:
        df_artist_shift_diff_pool_vec.to_csv(outdir / "artist_shift_vectors_diff_pooled.csv", index=False)
    plot_artist_shift_cossim_grid(
        df_by=df_artist_shift_by,
        df_pool=df_artist_shift_pool,
        artists=intersect_artists,
        steps=steps,
        outpath=outdir / "fig_artist_shift_cossim_grid.png",
        ncols=panel_cols,
        nrows=panel_rows,
        x_tick_fontsize=8,
    )

    df_global_real_shift_vec, real_global_shift_map = compute_global_shift_vectors_real(years, realG)
    df_global_real_shift_vec.to_csv(outdir / "global_shift_vectors_real.csv", index=False)
    df_global_sim_shift_vec_by_num, sim_global_shift_map = compute_global_shift_vectors_sim_by_number(
        years, year_to_round, simG, numbers
    )
    df_global_sim_shift_vec_by_num.to_csv(outdir / "global_shift_vectors_sim_by_number.csv", index=False)
    df_global_shift_by, df_global_shift_pool, df_global_shift_sim_pool_vec, df_global_shift_diff_pool_vec = compute_global_shift_metrics(
        steps=steps,
        year_to_round=year_to_round,
        real_shift=real_global_shift_map,
        sim_shift=sim_global_shift_map,
        numbers=numbers,
    )
    df_global_shift_by.to_csv(outdir / "global_shift_metrics_by_number.csv", index=False)
    df_global_shift_pool.to_csv(outdir / "global_shift_metrics_pooled.csv", index=False)
    if not df_global_shift_sim_pool_vec.empty:
        df_global_shift_sim_pool_vec.to_csv(outdir / "global_shift_vectors_sim_pooled.csv", index=False)
    if not df_global_shift_diff_pool_vec.empty:
        df_global_shift_diff_pool_vec.to_csv(outdir / "global_shift_vectors_diff_pooled.csv", index=False)
    plot_global_shift_cossim(
        df_by=df_global_shift_by,
        df_pool=df_global_shift_pool,
        steps=steps,
        outpath=outdir / "fig_global_shift_cossim.png",
        x_tick_fontsize=9,
    )

    pd.DataFrame({"artist": intersect_artists}).to_csv(outdir / "artists_intersection_used.csv", index=False)
    K = int(panel_cols) * int(panel_rows)
    pd.DataFrame({"artist_plotted": intersect_artists[:K]}).to_csv(outdir / "artists_plotted_in_grid.csv", index=False)

    return {
        "setting": setting,
        "years": years,
        "steps": steps,
        "artists": intersect_artists,
        "numbers": numbers,
        "wb_summary": wb_summary,
        "global_pool": df_global_pool,
        "global_shift_pool": df_global_shift_pool,
    }

# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-file", required=True,
                    help="Path to the simulation NPZ file to use, e.g. /mnt/data/paris_simulation_style.npz")
    ap.add_argument("--real-meta", required=True)
    ap.add_argument("--real-npz", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--mode", choices=["validation", "comparison", "all"], default="all")
    ap.add_argument("--setting", default="validation")
    ap.add_argument("--comparison-conditions", nargs="+", default=COMPARISON_CONDITIONS)

    ap.add_argument("--year-lo", type=int, default=1501)
    ap.add_argument("--year-hi", type=int, default=1510)
    ap.add_argument("--round-min", type=int, default=0)
    ap.add_argument("--round-max", type=int, default=9)

    ap.add_argument("--max-per-artist-round", type=int, default=200)
    ap.add_argument("--max-other-per-round", type=int, default=800)
    ap.add_argument("--embed-mode", choices=["squeeze_last", "flatten"], default="squeeze_last")
    ap.add_argument("--real-artist-field", default="Artist_name")
    ap.add_argument("--slugify-real-artists", action="store_true")
    ap.add_argument("--panel-cols", type=int, default=3)
    ap.add_argument("--panel-rows", type=int, default=6)
    args = ap.parse_args()

    sim_path = resolve_sim_path(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_sim, X_sim = load_sim_npz(sim_path, embed_mode=args.embed_mode)
    meta_real = load_real_meta(
        args.real_meta,
        year_lo=args.year_lo,
        year_hi=args.year_hi,
        real_artist_field=args.real_artist_field,
        slugify_real=args.slugify_real_artists,
    )
    X_real, ids_real = load_real_npz(args.real_npz)
    df_real = merge_real(meta_real, ids_real)

    available_settings = get_available_settings(df_sim)
    print(f"Available settings in sim file: {available_settings}")

    run_validation = args.mode in {"validation", "all"}
    run_comparison = args.mode in {"comparison", "all"}
    results = {}

    if run_validation:
        if args.setting in available_settings:
            validation_out = outdir / "validation"
            results["validation"] = run_single_setting(
                df_sim=df_sim,
                X_sim=X_sim,
                df_real=df_real,
                X_real=X_real,
                setting=args.setting,
                year_lo=args.year_lo,
                year_hi=args.year_hi,
                round_min=args.round_min,
                round_max=args.round_max,
                max_per_artist_round=args.max_per_artist_round,
                max_other_per_round=args.max_other_per_round,
                panel_cols=args.panel_cols,
                panel_rows=args.panel_rows,
                outdir=validation_out,
            )
        elif args.mode == "validation":
            raise ValueError(
                f"Requested validation setting='{args.setting}' was not found. "
                f"Available settings are: {available_settings}"
            )
        else:
            print(
                f"Skipping validation because setting='{args.setting}' was not found. "
                f"Available settings are: {available_settings}"
            )

    if run_comparison:
        comp_conditions = [cond for cond in args.comparison_conditions if cond in available_settings]
        missing_conditions = [cond for cond in args.comparison_conditions if cond not in available_settings]
        if missing_conditions:
            print(f"Skipping comparison settings not present in sim file: {missing_conditions}")
        if not comp_conditions:
            raise ValueError(
                "None of the requested comparison conditions were found in the sim file. "
                f"Requested: {args.comparison_conditions}. Available: {available_settings}"
            )

        comp_root = outdir / "comparison"
        comp_root.mkdir(parents=True, exist_ok=True)
        wb_map = {}
        global_pool_map = {}
        global_shift_pool_map = {}
        summary_rows = []
        for cond in comp_conditions:
            cond_out = comp_root / cond
            res = run_single_setting(
                df_sim=df_sim,
                X_sim=X_sim,
                df_real=df_real,
                X_real=X_real,
                setting=cond,
                year_lo=args.year_lo,
                year_hi=args.year_hi,
                round_min=args.round_min,
                round_max=args.round_max,
                max_per_artist_round=args.max_per_artist_round,
                max_other_per_round=args.max_other_per_round,
                panel_cols=args.panel_cols,
                panel_rows=args.panel_rows,
                outdir=cond_out,
            )
            results[cond] = res
            if isinstance(res["wb_summary"], pd.DataFrame) and not res["wb_summary"].empty:
                wb_map[cond] = res["wb_summary"]
            global_pool_map[cond] = res["global_pool"]
            global_shift_pool_map[cond] = res["global_shift_pool"]
            summary_rows.append({
                "setting": cond,
                "n_intersect_artists": len(res["artists"]),
                "n_runs_used": len(res["numbers"]),
                "runs_used": ",".join(map(str, res["numbers"])),
            })

        years = list(range(args.year_lo, args.year_hi + 1))
        steps = list(zip(years[:-1], years[1:]))
        pd.DataFrame(summary_rows).to_csv(comp_root / "comparison_condition_summary.csv", index=False)
        if wb_map:
            plot_within_between_panel(wb_map, comp_root / "fig_within_between_comparison_overlay.png", conditions=comp_conditions)
        if global_pool_map:
            plot_global_metric_panel(global_pool_map, years, comp_root / "fig_global_centroid_cosdist_comparison_overlay.png", "cos_dist", "Cosine distance", conditions=comp_conditions)
            plot_global_metric_panel(global_pool_map, years, comp_root / "fig_global_centroid_l2norm_comparison_overlay.png", "l2_norm", "L2 norm(sim - real)", conditions=comp_conditions)
        if global_shift_pool_map:
            plot_global_shift_panel(global_shift_pool_map, steps, comp_root / "fig_global_shift_cosdist_comparison_overlay.png", conditions=comp_conditions)

    print("DONE.")
    print(f"Sim NPZ used: {sim_path}")
    print(f"Wrote outputs to: {outdir}")
    print("Replicate selection: using all runs available within each setting.")


if __name__ == "__main__":
    main()
