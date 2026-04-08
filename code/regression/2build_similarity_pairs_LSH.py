import json
import numpy as np
import faiss
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ---------------------------------------------------------------------
# PATHS & PARAMS
# ---------------------------------------------------------------------
npz_path = "/home/wangyd/Projects/macs_thesis/yangyu/artwork_data/artwork_style_embeddings.npz"
meta_path = "/home/wangyd/Projects/macs_thesis/yangyu/artist_data/artwork_data_merged.csv"
json_path = "/home/wangyd/Projects/macs_thesis/yangyu/artist_demographics/demographic_location.json"
output_path = "/home/wangyd/Projects/macs_thesis/yangyu/artwork_data/artwork_similarity_pairs_50.parquet"

similarity_threshold = 0.9
year_diff_threshold = 50  # NEW: only keep pairs within 50 years
n_bits = 512
k = 1000
batch_size = 1024
flush_every = 20000

# ---------------------------------------------------------------------
# 1. LOAD METADATA + JSON, FILTER VALID PAINTINGS
# ---------------------------------------------------------------------
# CSV
meta = pd.read_csv(meta_path)

# make Year numeric
meta["Year"] = pd.to_numeric(meta["Year"], errors="coerce")

# JSON with demographic info
with open(json_path, "r") as f:
    demo = json.load(f)

json_artist_keys = set(demo.keys())

def csv_artist_to_slug(x: str) -> str:
    """
    Your CSV has things like 'en/chaim-goldberg'.
    JSON has 'chaim-goldberg'.
    So: lowercase -> split('/') -> take last part.
    """
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    if "/" in x:
        x = x.split("/")[-1]
    return x

# normalize artist
meta["artist_slug"] = meta["Artist_name"].map(csv_artist_to_slug)

# FILTER 1: year >= 1400
# FILTER 2: artist in JSON
meta_filtered = meta[
    (meta["Year"] >= 1400) &
    (meta["artist_slug"].isin(json_artist_keys))
].copy()

# this is your painting id in the CSV
valid_painting_ids = set()
for i in meta_filtered["image_n"].tolist():
    if not pd.isna(i):
        valid_painting_ids.add(int(i))

print(f"CSV total rows: {len(meta)}")
print(f"after year>=1400 & artist-in-JSON: {len(meta_filtered)}")
print(f"distinct valid painting ids: {len(valid_painting_ids)}")

# build id -> year ONLY from filtered rows
id_to_year = {
    int(row["image_n"]): int(row["Year"])
    for _, row in meta_filtered.iterrows()
    if pd.notna(row["image_n"]) and pd.notna(row["Year"])
}

print(f"built year map for {len(id_to_year)} filtered paintings")

# ---------------------------------------------------------------------
# 2. LOAD EMBEDDINGS AND FILTER TO VALID PAINTINGS
# ---------------------------------------------------------------------
data = np.load(npz_path, allow_pickle=True)
ids = data["ids"]                      # should match image_n
embeddings = data["embeddings"].astype("float32")

# keep only embeddings whose id is in valid_painting_ids
keep_mask = np.array([id_ in valid_painting_ids for id_ in ids])

filtered_ids = ids[keep_mask]
filtered_embeddings = embeddings[keep_mask]

d = filtered_embeddings.shape[1]
n = filtered_embeddings.shape[0]
assert d == 768, f"Expected 768-D embeddings, got {d}"

print(f"embeddings total in NPZ: {len(ids)}")
print(f"embeddings after metadata+json filter: {n}")

# ---------------------------------------------------------------------
# 3. BUILD FAISS INDEX ON FILTERED EMBEDDINGS
# ---------------------------------------------------------------------
faiss.normalize_L2(filtered_embeddings)

res = faiss.StandardGpuResources()
cpu_index = faiss.IndexLSH(d, n_bits)
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

gpu_index.add(filtered_embeddings)
print("Index built and vectors added to GPU.")

# ---------------------------------------------------------------------
# 5. PREP PARQUET WRITER (moved up)
# ---------------------------------------------------------------------
schema = pa.schema([
    pa.field("src", pa.int64()),
    pa.field("dst", pa.int64()),
    pa.field("sim", pa.float32()),
    pa.field("src_year", pa.int32()),
    pa.field("dst_year", pa.int32()),
])

def write_batch(writer, rows):
    if not rows:
        return
    src_col = pa.array([r[0] for r in rows], type=pa.int64())
    dst_col = pa.array([r[1] for r in rows], type=pa.int64())
    sim_col = pa.array([np.float32(r[2]) for r in rows], type=pa.float32())
    src_year_col = pa.array([r[3] for r in rows], type=pa.int32())
    dst_year_col = pa.array([r[4] for r in rows], type=pa.int32())
    table = pa.Table.from_arrays(
        [src_col, dst_col, sim_col, src_year_col, dst_year_col],
        schema=schema
    )
    writer.write_table(table)

writer = pq.ParquetWriter(output_path, schema)

# ---------------------------------------------------------------------
# 4. SEARCH AND COLLECT (src_year > dst_year AND within 50 years)
# ---------------------------------------------------------------------
results = []
filtered_by_year = 0

for i in tqdm(range(0, n, batch_size), desc="Searching (LSH Approx)"):
    end = min(i + batch_size, n)
    batch_vecs = filtered_embeddings[i:end]          # (B, d), already L2-normalized

    # 1) get candidates from LSH
    sim_lsh, idx = gpu_index.search(batch_vecs, k)

    # 2) re-score with exact cosine
    for bi, (lsh_sims, indices) in enumerate(zip(sim_lsh, idx)):
        src_id = filtered_ids[i + bi]
        src_year = id_to_year.get(src_id, None)
        if src_year is None:
            continue

        qvec = batch_vecs[bi]   # this is normalized

        # drop invalids and self
        mask = (indices != -1) & (indices != (i + bi))

        for dst_idx in indices[mask]:
            dst_id = filtered_ids[dst_idx]
            dst_year = id_to_year.get(dst_id, None)
            if dst_year is None:
                continue

            # 2a) exact cosine in [-1, 1]
            dst_vec = filtered_embeddings[dst_idx]
            cos_sim = float(np.dot(qvec, dst_vec))

            # your threshold is now on true cosine
            if cos_sim < similarity_threshold:
                continue

            # direction constraint AND year difference constraint (NEW)
            year_diff = abs(src_year - dst_year)
            if src_year > dst_year and year_diff <= year_diff_threshold:
                results.append(
                    (int(src_id), int(dst_id), cos_sim, int(src_year), int(dst_year))
                )

                if len(results) >= flush_every:
                    write_batch(writer, results)
                    results = []
            elif year_diff > year_diff_threshold:
                filtered_by_year += 1

# flush remaining
if results:
    write_batch(writer, results)

writer.close()

print(f"Pairs filtered out by year difference > {year_diff_threshold}: {filtered_by_year}")
print(f"Saved similarity pairs (sim ≥ {similarity_threshold}, src_year > dst_year, within {year_diff_threshold} years, artist in JSON, year ≥ 1400) to:")
print(f"   {output_path}")

tbl = pq.read_table(output_path)
print("Parquet rows:", tbl.num_rows, "\nExamples:\n", tbl.to_pandas().sample(10, random_state = 42))
