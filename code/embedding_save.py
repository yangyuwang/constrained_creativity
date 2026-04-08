import os
import glob
import torch
import numpy as np
from tqdm import tqdm

base_dir = "yangyu/artwork_embeddings"
output_path_s = "yangyu/artwork_data/artwork_style_embeddings.npz"
output_path_c = "yangyu/artwork_data/artwork_content_embeddings.npz"


def collect_embeddings(base_dir, pattern):
    """Collect embeddings for either style or content."""
    ids, embeddings = [], []
    paths = sorted(glob.glob(os.path.join(base_dir, f"*/{pattern}")))

    for path in tqdm(paths, desc=f"Processing {pattern}", total=len(paths)):
        try:
            n = os.path.basename(os.path.dirname(path))
            tensor = torch.load(path, map_location="cpu")

            if isinstance(tensor, torch.Tensor):
                arr = tensor.view(-1).detach().numpy()
            else:
                continue

            ids.append(int(n))
            embeddings.append(arr)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")

    embeddings = np.stack(embeddings)
    ids = np.array(ids)
    print(f"[INFO] {pattern}: collected {len(ids)} items, shape {embeddings.shape}")
    return ids, embeddings


# ---- Style embeddings ----
ids_s, embeddings_s = collect_embeddings(base_dir, "clip_pred_s_tensor.pt")
np.savez_compressed(output_path_s, ids=ids_s, embeddings=embeddings_s)
print(f"[INFO] Saved style embeddings to {output_path_s}")

# ---- Content embeddings ----
ids_c, embeddings_c = collect_embeddings(base_dir, "clip_pred_c_tensor.pt")
np.savez_compressed(output_path_c, ids=ids_c, embeddings=embeddings_c)
print(f"[INFO] Saved content embeddings to {output_path_c}")
