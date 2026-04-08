python - <<'PY'
from huggingface_hub import snapshot_download
# This repo ID corresponds to the common OpenCLIP ViT-H-14 weights:
# (If your YAML uses a different version, adjust accordingly.)
repo_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
snapshot_download(repo_id, cache_dir="yangyu/tmp/hf_cache")
print("Downloaded to yangyu/tmp/hf_cache")
PY
