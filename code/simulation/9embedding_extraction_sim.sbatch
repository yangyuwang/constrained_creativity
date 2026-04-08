#!/bin/bash
#SBATCH --job-name=scflow_posthoc_all
#SBATCH --partition=gpu
#SBATCH --account=pi-jevans
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=16 
#SBATCH --mem=64G 
#SBATCH --time=30:00:00 
#SBATCH --output=logs/9scflow_posthoc.out 
#SBATCH --error=logs/9scflow_posthoc.err 
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=wangyd@rcc.uchicago.edu 

module load python/anaconda-2022.05 cuda/12.2
source activate gallery_gpt

export CUDA_VISIBLE_DEVICES=0

export HF_HOME=/home/wangyd/Projects/macs_thesis/yangyu/tmp/hf_cache
export HF_HUB_CACHE=/home/wangyd/Projects/macs_thesis/yangyu/tmp/hf_cache

# ---- PATHS ----
SCFLOW_INFER_SCRIPT="yangyu/SCFlow/inference.py"
SCFLOW_CONFIG="yangyu/SCFlow/configs/inference.yaml"
SCFLOW_CKPT="yangyu/ckpts/scflow_last.ckpt"
UNCLIP_CKPT="yangyu/ckpts/sd21-unclip-l.ckpt"

EMBED_FRACTION=0.4
EMBED_SEED=42

echo "[$(date)] Starting post-hoc SCFlow embeddings (all images) over simulations for paris..."

PARENT_DIR="/home/wangyd/Projects/macs_thesis/yangyu/simulation_data/paris"
SCFLOW_OUTPUT_ROOT="/home/wangyd/Projects/macs_thesis/data/sim_embeddings/paris"

python code/simulation/9embedding_extraction_sim.py \
  --parent_dir "$PARENT_DIR" \
  --scflow_inference_script "$SCFLOW_INFER_SCRIPT" \
  --scflow_config "$SCFLOW_CONFIG" \
  --scflow_ckpt "$SCFLOW_CKPT" \
  --scflow_unclip_ckpt "$UNCLIP_CKPT" \
  --scflow_output_root "$SCFLOW_OUTPUT_ROOT" \
  --embedding_fraction "$EMBED_FRACTION" \
  --embedding_seed "$EMBED_SEED"

echo "[$(date)] Starting post-hoc SCFlow embeddings (all images) over simulations for shanghai..."

PARENT_DIR="/home/wangyd/Projects/macs_thesis/yangyu/simulation_data/shanghai"
SCFLOW_OUTPUT_ROOT="/home/wangyd/Projects/macs_thesis/data/sim_embeddings/shanghai"

python code/simulation/9embedding_extraction_sim.py \
  --parent_dir "$PARENT_DIR" \
  --scflow_inference_script "$SCFLOW_INFER_SCRIPT" \
  --scflow_config "$SCFLOW_CONFIG" \
  --scflow_ckpt "$SCFLOW_CKPT" \
  --scflow_unclip_ckpt "$UNCLIP_CKPT" \
  --scflow_output_root "$SCFLOW_OUTPUT_ROOT" \
  --embedding_fraction "$EMBED_FRACTION" \
  --embedding_seed "$EMBED_SEED"

echo "[$(date)] Finished."
