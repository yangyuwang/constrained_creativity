#!/bin/bash
#SBATCH --job-name=image_style
#SBATCH --partition=gpu
#SBATCH --account=pi-jevans
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=logs/extraction.out
#SBATCH --error=logs/extraction.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wangyd@rcc.uchicago.edu

module load python/anaconda-2022.05 cuda/12.2
source activate gallery_gpt

export CUDA_VISIBLE_DEVICES=0

export HF_HOME=/home/wangyd/Projects/macs_thesis/yangyu/tmp/hf_cache
export HF_HUB_CACHE=/home/wangyd/Projects/macs_thesis/yangyu/tmp/hf_cache

CONFIG_PATH="SCFlow/configs/inference.yaml"
CHECKPOINT_PATH="yangyu/ckpts/scflow_last.ckpt"
UNCLIP_CHECKPOINT_PATH="yangyu/ckpts/sd21-unclip-l.ckpt"
OUTPUT_FOLDER_NAME="yangyu/artwork_embeddings"    
REVERSE_INFERENCE=true
IMAGE_MIX_PATH="yangyu/artwork_images"
SEED=42

echo "[$(date)] Starting reverse inference..."


python SCFlow/inference.py \
    --config "$CONFIG_PATH" \
    --resume_checkpoint "$CHECKPOINT_PATH" \
    --name "$OUTPUT_FOLDER_NAME" \
    --reverse_inference \
    --image_mix_path "$IMAGE_MIX_PATH" \
    --unclip_ckpt "$UNCLIP_CHECKPOINT_PATH" \
    --seed $SEED

python embedding_save.py

echo "[$(date)] Reverse inference finished."
