#!/bin/bash
#SBATCH --job-name=baseline_trainign
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=logs/6baseline_training.out
#SBATCH --error=logs/6baseline_training.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wangyd@rcc.uchicago.edu

module load python/anaconda-2022.05
source activate gallery_gpt

SD15_DIR="/home/wangyd/Projects/macs_thesis/yangyu/diffusion_model/sd_v15"
LORA_PATH="/home/wangyd/Projects/macs_thesis/yangyu/diffusion_model/lora_sd15_pre1900"

accelerate launch code/simulation/6train_lora_sd15_paintings.py \
  --sd15_dir "$SD15_DIR" \
  --jsonl_path /home/wangyd/Projects/macs_thesis/yangyu/painting_content.jsonl \
  --images_dir /home/wangyd/Projects/macs_thesis/yangyu/artwork_images \
  --metadata_csv /home/wangyd/Projects/macs_thesis/yangyu/artist_data/artwork_data_merged.csv \
  --year_max 1900 \
  --output_dir "$LORA_PATH" \
  --train_batch_size 8 \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --mixed_precision fp16 \
  --save_steps 2000 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 500

python code/simulation/6test_lora_sd15_paintings.py \
  --sd15_dir "$SD15_DIR" \
  --lora_unet_path "$LORA_PATH/lora_weight.pt" \
  --output_dir "$LORA_PATH/test/" \
  --seed 42 \
  --prompt \
    "a painting of a dog" \
    "a painting of a flower" \
    "a painting of a woman" \
    "a painting of a tower" \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --lora_rank 8 
