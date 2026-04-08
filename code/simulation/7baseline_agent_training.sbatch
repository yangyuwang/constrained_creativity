#!/bin/bash
#SBATCH --job-name=pre_artist_training
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=logs/7agent_training.out
#SBATCH --error=logs/7agent_training.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wangyd@rcc.uchicago.edu

module load python/anaconda-2022.05
source activate gallery_gpt

python code/simulation/7baseline_agent_training.py \
  --nodes_csv /home/wangyd/Projects/macs_thesis/data/nodelist_key_paris.csv \
  --jsonl_path /home/wangyd/Projects/macs_thesis/yangyu/painting_content.jsonl \
  --images_dir /home/wangyd/Projects/macs_thesis/yangyu/artwork_images \
  --metadata_csv /home/wangyd/Projects/macs_thesis/yangyu/artist_data/artwork_data_merged.csv \
  --pre_artist_jsonl_dir /home/wangyd/Projects/macs_thesis/yangyu/simulation_data/paris/pre_artist \
  --pre_artist_lora_dir /home/wangyd/Projects/macs_thesis/yangyu/simulation_data/paris/pre_artist_lora \
  --sd15_dir /home/wangyd/Projects/macs_thesis/yangyu/diffusion_model/sd_v15 \
  --resume_unet /home/wangyd/Projects/macs_thesis/yangyu/diffusion_model/lora_sd15_pre1900/lora_weight.pt \
  --max_year 1920 \
  --train_batch_size 8 \
  --num_train_epochs 10 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --mixed_precision fp16 \
  --save_steps 2000 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 500 \
  --skip_existing

python code/simulation/7baseline_agent_training.py \
  --nodes_csv /home/wangyd/Projects/macs_thesis/data/nodelist_key_shanghai.csv \
  --jsonl_path /home/wangyd/Projects/macs_thesis/yangyu/painting_content.jsonl \
  --images_dir /home/wangyd/Projects/macs_thesis/yangyu/artwork_images \
  --metadata_csv /home/wangyd/Projects/macs_thesis/yangyu/artist_data/artwork_data_merged.csv \
  --pre_artist_jsonl_dir /home/wangyd/Projects/macs_thesis/yangyu/simulation_data/shanghai/pre_artist \
  --pre_artist_lora_dir /home/wangyd/Projects/macs_thesis/yangyu/simulation_data/shanghai/pre_artist_lora \
  --sd15_dir /home/wangyd/Projects/macs_thesis/yangyu/diffusion_model/sd_v15 \
  --resume_unet /home/wangyd/Projects/macs_thesis/yangyu/diffusion_model/lora_sd15_pre1900/lora_weight.pt \
  --max_year 1920 \
  --train_batch_size 8 \
  --num_train_epochs 10 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --mixed_precision fp16 \
  --save_steps 2000 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 500 \
  --skip_existing
