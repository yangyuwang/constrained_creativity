#!/bin/bash
#SBATCH --job-name=sim_pairs
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/2build_sim_pairs.out
#SBATCH --error=logs/2build_sim_pairs.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wangyd@rcc.uchicago.edu

module load python/anaconda-2022.05
source activate gallery_gpt

python code/regression/2build_similarity_pairs_LSH.py
