#!/bin/bash
#SBATCH --job-name=reg_geo
#SBATCH --partition=ssd
#SBATCH --account=ssd
#SBATCH --qos=ssd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=logs/4reg_geo.out
#SBATCH --error=logs/4reg_geo.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=wangyd@rcc.uchicago.edu

module load python/anaconda-2022.05 spark/3.3.2 graphviz
source activate gallery_gpt

export PYSPARK_DRIVER_PYTHON=python
export PYSPARK_DRIVER_PYTHON_OPTS=""

spark-submit \
  --total-executor-cores 16 \
  --executor-memory 64G \
  --driver-memory 64G \
  code/regression/4pair_regression_test.py
