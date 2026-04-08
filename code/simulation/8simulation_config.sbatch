#!/bin/bash
#SBATCH --job-name=sim_array_json
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=logs/8sim_%a.out
#SBATCH --error=logs/8sim_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=wangyd@rcc.uchicago.edu
#SBATCH --array=0-39

module load python/anaconda-2022.05
source activate gallery_gpt

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module load jq 2>/dev/null || true
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq not found on PATH. Please module load jq or install it."
  exit 1
fi

# ----- CONSTANT PATHS -----
SD15_DIR=/home/wangyd/Projects/macs_thesis/yangyu/diffusion_model/sd_v15
BASELINE_LORA=/home/wangyd/Projects/macs_thesis/yangyu/diffusion_model/lora_sd15_pre1900/lora_weight.pt
PROMPTS_JSONL=/home/wangyd/Projects/macs_thesis/yangyu/painting_content.jsonl
CONFIG_JSON=/home/wangyd/Projects/macs_thesis/yangyu/simulation_data/8simulation_config.json

IDX=${SLURM_ARRAY_TASK_ID}

CONFIG=$(jq -c ".[$IDX]" "$CONFIG_JSON")
if [ "$CONFIG" = "null" ] || [ -z "$CONFIG" ]; then
  echo "No config object at index $IDX in $CONFIG_JSON"
  exit 1
fi

jq_field() {
  echo "$CONFIG" | jq -r "$1"
}

SIM_ID=$(jq_field '.sim_id')
NODES_CSV=$(jq_field '.nodes_csv')
EDGES_CSV=$(jq_field '.edges_csv')
RANDOM_EDGES=$(jq_field '.random_edges')
EDGE_PROB=$(jq_field '.edge_prob')
MOVE_LABEL=$(jq_field '.move_label')
MOVE_PROB=$(jq_field '.move_prob')
MOVE_LAT_SD=$(jq_field '.move_lat_sd')
MOVE_LON_SD=$(jq_field '.move_lon_sd')
MOVE_START_ROUND=$(jq_field '.move_start_round')
ROUNDS=$(jq_field '.rounds')
IMAGES_PER_ROUND=$(jq_field '.images_per_round')
SEED=$(jq_field '.seed')

if [[ "$SIM_ID" == paris_* ]]; then
  CASE_NAME="paris"
elif [[ "$SIM_ID" == shanghai_* ]]; then
  CASE_NAME="shanghai"
else
  echo "ERROR: cannot infer case from SIM_ID=$SIM_ID"
  echo "Expected SIM_ID to start with paris_ or shanghai_"
  exit 1
fi

PRE_LORA_ROOT=/home/wangyd/Projects/macs_thesis/yangyu/simulation_data/${CASE_NAME}/pre_artist_lora

SIM_DIR=/home/wangyd/Projects/macs_thesis/yangyu/simulation_data/${CASE_NAME}/sim_${SIM_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$SIM_DIR"

echo "[INFO] Loaded config index $IDX:"
echo "  CASE_NAME        = $CASE_NAME"
echo "  SIM_ID           = $SIM_ID"
echo "  NODES_CSV        = $NODES_CSV"
echo "  EDGES_CSV        = $EDGES_CSV"
echo "  RANDOM_EDGES     = $RANDOM_EDGES"
echo "  EDGE_PROB        = $EDGE_PROB"
echo "  MOVE_LABEL       = $MOVE_LABEL"
echo "  MOVE_PROB        = $MOVE_PROB"
echo "  MOVE_LAT_SD      = $MOVE_LAT_SD"
echo "  MOVE_LON_SD      = $MOVE_LON_SD"
echo "  MOVE_START_ROUND = $MOVE_START_ROUND"
echo "  ROUNDS           = $ROUNDS"
echo "  IMAGES_PER_ROUND = $IMAGES_PER_ROUND"
echo "  SEED             = $SEED"
echo "  PRE_LORA_ROOT    = $PRE_LORA_ROOT"
echo "  SIM_DIR          = $SIM_DIR"

NETWORK_ARGS=()
if [ "$RANDOM_EDGES" = "true" ]; then
  NETWORK_ARGS=(--random_edges --edge_prob "$EDGE_PROB")
else
  if [ "$EDGES_CSV" != "null" ] && [ -n "$EDGES_CSV" ]; then
    NETWORK_ARGS=(--edges_csv "$EDGES_CSV")
  fi
fi

python code/simulation/8simulation.py \
  --sd15_dir "$SD15_DIR" \
  --baseline_lora "$BASELINE_LORA" \
  --pre_lora_root "$PRE_LORA_ROOT" \
  --lora_root "$SIM_DIR/lora_states" \
  --prompts_jsonl "$PROMPTS_JSONL" \
  --nodes_csv "$NODES_CSV" \
  "${NETWORK_ARGS[@]}" \
  --output_root "$SIM_DIR/simulation_output" \
  --rounds "$ROUNDS" \
  --images_per_round "$IMAGES_PER_ROUND" \
  --train_samples_per_round 30 \
  --seed "$SEED" \
  --move_prob "$MOVE_PROB" \
  --move_lat_sd "$MOVE_LAT_SD" \
  --move_lon_sd "$MOVE_LON_SD" \
  --move_start_round "$MOVE_START_ROUND" \
  --clip_model_dir /home/wangyd/Projects/macs_thesis/yangyu/models/clip-vit-large-patch14 \
  --unclip_model_dir /home/wangyd/Projects/macs_thesis/yangyu/models/sd21-unclip-small \
  --device cuda