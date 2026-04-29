#!/bin/bash
#SBATCH --job-name=gcs-backfill
#SBATCH --array=0-8%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH -p nvidia
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/afz225/res_stream_project/slurm/logs/%x_%A_%a.out
#SBATCH --error=/scratch/afz225/res_stream_project/slurm/logs/%x_%A_%a.err

set -euo pipefail

ROOT=/scratch/afz225/res_stream_project
ENV_DIR="${ROOT}/.conda-env"
OUTPUT_DIR="${ROOT}/outputs/gemma4_e2b"
HF_CACHE=/scratch/afz225/.cache

TASKS=(fact sentiment math)
SEEDS=(0 1 2)

TASK_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
TASK="${TASKS[$TASK_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

mkdir -p "${ROOT}/slurm/logs" "${HF_CACHE}" "${OUTPUT_DIR}/cache/matplotlib"

export HF_HOME="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export TRANSFORMERS_CACHE="${HF_CACHE}/transformers"
if [[ -f "${HF_CACHE}/hf_token" ]]; then
  export HF_TOKEN="$(< "${HF_CACHE}/hf_token")"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
export MPLCONFIGDIR="${OUTPUT_DIR}/cache/matplotlib"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"

cd "${ROOT}"

echo "Backfilling incorrect activations task=${TASK} seed=${SEED} array_id=${SLURM_ARRAY_TASK_ID}"
"${ENV_DIR}/bin/python" scripts/backfill_incorrect_activations.py \
  --tasks "${TASK}" \
  --seeds "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  --dtype bfloat16 \
  --device-map auto \
  --overwrite
