#!/bin/bash
set -euo pipefail

ROOT=/scratch/afz225/res_stream_project
LOG_DIR="${ROOT}/slurm/logs"
mkdir -p "${LOG_DIR}"

BACKFILL_JOB_ID=$(sbatch --parsable "${ROOT}/slurm/backfill_incorrect_activations.sbatch")
POSTPROCESS_JOB_ID=$(sbatch --parsable --dependency=afterok:${BACKFILL_JOB_ID} "${ROOT}/slurm/run_postprocess.sbatch")

echo "Submitted incorrect-activation backfill array job: ${BACKFILL_JOB_ID}"
echo "Submitted dependent postprocess job: ${POSTPROCESS_JOB_ID}"
echo "Logs: ${LOG_DIR}"
echo "Outputs: ${ROOT}/outputs/gemma4_e2b"
