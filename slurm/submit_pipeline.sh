#!/bin/bash
set -euo pipefail

ROOT=/scratch/afz225/res_stream_project
LOG_DIR="${ROOT}/slurm/logs"
mkdir -p "${LOG_DIR}"

INFERENCE_JOB_ID=$(sbatch --parsable "${ROOT}/slurm/run_inference_array.sbatch")
POSTPROCESS_JOB_ID=$(sbatch --parsable --dependency=afterok:${INFERENCE_JOB_ID} "${ROOT}/slurm/run_postprocess.sbatch")

echo "Submitted inference array job: ${INFERENCE_JOB_ID}"
echo "Submitted dependent postprocess job: ${POSTPROCESS_JOB_ID}"
echo "Logs: ${LOG_DIR}"
echo "Outputs: ${ROOT}/outputs/gemma4_e2b"
