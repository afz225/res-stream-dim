from pathlib import Path


def test_inference_array_uses_four_way_parallelism():
    script = Path("slurm/run_inference_array.sbatch").read_text()
    assert "#SBATCH --array=0-8%4" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH -p nvidia" in script
    assert "TASK_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))" in script
    assert "SEED_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))" in script


def test_submit_pipeline_uses_afterok_dependency():
    script = Path("slurm/submit_pipeline.sh").read_text()
    assert "sbatch --parsable" in script
    assert "--dependency=afterok:${INFERENCE_JOB_ID}" in script
