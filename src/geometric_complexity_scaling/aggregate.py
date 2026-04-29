from __future__ import annotations

from pathlib import Path

from .utils import ensure_dir, read_jsonl, require_import


def aggregate_inference(output_dir: str | Path) -> Path:
    pd = require_import("pandas", "pandas")
    output_dir = Path(output_dir)
    rows = []
    for path in sorted((output_dir / "inference").glob("*.jsonl")):
        records = read_jsonl(path)
        if not records:
            continue
        total = len(records)
        correct = sum(1 for r in records if r.get("correct"))
        first = records[0]
        rows.append(
            {
                "task": first["task"],
                "task_type": first["task_type"],
                "seed": first["seed"],
                "dataset": first["dataset"],
                "num_examples": total,
                "num_correct": correct,
                "accuracy": correct / total if total else 0.0,
            }
        )
    metrics_dir = ensure_dir(output_dir / "metrics")
    path = metrics_dir / "inference_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def combine_metric_csvs(output_dir: str | Path) -> Path:
    pd = require_import("pandas", "pandas")
    output_dir = Path(output_dir)
    frames = []
    for path in sorted((output_dir / "metrics").glob("*_metrics.csv")):
        if path.name in {"geometry_metrics_all.csv", "inference_summary.csv"}:
            continue
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    path = output_dir / "metrics" / "geometry_metrics_all.csv"
    combined.to_csv(path, index=False)
    return path
