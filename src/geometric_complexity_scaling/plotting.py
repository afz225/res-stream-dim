from __future__ import annotations

import os
from pathlib import Path

from .utils import ensure_dir, require_import


def plot_results(output_dir: str | Path) -> list[Path]:
    pd = require_import("pandas", "pandas")
    output_dir = Path(output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(output_dir / ".matplotlib")))
    require_import("matplotlib", "matplotlib")
    sns = require_import("seaborn", "seaborn")
    import matplotlib.pyplot as plt

    plot_dir = ensure_dir(output_dir / "plots")
    written: list[Path] = []

    inference_path = output_dir / "metrics" / "inference_summary.csv"
    if inference_path.exists():
        summary = pd.read_csv(inference_path)
        written.append(_barplot(summary, "accuracy", "Accuracy", plot_dir / "accuracy_by_task.png", sns, plt))
        written.append(
            _barplot(summary, "num_correct", "Correct examples retained", plot_dir / "correct_counts_by_task.png", sns, plt)
        )

    metrics_path = output_dir / "metrics" / "geometry_metrics_all.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        for metric_name in [
            "pca_residual_variance",
            "pca_reconstruction_mse",
            "isomap_reconstruction_error",
            "lle_reconstruction_error",
            "intrinsic_dim_twonn",
            "local_id_mle_k10",
            "local_id_mle_k20",
            "euclidean_layer_curvature",
            "semantic_pullback_layer_curvature",
        ]:
            subset = metrics[metrics["metric_name"] == metric_name].dropna(subset=["metric_value"])
            if not subset.empty and "layer" in subset:
                out = plot_dir / f"{metric_name}_by_layer.png"
                _lineplot(subset, metric_name, out, sns, plt)
                written.append(out)
    return written


def _barplot(df, y_col, title, output_path, sns, plt):
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="task", y=y_col, errorbar="sd")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _lineplot(df, title, output_path, sns, plt):
    plt.figure(figsize=(9, 5))
    hue = "task" if "outcome_group" not in df.columns else "task"
    style = "outcome_group" if "outcome_group" in df.columns else None
    sns.lineplot(
        data=df,
        x="layer",
        y="metric_value",
        hue=hue,
        style=style,
        errorbar="sd",
        estimator="mean",
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
