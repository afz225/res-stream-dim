from pathlib import Path

import numpy as np
import pandas as pd

from geometric_complexity_scaling.aggregate import combine_metric_csvs
from geometric_complexity_scaling.geometry import compute_metrics_for_activation_file
from geometric_complexity_scaling.plotting import plot_results


def write_activation_file(path: Path, n_examples: int = 8):
    rng = np.random.default_rng(0)
    np.savez_compressed(
        path,
        activations=rng.normal(size=(n_examples, 4, 16)).astype("float32"),
        semantic_activations=rng.normal(size=(n_examples, 4, 6)).astype("float32"),
        row_ids=np.arange(n_examples),
        generated_lengths=np.ones(n_examples),
        task=np.array("fact"),
        seed=np.array(0),
        model_id=np.array("dummy"),
    )


def test_geometry_metrics_and_plots_on_synthetic_activations(tmp_path):
    activation_dir = tmp_path / "activations"
    activation_dir.mkdir()
    activation_path = activation_dir / "fact_seed0_correct_activations.npz"
    write_activation_file(activation_path)

    metrics_path = compute_metrics_for_activation_file(activation_path, tmp_path, pca_components=2, neighbors=3)
    metrics = pd.read_csv(metrics_path)
    metric_names = set(metrics["metric_name"])
    assert set(metrics["outcome_group"]) == {"correct"}
    assert "pca_residual_variance" in metric_names
    assert "intrinsic_dim_twonn" in metric_names
    assert "euclidean_layer_curvature" in metric_names
    assert "semantic_pullback_layer_curvature" in metric_names

    combine_metric_csvs(tmp_path)
    plots = plot_results(tmp_path)
    assert any("pca_residual_variance" in str(path) for path in plots)


def test_empty_activation_file_writes_summary_metrics(tmp_path):
    activation_dir = tmp_path / "activations"
    activation_dir.mkdir()
    activation_path = activation_dir / "fact_seed0_correct_activations.npz"
    write_activation_file(activation_path, n_examples=0)

    metrics_path = compute_metrics_for_activation_file(activation_path, tmp_path)
    metrics = pd.read_csv(metrics_path)
    assert set(metrics["metric_name"]) == {"num_examples_retained", "num_layers"}
    assert set(metrics["outcome_group"]) == {"correct"}
