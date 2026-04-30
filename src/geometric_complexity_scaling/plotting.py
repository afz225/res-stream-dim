from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

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


def plot_trajectory_dr(
    output_dir: str | Path,
    max_per_group: int = 75,
    random_seed: int = 0,
    isomap_neighbors: int = 12,
    normalize_layers: bool = True,
) -> list[Path]:
    decomposition = require_import("sklearn.decomposition", "scikit-learn")
    manifold = require_import("sklearn.manifold", "scikit-learn")
    output_dir = Path(output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(output_dir / ".matplotlib")))
    require_import("matplotlib", "matplotlib")
    import matplotlib.pyplot as plt

    plot_dir = ensure_dir(output_dir / "plots")
    tasks = ["fact", "math", "sentiment"]
    written = []
    for method in ("pca", "isomap"):
        out = plot_dir / f"trajectory_{method}_correctness.png"
        _plot_trajectory_method(
            output_dir=output_dir,
            output_path=out,
            tasks=tasks,
            method=method,
            PCA=decomposition.PCA,
            Isomap=manifold.Isomap,
            plt=plt,
            max_per_group=max_per_group,
            random_seed=random_seed,
            isomap_neighbors=isomap_neighbors,
            normalize_layers=normalize_layers,
        )
        written.append(out)
    return written


def plot_custom_pca_trajectory(
    activation_path: str | Path,
    output_dir: str | Path,
    plot_name: str = "custom_math",
    max_per_group: int = 75,
    random_seed: int = 0,
    normalize_layers: bool = True,
) -> Path:
    decomposition = require_import("sklearn.decomposition", "scikit-learn")
    output_dir = Path(output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(output_dir / ".matplotlib")))
    require_import("matplotlib", "matplotlib")
    import matplotlib.pyplot as plt

    payload = np.load(activation_path, allow_pickle=True)
    activations = payload["activations"].astype(np.float32)
    labels = _labels_from_activation_payload(payload)
    trajectories, labels = _subsample_trajectories_by_label(activations, labels, max_per_group, random_seed)
    plot_dir = ensure_dir(output_dir / "plots")
    output_path = plot_dir / f"{plot_name}_trajectory_pca_correctness.png"
    _plot_single_pca_trajectory(
        trajectories=trajectories,
        labels=labels,
        output_path=output_path,
        title=f"{plot_name} residual-stream layer trajectories (PCA, layer-normalized)",
        PCA=decomposition.PCA,
        plt=plt,
        random_seed=random_seed,
        normalize_layers=normalize_layers,
    )
    return output_path


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


def _plot_trajectory_method(
    output_dir,
    output_path,
    tasks,
    method,
    PCA,
    Isomap,
    plt,
    max_per_group,
    random_seed,
    isomap_neighbors,
    normalize_layers,
):
    fig, axes = plt.subplots(1, len(tasks), figsize=(18, 5), squeeze=False)
    rng = np.random.default_rng(random_seed)
    colors = {"correct": "#14833b", "incorrect": "#c9342f"}
    for axis, task in zip(axes[0], tasks):
        trajectories, labels = _load_subsampled_task_trajectories(
            output_dir=output_dir,
            task=task,
            max_per_group=max_per_group,
            rng=rng,
        )
        if trajectories.size == 0:
            axis.set_title(f"{task}: no activations")
            axis.axis("off")
            continue
        if normalize_layers:
            trajectories = _layer_normalize(trajectories)
        n_examples, n_layers, hidden_dim = trajectories.shape
        states = trajectories.reshape(n_examples * n_layers, hidden_dim)
        if method == "pca":
            embedding = PCA(n_components=2, random_state=random_seed).fit_transform(states)
        elif method == "isomap":
            neighbors = max(2, min(isomap_neighbors, states.shape[0] - 1))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.manifold._isomap")
                warnings.filterwarnings("ignore", category=UserWarning, module="scipy.sparse._index")
                embedding = Isomap(n_neighbors=neighbors, n_components=2).fit_transform(states)
        else:
            raise ValueError(f"Unknown trajectory DR method: {method}")
        projected = embedding.reshape(n_examples, n_layers, 2)
        _draw_trajectories(axis, projected, labels, colors)
        axis.set_title(task)
        axis.set_xlabel(f"{method.upper()} 1")
        axis.set_ylabel(f"{method.upper()} 2")
    handles = [
        plt.Line2D([0], [0], color=colors["correct"], lw=2, label="correct"),
        plt.Line2D([0], [0], color=colors["incorrect"], lw=2, label="incorrect"),
        plt.Line2D([0], [0], marker="o", color="black", lw=0, markersize=5, label="start"),
        plt.Line2D([0], [0], marker="s", color="black", lw=0, markersize=5, label="end"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    title = f"Residual-stream layer trajectories ({method.upper()}, layer-normalized)"
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_single_pca_trajectory(
    trajectories,
    labels,
    output_path,
    title,
    PCA,
    plt,
    random_seed,
    normalize_layers,
):
    fig, axis = plt.subplots(1, 1, figsize=(7, 6))
    colors = {"correct": "#14833b", "incorrect": "#c9342f"}
    if trajectories.size == 0:
        axis.set_title("No activations")
        axis.axis("off")
    else:
        if normalize_layers:
            trajectories = _layer_normalize(trajectories)
        n_examples, n_layers, hidden_dim = trajectories.shape
        states = trajectories.reshape(n_examples * n_layers, hidden_dim)
        embedding = PCA(n_components=2, random_state=random_seed).fit_transform(states)
        projected = embedding.reshape(n_examples, n_layers, 2)
        _draw_trajectories(axis, projected, labels, colors)
        axis.set_title(title)
        axis.set_xlabel("PCA 1")
        axis.set_ylabel("PCA 2")
    handles = [
        plt.Line2D([0], [0], color=colors["correct"], lw=2, label="correct"),
        plt.Line2D([0], [0], color=colors["incorrect"], lw=2, label="incorrect"),
        plt.Line2D([0], [0], marker="o", color="black", lw=0, markersize=5, label="start"),
        plt.Line2D([0], [0], marker="s", color="black", lw=0, markersize=5, label="end"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _load_subsampled_task_trajectories(output_dir, task, max_per_group, rng):
    activation_dir = Path(output_dir) / "activations"
    trajectories = []
    labels = []
    for outcome_group in ("correct", "incorrect"):
        group_arrays = []
        for path in sorted(activation_dir.glob(f"{task}_seed*_{outcome_group}_activations.npz")):
            payload = np.load(path, allow_pickle=True)
            activations = payload["activations"].astype(np.float32)
            if activations.ndim == 3 and activations.shape[0] > 0:
                group_arrays.append(activations)
        if not group_arrays:
            continue
        group = np.concatenate(group_arrays, axis=0)
        count = min(max_per_group, group.shape[0])
        selected = rng.choice(group.shape[0], size=count, replace=False)
        trajectories.append(group[selected])
        labels.extend([outcome_group] * count)
    if not trajectories:
        return np.empty((0, 0, 0), dtype=np.float32), []
    return np.concatenate(trajectories, axis=0), labels


def _labels_from_activation_payload(payload) -> list[str]:
    if "outcome_group" in payload:
        values = payload["outcome_group"]
        if np.ndim(values) == 0:
            return [str(values)] * int(payload["activations"].shape[0])
        return [str(value) for value in values.tolist()]
    if "correct" in payload:
        return ["correct" if bool(value) else "incorrect" for value in payload["correct"].tolist()]
    return ["unknown"] * int(payload["activations"].shape[0])


def _subsample_trajectories_by_label(activations, labels, max_per_group, random_seed):
    if activations.ndim != 3 or activations.shape[0] == 0:
        return np.empty((0, 0, 0), dtype=np.float32), []
    rng = np.random.default_rng(random_seed)
    selected_arrays = []
    selected_labels = []
    label_array = np.array(labels)
    for label in ("correct", "incorrect"):
        indices = np.flatnonzero(label_array == label)
        if indices.size == 0:
            continue
        count = min(max_per_group, indices.size)
        chosen = rng.choice(indices, size=count, replace=False)
        selected_arrays.append(activations[chosen])
        selected_labels.extend([label] * count)
    if not selected_arrays:
        return np.empty((0, 0, 0), dtype=np.float32), []
    return np.concatenate(selected_arrays, axis=0), selected_labels


def _layer_normalize(trajectories):
    normalized = trajectories.astype(np.float32, copy=True)
    for layer_idx in range(normalized.shape[1]):
        layer = normalized[:, layer_idx, :]
        mean = layer.mean(axis=0, keepdims=True)
        std = layer.std(axis=0, keepdims=True)
        normalized[:, layer_idx, :] = (layer - mean) / np.maximum(std, 1e-6)
    return normalized


def _draw_trajectories(axis, projected, labels, colors):
    n_layers = projected.shape[1]
    alphas = np.linspace(0.25, 0.95, max(n_layers - 1, 1))
    for trajectory, label in zip(projected, labels):
        color = colors[label]
        for layer_idx in range(n_layers - 1):
            axis.plot(
                trajectory[layer_idx : layer_idx + 2, 0],
                trajectory[layer_idx : layer_idx + 2, 1],
                color=color,
                alpha=float(alphas[layer_idx]),
                linewidth=0.8,
            )
        axis.scatter(trajectory[0, 0], trajectory[0, 1], color=color, marker="o", s=12, alpha=0.9)
        axis.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, marker="s", s=14, alpha=0.9)
    axis.grid(alpha=0.2, linewidth=0.5)
