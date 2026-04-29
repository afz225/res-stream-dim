from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .utils import ensure_dir, require_import


def compute_metrics_for_activation_file(
    activation_path: str | Path,
    output_dir: str | Path,
    pca_components: int = 10,
    neighbors: int = 12,
    semantic_projection_dim: int = 256,
) -> Path:
    pd = require_import("pandas", "pandas")
    decomposition = require_import("sklearn.decomposition", "scikit-learn")
    manifold = require_import("sklearn.manifold", "scikit-learn")
    neighbors_mod = require_import("sklearn.neighbors", "scikit-learn")

    activation_path = Path(activation_path)
    payload = np.load(activation_path, allow_pickle=True)
    activations = payload["activations"].astype(np.float32)
    task = str(payload["task"])
    seed = int(payload["seed"])
    outcome_group = _outcome_group_for_payload(activation_path, payload)
    metrics_dir = ensure_dir(Path(output_dir) / "metrics")
    metrics_path = metrics_dir / f"{task}_seed{seed}_{outcome_group}_metrics.csv"

    rows: list[dict[str, Any]] = []
    n_examples = int(activations.shape[0])
    summary_rows = [
        _summary_row(task, seed, outcome_group, "num_examples_retained", n_examples),
        _summary_row(task, seed, outcome_group, "num_layers", int(activations.shape[1]) if activations.ndim == 3 else 0),
    ]
    if n_examples < 3:
        pd.DataFrame(summary_rows).to_csv(metrics_path, index=False)
        return metrics_path

    n_layers = int(activations.shape[1])
    for layer_idx in range(n_layers):
        x = activations[:, layer_idx, :]
        rows.extend(_linear_metrics(x, task, seed, outcome_group, layer_idx, pca_components, decomposition.PCA))
        rows.extend(_manifold_metrics(x, task, seed, outcome_group, layer_idx, neighbors, manifold))
        rows.append(
            _metric_row(task, seed, outcome_group, layer_idx, "intrinsic_dim_twonn", _two_nn_intrinsic_dim(x, neighbors_mod))
        )
        rows.append(
            _metric_row(task, seed, outcome_group, layer_idx, "local_id_mle_k10", _local_id_mle(x, neighbors_mod, k=10))
        )
        rows.append(
            _metric_row(task, seed, outcome_group, layer_idx, "local_id_mle_k20", _local_id_mle(x, neighbors_mod, k=20))
        )

    curvature_by_layer = _layer_path_curvature(activations)
    for layer_idx, value in enumerate(curvature_by_layer):
        rows.append(_metric_row(task, seed, outcome_group, layer_idx, "euclidean_layer_curvature", value))
    semantic_activations = payload["semantic_activations"].astype(np.float32) if "semantic_activations" in payload else None
    semantic_curvature_by_layer = (
        _layer_path_curvature(semantic_activations)
        if semantic_activations is not None and semantic_activations.size
        else _semantic_projection_curvature(activations, seed=seed, projection_dim=semantic_projection_dim)
    )
    for layer_idx, value in enumerate(semantic_curvature_by_layer):
        rows.append(_metric_row(task, seed, outcome_group, layer_idx, "semantic_pullback_layer_curvature", value))

    df = pd.concat([pd.DataFrame(rows), pd.DataFrame(summary_rows)], ignore_index=True)
    df.to_csv(metrics_path, index=False)
    return metrics_path


def _metric_row(task: str, seed: int, outcome_group: str, layer_idx: int, metric_name: str, value: float) -> dict[str, Any]:
    return {
        "task": task,
        "seed": seed,
        "outcome_group": outcome_group,
        "layer": layer_idx,
        "metric_name": metric_name,
        "metric_value": float(value) if np.isfinite(value) else np.nan,
    }


def _summary_row(task: str, seed: int, outcome_group: str, metric_name: str, value: float) -> dict[str, Any]:
    return {
        "task": task,
        "seed": seed,
        "outcome_group": outcome_group,
        "metric_name": metric_name,
        "metric_value": float(value) if np.isfinite(value) else np.nan,
    }


def _linear_metrics(x, task, seed, outcome_group, layer_idx, pca_components, PCA):
    n_components = max(1, min(pca_components, x.shape[0] - 1, x.shape[1]))
    pca = PCA(n_components=n_components, random_state=seed)
    z = pca.fit_transform(x)
    x_hat = pca.inverse_transform(z)
    recon_mse = np.mean((x - x_hat) ** 2)
    explained = float(np.sum(pca.explained_variance_ratio_))
    return [
        _metric_row(task, seed, outcome_group, layer_idx, "pca_explained_variance", explained),
        _metric_row(task, seed, outcome_group, layer_idx, "pca_reconstruction_mse", recon_mse),
        _metric_row(task, seed, outcome_group, layer_idx, "pca_residual_variance", 1.0 - explained),
    ]


def _manifold_metrics(x, task, seed, outcome_group, layer_idx, neighbors, manifold):
    n_neighbors = max(2, min(neighbors, x.shape[0] - 1))
    n_components = max(1, min(2, x.shape[0] - 1, x.shape[1]))
    rows = []
    try:
        iso = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
        iso.fit_transform(x)
        rows.append(_metric_row(task, seed, outcome_group, layer_idx, "isomap_reconstruction_error", iso.reconstruction_error()))
    except Exception:
        rows.append(_metric_row(task, seed, outcome_group, layer_idx, "isomap_reconstruction_error", np.nan))
    try:
        lle = manifold.LocallyLinearEmbedding(
            n_neighbors=n_neighbors,
            n_components=n_components,
            random_state=seed,
            method="standard",
        )
        lle.fit_transform(x)
        rows.append(_metric_row(task, seed, outcome_group, layer_idx, "lle_reconstruction_error", lle.reconstruction_error_))
    except Exception:
        rows.append(_metric_row(task, seed, outcome_group, layer_idx, "lle_reconstruction_error", np.nan))
    return rows


def _two_nn_intrinsic_dim(x, neighbors_mod) -> float:
    if x.shape[0] < 3:
        return np.nan
    nn = neighbors_mod.NearestNeighbors(n_neighbors=3, metric="euclidean")
    distances, _ = nn.fit(x).kneighbors(x)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    mask = (r1 > 0) & (r2 > r1)
    if not np.any(mask):
        return np.nan
    ratios = r2[mask] / r1[mask]
    denom = np.mean(np.log(ratios))
    if denom <= 0:
        return np.nan
    return 1.0 / denom


def _local_id_mle(x, neighbors_mod, k: int) -> float:
    if x.shape[0] <= k:
        return np.nan
    n_neighbors = min(k + 1, x.shape[0])
    distances, _ = neighbors_mod.NearestNeighbors(n_neighbors=n_neighbors).fit(x).kneighbors(x)
    neighbor_distances = distances[:, 1:]
    rk = neighbor_distances[:, -1]
    base_distances = neighbor_distances[:, :-1]
    usable = (base_distances > 0) & (rk[:, None] > base_distances)
    logs = np.full_like(base_distances, np.nan, dtype=np.float32)
    ratios = rk[:, None] / np.maximum(base_distances, 1e-12)
    logs[usable] = np.log(ratios[usable])
    denom = np.nanmean(logs, axis=1)
    ids = np.where(denom > 0, 1.0 / denom, np.nan)
    return float(np.nanmean(ids))


def _layer_path_curvature(activations: np.ndarray) -> np.ndarray:
    n_examples, n_layers, _ = activations.shape
    curvatures = np.full(n_layers, np.nan, dtype=np.float32)
    if n_layers < 3 or n_examples == 0:
        return curvatures
    for layer_idx in range(1, n_layers - 1):
        prev_vec = activations[:, layer_idx, :] - activations[:, layer_idx - 1, :]
        next_vec = activations[:, layer_idx + 1, :] - activations[:, layer_idx, :]
        diff = next_vec - prev_vec
        denom = np.linalg.norm(prev_vec, axis=1) ** 2 + 1e-8
        curvatures[layer_idx] = float(np.mean(np.linalg.norm(diff, axis=1) / denom))
    return curvatures


def _semantic_projection_curvature(
    activations: np.ndarray,
    seed: int,
    projection_dim: int,
) -> np.ndarray:
    if activations.ndim != 3 or activations.shape[0] == 0:
        return np.array([], dtype=np.float32)
    hidden_dim = activations.shape[-1]
    dim = max(2, min(projection_dim, hidden_dim))
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(hidden_dim, dim)).astype(np.float32)
    projected = np.einsum("nlh,hk->nlk", activations.astype(np.float32), projection)
    return _layer_path_curvature(projected)


def _outcome_group_for_payload(path: Path, payload) -> str:
    if "outcome_group" in payload:
        return str(payload["outcome_group"])
    name = path.name
    if "_incorrect_" in name:
        return "incorrect"
    if "_correct_" in name:
        return "correct"
    return "all"
