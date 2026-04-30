import csv
import json

import numpy as np

from geometric_complexity_scaling.cli import custom_math_pca_trajectory_main
from geometric_complexity_scaling.custom_math import configure_cache_dirs, load_gsm8k_like_rows
from geometric_complexity_scaling.plotting import plot_custom_pca_trajectory


def test_load_gsm8k_like_jsonl_with_custom_columns(tmp_path):
    path = tmp_path / "math.jsonl"
    rows = [
        {"problem": "A has 2 and gets 3. How many?", "gold": "#### 5", "id": "a"},
        {"problem": "10 minus 4?", "gold": "6", "id": "b"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    loaded = load_gsm8k_like_rows(
        data_file=path,
        question_column="problem",
        answer_column="gold",
        row_id_column="id",
    )

    assert loaded[0]["question"] == rows[0]["problem"]
    assert loaded[0]["answer"] == "#### 5"
    assert loaded[0]["row_id"] == "a"


def test_load_gsm8k_like_csv(tmp_path):
    path = tmp_path / "math.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerow({"question": "2+2?", "answer": "#### 4"})

    loaded = load_gsm8k_like_rows(data_file=path)

    assert loaded == [
        {
            "question": "2+2?",
            "answer": "#### 4",
            "row_id": 0,
            "source_row": {"question": "2+2?", "answer": "#### 4"},
        }
    ]


def test_configure_cache_dirs_sets_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)

    configure_cache_dirs(
        hf_home=tmp_path / "hf",
        datasets_cache=tmp_path / "datasets",
        transformers_cache=tmp_path / "transformers",
        mpl_cache=tmp_path / "mpl",
    )

    assert (tmp_path / "hf").is_dir()
    assert (tmp_path / "datasets").is_dir()
    assert (tmp_path / "transformers").is_dir()
    assert (tmp_path / "mpl").is_dir()


def test_custom_pca_trajectory_plot_synthetic(tmp_path):
    activation_dir = tmp_path / "activations"
    activation_dir.mkdir()
    activations = np.random.default_rng(0).normal(size=(12, 4, 8)).astype(np.float32)
    path = activation_dir / "custom_math_seed0_all_activations.npz"
    np.savez_compressed(
        path,
        activations=activations,
        correct=np.array([True] * 6 + [False] * 6),
        task=np.array("custom_math"),
        seed=np.array(0),
    )

    out = plot_custom_pca_trajectory(path, output_dir=tmp_path, max_per_group=3)

    assert out.exists()
    assert out.name == "custom_math_trajectory_pca_correctness.png"


def test_custom_math_cli_help(capsys):
    try:
        custom_math_pca_trajectory_main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "--data-file" in capsys.readouterr().out
