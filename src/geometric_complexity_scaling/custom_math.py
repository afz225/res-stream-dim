from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .inference import _extract_final_token_hidden_states
from .modeling import apply_gemma_chat_template, get_pad_token_id, load_gemma, parse_gemma_response
from .plotting import plot_custom_pca_trajectory
from .scoring import score_answer
from .tasks import SYSTEM_PROMPT, make_messages, parse_answer, target_for_row
from .utils import append_jsonl, ensure_dir, json_safe, seed_everything, tensor_to_numpy


def configure_cache_dirs(
    hf_home: str | Path | None = None,
    datasets_cache: str | Path | None = None,
    transformers_cache: str | Path | None = None,
    mpl_cache: str | Path | None = None,
) -> None:
    assignments = {
        "HF_HOME": hf_home,
        "HF_DATASETS_CACHE": datasets_cache,
        "TRANSFORMERS_CACHE": transformers_cache,
        "MPLCONFIGDIR": mpl_cache,
    }
    for name, value in assignments.items():
        if value is None:
            continue
        path = ensure_dir(value)
        os.environ[name] = str(path)


def load_gsm8k_like_rows(
    data_file: str | Path | None = None,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    split: str = "train",
    question_column: str = "question",
    answer_column: str = "answer",
    row_id_column: str | None = None,
) -> list[dict[str, Any]]:
    if data_file and dataset_name:
        raise ValueError("Use either --data-file or --dataset-name, not both.")
    if data_file:
        raw_rows = _load_local_rows(Path(data_file))
    elif dataset_name:
        raw_rows = _load_hf_rows(dataset_name, dataset_config, split)
    else:
        raise ValueError("Provide --data-file or --dataset-name.")

    rows = []
    for idx, row in enumerate(raw_rows):
        if question_column not in row:
            raise KeyError(f"Missing question column '{question_column}' in row {idx}")
        if answer_column not in row:
            raise KeyError(f"Missing answer column '{answer_column}' in row {idx}")
        row_id = row.get(row_id_column) if row_id_column else row.get("row_id", idx)
        rows.append(
            {
                "question": str(row[question_column]),
                "answer": str(row[answer_column]),
                "row_id": row_id,
                "source_row": dict(row),
            }
        )
    return rows


def run_custom_math_pca_trajectory(
    output_dir: str | Path,
    model_id: str,
    data_file: str | Path | None = None,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    split: str = "train",
    question_column: str = "question",
    answer_column: str = "answer",
    row_id_column: str | None = None,
    sample_size: int = 200,
    seed: int = 0,
    max_new_tokens: int = 256,
    system_prompt: str = SYSTEM_PROMPT,
    dtype: str = "auto",
    device_map: str = "auto",
    geometry_dtype: str = "float16",
    max_per_group: int = 75,
    normalize_layers: bool = True,
    overwrite: bool = False,
    hf_home: str | Path | None = None,
    datasets_cache: str | Path | None = None,
    transformers_cache: str | Path | None = None,
    mpl_cache: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    configure_cache_dirs(hf_home, datasets_cache, transformers_cache, mpl_cache)
    seed_everything(seed)
    output_dir = Path(output_dir)
    inference_dir = ensure_dir(output_dir / "inference")
    activation_dir = ensure_dir(output_dir / "activations")
    jsonl_path = inference_dir / f"custom_math_seed{seed}.jsonl"
    npz_path = activation_dir / f"custom_math_seed{seed}_all_activations.npz"
    if overwrite:
        jsonl_path.unlink(missing_ok=True)
        npz_path.unlink(missing_ok=True)
    elif jsonl_path.exists() and npz_path.exists():
        plot_path = plot_custom_pca_trajectory(
            npz_path,
            output_dir=output_dir,
            plot_name="custom_math",
            max_per_group=max_per_group,
            random_seed=seed,
            normalize_layers=normalize_layers,
        )
        return jsonl_path, npz_path, plot_path

    rows = load_gsm8k_like_rows(
        data_file=data_file,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        question_column=question_column,
        answer_column=answer_column,
        row_id_column=row_id_column,
    )
    rng = np.random.default_rng(seed)
    if rows:
        selected = rng.permutation(len(rows))[: min(sample_size, len(rows))]
        rows = [rows[int(i)] for i in selected]

    bundle = load_gemma(model_id=model_id, dtype=dtype, device_map=device_map)
    activations: list[np.ndarray] = []
    row_ids: list[Any] = []
    correct_values: list[bool] = []
    generated_lengths: list[int] = []

    for local_idx, row in enumerate(tqdm(rows, desc=f"custom_math seed={seed}")):
        math_row = {"question": row["question"], "answer": row["answer"]}
        messages = make_messages("math", math_row, system_prompt=system_prompt)
        prompt_text = apply_gemma_chat_template(bundle.processor, messages)
        inputs = bundle.processor(text=prompt_text, return_tensors="pt").to(bundle.device)
        input_len = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            outputs = bundle.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True,
                pad_token_id=get_pad_token_id(bundle.processor),
            )
        sequence = outputs.sequences[0]
        generated_ids = sequence[input_len:]
        response = bundle.processor.decode(generated_ids, skip_special_tokens=False)
        parsed_response = parse_gemma_response(bundle.processor, response)
        parsed_answer = parse_answer("math", response, parsed_response)
        target = target_for_row("math", math_row)
        score_details = score_answer("math", parsed_answer, target)
        correct = bool(score_details["correct"])
        generated_token_ids = [int(x) for x in generated_ids.detach().cpu().tolist()]
        record = {
            "task": "custom_math",
            "task_type": "custom_gsm8k_like_math",
            "dataset": dataset_name or str(data_file),
            "dataset_config": dataset_config,
            "split": split if dataset_name else None,
            "seed": seed,
            "row_id": row["row_id"],
            "local_idx": local_idx,
            "messages": messages,
            "prompt": prompt_text,
            "target": target,
            "raw_output": response,
            "parsed_response": json_safe(parsed_response),
            "parsed_answer": parsed_answer,
            "score_details": json_safe(score_details),
            "correct": correct,
            "input_len": input_len,
            "generated_len": len(generated_token_ids),
            "generated_token_ids": generated_token_ids,
            "source_row": json_safe(row["source_row"]),
            "created_unix": time.time(),
        }
        append_jsonl(jsonl_path, record)
        hidden_states = _extract_final_token_hidden_states(outputs)
        if hidden_states is None:
            continue
        activations.append(tensor_to_numpy(hidden_states, geometry_dtype))
        row_ids.append(row["row_id"])
        correct_values.append(correct)
        generated_lengths.append(len(generated_token_ids))

    if activations:
        stacked = np.stack(activations, axis=0)
    else:
        stacked = np.empty((0, 0, 0), dtype=np.float16 if geometry_dtype == "float16" else np.float32)
    correct_array = np.array(correct_values, dtype=bool)
    outcome_groups = np.where(correct_array, "correct", "incorrect")
    np.savez_compressed(
        npz_path,
        activations=stacked,
        row_ids=np.array(row_ids, dtype=object),
        correct=correct_array,
        outcome_group=outcome_groups,
        generated_lengths=np.array(generated_lengths),
        task=np.array("custom_math"),
        seed=np.array(seed),
        model_id=np.array(model_id),
    )
    plot_path = plot_custom_pca_trajectory(
        npz_path,
        output_dir=output_dir,
        plot_name="custom_math",
        max_per_group=max_per_group,
        random_seed=seed,
        normalize_layers=normalize_layers,
    )
    return jsonl_path, npz_path, plot_path


def _load_local_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return [dict(row) for row in payload["data"]]
        raise ValueError("JSON data must be a list of rows or an object with a 'data' list.")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported data file extension '{suffix}'. Use .jsonl, .json, or .csv.")


def _load_hf_rows(dataset_name: str, dataset_config: str | None, split: str) -> list[dict[str, Any]]:
    datasets = __import__("datasets")
    if dataset_config:
        ds = datasets.load_dataset(dataset_name, dataset_config, split=split)
    else:
        ds = datasets.load_dataset(dataset_name, split=split)
    return [dict(row) for row in ds]
