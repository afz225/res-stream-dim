from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .inference import _project_with_unembedding_subset, _sample_unembedding_subset
from .modeling import load_gemma
from .tasks import TASKS
from .utils import ensure_dir, read_jsonl, seed_everything, tensor_to_numpy


def backfill_incorrect_activations_for_task(
    task: str,
    seed: int,
    output_dir: str | Path,
    model_id: str = "google/gemma-4-E2B-it",
    dtype: str = "auto",
    device_map: str = "auto",
    geometry_dtype: str = "float16",
    semantic_vocab_sample: int = 256,
    max_examples: int | None = None,
    overwrite: bool = False,
) -> Path:
    if task not in TASKS:
        raise KeyError(f"Unknown task '{task}'. Choices: {sorted(TASKS)}")
    seed_everything(seed)
    output_dir = Path(output_dir)
    inference_path = output_dir / "inference" / f"{task}_seed{seed}.jsonl"
    if not inference_path.exists():
        raise FileNotFoundError(f"Missing inference log: {inference_path}")

    activation_dir = ensure_dir(output_dir / "activations")
    npz_path = activation_dir / f"{task}_seed{seed}_incorrect_activations.npz"
    if npz_path.exists() and not overwrite:
        return npz_path
    if overwrite:
        npz_path.unlink(missing_ok=True)

    records = [record for record in read_jsonl(inference_path) if not record.get("correct")]
    if max_examples is not None:
        records = records[:max_examples]

    bundle = load_gemma(model_id=model_id, dtype=dtype, device_map=device_map)
    unembedding_subset = _sample_unembedding_subset(bundle.model, seed, semantic_vocab_sample)

    activations: list[np.ndarray] = []
    semantic_activations: list[np.ndarray] = []
    row_ids: list[int] = []
    local_indices: list[int] = []
    generated_lengths: list[int] = []

    for record in tqdm(records, desc=f"backfill {task} seed={seed} incorrect"):
        vectors = _teacher_forced_final_token_hidden_states(bundle, record)
        if vectors is None:
            continue
        activations.append(tensor_to_numpy(vectors, geometry_dtype))
        semantic = _project_with_unembedding_subset(vectors, unembedding_subset)
        if semantic is not None:
            semantic_activations.append(tensor_to_numpy(semantic, geometry_dtype))
        row_ids.append(int(record.get("row_id", record.get("local_idx", -1))))
        local_indices.append(int(record.get("local_idx", -1)))
        generated_lengths.append(int(record.get("generated_len", len(record.get("generated_token_ids", [])))))

    stacked = _stack_or_empty(activations, geometry_dtype)
    semantic_stacked = _stack_or_empty(semantic_activations, geometry_dtype)
    np.savez_compressed(
        npz_path,
        activations=stacked,
        semantic_activations=semantic_stacked,
        row_ids=np.array(row_ids),
        local_indices=np.array(local_indices),
        generated_lengths=np.array(generated_lengths),
        correct=np.zeros(len(row_ids), dtype=bool),
        outcome_group=np.array("incorrect"),
        task=np.array(task),
        seed=np.array(seed),
        model_id=np.array(model_id),
        semantic_vocab_sample=np.array(semantic_vocab_sample),
    )
    return npz_path


def _teacher_forced_final_token_hidden_states(bundle: Any, record: dict[str, Any]) -> torch.Tensor | None:
    generated_token_ids = [int(token_id) for token_id in record.get("generated_token_ids", [])]
    if not generated_token_ids:
        return None
    prompt = record["prompt"]
    prompt_inputs = bundle.processor(text=prompt, return_tensors="pt")
    prompt_ids = prompt_inputs["input_ids"][0].to(bundle.device)
    generated_ids = torch.tensor(generated_token_ids, dtype=prompt_ids.dtype, device=bundle.device)
    input_ids = torch.cat([prompt_ids, generated_ids], dim=0).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, device=bundle.device)
    with torch.inference_mode():
        outputs = bundle.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        return None
    return torch.stack([layer_state[0, -1, :].detach() for layer_state in hidden_states], dim=0)


def _stack_or_empty(arrays: list[np.ndarray], dtype: str) -> np.ndarray:
    if arrays:
        return np.stack(arrays, axis=0)
    np_dtype = np.float16 if dtype == "float16" else np.float32
    return np.empty((0, 0, 0), dtype=np_dtype)
