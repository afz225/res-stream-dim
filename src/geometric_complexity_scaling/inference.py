from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .modeling import (
    apply_gemma_chat_template,
    get_output_embedding_weight,
    get_pad_token_id,
    load_gemma,
    parse_gemma_response,
)
from .scoring import score_answer
from .tasks import TASKS, load_task_dataset, make_messages, parse_answer, target_for_row
from .utils import append_jsonl, ensure_dir, json_safe, seed_everything, tensor_to_numpy


def run_inference_for_task(
    task: str,
    seed: int,
    output_dir: str | Path,
    model_id: str = "google/gemma-4-E2B-it",
    sample_size: int = 2000,
    max_new_tokens: int | None = None,
    system_prompt: str = "You are a helpful assistant.",
    dtype: str = "auto",
    device_map: str = "auto",
    geometry_dtype: str = "float16",
    max_correct_geometry: int | None = None,
    semantic_vocab_sample: int = 256,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    if task not in TASKS:
        raise KeyError(f"Unknown task '{task}'. Choices: {sorted(TASKS)}")
    seed_everything(seed)
    output_dir = Path(output_dir)
    inference_dir = ensure_dir(output_dir / "inference")
    activation_dir = ensure_dir(output_dir / "activations")
    jsonl_path = inference_dir / f"{task}_seed{seed}.jsonl"
    npz_path = activation_dir / f"{task}_seed{seed}_correct_activations.npz"
    if overwrite:
        jsonl_path.unlink(missing_ok=True)
        npz_path.unlink(missing_ok=True)
    elif jsonl_path.exists() and npz_path.exists():
        return jsonl_path, npz_path

    bundle = load_gemma(model_id=model_id, dtype=dtype, device_map=device_map)
    ds = load_task_dataset(task=task, sample_size=sample_size, seed=seed)
    max_new = max_new_tokens or TASKS[task].default_max_new_tokens

    correct_activations: list[np.ndarray] = []
    correct_semantic_activations: list[np.ndarray] = []
    correct_row_ids: list[int] = []
    correct_generated_lengths: list[int] = []
    unembedding_subset = _sample_unembedding_subset(bundle.model, seed, semantic_vocab_sample)

    for local_idx, row in enumerate(tqdm(ds, desc=f"{task} seed={seed}")):
        row_dict = dict(row)
        row_id = int(row_dict.get("question_id", local_idx)) if str(row_dict.get("question_id", "")).isdigit() else local_idx
        messages = make_messages(task, row_dict, system_prompt=system_prompt)
        prompt_text = apply_gemma_chat_template(bundle.processor, messages)
        inputs = bundle.processor(text=prompt_text, return_tensors="pt").to(bundle.device)
        input_len = int(inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            outputs = bundle.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True,
                pad_token_id=get_pad_token_id(bundle.processor),
            )

        sequence = outputs.sequences[0]
        generated_ids = sequence[input_len:]
        response = bundle.processor.decode(generated_ids, skip_special_tokens=False)
        parsed_response = parse_gemma_response(bundle.processor, response)
        parsed_answer = parse_answer(task, response, parsed_response)
        target = target_for_row(task, row_dict)
        score_details = score_answer(task, parsed_answer, target)
        correct = bool(score_details["correct"])

        generated_token_ids = [int(x) for x in generated_ids.detach().cpu().tolist()]
        record: dict[str, Any] = {
            "task": task,
            "task_type": TASKS[task].task_type,
            "dataset": TASKS[task].dataset_name,
            "dataset_config": TASKS[task].dataset_config,
            "seed": seed,
            "row_id": row_id,
            "local_idx": local_idx,
            "messages": messages,
            "prompt": prompt_text,
            "target": target,
            "raw_output": response,
            "parsed_response": json_safe(parsed_response),
            "parsed_answer": parsed_answer,
            "score_details": json_safe(score_details),
            "correct": bool(correct),
            "input_len": input_len,
            "generated_len": len(generated_token_ids),
            "generated_token_ids": generated_token_ids,
            "created_unix": time.time(),
        }
        append_jsonl(jsonl_path, record)

        if correct and (max_correct_geometry is None or len(correct_activations) < max_correct_geometry):
            activations = _extract_final_token_hidden_states(outputs)
            if activations is not None:
                correct_activations.append(tensor_to_numpy(activations, geometry_dtype))
                semantic = _project_with_unembedding_subset(activations, unembedding_subset)
                if semantic is not None:
                    correct_semantic_activations.append(tensor_to_numpy(semantic, geometry_dtype))
                correct_row_ids.append(row_id)
                correct_generated_lengths.append(len(generated_token_ids))

    if correct_activations:
        stacked = np.stack(correct_activations, axis=0)
    else:
        stacked = np.empty((0, 0, 0), dtype=np.float16 if geometry_dtype == "float16" else np.float32)
    if correct_semantic_activations:
        semantic_stacked = np.stack(correct_semantic_activations, axis=0)
    else:
        semantic_stacked = np.empty((0, 0, 0), dtype=stacked.dtype)
    np.savez_compressed(
        npz_path,
        activations=stacked,
        semantic_activations=semantic_stacked,
        row_ids=np.array(correct_row_ids),
        generated_lengths=np.array(correct_generated_lengths),
        task=np.array(task),
        seed=np.array(seed),
        model_id=np.array(model_id),
        semantic_vocab_sample=np.array(semantic_vocab_sample),
    )
    return jsonl_path, npz_path


def _extract_final_token_hidden_states(outputs: Any) -> torch.Tensor | None:
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        return None
    final_step = hidden_states[-1]
    layer_vectors = []
    for layer_state in final_step:
        # Shape is usually [batch, generated_seq_len_for_step, hidden_dim].
        layer_vectors.append(layer_state[0, -1, :].detach())
    return torch.stack(layer_vectors, dim=0)


def _sample_unembedding_subset(model: Any, seed: int, sample_size: int) -> torch.Tensor | None:
    weight = get_output_embedding_weight(model)
    if weight is None or sample_size <= 0:
        return None
    vocab_size = int(weight.shape[0])
    count = min(sample_size, vocab_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(vocab_size, generator=generator)[:count].to(weight.device)
    return weight.index_select(0, indices).detach()


def _project_with_unembedding_subset(activations: torch.Tensor, subset: torch.Tensor | None) -> torch.Tensor | None:
    if subset is None:
        return None
    projected = activations.to(dtype=subset.dtype, device=subset.device) @ subset.transpose(0, 1)
    return projected.detach()
