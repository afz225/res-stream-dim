from __future__ import annotations

import json
import importlib
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def require_import(module_name: str, package_hint: str | None = None) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        package = package_hint or module_name
        raise SystemExit(
            f"Missing dependency '{module_name}'. Install it with: python -m pip install {package}"
        ) from exc


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        return str(value)


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9.\-/% ]+", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_first_label(text: str, labels: set[str]) -> str:
    normalized = normalize_text(text)
    for token in re.split(r"\s+", normalized):
        stripped = token.strip(".,:;!?\"'")
        if stripped in labels:
            return stripped
    return normalized


def extract_final_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(text))
    if not matches:
        return None
    return matches[-1].replace(",", "")


def numbers_equal(pred: str | None, gold: str | None, atol: float = 1e-6) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) <= atol
    except ValueError:
        return normalize_text(pred) == normalize_text(gold)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_numpy(tensor: torch.Tensor, dtype: str = "float16") -> np.ndarray:
    cpu_tensor = tensor.detach().cpu()
    if cpu_tensor.dtype in (torch.bfloat16,):
        cpu_tensor = cpu_tensor.to(torch.float32)
    array = cpu_tensor.numpy()
    if dtype == "float16":
        return array.astype(np.float16)
    if dtype == "float32":
        return array.astype(np.float32)
    raise ValueError(f"Unsupported geometry dtype: {dtype}")
