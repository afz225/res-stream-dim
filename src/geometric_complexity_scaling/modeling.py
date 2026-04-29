from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .utils import require_import


@dataclass
class GemmaBundle:
    model: Any
    processor: Any
    device: torch.device


def load_gemma(
    model_id: str,
    dtype: str = "auto",
    device_map: str = "auto",
) -> GemmaBundle:
    transformers = require_import("transformers", "transformers")
    torch_dtype = _resolve_dtype(dtype)
    processor = transformers.AutoProcessor.from_pretrained(model_id)
    model_kwargs = {"torch_dtype": torch_dtype, "device_map": device_map}
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except Exception:
        image_text_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if image_text_cls is None:
            raise
        model = image_text_cls.from_pretrained(model_id, **model_kwargs)
    model.eval()
    device = next(model.parameters()).device
    return GemmaBundle(model=model, processor=processor, device=device)


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def apply_gemma_chat_template(processor: Any, messages: list[dict[str, str]]) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def parse_gemma_response(processor: Any, response: str) -> Any | None:
    parser = getattr(processor, "parse_response", None)
    if parser is None:
        return None
    try:
        return parser(response)
    except Exception:
        return None


def get_pad_token_id(processor: Any) -> int | None:
    for obj in (processor, getattr(processor, "tokenizer", None)):
        if obj is None:
            continue
        for attr in ("pad_token_id", "eos_token_id"):
            value = getattr(obj, attr, None)
            if value is not None:
                return int(value)
    return None


def get_output_embedding_weight(model: Any) -> torch.Tensor | None:
    getter = getattr(model, "get_output_embeddings", None)
    if getter is None:
        return None
    embeddings = getter()
    if embeddings is None or not hasattr(embeddings, "weight"):
        return None
    return embeddings.weight
