from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .scoring import clean_short_answer, score_answer
from .utils import extract_final_number, extract_first_label, normalize_text


SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: str
    dataset_name: str
    dataset_config: str | None
    split: str
    default_max_new_tokens: int


TASKS: dict[str, TaskSpec] = {
    "fact": TaskSpec(
        name="fact",
        task_type="type1_fact_retrieval",
        dataset_name="mandarjoshi/trivia_qa",
        dataset_config="rc.nocontext",
        split="train",
        default_max_new_tokens=32,
    ),
    "sentiment": TaskSpec(
        name="sentiment",
        task_type="type2_sentiment",
        dataset_name="SetFit/sst2",
        dataset_config=None,
        split="train",
        default_max_new_tokens=8,
    ),
    "math": TaskSpec(
        name="math",
        task_type="type3_math_reasoning",
        dataset_name="openai/gsm8k",
        dataset_config="main",
        split="train",
        default_max_new_tokens=256,
    ),
}


def load_task_dataset(task: str, sample_size: int, seed: int):
    datasets = __import__("datasets")
    spec = TASKS[task]
    if spec.dataset_config:
        ds = datasets.load_dataset(spec.dataset_name, spec.dataset_config, split=spec.split)
    else:
        ds = datasets.load_dataset(spec.dataset_name, split=spec.split)
    ds = ds.shuffle(seed=seed)
    return ds.select(range(min(sample_size, len(ds))))


def make_messages(task: str, row: dict[str, Any], system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    if task == "fact":
        question = row["question"]
        user = (
            "Answer the factual question with only the shortest correct answer. "
            "Do not explain.\n\n"
            f"Question: {question}"
        )
    elif task == "sentiment":
        text = row["text"]
        user = (
            "Classify the sentiment of the sentence. Reply with exactly one word: "
            "positive or negative.\n\n"
            f"Sentence: {text}"
        )
    elif task == "math":
        question = row["question"]
        user = (
            "Solve the math word problem. You may reason briefly, but the final line "
            "must be: Final answer: <number>\n\n"
            f"Problem: {question}"
        )
    else:
        raise KeyError(f"Unknown task: {task}")
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def target_for_row(task: str, row: dict[str, Any]) -> Any:
    if task == "fact":
        answer = row.get("answer", {})
        return {
            "value": answer.get("value", ""),
            "normalized_value": answer.get("normalized_value", ""),
            "matched_wiki_entity_name": answer.get("matched_wiki_entity_name", ""),
            "normalized_matched_wiki_entity_name": answer.get("normalized_matched_wiki_entity_name", ""),
            "aliases": list(answer.get("aliases", []) or []),
            "normalized_aliases": list(answer.get("normalized_aliases", []) or []),
        }
    if task == "sentiment":
        if "label_text" in row and row["label_text"] is not None:
            return str(row["label_text"]).lower()
        return "positive" if int(row["label"]) == 1 else "negative"
    if task == "math":
        return extract_final_number(row["answer"])
    raise KeyError(f"Unknown task: {task}")


def parse_answer(task: str, raw_text: str, parsed_response: Any | None = None) -> str:
    text = _coerce_parsed_response(parsed_response) if parsed_response is not None else raw_text
    if task == "sentiment":
        return extract_first_label(text, {"positive", "negative"})
    if task == "math":
        return extract_final_number(text) or normalize_text(text)
    return clean_short_answer(text)


def is_correct(task: str, parsed_answer: str, target: Any) -> bool:
    if task == "fact":
        return bool(score_answer(task, parsed_answer, target)["correct"])
    if task == "sentiment":
        return normalize_text(parsed_answer) == normalize_text(target)
    if task == "math":
        return bool(score_answer(task, parsed_answer, target)["correct"])
    raise KeyError(f"Unknown task: {task}")


def _coerce_parsed_response(parsed_response: Any) -> str:
    if isinstance(parsed_response, str):
        return parsed_response
    if isinstance(parsed_response, dict):
        for key in ("final", "answer", "content", "response"):
            if key in parsed_response and parsed_response[key]:
                return str(parsed_response[key])
        return str(parsed_response)
    return str(parsed_response)
