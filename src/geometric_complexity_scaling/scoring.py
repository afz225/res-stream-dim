from __future__ import annotations

import collections
import re
import string
from typing import Any


ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:final\s+answer|answer|ans|response)\s*[:\-]\s*",
    flags=re.IGNORECASE,
)
SPECIAL_TOKEN_RE = re.compile(r"<[^>\n]{1,80}>")


def normalize_triviaqa_answer(text: Any) -> str:
    """TriviaQA-style answer normalization."""
    text = "" if text is None else str(text)
    text = text.replace("_", " ").lower()
    punctuation = set(string.punctuation + "‘’´`“”")
    text = "".join(ch if ch not in punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split()).strip()


def clean_short_answer(raw_text: Any) -> str:
    text = "" if raw_text is None else str(raw_text)
    text = SPECIAL_TOKEN_RE.sub(" ", text)
    text = text.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[-1] if any("answer" in line.lower() for line in lines) else lines[0]
    text = ANSWER_PREFIX_RE.sub("", text).strip()
    text = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    text = ANSWER_PREFIX_RE.sub("", text).strip()
    return text.strip(" \t\r\n\"'")


def triviaqa_f1(prediction: Any, ground_truth: Any) -> float:
    pred_tokens = normalize_triviaqa_answer(prediction).split()
    gold_tokens = normalize_triviaqa_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def triviaqa_exact_match(prediction: Any, ground_truth: Any) -> bool:
    return normalize_triviaqa_answer(prediction) == normalize_triviaqa_answer(ground_truth)


def triviaqa_ground_truths(target: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("value", "normalized_value", "matched_wiki_entity_name", "normalized_matched_wiki_entity_name"):
        value = target.get(key)
        if value and value != "<unk>":
            values.append(str(value))
    for key in ("aliases", "normalized_aliases"):
        for value in target.get(key, []) or []:
            if value and value != "<unk>":
                values.append(str(value))
    deduped = []
    seen = set()
    for value in values:
        normalized = normalize_triviaqa_answer(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(value)
    return deduped


def score_triviaqa_answer(prediction: Any, target: dict[str, Any], f1_threshold: float = 0.5) -> dict[str, Any]:
    cleaned = clean_short_answer(prediction)
    ground_truths = triviaqa_ground_truths(target)
    if not ground_truths:
        return {
            "prediction": cleaned,
            "exact_match": False,
            "max_f1": 0.0,
            "matched_alias": None,
            "correct": False,
        }
    em_scores = [triviaqa_exact_match(cleaned, truth) for truth in ground_truths]
    f1_scores = [triviaqa_f1(cleaned, truth) for truth in ground_truths]
    best_idx = max(range(len(f1_scores)), key=f1_scores.__getitem__)
    exact_match = any(em_scores)
    max_f1 = float(f1_scores[best_idx])
    return {
        "prediction": cleaned,
        "exact_match": bool(exact_match),
        "max_f1": max_f1,
        "matched_alias": ground_truths[best_idx],
        "correct": bool(exact_match or max_f1 >= f1_threshold),
    }


def score_answer(task: str, parsed_answer: str, target: Any) -> dict[str, Any]:
    if task == "fact":
        return score_triviaqa_answer(parsed_answer, target)
    if task == "sentiment":
        correct = str(parsed_answer).strip().lower() == str(target).strip().lower()
        return {"correct": bool(correct), "exact_match": bool(correct)}
    if task == "math":
        from .utils import numbers_equal

        correct = numbers_equal(parsed_answer, target)
        return {"correct": bool(correct), "exact_match": bool(correct)}
    raise KeyError(f"Unknown task: {task}")
