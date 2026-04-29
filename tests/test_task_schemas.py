import os

import pytest

from geometric_complexity_scaling.tasks import TASKS, make_messages, target_for_row


def test_synthetic_triviaqa_schema():
    row = {
        "question": "Where in England was Dame Judi Dench born?",
        "question_id": "tc_3",
        "answer": {
            "value": "York",
            "normalized_value": "york",
            "aliases": ["York, UK"],
            "normalized_aliases": ["york uk"],
            "matched_wiki_entity_name": "York",
            "normalized_matched_wiki_entity_name": "york",
        },
    }
    messages = make_messages("fact", row)
    target = target_for_row("fact", row)
    assert messages[0]["role"] == "system"
    assert "Question:" in messages[1]["content"]
    assert target["value"] == "York"
    assert target["aliases"] == ["York, UK"]


def test_synthetic_sst2_schema():
    row = {"text": "A tiny, joyful movie.", "label": 1, "label_text": "positive"}
    assert target_for_row("sentiment", row) == "positive"
    assert "positive or negative" in make_messages("sentiment", row)[1]["content"]


def test_synthetic_gsm8k_schema():
    row = {"question": "If Ana has 2 apples and gets 3 more, how many?", "answer": "#### 5"}
    assert target_for_row("math", row) == "5"
    assert "Final answer" in make_messages("math", row)[1]["content"]


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_DATASET_CONTRACT_TESTS") != "1", reason="dataset contract tests are opt-in")
def test_real_dataset_contracts():
    from geometric_complexity_scaling.tasks import load_task_dataset

    for task in TASKS:
        row = dict(load_task_dataset(task, sample_size=1, seed=0)[0])
        assert make_messages(task, row)
        target = target_for_row(task, row)
        assert target is not None
        if task == "fact":
            assert "question" in row
            assert "answer" in row
            answer = row["answer"]
            for key in ("value", "normalized_value", "aliases", "normalized_aliases"):
                assert key in answer
