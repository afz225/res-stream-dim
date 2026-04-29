from geometric_complexity_scaling.modeling import apply_gemma_chat_template


class FakeProcessor:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return "formatted prompt"


def test_chat_template_disables_thinking():
    processor = FakeProcessor()
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert apply_gemma_chat_template(processor, messages) == "formatted prompt"
    assert processor.kwargs["tokenize"] is False
    assert processor.kwargs["add_generation_prompt"] is True
    assert processor.kwargs["enable_thinking"] is False


def test_generated_token_slicing_contract():
    sequence = [10, 11, 12, 99, 100]
    input_len = 3
    assert sequence[input_len:] == [99, 100]
