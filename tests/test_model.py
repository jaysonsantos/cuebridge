from __future__ import annotations

import torch
from cuebridge.agent import trim_messages_to_token_budget
from cuebridge.model import OpenAICompatibleChatModel, TranslateGemmaChatModel
from langchain_core.messages import AIMessage, HumanMessage


class FakeEncoding(dict):
    def to(self, device):
        self["moved_to"] = device
        return self


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return FakeEncoding(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        )

    def decode(self, tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        assert tokens.tolist() == [7, 8]
        return "translated text"


class FakeProcessor:
    def __init__(self, tokenizer: FakeTokenizer) -> None:
        self.tokenizer = tokenizer


class FakeModel:
    def __init__(self) -> None:
        self.device = "cpu"
        self.generate_calls: list[dict] = []

    def to(self, device):
        self.device = device
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor([[1, 2, 3, 7, 8]])


def test_model_formats_full_history_for_translategemma() -> None:
    tokenizer = FakeTokenizer()
    fake_model = FakeModel()

    model = TranslateGemmaChatModel(
        source_lang_code="en",
        target_lang_code="pt-BR",
        device="cpu",
        processor_loader=lambda model_id: FakeProcessor(tokenizer),
        model_loader=lambda model_id, dtype: fake_model,
    )

    result = model._generate(
        [
            HumanMessage(content="Hello"),
            AIMessage(content="Ola"),
            HumanMessage(content="Goodbye"),
        ]
    )

    assert result.generations[0].message.content == "translated text"
    assert tokenizer.calls[0]["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": "en",
                    "target_lang_code": "pt-BR",
                    "text": "Hello",
                }
            ],
        },
        {"role": "assistant", "content": "Ola"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": "en",
                    "target_lang_code": "pt-BR",
                    "text": "Goodbye",
                }
            ],
        },
    ]
    assert fake_model.generate_calls[0]["pad_token_id"] == 99
    assert fake_model.generate_calls[0]["max_new_tokens"] == 256


def test_model_rejects_unsupported_dtype() -> None:
    model = TranslateGemmaChatModel(
        source_lang_code="en",
        target_lang_code="pt-BR",
        dtype="definitely_not_a_dtype",
        processor_loader=lambda *args, **kwargs: None,
        model_loader=lambda *args, **kwargs: None,
    )

    try:
        model._get_model()
    except ValueError as exc:
        assert "Unsupported torch dtype" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid dtype")


def test_trim_messages_to_token_budget_keeps_recent_history() -> None:
    messages = [
        HumanMessage(content="one"),
        AIMessage(content="ONE"),
        HumanMessage(content="two"),
        AIMessage(content="TWO"),
        HumanMessage(content="three"),
    ]

    def token_counter(candidate):
        total = 0
        for message in candidate:
            total += len(str(message.content).split())
        return total

    trimmed = trim_messages_to_token_budget(
        messages=messages,
        token_counter=token_counter,
        max_input_tokens=3,
    )

    assert trimmed == messages[2:]


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": "ola mundo",
                    }
                }
            ]
        }


def test_openai_compatible_model_formats_chat_completions_request() -> None:
    captured_calls: list[dict] = []

    def fake_request_sender(url, **kwargs):
        captured_calls.append({"url": url, **kwargs})
        return FakeResponse()

    model = OpenAICompatibleChatModel(
        source_lang_code="de",
        target_lang_code="pt-BR",
        model_id="mlx-community/translategemma-4b-it-4bit",
        api_base_url="http://localhost:1234/v1",
        message_format="translategemma",
        request_sender=fake_request_sender,
    )

    result = model._generate(
        [
            HumanMessage(content="Hallo Welt"),
            AIMessage(content="ola mundo"),
            HumanMessage(content="Wie geht es dir?"),
        ]
    )

    assert result.generations[0].message.content == "ola mundo"
    assert captured_calls[0]["url"] == "http://localhost:1234/v1/chat/completions"
    assert captured_calls[0]["json"]["model"] == "mlx-community/translategemma-4b-it-4bit"
    assert captured_calls[0]["json"]["messages"][0]["role"] == "user"
    assert captured_calls[0]["json"]["messages"][0]["content"] == [
        {
            "type": "text",
            "source_lang_code": "de",
            "target_lang_code": "pt-BR",
            "text": "Hallo Welt",
        }
    ]
    assert captured_calls[0]["json"]["messages"][1] == {
        "role": "assistant",
        "content": "ola mundo",
    }
