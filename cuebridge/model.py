from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any, Callable

import requests
import torch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr
from transformers import AutoModelForImageTextToText, AutoProcessor

ModelLoader = Callable[..., Any]
ProcessorLoader = Callable[..., Any]
RequestSender = Callable[..., Any]


class TranslateGemmaChatModel(BaseChatModel):
    """LangChain chat wrapper around TranslateGemma with tokenizer-based history."""

    source_lang_code: str
    target_lang_code: str
    model_id: str = "google/translategemma-4b-it"
    dtype: str = "bfloat16"
    device: str | None = None
    max_new_tokens: int = 256
    batch_size: int = 1
    processor_loader: ProcessorLoader | None = Field(default=None, exclude=True, repr=False)
    model_loader: ModelLoader | None = Field(default=None, exclude=True, repr=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _processor: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "translategemma"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source_lang_code": self.source_lang_code,
            "target_lang_code": self.target_lang_code,
            "dtype": self.dtype,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        translated_text = self._generate_translated_text(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=translated_text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del stop, run_manager, kwargs
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=self._generate_translated_text(messages),
                chunk_position="last",
            )
        )

    def count_input_tokens(self, messages: list[BaseMessage]) -> int:
        inputs = self._tokenize_messages(messages)
        return int(inputs["input_ids"].shape[1])

    def _get_processor(self) -> Any:
        if self._processor is None:
            loader = self.processor_loader or AutoProcessor.from_pretrained
            self._processor = loader(self.model_id)

        return self._processor

    def _get_model(self) -> Any:
        if self._model is None:
            loader = self.model_loader or AutoModelForImageTextToText.from_pretrained
            dtype = _resolve_torch_dtype(self.dtype)

            if self.device is None:
                self._model = loader(
                    self.model_id,
                    device_map="auto",
                    dtype=dtype,
                )
            else:
                self._model = loader(self.model_id, dtype=dtype)
                self._model.to(self.device)

            generation_config = getattr(self._model, "generation_config", None)
            if generation_config is not None:
                for field_name in ("top_k", "top_p"):
                    if hasattr(generation_config, field_name):
                        setattr(generation_config, field_name, None)

        return self._model

    def _model_device(self) -> Any:
        return self._get_model().device

    def _get_tokenizer(self) -> Any:
        return self._get_processor().tokenizer

    def _tokenize_messages(self, messages: list[BaseMessage]) -> Any:
        return self._get_tokenizer().apply_chat_template(
            [self._format_message(message) for message in messages],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    def _generate_translated_text(self, messages: list[BaseMessage]) -> str:
        inputs = self._tokenize_messages(messages).to(self._model_device())
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generation = self._get_model().generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._get_tokenizer().eos_token_id,
            )

        return (
            self._get_tokenizer()
            .decode(
                generation[0][input_len:],
                skip_special_tokens=True,
            )
            .strip()
        )

    def _format_message(self, message: BaseMessage) -> dict[str, Any]:
        text = _message_to_text(message.content)

        if isinstance(message, HumanMessage):
            return {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": self.source_lang_code,
                        "target_lang_code": self.target_lang_code,
                        "text": text,
                    }
                ],
            }

        if isinstance(message, AIMessage):
            return {
                "role": "assistant",
                "content": text,
            }

        raise TypeError(
            f"TranslateGemma only supports human and assistant messages, got {type(message)!r}"
        )


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}") from exc


def _message_to_text(content: str | list[str | dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue

        text = item.get("text")
        if text:
            parts.append(str(text))

    return "\n".join(parts).strip()


class OpenAICompatibleChatModel(BaseChatModel):
    """Chat model adapter for OpenAI-compatible /v1/chat/completions APIs."""

    source_lang_code: str
    target_lang_code: str
    model_id: str
    api_base_url: str = "http://localhost:1234/v1"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    request_timeout_seconds: float = 120.0
    max_new_tokens: int = 256
    message_format: str = "auto"
    request_sender: RequestSender | None = Field(default=None, exclude=True, repr=False)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _session: requests.Session | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "api_base_url": self.api_base_url,
            "source_lang_code": self.source_lang_code,
            "target_lang_code": self.target_lang_code,
            "max_new_tokens": self.max_new_tokens,
            "message_format": self.message_format,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        translated_text = self._generate_translated_text(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=translated_text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del stop, run_manager, kwargs
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=self._generate_translated_text(messages),
                chunk_position="last",
            )
        )

    def count_input_tokens(self, messages: list[BaseMessage]) -> int:
        total = 0
        for message in messages:
            formatted = self._format_message(message)
            total += 8
            total += _estimate_token_count(str(formatted["content"]))
        return total + 8

    def _format_message(self, message: BaseMessage) -> dict[str, Any]:
        text = _message_to_text(message.content)

        if isinstance(message, HumanMessage):
            if self._resolved_message_format() == "translategemma":
                return {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": self.source_lang_code,
                            "target_lang_code": self.target_lang_code,
                            "text": text,
                        }
                    ],
                }

            return {
                "role": "user",
                "content": _build_translation_prompt(
                    text=text,
                    source_lang_code=self.source_lang_code,
                    target_lang_code=self.target_lang_code,
                ),
            }

        if isinstance(message, AIMessage):
            return {
                "role": "assistant",
                "content": text,
            }

        raise TypeError(
            f"OpenAI-compatible backend only supports human and assistant messages, got {type(message)!r}"
        )

    def _chat_completions_url(self) -> str:
        return f"{self.api_base_url.rstrip('/')}/chat/completions"

    def _generate_translated_text(self, messages: list[BaseMessage]) -> str:
        payload = {
            "model": self.model_id,
            "messages": [self._format_message(message) for message in messages],
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
        }
        headers = {"Content-Type": "application/json"}
        api_key = self._resolved_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = self._request_sender()(
            self._chat_completions_url(),
            headers=headers,
            json=payload,
            timeout=self.request_timeout_seconds,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            raise ValueError(
                f"OpenAI-compatible backend request failed with {response.status_code}: {detail}"
            ) from exc
        data = response.json()
        return _message_to_text(data["choices"][0]["message"]["content"]).strip()

    def _request_sender(self) -> RequestSender:
        if self.request_sender is not None:
            return self.request_sender

        if self._session is None:
            self._session = requests.Session()
        return self._session.post

    def _resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None

    def _resolved_message_format(self) -> str:
        if self.message_format != "auto":
            return self.message_format
        if "translategemma" in self.model_id.lower():
            return "translategemma"
        return "plain"


def _build_translation_prompt(
    *,
    text: str,
    source_lang_code: str,
    target_lang_code: str,
) -> str:
    return (
        f"You are a professional translator from {source_lang_code} to {target_lang_code}. "
        f"Translate the text faithfully and naturally into {target_lang_code}. "
        "Return only the translation, with no commentary or explanations.\n\n"
        f"{text}"
    )


def _estimate_token_count(text: str) -> int:
    return max(1, (len(text) + 3) // 4)
