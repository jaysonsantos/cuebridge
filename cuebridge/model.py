from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any, Callable
from urllib.parse import urlparse

import requests
import torch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from opentelemetry import trace
from pydantic import ConfigDict, Field, PrivateAttr
from transformers import AutoModelForImageTextToText, AutoProcessor

ModelLoader = Callable[..., Any]
ProcessorLoader = Callable[..., Any]
RequestSender = Callable[..., Any]
TRACER = trace.get_tracer(__name__)

DEFAULT_MAX_LENGTH_CONTINUATIONS = 75

MESSAGE_FORMAT_AUTO = "auto"
MESSAGE_FORMAT_PLAIN = "plain"
MESSAGE_FORMAT_TRANSLATEGEMMA = "translategemma"

CHAT_ROLE_KEY = "role"
CHAT_CONTENT_KEY = "content"
CHAT_USER_ROLE = "user"
CHAT_ASSISTANT_ROLE = "assistant"

OPENAI_MODEL_KEY = "model"
OPENAI_MESSAGES_KEY = "messages"
OPENAI_TEMPERATURE_KEY = "temperature"
OPENAI_MAX_TOKENS_KEY = "max_tokens"
OPENAI_REASONING_EFFORT_KEY = "reasoning_effort"
OPENAI_CHOICES_KEY = "choices"
OPENAI_MESSAGE_KEY = "message"
OPENAI_FINISH_REASON_KEY = "finish_reason"
OPENAI_FINISH_REASON_LENGTH = "length"

HTTP_HEADER_CONTENT_TYPE = "Content-Type"
HTTP_HEADER_AUTHORIZATION = "Authorization"
HTTP_HEADER_OPENROUTER_REFERER = "HTTP-Referer"
HTTP_HEADER_OPENROUTER_TITLE = "X-OpenRouter-Title"
HTTP_CONTENT_TYPE_JSON = "application/json"
OPENROUTER_HOST = "openrouter.ai"
OPENROUTER_APP_REFERER = "https://github.com/jaysonsantos/cuebridge"
OPENROUTER_APP_TITLE = "CueBridge"


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
        with TRACER.start_as_current_span("cuebridge.model.translategemma.generate") as span:
            inputs = self._tokenize_messages(messages).to(self._model_device())
            input_len = int(inputs["input_ids"].shape[1])
            span.set_attribute("cuebridge.model_id", self.model_id)
            span.set_attribute("cuebridge.input_messages", len(messages))
            span.set_attribute("cuebridge.input_tokens", input_len)
            span.set_attribute("cuebridge.max_new_tokens", self.max_new_tokens)
            if self.device is not None:
                span.set_attribute("cuebridge.device", self.device)

            with torch.inference_mode():
                generation = self._get_model().generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self._get_tokenizer().eos_token_id,
                )

            translated = (
                self._get_tokenizer()
                .decode(
                    generation[0][input_len:],
                    skip_special_tokens=True,
                )
                .strip()
            )
            span.set_attribute("cuebridge.output_length", len(translated))
            return translated

    def _format_message(self, message: BaseMessage) -> dict[str, Any]:
        text = _message_to_text(message.content)

        if isinstance(message, HumanMessage):
            return {
                CHAT_ROLE_KEY: CHAT_USER_ROLE,
                CHAT_CONTENT_KEY: [
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
                CHAT_ROLE_KEY: CHAT_ASSISTANT_ROLE,
                CHAT_CONTENT_KEY: text,
            }

        raise TypeError(
            f"TranslateGemma only supports human and assistant messages, got {type(message)!r}"
        )


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}") from exc


def _message_to_text(content: str | list[str | dict[str, Any]] | None) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""

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
    reasoning_effort: str | None = None
    max_new_tokens: int = 256
    max_length_continuations: int = DEFAULT_MAX_LENGTH_CONTINUATIONS
    message_format: str = MESSAGE_FORMAT_AUTO
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
            "max_length_continuations": self.max_length_continuations,
            "reasoning_effort": self.reasoning_effort,
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
            total += _estimate_token_count(str(formatted[CHAT_CONTENT_KEY]))
        return total + 8

    def _format_message(self, message: BaseMessage) -> dict[str, Any]:
        text = _message_to_text(message.content)

        if isinstance(message, HumanMessage):
            if self._resolved_message_format() == MESSAGE_FORMAT_TRANSLATEGEMMA:
                return {
                    CHAT_ROLE_KEY: CHAT_USER_ROLE,
                    CHAT_CONTENT_KEY: [
                        {
                            "type": "text",
                            "source_lang_code": self.source_lang_code,
                            "target_lang_code": self.target_lang_code,
                            "text": text,
                        }
                    ],
                }

            return {
                CHAT_ROLE_KEY: CHAT_USER_ROLE,
                CHAT_CONTENT_KEY: _build_translation_prompt(
                    text=text,
                    source_lang_code=self.source_lang_code,
                    target_lang_code=self.target_lang_code,
                ),
            }

        if isinstance(message, AIMessage):
            return {
                CHAT_ROLE_KEY: CHAT_ASSISTANT_ROLE,
                CHAT_CONTENT_KEY: text,
            }

        raise TypeError(
            f"OpenAI-compatible backend only supports human and assistant messages, got {type(message)!r}"
        )

    def _chat_completions_url(self) -> str:
        return f"{self.api_base_url.rstrip('/')}/chat/completions"

    def _generate_translated_text(self, messages: list[BaseMessage]) -> str:
        with TRACER.start_as_current_span("cuebridge.model.openai_compatible.generate") as span:
            request_messages = [self._format_message(message) for message in messages]
            headers = {HTTP_HEADER_CONTENT_TYPE: HTTP_CONTENT_TYPE_JSON}
            api_key = self._resolved_api_key()
            if api_key:
                headers[HTTP_HEADER_AUTHORIZATION] = f"Bearer {api_key}"

            url = self._chat_completions_url()
            if urlparse(url).hostname == OPENROUTER_HOST:
                headers[HTTP_HEADER_OPENROUTER_REFERER] = OPENROUTER_APP_REFERER
                headers[HTTP_HEADER_OPENROUTER_TITLE] = OPENROUTER_APP_TITLE
            span.set_attribute("cuebridge.model_id", self.model_id)
            span.set_attribute("cuebridge.api_host", urlparse(url).netloc)
            span.set_attribute("cuebridge.input_messages", len(messages))
            span.set_attribute(
                "cuebridge.estimated_input_tokens", self.count_input_tokens(messages)
            )
            span.set_attribute("cuebridge.max_new_tokens", self.max_new_tokens)
            span.set_attribute("cuebridge.timeout_seconds", self.request_timeout_seconds)
            span.set_attribute("cuebridge.message_format", self._resolved_message_format())
            span.set_attribute("cuebridge.max_length_continuations", self.max_length_continuations)
            if self.reasoning_effort is not None:
                span.set_attribute("cuebridge.reasoning_effort", self.reasoning_effort)

            translated_parts: list[str] = []
            continuations_used = 0
            finish_reason: str | None = None
            while True:
                response = self._request_sender()(
                    url,
                    headers=headers,
                    json=self._build_payload(request_messages),
                    timeout=self.request_timeout_seconds,
                )
                span.set_attribute("cuebridge.http_status_code", response.status_code)
                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    detail = response.text.strip()
                    raise ValueError(
                        "OpenAI-compatible backend request failed with "
                        f"{response.status_code}: {detail}"
                    ) from exc

                data = response.json()
                choice = data[OPENAI_CHOICES_KEY][0]
                finish_reason = choice.get(OPENAI_FINISH_REASON_KEY)
                part = _message_to_text(choice[OPENAI_MESSAGE_KEY].get(CHAT_CONTENT_KEY, ""))
                if part:
                    translated_parts.append(part)

                if finish_reason != OPENAI_FINISH_REASON_LENGTH:
                    break
                if continuations_used >= self.max_length_continuations:
                    break
                if not part:
                    break

                request_messages.extend(
                    [
                        _chat_message(CHAT_ASSISTANT_ROLE, part),
                        _chat_message(CHAT_USER_ROLE, _build_length_continuation_prompt()),
                    ]
                )
                continuations_used += 1

            translated = "".join(translated_parts).strip()
            if finish_reason is not None:
                span.set_attribute("cuebridge.finish_reason", finish_reason)
            span.set_attribute("cuebridge.length_continuations", continuations_used)
            span.set_attribute("cuebridge.output_length", len(translated))
            return translated

    def _build_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            OPENAI_MODEL_KEY: self.model_id,
            OPENAI_MESSAGES_KEY: [*messages],
            OPENAI_TEMPERATURE_KEY: 0,
            OPENAI_MAX_TOKENS_KEY: self.max_new_tokens,
        }
        if self.reasoning_effort is not None:
            payload[OPENAI_REASONING_EFFORT_KEY] = self.reasoning_effort

        return payload

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
        if self.message_format != MESSAGE_FORMAT_AUTO:
            return self.message_format
        if MESSAGE_FORMAT_TRANSLATEGEMMA in self.model_id.lower():
            return MESSAGE_FORMAT_TRANSLATEGEMMA
        return MESSAGE_FORMAT_PLAIN


def _chat_message(role: str, content: Any) -> dict[str, Any]:
    return {CHAT_ROLE_KEY: role, CHAT_CONTENT_KEY: content}


def _build_translation_prompt(
    *,
    text: str,
    source_lang_code: str,
    target_lang_code: str,
) -> str:
    return (
        f"You are a professional translator from {source_lang_code} to {target_lang_code}. "
        f"Translate the text faithfully and naturally into {target_lang_code}. "
        "If the input contains segment markers like [[SEG_1]], preserve every marker exactly "
        "and in the same order; translate only the text after each marker. "
        "Return only the translation, with no commentary or explanations.\n\n"
        f"{text}"
    )


def _build_length_continuation_prompt() -> str:
    return (
        "Continue the previous translation exactly where it stopped. "
        "Do not repeat text that was already returned. "
        "If the previous response ended mid-word or mid-segment, continue from the next "
        "character. Preserve any [[SEG_n]] markers exactly. "
        "Return only the remaining translation."
    )


def _estimate_token_count(text: str) -> int:
    return max(1, (len(text) + 3) // 4)
