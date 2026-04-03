from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Protocol
from uuid import uuid4

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from opentelemetry import trace

from cuebridge.cancellation import CancellationToken
from cuebridge.contracts import (
    StreamingTextTranslator,
    TranslationChunk,
    collect_translation_text,
)
from cuebridge.model import OpenAICompatibleChatModel, TranslateGemmaChatModel

TRACER = trace.get_tracer(__name__)
OPENAI_COMPATIBLE_BACKEND_DEFAULTS = {
    "openai-compatible": {
        "api_base_url": "http://localhost:1234/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "cerebras": {
        "api_base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
    },
    "openrouter": {
        "api_base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}


class SupportsTokenCounting(Protocol):
    def count_input_tokens(self, messages: list[BaseMessage]) -> int: ...


def make_trim_messages_middleware(
    *,
    token_counter: Callable[[list[BaseMessage]], int],
    max_input_tokens: int,
) -> Any:
    @before_model
    def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        del runtime
        with TRACER.start_as_current_span("cuebridge.agent.trim_messages") as span:
            messages = state["messages"]
            before_count = len(messages)
            before_tokens = token_counter(messages) if messages else 0
            trimmed = trim_messages_to_token_budget(
                messages=messages,
                token_counter=token_counter,
                max_input_tokens=max_input_tokens,
            )
            span.set_attribute("cuebridge.messages.before_count", before_count)
            span.set_attribute("cuebridge.messages.before_tokens", before_tokens)
            span.set_attribute("cuebridge.max_input_tokens", max_input_tokens)
            span.set_attribute("cuebridge.messages.after_count", len(trimmed))
            span.set_attribute(
                "cuebridge.messages.after_tokens", token_counter(trimmed) if trimmed else 0
            )
            span.set_attribute("cuebridge.messages.trimmed", trimmed != messages)
            if trimmed == messages:
                return None

            return {
                "messages": [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *trimmed,
                ]
            }

    return trim_messages


def trim_messages_to_token_budget(
    *,
    messages: list[BaseMessage],
    token_counter: Callable[[list[BaseMessage]], int],
    max_input_tokens: int,
) -> list[BaseMessage]:
    if not messages or token_counter(messages) <= max_input_tokens:
        return messages

    best_fit: list[BaseMessage] | None = None
    for start in range(len(messages) - 1, -1, -1):
        candidate = messages[start:]
        if not isinstance(candidate[0], HumanMessage):
            continue

        if token_counter(candidate) <= max_input_tokens:
            best_fit = candidate
        elif best_fit is not None:
            break

    if best_fit is not None:
        return best_fit

    # Fall back to the current user turn even if it alone exceeds the budget.
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return [message]

    return messages[-1:]


class LangChainSubtitleTranslator:
    def __init__(
        self,
        model: SupportsTokenCounting,
        *,
        thread_id: str | None = None,
        max_input_tokens: int = 1800,
        retain_history: bool = False,
    ) -> None:
        self._agent = create_agent(
            model,
            tools=[],
            middleware=[
                make_trim_messages_middleware(
                    token_counter=model.count_input_tokens,
                    max_input_tokens=max_input_tokens,
                )
            ],
            checkpointer=InMemorySaver(),
        )
        self._thread_id = thread_id or str(uuid4())
        self._retain_history = retain_history

    def _request_config(self) -> RunnableConfig:
        return {
            "configurable": {"thread_id": self._thread_id if self._retain_history else str(uuid4())}
        }

    def translate_text(
        self,
        text: str,
        cancellation_token: CancellationToken | None = None,
    ) -> str:
        with TRACER.start_as_current_span("cuebridge.agent.translate_text") as span:
            span.set_attribute("cuebridge.input_length", len(text))
            span.set_attribute("cuebridge.retain_history", self._retain_history)
            span.set_attribute("cuebridge.cancellation_requested", bool(cancellation_token))
            if cancellation_token is not None and cancellation_token.cancelled:
                span.set_attribute("cuebridge.cancelled_before_start", True)
                return ""

            translated = collect_translation_text(
                self.translate_text_stream(text, cancellation_token=cancellation_token)
            )
            span.set_attribute("cuebridge.stream_output_length", len(translated))
            if translated or (cancellation_token is not None and cancellation_token.cancelled):
                span.set_attribute("cuebridge.used_invoke_fallback", False)
                span.set_attribute("cuebridge.output_length", len(translated))
                return translated

            span.set_attribute("cuebridge.used_invoke_fallback", True)
            response = self._agent.invoke({"messages": text}, self._request_config())
            message = response["messages"][-1]
            if isinstance(message, AIMessage):
                resolved = _message_text(message.content)
                span.set_attribute("cuebridge.output_length", len(resolved))
                return resolved

            raise TypeError("Expected streamed translation output from LangChain agent")

    def translate_text_stream(
        self,
        text: str,
        cancellation_token: CancellationToken | None = None,
    ) -> Iterator[TranslationChunk]:
        if cancellation_token is not None and cancellation_token.cancelled:
            return

        if not hasattr(self._agent, "stream"):
            return

        for event in self._agent.stream(
            {"messages": text},
            self._request_config(),
            stream_mode="messages",
        ):
            if cancellation_token is not None and cancellation_token.cancelled:
                return

            if not isinstance(event, tuple) or len(event) != 2:
                continue

            message, _metadata = event
            if not isinstance(message, AIMessage | AIMessageChunk):
                continue

            chunk_text = _message_text(message.content)
            if chunk_text:
                yield TranslationChunk(text=chunk_text)


def build_subtitle_translator(
    *,
    source_lang_code: str,
    target_lang_code: str,
    model_id: str,
    backend: str = "hf-local",
    dtype: str,
    device: str | None,
    max_new_tokens: int,
    batch_size: int,
    api_base_url: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    request_timeout_seconds: float = 120.0,
    message_format: str = "auto",
    max_input_tokens: int = 1800,
    thread_id: str | None = None,
    retain_history: bool = False,
) -> StreamingTextTranslator:
    backend_name = backend.lower()

    if backend_name == "hf-local":
        model: SupportsTokenCounting = TranslateGemmaChatModel(
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
            model_id=model_id,
            dtype=dtype,
            device=device,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
    elif backend_name in OPENAI_COMPATIBLE_BACKEND_DEFAULTS:
        resolved_backend = _resolve_openai_compatible_backend(
            backend=backend_name,
            api_base_url=api_base_url,
            api_key_env=api_key_env,
        )
        model = OpenAICompatibleChatModel(
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
            model_id=model_id,
            api_base_url=resolved_backend["api_base_url"],
            api_key=api_key,
            api_key_env=resolved_backend["api_key_env"],
            request_timeout_seconds=request_timeout_seconds,
            message_format=message_format,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return LangChainSubtitleTranslator(
        model,
        thread_id=thread_id,
        max_input_tokens=max_input_tokens,
        retain_history=retain_history,
    )


def _resolve_openai_compatible_backend(
    *,
    backend: str,
    api_base_url: str | None,
    api_key_env: str | None,
) -> dict[str, str]:
    defaults = OPENAI_COMPATIBLE_BACKEND_DEFAULTS[backend]
    return {
        "api_base_url": api_base_url or defaults["api_base_url"],
        "api_key_env": api_key_env or defaults["api_key_env"],
    }


def _message_text(content: str | list[dict[str, Any] | str]) -> str:
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
