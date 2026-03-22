from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol
from uuid import uuid4

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from cuebridge.contracts import TextTranslator
from cuebridge.model import OpenAICompatibleChatModel, TranslateGemmaChatModel


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
        messages = state["messages"]
        trimmed = trim_messages_to_token_budget(
            messages=messages,
            token_counter=token_counter,
            max_input_tokens=max_input_tokens,
        )
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
        self._config: RunnableConfig = {"configurable": {"thread_id": thread_id or str(uuid4())}}

    def translate_text(self, text: str) -> str:
        response = self._agent.invoke({"messages": text}, self._config)
        message = response["messages"][-1]

        if isinstance(message, AIMessage):
            return _message_text(message.content)

        raise TypeError(f"Expected AIMessage, got {type(message)!r}")


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
    api_key_env: str = "OPENAI_API_KEY",
    request_timeout_seconds: float = 120.0,
    message_format: str = "auto",
    max_input_tokens: int = 1800,
    thread_id: str | None = None,
) -> TextTranslator:
    if backend == "hf-local":
        model: SupportsTokenCounting = TranslateGemmaChatModel(
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
            model_id=model_id,
            dtype=dtype,
            device=device,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
    elif backend == "openai-compatible":
        model = OpenAICompatibleChatModel(
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
            model_id=model_id,
            api_base_url=api_base_url or "http://localhost:1234/v1",
            api_key=api_key,
            api_key_env=api_key_env,
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
    )


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
