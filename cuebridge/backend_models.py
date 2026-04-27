from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import BaseMessage

from cuebridge.model import OpenAICompatibleChatModel, TranslateGemmaChatModel

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


@dataclass(frozen=True, slots=True)
class BackendModelConfig:
    source_lang_code: str
    target_lang_code: str
    model_id: str
    backend: str = "hf-local"
    dtype: str = "bfloat16"
    device: str | None = None
    max_new_tokens: int = 256
    batch_size: int = 1
    api_base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    request_timeout_seconds: float = 120.0
    reasoning_effort: str | None = None
    message_format: str = "auto"


def build_backend_model(config: BackendModelConfig) -> SupportsTokenCounting:
    backend_name = config.backend.lower()

    if backend_name == "hf-local":
        return TranslateGemmaChatModel(
            source_lang_code=config.source_lang_code,
            target_lang_code=config.target_lang_code,
            model_id=config.model_id,
            dtype=config.dtype,
            device=config.device,
            max_new_tokens=config.max_new_tokens,
            batch_size=config.batch_size,
        )

    if backend_name in OPENAI_COMPATIBLE_BACKEND_DEFAULTS:
        resolved_backend = _resolve_openai_compatible_backend(
            backend=backend_name,
            api_base_url=config.api_base_url,
            api_key_env=config.api_key_env,
        )
        return OpenAICompatibleChatModel(
            source_lang_code=config.source_lang_code,
            target_lang_code=config.target_lang_code,
            model_id=config.model_id,
            api_base_url=resolved_backend["api_base_url"],
            api_key=config.api_key,
            api_key_env=resolved_backend["api_key_env"],
            request_timeout_seconds=config.request_timeout_seconds,
            reasoning_effort=config.reasoning_effort,
            message_format=config.message_format,
            max_new_tokens=config.max_new_tokens,
        )

    raise ValueError(f"Unsupported backend: {config.backend}")


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
