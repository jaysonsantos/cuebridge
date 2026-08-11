from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ModelPreset:
    """CLI defaults for a known model and provider combination."""

    backend: str
    model_id: str
    api_base_url: str | None = None
    api_key_env: str | None = None
    reasoning_effort: str | None = None
    max_new_tokens: int | None = None
    max_input_tokens: int | None = None
    window_size: int | None = None


MODEL_PRESETS: Final = {
    "deepseek-v4-flash": ModelPreset(
        backend="openai-compatible",
        model_id="~deepseek/deepseek-v4-flash-latest",
        api_base_url="https://openrouter.ai/api/v1",
        api_key_env="SUB_TRANSLATOR_API_KEY",
        reasoning_effort="high",
        max_new_tokens=4096,
        max_input_tokens=1_000_000,
        window_size=256,
    ),
}


def get_model_preset(name: str) -> ModelPreset:
    try:
        return MODEL_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model preset: {name}") from exc
