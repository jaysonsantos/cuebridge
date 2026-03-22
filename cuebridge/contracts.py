from __future__ import annotations

from typing import Protocol


class TextTranslator(Protocol):
    def translate_text(self, text: str) -> str: ...
