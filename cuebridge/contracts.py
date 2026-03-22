from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from cuebridge.cancellation import CancellationToken


@dataclass(frozen=True, slots=True)
class TranslationChunk:
    """Append-only translation fragment yielded in stream order."""

    text: str


class TextTranslator(Protocol):
    def translate_text(
        self,
        text: str,
        cancellation_token: CancellationToken | None = None,
    ) -> str: ...


class StreamingTextTranslator(TextTranslator, Protocol):
    """Streaming translation with best-effort cancellation.

    Backends should yield partial chunks in order when possible. Bridge adapters may
    yield a single final chunk when the underlying backend only supports one-shot
    generation. Cancellation is cooperative: callers should expect the current backend
    request to finish before iteration stops, so partial chunks may already have been
    yielded when cancellation is observed.
    """

    def translate_text_stream(
        self,
        text: str,
        cancellation_token: CancellationToken | None = None,
    ) -> Iterator[TranslationChunk]: ...


def collect_translation_text(chunks: Iterable[TranslationChunk]) -> str:
    return "".join(chunk.text for chunk in chunks)
