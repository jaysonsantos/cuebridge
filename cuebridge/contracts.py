from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

from cuebridge.cancellation import CancellationToken


@dataclass(frozen=True, slots=True)
class TranslationChunk:
    """Append-only translation fragment yielded in stream order."""

    text: str


TranslationStatus: TypeAlias = Literal["translated", "flushed", "completed", "cancelled"]


@dataclass(frozen=True, slots=True)
class TranslationCheckpoint:
    output_path: Path
    translated_events: int
    translated_chunks: int
    persisted: bool


@dataclass(frozen=True, slots=True)
class TranslationEvent:
    status: TranslationStatus
    output_path: Path
    translated_events: int
    translated_chunks: int
    cue_index: int | None = None
    cue_range: tuple[int, int] | None = None
    source_text: str | None = None
    translated_text: str | None = None
    checkpoint: TranslationCheckpoint | None = None
    cancellation_reason: str | None = None


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
