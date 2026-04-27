from __future__ import annotations

import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pysubs2
from loguru import logger
from opentelemetry import trace
from tqdm import tqdm

from cuebridge.cancellation import CancellationToken
from cuebridge.contracts import (
    TextTranslator,
    TranslationCheckpoint,
    TranslationEvent,
    TranslationStatus,
)
from cuebridge.naming import build_output_path
from cuebridge.subtitle_windows import AdaptiveSubtitleWindowTranslator

TRACER = trace.get_tracer(__name__)


@dataclass(frozen=True)
class TranslationResult:
    output_path: Path
    translated_events: int
    status: TranslationStatus = "completed"
    cancellation_reason: str | None = None


def translate_subtitle_file(
    *,
    input_path: Path,
    target_lang_code: str,
    translator: TextTranslator,
    window_size: int = 4,
    flush_every_chunks: int = 1,
    output_path: Path | None = None,
    cancellation_token: CancellationToken | None = None,
) -> TranslationResult:
    final_event: TranslationEvent | None = None

    resolved_output_path = build_output_path(
        input_path=input_path,
        target_lang_code=target_lang_code,
        output_path=output_path,
    )
    subtitles = pysubs2.load(str(input_path))
    translatable_events = [
        (index, event, decoded_text)
        for index, event in enumerate(subtitles, start=1)
        if (decoded_text := _decode_subtitle_text(getattr(event, "text", ""))).strip()
    ]

    if window_size < 1:
        raise ValueError(f"window_size must be at least 1, got {window_size}")

    with tqdm(
        total=len(translatable_events),
        desc="Translating",
        unit="line",
        dynamic_ncols=True,
    ) as progress:
        for event in iter_translate_subtitles(
            subtitles=subtitles,
            output_path=resolved_output_path,
            translator=translator,
            window_size=window_size,
            flush_every_chunks=flush_every_chunks,
            cancellation_token=cancellation_token,
        ):
            if event.status == "translated":
                progress.update(1)
            elif event.status in {"completed", "cancelled"}:
                final_event = event

    if final_event is None:
        raise RuntimeError("Subtitle translation ended without a terminal event")

    return TranslationResult(
        output_path=resolved_output_path,
        translated_events=final_event.translated_events,
        status=final_event.status,
        cancellation_reason=final_event.cancellation_reason,
    )


def iter_translate_subtitles(
    *,
    subtitles: pysubs2.SSAFile,
    output_path: Path,
    translator: TextTranslator,
    window_size: int = 4,
    flush_every_chunks: int = 1,
    cancellation_token: CancellationToken | None = None,
) -> Iterator[TranslationEvent]:
    # Keep the span open for the full generator iteration, not just generator creation.
    with TRACER.start_as_current_span("cuebridge.subtitles.iter_translate_subtitles"):
        if window_size < 1:
            raise ValueError(f"window_size must be at least 1, got {window_size}")

        translated_events = 0
        translated_chunks = 0
        last_flushed_state: tuple[int, int] | None = None
        window_translator = AdaptiveSubtitleWindowTranslator(
            translator=translator,
            window_size=window_size,
            cancellation_token=cancellation_token,
        )
        translatable_events = [
            (index, event, decoded_text)
            for index, event in enumerate(subtitles, start=1)
            if (decoded_text := _decode_subtitle_text(getattr(event, "text", ""))).strip()
        ]

        chunk_start = 0
        while chunk_start < len(translatable_events):
            if cancellation_token is not None and cancellation_token.cancelled:
                logger.info("Stopping subtitle translation early due to cancellation request")
                flush_event, last_flushed_state = _flush_event_if_needed(
                    subtitles=subtitles,
                    output_path=output_path,
                    translated_events=translated_events,
                    translated_chunks=translated_chunks,
                    last_flushed_state=last_flushed_state,
                )
                if flush_event is not None:
                    yield flush_event
                yield _terminal_event(
                    status="cancelled",
                    output_path=output_path,
                    translated_events=translated_events,
                    translated_chunks=translated_chunks,
                    cancellation_token=cancellation_token,
                )
                return

            chunk = translatable_events[chunk_start : chunk_start + window_translator.window_size]
            window_result = window_translator.translate(
                [source_text for _index, _event, source_text in chunk]
            )
            if window_result.cancelled:
                logger.info(
                    "Discarding partially cancelled subtitle window without overwriting text"
                )
                flush_event, last_flushed_state = _flush_event_if_needed(
                    subtitles=subtitles,
                    output_path=output_path,
                    translated_events=translated_events,
                    translated_chunks=translated_chunks,
                    last_flushed_state=last_flushed_state,
                )
                if flush_event is not None:
                    yield flush_event
                yield _terminal_event(
                    status="cancelled",
                    output_path=output_path,
                    translated_events=translated_events,
                    translated_chunks=translated_chunks,
                    cancellation_token=cancellation_token,
                )
                return

            cue_range = (chunk[0][0], chunk[-1][0])
            for (cue_index, event, source_text), translated_text in zip(
                chunk, window_result.texts, strict=True
            ):
                translated_events += 1
                logger.debug("Translated subtitle event {}", translated_events)
                event.text = _encode_subtitle_text(translated_text)
                yield TranslationEvent(
                    status="translated",
                    output_path=output_path,
                    translated_events=translated_events,
                    translated_chunks=translated_chunks,
                    cue_index=cue_index,
                    cue_range=cue_range,
                    source_text=source_text,
                    translated_text=translated_text,
                )

            translated_chunks += 1

            if translated_chunks % flush_every_chunks == 0:
                flush_event, last_flushed_state = _flush_event_if_needed(
                    subtitles=subtitles,
                    output_path=output_path,
                    translated_events=translated_events,
                    translated_chunks=translated_chunks,
                    last_flushed_state=last_flushed_state,
                )
                if flush_event is not None:
                    yield flush_event
            chunk_start += len(chunk)

        logger.info(
            "Saving {} translated subtitle events to {}",
            translated_events,
            output_path,
        )
        flush_event, last_flushed_state = _flush_event_if_needed(
            subtitles=subtitles,
            output_path=output_path,
            translated_events=translated_events,
            translated_chunks=translated_chunks,
            last_flushed_state=last_flushed_state,
        )
        if flush_event is not None:
            yield flush_event
        yield _terminal_event(
            status="completed",
            output_path=output_path,
            translated_events=translated_events,
            translated_chunks=translated_chunks,
            cancellation_token=cancellation_token,
        )


def _decode_subtitle_text(text: str) -> str:
    return text.replace(r"\N", "\n").replace(r"\n", "\n")


def _encode_subtitle_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", r"\N")


@TRACER.start_as_current_span("cuebridge.subtitles.save_subtitles_atomic")
def _save_subtitles_atomic(subtitles: pysubs2.SSAFile, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=output_path.suffix,
        prefix=f"{output_path.stem}.",
        dir=output_path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        temp_path = Path(handle.name)

    try:
        subtitles.save(str(temp_path))
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


@TRACER.start_as_current_span("cuebridge.subtitles.flush_event")
def _flush_event(
    *,
    subtitles: pysubs2.SSAFile,
    output_path: Path,
    translated_events: int,
    translated_chunks: int,
) -> TranslationEvent:
    _save_subtitles_atomic(subtitles, output_path)
    logger.debug("Flushed partial subtitle output to {}", output_path)
    return TranslationEvent(
        status="flushed",
        output_path=output_path,
        translated_events=translated_events,
        translated_chunks=translated_chunks,
        checkpoint=TranslationCheckpoint(
            output_path=output_path,
            translated_events=translated_events,
            translated_chunks=translated_chunks,
            persisted=True,
        ),
    )


def _flush_event_if_needed(
    *,
    subtitles: pysubs2.SSAFile,
    output_path: Path,
    translated_events: int,
    translated_chunks: int,
    last_flushed_state: tuple[int, int] | None,
) -> tuple[TranslationEvent | None, tuple[int, int]]:
    current_state = (translated_events, translated_chunks)
    if last_flushed_state == current_state and output_path.exists():
        return None, current_state

    return (
        _flush_event(
            subtitles=subtitles,
            output_path=output_path,
            translated_events=translated_events,
            translated_chunks=translated_chunks,
        ),
        current_state,
    )


def _terminal_event(
    *,
    status: TranslationStatus,
    output_path: Path,
    translated_events: int,
    translated_chunks: int,
    cancellation_token: CancellationToken | None,
) -> TranslationEvent:
    return TranslationEvent(
        status=status,
        output_path=output_path,
        translated_events=translated_events,
        translated_chunks=translated_chunks,
        checkpoint=TranslationCheckpoint(
            output_path=output_path,
            translated_events=translated_events,
            translated_chunks=translated_chunks,
            persisted=True,
        ),
        cancellation_reason=(
            cancellation_token.reason
            if status == "cancelled" and cancellation_token is not None
            else None
        ),
    )
