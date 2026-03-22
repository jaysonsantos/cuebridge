from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pysubs2
from loguru import logger
from tqdm import tqdm

from cuebridge.contracts import TextTranslator
from cuebridge.naming import build_output_path

SEGMENT_MARKER_RE = re.compile(r"\[\[SEG_(\d+)]]")


@dataclass(frozen=True)
class TranslationResult:
    output_path: Path
    translated_events: int


def translate_subtitle_file(
    *,
    input_path: Path,
    target_lang_code: str,
    translator: TextTranslator,
    window_size: int = 4,
    flush_every_chunks: int = 1,
    output_path: Path | None = None,
) -> TranslationResult:
    subtitles = pysubs2.load(str(input_path))
    translated_events = 0
    translated_chunks = 0
    translatable_events = [
        (event, _decode_subtitle_text(getattr(event, "text", "")))
        for event in subtitles
        if _decode_subtitle_text(getattr(event, "text", "")).strip()
    ]

    resolved_output_path = build_output_path(
        input_path=input_path,
        target_lang_code=target_lang_code,
        output_path=output_path,
    )

    with tqdm(
        total=len(translatable_events),
        desc="Translating",
        unit="line",
        dynamic_ncols=True,
    ) as progress:
        for chunk_start in range(0, len(translatable_events), window_size):
            chunk = translatable_events[chunk_start : chunk_start + window_size]
            translated_texts = translate_event_window(chunk=chunk, translator=translator)
            for (event, _source_text), translated_text in zip(chunk, translated_texts, strict=True):
                translated_events += 1
                logger.debug("Translated subtitle event {}", translated_events)
                event.text = _encode_subtitle_text(translated_text)
            translated_chunks += 1
            progress.update(len(chunk))
            if translated_chunks % flush_every_chunks == 0:
                _save_subtitles_atomic(subtitles, resolved_output_path)
                logger.debug("Flushed partial subtitle output to {}", resolved_output_path)

    logger.info(
        "Saving {} translated subtitle events to {}",
        translated_events,
        resolved_output_path,
    )
    _save_subtitles_atomic(subtitles, resolved_output_path)
    return TranslationResult(
        output_path=resolved_output_path,
        translated_events=translated_events,
    )


def _decode_subtitle_text(text: str) -> str:
    return text.replace(r"\N", "\n").replace(r"\n", "\n")


def _encode_subtitle_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", r"\N")


def translate_event_window(
    *,
    chunk: list[tuple[object, str]],
    translator: TextTranslator,
) -> list[str]:
    if len(chunk) == 1:
        return [translator.translate_text(chunk[0][1])]

    prompt = _build_window_prompt([text for _event, text in chunk])
    translated = translator.translate_text(prompt)
    segments = _parse_window_translation(translated, expected_segments=len(chunk))
    if segments is not None:
        return segments

    logger.debug(
        "Window translation markers did not round-trip cleanly; falling back to single-event translation"
    )
    return [translator.translate_text(text) for _event, text in chunk]


def _build_window_prompt(texts: list[str]) -> str:
    parts: list[str] = []
    for idx, text in enumerate(texts, start=1):
        parts.append(f"[[SEG_{idx}]]")
        parts.append(text)
    return "\n".join(parts)


def _parse_window_translation(translated_text: str, *, expected_segments: int) -> list[str] | None:
    matches = list(SEGMENT_MARKER_RE.finditer(translated_text))
    if len(matches) != expected_segments:
        return None

    segments: list[str] = []
    for idx, match in enumerate(matches):
        expected_number = idx + 1
        if int(match.group(1)) != expected_number:
            return None

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(translated_text)
        segment_text = translated_text[start:end].strip()
        segments.append(segment_text)

    if any(not segment for segment in segments):
        return None

    return segments


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
