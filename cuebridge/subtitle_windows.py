from __future__ import annotations

import re
from dataclasses import dataclass

from loguru import logger
from opentelemetry import trace

from cuebridge.cancellation import CancellationToken
from cuebridge.contracts import TextTranslator

SEGMENT_MARKER_RE = re.compile(r"\[\[SEG_(\d+)]]")
TRACER = trace.get_tracer(__name__)


@dataclass(frozen=True)
class WindowTranslationResult:
    texts: list[str]
    had_retry: bool
    cancelled: bool = False


@dataclass
class AdaptiveSubtitleWindowTranslator:
    translator: TextTranslator
    window_size: int
    cancellation_token: CancellationToken | None = None
    consecutive_window_failures: int = 0

    def translate(self, texts: list[str]) -> WindowTranslationResult:
        attempted_window_size = len(texts)
        result = translate_subtitle_window(
            texts=texts,
            translator=self.translator,
            cancellation_token=self.cancellation_token,
        )
        self._record_result(
            attempted_window_size=attempted_window_size,
            result=result,
        )
        return result

    def _record_result(
        self,
        *,
        attempted_window_size: int,
        result: WindowTranslationResult,
    ) -> None:
        if attempted_window_size != self.window_size or self.window_size <= 1:
            return

        if not result.had_retry:
            self.consecutive_window_failures = 0
            return

        self.consecutive_window_failures += 1
        if self.consecutive_window_failures < 2:
            return

        new_window_size = _next_smaller_window_size(self.window_size)
        logger.info(
            "Reducing adaptive subtitle window size from {} to {} after repeated marker retries",
            self.window_size,
            new_window_size,
        )
        self.window_size = new_window_size
        self.consecutive_window_failures = 0


@TRACER.start_as_current_span("cuebridge.subtitle_windows.translate_event_window")
def translate_event_window(
    *,
    chunk: list[tuple[object, str]],
    translator: TextTranslator,
    cancellation_token: CancellationToken | None = None,
) -> list[str] | None:
    result = translate_subtitle_window(
        texts=[text for _event, text in chunk],
        translator=translator,
        cancellation_token=cancellation_token,
    )
    if result.cancelled:
        return None

    return result.texts


def translate_subtitle_window(
    *,
    texts: list[str],
    translator: TextTranslator,
    cancellation_token: CancellationToken | None = None,
) -> WindowTranslationResult:
    if cancellation_token is not None and cancellation_token.cancelled:
        logger.debug("Skipping subtitle backend call because cancellation was already requested")
        return WindowTranslationResult(texts=[], had_retry=False, cancelled=True)

    if len(texts) == 1:
        return WindowTranslationResult(
            texts=[translator.translate_text(texts[0], cancellation_token=cancellation_token)],
            had_retry=False,
        )

    prompt = build_window_prompt(texts)
    if cancellation_token is not None and cancellation_token.cancelled:
        logger.debug("Skipping subtitle backend call because cancellation was already requested")
        return WindowTranslationResult(texts=[], had_retry=False, cancelled=True)

    translated = translator.translate_text(prompt, cancellation_token=cancellation_token)
    segments = parse_window_translation(translated, expected_segments=len(texts))
    if segments is not None:
        return WindowTranslationResult(texts=segments, had_retry=False)

    if cancellation_token is not None and cancellation_token.cancelled:
        logger.debug("Skipping smaller-window retry for cancelled subtitle window")
        return WindowTranslationResult(texts=[], had_retry=False, cancelled=True)

    split_at = len(texts) // 2
    logger.debug(
        "Window translation markers did not round-trip cleanly for {} events; retrying smaller windows",
        len(texts),
    )
    left = translate_subtitle_window(
        texts=texts[:split_at],
        translator=translator,
        cancellation_token=cancellation_token,
    )
    if left.cancelled:
        return WindowTranslationResult(texts=[], had_retry=True, cancelled=True)

    right = translate_subtitle_window(
        texts=texts[split_at:],
        translator=translator,
        cancellation_token=cancellation_token,
    )
    if right.cancelled:
        return WindowTranslationResult(texts=[], had_retry=True, cancelled=True)

    return WindowTranslationResult(
        texts=[*left.texts, *right.texts],
        had_retry=True,
    )


def build_window_prompt(texts: list[str]) -> str:
    parts: list[str] = []
    for idx, text in enumerate(texts, start=1):
        parts.append(f"[[SEG_{idx}]]")
        parts.append(text)
    return "\n".join(parts)


@TRACER.start_as_current_span("cuebridge.subtitle_windows.parse_window_translation")
def parse_window_translation(translated_text: str, *, expected_segments: int) -> list[str] | None:
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


def _next_smaller_window_size(window_size: int) -> int:
    if window_size <= 2:
        return 1

    return (window_size + 1) // 2
