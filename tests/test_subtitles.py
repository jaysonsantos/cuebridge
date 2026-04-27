from __future__ import annotations

from pathlib import Path

import pysubs2
from cuebridge.cancellation import CancellationToken
from cuebridge.contracts import TranslationEvent
from cuebridge.subtitles import (
    iter_translate_subtitles,
    translate_subtitle_file,
)


class FakeTranslator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[str] = []

    def translate_text(self, text: str, cancellation_token: CancellationToken | None = None) -> str:
        del cancellation_token
        self.calls.append(text)
        return self.responses.pop(0)


class AdaptiveFakeTranslator:
    def __init__(self, *, max_marker_window: int) -> None:
        self.max_marker_window = max_marker_window
        self.window_calls: list[int] = []

    def translate_text(self, text: str, cancellation_token: CancellationToken | None = None) -> str:
        del cancellation_token
        marker_count = text.count("[[SEG_")
        if marker_count == 0:
            return f"[pt-BR] {text}"

        self.window_calls.append(marker_count)
        if marker_count > self.max_marker_window:
            return "broken translation without markers"

        parts: list[str] = []
        for idx in range(1, marker_count + 1):
            parts.append(f"[[SEG_{idx}]][pt-BR] segment {idx}")
        return "\n".join(parts)


def test_translate_subtitle_file_stops_before_starting_next_chunk(tmp_path: Path) -> None:
    input_path = tmp_path / "movie.en.srt"
    input_path.write_text(
        """1
00:00:01,000 --> 00:00:02,500
Hello there!

2
00:00:03,000 --> 00:00:04,500
How are you?
""",
        encoding="utf-8",
    )

    token = CancellationToken()

    class CancellingTranslator:
        def __init__(self) -> None:
            self.calls = 0

        def translate_text(
            self,
            text: str,
            cancellation_token: CancellationToken | None = None,
        ) -> str:
            self.calls += 1
            if cancellation_token is not None:
                cancellation_token.cancel("stop after current chunk")
            return f"[pt-BR] {text}"

    result = translate_subtitle_file(
        input_path=input_path,
        target_lang_code="pt-BR",
        translator=CancellingTranslator(),
        window_size=1,
        flush_every_chunks=1,
        cancellation_token=token,
    )

    translated = pysubs2.load(str(result.output_path))

    assert token.cancelled is True
    assert result.translated_events == 1
    assert result.status == "cancelled"
    assert result.cancellation_reason == "stop after current chunk"
    assert translated[0].text == "[pt-BR] Hello there!"
    assert translated[1].text == "How are you?"


def test_iter_translate_subtitles_emits_translation_flush_and_terminal_events(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "movie.en.srt"
    input_path.write_text(
        """1
00:00:01,000 --> 00:00:02,500
Hello there!

2
00:00:03,000 --> 00:00:04,500
How are you?
""",
        encoding="utf-8",
    )

    subtitles = pysubs2.load(str(input_path))
    output_path = tmp_path / "out.pt-BR.srt"
    translator = FakeTranslator(["[pt-BR] Hello there!", "[pt-BR] How are you?"])

    events = list(
        iter_translate_subtitles(
            subtitles=subtitles,
            output_path=output_path,
            translator=translator,
            window_size=1,
            flush_every_chunks=2,
        )
    )

    assert [event.status for event in events] == [
        "translated",
        "translated",
        "flushed",
        "completed",
    ]
    translated_events = [event for event in events if event.status == "translated"]
    assert translated_events == [
        TranslationEvent(
            status="translated",
            output_path=output_path,
            translated_events=1,
            translated_chunks=0,
            cue_index=1,
            cue_range=(1, 1),
            source_text="Hello there!",
            translated_text="[pt-BR] Hello there!",
        ),
        TranslationEvent(
            status="translated",
            output_path=output_path,
            translated_events=2,
            translated_chunks=1,
            cue_index=2,
            cue_range=(2, 2),
            source_text="How are you?",
            translated_text="[pt-BR] How are you?",
        ),
    ]
    assert events[-2].checkpoint is not None
    assert events[-2].checkpoint.translated_events == 2
    assert events[-1].checkpoint is not None
    assert events[-1].checkpoint.translated_chunks == 2


def test_iter_translate_subtitles_flushes_before_terminal_cancelled_event(tmp_path: Path) -> None:
    input_path = tmp_path / "movie.en.srt"
    input_path.write_text(
        """1
00:00:01,000 --> 00:00:02,500
Hello there!

2
00:00:03,000 --> 00:00:04,500
How are you?
""",
        encoding="utf-8",
    )

    subtitles = pysubs2.load(str(input_path))
    output_path = tmp_path / "out.pt-BR.srt"
    token = CancellationToken()

    class CancellingTranslator:
        def __init__(self) -> None:
            self.calls = 0

        def translate_text(
            self,
            text: str,
            cancellation_token: CancellationToken | None = None,
        ) -> str:
            self.calls += 1
            if cancellation_token is not None:
                cancellation_token.cancel("stop after current chunk")
            return f"[pt-BR] {text}"

    events = list(
        iter_translate_subtitles(
            subtitles=subtitles,
            output_path=output_path,
            translator=CancellingTranslator(),
            window_size=1,
            flush_every_chunks=10,
            cancellation_token=token,
        )
    )

    assert [event.status for event in events] == ["translated", "flushed", "cancelled"]
    assert events[-1].cancellation_reason == "stop after current chunk"

    translated = pysubs2.load(str(output_path))
    assert translated[0].text == "[pt-BR] Hello there!"
    assert translated[1].text == "How are you?"


def test_translate_subtitle_file_reduces_future_window_size_after_repeated_failures(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "movie.en.srt"
    subtitle_lines = []
    for idx in range(1, 19):
        subtitle_lines.append(
            f"""{idx}
00:00:{idx:02d},000 --> 00:00:{idx:02d},500
Line {idx}
"""
        )
    input_path.write_text("\n".join(subtitle_lines), encoding="utf-8")

    translator = AdaptiveFakeTranslator(max_marker_window=3)
    result = translate_subtitle_file(
        input_path=input_path,
        target_lang_code="pt-BR",
        translator=translator,
        window_size=6,
        output_path=tmp_path / "out.pt-BR.srt",
    )

    assert result.translated_events == 18
    assert translator.window_calls == [6, 3, 3, 6, 3, 3, 3, 3]

    translated = pysubs2.load(str(result.output_path))
    assert translated[0].text == "[pt-BR] segment 1"


def test_translate_subtitle_file_rejects_non_positive_window_size(tmp_path: Path) -> None:
    input_path = tmp_path / "movie.en.srt"
    input_path.write_text(
        """1
00:00:01,000 --> 00:00:02,500
Hello there!
""",
        encoding="utf-8",
    )

    try:
        translate_subtitle_file(
            input_path=input_path,
            target_lang_code="pt-BR",
            translator=FakeTranslator(["[pt-BR] Hello there!"]),
            window_size=0,
        )
    except ValueError as exc:
        assert str(exc) == "window_size must be at least 1, got 0"
    else:
        raise AssertionError("Expected ValueError for non-positive window size")
