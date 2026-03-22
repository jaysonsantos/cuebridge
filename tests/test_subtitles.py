from __future__ import annotations

from pathlib import Path

import pysubs2
from cuebridge.cancellation import CancellationToken
from cuebridge.subtitles import (
    _build_window_prompt,
    _parse_window_translation,
    translate_event_window,
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


def test_parse_window_translation_splits_segments() -> None:
    translated = "[[SEG_1]]First line\n[[SEG_2]]Second line"
    assert _parse_window_translation(translated, expected_segments=2) == [
        "First line",
        "Second line",
    ]


def test_translate_event_window_falls_back_when_markers_are_missing() -> None:
    translator = FakeTranslator(
        [
            "broken translation without markers",
            "one",
            "two",
        ]
    )
    chunk = [(object(), "eins"), (object(), "zwei")]

    result = translate_event_window(chunk=chunk, translator=translator)

    assert result == ["one", "two"]
    assert translator.calls[0] == _build_window_prompt(["eins", "zwei"])


def test_translate_event_window_uses_marked_segments() -> None:
    translator = FakeTranslator(
        [
            "[[SEG_1]]Hello\n[[SEG_2]]World",
        ]
    )
    chunk = [(object(), "Hallo"), (object(), "Welt")]

    result = translate_event_window(chunk=chunk, translator=translator)

    assert result == ["Hello", "World"]


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
    assert translated[0].text == "[pt-BR] Hello there!"
    assert translated[1].text == "How are you?"


def test_translate_event_window_skips_fallback_after_cancellation() -> None:
    class CancellingTranslator:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def translate_text(
            self,
            text: str,
            cancellation_token: CancellationToken | None = None,
        ) -> str:
            self.calls.append(text)
            if cancellation_token is not None:
                cancellation_token.cancel("cancel during window translation")
            return ""

    translator = CancellingTranslator()
    token = CancellationToken()
    chunk = [(object(), "eins"), (object(), "zwei")]

    result = translate_event_window(
        chunk=chunk,
        translator=translator,
        cancellation_token=token,
    )

    assert result is None
    assert translator.calls == [_build_window_prompt(["eins", "zwei"])]
