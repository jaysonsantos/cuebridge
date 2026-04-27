from __future__ import annotations

from cuebridge.cancellation import CancellationToken
from cuebridge.subtitle_windows import (
    AdaptiveSubtitleWindowTranslator,
    build_window_prompt,
    parse_window_translation,
    translate_event_window,
)


class FakeTranslator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def translate_text(self, text: str, cancellation_token: CancellationToken | None = None) -> str:
        del cancellation_token
        self.calls.append(text)
        return self.responses.pop(0)


def test_parse_window_translation_splits_segments() -> None:
    translated = "[[SEG_1]]First line\n[[SEG_2]]Second line"
    assert parse_window_translation(translated, expected_segments=2) == [
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
    assert translator.calls[0] == build_window_prompt(["eins", "zwei"])


def test_translate_event_window_recursively_splits_large_broken_windows() -> None:
    translator = FakeTranslator(
        [
            "broken translation without markers",
            "[[SEG_1]]one\n[[SEG_2]]two",
            "[[SEG_1]]three\n[[SEG_2]]four",
        ]
    )
    chunk = [(object(), "eins"), (object(), "zwei"), (object(), "drei"), (object(), "vier")]

    result = translate_event_window(chunk=chunk, translator=translator)

    assert result == ["one", "two", "three", "four"]
    assert translator.calls[0] == build_window_prompt(["eins", "zwei", "drei", "vier"])
    assert translator.calls[1] == build_window_prompt(["eins", "zwei"])
    assert translator.calls[2] == build_window_prompt(["drei", "vier"])


def test_translate_event_window_uses_marked_segments() -> None:
    translator = FakeTranslator(
        [
            "[[SEG_1]]Hello\n[[SEG_2]]World",
        ]
    )
    chunk = [(object(), "Hallo"), (object(), "Welt")]

    result = translate_event_window(chunk=chunk, translator=translator)

    assert result == ["Hello", "World"]


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
    assert translator.calls == [build_window_prompt(["eins", "zwei"])]


def test_adaptive_window_translator_reduces_size_after_repeated_marker_failures() -> None:
    translator = FakeTranslator(
        [
            "broken translation without markers",
            "one",
            "two",
            "broken translation without markers",
            "three",
            "four",
        ]
    )
    window_translator = AdaptiveSubtitleWindowTranslator(
        translator=translator,
        window_size=2,
    )

    first = window_translator.translate(["eins", "zwei"])
    second = window_translator.translate(["drei", "vier"])

    assert first.texts == ["one", "two"]
    assert second.texts == ["three", "four"]
    assert window_translator.window_size == 1
