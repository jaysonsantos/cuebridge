from __future__ import annotations

from cuebridge.subtitles import (
    _build_window_prompt,
    _parse_window_translation,
    translate_event_window,
)


class FakeTranslator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[str] = []

    def translate_text(self, text: str) -> str:
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
