from __future__ import annotations

from pathlib import Path

from cuebridge.naming import build_output_path


def test_build_output_path_replaces_existing_language_suffix() -> None:
    output_path = build_output_path(Path("abc.de.srt"), "pt-BR")
    assert output_path == Path("abc.pt-BR.srt")


def test_build_output_path_removes_subtitle_variant_suffixes() -> None:
    output_path = build_output_path(Path("abc.de.hi.srt"), "pt-BR")
    assert output_path == Path("abc.pt-BR.srt")


def test_build_output_path_appends_language_suffix_when_missing() -> None:
    output_path = build_output_path(Path("abc.srt"), "pt-BR")
    assert output_path == Path("abc.pt-BR.srt")
