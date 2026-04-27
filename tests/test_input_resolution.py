from __future__ import annotations

import io
from pathlib import Path

from cuebridge import input_resolution
from cuebridge.input_resolution import resolve_subtitle_input
from cuebridge.media import SubtitleStreamInfo


def test_resolve_subtitle_input_uses_subtitle_path_without_output_override(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "movie.en.srt"
    input_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8")

    with resolve_subtitle_input(
        input_source=input_path,
        source_lang_code="en",
        target_lang_code="pt-BR",
    ) as resolved:
        assert resolved.input_path == input_path
        assert resolved.output_path is None


def test_resolve_subtitle_input_materializes_file_like_input(tmp_path: Path) -> None:
    input_source = io.StringIO("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
    input_source.name = "movie.en.srt"

    with resolve_subtitle_input(
        input_source=input_source,
        source_lang_code="en",
        target_lang_code="pt-BR",
    ) as resolved:
        assert resolved.input_path.name == "movie.en.srt"
        assert resolved.input_path.read_text(encoding="utf-8").endswith("Hello\n")
        assert resolved.output_path == Path("movie.pt-BR.srt")


def test_resolve_subtitle_input_extracts_video_subtitle_stream(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "episode.en.mkv"
    input_path.write_bytes(b"fake video payload")

    def fake_probe_subtitle_streams(_input_path: Path) -> list[SubtitleStreamInfo]:
        return [
            SubtitleStreamInfo(
                relative_index=0,
                stream_index=3,
                codec_name="subrip",
                language="eng",
                title=None,
                is_default=True,
                duration_seconds=120.0,
            )
        ]

    def fake_extract_text_subtitle_stream_to_srt(*, output_path: Path, **kwargs) -> None:
        del kwargs
        output_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8")

    monkeypatch.setattr(input_resolution, "probe_subtitle_streams", fake_probe_subtitle_streams)
    monkeypatch.setattr(
        input_resolution,
        "extract_text_subtitle_stream_to_srt",
        fake_extract_text_subtitle_stream_to_srt,
    )

    with resolve_subtitle_input(
        input_source=input_path,
        source_lang_code="en",
        target_lang_code="pt-BR",
    ) as resolved:
        assert resolved.input_path.suffix == ".srt"
        assert resolved.input_path.read_text(encoding="utf-8").endswith("Hello\n")
        assert resolved.output_path == tmp_path / "episode.pt-BR.srt"
