from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

from cuebridge.media import (
    extract_bitmap_subtitle_stream_to_srt,
    extract_text_subtitle_stream_to_srt,
    is_bitmap_subtitle_codec,
    is_subtitle_file_path,
    probe_subtitle_streams,
    select_subtitle_stream,
)
from cuebridge.naming import build_output_path

SubtitleInputSource = Path | str | TextIO | BinaryIO


@dataclass(frozen=True, slots=True)
class SubtitleInputResolution:
    input_path: Path
    output_path: Path | None


@contextmanager
def resolve_subtitle_input(
    *,
    input_source: SubtitleInputSource,
    source_lang_code: str,
    target_lang_code: str,
    output_path: Path | None = None,
    subtitle_stream: int | None = None,
    ocr_language: str | None = None,
) -> Iterator[SubtitleInputResolution]:
    resolved_output_path = _resolve_output_path(
        input_source=input_source,
        target_lang_code=target_lang_code,
        output_path=output_path,
    )

    with _resolved_input_path(
        input_source=input_source,
        source_lang_code=source_lang_code,
        subtitle_stream=subtitle_stream,
        ocr_language=ocr_language,
    ) as input_path:
        yield SubtitleInputResolution(
            input_path=input_path,
            output_path=resolved_output_path,
        )


@contextmanager
def _resolved_input_path(
    *,
    input_source: SubtitleInputSource,
    source_lang_code: str,
    subtitle_stream: int | None,
    ocr_language: str | None,
) -> Iterator[Path]:
    if isinstance(input_source, Path | str):
        input_path = Path(input_source)
        if is_subtitle_file_path(input_path):
            yield input_path
            return

        with tempfile.TemporaryDirectory(prefix="cuebridge-video-subtitles-") as tmp_dir:
            extracted_path = Path(tmp_dir) / f"{input_path.stem}.source.srt"
            selected_stream = select_subtitle_stream(
                streams=probe_subtitle_streams(input_path),
                source_lang_code=source_lang_code,
                preferred_stream_index=subtitle_stream,
            )
            if is_bitmap_subtitle_codec(selected_stream.codec_name):
                extract_bitmap_subtitle_stream_to_srt(
                    input_path=input_path,
                    stream=selected_stream,
                    output_path=extracted_path,
                    source_lang_code=source_lang_code,
                    ocr_language=ocr_language,
                )
            else:
                extract_text_subtitle_stream_to_srt(
                    input_path=input_path,
                    stream=selected_stream,
                    output_path=extracted_path,
                )
            yield extracted_path
            return

    filename = _input_filename(input_source)
    if not is_subtitle_file_path(Path(filename)):
        raise ValueError("Video input must be provided as a filesystem path")
    content = input_source.read()
    if isinstance(content, bytes):
        text = content.decode("utf-8")
    elif isinstance(content, str):
        text = content
    else:
        raise TypeError(f"Unsupported file-like input content type: {type(content)!r}")

    with tempfile.TemporaryDirectory(prefix="cuebridge-") as tmp_dir:
        input_path = Path(tmp_dir) / filename
        input_path.write_text(text, encoding="utf-8")
        yield input_path


def _resolve_output_path(
    *,
    input_source: SubtitleInputSource,
    target_lang_code: str,
    output_path: Path | None,
) -> Path | None:
    if output_path is not None:
        return output_path

    if isinstance(input_source, Path | str):
        input_path = Path(input_source)
        if is_subtitle_file_path(input_path):
            return None

        return build_output_path(input_path.with_suffix(".srt"), target_lang_code)

    source_name = getattr(input_source, "name", None)
    if not source_name:
        raise ValueError("output_path is required when input_source is file-like without a name")

    source_path = Path(source_name)
    if is_subtitle_file_path(source_path):
        return build_output_path(source_path, target_lang_code)

    return build_output_path(source_path.with_suffix(".srt"), target_lang_code)


def _input_filename(input_source: TextIO | BinaryIO) -> str:
    source_name = getattr(input_source, "name", None)
    if source_name:
        return Path(source_name).name

    return "input.srt"
