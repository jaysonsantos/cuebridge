from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, TextIO

from cuebridge.agent import build_subtitle_translator
from cuebridge.cancellation import CancellationToken
from cuebridge.media import (
    extract_bitmap_subtitle_stream_to_srt,
    extract_text_subtitle_stream_to_srt,
    is_bitmap_subtitle_codec,
    is_subtitle_file_path,
    probe_subtitle_streams,
    select_subtitle_stream,
)
from cuebridge.naming import build_output_path
from cuebridge.subtitles import TranslationResult, translate_subtitle_file

SubtitleInputSource = Path | str | TextIO | BinaryIO
OPENAI_COMPATIBLE_BACKENDS = {"openai-compatible", "cerebras", "openrouter"}
DEFAULT_HF_LOCAL_WINDOW_SIZE = 4
DEFAULT_OPENAI_COMPATIBLE_WINDOW_SIZE = 12
DEFAULT_HISTORY_WINDOW_SIZE = 4


@dataclass(frozen=True)
class TranslatorConfig:
    model_id: str
    backend: str = "hf-local"
    dtype: str = "bfloat16"
    device: str | None = None
    max_new_tokens: int = 256
    batch_size: int = 1
    api_base_url: str | None = None
    message_format: str = "auto"
    api_key: str | None = None
    api_key_env: str | None = None
    request_timeout_seconds: float = 120.0
    max_input_tokens: int = 1800
    thread_id: str | None = None
    retain_history: bool = False
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class RuntimeOptions:
    window_size: int | None = None
    flush_every_chunks: int = 1
    subtitle_stream: int | None = None
    ocr_language: str | None = None


@dataclass(frozen=True)
class SubtitleTranslationRequest:
    input_source: SubtitleInputSource
    source_lang_code: str
    target_lang_code: str
    translator_config: TranslatorConfig
    runtime_options: RuntimeOptions = field(default_factory=RuntimeOptions)
    output_path: Path | None = None
    cancellation_token: CancellationToken | None = None


def run_subtitle_translation(request: SubtitleTranslationRequest) -> TranslationResult:
    translator = build_subtitle_translator(
        source_lang_code=request.source_lang_code,
        target_lang_code=request.target_lang_code,
        model_id=request.translator_config.model_id,
        backend=request.translator_config.backend,
        dtype=request.translator_config.dtype,
        device=request.translator_config.device,
        max_new_tokens=request.translator_config.max_new_tokens,
        batch_size=request.translator_config.batch_size,
        api_base_url=request.translator_config.api_base_url,
        message_format=request.translator_config.message_format,
        api_key=request.translator_config.api_key,
        api_key_env=request.translator_config.api_key_env,
        request_timeout_seconds=request.translator_config.request_timeout_seconds,
        reasoning_effort=request.translator_config.reasoning_effort,
        max_input_tokens=request.translator_config.max_input_tokens,
        thread_id=request.translator_config.thread_id,
        retain_history=request.translator_config.retain_history,
    )

    with _resolved_input_path(request) as input_path:
        resolved_output_path = _resolve_output_path(
            input_source=request.input_source,
            target_lang_code=request.target_lang_code,
            output_path=request.output_path,
        )
        return translate_subtitle_file(
            input_path=input_path,
            target_lang_code=request.target_lang_code,
            translator=translator,
            window_size=_resolve_window_size(request),
            flush_every_chunks=request.runtime_options.flush_every_chunks,
            output_path=resolved_output_path,
            cancellation_token=request.cancellation_token,
        )


@contextmanager
def _resolved_input_path(request: SubtitleTranslationRequest) -> Iterator[Path]:
    if isinstance(request.input_source, Path | str):
        input_path = Path(request.input_source)
        if is_subtitle_file_path(input_path):
            yield input_path
            return

        with tempfile.TemporaryDirectory(prefix="cuebridge-video-subtitles-") as tmp_dir:
            extracted_path = Path(tmp_dir) / f"{input_path.stem}.source.srt"
            selected_stream = select_subtitle_stream(
                streams=probe_subtitle_streams(input_path),
                source_lang_code=request.source_lang_code,
                preferred_stream_index=request.runtime_options.subtitle_stream,
            )
            if is_bitmap_subtitle_codec(selected_stream.codec_name):
                extract_bitmap_subtitle_stream_to_srt(
                    input_path=input_path,
                    stream=selected_stream,
                    output_path=extracted_path,
                    source_lang_code=request.source_lang_code,
                    ocr_language=request.runtime_options.ocr_language,
                )
            else:
                extract_text_subtitle_stream_to_srt(
                    input_path=input_path,
                    stream=selected_stream,
                    output_path=extracted_path,
                )
            yield extracted_path
            return

    filename = _input_filename(request.input_source)
    if not is_subtitle_file_path(Path(filename)):
        raise ValueError("Video input must be provided as a filesystem path")
    content = request.input_source.read()
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


def _resolve_window_size(request: SubtitleTranslationRequest) -> int:
    if request.runtime_options.window_size is not None:
        return request.runtime_options.window_size

    if request.translator_config.retain_history:
        return DEFAULT_HISTORY_WINDOW_SIZE

    if request.translator_config.backend.lower() in OPENAI_COMPATIBLE_BACKENDS:
        return DEFAULT_OPENAI_COMPATIBLE_WINDOW_SIZE

    return DEFAULT_HF_LOCAL_WINDOW_SIZE
