from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from opentelemetry import trace

from cuebridge import bitmap_ocr

SUBTITLE_FILE_EXTENSIONS = {".ass", ".srt", ".ssa", ".sub", ".vtt"}
BITMAP_SUBTITLE_CODECS = {
    "dvb_subtitle",
    "dvd_subtitle",
    "hdmv_pgs_subtitle",
    "xsub",
}
LANGUAGE_ALIASES = {
    "de": {"de", "deu", "ger"},
    "en": {"en", "eng"},
    "es": {"es", "spa"},
    "fr": {"fr", "fra", "fre"},
    "it": {"it", "ita"},
    "ja": {"ja", "jpn"},
    "nl": {"nl", "nld", "dut"},
    "pt": {"pt", "por"},
}
TRACER = trace.get_tracer(__name__)


@dataclass(frozen=True, slots=True)
class SubtitleStreamInfo:
    relative_index: int
    stream_index: int
    codec_name: str
    language: str | None
    title: str | None
    is_default: bool
    duration_seconds: float | None = None


def is_subtitle_file_path(path: Path) -> bool:
    return path.suffix.lower() in SUBTITLE_FILE_EXTENSIONS


def is_bitmap_subtitle_codec(codec_name: str) -> bool:
    return codec_name.lower() in BITMAP_SUBTITLE_CODECS


def probe_subtitle_streams(input_path: Path) -> list[SubtitleStreamInfo]:
    _ensure_command_available("ffprobe", reason="to inspect subtitle streams in video files")
    payload = _run_json_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-select_streams",
            "s",
            "-of",
            "json",
            str(input_path),
        ]
    )
    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, list):
        raise RuntimeError("ffprobe did not return subtitle stream information")

    streams: list[SubtitleStreamInfo] = []
    for relative_index, raw_stream in enumerate(raw_streams):
        if not isinstance(raw_stream, dict):
            continue
        tags = raw_stream.get("tags")
        disposition = raw_stream.get("disposition")
        streams.append(
            SubtitleStreamInfo(
                relative_index=relative_index,
                stream_index=int(raw_stream["index"]),
                codec_name=str(raw_stream.get("codec_name", "unknown")),
                language=_optional_string(tags, "language"),
                title=_optional_string(tags, "title"),
                is_default=bool(_optional_int(disposition, "default", default=0)),
                duration_seconds=_optional_float(raw_stream, "duration"),
            )
        )

    return streams


def select_subtitle_stream(
    *,
    streams: list[SubtitleStreamInfo],
    source_lang_code: str,
    preferred_stream_index: int | None = None,
) -> SubtitleStreamInfo:
    if not streams:
        raise ValueError("No subtitle streams were found in the video input")

    if preferred_stream_index is not None:
        for stream in streams:
            if stream.relative_index == preferred_stream_index:
                return stream
        raise ValueError(
            "Subtitle stream "
            f"{preferred_stream_index} was not found. Available subtitle streams: "
            f"{_format_streams(streams)}"
        )

    language_candidates = _language_candidates(source_lang_code)
    exact_matches = [
        stream
        for stream in streams
        if stream.language is not None and stream.language.lower() in language_candidates
    ]
    if exact_matches:
        default_match = next((stream for stream in exact_matches if stream.is_default), None)
        return default_match or exact_matches[0]

    if len(streams) == 1:
        return streams[0]

    raise ValueError(
        "Could not choose a subtitle stream automatically for source language "
        f"{source_lang_code!r}. Available subtitle streams: {_format_streams(streams)}. "
        "Pass --subtitle-stream to choose one explicitly."
    )


@TRACER.start_as_current_span("cuebridge.media.extract_text_subtitle_stream_to_srt")
def extract_text_subtitle_stream_to_srt(
    *,
    input_path: Path,
    stream: SubtitleStreamInfo,
    output_path: Path,
) -> None:
    _ensure_command_available("ffmpeg", reason="to extract subtitles from video files")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Extracting text subtitle stream {} ({}) to {}",
        stream.relative_index,
        stream.codec_name,
        output_path,
    )
    _run_checked_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-map",
            f"0:s:{stream.relative_index}",
            "-c:s",
            "srt",
            str(output_path),
        ]
    )


def extract_bitmap_subtitle_stream_to_srt(
    *,
    input_path: Path,
    stream: SubtitleStreamInfo,
    output_path: Path,
    source_lang_code: str,
    ocr_language: str | None = None,
) -> None:
    bitmap_ocr.extract_bitmap_subtitle_stream_to_srt(
        input_path=input_path,
        stream=stream,
        output_path=output_path,
        source_lang_code=source_lang_code,
        ocr_language=ocr_language,
    )


def _language_candidates(source_lang_code: str) -> set[str]:
    normalized = source_lang_code.strip().lower()
    if not normalized:
        return set()

    base_language = normalized.split("-", maxsplit=1)[0].split("_", maxsplit=1)[0]
    candidates = {normalized, base_language}
    candidates.update(LANGUAGE_ALIASES.get(normalized, set()))
    candidates.update(LANGUAGE_ALIASES.get(base_language, set()))
    return {candidate.lower() for candidate in candidates}


def _format_streams(streams: list[SubtitleStreamInfo]) -> str:
    formatted_streams: list[str] = []
    for stream in streams:
        description = (
            f"{stream.relative_index}: "
            f"lang={stream.language or 'unknown'}, codec={stream.codec_name}"
        )
        if stream.title:
            description += f", title={stream.title!r}"
        if stream.is_default:
            description += ", default=yes"
        formatted_streams.append(description)
    return "; ".join(formatted_streams)


def _ensure_command_available(command_name: str, *, reason: str) -> str:
    resolved_path = shutil.which(command_name)
    if resolved_path is None:
        raise RuntimeError(f"{command_name} is required {reason}")
    return resolved_path


def _run_json_command(command: list[str]) -> dict[str, Any]:
    completed = _run_checked_command(command, capture_output=True)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Command did not return valid JSON: {' '.join(command)}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object from command: {' '.join(command)}")
    return payload


def _run_checked_command(
    command: list[str],
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=capture_output,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = exc.stderr or exc.stdout or ""
        raise RuntimeError(
            f"Command failed with exit code {exc.returncode}: {' '.join(command)}\n{details.strip()}"
        ) from exc


def _optional_float(payload: object, key: str) -> float | None:
    value = _optional_string(payload, key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _optional_int(payload: object, key: str, *, default: int | None = None) -> int | None:
    if not isinstance(payload, dict):
        return default
    value = payload.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_string(payload: object, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value is None:
        return None
    return str(value)
