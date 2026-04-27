from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pysubs2
from loguru import logger
from opentelemetry import trace
from PIL import Image, ImageOps

BLANK_FRAME_CHECKSUM = "00000000"
DEFAULT_BITMAP_CANVAS_SIZE = (1920, 1080)
SHOWINFO_FRAME_RE = re.compile(
    r"n:\s*(?P<frame_number>\d+)\s+pts:\s*-?\d+\s+pts_time:(?P<pts_time>-?\d+(?:\.\d+)?)"
    r".*checksum:(?P<checksum>[0-9A-F]+)"
)
OCR_WHITESPACE_RE = re.compile(r"[ \t]+")
TESSERACT_LANGUAGE_ALIASES = {
    "de": "deu",
    "deu": "deu",
    "en": "eng",
    "eng": "eng",
    "es": "spa",
    "spa": "spa",
    "fr": "fra",
    "fra": "fra",
    "fre": "fra",
    "ger": "deu",
    "it": "ita",
    "ita": "ita",
    "ja": "jpn",
    "jpn": "jpn",
    "nl": "nld",
    "nld": "nld",
    "dut": "nld",
    "por": "por",
    "pt": "por",
}
TRACER = trace.get_tracer(__name__)


class SubtitleStream(Protocol):
    relative_index: int
    duration_seconds: float | None


@dataclass(frozen=True, slots=True)
class RenderedSubtitleFrame:
    pts_time: float
    checksum: str
    image_path: Path


@dataclass(frozen=True, slots=True)
class BitmapSubtitleCue:
    start_seconds: float
    end_seconds: float
    image_path: Path
    checksum: str


def extract_bitmap_subtitle_stream_to_srt(
    *,
    input_path: Path,
    stream: SubtitleStream,
    output_path: Path,
    source_lang_code: str,
    ocr_language: str | None = None,
) -> None:
    _ensure_command_available("ffmpeg", reason="to render bitmap subtitles from video files")
    _ensure_command_available("tesseract", reason="to OCR bitmap subtitle streams")

    with tempfile.TemporaryDirectory(prefix="cuebridge-bitmap-subtitles-") as temp_dir:
        render_dir = Path(temp_dir) / "rendered"
        render_dir.mkdir(parents=True, exist_ok=True)
        processed_dir = Path(temp_dir) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        rendered_frames = _render_bitmap_subtitle_frames(
            input_path=input_path,
            stream=stream,
            render_dir=render_dir,
        )
        cues = _build_bitmap_subtitle_cues(
            rendered_frames,
            stream_duration_seconds=stream.duration_seconds,
        )

        subtitles = pysubs2.SSAFile()
        resolved_ocr_language = ocr_language or _default_tesseract_language(source_lang_code)
        for cue_index, cue in enumerate(cues, start=1):
            processed_image_path = processed_dir / f"cue-{cue_index:06d}.png"
            _prepare_ocr_image(
                source_path=cue.image_path,
                output_path=processed_image_path,
            )
            subtitle_text = _ocr_image_to_text(
                image_path=processed_image_path,
                ocr_language=resolved_ocr_language,
            )
            subtitles.append(
                pysubs2.SSAEvent(
                    start=max(0, round(cue.start_seconds * 1000)),
                    end=max(
                        round(cue.start_seconds * 1000) + 1,
                        round(cue.end_seconds * 1000),
                    ),
                    text=subtitle_text,
                )
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "OCR extracted {} bitmap subtitle events from stream {} into {}",
            len(subtitles),
            stream.relative_index,
            output_path,
        )
        subtitles.save(str(output_path))


def _render_bitmap_subtitle_frames(
    *,
    input_path: Path,
    stream: SubtitleStream,
    render_dir: Path,
) -> list[RenderedSubtitleFrame]:
    output_pattern = render_dir / "frame-%06d.png"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-nostats",
        "-y",
        "-analyzeduration",
        "100M",
        "-probesize",
        "100M",
        "-i",
        str(input_path),
        "-filter_complex",
        (
            f"[0:s:{stream.relative_index}]scale="
            f"{DEFAULT_BITMAP_CANVAS_SIZE[0]}:{DEFAULT_BITMAP_CANVAS_SIZE[1]},showinfo"
        ),
        "-fps_mode",
        "passthrough",
        str(output_pattern),
    ]
    completed = _run_checked_command(command, capture_output=True)
    image_paths = sorted(render_dir.glob("frame-*.png"))
    return _parse_showinfo_frames(completed.stderr, image_paths)


def _parse_showinfo_frames(
    stderr_output: str, image_paths: list[Path]
) -> list[RenderedSubtitleFrame]:
    parsed_frames: list[RenderedSubtitleFrame] = []
    for line in stderr_output.splitlines():
        match = SHOWINFO_FRAME_RE.search(line)
        if match is None:
            continue
        parsed_frames.append(
            RenderedSubtitleFrame(
                pts_time=float(match.group("pts_time")),
                checksum=match.group("checksum"),
                image_path=Path(),
            )
        )

    if len(parsed_frames) != len(image_paths):
        raise RuntimeError(
            "Rendered subtitle frame count did not match ffmpeg showinfo output "
            f"({len(parsed_frames)} timestamps for {len(image_paths)} images)"
        )

    return [
        RenderedSubtitleFrame(
            pts_time=frame.pts_time,
            checksum=frame.checksum,
            image_path=image_path,
        )
        for frame, image_path in zip(parsed_frames, image_paths, strict=True)
    ]


def _build_bitmap_subtitle_cues(
    frames: list[RenderedSubtitleFrame],
    *,
    stream_duration_seconds: float | None,
) -> list[BitmapSubtitleCue]:
    cues: list[BitmapSubtitleCue] = []
    previous_checksum: str | None = None
    active_cue: BitmapSubtitleCue | None = None

    for frame in frames:
        checksum = frame.checksum
        if _is_blank_frame(frame):
            checksum = BLANK_FRAME_CHECKSUM

        if checksum == previous_checksum:
            continue

        if active_cue is not None:
            cues.append(
                BitmapSubtitleCue(
                    start_seconds=active_cue.start_seconds,
                    end_seconds=frame.pts_time,
                    image_path=active_cue.image_path,
                    checksum=active_cue.checksum,
                )
            )
            active_cue = None

        if checksum != BLANK_FRAME_CHECKSUM:
            active_cue = BitmapSubtitleCue(
                start_seconds=frame.pts_time,
                end_seconds=frame.pts_time,
                image_path=frame.image_path,
                checksum=checksum,
            )

        previous_checksum = checksum

    if active_cue is not None:
        final_end = stream_duration_seconds or active_cue.start_seconds + 2.0
        cues.append(
            BitmapSubtitleCue(
                start_seconds=active_cue.start_seconds,
                end_seconds=max(final_end, active_cue.start_seconds + 0.001),
                image_path=active_cue.image_path,
                checksum=active_cue.checksum,
            )
        )

    return cues


def _prepare_ocr_image(*, source_path: Path, output_path: Path) -> None:
    with Image.open(source_path) as image:
        grayscale = ImageOps.grayscale(image)
        bbox = grayscale.getbbox()
        if bbox is None:
            cropped = grayscale
        else:
            padding = 12
            left = max(0, bbox[0] - padding)
            top = max(0, bbox[1] - padding)
            right = min(grayscale.width, bbox[2] + padding)
            bottom = min(grayscale.height, bbox[3] + padding)
            cropped = grayscale.crop((left, top, right, bottom))

        normalized = ImageOps.autocontrast(cropped)
        binary = normalized.point(lambda pixel: 255 if pixel >= 160 else 0)
        inverted = ImageOps.invert(binary)
        upscaled = inverted.resize(
            (max(1, inverted.width * 2), max(1, inverted.height * 2)),
            Image.Resampling.LANCZOS,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        upscaled.save(output_path)


def _ocr_image_to_text(*, image_path: Path, ocr_language: str | None) -> str:
    command = [
        "tesseract",
        str(image_path),
        "stdout",
        "--psm",
        "6",
        "-c",
        "preserve_interword_spaces=1",
    ]
    if ocr_language:
        command.extend(["-l", ocr_language])
    completed = _run_checked_command(command, capture_output=True)
    return _clean_ocr_text(completed.stdout)


def _clean_ocr_text(text: str) -> str:
    cleaned_lines = [
        OCR_WHITESPACE_RE.sub(" ", line).strip() for line in text.replace("\x0c", "").splitlines()
    ]
    return "\n".join(line for line in cleaned_lines if line)


def _default_tesseract_language(source_lang_code: str) -> str | None:
    normalized = source_lang_code.strip().lower()
    if not normalized:
        return None
    base_language = normalized.split("-", maxsplit=1)[0].split("_", maxsplit=1)[0]
    return TESSERACT_LANGUAGE_ALIASES.get(normalized) or TESSERACT_LANGUAGE_ALIASES.get(
        base_language
    )


def _is_blank_frame(frame: RenderedSubtitleFrame) -> bool:
    if frame.checksum == BLANK_FRAME_CHECKSUM:
        return True

    with Image.open(frame.image_path) as image:
        grayscale = ImageOps.grayscale(image)
        return grayscale.getbbox() is None


def _ensure_command_available(command_name: str, *, reason: str) -> str:
    resolved_path = shutil.which(command_name)
    if resolved_path is None:
        raise RuntimeError(f"{command_name} is required {reason}")
    return resolved_path


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
