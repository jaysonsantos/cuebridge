from __future__ import annotations

from pathlib import Path

import pytest
from cuebridge import bitmap_ocr
from cuebridge.bitmap_ocr import (
    BitmapSubtitleCue,
    RenderedSubtitleFrame,
    _build_bitmap_subtitle_cues,
    _clean_ocr_text,
    _parse_showinfo_frames,
)
from cuebridge.media import SubtitleStreamInfo
from PIL import Image


def test_parse_showinfo_frames_pairs_timestamps_with_rendered_images(tmp_path: Path) -> None:
    image_paths = []
    for index in range(1, 3):
        image_path = tmp_path / f"frame-{index:06d}.png"
        Image.new("RGB", (4, 4), color="black").save(image_path)
        image_paths.append(image_path)

    stderr_output = """
[Parsed_showinfo_1 @ 0x0] n:   0 pts:  75000 pts_time:0.075   duration:      0 duration_time:0       fmt:rgba checksum:00000000 plane_checksum:[00000000]
[Parsed_showinfo_1 @ 0x0] n:   1 pts: 1785000 pts_time:1.785   duration:      0 duration_time:0       fmt:rgba checksum:B466570A plane_checksum:[B466570A]
"""

    frames = _parse_showinfo_frames(stderr_output, image_paths)

    assert frames == [
        RenderedSubtitleFrame(
            pts_time=0.075,
            checksum="00000000",
            image_path=image_paths[0],
        ),
        RenderedSubtitleFrame(
            pts_time=1.785,
            checksum="B466570A",
            image_path=image_paths[1],
        ),
    ]


def test_build_bitmap_subtitle_cues_deduplicates_hold_frames(tmp_path: Path) -> None:
    blank_image = tmp_path / "blank.png"
    cue_image = tmp_path / "cue.png"
    Image.new("RGB", (8, 8), color="black").save(blank_image)
    cue_bitmap = Image.new("RGB", (8, 8), color="black")
    cue_bitmap.putpixel((4, 4), (255, 255, 255))
    cue_bitmap.save(cue_image)

    cues = _build_bitmap_subtitle_cues(
        [
            RenderedSubtitleFrame(pts_time=0.075, checksum="00000000", image_path=blank_image),
            RenderedSubtitleFrame(pts_time=0.075, checksum="ABCD1234", image_path=cue_image),
            RenderedSubtitleFrame(pts_time=1.784999, checksum="ABCD1234", image_path=cue_image),
            RenderedSubtitleFrame(pts_time=1.785, checksum="00000000", image_path=blank_image),
        ],
        stream_duration_seconds=10.0,
    )

    assert cues == [
        BitmapSubtitleCue(
            start_seconds=0.075,
            end_seconds=1.785,
            image_path=cue_image,
            checksum="ABCD1234",
        )
    ]


def test_clean_ocr_text_normalizes_whitespace() -> None:
    assert _clean_ocr_text(" Hello   there \n\nHow\tare you?\x0c") == "Hello there\nHow are you?"


def test_extract_bitmap_subtitle_stream_requires_tesseract_only_when_needed(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def fake_ensure(command_name: str, *, reason: str) -> str:
        del reason
        calls.append(command_name)
        if command_name == "tesseract":
            raise RuntimeError("tesseract is required to OCR bitmap subtitle streams")
        return f"/usr/bin/{command_name}"

    monkeypatch.setattr(bitmap_ocr, "_ensure_command_available", fake_ensure)

    with pytest.raises(RuntimeError, match="tesseract is required"):
        bitmap_ocr.extract_bitmap_subtitle_stream_to_srt(
            input_path=tmp_path / "movie.mkv",
            stream=SubtitleStreamInfo(
                relative_index=0,
                stream_index=3,
                codec_name="hdmv_pgs_subtitle",
                language="eng",
                title="SDH",
                is_default=False,
                duration_seconds=120.0,
            ),
            output_path=tmp_path / "out.srt",
            source_lang_code="en",
        )

    assert calls == ["ffmpeg", "tesseract"]
