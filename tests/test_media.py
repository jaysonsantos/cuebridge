from __future__ import annotations

import subprocess

from cuebridge import media
from cuebridge.media import (
    SubtitleStreamInfo,
    select_subtitle_stream,
)


def test_select_subtitle_stream_matches_language_aliases() -> None:
    streams = [
        SubtitleStreamInfo(
            relative_index=0,
            stream_index=3,
            codec_name="hdmv_pgs_subtitle",
            language="eng",
            title="SDH",
            is_default=False,
            duration_seconds=120.0,
        ),
        SubtitleStreamInfo(
            relative_index=1,
            stream_index=4,
            codec_name="hdmv_pgs_subtitle",
            language="ger",
            title="SDH",
            is_default=True,
            duration_seconds=120.0,
        ),
    ]

    selected = select_subtitle_stream(streams=streams, source_lang_code="de-DE")

    assert selected == streams[1]


def test_run_checked_command_defaults_to_no_capture(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_run(command, *, check, capture_output, text):
        captured_kwargs["command"] = command
        captured_kwargs["check"] = check
        captured_kwargs["capture_output"] = capture_output
        captured_kwargs["text"] = text
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(media.subprocess, "run", fake_run)

    result = media._run_checked_command(["echo", "ok"])

    assert result.returncode == 0
    assert captured_kwargs == {
        "command": ["echo", "ok"],
        "check": True,
        "capture_output": False,
        "text": True,
    }
