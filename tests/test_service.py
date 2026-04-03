from __future__ import annotations

import io
import re
from pathlib import Path

import pysubs2
from cuebridge import service
from cuebridge.cancellation import CancellationToken
from cuebridge.media import SubtitleStreamInfo
from cuebridge.service import RuntimeOptions, SubtitleTranslationRequest, TranslatorConfig

SEGMENT_MARKER_RE = re.compile(r"(\[\[SEG_\d+]])")


class FakeTranslator:
    def __init__(self, target_lang_code: str) -> None:
        self.target_lang_code = target_lang_code

    def translate_text(self, text: str, cancellation_token: CancellationToken | None = None) -> str:
        del cancellation_token
        if "[[SEG_" in text:
            parts = SEGMENT_MARKER_RE.split(text)
            output_parts: list[str] = []
            current_marker: str | None = None
            for part in parts:
                if not part:
                    continue
                if SEGMENT_MARKER_RE.fullmatch(part):
                    current_marker = part
                    output_parts.append(part)
                    continue
                if current_marker is not None:
                    output_parts.append(f"[{self.target_lang_code}] {part.strip()}")
                    current_marker = None
            return "\n".join(output_parts)

        return f"[{self.target_lang_code}] {text}"


def test_service_translates_samples_and_uses_target_language_suffix(
    monkeypatch,
    subtitle_samples,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_builder(**kwargs):
        calls.append((kwargs["source_lang_code"], kwargs["target_lang_code"]))
        return FakeTranslator(kwargs["target_lang_code"])

    monkeypatch.setattr(service, "build_subtitle_translator", fake_builder)

    for sample in subtitle_samples:
        input_path = Path(sample.filename)
        result = service.run_subtitle_translation(
            SubtitleTranslationRequest(
                input_source=input_path,
                source_lang_code=sample.source_lang_code,
                target_lang_code="pt-BR",
                translator_config=TranslatorConfig(model_id="fake-model"),
                runtime_options=RuntimeOptions(),
            )
        )

        assert result.output_path.exists()

        translated = pysubs2.load(str(result.output_path))
        assert translated[0].text.startswith("[pt-BR]")

    assert calls == [("en", "pt-BR"), ("es", "pt-BR")]


def test_service_accepts_file_like_input(monkeypatch, tmp_path: Path) -> None:
    def fake_builder(**kwargs):
        return FakeTranslator(kwargs["target_lang_code"])

    monkeypatch.setattr(service, "build_subtitle_translator", fake_builder)

    input_source = io.StringIO(
        """1
00:00:01,000 --> 00:00:02,500
Hello there!
"""
    )
    output_path = tmp_path / "translated.pt-BR.srt"

    result = service.run_subtitle_translation(
        SubtitleTranslationRequest(
            input_source=input_source,
            source_lang_code="en",
            target_lang_code="pt-BR",
            translator_config=TranslatorConfig(model_id="fake-model"),
            runtime_options=RuntimeOptions(window_size=1, flush_every_chunks=1),
            output_path=output_path,
        )
    )

    assert result.output_path == output_path
    translated = pysubs2.load(str(output_path))
    assert translated[0].text == "[pt-BR] Hello there!"


def test_service_passes_known_backend_name_to_translator_builder(
    monkeypatch, tmp_path: Path
) -> None:
    captured_kwargs: list[dict[str, object]] = []

    def fake_builder(**kwargs):
        captured_kwargs.append(kwargs)
        return FakeTranslator(kwargs["target_lang_code"])

    monkeypatch.setattr(service, "build_subtitle_translator", fake_builder)

    input_path = tmp_path / "movie.en.srt"
    input_path.write_text(
        """1
00:00:01,000 --> 00:00:02,500
Hello there!
""",
        encoding="utf-8",
    )

    service.run_subtitle_translation(
        SubtitleTranslationRequest(
            input_source=input_path,
            source_lang_code="en",
            target_lang_code="pt-BR",
            translator_config=TranslatorConfig(model_id="fake-model", backend="openrouter"),
        )
    )

    assert captured_kwargs[0]["backend"] == "openrouter"
    assert captured_kwargs[0]["api_base_url"] is None
    assert captured_kwargs[0]["api_key_env"] is None


def test_service_passes_retain_history_to_translator_builder(monkeypatch, tmp_path: Path) -> None:
    captured_kwargs: list[dict[str, object]] = []

    def fake_builder(**kwargs):
        captured_kwargs.append(kwargs)
        return FakeTranslator(kwargs["target_lang_code"])

    monkeypatch.setattr(service, "build_subtitle_translator", fake_builder)

    input_path = tmp_path / "movie.en.srt"
    input_path.write_text(
        """1
00:00:01,000 --> 00:00:02,500
Hello there!
""",
        encoding="utf-8",
    )

    service.run_subtitle_translation(
        SubtitleTranslationRequest(
            input_source=input_path,
            source_lang_code="en",
            target_lang_code="pt-BR",
            translator_config=TranslatorConfig(model_id="fake-model", retain_history=True),
        )
    )

    assert captured_kwargs[0]["retain_history"] is True


def test_service_passes_reasoning_effort_to_translator_builder(monkeypatch, tmp_path: Path) -> None:
    captured_kwargs: list[dict[str, object]] = []

    def fake_builder(**kwargs):
        captured_kwargs.append(kwargs)
        return FakeTranslator(kwargs["target_lang_code"])

    monkeypatch.setattr(service, "build_subtitle_translator", fake_builder)

    input_path = tmp_path / "movie.en.srt"
    input_path.write_text(
        """1
00:00:01,000 --> 00:00:02,500
Hello there!
""",
        encoding="utf-8",
    )

    service.run_subtitle_translation(
        SubtitleTranslationRequest(
            input_source=input_path,
            source_lang_code="en",
            target_lang_code="pt-BR",
            translator_config=TranslatorConfig(model_id="fake-model", reasoning_effort="none"),
        )
    )

    assert captured_kwargs[0]["reasoning_effort"] == "none"


def test_service_uses_auto_window_size_for_openai_compatible_backends(
    monkeypatch, tmp_path: Path
) -> None:
    captured_kwargs: list[dict[str, object]] = []

    def fake_builder(**kwargs):
        return FakeTranslator(kwargs["target_lang_code"])

    def fake_translate_subtitle_file(**kwargs):
        captured_kwargs.append(kwargs)
        return service.TranslationResult(
            output_path=Path("/tmp/out.srt"),
            translated_events=1,
        )

    monkeypatch.setattr(service, "build_subtitle_translator", fake_builder)
    monkeypatch.setattr(service, "translate_subtitle_file", fake_translate_subtitle_file)

    input_path = tmp_path / "movie.en.srt"
    input_path.write_text(
        """1
00:00:01,000 --> 00:00:02,500
Hello there!
""",
        encoding="utf-8",
    )

    service.run_subtitle_translation(
        SubtitleTranslationRequest(
            input_source=input_path,
            source_lang_code="en",
            target_lang_code="pt-BR",
            translator_config=TranslatorConfig(model_id="fake-model", backend="openrouter"),
        )
    )

    assert captured_kwargs[0]["window_size"] == service.DEFAULT_OPENAI_COMPATIBLE_WINDOW_SIZE


def test_service_uses_smaller_auto_window_size_when_history_is_retained(
    monkeypatch, tmp_path: Path
) -> None:
    captured_kwargs: list[dict[str, object]] = []

    def fake_builder(**kwargs):
        return FakeTranslator(kwargs["target_lang_code"])

    def fake_translate_subtitle_file(**kwargs):
        captured_kwargs.append(kwargs)
        return service.TranslationResult(
            output_path=Path("/tmp/out.srt"),
            translated_events=1,
        )

    monkeypatch.setattr(service, "build_subtitle_translator", fake_builder)
    monkeypatch.setattr(service, "translate_subtitle_file", fake_translate_subtitle_file)

    input_path = tmp_path / "movie.en.srt"
    input_path.write_text(
        """1
00:00:01,000 --> 00:00:02,500
Hello there!
""",
        encoding="utf-8",
    )

    service.run_subtitle_translation(
        SubtitleTranslationRequest(
            input_source=input_path,
            source_lang_code="en",
            target_lang_code="pt-BR",
            translator_config=TranslatorConfig(
                model_id="fake-model",
                backend="openrouter",
                retain_history=True,
            ),
        )
    )

    assert captured_kwargs[0]["window_size"] == service.DEFAULT_HISTORY_WINDOW_SIZE


def test_service_defaults_video_output_to_srt_path(monkeypatch, tmp_path: Path) -> None:
    captured_kwargs: list[dict[str, object]] = []

    def fake_builder(**kwargs):
        return FakeTranslator(kwargs["target_lang_code"])

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
        output_path.write_text(
            """1
00:00:01,000 --> 00:00:02,500
Hello there!
""",
            encoding="utf-8",
        )

    def fake_translate_subtitle_file(**kwargs):
        captured_kwargs.append(kwargs)
        return service.TranslationResult(
            output_path=kwargs["output_path"],
            translated_events=1,
        )

    monkeypatch.setattr(service, "build_subtitle_translator", fake_builder)
    monkeypatch.setattr(service, "probe_subtitle_streams", fake_probe_subtitle_streams)
    monkeypatch.setattr(
        service, "extract_text_subtitle_stream_to_srt", fake_extract_text_subtitle_stream_to_srt
    )
    monkeypatch.setattr(service, "translate_subtitle_file", fake_translate_subtitle_file)

    input_path = tmp_path / "episode.en.mkv"
    input_path.write_bytes(b"fake video payload")

    result = service.run_subtitle_translation(
        SubtitleTranslationRequest(
            input_source=input_path,
            source_lang_code="en",
            target_lang_code="pt-BR",
            translator_config=TranslatorConfig(model_id="fake-model"),
        )
    )

    assert result.output_path == tmp_path / "episode.pt-BR.srt"
    assert captured_kwargs[0]["output_path"] == tmp_path / "episode.pt-BR.srt"
    assert Path(captured_kwargs[0]["input_path"]).suffix == ".srt"
