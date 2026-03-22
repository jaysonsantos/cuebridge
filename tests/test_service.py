from __future__ import annotations

import io
import re
from pathlib import Path

import pysubs2
from cuebridge import service
from cuebridge.cancellation import CancellationToken
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
