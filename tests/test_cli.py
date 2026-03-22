from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
from cuebridge import cli
from cuebridge.naming import build_output_path
from cuebridge.service import RuntimeOptions, SubtitleTranslationRequest, TranslatorConfig
from cuebridge.subtitles import TranslationResult


def test_cli_calls_service_with_parsed_request_and_prints_output_path(
    monkeypatch,
    subtitle_samples,
) -> None:
    requests: list[SubtitleTranslationRequest] = []

    def fake_service(request: SubtitleTranslationRequest) -> TranslationResult:
        requests.append(request)
        input_path = Path(request.input_source)
        return TranslationResult(
            output_path=build_output_path(input_path, request.target_lang_code),
            translated_events=2,
        )

    monkeypatch.setattr(cli, "run_subtitle_translation", fake_service)
    runner = CliRunner()

    for sample in subtitle_samples:
        input_path = Path(sample.filename)
        result = runner.invoke(
            cli.main,
            [
                str(input_path),
                "--source-lang",
                sample.source_lang_code,
                "--target-lang",
                "pt-BR",
            ],
        )

        assert result.exit_code == 0, result.output

        output_path = build_output_path(input_path, "pt-BR")
        assert result.output.strip() == str(output_path)

    assert requests == [
        SubtitleTranslationRequest(
            input_source=Path(subtitle_samples[0].filename),
            source_lang_code="en",
            target_lang_code="pt-BR",
            translator_config=TranslatorConfig(model_id=cli.DEFAULT_MODEL_ID),
            runtime_options=RuntimeOptions(),
            output_path=None,
        ),
        SubtitleTranslationRequest(
            input_source=Path(subtitle_samples[1].filename),
            source_lang_code="es",
            target_lang_code="pt-BR",
            translator_config=TranslatorConfig(model_id=cli.DEFAULT_MODEL_ID),
            runtime_options=RuntimeOptions(),
            output_path=None,
        ),
    ]


def test_cli_accepts_known_openai_compatible_backend_names(monkeypatch, subtitle_samples) -> None:
    requests: list[SubtitleTranslationRequest] = []

    def fake_service(request: SubtitleTranslationRequest) -> TranslationResult:
        requests.append(request)
        return TranslationResult(output_path=Path("/tmp/out.srt"), translated_events=1)

    monkeypatch.setattr(cli, "run_subtitle_translation", fake_service)
    runner = CliRunner()

    result = runner.invoke(
        cli.main,
        [
            subtitle_samples[0].filename,
            "--source-lang",
            "en",
            "--target-lang",
            "pt-BR",
            "--backend",
            "cerebras",
        ],
    )

    assert result.exit_code == 0, result.output
    assert requests == [
        SubtitleTranslationRequest(
            input_source=Path(subtitle_samples[0].filename),
            source_lang_code="en",
            target_lang_code="pt-BR",
            translator_config=TranslatorConfig(model_id=cli.DEFAULT_MODEL_ID, backend="cerebras"),
            runtime_options=RuntimeOptions(),
            output_path=None,
        )
    ]


def test_cli_can_enable_retained_history(monkeypatch, subtitle_samples) -> None:
    requests: list[SubtitleTranslationRequest] = []

    def fake_service(request: SubtitleTranslationRequest) -> TranslationResult:
        requests.append(request)
        return TranslationResult(output_path=Path("/tmp/out.srt"), translated_events=1)

    monkeypatch.setattr(cli, "run_subtitle_translation", fake_service)
    runner = CliRunner()

    result = runner.invoke(
        cli.main,
        [
            subtitle_samples[0].filename,
            "--source-lang",
            "en",
            "--target-lang",
            "pt-BR",
            "--retain-history",
        ],
    )

    assert result.exit_code == 0, result.output
    assert requests[0].translator_config.retain_history is True
