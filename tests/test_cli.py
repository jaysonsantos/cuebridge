from __future__ import annotations

import re
from pathlib import Path

import pysubs2
from click.testing import CliRunner
from cuebridge import cli
from cuebridge.naming import build_output_path

SEGMENT_MARKER_RE = re.compile(r"(\[\[SEG_\d+]])")


class FakeTranslator:
    def __init__(self, target_lang_code: str) -> None:
        self.target_lang_code = target_lang_code

    def translate_text(self, text: str) -> str:
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


def test_cli_translates_samples_and_uses_target_language_suffix(
    monkeypatch,
    subtitle_samples,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_builder(**kwargs):
        calls.append((kwargs["source_lang_code"], kwargs["target_lang_code"]))
        return FakeTranslator(kwargs["target_lang_code"])

    monkeypatch.setattr(cli, "build_subtitle_translator", fake_builder)
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
        assert output_path.exists()

        translated = pysubs2.load(str(output_path))
        assert translated[0].text.startswith("[pt-BR]")

    assert calls == [("en", "pt-BR"), ("es", "pt-BR")]
