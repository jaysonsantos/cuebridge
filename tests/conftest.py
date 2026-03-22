from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class SubtitleSample:
    source_lang_code: str
    filename: str
    content: str


@pytest.fixture
def subtitle_samples(tmp_path: Path) -> list[SubtitleSample]:
    samples = [
        SubtitleSample(
            source_lang_code="en",
            filename="movie.en.srt",
            content="""1
00:00:01,000 --> 00:00:02,500
Hello there!

2
00:00:03,000 --> 00:00:04,500
How are you?
""",
        ),
        SubtitleSample(
            source_lang_code="es",
            filename="movie.es.srt",
            content="""1
00:00:01,000 --> 00:00:02,500
Hola a todos.

2
00:00:03,000 --> 00:00:04,500
Nos vemos pronto.
""",
        ),
    ]

    materialized_samples: list[SubtitleSample] = []
    for sample in samples:
        path = tmp_path / sample.filename
        path.write_text(sample.content, encoding="utf-8")
        materialized_samples.append(
            SubtitleSample(
                source_lang_code=sample.source_lang_code,
                filename=str(path),
                content=sample.content,
            )
        )

    return materialized_samples
