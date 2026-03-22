from __future__ import annotations

import re
from pathlib import Path

LANGUAGE_SUFFIX_PATTERN = re.compile(r"(?i)\.[a-z]{2,3}(?:[-_][a-z0-9]{2,8})?$")
LANGUAGE_TOKEN_PATTERN = re.compile(r"(?i)^[a-z]{2,3}(?:[-_][a-z0-9]{2,8})?$")
SUBTITLE_VARIANT_TOKENS = {"cc", "forced", "hi", "sdh"}


def build_output_path(
    input_path: Path, target_lang_code: str, output_path: Path | None = None
) -> Path:
    if output_path is not None:
        return output_path

    stem = input_path.stem
    new_stem = _replace_trailing_subtitle_tags(stem, target_lang_code)

    return input_path.with_name(f"{new_stem}{input_path.suffix}")


def _replace_trailing_subtitle_tags(stem: str, target_lang_code: str) -> str:
    parts = stem.split(".")
    while parts and parts[-1].lower() in SUBTITLE_VARIANT_TOKENS:
        parts.pop()

    if len(parts) > 1 and LANGUAGE_TOKEN_PATTERN.fullmatch(parts[-1]):
        parts.pop()
    elif LANGUAGE_SUFFIX_PATTERN.search(stem):
        return LANGUAGE_SUFFIX_PATTERN.sub(f".{target_lang_code}", stem)

    return ".".join([*parts, target_lang_code])
