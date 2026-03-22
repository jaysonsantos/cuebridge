from __future__ import annotations

import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from loguru import logger

from cuebridge.service import (
    RuntimeOptions,
    SubtitleTranslationRequest,
    TranslatorConfig,
    run_subtitle_translation,
)

DEFAULT_MODEL_ID = "google/translategemma-4b-it"


@click.command(context_settings={"show_default": True})
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--source-lang", required=True, help="Source language code, for example en or pt-BR.")
@click.option(
    "--target-lang", required=True, help="Target language code, for example de-DE or pt-BR."
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional output path. Defaults to replacing or appending the language code in the filename.",
)
@click.option(
    "--backend",
    default="hf-local",
    type=click.Choice(
        ["hf-local", "openai-compatible", "cerebras", "openrouter"],
        case_sensitive=False,
    ),
    help="Model backend. Known OpenAI-compatible providers fill in their default base URL.",
)
@click.option("--model-id", default=DEFAULT_MODEL_ID, help="Backend-specific model id.")
@click.option(
    "--dtype", default="bfloat16", help="Torch dtype passed to the Transformers pipeline."
)
@click.option("--device", default=None, help="Optional explicit device like cpu, cuda, or cuda:0.")
@click.option("--max-new-tokens", default=256, type=click.IntRange(min=1))
@click.option(
    "--batch-size",
    default=1,
    type=click.IntRange(min=1),
    help="Batch size for the hf-local backend. OpenAI-compatible backends ignore this.",
)
@click.option(
    "--api-base-url",
    default=None,
    help="Optional override for OpenAI-compatible endpoints. Known providers set this automatically.",
)
@click.option(
    "--message-format",
    default="auto",
    type=click.Choice(["auto", "plain", "translategemma"], case_sensitive=False),
    help="How to format user messages for OpenAI-compatible endpoints.",
)
@click.option("--api-key", default=None, help="Optional API key for OpenAI-compatible endpoints.")
@click.option(
    "--api-key-env",
    default=None,
    help="Optional environment variable to read the API key from when --api-key is omitted.",
)
@click.option("--request-timeout-seconds", default=120.0, type=float)
@click.option(
    "--window-size",
    default=None,
    type=click.IntRange(min=1),
    help="Optional subtitle window size. Defaults to an auto-tuned value for the selected backend.",
)
@click.option(
    "--flush-every-chunks",
    default=1,
    type=click.IntRange(min=1),
    help="How many translated chunks to process before rewriting the output .srt.",
)
@click.option(
    "--max-input-tokens",
    default=1800,
    type=click.IntRange(min=128),
    help="Token budget for retained chat history when --retain-history is enabled.",
)
@click.option("--thread-id", default=None, help="Optional LangGraph checkpoint thread id.")
@click.option(
    "--retain-history",
    is_flag=True,
    help="Reuse prior translation turns as chat history across subtitle chunks.",
)
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(
    input_path: Path,
    source_lang: str,
    target_lang: str,
    output_path: Path | None,
    backend: str,
    model_id: str,
    dtype: str,
    device: str | None,
    max_new_tokens: int,
    batch_size: int,
    api_base_url: str | None,
    message_format: str,
    api_key: str | None,
    api_key_env: str | None,
    request_timeout_seconds: float,
    window_size: int | None,
    flush_every_chunks: int,
    max_input_tokens: int,
    thread_id: str | None,
    retain_history: bool,
    verbose: bool,
) -> None:
    """Translate a subtitle file with TranslateGemma."""
    load_dotenv()
    configure_logging(verbose=verbose)

    result = run_subtitle_translation(
        SubtitleTranslationRequest(
            input_source=input_path,
            source_lang_code=source_lang,
            target_lang_code=target_lang,
            output_path=output_path,
            translator_config=TranslatorConfig(
                backend=backend,
                model_id=model_id,
                dtype=dtype,
                device=device,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                api_base_url=api_base_url,
                message_format=message_format,
                api_key=api_key,
                api_key_env=api_key_env,
                request_timeout_seconds=request_timeout_seconds,
                max_input_tokens=max_input_tokens,
                thread_id=thread_id,
                retain_history=retain_history,
            ),
            runtime_options=RuntimeOptions(
                window_size=window_size,
                flush_every_chunks=flush_every_chunks,
            ),
        )
    )

    click.echo(str(result.output_path))


def configure_logging(*, verbose: bool) -> None:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")
