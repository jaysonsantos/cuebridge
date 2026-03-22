# CueBridge

Translate subtitle files with `google/translategemma-4b-it`, `click`, `loguru`, `pysubs2`, and a small LangChain/LangGraph wrapper that keeps recent conversation history trimmed across subtitle lines.

## Setup

1. Accept the TranslateGemma license on Hugging Face: <https://huggingface.co/google/translategemma-4b-it>
2. Enter the dev shell if you use Nix:

```bash
nix develop
```

3. Sync dependencies:

```bash
uv sync --dev
```

4. Run the common project tasks with `just`:

```bash
just lint
just test
just all
```

## Usage

```bash
uv run cuebridge subtitles/movie.de.srt \
  --source-lang de \
  --target-lang pt-BR
```

The default translation mode groups `4` subtitle events per request to give TranslateGemma more local context. Set `--window-size 1` if you want strict per-cue translation.

The output `.srt` is rewritten after each translated chunk by default, so you can watch progress in the final destination file while the job is still running. Use `--flush-every-chunks` to change that cadence.

### LM Studio / OpenAI-Compatible

You can also target an OpenAI-compatible `/v1/chat/completions` server such as LM Studio:

```bash
uv run cuebridge subtitles/movie.de.srt \
  --backend openai-compatible \
  --api-base-url http://localhost:1234/v1 \
  --model-id mlx-community/translategemma-4b-it-4bit \
  --source-lang de \
  --target-lang pt-BR
```

For OpenRouter or similar services, point `--api-base-url` at the provider and either export an API key in `OPENAI_API_KEY` or pass `--api-key` directly:

```bash
uv run cuebridge subtitles/movie.de.srt \
  --backend openai-compatible \
  --api-base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --model-id your/provider-model \
  --source-lang de \
  --target-lang pt-BR
```

You can also run it as a module:

```bash
uv run python -m cuebridge subtitles/movie.de.srt \
  --source-lang de \
  --target-lang pt-BR
```

If no `--output` path is given, the CLI replaces the trailing language code in the filename when present:

- `abc.de.srt` -> `abc.pt-BR.srt`
- `abc.srt` -> `abc.pt-BR.srt`

## Tests

```bash
just test
```

## Formatting And Hooks

```bash
just lint
```

Repository text files are also normalized through `.editorconfig`: LF line endings, a final newline, trimmed trailing whitespace, 4-space Python indentation, and 2-space YAML indentation.

The hook stack also includes a staged `gitleaks` scan and private-key detection to catch accidentally committed credentials before they land in Git history. `just lint` also runs a full working-tree `gitleaks` scan for CI and local verification.
