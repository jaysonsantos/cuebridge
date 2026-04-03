# CueBridge

CLI subtitle translator powered by LLMs.

## Setup

1. Accept the TranslateGemma license on Hugging Face: <https://huggingface.co/google/translategemma-4b-it>
2. If you use Nix, enter the dev shell:

```bash
nix develop
```

This shell now provides a locked Python environment from `pyproject.toml` and `uv.lock`, so you can run `just test`, `just lint`, `just all`, or `cuebridge` immediately without `uv sync`.

3. If you are not using Nix, sync dependencies with `uv`:

```bash
uv sync --dev
```

4. Run the common project tasks with `just`:

```bash
just lint
just test
just all
```

## Citation

CueBridge can be used with Google's TranslateGemma model family. For attribution and scholarly citation, cite the TranslateGemma technical report rather than only the model access page.

- Model access and license acceptance: <https://huggingface.co/google/translategemma-4b-it>
- Technical report: <https://arxiv.org/pdf/2601.09012>

```bibtex
@article{gemmatranslate2026,
    title={{TranslateGemma Technical Report}},
    url={https://arxiv.org/pdf/2601.09012},
    publisher={Google DeepMind},
    author={{Google Translate Research Team} and
    Finkelstein, Mara and
    Caswell, Isaac and
    Domhan, Tobias and
    Peter, Jan-Thorsten and
    Juraska, Juraj and
    Riley, Parker and
    Deutsch, Daniel and
    Dilanni, Cole and
    Cherry, Colin and
    Briakou, Eleftheria and
    Nielsen, Elizabeth and
    Luo, Jiaming and
    Agrawal, Sweta and
    Xu, Wenda and
    Kats, Erin and
    Jaskiewicz, Stephane and
    Freitag, Markus and
    Vilar, David
},
    year={2026}
}
```

## Nix

Build the installable package:

```bash
nix build .#cuebridge
```

Run the CLI directly from the flake:

```bash
nix run .#cuebridge -- subtitles/movie.de.srt --source-lang de --target-lang pt-BR
```

The flake also exports an overlay, so another flake can consume it as `cuebridge`:

```nix
{
  inputs.cuebridge.url = "github:your-user/cuebridge";

  outputs = { self, nixpkgs, cuebridge, ... }: {
    overlays.default = nixpkgs.lib.composeManyExtensions [
      cuebridge.overlays.default
    ];
  };
}
```

## Usage

```bash
uv run cuebridge subtitles/movie.de.srt \
  --source-lang de \
  --target-lang pt-BR
```

Video containers with embedded subtitle tracks work too:

```bash
uv run cuebridge "/path/to/S01E03 - Second of His Name.mkv" \
  --source-lang en \
  --target-lang pt-BR
```

The default translation mode groups `4` subtitle events per request to give TranslateGemma more local context. Set `--window-size 1` if you want strict per-cue translation.

The output `.srt` is rewritten after each translated chunk by default, so you can watch progress in the final destination file while the job is still running. Use `--flush-every-chunks` to change that cadence.

Internally, translators now expose both a one-shot `translate_text(...)` path and a streaming `translate_text_stream(...)` path. Streaming yields append-only translation fragments in order; backends that cannot truly stream yet bridge by yielding a single final chunk.

Cancellation is cooperative and best-effort rather than immediate. CueBridge checks for cancellation before starting the next subtitle window and between emitted translation chunks, but an in-flight backend request may still complete. That means partial stream output may already have been yielded, and completed subtitle windows may already have been flushed to disk, by the time cancellation is observed.

For video inputs, CueBridge probes subtitle streams with `ffprobe` and picks the stream that best matches `--source-lang`. Pass `--subtitle-stream` if you want a specific 0-based subtitle stream instead.

Text subtitle streams are extracted with `ffmpeg` and then translated through the normal `.srt` pipeline. Bitmap subtitle streams such as PGS, VobSub, or DVB subtitles are OCRed into a temporary `.srt` first. That bitmap path only requires `tesseract` when you actually select an image-based subtitle stream, so plain `.srt` files and text subtitle streams do not need it installed.

If bitmap OCR needs a specific Tesseract language pack, pass `--ocr-language` explicitly, for example `--ocr-language eng` or `--ocr-language deu`.

## Known Models

These are the model setups currently known to work well enough to be worth trying:

| Model | Backend | Status | Notes |
| --- | --- | --- | --- |
| `google/translategemma-4b-it` | `hf-local` | Stable default | Best fit when you want the project default and can run the Hugging Face model locally |
| `mlx-community/translategemma-4b-it-4bit` | OpenAI-compatible | Known-good | Good fit for LM Studio or other OpenAI-compatible local runtimes |
| `liquid/lfm2.5-1.2b` | OpenAI-compatible | Fastest benchmarked local model | Current recommendation in [`docs/models-benchmark.md`](docs/models-benchmark.md) |
| `openai/gpt-5.4-nano` | OpenRouter | Viable hosted option | Correct output, but slower than the best local benchmark |
| `gemma4:latest` | Ollama via OpenAI-compatible | Usable with tuning | Works, but needs extra speed tuning because it is a thinking-capable general model |
| `gemma4:e2b` | Ollama via OpenAI-compatible | Faster tradeoff | Smaller Gemma 4 variant if you want more speed and can accept some quality risk |

If you are unsure where to start, use `google/translategemma-4b-it` with `hf-local`, or `liquid/lfm2.5-1.2b` on an OpenAI-compatible local server when throughput matters more.

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

CueBridge also supports an optional `--reasoning-effort` flag for OpenAI-compatible backends that expose thinking control. Accepted values are `none`, `low`, `medium`, and `high`.

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

### Ollama / Gemma 4

Ollama's `gemma4` models work through the same OpenAI-compatible path. They are usable for subtitle translation, but the speed profile is different from TranslateGemma or smaller non-thinking models.

- `gemma4:latest` is a thinking-capable general model, so the default behavior can spend tokens on reasoning that do not help subtitle translation.
- `--reasoning-effort none` is the first knob to try when you want better throughput.
- `--retain-history` usually hurts speed here because it shrinks the auto window size and adds extra prompt baggage across requests.
- `--max-new-tokens 4096` is usually much higher than needed for subtitle windows. Start closer to `512` or `768`.
- `--flush-every-chunks` mainly changes output file rewrite frequency. It is worth increasing for less I/O, but it is a smaller win than windowing or reasoning control.
- If you want an even faster Gemma 4 tradeoff, try `gemma4:e2b` instead of `gemma4:latest`.

Recommended starting point for Ollama:

```bash
uv run cuebridge "/path/to/episode.mkv" \
  --source-lang en \
  --target-lang pt-BR \
  --backend openai-compatible \
  --api-base-url http://localhost:11434/v1 \
  --model-id gemma4:latest \
  --reasoning-effort none \
  --window-size 16 \
  --flush-every-chunks 25 \
  --max-new-tokens 768
```

If the model still feels slow, make changes in this order:

1. Remove `--retain-history` if you enabled it.
2. Lower `--max-new-tokens`.
3. Raise `--window-size` modestly, for example from `12` to `16`.
4. Try `gemma4:e2b`.

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
