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
