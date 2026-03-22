# Repository Guidelines

## Project Structure & Module Organization
`cuebridge/` contains the application code. Keep CLI behavior in `cli.py`, backend/model wiring in `agent.py` and `model.py`, subtitle processing in `subtitles.py`, and filename logic in `naming.py`. Shared interfaces belong in `contracts.py`. Tests live in `tests/` and generally mirror the module they cover, with shared fixtures in `tests/conftest.py`. The `.srt` files in the repository root are sample inputs/outputs for manual validation.

## Build, Test, and Development Commands
Use `uv` for local setup and execution.

- `uv sync --dev`: install runtime and test dependencies into the local virtual environment.
- `nix develop`: enter the Nix dev shell with `python`, `uv`, `just`, `gitleaks`, `ruff`, and `prek` available.
- `just lint`: run a full working-tree `gitleaks` scan and then all configured `prek` checks, including YAML validation, EditorConfig validation, staged secret scanning, and Ruff hooks.
- `just test`: run the full `pytest` suite.
- `just all`: run linting and tests in sequence.
- `uv run cuebridge movie.de.srt --source-lang de --target-lang pt-BR`: run the CLI through the packaged entrypoint.
- `uv run python -m cuebridge movie.de.srt --source-lang de --target-lang pt-BR`: run the module directly while developing.
- `uv run pytest`: run the full test suite.
- `uv run prek run --all-files`: run the configured repository checks through `prek`.
- `uv run prek run check-yaml editorconfig-checker --all-files`: run the non-Python repository hygiene hooks directly.
- `uv run pytest tests/test_cli.py -q`: run a focused test file during iteration.

## Coding Style & Naming Conventions
Target Python 3.12+ and follow the existing style: 4-space indentation, explicit type hints, and `from __future__ import annotations` in modules. Use `snake_case` for modules, functions, and variables, `PascalCase` for classes and dataclasses, and `UPPER_SNAKE_CASE` for constants such as regex patterns. Keep functions small and behavior-specific. `.editorconfig` defines LF endings, final newlines, trailing-whitespace trimming, 4-space Python indentation, and 2-space YAML indentation. Formatting and repository hygiene are enforced through `prek`.

## Testing Guidelines
Tests use `pytest`. Name files `test_<feature>.py` and test functions `test_<behavior>()`. Prefer fixtures and temporary files over real model calls; `tests/test_cli.py` shows the expected pattern with a fake translator and `CliRunner`. When changing translation flow, add coverage for output naming, chunk/window behavior, and written subtitle text.

## Commit & Pull Request Guidelines
This repository currently has no commit history, so there is no established message convention to copy. Use short, imperative subjects and prefer Conventional Commit prefixes like `feat:`, `fix:`, or `test:`. When opening a pull request, use a semantic/conventional-commit PR title as well, because squash merging will use the PR title as the final commit message. Pull requests should include a brief summary, the commands you ran (for example `uv run pytest`), and any representative CLI example or sample subtitle output when behavior changes.

## Security & Configuration Tips
Store API keys in `.env` or environment variables such as `OPENAI_API_KEY`; do not hardcode secrets. If you use Hugging Face locally, accept the TranslateGemma model license before running the CLI.
