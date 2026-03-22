default:
  @just --list

test:
  uv run pytest

lint:
  gitleaks dir . --redact --no-banner
  uv run prek run --all-files

all: lint test
  @echo "All checks passed."
