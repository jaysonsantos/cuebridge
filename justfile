default:
  @just --list

test:
  if [ -n "${IN_NIX_SHELL:-}" ] || [ -n "${VIRTUAL_ENV:-}" ]; then pytest; else uv run pytest; fi

lint:
  gitleaks dir . --redact --no-banner
  if [ -n "${IN_NIX_SHELL:-}" ] || [ -n "${VIRTUAL_ENV:-}" ]; then prek run --all-files; else uv run prek run --all-files; fi

all: lint test
  @echo "All checks passed."
