#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

if [[ ! -x "$UV_BIN" ]]; then
  echo "uv not found at $UV_BIN. Install uv or set UV_BIN=/path/to/uv." >&2
  exit 1
fi

if ! command -v npx >/dev/null 2>&1; then
  echo "Note: automated mRI downloads additionally require Node.js 18+ with npx." >&2
fi

cd "$REPO_ROOT"
exec "$UV_BIN" sync --group dev
