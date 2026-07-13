#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

if [[ ! -x "$UV_BIN" ]]; then
  echo "uv not found at $UV_BIN. Run ./scripts/setup.sh first." >&2
  exit 1
fi

cd "$REPO_ROOT"
exec "$UV_BIN" run --frozen gravity-mocap "$@"
