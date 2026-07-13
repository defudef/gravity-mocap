#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UVX_BIN="${UVX_BIN:-$HOME/.local/bin/uvx}"
DATA_ROOT="${GRAVITY_MOCAP_DATA_ROOT:-$REPO_ROOT/Saved/GravityMocap}"
MLFLOW_ROOT="$DATA_ROOT/mlflow"
PORT="${MLFLOW_PORT:-5000}"
MLFLOW_UI_VERSION="${MLFLOW_UI_VERSION:-3.14.0}"

if [[ ! -x "$UVX_BIN" ]]; then
  echo "uvx not found at $UVX_BIN. Run ./scripts/setup.sh first." >&2
  exit 1
fi

mkdir -p "$MLFLOW_ROOT/artifacts"
cd "$REPO_ROOT"
exec "$UVX_BIN" --from "mlflow==$MLFLOW_UI_VERSION" mlflow server \
  --host 127.0.0.1 \
  --port "$PORT" \
  --backend-store-uri "sqlite:///$MLFLOW_ROOT/mlflow.db" \
  --default-artifact-root "file://$MLFLOW_ROOT/artifacts"
