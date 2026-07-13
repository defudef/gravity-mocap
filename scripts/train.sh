#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${GRAVITY_MOCAP_CONFIG:-$REPO_ROOT/configs/train-paper.yaml}"
DATA_ROOT="${GRAVITY_MOCAP_DATA_ROOT:-$REPO_ROOT/Saved/GravityMocap}"
OUTPUT="${GRAVITY_MOCAP_OUTPUT:-$DATA_ROOT/runs/motion}"

exec "$REPO_ROOT/scripts/mocap.sh" train \
  --config "$CONFIG" \
  --output "$OUTPUT" \
  "$@"
