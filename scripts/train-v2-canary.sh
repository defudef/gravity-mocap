#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${GRAVITY_MOCAP_DATA_ROOT:-$REPO_ROOT/Saved/GravityMocap}"
case "$DATA_ROOT" in
  /*) ;;
  *) DATA_ROOT="$REPO_ROOT/$DATA_ROOT" ;;
esac

export GRAVITY_MOCAP_OUTPUT="${GRAVITY_MOCAP_OUTPUT:-$DATA_ROOT/runs/motion-small-v2-canary}"
exec "$REPO_ROOT/scripts/train-small.sh" --max-epochs 3 --resume never "$@"
