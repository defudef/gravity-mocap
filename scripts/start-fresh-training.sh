#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${GRAVITY_MOCAP_DATA_ROOT:-$REPO_ROOT/Saved/GravityMocap}"
CONFIG="${GRAVITY_MOCAP_CONFIG:-$REPO_ROOT/configs/train-paper.yaml}"
MAX_HOURS=""
MAX_EPOCHS=""
EXECUTE=false

usage() {
  cat <<'EOF'
Usage: ./scripts/start-fresh-training.sh [--execute] [--max-hours HOURS | --max-epochs EPOCHS]

Prepare the currently supported motion corpus (CMU + AddBiomechanics +
100STYLE), archive the existing default run, and start a fresh training session.
The default is a side-effect-free plan. Add --execute to download, preprocess,
archive, and train.

Options:
  --execute            Perform the plan and start training.
  --max-hours HOURS    Stop safely after this many hours (default: 2).
  --max-epochs EPOCHS  Stop safely after this many completed epochs.
  -h, --help           Show this help.

Environment overrides:
  GRAVITY_MOCAP_DATA_ROOT  Raw, processed, run, and MLflow root.
  GRAVITY_MOCAP_OUTPUT     Fresh run output to archive/recreate.
  GRAVITY_MOCAP_CONFIG     Training config passed through scripts/train.sh.
  UV_BIN                  uv executable used by setup and the CLI wrappers.
EOF
}

while (($#)); do
  case "$1" in
    --execute)
      EXECUTE=true
      shift
      ;;
    --max-hours)
      if (($# < 2)); then
        echo "--max-hours requires a value" >&2
        exit 2
      fi
      MAX_HOURS="$2"
      shift 2
      ;;
    --max-hours=*)
      MAX_HOURS="${1#*=}"
      shift
      ;;
    --max-epochs)
      if (($# < 2)); then
        echo "--max-epochs requires a value" >&2
        exit 2
      fi
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --max-epochs=*)
      MAX_EPOCHS="${1#*=}"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$MAX_HOURS" && -n "$MAX_EPOCHS" ]]; then
  echo "--max-hours and --max-epochs are mutually exclusive" >&2
  exit 2
fi
if [[ -z "$MAX_HOURS" && -z "$MAX_EPOCHS" ]]; then
  MAX_HOURS="2"
fi
if [[ -n "$MAX_HOURS" ]]; then
  if [[ ! "$MAX_HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
    ! awk -v value="$MAX_HOURS" 'BEGIN { exit !(value > 0) }'; then
    echo "--max-hours must be a positive number" >&2
    exit 2
  fi
  LIMIT_ARGS=(--max-hours "$MAX_HOURS")
  LIMIT_DESCRIPTION="$MAX_HOURS hour(s)"
else
  if [[ ! "$MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-epochs must be a positive integer" >&2
    exit 2
  fi
  LIMIT_ARGS=(--max-epochs "$MAX_EPOCHS")
  LIMIT_DESCRIPTION="$MAX_EPOCHS completed epoch(s)"
fi

case "$DATA_ROOT" in
  /*) ;;
  *) DATA_ROOT="$REPO_ROOT/$DATA_ROOT" ;;
esac
OUTPUT="${GRAVITY_MOCAP_OUTPUT:-$DATA_ROOT/runs/motion}"
case "$OUTPUT" in
  /*) ;;
  *) OUTPUT="$REPO_ROOT/$OUTPUT" ;;
esac

export GRAVITY_MOCAP_DATA_ROOT="$DATA_ROOT"
export GRAVITY_MOCAP_OUTPUT="$OUTPUT"

count_shards() {
  local directory="$1"
  if [[ ! -d "$directory" ]]; then
    echo 0
    return
  fi
  find "$directory" -type f -name '*.npz' | wc -l | tr -d ' '
}

training_is_active() {
  ps -ax -o command= | awk -v output="$OUTPUT" '
    index($0, "gravity-mocap train") && index($0, "--output " output) { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

echo "[fresh-train] repository: $REPO_ROOT"
echo "[fresh-train] data root:  $DATA_ROOT"
echo "[fresh-train] output:     $OUTPUT"
echo "[fresh-train] session:    $LIMIT_DESCRIPTION"

if [[ "$EXECUTE" != true ]]; then
  echo
  echo "PLAN:"
  echo "1. Sync the locked environment and run audit/forward validation."
  echo "2. Download and preprocess CMU, study-stratified AddBiomechanics train data, and 100STYLE."
  echo "3. Require at least one processed shard from each source."
  if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
    echo "4. Archive the existing output below $(dirname "$OUTPUT")/archive/."
  else
    echo "4. No existing output needs archiving."
  fi
  echo "5. Validate a fresh training plan and train for at most $LIMIT_DESCRIPTION."
  echo
  echo "DRY RUN: nothing installed, downloaded, preprocessed, archived, or trained."
  echo "Add --execute to perform this plan."
  exit 0
fi

if training_is_active; then
  echo "A training process is already using $OUTPUT; stop it before starting fresh." >&2
  exit 1
fi

echo "[fresh-train] syncing environment"
"$REPO_ROOT/scripts/setup.sh"

echo "[fresh-train] validating catalog and model"
"$REPO_ROOT/scripts/mocap.sh" audit
"$REPO_ROOT/scripts/mocap.sh" validate

echo "[fresh-train] downloading trainable motion sources"
"$REPO_ROOT/scripts/mocap.sh" download \
  --profile core \
  --dataset cmu_mocap \
  --dataset addbiomechanics \
  --dataset 100style \
  --execute

echo "[fresh-train] preprocessing trainable motion sources"
"$REPO_ROOT/scripts/mocap.sh" preprocess \
  --config "$CONFIG" \
  --profile core \
  --dataset cmu_mocap \
  --dataset addbiomechanics \
  --dataset 100style

CMU_SHARDS="$(count_shards "$DATA_ROOT/processed/cmu_mocap")"
ADDBIO_SHARDS="$(count_shards "$DATA_ROOT/processed/addbiomechanics")"
STYLE_SHARDS="$(count_shards "$DATA_ROOT/processed/100style")"
echo "[fresh-train] processed shards: cmu_mocap=$CMU_SHARDS addbiomechanics=$ADDBIO_SHARDS 100style=$STYLE_SHARDS"
if ((CMU_SHARDS < 1 || ADDBIO_SHARDS < 1 || STYLE_SHARDS < 1)); then
  echo "CMU, AddBiomechanics, and 100STYLE must each produce at least one shard; training aborted." >&2
  exit 1
fi

PLAN_OUTPUT="${OUTPUT}.fresh-plan-$$"
if [[ -e "$PLAN_OUTPUT" || -L "$PLAN_OUTPUT" ]]; then
  echo "Temporary plan output already exists: $PLAN_OUTPUT" >&2
  exit 1
fi
echo "[fresh-train] validating fresh training plan"
GRAVITY_MOCAP_OUTPUT="$PLAN_OUTPUT" \
  "$REPO_ROOT/scripts/train.sh" "${LIMIT_ARGS[@]}" --resume never
if [[ -e "$PLAN_OUTPUT" || -L "$PLAN_OUTPUT" ]]; then
  echo "Training dry-run unexpectedly created $PLAN_OUTPUT; refusing to archive the old run." >&2
  exit 1
fi

if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  ARCHIVE_ROOT="$(dirname "$OUTPUT")/archive"
  mkdir -p "$ARCHIVE_ROOT"
  ARCHIVE="$ARCHIVE_ROOT/$(basename "$OUTPUT")-$(date -u +%Y%m%d-%H%M%S)"
  if [[ -e "$ARCHIVE" || -L "$ARCHIVE" ]]; then
    ARCHIVE="${ARCHIVE}-$$"
  fi
  mv "$OUTPUT" "$ARCHIVE"
  echo "[fresh-train] previous run archived at $ARCHIVE"
fi

echo "[fresh-train] starting a new training session"
exec "$REPO_ROOT/scripts/train.sh" --execute "${LIMIT_ARGS[@]}" --resume never
