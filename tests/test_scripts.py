import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRESH_TRAIN = PROJECT_ROOT / "scripts/start-fresh-training.sh"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(textwrap.dedent(source).lstrip())
    path.chmod(0o755)


def test_fresh_training_dry_run_is_path_agnostic_and_side_effect_free(tmp_path: Path) -> None:
    data_root = tmp_path / "external-data"
    result = subprocess.run(
        [str(FRESH_TRAIN), "--max-hours", "2"],
        cwd=tmp_path,
        env={**os.environ, "GRAVITY_MOCAP_DATA_ROOT": str(data_root)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"repository: {PROJECT_ROOT}" in result.stdout
    assert f"data root:  {data_root}" in result.stdout
    assert "DRY RUN: nothing installed, downloaded" in result.stdout
    assert not data_root.exists()


def test_fresh_training_rejects_non_positive_duration(tmp_path: Path) -> None:
    result = subprocess.run(
        [str(FRESH_TRAIN), "--max-hours", "0"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--max-hours must be a positive number" in result.stderr


def test_fresh_training_accepts_epoch_limit_without_side_effects(tmp_path: Path) -> None:
    data_root = tmp_path / "external-data"
    result = subprocess.run(
        [str(FRESH_TRAIN), "--max-epochs", "3"],
        cwd=tmp_path,
        env={**os.environ, "GRAVITY_MOCAP_DATA_ROOT": str(data_root)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "session:    3 completed epoch(s)" in result.stdout
    assert "train for at most 3 completed epoch(s)" in result.stdout
    assert not data_root.exists()


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "nope"])
def test_fresh_training_rejects_invalid_epoch_limit(tmp_path: Path, value: str) -> None:
    result = subprocess.run(
        [str(FRESH_TRAIN), "--max-epochs", value],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--max-epochs must be a positive integer" in result.stderr


def test_fresh_training_rejects_two_session_limits(tmp_path: Path) -> None:
    result = subprocess.run(
        [str(FRESH_TRAIN), "--max-hours", "2", "--max-epochs", "3"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--max-hours and --max-epochs are mutually exclusive" in result.stderr


@pytest.mark.parametrize(
    ("limit_args", "expected_limit"),
    [
        (["--max-hours", "0.01"], "--max-hours 0.01"),
        (["--max-epochs", "3"], "--max-epochs 3"),
    ],
)
def test_fresh_training_executes_in_order_and_archives_previous_run(
    tmp_path: Path, limit_args: list[str], expected_limit: str
) -> None:
    fake_repo = tmp_path / "portable-checkout"
    scripts = fake_repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(FRESH_TRAIN, scripts / FRESH_TRAIN.name)
    log = tmp_path / "calls.log"
    data_root = tmp_path / "external-data"
    old_output = data_root / "runs/motion"
    old_output.mkdir(parents=True)
    (old_output / "old-checkpoint.pt").write_bytes(b"old")

    _write_executable(
        scripts / "setup.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        echo setup >> "$FAKE_LOG"
        """,
    )
    _write_executable(
        scripts / "mocap.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        echo "mocap $*" >> "$FAKE_LOG"
        if [[ "${1:-}" == "preprocess" ]]; then
          mkdir -p \
            "$GRAVITY_MOCAP_DATA_ROOT/processed/cmu_mocap" \
            "$GRAVITY_MOCAP_DATA_ROOT/processed/addbiomechanics"
          touch "$GRAVITY_MOCAP_DATA_ROOT/processed/cmu_mocap/sequence.npz"
          touch "$GRAVITY_MOCAP_DATA_ROOT/processed/addbiomechanics/sequence.npz"
        fi
        """,
    )
    _write_executable(
        scripts / "train.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        echo "train output=$GRAVITY_MOCAP_OUTPUT args=$*" >> "$FAKE_LOG"
        if [[ " $* " == *" --execute "* ]]; then
          mkdir -p "$GRAVITY_MOCAP_OUTPUT"
          touch "$GRAVITY_MOCAP_OUTPUT/latest.pt"
        fi
        """,
    )

    result = subprocess.run(
        [str(scripts / FRESH_TRAIN.name), "--execute", *limit_args],
        cwd=tmp_path,
        env={
            **os.environ,
            "FAKE_LOG": str(log),
            "GRAVITY_MOCAP_DATA_ROOT": str(data_root),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert calls[:4] == [
        "setup",
        "mocap audit",
        "mocap validate",
        "mocap download --profile core --dataset cmu_mocap --dataset addbiomechanics --execute",
    ]
    assert calls[4] == (
        "mocap preprocess --profile core --dataset cmu_mocap --dataset addbiomechanics"
    )
    assert ".fresh-plan-" in calls[5]
    assert f"args={expected_limit} --resume never" in calls[5]
    assert calls[6] == (f"train output={old_output} args=--execute {expected_limit} --resume never")

    archives = list((data_root / "runs/archive").glob("motion-*"))
    assert len(archives) == 1
    assert (archives[0] / "old-checkpoint.pt").read_bytes() == b"old"
    assert (old_output / "latest.pt").exists()
