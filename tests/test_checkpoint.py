import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gravity_mocap.checkpoint import (
    CHECKPOINT_VERSION,
    TrainingProgress,
    cleanup_stale_checkpoint_temps,
    compatibility_hash,
    load_training_checkpoint,
    promote_best_checkpoint,
    promote_best_pose_checkpoint,
    read_training_state,
    resolve_resume_path,
    save_training_checkpoint,
)
from gravity_mocap.fixture import create_fixture
from gravity_mocap.schema import read_shard, write_shard
from gravity_mocap.trainer import (
    _epoch_session_limit_reached,
    _restore_legacy_validation_progress,
    _restore_pose_validation_progress,
    _update_early_stopping,
    _validate_session_limits,
    build_model,
    load_config,
    training_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def _state() -> tuple[dict, torch.nn.Module, torch.optim.Optimizer, torch.amp.GradScaler]:
    config = load_config(ROOT / "configs/train-smoke.yaml")
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return config, model, optimizer, scaler


def test_checkpoint_round_trip_restores_full_state_without_training(tmp_path: Path) -> None:
    config, model, optimizer, scaler = _state()
    data_bom = {"schema_version": 1, "shard_count": 1, "shards": [{"id": "fixture"}]}
    progress = TrainingProgress(
        next_epoch=7,
        next_batch=32,
        global_step=123,
        elapsed_seconds=456.5,
        last_loss=0.75,
        stop_reason="SIGINT",
        mlflow_run_id="test-run-id",
        best_validation_loss=0.42,
        best_validation_epoch=6,
        best_pose_mpjpe_m=0.05,
        best_pose_epoch=5,
        validations_without_improvement=1,
    )
    random.seed(10)
    np.random.seed(11)
    torch.manual_seed(12)
    original_parameter = next(model.parameters()).detach().clone()
    checkpoint = save_training_checkpoint(
        tmp_path, model, optimizer, scaler, config, data_bom, progress
    )
    expected_random = (random.random(), float(np.random.random()), torch.rand(1))
    with torch.no_grad():
        next(model.parameters()).add_(1)
    random.random()
    np.random.random()
    torch.rand(1)

    loaded = load_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        scaler,
        config,
        data_bom,
        torch.device("cpu"),
    )
    assert loaded == progress
    assert torch.equal(next(model.parameters()), original_parameter)
    assert random.random() == expected_random[0]
    assert float(np.random.random()) == expected_random[1]
    assert torch.equal(torch.rand(1), expected_random[2])
    state = read_training_state(tmp_path)
    assert state is not None
    assert state["next_epoch"] == 7
    assert resolve_resume_path(tmp_path, "auto") == checkpoint


def test_checkpoint_rejects_changed_bom(tmp_path: Path) -> None:
    config, model, optimizer, scaler = _state()
    data_bom = {"schema_version": 1, "shard_count": 1, "shards": []}
    checkpoint = save_training_checkpoint(
        tmp_path,
        model,
        optimizer,
        scaler,
        config,
        data_bom,
        TrainingProgress(),
    )
    with pytest.raises(RuntimeError, match="data BOM differs"):
        load_training_checkpoint(
            checkpoint,
            model,
            optimizer,
            scaler,
            config,
            {**data_bom, "shard_count": 2},
            torch.device("cpu"),
        )


def test_checkpoint_from_before_early_stopping_loads_default_state(tmp_path: Path) -> None:
    config, model, optimizer, scaler = _state()
    data_bom = {"schema_version": 1, "shard_count": 1, "shards": []}
    checkpoint = save_training_checkpoint(
        tmp_path,
        model,
        optimizer,
        scaler,
        config,
        data_bom,
        TrainingProgress(next_epoch=4, global_step=180),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for name in (
        "best_validation_loss",
        "best_validation_epoch",
        "validations_without_improvement",
    ):
        payload["progress"].pop(name)
    torch.save(payload, checkpoint)

    loaded = load_training_checkpoint(
        checkpoint,
        model,
        optimizer,
        scaler,
        config,
        data_bom,
        torch.device("cpu"),
    )

    assert loaded.next_epoch == 4
    assert loaded.best_validation_loss is None
    assert loaded.best_validation_epoch is None
    assert loaded.validations_without_improvement == 0


def test_epoch_limit_can_change_but_optimizer_settings_cannot() -> None:
    config, *_ = _state()
    extended = {**config, "train": {**config["train"], "epochs": 600}}
    assert compatibility_hash(extended) == compatibility_hash(config)
    changed_lr = {
        **config,
        "train": {**config["train"], "learning_rate": 0.0001},
    }
    assert compatibility_hash(changed_lr) != compatibility_hash(config)
    changed_logging = {
        **config,
        "logging": {**config["logging"], "log_every_steps": 20},
    }
    assert compatibility_hash(changed_logging) == compatibility_hash(config)
    changed_augmentation = {
        **config,
        "data": {
            **config["data"],
            "augmentation": {
                **config["data"]["augmentation"],
                "keypoint_noise_std": 0.25,
            },
        },
    }
    assert compatibility_hash(changed_augmentation) != compatibility_hash(config)
    moved_detector_loop = {
        **config,
        "data": {
            **config["data"],
            "detector_loop": {
                **config["data"]["detector_loop"],
                "root": "/another/checkout/Saved/GravityMocap/detector-loop",
            },
        },
    }
    assert compatibility_hash(moved_detector_loop) == compatibility_hash(config)
    changed_detector_loop_mix = {
        **config,
        "data": {
            **config["data"],
            "detector_loop": {
                **config["data"]["detector_loop"],
                "mix_probability": 0.25,
            },
        },
    }
    assert compatibility_hash(changed_detector_loop_mix) != compatibility_hash(config)
    changed_validation_schedule = {
        **config,
        "validation": {**config["validation"], "every_epochs": 10},
    }
    assert compatibility_hash(changed_validation_schedule) == compatibility_hash(config)
    changed_early_stopping = {
        **config,
        "validation": {
            **config["validation"],
            "early_stopping": {
                **config["validation"]["early_stopping"],
                "patience": 30,
            },
        },
    }
    assert compatibility_hash(changed_early_stopping) == compatibility_hash(config)


def test_cleanup_removes_only_uncommitted_checkpoint_temps(tmp_path: Path) -> None:
    latest = tmp_path / "latest.pt"
    latest.write_bytes(b"valid")
    latest_temp = tmp_path / "latest.pt.tmp"
    latest_temp.write_bytes(b"partial")
    state_temp = tmp_path / "training-state.json.tmp"
    state_temp.write_text("partial")
    archive_temp = tmp_path / "epoch-0005.pt.tmp"
    archive_temp.write_bytes(b"partial")
    best_temp = tmp_path / "best.pt.tmp"
    best_temp.write_bytes(b"partial")
    best_pose_temp = tmp_path / "best-pose.pt.tmp"
    best_pose_temp.write_bytes(b"partial")

    removed = cleanup_stale_checkpoint_temps(tmp_path)

    assert set(removed) == {
        latest_temp,
        best_temp,
        best_pose_temp,
        state_temp,
        archive_temp,
    }
    assert latest.read_bytes() == b"valid"
    assert all(not path.exists() for path in removed)


def test_best_checkpoint_is_promoted_atomically(tmp_path: Path) -> None:
    latest = tmp_path / "latest.pt"
    latest.write_bytes(b"best-model-state")
    stale_best = tmp_path / "best.pt"
    stale_best.write_bytes(b"stale")

    best = promote_best_checkpoint(tmp_path)

    assert best == stale_best
    assert best.read_bytes() == b"best-model-state"
    assert not (tmp_path / "best.pt.tmp").exists()


def test_best_pose_checkpoint_is_promoted_separately(tmp_path: Path) -> None:
    latest = tmp_path / "latest.pt"
    latest.write_bytes(b"best-pose-state")

    best_pose = promote_best_pose_checkpoint(tmp_path)

    assert best_pose == tmp_path / "best-pose.pt"
    assert best_pose.read_bytes() == b"best-pose-state"
    assert not (tmp_path / "best-pose.pt.tmp").exists()


def test_training_plan_never_constructs_an_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs/train-smoke.yaml")
    data_root = tmp_path / "fixtures"
    create_fixture(data_root / "synthetic" / "walk.npz", frames=16, image_feature_dim=32)
    config["data"]["root"] = str(data_root)
    config["data"]["catalog"] = str(ROOT / "configs/datasets.yaml")
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not construct an optimizer")

    monkeypatch.setattr(torch.optim, "AdamW", fail_if_called)
    plan = training_plan(config_path, tmp_path / "run", max_epochs=8)

    assert plan["ready"] is True
    assert plan["action_on_execute"] == "start_new"
    assert plan["max_hours_this_session"] is None
    assert plan["max_epochs_this_session"] == 8
    assert plan["validation"]["early_stopping"]["enabled"] is False
    assert plan["progress_logging"]["mlflow_enabled"] is True
    assert not (tmp_path / "run" / "mlflow" / "mlflow.db").exists()


def test_training_plan_reports_old_checkpoint_target_contract(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/train-smoke.yaml")
    data_root = tmp_path / "fixtures"
    create_fixture(data_root / "synthetic" / "walk.npz", frames=16, image_feature_dim=32)
    config["data"]["root"] = str(data_root)
    config["data"]["catalog"] = str(ROOT / "configs/datasets.yaml")
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    output = tmp_path / "run"
    output.mkdir()
    (output / "latest.pt").write_bytes(b"metadata-only dry-run fixture")
    (output / "training-state.json").write_text(
        json.dumps(
            {
                "checkpoint_version": CHECKPOINT_VERSION - 1,
                "compatibility_hash": compatibility_hash(config),
                "data_bom_hash": "stale",
                "next_epoch": 1,
                "complete": False,
                "stop_reason": None,
            }
        )
    )

    plan = training_plan(config_path, output)

    assert plan["ready"] is False
    assert any("saved checkpoint version" in error for error in plan["resume_audit"]["errors"])


def test_training_session_limits_are_positive_and_mutually_exclusive() -> None:
    _validate_session_limits(max_hours=1, max_epochs=None)
    _validate_session_limits(max_hours=None, max_epochs=3)

    with pytest.raises(ValueError, match="max_hours must be positive"):
        _validate_session_limits(max_hours=0, max_epochs=None)
    with pytest.raises(ValueError, match="max_epochs must be positive"):
        _validate_session_limits(max_hours=None, max_epochs=0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_session_limits(max_hours=1, max_epochs=3)


def test_epoch_session_limit_counts_a_resumed_partial_epoch() -> None:
    assert not _epoch_session_limit_reached(epoch=9, session_start_epoch=9, max_epochs=2)
    assert _epoch_session_limit_reached(epoch=10, session_start_epoch=9, max_epochs=2)
    assert _epoch_session_limit_reached(epoch=9, session_start_epoch=9, max_epochs=1)
    assert not _epoch_session_limit_reached(epoch=100, session_start_epoch=9, max_epochs=None)


def test_early_stopping_tracks_improvement_and_patience() -> None:
    progress = TrainingProgress()

    assert _update_early_stopping(progress, epoch=1, validation_loss=0.5, min_delta=0.001)
    assert progress.best_validation_loss == 0.5
    assert progress.best_validation_epoch == 1
    assert progress.validations_without_improvement == 0

    assert not _update_early_stopping(progress, epoch=2, validation_loss=0.4995, min_delta=0.001)
    assert progress.best_validation_loss == 0.5
    assert progress.validations_without_improvement == 1

    assert _update_early_stopping(progress, epoch=3, validation_loss=0.48, min_delta=0.001)
    assert progress.best_validation_loss == 0.48
    assert progress.best_validation_epoch == 3
    assert progress.validations_without_improvement == 0


def test_early_stopping_rejects_non_finite_validation_loss() -> None:
    with pytest.raises(RuntimeError, match="not finite"):
        _update_early_stopping(
            TrainingProgress(), epoch=1, validation_loss=float("nan"), min_delta=0.0
        )


def test_legacy_validation_state_seeds_resumable_early_stopping(tmp_path: Path) -> None:
    (tmp_path / "validation-state.json").write_text('{"epoch": 3, "metrics": {"loss.total": 0.25}}')
    progress = TrainingProgress(next_epoch=4)

    restored = _restore_legacy_validation_progress(progress, tmp_path, monitor="loss.total")

    assert restored is True
    assert progress.best_validation_loss == 0.25
    assert progress.best_validation_epoch == 3
    assert progress.validations_without_improvement == 0


def test_validation_state_seeds_resumable_best_pose(tmp_path: Path) -> None:
    (tmp_path / "validation-state.json").write_text('{"epoch": 3, "metrics": {"mpjpe_m": 0.041}}')
    progress = TrainingProgress(next_epoch=4)

    restored = _restore_pose_validation_progress(progress, tmp_path)

    assert restored is True
    assert progress.best_pose_mpjpe_m == 0.041
    assert progress.best_pose_epoch == 3


def test_early_stopping_config_is_validated(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/train-paper.yaml")
    config["validation"]["early_stopping"]["patience"] = 0
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(ValueError, match="patience must be at least 1"):
        load_config(config_path)


def test_training_plan_reports_sequence_disjoint_validation_split(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/train-smoke.yaml")
    data_root = tmp_path / "fixtures"
    create_fixture(data_root / "synthetic" / "walk-a.npz", frames=16, image_feature_dim=32)
    second_path = create_fixture(
        data_root / "synthetic" / "walk-b.npz", frames=16, image_feature_dim=32
    )
    second_arrays, second_provenance = read_shard(second_path)
    second_provenance["source_sequence"] = "walk-cycle-002"
    write_shard(second_path, second_arrays, second_provenance)
    config["data"].update(
        {
            "root": str(data_root),
            "catalog": str(ROOT / "configs/datasets.yaml"),
            "validation_fraction": 0.5,
        }
    )
    config["validation"]["enabled"] = True
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    plan = training_plan(config_path, tmp_path / "run")

    assert plan["ready"] is True
    assert plan["data_split"]["status"] == "ok"
    assert plan["data_split"]["train_sequences"] == 1
    assert plan["data_split"]["validation_sequences"] == 1
    assert plan["data_split"]["train_windows"] > 0
    assert plan["data_split"]["validation_windows"] > 0
    assert plan["data_split"]["target_fps"] == 30


def test_training_plan_keeps_trials_from_one_source_file_in_one_split(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/train-smoke.yaml")
    data_root = tmp_path / "fixtures"
    paths = [
        create_fixture(
            data_root / "synthetic" / f"trial-{index}.npz",
            frames=16,
            image_feature_dim=32,
        )
        for index in range(4)
    ]
    for index, path in enumerate(paths):
        arrays, provenance = read_shard(path)
        provenance["source_sequence"] = f"subject-a/trial-{index}"
        provenance["split_group"] = "subject-a"
        write_shard(path, arrays, provenance)
    config["data"].update(
        {
            "root": str(data_root),
            "catalog": str(ROOT / "configs/datasets.yaml"),
            "validation_fraction": 0.5,
        }
    )
    config["validation"]["enabled"] = True
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    plan = training_plan(config_path, tmp_path / "run")

    assert plan["ready"] is False
    assert "cannot produce a holdout split" in plan["data_split"]["errors"][0]
