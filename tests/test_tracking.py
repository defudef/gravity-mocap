import io
import json
from pathlib import Path

import torch
from mlflow.tracking import MlflowClient

from gravity_mocap.checkpoint import TrainingProgress
from gravity_mocap.tracking import MLflowTracker, ProgressLogger, format_duration
from gravity_mocap.trainer import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_progress_logger_prints_immediately_readable_progress() -> None:
    stream = io.StringIO()
    logger = ProgressLogger(stream)
    progress = TrainingProgress(next_epoch=8, global_step=7, elapsed_seconds=52.9)

    logger.start(
        resumed=True,
        device=torch.device("mps"),
        parameters=123456,
        windows=10,
        progress=progress,
        epochs=500,
        total_steps=500,
    )
    logger.step(
        epoch=8,
        epochs=500,
        batch=1,
        batches=1,
        global_step=8,
        total_steps=500,
        loss=0.3694389,
        learning_rate=0.0002,
        elapsed_seconds=61.3,
    )
    logger.validation_start(epoch=8, epochs=500, windows=1025)
    logger.validation(
        epoch=8,
        epochs=500,
        metrics={
            "loss.total": 0.488491,
            "mpjpe_m": 0.29733,
            "root_local_drift_m": 0.59966,
            "contact_f1": 0.25,
        },
        elapsed_seconds=25.6,
        improved=True,
        best_loss=0.488491,
        validations_without_improvement=0,
        patience=20,
    )

    output = stream.getvalue()
    assert "[train] RESUME" in output
    assert "epoch 8/500" in output
    assert "step 8/500" in output
    assert "loss 0.369439" in output
    assert "ETA" in output
    assert "[validation] START | epoch 8/500 | windows=1,025" in output
    assert "val_loss=0.488491" in output
    assert "MPJPE=29.73cm" in output
    assert "early_stop=0/20" in output
    assert format_duration(3661) == "1h 01m"


def test_mlflow_logs_and_resumes_one_run_without_training(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    config = load_config(ROOT / "configs/train-smoke.yaml")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    config["logging"]["mlflow"].update(
        {
            "tracking_uri": tracking_uri,
            "experiment_name": "gravity-mocap-test",
            "run_name": "test-run",
        }
    )
    data_bom = {"schema_version": 1, "shard_count": 0, "shards": []}
    (output / "resolved-config.json").write_text(json.dumps(config))
    (output / "data-bom.json").write_text(json.dumps(data_bom))
    logger = ProgressLogger(io.StringIO())
    tracker = MLflowTracker(
        config=config,
        output=output,
        config_path=ROOT / "configs/train-smoke.yaml",
        data_bom=data_bom,
        logger=logger,
    )

    run_id = tracker.start(
        existing_run_id=None,
        may_reuse_run_file=False,
        resumed_from=None,
        device=torch.device("cpu"),
    )
    assert run_id is not None
    tracker.log_step(
        metrics={"total": 0.5, "rotations": 0.25},
        global_step=7,
        epoch=3,
        learning_rate=0.001,
        elapsed_seconds=14.0,
    )
    tracker.log_epoch(epoch=3, metrics={"total": 0.6}, global_step=7)
    tracker.log_validation(
        epoch=3,
        metrics={"loss.total": 0.7, "mpjpe_m": 0.08},
        global_step=7,
    )
    checkpoint = output / "latest.pt"
    checkpoint.write_bytes(b"not a real checkpoint; tracking-only test")
    (output / "training-state.json").write_text(json.dumps({"global_step": 7}))
    tracker.log_checkpoint(
        checkpoint,
        TrainingProgress(global_step=7, mlflow_run_id=run_id),
        reason="test",
    )
    tracker.finish(status="KILLED", reason="max_hours")

    client = MlflowClient(tracking_uri=tracking_uri)
    first = client.get_run(run_id)
    assert first.info.status == "KILLED"
    assert first.data.metrics["loss.total"] == 0.5
    assert first.data.metrics["epoch_loss.total"] == 0.6
    assert first.data.metrics["validation.loss.total"] == 0.7
    assert first.data.metrics["validation.mpjpe_m"] == 0.08
    assert first.data.params["config.model.hidden_dim"] == "64"
    assert {item.path for item in client.list_artifacts(run_id, "inputs")} == {
        "inputs/data-bom.json",
        "inputs/resolved-config.json",
    }
    checkpoint_artifacts = {item.path for item in client.list_artifacts(run_id, "checkpoint")}
    assert "checkpoint/checkpoint-manifest.json" in checkpoint_artifacts
    assert "checkpoint/training-state.json" in checkpoint_artifacts

    resumed = MLflowTracker(
        config=config,
        output=output,
        config_path=ROOT / "configs/train-smoke.yaml",
        data_bom=data_bom,
        logger=logger,
    )
    resumed_id = resumed.start(
        existing_run_id=run_id,
        may_reuse_run_file=True,
        resumed_from=str(checkpoint),
        device=torch.device("cpu"),
    )
    assert resumed_id == run_id
    resumed.log_step(
        metrics={"total": 0.4},
        global_step=8,
        epoch=4,
        learning_rate=0.001,
        elapsed_seconds=16.0,
    )
    resumed.finish(status="FINISHED", reason="completed")

    final = client.get_run(run_id)
    assert final.info.status == "FINISHED"
    assert final.data.metrics["loss.total"] == 0.4
