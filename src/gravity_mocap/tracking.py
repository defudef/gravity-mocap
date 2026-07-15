from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import torch

from .checkpoint import CHECKPOINT_VERSION, STATE_FILE, TrainingProgress

MLFLOW_RUN_FILE = "mlflow-run.json"


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "?"
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _flatten(value: Any, prefix: str = "") -> dict[str, str | int | float | bool]:
    flattened: dict[str, str | int | float | bool] = {}
    if isinstance(value, dict):
        for key, nested in sorted(value.items()):
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(nested, name))
    elif isinstance(value, (str, int, float, bool)):
        flattened[prefix] = value
    elif value is None:
        flattened[prefix] = "null"
    else:
        flattened[prefix] = json.dumps(value, sort_keys=True)
    return flattened


def resolve_tracking(config: dict[str, Any], output: Path) -> tuple[str, Path]:
    configured = str(config["logging"]["mlflow"].get("tracking_uri", "auto"))
    environment_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if configured != "auto":
        tracking_uri = configured
        artifact_root = (output / "mlflow-artifacts").resolve()
        return tracking_uri, artifact_root
    if environment_uri:
        return environment_uri, (output / "mlflow-artifacts").resolve()
    if output.parent.name == "runs":
        mlflow_root = output.parent.parent / "mlflow"
    else:
        mlflow_root = output / "mlflow"
    database = (mlflow_root / "mlflow.db").resolve()
    return f"sqlite:///{database}", (mlflow_root / "artifacts").resolve()


class ProgressLogger:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout
        self.last_step_at: float | None = None
        self.last_logged_step: int | None = None

    def _write(self, message: str) -> None:
        print(message, file=self.stream, flush=True)

    def cleanup(self, paths: list[Path]) -> None:
        for path in paths:
            self._write(f"[checkpoint] removed stale temporary file: {path}")

    def start(
        self,
        *,
        resumed: bool,
        device: torch.device,
        parameters: int,
        windows: int,
        progress: TrainingProgress,
        epochs: int,
        total_steps: int,
    ) -> None:
        mode = "RESUME" if resumed else "NEW"
        self._write(
            f"[train] {mode} | device={device} | parameters={parameters:,} | "
            f"windows={windows:,} | epoch={progress.next_epoch}/{epochs} | "
            f"step={progress.global_step}/{total_steps}"
        )

    def mlflow(self, *, run_id: str, tracking_uri: str, experiment: str) -> None:
        self._write(f"[mlflow] run={run_id} | experiment={experiment} | tracking={tracking_uri}")

    def warning(self, message: str) -> None:
        self._write(f"[warning] {message}")

    def step(
        self,
        *,
        epoch: int,
        epochs: int,
        batch: int,
        batches: int,
        global_step: int,
        total_steps: int,
        loss: float,
        learning_rate: float,
        elapsed_seconds: float,
    ) -> None:
        now = time.monotonic()
        step_seconds = None
        if self.last_step_at is not None and self.last_logged_step is not None:
            logged_steps = max(1, global_step - self.last_logged_step)
            step_seconds = (now - self.last_step_at) / logged_steps
        self.last_step_at = now
        self.last_logged_step = global_step
        steps_per_second = global_step / elapsed_seconds if elapsed_seconds > 0 else 0.0
        remaining = max(0, total_steps - global_step)
        eta = remaining / steps_per_second if steps_per_second > 0 else None
        pace = "warming up" if step_seconds is None else f"{step_seconds:.1f}s/step"
        self._write(
            f"[train] epoch {epoch}/{epochs} | batch {batch}/{batches} | "
            f"step {global_step}/{total_steps} | loss {loss:.6f} | "
            f"lr {learning_rate:.2e} | {pace} | elapsed {format_duration(elapsed_seconds)} | "
            f"ETA {format_duration(eta)}"
        )

    def checkpoint(self, path: Path, progress: TrainingProgress, *, reason: str) -> None:
        size = _format_size(path.stat().st_size) if path.exists() else "missing"
        self._write(
            f"[checkpoint] saved {path} ({size}) | next={progress.next_epoch}:"
            f"{progress.next_batch} | reason={reason}"
        )

    def best_checkpoint(
        self,
        path: Path,
        *,
        epoch: int | None,
        validation_loss: float | None,
    ) -> None:
        loss = "?" if validation_loss is None else f"{validation_loss:.6f}"
        self._write(f"[checkpoint] best={path} | epoch={epoch} | val_loss={loss}")

    def best_pose_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        mpjpe_m: float,
    ) -> None:
        self._write(f"[checkpoint] best_pose={path} | epoch={epoch} | MPJPE={mpjpe_m * 100:.2f}cm")

    def validation_start(self, *, epoch: int, epochs: int, windows: int) -> None:
        self._write(f"[validation] START | epoch {epoch}/{epochs} | windows={windows:,}")

    def validation(
        self,
        *,
        epoch: int,
        epochs: int,
        metrics: dict[str, float],
        train_loss: float | None,
        elapsed_seconds: float,
        improved: bool,
        best_loss: float | None,
        validations_without_improvement: int | None,
        patience: int | None,
    ) -> None:
        train_loss_summary = "partial" if train_loss is None else f"{train_loss:.6f}"
        parts = [
            f"[validation] DONE | epoch {epoch}/{epochs}",
            f"train_loss={train_loss_summary}",
            f"val_loss={metrics['loss.total']:.6f}",
            f"MPJPE={metrics['mpjpe_m'] * 100:.2f}cm",
            f"root_drift={metrics['root_local_drift_m'] * 100:.2f}cm",
            f"contact_F1={metrics['contact_f1']:.4f}",
            f"time={format_duration(elapsed_seconds)}",
        ]
        if "detector_prior_mpjpe_m" in metrics:
            parts.insert(
                4,
                f"detector_MPJPE={metrics['detector_prior_mpjpe_m'] * 100:.2f}cm",
            )
            parts.insert(
                5,
                f"MPJPE_gain={metrics['mpjpe_gain_vs_detector_m'] * 100:+.2f}cm",
            )
        if "detector_neutral_mpjpe_m" in metrics:
            parts.insert(
                5,
                f"neutral_prior={metrics['detector_neutral_mpjpe_m'] * 100:.2f}cm",
            )
            parts.insert(
                7,
                f"neutral_gain={metrics['mpjpe_gain_vs_detector_neutral_m'] * 100:+.2f}cm",
            )
        if "acceleration_gain_vs_detector_neutral_mps2" in metrics:
            parts.insert(
                -3,
                f"accel_gain={metrics['acceleration_gain_vs_detector_neutral_mps2']:+.2f}m/s^2",
            )
        if best_loss is not None:
            parts.append(f"best={best_loss:.6f}")
        if improved:
            parts.append("IMPROVED")
        if validations_without_improvement is not None and patience is not None:
            parts.append(f"early_stop={validations_without_improvement}/{patience}")
        self._write(" | ".join(parts))

    def finish(self, *, status: str, reason: str | None, progress: TrainingProgress) -> None:
        suffix = f" | reason={reason}" if reason else ""
        self._write(
            f"[train] {status.upper()} | step={progress.global_step} | "
            f"elapsed={format_duration(progress.elapsed_seconds)}{suffix}"
        )


class MLflowTracker:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        output: Path,
        config_path: Path,
        data_bom: dict[str, Any],
        logger: ProgressLogger,
    ) -> None:
        settings = config["logging"]["mlflow"]
        self.enabled = bool(settings.get("enabled", True))
        self.log_checkpoints = bool(settings.get("log_checkpoints", False))
        self.experiment_name = str(settings.get("experiment_name", "gravity-mocap"))
        configured_name = settings.get("run_name")
        self.run_name = str(configured_name) if configured_name else output.name
        self.tracking_uri, self.artifact_root = resolve_tracking(config, output)
        self.config = config
        self.output = output
        self.config_path = config_path
        self.data_bom = data_bom
        self.logger = logger
        self.run_id: str | None = None
        self._mlflow: Any = None

    def start(
        self,
        *,
        existing_run_id: str | None,
        may_reuse_run_file: bool,
        resumed_from: str | None,
        device: torch.device,
    ) -> str | None:
        if not self.enabled:
            return None
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError as error:  # pragma: no cover - package is a required dependency
            raise RuntimeError(
                "MLflow is enabled but not installed; run ./scripts/setup.sh"
            ) from error

        self._mlflow = mlflow
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        experiment = client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            artifact_location = (
                self.artifact_root.as_uri()
                if self.tracking_uri.startswith(("sqlite:", "file:"))
                else None
            )
            experiment_id = client.create_experiment(
                self.experiment_name,
                artifact_location=artifact_location,
            )
        else:
            experiment_id = experiment.experiment_id

        if existing_run_id is None and may_reuse_run_file:
            existing_run_id = self._read_run_id()
        run = None
        if existing_run_id:
            try:
                run = mlflow.start_run(run_id=existing_run_id)
            except Exception as error:
                self.logger.warning(
                    f"cannot resume MLflow run {existing_run_id}: {error}; creating a new run"
                )
        if run is None:
            run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=self.run_name,
                tags={"gravity_mocap.new_run": "true"},
            )
            for key, value in _flatten(self.config, "config").items():
                mlflow.log_param(key, value)

        self.run_id = run.info.run_id
        mlflow.set_tags(
            {
                "framework": "pytorch",
                "torch.version": torch.__version__,
                "device": str(device),
                "gravity_mocap.checkpoint_version": str(CHECKPOINT_VERSION),
                "gravity_mocap.config_path": str(self.config_path.resolve()),
                "gravity_mocap.data_shards": str(self.data_bom.get("shard_count", "unknown")),
                "gravity_mocap.resumed": str(resumed_from is not None).lower(),
                "gravity_mocap.resumed_from": resumed_from or "none",
            }
        )
        mlflow.log_artifact(str(self.output / "resolved-config.json"), artifact_path="inputs")
        mlflow.log_artifact(str(self.output / "data-bom.json"), artifact_path="inputs")
        self._write_run_file()
        self.logger.mlflow(
            run_id=self.run_id,
            tracking_uri=self.tracking_uri,
            experiment=self.experiment_name,
        )
        return self.run_id

    def log_step(
        self,
        *,
        metrics: dict[str, float],
        global_step: int,
        epoch: int,
        learning_rate: float,
        elapsed_seconds: float,
    ) -> None:
        if self.run_id is None:
            return
        values = {
            **{f"loss.{name}": value for name, value in metrics.items()},
            "progress.epoch": float(epoch),
            "optimizer.learning_rate": learning_rate,
            "time.elapsed_seconds": elapsed_seconds,
            "performance.steps_per_second": (
                global_step / elapsed_seconds if elapsed_seconds > 0 else 0.0
            ),
        }
        self._mlflow.log_metrics(values, step=global_step)

    def log_epoch(self, *, epoch: int, metrics: dict[str, float], global_step: int) -> None:
        if self.run_id is None or not metrics:
            return
        self._mlflow.log_metrics(
            {f"epoch_loss.{name}": value for name, value in metrics.items()},
            step=global_step,
        )
        self._mlflow.log_metric("progress.completed_epoch", float(epoch), step=global_step)

    def log_validation(self, *, epoch: int, metrics: dict[str, float], global_step: int) -> None:
        if self.run_id is None or not metrics:
            return
        self._mlflow.log_metrics(
            {f"validation.{name}": value for name, value in metrics.items()},
            step=global_step,
        )
        self._mlflow.log_metric("validation.epoch", float(epoch), step=global_step)

    def log_checkpoint(self, path: Path, progress: TrainingProgress, *, reason: str) -> None:
        if self.run_id is None:
            return
        manifest = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "recorded_at": datetime.now(UTC).isoformat(),
            "reason": reason,
            "full_checkpoint_logged": self.log_checkpoints,
            "progress": asdict(progress),
        }
        manifest_path = self.output / "checkpoint-manifest.json"
        _write_json(manifest_path, manifest)
        self._mlflow.log_artifact(str(manifest_path), artifact_path="checkpoint")
        state_path = self.output / STATE_FILE
        if state_path.exists():
            self._mlflow.log_artifact(str(state_path), artifact_path="checkpoint")
        if self.log_checkpoints:
            self._mlflow.log_artifact(str(path), artifact_path="checkpoint/full")

    def finish(self, *, status: str, reason: str | None) -> None:
        if self.run_id is None:
            return
        self._mlflow.set_tag("gravity_mocap.stop_reason", reason or "none")
        self._mlflow.end_run(status=status)

    def fail(self, error: BaseException) -> None:
        if self.run_id is None:
            return
        self._mlflow.set_tag("gravity_mocap.error", f"{type(error).__name__}: {error}"[:5000])
        self._mlflow.end_run(status="FAILED")

    def _read_run_id(self) -> str | None:
        path = self.output / MLFLOW_RUN_FILE
        if not path.exists():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            self.logger.warning(f"cannot read {path}: {error}")
            return None
        run_id = state.get("run_id")
        return str(run_id) if run_id else None

    def _write_run_file(self) -> None:
        if self.run_id is None:
            return
        _write_json(
            self.output / MLFLOW_RUN_FILE,
            {
                "run_id": self.run_id,
                "tracking_uri": self.tracking_uri,
                "experiment_name": self.experiment_name,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
