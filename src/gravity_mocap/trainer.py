from __future__ import annotations

import json
import math
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .catalog import DatasetCatalog
from .checkpoint import (
    LATEST_CHECKPOINT,
    TrainingProgress,
    archive_latest_checkpoint,
    cleanup_stale_checkpoint_temps,
    compatibility_hash,
    data_bom_hash,
    load_training_checkpoint,
    promote_best_checkpoint,
    read_training_state,
    resolve_resume_path,
    save_training_checkpoint,
)
from .data import (
    MotionWindowDataset,
    audit_training_data,
    discover_shards,
    partition_sequence_paths,
)
from .losses import compute_losses
from .metrics import compute_motion_metrics
from .model import GravityViewMotionModel
from .tracking import MLflowTracker, ProgressLogger, resolve_tracking


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_root = Path(config["data"]["root"])
    if not data_root.is_absolute():
        config["data"]["root"] = str((path.parent / data_root).resolve())
    catalog_path = Path(config["data"]["catalog"])
    if not catalog_path.is_absolute():
        config["data"]["catalog"] = str((path.parent / catalog_path).resolve())
    expected_batch = int(config["train"]["batch_size"]) * int(
        config["train"]["gradient_accumulation_steps"]
    )
    if int(config["train"]["effective_batch_size"]) != expected_batch:
        raise ValueError("effective_batch_size must equal batch_size * gradient_accumulation_steps")
    if float(config["train"]["checkpoint_every_minutes"]) <= 0:
        raise ValueError("checkpoint_every_minutes must be positive")
    if int(config["train"]["keep_last_checkpoints"]) < 1:
        raise ValueError("keep_last_checkpoints must be at least 1")
    data_config = config["data"]
    data_config.setdefault("target_fps", 30.0)
    data_config.setdefault("validation_fraction", 0.0)
    data_config.setdefault("split_seed", int(config["seed"]))
    if float(data_config["target_fps"]) <= 0:
        raise ValueError("data.target_fps must be positive")
    validation_fraction = float(data_config["validation_fraction"])
    if not 0 <= validation_fraction < 1:
        raise ValueError("data.validation_fraction must be in [0, 1)")
    augmentation = data_config.setdefault("augmentation", {"enabled": False})
    augmentation.setdefault("enabled", False)
    probability_names = (
        "keypoint_dropout_probability",
        "frame_dropout_probability",
        "occlusion_probability",
        "camera_dropout_probability",
    )
    for name in probability_names:
        augmentation.setdefault(name, 0.0)
        if not 0 <= float(augmentation[name]) <= 1:
            raise ValueError(f"data.augmentation.{name} must be in [0, 1]")
    augmentation.setdefault("keypoint_noise_std", 0.0)
    augmentation.setdefault("confidence_min", 1.0)
    augmentation.setdefault("occlusion_min_frames", 1)
    augmentation.setdefault("occlusion_max_frames", 1)
    augmentation.setdefault("occlusion_joint_fraction", 0.25)
    augmentation.setdefault("bbox_jitter_std", 0.0)
    if float(augmentation["keypoint_noise_std"]) < 0:
        raise ValueError("data.augmentation.keypoint_noise_std cannot be negative")
    if not 0 <= float(augmentation["confidence_min"]) <= 1:
        raise ValueError("data.augmentation.confidence_min must be in [0, 1]")
    if int(augmentation["occlusion_min_frames"]) < 1 or int(
        augmentation["occlusion_max_frames"]
    ) < int(augmentation["occlusion_min_frames"]):
        raise ValueError("data.augmentation occlusion frame range is invalid")
    if not 0 < float(augmentation["occlusion_joint_fraction"]) <= 1:
        raise ValueError("data.augmentation.occlusion_joint_fraction must be in (0, 1]")
    if float(augmentation["bbox_jitter_std"]) < 0:
        raise ValueError("data.augmentation.bbox_jitter_std cannot be negative")
    validation_config = config.setdefault("validation", {})
    validation_config.setdefault("enabled", validation_fraction > 0)
    validation_config.setdefault("every_epochs", 1)
    validation_config.setdefault("batch_size", int(config["train"]["batch_size"]))
    if int(validation_config["every_epochs"]) < 1:
        raise ValueError("validation.every_epochs must be at least 1")
    if int(validation_config["batch_size"]) < 1:
        raise ValueError("validation.batch_size must be at least 1")
    if bool(validation_config["enabled"]) and validation_fraction <= 0:
        raise ValueError("enabled validation requires data.validation_fraction > 0")
    early_stopping = validation_config.setdefault("early_stopping", {})
    early_stopping.setdefault("enabled", bool(validation_config["enabled"]))
    early_stopping.setdefault("monitor", "loss.total")
    early_stopping.setdefault("patience", 20)
    early_stopping.setdefault("min_delta", 0.001)
    if bool(early_stopping["enabled"]) and not bool(validation_config["enabled"]):
        raise ValueError("early stopping requires validation.enabled: true")
    if str(early_stopping["monitor"]) != "loss.total":
        raise ValueError("validation.early_stopping.monitor must be loss.total")
    if int(early_stopping["patience"]) < 1:
        raise ValueError("validation.early_stopping.patience must be at least 1")
    if float(early_stopping["min_delta"]) < 0:
        raise ValueError("validation.early_stopping.min_delta cannot be negative")
    logging_config = config.setdefault("logging", {})
    logging_config.setdefault("log_every_steps", 1)
    mlflow_config = logging_config.setdefault("mlflow", {})
    mlflow_config.setdefault("enabled", True)
    mlflow_config.setdefault("tracking_uri", "auto")
    mlflow_config.setdefault("experiment_name", "gravity-mocap")
    mlflow_config.setdefault("run_name", None)
    mlflow_config.setdefault("log_checkpoints", False)
    if int(logging_config["log_every_steps"]) < 1:
        raise ValueError("logging.log_every_steps must be at least 1")
    if not str(mlflow_config["experiment_name"]).strip():
        raise ValueError("logging.mlflow.experiment_name cannot be empty")
    return config


def build_model(config: dict[str, Any]) -> GravityViewMotionModel:
    return GravityViewMotionModel(**config["model"])


def _audit_data(config: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    root = Path(config["data"]["root"])
    inventory = discover_shards(root)
    errors, bill_of_materials = audit_training_data(
        root,
        DatasetCatalog(Path(config["data"]["catalog"])),
        allow_synthetic=bool(config["data"].get("allow_synthetic", False)),
        image_feature_dim=int(config["model"]["image_feature_dim"]),
        expected_fps=float(config["data"]["target_fps"]),
    )
    return inventory, errors, bill_of_materials


def _split_paths(
    config: dict[str, Any], bill_of_materials: dict[str, Any]
) -> tuple[list[Path], list[Path], dict[Path, str]]:
    root = Path(config["data"]["root"])
    paths: list[Path] = []
    group_keys: dict[Path, str] = {}
    for shard in bill_of_materials["shards"]:
        path = root / shard["path"]
        source_id = str(shard.get("source_id"))
        sequence = Path(str(shard.get("source_sequence") or shard["path"]))
        if source_id == "cmu_mocap" and "subjects" in sequence.parts:
            subject_index = sequence.parts.index("subjects") + 1
            unit = "/".join(sequence.parts[: subject_index + 1])
        elif source_id == "addbiomechanics":
            unit = str(sequence.parent)
        else:
            unit = str(sequence)
        paths.append(path)
        group_keys[path] = f"{source_id}:{unit}"
    train_paths, validation_paths = partition_sequence_paths(
        paths,
        root,
        validation_fraction=float(config["data"]["validation_fraction"]),
        split_seed=int(config["data"]["split_seed"]),
        group_keys=group_keys,
    )
    return train_paths, validation_paths, group_keys


def _window_count(frame_count: int, sequence_length: int, stride: int) -> int:
    starts = list(range(0, max(frame_count - sequence_length + 1, 1), stride))
    final = max(frame_count - sequence_length, 0)
    return len(starts) + (not starts or starts[-1] != final)


def _split_window_counts(
    config: dict[str, Any],
    bill_of_materials: dict[str, Any],
    train_paths: list[Path],
    validation_paths: list[Path],
) -> tuple[int, int]:
    root = Path(config["data"]["root"])
    frames = {str(shard["path"]): int(shard["frames"]) for shard in bill_of_materials["shards"]}
    sequence_length = int(config["data"]["sequence_length"])
    stride = int(config["data"]["stride"])

    def count(paths: list[Path]) -> int:
        return sum(
            _window_count(frames[str(path.relative_to(root))], sequence_length, stride)
            for path in paths
        )

    return count(train_paths), count(validation_paths)


def training_plan(
    config_path: Path,
    output: Path,
    *,
    resume: str = "auto",
    max_hours: float | None = None,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    _validate_session_limits(max_hours=max_hours, max_epochs=max_epochs)
    config = load_config(config_path)
    root = Path(config["data"]["root"])
    inventory, license_errors, bill_of_materials = _audit_data(config)
    train_paths, validation_paths, group_keys = _split_paths(config, bill_of_materials)
    train_windows, validation_windows = _split_window_counts(
        config, bill_of_materials, train_paths, validation_paths
    )
    split_errors: list[str] = []
    if bool(config["validation"]["enabled"]) and not validation_paths:
        split_errors.append("validation is enabled but the dataset cannot produce a holdout split")
    model = build_model(config)
    resume_path = resolve_resume_path(output, resume)
    state = read_training_state(output)
    resume_errors: list[str] = []
    action = "start_new"
    if resume_path is not None:
        action = "resume"
        if state is not None:
            if state.get("compatibility_hash") != compatibility_hash(config):
                resume_errors.append("saved checkpoint is incompatible with the current config")
            if state.get("data_bom_hash") != data_bom_hash(bill_of_materials):
                resume_errors.append("saved checkpoint data BOM differs from current shards")
            early_stopped = (
                bool(state.get("complete"))
                and state.get("stop_reason") == "early_stopping"
                and bool(config["validation"]["early_stopping"]["enabled"])
            )
            if early_stopped or int(state.get("next_epoch", 1)) > int(config["train"]["epochs"]):
                action = "already_complete"
    ready = (
        not license_errors
        and not split_errors
        and not resume_errors
        and inventory["shards"] > 0
        and bool(train_paths)
    )
    tracking_uri, _ = resolve_tracking(config, output)
    return {
        "mode": "DRY RUN - no optimizer or training loop executed",
        "ready": ready,
        "action_on_execute": action,
        "resume_checkpoint": str(resume_path) if resume_path else None,
        "saved_progress": state,
        "config": str(config_path.resolve()),
        "output": str(output.resolve()),
        "data_root": str(root),
        "inventory": inventory,
        "data_split": {
            "status": "ok" if not split_errors else "failed",
            "errors": split_errors,
            "strategy": "stable source-stratified subject/sequence-disjoint hash partition",
            "target_fps": config["data"]["target_fps"],
            "train_sequences": len(train_paths),
            "validation_sequences": len(validation_paths),
            "train_units": len({group_keys[path] for path in train_paths}),
            "validation_units": len({group_keys[path] for path in validation_paths}),
            "train_windows": train_windows,
            "validation_windows": validation_windows,
            "validation_fraction": config["data"]["validation_fraction"],
            "split_seed": config["data"]["split_seed"],
        },
        "license_audit": {
            "status": "ok" if not license_errors else "failed",
            "errors": license_errors,
            "shards_in_bom": bill_of_materials["shard_count"],
        },
        "resume_audit": {
            "status": "ok" if not resume_errors else "failed",
            "errors": resume_errors,
        },
        "model_parameters": model.parameter_count,
        "sequence_length": config["data"]["sequence_length"],
        "micro_batch_size": config["train"]["batch_size"],
        "gradient_accumulation_steps": config["train"]["gradient_accumulation_steps"],
        "effective_batch_size": config["train"]["effective_batch_size"],
        "epochs": config["train"]["epochs"],
        "checkpoint_every_minutes": config["train"]["checkpoint_every_minutes"],
        "max_hours_this_session": max_hours,
        "max_epochs_this_session": max_epochs,
        "detector_augmentation": config["data"]["augmentation"],
        "validation": config["validation"],
        "progress_logging": {
            "every_optimizer_steps": config["logging"]["log_every_steps"],
            "mlflow_enabled": config["logging"]["mlflow"]["enabled"],
            "mlflow_tracking_uri": tracking_uri,
            "note": "dry-run does not create an MLflow database or run",
        },
    }


def _validate_session_limits(*, max_hours: float | None, max_epochs: int | None) -> None:
    if max_hours is not None and max_hours <= 0:
        raise ValueError("max_hours must be positive")
    if max_epochs is not None and max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if max_hours is not None and max_epochs is not None:
        raise ValueError("max_hours and max_epochs are mutually exclusive")


def _epoch_session_limit_reached(
    *, epoch: int, session_start_epoch: int, max_epochs: int | None
) -> bool:
    return max_epochs is not None and epoch - session_start_epoch + 1 >= max_epochs


def _update_early_stopping(
    progress: TrainingProgress,
    *,
    epoch: int,
    validation_loss: float,
    min_delta: float,
) -> bool:
    if not math.isfinite(validation_loss):
        raise RuntimeError("Early-stopping validation loss is not finite")
    improved = (
        progress.best_validation_loss is None
        or validation_loss < progress.best_validation_loss - min_delta
    )
    if improved:
        progress.best_validation_loss = validation_loss
        progress.best_validation_epoch = epoch
        progress.validations_without_improvement = 0
    else:
        progress.validations_without_improvement += 1
    return improved


def _restore_legacy_validation_progress(
    progress: TrainingProgress,
    output: Path,
    *,
    monitor: str,
) -> bool:
    """Seed early stopping from the last pre-feature validation checkpoint."""
    if progress.best_validation_loss is not None:
        return False
    path = output / "validation-state.json"
    if not path.is_file():
        return False
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        epoch = int(state["epoch"])
        value = float(state["metrics"][monitor])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False
    if epoch != progress.next_epoch - 1 or not math.isfinite(value):
        return False
    progress.best_validation_loss = value
    progress.best_validation_epoch = epoch
    progress.validations_without_improvement = 0
    return True


def _device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _epoch_batches(dataset_size: int, batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    order = torch.randperm(dataset_size, generator=generator).tolist()
    return [order[start : start + batch_size] for start in range(0, dataset_size, batch_size)]


class _StopController:
    def __init__(self) -> None:
        self.reason: str | None = None
        self.previous: dict[signal.Signals, Any] = {}

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        reason = signal.Signals(signum).name
        if self.reason is not None:
            raise KeyboardInterrupt(f"Second stop signal received ({reason})")
        self.reason = reason

    def __enter__(self) -> _StopController:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        return self

    def __exit__(self, *_args: object) -> None:
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def evaluate_model(
    model: torch.nn.Module,
    dataset: MotionWindowDataset,
    config: dict[str, Any],
    device: torch.device,
    *,
    use_amp: bool,
) -> dict[str, float]:
    """Run a forward-only held-out evaluation without mutating training state."""
    loader = DataLoader(
        dataset,
        batch_size=int(config["validation"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
    )
    model.eval()
    metric_sums: dict[str, float] = {}
    windows = 0
    with torch.no_grad():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            batch_windows = int(batch["frame_mask"].shape[0])
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                prediction = model(batch)
                losses = compute_losses(prediction, batch, config["loss"])
                motion_metrics = compute_motion_metrics(
                    prediction,
                    batch,
                    fps=float(config["data"]["target_fps"]),
                )
            values = {
                **{f"loss.{name}": value for name, value in losses.items()},
                **motion_metrics,
            }
            for name, value in values.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value.cpu()) * batch_windows
            windows += batch_windows
    if windows == 0:
        raise RuntimeError("Validation dataset contains no windows")
    return {name: value / windows for name, value in metric_sums.items()}


def run_training(
    config_path: Path,
    output: Path,
    *,
    resume: str = "auto",
    max_hours: float | None = None,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    _validate_session_limits(max_hours=max_hours, max_epochs=max_epochs)
    config = load_config(config_path)
    root = Path(config["data"]["root"])
    _, license_errors, bill_of_materials = _audit_data(config)
    if license_errors:
        raise RuntimeError("Training data license audit failed: " + "; ".join(license_errors))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_paths, validation_paths, _ = _split_paths(config, bill_of_materials)
    dataset = MotionWindowDataset(
        root,
        int(config["data"]["sequence_length"]),
        int(config["data"]["stride"]),
        paths=train_paths,
        augmentation=config["data"]["augmentation"],
        augmentation_seed=seed,
    )
    if not dataset:
        raise RuntimeError("No valid preprocessed shards found; training did not start")
    validation_dataset = MotionWindowDataset(
        root,
        int(config["data"]["sequence_length"]),
        int(config["data"]["stride"]),
        paths=validation_paths,
    )
    if bool(config["validation"]["enabled"]) and not validation_dataset:
        raise RuntimeError("Validation is enabled but the holdout split contains no windows")

    device = _device(config["train"]["device"])
    model = build_model(config).to(device)
    training_model: torch.nn.Module = model
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        training_model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    use_amp = bool(config["train"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger()
    logger.cleanup(cleanup_stale_checkpoint_temps(output))
    _write_json(output / "resolved-config.json", config)
    _write_json(output / "data-bom.json", bill_of_materials)
    resume_path = resolve_resume_path(output, resume)
    progress = TrainingProgress()
    resumed_from: str | None = None
    if resume_path is not None:
        progress = load_training_checkpoint(
            resume_path,
            model,
            optimizer,
            scaler,
            config,
            bill_of_materials,
            device,
        )
        resumed_from = str(resume_path)

    epochs = int(config["train"]["epochs"])
    early_stopping = config["validation"]["early_stopping"]
    if (
        progress.complete
        and progress.stop_reason == "early_stopping"
        and bool(early_stopping["enabled"])
    ):
        return {
            **asdict(progress),
            "status": "already_complete",
            "resumed_from": resumed_from,
        }
    if progress.next_epoch > epochs:
        if not progress.complete:
            progress.complete = True
            progress.stop_reason = "completed"
            save_training_checkpoint(
                output,
                model,
                optimizer,
                scaler,
                config,
                bill_of_materials,
                progress,
            )
        return {
            **asdict(progress),
            "status": "already_complete",
            "resumed_from": resumed_from,
        }
    restored_legacy_best = False
    if resume_path is not None and bool(early_stopping["enabled"]):
        restored_legacy_best = _restore_legacy_validation_progress(
            progress,
            output,
            monitor=str(early_stopping["monitor"]),
        )
    progress.complete = False
    progress.stop_reason = None

    accumulation = int(config["train"]["gradient_accumulation_steps"])
    batch_size = int(config["train"]["batch_size"])
    batches_per_epoch = len(_epoch_batches(len(dataset), batch_size, seed, 1))
    steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    total_steps = epochs * steps_per_epoch
    log_every_steps = int(config["logging"]["log_every_steps"])
    checkpoint_seconds = float(config["train"]["checkpoint_every_minutes"]) * 60
    session_started = time.monotonic()
    session_start_step = progress.global_step
    session_start_epoch = progress.next_epoch
    elapsed_before_session = progress.elapsed_seconds
    deadline = session_started + max_hours * 3600 if max_hours is not None else None
    last_checkpoint_at = session_started
    tracker = MLflowTracker(
        config=config,
        output=output,
        config_path=config_path,
        data_bom=bill_of_materials,
        logger=logger,
    )
    logger.start(
        resumed=resumed_from is not None,
        device=device,
        parameters=model.parameter_count,
        windows=len(dataset),
        progress=progress,
        epochs=epochs,
        total_steps=total_steps,
    )
    progress.mlflow_run_id = tracker.start(
        existing_run_id=progress.mlflow_run_id,
        may_reuse_run_file=resumed_from is not None,
        resumed_from=resumed_from,
        device=device,
    )
    if (
        restored_legacy_best
        and resume_path is not None
        and resume_path.resolve() == (output / LATEST_CHECKPOINT).resolve()
    ):
        best_path = promote_best_checkpoint(output)
        logger.best_checkpoint(
            best_path,
            epoch=progress.best_validation_epoch,
            validation_loss=progress.best_validation_loss,
        )

    def save_progress(
        *,
        next_epoch: int,
        next_batch: int,
        last_loss: float | None,
        complete: bool,
        stop_reason: str | None,
        reason: str,
    ) -> TrainingProgress:
        nonlocal last_checkpoint_at
        saved = TrainingProgress(
            next_epoch=next_epoch,
            next_batch=next_batch,
            global_step=progress.global_step,
            elapsed_seconds=elapsed_before_session + (time.monotonic() - session_started),
            last_loss=last_loss,
            complete=complete,
            stop_reason=stop_reason,
            mlflow_run_id=progress.mlflow_run_id,
            best_validation_loss=progress.best_validation_loss,
            best_validation_epoch=progress.best_validation_epoch,
            validations_without_improvement=progress.validations_without_improvement,
        )
        checkpoint_path = save_training_checkpoint(
            output,
            model,
            optimizer,
            scaler,
            config,
            bill_of_materials,
            saved,
        )
        last_checkpoint_at = time.monotonic()
        logger.checkpoint(checkpoint_path, saved, reason=reason)
        tracker.log_checkpoint(checkpoint_path, saved, reason=reason)
        return saved

    last_loss = progress.last_loss
    next_epoch = progress.next_epoch
    next_batch = progress.next_batch
    try:
        with _StopController() as stop:
            for epoch in range(progress.next_epoch, epochs + 1):
                epoch_metric_sums: dict[str, float] = {}
                epoch_metric_steps = 0
                validation_improved = False
                dataset.set_epoch(epoch)
                all_batches = _epoch_batches(len(dataset), batch_size, seed, epoch)
                start_batch = progress.next_batch if epoch == progress.next_epoch else 0
                if start_batch > len(all_batches):
                    raise RuntimeError(
                        f"Checkpoint batch {start_batch} exceeds epoch size {len(all_batches)}"
                    )
                if start_batch not in {len(all_batches)} and start_batch % accumulation:
                    raise RuntimeError("Checkpoint is not aligned to an optimizer-step boundary")
                remaining_batches = all_batches[start_batch:]
                loader = DataLoader(
                    dataset,
                    batch_sampler=remaining_batches,
                    num_workers=int(config["data"]["num_workers"]),
                )
                training_model.train()
                optimizer.zero_grad(set_to_none=True)
                step_metrics: dict[str, float] = {}
                for local_batch, batch in enumerate(loader):
                    absolute_batch = start_batch + local_batch
                    group_start = (absolute_batch // accumulation) * accumulation
                    group_end = min(group_start + accumulation, len(all_batches))
                    group_size = group_end - group_start
                    if absolute_batch == group_start:
                        step_metrics = {}
                    batch = {name: value.to(device) for name, value in batch.items()}
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=use_amp,
                    ):
                        losses = compute_losses(training_model(batch), batch, config["loss"])
                        scaled_loss = losses["total"] / group_size
                    for name, value in losses.items():
                        detached = float(value.detach().cpu()) / group_size
                        step_metrics[name] = step_metrics.get(name, 0.0) + detached
                    scaler.scale(scaled_loss).backward()
                    if absolute_batch + 1 != group_end:
                        continue
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config["train"]["grad_clip"])
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    progress.global_step += 1
                    last_loss = step_metrics["total"]
                    next_epoch, next_batch = epoch, absolute_batch + 1
                    if next_batch >= len(all_batches):
                        next_epoch, next_batch = epoch + 1, 0

                    now = time.monotonic()
                    elapsed_seconds = elapsed_before_session + (now - session_started)
                    for name, value in step_metrics.items():
                        epoch_metric_sums[name] = epoch_metric_sums.get(name, 0.0) + value
                    epoch_metric_steps += 1
                    should_log = (
                        progress.global_step == session_start_step + 1
                        or progress.global_step % log_every_steps == 0
                        or next_epoch == epoch + 1
                    )
                    if should_log:
                        learning_rate = float(optimizer.param_groups[0]["lr"])
                        logger.step(
                            epoch=epoch,
                            epochs=epochs,
                            batch=group_end,
                            batches=len(all_batches),
                            global_step=progress.global_step,
                            total_steps=total_steps,
                            loss=last_loss,
                            learning_rate=learning_rate,
                            elapsed_seconds=elapsed_seconds,
                        )
                        tracker.log_step(
                            metrics=step_metrics,
                            global_step=progress.global_step,
                            epoch=epoch,
                            learning_rate=learning_rate,
                            elapsed_seconds=elapsed_seconds,
                        )

                    if deadline is not None and now >= deadline:
                        stop.reason = "max_hours"
                    periodic = now - last_checkpoint_at >= checkpoint_seconds
                    if periodic or stop.reason is not None:
                        progress = save_progress(
                            next_epoch=next_epoch,
                            next_batch=next_batch,
                            last_loss=last_loss,
                            complete=False,
                            stop_reason=stop.reason,
                            reason=stop.reason or "periodic",
                        )
                    if stop.reason is not None:
                        tracker.finish(status="KILLED", reason=stop.reason)
                        logger.finish(
                            status="stopped safely", reason=stop.reason, progress=progress
                        )
                        return {
                            **asdict(progress),
                            "status": "stopped_safely",
                            "resumed_from": resumed_from,
                        }

                epoch_metrics: dict[str, float] | None = None
                if epoch_metric_steps and start_batch == 0:
                    epoch_metrics = {
                        name: value / epoch_metric_steps
                        for name, value in epoch_metric_sums.items()
                    }
                    tracker.log_epoch(
                        epoch=epoch,
                        metrics=epoch_metrics,
                        global_step=progress.global_step,
                    )
                if (
                    bool(config["validation"]["enabled"])
                    and epoch % int(config["validation"]["every_epochs"]) == 0
                ):
                    logger.validation_start(
                        epoch=epoch,
                        epochs=epochs,
                        windows=len(validation_dataset),
                    )
                    validation_started = time.monotonic()
                    validation_metrics = evaluate_model(
                        training_model,
                        validation_dataset,
                        config,
                        device,
                        use_amp=use_amp,
                    )
                    if bool(early_stopping["enabled"]):
                        monitor = str(early_stopping["monitor"])
                        validation_improved = _update_early_stopping(
                            progress,
                            epoch=epoch,
                            validation_loss=float(validation_metrics[monitor]),
                            min_delta=float(early_stopping["min_delta"]),
                        )
                        if (
                            progress.validations_without_improvement
                            >= int(early_stopping["patience"])
                            and epoch < epochs
                            and stop.reason is None
                        ):
                            stop.reason = "early_stopping"
                    validation_elapsed = time.monotonic() - validation_started
                    logger.validation(
                        epoch=epoch,
                        epochs=epochs,
                        metrics=validation_metrics,
                        train_loss=(epoch_metrics or {}).get("total"),
                        elapsed_seconds=validation_elapsed,
                        improved=validation_improved,
                        best_loss=progress.best_validation_loss,
                        validations_without_improvement=(
                            progress.validations_without_improvement
                            if bool(early_stopping["enabled"])
                            else None
                        ),
                        patience=(
                            int(early_stopping["patience"])
                            if bool(early_stopping["enabled"])
                            else None
                        ),
                    )
                    tracking_metrics = dict(validation_metrics)
                    if bool(early_stopping["enabled"]):
                        tracking_metrics.update(
                            {
                                "early_stopping.best_loss": float(progress.best_validation_loss),
                                "early_stopping.without_improvement": float(
                                    progress.validations_without_improvement
                                ),
                            }
                        )
                    tracker.log_validation(
                        epoch=epoch,
                        metrics=tracking_metrics,
                        global_step=progress.global_step,
                    )
                    _write_json(
                        output / "validation-state.json",
                        {
                            "epoch": epoch,
                            "global_step": progress.global_step,
                            "sequences": validation_dataset.sequence_count,
                            "windows": len(validation_dataset),
                            "metrics": validation_metrics,
                            "elapsed_seconds": validation_elapsed,
                            "early_stopping": {
                                "enabled": bool(early_stopping["enabled"]),
                                "monitor": early_stopping["monitor"],
                                "best_loss": progress.best_validation_loss,
                                "best_epoch": progress.best_validation_epoch,
                                "without_improvement": (progress.validations_without_improvement),
                                "patience": early_stopping["patience"],
                                "min_delta": early_stopping["min_delta"],
                            },
                        },
                    )
                if epoch < epochs and stop.reason is None:
                    if _epoch_session_limit_reached(
                        epoch=epoch,
                        session_start_epoch=session_start_epoch,
                        max_epochs=max_epochs,
                    ):
                        stop.reason = "max_epochs"
                    elif deadline is not None and time.monotonic() >= deadline:
                        stop.reason = "max_hours"
                early_stopped = stop.reason == "early_stopping"
                progress = save_progress(
                    next_epoch=epoch + 1,
                    next_batch=0,
                    last_loss=last_loss,
                    complete=epoch == epochs or early_stopped,
                    stop_reason=("completed" if epoch == epochs else stop.reason),
                    reason=("completed" if epoch == epochs else stop.reason or "epoch"),
                )
                if validation_improved:
                    best_path = promote_best_checkpoint(output)
                    logger.best_checkpoint(
                        best_path,
                        epoch=progress.best_validation_epoch,
                        validation_loss=progress.best_validation_loss,
                    )
                if epoch % int(config["train"]["checkpoint_every"]) == 0:
                    archive_latest_checkpoint(
                        output,
                        epoch,
                        keep=int(config["train"]["keep_last_checkpoints"]),
                    )
                next_epoch, next_batch = epoch + 1, 0
                if early_stopped:
                    tracker.finish(status="FINISHED", reason="early_stopping")
                    logger.finish(
                        status="early stopped",
                        reason="early_stopping",
                        progress=progress,
                    )
                    return {
                        **asdict(progress),
                        "status": "early_stopped",
                        "resumed_from": resumed_from,
                    }
                if stop.reason is not None and epoch < epochs:
                    tracker.finish(status="KILLED", reason=stop.reason)
                    logger.finish(status="stopped safely", reason=stop.reason, progress=progress)
                    return {
                        **asdict(progress),
                        "status": "stopped_safely",
                        "resumed_from": resumed_from,
                    }
    except BaseException as error:
        tracker.fail(error)
        logger.finish(status="failed", reason=str(error), progress=progress)
        raise

    tracker.finish(status="FINISHED", reason="completed")
    logger.finish(status="complete", reason="completed", progress=progress)
    return {
        **asdict(progress),
        "status": "complete",
        "resumed_from": resumed_from,
        "next_epoch": next_epoch,
        "next_batch": next_batch,
    }
