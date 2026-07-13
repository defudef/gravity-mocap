from __future__ import annotations

import json
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
    TrainingProgress,
    archive_latest_checkpoint,
    compatibility_hash,
    data_bom_hash,
    load_training_checkpoint,
    read_training_state,
    resolve_resume_path,
    save_training_checkpoint,
)
from .data import MotionWindowDataset, audit_training_data, discover_shards
from .losses import compute_losses
from .model import GravityViewMotionModel


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
    )
    return inventory, errors, bill_of_materials


def training_plan(
    config_path: Path,
    output: Path,
    *,
    resume: str = "auto",
    max_hours: float | None = None,
) -> dict[str, Any]:
    if max_hours is not None and max_hours <= 0:
        raise ValueError("max_hours must be positive")
    config = load_config(config_path)
    root = Path(config["data"]["root"])
    inventory, license_errors, bill_of_materials = _audit_data(config)
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
            if int(state.get("next_epoch", 1)) > int(config["train"]["epochs"]):
                action = "already_complete"
    ready = not license_errors and not resume_errors and inventory["shards"] > 0
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
    }


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


def run_training(
    config_path: Path,
    output: Path,
    *,
    resume: str = "auto",
    max_hours: float | None = None,
) -> dict[str, Any]:
    if max_hours is not None and max_hours <= 0:
        raise ValueError("max_hours must be positive")
    config = load_config(config_path)
    root = Path(config["data"]["root"])
    _, license_errors, bill_of_materials = _audit_data(config)
    if license_errors:
        raise RuntimeError("Training data license audit failed: " + "; ".join(license_errors))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = MotionWindowDataset(
        root,
        int(config["data"]["sequence_length"]),
        int(config["data"]["stride"]),
    )
    if not dataset:
        raise RuntimeError("No valid preprocessed shards found; training did not start")

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
    progress.complete = False
    progress.stop_reason = None

    accumulation = int(config["train"]["gradient_accumulation_steps"])
    batch_size = int(config["train"]["batch_size"])
    checkpoint_seconds = float(config["train"]["checkpoint_every_minutes"]) * 60
    session_started = time.monotonic()
    elapsed_before_session = progress.elapsed_seconds
    deadline = session_started + max_hours * 3600 if max_hours is not None else None
    last_checkpoint_at = session_started

    def save_progress(
        *,
        next_epoch: int,
        next_batch: int,
        last_loss: float | None,
        complete: bool,
        stop_reason: str | None,
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
        )
        save_training_checkpoint(
            output,
            model,
            optimizer,
            scaler,
            config,
            bill_of_materials,
            saved,
        )
        last_checkpoint_at = time.monotonic()
        return saved

    last_loss = progress.last_loss
    next_epoch = progress.next_epoch
    next_batch = progress.next_batch
    with _StopController() as stop:
        for epoch in range(progress.next_epoch, epochs + 1):
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
            for local_batch, batch in enumerate(loader):
                absolute_batch = start_batch + local_batch
                group_start = (absolute_batch // accumulation) * accumulation
                group_end = min(group_start + accumulation, len(all_batches))
                group_size = group_end - group_start
                batch = {name: value.to(device) for name, value in batch.items()}
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    losses = compute_losses(training_model(batch), batch, config["loss"])
                    scaled_loss = losses["total"] / group_size
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
                last_loss = float(losses["total"].detach().cpu())
                next_epoch, next_batch = epoch, absolute_batch + 1
                if next_batch >= len(all_batches):
                    next_epoch, next_batch = epoch + 1, 0

                now = time.monotonic()
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
                    )
                if stop.reason is not None:
                    return {
                        **asdict(progress),
                        "status": "stopped_safely",
                        "resumed_from": resumed_from,
                    }

            progress = save_progress(
                next_epoch=epoch + 1,
                next_batch=0,
                last_loss=last_loss,
                complete=epoch == epochs,
                stop_reason="completed" if epoch == epochs else None,
            )
            if epoch % int(config["train"]["checkpoint_every"]) == 0:
                archive_latest_checkpoint(
                    output,
                    epoch,
                    keep=int(config["train"]["keep_last_checkpoints"]),
                )
            next_epoch, next_batch = epoch + 1, 0

    return {
        **asdict(progress),
        "status": "complete",
        "resumed_from": resumed_from,
        "next_epoch": next_epoch,
        "next_batch": next_batch,
    }
