from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .schema import stable_hash

CHECKPOINT_VERSION = 2
LATEST_CHECKPOINT = "latest.pt"
BEST_CHECKPOINT = "best.pt"
STATE_FILE = "training-state.json"


@dataclass
class TrainingProgress:
    next_epoch: int = 1
    next_batch: int = 0
    global_step: int = 0
    elapsed_seconds: float = 0.0
    last_loss: float | None = None
    complete: bool = False
    stop_reason: str | None = None
    mlflow_run_id: str | None = None
    best_validation_loss: float | None = None
    best_validation_epoch: int | None = None
    validations_without_improvement: int = 0


def cleanup_stale_checkpoint_temps(output: Path) -> list[Path]:
    """Remove files that were never atomically promoted after an interrupted save."""
    candidates = [
        output / f"{LATEST_CHECKPOINT}.tmp",
        output / f"{BEST_CHECKPOINT}.tmp",
        output / f"{STATE_FILE}.tmp",
        *output.glob("epoch-*.pt.tmp"),
    ]
    removed = []
    for path in candidates:
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def compatibility_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Return only settings that must stay fixed across resumed sessions."""
    train = config["train"]
    data = config["data"]
    return {
        "seed": config["seed"],
        "data": {
            "target_fps": data["target_fps"],
            "sequence_length": data["sequence_length"],
            "stride": data["stride"],
            "allow_synthetic": data.get("allow_synthetic", False),
            "validation_fraction": data.get("validation_fraction", 0.0),
            "split_seed": data.get("split_seed", 0),
            "augmentation": data.get("augmentation", {}),
        },
        "model": config["model"],
        "optimizer": {
            "batch_size": train["batch_size"],
            "gradient_accumulation_steps": train["gradient_accumulation_steps"],
            "effective_batch_size": train["effective_batch_size"],
            "mixed_precision": train.get("mixed_precision", True),
            "learning_rate": train["learning_rate"],
            "weight_decay": train["weight_decay"],
            "grad_clip": train["grad_clip"],
        },
        "loss": config["loss"],
    }


def compatibility_hash(config: dict[str, Any]) -> str:
    return stable_hash(compatibility_payload(config))


def data_bom_hash(data_bom: dict[str, Any]) -> str:
    return stable_hash(data_bom)


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    result: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        result["torch_mps"] = torch.mps.get_rng_state()
    return result


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["generator"],
            numpy_state["keys"].cpu().numpy(),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "torch_mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["torch_mps"].cpu())


def resolve_resume_path(output: Path, resume: str) -> Path | None:
    latest = output / LATEST_CHECKPOINT
    if resume == "auto":
        return latest if latest.exists() else None
    if resume == "never":
        if latest.exists():
            raise RuntimeError(
                f"{latest} already exists. Use --resume auto, an explicit checkpoint, "
                "or a different --output directory."
            )
        return None
    explicit = Path(resume).expanduser().resolve()
    if not explicit.is_file():
        raise RuntimeError(f"Resume checkpoint does not exist: {explicit}")
    return explicit


def read_training_state(output: Path) -> dict[str, Any] | None:
    path = output / STATE_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise RuntimeError(f"Cannot read training state {path}: {error}") from error


def _write_training_state(
    output: Path,
    progress: TrainingProgress,
    *,
    checkpoint_path: Path,
    config_hash: str,
    bom_hash: str,
) -> None:
    state = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "latest_checkpoint": str(checkpoint_path.resolve()),
        "updated_at": datetime.now(UTC).isoformat(),
        "compatibility_hash": config_hash,
        "data_bom_hash": bom_hash,
        **asdict(progress),
    }
    path = output / STATE_FILE
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(temporary, path)


def save_training_checkpoint(
    output: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    data_bom: dict[str, Any],
    progress: TrainingProgress,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    config_hash = compatibility_hash(config)
    bom_hash = data_bom_hash(data_bom)
    checkpoint = output / LATEST_CHECKPOINT
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": capture_rng_state(),
            "progress": asdict(progress),
            "compatibility_hash": config_hash,
            "data_bom_hash": bom_hash,
            "config": config,
            "data_bom": data_bom,
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    _write_training_state(
        output,
        progress,
        checkpoint_path=checkpoint,
        config_hash=config_hash,
        bom_hash=bom_hash,
    )
    return checkpoint


def load_training_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    data_bom: dict[str, Any],
    device: torch.device,
) -> TrainingProgress:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    version = checkpoint.get("checkpoint_version")
    if version != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"Unsupported checkpoint version {version!r}; expected {CHECKPOINT_VERSION}"
        )
    expected_config = compatibility_hash(config)
    if checkpoint.get("compatibility_hash") != expected_config:
        raise RuntimeError(
            "Checkpoint is incompatible with the current model/data/optimizer configuration"
        )
    expected_bom = data_bom_hash(data_bom)
    if checkpoint.get("data_bom_hash") != expected_bom:
        raise RuntimeError("Checkpoint data BOM differs from the current training dataset")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng"])
    return TrainingProgress(**checkpoint["progress"])


def archive_latest_checkpoint(output: Path, completed_epoch: int, keep: int) -> Path:
    latest = output / LATEST_CHECKPOINT
    archive = output / f"epoch-{completed_epoch:04d}.pt"
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    shutil.copy2(latest, temporary)
    os.replace(temporary, archive)
    archives = sorted(output.glob("epoch-*.pt"))
    for stale in archives[: max(0, len(archives) - keep)]:
        stale.unlink()
    return archive


def promote_best_checkpoint(output: Path) -> Path:
    """Atomically copy the latest full-state checkpoint to the best slot."""
    latest = output / LATEST_CHECKPOINT
    if not latest.is_file():
        raise RuntimeError(f"Cannot promote missing checkpoint: {latest}")
    best = output / BEST_CHECKPOINT
    temporary = best.with_suffix(best.suffix + ".tmp")
    shutil.copy2(latest, temporary)
    os.replace(temporary, best)
    return best
