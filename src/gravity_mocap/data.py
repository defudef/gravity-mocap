from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .catalog import DatasetCatalog
from .schema import read_shard


class MotionWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        root: Path,
        sequence_length: int,
        stride: int,
        *,
        paths: Sequence[Path] | None = None,
        augmentation: dict[str, Any] | None = None,
        augmentation_seed: int = 0,
    ):
        self.root = root
        self.sequence_length = sequence_length
        self.augmentation = dict(augmentation or {})
        self.augmentation_seed = int(augmentation_seed)
        self.epoch = 0
        self.index: list[tuple[Path, int, int]] = []
        selected_paths = sorted(paths) if paths is not None else sorted(root.rglob("*.npz"))
        self.sequence_paths: list[Path] = []
        for path in selected_paths:
            try:
                arrays, _ = read_shard(path)
            except (KeyError, ValueError):
                continue
            self.sequence_paths.append(path)
            frame_count = len(arrays["frame_mask"])
            starts = list(range(0, max(frame_count - sequence_length + 1, 1), stride))
            final = max(frame_count - sequence_length, 0)
            if not starts or starts[-1] != final:
                starts.append(final)
            self.index.extend((path, start, frame_count) for start in starts)

    @property
    def sequence_count(self) -> int:
        return len(self.sequence_paths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, start, frame_count = self.index[index]
        arrays, _ = read_shard(path)
        end = min(start + self.sequence_length, frame_count)
        result: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            window = value[start:end]
            if len(window) < self.sequence_length:
                pad_shape = (self.sequence_length - len(window), *window.shape[1:])
                padding = np.zeros(pad_shape, dtype=window.dtype)
                window = np.concatenate((window, padding), axis=0)
            result[name] = np.asarray(window, dtype=np.float32).copy()
        result["keypoints_2d_target"] = result["keypoints_2d"].copy()
        if self.augmentation.get("enabled", False):
            relative = path.relative_to(self.root)
            digest = hashlib.sha256(
                f"{self.augmentation_seed}:{self.epoch}:{relative}:{start}".encode()
            ).digest()
            generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            _augment_detector_inputs(result, self.augmentation, generator)
        return {name: torch.from_numpy(value) for name, value in result.items()}


def partition_sequence_paths(
    paths: Sequence[Path],
    root: Path,
    *,
    validation_fraction: float,
    split_seed: int,
    group_keys: Mapping[Path, str] | None = None,
) -> tuple[list[Path], list[Path]]:
    """Create a stable source-stratified subject/sequence holdout."""
    selected = sorted(set(paths))
    if validation_fraction <= 0 or len(selected) < 2:
        return selected, []
    groups: dict[str, list[Path]] = {}
    for path in selected:
        relative = path.relative_to(root)
        source = relative.parts[0] if len(relative.parts) > 1 else "__default__"
        groups.setdefault(source, []).append(path)
    validation: set[Path] = set()
    for source, source_paths in sorted(groups.items()):
        units: dict[str, list[Path]] = {}
        for path in source_paths:
            unit = group_keys[path] if group_keys is not None else str(path.relative_to(root))
            units.setdefault(unit, []).append(path)
        if len(units) < 2:
            continue
        validation_count = min(
            len(units) - 1,
            max(1, int(round(len(units) * validation_fraction))),
        )
        ranked_units = sorted(
            units,
            key=lambda unit: hashlib.sha256(f"{split_seed}:{source}:{unit}".encode()).digest(),
        )
        for unit in ranked_units[:validation_count]:
            validation.update(units[unit])
    return (
        [path for path in selected if path not in validation],
        [path for path in selected if path in validation],
    )


def _augment_detector_inputs(
    sample: dict[str, np.ndarray],
    config: dict[str, Any],
    generator: np.random.Generator,
) -> None:
    keypoints = sample["keypoints_2d"]
    frame_valid = sample["frame_mask"] > 0
    visible = (keypoints[..., 2] > 0) & frame_valid[:, None]

    noise_std = float(config.get("keypoint_noise_std", 0.0))
    if noise_std:
        noise = generator.normal(0.0, noise_std, size=keypoints[..., :2].shape).astype(np.float32)
        keypoints[..., :2] += noise * visible[..., None]

    confidence_min = float(config.get("confidence_min", 1.0))
    if confidence_min < 1.0:
        confidence_scale = generator.uniform(
            confidence_min, 1.0, size=keypoints[..., 2].shape
        ).astype(np.float32)
        keypoints[..., 2] *= confidence_scale

    dropout = generator.random(visible.shape) < float(
        config.get("keypoint_dropout_probability", 0.0)
    )
    frame_dropout = generator.random(len(keypoints)) < float(
        config.get("frame_dropout_probability", 0.0)
    )
    dropout |= frame_dropout[:, None]

    if generator.random() < float(config.get("occlusion_probability", 0.0)):
        valid_frames = int(frame_valid.sum())
        minimum = max(1, int(config.get("occlusion_min_frames", 1)))
        maximum = min(valid_frames, max(minimum, int(config.get("occlusion_max_frames", 1))))
        if maximum >= minimum and valid_frames:
            duration = int(generator.integers(minimum, maximum + 1))
            start = int(generator.integers(0, max(1, valid_frames - duration + 1)))
            joint_fraction = float(config.get("occlusion_joint_fraction", 0.25))
            joint_count = max(
                1,
                min(keypoints.shape[1], round(keypoints.shape[1] * joint_fraction)),
            )
            joints = generator.choice(keypoints.shape[1], size=joint_count, replace=False)
            dropout[start : start + duration, joints] = True

    dropout &= frame_valid[:, None]
    keypoints[..., :2][dropout] = 0.0
    keypoints[..., 2][dropout] = 0.0
    keypoints[..., :2] = np.clip(keypoints[..., :2], -2.0, 2.0)

    bbox = sample["bbox"]
    bbox_jitter = float(config.get("bbox_jitter_std", 0.0))
    if bbox_jitter:
        minimum, maximum = bbox[:, :2], bbox[:, 2:]
        size = np.maximum(maximum - minimum, 1e-4)
        center = 0.5 * (minimum + maximum)
        center += generator.normal(0.0, bbox_jitter, size=center.shape).astype(np.float32) * size
        scale = np.exp(generator.normal(0.0, bbox_jitter, size=(len(bbox), 1)).astype(np.float32))
        half_size = 0.5 * size * scale
        bbox[:, :2] = center - half_size
        bbox[:, 2:] = center + half_size

    if generator.random() < float(config.get("camera_dropout_probability", 0.0)):
        identity_6d = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        sample["camera_delta_6d"][frame_valid] = identity_6d

    keypoints[~frame_valid] = 0.0
    bbox[~frame_valid] = 0.0


def discover_shards(root: Path) -> dict[str, Any]:
    sources: dict[str, int] = {}
    licenses: dict[str, int] = {}
    frames = 0
    valid = 0
    invalid: list[str] = []
    for path in sorted(root.rglob("*.npz")) if root.exists() else []:
        try:
            arrays, provenance = read_shard(path)
        except (KeyError, ValueError) as error:
            invalid.append(f"{path}: {error}")
            continue
        valid += 1
        frames += len(arrays["frame_mask"])
        source = provenance["source_id"]
        sources[source] = sources.get(source, 0) + 1
        license_id = provenance["license_id"]
        licenses[license_id] = licenses.get(license_id, 0) + 1
    return {
        "shards": valid,
        "frames": frames,
        "sources": sources,
        "licenses": licenses,
        "invalid": invalid,
    }


def audit_training_data(
    root: Path,
    catalog: DatasetCatalog,
    *,
    allow_synthetic: bool,
    image_feature_dim: int | None = None,
    expected_fps: float | None = None,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    bill_of_materials: dict[str, Any] = {"schema_version": 1, "shards": []}
    for path in sorted(root.rglob("*.npz")) if root.exists() else []:
        try:
            arrays, provenance = read_shard(path)
        except (KeyError, ValueError) as error:
            errors.append(f"{path}: {error}")
            continue
        source_id = provenance.get("source_id")
        license_id = provenance.get("license_id")
        if expected_fps is not None:
            try:
                shard_fps = float(provenance["fps"])
            except (KeyError, TypeError, ValueError):
                shard_fps = float("nan")
            if not np.isclose(shard_fps, expected_fps):
                errors.append(
                    f"{path}: preprocessed fps {provenance.get('fps')!r} "
                    f"does not match configured target_fps {expected_fps}"
                )
        if (
            image_feature_dim is not None
            and arrays["image_features"].shape[-1] != image_feature_dim
        ):
            errors.append(
                f"{path}: feature dimension {arrays['image_features'].shape[-1]} "
                f"does not match model dimension {image_feature_dim}"
            )
        if source_id == "synthetic_fixture" and allow_synthetic:
            if license_id != "CC0-1.0":
                errors.append(f"{path}: synthetic fixture must be CC0-1.0")
        else:
            try:
                entry = catalog.require_approved(str(source_id))
            except ValueError as error:
                errors.append(f"{path}: {error}")
                continue
            if license_id != entry.license_id:
                errors.append(
                    f"{path}: shard license {license_id!r} does not match "
                    f"catalog license {entry.license_id!r}"
                )
            if provenance.get("license_url") != entry.license_url:
                errors.append(f"{path}: shard license URL does not match the approved catalog")
            if bool(provenance.get("attribution_required", False)) != (entry.attribution_required):
                errors.append(f"{path}: shard attribution flag does not match the approved catalog")
            if bool(provenance.get("requires_acceptance", False)) != entry.requires_acceptance:
                errors.append(
                    f"{path}: shard data-use acceptance flag does not match the approved catalog"
                )
        bill_of_materials["shards"].append(
            {
                "path": str(path.relative_to(root)),
                "source_id": source_id,
                "source_title": provenance.get("source_title"),
                "source_sequence": provenance.get("source_sequence"),
                "license_id": license_id,
                "license_url": provenance.get("license_url"),
                "attribution_required": bool(provenance.get("attribution_required", False)),
                "requires_acceptance": bool(provenance.get("requires_acceptance", False)),
                "frames": int(len(arrays["frame_mask"])),
                "provenance_hash": provenance.get("provenance_hash"),
                "source_sha256": provenance.get("source_sha256", []),
                "source_fps": provenance.get("source_fps", provenance.get("fps")),
                "fps": provenance.get("fps"),
                "input_mode": provenance.get("input_mode"),
                "converter_version": provenance.get("converter_version"),
                "preprocess_hash": provenance.get("preprocess_hash"),
            }
        )
    bill_of_materials["shard_count"] = len(bill_of_materials["shards"])
    return errors, bill_of_materials
