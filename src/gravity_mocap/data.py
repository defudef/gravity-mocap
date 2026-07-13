from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .catalog import DatasetCatalog
from .schema import read_shard


class MotionWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, root: Path, sequence_length: int, stride: int):
        self.root = root
        self.sequence_length = sequence_length
        self.index: list[tuple[Path, int, int]] = []
        for path in sorted(root.rglob("*.npz")):
            try:
                arrays, _ = read_shard(path)
            except (KeyError, ValueError):
                continue
            frame_count = len(arrays["frame_mask"])
            starts = list(range(0, max(frame_count - sequence_length + 1, 1), stride))
            final = max(frame_count - sequence_length, 0)
            if not starts or starts[-1] != final:
                starts.append(final)
            self.index.extend((path, start, frame_count) for start in starts)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, start, frame_count = self.index[index]
        arrays, _ = read_shard(path)
        end = min(start + self.sequence_length, frame_count)
        result: dict[str, torch.Tensor] = {}
        for name, value in arrays.items():
            window = value[start:end]
            if len(window) < self.sequence_length:
                pad_shape = (self.sequence_length - len(window), *window.shape[1:])
                padding = np.zeros(pad_shape, dtype=window.dtype)
                window = np.concatenate((window, padding), axis=0)
            result[name] = torch.from_numpy(np.asarray(window, dtype=np.float32))
        return result


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
            }
        )
    bill_of_materials["shard_count"] = len(bill_of_materials["shards"])
    return errors, bill_of_materials
