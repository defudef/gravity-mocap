from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .skeleton import SKELETON

SCHEMA_VERSION = 1
REQUIRED_ARRAYS = {
    "bbox": (4,),
    "keypoints_2d": (SKELETON.joint_count, 3),
    "camera_delta_6d": (6,),
    "image_features": None,
    "local_rotations_6d": (SKELETON.joint_count, 6),
    "camera_orientation_6d": (6,),
    "gravity_view_orientation_6d": (6,),
    "root_velocity_local": (3,),
    "weak_camera": (3,),
    "contacts": (len(SKELETON.contact_joints),),
    "joints_3d": (SKELETON.joint_count, 3),
    "frame_mask": (),
    "image_mask": (),
}


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_arrays(arrays: dict[str, np.ndarray]) -> int:
    missing = sorted(set(REQUIRED_ARRAYS) - arrays.keys())
    if missing:
        raise ValueError(f"Missing shard arrays: {', '.join(missing)}")
    frame_count = int(arrays["frame_mask"].shape[0])
    if frame_count < 2:
        raise ValueError("A shard needs at least two frames")
    for name, tail_shape in REQUIRED_ARRAYS.items():
        value = np.asarray(arrays[name])
        if value.shape[0] != frame_count:
            raise ValueError(f"{name} has {value.shape[0]} frames, expected {frame_count}")
        if tail_shape is not None and value.shape[1:] != tail_shape:
            raise ValueError(f"{name} has shape {value.shape}; expected (T, {tail_shape})")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinity")
    return frame_count


def write_shard(path: Path, arrays: dict[str, np.ndarray], provenance: dict[str, Any]) -> None:
    validate_arrays(arrays)
    metadata = dict(provenance)
    metadata["schema_version"] = SCHEMA_VERSION
    metadata["joint_names"] = list(SKELETON.names)
    metadata.pop("provenance_hash", None)
    metadata["provenance_hash"] = stable_hash(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()}
    payload["provenance_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def read_shard(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in REQUIRED_ARRAYS}
        provenance = json.loads(str(archive["provenance_json"]))
    validate_arrays(arrays)
    if provenance.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version in {path}")
    return arrays, provenance
