from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import FORBIDDEN_PARAMETRIC_BODY_KEYS
from .artifacts import write_npz_atomic
from .skeleton import SKELETON

RIG_2D_VERSION = 1
RIG_2D_FILENAME = "rig-2d.npz"
RIG_2D_MANIFEST_FILENAME = "rig-2d-manifest.json"
RIG_2D_PREVIEW_FILENAME = "preview-rig-2d.mp4"

_ARRAY_NAMES = (
    "keypoints_2d",
    "bbox",
    "camera_delta_6d",
    "image_mask",
    "frame_mask",
    "pixel_keypoints",
    "pixel_bbox",
    "source_frame_indices",
)
_SCALAR_NAMES = ("fps", "source_fps", "frame_width", "frame_height")


@dataclass(frozen=True)
class Rig2D:
    """Validated neutral 22-joint video frontend artifact."""

    keypoints_2d: np.ndarray
    bbox: np.ndarray
    camera_delta_6d: np.ndarray
    image_mask: np.ndarray
    frame_mask: np.ndarray
    pixel_keypoints: np.ndarray
    pixel_bbox: np.ndarray
    source_frame_indices: np.ndarray
    fps: float
    source_fps: float
    frame_width: int
    frame_height: int
    provenance: dict[str, Any]

    @property
    def frames(self) -> int:
        return int(self.frame_mask.shape[0])

    def model_inputs(self, image_feature_dim: int) -> dict[str, np.ndarray]:
        if self.frames < 2:
            raise ValueError("3D inference requires at least two 2D rig frames")
        if image_feature_dim < 1:
            raise ValueError("image_feature_dim must be positive")
        return {
            "keypoints_2d": self.keypoints_2d,
            "bbox": self.bbox,
            "camera_delta_6d": self.camera_delta_6d,
            "image_features": np.zeros((self.frames, image_feature_dim), dtype=np.float32),
            "image_mask": np.zeros(self.frames, dtype=np.float32),
            "frame_mask": self.frame_mask,
            "pixel_keypoints": self.pixel_keypoints,
            "pixel_bbox": self.pixel_bbox,
            "source_frame_indices": self.source_frame_indices,
        }


def _validate_rig(rig: Rig2D) -> Rig2D:
    frames = rig.frames
    if frames < 1:
        raise ValueError("A 2D rig artifact needs at least one frame")
    expected_shapes = {
        "keypoints_2d": (frames, SKELETON.joint_count, 3),
        "bbox": (frames, 4),
        "camera_delta_6d": (frames, 6),
        "image_mask": (frames,),
        "frame_mask": (frames,),
        "pixel_keypoints": (frames, SKELETON.joint_count, 3),
        "pixel_bbox": (frames, 4),
        "source_frame_indices": (frames,),
    }
    for name, shape in expected_shapes.items():
        value = np.asarray(getattr(rig, name))
        if value.shape != shape:
            raise ValueError(f"{name} has shape {value.shape}; expected {shape}")
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"{name} must be numeric")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinity")

    if not np.isfinite(rig.fps) or rig.fps <= 0:
        raise ValueError("fps must be a positive finite number")
    if not np.isfinite(rig.source_fps) or rig.source_fps <= 0:
        raise ValueError("source_fps must be a positive finite number")
    if rig.frame_width < 1 or rig.frame_height < 1:
        raise ValueError("frame dimensions must be positive")
    if not np.issubdtype(rig.source_frame_indices.dtype, np.integer):
        raise ValueError("source_frame_indices must contain integers")
    if np.any(rig.source_frame_indices < 0) or np.any(np.diff(rig.source_frame_indices) < 0):
        raise ValueError("source_frame_indices must be non-negative and monotonic")
    if np.any((rig.frame_mask < 0) | (rig.frame_mask > 1)):
        raise ValueError("frame_mask must be in [0, 1]")
    if np.any(rig.keypoints_2d[..., 2] < 0) or np.any(rig.keypoints_2d[..., 2] > 1):
        raise ValueError("2D rig confidence must be in [0, 1]")
    visible = rig.keypoints_2d[..., 2] > 0
    if np.any(np.abs(rig.keypoints_2d[..., :2][visible]) > 1.00001):
        raise ValueError("Visible 2D rig coordinates must be within [-1, 1]")
    if np.any(rig.keypoints_2d[..., :2][~visible] != 0):
        raise ValueError("Missing 2D rig joints must have zero coordinates")
    if np.any(np.abs(rig.bbox) > 1.00001):
        raise ValueError("2D rig bboxes must be within [-1, 1]")
    if np.any(rig.bbox[:, 2:] < rig.bbox[:, :2]):
        raise ValueError("2D rig bboxes must be ordered [x1, y1, x2, y2]")
    if np.any(rig.image_mask != 0):
        raise ValueError("2D rig artifacts have no visual features; image_mask must be zero")

    version = rig.provenance.get("rig_2d_version", rig.provenance.get("input_version"))
    if version != RIG_2D_VERSION:
        raise ValueError(f"Unsupported 2D rig version {version!r}; expected {RIG_2D_VERSION}")
    artifact_type = rig.provenance.get("artifact_type")
    if artifact_type not in (None, "gravity-mocap-rig-2d"):
        raise ValueError(f"Unsupported 2D rig artifact type: {artifact_type!r}")
    return rig


def write_rig_2d(path: Path, rig: Rig2D) -> Path:
    """Atomically write a validated, model-compatible 2D rig artifact."""
    rig = _validate_rig(rig)
    write_npz_atomic(
        path,
        keypoints_2d=rig.keypoints_2d,
        bbox=rig.bbox,
        camera_delta_6d=rig.camera_delta_6d,
        image_mask=rig.image_mask,
        frame_mask=rig.frame_mask,
        pixel_keypoints=rig.pixel_keypoints,
        pixel_bbox=rig.pixel_bbox,
        source_frame_indices=rig.source_frame_indices,
        fps=np.asarray(rig.fps, dtype=np.float32),
        source_fps=np.asarray(rig.source_fps, dtype=np.float32),
        frame_width=np.asarray(rig.frame_width, dtype=np.int32),
        frame_height=np.asarray(rig.frame_height, dtype=np.int32),
        joint_names=np.asarray(SKELETON.names),
        provenance_json=np.asarray(json.dumps(rig.provenance, sort_keys=True)),
    )
    return path


def load_rig_2d(path: Path) -> Rig2D:
    """Load and validate a neutral 2D rig without importing the video backend."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"2D rig artifact does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        forbidden = sorted(
            key
            for key in archive.files
            if any(token in key.lower() for token in FORBIDDEN_PARAMETRIC_BODY_KEYS)
        )
        if forbidden:
            raise ValueError(
                f"{path} contains prohibited SMPL/SMPL-X/body-model fields: {forbidden}"
            )
        required = (*_ARRAY_NAMES, *_SCALAR_NAMES, "joint_names", "provenance_json")
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"2D rig artifact is missing: {', '.join(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in _ARRAY_NAMES}
        scalars = {name: np.asarray(archive[name]) for name in _SCALAR_NAMES}
        joint_names = tuple(str(name) for name in np.asarray(archive["joint_names"]).tolist())
        raw_provenance = np.asarray(archive["provenance_json"])

    if joint_names != SKELETON.names:
        raise ValueError("2D rig joint_names do not match the canonical 22-joint order")
    if raw_provenance.shape != ():
        raise ValueError("provenance_json must be a scalar JSON string")
    try:
        provenance = json.loads(str(raw_provenance.item()))
    except json.JSONDecodeError as error:
        raise ValueError("provenance_json is not valid JSON") from error
    if not isinstance(provenance, dict):
        raise ValueError("provenance_json must contain a JSON object")
    for name, value in scalars.items():
        if value.shape != ():
            raise ValueError(f"{name} must be a scalar")

    return _validate_rig(
        Rig2D(
            **arrays,
            fps=float(scalars["fps"]),
            source_fps=float(scalars["source_fps"]),
            frame_width=int(scalars["frame_width"]),
            frame_height=int(scalars["frame_height"]),
            provenance=provenance,
        )
    )
