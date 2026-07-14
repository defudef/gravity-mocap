from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import FORBIDDEN_PARAMETRIC_BODY_KEYS
from .artifacts import write_npz_atomic
from .skeleton import SKELETON

DETECTOR_WORLD_3D_VERSION = 1
DETECTOR_WORLD_3D_FILENAME = "detector-world-3d.npz"


@dataclass(frozen=True)
class DetectorWorld3D:
    """Versioned, root-relative 3D landmarks emitted by the video detector."""

    joints_3d: np.ndarray
    confidence: np.ndarray
    frame_mask: np.ndarray
    source_frame_indices: np.ndarray
    fps: float
    source_fps: float
    provenance: dict[str, Any]

    @property
    def frames(self) -> int:
        return int(self.frame_mask.shape[0])


def _validate_world_3d(artifact: DetectorWorld3D) -> DetectorWorld3D:
    frames = artifact.frames
    if frames < 1:
        raise ValueError("A detector world-3D artifact needs at least one frame")
    expected_shapes = {
        "joints_3d": (frames, SKELETON.joint_count, 3),
        "confidence": (frames, SKELETON.joint_count),
        "frame_mask": (frames,),
        "source_frame_indices": (frames,),
    }
    for name, shape in expected_shapes.items():
        value = np.asarray(getattr(artifact, name))
        if value.shape != shape:
            raise ValueError(f"{name} has shape {value.shape}; expected {shape}")
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"{name} must be numeric")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinity")

    if np.any((artifact.confidence < 0) | (artifact.confidence > 1)):
        raise ValueError("detector world-3D confidence must be in [0, 1]")
    if np.any((artifact.frame_mask < 0) | (artifact.frame_mask > 1)):
        raise ValueError("detector world-3D frame_mask must be in [0, 1]")
    missing = artifact.frame_mask <= 0
    if np.any(artifact.joints_3d[missing] != 0) or np.any(artifact.confidence[missing] != 0):
        raise ValueError("Missing detector world-3D frames must contain only zeros")
    if not np.issubdtype(artifact.source_frame_indices.dtype, np.integer):
        raise ValueError("source_frame_indices must contain integers")
    if np.any(artifact.source_frame_indices < 0) or np.any(
        np.diff(artifact.source_frame_indices) < 0
    ):
        raise ValueError("source_frame_indices must be non-negative and monotonic")
    if not np.isfinite(artifact.fps) or artifact.fps <= 0:
        raise ValueError("fps must be a positive finite number")
    if not np.isfinite(artifact.source_fps) or artifact.source_fps <= 0:
        raise ValueError("source_fps must be a positive finite number")

    version = artifact.provenance.get("detector_world_3d_version")
    if version != DETECTOR_WORLD_3D_VERSION:
        raise ValueError(
            f"Unsupported detector world-3D version {version!r}; "
            f"expected {DETECTOR_WORLD_3D_VERSION}"
        )
    artifact_type = artifact.provenance.get("artifact_type")
    if artifact_type not in (None, "gravity-mocap-detector-world-3d"):
        raise ValueError(f"Unsupported detector world-3D artifact type: {artifact_type!r}")
    return artifact


def write_detector_world_3d(path: Path, artifact: DetectorWorld3D) -> Path:
    artifact = _validate_world_3d(artifact)
    write_npz_atomic(
        path,
        joints_3d=np.asarray(artifact.joints_3d, dtype=np.float32),
        confidence=np.asarray(artifact.confidence, dtype=np.float32),
        frame_mask=np.asarray(artifact.frame_mask, dtype=np.float32),
        source_frame_indices=np.asarray(artifact.source_frame_indices, dtype=np.int64),
        fps=np.asarray(artifact.fps, dtype=np.float32),
        source_fps=np.asarray(artifact.source_fps, dtype=np.float32),
        joint_names=np.asarray(SKELETON.names),
        provenance_json=np.asarray(json.dumps(artifact.provenance, sort_keys=True)),
    )
    return path


def load_detector_world_3d(path: Path) -> DetectorWorld3D:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Detector world-3D artifact does not exist: {path}")
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
        required = (
            "joints_3d",
            "confidence",
            "frame_mask",
            "source_frame_indices",
            "fps",
            "source_fps",
            "joint_names",
            "provenance_json",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"Detector world-3D artifact is missing: {', '.join(missing)}")
        joints_3d = np.asarray(archive["joints_3d"])
        confidence = np.asarray(archive["confidence"])
        frame_mask = np.asarray(archive["frame_mask"])
        source_frame_indices = np.asarray(archive["source_frame_indices"])
        fps = np.asarray(archive["fps"])
        source_fps = np.asarray(archive["source_fps"])
        joint_names = tuple(str(name) for name in np.asarray(archive["joint_names"]).tolist())
        raw_provenance = np.asarray(archive["provenance_json"])

    if joint_names != SKELETON.names:
        raise ValueError("Detector world-3D joint_names do not match the canonical order")
    for name, value in (("fps", fps), ("source_fps", source_fps)):
        if value.shape != ():
            raise ValueError(f"{name} must be a scalar")
    if raw_provenance.shape != ():
        raise ValueError("provenance_json must be a scalar JSON string")
    try:
        provenance = json.loads(str(raw_provenance.item()))
    except json.JSONDecodeError as error:
        raise ValueError("provenance_json is not valid JSON") from error
    if not isinstance(provenance, dict):
        raise ValueError("provenance_json must contain a JSON object")

    return _validate_world_3d(
        DetectorWorld3D(
            joints_3d=joints_3d,
            confidence=confidence,
            frame_mask=frame_mask,
            source_frame_indices=source_frame_indices,
            fps=float(fps),
            source_fps=float(source_fps),
            provenance=provenance,
        )
    )


def prepare_detector_world_3d(
    artifact: DetectorWorld3D,
    *,
    smoothing_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill short gaps, smooth landmarks, and retarget them to the neutral skeleton.

    Returns ``neutral_joints, confidence, local_rotations_6d, smoothed_detector_joints``.
    The function is shared by the no-training baseline and the learned v2 prior.
    """
    from .preprocess import canonicalize_motion_skeleton

    max_gap = int(artifact.provenance.get("max_missing_frames", 30))
    values = np.asarray(artifact.joints_3d, dtype=np.float32).copy()
    confidence = np.asarray(artifact.confidence, dtype=np.float32).copy()
    valid = np.asarray(artifact.frame_mask) > 0
    if not valid.any():
        raise ValueError("Detector produced no world-3D frames")

    index = 0
    while index < len(values):
        if valid[index]:
            index += 1
            continue
        start = index
        while index < len(values) and not valid[index]:
            index += 1
        end = index
        gap = end - start
        if gap > max_gap:
            raise ValueError(
                f"World-3D detector missed {gap} consecutive frame(s); maximum allowed is {max_gap}"
            )
        previous = start - 1 if start > 0 else None
        following = end if end < len(values) else None
        if previous is None:
            values[start:end] = values[following]
            confidence[start:end] = confidence[following]
        elif following is None:
            values[start:end] = values[previous]
            confidence[start:end] = confidence[previous]
        else:
            for offset, frame in enumerate(range(start, end), start=1):
                amount = offset / float(gap + 1)
                values[frame] = values[previous] + (values[following] - values[previous]) * amount
                confidence[frame] = (
                    confidence[previous] + (confidence[following] - confidence[previous]) * amount
                )

    window = int(smoothing_window)
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd number")
    if window > 1:
        radius = window // 2
        ramp = np.arange(1, radius + 2, dtype=np.float32)
        kernel = np.concatenate((ramp, ramp[-2::-1]))
        kernel /= kernel.sum()
        padded_values = np.pad(values, ((radius, radius), (0, 0), (0, 0)), mode="edge")
        padded_confidence = np.pad(confidence, ((radius, radius), (0, 0)), mode="edge")
        numerator = np.zeros_like(values, dtype=np.float32)
        denominator = np.zeros((*values.shape[:2], 1), dtype=np.float32)
        for offset, kernel_weight in enumerate(kernel):
            joint_weight = (0.1 + 0.9 * padded_confidence[offset : offset + len(values)])[..., None]
            weighted = float(kernel_weight) * joint_weight
            numerator += padded_values[offset : offset + len(values)] * weighted
            denominator += weighted
        values = numerator / denominator.clip(1e-6)
    values -= values[:, :1]
    values = values.astype(np.float32)
    local_rotations_6d, neutral_joints = canonicalize_motion_skeleton(values)
    return neutral_joints, confidence, local_rotations_6d, values
