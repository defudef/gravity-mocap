from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import write_json_atomic, write_npz_atomic
from .avatar import AVATAR_RENDER_VERSION
from .inference import _render_motion_preview
from .rig2d import load_rig_2d
from .rotations import rotation_6d_to_matrix
from .schema import stable_hash
from .skeleton import SKELETON
from .video import file_sha256
from .world3d import load_detector_world_3d, prepare_detector_world_3d

DETECTOR_BASELINE_VERSION = 1
DETECTOR_BASELINE_MOTION_FILENAME = "detector-baseline-motion.npz"
DETECTOR_BASELINE_MANIFEST_FILENAME = "detector-baseline-manifest.json"
DETECTOR_BASELINE_PREVIEW_FILENAME = "preview-detector-baseline.mp4"


def _estimate_contacts(joints: np.ndarray, fps: float) -> np.ndarray:
    contact_joints = SKELETON.contact_joints
    velocity = np.zeros((len(joints), len(contact_joints)), dtype=np.float32)
    velocity[1:] = np.linalg.norm(
        joints[1:, contact_joints] - joints[:-1, contact_joints], axis=-1
    ) * float(fps)
    floor = float(np.percentile(joints[:, [7, 8, 10, 11], 1], 2))
    low = joints[:, contact_joints, 1] < floor + 0.09
    low[:, :2] = True
    return ((velocity < 0.08) & low).astype(np.float32)


def infer_detector_baseline(
    detector_world_3d_path: Path,
    output_dir: Path,
    *,
    rig_2d_path: Path | None = None,
    source_video: Path | None = None,
    smoothing_window: int = 5,
    force: bool = False,
    preview: bool = False,
) -> dict[str, Any]:
    """Retarget detector world landmarks to the neutral skeleton without training."""
    world_path = detector_world_3d_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    rig_path = rig_2d_path.expanduser().resolve() if rig_2d_path is not None else None
    video = source_video.expanduser().resolve() if source_video is not None else None
    if preview and (rig_path is None or video is None):
        raise ValueError("A matching 2D rig and source video are required for baseline preview")
    if video is not None and not video.is_file():
        raise ValueError(f"Video does not exist: {video}")

    world = load_detector_world_3d(world_path)
    rig = load_rig_2d(rig_path) if rig_path is not None else None
    if rig is not None:
        if rig.frames != world.frames:
            raise ValueError("2D rig and detector world-3D frame counts do not match")
        if not np.array_equal(rig.source_frame_indices, world.source_frame_indices):
            raise ValueError("2D rig and detector world-3D source frame indices do not match")
        if not np.isclose(rig.fps, world.fps):
            raise ValueError("2D rig and detector world-3D FPS do not match")
    if video is not None:
        source_hash = file_sha256(video)
        expected_hash = world.provenance.get("source_sha256")
        if expected_hash and source_hash != expected_hash:
            raise ValueError("Source video SHA-256 does not match detector world-3D provenance")

    request = {
        "detector_baseline_version": DETECTOR_BASELINE_VERSION,
        "avatar_render_version": AVATAR_RENDER_VERSION,
        "detector_world_3d_sha256": file_sha256(world_path),
        "rig_2d_sha256": file_sha256(rig_path) if rig_path is not None else None,
        "smoothing_window": int(smoothing_window),
    }
    request_hash = stable_hash(request)
    motion_path = output_dir / DETECTOR_BASELINE_MOTION_FILENAME
    manifest_path = output_dir / DETECTOR_BASELINE_MANIFEST_FILENAME
    preview_path = output_dir / DETECTOR_BASELINE_PREVIEW_FILENAME
    if not force and motion_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("request_hash") == request_hash and (
                not preview or preview_path.is_file()
            ):
                return {
                    "status": "cached",
                    "frames": world.frames,
                    "motion": str(motion_path),
                    "manifest": str(manifest_path),
                    "preview": str(preview_path) if preview_path.is_file() else None,
                }
        except (json.JSONDecodeError, OSError):
            pass

    neutral_joints, confidence, local_rotations_6d, smoothed = prepare_detector_world_3d(
        world, smoothing_window=int(smoothing_window)
    )
    local_rotation_matrices = rotation_6d_to_matrix(torch.from_numpy(local_rotations_6d)).numpy()
    root_translation = np.zeros((world.frames, 3), dtype=np.float32)
    contacts = _estimate_contacts(neutral_joints, world.fps)
    arrays = {
        "local_rotations_6d": local_rotations_6d,
        "local_rotation_matrices": local_rotation_matrices,
        "gravity_view_orientation_6d": local_rotations_6d[:, 0],
        "root_velocity_local": np.zeros((world.frames, 3), dtype=np.float32),
        "root_translation": root_translation,
        "joints_3d": neutral_joints,
        "joints_world": neutral_joints + root_translation[:, None],
        "detector_joints_3d": smoothed,
        "detector_confidence": confidence,
        "contacts": contacts,
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("Detector baseline produced NaN or infinity")

    provenance = {
        **request,
        "request_hash": request_hash,
        "artifact_type": "gravity-mocap-detector-baseline-motion",
        "detector_world_3d": str(world_path),
        "detector_world_3d_provenance": world.provenance,
        "rig_2d": str(rig_path) if rig_path is not None else None,
        "fps": world.fps,
        "frames": world.frames,
        "joint_names": list(SKELETON.names),
        "root_motion": "stationary-baseline",
    }
    write_npz_atomic(
        motion_path,
        **arrays,
        joint_names=np.asarray(SKELETON.names),
        parents=SKELETON.parents,
        rest_offsets=SKELETON.rest_offsets,
        fps=np.asarray(world.fps, dtype=np.float32),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    if preview:
        assert rig is not None and video is not None
        _render_motion_preview(
            video,
            rig.source_frame_indices.astype(np.int64),
            rig.pixel_keypoints,
            neutral_joints,
            preview_path,
            world.fps,
            avatar_title="A - MediaPipe detector baseline",
        )
    write_json_atomic(
        manifest_path,
        {
            **provenance,
            "motion": str(motion_path),
            "preview": str(preview_path) if preview else None,
        },
    )
    return {
        "status": "created",
        "frames": world.frames,
        "motion": str(motion_path),
        "manifest": str(manifest_path),
        "preview": str(preview_path) if preview else None,
    }
