import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gravity_mocap.baseline import infer_detector_baseline
from gravity_mocap.skeleton import SKELETON
from gravity_mocap.world3d import (
    DETECTOR_WORLD_3D_VERSION,
    DetectorWorld3D,
    load_detector_world_3d,
    write_detector_world_3d,
)


def _rest_joints() -> np.ndarray:
    joints = np.zeros((SKELETON.joint_count, 3), dtype=np.float32)
    for joint, parent in enumerate(SKELETON.parents):
        if parent >= 0:
            joints[joint] = joints[int(parent)] + SKELETON.rest_offsets[joint]
    return joints


def _artifact(frames: int = 7) -> DetectorWorld3D:
    joints = np.repeat(_rest_joints()[None], frames, axis=0)
    joints[:, SKELETON.names.index("left_wrist"), 2] += np.linspace(0, 0.2, frames)
    return DetectorWorld3D(
        joints_3d=joints,
        confidence=np.ones((frames, SKELETON.joint_count), dtype=np.float32),
        frame_mask=np.ones(frames, dtype=np.float32),
        source_frame_indices=np.arange(frames, dtype=np.int64),
        fps=30.0,
        source_fps=30.0,
        provenance={
            "detector_world_3d_version": DETECTOR_WORLD_3D_VERSION,
            "artifact_type": "gravity-mocap-detector-world-3d",
            "request_hash": "world-test",
            "source_sha256": "source-test",
            "max_missing_frames": 2,
        },
    )


def test_detector_world_3d_round_trip(tmp_path: Path) -> None:
    path = write_detector_world_3d(tmp_path / "detector-world-3d.npz", _artifact())

    loaded = load_detector_world_3d(path)

    assert loaded.frames == 7
    assert loaded.joints_3d.shape == (7, 22, 3)
    assert loaded.provenance["request_hash"] == "world-test"


def test_detector_world_3d_rejects_nonzero_missing_frame(tmp_path: Path) -> None:
    artifact = _artifact()
    frame_mask = artifact.frame_mask.copy()
    frame_mask[2] = 0

    with pytest.raises(ValueError, match="Missing detector world-3D frames"):
        write_detector_world_3d(
            tmp_path / "detector-world-3d.npz",
            replace(artifact, frame_mask=frame_mask),
        )


def test_detector_world_3d_rejects_parametric_body_fields(tmp_path: Path) -> None:
    path = write_detector_world_3d(tmp_path / "detector-world-3d.npz", _artifact())
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["smpl_pose"] = np.zeros((7, 72), dtype=np.float32)
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="prohibited SMPL/SMPL-X/body-model fields"):
        load_detector_world_3d(path)


def test_detector_baseline_writes_finite_neutral_motion(tmp_path: Path) -> None:
    world = write_detector_world_3d(tmp_path / "detector-world-3d.npz", _artifact())
    output = tmp_path / "output"

    result = infer_detector_baseline(world, output, smoothing_window=3, preview=False)

    assert result["status"] == "created"
    with np.load(result["motion"], allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_3d"])
        contacts = np.asarray(archive["contacts"])
        provenance = json.loads(str(archive["provenance_json"]))
    assert joints.shape == (7, 22, 3)
    assert contacts.shape == (7, len(SKELETON.contact_joints))
    assert np.isfinite(joints).all()
    assert np.allclose(joints[:, 0], 0)
    assert provenance["root_motion"] == "stationary-baseline"

    cached = infer_detector_baseline(world, output, smoothing_window=3, preview=False)
    assert cached["status"] == "cached"


def test_detector_baseline_rejects_long_missing_run(tmp_path: Path) -> None:
    artifact = _artifact()
    joints = artifact.joints_3d.copy()
    confidence = artifact.confidence.copy()
    frame_mask = artifact.frame_mask.copy()
    joints[2:5] = 0
    confidence[2:5] = 0
    frame_mask[2:5] = 0
    world = write_detector_world_3d(
        tmp_path / "detector-world-3d.npz",
        replace(
            artifact,
            joints_3d=joints,
            confidence=confidence,
            frame_mask=frame_mask,
        ),
    )

    with pytest.raises(ValueError, match="missed 3 consecutive"):
        infer_detector_baseline(world, tmp_path / "output", preview=False)
