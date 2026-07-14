import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gravity_mocap.rig2d import RIG_2D_VERSION, Rig2D, load_rig_2d, write_rig_2d
from gravity_mocap.rotations import identity_rotation_6d
from gravity_mocap.skeleton import JOINT_NAMES, SKELETON


def _rig(frames: int = 3) -> Rig2D:
    return Rig2D(
        keypoints_2d=np.zeros((frames, SKELETON.joint_count, 3), dtype=np.float32),
        bbox=np.zeros((frames, 4), dtype=np.float32),
        camera_delta_6d=identity_rotation_6d(frames).numpy(),
        image_mask=np.zeros(frames, dtype=np.float32),
        frame_mask=np.ones(frames, dtype=np.float32),
        pixel_keypoints=np.zeros((frames, SKELETON.joint_count, 3), dtype=np.float32),
        pixel_bbox=np.tile(np.asarray([10, 20, 100, 200], dtype=np.float32), (frames, 1)),
        source_frame_indices=np.arange(frames, dtype=np.int64),
        fps=30.0,
        source_fps=60.0,
        frame_width=640,
        frame_height=480,
        provenance={
            "rig_2d_version": RIG_2D_VERSION,
            "request_hash": "rig-test",
            "source_sha256": "source-test",
            "model_sha256": "model-test",
        },
    )


def test_rig_2d_round_trip_and_model_adapter(tmp_path: Path) -> None:
    path = write_rig_2d(tmp_path / "rig-2d.npz", _rig())

    loaded = load_rig_2d(path)
    inputs = loaded.model_inputs(image_feature_dim=8)

    assert loaded.frames == 3
    assert inputs["keypoints_2d"].shape == (3, 22, 3)
    assert inputs["image_features"].shape == (3, 8)
    assert np.array_equal(inputs["image_mask"], np.zeros(3, dtype=np.float32))


def test_rig_2d_rejects_wrong_joint_order(tmp_path: Path) -> None:
    path = write_rig_2d(tmp_path / "rig-2d.npz", _rig())
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["joint_names"] = np.asarray(tuple(reversed(JOINT_NAMES)))
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="canonical 22-joint order"):
        load_rig_2d(path)


def test_rig_2d_rejects_parametric_body_fields(tmp_path: Path) -> None:
    path = write_rig_2d(tmp_path / "rig-2d.npz", _rig())
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["smpl_body_pose"] = np.zeros((3, 63), dtype=np.float32)
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="prohibited SMPL/SMPL-X/body-model fields"):
        load_rig_2d(path)


def test_rig_2d_requires_zero_mask_for_absent_visual_features(tmp_path: Path) -> None:
    rig = replace(_rig(), image_mask=np.ones(3, dtype=np.float32))

    with pytest.raises(ValueError, match="image_mask must be zero"):
        write_rig_2d(tmp_path / "rig-2d.npz", rig)


def test_rig_2d_requires_zero_coordinates_for_missing_joints(tmp_path: Path) -> None:
    keypoints = _rig().keypoints_2d.copy()
    keypoints[0, 0, :2] = [0.25, -0.5]
    rig = replace(_rig(), keypoints_2d=keypoints)

    with pytest.raises(ValueError, match="Missing 2D rig joints must have zero coordinates"):
        write_rig_2d(tmp_path / "rig-2d.npz", rig)


def test_rig_2d_provenance_is_plain_json(tmp_path: Path) -> None:
    path = write_rig_2d(tmp_path / "rig-2d.npz", _rig())

    with np.load(path, allow_pickle=False) as archive:
        provenance = json.loads(str(archive["provenance_json"]))

    assert provenance["rig_2d_version"] == RIG_2D_VERSION
