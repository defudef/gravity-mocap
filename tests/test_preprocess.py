from pathlib import Path

import numpy as np

from gravity_mocap.adapters import canonicalize_positions, load_generic_npz
from gravity_mocap.preprocess import motion_to_arrays
from gravity_mocap.schema import validate_arrays
from gravity_mocap.skeleton import JOINT_NAMES, PARENTS, REST_OFFSETS


def test_generic_canonical_motion_converts_without_smpl(tmp_path: Path) -> None:
    del tmp_path
    rest = np.zeros_like(REST_OFFSETS)
    for joint, parent in enumerate(PARENTS):
        rest[joint] = REST_OFFSETS[joint] if parent < 0 else rest[parent] + REST_OFFSETS[joint]
    positions = np.repeat(rest[None], 10, axis=0)
    positions[:, :, 0] += np.arange(10, dtype=np.float32)[:, None] * 0.01
    canonical = canonicalize_positions(positions, list(JOINT_NAMES))
    arrays = motion_to_arrays(canonical, fps=30, image_feature_dim=32)
    assert validate_arrays(arrays) == 10
    assert arrays["image_mask"].sum() == 0
    assert not np.allclose(arrays["camera_delta_6d"][1:], arrays["camera_delta_6d"][0])


def test_generic_npz_rejects_parametric_body_fields(tmp_path: Path) -> None:
    path = tmp_path / "contains-smpl.npz"
    np.savez(
        path,
        joints_3d=np.zeros((2, len(JOINT_NAMES), 3), dtype=np.float32),
        joint_names=np.asarray(JOINT_NAMES),
        smpl_body_pose=np.zeros((2, 63), dtype=np.float32),
    )

    try:
        load_generic_npz(path)
    except ValueError as error:
        assert "prohibited SMPL/SMPL-X/body-model fields" in str(error)
    else:
        raise AssertionError("Parametric body fields must fail closed")
