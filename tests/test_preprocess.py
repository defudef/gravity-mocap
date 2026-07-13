from pathlib import Path

import numpy as np
import pytest

from gravity_mocap import preprocess
from gravity_mocap.adapters import canonicalize_positions, load_generic_npz
from gravity_mocap.catalog import DatasetEntry
from gravity_mocap.preprocess import motion_to_arrays, resample_positions
from gravity_mocap.schema import read_shard, validate_arrays
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


def test_resampling_normalizes_time_and_root_velocity_to_30_fps() -> None:
    rest = np.zeros_like(REST_OFFSETS)
    for joint, parent in enumerate(PARENTS):
        rest[joint] = REST_OFFSETS[joint] if parent < 0 else rest[parent] + REST_OFFSETS[joint]
    positions = np.repeat(rest[None], 121, axis=0)
    positions[:, :, 0] += np.arange(121, dtype=np.float32)[:, None] / 120.0

    resampled = resample_positions(positions, source_fps=120, target_fps=30)
    arrays = motion_to_arrays(resampled, fps=30, image_feature_dim=32)

    assert resampled.shape == (31, len(JOINT_NAMES), 3)
    assert resampled[-1, 0, 0] == pytest.approx(1.0)
    assert arrays["root_velocity_local"][:-1, 0] == pytest.approx(1.0, abs=1e-5)
    bbox_center = 0.5 * (arrays["bbox"][:, 0] + arrays["bbox"][:, 2])
    assert np.ptp(bbox_center) > 0.01


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


def test_preprocess_reuses_matching_atomic_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    dataset_root = raw_root / "fixture"
    dataset_root.mkdir(parents=True)
    rest = np.zeros_like(REST_OFFSETS)
    for joint, parent in enumerate(PARENTS):
        rest[joint] = REST_OFFSETS[joint] if parent < 0 else rest[parent] + REST_OFFSETS[joint]
    positions = np.repeat(rest[None], 10, axis=0)
    np.savez(
        dataset_root / "walk.npz",
        positions=positions,
        joint_names=np.asarray(JOINT_NAMES),
        fps=np.asarray(30.0),
    )
    entry = DatasetEntry(
        dataset_id="fixture",
        title="Fixture",
        task="motion",
        license_id="CC0-1.0",
        license_url="https://example.com/license",
        approved_for_training=True,
        attribution_required=False,
        requires_acceptance=False,
        downloader={"type": "http"},
    )

    first = preprocess.preprocess_dataset(entry, raw_root, output_root, image_feature_dim=32)
    _, provenance = read_shard(first[0])
    assert provenance["source_fps"] == 30.0
    assert provenance["fps"] == 30.0
    assert provenance["converter_version"] == "cleanroom-v2"
    monkeypatch.setattr(
        preprocess,
        "motion_to_arrays",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("matching shard should be reused")
        ),
    )
    second = preprocess.preprocess_dataset(entry, raw_root, output_root, image_feature_dim=32)

    assert first == second
    assert len(second) == 1
