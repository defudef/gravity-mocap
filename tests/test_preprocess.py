from pathlib import Path

import numpy as np
import pytest

from gravity_mocap import preprocess
from gravity_mocap.adapters import (
    canonical_indices,
    canonicalize_positions,
    load_bvh,
    load_generic_npz,
)
from gravity_mocap.catalog import DatasetEntry
from gravity_mocap.preprocess import (
    _source_split_group,
    _trim_100style,
    motion_to_arrays,
    resample_positions,
)
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


def test_bvh_loader_respects_channel_order_and_root_translation(tmp_path: Path) -> None:
    path = tmp_path / "walk.bvh"
    path.write_text(
        """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT LeftHip
  {
    OFFSET 1 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 1 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.016666667
0 0 0 0 0 0 0 0 0
1 0 0 90 0 0 0 0 0
"""
    )

    positions, names, fps = load_bvh(path)

    assert names == ["Hips", "LeftHip"]
    assert fps == pytest.approx(60.0)
    assert positions[0, 1] == pytest.approx([1, 0, 0], abs=1e-6)
    assert positions[1, 0] == pytest.approx([1, 0, 0], abs=1e-6)
    assert positions[1, 1] == pytest.approx([1, 1, 0], abs=1e-6)


def test_100style_frame_cuts_remove_calibration_frames(tmp_path: Path) -> None:
    path = tmp_path / "Kick" / "Kick_FW.bvh"
    positions = np.arange(30, dtype=np.float32).reshape(10, 1, 3)

    trimmed = _trim_100style(positions, path, {("Kick", "FW"): (2, 7)})

    assert np.array_equal(trimmed, positions[2:7])


def test_100style_joint_names_map_to_distinct_canonical_joints() -> None:
    source_names = [
        "Hips",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "LeftToeBase",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "RightToeBase",
        "Chest",
        "Chest2",
        "Chest3",
        "Chest4",
        "Neck",
        "Head",
        "LeftCollar",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightCollar",
        "RightArm",
        "RightForeArm",
        "RightHand",
    ]

    indices = canonical_indices(source_names)

    assert len(indices) == len(JOINT_NAMES)
    assert len(set(indices.tolist())) == len(JOINT_NAMES)
    mapping = dict(zip(JOINT_NAMES, indices, strict=True))
    assert source_names[mapping["spine_1"]] == "Chest"
    assert source_names[mapping["spine_2"]] == "Chest2"
    assert source_names[mapping["spine_3"]] == "Chest3"
    assert source_names[mapping["left_clavicle"]] == "LeftCollar"
    assert source_names[mapping["left_shoulder"]] == "LeftArm"


def test_camera_simulation_is_seeded_and_configurable() -> None:
    rest = np.zeros_like(REST_OFFSETS)
    for joint, parent in enumerate(PARENTS):
        rest[joint] = REST_OFFSETS[joint] if parent < 0 else rest[parent] + REST_OFFSETS[joint]
    positions = np.repeat(rest[None], 10, axis=0)
    positions[:, :, 0] += np.arange(10, dtype=np.float32)[:, None] * 0.01
    close_camera = {"distance_meters": [2.0, 2.1]}
    far_camera = {"distance_meters": [8.0, 8.1]}

    first = motion_to_arrays(
        positions,
        fps=30,
        image_feature_dim=32,
        seed=17,
        camera_config=close_camera,
    )
    repeated = motion_to_arrays(
        positions,
        fps=30,
        image_feature_dim=32,
        seed=17,
        camera_config=close_camera,
    )
    far = motion_to_arrays(
        positions,
        fps=30,
        image_feature_dim=32,
        seed=17,
        camera_config=far_camera,
    )

    assert np.array_equal(first["keypoints_2d"], repeated["keypoints_2d"])
    assert np.array_equal(first["bbox"], repeated["bbox"])
    assert not np.array_equal(first["bbox"], far["bbox"])
    assert np.abs(first["keypoints_2d"][..., :2]).max() == pytest.approx(
        1.0 / 1.24,
        abs=1e-5,
    )


def test_split_groups_preserve_cmu_subjects_and_100style_styles() -> None:
    cmu = DatasetEntry(
        "cmu_mocap",
        "CMU",
        "motion",
        "CMU-MOCAP",
        "https://example.com",
        True,
        True,
        False,
        {"type": "http"},
    )
    style = DatasetEntry(
        "100style",
        "100STYLE",
        "motion",
        "CC-BY-4.0",
        "https://example.com",
        True,
        True,
        False,
        {"type": "http"},
    )

    assert (
        _source_split_group(cmu, Path("archive/subjects/135/135_01.amc")) == "archive/subjects/135"
    )
    assert _source_split_group(style, Path("100STYLE/Kick/Kick_FW.bvh")) == ("100STYLE/Kick")


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
    assert provenance["converter_version"] == preprocess.CONVERTER_VERSION
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


def test_preprocess_emits_every_b3d_trial_with_one_split_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    path = raw_root / "addbiomechanics/train/With_Arm/Study_With_Arm/subject/subject.b3d"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fake-b3d")
    rest = np.zeros_like(REST_OFFSETS)
    for joint, parent in enumerate(PARENTS):
        rest[joint] = REST_OFFSETS[joint] if parent < 0 else rest[parent] + REST_OFFSETS[joint]

    class FakeB3DReader:
        def __init__(self, _path: Path):
            self.trial_count = 3
            self.source_href = "https://example.com/study/subject-a/"

        def trial_name(self, trial_index: int) -> str:
            return f"Walk {trial_index}"

        def load_trial(self, trial_index: int) -> tuple[np.ndarray, list[str], float]:
            positions = np.repeat(rest[None], 12 + trial_index, axis=0)
            positions[:, :, 0] += np.arange(len(positions))[:, None] * 0.01
            return positions, list(JOINT_NAMES), 100.0

    monkeypatch.setattr(preprocess, "B3DReader", FakeB3DReader)
    monkeypatch.setattr(
        preprocess,
        "canonicalize_addbiomechanics",
        lambda positions, _names: positions,
    )
    entry = DatasetEntry(
        dataset_id="addbiomechanics",
        title="AddBiomechanics",
        task="motion",
        license_id="CC-BY-4.0",
        license_url="https://example.com/license",
        approved_for_training=True,
        attribution_required=True,
        requires_acceptance=False,
        downloader={
            "type": "remote_zip_members",
            "member_regex": r"^train/With_Arm/.+\.b3d$",
        },
    )

    outputs = preprocess.preprocess_dataset(
        entry,
        raw_root,
        output_root,
        image_feature_dim=32,
    )

    assert len(outputs) == 3
    provenances = [read_shard(output)[1] for output in outputs]
    assert [item["source_trial_index"] for item in provenances] == [0, 1, 2]
    assert [item["source_trial_name"] for item in provenances] == ["Walk 0", "Walk 1", "Walk 2"]
    assert len({item["split_group"] for item in provenances}) == 1
    assert all(item["source_file"].startswith("train/With_Arm/") for item in provenances)
    assert all(item["source_study"] == "Study_With_Arm" for item in provenances)
    assert all(item["source_href"].startswith("https://example.com/") for item in provenances)
