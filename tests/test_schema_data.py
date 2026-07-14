from pathlib import Path

import numpy as np
import pytest

from gravity_mocap.data import MotionWindowDataset, partition_sequence_paths
from gravity_mocap.detector import normalize_detector_inputs
from gravity_mocap.fixture import create_fixture
from gravity_mocap.schema import read_shard, validate_arrays


def test_fixture_round_trip_and_padding(tmp_path: Path) -> None:
    path = create_fixture(tmp_path / "synthetic/walk.npz", frames=6, image_feature_dim=32)
    arrays, provenance = read_shard(path)
    assert arrays["joints_3d"].shape == (6, 22, 3)
    assert provenance["license_id"] == "CC0-1.0"
    assert provenance["schema_version"] == 1
    dataset = MotionWindowDataset(tmp_path, sequence_length=8, stride=8)
    sample = dataset[0]
    assert sample["image_features"].shape == (8, 32)
    assert np.allclose(sample["frame_mask"][6:].numpy(), 0)


def test_shard_validation_rejects_degenerate_bbox(tmp_path: Path) -> None:
    path = create_fixture(tmp_path / "synthetic/walk.npz", frames=6, image_feature_dim=32)
    arrays, _ = read_shard(path)
    arrays["bbox"] = arrays["bbox"].copy()
    arrays["bbox"][0, 2] = arrays["bbox"][0, 0]

    with pytest.raises(ValueError, match="positive non-degenerate"):
        validate_arrays(arrays)


def test_sequence_split_is_stable_and_disjoint(tmp_path: Path) -> None:
    paths = [tmp_path / f"sequence-{index}.npz" for index in range(10)]
    first_train, first_validation = partition_sequence_paths(
        paths, tmp_path, validation_fraction=0.2, split_seed=17
    )
    second_train, second_validation = partition_sequence_paths(
        list(reversed(paths)), tmp_path, validation_fraction=0.2, split_seed=17
    )

    assert first_train == second_train
    assert first_validation == second_validation
    assert len(first_train) == 8
    assert len(first_validation) == 2
    assert set(first_train).isdisjoint(first_validation)


def test_grouped_split_keeps_all_sequences_from_one_subject_together(tmp_path: Path) -> None:
    paths = [
        tmp_path / "cmu" / "subject-1" / "walk.npz",
        tmp_path / "cmu" / "subject-1" / "run.npz",
        tmp_path / "cmu" / "subject-2" / "walk.npz",
        tmp_path / "cmu" / "subject-2" / "run.npz",
    ]
    groups = {path: f"cmu:{path.parent.name}" for path in paths}

    train, validation = partition_sequence_paths(
        paths,
        tmp_path,
        validation_fraction=0.5,
        split_seed=17,
        group_keys=groups,
    )

    assert len(train) == 2
    assert len(validation) == 2
    assert {path.parent.name for path in train}.isdisjoint(path.parent.name for path in validation)


def test_detector_augmentation_is_deterministic_and_preserves_clean_target(
    tmp_path: Path,
) -> None:
    path = create_fixture(tmp_path / "synthetic/walk.npz", frames=16, image_feature_dim=32)
    augmentation = {
        "enabled": True,
        "keypoint_noise_std": 0.05,
        "confidence_min": 0.5,
        "keypoint_dropout_probability": 0.1,
        "frame_dropout_probability": 0.0,
        "occlusion_probability": 1.0,
        "occlusion_min_frames": 2,
        "occlusion_max_frames": 4,
        "occlusion_joint_fraction": 0.3,
        "bbox_jitter_std": 0.02,
        "camera_dropout_probability": 1.0,
    }
    dataset = MotionWindowDataset(
        tmp_path,
        sequence_length=8,
        stride=8,
        paths=[path],
        augmentation=augmentation,
        augmentation_seed=123,
    )
    dataset.set_epoch(3)
    first = dataset[0]
    repeated = dataset[0]
    dataset.set_epoch(4)
    next_epoch = dataset[0]
    clean, _ = read_shard(path)

    assert np.array_equal(first["keypoints_2d"], repeated["keypoints_2d"])
    assert not np.array_equal(first["keypoints_2d"], next_epoch["keypoints_2d"])
    assert np.array_equal(first["keypoints_2d_target"], clean["keypoints_2d"][:8])
    assert not np.array_equal(first["keypoints_2d"], first["keypoints_2d_target"])
    assert np.array_equal(first["bbox_target"], clean["bbox"][:8])
    assert not np.array_equal(first["bbox"], first["bbox_target"])
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    assert np.allclose(first["camera_delta_6d"], identity)


def test_detector_pixel_contract_normalizes_bbox_and_22_keypoints() -> None:
    bbox = np.asarray([100, 50, 300, 450], dtype=np.float32)
    keypoints = np.tile(np.asarray([200, 250, 0.8], dtype=np.float32), (22, 1))
    keypoints[0, 2] = 0

    normalized_keypoints, normalized_bbox = normalize_detector_inputs(
        keypoints,
        bbox,
        frame_width=640,
        frame_height=480,
    )

    assert normalized_keypoints.shape == (22, 3)
    assert np.allclose(normalized_keypoints[1:, :2], 0)
    assert np.allclose(normalized_keypoints[0], 0)
    assert np.allclose(normalized_bbox, [-0.6875, -0.7916667, -0.0625, 0.875])


def test_detector_pixel_contract_clips_points_outside_frame_bbox() -> None:
    keypoints = np.tile(np.asarray([300, -20, 0.8], dtype=np.float32), (22, 1))
    normalized_keypoints, _ = normalize_detector_inputs(
        keypoints,
        np.asarray([100, 50, 200, 250], dtype=np.float32),
        frame_width=640,
        frame_height=480,
    )

    assert np.all(normalized_keypoints[:, 0] == 1)
    assert np.all(normalized_keypoints[:, 1] == -1)
