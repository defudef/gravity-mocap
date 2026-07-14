import numpy as np
import pytest

from gravity_mocap.pose import (
    fill_short_bbox_gaps,
    mediapipe_to_canonical,
    mediapipe_world_to_canonical,
    padded_bbox_from_landmarks,
)
from gravity_mocap.skeleton import JOINT_NAMES


def _landmarks() -> np.ndarray:
    points = np.zeros((33, 3), dtype=np.float32)
    points[:, 2] = 0.9
    for index in range(33):
        points[index, :2] = [index * 10.0, index * 5.0]
    return points


def test_mediapipe_mapping_preserves_feet_and_derives_spine() -> None:
    points = _landmarks()
    mapped = mediapipe_to_canonical(points)
    by_name = {name: mapped[index] for index, name in enumerate(JOINT_NAMES)}

    assert mapped.shape == (22, 3)
    assert np.allclose(by_name["left_toe"], points[31])
    assert np.allclose(by_name["right_toe"], points[32])
    assert np.allclose(by_name["root"][:2], (points[23, :2] + points[24, :2]) / 2)
    assert by_name["spine_1"][2] == pytest.approx(0.9)


def test_mediapipe_mapping_zeros_missing_joint() -> None:
    points = _landmarks()
    points[31, 2] = 0.1

    mapped = mediapipe_to_canonical(points, confidence_threshold=0.2)

    assert np.array_equal(mapped[JOINT_NAMES.index("left_toe")], np.zeros(3))


def test_mediapipe_world_mapping_is_root_relative_gravity_up() -> None:
    points = np.zeros((33, 4), dtype=np.float32)
    points[:, 3] = 0.9
    for index in range(33):
        points[index, :3] = [index, index * 2, index * 3]

    joints, confidence = mediapipe_world_to_canonical(points)
    by_name = {name: joints[index] for index, name in enumerate(JOINT_NAMES)}

    assert joints.shape == (22, 3)
    assert confidence.shape == (22,)
    assert np.array_equal(by_name["root"], np.zeros(3, dtype=np.float32))
    expected_left_hip = points[23, :3] - 0.5 * (points[23, :3] + points[24, :3])
    expected_left_hip[1:] *= -1
    assert np.allclose(by_name["left_hip"], expected_left_hip)
    assert np.allclose(confidence, 0.9)


def test_bbox_uses_visible_landmarks_and_padding() -> None:
    points = _landmarks()
    points[:, 0] += 100
    points[:, 1] += 50

    bbox = padded_bbox_from_landmarks(points, frame_width=640, frame_height=480)

    assert bbox is not None
    assert bbox.shape == (4,)
    assert bbox[0] < points[:, 0].min()
    assert bbox[2] > points[:, 0].max()


def test_bbox_gap_fill_interpolates_short_run() -> None:
    bboxes = np.asarray(
        [[0, 0, 10, 10], [np.nan] * 4, [np.nan] * 4, [6, 6, 16, 16]],
        dtype=np.float32,
    )

    filled = fill_short_bbox_gaps(bboxes, max_gap=2)

    assert np.allclose(filled[1], [2, 2, 12, 12])
    assert np.allclose(filled[2], [4, 4, 14, 14])


def test_bbox_gap_fill_rejects_long_run() -> None:
    bboxes = np.asarray([[0, 0, 10, 10], [np.nan] * 4, [np.nan] * 4], dtype=np.float32)

    with pytest.raises(ValueError, match="missed 2 consecutive"):
        fill_short_bbox_gaps(bboxes, max_gap=1)
