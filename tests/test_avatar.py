from __future__ import annotations

import numpy as np
import pytest

from gravity_mocap.avatar import _build_avatar_primitives, render_avatar_panel
from gravity_mocap.skeleton import SKELETON


def _rest_pose() -> np.ndarray:
    joints = np.zeros((SKELETON.joint_count, 3), dtype=np.float32)
    for joint, parent in enumerate(SKELETON.parents):
        if parent >= 0:
            joints[joint] = joints[int(parent)] + SKELETON.rest_offsets[joint]
    joints[:, 2] += np.linspace(-0.1, 0.1, SKELETON.joint_count, dtype=np.float32)
    return joints


def test_avatar_primitives_cover_readable_body_parts_and_sort_depth() -> None:
    pose = _rest_pose()
    projected = np.empty((SKELETON.joint_count, 2), dtype=np.float32)
    projected[:, 0] = 200 + pose[:, 0] * 180
    projected[:, 1] = 220 - pose[:, 1] * 180

    primitives = _build_avatar_primitives(pose, projected, 180.0)
    names = {primitive.name for primitive in primitives}

    assert {
        "torso",
        "head",
        "left_upper_arm",
        "right_upper_arm",
        "left_thigh",
        "right_thigh",
        "left_foot",
        "right_foot",
        "left_hand",
        "right_hand",
    } <= names
    assert [primitive.depth for primitive in primitives] == sorted(
        primitive.depth for primitive in primitives
    )
    camera_primitives = _build_avatar_primitives(
        pose,
        projected,
        180.0,
        nearer_positive=False,
    )
    assert [primitive.depth for primitive in camera_primitives] == sorted(
        (primitive.depth for primitive in camera_primitives),
        reverse=True,
    )


def test_avatar_panel_is_finite_sized_and_visibly_rendered() -> None:
    cv2 = pytest.importorskip("cv2")
    pose = _rest_pose()

    panel = render_avatar_panel(
        cv2,
        pose,
        width=400,
        height=480,
        scale=180.0,
        center=np.asarray([200.0, 245.0], dtype=np.float32),
    )

    assert panel.shape == (480, 400, 3)
    assert panel.dtype == np.uint8
    assert np.isfinite(panel).all()
    assert len(np.unique(panel.reshape(-1, 3), axis=0)) > 100
