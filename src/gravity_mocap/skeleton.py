from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

JOINT_NAMES = (
    "root",
    "left_hip",
    "right_hip",
    "spine_1",
    "left_knee",
    "right_knee",
    "spine_2",
    "left_ankle",
    "right_ankle",
    "spine_3",
    "left_toe",
    "right_toe",
    "neck",
    "left_clavicle",
    "right_clavicle",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)

# Canonical rest offsets in metres. They establish topology and scale only; each
# converted sequence records a scale estimate so raw datasets are not forced to
# share one actor's proportions.
REST_OFFSETS = np.asarray(
    [
        [0.00, 0.00, 0.00],
        [-0.09, -0.08, 0.00],
        [0.09, -0.08, 0.00],
        [0.00, 0.11, 0.00],
        [0.00, -0.42, 0.00],
        [0.00, -0.42, 0.00],
        [0.00, 0.12, 0.00],
        [0.00, -0.43, 0.00],
        [0.00, -0.43, 0.00],
        [0.00, 0.13, 0.00],
        [0.00, -0.04, 0.14],
        [0.00, -0.04, 0.14],
        [0.00, 0.16, 0.00],
        [-0.08, 0.04, 0.00],
        [0.08, 0.04, 0.00],
        [0.00, 0.18, 0.01],
        [-0.13, 0.00, 0.00],
        [0.13, 0.00, 0.00],
        [-0.27, 0.00, 0.00],
        [0.27, 0.00, 0.00],
        [-0.25, 0.00, 0.00],
        [0.25, 0.00, 0.00],
    ],
    dtype=np.float32,
)

CONTACT_JOINTS = np.asarray([20, 21, 10, 11, 7, 8], dtype=np.int64)
CONTACT_NAMES = ("left_hand", "right_hand", "left_toe", "right_toe", "left_heel", "right_heel")


@dataclass(frozen=True)
class CanonicalSkeleton:
    names: tuple[str, ...] = JOINT_NAMES
    parents: np.ndarray = field(default_factory=lambda: PARENTS.copy())
    rest_offsets: np.ndarray = field(default_factory=lambda: REST_OFFSETS.copy())
    contact_joints: np.ndarray = field(default_factory=lambda: CONTACT_JOINTS.copy())

    @property
    def joint_count(self) -> int:
        return len(self.names)


SKELETON = CanonicalSkeleton()
