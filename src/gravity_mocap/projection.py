from __future__ import annotations

import torch
from torch import Tensor

from .rotations import rotation_6d_to_matrix


def camera_space_joints(
    local_rotations_6d: Tensor,
    camera_orientation_6d: Tensor,
    joints: Tensor,
) -> Tensor:
    """Rotate root-relative world joints into the predicted source-camera frame."""
    root_orientation = rotation_6d_to_matrix(local_rotations_6d)[..., 0, :, :]
    camera_orientation = rotation_6d_to_matrix(camera_orientation_6d)
    world_to_camera = camera_orientation @ root_orientation.transpose(-1, -2)
    return torch.einsum("...ij,...kj->...ki", world_to_camera, joints)


def weak_project_camera_joints(camera_joints: Tensor, weak_camera: Tensor) -> Tensor:
    """Project camera-space joints to frame-relative coordinates."""
    scale = weak_camera[..., :1].unsqueeze(-2)
    translation = weak_camera[..., 1:].unsqueeze(-2)
    return camera_joints[..., :2] * scale + translation


def normalize_frame_keypoints(frame_xy: Tensor, bbox: Tensor) -> Tensor:
    """Convert frame-relative points to bbox-relative ``[-1, 1]`` coordinates."""
    minimum = bbox[..., :2].unsqueeze(-2)
    size = (bbox[..., 2:] - bbox[..., :2]).clamp_min(1e-4).unsqueeze(-2)
    return (frame_xy - minimum) * (2.0 / size) - 1.0


def denormalize_bbox_keypoints(bbox_xy: Tensor, bbox: Tensor) -> Tensor:
    """Convert bbox-relative points back to frame-relative coordinates."""
    minimum = bbox[..., :2].unsqueeze(-2)
    size = (bbox[..., 2:] - bbox[..., :2]).unsqueeze(-2)
    return minimum + (bbox_xy + 1.0) * 0.5 * size


def project_motion_to_frame(prediction: dict[str, Tensor], joints: Tensor) -> Tensor:
    pose_orientation = prediction.get("local_rotations_6d")
    if pose_orientation is None:
        pose_orientation = prediction["gravity_view_orientation_6d"].unsqueeze(-2)
    camera_joints = camera_space_joints(
        pose_orientation,
        prediction["camera_orientation_6d"],
        joints,
    )
    return weak_project_camera_joints(camera_joints, prediction["weak_camera"])


def project_motion_to_bbox(prediction: dict[str, Tensor], joints: Tensor, bbox: Tensor) -> Tensor:
    return normalize_frame_keypoints(project_motion_to_frame(prediction, joints), bbox)
