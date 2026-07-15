from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .projection import denormalize_bbox_keypoints, project_motion_to_frame
from .rotations import forward_kinematics, rotation_6d_to_matrix
from .skeleton import CONTACT_JOINTS, PARENTS, REST_OFFSETS


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def _pair_mask(mask: Tensor) -> Tensor:
    return mask[:, 1:] * mask[:, :-1]


def _triple_mask(mask: Tensor) -> Tensor:
    return mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]


def compute_losses(
    prediction: dict[str, Tensor],
    target: dict[str, Tensor],
    weights: dict[str, Any],
) -> dict[str, Tensor]:
    mask = target["frame_mask"]
    losses: dict[str, Tensor] = {}
    if "local_rotations_6d" in prediction:
        predicted_rotation_matrices = rotation_6d_to_matrix(prediction["local_rotations_6d"])
        target_rotation_matrices = rotation_6d_to_matrix(target["local_rotations_6d"])
        losses["rotations"] = _masked_mean(
            (predicted_rotation_matrices - target_rotation_matrices).square(), mask
        )
    else:
        losses["rotations"] = prediction["joints_3d"].new_zeros(())
    losses["root_velocity"] = _masked_mean(
        (prediction["root_velocity_local"] - target["root_velocity_local"]).square(), mask
    )
    predicted_gravity = rotation_6d_to_matrix(prediction["gravity_view_orientation_6d"])
    target_gravity = rotation_6d_to_matrix(target["gravity_view_orientation_6d"])
    losses["orientation"] = _masked_mean((predicted_gravity - target_gravity).square(), mask)
    predicted_camera = rotation_6d_to_matrix(prediction["camera_orientation_6d"])
    target_camera = rotation_6d_to_matrix(target["camera_orientation_6d"])
    losses["camera_orientation"] = _masked_mean((predicted_camera - target_camera).square(), mask)
    losses["weak_camera"] = _masked_mean(
        (prediction["weak_camera"] - target["weak_camera"]).square(), mask
    )
    positive_weight = prediction["contacts"].new_tensor(
        weights.get("contacts_positive_weight", 1.0)
    )
    contact_bce = F.binary_cross_entropy_with_logits(
        prediction["contacts"],
        target["contacts"],
        reduction="none",
        pos_weight=positive_weight,
    )
    losses["contacts"] = _masked_mean(contact_bce, mask)

    if "joints_3d" in prediction:
        predicted_joints = prediction["joints_3d"]
    else:
        rotations = predicted_rotation_matrices
        root = torch.zeros_like(prediction["root_velocity_local"])
        offsets = torch.as_tensor(REST_OFFSETS, device=rotations.device, dtype=rotations.dtype)
        parents = torch.as_tensor(PARENTS, device=rotations.device)
        predicted_joints = forward_kinematics(rotations, root, offsets, parents)
    losses["joints_3d"] = _masked_mean((predicted_joints - target["joints_3d"]).square(), mask)

    if predicted_joints.shape[1] > 1:
        predicted_joint_velocity = predicted_joints[:, 1:] - predicted_joints[:, :-1]
        target_joint_velocity = target["joints_3d"][:, 1:] - target["joints_3d"][:, :-1]
        pair_mask = _pair_mask(mask)
        losses["joint_velocity"] = _masked_mean(
            (predicted_joint_velocity - target_joint_velocity).square(), pair_mask
        )
        root_velocity_delta = (
            prediction["root_velocity_local"][:, 1:] - prediction["root_velocity_local"][:, :-1]
        )
        target_root_velocity_delta = (
            target["root_velocity_local"][:, 1:] - target["root_velocity_local"][:, :-1]
        )
        losses["root_acceleration"] = _masked_mean(
            (root_velocity_delta - target_root_velocity_delta).square(), pair_mask
        )
        predicted_contact_delta = torch.sigmoid(prediction["contacts"][:, 1:]) - torch.sigmoid(
            prediction["contacts"][:, :-1]
        )
        target_contact_delta = target["contacts"][:, 1:] - target["contacts"][:, :-1]
        losses["contact_transitions"] = _masked_mean(
            (predicted_contact_delta - target_contact_delta).square(), pair_mask
        )

        fps = float(weights.get("temporal_fps", 30.0))
        ground_columns = torch.as_tensor((2, 3, 4, 5), device=mask.device)
        ground_joints = torch.as_tensor(CONTACT_JOINTS[2:], device=mask.device)
        ground_contact = target["contacts"].index_select(-1, ground_columns)
        contact_pair = torch.minimum(ground_contact[:, 1:], ground_contact[:, :-1])
        contact_pair = contact_pair * pair_mask.unsqueeze(-1)
        local_foot_velocity = (
            predicted_joints.index_select(-2, ground_joints)[:, 1:]
            - predicted_joints.index_select(-2, ground_joints)[:, :-1]
        ) * fps
        root_velocity = prediction["root_velocity_local"][:, 1:]
        gravity_root = predicted_gravity[:, 1:]
        gravity_root_velocity = (gravity_root @ root_velocity.unsqueeze(-1)).squeeze(-1)
        horizontal_world_velocity = (local_foot_velocity + gravity_root_velocity.unsqueeze(-2))[
            ..., (0, 2)
        ]
        losses["foot_sliding"] = _masked_mean(horizontal_world_velocity.square(), contact_pair)
        losses["smoothness"] = _masked_mean(root_velocity_delta.square(), pair_mask)
    else:
        zero = predicted_joints.new_zeros(())
        losses.update(
            joint_velocity=zero,
            root_acceleration=zero,
            contact_transitions=zero,
            foot_sliding=zero,
            smoothness=zero,
        )

    if predicted_joints.shape[1] > 2:
        predicted_acceleration = (
            predicted_joints[:, 2:] - 2.0 * predicted_joints[:, 1:-1] + predicted_joints[:, :-2]
        )
        target_joints = target["joints_3d"]
        target_acceleration = (
            target_joints[:, 2:] - 2.0 * target_joints[:, 1:-1] + target_joints[:, :-2]
        )
        losses["joint_acceleration"] = _masked_mean(
            (predicted_acceleration - target_acceleration).square(), _triple_mask(mask)
        )
    else:
        losses["joint_acceleration"] = predicted_joints.new_zeros(())
    if "detector_residual_3d" in prediction:
        confidence = target["detector_3d_confidence"] * mask.unsqueeze(-1)
        losses["detector_residual"] = _masked_mean(
            prediction["detector_residual_3d"].square(), confidence
        )

    reprojection_bbox = target.get("bbox_target", target["bbox"])
    projected = project_motion_to_frame(prediction, predicted_joints)
    reprojection_target = target.get("keypoints_2d_target", target["keypoints_2d"])
    reprojection_target_xy = denormalize_bbox_keypoints(
        reprojection_target[..., :2], reprojection_bbox
    )
    keypoint_confidence = reprojection_target[..., 2] * mask.unsqueeze(-1)
    losses["reprojection_2d"] = _masked_mean(
        (projected - reprojection_target_xy).square(), keypoint_confidence
    )
    losses["total"] = sum(losses[name] * float(weights.get(name, 0.0)) for name in losses)
    return losses
