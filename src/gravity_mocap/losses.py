from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .projection import denormalize_bbox_keypoints, project_motion_to_frame
from .rotations import forward_kinematics, rotation_6d_to_matrix
from .skeleton import PARENTS, REST_OFFSETS


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def compute_losses(
    prediction: dict[str, Tensor],
    target: dict[str, Tensor],
    weights: dict[str, float],
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
        float(weights.get("contacts_positive_weight", 1.0))
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
    velocity_delta = (
        prediction["root_velocity_local"][:, 1:] - prediction["root_velocity_local"][:, :-1]
    )
    losses["smoothness"] = velocity_delta.square().mean()
    losses["total"] = sum(losses[name] * float(weights.get(name, 0.0)) for name in losses)
    return losses
