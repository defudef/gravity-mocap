from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

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
    losses["rotations"] = _masked_mean(
        (prediction["local_rotations_6d"] - target["local_rotations_6d"]).square(), mask
    )
    losses["root_velocity"] = _masked_mean(
        (prediction["root_velocity_local"] - target["root_velocity_local"]).square(), mask
    )
    losses["orientation"] = _masked_mean(
        (
            prediction["gravity_view_orientation_6d"] - target["gravity_view_orientation_6d"]
        ).square(),
        mask,
    )
    losses["camera_orientation"] = _masked_mean(
        (prediction["camera_orientation_6d"] - target["camera_orientation_6d"]).square(),
        mask,
    )
    losses["weak_camera"] = _masked_mean(
        (prediction["weak_camera"] - target["weak_camera"]).square(), mask
    )
    contact_bce = F.binary_cross_entropy_with_logits(
        prediction["contacts"], target["contacts"], reduction="none"
    )
    losses["contacts"] = _masked_mean(contact_bce, mask)

    rotations = rotation_6d_to_matrix(prediction["local_rotations_6d"])
    root = torch.zeros_like(prediction["root_velocity_local"])
    offsets = torch.as_tensor(REST_OFFSETS, device=rotations.device, dtype=rotations.dtype)
    parents = torch.as_tensor(PARENTS, device=rotations.device)
    predicted_joints = forward_kinematics(rotations, root, offsets, parents)
    losses["joints_3d"] = _masked_mean((predicted_joints - target["joints_3d"]).square(), mask)

    projected = predicted_joints[..., :2] / predicted_joints[..., 2:].abs().clamp_min(0.25)
    reprojection_target = target.get("keypoints_2d_target", target["keypoints_2d"])
    keypoint_confidence = reprojection_target[..., 2] * target["image_mask"].unsqueeze(-1)
    losses["reprojection_2d"] = _masked_mean(
        (projected - reprojection_target[..., :2]).square(), keypoint_confidence
    )
    velocity_delta = (
        prediction["root_velocity_local"][:, 1:] - prediction["root_velocity_local"][:, :-1]
    )
    losses["smoothness"] = velocity_delta.square().mean()
    losses["total"] = sum(losses[name] * float(weights.get(name, 0.0)) for name in losses)
    return losses
