from __future__ import annotations

import torch
from torch import Tensor

from .rotations import forward_kinematics, rotation_6d_to_matrix
from .skeleton import PARENTS, REST_OFFSETS


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def compute_motion_metrics(
    prediction: dict[str, Tensor], target: dict[str, Tensor], *, fps: float
) -> dict[str, Tensor]:
    """Metrics for held-out, sequence-disjoint validation windows."""
    frame_mask = target["frame_mask"]
    rotations = rotation_6d_to_matrix(prediction["local_rotations_6d"])
    root = torch.zeros_like(prediction["root_velocity_local"])
    offsets = torch.as_tensor(REST_OFFSETS, device=rotations.device, dtype=rotations.dtype)
    parents = torch.as_tensor(PARENTS, device=rotations.device)
    predicted_joints = forward_kinematics(rotations, root, offsets, parents)

    joint_error = torch.linalg.vector_norm(predicted_joints - target["joints_3d"], dim=-1)
    velocity_error = torch.linalg.vector_norm(
        prediction["root_velocity_local"] - target["root_velocity_local"], dim=-1
    )
    integrated_error = torch.cumsum(
        (prediction["root_velocity_local"] - target["root_velocity_local"]) / float(fps),
        dim=1,
    )
    root_local_drift = torch.linalg.vector_norm(integrated_error, dim=-1)

    metrics = {
        "mpjpe_m": _masked_mean(joint_error, frame_mask),
        "root_velocity_error_mps": _masked_mean(velocity_error, frame_mask),
        "root_local_drift_m": _masked_mean(root_local_drift, frame_mask),
    }
    if "detector_joints_3d" in target and "detector_3d_confidence" in target:
        detector_error = torch.linalg.vector_norm(
            target["detector_joints_3d"] - target["joints_3d"], dim=-1
        )
        detector_mask = frame_mask.unsqueeze(-1) * (target["detector_3d_confidence"] > 0).to(
            frame_mask.dtype
        )
        detector_mpjpe = _masked_mean(detector_error, detector_mask)
        metrics["detector_prior_mpjpe_m"] = detector_mpjpe
        metrics["detector_prior_coverage"] = _masked_mean(
            (target["detector_3d_confidence"] > 0).to(frame_mask.dtype),
            frame_mask,
        )
        # Positive means the learned output beats the detector prior.
        metrics["mpjpe_gain_vs_detector_m"] = detector_mpjpe - metrics["mpjpe_m"]

    if predicted_joints.shape[1] >= 3:
        predicted_acceleration = (
            predicted_joints[:, 2:] - 2 * predicted_joints[:, 1:-1] + predicted_joints[:, :-2]
        ) * float(fps) ** 2
        target_acceleration = (
            target["joints_3d"][:, 2:]
            - 2 * target["joints_3d"][:, 1:-1]
            + target["joints_3d"][:, :-2]
        ) * float(fps) ** 2
        acceleration_mask = frame_mask[:, 2:] * frame_mask[:, 1:-1] * frame_mask[:, :-2]
        acceleration_error = torch.linalg.vector_norm(
            predicted_acceleration - target_acceleration, dim=-1
        )
        metrics["acceleration_error_mps2"] = _masked_mean(acceleration_error, acceleration_mask)

    contact_mask = frame_mask.unsqueeze(-1).bool()
    predicted_contacts = prediction["contacts"].sigmoid() >= 0.5
    target_contacts = target["contacts"] >= 0.5
    true_positive = (predicted_contacts & target_contacts & contact_mask).sum().float()
    false_positive = (predicted_contacts & ~target_contacts & contact_mask).sum().float()
    false_negative = (~predicted_contacts & target_contacts & contact_mask).sum().float()
    metrics["contact_f1"] = (
        2 * true_positive / (2 * true_positive + false_positive + false_negative).clamp_min(1.0)
    )
    return metrics
