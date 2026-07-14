from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def rotation_6d_to_matrix(value: Tensor) -> Tensor:
    """Convert the continuous 6D representation to a 3x3 rotation matrix."""
    first, second = value[..., :3], value[..., 3:]
    x_axis = F.normalize(first, dim=-1)
    y_axis = F.normalize(second - (x_axis * second).sum(-1, keepdim=True) * x_axis, dim=-1)
    z_axis = torch.cross(x_axis, y_axis, dim=-1)
    return torch.stack((x_axis, y_axis, z_axis), dim=-1)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def identity_rotation_6d(*shape: int, device: torch.device | str | None = None) -> Tensor:
    value = torch.zeros((*shape, 6), dtype=torch.float32, device=device)
    value[..., 0] = 1.0
    value[..., 4] = 1.0
    return value


def forward_kinematics(
    local_rotations: Tensor, root_position: Tensor, offsets: Tensor, parents: Tensor
) -> Tensor:
    """Run FK for column-vector rotations.

    local_rotations: (..., J, 3, 3), root_position: (..., 3)
    """
    world_rotations: list[Tensor] = []
    world_positions: list[Tensor] = []
    for joint in range(local_rotations.shape[-3]):
        parent = int(parents[joint])
        if parent < 0:
            world_rotations.append(local_rotations[..., joint, :, :])
            world_positions.append(root_position)
            continue
        parent_rotation = world_rotations[parent]
        world_rotations.append(parent_rotation @ local_rotations[..., joint, :, :])
        offset = offsets[joint].expand(root_position.shape)
        world_positions.append(
            world_positions[parent] + (parent_rotation @ offset.unsqueeze(-1)).squeeze(-1)
        )
    return torch.stack(world_positions, dim=-2)


def integrate_root_velocity(
    local_velocity: Tensor, world_orientation: Tensor, *, fps: float
) -> Tensor:
    """Integrate local velocity expressed in metres per second."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    world_velocity = (world_orientation @ local_velocity.unsqueeze(-1)).squeeze(-1)
    origin = torch.zeros_like(world_velocity[..., :1, :])
    return torch.cat(
        (origin, torch.cumsum(world_velocity[..., :-1, :] / float(fps), dim=-2)), dim=-2
    )
