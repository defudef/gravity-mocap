from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .rotations import retarget_joints_to_neutral_skeleton
from .skeleton import PARENTS


def visual_feature_gate(
    image_mask: Tensor,
    detector_confidence: Tensor | None,
    *,
    confidence_aware: bool,
) -> Tensor:
    """Gate visual evidence harder when the geometric detector is already reliable."""
    gate = image_mask.clamp(0.0, 1.0)
    if confidence_aware and detector_confidence is not None:
        geometric_confidence = detector_confidence.mean(dim=-1).clamp(0.0, 1.0)
        gate = gate * (0.25 + 0.75 * (1.0 - geometric_confidence))
    return gate


class ModalityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


def _apply_rope(value: Tensor) -> Tensor:
    # value: (B, H, T, D), with D even.
    length, dimension = value.shape[-2:]
    half = dimension // 2
    positions = torch.arange(length, device=value.device, dtype=value.dtype)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=value.device, dtype=value.dtype)
        / max(half, 1)
    )
    angles = positions[:, None] * frequencies[None, :]
    cosine, sine = angles.cos()[None, None], angles.sin()[None, None]
    first, second = value[..., :half], value[..., half : 2 * half]
    rotated = torch.cat((first * cosine - second * sine, first * sine + second * cosine), dim=-1)
    return torch.cat((rotated, value[..., 2 * half :]), dim=-1)


class RelativeSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float, attention_radius: int):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if (hidden_dim // heads) % 2:
            raise ValueError("head dimension must be even for RoPE")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.attention_radius = attention_radius
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, value: Tensor) -> Tensor:
        batch, length, hidden = value.shape
        qkv = self.qkv(value).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, content = qkv.unbind(dim=2)
        query = _apply_rope(query.transpose(1, 2))
        key = _apply_rope(key.transpose(1, 2))
        content = content.transpose(1, 2)
        indices = torch.arange(length, device=value.device)
        permitted = (indices[:, None] - indices[None, :]).abs() < self.attention_radius
        mask = torch.zeros((length, length), dtype=value.dtype, device=value.device)
        mask.masked_fill_(~permitted, float("-inf"))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            content,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, hidden)
        return self.output(attended)


class RelativeTransformerBlock(nn.Module):
    def __init__(
        self, hidden_dim: int, heads: int, mlp_ratio: int, dropout: float, attention_radius: int
    ):
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.attention = RelativeSelfAttention(hidden_dim, heads, dropout, attention_radius)
        self.norm_mlp = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        value = value + self.attention(self.norm_attention(value))
        return value + self.mlp(self.norm_mlp(value))


class GravityViewMotionModel(nn.Module):
    """Paper-derived architecture with a generic skeleton and no SMPL dependency."""

    def __init__(
        self,
        *,
        joints: int = 22,
        image_feature_dim: int = 512,
        hidden_dim: int = 512,
        layers: int = 12,
        heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        attention_radius: int = 120,
        use_detector_world_3d: bool = False,
        pose_representation: str = "rotations",
        max_detector_residual_meters: float = 0.12,
        residual_confidence_floor: float = 0.25,
        max_root_speed_mps: float = 5.0,
        confidence_aware_visual_gating: bool = False,
    ):
        super().__init__()
        if pose_representation not in {"rotations", "detector_residual"}:
            raise ValueError("pose_representation must be 'rotations' or 'detector_residual'")
        if pose_representation == "detector_residual" and not use_detector_world_3d:
            raise ValueError("detector_residual pose requires detector world 3D")
        if max_detector_residual_meters <= 0:
            raise ValueError("max_detector_residual_meters must be positive")
        if not 0 <= residual_confidence_floor <= 1:
            raise ValueError("residual_confidence_floor must be in [0, 1]")
        if max_root_speed_mps <= 0:
            raise ValueError("max_root_speed_mps must be positive")
        self.joints = joints
        self.use_detector_world_3d = bool(use_detector_world_3d)
        self.pose_representation = pose_representation
        self.max_detector_residual_meters = float(max_detector_residual_meters)
        self.residual_confidence_floor = float(residual_confidence_floor)
        self.max_root_speed_mps = float(max_root_speed_mps)
        self.confidence_aware_visual_gating = bool(confidence_aware_visual_gating)
        self.modalities = nn.ModuleDict(
            {
                "bbox": ModalityMLP(4, hidden_dim),
                "keypoints_2d": ModalityMLP(joints * 3, hidden_dim),
                "image_features": ModalityMLP(image_feature_dim, hidden_dim),
                "camera_delta_6d": ModalityMLP(6, hidden_dim),
            }
        )
        if self.use_detector_world_3d:
            self.modalities["detector_world_3d"] = ModalityMLP(joints * 4, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                RelativeTransformerBlock(hidden_dim, heads, mlp_ratio, dropout, attention_radius)
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        pose_head_name = (
            "local_rotations_6d"
            if self.pose_representation == "rotations"
            else "detector_residual_3d"
        )
        pose_head_size = joints * (6 if self.pose_representation == "rotations" else 3)
        self.heads = nn.ModuleDict(
            {
                pose_head_name: nn.Linear(hidden_dim, pose_head_size),
                "camera_orientation_6d": nn.Linear(hidden_dim, 6),
                "gravity_view_orientation_6d": nn.Linear(hidden_dim, 6),
                "root_velocity_local": nn.Linear(hidden_dim, 3),
                "contacts": nn.Linear(hidden_dim, 6),
                "weak_camera": nn.Linear(hidden_dim, 3),
            }
        )
        if self.pose_representation == "detector_residual":
            nn.init.zeros_(self.heads["detector_residual_3d"].weight)
            nn.init.zeros_(self.heads["detector_residual_3d"].bias)
            nn.init.zeros_(self.heads["root_velocity_local"].weight)
            nn.init.zeros_(self.heads["root_velocity_local"].bias)
            nn.init.zeros_(self.heads["contacts"].weight)
            nn.init.zeros_(self.heads["contacts"].bias)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        tokens = self.modalities["bbox"](batch["bbox"])
        tokens = tokens + self.modalities["keypoints_2d"](batch["keypoints_2d"].flatten(-2))
        image_tokens = self.modalities["image_features"](batch["image_features"])
        detector_confidence = (
            batch["detector_3d_confidence"] if self.use_detector_world_3d else None
        )
        image_gate = visual_feature_gate(
            batch["image_mask"],
            detector_confidence,
            confidence_aware=self.confidence_aware_visual_gating,
        )
        tokens = tokens + image_tokens * image_gate.unsqueeze(-1)
        tokens = tokens + self.modalities["camera_delta_6d"](batch["camera_delta_6d"])
        if self.use_detector_world_3d:
            detector_world = torch.cat(
                (batch["detector_joints_3d"], batch["detector_3d_confidence"].unsqueeze(-1)),
                dim=-1,
            )
            tokens = tokens + self.modalities["detector_world_3d"](detector_world.flatten(-2))
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        outputs = {name: head(tokens) for name, head in self.heads.items()}
        if self.pose_representation == "rotations":
            outputs["local_rotations_6d"] = outputs["local_rotations_6d"].unflatten(
                -1, (self.joints, 6)
            )
        else:
            raw_residual = outputs["detector_residual_3d"].unflatten(-1, (self.joints, 3))
            confidence = batch["detector_3d_confidence"].clamp(0.0, 1.0)
            parent_indices = torch.as_tensor(
                [max(int(parent), 0) for parent in PARENTS],
                device=confidence.device,
            )
            parent_confidence = confidence.index_select(-1, parent_indices)
            bone_confidence = torch.minimum(confidence, parent_confidence)
            residual_scale = self.max_detector_residual_meters * (
                self.residual_confidence_floor
                + (1.0 - self.residual_confidence_floor) * (1.0 - bone_confidence)
            )
            residual = torch.tanh(raw_residual) * residual_scale.unsqueeze(-1)
            residual = residual.clone()
            residual[..., 0, :] = 0.0
            detector_pose = retarget_joints_to_neutral_skeleton(batch["detector_joints_3d"])
            outputs["detector_residual_3d"] = residual
            outputs["joints_3d"] = retarget_joints_to_neutral_skeleton(detector_pose + residual)
            raw_velocity = outputs["root_velocity_local"]
            speed = torch.linalg.vector_norm(raw_velocity, dim=-1, keepdim=True)
            bounded_scale = torch.where(
                speed > 1e-6,
                self.max_root_speed_mps
                * torch.tanh(speed / self.max_root_speed_mps)
                / speed.clamp_min(1e-6),
                torch.ones_like(speed),
            )
            outputs["root_velocity_local"] = raw_velocity * bounded_scale
        return outputs

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
