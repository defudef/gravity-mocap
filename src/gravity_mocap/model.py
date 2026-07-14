from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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
    ):
        super().__init__()
        self.joints = joints
        self.use_detector_world_3d = bool(use_detector_world_3d)
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
        self.heads = nn.ModuleDict(
            {
                "local_rotations_6d": nn.Linear(hidden_dim, joints * 6),
                "camera_orientation_6d": nn.Linear(hidden_dim, 6),
                "gravity_view_orientation_6d": nn.Linear(hidden_dim, 6),
                "root_velocity_local": nn.Linear(hidden_dim, 3),
                "contacts": nn.Linear(hidden_dim, 6),
                "weak_camera": nn.Linear(hidden_dim, 3),
            }
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        tokens = self.modalities["bbox"](batch["bbox"])
        tokens = tokens + self.modalities["keypoints_2d"](batch["keypoints_2d"].flatten(-2))
        image_tokens = self.modalities["image_features"](batch["image_features"])
        tokens = tokens + image_tokens * batch["image_mask"].unsqueeze(-1)
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
        outputs["local_rotations_6d"] = outputs["local_rotations_6d"].unflatten(
            -1, (self.joints, 6)
        )
        return outputs

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
