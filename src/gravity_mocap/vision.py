from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .catalog import DatasetCatalog
from .schema import read_shard, write_shard


class FrameEncoder(nn.Module):
    """Small from-scratch crop encoder; no third-party pretrained weights."""

    def __init__(self, feature_dim: int = 512, joints: int = 22):
        super().__init__()
        self.joints = joints
        channels = [3, 32, 64, 128, 256]
        blocks: list[nn.Module] = []
        for input_channels, output_channels in zip(channels, channels[1:], strict=True):
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.GELU(),
                    nn.Conv2d(
                        output_channels,
                        output_channels,
                        3,
                        padding=1,
                        groups=output_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.GELU(),
                ]
            )
        self.backbone = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.feature_head = nn.Linear(channels[-1], feature_dim)
        self.keypoint_head = nn.Linear(feature_dim, joints * 2)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.backbone(images)
        features = self.feature_head(encoded)
        keypoints = self.keypoint_head(features).reshape(*encoded.shape[:-1], self.joints, 2).tanh()
        return features, keypoints


class FrameManifestDataset(Dataset[dict[str, Tensor]]):
    """JSONL rows: image, keypoints_2d (Jx3), source_id, license_id."""

    def __init__(self, manifest: Path, catalog: DatasetCatalog, image_size: int = 256):
        self.root = manifest.parent
        self.image_size = image_size
        self.rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"Empty vision manifest: {manifest}")
        for row in self.rows:
            if not row.get("source_id") or not row.get("license_id"):
                raise ValueError("Every vision row needs source_id and license_id")
            entry = catalog.require_approved(row["source_id"])
            if entry.task != "paired_video":
                raise ValueError(f"{entry.dataset_id} is not approved as paired-video supervision")
            if row["license_id"] != entry.license_id:
                raise ValueError(
                    f"Manifest license {row['license_id']!r} does not match "
                    f"catalog license {entry.license_id!r}"
                )
            keypoints = np.asarray(row.get("keypoints_2d", []))
            if keypoints.shape != (22, 3):
                raise ValueError("Every vision row needs keypoints_2d shaped (22, 3)")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.rows[index]
        path = Path(row["image"])
        if not path.is_absolute():
            path = self.root / path
        image = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - 0.5) / 0.5
        keypoints = np.asarray(row["keypoints_2d"], dtype=np.float32)
        return {
            "image": torch.from_numpy(pixels).permute(2, 0, 1),
            "keypoints": torch.from_numpy(keypoints[:, :2]),
            "confidence": torch.from_numpy(keypoints[:, 2]),
        }


def train_frame_encoder(
    manifest: Path,
    output: Path,
    *,
    catalog: DatasetCatalog,
    feature_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> None:
    dataset = FrameManifestDataset(manifest, catalog)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = FrameEncoder(feature_dim=feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            images = batch["image"].to(device)
            target = batch["keypoints"].to(device)
            confidence = batch["confidence"].to(device).unsqueeze(-1)
            _, predicted = model(images)
            loss = ((predicted - target).square() * confidence).sum() / confidence.sum().clamp_min(
                1
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "feature_dim": feature_dim,
            "joints": 22,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "sources": sorted({row["source_id"] for row in dataset.rows}),
            "licenses": sorted({row["license_id"] for row in dataset.rows}),
        },
        output,
    )


def extract_frame_features(
    manifest: Path,
    checkpoint_path: Path,
    output: Path,
    *,
    catalog: DatasetCatalog,
    device: str,
    batch_size: int,
) -> None:
    dataset = FrameManifestDataset(manifest, catalog)
    sources = {row["source_id"] for row in dataset.rows}
    licenses = {row["license_id"] for row in dataset.rows}
    if len(sources) != 1 or len(licenses) != 1:
        raise ValueError("Feature export manifest must contain one source and license")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = FrameEncoder(
        feature_dim=int(checkpoint["feature_dim"]), joints=int(checkpoint["joints"])
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    features: list[np.ndarray] = []
    keypoints: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            encoded, predicted_keypoints = model(batch["image"].to(device))
            features.append(encoded.cpu().numpy())
            confidence = batch["confidence"].unsqueeze(-1).numpy()
            keypoints.append(
                np.concatenate((predicted_keypoints.cpu().numpy(), confidence), axis=-1)
            )
    target_keypoints = np.asarray([row["keypoints_2d"] for row in dataset.rows], dtype=np.float32)
    bbox = np.concatenate(
        (target_keypoints[..., :2].min(axis=1), target_keypoints[..., :2].max(axis=1)), axis=-1
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        image_features=np.concatenate(features),
        keypoints_2d=np.concatenate(keypoints),
        bbox=bbox,
        source_id=np.asarray(next(iter(sources))),
        license_id=np.asarray(next(iter(licenses))),
        checkpoint_sha256=np.asarray(hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()),
    )


def attach_features(shard_path: Path, feature_path: Path, *, catalog: DatasetCatalog) -> None:
    arrays, provenance = read_shard(shard_path)
    entry = catalog.require_approved(provenance["source_id"])
    if provenance["license_id"] != entry.license_id:
        raise ValueError("Shard license does not match the approved catalog")
    with np.load(feature_path, allow_pickle=False) as archive:
        feature_source = str(archive["source_id"])
        feature_license = str(archive["license_id"])
        if feature_source != provenance["source_id"] or feature_license != provenance["license_id"]:
            raise ValueError("Feature source/license does not match the motion shard")
        features = np.asarray(archive["image_features"], dtype=np.float32)
        if features.shape[0] != arrays["frame_mask"].shape[0]:
            raise ValueError("Feature count does not match shard frame count")
        arrays["image_features"] = features
        arrays["image_mask"] = np.ones(len(features), dtype=np.float32)
        if "keypoints_2d" in archive:
            arrays["keypoints_2d"] = np.asarray(archive["keypoints_2d"], dtype=np.float32)
        if "bbox" in archive:
            arrays["bbox"] = np.asarray(archive["bbox"], dtype=np.float32)
    provenance["vision_features_sha256"] = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    provenance["input_mode"] = "paired_video"
    write_shard(shard_path, arrays, provenance)
