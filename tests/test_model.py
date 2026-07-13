from pathlib import Path

import torch

from gravity_mocap.data import MotionWindowDataset
from gravity_mocap.fixture import create_fixture
from gravity_mocap.losses import compute_losses
from gravity_mocap.model import GravityViewMotionModel


def test_forward_and_losses_are_finite_without_training(tmp_path: Path) -> None:
    create_fixture(tmp_path / "synthetic/walk.npz", frames=8, image_feature_dim=32)
    dataset = MotionWindowDataset(tmp_path, sequence_length=8, stride=8)
    batch = {name: value.unsqueeze(0) for name, value in dataset[0].items()}
    model = GravityViewMotionModel(
        image_feature_dim=32,
        hidden_dim=64,
        layers=2,
        heads=4,
        mlp_ratio=2,
        dropout=0,
        attention_radius=8,
    ).eval()
    with torch.no_grad():
        prediction = model(batch)
        losses = compute_losses(
            prediction,
            batch,
            {
                "rotations": 1,
                "root_velocity": 1,
                "orientation": 1,
                "camera_orientation": 1,
                "weak_camera": 1,
                "contacts": 0.1,
                "joints_3d": 1,
                "reprojection_2d": 0.1,
                "smoothness": 0.05,
            },
        )
    assert prediction["local_rotations_6d"].shape == (1, 8, 22, 6)
    assert all(torch.isfinite(value) for value in losses.values())
