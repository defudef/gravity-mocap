from pathlib import Path

import numpy as np

from gravity_mocap.data import MotionWindowDataset
from gravity_mocap.fixture import create_fixture
from gravity_mocap.schema import read_shard


def test_fixture_round_trip_and_padding(tmp_path: Path) -> None:
    path = create_fixture(tmp_path / "synthetic/walk.npz", frames=6, image_feature_dim=32)
    arrays, provenance = read_shard(path)
    assert arrays["joints_3d"].shape == (6, 22, 3)
    assert provenance["license_id"] == "CC0-1.0"
    assert provenance["schema_version"] == 1
    dataset = MotionWindowDataset(tmp_path, sequence_length=8, stride=8)
    sample = dataset[0]
    assert sample["image_features"].shape == (8, 32)
    assert np.allclose(sample["frame_mask"][6:].numpy(), 0)
