from pathlib import Path

import pytest
import torch

from gravity_mocap.data import MotionWindowDataset
from gravity_mocap.fixture import create_fixture
from gravity_mocap.losses import compute_losses
from gravity_mocap.metrics import compute_motion_metrics
from gravity_mocap.model import GravityViewMotionModel
from gravity_mocap.rotations import retarget_joints_to_neutral_skeleton
from gravity_mocap.skeleton import SKELETON
from gravity_mocap.trainer import build_model, evaluate_model, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_small_config_changes_only_model_capacity() -> None:
    paper = load_config(ROOT / "configs/train-paper.yaml")
    small = load_config(ROOT / "configs/train-small.yaml")

    assert small["seed"] == paper["seed"]
    for section in ("data", "train", "validation", "loss"):
        assert small[section] == paper[section]
    assert small["model"] == {
        **paper["model"],
        "hidden_dim": 384,
        "layers": 6,
    }
    assert build_model(small).parameter_count == 11_708_316


def test_forward_and_losses_are_finite_without_training(tmp_path: Path) -> None:
    create_fixture(tmp_path / "synthetic/walk.npz", frames=8, image_feature_dim=32)
    dataset = MotionWindowDataset(
        tmp_path,
        sequence_length=8,
        stride=8,
        gravity_view_contract=True,
    )
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
    assert losses["reprojection_2d"] < 10
    metrics = compute_motion_metrics(prediction, batch, fps=30)
    assert {"mpjpe_m", "root_velocity_error_mps", "root_local_drift_m", "contact_f1"} <= set(
        metrics
    )
    assert all(torch.isfinite(value) for value in metrics.values())


def test_motion_targets_are_fk_consistent_and_reprojection_stays_active(
    tmp_path: Path,
) -> None:
    create_fixture(tmp_path / "synthetic/walk.npz", frames=8, image_feature_dim=32)
    dataset = MotionWindowDataset(
        tmp_path,
        sequence_length=8,
        stride=8,
        gravity_view_contract=True,
    )
    batch = {name: value.unsqueeze(0) for name, value in dataset[0].items()}
    prediction_names = (
        "local_rotations_6d",
        "root_velocity_local",
        "gravity_view_orientation_6d",
        "camera_orientation_6d",
        "weak_camera",
    )
    prediction = {name: batch[name].clone() for name in prediction_names}
    prediction["contacts"] = torch.where(
        batch["contacts"] > 0.5,
        torch.full_like(batch["contacts"], 10.0),
        torch.full_like(batch["contacts"], -10.0),
    )
    weights = {
        "rotations": 1.0,
        "root_velocity": 1.0,
        "orientation": 1.0,
        "camera_orientation": 1.0,
        "weak_camera": 1.0,
        "contacts": 1.0,
        "joints_3d": 1.0,
        "reprojection_2d": 1.0,
        "smoothness": 1.0,
    }

    losses = compute_losses(prediction, batch, weights)

    assert batch["image_mask"].sum() == 0
    assert losses["joints_3d"] < 1e-12
    assert losses["reprojection_2d"] < 1e-10

    shifted = {name: value.clone() for name, value in prediction.items()}
    shifted["weak_camera"][..., 1] += 0.1
    shifted_losses = compute_losses(shifted, batch, weights)
    assert shifted_losses["reprojection_2d"] > 1e-3


def test_zero_image_mask_completely_gates_visual_features(tmp_path: Path) -> None:
    create_fixture(tmp_path / "synthetic/walk.npz", frames=8, image_feature_dim=32)
    dataset = MotionWindowDataset(tmp_path, sequence_length=8, stride=8)
    batch = {name: value.unsqueeze(0) for name, value in dataset[0].items()}
    changed = {name: value.clone() for name, value in batch.items()}
    changed["image_features"].fill_(1000)
    assert batch["image_mask"].sum() == 0
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
        baseline = model(batch)
        visual_noise = model(changed)

    for name in baseline:
        assert torch.equal(baseline[name], visual_noise[name])


def test_held_out_evaluation_is_forward_only_and_reports_motion_metrics(tmp_path: Path) -> None:
    path = create_fixture(tmp_path / "synthetic/walk.npz", frames=16, image_feature_dim=32)
    config = load_config(ROOT / "configs/train-smoke.yaml")
    dataset = MotionWindowDataset(
        tmp_path,
        sequence_length=8,
        stride=8,
        paths=[path],
        gravity_view_contract=True,
        detector_world_3d=config["data"]["detector_world_3d"],
    )
    model = build_model(config)

    metrics = evaluate_model(model, dataset, config, torch.device("cpu"), use_amp=False)

    assert "loss.total" in metrics
    assert "mpjpe_m" in metrics
    assert "detector_prior_mpjpe_m" in metrics
    assert "mpjpe_gain_vs_detector_m" in metrics
    assert 0 <= metrics["detector_prior_coverage"] <= 1
    assert "root_local_drift_m" in metrics
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_detector_residual_pose_starts_as_safe_neutral_prior(tmp_path: Path) -> None:
    create_fixture(tmp_path / "synthetic/walk.npz", frames=8, image_feature_dim=32)
    detector_config = {
        "enabled": True,
        "noise_std_meters": 0.02,
        "confidence_min": 0.5,
        "joint_dropout_probability": 0.05,
    }
    dataset = MotionWindowDataset(
        tmp_path,
        sequence_length=8,
        stride=8,
        gravity_view_contract=True,
        detector_world_3d=detector_config,
    )
    batch = {name: value.unsqueeze(0) for name, value in dataset[0].items()}
    model = GravityViewMotionModel(
        image_feature_dim=32,
        hidden_dim=64,
        layers=2,
        heads=4,
        mlp_ratio=2,
        dropout=0,
        attention_radius=8,
        use_detector_world_3d=True,
        pose_representation="detector_residual",
    ).eval()

    with torch.no_grad():
        prediction = model(batch)
    expected = retarget_joints_to_neutral_skeleton(batch["detector_joints_3d"])

    assert "local_rotations_6d" not in prediction
    assert torch.equal(prediction["detector_residual_3d"], torch.zeros_like(expected))
    assert torch.allclose(prediction["joints_3d"], expected, atol=1e-6)
    bone_lengths = torch.linalg.vector_norm(
        prediction["joints_3d"][..., 1:, :] - prediction["joints_3d"][..., SKELETON.parents[1:], :],
        dim=-1,
    )
    expected_lengths = torch.linalg.vector_norm(torch.from_numpy(SKELETON.rest_offsets[1:]), dim=-1)
    assert torch.allclose(bone_lengths, expected_lengths, atol=1e-6)


def test_detector_residual_is_bounded_by_confidence() -> None:
    model = GravityViewMotionModel(
        image_feature_dim=8,
        hidden_dim=32,
        layers=1,
        heads=4,
        mlp_ratio=2,
        dropout=0,
        attention_radius=8,
        use_detector_world_3d=True,
        pose_representation="detector_residual",
        max_detector_residual_meters=0.12,
        residual_confidence_floor=0.25,
    ).eval()
    model.heads["detector_residual_3d"].bias.data.fill_(10.0)
    batch = {
        "bbox": torch.zeros(1, 2, 4),
        "keypoints_2d": torch.zeros(1, 2, 22, 3),
        "image_features": torch.zeros(1, 2, 8),
        "image_mask": torch.zeros(1, 2),
        "camera_delta_6d": torch.zeros(1, 2, 6),
        "detector_joints_3d": torch.zeros(1, 2, 22, 3),
        "detector_3d_confidence": torch.stack((torch.ones(22), torch.zeros(22)), dim=0).unsqueeze(
            0
        ),
    }

    with torch.no_grad():
        residual = model(batch)["detector_residual_3d"]

    assert residual[0, 0, 1, 0] == pytest.approx(0.03, abs=1e-5)
    assert residual[0, 1, 1, 0] == pytest.approx(0.12, abs=1e-5)
    assert torch.equal(residual[..., 0, :], torch.zeros_like(residual[..., 0, :]))
