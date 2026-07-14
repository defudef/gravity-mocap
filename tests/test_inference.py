import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gravity_mocap.checkpoint import CHECKPOINT_VERSION
from gravity_mocap.inference import infer_motion, infer_rig, window_starts
from gravity_mocap.model import GravityViewMotionModel
from gravity_mocap.rotations import identity_rotation_6d, integrate_root_velocity
from gravity_mocap.skeleton import JOINT_NAMES, SKELETON


def _checkpoint(path: Path) -> None:
    model_config = {
        "joints": 22,
        "image_feature_dim": 8,
        "hidden_dim": 32,
        "layers": 1,
        "heads": 4,
        "mlp_ratio": 2,
        "dropout": 0.0,
        "attention_radius": 8,
    }
    model = GravityViewMotionModel(**model_config)
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model": model.state_dict(),
            "config": {
                "model": model_config,
                "data": {"sequence_length": 8, "stride": 4},
            },
            "compatibility_hash": "test-config",
            "data_bom_hash": "test-data",
        },
        path,
    )


def _detector_inputs(path: Path, frames: int = 11) -> None:
    provenance = {"rig_2d_version": 1, "request_hash": "detector-test"}
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            keypoints_2d=np.zeros((frames, 22, 3), dtype=np.float32),
            bbox=np.zeros((frames, 4), dtype=np.float32),
            camera_delta_6d=identity_rotation_6d(frames).numpy(),
            image_mask=np.zeros(frames, dtype=np.float32),
            frame_mask=np.ones(frames, dtype=np.float32),
            pixel_keypoints=np.zeros((frames, 22, 3), dtype=np.float32),
            pixel_bbox=np.tile(np.asarray([10, 10, 100, 180], dtype=np.float32), (frames, 1)),
            source_frame_indices=np.arange(frames, dtype=np.int64),
            fps=np.asarray(30.0, dtype=np.float32),
            source_fps=np.asarray(30.0, dtype=np.float32),
            frame_width=np.asarray(640, dtype=np.int32),
            frame_height=np.asarray(480, dtype=np.int32),
            joint_names=np.asarray(JOINT_NAMES),
            provenance_json=np.asarray(json.dumps(provenance)),
        )


def test_window_starts_cover_tail_exactly() -> None:
    assert window_starts(11, 8, 4) == [0, 3]
    assert window_starts(4, 8, 4) == [0]


def test_velocity_integration_uses_seconds() -> None:
    velocity = torch.tensor([[3.0, 0.0, 0.0]]).repeat(4, 1)
    orientation = torch.eye(3).repeat(4, 1, 1)
    translation = integrate_root_velocity(velocity, orientation, fps=30.0)
    assert torch.allclose(translation[:, 0], torch.tensor([0.0, 0.1, 0.2, 0.3]))


@pytest.mark.parametrize("frames", [5, 11])
def test_inference_writes_finite_neutral_motion(tmp_path: Path, frames: int) -> None:
    checkpoint = tmp_path / "best.pt"
    detector = tmp_path / "detector-inputs.npz"
    video = tmp_path / "input.mp4"
    output = tmp_path / "output"
    _checkpoint(checkpoint)
    _detector_inputs(detector, frames=frames)
    video.write_bytes(b"preview-disabled")

    result = infer_motion(
        detector,
        video,
        checkpoint,
        output,
        device_name="cpu",
        preview=False,
    )

    assert result["status"] == "created"
    with np.load(result["motion"], allow_pickle=False) as archive:
        assert archive["local_rotations_6d"].shape == (frames, SKELETON.joint_count, 6)
        assert archive["joints_world"].shape == (frames, SKELETON.joint_count, 3)
        assert archive["contacts"].shape == (frames, len(SKELETON.contact_joints))
        assert np.isfinite(archive["joints_world"]).all()

    cached = infer_motion(
        detector,
        video,
        checkpoint,
        output,
        device_name="cpu",
        preview=False,
    )
    assert cached["status"] == "cached"


def test_infer_rig_does_not_require_source_video_without_preview(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    rig = tmp_path / "rig-2d.npz"
    output = tmp_path / "output"
    _checkpoint(checkpoint)
    _detector_inputs(rig)

    result = infer_rig(rig, checkpoint, output, device_name="cpu", preview=False)

    assert result["status"] == "created"
    with np.load(result["motion"], allow_pickle=False) as archive:
        provenance = json.loads(str(archive["provenance_json"]))
    assert provenance["rig_2d"] == str(rig)
    assert provenance["rig_2d_provenance"]["request_hash"] == "detector-test"
