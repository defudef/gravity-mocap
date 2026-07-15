from pathlib import Path

import numpy as np

from gravity_mocap.evaluation import diagnose_motion
from gravity_mocap.rig2d import RIG_2D_VERSION, Rig2D, write_rig_2d
from gravity_mocap.skeleton import SKELETON


def _write_motion(path: Path, *, learned: bool, offset: float = 0.0) -> None:
    frames = 5
    joints = np.zeros((frames, SKELETON.joint_count, 3), dtype=np.float32)
    joints[..., 0] = offset
    contacts = np.zeros((frames, len(SKELETON.contact_joints)), dtype=np.float32)
    arrays = {
        "joints_3d": joints,
        "contacts": contacts,
        "fps": np.asarray(30.0, dtype=np.float32),
        "joint_names": np.asarray(SKELETON.names),
    }
    if learned:
        root = np.zeros((frames, 3), dtype=np.float32)
        root[:, 0] = np.arange(frames, dtype=np.float32) * 0.1
        arrays["root_translation"] = root
        arrays["joints_world"] = joints + root[:, None]
    np.savez_compressed(path, **arrays)


def _write_rig(path: Path, *, temporal_resampling_version: int) -> None:
    frames = 5
    zeros_2d = np.zeros((frames, SKELETON.joint_count, 3), dtype=np.float32)
    write_rig_2d(
        path,
        Rig2D(
            keypoints_2d=zeros_2d,
            bbox=np.zeros((frames, 4), dtype=np.float32),
            camera_delta_6d=np.zeros((frames, 6), dtype=np.float32),
            image_mask=np.zeros(frames, dtype=np.float32),
            frame_mask=np.ones(frames, dtype=np.float32),
            pixel_keypoints=zeros_2d,
            pixel_bbox=np.zeros((frames, 4), dtype=np.float32),
            source_frame_indices=np.asarray([0, 0, 1, 2, 2], dtype=np.int64),
            fps=30.0,
            source_fps=15.0,
            frame_width=320,
            frame_height=240,
            provenance={
                "rig_2d_version": RIG_2D_VERSION,
                "artifact_type": "gravity-mocap-rig-2d",
                "temporal_resampling_version": temporal_resampling_version,
            },
        ),
    )


def test_diagnostics_exposes_world_root_and_legacy_resampling(tmp_path: Path) -> None:
    motion = tmp_path / "motion.npz"
    baseline = tmp_path / "baseline.npz"
    rig = tmp_path / "rig-2d.npz"
    output = tmp_path / "diagnostics.json"
    _write_motion(motion, learned=True, offset=0.01)
    _write_motion(baseline, learned=False)
    _write_rig(rig, temporal_resampling_version=1)

    result = diagnose_motion(
        motion,
        rig,
        output,
        baseline_motion_path=baseline,
    )

    assert output.is_file()
    assert result["ground_truth_available"] is False
    assert result["frontend"]["repeated_preview_source_indices"] == 2
    assert np.isclose(result["world_root"]["path_length_m"], 0.4)
    assert np.isclose(result["baseline_comparison"]["correction_mean_m"], 0.01)
    assert any("legacy frontend" in warning for warning in result["warnings"])
    assert any("contact predictions" in warning for warning in result["warnings"])


def test_diagnostics_accepts_interpolated_low_fps_preview_indices(tmp_path: Path) -> None:
    motion = tmp_path / "motion.npz"
    rig = tmp_path / "rig-2d.npz"
    _write_motion(motion, learned=True)
    _write_rig(rig, temporal_resampling_version=2)

    result = diagnose_motion(motion, rig, tmp_path / "diagnostics.json")

    assert result["frontend"]["repeated_preview_source_indices"] == 2
    assert not any("legacy frontend" in warning for warning in result["warnings"])


def test_diagnostics_rejects_body_model_fields(tmp_path: Path) -> None:
    motion = tmp_path / "motion.npz"
    rig = tmp_path / "rig-2d.npz"
    _write_rig(rig, temporal_resampling_version=2)
    np.savez_compressed(
        motion,
        joints_3d=np.zeros((5, SKELETON.joint_count, 3), dtype=np.float32),
        joints_world=np.zeros((5, SKELETON.joint_count, 3), dtype=np.float32),
        root_translation=np.zeros((5, 3), dtype=np.float32),
        contacts=np.zeros((5, len(SKELETON.contact_joints)), dtype=np.float32),
        fps=np.asarray(30.0),
        joint_names=np.asarray(SKELETON.names),
        smpl_pose=np.zeros((5, 72), dtype=np.float32),
    )

    try:
        diagnose_motion(motion, rig, tmp_path / "diagnostics.json")
    except ValueError as error:
        assert "prohibited body-model fields" in str(error)
    else:
        raise AssertionError("diagnostics accepted a prohibited parametric body field")
