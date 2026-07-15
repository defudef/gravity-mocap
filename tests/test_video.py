import argparse
from pathlib import Path

import numpy as np
import yaml

from gravity_mocap.cli import (
    build_parser,
    command_compare_previews,
    command_diagnose_motion,
    command_generate_detector_loop,
    command_infer_detector_world,
    command_infer_rig,
    command_infer_video_baseline,
    command_validate,
    command_video_to_rig,
)
from gravity_mocap.trainer import load_config
from gravity_mocap.video import (
    default_inference_directory,
    resample_detector_boxes,
    resample_detector_values,
    selected_frame_indices,
)


def test_frame_selection_resamples_deterministically() -> None:
    indices = selected_frame_indices(60, 60.0, 30.0)
    assert np.array_equal(indices, np.arange(0, 60, 2))


def test_frame_selection_can_repeat_low_fps_source_and_limit_output() -> None:
    indices = selected_frame_indices(3, 15.0, 30.0, max_frames=5)
    assert np.array_equal(indices, np.asarray([0, 0, 1, 2, 2]))


def test_native_detector_values_are_interpolated_instead_of_duplicated() -> None:
    values = np.asarray([[[0.0, 0.0]], [[10.0, 4.0]]], dtype=np.float32)
    confidence = np.ones((2, 1), dtype=np.float32)

    output, output_confidence = resample_detector_values(
        values,
        confidence,
        source_fps=15.0,
        target_fps=30.0,
        output_frames=3,
    )

    assert output[:, 0].tolist() == [[0.0, 0.0], [5.0, 2.0], [10.0, 4.0]]
    assert np.array_equal(output_confidence, np.ones((3, 1), dtype=np.float32))


def test_detector_resampling_uses_confidence_weighted_coordinates() -> None:
    values = np.asarray([[[2.0, 4.0]], [[100.0, 200.0]]], dtype=np.float32)
    confidence = np.asarray([[1.0], [0.0]], dtype=np.float32)

    output, output_confidence = resample_detector_values(
        values,
        confidence,
        source_fps=15.0,
        target_fps=30.0,
        output_frames=3,
    )
    boxes = resample_detector_boxes(
        np.asarray([[0.0, 0.0, 2.0, 2.0], [2.0, 4.0, 6.0, 8.0]], dtype=np.float32),
        source_fps=15.0,
        target_fps=30.0,
        output_frames=3,
    )

    assert output[1, 0].tolist() == [2.0, 4.0]
    assert output_confidence[:, 0].tolist() == [1.0, 0.5, 0.0]
    assert output[2, 0].tolist() == [0.0, 0.0]
    assert boxes[1].tolist() == [1.0, 2.0, 4.0, 5.0]


def test_default_output_is_content_addressed(tmp_path: Path) -> None:
    video = tmp_path / "walk.mp4"
    video.write_bytes(b"first")
    first = default_inference_directory(video, tmp_path / "data")
    video.write_bytes(b"second")
    second = default_inference_directory(video, tmp_path / "data")

    assert first.name.startswith("walk-")
    assert first != second


def test_video_to_rig_cli_has_backward_compatible_detector_alias() -> None:
    parser = build_parser()

    canonical = parser.parse_args(["video-to-rig", "walk.mp4"])
    legacy = parser.parse_args(["detect-video", "walk.mp4"])
    infer = parser.parse_args(["infer-rig", "rig-2d.npz"])
    detector_world = parser.parse_args(["infer-detector-world", "detector-world-3d.npz"])
    baseline = parser.parse_args(["infer-video-baseline", "walk.mp4"])
    compare = parser.parse_args(
        ["compare-previews", "baseline.mp4", "learned.mp4", "comparison.mp4"]
    )
    diagnostics = parser.parse_args(["diagnose-motion", "motion.npz", "--rig-2d", "rig-2d.npz"])
    detector_loop = parser.parse_args(["generate-detector-loop", "walk.npz"])

    assert canonical.handler is command_video_to_rig
    assert legacy.handler is command_video_to_rig
    assert infer.handler is command_infer_rig
    assert detector_world.handler is command_infer_detector_world
    assert baseline.handler is command_infer_video_baseline
    assert compare.handler is command_compare_previews
    assert diagnostics.handler is command_diagnose_motion
    assert detector_loop.handler is command_generate_detector_loop
    assert detector_loop.execute is False
    assert infer.root_motion == "safe"
    assert infer.avatar_renderer == "auto"
    assert baseline.avatar_renderer == "auto"

    required_mesh = parser.parse_args(["infer-rig", "rig-2d.npz", "--avatar-renderer", "mesh"])
    assert required_mesh.avatar_renderer == "mesh"


def test_forward_validation_does_not_write_into_training_data_root(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/train-smoke.yaml")
    training_root = tmp_path / "processed"
    config["data"]["root"] = str(training_root)
    config_path = tmp_path / "validate.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    assert command_validate(argparse.Namespace(config=config_path)) == 0
    assert not training_root.exists()
