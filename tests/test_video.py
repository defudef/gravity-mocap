from pathlib import Path

import numpy as np

from gravity_mocap.cli import (
    build_parser,
    command_infer_detector_world,
    command_infer_rig,
    command_infer_video_baseline,
    command_video_to_rig,
)
from gravity_mocap.video import default_inference_directory, selected_frame_indices


def test_frame_selection_resamples_deterministically() -> None:
    indices = selected_frame_indices(60, 60.0, 30.0)
    assert np.array_equal(indices, np.arange(0, 60, 2))


def test_frame_selection_can_repeat_low_fps_source_and_limit_output() -> None:
    indices = selected_frame_indices(3, 15.0, 30.0, max_frames=5)
    assert np.array_equal(indices, np.asarray([0, 0, 1, 2, 2]))


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

    assert canonical.handler is command_video_to_rig
    assert legacy.handler is command_video_to_rig
    assert infer.handler is command_infer_rig
    assert detector_world.handler is command_infer_detector_world
    assert baseline.handler is command_infer_video_baseline
