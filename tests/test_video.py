from pathlib import Path

import numpy as np

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
