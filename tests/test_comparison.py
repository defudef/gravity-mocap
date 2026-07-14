from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gravity_mocap.comparison import compare_motion_previews


def _preview(
    path: Path,
    *,
    avatar_color: tuple[int, int, int],
    source_offset: int = 0,
) -> None:
    cv2 = pytest.importorskip("cv2")
    height = 64
    source_width = 96
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (source_width + height, height),
    )
    assert writer.isOpened()
    for frame_index in range(4):
        frame = np.zeros((height, source_width + height, 3), dtype=np.uint8)
        frame[:, :source_width] = (frame_index * 20, 40 + source_offset, 60)
        frame[:, source_width:] = avatar_color
        writer.write(frame)
    writer.release()


def test_comparison_preview_is_three_panels_and_idempotent(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    baseline = tmp_path / "baseline.mp4"
    learned = tmp_path / "learned.mp4"
    output = tmp_path / "comparison.mp4"
    _preview(baseline, avatar_color=(20, 120, 220))
    _preview(learned, avatar_color=(220, 120, 20))

    created = compare_motion_previews(baseline, learned, output)
    cached = compare_motion_previews(baseline, learned, output)

    assert created["status"] == "created"
    assert created["frames"] == 4
    assert cached["status"] == "cached"
    capture = cv2.VideoCapture(str(output))
    assert round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 96 + 2 * 64
    assert round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 64
    assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    capture.release()


def test_comparison_rejects_same_preview(tmp_path: Path) -> None:
    preview = tmp_path / "preview.mp4"
    preview.write_bytes(b"not-a-video")

    with pytest.raises(ValueError, match="must be different"):
        compare_motion_previews(preview, preview, tmp_path / "comparison.mp4")


def test_comparison_rejects_overwriting_input(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.mp4"
    learned = tmp_path / "learned.mp4"
    baseline.write_bytes(b"baseline")
    learned.write_bytes(b"learned")

    with pytest.raises(ValueError, match="must not overwrite"):
        compare_motion_previews(baseline, learned, baseline)


def test_comparison_rejects_different_source_video(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.mp4"
    learned = tmp_path / "learned.mp4"
    _preview(baseline, avatar_color=(20, 120, 220))
    _preview(learned, avatar_color=(220, 120, 20), source_offset=80)

    with pytest.raises(ValueError, match="different source videos"):
        compare_motion_previews(baseline, learned, tmp_path / "comparison.mp4")
