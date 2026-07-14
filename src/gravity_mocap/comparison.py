from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_json_atomic
from .schema import stable_hash
from .video import _video_dependencies, file_sha256

COMPARISON_PREVIEW_VERSION = 1


def _capture_metadata(cv2: Any, capture: Any, path: Path) -> dict[str, float | int]:
    if not capture.isOpened():
        raise ValueError(f"Cannot open motion preview: {path}")
    width = round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if width <= height or height < 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid motion preview metadata: {path}")
    return {"width": width, "height": height, "fps": fps, "frames": frames}


def compare_motion_previews(
    baseline_preview: Path,
    learned_preview: Path,
    output: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Compose source video, detector baseline, and learned avatar side by side."""
    baseline = baseline_preview.expanduser().resolve()
    learned = learned_preview.expanduser().resolve()
    output = output.expanduser().resolve()
    if not baseline.is_file():
        raise ValueError(f"Baseline preview does not exist: {baseline}")
    if not learned.is_file():
        raise ValueError(f"Learned preview does not exist: {learned}")
    if baseline == learned:
        raise ValueError("Baseline and learned previews must be different files")
    if output in {baseline, learned}:
        raise ValueError("Comparison output must not overwrite an input preview")

    request = {
        "comparison_preview_version": COMPARISON_PREVIEW_VERSION,
        "baseline_preview_sha256": file_sha256(baseline),
        "learned_preview_sha256": file_sha256(learned),
    }
    request_hash = stable_hash(request)
    manifest_path = output.with_name(f"{output.stem}-manifest.json")
    if not force and output.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("request_hash") == request_hash:
                return {
                    "status": "cached",
                    "frames": int(manifest["frames"]),
                    "comparison": str(output),
                    "manifest": str(manifest_path),
                }
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    cv2, _, _ = _video_dependencies()
    baseline_capture = cv2.VideoCapture(str(baseline))
    learned_capture = cv2.VideoCapture(str(learned))
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    writer = None
    frames = 0
    try:
        baseline_meta = _capture_metadata(cv2, baseline_capture, baseline)
        learned_meta = _capture_metadata(cv2, learned_capture, learned)
        if baseline_meta["height"] != learned_meta["height"]:
            raise ValueError("Baseline and learned preview heights do not match")
        if not np.isclose(baseline_meta["fps"], learned_meta["fps"], atol=1e-3):
            raise ValueError("Baseline and learned preview FPS do not match")
        if (
            baseline_meta["frames"] > 0
            and learned_meta["frames"] > 0
            and baseline_meta["frames"] != learned_meta["frames"]
        ):
            raise ValueError("Baseline and learned preview frame counts do not match")

        height = int(baseline_meta["height"])
        baseline_source_width = int(baseline_meta["width"]) - height
        learned_source_width = int(learned_meta["width"]) - height
        if baseline_source_width != learned_source_width or baseline_source_width <= 0:
            raise ValueError("Preview source-panel dimensions do not match")
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(baseline_meta["fps"]),
            (baseline_source_width + 2 * height, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create comparison preview: {temporary}")

        while True:
            baseline_ok, baseline_frame = baseline_capture.read()
            learned_ok, learned_frame = learned_capture.read()
            if not baseline_ok and not learned_ok:
                break
            if baseline_ok != learned_ok:
                raise RuntimeError("Comparison previews ended on different frames")
            source = baseline_frame[:, :baseline_source_width]
            learned_source = learned_frame[:, :learned_source_width]
            source_difference = np.mean(
                np.abs(source.astype(np.int16) - learned_source.astype(np.int16))
            )
            if source_difference > 5.0:
                raise ValueError("Baseline and learned previews contain different source videos")
            baseline_avatar = baseline_frame[:, baseline_source_width:]
            learned_avatar = learned_frame[:, learned_source_width:]
            writer.write(np.concatenate((source, baseline_avatar, learned_avatar), axis=1))
            frames += 1
    except Exception:
        if writer is not None:
            writer.release()
            writer = None
        temporary.unlink(missing_ok=True)
        raise
    finally:
        baseline_capture.release()
        learned_capture.release()
        if writer is not None:
            writer.release()

    if frames < 1:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Comparison previews contain no frames")
    os.replace(temporary, output)
    manifest = {
        **request,
        "request_hash": request_hash,
        "artifact_type": "gravity-mocap-comparison-preview",
        "baseline_preview": str(baseline),
        "learned_preview": str(learned),
        "comparison": str(output),
        "frames": frames,
    }
    write_json_atomic(manifest_path, manifest)
    return {
        "status": "created",
        "frames": frames,
        "comparison": str(output),
        "manifest": str(manifest_path),
    }
