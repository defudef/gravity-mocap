from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_json_atomic
from .detector import normalize_detector_inputs
from .pose import (
    fill_short_bbox_gaps,
    mediapipe_to_canonical,
    mediapipe_world_to_canonical,
    padded_bbox_from_landmarks,
)
from .rig2d import (
    RIG_2D_FILENAME,
    RIG_2D_MANIFEST_FILENAME,
    RIG_2D_PREVIEW_FILENAME,
    RIG_2D_VERSION,
    Rig2D,
    load_rig_2d,
    write_rig_2d,
)
from .rotations import identity_rotation_6d
from .schema import stable_hash
from .skeleton import SKELETON
from .world3d import (
    DETECTOR_WORLD_3D_FILENAME,
    DETECTOR_WORLD_3D_VERSION,
    DetectorWorld3D,
    load_detector_world_3d,
    write_detector_world_3d,
)

POSE_BACKEND = "mediapipe-pose-landmarker-heavy"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
POSE_MODEL_FILENAME = "pose_landmarker_heavy.task"
POSE_MODEL_SIZE = 30_664_242
POSE_MODEL_SHA256 = "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b"
TEMPORAL_RESAMPLING_VERSION = 2


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_pose_model(model_dir: Path) -> Path:
    """Download the exact audited MediaPipe model bundle, then verify it."""
    model_dir.mkdir(parents=True, exist_ok=True)
    output = model_dir / POSE_MODEL_FILENAME
    if output.is_file() and output.stat().st_size == POSE_MODEL_SIZE:
        if file_sha256(output) == POSE_MODEL_SHA256:
            return output
        output.unlink()

    temporary = output.with_suffix(output.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        print(f"Downloading audited pose model ({POSE_MODEL_SIZE / 1024 / 1024:.1f} MiB)...")
        with urllib.request.urlopen(POSE_MODEL_URL, timeout=60) as response:  # noqa: S310
            with temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        if temporary.stat().st_size != POSE_MODEL_SIZE:
            raise RuntimeError(
                f"Pose model has {temporary.stat().st_size} bytes; expected {POSE_MODEL_SIZE}"
            )
        digest = file_sha256(temporary)
        if digest != POSE_MODEL_SHA256:
            raise RuntimeError(f"Pose model SHA-256 mismatch: {digest}")
        os.replace(temporary, output)
        print(f"Pose model ready: {output}")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output


def selected_frame_indices(
    source_frame_count: int,
    source_fps: float,
    target_fps: float,
    *,
    max_frames: int | None = None,
) -> np.ndarray:
    """Return deterministic source indices sampled on a target-FPS timeline."""
    if source_frame_count < 1:
        raise ValueError("Video contains no frames")
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Invalid source FPS: {source_fps}")
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError(f"Invalid target FPS: {target_fps}")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")

    duration = (source_frame_count - 1) / float(source_fps)
    output_count = max(1, int(math.floor(duration * target_fps)) + 1)
    if max_frames is not None:
        output_count = min(output_count, max_frames)
    timeline = np.arange(output_count, dtype=np.float64) / target_fps
    return np.clip(np.rint(timeline * source_fps), 0, source_frame_count - 1).astype(np.int64)


def resample_detector_values(
    values: np.ndarray,
    confidence: np.ndarray,
    *,
    source_fps: float,
    target_fps: float,
    output_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence-weighted temporal interpolation of native detector outputs."""
    source = np.asarray(values, dtype=np.float32)
    weights = np.asarray(confidence, dtype=np.float32)
    if source.ndim < 2 or source.shape[:-1] != weights.shape:
        raise ValueError(
            f"Detector values/confidence shapes do not match: {source.shape}, {weights.shape}"
        )
    if len(source) < 1 or output_frames < 1:
        raise ValueError("Detector resampling requires input and output frames")
    if not np.isfinite(source).all() or not np.isfinite(weights).all():
        raise ValueError("Detector resampling inputs must be finite")
    if np.any((weights < 0) | (weights > 1)):
        raise ValueError("Detector confidence must stay within [0, 1]")
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be positive")
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be positive")

    positions = np.arange(output_frames, dtype=np.float64) * float(source_fps) / float(target_fps)
    positions = np.clip(positions, 0.0, len(source) - 1)
    left = np.floor(positions).astype(np.int64)
    right = np.ceil(positions).astype(np.int64)
    amount = (positions - left).astype(np.float32)
    amount_shape = (output_frames,) + (1,) * (weights.ndim - 1)
    amount = amount.reshape(amount_shape)

    left_weight = weights[left] * (1.0 - amount)
    right_weight = weights[right] * amount
    output_confidence = left_weight + right_weight
    numerator = source[left] * left_weight[..., None] + source[right] * right_weight[..., None]
    output = np.divide(
        numerator,
        output_confidence[..., None],
        out=np.zeros_like(numerator),
        where=output_confidence[..., None] > 1e-8,
    )
    return output.astype(np.float32), output_confidence.astype(np.float32)


def resample_detector_boxes(
    boxes: np.ndarray,
    *,
    source_fps: float,
    target_fps: float,
    output_frames: int,
) -> np.ndarray:
    """Linearly interpolate already gap-validated detector boxes."""
    source = np.asarray(boxes, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 4 or len(source) < 1:
        raise ValueError(f"Expected detector boxes shaped (T, 4), got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("Detector boxes must be finite before resampling")
    positions = np.arange(output_frames, dtype=np.float64) * float(source_fps) / float(target_fps)
    positions = np.clip(positions, 0.0, len(source) - 1)
    left = np.floor(positions).astype(np.int64)
    right = np.ceil(positions).astype(np.int64)
    amount = (positions - left).astype(np.float32)[:, None]
    return (source[left] * (1.0 - amount) + source[right] * amount).astype(np.float32)


def _video_dependencies() -> tuple[Any, Any, Any]:
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import vision
    except ImportError as error:
        raise RuntimeError(
            "Video extras are missing. Run ./scripts/setup-video.sh first."
        ) from error
    return cv2, mp, vision


def _video_metadata(video: Path) -> tuple[int, int, int, float]:
    cv2, _, _ = _video_dependencies()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or frame_count <= 0 or not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(
            f"Invalid video metadata: {width}x{height}, {frame_count} frame(s), {fps} FPS"
        )
    return width, height, frame_count, fps


def _sampled_frames(video: Path, indices: np.ndarray) -> Iterator[tuple[int, int, np.ndarray]]:
    cv2, _, _ = _video_dependencies()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    wanted = 0
    source_index = 0
    last_frame: np.ndarray | None = None
    try:
        while wanted < len(indices):
            target_index = int(indices[wanted])
            if last_frame is not None and target_index == source_index - 1:
                yield wanted, target_index, last_frame.copy()
                wanted += 1
                continue
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended before source frame {target_index}: {video}")
            last_frame = frame
            if source_index == target_index:
                yield wanted, target_index, frame
                wanted += 1
            source_index += 1
    finally:
        capture.release()


def default_video_output_directory(video: Path, data_root: Path) -> Path:
    source_digest = file_sha256(video)
    return data_root / "inference" / f"{video.stem}-{source_digest[:12]}"


def default_inference_directory(video: Path, data_root: Path) -> Path:
    """Backward-compatible name for the content-addressed video output directory."""
    return default_video_output_directory(video, data_root)


def _render_pose_preview(
    video: Path,
    indices: np.ndarray,
    keypoints: np.ndarray,
    bboxes: np.ndarray,
    output: Path,
    fps: float,
) -> None:
    cv2, _, _ = _video_dependencies()
    width, height, _, _ = _video_metadata(video)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video preview: {temporary}")
    try:
        for output_index, _, frame in _sampled_frames(video, indices):
            bbox = bboxes[output_index].round().astype(int)
            cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), (50, 220, 255), 2)
            points = keypoints[output_index]
            for joint, parent in enumerate(SKELETON.parents):
                if parent < 0 or points[joint, 2] <= 0 or points[parent, 2] <= 0:
                    continue
                first = tuple(points[int(parent), :2].round().astype(int))
                second = tuple(points[joint, :2].round().astype(int))
                cv2.line(frame, first, second, (100, 255, 100), 2, cv2.LINE_AA)
            for x, y, confidence in points:
                if confidence > 0:
                    cv2.circle(frame, (round(float(x)), round(float(y))), 3, (30, 80, 255), -1)
            writer.write(frame)
    finally:
        writer.release()
    os.replace(temporary, output)


def video_to_rig(
    video: Path,
    output_dir: Path,
    *,
    data_root: Path,
    target_fps: float = 30.0,
    confidence_threshold: float = 0.2,
    bbox_padding: float = 0.12,
    max_missing_frames: int = 30,
    max_frames: int | None = None,
    force: bool = False,
    preview: bool = True,
) -> dict[str, Any]:
    """Run MediaPipe Heavy and save a standalone neutral 2D rig artifact."""
    video = video.expanduser().resolve()
    if not video.is_file():
        raise ValueError(f"Video does not exist: {video}")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if bbox_padding < 0:
        raise ValueError("bbox_padding must be non-negative")
    if max_missing_frames < 0:
        raise ValueError("max_missing_frames must be non-negative")

    model = ensure_pose_model(data_root / "models" / "mediapipe")
    width, height, source_frame_count, source_fps = _video_metadata(video)
    indices = selected_frame_indices(
        source_frame_count, source_fps, target_fps, max_frames=max_frames
    )
    source_digest = file_sha256(video)
    request = {
        "rig_2d_version": RIG_2D_VERSION,
        "detector_world_3d_version": DETECTOR_WORLD_3D_VERSION,
        "temporal_resampling_version": TEMPORAL_RESAMPLING_VERSION,
        "source_sha256": source_digest,
        "backend": POSE_BACKEND,
        "model_sha256": POSE_MODEL_SHA256,
        "target_fps": float(target_fps),
        "confidence_threshold": float(confidence_threshold),
        "bbox_padding": float(bbox_padding),
        "max_missing_frames": int(max_missing_frames),
        "max_frames": max_frames,
    }
    request_hash = stable_hash(request)
    output_dir = output_dir.expanduser().resolve()
    rig_path = output_dir / RIG_2D_FILENAME
    world_3d_path = output_dir / DETECTOR_WORLD_3D_FILENAME
    manifest_path = output_dir / RIG_2D_MANIFEST_FILENAME
    preview_path = output_dir / RIG_2D_PREVIEW_FILENAME
    if not force and rig_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("request_hash") == request_hash
                and (not preview or preview_path.is_file())
                and world_3d_path.is_file()
            ):
                cached_rig = load_rig_2d(rig_path)
                cached_world_3d = load_detector_world_3d(world_3d_path)
                return {
                    "status": "cached",
                    "frames": cached_rig.frames,
                    "detector_world_3d_frames": int(cached_world_3d.frame_mask.sum()),
                    "rig_2d": str(rig_path),
                    "detector_inputs": str(rig_path),
                    "detector_world_3d": str(world_3d_path),
                    "manifest": str(manifest_path),
                    "preview": str(preview_path) if preview_path.is_file() else None,
                    "request_hash": manifest.get("request_hash"),
                }
        except (json.JSONDecodeError, KeyError, OSError, ValueError):
            pass

    cv2, mp, vision = _video_dependencies()
    del cv2
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    last_target_time = (len(indices) - 1) / float(target_fps)
    last_native_index = min(
        source_frame_count - 1,
        int(math.ceil(last_target_time * source_fps)),
    )
    native_indices = np.arange(last_native_index + 1, dtype=np.int64)
    native_keypoints = np.zeros((len(native_indices), SKELETON.joint_count, 3), dtype=np.float32)
    native_world_joints = np.zeros((len(native_indices), SKELETON.joint_count, 3), dtype=np.float32)
    native_world_confidence = np.zeros(
        (len(native_indices), SKELETON.joint_count), dtype=np.float32
    )
    native_raw_bboxes = np.full((len(native_indices), 4), np.nan, dtype=np.float32)
    native_detected_frames = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for native_index, source_index, bgr in _sampled_frames(video, native_indices):
            rgb = bgr[:, :, ::-1].copy()
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                round(1000.0 * source_index / source_fps),
            )
            if not result.pose_landmarks:
                continue
            native_detected_frames += 1
            landmarks = np.asarray(
                [
                    [
                        landmark.x * width,
                        landmark.y * height,
                        min(float(landmark.visibility), float(landmark.presence)),
                    ]
                    for landmark in result.pose_landmarks[0]
                ],
                dtype=np.float32,
            )
            native_keypoints[native_index] = mediapipe_to_canonical(
                landmarks, confidence_threshold=confidence_threshold
            )
            if result.pose_world_landmarks:
                world_landmarks = np.asarray(
                    [
                        [
                            landmark.x,
                            landmark.y,
                            landmark.z,
                            min(float(landmark.visibility), float(landmark.presence)),
                        ]
                        for landmark in result.pose_world_landmarks[0]
                    ],
                    dtype=np.float32,
                )
                (
                    native_world_joints[native_index],
                    native_world_confidence[native_index],
                ) = mediapipe_world_to_canonical(world_landmarks)
            bbox = padded_bbox_from_landmarks(
                landmarks,
                frame_width=width,
                frame_height=height,
                confidence_threshold=confidence_threshold,
                padding=bbox_padding,
            )
            if bbox is not None:
                native_raw_bboxes[native_index] = bbox

    native_gap_limit = int(math.ceil(max_missing_frames * source_fps / target_fps))
    native_bboxes = fill_short_bbox_gaps(native_raw_bboxes, max_gap=native_gap_limit)
    keypoint_xy, keypoint_confidence = resample_detector_values(
        native_keypoints[..., :2],
        native_keypoints[..., 2],
        source_fps=source_fps,
        target_fps=target_fps,
        output_frames=len(indices),
    )
    keypoints = np.concatenate((keypoint_xy, keypoint_confidence[..., None]), axis=-1)
    world_joints, world_confidence = resample_detector_values(
        native_world_joints,
        native_world_confidence,
        source_fps=source_fps,
        target_fps=target_fps,
        output_frames=len(indices),
    )
    world_frame_mask = np.any(world_confidence > 0, axis=-1).astype(np.float32)
    world_joints[world_frame_mask <= 0] = 0.0
    world_confidence[world_frame_mask <= 0] = 0.0
    bboxes = resample_detector_boxes(
        native_bboxes,
        source_fps=source_fps,
        target_fps=target_fps,
        output_frames=len(indices),
    )
    normalized_keypoints = np.empty_like(keypoints)
    normalized_bboxes = np.empty_like(bboxes)
    for index, (points, bbox) in enumerate(zip(keypoints, bboxes, strict=True)):
        normalized_keypoints[index], normalized_bboxes[index] = normalize_detector_inputs(
            points, bbox, frame_width=width, frame_height=height
        )
    detected_frames = int(np.any(keypoint_confidence > 0, axis=-1).sum())
    provenance = {
        **request,
        "artifact_type": "gravity-mocap-rig-2d",
        "request_hash": request_hash,
        "source_path": str(video),
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "output_frames": len(indices),
        "detected_frames": detected_frames,
        "native_detector_frames": len(native_indices),
        "native_detected_frames": native_detected_frames,
        "temporal_resampling": "confidence-weighted-linear-after-native-video-tracking",
        "frame_width": width,
        "frame_height": height,
        "joint_names": list(SKELETON.names),
        "model_url": POSE_MODEL_URL,
        "model_size": POSE_MODEL_SIZE,
        "license": "Apache-2.0",
    }
    write_rig_2d(
        rig_path,
        Rig2D(
            keypoints_2d=normalized_keypoints,
            bbox=normalized_bboxes,
            camera_delta_6d=identity_rotation_6d(len(indices)).cpu().numpy(),
            image_mask=np.zeros(len(indices), dtype=np.float32),
            frame_mask=np.ones(len(indices), dtype=np.float32),
            pixel_keypoints=keypoints,
            pixel_bbox=bboxes,
            source_frame_indices=indices,
            fps=target_fps,
            source_fps=source_fps,
            frame_width=width,
            frame_height=height,
            provenance=provenance,
        ),
    )
    world_3d_provenance = {
        **request,
        "artifact_type": "gravity-mocap-detector-world-3d",
        "request_hash": request_hash,
        "source_path": str(video),
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "output_frames": len(indices),
        "detected_frames": int(world_frame_mask.sum()),
        "native_detector_frames": len(native_indices),
        "native_detected_frames": int(np.any(native_world_confidence > 0, axis=-1).sum()),
        "temporal_resampling": "confidence-weighted-linear-after-native-video-tracking",
        "coordinate_transform": ["x", "-y", "-z"],
        "coordinate_units": "metres",
        "joint_names": list(SKELETON.names),
        "model_url": POSE_MODEL_URL,
        "model_size": POSE_MODEL_SIZE,
        "license": "Apache-2.0",
    }
    write_detector_world_3d(
        world_3d_path,
        DetectorWorld3D(
            joints_3d=world_joints,
            confidence=world_confidence,
            frame_mask=world_frame_mask,
            source_frame_indices=indices,
            fps=target_fps,
            source_fps=source_fps,
            provenance=world_3d_provenance,
        ),
    )
    if preview:
        output_dir.mkdir(parents=True, exist_ok=True)
        _render_pose_preview(video, indices, keypoints, bboxes, preview_path, target_fps)
    manifest = {
        **provenance,
        "frames": len(indices),
        "rig_2d": str(rig_path),
        "detector_inputs": str(rig_path),
        "detector_world_3d": str(world_3d_path),
        "preview": str(preview_path) if preview else None,
        "request_hash": request_hash,
    }
    write_json_atomic(manifest_path, manifest)
    return {
        "status": "created",
        "frames": len(indices),
        "detected_frames": detected_frames,
        "detector_world_3d_frames": int(world_frame_mask.sum()),
        "rig_2d": str(rig_path),
        "detector_inputs": str(rig_path),
        "detector_world_3d": str(world_3d_path),
        "manifest": str(manifest_path),
        "preview": str(preview_path) if preview else None,
        "request_hash": request_hash,
    }


def detect_video(
    video: Path,
    output_dir: Path,
    *,
    data_root: Path,
    target_fps: float = 30.0,
    confidence_threshold: float = 0.2,
    bbox_padding: float = 0.12,
    max_missing_frames: int = 30,
    max_frames: int | None = None,
    force: bool = False,
    preview: bool = True,
) -> dict[str, Any]:
    """Backward-compatible alias for :func:`video_to_rig`."""
    return video_to_rig(
        video,
        output_dir,
        data_root=data_root,
        target_fps=target_fps,
        confidence_threshold=confidence_threshold,
        bbox_padding=bbox_padding,
        max_missing_frames=max_missing_frames,
        max_frames=max_frames,
        force=force,
        preview=preview,
    )
