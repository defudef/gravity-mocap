from __future__ import annotations

import numpy as np

from .skeleton import SKELETON

MEDIAPIPE_LANDMARK_COUNT = 33


def _canonical_landmark_values(points: np.ndarray) -> dict[str, np.ndarray]:
    """Build the neutral 22-joint topology from MediaPipe's 33 landmarks."""
    left_shoulder = points[11]
    right_shoulder = points[12]
    left_hip = points[23]
    right_hip = points[24]
    root = 0.5 * (left_hip + right_hip)
    neck = 0.5 * (left_shoulder + right_shoulder)
    head = 0.5 * (points[7] + points[8])
    return {
        "root": root,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "spine_1": root + (neck - root) * 0.25,
        "left_knee": points[25],
        "right_knee": points[26],
        "spine_2": root + (neck - root) * 0.50,
        "left_ankle": points[27],
        "right_ankle": points[28],
        "spine_3": root + (neck - root) * 0.75,
        "left_toe": points[31],
        "right_toe": points[32],
        "neck": neck,
        "left_clavicle": neck + (left_shoulder - neck) * 0.30,
        "right_clavicle": neck + (right_shoulder - neck) * 0.30,
        "head": head,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_elbow": points[13],
        "right_elbow": points[14],
        "left_wrist": points[15],
        "right_wrist": points[16],
    }


def _present(point: np.ndarray, threshold: float) -> bool:
    return bool(np.isfinite(point).all() and point[2] >= threshold)


def _direct(points: np.ndarray, index: int, threshold: float) -> np.ndarray:
    point = np.asarray(points[index], dtype=np.float32).copy()
    if not _present(point, threshold):
        return np.zeros(3, dtype=np.float32)
    point[2] = np.clip(point[2], 0.0, 1.0)
    return point


def _lerp(first: np.ndarray, second: np.ndarray, amount: float) -> np.ndarray:
    if first[2] <= 0 or second[2] <= 0:
        return np.zeros(3, dtype=np.float32)
    result = np.empty(3, dtype=np.float32)
    result[:2] = first[:2] + (second[:2] - first[:2]) * float(amount)
    result[2] = min(float(first[2]), float(second[2]))
    return result


def _midpoint(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return _lerp(first, second, 0.5)


def mediapipe_to_canonical(
    landmarks_xyc: np.ndarray,
    *,
    confidence_threshold: float = 0.2,
) -> np.ndarray:
    """Map MediaPipe's 33 pixel-space landmarks to the canonical 22 joints."""
    points = np.asarray(landmarks_xyc, dtype=np.float32)
    if points.shape != (MEDIAPIPE_LANDMARK_COUNT, 3):
        raise ValueError(
            f"Expected MediaPipe landmarks shaped ({MEDIAPIPE_LANDMARK_COUNT}, 3), "
            f"got {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError("MediaPipe landmarks must be finite")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")

    left_shoulder = _direct(points, 11, confidence_threshold)
    right_shoulder = _direct(points, 12, confidence_threshold)
    left_hip = _direct(points, 23, confidence_threshold)
    right_hip = _direct(points, 24, confidence_threshold)
    root = _midpoint(left_hip, right_hip)
    neck = _midpoint(left_shoulder, right_shoulder)

    left_ear = _direct(points, 7, confidence_threshold)
    right_ear = _direct(points, 8, confidence_threshold)
    left_eye = _direct(points, 2, confidence_threshold)
    right_eye = _direct(points, 5, confidence_threshold)
    nose = _direct(points, 0, confidence_threshold)
    head = _midpoint(left_ear, right_ear)
    if head[2] <= 0:
        head = _midpoint(left_eye, right_eye)
    if head[2] <= 0:
        head = nose

    result = np.zeros((SKELETON.joint_count, 3), dtype=np.float32)
    values = {
        "root": root,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "spine_1": _lerp(root, neck, 0.25),
        "left_knee": _direct(points, 25, confidence_threshold),
        "right_knee": _direct(points, 26, confidence_threshold),
        "spine_2": _lerp(root, neck, 0.50),
        "left_ankle": _direct(points, 27, confidence_threshold),
        "right_ankle": _direct(points, 28, confidence_threshold),
        "spine_3": _lerp(root, neck, 0.75),
        "left_toe": _direct(points, 31, confidence_threshold),
        "right_toe": _direct(points, 32, confidence_threshold),
        "neck": neck,
        "left_clavicle": _lerp(neck, left_shoulder, 0.30),
        "right_clavicle": _lerp(neck, right_shoulder, 0.30),
        "head": head,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_elbow": _direct(points, 13, confidence_threshold),
        "right_elbow": _direct(points, 14, confidence_threshold),
        "left_wrist": _direct(points, 15, confidence_threshold),
        "right_wrist": _direct(points, 16, confidence_threshold),
    }
    for index, name in enumerate(SKELETON.names):
        result[index] = values[name]
    result[result[:, 2] <= 0] = 0.0
    return result


def mediapipe_world_to_canonical(
    landmarks_xyzc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map MediaPipe world landmarks to root-relative, gravity-up neutral joints.

    MediaPipe exposes metric ``x, y, z`` values plus visibility/presence.  The
    detector coordinate transform ``[x, -y, -z]`` keeps the basis right-handed
    while making gravity point along positive Y.  Confidence is preserved as a
    separate array instead of deleting useful detector estimates for occluded
    limbs.
    """
    points = np.asarray(landmarks_xyzc, dtype=np.float32)
    if points.shape != (MEDIAPIPE_LANDMARK_COUNT, 4):
        raise ValueError(
            f"Expected MediaPipe world landmarks shaped ({MEDIAPIPE_LANDMARK_COUNT}, 4), "
            f"got {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError("MediaPipe world landmarks must be finite")
    if np.any((points[:, 3] < 0) | (points[:, 3] > 1)):
        raise ValueError("MediaPipe world landmark confidence must be in [0, 1]")

    values = _canonical_landmark_values(points)
    mapped = np.stack([values[name] for name in SKELETON.names]).astype(np.float32)
    coordinates = mapped[:, :3]
    coordinates -= coordinates[:1]
    coordinates[:, 1:] *= -1.0
    confidence = np.clip(mapped[:, 3], 0.0, 1.0)
    return coordinates, confidence


def padded_bbox_from_landmarks(
    landmarks_xyc: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    confidence_threshold: float = 0.2,
    padding: float = 0.12,
) -> np.ndarray | None:
    """Create a clipped pixel-space bbox from visible landmarks."""
    points = np.asarray(landmarks_xyc, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected keypoints shaped (J, 3), got {points.shape}")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    visible = points[:, 2] >= confidence_threshold
    if visible.sum() < 4:
        return None
    minimum = points[visible, :2].min(axis=0)
    maximum = points[visible, :2].max(axis=0)
    size = maximum - minimum
    if np.any(size <= 1.0):
        return None
    margin = size * float(padding)
    minimum = np.maximum(minimum - margin, [0.0, 0.0])
    maximum = np.minimum(maximum + margin, [float(frame_width - 1), float(frame_height - 1)])
    if np.any(maximum - minimum <= 1.0):
        return None
    return np.concatenate((minimum, maximum)).astype(np.float32)


def fill_short_bbox_gaps(bboxes: np.ndarray, *, max_gap: int) -> np.ndarray:
    """Linearly fill short missing bbox runs; reject long or fully missing tracks."""
    values = np.asarray(bboxes, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Expected bboxes shaped (T, 4), got {values.shape}")
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    valid = np.isfinite(values).all(axis=1)
    if not valid.any():
        raise ValueError("No pose was detected in the video")

    index = 0
    while index < len(values):
        if valid[index]:
            index += 1
            continue
        start = index
        while index < len(values) and not valid[index]:
            index += 1
        end = index
        gap = end - start
        if gap > max_gap:
            raise ValueError(
                f"Pose detector missed {gap} consecutive frame(s); maximum allowed is {max_gap}"
            )
        previous = start - 1 if start > 0 else None
        following = end if end < len(values) else None
        if previous is None and following is None:
            raise ValueError("No bbox is available around the missing detection run")
        if previous is None:
            values[start:end] = values[following]
        elif following is None:
            values[start:end] = values[previous]
        else:
            for offset, frame in enumerate(range(start, end), start=1):
                amount = offset / float(gap + 1)
                values[frame] = values[previous] + (values[following] - values[previous]) * amount
    return values
