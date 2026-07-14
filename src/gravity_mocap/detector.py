from __future__ import annotations

import numpy as np

from .skeleton import SKELETON


def normalize_detector_inputs(
    keypoints_xyc: np.ndarray,
    bbox_xyxy: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize one detector frame to the model's public 22-joint contract.

    Keypoint coordinates become bbox-relative ``[-1, 1]`` values. The returned
    bbox is frame-relative ``[-1, 1]`` in ``[x1, y1, x2, y2]`` order.
    """
    keypoints = np.asarray(keypoints_xyc, dtype=np.float32)
    bbox = np.asarray(bbox_xyxy, dtype=np.float32)
    if keypoints.shape != (SKELETON.joint_count, 3):
        raise ValueError(
            f"Expected detector keypoints shaped ({SKELETON.joint_count}, 3), got {keypoints.shape}"
        )
    if bbox.shape != (4,):
        raise ValueError(f"Expected bbox shaped (4,), got {bbox.shape}")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if not np.isfinite(keypoints).all() or not np.isfinite(bbox).all():
        raise ValueError("detector inputs must be finite")
    size = bbox[2:] - bbox[:2]
    if np.any(size <= 0):
        raise ValueError("bbox must have positive width and height")

    normalized_keypoints = keypoints.copy()
    normalized_keypoints[:, :2] = 2.0 * (keypoints[:, :2] - bbox[:2]) / size - 1.0
    normalized_keypoints[:, :2] = np.clip(normalized_keypoints[:, :2], -1.0, 1.0)
    normalized_keypoints[:, 2] = np.clip(keypoints[:, 2], 0.0, 1.0)
    missing = normalized_keypoints[:, 2] <= 0
    normalized_keypoints[missing, :2] = 0.0

    frame_scale = np.asarray(
        [frame_width, frame_height, frame_width, frame_height], dtype=np.float32
    )
    normalized_bbox = 2.0 * bbox / frame_scale - 1.0
    return normalized_keypoints.astype(np.float32), normalized_bbox.astype(np.float32)
