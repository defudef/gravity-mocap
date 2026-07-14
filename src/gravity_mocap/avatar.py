from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .skeleton import SKELETON

AVATAR_RENDER_VERSION = 2

_INDEX = {name: index for index, name in enumerate(SKELETON.names)}
_OUTLINE = (38, 39, 52)
_CORE = (174, 104, 218)
_LEFT = (218, 184, 58)
_RIGHT = (55, 145, 244)
_SKIN = (155, 203, 238)
_LEFT_FOOT = (168, 132, 40)
_RIGHT_FOOT = (38, 104, 194)


@dataclass(frozen=True)
class AvatarPrimitive:
    name: str
    kind: str
    depth: float
    points: np.ndarray
    radii: tuple[float, float]
    color: tuple[int, int, int]


def _build_avatar_primitives(
    pose: np.ndarray,
    projected: np.ndarray,
    scale: float,
    *,
    nearer_positive: bool = True,
) -> list[AvatarPrimitive]:
    pose = np.asarray(pose, dtype=np.float32)
    projected = np.asarray(projected, dtype=np.float32)
    if pose.shape != (SKELETON.joint_count, 3):
        raise ValueError(f"Expected pose shaped ({SKELETON.joint_count}, 3), got {pose.shape}")
    if projected.shape != (SKELETON.joint_count, 2):
        raise ValueError(
            f"Expected projected pose shaped ({SKELETON.joint_count}, 2), got {projected.shape}"
        )
    if not np.isfinite(pose).all() or not np.isfinite(projected).all():
        raise ValueError("Avatar pose must contain only finite values")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Avatar scale must be a positive finite number")

    def segment(
        name: str,
        start: str,
        end: str,
        start_radius: float,
        end_radius: float,
        color: tuple[int, int, int],
    ) -> AvatarPrimitive:
        indices = [_INDEX[start], _INDEX[end]]
        return AvatarPrimitive(
            name=name,
            kind="segment",
            depth=float(pose[indices, 2].mean()),
            points=projected[indices],
            radii=(start_radius * scale, end_radius * scale),
            color=color,
        )

    def ellipse(
        name: str,
        joint: str,
        radius_x: float,
        radius_y: float,
        color: tuple[int, int, int],
    ) -> AvatarPrimitive:
        index = _INDEX[joint]
        return AvatarPrimitive(
            name=name,
            kind="ellipse",
            depth=float(pose[index, 2]),
            points=projected[index : index + 1],
            radii=(radius_x * scale, radius_y * scale),
            color=color,
        )

    torso_indices = np.asarray(
        [
            _INDEX["left_hip"],
            _INDEX["left_shoulder"],
            _INDEX["right_shoulder"],
            _INDEX["right_hip"],
        ]
    )
    primitives = [
        AvatarPrimitive(
            name="torso",
            kind="polygon",
            depth=float(pose[torso_indices, 2].mean()),
            points=projected[torso_indices],
            radii=(0.0, 0.0),
            color=_CORE,
        ),
        segment("pelvis", "left_hip", "right_hip", 0.085, 0.085, _CORE),
        segment("neck", "neck", "head", 0.055, 0.065, _SKIN),
        segment("left_upper_arm", "left_shoulder", "left_elbow", 0.058, 0.050, _LEFT),
        segment("left_forearm", "left_elbow", "left_wrist", 0.049, 0.037, _LEFT),
        segment("right_upper_arm", "right_shoulder", "right_elbow", 0.058, 0.050, _RIGHT),
        segment("right_forearm", "right_elbow", "right_wrist", 0.049, 0.037, _RIGHT),
        segment("left_thigh", "left_hip", "left_knee", 0.086, 0.069, _LEFT),
        segment("left_shin", "left_knee", "left_ankle", 0.066, 0.047, _LEFT),
        segment("right_thigh", "right_hip", "right_knee", 0.086, 0.069, _RIGHT),
        segment("right_shin", "right_knee", "right_ankle", 0.066, 0.047, _RIGHT),
        segment("left_foot", "left_ankle", "left_toe", 0.058, 0.042, _LEFT_FOOT),
        segment("right_foot", "right_ankle", "right_toe", 0.058, 0.042, _RIGHT_FOOT),
        ellipse("left_hand", "left_wrist", 0.050, 0.058, _SKIN),
        ellipse("right_hand", "right_wrist", 0.050, 0.058, _SKIN),
        ellipse("head", "head", 0.105, 0.125, _SKIN),
    ]
    # Draw far-to-near. MediaPipe's transformed world landmarks use positive Z
    # toward the camera, while the learned synthetic-camera contract uses
    # conventional positive Z away from it.
    return sorted(
        primitives,
        key=lambda primitive: primitive.depth,
        reverse=not nearer_positive,
    )


def _shade(
    color: tuple[int, int, int],
    depth: float,
    minimum: float,
    maximum: float,
) -> tuple[int, int, int]:
    amount = (depth - minimum) / max(maximum - minimum, 1e-6)
    brightness = 0.78 + 0.22 * float(np.clip(amount, 0.0, 1.0))
    return tuple(int(np.clip(channel * brightness, 0, 255)) for channel in color)


def _draw_tapered_segment(
    cv2: Any,
    panel: np.ndarray,
    points: np.ndarray,
    radii: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    start, end = points.astype(np.float32)
    direction = end - start
    length = float(np.linalg.norm(direction))
    start_radius = max(2, round(float(radii[0])))
    end_radius = max(2, round(float(radii[1])))
    if length < 1e-4:
        cv2.circle(
            panel,
            tuple(start.round().astype(int)),
            max(start_radius, end_radius),
            color,
            -1,
            cv2.LINE_AA,
        )
        return
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float32) / length
    polygon = (
        np.stack(
            (
                start + perpendicular * start_radius,
                end + perpendicular * end_radius,
                end - perpendicular * end_radius,
                start - perpendicular * start_radius,
            )
        )
        .round()
        .astype(np.int32)
    )
    cv2.fillConvexPoly(panel, polygon, color, cv2.LINE_AA)
    cv2.polylines(panel, [polygon], True, _OUTLINE, 2, cv2.LINE_AA)
    cv2.circle(
        panel,
        tuple(start.round().astype(int)),
        start_radius,
        color,
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(
        panel,
        tuple(end.round().astype(int)),
        end_radius,
        color,
        -1,
        cv2.LINE_AA,
    )


def _background(cv2: Any, width: int, height: int, floor_y: int) -> np.ndarray:
    top = np.asarray([35, 31, 45], dtype=np.float32)
    bottom = np.asarray([16, 17, 24], dtype=np.float32)
    blend = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    panel = np.repeat(top[None, None] * (1.0 - blend) + bottom[None, None] * blend, width, axis=1)
    panel = panel.astype(np.uint8)
    horizon = min(max(round(height * 0.68), 1), floor_y)
    for amount in np.linspace(0.0, 1.0, 5)[1:]:
        y = round(horizon + (floor_y - horizon) * amount**1.7)
        cv2.line(panel, (0, y), (width, y), (55, 49, 67), 1, cv2.LINE_AA)
    vanishing_x = width // 2
    for x in np.linspace(-width, width * 2, 9):
        cv2.line(
            panel,
            (vanishing_x, horizon),
            (round(float(x)), height),
            (48, 43, 58),
            1,
            cv2.LINE_AA,
        )
    return panel


def render_avatar_panel(
    cv2: Any,
    pose: np.ndarray,
    *,
    width: int,
    height: int,
    scale: float,
    center: np.ndarray,
    nearer_positive: bool = True,
    title: str = "Gravity Mocap - procedural 3D avatar",
) -> np.ndarray:
    """Render a depth-sorted procedural mannequin from the neutral 22-joint pose."""
    projected = np.empty((SKELETON.joint_count, 2), dtype=np.float32)
    projected[:, 0] = float(center[0]) + pose[:, 0] * float(scale)
    projected[:, 1] = float(center[1]) - pose[:, 1] * float(scale)
    floor_joints = [_INDEX[name] for name in ("left_ankle", "right_ankle", "left_toe", "right_toe")]
    floor_y = int(np.clip(np.max(projected[floor_joints, 1]) + 8, height * 0.72, height - 8))
    panel = _background(cv2, width, height, floor_y)

    shadow = panel.copy()
    root_x = round(float(projected[_INDEX["root"], 0]))
    cv2.ellipse(
        shadow,
        (root_x, floor_y),
        (max(18, round(scale * 0.22)), max(5, round(scale * 0.035))),
        0,
        0,
        360,
        (5, 6, 10),
        -1,
        cv2.LINE_AA,
    )
    panel = cv2.addWeighted(shadow, 0.62, panel, 0.38, 0)

    primitives = _build_avatar_primitives(
        pose,
        projected,
        scale,
        nearer_positive=nearer_positive,
    )
    minimum_depth = min(primitive.depth for primitive in primitives)
    maximum_depth = max(primitive.depth for primitive in primitives)
    for primitive in primitives:
        color = _shade(primitive.color, primitive.depth, minimum_depth, maximum_depth)
        if primitive.kind == "segment":
            _draw_tapered_segment(cv2, panel, primitive.points, primitive.radii, color)
        elif primitive.kind == "polygon":
            polygon = primitive.points.round().astype(np.int32)
            cv2.fillConvexPoly(panel, polygon, color, cv2.LINE_AA)
            cv2.polylines(panel, [polygon], True, _OUTLINE, 2, cv2.LINE_AA)
        elif primitive.kind == "ellipse":
            center_xy = tuple(primitive.points[0].round().astype(int))
            radii = tuple(max(2, round(float(radius))) for radius in primitive.radii)
            cv2.ellipse(panel, center_xy, radii, 0, 0, 360, color, -1, cv2.LINE_AA)
            cv2.ellipse(panel, center_xy, radii, 0, 0, 360, _OUTLINE, 2, cv2.LINE_AA)
        else:
            raise RuntimeError(f"Unknown avatar primitive kind: {primitive.kind}")

    cv2.putText(
        panel,
        title,
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (238, 238, 244),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(panel, (24, 57), 6, _LEFT, -1, cv2.LINE_AA)
    cv2.putText(
        panel,
        "left",
        (36, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (212, 212, 222),
        1,
        cv2.LINE_AA,
    )
    cv2.circle(panel, (88, 57), 6, _RIGHT, -1, cv2.LINE_AA)
    cv2.putText(
        panel,
        "right",
        (100, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (212, 212, 222),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "neutral 22-joint rig - no SMPL",
        (20, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (168, 168, 180),
        1,
        cv2.LINE_AA,
    )
    return panel
