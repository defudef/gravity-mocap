from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .adapters import FORBIDDEN_PARAMETRIC_BODY_KEYS
from .artifacts import write_json_atomic
from .inference import _render_motion_preview
from .rig2d import load_rig_2d
from .skeleton import SKELETON
from .video import file_sha256
from .world3d import load_detector_world_3d

MOTION_DIAGNOSTICS_VERSION = 1


def _load_motion(path: Path, *, learned: bool) -> dict[str, np.ndarray | float]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Motion artifact does not exist: {resolved}")
    required = {"joints_3d", "contacts", "fps", "joint_names"}
    if learned:
        required.update(("joints_world", "root_translation"))
    with np.load(resolved, allow_pickle=False) as archive:
        forbidden = sorted(
            name
            for name in archive.files
            if any(token in name.lower() for token in FORBIDDEN_PARAMETRIC_BODY_KEYS)
        )
        if forbidden:
            raise ValueError(f"Motion contains prohibited body-model fields: {forbidden}")
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Motion artifact is missing: {', '.join(missing)}")
        result: dict[str, np.ndarray | float] = {
            name: np.asarray(archive[name]) for name in required if name not in {"fps"}
        }
        for name in ("root_translation_raw", "root_velocity_local_raw", "joints_world_raw"):
            if name in archive:
                result[name] = np.asarray(archive[name])
        result["fps"] = float(np.asarray(archive["fps"]))
    names = tuple(str(name) for name in np.asarray(result.pop("joint_names")).tolist())
    if names != SKELETON.names:
        raise ValueError("Motion joint_names do not match the canonical 22-joint order")
    joints = np.asarray(result["joints_3d"])
    if joints.ndim != 3 or joints.shape[1:] != (SKELETON.joint_count, 3):
        raise ValueError(f"Motion joints_3d has invalid shape: {joints.shape}")
    frames = len(joints)
    contacts = np.asarray(result["contacts"])
    if contacts.shape != (frames, len(SKELETON.contact_joints)):
        raise ValueError(f"Motion contacts has invalid shape: {contacts.shape}")
    if np.any((contacts < 0) | (contacts > 1)):
        raise ValueError("Motion contacts must contain probabilities in [0, 1]")
    for name, value in result.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise ValueError(f"Motion {name} contains NaN or infinity")
    fps = float(result["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Motion FPS must be positive")
    if learned:
        if np.asarray(result["joints_world"]).shape != joints.shape:
            raise ValueError("Motion joints_world shape does not match joints_3d")
        if np.asarray(result["root_translation"]).shape != (frames, 3):
            raise ValueError("Motion root_translation must be shaped (T, 3)")
        if not np.allclose(
            result["joints_world"],
            joints + np.asarray(result["root_translation"])[:, None],
            atol=1e-5,
        ):
            raise ValueError("Motion joints_world is inconsistent with root translation")
        if "root_translation_raw" in result and np.asarray(
            result["root_translation_raw"]
        ).shape != (frames, 3):
            raise ValueError("Motion root_translation_raw must be shaped (T, 3)")
        if "root_velocity_local_raw" in result and np.asarray(
            result["root_velocity_local_raw"]
        ).shape != (frames, 3):
            raise ValueError("Motion root_velocity_local_raw must be shaped (T, 3)")
        if "joints_world_raw" in result and np.asarray(result["joints_world_raw"]).shape != (
            frames,
            SKELETON.joint_count,
            3,
        ):
            raise ValueError("Motion joints_world_raw shape does not match joints_3d")
        if "joints_world_raw" in result and "root_translation_raw" in result:
            if not np.allclose(
                result["joints_world_raw"],
                joints + np.asarray(result["root_translation_raw"])[:, None],
                atol=1e-5,
            ):
                raise ValueError("Motion raw world joints are inconsistent with raw root")
    return result


def _temporal_stats(joints: np.ndarray, fps: float) -> dict[str, float]:
    velocity = np.linalg.norm(np.diff(joints, axis=0) * fps, axis=-1)
    acceleration = np.linalg.norm(np.diff(joints, n=2, axis=0) * fps**2, axis=-1)
    jerk = np.linalg.norm(np.diff(joints, n=3, axis=0) * fps**3, axis=-1)

    def stats(value: np.ndarray) -> tuple[float, float]:
        if value.size == 0:
            return 0.0, 0.0
        return float(value.mean()), float(np.percentile(value, 95))

    velocity_mean, velocity_p95 = stats(velocity)
    acceleration_mean, acceleration_p95 = stats(acceleration)
    jerk_mean, jerk_p95 = stats(jerk)
    return {
        "velocity_mean_mps": velocity_mean,
        "velocity_p95_mps": velocity_p95,
        "acceleration_mean_mps2": acceleration_mean,
        "acceleration_p95_mps2": acceleration_p95,
        "jerk_mean_mps3": jerk_mean,
        "jerk_p95_mps3": jerk_p95,
    }


def _root_stats(root: np.ndarray, fps: float) -> dict[str, float]:
    velocity = np.diff(root, axis=0) * fps
    speed = np.linalg.norm(velocity, axis=-1)
    return {
        "path_length_m": float(np.linalg.norm(np.diff(root, axis=0), axis=-1).sum()),
        "net_displacement_m": float(np.linalg.norm(root[-1] - root[0])),
        "max_displacement_from_start_m": float(np.linalg.norm(root - root[0], axis=-1).max()),
        "speed_mean_mps": float(speed.mean()) if len(speed) else 0.0,
        "speed_p95_mps": float(np.percentile(speed, 95)) if len(speed) else 0.0,
        "speed_max_mps": float(speed.max()) if len(speed) else 0.0,
    }


def diagnose_motion(
    motion_path: Path,
    rig_2d_path: Path,
    output_path: Path,
    *,
    baseline_motion_path: Path | None = None,
    detector_world_3d_path: Path | None = None,
    source_video: Path | None = None,
    world_preview: bool = False,
    avatar_renderer: str = "mesh",
) -> dict[str, Any]:
    """Write real-video diagnostics without claiming unavailable ground truth."""
    learned_path = motion_path.expanduser().resolve()
    rig_path = rig_2d_path.expanduser().resolve()
    learned = _load_motion(learned_path, learned=True)
    rig = load_rig_2d(rig_path)
    joints = np.asarray(learned["joints_3d"], dtype=np.float32)
    fps = float(learned["fps"])
    if rig.frames != len(joints) or not np.isclose(rig.fps, fps):
        raise ValueError("Motion and 2D rig frame count/FPS do not match")

    warnings: list[str] = []
    if not np.any(rig.image_mask > 0):
        warnings.append("no visual features are active; inference relies on detector inputs")
    repeats = int(np.sum(np.diff(rig.source_frame_indices) == 0))
    if repeats and int(rig.provenance.get("temporal_resampling_version", 1)) < 2:
        warnings.append("legacy frontend repeated source frames before detector tracking")

    contacts = np.asarray(learned["contacts"], dtype=np.float32)
    contact_stats = {}
    for column, joint in enumerate(SKELETON.contact_joints):
        probabilities = contacts[:, column]
        contact_stats[SKELETON.names[int(joint)]] = {
            "mean_probability": float(probabilities.mean()),
            "max_probability": float(probabilities.max()),
            "fraction_above_0_5": float(np.mean(probabilities >= 0.5)),
        }
    if max(value["fraction_above_0_5"] for value in contact_stats.values()) < 0.05:
        warnings.append("contact predictions are effectively inactive on this clip")

    root = np.asarray(learned["root_translation"], dtype=np.float32)
    world = _root_stats(root, fps)
    raw_world = (
        _root_stats(np.asarray(learned["root_translation_raw"], dtype=np.float32), fps)
        if "root_translation_raw" in learned
        else None
    )
    if raw_world is not None and raw_world["path_length_m"] > world["path_length_m"] + 0.25:
        warnings.append("root-motion fail-safe suppressed an ungrounded learned trajectory")

    comparison: dict[str, Any] | None = None
    if baseline_motion_path is not None:
        baseline_path = baseline_motion_path.expanduser().resolve()
        baseline = _load_motion(baseline_path, learned=False)
        baseline_joints = np.asarray(baseline["joints_3d"], dtype=np.float32)
        if baseline_joints.shape != joints.shape or not np.isclose(float(baseline["fps"]), fps):
            raise ValueError("Baseline and learned motion shapes/FPS do not match")
        correction = np.linalg.norm(joints - baseline_joints, axis=-1)
        learned_temporal = _temporal_stats(joints, fps)
        baseline_temporal = _temporal_stats(baseline_joints, fps)
        baseline_acceleration = baseline_temporal["acceleration_mean_mps2"]
        learned_acceleration = learned_temporal["acceleration_mean_mps2"]
        comparison = {
            "baseline_motion": str(baseline_path),
            "baseline_sha256": file_sha256(baseline_path),
            "correction_mean_m": float(correction.mean()),
            "correction_p95_m": float(np.percentile(correction, 95)),
            "correction_max_m": float(correction.max()),
            "learned_temporal": learned_temporal,
            "baseline_temporal": baseline_temporal,
            "acceleration_reduction_fraction": float(
                (baseline_acceleration - learned_acceleration) / max(baseline_acceleration, 1e-8)
            ),
        }

    detector: dict[str, Any] | None = None
    if detector_world_3d_path is not None:
        detector_path = detector_world_3d_path.expanduser().resolve()
        artifact = load_detector_world_3d(detector_path)
        if artifact.frames != len(joints):
            raise ValueError("Detector world-3D and learned motion frame counts do not match")
        detector = {
            "path": str(detector_path),
            "sha256": file_sha256(detector_path),
            "frame_coverage": float(np.mean(artifact.frame_mask > 0)),
            "confidence_mean": float(artifact.confidence.mean()),
            "confidence_below_0_5_fraction": float(np.mean(artifact.confidence < 0.5)),
        }

    preview_path: Path | None = None
    preview_motion = "selected"
    if world_preview:
        if source_video is None:
            raise ValueError("A source video is required for world-space preview")
        video = source_video.expanduser().resolve()
        expected_hash = rig.provenance.get("source_sha256")
        if expected_hash and file_sha256(video) != expected_hash:
            raise ValueError("Source video SHA-256 does not match the 2D rig provenance")
        preview_joints = np.asarray(learned["joints_world"], dtype=np.float32)
        if "joints_world_raw" in learned:
            preview_joints = np.asarray(learned["joints_world_raw"], dtype=np.float32)
            preview_motion = "raw-learned"
        preview_path = (
            output_path.expanduser()
            .resolve()
            .with_name(
                f"preview-motion-world{'-raw' if preview_motion == 'raw-learned' else ''}.mp4"
            )
        )
        _render_motion_preview(
            video,
            rig.source_frame_indices,
            rig.pixel_keypoints,
            preview_joints,
            preview_path,
            fps,
            avatar_title=f"Gravity Mocap - fixed-camera {preview_motion} world motion",
            avatar_renderer=avatar_renderer,
            world_space=True,
        )

    result = {
        "motion_diagnostics_version": MOTION_DIAGNOSTICS_VERSION,
        "artifact_type": "gravity-mocap-real-video-diagnostics",
        "ground_truth_available": False,
        "motion": str(learned_path),
        "motion_sha256": file_sha256(learned_path),
        "rig_2d": str(rig_path),
        "rig_2d_sha256": file_sha256(rig_path),
        "frames": len(joints),
        "fps": fps,
        "duration_seconds": (len(joints) - 1) / fps,
        "frontend": {
            "source_fps": rig.source_fps,
            "target_fps": rig.fps,
            "repeated_preview_source_indices": repeats,
            "temporal_resampling_version": rig.provenance.get("temporal_resampling_version", 1),
            "image_feature_frames": int(np.sum(rig.image_mask > 0)),
        },
        "detector_world_3d": detector,
        "local_pose_temporal": _temporal_stats(joints, fps),
        "baseline_comparison": comparison,
        "world_root": world,
        "world_root_raw": raw_world,
        "contacts": contact_stats,
        "world_preview": str(preview_path) if preview_path is not None else None,
        "world_preview_motion": preview_motion if preview_path is not None else None,
        "warnings": warnings,
    }
    write_json_atomic(output_path.expanduser().resolve(), result)
    return result
