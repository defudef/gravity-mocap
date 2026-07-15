from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch

from .artifacts import write_json_atomic, write_npz_atomic
from .avatar import AVATAR_RENDER_VERSION, render_avatar_panel
from .checkpoint import CHECKPOINT_VERSION
from .mesh_avatar import (
    AVATAR_RENDERERS,
    avatar_provenance,
    render_mesh_avatar_frames,
    resolve_avatar_renderer,
)
from .model import GravityViewMotionModel
from .projection import camera_space_joints
from .rig2d import load_rig_2d
from .rotations import (
    forward_kinematics,
    integrate_root_velocity,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from .schema import stable_hash
from .skeleton import SKELETON
from .video import _sampled_frames, _video_dependencies, file_sha256
from .world3d import load_detector_world_3d, prepare_detector_world_3d

INFERENCE_VERSION = 6
ROOT_MOTION_POLICIES = ("safe", "learned", "stationary")


def select_root_motion(
    raw_velocity: torch.Tensor,
    root_rotations: torch.Tensor,
    contacts: torch.Tensor,
    *,
    fps: float,
    policy: str,
    minimum_ground_contact_fraction: float = 0.15,
    minimum_per_foot_contact_fraction: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if policy not in ROOT_MOTION_POLICIES:
        raise ValueError(
            f"Unknown root-motion policy {policy!r}; "
            f"expected one of {', '.join(ROOT_MOTION_POLICIES)}"
        )
    raw_translation = integrate_root_velocity(raw_velocity, root_rotations, fps=fps)
    ground_support = contacts[..., 2:].amax(dim=-1)
    supported_fraction = float((ground_support >= 0.5).float().mean().cpu())
    left_support = contacts[..., (2, 4)].amax(dim=-1)
    right_support = contacts[..., (3, 5)].amax(dim=-1)
    left_fraction = float((left_support >= 0.5).float().mean().cpu())
    right_fraction = float((right_support >= 0.5).float().mean().cpu())
    applied = policy
    reason = "explicit-policy"
    if policy == "safe":
        if (
            supported_fraction < minimum_ground_contact_fraction
            or left_fraction < minimum_per_foot_contact_fraction
            or right_fraction < minimum_per_foot_contact_fraction
        ):
            applied = "stationary"
            reason = "insufficient-ground-contact-support"
        else:
            applied = "learned"
            reason = "ground-contact-support-passed"
    if applied == "stationary":
        velocity = torch.zeros_like(raw_velocity)
        translation = torch.zeros_like(raw_translation)
    else:
        velocity = raw_velocity
        translation = raw_translation
    return (
        velocity,
        translation,
        {
            "requested": policy,
            "applied": applied,
            "reason": reason,
            "ground_contact_fraction": supported_fraction,
            "left_foot_contact_fraction": left_fraction,
            "right_foot_contact_fraction": right_fraction,
            "minimum_ground_contact_fraction": minimum_ground_contact_fraction,
            "minimum_per_foot_contact_fraction": minimum_per_foot_contact_fraction,
        },
    )


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_inference_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[GravityViewMotionModel, dict[str, Any], dict[str, Any]]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    version = payload.get("checkpoint_version")
    if version != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"Unsupported checkpoint version {version!r}; expected {CHECKPOINT_VERSION}"
        )
    config = payload.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise RuntimeError("Checkpoint does not contain a resolved model config")
    model = GravityViewMotionModel(**config["model"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, config, payload


def window_starts(frame_count: int, sequence_length: int, stride: int) -> list[int]:
    if frame_count < 1 or sequence_length < 2 or stride < 1:
        raise ValueError("Invalid inference window dimensions")
    starts = list(range(0, max(frame_count - sequence_length + 1, 1), stride))
    final = max(frame_count - sequence_length, 0)
    if not starts or starts[-1] != final:
        starts.append(final)
    return starts


def _blend_predictions(
    model: GravityViewMotionModel,
    inputs: dict[str, np.ndarray],
    *,
    sequence_length: int,
    stride: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    frame_count = len(inputs["frame_mask"])
    starts = window_starts(frame_count, sequence_length, stride)
    weights = np.hanning(sequence_length + 2)[1:-1].astype(np.float32)
    input_names = ["bbox", "keypoints_2d", "image_features", "image_mask", "camera_delta_6d"]
    if model.use_detector_world_3d:
        input_names.extend(("detector_joints_3d", "detector_3d_confidence"))
    sums: dict[str, torch.Tensor] = {}
    weight_sum = torch.zeros(frame_count, dtype=torch.float32, device=device)

    with torch.inference_mode():
        for start in starts:
            valid = min(sequence_length, frame_count - start)
            frame_indices = np.clip(np.arange(start, start + sequence_length), 0, frame_count - 1)
            batch = {
                name: torch.from_numpy(inputs[name][frame_indices].astype(np.float32, copy=False))
                .unsqueeze(0)
                .to(device)
                for name in input_names
            }
            prediction = model(batch)
            window_weight = torch.from_numpy(weights[:valid]).to(device)
            weight_sum[start : start + valid] += window_weight
            for name, value in prediction.items():
                if name not in sums:
                    sums[name] = torch.zeros(
                        (frame_count, *value.shape[2:]), dtype=value.dtype, device=device
                    )
                expanded_weight = window_weight.reshape(valid, *((1,) * (value.ndim - 2)))
                sums[name][start : start + valid] += value[0, :valid] * expanded_weight

    if torch.any(weight_sum <= 0):
        raise RuntimeError("Inference window blending left uncovered frames")
    outputs = {
        name: value / weight_sum.reshape(frame_count, *((1,) * (value.ndim - 1)))
        for name, value in sums.items()
    }
    for name in (
        "local_rotations_6d",
        "camera_orientation_6d",
        "gravity_view_orientation_6d",
    ):
        if name in outputs:
            outputs[name] = matrix_to_rotation_6d(rotation_6d_to_matrix(outputs[name]))
    return outputs


def _render_motion_preview(
    video: Path,
    indices: np.ndarray,
    pixel_keypoints: np.ndarray,
    joints_3d: np.ndarray,
    output: Path,
    fps: float,
    *,
    nearer_positive: bool = True,
    avatar_title: str = "Gravity Mocap - learned 3D avatar",
    avatar_renderer: str = "auto",
    world_space: bool = False,
) -> str:
    cv2, _, _ = _video_dependencies()
    renderer = resolve_avatar_renderer(avatar_renderer)
    if world_space and renderer != "mesh":
        raise RuntimeError("World-space preview requires the Blender mesh renderer")
    first_frame = next(_sampled_frames(video, indices[:1]))[2]
    source_height, source_width = first_frame.shape[:2]
    panel_width = source_height
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    with TemporaryDirectory(prefix="gravity-mocap-avatar-") as temporary_directory:
        mesh_frames = (
            render_mesh_avatar_frames(
                joints_3d,
                Path(temporary_directory),
                width=panel_width,
                height=source_height,
                fps=fps,
                world_space=world_space,
            )
            if renderer == "mesh"
            else None
        )
        writer = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (source_width + panel_width, source_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create motion preview: {temporary}")
        extents = np.ptp(joints_3d[..., :2], axis=1)
        scale = 0.75 * source_height / max(float(np.percentile(extents, 95)), 0.5)
        center = np.asarray([panel_width / 2.0, source_height * 0.55], dtype=np.float32)
        try:
            for output_index, _, frame in _sampled_frames(video, indices):
                points_2d = pixel_keypoints[output_index]
                for joint, parent in enumerate(SKELETON.parents):
                    if parent < 0 or points_2d[joint, 2] <= 0 or points_2d[parent, 2] <= 0:
                        continue
                    cv2.line(
                        frame,
                        tuple(points_2d[int(parent), :2].round().astype(int)),
                        tuple(points_2d[joint, :2].round().astype(int)),
                        (100, 255, 100),
                        2,
                        cv2.LINE_AA,
                    )
                if mesh_frames is None:
                    panel = render_avatar_panel(
                        cv2,
                        joints_3d[output_index],
                        width=panel_width,
                        height=source_height,
                        scale=scale,
                        center=center,
                        nearer_positive=nearer_positive,
                        title=avatar_title,
                    )
                else:
                    panel = cv2.imread(str(mesh_frames[output_index]), cv2.IMREAD_COLOR)
                    if panel is None or panel.shape[:2] != (source_height, panel_width):
                        raise RuntimeError(
                            f"Cannot read rendered avatar frame: {mesh_frames[output_index]}"
                        )
                    cv2.putText(
                        panel,
                        avatar_title,
                        (20, 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (238, 238, 244),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        panel,
                        "Quaternius CC0 - "
                        f"{'fixed-camera world space' if world_space else 'neutral retarget'} "
                        "- no SMPL",
                        (20, source_height - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (168, 168, 180),
                        1,
                        cv2.LINE_AA,
                    )
                writer.write(np.concatenate((frame, panel), axis=1))
        finally:
            writer.release()
    os.replace(temporary, output)
    return renderer


def infer_rig(
    rig_2d_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    source_video: Path | None = None,
    detector_world_3d_path: Path | None = None,
    device_name: str = "auto",
    force: bool = False,
    preview: bool = False,
    avatar_renderer: str = "auto",
    root_motion: str = "safe",
) -> dict[str, Any]:
    """Recover neutral 3D motion from a standalone 2D rig artifact."""
    rig_2d_path = rig_2d_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    video = source_video.expanduser().resolve() if source_video is not None else None
    world_3d_path = (
        detector_world_3d_path.expanduser().resolve()
        if detector_world_3d_path is not None
        else None
    )
    if preview and video is None:
        raise ValueError("A source video is required to render the 3D motion preview")
    if avatar_renderer not in AVATAR_RENDERERS:
        raise ValueError(
            f"Unknown avatar renderer {avatar_renderer!r}; "
            f"expected one of {', '.join(AVATAR_RENDERERS)}"
        )
    if root_motion not in ROOT_MOTION_POLICIES:
        raise ValueError(
            f"Unknown root-motion policy {root_motion!r}; "
            f"expected one of {', '.join(ROOT_MOTION_POLICIES)}"
        )
    actual_avatar_renderer = (
        resolve_avatar_renderer(avatar_renderer) if preview else avatar_renderer
    )
    if video is not None and not video.is_file():
        raise ValueError(f"Video does not exist: {video}")
    rig = load_rig_2d(rig_2d_path)
    if video is not None:
        expected_source_hash = rig.provenance.get("source_sha256")
        actual_source_hash = file_sha256(video)
        if expected_source_hash and actual_source_hash != expected_source_hash:
            raise ValueError("Source video SHA-256 does not match the 2D rig provenance")
    device = resolve_device(device_name)
    model, config, checkpoint = load_inference_checkpoint(checkpoint_path, device)
    image_feature_dim = int(config["model"]["image_feature_dim"])
    inputs = rig.model_inputs(image_feature_dim)
    world_3d = None
    if model.use_detector_world_3d:
        if world_3d_path is None:
            raise ValueError(
                "This checkpoint requires a detector world-3D artifact; pass --detector-world-3d"
            )
        world_3d = load_detector_world_3d(world_3d_path)
        if world_3d.frames != rig.frames:
            raise ValueError("2D rig and detector world-3D frame counts do not match")
        if not np.array_equal(world_3d.source_frame_indices, rig.source_frame_indices):
            raise ValueError("2D rig and detector world-3D source frame indices do not match")
        if not np.isclose(world_3d.fps, rig.fps):
            raise ValueError("2D rig and detector world-3D FPS do not match")
        rig_source_hash = rig.provenance.get("source_sha256")
        world_source_hash = world_3d.provenance.get("source_sha256")
        if rig_source_hash and world_source_hash and rig_source_hash != world_source_hash:
            raise ValueError("2D rig and detector world-3D source provenance do not match")
        detector_joints, detector_confidence, _, _ = prepare_detector_world_3d(world_3d)
        inputs["detector_joints_3d"] = detector_joints
        inputs["detector_3d_confidence"] = detector_confidence.astype(np.float32)
    sequence_length = int(config["data"]["sequence_length"])
    stride = int(config["data"]["stride"])
    fps = rig.fps

    checkpoint_digest = file_sha256(checkpoint_path)
    rig_digest = file_sha256(rig_2d_path)
    request = {
        "inference_version": INFERENCE_VERSION,
        "avatar_render_version": AVATAR_RENDER_VERSION,
        "avatar_renderer": actual_avatar_renderer,
        "avatar_asset": (avatar_provenance() if actual_avatar_renderer == "mesh" else None),
        "rig_2d_sha256": rig_digest,
        "rig_2d_request_hash": rig.provenance.get("request_hash"),
        "detector_world_3d_sha256": (
            file_sha256(world_3d_path) if world_3d_path is not None else None
        ),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_version": CHECKPOINT_VERSION,
        "sequence_length": sequence_length,
        "stride": stride,
        "root_motion": root_motion,
    }
    request_hash = stable_hash(request)
    output_dir = output_dir.expanduser().resolve()
    motion_path = output_dir / "motion.npz"
    manifest_path = output_dir / "motion-manifest.json"
    preview_path = output_dir / "preview-motion.mp4"
    if not force and motion_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("request_hash") == request_hash and (
                not preview or preview_path.is_file()
            ):
                return {
                    "status": "cached",
                    "frames": int(manifest["frames"]),
                    "motion": str(motion_path),
                    "manifest": str(manifest_path),
                    "preview": str(preview_path) if preview_path.is_file() else None,
                    "avatar_renderer": manifest.get("avatar_renderer"),
                }
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    prediction = _blend_predictions(
        model,
        inputs,
        sequence_length=sequence_length,
        stride=stride,
        device=device,
    )
    model_joints = prediction.get("joints_3d")
    if model_joints is not None:
        from .preprocess import canonicalize_motion_skeleton

        local_rotations_numpy, centered_joints_numpy = canonicalize_motion_skeleton(
            model_joints.cpu().numpy()
        )
        local_rotations_6d = torch.from_numpy(local_rotations_numpy).to(device)
        centered_joints = torch.from_numpy(centered_joints_numpy).to(device)
        gravity_view_orientation_6d = local_rotations_6d[:, 0]
    else:
        local_rotations_6d = prediction["local_rotations_6d"]
        local_rotations = rotation_6d_to_matrix(local_rotations_6d)
        offsets = torch.from_numpy(SKELETON.rest_offsets).to(device)
        parents = torch.from_numpy(SKELETON.parents).to(device)
        centered_joints = forward_kinematics(
            local_rotations,
            torch.zeros((len(inputs["frame_mask"]), 3), dtype=torch.float32, device=device),
            offsets,
            parents,
        )
        gravity_view_orientation_6d = prediction["gravity_view_orientation_6d"]
    local_rotations = rotation_6d_to_matrix(local_rotations_6d)
    raw_root_velocity = prediction["root_velocity_local"]
    contacts = torch.sigmoid(prediction["contacts"])
    root_velocity, root_translation, root_motion_result = select_root_motion(
        raw_root_velocity,
        local_rotations[:, 0],
        contacts,
        fps=fps,
        policy=root_motion,
    )
    raw_root_translation = integrate_root_velocity(
        raw_root_velocity, local_rotations[:, 0], fps=fps
    )
    camera_joints = camera_space_joints(
        local_rotations_6d,
        prediction["camera_orientation_6d"],
        centered_joints,
    )
    world_joints = centered_joints + root_translation.unsqueeze(-2)

    arrays = {
        "local_rotations_6d": local_rotations_6d.cpu().numpy(),
        "local_rotation_matrices": local_rotations.cpu().numpy(),
        "camera_orientation_6d": prediction["camera_orientation_6d"].cpu().numpy(),
        "gravity_view_orientation_6d": gravity_view_orientation_6d.cpu().numpy(),
        "root_velocity_local": root_velocity.cpu().numpy(),
        "root_translation": root_translation.cpu().numpy(),
        "root_velocity_local_raw": raw_root_velocity.cpu().numpy(),
        "root_translation_raw": raw_root_translation.cpu().numpy(),
        "joints_3d": centered_joints.cpu().numpy(),
        "joints_camera": camera_joints.cpu().numpy(),
        "joints_world": world_joints.cpu().numpy(),
        "joints_world_raw": (centered_joints + raw_root_translation.unsqueeze(-2)).cpu().numpy(),
        "contacts": contacts.cpu().numpy(),
        "weak_camera": prediction["weak_camera"].cpu().numpy(),
    }
    if model_joints is not None:
        arrays["model_joints_3d"] = model_joints.cpu().numpy()
        arrays["detector_residual_3d"] = prediction["detector_residual_3d"].cpu().numpy()
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("Motion inference produced NaN or infinity")
    provenance = {
        **request,
        "request_hash": request_hash,
        "rig_2d": str(rig_2d_path),
        "rig_2d_provenance": rig.provenance,
        "detector_world_3d": str(world_3d_path) if world_3d_path is not None else None,
        "detector_world_3d_provenance": world_3d.provenance if world_3d is not None else None,
        "checkpoint": str(checkpoint_path),
        "checkpoint_compatibility_hash": checkpoint.get("compatibility_hash"),
        "checkpoint_data_bom_hash": checkpoint.get("data_bom_hash"),
        "device": str(device),
        "fps": fps,
        "frames": len(inputs["frame_mask"]),
        "joint_names": list(SKELETON.names),
        "root_motion_result": root_motion_result,
    }
    write_npz_atomic(
        motion_path,
        **arrays,
        joint_names=np.asarray(SKELETON.names),
        parents=SKELETON.parents,
        rest_offsets=SKELETON.rest_offsets,
        fps=np.asarray(fps, dtype=np.float32),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    if preview:
        output_dir.mkdir(parents=True, exist_ok=True)
        assert video is not None
        _render_motion_preview(
            video,
            inputs["source_frame_indices"].astype(np.int64),
            inputs["pixel_keypoints"],
            arrays["joints_3d"],
            preview_path,
            fps,
            nearer_positive=True,
            avatar_title="B - detector-safe residual v7",
            avatar_renderer=actual_avatar_renderer,
        )
    write_json_atomic(
        manifest_path,
        {
            **provenance,
            "motion": str(motion_path),
            "preview": str(preview_path) if preview else None,
        },
    )
    return {
        "status": "created",
        "frames": len(inputs["frame_mask"]),
        "device": str(device),
        "motion": str(motion_path),
        "manifest": str(manifest_path),
        "preview": str(preview_path) if preview else None,
        "avatar_renderer": actual_avatar_renderer,
    }


def infer_motion(
    detector_inputs: Path,
    video: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    device_name: str = "auto",
    force: bool = False,
    preview: bool = True,
    avatar_renderer: str = "auto",
    root_motion: str = "safe",
) -> dict[str, Any]:
    """Backward-compatible video-bound wrapper around :func:`infer_rig`."""
    return infer_rig(
        detector_inputs,
        checkpoint_path,
        output_dir,
        source_video=video,
        device_name=device_name,
        force=force,
        preview=preview,
        avatar_renderer=avatar_renderer,
        root_motion=root_motion,
    )
