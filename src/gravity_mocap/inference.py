from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import write_json_atomic, write_npz_atomic
from .checkpoint import CHECKPOINT_VERSION
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

INFERENCE_VERSION = 2


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
    input_names = ("bbox", "keypoints_2d", "image_features", "image_mask", "camera_delta_6d")
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
        outputs[name] = matrix_to_rotation_6d(rotation_6d_to_matrix(outputs[name]))
    return outputs


def _render_motion_preview(
    video: Path,
    indices: np.ndarray,
    pixel_keypoints: np.ndarray,
    joints_3d: np.ndarray,
    output: Path,
    fps: float,
) -> None:
    cv2, _, _ = _video_dependencies()
    first_frame = next(_sampled_frames(video, indices[:1]))[2]
    source_height, source_width = first_frame.shape[:2]
    panel_width = source_height
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
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
            panel = np.full((source_height, panel_width, 3), 24, dtype=np.uint8)
            pose = joints_3d[output_index]
            projected = np.empty((len(pose), 2), dtype=np.float32)
            projected[:, 0] = center[0] + pose[:, 0] * scale
            projected[:, 1] = center[1] - pose[:, 1] * scale
            for joint, parent in enumerate(SKELETON.parents):
                if parent < 0:
                    continue
                cv2.line(
                    panel,
                    tuple(projected[int(parent)].round().astype(int)),
                    tuple(projected[joint].round().astype(int)),
                    (50, 220, 255),
                    3,
                    cv2.LINE_AA,
                )
            for x, y in projected:
                cv2.circle(panel, (round(float(x)), round(float(y))), 4, (255, 180, 80), -1)
            cv2.putText(
                panel,
                "Gravity Mocap 3D",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )
            writer.write(np.concatenate((frame, panel), axis=1))
    finally:
        writer.release()
    os.replace(temporary, output)


def infer_rig(
    rig_2d_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    source_video: Path | None = None,
    device_name: str = "auto",
    force: bool = False,
    preview: bool = False,
) -> dict[str, Any]:
    """Recover neutral 3D motion from a standalone 2D rig artifact."""
    rig_2d_path = rig_2d_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    video = source_video.expanduser().resolve() if source_video is not None else None
    if preview and video is None:
        raise ValueError("A source video is required to render the 3D motion preview")
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
    sequence_length = int(config["data"]["sequence_length"])
    stride = int(config["data"]["stride"])
    fps = rig.fps

    checkpoint_digest = file_sha256(checkpoint_path)
    rig_digest = file_sha256(rig_2d_path)
    request = {
        "inference_version": INFERENCE_VERSION,
        "rig_2d_sha256": rig_digest,
        "rig_2d_request_hash": rig.provenance.get("request_hash"),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_version": CHECKPOINT_VERSION,
        "sequence_length": sequence_length,
        "stride": stride,
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
    local_rotations_6d = prediction["local_rotations_6d"]
    local_rotations = rotation_6d_to_matrix(local_rotations_6d)
    root_velocity = prediction["root_velocity_local"]
    root_translation = integrate_root_velocity(root_velocity, local_rotations[:, 0], fps=fps)
    offsets = torch.from_numpy(SKELETON.rest_offsets).to(device)
    parents = torch.from_numpy(SKELETON.parents).to(device)
    centered_joints = forward_kinematics(
        local_rotations,
        torch.zeros((len(inputs["frame_mask"]), 3), dtype=torch.float32, device=device),
        offsets,
        parents,
    )
    camera_joints = camera_space_joints(
        local_rotations_6d,
        prediction["camera_orientation_6d"],
        centered_joints,
    )
    world_joints = centered_joints + root_translation.unsqueeze(-2)
    contacts = torch.sigmoid(prediction["contacts"])

    arrays = {
        "local_rotations_6d": local_rotations_6d.cpu().numpy(),
        "local_rotation_matrices": local_rotations.cpu().numpy(),
        "camera_orientation_6d": prediction["camera_orientation_6d"].cpu().numpy(),
        "gravity_view_orientation_6d": prediction["gravity_view_orientation_6d"].cpu().numpy(),
        "root_velocity_local": root_velocity.cpu().numpy(),
        "root_translation": root_translation.cpu().numpy(),
        "joints_3d": centered_joints.cpu().numpy(),
        "joints_camera": camera_joints.cpu().numpy(),
        "joints_world": world_joints.cpu().numpy(),
        "contacts": contacts.cpu().numpy(),
        "weak_camera": prediction["weak_camera"].cpu().numpy(),
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("Motion inference produced NaN or infinity")
    provenance = {
        **request,
        "request_hash": request_hash,
        "rig_2d": str(rig_2d_path),
        "rig_2d_provenance": rig.provenance,
        "checkpoint": str(checkpoint_path),
        "checkpoint_compatibility_hash": checkpoint.get("compatibility_hash"),
        "checkpoint_data_bom_hash": checkpoint.get("data_bom_hash"),
        "device": str(device),
        "fps": fps,
        "frames": len(inputs["frame_mask"]),
        "joint_names": list(SKELETON.names),
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
            arrays["joints_camera"],
            preview_path,
            fps,
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
    )
