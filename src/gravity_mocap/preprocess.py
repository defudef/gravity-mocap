from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .adapters import (
    B3DReader,
    canonicalize_addbiomechanics,
    canonicalize_positions,
    find_cmu_sequences,
    load_asf_amc,
    load_bvh,
    load_generic_npz,
)
from .catalog import DatasetEntry
from .schema import REQUIRED_ARRAYS, SCHEMA_VERSION, stable_hash, write_shard
from .skeleton import CONTACT_JOINTS, JOINT_NAMES, PARENTS, REST_OFFSETS

CONVERTER_VERSION = "cleanroom-v4-auto-framed-rig"
B3D_BVH_CONVERTER_VERSION = "cleanroom-v5-b3d-trials-auto-framed-rig"
DEFAULT_TARGET_FPS = 30.0
DEFAULT_CAMERA_SIMULATION: dict[str, object] = {
    "yaw_degrees": [-180.0, 180.0],
    "pitch_degrees": [-20.0, 15.0],
    "roll_degrees": [-8.0, 8.0],
    "distance_meters": [3.0, 7.0],
    "horizontal_offset_meters": [-0.35, 0.35],
    "vertical_offset_meters": [-0.2, 0.2],
    "yaw_drift_std_degrees": 0.3,
    "pitch_drift_std_degrees": 0.1,
    "roll_drift_std_degrees": 0.05,
    "translation_drift_std_meters": 0.001,
    "bbox_padding": 0.12,
    "minimum_bbox_aspect_ratio": 0.2,
}


@dataclass(frozen=True)
class RawSource:
    primary: Path
    source_paths: tuple[Path, ...]


def _safe_extract_zip(archive_path: Path, extraction_root: Path) -> None:
    marker = extraction_root / ".gravity-mocap-extracted.json"
    expected_marker = {
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_mtime_ns": archive_path.stat().st_mtime_ns,
    }
    if marker.exists():
        try:
            if json.loads(marker.read_text()) == expected_marker:
                return
        except (OSError, json.JSONDecodeError):
            pass
    temporary = extraction_root.with_name(f".{extraction_root.name}.extracting")
    if temporary.exists():
        shutil.rmtree(temporary)
    if extraction_root.exists():
        shutil.rmtree(extraction_root)
    temporary.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [
                member
                for member in archive.infolist()
                if not member.filename.startswith("__MACOSX/")
                and Path(member.filename).name != ".DS_Store"
            ]
            if any(
                Path(member.filename).is_absolute() or ".." in Path(member.filename).parts
                for member in members
            ):
                raise RuntimeError(f"Unsafe member path in {archive_path}")
            file_members = [member for member in members if not member.is_dir()]
            for index, member in enumerate(file_members, start=1):
                output = temporary / member.filename
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                if index == 1 or index % 100 == 0 or index == len(file_members):
                    print(
                        f"[preprocess] extracting {archive_path.name}: {index}/{len(file_members)}",
                        flush=True,
                    )
        (temporary / marker.name).write_text(json.dumps(expected_marker, sort_keys=True))
        os.replace(temporary, extraction_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _ensure_extracted(entry: DatasetEntry, dataset_root: Path) -> None:
    expected_suffix = ".asf" if entry.dataset_id == "cmu_mocap" else ".bvh"
    if entry.dataset_id not in {"cmu_mocap", "100style"}:
        return
    if any(dataset_root.rglob(f"*{expected_suffix}")):
        return
    for archive_path in sorted(dataset_root.glob("*.zip")):
        _safe_extract_zip(archive_path, dataset_root / archive_path.stem)


def _load_100style_cuts(dataset_root: Path) -> dict[tuple[str, str], tuple[int, int]]:
    matches = list(dataset_root.rglob("Frame_Cuts.csv"))
    if len(matches) != 1:
        raise ValueError(
            f"100STYLE needs exactly one Frame_Cuts.csv below {dataset_root}, found {len(matches)}"
        )
    cuts: dict[tuple[str, str], tuple[int, int]] = {}
    with matches[0].open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            style = str(row["STYLE_NAME"])
            for key, start_value in row.items():
                if not key.endswith("_START") or start_value in {None, "", "N/A"}:
                    continue
                motion = key.removesuffix("_START")
                stop_value = row.get(f"{motion}_STOP")
                if stop_value in {None, "", "N/A"}:
                    continue
                start, stop = int(start_value), int(stop_value)
                if start < 0 or stop <= start:
                    raise ValueError(f"Invalid 100STYLE frame cut for {style}/{motion}")
                cuts[(style, motion)] = (start, stop)
    return cuts


def _trim_100style(
    positions: np.ndarray,
    path: Path,
    cuts: dict[tuple[str, str], tuple[int, int]],
) -> np.ndarray:
    style = path.parent.name
    prefix = f"{style}_"
    if not path.stem.startswith(prefix):
        raise ValueError(f"Unexpected 100STYLE filename: {path}")
    motion = path.stem.removeprefix(prefix)
    try:
        start, stop = cuts[(style, motion)]
    except KeyError as error:
        raise ValueError(f"Missing 100STYLE frame cut for {style}/{motion}") from error
    if stop > len(positions):
        raise ValueError(
            f"100STYLE frame cut {start}:{stop} exceeds {len(positions)} frames in {path}"
        )
    return positions[start:stop]


def _hash_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _cached_shard_matches(path: Path, expected: dict[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not set(REQUIRED_ARRAYS).issubset(archive.files) or "provenance_json" not in archive:
                return False
            provenance = json.loads(str(archive["provenance_json"]))
    except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError):
        return False
    if provenance.get("schema_version") != SCHEMA_VERSION:
        return False
    stored_hash = provenance.pop("provenance_hash", None)
    if stored_hash != stable_hash(provenance):
        return False
    return all(provenance.get(name) == value for name, value in expected.items())


def _align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / max(np.linalg.norm(source), 1e-8)
    target = target / max(np.linalg.norm(target), 1e-8)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    norm = float(np.linalg.norm(cross))
    if norm < 1e-8:
        if dot > 0:
            return np.eye(3, dtype=np.float32)
        candidate = np.asarray([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.8:
            candidate = np.asarray([0.0, 1.0, 0.0])
        axis = np.cross(source, candidate)
        axis /= np.linalg.norm(axis)
        return (2 * np.outer(axis, axis) - np.eye(3)).astype(np.float32)
    skew = np.asarray(
        [[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]]
    )
    return (np.eye(3) + skew + skew @ skew * ((1 - dot) / (norm * norm))).astype(np.float32)


def _fit_world_rotation(rest_vectors: np.ndarray, observed_vectors: np.ndarray) -> np.ndarray:
    rest = np.asarray(rest_vectors, dtype=np.float64)
    observed = np.asarray(observed_vectors, dtype=np.float64)
    valid = (np.linalg.norm(rest, axis=-1) > 1e-8) & (np.linalg.norm(observed, axis=-1) > 1e-8)
    rest = rest[valid]
    observed = observed[valid]
    if not len(rest):
        return np.eye(3, dtype=np.float32)
    if len(rest) == 1:
        return _align_vectors(rest[0], observed[0])
    rest /= np.linalg.norm(rest, axis=-1, keepdims=True)
    observed /= np.linalg.norm(observed, axis=-1, keepdims=True)
    left, _, right_transpose = np.linalg.svd(observed.T @ rest)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_transpose))
    return (left @ correction @ right_transpose).astype(np.float32)


def _estimate_local_rotations(joints: np.ndarray) -> np.ndarray:
    frames, joint_count = joints.shape[:2]
    children = [
        [index for index, parent in enumerate(PARENTS) if parent == joint]
        for joint in range(joint_count)
    ]
    local = np.repeat(
        np.eye(3, dtype=np.float32)[None, None], frames * joint_count, axis=0
    ).reshape(frames, joint_count, 3, 3)
    for frame in range(frames):
        world = [np.eye(3, dtype=np.float32) for _ in range(joint_count)]
        for joint in range(joint_count):
            if children[joint]:
                rest = np.stack([REST_OFFSETS[child] for child in children[joint]])
                observed = np.stack(
                    [joints[frame, child] - joints[frame, joint] for child in children[joint]]
                )
                world[joint] = _fit_world_rotation(rest, observed)
            elif joint == 0:
                world[joint] = np.eye(3, dtype=np.float32)
            else:
                parent = int(PARENTS[joint])
                world[joint] = world[parent]
            parent = int(PARENTS[joint])
            local[frame, joint] = world[joint] if parent < 0 else world[parent].T @ world[joint]
    return np.concatenate((local[..., :, 0], local[..., :, 1]), axis=-1).astype(np.float32)


def _rotation_6d_to_matrix(value: np.ndarray) -> np.ndarray:
    first = value[..., :3]
    second = value[..., 3:]
    first = first / np.linalg.norm(first, axis=-1, keepdims=True).clip(1e-8)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.linalg.norm(second, axis=-1, keepdims=True).clip(1e-8)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1).astype(np.float32)


def _forward_kinematics(local_rotations_6d: np.ndarray) -> np.ndarray:
    local = _rotation_6d_to_matrix(local_rotations_6d)
    world_rotations: list[np.ndarray] = []
    world_positions: list[np.ndarray] = []
    root = np.zeros((len(local), 3), dtype=np.float32)
    for joint, parent_value in enumerate(PARENTS):
        parent = int(parent_value)
        if parent < 0:
            world_rotations.append(local[:, joint])
            world_positions.append(root)
            continue
        parent_rotation = world_rotations[parent]
        world_rotations.append(parent_rotation @ local[:, joint])
        offset = np.broadcast_to(REST_OFFSETS[joint], root.shape)
        world_positions.append(
            world_positions[parent] + (parent_rotation @ offset[..., None])[..., 0]
        )
    return np.stack(world_positions, axis=1).astype(np.float32)


def canonicalize_motion_skeleton(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retarget observed joint directions to the fixed neutral skeleton."""
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (len(JOINT_NAMES), 3):
        raise ValueError(f"Expected joints shaped (T, {len(JOINT_NAMES)}, 3), got {joints.shape}")
    if not np.isfinite(joints).all():
        raise ValueError("Motion joints must be finite")
    local_rotations = _estimate_local_rotations(joints)
    return local_rotations, _forward_kinematics(local_rotations)


def resample_positions(
    joints: np.ndarray, source_fps: float, target_fps: float = DEFAULT_TARGET_FPS
) -> np.ndarray:
    """Linearly resample joint positions to one canonical temporal rate."""
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3 or len(joints) < 2:
        raise ValueError(f"Expected at least two frames shaped (T, J, 3), got {joints.shape}")
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be a positive finite number")
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be a positive finite number")
    if np.isclose(source_fps, target_fps):
        return joints.copy()

    duration = (len(joints) - 1) / float(source_fps)
    target_frames = max(2, int(np.floor(duration * target_fps)) + 1)
    source_times = np.arange(len(joints), dtype=np.float64) / float(source_fps)
    target_times = np.arange(target_frames, dtype=np.float64) / float(target_fps)
    flattened = joints.reshape(len(joints), -1)
    resampled = np.empty((target_frames, flattened.shape[1]), dtype=np.float32)
    for column in range(flattened.shape[1]):
        resampled[:, column] = np.interp(target_times, source_times, flattened[:, column]).astype(
            np.float32
        )
    return resampled.reshape(target_frames, *joints.shape[1:])


def _camera_rotations(yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    frames = len(yaw)
    yaw_matrix = np.zeros((frames, 3, 3), dtype=np.float32)
    yaw_cosine, yaw_sine = np.cos(yaw), np.sin(yaw)
    yaw_matrix[:, 0, 0] = yaw_cosine
    yaw_matrix[:, 0, 2] = -yaw_sine
    yaw_matrix[:, 1, 1] = 1.0
    yaw_matrix[:, 2, 0] = yaw_sine
    yaw_matrix[:, 2, 2] = yaw_cosine

    pitch_matrix = np.zeros_like(yaw_matrix)
    pitch_cosine, pitch_sine = np.cos(pitch), np.sin(pitch)
    pitch_matrix[:, 0, 0] = 1.0
    pitch_matrix[:, 1, 1] = pitch_cosine
    pitch_matrix[:, 1, 2] = -pitch_sine
    pitch_matrix[:, 2, 1] = pitch_sine
    pitch_matrix[:, 2, 2] = pitch_cosine

    roll_matrix = np.zeros_like(yaw_matrix)
    roll_cosine, roll_sine = np.cos(roll), np.sin(roll)
    roll_matrix[:, 0, 0] = roll_cosine
    roll_matrix[:, 0, 1] = -roll_sine
    roll_matrix[:, 1, 0] = roll_sine
    roll_matrix[:, 1, 1] = roll_cosine
    roll_matrix[:, 2, 2] = 1.0
    return (roll_matrix @ pitch_matrix @ yaw_matrix).astype(np.float32)


def _camera_range(config: dict[str, object], name: str) -> tuple[float, float]:
    value = config[name]
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"synthetic_camera.{name} must contain [minimum, maximum]")
    minimum, maximum = float(value[0]), float(value[1])
    if not np.isfinite([minimum, maximum]).all() or maximum <= minimum:
        raise ValueError(f"synthetic_camera.{name} must be a finite increasing range")
    return minimum, maximum


def resolve_camera_simulation(config: dict[str, object] | None) -> dict[str, object]:
    resolved = dict(DEFAULT_CAMERA_SIMULATION)
    resolved.update(config or {})
    for name in (
        "yaw_degrees",
        "pitch_degrees",
        "roll_degrees",
        "distance_meters",
        "horizontal_offset_meters",
        "vertical_offset_meters",
    ):
        _camera_range(resolved, name)
    for name in (
        "yaw_drift_std_degrees",
        "pitch_drift_std_degrees",
        "roll_drift_std_degrees",
        "translation_drift_std_meters",
    ):
        value = float(resolved[name])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"synthetic_camera.{name} must be finite and non-negative")
        resolved[name] = value
    bbox_padding = float(resolved["bbox_padding"])
    if not np.isfinite(bbox_padding) or not 0 <= bbox_padding < 1:
        raise ValueError("synthetic_camera.bbox_padding must be in [0, 1)")
    resolved["bbox_padding"] = bbox_padding
    minimum_aspect = float(resolved["minimum_bbox_aspect_ratio"])
    if not np.isfinite(minimum_aspect) or not 0 < minimum_aspect <= 1:
        raise ValueError("synthetic_camera.minimum_bbox_aspect_ratio must be in (0, 1]")
    resolved["minimum_bbox_aspect_ratio"] = minimum_aspect
    if _camera_range(resolved, "distance_meters")[0] <= 0:
        raise ValueError("synthetic_camera.distance_meters must stay positive")
    return resolved


def _simulate_inputs(
    joints: np.ndarray,
    image_feature_dim: int,
    seed: int,
    camera_config: dict[str, object] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    config = resolve_camera_simulation(camera_config)
    generator = np.random.default_rng(seed)
    yaw_range = np.deg2rad(_camera_range(config, "yaw_degrees"))
    pitch_range = np.deg2rad(_camera_range(config, "pitch_degrees"))
    roll_range = np.deg2rad(_camera_range(config, "roll_degrees"))
    yaw = float(generator.uniform(*yaw_range)) + np.cumsum(
        generator.normal(
            0.0,
            np.deg2rad(float(config["yaw_drift_std_degrees"])),
            size=len(joints),
        )
    )
    pitch = float(generator.uniform(*pitch_range)) + np.cumsum(
        generator.normal(
            0.0,
            np.deg2rad(float(config["pitch_drift_std_degrees"])),
            size=len(joints),
        )
    )
    roll = float(generator.uniform(*roll_range)) + np.cumsum(
        generator.normal(
            0.0,
            np.deg2rad(float(config["roll_drift_std_degrees"])),
            size=len(joints),
        )
    )
    camera = _camera_rotations(yaw, pitch, roll)

    # Keep actor translation relative to the first frame. Per-frame root centering
    # made root motion unobservable from the simulated detector inputs.
    world = joints - joints[:1, :1]
    centered = world - world[:, :1]
    camera_centered = np.einsum("tij,tkj->tki", camera, centered)
    horizontal_range = _camera_range(config, "horizontal_offset_meters")
    vertical_range = _camera_range(config, "vertical_offset_meters")
    distance_range = _camera_range(config, "distance_meters")
    translation_drift = float(config["translation_drift_std_meters"])
    translation = np.stack(
        (
            float(generator.uniform(*horizontal_range))
            + np.cumsum(generator.normal(0.0, translation_drift, size=len(joints))),
            float(generator.uniform(*vertical_range))
            + np.cumsum(generator.normal(0.0, translation_drift * 0.5, size=len(joints))),
            float(generator.uniform(*distance_range))
            + np.cumsum(generator.normal(0.0, translation_drift, size=len(joints))),
        ),
        axis=-1,
    ).astype(np.float32)
    camera_root = np.einsum("tij,tj->ti", camera, world[:, 0]) + translation
    distance_minimum, distance_maximum = _camera_range(config, "distance_meters")
    depth = np.clip(camera_root[:, 2], distance_minimum, distance_maximum)
    scale = 1.0 / depth
    centered_xy = camera_centered[..., :2] * scale[:, None, None]
    minimum = centered_xy.min(axis=1)
    maximum = centered_xy.max(axis=1)
    center = 0.5 * (minimum + maximum)
    size = maximum - minimum
    major_size = size.max(axis=-1, keepdims=True)
    size = np.maximum(size, major_size * float(config["minimum_bbox_aspect_ratio"]))
    minimum = center - 0.5 * size
    maximum = center + 0.5 * size
    padding = float(config["bbox_padding"])
    unshifted_bbox_minimum = minimum - size * padding
    unshifted_bbox_maximum = maximum + size * padding
    shift_minimum = -1.0 - unshifted_bbox_minimum
    shift_maximum = 1.0 - unshifted_bbox_maximum
    if np.any(shift_minimum > shift_maximum):
        raise ValueError("Synthetic actor cannot be framed inside the configured camera view")
    proposed_shift = camera_root[:, :2] / depth[:, None]
    shift = np.clip(proposed_shift, shift_minimum, shift_maximum)
    bbox_minimum = unshifted_bbox_minimum + shift
    bbox_maximum = unshifted_bbox_maximum + shift
    bbox_size = bbox_maximum - bbox_minimum
    xy = centered_xy + shift[:, None, :]
    normalized = (xy - bbox_minimum[:, None]) / bbox_size[:, None]
    confidence = np.ones((*normalized.shape[:-1], 1), dtype=np.float32)
    keypoints = np.concatenate((np.clip(normalized * 2 - 1, -1.0, 1.0), confidence), axis=-1)
    keypoints = keypoints.astype(np.float32)
    bbox = np.concatenate((bbox_minimum, bbox_maximum), axis=-1).astype(np.float32)
    weak_camera = np.concatenate((scale[:, None], shift), axis=-1).astype(np.float32)
    image_features = np.zeros((len(joints), image_feature_dim), dtype=np.float32)
    camera_delta = np.repeat(np.eye(3, dtype=np.float32)[None], len(joints), axis=0)
    camera_delta[1:] = camera[1:] @ np.swapaxes(camera[:-1], -1, -2)
    camera_delta_6d = np.concatenate((camera_delta[..., :, 0], camera_delta[..., :, 1]), axis=-1)
    gravity_view = np.zeros_like(camera)
    gravity_view[:, :, 1] = np.asarray([0.0, 1.0, 0.0])
    view_world = camera[:, 2, :]
    gravity_view[:, :, 0] = np.cross(gravity_view[:, :, 1], view_world)
    gravity_view[:, :, 0] /= np.linalg.norm(gravity_view[:, :, 0], axis=-1, keepdims=True).clip(
        1e-8
    )
    gravity_view[:, :, 2] = np.cross(gravity_view[:, :, 0], gravity_view[:, :, 1])
    return bbox, keypoints, image_features, camera_delta_6d, camera, gravity_view, weak_camera


def motion_to_arrays(
    joints: np.ndarray,
    fps: float,
    image_feature_dim: int,
    seed: int = 0,
    camera_config: dict[str, object] | None = None,
) -> dict[str, np.ndarray]:
    joints = np.asarray(joints, dtype=np.float32)
    root_world = joints[:, 0]
    local_rotations, centered_joints = canonicalize_motion_skeleton(joints)
    canonical_world = centered_joints + root_world[:, None]
    root_displacement = np.zeros_like(root_world)
    root_displacement[:-1] = (root_world[1:] - root_world[:-1]) * float(fps)
    root_matrices = np.stack(
        (local_rotations[:, 0, :3], local_rotations[:, 0, 3:], np.zeros_like(root_world)), axis=-1
    )
    root_matrices[..., 2] = np.cross(root_matrices[..., 0], root_matrices[..., 1])
    root_velocity_local = (np.swapaxes(root_matrices, -1, -2) @ root_displacement[..., None])[
        ..., 0
    ]
    (
        bbox,
        keypoints,
        image_features,
        camera_delta_6d,
        camera,
        gravity_view,
        weak_camera,
    ) = _simulate_inputs(canonical_world, image_feature_dim, seed, camera_config)
    velocity = np.zeros((len(joints), len(CONTACT_JOINTS)), dtype=np.float32)
    velocity[1:] = (
        np.linalg.norm(
            canonical_world[1:, CONTACT_JOINTS] - canonical_world[:-1, CONTACT_JOINTS], axis=-1
        )
        * fps
    )
    floor = float(np.percentile(canonical_world[:, [7, 8, 10, 11], 1], 2))
    low = canonical_world[:, CONTACT_JOINTS, 1] < floor + 0.09
    low[:, :2] = True
    contacts = ((velocity < 0.08) & low).astype(np.float32)
    camera_orientation = camera @ root_matrices
    camera_orientation_6d = np.concatenate(
        (camera_orientation[..., :, 0], camera_orientation[..., :, 1]), axis=-1
    )
    gravity_orientation = np.swapaxes(gravity_view, -1, -2) @ root_matrices
    gravity_orientation_6d = np.concatenate(
        (gravity_orientation[..., :, 0], gravity_orientation[..., :, 1]), axis=-1
    )
    return {
        "bbox": bbox,
        "keypoints_2d": keypoints,
        "camera_delta_6d": camera_delta_6d.astype(np.float32),
        "image_features": image_features,
        "local_rotations_6d": local_rotations,
        "camera_orientation_6d": camera_orientation_6d.astype(np.float32),
        "gravity_view_orientation_6d": gravity_orientation_6d.astype(np.float32),
        "root_velocity_local": root_velocity_local.astype(np.float32),
        "weak_camera": weak_camera,
        "contacts": contacts,
        "joints_3d": centered_joints.astype(np.float32),
        "frame_mask": np.ones(len(joints), dtype=np.float32),
        "image_mask": np.zeros(len(joints), dtype=np.float32),
    }


def _sources(entry: DatasetEntry, raw_root: Path) -> list[RawSource]:
    dataset_root = raw_root / entry.dataset_id
    _ensure_extracted(entry, dataset_root)
    if entry.dataset_id == "cmu_mocap":
        return [RawSource(amc, (asf, amc)) for asf, amc in find_cmu_sequences(dataset_root)]
    paths = sorted(
        [
            *dataset_root.rglob("*.npz"),
            *dataset_root.rglob("*.b3d"),
            *dataset_root.rglob("*.bvh"),
        ]
    )
    if member_regex := entry.downloader.get("member_regex"):
        pattern = re.compile(str(member_regex))
        paths = [
            path for path in paths if pattern.search(path.relative_to(dataset_root).as_posix())
        ]
    return [RawSource(path, (path,)) for path in paths]


def _sequence_slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return normalized[:80] or "unnamed"


def _source_split_group(entry: DatasetEntry, relative: Path) -> str:
    if entry.dataset_id == "100style":
        return str(relative.parent)
    if entry.dataset_id == "cmu_mocap" and "subjects" in relative.parts:
        subject_index = relative.parts.index("subjects") + 1
        return "/".join(relative.parts[: subject_index + 1])
    return str(relative)


def _prune_disallowed_outputs(
    entry: DatasetEntry,
    output_root: Path,
    *,
    converted_any: bool,
) -> int:
    member_regex = entry.downloader.get("member_regex")
    if not converted_any or not member_regex:
        return 0
    pattern = re.compile(str(member_regex))
    removed = 0
    dataset_output = output_root / entry.dataset_id
    for path in sorted(dataset_output.rglob("*.npz")) if dataset_output.exists() else []:
        try:
            with np.load(path, allow_pickle=False) as archive:
                provenance = json.loads(str(archive["provenance_json"]))
        except (KeyError, OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError):
            continue
        source_file = str(provenance.get("source_file") or provenance.get("source_sequence", ""))
        if provenance.get("source_id") == entry.dataset_id and not pattern.search(source_file):
            path.unlink()
            removed += 1
    if removed:
        print(
            f"[preprocess] {entry.dataset_id}: removed {removed} stale shard(s) outside the "
            "configured source partition",
            flush=True,
        )
    return removed


def preprocess_dataset(
    entry: DatasetEntry,
    raw_root: Path,
    output_root: Path,
    *,
    image_feature_dim: int,
    target_fps: float = DEFAULT_TARGET_FPS,
    camera_config: dict[str, object] | None = None,
    limit: int | None = None,
) -> list[Path]:
    written: list[Path] = []
    sources = _sources(entry, raw_root)
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be a positive finite number")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    resolved_camera_config = resolve_camera_simulation(camera_config)
    cuts = (
        _load_100style_cuts(raw_root / entry.dataset_id) if entry.dataset_id == "100style" else {}
    )
    hash_cache: dict[Path, str] = {}
    cached = 0
    converted = 0
    attempted = 0
    stop = False
    print(f"[preprocess] {entry.dataset_id}: {len(sources)} raw file(s)", flush=True)
    for source_index, source in enumerate(sources, start=1):
        primary, source_paths = source.primary, source.source_paths
        relative = primary.relative_to(raw_root / entry.dataset_id)
        source_study = (
            relative.parts[2]
            if entry.dataset_id == "addbiomechanics" and len(relative.parts) > 2
            else None
        )
        source_hashes: list[str] = []
        for path in source_paths:
            if path not in hash_cache:
                hash_cache[path] = _hash_file(path)
            source_hashes.append(hash_cache[path])
        suffix = primary.suffix.lower()
        converter_version = (
            B3D_BVH_CONVERTER_VERSION if suffix in {".b3d", ".bvh"} else CONVERTER_VERSION
        )
        preprocess_settings: dict[str, object] = {
            "image_feature_dim": image_feature_dim,
            "joints": JOINT_NAMES,
            "target_fps": float(target_fps),
            "converter_version": converter_version,
            "synthetic_camera": resolved_camera_config,
        }
        if entry.dataset_id == "100style":
            preprocess_settings["trim_100style"] = True
        preprocess_hash = stable_hash(preprocess_settings)
        reader = B3DReader(primary) if suffix == ".b3d" else None
        sequence_specs: list[tuple[int | None, str, str]]
        if reader is not None:
            sequence_specs = [
                (
                    trial_index,
                    str(
                        relative.with_suffix("")
                        / (
                            f"trial-{trial_index:04d}-"
                            f"{_sequence_slug(reader.trial_name(trial_index))}"
                        )
                    ),
                    str(relative.with_suffix("")),
                )
                for trial_index in range(reader.trial_count)
            ]
        else:
            split_group = _source_split_group(entry, relative)
            sequence_specs = [(None, str(relative), split_group)]

        for trial_index, source_sequence, split_group in sequence_specs:
            if limit is not None and attempted >= limit:
                stop = True
                break
            attempted += 1
            if trial_index is None:
                output_relative = relative.with_suffix(".npz")
            else:
                output_relative = Path(f"{source_sequence}.npz")
            output = output_root / entry.dataset_id / output_relative
            cache_key: dict[str, object] = {
                "source_id": entry.dataset_id,
                "source_sequence": source_sequence,
                "source_sha256": source_hashes,
                "license_id": entry.license_id,
                "converter_version": converter_version,
                "preprocess_hash": preprocess_hash,
            }
            if suffix in {".b3d", ".bvh"}:
                cache_key["source_file"] = str(relative)
            if source_study is not None:
                cache_key["source_study"] = source_study
                if reader is not None and reader.source_href:
                    cache_key["source_href"] = reader.source_href
            if trial_index is not None:
                cache_key["source_trial_index"] = trial_index
            if _cached_shard_matches(output, cache_key):
                cached += 1
                written.append(output)
                status = "cached"
            else:
                if suffix == ".amc":
                    positions, names, fps = load_asf_amc(source_paths[0], source_paths[1])
                elif suffix == ".b3d":
                    assert reader is not None and trial_index is not None
                    positions, names, fps = reader.load_trial(trial_index)
                elif suffix == ".bvh":
                    positions, names, fps = load_bvh(primary)
                    if entry.dataset_id == "100style":
                        positions = _trim_100style(positions, primary, cuts)
                else:
                    positions, names, fps = load_generic_npz(primary)
                canonical = (
                    canonicalize_addbiomechanics(positions, names)
                    if suffix == ".b3d"
                    else canonicalize_positions(positions, names)
                )
                resampled = resample_positions(canonical, fps, target_fps)
                arrays = motion_to_arrays(
                    resampled,
                    target_fps,
                    image_feature_dim,
                    seed=int(stable_hash(source_sequence)[:8], 16),
                    camera_config=resolved_camera_config,
                )
                provenance = {
                    "source_id": entry.dataset_id,
                    "source_title": entry.title,
                    "source_sequence": source_sequence,
                    "source_file": str(relative),
                    "split_group": split_group,
                    "source_sha256": source_hashes,
                    "license_id": entry.license_id,
                    "license_url": entry.license_url,
                    "attribution_required": entry.attribution_required,
                    "requires_acceptance": entry.requires_acceptance,
                    "converter_version": converter_version,
                    "preprocess_hash": preprocess_hash,
                    "synthetic_camera": resolved_camera_config,
                    "source_fps": float(fps),
                    "fps": float(target_fps),
                    "input_mode": "motion_only",
                }
                if entry.dataset_id == "addbiomechanics":
                    provenance["source_study"] = source_study or relative.parent.name
                    if reader is not None and reader.source_href:
                        provenance["source_href"] = reader.source_href
                if trial_index is not None:
                    provenance["source_trial_index"] = trial_index
                    provenance["source_trial_name"] = reader.trial_name(trial_index)
                write_shard(output, arrays, provenance)
                converted += 1
                written.append(output)
                status = "converted"
            if attempted == 1 or attempted % 10 == 0:
                print(
                    f"[preprocess] {entry.dataset_id}: {attempted} sequence(s) "
                    f"({converted} converted, {cached} cached) | {source_sequence} [{status}] "
                    f"raw-file={source_index}/{len(sources)}",
                    flush=True,
                )
        if stop:
            break
    _prune_disallowed_outputs(entry, output_root, converted_any=bool(written))
    print(
        f"[preprocess] {entry.dataset_id}: ready={len(written)} "
        f"converted={converted} cached={cached}",
        flush=True,
    )
    return written
