from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from .adapters import (
    canonicalize_addbiomechanics,
    canonicalize_positions,
    find_cmu_sequences,
    load_asf_amc,
    load_b3d,
    load_generic_npz,
)
from .catalog import DatasetEntry
from .schema import REQUIRED_ARRAYS, SCHEMA_VERSION, stable_hash, write_shard
from .skeleton import CONTACT_JOINTS, JOINT_NAMES, PARENTS, REST_OFFSETS

CONVERTER_VERSION = "cleanroom-v2"
DEFAULT_TARGET_FPS = 30.0


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
        right = joints[frame, 2] - joints[frame, 1]
        up = joints[frame, 9] - joints[frame, 0]
        right /= max(np.linalg.norm(right), 1e-8)
        up -= right * np.dot(right, up)
        up /= max(np.linalg.norm(up), 1e-8)
        forward = np.cross(right, up)
        world[0] = np.stack((right, up, forward), axis=-1)
        local[frame, 0] = world[0]
        for joint in range(1, joint_count):
            parent = int(PARENTS[joint])
            if children[joint]:
                child = children[joint][0]
                rest = REST_OFFSETS[child]
                observed = joints[frame, child] - joints[frame, joint]
                world[joint] = _align_vectors(rest, observed)
            else:
                world[joint] = world[parent]
            local[frame, joint] = world[parent].T @ world[joint]
    return np.concatenate((local[..., :, 0], local[..., :, 1]), axis=-1).astype(np.float32)


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


def _simulate_inputs(
    joints: np.ndarray, image_feature_dim: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    yaw = float(generator.uniform(-np.pi, np.pi)) + np.cumsum(
        generator.normal(0.0, 0.005, size=len(joints))
    )
    pitch = float(generator.uniform(-0.2, 0.2)) + np.cumsum(
        generator.normal(0.0, 0.0015, size=len(joints))
    )
    roll = float(generator.uniform(-0.08, 0.08)) + np.cumsum(
        generator.normal(0.0, 0.0008, size=len(joints))
    )
    camera = _camera_rotations(yaw, pitch, roll)

    # Keep actor translation relative to the first frame. Per-frame root centering
    # made root motion unobservable from the simulated detector inputs.
    world = joints - joints[:1, :1]
    camera_points = np.einsum("tij,tkj->tki", camera, world)
    translation = np.stack(
        (
            float(generator.uniform(-0.25, 0.25))
            + np.cumsum(generator.normal(0.0, 0.001, size=len(joints))),
            float(generator.uniform(-0.15, 0.15))
            + np.cumsum(generator.normal(0.0, 0.0005, size=len(joints))),
            float(generator.uniform(3.5, 6.0))
            + np.cumsum(generator.normal(0.0, 0.001, size=len(joints))),
        ),
        axis=-1,
    ).astype(np.float32)
    camera_points += translation[:, None]
    xy = camera_points[..., :2] / np.maximum(camera_points[..., 2:], 0.25)
    minimum = xy.min(axis=1)
    maximum = xy.max(axis=1)
    size = np.maximum(maximum - minimum, 1e-4)
    normalized = (xy - minimum[:, None]) / size[:, None]
    confidence = np.ones((*normalized.shape[:-1], 1), dtype=np.float32)
    keypoints = np.concatenate((normalized * 2 - 1, confidence), axis=-1).astype(np.float32)
    bbox = np.concatenate((minimum, maximum), axis=-1).astype(np.float32)
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
    return bbox, keypoints, image_features, camera_delta_6d, camera, gravity_view


def motion_to_arrays(
    joints: np.ndarray, fps: float, image_feature_dim: int, seed: int = 0
) -> dict[str, np.ndarray]:
    joints = np.asarray(joints, dtype=np.float32)
    local_rotations = _estimate_local_rotations(joints)
    root_world = joints[:, 0]
    root_displacement = np.zeros_like(root_world)
    root_displacement[:-1] = (root_world[1:] - root_world[:-1]) * float(fps)
    root_matrices = np.stack(
        (local_rotations[:, 0, :3], local_rotations[:, 0, 3:], np.zeros_like(root_world)), axis=-1
    )
    root_matrices[..., 2] = np.cross(root_matrices[..., 0], root_matrices[..., 1])
    root_velocity_local = (np.swapaxes(root_matrices, -1, -2) @ root_displacement[..., None])[
        ..., 0
    ]
    bbox, keypoints, image_features, camera_delta_6d, camera, gravity_view = _simulate_inputs(
        joints, image_feature_dim, seed
    )
    velocity = np.zeros((len(joints), len(CONTACT_JOINTS)), dtype=np.float32)
    velocity[1:] = (
        np.linalg.norm(joints[1:, CONTACT_JOINTS] - joints[:-1, CONTACT_JOINTS], axis=-1) * fps
    )
    floor = float(np.percentile(joints[:, [7, 8, 10, 11], 1], 2))
    low = joints[:, CONTACT_JOINTS, 1] < floor + 0.09
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
    centered_joints = joints - root_world[:, None]
    weak_camera = np.zeros((len(joints), 3), dtype=np.float32)
    weak_camera[:, 0] = 0.25
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


def _sources(entry: DatasetEntry, raw_root: Path) -> list[tuple[Path, tuple[Path, ...]]]:
    dataset_root = raw_root / entry.dataset_id
    if entry.dataset_id == "cmu_mocap":
        if not any(dataset_root.rglob("*.asf")):
            for archive_path in dataset_root.glob("*.zip"):
                extraction_root = dataset_root / archive_path.stem
                with zipfile.ZipFile(archive_path) as archive:
                    members = archive.infolist()
                    if any(
                        Path(member.filename).is_absolute() or ".." in Path(member.filename).parts
                        for member in members
                    ):
                        raise RuntimeError(f"Unsafe member path in {archive_path}")
                    archive.extractall(extraction_root)
        return [(amc, (asf, amc)) for asf, amc in find_cmu_sequences(dataset_root)]
    paths = sorted([*dataset_root.rglob("*.npz"), *dataset_root.rglob("*.b3d")])
    return [(path, (path,)) for path in paths]


def preprocess_dataset(
    entry: DatasetEntry,
    raw_root: Path,
    output_root: Path,
    *,
    image_feature_dim: int,
    target_fps: float = DEFAULT_TARGET_FPS,
    limit: int | None = None,
) -> list[Path]:
    written: list[Path] = []
    sources = _sources(entry, raw_root)
    if limit is not None:
        sources = sources[:limit]
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be a positive finite number")
    preprocess_hash = stable_hash(
        {
            "image_feature_dim": image_feature_dim,
            "joints": JOINT_NAMES,
            "target_fps": float(target_fps),
            "converter_version": CONVERTER_VERSION,
        }
    )
    hash_cache: dict[Path, str] = {}
    cached = 0
    converted = 0
    print(f"[preprocess] {entry.dataset_id}: {len(sources)} raw sequence(s)", flush=True)
    for index, (primary, source_paths) in enumerate(sources, start=1):
        relative = primary.relative_to(raw_root / entry.dataset_id)
        output = output_root / entry.dataset_id / relative.with_suffix(".npz")
        source_hashes: list[str] = []
        for path in source_paths:
            if path not in hash_cache:
                hash_cache[path] = _hash_file(path)
            source_hashes.append(hash_cache[path])
        cache_key: dict[str, object] = {
            "source_id": entry.dataset_id,
            "source_sequence": str(relative),
            "source_sha256": source_hashes,
            "license_id": entry.license_id,
            "converter_version": CONVERTER_VERSION,
            "preprocess_hash": preprocess_hash,
        }
        if _cached_shard_matches(output, cache_key):
            cached += 1
            written.append(output)
            status = "cached"
            if index == 1 or index % 10 == 0 or index == len(sources):
                print(
                    f"[preprocess] {entry.dataset_id}: {index}/{len(sources)} "
                    f"({converted} converted, {cached} cached) | {relative} [{status}]",
                    flush=True,
                )
            continue
        if primary.suffix.lower() == ".amc":
            positions, names, fps = load_asf_amc(source_paths[0], source_paths[1])
        elif primary.suffix.lower() == ".b3d":
            positions, names, fps = load_b3d(primary)
        else:
            positions, names, fps = load_generic_npz(primary)
        canonical = (
            canonicalize_addbiomechanics(positions, names)
            if primary.suffix.lower() == ".b3d"
            else canonicalize_positions(positions, names)
        )
        resampled = resample_positions(canonical, fps, target_fps)
        arrays = motion_to_arrays(
            resampled,
            target_fps,
            image_feature_dim,
            seed=int(stable_hash(str(relative))[:8], 16),
        )
        provenance = {
            "source_id": entry.dataset_id,
            "source_title": entry.title,
            "source_sequence": str(relative),
            "source_sha256": source_hashes,
            "license_id": entry.license_id,
            "license_url": entry.license_url,
            "attribution_required": entry.attribution_required,
            "requires_acceptance": entry.requires_acceptance,
            "converter_version": CONVERTER_VERSION,
            "preprocess_hash": preprocess_hash,
            "source_fps": float(fps),
            "fps": float(target_fps),
            "input_mode": "motion_only",
        }
        write_shard(output, arrays, provenance)
        converted += 1
        written.append(output)
        if index == 1 or index % 10 == 0 or index == len(sources):
            print(
                f"[preprocess] {entry.dataset_id}: {index}/{len(sources)} "
                f"({converted} converted, {cached} cached) | {relative} [converted]",
                flush=True,
            )
    print(
        f"[preprocess] {entry.dataset_id}: ready={len(written)} "
        f"converted={converted} cached={cached}",
        flush=True,
    )
    return written
