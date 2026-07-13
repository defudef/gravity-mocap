from __future__ import annotations

import hashlib
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
from .schema import stable_hash, write_shard
from .skeleton import CONTACT_JOINTS, JOINT_NAMES, PARENTS, REST_OFFSETS

CONVERTER_VERSION = "cleanroom-v1"


def _hash_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


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


def _simulate_inputs(
    joints: np.ndarray, image_feature_dim: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    yaw = float(generator.uniform(-np.pi, np.pi)) + np.cumsum(
        generator.normal(0.0, 0.005, size=len(joints))
    )
    cosine, sine = np.cos(yaw), np.sin(yaw)
    camera = np.zeros((len(joints), 3, 3), dtype=np.float32)
    camera[:, 0, 0] = cosine
    camera[:, 0, 2] = -sine
    camera[:, 1, 1] = 1
    camera[:, 2, 0] = sine
    camera[:, 2, 2] = cosine
    centered = joints - joints[:, :1]
    camera_points = np.einsum("tij,tkj->tki", camera, centered)
    camera_points[..., 2] += 4.0
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
    root_displacement[:-1] = root_world[1:] - root_world[:-1]
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
    limit: int | None = None,
) -> list[Path]:
    written: list[Path] = []
    sources = _sources(entry, raw_root)
    if limit is not None:
        sources = sources[:limit]
    for primary, source_paths in sources:
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
        arrays = motion_to_arrays(
            canonical, fps, image_feature_dim, seed=int(stable_hash(str(primary))[:8], 16)
        )
        relative = primary.relative_to(raw_root / entry.dataset_id)
        output = output_root / entry.dataset_id / relative.with_suffix(".npz")
        provenance = {
            "source_id": entry.dataset_id,
            "source_title": entry.title,
            "source_sequence": str(relative),
            "source_sha256": [_hash_file(path) for path in source_paths],
            "license_id": entry.license_id,
            "license_url": entry.license_url,
            "attribution_required": entry.attribution_required,
            "requires_acceptance": entry.requires_acceptance,
            "converter_version": CONVERTER_VERSION,
            "preprocess_hash": stable_hash(
                {"image_feature_dim": image_feature_dim, "joints": JOINT_NAMES}
            ),
            "fps": fps,
            "input_mode": "motion_only",
        }
        write_shard(output, arrays, provenance)
        written.append(output)
    return written
