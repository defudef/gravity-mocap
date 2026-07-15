from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import FORBIDDEN_PARAMETRIC_BODY_KEYS
from .artifacts import write_json_atomic, write_npz_atomic
from .catalog import DatasetCatalog
from .data import convert_targets_to_gravity_view
from .mesh_avatar import avatar_provenance, render_mesh_avatar_frames
from .schema import read_shard, stable_hash
from .skeleton import SKELETON
from .video import file_sha256, video_to_rig
from .world3d import load_detector_world_3d, prepare_detector_world_3d

DETECTOR_LOOP_VERSION = 1
DETECTOR_LOOP_GENERATOR_VERSION = 4
DETECTOR_LOOP_SUFFIX = ".detector-loop.npz"
DETECTOR_LOOP_BATCH_VERSION = 1
DETECTOR_LOOP_BATCH_MANIFEST = "batch-manifest.json"


@dataclass(frozen=True)
class DetectorLoopSidecar:
    """Real detector output aligned to one approved clean-room motion shard."""

    detector_joints_3d: np.ndarray
    detector_3d_confidence: np.ndarray
    frame_mask: np.ndarray
    source_shard_sha256: str
    source_provenance_hash: str
    provenance: dict[str, Any]

    @property
    def frames(self) -> int:
        return int(self.frame_mask.shape[0])


@dataclass(frozen=True)
class DetectorLoopBatchCandidate:
    """Small, portable inventory row used for deterministic batch selection."""

    relative_path: str
    source_id: str
    frames: int
    source_provenance_hash: str


def _validate_sidecar(sidecar: DetectorLoopSidecar) -> DetectorLoopSidecar:
    frames = sidecar.frames
    expected = {
        "detector_joints_3d": (frames, SKELETON.joint_count, 3),
        "detector_3d_confidence": (frames, SKELETON.joint_count),
        "frame_mask": (frames,),
    }
    if frames < 2:
        raise ValueError("A detector-loop sidecar needs at least two frames")
    for name, shape in expected.items():
        value = np.asarray(getattr(sidecar, name))
        if value.shape != shape:
            raise ValueError(f"{name} has shape {value.shape}; expected {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinity")
    confidence = np.asarray(sidecar.detector_3d_confidence)
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("detector_3d_confidence must be in [0, 1]")
    frame_mask = np.asarray(sidecar.frame_mask)
    if np.any((frame_mask < 0) | (frame_mask > 1)):
        raise ValueError("frame_mask must be in [0, 1]")
    hash_values = (sidecar.source_shard_sha256, sidecar.source_provenance_hash)
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in hash_values
    ):
        raise ValueError("Detector-loop source hashes must be SHA-256 values")
    missing = frame_mask <= 0
    if np.any(confidence[missing] != 0) or np.any(sidecar.detector_joints_3d[missing] != 0):
        raise ValueError("Missing detector-loop frames must contain only zeros")
    if sidecar.provenance.get("detector_loop_version") != DETECTOR_LOOP_VERSION:
        raise ValueError("Unsupported detector-loop sidecar version")
    if sidecar.provenance.get("artifact_type") != "gravity-mocap-detector-loop-sidecar":
        raise ValueError("Unsupported detector-loop artifact type")
    if sidecar.provenance.get("source_shard_sha256") != sidecar.source_shard_sha256:
        raise ValueError("Detector-loop source shard hash fields do not match")
    if sidecar.provenance.get("source_provenance_hash") != sidecar.source_provenance_hash:
        raise ValueError("Detector-loop source provenance hash fields do not match")
    metadata = dict(sidecar.provenance)
    stored_provenance_hash = metadata.pop("provenance_hash", None)
    if stored_provenance_hash is not None and stored_provenance_hash != stable_hash(metadata):
        raise ValueError("Detector-loop provenance hash is invalid")
    return sidecar


def write_detector_loop_sidecar(path: Path, sidecar: DetectorLoopSidecar) -> Path:
    sidecar = _validate_sidecar(sidecar)
    metadata = dict(sidecar.provenance)
    metadata.pop("provenance_hash", None)
    metadata["provenance_hash"] = stable_hash(metadata)
    write_npz_atomic(
        path,
        detector_joints_3d=np.asarray(sidecar.detector_joints_3d, dtype=np.float32),
        detector_3d_confidence=np.asarray(sidecar.detector_3d_confidence, dtype=np.float32),
        frame_mask=np.asarray(sidecar.frame_mask, dtype=np.float32),
        source_shard_sha256=np.asarray(sidecar.source_shard_sha256),
        source_provenance_hash=np.asarray(sidecar.source_provenance_hash),
        joint_names=np.asarray(SKELETON.names),
        provenance_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return path


def load_detector_loop_sidecar(path: Path) -> DetectorLoopSidecar:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Detector-loop sidecar does not exist: {resolved}")
    with np.load(resolved, allow_pickle=False) as archive:
        forbidden = sorted(
            name
            for name in archive.files
            if any(token in name.lower() for token in FORBIDDEN_PARAMETRIC_BODY_KEYS)
        )
        if forbidden:
            raise ValueError(f"Detector-loop sidecar contains prohibited fields: {forbidden}")
        required = {
            "detector_joints_3d",
            "detector_3d_confidence",
            "frame_mask",
            "source_shard_sha256",
            "source_provenance_hash",
            "joint_names",
            "provenance_json",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Detector-loop sidecar is missing: {', '.join(missing)}")
        names = tuple(str(name) for name in np.asarray(archive["joint_names"]).tolist())
        if names != SKELETON.names:
            raise ValueError("Detector-loop joint order is not canonical")
        provenance = json.loads(str(np.asarray(archive["provenance_json"]).item()))
        sidecar = DetectorLoopSidecar(
            detector_joints_3d=np.asarray(archive["detector_joints_3d"]),
            detector_3d_confidence=np.asarray(archive["detector_3d_confidence"]),
            frame_mask=np.asarray(archive["frame_mask"]),
            source_shard_sha256=str(np.asarray(archive["source_shard_sha256"]).item()),
            source_provenance_hash=str(np.asarray(archive["source_provenance_hash"]).item()),
            provenance=provenance,
        )
    return _validate_sidecar(sidecar)


def index_detector_loop_sidecars(root: Path) -> dict[str, tuple[Path, DetectorLoopSidecar]]:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"Detector-loop root does not exist: {resolved}")
    result: dict[str, tuple[Path, DetectorLoopSidecar]] = {}
    for path in sorted(resolved.rglob(f"*{DETECTOR_LOOP_SUFFIX}")):
        sidecar = load_detector_loop_sidecar(path)
        if sidecar.source_shard_sha256 in result:
            previous = result[sidecar.source_shard_sha256][0]
            raise ValueError(f"Duplicate detector-loop sidecars for one shard: {previous}, {path}")
        result[sidecar.source_shard_sha256] = (path, sidecar)
    return result


def audit_detector_loop_sidecars(
    source_paths: list[Path],
    root: Path,
    *,
    require_all: bool,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    try:
        index = index_detector_loop_sidecars(root)
    except ValueError as error:
        return [str(error)], {"schema_version": DETECTOR_LOOP_VERSION, "sidecars": []}
    sidecars: list[dict[str, Any]] = []
    matched: set[str] = set()
    for source_path in source_paths:
        source_hash = file_sha256(source_path)
        item = index.get(source_hash)
        if item is None:
            if require_all:
                errors.append(f"{source_path}: detector-loop sidecar is required but missing")
            continue
        sidecar_path, sidecar = item
        _, source_provenance = read_shard(source_path)
        if sidecar.source_provenance_hash != source_provenance.get("provenance_hash"):
            errors.append(f"{sidecar_path}: source shard provenance hash does not match")
            continue
        matched.add(source_hash)
        sidecars.append(
            {
                "path": str(sidecar_path.relative_to(root.expanduser().resolve())),
                "sha256": file_sha256(sidecar_path),
                "source_shard_sha256": source_hash,
                "source_provenance_hash": sidecar.source_provenance_hash,
                "detector_world_3d_sha256": sidecar.provenance.get("detector_world_3d_sha256"),
                "frames": sidecar.frames,
            }
        )
    unknown = sorted(set(index) - matched)
    if unknown:
        errors.append(
            f"Detector-loop root contains {len(unknown)} sidecar(s) outside the training BOM"
        )
    return errors, {
        "schema_version": DETECTOR_LOOP_VERSION,
        "source_shards": len(source_paths),
        "covered_source_shards": len(sidecars),
        "coverage_fraction": len(sidecars) / max(len(source_paths), 1),
        "sidecars": sidecars,
    }


def _align_frame_rotation(
    source: np.ndarray, target: np.ndarray, confidence: np.ndarray
) -> np.ndarray:
    visible = confidence > 0
    if int(visible.sum()) < 3:
        return np.eye(3, dtype=np.float32)
    source_visible = source[visible]
    target_visible = target[visible]
    weights = confidence[visible]
    weights = weights / max(float(weights.sum()), 1e-8)
    covariance = (source_visible * weights[:, None]).T @ target_visible
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return rotation.astype(np.float32)


def align_detector_to_targets(
    detector_joints: np.ndarray,
    target_joints: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """Remove render-camera orientation while retaining detector joint errors."""
    detector = np.asarray(detector_joints, dtype=np.float32)
    target = np.asarray(target_joints, dtype=np.float32)
    weights = np.asarray(confidence, dtype=np.float32)
    if detector.shape != target.shape or detector.shape[:2] != weights.shape:
        raise ValueError("Detector/target/confidence shapes do not align")
    detector = detector - detector[:, :1]
    target = target - target[:, :1]
    aligned = np.empty_like(detector)
    previous = np.eye(3, dtype=np.float32)
    for frame in range(len(detector)):
        if int(np.sum(weights[frame] > 0)) >= 3:
            previous = _align_frame_rotation(detector[frame], target[frame], weights[frame])
        aligned[frame] = detector[frame] @ previous
    aligned -= aligned[:, :1]
    return aligned


def build_detector_loop_sidecar(
    source_shard: Path,
    detector_world_3d: Path,
    output_root: Path,
    *,
    catalog_path: Path,
    generator_provenance: dict[str, Any] | None = None,
) -> Path:
    """Create an audited detector prior sidecar from a rendered-shard detector pass."""
    source_path = source_shard.expanduser().resolve()
    detector_path = detector_world_3d.expanduser().resolve()
    arrays, source_provenance = read_shard(source_path)
    source_metadata = dict(source_provenance)
    stored_provenance_hash = source_metadata.pop("provenance_hash", None)
    if stored_provenance_hash != stable_hash(source_metadata):
        raise ValueError("Source shard provenance hash is invalid")
    source_id = str(source_provenance.get("source_id"))
    entry = DatasetCatalog(catalog_path).require_approved(source_id)
    if source_provenance.get("license_id") != entry.license_id:
        raise ValueError("Source shard license does not match the approved catalog entry")
    detector = load_detector_world_3d(detector_path)
    if detector.frames != len(arrays["frame_mask"]):
        raise ValueError("Detector pass and source shard frame counts do not match")
    detector_joints, confidence, _, _ = prepare_detector_world_3d(detector)
    gravity_targets = {name: np.asarray(value).copy() for name, value in arrays.items()}
    convert_targets_to_gravity_view(gravity_targets)
    aligned = align_detector_to_targets(detector_joints, gravity_targets["joints_3d"], confidence)
    source_frame_mask = np.asarray(arrays["frame_mask"], dtype=np.float32)
    aligned[source_frame_mask <= 0] = 0.0
    confidence[source_frame_mask <= 0] = 0.0
    source_hash = file_sha256(source_path)
    source_provenance_hash = str(source_provenance.get("provenance_hash", ""))
    detector_provenance = dict(detector.provenance)
    detector_source_path = detector_provenance.pop("source_path", None)
    if detector_source_path is not None:
        detector_provenance["source_filename"] = Path(str(detector_source_path)).name
    provenance = {
        "detector_loop_version": DETECTOR_LOOP_VERSION,
        "artifact_type": "gravity-mocap-detector-loop-sidecar",
        "scope": "detector-world-3d-prior",
        "source_sequence": source_provenance.get("source_sequence"),
        "source_shard_sha256": source_hash,
        "source_id": source_id,
        "source_provenance_hash": source_provenance_hash,
        "source_provenance": source_provenance,
        "source_sha256": source_provenance.get("source_sha256", []),
        "license_id": entry.license_id,
        "license_url": entry.license_url,
        "attribution_required": entry.attribution_required,
        "requires_acceptance": entry.requires_acceptance,
        "detector_world_3d_filename": detector_path.name,
        "detector_world_3d_sha256": file_sha256(detector_path),
        "detector_provenance": detector_provenance,
        "alignment": "per-frame-rigid-kabsch-no-scale",
        "generator": generator_provenance,
        "frames": len(aligned),
    }
    output = output_root.expanduser().resolve() / f"{source_hash}{DETECTOR_LOOP_SUFFIX}"
    return write_detector_loop_sidecar(
        output,
        DetectorLoopSidecar(
            detector_joints_3d=aligned,
            detector_3d_confidence=confidence,
            frame_mask=source_frame_mask,
            source_shard_sha256=source_hash,
            source_provenance_hash=source_provenance_hash,
            provenance=provenance,
        ),
    )


def _encode_frames(frames: list[Path], output: Path, fps: float) -> Path:
    import cv2

    if not frames:
        raise ValueError("Cannot encode an empty detector-loop render")
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read rendered frame: {frames[0]}")
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.part{output.suffix}")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create detector-loop video: {temporary}")
    try:
        for path in frames:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise RuntimeError(f"Invalid detector-loop render frame: {path}")
            writer.write(image)
    finally:
        writer.release()
    os.replace(temporary, output)
    return output


def detector_loop_generation_plan(
    source_shard: Path,
    output_root: Path,
    *,
    catalog_path: Path,
    data_root: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    source_path = source_shard.expanduser().resolve()
    saved_root = data_root.expanduser().resolve()
    if not source_path.is_relative_to(saved_root):
        raise ValueError("Detector-loop source shards must stay below Saved/GravityMocap")
    arrays, provenance = read_shard(source_path)
    source_metadata = dict(provenance)
    stored_provenance_hash = source_metadata.pop("provenance_hash", None)
    if stored_provenance_hash != stable_hash(source_metadata):
        raise ValueError("Source shard provenance hash is invalid")
    entry = DatasetCatalog(catalog_path).require_approved(str(provenance.get("source_id")))
    if provenance.get("license_id") != entry.license_id:
        raise ValueError("Source shard license does not match the approved catalog entry")
    fps = float(provenance.get("fps", 30.0))
    if not np.isclose(fps, 30.0):
        raise ValueError("Detector-loop generation requires canonical 30 FPS shards")
    root = output_root.expanduser().resolve()
    if not root.is_relative_to(saved_root):
        raise ValueError("Detector-loop outputs must stay below Saved/GravityMocap")
    if width < 256 or height < 256:
        raise ValueError("Detector-loop renders must be at least 256x256")
    source_hash = file_sha256(source_path)
    return {
        "source_shard": str(source_path),
        "source_shard_sha256": source_hash,
        "source_id": entry.dataset_id,
        "license_id": entry.license_id,
        "frames": len(arrays["frame_mask"]),
        "fps": fps,
        "output_root": str(root),
        "sidecar": str(root / f"{source_hash}{DETECTOR_LOOP_SUFFIX}"),
        "render_size": [width, height],
    }


def select_detector_loop_candidates(
    candidates: list[DetectorLoopBatchCandidate],
    *,
    coverage_fraction: float,
    selection_seed: int,
) -> list[DetectorLoopBatchCandidate]:
    """Select a stable ceil-per-source sample and interleave sources for partial runs."""
    if not 0 < coverage_fraction <= 1:
        raise ValueError("Detector-loop batch coverage must be in (0, 1]")
    if not candidates:
        raise ValueError("Detector-loop batch inventory is empty")
    relative_paths = [candidate.relative_path for candidate in candidates]
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("Detector-loop batch inventory contains duplicate relative paths")

    grouped: dict[str, list[tuple[str, DetectorLoopBatchCandidate]]] = {}
    for candidate in candidates:
        identity = (
            f"detector-loop-batch-v{DETECTOR_LOOP_BATCH_VERSION}:"
            f"{selection_seed}:{candidate.source_id}:{candidate.relative_path}"
        )
        rank = sha256(identity.encode("utf-8")).hexdigest()
        grouped.setdefault(candidate.source_id, []).append((rank, candidate))

    selected_by_source: dict[str, list[DetectorLoopBatchCandidate]] = {}
    for source_id, ranked in grouped.items():
        ranked.sort(key=lambda item: (item[0], item[1].relative_path))
        count = ceil(len(ranked) * coverage_fraction)
        chosen = [candidate for _, candidate in ranked[:count]]
        # Membership remains hash-random; processing shorter selected shards first
        # keeps a smoke or interrupted preparation session usefully bounded.
        selected_by_source[source_id] = sorted(
            chosen, key=lambda candidate: (candidate.frames, candidate.relative_path)
        )

    # A bounded session should not accidentally process one source exclusively.
    interleaved: list[DetectorLoopBatchCandidate] = []
    max_items = max(len(items) for items in selected_by_source.values())
    for offset in range(max_items):
        for source_id in sorted(selected_by_source):
            items = selected_by_source[source_id]
            if offset < len(items):
                interleaved.append(items[offset])
    return interleaved


def _batch_candidate(path: Path, processed_root: Path) -> DetectorLoopBatchCandidate:
    with np.load(path, allow_pickle=False) as archive:
        required = {"frame_mask", "provenance_json"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{path}: training shard is missing: {', '.join(missing)}")
        forbidden = sorted(
            name
            for name in archive.files
            if any(token in name.lower() for token in FORBIDDEN_PARAMETRIC_BODY_KEYS)
        )
        if forbidden:
            raise ValueError(f"{path}: training shard contains prohibited fields: {forbidden}")
        provenance = json.loads(str(np.asarray(archive["provenance_json"]).item()))
        frames = int(archive["frame_mask"].shape[0])
    metadata = dict(provenance)
    stored_provenance_hash = metadata.pop("provenance_hash", None)
    if stored_provenance_hash != stable_hash(metadata):
        raise ValueError(f"{path}: source shard provenance hash is invalid")
    if frames < 2:
        raise ValueError(f"{path}: source shard needs at least two frames")
    fps = float(provenance.get("fps", 30.0))
    if not np.isclose(fps, 30.0):
        raise ValueError(f"{path}: detector-loop generation requires canonical 30 FPS shards")
    return DetectorLoopBatchCandidate(
        relative_path=path.relative_to(processed_root).as_posix(),
        source_id=str(provenance.get("source_id")),
        frames=frames,
        source_provenance_hash=str(stored_provenance_hash),
    )


def _compatible_generated_sidecar(
    sidecar: DetectorLoopSidecar,
    candidate: DetectorLoopBatchCandidate,
    *,
    width: int,
    height: int,
) -> bool:
    generator = sidecar.provenance.get("generator") or {}
    return bool(
        sidecar.source_provenance_hash == candidate.source_provenance_hash
        and sidecar.frames == candidate.frames
        and generator.get("version") == DETECTOR_LOOP_GENERATOR_VERSION
        and generator.get("render_width") == width
        and generator.get("render_height") == height
    )


def detector_loop_batch_plan(
    processed_root: Path,
    output_root: Path,
    *,
    catalog_path: Path,
    data_root: Path,
    profile: str = "core",
    coverage_fraction: float = 0.1,
    selection_seed: int = 8088,
    width: int = 512,
    height: int = 512,
) -> dict[str, Any]:
    """Build a read-only, source-stratified plan for detector-loop sidecars."""
    saved_root = data_root.expanduser().resolve()
    processed = processed_root.expanduser().resolve()
    root = output_root.expanduser().resolve()
    if not processed.is_dir() or not processed.is_relative_to(saved_root):
        raise ValueError("Detector-loop processed root must exist below Saved/GravityMocap")
    if not root.is_relative_to(saved_root):
        raise ValueError("Detector-loop outputs must stay below Saved/GravityMocap")
    if width < 256 or height < 256:
        raise ValueError("Detector-loop renders must be at least 256x256")
    if profile != "core":
        raise ValueError("Detector-loop batches are restricted to the official core profile")

    catalog = DatasetCatalog(catalog_path)
    entries = {entry.dataset_id: entry for entry in catalog.profile(profile)}
    candidates: list[DetectorLoopBatchCandidate] = []
    skipped_by_source: dict[str, int] = {}
    for path in sorted(processed.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if "provenance_json" not in archive.files:
                raise ValueError(f"{path}: training shard is missing provenance_json")
            provenance = json.loads(str(np.asarray(archive["provenance_json"]).item()))
        source_id = str(provenance.get("source_id"))
        if source_id not in entries:
            skipped_by_source[source_id] = skipped_by_source.get(source_id, 0) + 1
            continue
        candidate = _batch_candidate(path, processed)
        entry = entries[candidate.source_id]
        if provenance.get("license_id") != entry.license_id:
            raise ValueError(f"{path}: source shard license does not match the approved catalog")
        candidates.append(candidate)

    selected = select_detector_loop_candidates(
        candidates,
        coverage_fraction=coverage_fraction,
        selection_seed=selection_seed,
    )
    sidecar_index = index_detector_loop_sidecars(root) if root.is_dir() else {}
    items: list[dict[str, Any]] = []
    source_hashes: set[str] = set()
    for candidate in selected:
        source_path = processed / candidate.relative_path
        source_hash = file_sha256(source_path)
        if source_hash in source_hashes:
            raise ValueError("Detector-loop batch selection contains duplicate shard content")
        source_hashes.add(source_hash)
        indexed = sidecar_index.get(source_hash)
        status = "pending"
        if indexed is not None:
            status = (
                "ready"
                if _compatible_generated_sidecar(
                    indexed[1], candidate, width=width, height=height
                )
                else "stale"
            )
        items.append(
            {
                "source_shard": candidate.relative_path,
                "source_shard_sha256": source_hash,
                "source_provenance_hash": candidate.source_provenance_hash,
                "source_id": candidate.source_id,
                "frames": candidate.frames,
                "sidecar": f"{source_hash}{DETECTOR_LOOP_SUFFIX}",
                "status": status,
            }
        )

    inventory_by_source: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        stats = inventory_by_source.setdefault(candidate.source_id, {"shards": 0, "frames": 0})
        stats["shards"] += 1
        stats["frames"] += candidate.frames
    selected_by_source: dict[str, dict[str, int]] = {}
    for item in items:
        stats = selected_by_source.setdefault(
            str(item["source_id"]), {"shards": 0, "frames": 0, "ready": 0}
        )
        stats["shards"] += 1
        stats["frames"] += int(item["frames"])
        stats["ready"] += int(item["status"] == "ready")
    pending = [item for item in items if item["status"] != "ready"]
    return {
        "schema_version": DETECTOR_LOOP_BATCH_VERSION,
        "artifact_type": "gravity-mocap-detector-loop-batch-plan",
        "detector_loop_version": DETECTOR_LOOP_VERSION,
        "detector_loop_generator_version": DETECTOR_LOOP_GENERATOR_VERSION,
        "profile": profile,
        "catalog_sha256": file_sha256(catalog_path.expanduser().resolve()),
        "processed_root": processed.relative_to(saved_root).as_posix(),
        "output_root": root.relative_to(saved_root).as_posix(),
        "coverage_fraction": float(coverage_fraction),
        "selection_seed": int(selection_seed),
        "selection_algorithm": "sha256-rank-ceil-per-source-interleaved-v1",
        "render_size": [width, height],
        "inventory_shards": len(candidates),
        "inventory_frames": sum(candidate.frames for candidate in candidates),
        "inventory_by_source": inventory_by_source,
        "skipped_by_source": skipped_by_source,
        "selected_shards": len(items),
        "selected_frames": sum(int(item["frames"]) for item in items),
        "selected_by_source": selected_by_source,
        "ready_shards": sum(item["status"] == "ready" for item in items),
        "remaining_shards": len(pending),
        "remaining_frames": sum(int(item["frames"]) for item in pending),
        "peak_transient_frames": max((int(item["frames"]) for item in pending), default=0),
        "items": items,
    }


def detector_loop_batch_summary(plan: dict[str, Any]) -> dict[str, Any]:
    summary = {key: value for key, value in plan.items() if key != "items"}
    summary["selected_examples"] = [
        {
            "source_shard": item["source_shard"],
            "source_id": item["source_id"],
            "frames": item["frames"],
            "status": item["status"],
        }
        for item in plan["items"][:6]
    ]
    return summary


def _batch_progress(manifest: dict[str, Any]) -> tuple[int, int]:
    ready = sum(item["status"] == "ready" for item in manifest["items"])
    return ready, len(manifest["items"])


def _refresh_batch_counts(manifest: dict[str, Any]) -> tuple[int, int]:
    ready, total = _batch_progress(manifest)
    manifest["ready_shards"] = ready
    manifest["remaining_shards"] = total - ready
    manifest["remaining_frames"] = sum(
        int(item["frames"]) for item in manifest["items"] if item["status"] != "ready"
    )
    ready_by_source: dict[str, int] = {}
    for item in manifest["items"]:
        if item["status"] == "ready":
            source_id = str(item["source_id"])
            ready_by_source[source_id] = ready_by_source.get(source_id, 0) + 1
    for source_id, stats in manifest["selected_by_source"].items():
        stats["ready"] = ready_by_source.get(source_id, 0)
    return ready, total


def generate_detector_loop_batch(
    processed_root: Path,
    output_root: Path,
    *,
    catalog_path: Path,
    data_root: Path,
    profile: str = "core",
    coverage_fraction: float = 0.1,
    selection_seed: int = 8088,
    width: int = 512,
    height: int = 512,
    max_hours: float | None = None,
    max_shards: int | None = None,
    keep_work: bool = False,
) -> dict[str, Any]:
    """Generate a resumable batch, enforcing the time limit between atomic sidecars."""
    if max_hours is not None and max_hours <= 0:
        raise ValueError("Detector-loop batch max_hours must be positive")
    if max_shards is not None and max_shards <= 0:
        raise ValueError("Detector-loop batch max_shards must be positive")
    plan = detector_loop_batch_plan(
        processed_root,
        output_root,
        catalog_path=catalog_path,
        data_root=data_root,
        profile=profile,
        coverage_fraction=coverage_fraction,
        selection_seed=selection_seed,
        width=width,
        height=height,
    )
    processed = processed_root.expanduser().resolve()
    root = output_root.expanduser().resolve()
    manifest_path = root / DETECTOR_LOOP_BATCH_MANIFEST
    manifest = {
        **plan,
        "artifact_type": "gravity-mocap-detector-loop-batch-manifest",
        "status": "running",
        "session_generated_shards": 0,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(manifest_path, manifest)
    started = time.monotonic()
    generated = 0
    stop_reason: str | None = None
    for index, item in enumerate(manifest["items"], start=1):
        if item["status"] == "ready":
            if not keep_work:
                shutil.rmtree(root / "work" / item["source_shard_sha256"], ignore_errors=True)
            continue
        if max_shards is not None and generated >= max_shards:
            stop_reason = "max_shards"
            break
        if max_hours is not None and time.monotonic() - started >= max_hours * 3600:
            stop_reason = "max_hours"
            break
        ready, total = _batch_progress(manifest)
        print(
            f"[detector-loop] item {index}/{total} | ready {ready}/{total} | "
            f"source {item['source_id']} | frames {item['frames']} | "
            f"{item['source_shard']}",
            flush=True,
        )
        source_path = processed / item["source_shard"]
        try:
            sidecar_path = generate_detector_loop_sidecar(
                source_path,
                root,
                catalog_path=catalog_path,
                data_root=data_root,
                width=width,
                height=height,
            )
            sidecar = load_detector_loop_sidecar(sidecar_path)
            candidate = DetectorLoopBatchCandidate(
                relative_path=str(item["source_shard"]),
                source_id=str(item["source_id"]),
                frames=int(item["frames"]),
                source_provenance_hash=str(item["source_provenance_hash"]),
            )
            if sidecar.source_shard_sha256 != item["source_shard_sha256"]:
                raise ValueError("Generated detector-loop sidecar is bound to the wrong shard")
            if not _compatible_generated_sidecar(
                sidecar, candidate, width=width, height=height
            ):
                raise ValueError("Generated detector-loop sidecar is stale or incompatible")
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            item["status"] = "failed"
            manifest["status"] = "failed"
            manifest["last_error"] = str(error)
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            write_json_atomic(manifest_path, manifest)
            raise
        item["status"] = "ready"
        generated += 1
        manifest["session_generated_shards"] = generated
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        if not keep_work:
            shutil.rmtree(root / "work" / item["source_shard_sha256"], ignore_errors=True)
        write_json_atomic(manifest_path, manifest)

    ready, total = _refresh_batch_counts(manifest)
    if ready == total:
        manifest["status"] = "complete"
    else:
        manifest["status"] = stop_reason or "incomplete"
    manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    write_json_atomic(manifest_path, manifest)
    return {
        **detector_loop_batch_summary(manifest),
        "manifest": str(manifest_path),
    }


def generate_detector_loop_sidecar(
    source_shard: Path,
    output_root: Path,
    *,
    catalog_path: Path,
    data_root: Path,
    width: int = 512,
    height: int = 512,
) -> Path:
    """Render an approved shard, run the pinned detector, and bind its prior to the shard."""
    plan = detector_loop_generation_plan(
        source_shard,
        output_root,
        catalog_path=catalog_path,
        data_root=data_root,
        width=width,
        height=height,
    )
    source_path = Path(plan["source_shard"])
    arrays, _ = read_shard(source_path)
    fps = float(plan["fps"])
    source_hash = str(plan["source_shard_sha256"])
    root = Path(plan["output_root"])
    work = root / "work" / source_hash
    sidecar_path = root / f"{source_hash}{DETECTOR_LOOP_SUFFIX}"
    if sidecar_path.is_file():
        try:
            cached = load_detector_loop_sidecar(sidecar_path)
            generator = cached.provenance.get("generator") or {}
            if generator.get("version") == DETECTOR_LOOP_GENERATOR_VERSION:
                if (
                    generator.get("render_width") == width
                    and generator.get("render_height") == height
                ):
                    return sidecar_path
        except (ValueError, OSError):
            pass

    frames = render_mesh_avatar_frames(
        np.asarray(arrays["joints_3d"], dtype=np.float32),
        work / "frames",
        width=width,
        height=height,
        fps=fps,
    )
    video = _encode_frames(frames, work / "render.mp4", fps)
    detection = video_to_rig(
        video,
        work / "detection",
        data_root=data_root,
        target_fps=fps,
        preview=False,
    )
    return build_detector_loop_sidecar(
        source_path,
        Path(detection["detector_world_3d"]),
        root,
        catalog_path=catalog_path,
        generator_provenance={
            "version": DETECTOR_LOOP_GENERATOR_VERSION,
            "renderer": "bundled-gray-quaternius-mesh",
            "avatar": avatar_provenance(),
            "render_width": width,
            "render_height": height,
            "rendered_video_sha256": file_sha256(video),
            "video_frontend_request_hash": detection.get("request_hash"),
        },
    )
