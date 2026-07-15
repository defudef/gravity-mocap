from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import FORBIDDEN_PARAMETRIC_BODY_KEYS
from .artifacts import write_npz_atomic
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
