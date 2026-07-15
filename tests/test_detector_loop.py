from pathlib import Path

import numpy as np

from gravity_mocap.data import MotionWindowDataset
from gravity_mocap.detector_loop import (
    DETECTOR_LOOP_VERSION,
    DetectorLoopSidecar,
    align_detector_to_targets,
    audit_detector_loop_sidecars,
    load_detector_loop_sidecar,
    write_detector_loop_sidecar,
)
from gravity_mocap.fixture import create_fixture
from gravity_mocap.schema import read_shard
from gravity_mocap.video import file_sha256


def test_detector_loop_round_trip_and_dataset_override(tmp_path: Path) -> None:
    shard = create_fixture(tmp_path / "processed/synthetic/walk.npz", frames=8)
    arrays, source_provenance = read_shard(shard)
    source_hash = file_sha256(shard)
    detector = np.asarray(arrays["joints_3d"], dtype=np.float32).copy()
    detector[:, 1:, 0] += 0.02
    confidence = np.full(detector.shape[:2], 0.7, dtype=np.float32)
    loop_root = tmp_path / "detector-loop"
    sidecar_path = write_detector_loop_sidecar(
        loop_root / f"{source_hash}.detector-loop.npz",
        DetectorLoopSidecar(
            detector_joints_3d=detector,
            detector_3d_confidence=confidence,
            frame_mask=np.ones(8, dtype=np.float32),
            source_shard_sha256=source_hash,
            source_provenance_hash=source_provenance["provenance_hash"],
            provenance={
                "detector_loop_version": DETECTOR_LOOP_VERSION,
                "artifact_type": "gravity-mocap-detector-loop-sidecar",
                "source_shard_sha256": source_hash,
                "source_provenance_hash": source_provenance["provenance_hash"],
            },
        ),
    )

    loaded = load_detector_loop_sidecar(sidecar_path)
    errors, bom = audit_detector_loop_sidecars([shard], loop_root, require_all=True)
    dataset = MotionWindowDataset(
        tmp_path / "processed",
        sequence_length=8,
        stride=8,
        gravity_view_contract=True,
        detector_world_3d={"enabled": True},
        detector_loop={
            "enabled": True,
            "root": str(loop_root),
            "mix_probability": 1.0,
            "require_all": True,
        },
    )
    sample = dataset[0]

    assert np.array_equal(loaded.detector_joints_3d, detector)
    assert errors == []
    assert bom["coverage_fraction"] == 1.0
    assert bom["sidecars"][0]["source_shard_sha256"] == source_hash
    assert np.allclose(sample["detector_joints_3d"].numpy(), detector)
    assert np.allclose(sample["detector_3d_confidence"].numpy(), confidence)


def test_detector_alignment_removes_only_rigid_frame_rotation() -> None:
    target = np.zeros((2, 22, 3), dtype=np.float32)
    target[:, 1, 0] = 1.0
    target[:, 2, 1] = 1.0
    target[:, 3, 2] = 1.0
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    detector = target @ rotation
    confidence = np.ones((2, 22), dtype=np.float32)

    aligned = align_detector_to_targets(detector, target, confidence)

    assert np.allclose(aligned, target, atol=1e-6)
