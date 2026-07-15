from pathlib import Path

import numpy as np
import yaml

from gravity_mocap.data import MotionWindowDataset
from gravity_mocap.detector_loop import (
    DETECTOR_LOOP_GENERATOR_VERSION,
    DETECTOR_LOOP_VERSION,
    DetectorLoopBatchCandidate,
    DetectorLoopSidecar,
    align_detector_to_targets,
    audit_detector_loop_sidecars,
    detector_loop_batch_plan,
    load_detector_loop_sidecar,
    select_detector_loop_candidates,
    write_detector_loop_sidecar,
)
from gravity_mocap.fixture import create_fixture
from gravity_mocap.schema import read_shard, write_shard
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


def test_detector_loop_batch_selection_is_stable_and_source_stratified() -> None:
    candidates = [
        DetectorLoopBatchCandidate(f"a/{index}.npz", "a", index + 2, "a" * 64)
        for index in range(10)
    ]
    candidates += [
        DetectorLoopBatchCandidate(f"b/{index}.npz", "b", index + 2, "b" * 64)
        for index in range(6)
    ]
    candidates += [
        DetectorLoopBatchCandidate(f"c/{index}.npz", "c", index + 2, "c" * 64)
        for index in range(3)
    ]

    selected = select_detector_loop_candidates(
        candidates,
        coverage_fraction=0.2,
        selection_seed=8088,
    )
    reordered = select_detector_loop_candidates(
        list(reversed(candidates)),
        coverage_fraction=0.2,
        selection_seed=8088,
    )

    assert [item.relative_path for item in selected] == [
        item.relative_path for item in reordered
    ]
    assert [item.source_id for item in selected[:3]] == ["a", "b", "c"]
    assert sum(item.source_id == "a" for item in selected) == 2
    assert sum(item.source_id == "b" for item in selected) == 2
    assert sum(item.source_id == "c" for item in selected) == 1


def test_detector_loop_batch_plan_reuses_only_compatible_sidecars(tmp_path: Path) -> None:
    data_root = tmp_path / "Saved/GravityMocap"
    processed_root = data_root / "processed"
    for source_id, count in (("a", 3), ("b", 2)):
        for index in range(count):
            shard = create_fixture(
                processed_root / source_id / f"{index}.npz",
                frames=6 + index,
            )
            arrays, provenance = read_shard(shard)
            provenance.update(
                {
                    "source_id": source_id,
                    "source_sequence": f"{source_id}/{index}",
                    "license_id": "CC0-1.0",
                }
            )
            write_shard(shard, arrays, provenance)
    catalog_path = tmp_path / "datasets.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "profiles": {"core": {"datasets": ["a", "b"]}},
                "datasets": {
                    source_id: {
                        "title": source_id,
                        "task": "motion",
                        "license_id": "CC0-1.0",
                        "license_url": "https://example.com/license",
                        "approved_for_training": True,
                        "attribution_required": False,
                        "requires_acceptance": False,
                        "downloader": {"type": "fixture"},
                    }
                    for source_id in ("a", "b")
                },
            }
        ),
        encoding="utf-8",
    )
    loop_root = data_root / "detector-loop"
    initial = detector_loop_batch_plan(
        processed_root,
        loop_root,
        catalog_path=catalog_path,
        data_root=data_root,
        coverage_fraction=0.5,
        selection_seed=7,
        width=256,
        height=256,
    )
    selected = initial["items"][0]
    source_path = processed_root / selected["source_shard"]
    arrays, provenance = read_shard(source_path)
    source_hash = file_sha256(source_path)
    confidence = np.ones(np.asarray(arrays["joints_3d"]).shape[:2], dtype=np.float32)
    write_detector_loop_sidecar(
        loop_root / selected["sidecar"],
        DetectorLoopSidecar(
            detector_joints_3d=np.asarray(arrays["joints_3d"], dtype=np.float32),
            detector_3d_confidence=confidence,
            frame_mask=np.asarray(arrays["frame_mask"], dtype=np.float32),
            source_shard_sha256=source_hash,
            source_provenance_hash=provenance["provenance_hash"],
            provenance={
                "detector_loop_version": DETECTOR_LOOP_VERSION,
                "artifact_type": "gravity-mocap-detector-loop-sidecar",
                "source_shard_sha256": source_hash,
                "source_provenance_hash": provenance["provenance_hash"],
                "generator": {
                    "version": DETECTOR_LOOP_GENERATOR_VERSION,
                    "render_width": 256,
                    "render_height": 256,
                },
            },
        ),
    )

    resumed = detector_loop_batch_plan(
        processed_root,
        loop_root,
        catalog_path=catalog_path,
        data_root=data_root,
        coverage_fraction=0.5,
        selection_seed=7,
        width=256,
        height=256,
    )

    assert initial["selected_shards"] == 3
    assert initial["ready_shards"] == 0
    assert resumed["ready_shards"] == 1
    assert resumed["remaining_shards"] == 2
    assert resumed["items"][0]["status"] == "ready"
