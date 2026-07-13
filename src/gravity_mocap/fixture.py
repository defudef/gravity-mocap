from __future__ import annotations

from pathlib import Path

import numpy as np

from .preprocess import motion_to_arrays
from .schema import write_shard
from .skeleton import PARENTS, REST_OFFSETS


def create_fixture(path: Path, frames: int = 16, image_feature_dim: int = 32) -> Path:
    rest = np.zeros_like(REST_OFFSETS)
    for joint, parent in enumerate(PARENTS):
        rest[joint] = REST_OFFSETS[joint] if parent < 0 else rest[parent] + REST_OFFSETS[joint]
    time = np.linspace(0, 2 * np.pi, frames, endpoint=False, dtype=np.float32)
    positions = np.repeat(rest[None], frames, axis=0)
    positions[:, :, 0] += np.linspace(0, 0.4, frames, dtype=np.float32)[:, None]
    positions[:, 20, 1] += 0.05 * np.sin(time)
    positions[:, 21, 1] -= 0.05 * np.sin(time)
    positions[:, 10, 2] += 0.03 * np.sin(time)
    positions[:, 11, 2] -= 0.03 * np.sin(time)
    arrays = motion_to_arrays(positions, fps=30.0, image_feature_dim=image_feature_dim, seed=7)
    write_shard(
        path,
        arrays,
        {
            "source_id": "synthetic_fixture",
            "source_title": "Procedural smoke-test fixture",
            "source_sequence": "walk-cycle-001",
            "source_sha256": [],
            "license_id": "CC0-1.0",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "attribution_required": False,
            "converter_version": "fixture-v1",
            "preprocess_hash": "synthetic",
            "fps": 30.0,
            "input_mode": "motion_only",
        },
    )
    return path
