from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .skeleton import SKELETON

AVATAR_ASSET_NAME = "Quaternius Universal Base Characters - Superhero Male (gray)"
AVATAR_ASSET_FILENAME = "quaternius_gray_man.glb"
AVATAR_ASSET_SHA256 = "080bd8d01037e188129b86930ed424316cb912d64f7967504bd22b7e1c1bd8e3"
AVATAR_ASSET_LICENSE = "CC0-1.0"
AVATAR_ASSET_SOURCE = "https://quaternius.com/packs/universalbasecharacters.html"
AVATAR_RENDERERS = ("auto", "mesh", "procedural")


def avatar_asset_path() -> Path:
    return Path(__file__).with_name("assets") / AVATAR_ASSET_FILENAME


def avatar_worker_path() -> Path:
    return Path(__file__).with_name("_blender_avatar.py")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_avatar_asset() -> Path:
    path = avatar_asset_path()
    if not path.is_file():
        raise RuntimeError(f"Bundled avatar asset is missing: {path}")
    actual_hash = file_sha256(path)
    if actual_hash != AVATAR_ASSET_SHA256:
        raise RuntimeError(
            f"Bundled avatar checksum mismatch: expected {AVATAR_ASSET_SHA256}, got {actual_hash}"
        )
    return path


def resolve_avatar_renderer(renderer: str) -> str:
    if renderer not in AVATAR_RENDERERS:
        raise ValueError(
            f"Unknown avatar renderer {renderer!r}; expected one of {', '.join(AVATAR_RENDERERS)}"
        )
    if renderer == "procedural":
        return renderer
    validate_avatar_asset()
    blender = shutil.which("blender")
    if renderer == "mesh" and blender is None:
        raise RuntimeError(
            "The mesh avatar renderer requires Blender on PATH; "
            "use --avatar-renderer procedural for the lightweight fallback"
        )
    return "mesh" if blender is not None else "procedural"


def avatar_provenance() -> dict[str, str]:
    return {
        "name": AVATAR_ASSET_NAME,
        "filename": AVATAR_ASSET_FILENAME,
        "sha256": AVATAR_ASSET_SHA256,
        "license_id": AVATAR_ASSET_LICENSE,
        "source_url": AVATAR_ASSET_SOURCE,
    }


def render_mesh_avatar_frames(
    joints_3d: np.ndarray,
    output_dir: Path,
    *,
    width: int,
    height: int,
    fps: float,
) -> list[Path]:
    joints = np.asarray(joints_3d, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[1:] != (SKELETON.joint_count, 3):
        raise ValueError(
            f"Expected joints shaped (T, {SKELETON.joint_count}, 3), got {joints.shape}"
        )
    if len(joints) == 0 or not np.isfinite(joints).all():
        raise ValueError("Avatar motion must contain finite joints and at least one frame")
    if width < 64 or height < 64 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Avatar render dimensions and FPS must be positive")

    blender = shutil.which("blender")
    if blender is None:
        raise RuntimeError("The mesh avatar renderer requires Blender on PATH")
    asset = validate_avatar_asset()
    worker = avatar_worker_path()
    if not worker.is_file():
        raise RuntimeError(f"Bundled Blender avatar worker is missing: {worker}")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    motion_path = output_dir / "avatar-motion.npz"
    with motion_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            joints_3d=joints,
            joint_names=np.asarray(SKELETON.names),
        )
    command = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(worker),
        "--",
        "--asset",
        str(asset),
        "--motion",
        str(motion_path),
        "--output-dir",
        str(output_dir),
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(float(fps)),
    ]
    print(
        f"[avatar] rendering {len(joints)} frame(s) with the bundled gray Quaternius mesh",
        flush=True,
    )
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-30:])
        raise RuntimeError(f"Blender avatar render failed:\n{tail}")
    frames = sorted(output_dir.glob("frame-*.png"))
    if len(frames) != len(joints):
        raise RuntimeError(
            f"Blender avatar render produced {len(frames)} frame(s), expected {len(joints)}"
        )
    print(f"[avatar] rendered {len(frames)} gray mesh frame(s)", flush=True)
    return frames
