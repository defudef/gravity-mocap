from pathlib import Path

import pytest

from gravity_mocap import mesh_avatar


def test_bundled_gray_avatar_has_pinned_cc0_provenance() -> None:
    path = mesh_avatar.validate_avatar_asset()
    provenance = mesh_avatar.avatar_provenance()

    assert path.name == "quaternius_gray_man.glb"
    assert path.stat().st_size > 100_000
    assert provenance == {
        "name": "Quaternius Universal Base Characters - Superhero Male (gray)",
        "filename": "quaternius_gray_man.glb",
        "sha256": "080bd8d01037e188129b86930ed424316cb912d64f7967504bd22b7e1c1bd8e3",
        "license_id": "CC0-1.0",
        "source_url": "https://quaternius.com/packs/universalbasecharacters.html",
    }


def test_avatar_renderer_selection_is_explicit_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mesh_avatar.shutil, "which", lambda _: None)

    assert mesh_avatar.resolve_avatar_renderer("auto") == "procedural"
    assert mesh_avatar.resolve_avatar_renderer("procedural") == "procedural"
    with pytest.raises(RuntimeError, match="requires Blender"):
        mesh_avatar.resolve_avatar_renderer("mesh")
    with pytest.raises(ValueError, match="Unknown avatar renderer"):
        mesh_avatar.resolve_avatar_renderer("rainbow")


def test_avatar_checksum_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corrupted = tmp_path / "quaternius_gray_man.glb"
    corrupted.write_bytes(b"not the pinned asset")
    monkeypatch.setattr(mesh_avatar, "avatar_asset_path", lambda: corrupted)

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        mesh_avatar.validate_avatar_asset()
