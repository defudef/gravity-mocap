"""Build the redistributable gray Quaternius avatar used by preview rendering.

Run with Blender, not the project Python interpreter:

    blender --background --python scripts/build_quaternius_avatar.py -- \
      --input /path/to/Superhero_Male_FullBody.gltf \
      --output src/gravity_mocap/assets/quaternius_gray_man.glb
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import bpy

SOURCE_GLTF_SHA256 = "e7fcea214ecf8855afbf910b50de6f9c7d1decfb71ca28bad8a4481452dafeb4"
SOURCE_BIN_SHA256 = "459003f9745853ae562a85506a2b94dd56515c1f37728f9fa3d2ce1a3e4cd92f"
GRAY_RGBA = (0.30, 0.32, 0.35, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(arguments)


def _validate_source(source: Path) -> None:
    if source.name != "Superhero_Male_FullBody.gltf":
        raise RuntimeError(f"Unexpected Quaternius source filename: {source.name}")
    binary = source.with_suffix(".bin")
    expected = ((source, SOURCE_GLTF_SHA256), (binary, SOURCE_BIN_SHA256))
    for path, expected_hash in expected:
        if not path.is_file():
            raise RuntimeError(f"Missing Quaternius source file: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Quaternius source checksum mismatch for {path.name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )


def _reset_scene() -> None:
    for item in list(bpy.data.objects):
        bpy.data.objects.remove(item, do_unlink=True)
    for collection in (
        bpy.data.armatures,
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.actions,
    ):
        for item in list(collection):
            collection.remove(item)


def _gray_material() -> bpy.types.Material:
    material = bpy.data.materials.new("GravityMocap_MatteGray")
    material.diffuse_color = GRAY_RGBA
    material.metallic = 0.0
    material.roughness = 0.78
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is None:
        raise RuntimeError("Blender did not create a Principled BSDF node")
    principled.inputs["Base Color"].default_value = GRAY_RGBA
    metallic = principled.inputs.get("Metallic") or principled.inputs.get("Metallic IOR Level")
    if metallic is None:
        raise RuntimeError("Blender Principled BSDF node has no metallic input")
    metallic.default_value = 0.0
    principled.inputs["Roughness"].default_value = 0.78
    return material


def _prepare_avatar(source: Path) -> None:
    bpy.ops.import_scene.gltf(filepath=str(source))
    armatures = [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]
    meshes = [
        item
        for item in bpy.context.scene.objects
        if item.type == "MESH" and any(modifier.type == "ARMATURE" for modifier in item.modifiers)
    ]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one armature, found {len(armatures)}")
    if sorted(item.name for item in meshes) != ["Eyebrows", "Eyes", "SuperHero_Male"]:
        raise RuntimeError(f"Unexpected Quaternius meshes: {[item.name for item in meshes]}")

    for item in list(bpy.context.scene.objects):
        if item.type == "MESH" and item not in meshes:
            bpy.data.objects.remove(item, do_unlink=True)

    material = _gray_material()
    for mesh in meshes:
        mesh.data.materials.clear()
        mesh.data.materials.append(material)

    armatures[0].name = "GravityMocap_Armature"
    armatures[0].data.name = "GravityMocap_Armature"
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    for image in list(bpy.data.images):
        bpy.data.images.remove(image)


def main() -> None:
    args = _arguments()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    _validate_source(source)
    _reset_scene()
    _prepare_avatar(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        export_animations=False,
        export_skins=True,
        export_materials="EXPORT",
        export_yup=True,
    )
    print(f"Built {output}")
    print(f"SHA-256 {_sha256(output)}")


if __name__ == "__main__":
    main()
