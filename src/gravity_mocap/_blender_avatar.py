"""Blender-side worker for rendering the bundled neutral gray avatar.

This module is executed by Blender in a subprocess. The regular Gravity Mocap
Python runtime must not import it because ``bpy`` is provided only by Blender.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

JOINT_NAMES = (
    "root",
    "left_hip",
    "right_hip",
    "spine_1",
    "left_knee",
    "right_knee",
    "spine_2",
    "left_ankle",
    "right_ankle",
    "spine_3",
    "left_toe",
    "right_toe",
    "neck",
    "left_clavicle",
    "right_clavicle",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}

# Quaternius' armature has the same major anatomical chains as Gravity Mocap.
# The extra clavicle marker in the canonical rig is intentionally folded into
# the shoulder target because the asset has one clavicle bone per side.
BONE_TARGETS = (
    ("pelvis", "root", "spine_1"),
    ("spine_01", "spine_1", "spine_2"),
    ("spine_02", "spine_2", "spine_3"),
    ("spine_03", "spine_3", "neck"),
    ("neck_01", "neck", "head"),
    ("clavicle_l", "neck", "left_shoulder"),
    ("upperarm_l", "left_shoulder", "left_elbow"),
    ("lowerarm_l", "left_elbow", "left_wrist"),
    ("clavicle_r", "neck", "right_shoulder"),
    ("upperarm_r", "right_shoulder", "right_elbow"),
    ("lowerarm_r", "right_elbow", "right_wrist"),
    ("thigh_l", "left_hip", "left_knee"),
    ("calf_l", "left_knee", "left_ankle"),
    ("foot_l", "left_ankle", "left_toe"),
    ("thigh_r", "right_hip", "right_knee"),
    ("calf_r", "right_knee", "right_ankle"),
    ("foot_r", "right_ankle", "right_toe"),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(values)


def _reset_scene() -> None:
    for item in list(bpy.data.objects):
        bpy.data.objects.remove(item, do_unlink=True)
    for collection in (bpy.data.armatures, bpy.data.meshes, bpy.data.materials, bpy.data.actions):
        for item in list(collection):
            collection.remove(item)


def _load_motion(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_blender"], dtype=np.float32)
        names = tuple(str(name) for name in archive["joint_names"].tolist())
    if joints.ndim != 3 or joints.shape[1:] != (len(JOINT_NAMES), 3):
        raise RuntimeError(f"Expected joints_blender shaped (T, 22, 3), got {joints.shape}")
    if names != JOINT_NAMES:
        raise RuntimeError("Motion joint_names do not match the canonical 22-joint order")
    if len(joints) == 0 or not np.isfinite(joints).all():
        raise RuntimeError("Motion must contain finite joints and at least one frame")
    return joints


def _frame(right: Vector, up: Vector) -> Matrix:
    right = right.normalized()
    up = up - right * up.dot(right)
    if up.length < 1e-6:
        raise RuntimeError("Cannot build avatar frame from degenerate torso joints")
    up.normalize()
    forward = up.cross(right).normalized()
    return Matrix((right, forward, up)).transposed()


def _motion_scale(joints: np.ndarray, armature: bpy.types.Object) -> float:
    bones = armature.data.bones
    asset_chain = sum(
        bones[name].length
        for name in (
            "thigh_l",
            "calf_l",
            "pelvis",
            "spine_01",
            "spine_02",
            "spine_03",
            "neck_01",
        )
    )
    index = INDEX
    pose_chain = (
        np.linalg.norm(joints[:, index["left_hip"]] - joints[:, index["left_knee"]], axis=-1)
        + np.linalg.norm(joints[:, index["left_knee"]] - joints[:, index["left_ankle"]], axis=-1)
        + np.linalg.norm(joints[:, index["root"]] - joints[:, index["spine_1"]], axis=-1)
        + np.linalg.norm(joints[:, index["spine_1"]] - joints[:, index["spine_2"]], axis=-1)
        + np.linalg.norm(joints[:, index["spine_2"]] - joints[:, index["spine_3"]], axis=-1)
        + np.linalg.norm(joints[:, index["spine_3"]] - joints[:, index["neck"]], axis=-1)
        + np.linalg.norm(joints[:, index["neck"]] - joints[:, index["head"]], axis=-1)
    )
    median_chain = float(np.median(pose_chain))
    if not np.isfinite(median_chain) or median_chain < 0.25:
        raise RuntimeError(f"Invalid avatar motion scale reference: {median_chain}")
    return float(asset_chain / median_chain)


def _targets(joints: np.ndarray, armature: bpy.types.Object) -> np.ndarray:
    scale = _motion_scale(joints, armature)
    anchor = np.asarray(armature.data.bones["pelvis"].head_local, dtype=np.float32)
    roots = joints[:, INDEX["root"] : INDEX["root"] + 1]
    targets = anchor[None, None] + (joints - roots) * scale
    floor_joints = [
        INDEX["left_ankle"],
        INDEX["right_ankle"],
        INDEX["left_toe"],
        INDEX["right_toe"],
    ]
    floor = np.min(targets[:, floor_joints, 2], axis=1)
    targets[..., 2] -= floor[:, None]
    return targets


def _body_rotation(target: np.ndarray, armature: bpy.types.Object) -> Matrix:
    bones = armature.data.bones
    rest_right = bones["upperarm_r"].head_local - bones["upperarm_l"].head_local
    rest_up = bones["neck_01"].head_local - bones["pelvis"].head_local
    target_right = Vector(target[INDEX["right_shoulder"]] - target[INDEX["left_shoulder"]])
    target_up = Vector(target[INDEX["neck"]] - target[INDEX["root"]])
    return _frame(target_right, target_up) @ _frame(rest_right, rest_up).inverted()


def _pose_bone(
    pose_bone: bpy.types.PoseBone,
    _start: Vector,
    end: Vector,
    body_rotation: Matrix,
) -> None:
    vector = end - _start
    if vector.length < 1e-5:
        return
    rest_rotation = pose_bone.bone.matrix_local.to_3x3().normalized()
    rotation = body_rotation @ rest_rotation
    current_direction = (rotation @ Vector((0.0, 1.0, 0.0))).normalized()
    alignment = current_direction.rotation_difference(vector.normalized()).to_matrix()
    rotation = alignment @ rotation
    pose_bone.matrix = Matrix.Translation(pose_bone.head) @ rotation.to_4x4()


def _animate(armature: bpy.types.Object, targets: np.ndarray) -> None:
    required = {name for name, _, _ in BONE_TARGETS}
    missing = sorted(required - set(armature.pose.bones.keys()))
    if missing:
        raise RuntimeError(f"Quaternius armature is missing mapped bones: {missing}")
    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"

    for frame_index, target in enumerate(targets, start=1):
        bpy.context.scene.frame_set(frame_index)
        root_anchor = armature.data.bones["pelvis"].head_local
        armature.location = Vector(target[INDEX["root"]]) - root_anchor
        armature.keyframe_insert(data_path="location", frame=frame_index)
        for pose_bone in armature.pose.bones:
            pose_bone.matrix_basis.identity()
        bpy.context.view_layer.update()
        body_rotation = _body_rotation(target, armature)
        for bone_name, start_name, end_name in BONE_TARGETS:
            pose_bone = armature.pose.bones[bone_name]
            _pose_bone(
                pose_bone,
                Vector(target[INDEX[start_name]]),
                Vector(target[INDEX[end_name]]),
                body_rotation,
            )
            bpy.context.view_layer.update()
            pose_bone.keyframe_insert(data_path="location", frame=frame_index)
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_index)
            pose_bone.keyframe_insert(data_path="scale", frame=frame_index)
    if (
        armature.animation_data
        and armature.animation_data.action
        and hasattr(armature.animation_data.action, "fcurves")
    ):
        for curve in armature.animation_data.action.fcurves:
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "LINEAR"


def _look_at(item: bpy.types.Object, target: Vector) -> None:
    item.rotation_euler = (target - item.location).to_track_quat("-Z", "Y").to_euler()


def _material(name: str, color: tuple[float, float, float, float], roughness: float):
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.metallic = 0.0
    material.roughness = roughness
    return material


def _stage(targets: np.ndarray, width: int, height: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 16
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 70
    scene.render.film_transparent = False
    scene.world.color = (0.012, 0.014, 0.022)
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.012, 0.014, 0.022, 1.0)
        background.inputs["Strength"].default_value = 0.22

    bpy.ops.mesh.primitive_plane_add(size=200.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "GravityMocap_Floor"
    floor.data.materials.append(_material("GravityMocap_Floor", (0.055, 0.06, 0.075, 1.0), 0.92))

    x_min = float(np.min(targets[..., 0]))
    x_max = float(np.max(targets[..., 0]))
    z_min = min(0.0, float(np.min(targets[..., 2])))
    z_max = float(np.max(targets[..., 2]))
    center = Vector(((x_min + x_max) * 0.5, 0.0, (z_min + z_max) * 0.5))
    vertical = max(z_max - z_min, 1.8)
    horizontal = max(x_max - x_min, 1.8)
    aspect = width / height

    bpy.ops.object.camera_add(location=(center.x, -5.5, center.z + 0.05))
    camera = bpy.context.object
    camera.data.type = "ORTHO"
    camera.data.lens = 50.0
    camera.data.ortho_scale = max(vertical * 1.14, horizontal * 1.14 / aspect)
    _look_at(camera, center)
    scene.camera = camera

    lights = (
        ("Key", (-3.2, -4.0, 5.2), 540.0, 4.0, (0.82, 0.88, 1.0)),
        ("Fill", (3.8, -2.0, 2.8), 300.0, 3.5, (0.66, 0.76, 1.0)),
        ("Rim", (0.0, 3.2, 4.5), 720.0, 3.0, (0.75, 0.82, 1.0)),
    )
    for name, location, energy, size, color in lights:
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        data.color = color
        light = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(light)
        light.location = location
        _look_at(light, center)

    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass


def _render(args: argparse.Namespace, frame_count: int) -> None:
    scene = bpy.context.scene
    scene.render.fps = max(1, round(args.fps))
    scene.render.fps_base = scene.render.fps / args.fps
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_dir / "frame-")
    scene.render.use_file_extension = True
    scene.frame_start = 1
    scene.frame_end = frame_count
    bpy.ops.render.render(animation=True)


def main() -> None:
    args = _arguments()
    if args.width < 64 or args.height < 64 or args.fps <= 0:
        raise RuntimeError("Avatar render dimensions and FPS must be positive")
    motion = _load_motion(args.motion.expanduser().resolve())
    _reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(args.asset.expanduser().resolve()))
    armatures = [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"Expected one avatar armature, found {len(armatures)}")
    armature = armatures[0]
    for item in list(bpy.context.scene.objects):
        if item.type == "MESH" and not any(
            modifier.type == "ARMATURE" for modifier in item.modifiers
        ):
            bpy.data.objects.remove(item, do_unlink=True)
    targets = _targets(motion, armature)
    _animate(armature, targets)
    _stage(targets, args.width, args.height)
    _render(args, len(motion))


if __name__ == "__main__":
    main()
