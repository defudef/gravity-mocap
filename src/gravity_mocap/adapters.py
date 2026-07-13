from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .skeleton import JOINT_NAMES

ALIASES = {
    "root": ("root", "pelvis", "hips", "hip", "lowerback"),
    "left_hip": ("lhipjoint", "lefthip", "lhip", "leftupleg", "lfemur"),
    "right_hip": ("rhipjoint", "righthip", "rhip", "rightupleg", "rfemur"),
    "spine_1": ("lowerback", "spine", "spine1", "abdomen"),
    "left_knee": ("leftleg", "leftknee", "lknee", "ltibia"),
    "right_knee": ("rightleg", "rightknee", "rknee", "rtibia"),
    "spine_2": ("upperback", "spine2", "chest"),
    "left_ankle": ("leftfoot", "leftankle", "lankle", "lfoot"),
    "right_ankle": ("rightfoot", "rightankle", "rankle", "rfoot"),
    "spine_3": ("thorax", "spine3", "upperchest"),
    "left_toe": ("lefttoebase", "lefttoe", "ltoes", "ltoe"),
    "right_toe": ("righttoebase", "righttoe", "rtoes", "rtoe"),
    "neck": ("lowerneck", "neck", "neck1"),
    "left_clavicle": ("lclavicle", "leftshoulder", "lcollar"),
    "right_clavicle": ("rclavicle", "rightshoulder", "rcollar"),
    "head": ("head", "headend", "upperneck"),
    "left_shoulder": ("lhumerus", "leftarm", "lshoulder"),
    "right_shoulder": ("rhumerus", "rightarm", "rshoulder"),
    "left_elbow": ("lradius", "leftforearm", "lelbow"),
    "right_elbow": ("rradius", "rightforearm", "relbow"),
    "left_wrist": ("lhand", "lefthand", "lwrist"),
    "right_wrist": ("rhand", "righthand", "rwrist"),
}

FORBIDDEN_PARAMETRIC_BODY_KEYS = (
    "smpl",
    "body_pose",
    "global_orient",
    "betas",
    "body_model",
)


def _normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def canonical_indices(source_names: list[str]) -> np.ndarray:
    lookup = {_normalized(name): index for index, name in enumerate(source_names)}
    result = []
    for canonical in JOINT_NAMES:
        candidates = (canonical, *ALIASES[canonical])
        matches = [
            lookup[_normalized(alias)] for alias in candidates if _normalized(alias) in lookup
        ]
        if not matches:
            raise ValueError(
                f"Cannot map canonical joint {canonical!r}; source joints: {source_names}"
            )
        result.append(matches[0])
    return np.asarray(result, dtype=np.int64)


def canonicalize_positions(positions: np.ndarray, source_names: list[str]) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Expected positions shaped (T, J, 3), got {positions.shape}")
    mapped = positions[:, canonical_indices(source_names)]
    # Normalize unknown source units/proportions using median head-to-ankle span.
    height = np.linalg.norm(mapped[:, 15] - 0.5 * (mapped[:, 7] + mapped[:, 8]), axis=-1)
    median_height = float(np.median(height[height > 1e-6]))
    if not np.isfinite(median_height) or median_height <= 1e-6:
        raise ValueError("Could not estimate actor scale")
    if median_height < 1.0 or median_height > 2.5:
        mapped = mapped * (1.65 / median_height)
    return mapped


def canonicalize_addbiomechanics(positions: np.ndarray, source_names: list[str]) -> np.ndarray:
    lookup = {_normalized(name): index for index, name in enumerate(source_names)}

    def joint(*names: str) -> np.ndarray:
        for name in names:
            if _normalized(name) in lookup:
                return positions[:, lookup[_normalized(name)]]
        raise ValueError(f"Missing AddBiomechanics joint; tried {names}")

    root = joint("ground_pelvis", "pelvis")
    left_hip = joint("hip_l")
    right_hip = joint("hip_r")
    left_shoulder = joint("acromial_l", "shoulder_l")
    right_shoulder = joint("acromial_r", "shoulder_r")
    shoulder_center = 0.5 * (left_shoulder + right_shoulder)
    up = shoulder_center - root
    up /= np.linalg.norm(up, axis=-1, keepdims=True).clip(1e-8)
    neck = shoulder_center + up * 0.06
    canonical = np.stack(
        (
            root,
            left_hip,
            right_hip,
            root * 0.75 + shoulder_center * 0.25,
            joint("walker_knee_l", "knee_l"),
            joint("walker_knee_r", "knee_r"),
            root * 0.5 + shoulder_center * 0.5,
            joint("ankle_l"),
            joint("ankle_r"),
            root * 0.25 + shoulder_center * 0.75,
            joint("mtp_l"),
            joint("mtp_r"),
            neck,
            neck * 0.7 + left_shoulder * 0.3,
            neck * 0.7 + right_shoulder * 0.3,
            neck + up * 0.22,
            left_shoulder,
            right_shoulder,
            joint("elbow_l"),
            joint("elbow_r"),
            joint("radius_hand_l", "hand_l"),
            joint("radius_hand_r", "hand_r"),
        ),
        axis=1,
    ).astype(np.float32)
    return canonicalize_positions(canonical, list(JOINT_NAMES))


def load_generic_npz(path: Path) -> tuple[np.ndarray, list[str], float]:
    with np.load(path, allow_pickle=False) as archive:
        forbidden = sorted(
            key
            for key in archive.files
            if any(token in key.lower() for token in FORBIDDEN_PARAMETRIC_BODY_KEYS)
        )
        if forbidden:
            raise ValueError(
                f"{path} contains prohibited SMPL/SMPL-X/body-model fields: {forbidden}"
            )
        position_key = next(
            (key for key in ("joints_3d", "joint_positions", "positions") if key in archive), None
        )
        if position_key is None or "joint_names" not in archive:
            raise ValueError(f"{path} needs positions/joints_3d and joint_names")
        positions = np.asarray(archive[position_key], dtype=np.float32)
        names = [str(value) for value in archive["joint_names"].tolist()]
        fps = float(archive["fps"]) if "fps" in archive else 30.0
    return positions, names, fps


def _axis_rotation(axis: str, degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    sine, cosine = np.sin(radians), np.cos(radians)
    if axis.lower() == "rx":
        return np.asarray([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]], dtype=np.float64)
    if axis.lower() == "ry":
        return np.asarray([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]], dtype=np.float64)
    if axis.lower() == "rz":
        return np.asarray([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"Unsupported rotation channel: {axis}")


def _ordered_rotation(channels: list[str], values: list[float]) -> np.ndarray:
    result = np.eye(3, dtype=np.float64)
    for channel, value in zip(channels, values, strict=False):
        if channel.lower().startswith("r"):
            result = result @ _axis_rotation(channel, value)
    return result


@dataclass
class AsfBone:
    name: str
    direction: np.ndarray
    length: float
    axis: np.ndarray
    axis_order: str
    dof: list[str] = field(default_factory=list)
    parent: str = "root"


@dataclass
class AsfSkeleton:
    bones: dict[str, AsfBone]
    order: list[str]
    root_order: list[str]
    root_axis: str
    length_scale: float


def parse_asf(path: Path) -> AsfSkeleton:
    lines = [
        line.split("#", 1)[0].strip() for line in path.read_text(errors="replace").splitlines()
    ]
    lines = [line for line in lines if line]
    section = ""
    units_length = 1.0
    root_order = ["TX", "TY", "TZ", "RX", "RY", "RZ"]
    root_axis = "XYZ"
    bones: dict[str, AsfBone] = {}
    hierarchy: dict[str, str] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(":"):
            section = line[1:].split()[0].lower()
            index += 1
            continue
        if section == "units" and line.lower().startswith("length"):
            units_length = float(line.split()[1])
        elif section == "root" and line.lower().startswith("order"):
            root_order = line.split()[1:]
        elif section == "root" and line.lower().startswith("axis"):
            root_axis = line.split()[1]
        elif section == "bonedata" and line.lower() == "begin":
            fields: dict[str, list[str]] = {}
            index += 1
            while index < len(lines) and lines[index].lower() != "end":
                parts = lines[index].split()
                fields[parts[0].lower()] = parts[1:]
                index += 1
            name = fields["name"][0]
            bones[name] = AsfBone(
                name=name,
                direction=np.asarray(
                    [float(value) for value in fields["direction"]], dtype=np.float64
                ),
                length=float(fields["length"][0]) * units_length,
                axis=np.asarray(
                    [float(value) for value in fields.get("axis", ["0", "0", "0"])[:3]]
                ),
                axis_order=fields.get("axis", ["0", "0", "0", "XYZ"])[-1],
                dof=[value.upper() for value in fields.get("dof", [])],
            )
        elif section == "hierarchy" and line.lower() not in {"begin", "end"}:
            names = line.split()
            for child in names[1:]:
                hierarchy[child] = names[0]
        index += 1
    for name, parent in hierarchy.items():
        if name in bones:
            bones[name].parent = parent
    order: list[str] = []
    pending = set(bones)
    while pending:
        ready = sorted(
            name for name in pending if bones[name].parent == "root" or bones[name].parent in order
        )
        if not ready:
            raise ValueError(f"Cyclic or incomplete ASF hierarchy in {path}")
        order.extend(ready)
        pending.difference_update(ready)
    return AsfSkeleton(bones, order, root_order, root_axis, units_length)


def parse_amc(path: Path) -> list[dict[str, list[float]]]:
    frames: list[dict[str, list[float]]] = []
    current: dict[str, list[float]] | None = None
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(":"):
            continue
        if line.isdigit():
            current = {}
            frames.append(current)
            continue
        if current is None:
            continue
        parts = line.split()
        current[parts[0]] = [float(value) for value in parts[1:]]
    if not frames:
        raise ValueError(f"No frames found in {path}")
    return frames


def load_asf_amc(
    asf_path: Path, amc_path: Path, fps: float = 120.0
) -> tuple[np.ndarray, list[str], float]:
    skeleton = parse_asf(asf_path)
    frames = parse_amc(amc_path)
    names = ["root", *skeleton.order]
    positions = np.zeros((len(frames), len(names), 3), dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        root_values = frame.get("root", [0.0] * len(skeleton.root_order))
        translations = {
            channel.lower(): value
            for channel, value in zip(skeleton.root_order, root_values, strict=False)
        }
        positions[frame_index, 0] = [
            translations.get("tx", 0),
            translations.get("ty", 0),
            translations.get("tz", 0),
        ]
        root_rotation = _ordered_rotation(skeleton.root_order, root_values)
        rotations: dict[str, np.ndarray] = {"root": root_rotation}
        name_to_index = {name: index for index, name in enumerate(names)}
        for name in skeleton.order:
            bone = skeleton.bones[name]
            parent_rotation = rotations[bone.parent]
            parent_position = positions[frame_index, name_to_index[bone.parent]]
            positions[frame_index, name_to_index[name]] = parent_position + parent_rotation @ (
                bone.direction * bone.length
            )
            axis_rotation = _ordered_rotation(
                [f"R{axis}" for axis in bone.axis_order], bone.axis.tolist()
            )
            motion_rotation = _ordered_rotation(bone.dof, frame.get(name, []))
            rotations[name] = parent_rotation @ axis_rotation @ motion_rotation @ axis_rotation.T
    return positions.astype(np.float32), names, fps


def find_cmu_sequences(root: Path) -> list[tuple[Path, Path]]:
    asf_by_dir = {path.parent: path for path in root.rglob("*.asf")}
    pairs: list[tuple[Path, Path]] = []
    for amc_path in sorted(root.rglob("*.amc")):
        asf_path = asf_by_dir.get(amc_path.parent)
        if asf_path is not None:
            pairs.append((asf_path, amc_path))
    return pairs


def load_b3d(path: Path, processing_pass: int = 0) -> tuple[np.ndarray, list[str], float]:
    try:
        import nimblephysics as nimble
    except ImportError as error:
        raise RuntimeError(
            "Reading .b3d requires the optional nimblephysics package. Install it in a compatible "
            "Python environment, or export the sequence to the generic NPZ contract first."
        ) from error
    subject = nimble.biomechanics.SubjectOnDisk(str(path))
    skeleton = subject.readSkel(processingPass=processing_pass, ignoreGeometry=True)
    names = [skeleton.getJoint(index).getName() for index in range(skeleton.getNumJoints())]
    frames = subject.readFrames(
        0,
        0,
        subject.getTrialLength(0),
        includeSensorData=False,
        includeProcessingPasses=True,
    )
    positions = [
        frame.processingPasses[processing_pass].jointCenters.reshape(len(names), 3)
        for frame in frames
    ]
    fps = 1.0 / float(subject.getTrialTimestep(0))
    return np.asarray(positions, dtype=np.float32), names, fps
