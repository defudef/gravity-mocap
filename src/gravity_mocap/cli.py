from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from .baseline import infer_detector_baseline
from .catalog import DatasetCatalog
from .comparison import compare_motion_previews
from .data import MotionWindowDataset
from .detector_loop import detector_loop_generation_plan, generate_detector_loop_sidecar
from .download import describe, download_dataset
from .evaluation import diagnose_motion
from .fixture import create_fixture
from .inference import infer_rig
from .losses import compute_losses
from .preprocess import preprocess_dataset
from .trainer import build_model, load_config, run_training, training_plan
from .video import default_video_output_directory, video_to_rig
from .vision import (
    FrameManifestDataset,
    attach_features,
    extract_frame_features,
    train_frame_encoder,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = PROJECT_ROOT / "configs" / "datasets.yaml"
DEFAULT_DATA_ROOT = (
    Path(os.environ.get("GRAVITY_MOCAP_DATA_ROOT", PROJECT_ROOT / "Saved/GravityMocap"))
    .expanduser()
    .resolve()
)
DEFAULT_RAW = DEFAULT_DATA_ROOT / "raw"
DEFAULT_PROCESSED = DEFAULT_DATA_ROOT / "processed"
DEFAULT_FIXTURE = DEFAULT_DATA_ROOT / "fixtures/synthetic/walk.npz"
DEFAULT_INFERENCE_CHECKPOINT = DEFAULT_DATA_ROOT / "runs/motion-small-v2/best.pt"


def _catalog(path: str) -> DatasetCatalog:
    return DatasetCatalog(Path(path))


def command_audit(args: argparse.Namespace) -> int:
    catalog = _catalog(args.catalog)
    errors = catalog.audit()
    result = {
        "status": "ok" if not errors else "failed",
        "approved": sorted(
            dataset_id
            for dataset_id, entry in catalog.datasets.items()
            if entry.approved_for_training
        ),
        "blocked": catalog.raw["blocked_sources"],
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


def command_download(args: argparse.Namespace) -> int:
    catalog = _catalog(args.catalog)
    entries = (
        [catalog.require_approved(dataset_id) for dataset_id in args.dataset]
        if args.dataset
        else catalog.profile(args.profile)
    )
    selection = f"datasets {', '.join(args.dataset)}" if args.dataset else f"profile {args.profile}"
    print(f"Selection {selection} -> {args.root}")
    for entry in entries:
        print(f"- {entry.dataset_id}: {describe(entry, args.profile)} [{entry.license_id}]")
    if not args.execute:
        print("DRY RUN: nothing downloaded. Add --execute to perform these downloads.")
        return 0
    for entry in entries:
        outputs = download_dataset(
            entry,
            args.root,
            args.profile,
            allow_large=args.allow_large,
            accepted_data_use=args.accept_data_use,
        )
        print(f"{entry.dataset_id}: {len(outputs)} file(s) ready")
    return 0


def command_preprocess(args: argparse.Namespace) -> int:
    catalog = _catalog(args.catalog)
    config = load_config(args.config) if args.config is not None else None
    image_feature_dim = (
        args.image_feature_dim
        if args.image_feature_dim is not None
        else int(config["model"]["image_feature_dim"] if config is not None else 512)
    )
    target_fps = (
        args.target_fps
        if args.target_fps is not None
        else float(config["data"]["target_fps"] if config is not None else 30.0)
    )
    entries = (
        [catalog.require_approved(dataset_id) for dataset_id in args.dataset]
        if args.dataset
        else catalog.profile(args.profile)
    )
    total = 0
    camera_config = dict(catalog.raw.get("preprocessing", {}).get("synthetic_camera", {}))
    for entry in entries:
        written = preprocess_dataset(
            entry,
            args.raw_root,
            args.output_root,
            image_feature_dim=image_feature_dim,
            target_fps=target_fps,
            camera_config=camera_config,
            limit=args.limit,
        )
        total += len(written)
        print(f"{entry.dataset_id}: {len(written)} shard(s) ready")
    if total == 0:
        print(
            "No supported raw sequences found. Download/extract data or use "
            "the generic NPZ contract.",
            file=sys.stderr,
        )
        return 2
    return 0


def command_fixture(args: argparse.Namespace) -> int:
    print(create_fixture(args.output, frames=args.frames, image_feature_dim=args.image_feature_dim))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    with TemporaryDirectory(prefix="gravity-mocap-validate-") as temporary:
        fixture_root = Path(temporary)
        fixture = fixture_root / "synthetic" / "walk.npz"
        create_fixture(
            fixture,
            frames=16,
            image_feature_dim=int(config["model"]["image_feature_dim"]),
        )
        dataset = MotionWindowDataset(
            fixture_root,
            int(config["data"]["sequence_length"]),
            int(config["data"]["stride"]),
            augmentation=config["data"]["augmentation"],
            augmentation_seed=int(config["seed"]),
            gravity_view_contract=bool(config["data"]["gravity_view_contract"]),
            detector_world_3d=config["data"]["detector_world_3d"],
        )
        batch = {name: value.unsqueeze(0) for name, value in dataset[0].items()}
    model = build_model(config).eval()
    with torch.no_grad():
        prediction = model(batch)
        losses = compute_losses(prediction, batch, config["loss"])
    result = {
        "status": "ok",
        "note": "forward-only validation; no backward pass or optimizer step",
        "parameters": model.parameter_count,
        "output_shapes": {name: list(value.shape) for name, value in prediction.items()},
        "losses_finite": all(torch.isfinite(value).item() for value in losses.values()),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["losses_finite"] else 1


def command_train(args: argparse.Namespace) -> int:
    if not args.execute:
        plan = training_plan(
            args.config,
            args.output,
            resume=args.resume,
            max_hours=args.max_hours,
            max_epochs=args.max_epochs,
        )
        print(json.dumps(plan, indent=2))
        print("DRY RUN: training did not start. Add --execute only when you want to train.")
        return 0 if plan["ready"] else 1
    result = run_training(
        args.config,
        args.output,
        resume=args.resume,
        max_hours=args.max_hours,
        max_epochs=args.max_epochs,
    )
    print(json.dumps(result, indent=2))
    return 0


def command_train_vision(args: argparse.Namespace) -> int:
    manifest_dataset = FrameManifestDataset(args.manifest, _catalog(args.catalog))
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "DRY RUN - vision training did not start",
                    "manifest": str(args.manifest.resolve()),
                    "validated_frames": len(manifest_dataset),
                    "output": str(args.output.resolve()),
                    "feature_dim": args.feature_dim,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                },
                indent=2,
            )
        )
        return 0
    train_frame_encoder(
        args.manifest,
        args.output,
        catalog=_catalog(args.catalog),
        feature_dim=args.feature_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    return 0


def command_extract_features(args: argparse.Namespace) -> int:
    extract_frame_features(
        args.manifest,
        args.checkpoint,
        args.output,
        catalog=_catalog(args.catalog),
        device=args.device,
        batch_size=args.batch_size,
    )
    return 0


def command_attach_features(args: argparse.Namespace) -> int:
    attach_features(args.shard, args.features, catalog=_catalog(args.catalog))
    return 0


def _video_output(args: argparse.Namespace) -> Path:
    return args.output or default_video_output_directory(
        args.video.expanduser().resolve(), DEFAULT_DATA_ROOT
    )


def command_video_to_rig(args: argparse.Namespace) -> int:
    result = video_to_rig(
        args.video,
        _video_output(args),
        data_root=DEFAULT_DATA_ROOT,
        target_fps=args.target_fps,
        confidence_threshold=args.confidence_threshold,
        bbox_padding=args.bbox_padding,
        max_missing_frames=args.max_missing_frames,
        max_frames=args.max_frames,
        force=args.force,
        preview=not args.no_preview,
    )
    print(json.dumps(result, indent=2))
    return 0


def command_infer_video(args: argparse.Namespace) -> int:
    output = _video_output(args)
    rig = video_to_rig(
        args.video,
        output,
        data_root=DEFAULT_DATA_ROOT,
        target_fps=args.target_fps,
        confidence_threshold=args.confidence_threshold,
        bbox_padding=args.bbox_padding,
        max_missing_frames=args.max_missing_frames,
        max_frames=args.max_frames,
        force=args.force,
        preview=not args.no_preview,
    )
    result = infer_rig(
        Path(rig["rig_2d"]),
        args.checkpoint,
        output,
        source_video=args.video,
        detector_world_3d_path=Path(rig["detector_world_3d"]),
        device_name=args.device,
        force=args.force,
        preview=not args.no_preview,
        avatar_renderer=args.avatar_renderer,
        root_motion=args.root_motion,
    )
    result["rig_2d_status"] = rig["status"]
    result["detector_status"] = rig["status"]
    print(json.dumps(result, indent=2))
    return 0


def command_infer_video_baseline(args: argparse.Namespace) -> int:
    output = _video_output(args)
    detection = video_to_rig(
        args.video,
        output,
        data_root=DEFAULT_DATA_ROOT,
        target_fps=args.target_fps,
        confidence_threshold=args.confidence_threshold,
        bbox_padding=args.bbox_padding,
        max_missing_frames=args.max_missing_frames,
        max_frames=args.max_frames,
        force=args.force,
        preview=not args.no_preview,
    )
    result = infer_detector_baseline(
        Path(detection["detector_world_3d"]),
        output,
        rig_2d_path=Path(detection["rig_2d"]),
        source_video=args.video,
        smoothing_window=args.smoothing_window,
        force=args.force,
        preview=not args.no_preview,
        avatar_renderer=args.avatar_renderer,
    )
    result["detector_status"] = detection["status"]
    result["rig_2d"] = detection["rig_2d"]
    result["detector_world_3d"] = detection["detector_world_3d"]
    print(json.dumps(result, indent=2))
    return 0


def command_infer_rig(args: argparse.Namespace) -> int:
    output = args.output or args.rig_2d.expanduser().resolve().parent
    result = infer_rig(
        args.rig_2d,
        args.checkpoint,
        output,
        source_video=args.source_video,
        detector_world_3d_path=args.detector_world_3d,
        device_name=args.device,
        force=args.force,
        preview=args.source_video is not None and not args.no_preview,
        avatar_renderer=args.avatar_renderer,
        root_motion=args.root_motion,
    )
    print(json.dumps(result, indent=2))
    return 0


def command_infer_detector_world(args: argparse.Namespace) -> int:
    output = args.output or args.detector_world_3d.expanduser().resolve().parent
    preview = args.source_video is not None and args.rig_2d is not None and not args.no_preview
    result = infer_detector_baseline(
        args.detector_world_3d,
        output,
        rig_2d_path=args.rig_2d,
        source_video=args.source_video,
        smoothing_window=args.smoothing_window,
        force=args.force,
        preview=preview,
        avatar_renderer=args.avatar_renderer,
    )
    print(json.dumps(result, indent=2))
    return 0


def command_compare_previews(args: argparse.Namespace) -> int:
    result = compare_motion_previews(
        args.baseline_preview,
        args.learned_preview,
        args.output,
        force=args.force,
    )
    print(json.dumps(result, indent=2))
    return 0


def command_diagnose_motion(args: argparse.Namespace) -> int:
    output = args.output or args.motion.expanduser().resolve().with_name("motion-diagnostics.json")
    result = diagnose_motion(
        args.motion,
        args.rig_2d,
        output,
        baseline_motion_path=args.baseline_motion,
        detector_world_3d_path=args.detector_world_3d,
        source_video=args.source_video,
        world_preview=args.world_preview,
        avatar_renderer=args.avatar_renderer,
    )
    print(json.dumps(result, indent=2))
    return 0


def command_generate_detector_loop(args: argparse.Namespace) -> int:
    plan = detector_loop_generation_plan(
        args.source_shard,
        args.output_root,
        catalog_path=args.catalog,
        data_root=DEFAULT_DATA_ROOT,
        width=args.width,
        height=args.height,
    )
    plan.update(
        {
            "catalog": str(args.catalog),
            "steps": [
                "render approved neutral motion with bundled CC0 gray avatar",
                "run checksum-pinned MediaPipe in native VIDEO mode",
                "align detector world-3D prior to the gravity-view target",
                "write a source-hash-bound detector-loop sidecar",
            ],
        }
    )
    if not args.execute:
        print(json.dumps(plan, indent=2))
        print("DRY RUN: no frames, video, detector cache, or sidecar were created.")
        return 0
    sidecar = generate_detector_loop_sidecar(
        args.source_shard,
        args.output_root,
        catalog_path=args.catalog,
        data_root=DEFAULT_DATA_ROOT,
        width=args.width,
        height=args.height,
    )
    print(json.dumps({**plan, "sidecar": str(sidecar), "status": "ready"}, indent=2))
    return 0


def _add_video_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--bbox-padding", type=float, default=0.12)
    parser.add_argument("--max-missing-frames", type=int, default=30)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-preview", action="store_true")


def _add_avatar_renderer_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--avatar-renderer",
        choices=("auto", "mesh", "procedural"),
        default="auto",
        help="auto uses the bundled gray mesh when Blender is available",
    )


def _add_root_motion_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root-motion",
        choices=("safe", "learned", "stationary"),
        default="safe",
        help="safe fails closed to stationary motion without reliable ground contacts",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean-room world-grounded mocap pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Validate the closed dataset/license allowlist")
    audit.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    audit.set_defaults(handler=command_audit)

    download = subparsers.add_parser("download", help="Plan or execute allowlisted downloads")
    download.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    download.add_argument("--profile", choices=("smoke", "core", "expanded"), default="core")
    download.add_argument("--dataset", action="append", default=[])
    download.add_argument("--root", type=Path, default=DEFAULT_RAW)
    download.add_argument("--execute", action="store_true")
    download.add_argument("--allow-large", action="store_true")
    download.add_argument("--accept-data-use", action="store_true")
    download.set_defaults(handler=command_download)

    preprocess = subparsers.add_parser(
        "preprocess", help="Convert allowlisted raw motion to neutral shards"
    )
    preprocess.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    preprocess.add_argument("--profile", choices=("smoke", "core", "expanded"), default="core")
    preprocess.add_argument("--dataset", action="append", default=[])
    preprocess.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    preprocess.add_argument("--output-root", type=Path, default=DEFAULT_PROCESSED)
    preprocess.add_argument(
        "--config",
        type=Path,
        help="Use model.image_feature_dim and data.target_fps as preprocessing defaults",
    )
    preprocess.add_argument("--image-feature-dim", type=int)
    preprocess.add_argument("--target-fps", type=float)
    preprocess.add_argument("--limit", type=int)
    preprocess.set_defaults(handler=command_preprocess)

    fixture = subparsers.add_parser("fixture", help="Create a tiny synthetic smoke-test shard")
    fixture.add_argument("--output", type=Path, default=DEFAULT_FIXTURE)
    fixture.add_argument("--frames", type=int, default=16)
    fixture.add_argument("--image-feature-dim", type=int, default=32)
    fixture.set_defaults(handler=command_fixture)

    validate = subparsers.add_parser("validate", help="Run a forward-only model/schema smoke test")
    validate.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/train-smoke.yaml")
    validate.set_defaults(handler=command_validate)

    train = subparsers.add_parser("train", help="Show a plan by default; --execute starts training")
    train.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/train-paper.yaml")
    train.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_ROOT / "runs/motion",
    )
    train.add_argument(
        "--resume",
        default="auto",
        help="auto (default), never, or an explicit checkpoint path",
    )
    session_limit = train.add_mutually_exclusive_group()
    session_limit.add_argument(
        "--max-hours",
        type=float,
        help="Stop safely after this many hours and continue on the next invocation",
    )
    session_limit.add_argument(
        "--max-epochs",
        type=int,
        help="Stop safely after this many completed epochs and continue next time",
    )
    train.add_argument("--execute", action="store_true")
    train.set_defaults(handler=command_train)

    train_vision = subparsers.add_parser("train-vision", help="Train the clean visual crop encoder")
    train_vision.add_argument("manifest", type=Path)
    train_vision.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    train_vision.add_argument("--output", type=Path, required=True)
    train_vision.add_argument("--feature-dim", type=int, default=512)
    train_vision.add_argument("--epochs", type=int, default=50)
    train_vision.add_argument("--batch-size", type=int, default=64)
    train_vision.add_argument("--learning-rate", type=float, default=0.001)
    train_vision.add_argument("--device", default="cpu")
    train_vision.add_argument("--execute", action="store_true")
    train_vision.set_defaults(handler=command_train_vision)

    extract = subparsers.add_parser(
        "extract-features", help="Export clean visual features from one sequence manifest"
    )
    extract.add_argument("manifest", type=Path)
    extract.add_argument("checkpoint", type=Path)
    extract.add_argument("output", type=Path)
    extract.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--batch-size", type=int, default=64)
    extract.set_defaults(handler=command_extract_features)

    attach = subparsers.add_parser(
        "attach-features", help="Attach audited visual features to a motion shard"
    )
    attach.add_argument("shard", type=Path)
    attach.add_argument("features", type=Path)
    attach.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    attach.set_defaults(handler=command_attach_features)

    rig = subparsers.add_parser(
        "video-to-rig",
        aliases=["detect-video"],
        help="Create a standalone cached neutral 22-joint 2D rig",
    )
    _add_video_arguments(rig)
    rig.set_defaults(handler=command_video_to_rig)

    infer_rig_parser = subparsers.add_parser(
        "infer-rig", help="Run 2D rig -> clean-room 3D motion without rerunning detection"
    )
    infer_rig_parser.add_argument("rig_2d", type=Path)
    infer_rig_parser.add_argument("--output", type=Path)
    infer_rig_parser.add_argument(
        "--source-video",
        type=Path,
        help="Optional matching source video; when provided, render the motion preview",
    )
    infer_rig_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_INFERENCE_CHECKPOINT)
    infer_rig_parser.add_argument(
        "--detector-world-3d",
        type=Path,
        help="Required detector prior for v2 checkpoints",
    )
    infer_rig_parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    infer_rig_parser.add_argument("--force", action="store_true")
    infer_rig_parser.add_argument("--no-preview", action="store_true")
    _add_avatar_renderer_argument(infer_rig_parser)
    _add_root_motion_argument(infer_rig_parser)
    infer_rig_parser.set_defaults(handler=command_infer_rig)

    infer_world_parser = subparsers.add_parser(
        "infer-detector-world",
        help="Retarget detector world 3D to neutral motion without a learned checkpoint",
    )
    infer_world_parser.add_argument("detector_world_3d", type=Path)
    infer_world_parser.add_argument("--rig-2d", type=Path)
    infer_world_parser.add_argument("--source-video", type=Path)
    infer_world_parser.add_argument("--output", type=Path)
    infer_world_parser.add_argument("--smoothing-window", type=int, default=5)
    infer_world_parser.add_argument("--force", action="store_true")
    infer_world_parser.add_argument("--no-preview", action="store_true")
    _add_avatar_renderer_argument(infer_world_parser)
    infer_world_parser.set_defaults(handler=command_infer_detector_world)

    infer = subparsers.add_parser(
        "infer-video", help="Run video -> 2D skeleton -> clean-room 3D motion"
    )
    _add_video_arguments(infer)
    _add_avatar_renderer_argument(infer)
    _add_root_motion_argument(infer)
    infer.add_argument("--checkpoint", type=Path, default=DEFAULT_INFERENCE_CHECKPOINT)
    infer.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    infer.set_defaults(handler=command_infer_video)

    baseline = subparsers.add_parser(
        "infer-video-baseline",
        help="Run video -> detector world 3D -> neutral motion without training",
    )
    _add_video_arguments(baseline)
    _add_avatar_renderer_argument(baseline)
    baseline.add_argument("--smoothing-window", type=int, default=5)
    baseline.set_defaults(handler=command_infer_video_baseline)

    compare = subparsers.add_parser(
        "compare-previews",
        help="Compose source, detector baseline, and learned 3D avatar previews",
    )
    compare.add_argument("baseline_preview", type=Path)
    compare.add_argument("learned_preview", type=Path)
    compare.add_argument("output", type=Path)
    compare.add_argument("--force", action="store_true")
    compare.set_defaults(handler=command_compare_previews)

    diagnostics = subparsers.add_parser(
        "diagnose-motion",
        help="Measure detector coverage, temporal stability, contacts, and world-root drift",
    )
    diagnostics.add_argument("motion", type=Path)
    diagnostics.add_argument("--rig-2d", type=Path, required=True)
    diagnostics.add_argument("--baseline-motion", type=Path)
    diagnostics.add_argument("--detector-world-3d", type=Path)
    diagnostics.add_argument("--source-video", type=Path)
    diagnostics.add_argument("--output", type=Path)
    diagnostics.add_argument(
        "--world-preview",
        action="store_true",
        help="Render a fixed-camera view that exposes predicted root translation",
    )
    _add_avatar_renderer_argument(diagnostics)
    diagnostics.set_defaults(handler=command_diagnose_motion)

    detector_loop = subparsers.add_parser(
        "generate-detector-loop",
        help="Dry-run or build a detector-in-the-loop sidecar for one approved shard",
    )
    detector_loop.add_argument("source_shard", type=Path)
    detector_loop.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "detector-loop",
    )
    detector_loop.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    detector_loop.add_argument("--width", type=int, default=512)
    detector_loop.add_argument("--height", type=int, default=512)
    detector_loop.add_argument("--execute", action="store_true")
    detector_loop.set_defaults(handler=command_generate_detector_loop)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
