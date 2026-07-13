from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from .catalog import DatasetCatalog
from .data import MotionWindowDataset
from .download import describe, download_dataset
from .fixture import create_fixture
from .losses import compute_losses
from .preprocess import preprocess_dataset
from .trainer import build_model, load_config, run_training, training_plan
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


def _catalog(path: str) -> DatasetCatalog:
    return DatasetCatalog(Path(path))


def command_audit(args: argparse.Namespace) -> int:
    catalog = _catalog(args.catalog)
    errors = catalog.audit()
    result = {
        "status": "ok" if not errors else "failed",
        "approved": sorted(catalog.datasets),
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
    entries = (
        [catalog.require_approved(dataset_id) for dataset_id in args.dataset]
        if args.dataset
        else catalog.profile(args.profile)
    )
    total = 0
    for entry in entries:
        written = preprocess_dataset(
            entry,
            args.raw_root,
            args.output_root,
            image_feature_dim=args.image_feature_dim,
            target_fps=args.target_fps,
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
    fixture = Path(config["data"]["root"]) / "synthetic" / "walk.npz"
    create_fixture(fixture, frames=16, image_feature_dim=int(config["model"]["image_feature_dim"]))
    dataset = MotionWindowDataset(
        Path(config["data"]["root"]),
        int(config["data"]["sequence_length"]),
        int(config["data"]["stride"]),
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
    preprocess.add_argument("--image-feature-dim", type=int, default=512)
    preprocess.add_argument("--target-fps", type=float, default=30.0)
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
