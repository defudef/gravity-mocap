# Gravity Mocap

Independent clean-room training code for world-grounded monocular human motion
recovery, inspired by the methods described in the
[GVHMR paper](https://arxiv.org/abs/2409.06662). This project is not affiliated
with or endorsed by the GVHMR authors. It includes no GVHMR/WHAM code, weights,
configs, caches, SMPL, or SMPL-X assets and replaces the parametric body with a
neutral 22-joint skeleton.

This repository contains training infrastructure, not trained weights. The
default CLI behavior is deliberately safe: download and training commands only
print a plan until `--execute` is supplied.

## Quickstart: from clone to resumable training

Run these commands in order. Commands without `--execute` are safe previews and
do not start a download or training job.

For the shortest path to a **fresh** two-hour training run, use the repository
script. It resolves every default path from its own location, so it contains no
developer-specific checkout path and can also be invoked from another working
directory:

```sh
./scripts/start-fresh-training.sh
./scripts/start-fresh-training.sh --execute --max-hours 2
./scripts/start-fresh-training.sh --execute --max-epochs 10
```

The first command only prints the plan. The execute form syncs the environment,
validates the project, downloads and preprocesses CMU, a study-stratified
AddBiomechanics train subset, and 100STYLE, requires shards from all three,
validates a fresh training plan, archives an existing default run, and finally
starts training. If preparation fails, the previous run is not archived and
training does not start.

Use the fresh-start script only once per new corpus/configuration. Resume the
run on later sessions without archiving it again:

```sh
./scripts/train.sh --execute --max-hours 2
./scripts/train.sh --execute --max-epochs 10
```

`--max-epochs N` limits only the current invocation to `N` completed epochs.
Repeating the command resumes `latest.pt` and trains another `N` epochs, up to
the total target in the config. Use either `--max-hours` or `--max-epochs`, not
both.

The equivalent manual flow follows.

1. Clone the repository and install the locked environment:

   ```sh
   git clone https://github.com/defudef/gravity-mocap.git
   cd gravity-mocap
   ./scripts/setup.sh
   ```

2. Check the dataset allowlist and run the forward-only model smoke test:

   ```sh
   ./scripts/mocap.sh audit
   ./scripts/mocap.sh validate
   ```

3. Preview, download, and convert the smallest supported motion profile:

   ```sh
   ./scripts/mocap.sh download --profile smoke
   ./scripts/mocap.sh download --profile smoke --execute
   ./scripts/mocap.sh preprocess --profile smoke
   ```

   The smoke profile downloads the approximately 1 GiB CMU archive and one
   selected AddBiomechanics B3D. Generated training shards are written to
   `Saved/GravityMocap/processed/`.

4. Inspect the paper-size training plan without starting it:

   ```sh
   ./scripts/train.sh
   ```

5. Start an eight-hour session that saves and stops safely:

   ```sh
   ./scripts/train.sh --execute --max-hours 8
   ```

   Run the same command again on the next night. It resumes `latest.pt`
   automatically. Press Ctrl+C once to request a checkpoint and graceful stop.

6. Watch metrics in a second terminal:

   ```sh
   ./scripts/mlflow-ui.sh
   ```

   Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

### mRI paired-video data is blocked

mRI is deliberately not in any training profile. Dryad currently labels the
record CC0, while the original publication describes the release as CC BY-NC
4.0. Until the conflict is resolved by the rightsholder, explicit mRI download,
preprocessing, and training requests fail the catalog gate. Any files retained
from an earlier investigation are ignored by the official training corpus.

## What is implemented

- closed dataset/license allowlist with fail-closed source validation;
- resumable HTTP/Hugging Face/Google Drive downloaders and selective remote ZIP
  extraction for the 388 GiB AddBiomechanics archive;
- generic NPZ, CMU ASF/AMC, every trial in an AddBiomechanics B3D, and 100STYLE
  BVH motion adapters;
- provenance-bearing, finite-value-validated NPZ shards;
- canonical 30 FPS preprocessing with source/target rates in provenance;
- FK-consistent neutral-skeleton targets plus simulated full camera motion,
  auto-framed detector-style bbox, and bbox-relative 2D inputs for motion-only sequences;
- deterministic detector corruption (noise, low confidence, missing joints,
  occlusion, bbox jitter, and optional unknown-camera input);
- source-stratified, subject/sequence-disjoint validation with 3D motion metrics,
  resumable early stopping, and an atomic best checkpoint;
- neutral skeleton local rotations, root motion, and six stationary contacts;
- early-fusion relative Transformer with RoPE and the paper-size 12-layer,
  8-head, 512-hidden configuration;
- multitask losses and an actual checkpointing training loop behind
  `train --execute`;
- flushed terminal progress with loss, epoch/batch/step, elapsed time, pace,
  ETA, checkpoint events, and resumable local MLflow tracking;
- mixed-precision multi-GPU execution plus gradient accumulation for an
  effective paper-size batch of 256;
- a from-scratch visual crop encoder path, with no pretrained third-party
  checkpoint dependency;
- forward-only smoke validation on a procedural fixture.

See [CLEANROOM.md](CLEANROOM.md) for the implementation boundary and
[LICENSE_AUDIT.md](LICENSE_AUDIT.md) for the engineering license review. The
Apache-2.0 license covers this project's source code, not datasets, dependency
binaries, or future checkpoints.

## Setup and validation (does not train)

From the repository root:

```sh
./scripts/setup.sh
./scripts/mocap.sh audit
./scripts/mocap.sh validate
./scripts/train.sh
```

The last command is a dry-run: it inventories shards, constructs the model to
count parameters, and exits before an optimizer exists.

## Data

Preview the practical corpus without downloading anything:

```sh
./scripts/mocap.sh download --profile core
```

Perform the approved downloads:

```sh
./scripts/mocap.sh download --profile core --execute
```

A single source can be selected with `--dataset addbiomechanics` or
`--dataset 100style`.

`smoke` selects only one tiny B3D member from the remote AddBiomechanics ZIP,
but the CMU archive is still about 1 GiB. `core` adds the 1.37 GiB 100STYLE
archive and up to 24 B3Ds selected round-robin across the eight available
AddBiomechanics studies, capped at 12 GiB uncompressed. CMU's server currently
omits its TLS intermediate certificate. The
downloader still tries HTTPS first and permits only a same-host/path HTTP
fallback protected by the pinned archive SHA-256; TLS verification is never
disabled. The current AddBiomechanics selection is 24 train-partition B3Ds and
10.65 GiB. They show per-file
progress, are written through `.part`, and are promoted only after ZIP size/CRC
verification; an interrupted member restarts individually.
SAM, mRI, and HUM4D are catalogued but blocked from official training: their
raw releases have unresolved participant/license or SMPL/SMPL-X provenance
questions. `expanded` adds only TUM radar/3D-pose motion and requires reading
and explicitly accepting its Data Usage Agreement.

Only these raw formats are converted directly today:

- CMU `.asf` + `.amc` pairs;
- every trial in AddBiomechanics `.b3d` through the pinned `nimblephysics`
  dependency;
- 100STYLE `.bvh`, trimmed by its published `Frame_Cuts.csv`;
- generic `.npz` with `positions` (or `joints_3d`/`joint_positions`),
  `joint_names`, and optional `fps`.

Convert downloaded motion sources:

```sh
./scripts/mocap.sh preprocess --profile core
```

Preprocessing reports progress every ten sequences. Repeating the command
reuses atomic shards only when their schema, provenance, source hashes,
converter version, and preprocessing configuration still match; interrupted or
stale outputs are regenerated. All supported motion is resampled to 30 FPS,
retargeted to the fixed neutral rig, and regenerated through forward kinematics
so local-rotation and 3D-joint targets describe the same skeleton exactly. Root
velocity is stored in metres per second. All trials from one B3D stay in the
same train/validation group, so the split cannot leak neighboring trials from
one source file. Changing that target contract, temporal contract, or the
synthetic-camera configuration invalidates old shards and checkpoints by
design; the training audit rejects stale converter versions.

## Video -> neutral 2D rig

The video frontend is a first-class Gravity Mocap module, separate from the
learned 2D-to-3D model. It uses the checksum-pinned MediaPipe Pose Landmarker
Heavy bundle in VIDEO tracking mode, tracks one person, and maps its 33
landmarks (including heels and toes) to the neutral 22-joint contract. The
frontend, model checksum, provenance, cache, artifact schema, CLI, and tests all
live in this repository. It has no dependency on `cozy-rpg`, SMPL, or SMPL-X.

Install the optional video dependencies once:

```sh
./scripts/setup-video.sh
```

Create only the standalone 2D rig:

```sh
./scripts/video.sh video-to-rig path/to/reference.mp4 --max-frames 90
```

This command stops after the deterministic frontend. It does not load a Gravity
Mocap checkpoint or run learned 3D inference. `detect-video` remains an alias
for backward compatibility. The 2D rig stays independently loadable; the same
detector pass also preserves MediaPipe's metric world landmarks as a separate
sidecar instead of discarding them.

The API boundary is equally explicit:

```python
from pathlib import Path

from gravity_mocap.rig2d import load_rig_2d
from gravity_mocap.video import video_to_rig

result = video_to_rig(video, output_dir, data_root=data_root)
rig = load_rig_2d(Path(result["rig_2d"]))
```

By default, the content-addressed output directory is
`Saved/GravityMocap/inference/<video-stem>-<source-sha-prefix>/` and the 2D stage
creates:

- `rig-2d.npz`: the independently reusable 2D rig and model-ready normalized
  inputs;
- `rig-2d-manifest.json`: source bytes, detector model URL/size/SHA-256, mapping
  version, parameters, frame selection, dimensions, FPS, and joint order;
- `preview-rig-2d.mp4`: the detected neutral skeleton over the source video.
- `detector-world-3d.npz`: separate root-relative 33-to-22 mapped detector
  landmarks, confidence, metric units, frame mask, source indices, and full
  source/model provenance.

The NPZ contains pixel-space keypoints for inspection and bbox-relative
keypoints in `[-1, 1]` for the motion model. Its bbox is frame-relative
`[-1, 1]`; missing joints have confidence and coordinates set to zero. The
loader validates finite values and the exact neutral joint order and rejects
any SMPL/SMPL-X/body-model fields.

## Detector-world baseline without training

Produce a neutral, fixed-bone-length 22-joint motion immediately, without a
checkpoint:

```sh
./scripts/video.sh infer-video-baseline path/to/reference.mp4
```

This writes `detector-baseline-motion.npz`,
`detector-baseline-manifest.json`, and `preview-detector-baseline.mp4`. Short
world-landmark gaps are interpolated under the same fail-closed gap limit,
confidence-weighted temporal smoothing is applied, and the result is retargeted
to the repository's neutral skeleton. This is a local-pose baseline: root
translation is intentionally stationary until a learned world-grounding model
predicts it.

The standalone artifact boundary is also available without rerunning detection:

```sh
./scripts/video.sh infer-detector-world \
  Saved/GravityMocap/inference/<video-id>/detector-world-3d.npz
```

## Neutral detector artifacts -> learned Gravity Mocap 3D

Run 3D recovery from an already-created rig without rerunning the detector or
requiring the source video:

```sh
./scripts/video.sh infer-rig \
  Saved/GravityMocap/inference/<video-id>/rig-2d.npz \
  --detector-world-3d \
    Saved/GravityMocap/inference/<video-id>/detector-world-3d.npz \
  --checkpoint Saved/GravityMocap/runs/motion-small-v2/best.pt
```

Add `--source-video path/to/reference.mp4` to render `preview-motion.mp4`; the
video hash must match the rig provenance. The 3D stage adds:

- `motion.npz` contains rotations, FK joints, root translation, contacts, FPS,
  topology, and provenance;
- `preview-motion.mp4` places the 2D input beside the neutral 3D skeleton in
  the predicted camera frame;
- `motion-manifest.json` carries the complete rig/source/model and checkpoint
  provenance.

For convenience, the composed command still runs both stages:

```sh
./scripts/video.sh infer-video path/to/reference.mp4 \
  --checkpoint Saved/GravityMocap/runs/motion-small-v2/best.pt
```

The operation is idempotent: unchanged source bytes, model, checkpoint, and
arguments reuse the existing artifacts. `--force` rebuilds them. `--no-preview`
skips MP4 rendering, `--device cpu|mps|cuda` overrides automatic device choice,
and `--output PATH` selects another directory. Static/unknown camera motion is
currently represented by identity camera deltas. `image_mask` gates only the
optional learned crop features; 2D keypoints and their camera-aware reprojection
loss remain active for motion-only training.

Checkpoint version 5 introduces the Gravity-View target contract and the
detector-world prior. Absolute world yaw is no longer used as a target that
cannot be identified from monocular observations. Version 4 and older
checkpoints are rejected by training resume and learned inference instead of
silently mixing incompatible coordinate systems. The no-checkpoint detector
baseline remains available independently. The current converter-v4/v5 shard
inventory remains valid: Gravity-View targets and the synthetic detector-world
prior are derived deterministically by the loader, so this checkpoint change
does not require another shard rebuild.
This repository still ships no trained weights, so a new checkpoint must be
trained from regenerated shards before the 3D stage is representative. The 2D
rig frontend does not require that checkpoint and remains usable independently.

MediaPipe tracking improves temporal stability but still assumes one principal
person. Long detector gaps fail closed instead of silently inventing a track;
adjust `--max-missing-frames` only after checking `preview-rig-2d.mp4`. A clean 2D
preview does not guarantee a good 3D result: that also depends on the capacity
and training state of the chosen Gravity Mocap checkpoint.

The v2 motion model consumes both canonical 2D rows and the separately
provenanced detector-world prior. The 2D `x/y` coordinates are normalized to
`[-1, 1]` within the person's bbox; the bbox itself is `[x1, y1, x2, y2]`
normalized to the full frame's `[-1, 1]` coordinates. World landmarks are
converted with `[x, -y, -z]`, confidence-weighted, smoothed, and retargeted to
the same fixed neutral skeleton before entering the model. Missing values keep
confidence `0`; long frame gaps fail closed.

The training loader deterministically degrades clean projected keypoints per
epoch to resemble a real detector. Because its random seed is derived from the
epoch, sequence, and window, a mid-epoch checkpoint resumes with identical
inputs even when DataLoader workers change. Visual features remain optional and
are completely gated when `image_mask` is zero.

The clean projection is also deterministic and configurable in
`configs/datasets.yaml` under `preprocessing.synthetic_camera`: initial camera
yaw/pitch/roll, distance, and offsets are sampled per sequence, then receive a
small random-walk drift. Resolved settings are included in every shard's
provenance and preprocessing hash.

Large paired-video datasets differ in folder layout and annotation format. They
should first be normalized to the generic contract or a visual JSONL manifest;
the original source/license ID must be preserved. Unknown identifiers are not
accepted by the catalog.

## Visual features

The visual encoder deliberately starts from random weights. A JSONL manifest
contains one frame per line:

```json
{"image":"frames/000001.jpg","keypoints_2d":"22 x [x,y,confidence] values","source_id":"mri","license_id":"CC0-1.0"}
```

The keypoint value above is abbreviated for readability; real JSON contains the
22 arrays. Coordinates are normalized to `[-1, 1]`.
Plan the visual training first, then execute it yourself:

```sh
./scripts/mocap.sh train-vision manifest.jsonl \
  --output Saved/GravityMocap/checkpoints/frame-encoder.pt

./scripts/mocap.sh train-vision manifest.jsonl \
  --output Saved/GravityMocap/checkpoints/frame-encoder.pt --device mps --execute

./scripts/mocap.sh extract-features manifest.jsonl \
  Saved/GravityMocap/checkpoints/frame-encoder.pt features.npz --device mps
```

Features exported by the encoder can be attached to an existing shard using
`attach-features SHARD FEATURES.npz`. The feature file needs `image_features`
and may also replace `keypoints_2d` and `bbox`.

## Start motion training yourself

### Smaller 11.7M-parameter v2 baseline

The recommended model for the current approximately 31--32 hour `core` corpus
keeps the paper config's dataset, seed,
split, augmentations, optimizer, validation, and early stopping unchanged. It
changes only the Transformer from 12x512 (39.6M parameters) to 6x384 (11.7M)
and writes to an isolated `Saved/GravityMocap/runs/motion-small-v2/` output.
The old `motion-small/` version-4 checkpoints remain untouched but cannot be
resumed into the v2 contract.

Preview and then start its first three-epoch session:

```sh
./scripts/train-small.sh --max-epochs 3 --resume never
./scripts/train-small.sh --execute --max-epochs 3 --resume never
```

The isolated three-epoch canary has a dedicated wrapper and is also dry-run by
default:

```sh
./scripts/train-v2-canary.sh
./scripts/train-v2-canary.sh --execute
```

Resume the same small-model run later without `--resume never`:

```sh
./scripts/train-small.sh --execute --max-epochs 10
```

The small and paper checkpoints are architecturally incompatible by design;
their separate output directories prevent accidental cross-resume.

The corpus and capacity calculation, including the full-AddBiomechanics
scenario, is recorded in [docs/model-capacity.md](docs/model-capacity.md).

### Paper-size model

Inspect the full paper-size job:

```sh
./scripts/train.sh
```

Start it only when the dataset inventory is correct:

```sh
./scripts/train.sh --execute
```

For an eight-hour overnight session that stops safely:

```sh
./scripts/train.sh --execute --max-hours 8
```

To stop after a fixed number of completed epochs instead of elapsed time:

```sh
./scripts/train.sh --execute --max-epochs 10
```

Run the same command on the next night. `--resume auto` is the default, so it
loads `latest.pt` together with optimizer, AMP scaler, RNG, epoch, batch, and
global-step state. A checkpoint is atomically replaced every 15 minutes and at
every epoch; the last three configured epoch archives are retained. Progress is
also readable in `training-state.json`.

Press Ctrl+C once for a graceful save at the next optimizer-step boundary.
SIGTERM behaves the same way. A second Ctrl+C is a hard abort. Resume rejects a
changed model, optimizer, loss, sequence setup, or dataset BOM; increasing only
the target epoch count is allowed. `--max-epochs` is a per-session limit and
does not change that configured total. Use a new output directory for an
intentionally incompatible run, or `--resume never` to require an empty one.

The paper and small-model configs hold out 10% of each source by complete
subject when identity is available, otherwise by complete sequence. Validation
never receives detector corruption and runs after every completed epoch. It
logs held-out component losses, MPJPE, root-velocity error, integrated local
root drift, acceleration error, and contact F1 to MLflow and writes the latest
snapshot to `validation-state.json`. No validation window comes from a training
subject/sequence holdout unit. The paper config monitors `loss.total` with
`min_delta: 0.001` and stops after 20 consecutive validations without a
meaningful improvement. The counter and best epoch/loss live in the full-state
checkpoint, so session limits and resume do not reset patience. Each improvement
atomically promotes `latest.pt` to `best.pt`; early stopping ends the MLflow run
as `FINISHED`.

Training prints one line per optimizer step by default, so a healthy run is
immediately visible:

```text
[train] RESUME | device=mps | parameters=... | windows=... | epoch=8/500 | step=7/500
[mlflow] run=... | experiment=gravity-mocap | tracking=sqlite:////.../mlflow.db
[train] epoch 8/500 | batch 1/1 | step 8/500 | loss 0.369439 | lr 2.00e-04 | warming up | elapsed 1m 01s | ETA 1h 03m
[validation] START | epoch 8/500 | windows=1,025
[validation] DONE | epoch 8/500 | train_loss=0.612345 | val_loss=0.488491 | MPJPE=29.73cm | root_drift=59.97cm | contact_F1=0.2500 | time=26s | best=0.488491 | IMPROVED | early_stop=0/20
```

Open the local MLflow UI in a second terminal:

```sh
./scripts/mlflow-ui.sh
```

Then visit [http://127.0.0.1:5000](http://127.0.0.1:5000). The first UI start
may take a moment because `uvx` installs the full UI in an isolated tool
environment; the training environment uses the smaller Apache-2.0
`mlflow-skinny` client. The SQLite store and artifacts live under
`Saved/GravityMocap/mlflow/` and are ignored by Git.

MLflow records flattened config parameters, all component losses, epoch means,
held-out validation metrics, learning rate, elapsed time, throughput,
`resolved-config.json`, `data-bom.json`, `training-state.json`, and a checkpoint
manifest. The same run ID is kept in
`latest.pt` and `mlflow-run.json`, so repeated overnight sessions append to one
run. Full checkpoint upload is off by default to avoid duplicating a roughly
450 MB file at every save; set `logging.mlflow.log_checkpoints: true` only when
that storage cost is intentional. `logging.log_every_steps` controls terminal
and step-metric frequency. A dry-run never creates the MLflow database or run.

An interrupted atomic checkpoint write can leave `latest.pt.tmp` or
`best.pt.tmp`. The next real training execution deletes stale checkpoint
temporary files before loading the last committed `latest.pt`; dry-runs remain
read-only.

Checkpoints and resolved configuration are written below
`Saved/GravityMocap/runs/motion/`. Training is intentionally never invoked by
project automation or validation tests.

The full operator handoff, architecture notes, and dataset roles are documented
in [docs/training.md](docs/training.md).
