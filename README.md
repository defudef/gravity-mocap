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

## What is implemented

- closed dataset/license allowlist with fail-closed source validation;
- resumable HTTP/Hugging Face/Google Drive downloaders and selective remote ZIP
  extraction for the 388 GiB AddBiomechanics archive;
- generic NPZ, CMU ASF/AMC, and optional AddBiomechanics B3D motion adapters;
- provenance-bearing, finite-value-validated NPZ shards;
- simulated camera, bbox, and 2D-keypoint inputs for motion-only sequences;
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

SAM crosses the 20 GiB safety threshold, so the full core command also needs
`--allow-large` after the dry-run and disk-space review. SAM contains motion and
spatial audio, not video. A single source can be selected with
`--dataset addbiomechanics`.

`smoke` selects only one tiny B3D member from the remote AddBiomechanics ZIP,
but the CMU archive is still about 1 GiB. `core` adds selected B3D members, SAM,
and mRI. The public mRI archives are downloaded automatically without an
account: a temporary visible Chrome window passes Dryad's intended browser
challenge, captures a short-lived signed asset URL, cancels the browser
transfer, and hands the URL to the resumable HTTP downloader. Node.js 18+ with
`npx` is required only for this Dryad step. The window closes automatically,
partial `.part` files survive interruption, and every completed archive is
checked against Dryad's pinned size and SHA-256. The pretrained `.pkl` files
shipped inside `dataset_release.zip` are ignored and must not be used.
`expanded` also includes HUM4D paired video and TUM radar/3D-pose motion; TUM
requires reading and explicitly accepting its Data Usage Agreement.

Download only mRI, first as a dry-run and then for real:

```sh
./scripts/mocap.sh download --dataset mri
./scripts/mocap.sh download --dataset mri --allow-large --execute
```

The execute command opens Chrome but requires no click, login, or account. If
the connection stops, repeat the same command to obtain a fresh signed URL and
resume the existing `.part` file.

Only these raw formats are converted directly today:

- CMU `.asf` + `.amc` pairs;
- AddBiomechanics `.b3d` through the pinned `nimblephysics` dependency;
- generic `.npz` with `positions` (or `joints_3d`/`joint_positions`),
  `joint_names`, and optional `fps`.

Convert downloaded motion sources:

```sh
./scripts/mocap.sh preprocess --profile core
```

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

Run the same command on the next night. `--resume auto` is the default, so it
loads `latest.pt` together with optimizer, AMP scaler, RNG, epoch, batch, and
global-step state. A checkpoint is atomically replaced every 15 minutes and at
every epoch; the last three configured epoch archives are retained. Progress is
also readable in `training-state.json`.

Press Ctrl+C once for a graceful save at the next optimizer-step boundary.
SIGTERM behaves the same way. A second Ctrl+C is a hard abort. Resume rejects a
changed model, optimizer, loss, sequence setup, or dataset BOM; increasing only
the target epoch count is allowed. Use a new output directory for an
intentionally incompatible run, or `--resume never` to require an empty one.

Training prints one line per optimizer step by default, so a healthy run is
immediately visible:

```text
[train] RESUME | device=mps | parameters=... | windows=... | epoch=8/500 | step=7/500
[mlflow] run=... | experiment=gravity-mocap | tracking=sqlite:////.../mlflow.db
[train] epoch 8/500 | batch 1/1 | step 8/500 | loss 0.369439 | lr 2.00e-04 | warming up | elapsed 1m 01s | ETA 1h 03m
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
learning rate, elapsed time, throughput, `resolved-config.json`, `data-bom.json`,
`training-state.json`, and a checkpoint manifest. The same run ID is kept in
`latest.pt` and `mlflow-run.json`, so repeated overnight sessions append to one
run. Full checkpoint upload is off by default to avoid duplicating a roughly
450 MB file at every save; set `logging.mlflow.log_checkpoints: true` only when
that storage cost is intentional. `logging.log_every_steps` controls terminal
and step-metric frequency. A dry-run never creates the MLflow database or run.

An interrupted atomic checkpoint write can leave `latest.pt.tmp`. The next real
training execution deletes stale checkpoint temporary files before loading the
last committed `latest.pt`; dry-runs remain read-only.

Checkpoints and resolved configuration are written below
`Saved/GravityMocap/runs/motion/`. Training is intentionally never invoked by
project automation or validation tests.

The full operator handoff, architecture notes, and dataset roles are documented
in [docs/training.md](docs/training.md).
