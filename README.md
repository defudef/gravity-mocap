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
validates the project, downloads and preprocesses the currently supported CMU
and AddBiomechanics motion sources, requires shards from both, validates a fresh
training plan, archives an existing default run, and finally starts training.
It deliberately does not download SAM or feed mRI into training until their
release-specific adapters exist. If preparation fails, the previous run is not
archived and training does not start.

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
   selected AddBiomechanics sequence. Generated training shards are written to
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

### Add the public mRI paired-video data

The mRI download is separate from the small motion-only quickstart. It needs
about 14.6 GiB plus temporary disk headroom and Node.js 18+ with `npx`:

```sh
./scripts/mocap.sh download --dataset mri
./scripts/mocap.sh download --dataset mri --allow-large --execute
```

The second command opens a temporary Chrome window, needs no account or click,
verifies both archives, and resumes partial `.part` files when repeated.
Corrupt completed or partial files are discarded and fetched again
automatically; a failed resumed checksum triggers one clean retry from byte
zero. The current generic `preprocess` command does not yet convert mRI's
release-specific paired-video annotations. Before mRI can contribute visual
supervision, those annotations must be normalized to the JSONL contract
described in [Visual features](#visual-features), then encoded and attached to
matching motion shards.

## What is implemented

- closed dataset/license allowlist with fail-closed source validation;
- resumable HTTP/Hugging Face/Google Drive downloaders and selective remote ZIP
  extraction for the 388 GiB AddBiomechanics archive;
- generic NPZ, CMU ASF/AMC, and optional AddBiomechanics B3D motion adapters;
- provenance-bearing, finite-value-validated NPZ shards;
- canonical 30 FPS preprocessing with source/target rates in provenance;
- simulated full camera motion, bbox, and 2D-keypoint inputs for motion-only sequences;
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

SAM crosses the 20 GiB safety threshold, so the full core command also needs
`--allow-large` after the dry-run and disk-space review. SAM contains motion and
spatial audio, not video. A single source can be selected with
`--dataset addbiomechanics`.

`smoke` selects only one tiny B3D member from the remote AddBiomechanics ZIP,
but the CMU archive is still about 1 GiB. `core` adds selected B3D members, SAM,
and mRI. CMU's server currently omits its TLS intermediate certificate. The
downloader still tries HTTPS first and permits only a same-host/path HTTP
fallback protected by the pinned archive SHA-256; TLS verification is never
disabled. The `core` profile deterministically selects the 12 smallest matching
AddBiomechanics members (currently about 0.81 GiB total). They show per-file
progress, are written through `.part`, and are promoted only after ZIP size/CRC
verification; an interrupted member restarts individually. The public mRI
archives are downloaded automatically without an account: a temporary visible
Chrome window passes Dryad's intended browser
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
resume the existing `.part` file. If an existing completed or partial file fails
the pinned size/checksum verification, it is discarded and downloaded again
automatically.

Only these raw formats are converted directly today:

- CMU `.asf` + `.amc` pairs;
- AddBiomechanics `.b3d` through the pinned `nimblephysics` dependency;
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
and root velocity is stored in metres per second. Changing that temporal
contract invalidates old shards and checkpoints by design.

## External 2D detector contract

The motion model does not require a particular detector. A frontend maps its
output to the canonical 22-joint order from `src/gravity_mocap/skeleton.py`,
then supplies one `[x, y, confidence]` row per joint. `x/y` are normalized to
`[-1, 1]` within the person's bbox; the bbox itself is `[x1, y1, x2, y2]`
normalized to the full frame's `[-1, 1]` coordinates. The
`normalize_detector_inputs` helper converts pixel values to this contract.
Missing joints use confidence `0` and zero coordinates.

The training loader deterministically degrades clean projected keypoints per
epoch to resemble a real detector. Because its random seed is derived from the
epoch, sequence, and window, a mid-epoch checkpoint resumes with identical
inputs even when DataLoader workers change. Visual features remain optional and
are completely gated when `image_mask` is zero.

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

The paper config holds out 5% of each source by complete subject when identity
is available, otherwise by complete sequence. Validation
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
