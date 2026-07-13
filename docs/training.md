# Clean-room world-grounded mocap training

## Decision

This repository is the auditable training path. It is an independent
Apache-2.0 clean-room implementation inspired by the public description in
*World-Grounded Human Motion Recovery via Gravity-View Coordinates*
(arXiv:2409.06662). It does not import code, configs, weights, caches, or SMPL
parameters from GVHMR/WHAM. Any separately installed GVHMR runner remains an
unrelated research-only tool and is outside this repository.

The project is not affiliated with or endorsed by the GVHMR authors. Apache-2.0
covers our source only. Dataset terms and future checkpoint release obligations
remain separate; see `LICENSE_AUDIT.md`.

This is engineering provenance, not a legal opinion. Before distributing a
trained checkpoint, preserve the generated data BOM and review attribution/data
use obligations with whoever owns release compliance.

## Body representation

Neither SMPL nor SMPL-X is used. The model predicts a neutral 22-joint tree:
pelvis/root, three spine joints, neck/head, clavicles, shoulders, elbows,
wrists, hips, knees, ankles, and toes. Per-frame outputs are local 6D joint
rotations, camera-frame orientation, Gravity-View orientation, local root
velocity, weak camera, and six stationary probabilities (hands, toes, heels).

The deliberate difference from the paper is removal of body shape and vertex
losses. FK joint loss, 2D reprojection, temporal smoothness, contacts, root
velocity, and orientation losses remain.

## Architecture

Each frame fuses four independently projected inputs by element-wise addition:

1. bounding box;
2. 22 normalized 2D keypoints plus confidence;
3. 512-dimensional clean visual features;
4. relative camera rotation in 6D form.

The production config uses 12 relative Transformer blocks, 8 attention heads,
512 hidden dimensions, RoPE, and a receptive-field attention mask. Motion-only
datasets receive deterministic simulated static/dynamic camera paths, projected
2D joints, and zero visual features. Paired-video shards use features from the
from-scratch crop encoder. The trainer uses CUDA mixed precision, all visible
CUDA devices through `DataParallel`, micro-batches of 16, and 16-step gradient
accumulation for an effective batch of 256; CPU/MPS remain valid slower paths.

## Dataset gate

`configs/datasets.yaml` is the only source of truth. Training fails closed when
a shard has an unknown source, mismatched license ID, invalid schema, or
non-finite value.

| Profile | Sources | Intended use |
| --- | --- | --- |
| `smoke` | CMU, one remote AddBiomechanics B3D | exercise download/conversion without the 389 GiB archive |
| `core` | CMU, selected AddBiomechanics, SAM, mRI | motion corpus plus mRI paired video; SAM is motion/audio only |
| `expanded` | core + HUM4D + TUM prehab | HUM4D adds paired video; TUM adds radar/3D pose and requires DUA acceptance |

AMASS, BEDLAM, Human3.6M, 3DPW, HumanML3D, and the GVHMR repo are explicitly
blocked. Dataset binaries stay in `Saved/GravityMocap` and are ignored by Git.
Every converted shard records the original SHA-256, source ID, license ID,
converter version, and preprocessing hash. Training writes `data-bom.json` and
embeds it in checkpoints.

## Operator flow

Install and validate without training:

```sh
./scripts/setup.sh
./scripts/mocap.sh audit
./scripts/mocap.sh validate
./scripts/train.sh
```

Plan data acquisition:

```sh
./scripts/mocap.sh download --profile core
```

The mRI release is public and requires no account. Dryad places a JavaScript
browser challenge in front of the file stream, so the downloader launches a
temporary headed Chrome session through pinned Apache-2.0 `@playwright/cli`.
Chrome passes the challenge and yields a signed Dryad S3 URL; its own download
is cancelled, and the existing HTTP path performs the actual resumable transfer
into `Saved/GravityMocap/raw/mri/*.part`. The browser session is temporary and
stores no project credential. Node.js 18+ and `npx` must be on `PATH` for this
step. Published sizes and SHA-256 values remain pinned and checked. Do not use
the pretrained `.pkl` models included in the release. Run:

```sh
./scripts/mocap.sh download --profile core --allow-large --execute
./scripts/mocap.sh preprocess --profile core
```

For mRI alone:

```sh
./scripts/mocap.sh download --dataset mri
./scripts/mocap.sh download --dataset mri --allow-large --execute
```

The first command remains side-effect-free and does not launch Chrome. The
second opens and closes Chrome automatically. Repeating it after an interrupted
connection obtains a new signed URL and resumes the `.part` file with an HTTP
Range request before final checksum verification.

Normalize paired-video annotations into the documented visual JSONL contract,
train the visual encoder, extract features, and attach one feature file to the
matching motion shard. These training commands also remain dry by default.

When the shard inventory and license audit are both correct, the operator—not
automation—starts the paper-size motion training:

```sh
./scripts/train.sh --execute
```

For repeatable overnight sessions:

```sh
./scripts/train.sh --execute --max-hours 8
```

The same command resumes `latest.pt` on every subsequent night. The checkpoint
contains model, optimizer, AMP scaler, RNG and exact next epoch/batch/global
step. It is replaced atomically every 15 minutes and after every epoch;
`training-state.json` provides a readable status summary. Ctrl+C or SIGTERM once
requests a save at the next optimizer boundary. A second Ctrl+C aborts without
waiting. Increasing the target epoch count is compatible; model, data BOM,
optimizer, loss, sequence, and seed drift fail closed.

## Progress and MLflow

The trainer flushes a terminal line after the first optimizer step, every
`logging.log_every_steps` steps, and at each epoch boundary. The paper config
uses `1`, which shows epoch/batch/global-step position, total loss, learning
rate, step pace, elapsed time, ETA, every checkpoint save, and the final stop
reason. This makes a slow MPS/CUDA step distinguishable from a hung process.

MLflow is enabled in both training configs. With `tracking_uri: auto`, it uses:

- backend: `Saved/GravityMocap/mlflow/mlflow.db`;
- artifacts: `Saved/GravityMocap/mlflow/artifacts/`;
- experiment: `gravity-mocap`.

Start the browser UI separately; it does not start or stop training:

```sh
./scripts/mlflow-ui.sh
# open http://127.0.0.1:5000
```

`MLFLOW_PORT=5001 ./scripts/mlflow-ui.sh` selects another port. Setting the
standard `MLFLOW_TRACKING_URI` environment variable overrides the automatic
local URI; an explicit config value overrides both. The project runtime pins
`mlflow-skinny`; the UI script runs full MLflow through an isolated, pinned
`uvx` environment so its large analytics dependency set does not enter the
training lock/runtime.

One logical training job remains one MLflow run across safe stops. Its run ID
is written immediately to `mlflow-run.json` and included in every subsequent
full-state checkpoint. Resume first uses the checkpoint ID and falls back to
the JSON file for checkpoints made before MLflow support existed. A completed
job ends as `FINISHED`; a safe `max_hours`/signal stop ends the current session
as `KILLED`, and the next start reopens the same run; an exception records
`FAILED` plus a bounded error tag.

Step metrics contain each loss component, total loss, epoch, learning rate,
elapsed seconds, and steps per second. Epoch means are logged separately.
Artifacts contain the resolved config, data BOM, readable training state, and
a checkpoint manifest with path, size, reason, and progress. To avoid copying a
large checkpoint on every periodic/epoch save, the `.pt` itself is not copied
unless `logging.mlflow.log_checkpoints: true` is explicitly configured.

Dry-run planning resolves and prints the intended MLflow URI but does not import
MLflow, create a database, create a run, delete stale files, construct an
optimizer, or enter the training loop. On an actual execution, stale atomic
save remnants (`latest.pt.tmp`, `training-state.json.tmp`, and epoch archive
temps) are removed before the committed checkpoint is resolved.

Omitting `--execute` prints the resume decision and never creates an optimizer
or enters a training loop.
