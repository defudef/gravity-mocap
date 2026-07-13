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
datasets are resampled to 30 FPS and receive deterministic simulated yaw,
pitch, roll, translation, projected 2D joints, and zero visual features. Root
translation is preserved in the bbox input and root velocity is measured in
metres per second. Paired-video shards use features from the from-scratch crop
encoder; zero-feature motion shards gate that modality completely. The trainer
uses CUDA mixed precision, all visible
CUDA devices through `DataParallel`, micro-batches of 16, and 16-step gradient
accumulation for an effective batch of 256; CPU/MPS remain valid slower paths.

The paper config treats a permissively licensed 2D pose estimator as a
replaceable frontend. Clean projected points are corrupted deterministically
with coordinate noise, confidence degradation, missing joints/frames,
contiguous occlusion, bbox jitter, and whole-window camera-input dropout. The
seed includes epoch, sequence path, and window start, so exact mid-epoch resume
does not depend on DataLoader worker RNG state.

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

For a path-independent fresh run using the currently supported CMU and
AddBiomechanics converters, preview and then execute the repository workflow:

```sh
./scripts/start-fresh-training.sh
./scripts/start-fresh-training.sh --execute --max-hours 2
./scripts/start-fresh-training.sh --execute --max-epochs 10
```

Execution synchronizes the environment, validates the allowlist/model,
downloads and preprocesses both sources, verifies that each produced shards,
dry-runs a new output, archives the previous default run, and starts training.
Any failure before the archive step leaves the previous run in place. This
command intentionally starts fresh; subsequent sessions resume with
`./scripts/train.sh --execute --max-hours 2` or
`./scripts/train.sh --execute --max-epochs 10`.

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

For AddBiomechanics, `core` deterministically selects the 12 smallest matching
B3D members rather than archive order (currently about 0.81 GiB instead of more
than 9 GiB). Each member is written atomically through `.part`, reports
progress, and must match the ZIP directory's size and CRC before use.

Preprocessing logs progress every ten sequences and is safely repeatable.
Existing atomic shards are reused only when their schema, provenance hash,
source hashes, converter version, and preprocessing hash match the current
input; stale or interrupted outputs are regenerated. The current converter
records both source FPS and canonical 30 FPS. An older non-resampled corpus is
therefore regenerated automatically, and its old checkpoint cannot be resumed
against the new BOM/configuration.

CMU's archive host currently serves an incomplete TLS chain. Downloads attempt
HTTPS normally. Only an SSL chain failure may select the configured HTTP URL,
and only when its hostname/path match HTTPS and the full archive SHA-256 is
pinned. The completed archive must pass both size and SHA-256 verification; the
implementation never uses `verify=False`.

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
Range request before final checksum verification. A corrupt completed or
partial file is discarded automatically. If a resumed file fails verification,
the downloader retries once from byte zero before reporting an error.

Normalize paired-video annotations into the documented visual JSONL contract,
train the visual encoder, extract features, and attach one feature file to the
matching motion shard. These training commands also remain dry by default.

When the shard inventory and license audit are both correct, the operator—not
automation—starts the paper-size motion training:

```sh
./scripts/train.sh --execute
```

For the recommended smaller capacity baseline, use the dedicated wrapper. It
selects `configs/train-small.yaml` and the isolated
`Saved/GravityMocap/runs/motion-small/` output while preserving the production
data, seed, split, augmentation, optimizer, validation, and early-stopping
settings. Only model capacity changes from 12x512 (39.3M parameters) to 6x384
(11.5M parameters):

```sh
./scripts/train-small.sh --max-epochs 3 --resume never
./scripts/train-small.sh --execute --max-epochs 3 --resume never
```

Subsequent sessions resume with
`./scripts/train-small.sh --execute --max-epochs N`. Never point the small
configuration at the paper model's output because their checkpoints are
intentionally incompatible.

For repeatable overnight sessions:

```sh
./scripts/train.sh --execute --max-hours 8
```

For a fixed number of completed epochs per session instead of a time budget:

```sh
./scripts/train.sh --execute --max-epochs 10
```

The same command resumes `latest.pt` on every subsequent night. The checkpoint
contains model, optimizer, AMP scaler, RNG and exact next epoch/batch/global
step. It is replaced atomically every 15 minutes and after every epoch;
`training-state.json` provides a readable status summary. Ctrl+C or SIGTERM once
requests a save at the next optimizer boundary. A second Ctrl+C aborts without
waiting. Increasing the target epoch count is compatible; model, data BOM,
optimizer, loss, sequence, and seed drift fail closed.

`--max-epochs N` counts completed epochs in the current invocation, including a
partially completed epoch resumed from a checkpoint. It saves at the epoch
boundary and the next invocation continues with the following epoch. The limit
does not replace the total `train.epochs` target. `--max-hours` and
`--max-epochs` are mutually exclusive.

The production split is deterministic and source-stratified: 5% of each source
is held out by whole subject where identity is available, otherwise by whole
sequence, never by overlapping window. Validation inputs
are clean and never receive detector augmentation. At every completed epoch the
trainer writes `validation-state.json` and logs held-out loss components,
MPJPE, root-velocity error, integrated local root drift, acceleration error,
and contact F1. The split fraction, split seed, target FPS, and augmentation
settings are checkpoint compatibility inputs.

The paper config performs early stopping on held-out `loss.total`. A validation
must improve the saved best by at least `0.001` to reset patience. Twenty
consecutive validation checks without such an improvement stop the logical job
successfully, mark it `FINISHED` in MLflow, and leave `best.pt` pointing at the
best full-state checkpoint. `latest.pt` still records the final stopping point.
Best loss, best epoch, and the patience counter are checkpointed, so
`--max-hours`, `--max-epochs`, Ctrl+C, and later resume do not reset them.
Resuming a checkpoint made before early-stopping support seeds the baseline from
its matching `validation-state.json` and starts patience at zero.

## Progress and MLflow

The trainer flushes a terminal line after the first optimizer step, every
`logging.log_every_steps` steps, and at each epoch boundary. The paper config
uses `1`, which shows epoch/batch/global-step position, total loss, learning
rate, step pace, elapsed time, ETA, every checkpoint save, and the final stop
reason. Validation prints an explicit `START` line followed by a `DONE` summary
with the full-epoch mean train loss beside validation loss, MPJPE, root drift,
contact F1, duration, best loss, and current early-stopping patience. Train loss
includes detector augmentation while validation is clean, so their trends and
growing gap matter more than equality of the raw values. This also makes a slow
MPS/CUDA step or validation pass distinguishable from a hung process. A
mid-epoch resume reports `train_loss=partial` for that first incomplete epoch.

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
or early-stopped job ends as `FINISHED`; a safe
`max_hours`/`max_epochs`/signal stop ends the current session as `KILLED`, and
the next start reopens the same run; an exception records `FAILED` plus a
bounded error tag.

Step metrics contain each loss component, total loss, epoch, learning rate,
elapsed seconds, and steps per second. Epoch means and held-out validation
metrics are logged separately.
Artifacts contain the resolved config, data BOM, readable training state, and
a checkpoint manifest with path, size, reason, and progress. To avoid copying a
large checkpoint on every periodic/epoch save, the `.pt` itself is not copied
unless `logging.mlflow.log_checkpoints: true` is explicitly configured.

Dry-run planning resolves and prints the intended MLflow URI but does not import
MLflow, create a database, create a run, delete stale files, construct an
optimizer, or enter the training loop. On an actual execution, stale atomic
save remnants (`latest.pt.tmp`, `best.pt.tmp`, `training-state.json.tmp`, and
epoch archive temps) are removed before the committed checkpoint is resolved.

Omitting `--execute` prints the resume decision and never creates an optimizer
or enters a training loop.
