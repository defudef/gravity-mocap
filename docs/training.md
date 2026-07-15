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
wrists, hips, knees, ankles, and toes. Per-frame outputs are Gravity-View local
6D joint rotations, camera-frame orientation, local root
velocity, weak camera, and six stationary probabilities (hands, toes, heels).

The deliberate difference from the paper is removal of body shape and vertex
losses. FK joint loss, camera-aware frame-relative 2D reprojection, temporal
smoothness, contacts, root velocity, and orientation losses remain.

## Architecture

Each frame fuses five independently projected inputs by element-wise addition:

1. bounding box;
2. 22 normalized 2D keypoints plus confidence;
3. 512-dimensional clean visual features;
4. relative camera rotation in 6D form.
5. confidence-bearing detector world 3D retargeted to the neutral skeleton.

The production config uses 12 relative Transformer blocks, 8 attention heads,
512 hidden dimensions, RoPE, and a receptive-field attention mask. Motion-only
datasets are resampled to 30 FPS, retargeted to the fixed neutral skeleton, and
regenerated through FK so rotation and joint targets are exactly consistent.
They receive deterministic simulated yaw, pitch, roll, translation, auto-framed
non-degenerate bbox, bbox-relative 2D joints, and zero visual features. The
reprojection target is converted back to bounded frame-relative coordinates so
thin boxes cannot amplify an initialization error. Root
translation is preserved in the bbox input and root velocity is measured in
metres per second. Paired-video shards use features from the from-scratch crop
encoder; zero-feature motion shards gate only that modality. The 2D
reprojection remains active and supervises local pose, camera orientation, and
weak camera even when `image_mask` is zero. The trainer
uses CUDA mixed precision, all visible
CUDA devices through `DataParallel`, micro-batches of 16, and 16-step gradient
accumulation for an effective batch of 256; CPU/MPS remain valid slower paths.

Checkpoint contract v5 removes unobservable absolute world yaw from the target:
the root rotation and FK joint target live in Gravity-View coordinates. The
same audited MediaPipe bundle also provides metric world landmarks; these are
stored in a separate inference artifact and become a noisy neutralized prior,
while the learned model focuses on residual pose, root motion, camera relation,
and contacts. Clean projected points and synthetic detector-world priors are
corrupted deterministically
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
| `core` | CMU, 24 study-stratified AddBiomechanics train B3Ds, 100STYLE | commercially usable motion-only corpus |
| `expanded` | core + TUM prehab | adds radar/3D pose and requires DUA acceptance |

AMASS, BEDLAM, Human3.6M, 3DPW, HumanML3D, and the GVHMR repo are explicitly
blocked. Dataset binaries stay in `Saved/GravityMocap` and are ignored by Git.
Every converted shard records the original SHA-256, source ID, license ID,
converter version, and preprocessing hash. Training writes `data-bom.json` and
embeds it in checkpoints.

## Operator flow

For a path-independent fresh run using CMU, AddBiomechanics, and 100STYLE,
preview and then execute the repository workflow:

```sh
./scripts/start-fresh-training.sh
./scripts/start-fresh-training.sh --execute --max-hours 2
./scripts/start-fresh-training.sh --execute --max-epochs 10
```

Execution synchronizes the environment, validates the allowlist/model,
downloads and preprocesses all three sources, verifies that each produced shards,
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

For AddBiomechanics, `core` considers only `train/With_Arm`, then selects up to
24 B3D members in deterministic round-robin order across the eight studies and
under a 12 GiB uncompressed budget. The current archive resolves to 24 members
and 10.65 GiB. Each member is written atomically through `.part`, reports
progress, and must match the ZIP directory's size and CRC before use. Every B3D
trial becomes a separate shard, but all trials from the same B3D share one split
group and cannot cross the train/validation boundary.

100STYLE is downloaded from its checksum-pinned Zenodo record, extracted
atomically, parsed directly from BVH, trimmed with the published
`Frame_Cuts.csv`, and resampled from 60 to 30 FPS. Its 100 style directories are
the holdout groups.

Preprocessing logs progress every ten sequences and is safely repeatable.
Existing atomic shards are reused only when their schema, provenance hash,
source hashes, converter version, and preprocessing hash match the current
input; stale or interrupted outputs are regenerated. The current converter
records both source FPS and canonical 30 FPS. The training audit also rejects a
converter version older than the current auto-framed target contract. Such a
corpus is therefore regenerated automatically by preprocessing. Checkpoint
versions 2 and 3 remain obsolete, and version 4 is isolated as the former
2D-only/absolute-root contract. Version 5 requires a fresh run but derives its
Gravity-View targets and synthetic detector-world prior from the current valid
shards at load time, so the v4 shard inventory itself does not need rebuilding.

CMU's archive host currently serves an incomplete TLS chain. Downloads attempt
HTTPS normally. Only an SSL chain failure may select the configured HTTP URL,
and only when its hostname/path match HTTPS and the full archive SHA-256 is
pinned. The completed archive must pass both size and SHA-256 verification; the
implementation never uses `verify=False`.

mRI, SAM, and HUM4D remain catalogued for audit history but are not approved for
official checkpoint training. mRI's Dryad CC0 metadata conflicts with the
original publication's CC BY-NC statement; SAM's raw participant clearance
needs confirmation; HUM4D's public processing depends on prohibited
SMPL/SMPL-X assets and the raw Vicon scope is unclear. Explicit download or
preprocess requests for those IDs fail closed. Existing raw files are ignored.

When the shard inventory and license audit are both correct, the operator—not
automation—starts the paper-size motion training:

```sh
./scripts/train.sh --execute
```

For the current approximately 31--32 hour `core` corpus, the recommended model
is the smaller capacity baseline. Use the dedicated wrapper. It
selects `configs/train-small.yaml` and the isolated
`Saved/GravityMocap/runs/motion-small-v2/` output while preserving the production
data, seed, split, augmentation, optimizer, validation, and early-stopping
settings. Only model capacity changes from 12x512 (39.6M parameters) to 6x384
(11.7M parameters):

```sh
./scripts/train-small.sh --max-epochs 3 --resume never
./scripts/train-small.sh --execute --max-epochs 3 --resume never
```

Subsequent sessions resume with
`./scripts/train-small.sh --execute --max-epochs N`. Never point the small
configuration at the paper model's output because their checkpoints are
intentionally incompatible.

Version-4 `motion-small/` checkpoints remain isolated and cannot resume into
the Gravity-View/detector-prior v5 job. Its three-epoch canary improved MPJPE
from 15.32 cm to 10.87 cm, but the raw detector prior was already 4.31 cm and
contact F1 ended at zero. Do not extend that run.

The recommended successor is the checkpoint-v7 detector-safe residual model.
It enforces neutral bone lengths, starts with an exact zero correction to the
neutralized detector pose, and bounds learned corrections by detector
confidence. Preview and execute its isolated canary with:

```sh
./scripts/train-residual-canary.sh
./scripts/train-residual-canary.sh --execute
```

Use `scripts/train-residual-small.sh` for a fresh production output and later
resumes. It writes below `runs/motion-small-v3-residual`; never point it at a v5
output. A run may continue only when held-out `neutral_gain` becomes positive,
not merely because the aggregate loss decreases. `best-pose.pt` tracks minimum
held-out MPJPE independently from the aggregate-loss `best.pt`. The root head
starts from a stationary path and is speed-bounded; contact loss uses the six
class frequencies measured from the approved training corpus. Validation also
reports `accel_gain` against the neutral detector on the same valid joint
frames, so lower positional error cannot hide a temporally noisier animation.

The capacity calculation is in `docs/model-capacity.md`. In short, current
`core` yields roughly 53--54 thousand 4-second training windows after the 5%
holdout. That supports the 6x384, 11.7M-parameter model; the 12x512, 39.6M model
is retained as an explicit scaling experiment, not the default recommendation.

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
