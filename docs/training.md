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

The mRI release is public and does not require an account, but Dryad rejects
anonymous scripted downloads. Download `dataset_release.zip` and
`blurred_videos.zip` in a browser from the URL printed by the command and place
both at their printed paths. Their published SHA-256 values are pinned and
checked. Do not use the pretrained `.pkl` models included in the release. Then
run:

```sh
./scripts/mocap.sh download --profile core --allow-large --execute
./scripts/mocap.sh preprocess --profile core
```

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

Omitting `--execute` prints the resume decision and never creates an optimizer
or enters a training loop.
