# Engineering license audit

Snapshot: 2026-07-13. This is an engineering provenance review, not legal
advice, a patent/FTO search, or a guarantee about how a court would classify
trained weights.

## Result

No known research-only, non-commercial, GPL/AGPL, SMPL, or SMPL-X component is
included in the default training implementation or loaded into a checkpoint.
That conclusion depends on the fail-closed rules below and on distributing
source, datasets, dependencies, and checkpoints under their own terms.

The source in this project is Apache-2.0. The method and coordinate formulation
are independently implemented from the public
[GVHMR paper](https://arxiv.org/abs/2409.06662). The project is not affiliated
with or endorsed by the GVHMR authors. The
[GVHMR repository license](https://github.com/zju3dv/GVHMR/blob/main/LICENSE)
limits its software to educational, research, and non-profit use, so its code,
configs, weights, caches, and derived SMPL parameters are explicitly blocked.

## Dataset allowlist

| Source | Terms used by this project | Approved role | Required handling |
| --- | --- | --- | --- |
| [CMU Mocap](https://mocap.cs.cmu.edu/) | CMU permits all uses and inclusion in commercial products, but not resale of the data itself | motion | Preserve CMU acknowledgement; do not redistribute the raw/converted dataset as a product |
| [AddBiomechanics data](https://addbiomechanics.org/data_sharing_mission.html) | CC BY 4.0 | motion | Preserve source-study identity, the B3D source href, and original-publication attribution; the AddBiomechanics application is GPLv3 but its code is not used |
| [100STYLE](https://www.ianxmason.com/100style/) | CC BY 4.0 | styled BVH motion | Preserve author/dataset attribution and the checksum-pinned Zenodo source |
| [TUM prehabilitation](https://zenodo.org/records/19866202) | CC BY 4.0 plus Data Usage Agreement | radar + camera-derived 3D pose | Explicit acceptance, DOI attribution, no re-identification/contact/tracking/profiling; released files are not paired RGB supervision |

The default `core` profile uses only CMU, AddBiomechanics `train/With_Arm`, and
100STYLE. TUM is opt-in through `expanded` and requires explicit DUA acceptance.

| Blocked source | Why it cannot enter official checkpoints |
| --- | --- |
| [SAM](https://huggingface.co/datasets/JimSYXu/SAM) | The repository label is CC BY 4.0, but the raw motion scope and participant clearance need confirmation; audio and SMPL-X derivatives are excluded |
| [mRI](https://datadryad.org/dataset/doi:10.5061/dryad.9ghx3ffpp) | Dryad metadata says CC0 while the original publication states CC BY-NC 4.0; bundled pretrained models are also excluded |
| [HUM4D](https://parkyeeun23.github.io/HUM4D/) | Public processing depends on non-commercial SMPL/SMPL-X assets and the standalone raw Vicon-joint scope is unclear |

AMASS, BEDLAM, Human3.6M, 3DPW, HumanML3D, and the GVHMR repository also remain
blocked. Unknown `source_id` or license IDs fail closed. Generic NPZ input is
rejected when any field name indicates SMPL, SMPL-X, body pose/shape, global
orientation, or another parametric body model.

The official CMU bulk archive host currently omits its intermediate TLS
certificate. The downloader attempts verified HTTPS first and never disables
certificate verification. A fallback is accepted only for the configured HTTP
URL with the same hostname/path and a pinned SHA-256 derived from the official
HTTPS archive; size and SHA-256 are checked before the file becomes usable.
This is a source-integrity exception for one static archive, not a general TLS
downgrade mechanism.

## Python and GPU dependencies

Dependencies are installed from their publishers and are not vendored or
relicensed. Direct runtime dependencies are permissive: PyTorch (its upstream
BSD-style license plus the notices included in the wheel), NumPy (BSD-3-Clause
and included permissive notices), Pillow (MIT-CMU), PyYAML (MIT), Requests
(Apache-2.0), gdown (MIT), huggingface-hub (Apache-2.0), remotezip (MIT),
nimblephysics (a BSD-style three-clause license in the wheel, despite its `MIT`
package metadata; its included Rajagopal model notice is MIT), MLflow Skinny
(Apache-2.0), SQLAlchemy (MIT), and Alembic (MIT).

The resolved transitive environment also contains `certifi` under MPL-2.0 and
`tqdm` under its MIT/MPL dual license. They are separate dynamically installed
packages, not copied into this source. Linux PyTorch resolution may download
NVIDIA CUDA/cuDNN/NCCL wheels governed by NVIDIA's separate terms. Do not bundle
or redistribute dependency wheels under Apache-2.0; retain the notices shipped
inside any binary distribution.

MLflow tracking deliberately uses `mlflow-skinny` plus the MIT SQLite adapter
stack, not full MLflow. The resolved training environment was checked for
AGPL/GPL/LGPL license classifiers after this change and none were reported.
`scripts/mlflow-ui.sh` runs full Apache-2.0 MLflow in a separate pinned `uvx`
tool environment only when an operator asks for the browser UI. That optional
tool downloads additional analytics packages, including binary components with
their own notices; it is not imported by training, embedded in checkpoints, or
part of the locked training environment. Do not redistribute the `uvx` cache as
if it were Apache-2.0 project output.

The optional automated mRI download path invokes pinned `@playwright/cli`
0.1.17 (Apache-2.0) through an operator's Node.js/npm installation. It is not a
Python runtime dependency, is not imported by model training, and runs only
after explicit `download --execute`. A temporary local Chrome/Chromium process
passes Dryad's public browser challenge without credentials, and the project
stores neither challenge cookies nor signed URLs. Browser binaries and Node
dependencies retain their own licenses and must not be redistributed as this
project's Apache-2.0 source.

Dryad's current [End User Terms](https://datadryad.org/terms) allow and
encourage reuse of published datasets while prohibiting use that impairs the
service or compromises its security/functionality. The downloader follows the
normal public browser challenge once per session, uses the resulting official
signed asset URL for one sequential transfer, and does not evade rate limits,
parallelize ranges, retain session credentials, or alter the challenge. Stop
using the automated path if Dryad changes its terms or explicitly asks clients
not to automate this flow.

## Checkpoint release gate

Apache-2.0 applies automatically only to this source. Before releasing a trained
checkpoint:

1. retain the generated `data-bom.json`, resolved config, and source hashes;
2. publish a model card listing every dataset, license, DOI/citation, changes,
   and prohibited-use conditions;
3. include all CC BY and CMU acknowledgements and the TUM ethical restrictions
   when those sources are present;
4. verify that no dataset files, frames, SMPL/SMPL-X parameters, third-party
   weights, or dependency binaries are embedded in the release;
5. obtain a focused legal review if the checkpoint will be commercially
   distributed, because whether training output is an adaptation of input data
   is jurisdiction- and fact-dependent.
