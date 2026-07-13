# Clean-room boundary

This directory is a new implementation based on the public method description
in *World-Grounded Human Motion Recovery via Gravity-View Coordinates*
(arXiv:2409.06662), general robotics/graphics mathematics, and public data format
documentation.

Hard rules:

- Do not copy, translate, adapt, or import source code from `zju3dv/GVHMR`, WHAM,
  or any local checkout of those repositories.
- Do not load their checkpoints, configuration files, cached features, or derived
  SMPL parameters into this project.
- Do not use SMPL or SMPL-X files. The implementation uses the neutral 22-joint
  skeleton defined in `src/gravity_mocap/skeleton.py`. Generic NPZ input containing
  SMPL/SMPL-X/body-model fields is rejected even when it also contains joints.
- Only datasets marked `approved_for_training: true` in
  `configs/datasets.yaml` may be converted. Unknown source identifiers fail
  closed.
- Every produced shard records source, license, converter version, source hash,
  and preprocessing configuration hash.
- Dataset files and model checkpoints stay below `Saved/GravityMocap/` and are
  never committed or redistributed by this repository.

The implementation is Apache-2.0. Dataset licenses remain independent and must
be followed when distributing a trained model or derived artifact. In
particular, attribution metadata must travel with releases trained from
CC-BY/CMU sources. `CC-BY-4.0+DUA` data is not downloaded until the operator has
explicitly accepted the linked agreement. Dependency wheels are installed from
their publishers and are not relicensed or redistributed by this repository.
