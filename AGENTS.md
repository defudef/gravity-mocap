# Gravity Mocap Agent Notes

## Clean-room boundary

- This is an independent Apache-2.0 implementation inspired by the public GVHMR paper. Never copy or import GVHMR/WHAM code, configs, weights, caches, or derived artifacts.
- Do not add SMPL or SMPL-X assets. Generic NPZ input containing parametric-body fields must continue to fail closed.
- Only sources approved in `configs/datasets.yaml` may produce training shards. Preserve source hashes, license IDs/URLs, attribution flags, DUA acceptance, and the generated data BOM.
- SAM and TUM are motion supervision, not paired video. mRI paired-video use requires both pinned Dryad archives. Ignore bundled mRI pretrained `.pkl` models. HUM4D SMPL/SMPL-X parameters are prohibited.
- Keep all datasets, feature caches, checkpoints, and exports below `Saved/GravityMocap/`; never commit or redistribute them from this repository.

## Safe operation

- `download` and `train` are dry-run by default. Never add `--execute` to a training command unless the user explicitly requests starting training.
- Validate changes with `./scripts/mocap.sh audit`, `./scripts/mocap.sh validate`, `./scripts/train.sh --max-hours 8`, and tests. These checks must remain forward-only and must not call `backward()` or `optimizer.step()`.
- The user starts an overnight session with `./scripts/train.sh --execute --max-hours 8`. Repeating it resumes `latest.pt` automatically. The first Ctrl+C/SIGTERM saves at the next optimizer boundary; a second Ctrl+C aborts immediately.
- Preserve atomic full-state checkpoints, config/BOM compatibility checks, RNG restoration, and deterministic mid-epoch resume.
- Preserve flushed terminal progress and reuse the MLflow run ID stored in the checkpoint. Dry-runs must not create an MLflow store/run or clean checkpoint temp files.
- The default tracker uses local SQLite below `Saved/GravityMocap/mlflow/` through `mlflow-skinny`. `scripts/mlflow-ui.sh` is a separate, isolated UI tool and must never start training.
- Keep full checkpoint artifact logging opt-in: `logging.mlflow.log_checkpoints: false` prevents repeated large checkpoint copies while the manifest and readable state remain logged.

## License release gate

- Apache-2.0 covers project source only. Dependencies, datasets, and trained checkpoints retain separate terms described in `LICENSE_AUDIT.md`.
- Before any checkpoint release, require its `data-bom.json`, resolved config, complete dataset citations/attributions, model card, and a focused legal review for the intended distribution.
