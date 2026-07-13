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
- The user starts an overnight session with `./scripts/train.sh --execute --max-hours 8` or a fixed-epoch session with `./scripts/train.sh --execute --max-epochs N`. Repeating either resumes `latest.pt` automatically. The first Ctrl+C/SIGTERM saves at the next optimizer boundary; a second Ctrl+C aborts immediately.
- `./scripts/start-fresh-training.sh --execute --max-hours HOURS` and `./scripts/start-fresh-training.sh --execute --max-epochs N` are the path-independent one-command bootstraps for a new run. They may archive the current default output only after CMU/AddBiomechanics download, preprocessing, shard checks, and a fresh training dry-run succeed. Never use them to resume; use `scripts/train.sh` for subsequent sessions.
- Preserve atomic full-state checkpoints, config/BOM compatibility checks, RNG restoration, deterministic mid-epoch resume, and validation-loss early-stopping state. `best.pt` must be promoted atomically; safe session stops must not reset patience.
- Preserve canonical 30 FPS preprocessing, per-shard source/target FPS provenance, source-stratified subject/sequence holdout, and epoch/sequence/window-derived detector augmentation. A resumed batch must receive the same corrupted detector inputs.
- Real detector inputs use the canonical 22-joint order, bbox-relative keypoints in `[-1, 1]`, frame-relative bbox coordinates in `[-1, 1]`, and confidence zero for missing joints. Zero visual features must be gated by `image_mask`.
- Preserve flushed terminal progress, explicit validation START/DONE summaries, and reuse the MLflow run ID stored in the checkpoint. Dry-runs must not create an MLflow store/run or clean checkpoint temp files.
- The default tracker uses local SQLite below `Saved/GravityMocap/mlflow/` through `mlflow-skinny`. `scripts/mlflow-ui.sh` is a separate, isolated UI tool and must never start training.
- Keep full checkpoint artifact logging opt-in: `logging.mlflow.log_checkpoints: false` prevents repeated large checkpoint copies while the manifest and readable state remain logged.
- mRI execution uses pinned `@playwright/cli` in a temporary headed Chrome session to pass Dryad's public browser challenge, cancel the browser download, and obtain a short-lived signed S3 URL. Preserve resumable `.part` transfer and size/SHA-256 verification; never persist challenge cookies or signed URLs.
- Preserve automatic recovery: corrupt completed or partial downloads are discarded, and a failed resumed checksum gets one clean retry before surfacing an error.
- Selective remote ZIP members must be written through `.part`, report per-member progress, and become final only after ZIP size/CRC verification. Remote ZIP members restart individually because compressed streams cannot safely resume mid-member.
- Preprocessing must report bounded progress and reuse only atomic shards whose schema, provenance hash, source hashes, converter version, and preprocessing hash still match. Interrupted/stale outputs must be regenerated.
- CMU's HTTPS server currently omits its intermediate certificate. Preserve the narrow fallback rule: HTTPS first, then only the configured same-host/path HTTP URL when SHA-256 is pinned; never disable TLS verification or allow an unpinned/general downgrade.
- Dry-run downloads must not launch Chrome, install a browser, or create raw-data directories. Browser-assisted integration tests must use the 4.4 KB public mRI README, never either large archive.

## License release gate

- Apache-2.0 covers project source only. Dependencies, datasets, and trained checkpoints retain separate terms described in `LICENSE_AUDIT.md`.
- Before any checkpoint release, require its `data-bom.json`, resolved config, complete dataset citations/attributions, model card, and a focused legal review for the intended distribution.
