# Dataset size and model capacity

Snapshot: 2026-07-13. The values below size the temporal 2D-to-3D model, not a
future image detector. No training was run to produce this estimate.

## Current `core` corpus

All durations are measured after applying each dataset's supported cuts and
before the deterministic 5% holdout.

| Source | Estimated hours | 30 FPS frames | Basis |
| --- | ---: | ---: | --- |
| CMU Mocap | 9.62 | 1,038,799 | current processed inventory |
| 100STYLE | 18.78 | 2,027,989 | sum of published frame cuts, resampled 60 to 30 FPS |
| AddBiomechanics core subset | 2.5--3.3 | 270k--355k | 24 selected train B3Ds; bounded by the local B3D seconds/byte ratio and selected/full archive share |
| **Total** | **30.9--31.7** | **3.34M--3.42M** | |

The AddBiomechanics duration remains an estimate until all 24 selected B3Ds are
downloaded and their trial headers can be summed. The selection itself is
exact: 24 B3Ds, eight contributing studies, 10.65 GiB uncompressed. Once
preprocessing finishes, the generated BOM provides the exact frame count.

Training uses 120-frame (4-second) windows with stride 60 (2 seconds). Ignoring
small losses at sequence boundaries, the corpus therefore produces:

- about 55.6k--57.0k windows in total;
- about 52.8k--54.2k training windows after the 5% grouped holdout;
- about 6.34M--6.50M temporal tokens per epoch.

Overlapping windows and adjacent frames are highly correlated. Camera sampling
and detector degradation increase input variety, but they do not create new 3D
motions, actors, or activities. In particular, 100STYLE contributes 100 useful
locomotion styles but only one actor. Capacity should therefore follow the
number of independent sequences/styles/subjects and validation behavior, not
the raw frame count alone.

## Recommendation

Use the existing **6 layers x 384 hidden, 11,526,300 parameter** model in
`configs/train-small.yaml` for the current `core` corpus. It is large enough for
the 22-joint temporal mapping while leaving a sensible regularization margin
for roughly 53--54k training clips.

Do not use the 12x512, 39,263,388-parameter paper-size configuration as the
default for this corpus. It has 3.4 times as many parameters without 3.4 times
as much independent motion, so it is a scaling experiment with a materially
higher overfitting risk. Early stopping protects a run, but does not make an
oversized model compute-efficient.

If the full 71.47 hours of AddBiomechanics are later ingested, the combined
corpus becomes approximately 99.87 hours, 10.79M 30 FPS frames, and 180k
overlapping windows. The next justified capacity is **8x384, 15,075,228
parameters**. Move to 39.3M only if a controlled capacity sweep shows
11.5M and 15.1M both underfit: train and validation losses should still fall
together, with no widening generalization gap and improved held-out motion
metrics.

The practical sweep is therefore 11.5M first, then 15.1M on the same seed,
split, augmentation, optimizer, and early-stopping settings. Dataset diversity
should be expanded before testing 39.3M.
