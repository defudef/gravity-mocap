# Dataset size and model capacity

Snapshot: 2026-07-14. The values below size the temporal 2D-to-3D model, not a
future image detector. No training was run to produce this estimate.

## Current `core` corpus

All durations are measured after applying each dataset's supported cuts and
before the deterministic 10% holdout.

| Source | Hours | 30 FPS frames | Sequences / groups |
| --- | ---: | ---: | --- |
| CMU Mocap | 9.62 | 1,038,799 | 2,514 / 112 subjects |
| 100STYLE | 18.78 | 2,028,184 | 810 / 100 styles |
| AddBiomechanics core subset | 2.53 | 273,118 | 606 trials / 24 source groups |
| **Total** | **30.93** | **3,340,101** | **3,930 / 236 groups** |

The generated BOM and processed shard inventory provide these exact values. The
AddBiomechanics selection contains 24 train B3Ds from eight contributing
studies, totaling 10.65 GiB uncompressed.

Training uses 120-frame (4-second) windows with stride 60 (2 seconds). Ignoring
small losses at sequence boundaries, the corpus therefore produces:

- 53,731 windows in total;
- 47,763 training windows after the 10% grouped holdout;
- 5,968 validation windows;
- 5.73M temporal training tokens per epoch.

The previous 5% split selected only 7.8 seconds of AddBiomechanics validation
motion from one unusually short source group. The 10% split holds out two
AddBiomechanics groups with 31.5 minutes of motion and produces 27.48 training
hours plus 3.44 validation hours overall. This makes early stopping materially
more representative while keeping complete subjects, source files, and styles
disjoint between train and validation.

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
for roughly 47.8k training clips.

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
