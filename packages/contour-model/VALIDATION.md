# Initial contour model validation — 2026-09-06

Selected artifact: `artifacts/release/model.pt`, exported from
`artifacts/final-joint2/best.pt`, step **88,000** of the user's 100,000-step run.
The 100,000-step run improved held-out recognition as well as training loss.
The original run folders have been preserved. Release metadata includes checkpoint
and dataset hashes; optimizer state is excluded from the 3.1 MB inference artifact.

## Checkpoint comparison and held-out recognition

Checkpoint selection used the mean of semantic F1 and query instance precision and
recall on 500 validation parts. We confirmed the ranking on all 3,892 validation
parts before evaluating the selected model on test and distribution-shift sets.

| Checkpoint / split | Parts | Semantic macro F1 | Query instance precision | Query instance recall |
|---|---:|---:|---:|---:|
| final-joint, 20,000 steps / validation | 3,892 | 99.04% | 90.51% | 94.04% |
| final-joint2, 88,000 steps / validation | 3,892 | 99.57% | 98.97% | 99.31% |
| Selected / test | 4,002 | 99.54% | 98.97% | 99.34% |
| Selected / distribution shift | 2,026 | 98.78% | 97.46% | 90.81% |

Semantic metrics use observed arc-length weights and macro-average supported primary
classes, excluding unlabeled. Polygon and rounded-profile ancestors are evaluated
separately as multilabel targets. Instance matches require the same class and at
least 0.5 weighted IoU with one-to-one Hungarian matching; raw query confidence
threshold is 0.35. These are query masks, not the final measurement selections.
All per-class and per-family results are in `artifacts/comparison/*-{val,test,ood}.json`.
The distribution shift consists of changed compositions/opening distributions in
three mixed families, not arbitrary unseen industrial part grammars.

## Measurement handoff

The handoff uses whole-loop consensus, geometry-supported surface splitting and
learned association of repeated profiles, at acceptance threshold 0.7. It returns
original raw edges and nearby boundary search regions. An array seam midway along
a same-class surface no longer introduces a false feature boundary.

| Checkpoint / split | Parts | Accepted instance precision | Accepted instance recall |
|---|---:|---:|---:|
| final-joint / validation | 300 | 93.06% | 97.49% |
| Selected / validation | 300 | 96.03% | 99.00% |
| Selected / test | 300 | 96.25% | 99.38% |
| Selected / distribution shift | 300 | 95.37% | 98.35% |

These are the first 300 matching records in the same corpus traversal for each
split, not a stratified or independently acquired sample. Ancestor overlaps are
excluded. Higher recall than raw queries is possible because the hybrid handoff
uses complete semantic runs and closed-loop geometry.

For selected test predictions, 2,423 of 2,424 matched reference-measurable features
produced a diagnostic fit. Comparing identical classical fitters using predicted
versus reference selections on the **same imperfect observations** gave:

| Quantity | Matched fits | Median absolute difference (mm) | 95th percentile (mm) |
|---|---:|---:|---:|
| Diameter (holes and fitted arcs pooled) | 592 | 0 | 0.00446 |
| Slot width / overall length | 94 | 0 | 0 |
| Visible thread pitch | 22 | 0.00102 | 0.01622 |
| Projected line support extent | 1,715 | 0.03907 | 0.21384 |

These measure **selection-induced differences**, not absolute metrology accuracy.
Zero differences for full loops mean the fitter received exactly the same points.
Only correctly matched features enter these fit comparisons; missed/spurious
features remain in precision/recall above. The thread-pitch 95th percentile rose
to **0.367 mm** on 38 distribution-shift matches. The reference pitch estimator is
therefore diagnostic and needs application-specific qualification. Exact endpoints,
virtual-sharp lengths, datum association and standard-specific GD&T need downstream
classical modules. Unsupported routes explicitly return `unsupported`.

CPU preprocessing, network inference and selection took a median **28.1 ms** and
95th percentile **41.2 ms** per part on this host with four PyTorch threads, excluding
file reading and classical fitting. This is an observed run, not a latency guarantee.

## Robustness and inspection

The same 100 held-out synthetic parts were evaluated with geometry/representation
changes, using raw-edge length-weighted semantic F1:

| Probe | F1 |
|---|---:|
| Trained 1,024-interval budget | 99.30% |
| Rotation by 0.731 radians | 99.31% |
| Shift ring starting vertex | 99.18% |
| Reverse ring winding | 99.30% |
| Insert midpoints on every raw edge | 99.21% |
| Change budget to 512 intervals | 89.15% |
| Change budget to 2,048 intervals | 88.17% |

**Sampling scale is part of this checkpoint's contract.** The exported inference
CLI enforces the 1,024 budget. Original vertex count may vary. Multi-resolution
training is future work; using more tokens with these weights degrades results.

A 15-family reference/prediction contour sheet was generated and visually inspected
at `artifacts/comparison/contour-review.png`. Hole/slot loops, repeated profiles and
major surface classes align well in those examples; small disagreements occur at
feature transitions. This sheet is a diagnostic sample, not exhaustive review.

The test suite checks label-blind resampling, overlap conservation, winding and
scale behavior, augmentation target alignment, compound label adaptation, both
backbones' forward/backward passes, ring permutation, ignored ground truth at
inference, wrapped spans, seam splitting, reference fits, abstention, image-bridge
topology rejection, checkpoint export round trips and configuration guards.
All **14 tests passed**. Saved slotted-bracket, mixed-shaft and threaded-shaft NPZ
examples also completed CPU inference and JSON/probability export with diagnostic
measurements; unsupported routes were reported explicitly.

The selected model additionally scored 99.68% semantic F1 on the original 121-part
synthetic test split and 98.92% on 28 test contours extracted from synthetic
grayscale images. This small bridge check includes optical rasterization/extraction,
but no real images. Reports are `cache-delivered-test.json` and `bridge-test.json`.

## Scope and remaining work

This is an implemented, trained **synthetic prototype**, not a production-qualified
inspection model. No real part capture dataset was supplied. The small optical
bridge is based on synthetic rendered images, with uncertain correspondences
rejected; it cannot establish real-world transfer. Confidence and defect heads
have not been calibrated/qualified for deployment. OOD detection is not established.

The release retains the current semantic taxonomy. Some intended uses are
indistinguishable from silhouettes alone; ambiguous grooves/notches or thread/rack
interpretations require an application policy or a future generic geometry label.
A part's topology is available, but its family, drawing datums and tolerance
requirements are not inferred. Fragmented contours, touching rings, multiple parts
and nested islands are outside the supported input contract.

The earlier local-convolution control retained global attention and had different
initialization history; it does not establish that a transformer is necessary or
superior. Comprehensive architecture ablations, resolution-aware training, real-data
fine-tuning, confidence calibration and standardized measurement validation remain
on the longer-term training-plan roadmap.
