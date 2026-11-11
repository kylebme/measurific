# Contour semantic recognition and measurement handoff

This project trains a small hierarchical transformer to select semantic features
on original observed contours. It exports per-edge probabilities, feature instances,
contour spans, adjacency, defect flags and boundary search regions for classical
measurement modules. Dimensions come from those modules using the original points.

The expanded-corpus checkpoint is `artifacts/release-v2/model.pt`, selected from
`artifacts/v2-full-local/best.pt` at step 27,000. See [VALIDATION_V2.md](VALIDATION_V2.md)
for held-out results and limitations. The prior fixed-sampling checkpoint remains
at `artifacts/release/model.pt`, with its original [validation report](VALIDATION.md). The synthetic renderer remains CPU-only;
model training supports CUDA and inference supports CPU.

## Use the trained model

From the repository root:

```bash
packages/contour-model/.venv/bin/python -m contour_model predict \
  packages/contour-model/artifacts/release-v2/model.pt \
  /absolute/path/part.contours.npz \
  /absolute/path/prediction.json --device cpu --measure
```

The existing `.venv` reuses this machine's installed PyTorch environment via a
`.pth` file. For an independent installation, create a Python 3.11+ virtual
environment, install an appropriate PyTorch build, then run:

```bash
python -m pip install -e ./packages/contour-model
# Only required for corpus generation, synthetic-data tests and image bridging:
python -m pip install -e ./research/synthetic-shapes
# Optional image-to-contour bridge and tests:
python -m pip install -e './packages/contour-model[bridge,test]'
```

Input is a NumPy NPZ containing only these required arrays:

| Array | Contract |
|---|---|
| `actual_xy_mm` | Finite `[N,2]` coordinates, preferably float64; concatenate rings |
| `ring_offsets` | Integer `[R+1]`, starts at 0, ends at N; each ring has at least 3 points |

Each ring closes implicitly from its last vertex to its first: do not repeat the
first point or include duplicate consecutive points. Supply one complete part:
one simple outer ring and disjoint, non-nested through holes. Either winding and
any ring ordering are accepted. Self-intersection, touching rings, fragmented
contours and multiple separate parts are rejected. Use a Cartesian, y-up frame.
For pixel coordinates use the same array name and `--units px`; outputs then remain
in pixels. Calibration and distortion correction belong upstream of measurement.
Extra synthetic labels or nominal arrays are ignored during prediction.

The v2 checkpoint selects per-ring resolution from geometry, with at least 64
intervals per opening and 768 on the outer ring, increasing with observed turning.
Its `adaptive-local-v2` protocol concentrates samples around corners to retain short
lands beside long fins. The limit is 65,536 intervals per part; larger inputs fail
explicitly. Omit `--tokens` for adaptive checkpoints. Class probabilities transfer
back by physical overlap length, while point-like boundary evidence uses a maximum.
An edge index denotes the segment from vertex `i` to the next vertex in its ring.

The prior v1 checkpoint must retain its fixed **1,024-interval** contract. Its
curvature neighborhoods and attention windows were trained at that scale. The CLI
rejects incompatible explicit budgets. Both releases accept arbitrary original point
counts within their topology and sampling limits, but finite resampling is not
perfectly invariant to point-density changes; see the format-probe results.

## Output and integration

`prediction.json` contains:

- `features`: semantic class, raw edge indices, ordered circular contour spans,
  confidence, accepted/abstain status, measurement route and units.
- `boundary_search_regions`: original edges near each predicted span endpoint;
  use these to refine boundaries with your classical intersection/change-point logic.
- `relations`: contour adjacency between selected leaf features. Overlapping polygon
  and rounded-profile ancestor selections have parent links.
- `part`: ring topology and observed bounding box; `coordinate_hash` binds selections
  to the exact float64 coordinates and int64 connectivity.
- Optional `measurements`, when `--measure` is supplied.

`prediction.probabilities.npz` contains primary `[N,17]`, independent semantic
`[N,16]`, defect `[N,3]`, boundary `[N]` probabilities and hard primary classes.
Semantic ordering is in `data.NAMES` and the exported `model.json`. Class 0 is
unlabeled; multilabel channel 0 corresponds to semantic ID 1. Defect channels are
normal, burr and nick. Probabilities and the default acceptance threshold 0.7
are **not calibrated**. Abstention is not a reliable detector of unseen part types.

Closed-loop hole/slot consensus preserves entire observed loops. Same-class surface
boundaries require learned boundary evidence and observed turns. Query masks can
associate disconnected thread/tooth runs. Other compound interpretations remain
observed spans; opposite groove sides are separate observed features. This does
not infer datum systems, design intent, thread standards or a 3D part identity.

Built-in diagnostic fitters support circles/arcs, TLS lines, stadium slots and
visible thread pitch. Other routes return `unsupported` until a module is registered:

```python
from contour_model.measurement import MeasurementRouter, selected_points

router = MeasurementRouter()

def measure_groove(contours, feature):
    points = selected_points(contours, feature)
    # Implement your validated geometry measurement; use feature['contour_spans']
    # for ordering and feature['boundary_search_regions'] for endpoint refinement.
    return {'operator': 'custom_groove', 'point_count': len(points)}

router.register('groove', measure_groove)
result = router.measure(raw_contours, predicted_feature)
```

Fitters receive all selected raw observations, including flagged defects. They must
choose any robust rejection themselves. Line `projected_length` is the extent of
the selected support, not a precision virtual-sharp length. Circle radial residuals
are not standardized minimum-zone roundness. No fitter issues tolerance pass/fail.

## Expanded data and training

The new 50,000-part corpus covers 47 families, up to 576 openings, and up to 96
axial stations. Its ordinary physical imperfections are milder, with approximately
3% severe-form training cases. See [DATASET_V2.md](DATASET_V2.md) for files, frozen
source snapshots, cache contracts and exact generation commands.

```bash
python -m contour_model.corpus_v2 /path/new-corpus --count 50000 --workers 6
python -m contour_model.recache /path/new-corpus /path/local-cache --workers 4
python -m contour_model train /path/local-cache /path/new-run \
  --adaptive --width 64 --queries 64 --repeat-queries --instances --relative-noise \
  --steps 60000 --device cuda --val-limit 500 --eval-every 3000
python -m contour_model evaluate /path/new-run/best.pt /path/local-cache \
  --split test --device cuda --output /path/test-report.json
```

Raw observations and the first adaptive cache are at
`/media/T7Shield/inspection-contour-training/v2-expanded`; the corner-focused cache
is at `/media/T7Shield/inspection-contour-training/v2-local-cache`. The separate
`observable-feature-groups.json` reference layer merges nominal collinear flat
subdivisions into physical measurement surfaces without altering observations,
semantic classes or frozen training caches. Current generators reuse one feature
ID across each continuous comb/fin land.

The new test semantic F1 is 99.72%, while the original 121-test-vector regression
is 91.51%. Retain the v1 checkpoint for its previously validated distribution;
the v2 release is not a universal replacement. Neither model has a real-capture
validation set in this workspace.

## Prior corpus and training compatibility

The original 1,200 rendered examples remain at
`/media/T7Shield/inspection-synthetic-shapes`. A separate 50,000-part contour-only
training corpus is at `/media/T7Shield/inspection-contour-training/v1`, with compact
cached targets and retained original contours. It includes the original 12 families
plus mixed shafts, mixed plates and perforated disks, with form errors and defects.
There are 40,080 training, 3,892 validation, 4,002 test and 2,026 distribution-shift
examples. The label adapter is versioned `observed-compounds-v1`; it adds consistent
external ancestry and groups adjacent tab/notch surfaces without modifying originals.

```bash
python -m contour_model corpus /path/new-corpus --count 50000 --tokens 1024 --workers 4
python -m contour_model prepare /path/rendered-dataset /path/cache --tokens 1024
python -m contour_model train /path/new-corpus /path/new-run \
  --steps 100000 --width 64 --instances --device cuda --val-limit 500
python -m contour_model evaluate /path/new-run/best.pt /path/new-corpus \
  --split test --device cuda --output /path/test-report.json
python -m contour_model measure-evaluate /path/new-run/best.pt /path/new-corpus \
  /path/handoff-report.json --split test --limit 300
python -m contour_model export /path/new-run/best.pt /path/model.pt --tokens 1024
```

The released run used staged initialization (32-part overfit, semantic training,
query-only warmup, joint training), then the full corpus. Starting from scratch with
the example command is supported but is not an exact reproduction of that history.
Training samples with replacement; a step is an optimizer update, not an epoch.
`--init checkpoint` starts a new schedule from weights. `--resume last.pt` restores
optimizer/RNG state and requires the same dataset, schedule and training settings.
Use a fresh output directory for a new experiment. `best.pt` uses validation F1 for semantic-only runs. Current repeat-only runs use
90% semantic F1 plus 10% repeat-instance F1; all-instance legacy runs retain the
older equal-weight score. Final v2 selection also reviewed raw feature selections.

Local attention windows follow ring connectivity; a pooled global transformer
exchanges context across rings. A query decoder supplies compound associations.
The `conv` control replaces local attention while retaining global attention.
Model input never includes nominal geometry, family names, feature IDs or labels.
The 64-wide released model has approximately 780k parameters and needs no custom
CUDA extensions. Corpus generation, image extraction and classical fitting use CPU.

`bridge` extracts contours from rendered images and rejects ambiguous transfers
against their paired vectors. This is a synthetic optical-domain check, not real
capture adaptation. Real images can be processed by your own contour extractor and
passed through the same input contract.

```bash
python -m contour_model bridge /path/rendered-dataset /path/bridge-cache --limit 300
python -m contour_model review /path/model.pt /path/new-corpus /path/review.png --limit 15
python -m pytest packages/contour-model/tests -q
python packages/contour-model/scripts/robustness.py /path/model.pt /path/new-corpus /path/probe.json
```

Before production use, collect representative real contours and review semantic
identity, instance boundaries and measurement results. Fine-tune and calibrate on
separate real designs, retaining a locked real test set. Expand the grammar and
measurement registry for unsupported features and ambiguous contour interpretations.

## Exhaustive contact sheets

The complete 50,000-part reference review is in
`/media/T7Shield/inspection-contour-training/v2-contact-sheets-reference`.
The original v1 review remains in `contact-sheets-reference` beside it.
Open `index.html` for previous/next navigation, sample-ID lookup, full-resolution
PNG links and optional browser-stored review notes. Export notes as CSV to keep a
portable backup. `review.csv` lists every sample, its sheet/tile, family, split and
source contour path. `manifest.json` records exact coverage and image hashes.

There are 2,084 sheets at 3,200 × 2,610 pixels, with 23–24 parts per sheet. Each part
is scaled independently and rendered from its original observed vertices, colored
by primary reference semantics. These are dataset labels, not model predictions.
Ancestor labels overlap leaf classes and are not painted over the primary colors.

To regenerate or resume with the same source and renderer:

```bash
python -m contour_model.contact_sheets \
  /media/T7Shield/inspection-contour-training/v1 \
  /media/T7Shield/inspection-contour-training/contact-sheets-reference --workers 4
```

Rendering is CPU-only. `--limit 48` in a fresh output directory makes a two-sheet
preview. Existing sheets are resumed only when their configuration, sample list
and PNG hash match; use a new directory after changing the source or renderer.
