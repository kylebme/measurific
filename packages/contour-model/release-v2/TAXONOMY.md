# Primitive and compound recognition

The active learning taxonomy is `primitives-compounds-v1`.

| Base ID | Class | Instance policy |
|---|---|---|
| 1 | Line | One continuous straight construction surface; collinear subdivisions merge |
| 2 | Arc | A partial circular span |
| 3 | Circle | A complete circular ring, on either the exterior or an opening |
| 4 | Arbitrary Profile | A noncircular curved span, including harmonic cams and involute flanks |

ID 0 is unlabeled. The classes describe nominal construction identity on imperfect
observations, not a claim that an observed curve is mathematically exact.

| Compound ID | Class | Constituents |
|---|---|---|
| 1 | Full round slot | Two lines and two semicircular ends |
| 2 | Triangular thread | Repeated triangular flank geometry, including small crest/root truncations |
| 3 | Trapezoidal thread | Trapezoidal and square thread templates |
| 4 | Tooth Profile | Lines, arcs, and/or arbitrary flank profiles |

Compound IDs occupy a separate namespace. ID 0 means no compound. Each detected
compound gets its most likely class and an uncalibrated confidence score; compound
classification does not abstain. Detection still allows no compound. Base features
retain the configurable acceptance threshold.

Every compound references complete base instances. A nominal thread endpoint
cannot cut a continuous straight root land into different learning instances.
One thread zone can include its two disconnected observed sides. Each opening
has its own slot instance without a query-count limit. Tooth groups on different
rings remain separate. The classifier sees only observed coordinates and ring
connectivity; family names, nominal geometry and construction labels are excluded.

## Generator

Keyhole panels are removed. Thread generation uses only `v`, `trapezoid`, and
`square`. Asymmetric sawtooth racks remain tooth profiles. Angular tooth geometry
includes explicit circular crowns, straight flanks, and exact involute flanks.
Harmonic cams remain arbitrary profiles. Polygonized line/circle outlines are
decomposed before dense sampling, preserving constituent geometry.

The original 16 manufacturing names remain as construction provenance in saved
metadata. They are not targets of the new model. Legacy rendered-image masks still
use that older schema; use the following contour generator for current training:

```bash
packages/contour-model/.venv/bin/python -m contour_model.corpus_primitives \
  /media/T7Shield/inspection-contour-training/new-primitive-corpus \
  --count 12000 --workers 6
```

The delivered observations are in
`/media/T7Shield/inspection-contour-training/primitives-v2`.
The final audited whole-instance targets and identical observations are in
`/media/T7Shield/inspection-contour-training/primitives-corrected-v1`.
These are 12,000 designs spanning 46 families. Splits are deterministic by design;
ordinary and stress strata remain separate in the metadata. Test/ood examples
are excluded from training and checkpoint selection.

An audit corrected 260 labyrinth plates (2,026 opening rings): an earlier
three-point decomposition confused long straights plus short rounded joins with
large-radius arcs. Repaired labels preserve every observed coordinate. The earlier
release candidate was withdrawn before publication. A separate 1,000-design cohort
(seed 89237) at `primitives-independent-v1` was excluded from weight training and
checkpoint selection, including records carrying a generator-origin train/val
split. Its evaluation-only index is
`artifacts/primitives-independent-cache/index.json`. Isolated thread fragments in
this cohort informed the final repeat-support detection guard, which was also
checked against 300 validation designs; it is therefore not an untouched test of
that final guard.

Source records add `edge_base_class`, `edge_base_instance`,
`edge_compound_class`, and `edge_compound_instance`. Metadata supplies
`compound_instances[].base_instance_ids`. The raw nominal/actual correspondence
and analytic primitive definitions are retained for audits. Inputs are prepared
with the label-blind `adaptive-turn-v1` sampler. This version is recorded in both
the dataset and checkpoint and is used consistently for training and inference.

## Train and evaluate

```bash
packages/contour-model/.venv/bin/python -m contour_model.train_primitives \
  /media/T7Shield/inspection-contour-training/primitives-corrected-v1 \
  packages/contour-model/artifacts/new-primitive-experiment \
  --width 64 --steps 16000 --device cuda --val-limit 400 --eval-every 2000

packages/contour-model/.venv/bin/python -m contour_model evaluate \
  packages/contour-model/artifacts/new-primitive-experiment/best.pt \
  /media/T7Shield/inspection-contour-training/primitives-corrected-v1 \
  --split test --device cuda --output packages/contour-model/artifacts/new-test.json

packages/contour-model/.venv/bin/python -m contour_model.validate_primitives \
  packages/contour-model/artifacts/new-primitive-experiment/best.pt \
  /media/T7Shield/inspection-contour-training/primitives-corrected-v1 \
  packages/contour-model/artifacts/new-raw-test.json --split test --limit 200 --device cpu
```

The 64-wide transformer has 780,567 parameters. Its secondary compound head takes
learned base probabilities and contour context. A query decoder learns repeated
span associations. Inference retains whole base supports, constrains thread-side
pairing to overlapping longitudinal intervals, and prevents cross-ring tooth
grouping. All operations run on CPU without custom CUDA extensions.
Thread candidates require at least four line/arc constituent spans on a side,
the minimum support for two pairs of repeated flanks. Isolated corners do not
become thread instances solely from a high classification score.

`--init` starts a new experiment schedule; it does not resume optimizer/RNG state.
Use a new output directory for each experiment. Selection combines validation
base macro F1 (70%), compound macro F1 (20%), and repeat-query instance F1 (10%).
Review raw-edge instance precision/recall as well: length-weighted classification
can conceal missed short flanks or merged feature instances.

## Output

Prediction schema 2 includes `features` for base instances, `compound_features`
for compounds, and adjacency/membership `relations`. Compound members refer to
base IDs; compound raw support is exactly their union. IDs and names are exported
with the checkpoint. The probability NPZ contains base `primary` `[N,5]`,
base `semantic` `[N,4]`, secondary `compound` `[N,5]`, defect `[N,3]`, and boundary
`[N]` arrays. A selection always indexes the original observed contour edges.

The measurement adapter accepts this schema and fits base and compound support.
The GUI palette includes the new names. Earlier checkpoints remain readable via
their legacy label schema; their 16-class probabilities must not be interpreted
using the new IDs.

Real-capture validation is still absent. Closely approximated curves, very short
flanks, heavy form error and thread/tooth silhouette ambiguity remain relevant
limits. Confidence values are not calibrated probabilities of correctness.
