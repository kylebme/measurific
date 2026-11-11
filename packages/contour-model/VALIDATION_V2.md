# Expanded contour-model validation

The selected checkpoint is `artifacts/release-v2/model.pt`, from the joint
corner-focused run at step 27,000. It was selected on validation; held-out test and
challenge results did not select the weights. The release uses `adaptive-local-v2`,
64 learned repeat queries, and the default acceptance threshold 0.7. All 25 tests
pass, and input-only CLI examples verify that selected coordinates remain unchanged.

## Data and evaluation protocol

The expanded corpus has 50,000 independently seeded designs in 47 part families:
40,011 train, 3,979 validation, 3,923 test and 2,087 challenge. See
[DATASET_V2.md](DATASET_V2.md) and [the taxonomy](../../research/synthetic-shapes/expanded-part-taxonomy.md).
There are 6,615 parts with at least 100 openings, 1,206 with at least 300, and
examples reaching 576 openings and 96 axial stations. Maximum retained reference
feature count is 584; maximum outer-contour count is 388.

Ordinary physical errors are substantially milder than v1. Approximately 3% of
training parts retain severe-form stress cases. The challenge partition contains
1,034 mild-form cases with extreme parameter settings and 1,053 severe-form cases.
Parameters overlap the training range for some grammars; this partition does not
test unseen part families or real captures.

Model inputs are observed coordinates and ring connectivity only. Neither nominal
geometry, semantic labels, part-family names nor primitive IDs enter inference.
Geometry determines adaptive sample counts and locations. All final feature selectors
refer back to the original raw edges, preserving actual form errors and defects.

Cached primary semantic F1 is macro-averaged over classes with reference support,
using physical interval lengths for the confusion matrix. Family scores similarly
use classes present in that family. Different sampling protocols produce different
interval targets, so raw-contour feature selection is the decisive cross-protocol
measurement-handoff comparison. Multilabel and boundary results are retained in the
JSON reports; the boundary tolerance is one model interval and is not a fixed
physical tolerance across protocols.

Raw leaf-instance matching uses one-to-one Hungarian assignment, identical semantic
class and physical arc-length IoU >= 0.5. Precision counts accepted predictions;
recall counts all observable reference leaf features. Confidence threshold is 0.7.
Overlapping semantic ancestor selections are excluded. This checks contour support
selection, not precision dimensions, datum systems or tolerance compliance.

The fixed feature-review subsets contain 288 validation, 284 test and 265 challenge
parts. For each family and split, they include the first five IDs and two highest
realized instance counts, with duplicate selections removed. IDs were saved before
held-out test inference. They deliberately emphasize difficult dense parts and
are not an unbiased estimate of the corpus's natural part-frequency distribution.
Full cached semantic evaluation uses every part in each respective split.

Measurement references apply `observable-flat-surfaces-v2`: adjacent nominally
collinear flat subdivisions become one surface. A scan of all 50,000 parts found
4,254 affected designs, principally comb/fin cell seams and trailing thread-zone
lands. The alias layer is supplied as `observable-feature-groups.json`; frozen raw
coordinates, semantic classes and training caches are unchanged. Nominal geometry
is used only to define these references, never to predict or fit measurements.
Current comb/fin generators reuse a shared feature ID across each continuous land.

This correction followed a validation-only audit: perfect semantic predictions
were being penalized for merging unobservable annotation seams. It also exposed
false corner proposals when a short land's neighborhood reached into adjacent deep
notches. Inference now caps that neighborhood to 20% of the observed semantic run's
length. The correction was established on validation geometry; held-out raw reports
were then rerun with the frozen correction. Earlier reports remain archived.

## Model selection and held-out results

| Completed candidate | Step | Full validation semantic F1 |
|---|---:|---:|
| Uniform adaptive, semantic only | 60,000 | 99.540% |
| Uniform adaptive, joint repeat association | 57,000 | 99.415% |
| Uniform adaptive, joint with relative coordinate noise | 60,000 | 99.573% |
| Corner-focused, semantic only | 18,000 | 99.699% |
| **Corner-focused, joint — selected** | **27,000** | **99.763%** |
| Corner-focused, frozen semantic backbone / trained grouping head | 9,000 | 99.699% |

Best and final checkpoints were also compared where they differed. The selected
joint checkpoint combined the highest full-validation semantic F1 with the best
feature-selection precision among the local candidates. Historical raw handoff
reports predate the physical-surface reference and corner-window corrections;
use the `final-*` reports for the released inference implementation.

| Held-out population | Parts | Semantic macro F1 |
|---|---:|---:|
| New test | 3,923 | 99.718% |
| Challenge, all | 2,087 | 97.590% |
| Challenge, extreme parameters / mild form | 1,034 | 99.781% |
| Challenge, severe form | 1,053 | 95.471% |
| Original rendered-corpus test vectors | 121 | 91.509% |

New-test semantic F1 is 99.86% for parts with at least 200 stored reference
instances, and 99.91% for parts with at least 300 openings. The weakest new-test
family is stepped-taper arbors at 96.94%; multi-thread shafts score 98.79%.

**Legacy regression:** the original 121-vector population is substantially weaker,
including fillet F1 of 61.19%. Its raw feature precision is 83.66%, with 96.60%
recall. The old v1 checkpoint and its validation remain available. The changed
generator/deformation population and representation are not a universal replacement
for the previous validated domain; real-capture validation is still absent.

| Raw measurement-surface review, IoU >= 0.5 | Parts | Precision | Recall |
|---|---:|---:|---:|
| Validation | 288 | 99.854% | 99.407% |
| New test | 284 | 99.776% | 99.442% |
| Challenge, all | 265 | 97.173% | 98.996% |
| Challenge, mild form | 135 | 99.793% | 99.476% |
| Challenge, severe form | 130 | 93.451% | 98.278% |

The new-test review contains 25,975 reference measurement surfaces. At the stricter
IoU >= 0.9 threshold, precision is 99.637% and recall is 99.303%. Performance is not
uniform across feature types: thread-group precision/recall are **79.07% / 73.91%**
(34 matches, 43 accepted predictions, 46 reference groups). Both sides can have
correct thread semantics while remaining separate or incorrectly associated.
Taper precision/recall are 95.20% / 93.63%; flat surfaces are 99.63% / 98.78%.
Use the explicit grouping kind and contour spans when integrating compound
measurement modules; do not infer perfect disconnected grouping from semantic F1.

The two uniform controls started from the same 1,000-part semantic pilot checkpoint
(step 5,000). The corner-focused pilot started from an immutable step-30,000 snapshot
of the uniform joint run, then trained on 3,000 indexed parts. Full corner-focused
joint and semantic-only runs use identical pilot initialization and 30,000-step
schedules. A third control freezes the completed semantic network while training
repeat associations. Full run configurations, histories, hashes and immutable
validation checkpoint copies are retained with the release artifacts.

The initial joint experiments selected intermediate checkpoints with the older
mean of semantic F1, query precision and query recall. This could reward an auxiliary
query gain while semantic classes regressed. Current repeat-only training uses
90% semantic F1 plus 10% repeat-instance F1; final release selection separately
compares full validation and raw feature selection. Raw query scores are explicitly
scoped to thread/tooth repeats, not all openings or surfaces.

## Representation and implementation checks

Both adaptive protocols retain at least 64 intervals per opening and 768 per outer
ring, increasing with observed turning. `adaptive-local-v2` places the same interval
count with 35% uniform arc-length mass and 65% smoothed turning mass. The observed
corpus uses 1,600 median, 14,912 p95 and 37,632 maximum intervals. Inputs exceeding
65,536 intervals fail explicitly. Topology supplies closed-opening connectivity;
learned queries do not impose a cap on hole or ordinary surface counts.

Class probabilities transfer back using physical overlap length. Boundary predictions
are point events: maximum overlapping evidence preserves corners that dense local
sampling would otherwise average away. The same semantic class can therefore still
split into separate flat measurement surfaces at observed, predicted boundaries.

A sampling-only audit on 50 validation parts from comb, heat-sink, connector,
stepped-taper and multi-step families found 99.73–100% reference feature-class support
retention for corner-focused sampling, compared with 96.84–99.25% for uniform sampling.
This is an oracle target projection check, not model accuracy or accepted-feature recall.

GPU smoke tests completed forward/backward/optimizer steps on 576 circular openings,
576 hexagonal openings and a 96-station shaft, with tensor-allocation peaks below
500 MB for the 64-wide model. Concurrent training uses the RTX 3070; generation,
contact sheets and classical fitting remain CPU-only.

The complete new reference contact set contains 2,084 PNG sheets: 2,068 with 24 parts
and 16 with 23. All 50,000 IDs occur exactly once. CSV mappings, family/split counts,
PNG dimensions/integrity and every image SHA256 were verified. Representative final
sheets and reference/predicted contour images were inspected visually.

A final 47-test-part format probe scores 99.844% baseline, 99.860% rotated,
99.873% with cyclic starts, and 99.850% with doubled point density. Reversal,
ring reordering, and scale/translation preserve baseline semantic results.
These are semantic probes, not a guarantee of invariant accepted instance grouping.
Twelve dense held-out examples have reference/predicted semantic and instance views,
with detailed thread-zone and shaft-station enlargements.

## Measurement interpretation and limits

Diagnostic modules fit circles/arcs, lines, stadium slots and visible thread pitch.
The short-zone thread fitter estimates axis direction from a continuous side;
combining both sides could otherwise rotate PCA toward the diameter. Other semantic
routes remain available for user-written modules and return `unsupported` by default.
Very short raw spans are retained, but a fitter may report insufficient support.

Selection-induced measurement differences compare predicted and reference selectors
on the **same imperfect raw observations**, conditional on successful instance matching
and supported reference fits. Zero circle or slot difference means the identical
observed closed loop was selected. It is not zero manufacturing error or proof of
metrology accuracy. Projected line length is support extent rather than virtual-sharp
length; endpoint search regions are supplied for classical boundary refinement.

On the new-test feature review, 18,552 of 18,556 matched, reference-measurable
selections fit successfully. Diameter/slot-width/slot-length differences have zero
p95 because complete observed loops are retained. Projected line-support length
also has zero p95, but its maximum difference is **12.486 mm** on sample `000000708`;
this illustrates why endpoint refinement and fit-quality checks remain necessary.
Thread pitch has 33 successful paired comparisons, median selection difference
0.00111 mm, p95 **0.14352 mm**, and maximum **0.28422 mm**. These figures are
conditional selector comparisons, not manufacturing error or calibrated accuracy.

Confidence, defect flags and the acceptance threshold are not calibrated. No real
capture test, optical-domain adaptation of these new weights, standardized GD&T
validation or unseen-family guarantee is claimed. The model does not infer datums,
tolerances, thread standards or arbitrary pattern relationships.
