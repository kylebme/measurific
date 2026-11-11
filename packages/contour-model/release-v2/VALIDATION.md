# Final primitive release validation

The release uses `primitives-corrected-final-b/best.pt`, step 2,000. Weights were
selected using validation results after repairing the labyrinth reference labels.
Check `model.json`, `reports/evaluation-provenance.json`, and `MANIFEST.json` for
checkpoint, source, and dataset hashes. All reported data is synthetic.

## Class predictions

Physical-length-weighted token confusion matrices are converted to per-class F1,
and the four non-background classes are averaged. These scores measure class
assignment; they do not measure whether instances are split or grouped correctly.

| Dataset | Designs | Base macro F1 | Compound macro F1 |
|---|---:|---:|---:|
| Held-out test | 923 | 98.81% | 99.28% |
| Generator stress/OOD | 499 | 97.98% | 99.21% |
| Additional seed 89237 cohort | 1000 | 98.82% | 99.36% |

## Raw observed-contour instances

Class-aware, one-to-one matching uses physical edge-length IoU of at least 0.5.
All predicted base proposals are counted, including those marked abstain at the
default 0.7 base threshold. Compound classification always chooses a class.
These are output-instance scores after whole-base association and repeat-support
detection. They differ from the auxiliary query-decoder metrics in training logs.

| Dataset | Designs | Base instance macro F1 | Compound instance macro F1 |
|---|---:|---:|---:|
| Full held-out test | 923 | 95.06% | 98.96% |
| Validation sample | 300 | 96.35% | 98.71% |
| Additional cohort sample | 250 | 96.87% | 98.30% |

Full held-out test, by class:

| Level / class | Matched | Predicted | Reference | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Base: line | 97,005 | 98,186 | 103,250 | 98.80% | 93.95% | 96.31% |
| Base: arc | 27,341 | 37,008 | 27,988 | 73.88% | 97.69% | 84.13% |
| Base: circle | 23,521 | 23,521 | 23,521 | 100.00% | 100.00% | 100.00% |
| Base: arbitrary_profile | 8,838 | 8,871 | 8,838 | 99.63% | 100.00% | 99.81% |
| Compound: full_round_slot | 4,253 | 4,253 | 4,253 | 100.00% | 100.00% | 100.00% |
| Compound: triangular_thread | 35 | 36 | 36 | 97.22% | 97.22% | 97.22% |
| Compound: trapezoidal_thread | 71 | 72 | 72 | 98.61% | 98.61% | 98.61% |
| Compound: tooth_profile | 167 | 167 | 167 | 100.00% | 100.00% | 100.00% |

Two of 108 reference thread instances in the full test were not matched with
the correct class at IoU 0.5.
The 250-design additional sample has one triangular thread classified as
trapezoidal (11 triangular and 24 trapezoidal reference instances). Thread class
accuracy on small samples should therefore not be read as a precise estimate.

## Label audit and evaluation discipline

The corrected 12,000-design corpus and the separate 1,000-design cohort passed
primitive/compound membership, split isolation, and cache normalization audits.
The repair audit verifies bitwise preservation of observed/nominal coordinates,
ring offsets, anomaly labels, and cached input x/offsets/weights for all 260
affected labyrinth plates (2,026 opening rings). Long straights are now Lines;
only true rounded spans are Arcs. The old candidate was withdrawn before release.

No examples from the additional seed 89237 cohort were used for weight training
or checkpoint selection. Its isolated thread-fragment failures did inform the
final detection guard requiring four line/arc spans in each candidate thread side.
It is therefore not an untouched test of that guard. The guard removed 15 extra
fragment instances on the 250-design sample, preserved the 300-design validation
results, and was then evaluated on all 923 held-out test designs. Both before/after
diagnostic reports remain under artifacts; the release reports use the final code.

## CPU cost and model size

Inference requires no CUDA runtime execution or custom GPU kernels. The model has
780,567 parameters versus 780,215 previously (+352, approximately 0.045%). Its
optimizer-free checkpoint is approximately 3.0 MiB. Training ran twelve experiments
for 76,000 optimizer updates, with two or three simultaneous RTX 3070 jobs.

The CPU comparison uses four PyTorch threads and seven identical input designs
covering 768–34,560 prepared tokens. Each model receives a warmup, five forward
repeats, and a warm full prediction. Full prediction includes sampling and output
instance construction; the much larger number of base selections can add cost.

| Family / tokens | Previous forward (ms) | New forward (ms) | Previous full (ms) | New full (ms) |
|---|---:|---:|---:|---:|
| threaded_shaft / 768 | 6.2 | 6.0 | 24.9 | 32.7 |
| mixed_shaft / 1,504 | 9.6 | 8.8 | 45.4 | 54.6 |
| drilled_plate / 3,008 | 23.0 | 23.9 | 107.3 | 94.0 |
| internal_spline_ring / 5,984 | 40.8 | 42.6 | 102.4 | 200.5 |
| serrated_washer / 11,968 | 86.9 | 89.9 | 202.3 | 438.3 |
| perforated_grid_plate / 23,808 | 168.1 | 166.3 | 1249.8 | 1115.9 |
| staggered_filter_plate / 34,560 | 276.1 | 302.7 | 2243.4 | 2034.2 |

Median of the seven forward medians: 42.6 ms new versus 40.8 ms previous.
The 250-design raw CPU evaluation recorded 63.6 ms median and 935.0 ms p95.
These timings are machine-specific and exclude model loading and JSON serialization.

## Verification and limits

All 56 contour-model and synthetic-generator tests pass. The packaged source was
extracted into a separate directory and used for CPU predictions with measurement
on slot, multi-thread, tooth, and labyrinth examples. Schema and compound membership
unions were checked. The separate pixel measurement adapter fitted all four thread
compounds in the integration example while preserving the original coordinate hash.
Earlier integration checks also passed 70 measurement tests, three GUI tests, and
the GUI production build. The final release smoke reports are included in `reports`.

Arc oversegmentation and short-span confusion are the main observed limitations:
full-test arc instance precision is 73.88%, even though arc recall is 97.69%.
Base class F1 alone conceals these small extra selections. Very short/partial
repeat spans can be suppressed by the minimum-support detection rule. Thread/tooth
silhouettes can be geometrically indistinguishable; the compound chooses its most
likely class without inferring thread standards or unseen design intent.

No real-camera contour validation or confidence calibration was performed.
Synthetic success does not establish real-capture performance or dimensional
measurement accuracy. Base threshold abstention and diagnostic fitter failures
remain explicit output states.
