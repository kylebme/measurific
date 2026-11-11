# Validation record

Validated 2026-09-07 in the isolated `packages/measurement/.venv` environment, Python 3.13.13.

* **28 tests passed.** Analytic tests cover line/circle/arc/ellipse/stadium/rounded
  rectangle fits, reversed elliptical arcs, minimum-strip straightness, finite
  segment intersections, signed profile errors, dimensions, datum relationships,
  groove/width/step/taper compounds, linear/polar/grid/arbitrary patterns, linear
  and angular repeats, curvature and directed wall-distance distributions.
* Failure cases include malformed topology, stale model hashes, abstention/user
  overrides, disconnected arc selections, unknown/unavailable geometry, cyclic
  compound membership, vector tolerance rejection, missing PAE support and clipping.
* The image integration test verifies that **every ordered contour vertex equals
  its original PAE sample**, plus ring/hole topology and deterministic extraction.
* The package built and installed successfully. Installed CLI runs exercised
  saved-contour inputs, image extraction and actual model prediction JSON.
* The existing `packages/contour-model/artifacts/release-v2/model.pt` was run on CPU against
  the demo's extracted pixel contours. Its two selected features were consumed
  directly; all four default measurements completed. The checkpoint was unchanged.

## Synthetic image example

`examples/synthetic_demo.py` uses the `synthetic-shapes` washer generator and renderer,
not `synthetic-esf`. It saves image, contours, request, report, annotated output,
extraction provenance and analytic projected diameter references under
`outputs/washer/` (generated artifacts are git-ignored).

The five requested measurements completed, with no tolerance failures:

| Quantity | Result (px) |
| --- | ---: |
| Outer least-squares diameter | 340.187120 |
| Bore least-squares diameter | 148.843904 |
| Bore/outer center offset | 0.002421 |
| Minimum observed wall separation | 95.549703 |
| Outer radial range about LS center | 0.195745 |

The nominal outer diameter was 340 px. The synthetic-shapes renderer introduces
polygon rasterization bias, so this difference is not a PAE-only error estimate.
These results establish implementation and integration behavior, not calibrated
metrology accuracy or a measurement uncertainty budget. Test and demo seeds are
fixed; exact last digits may depend on dependency versions.

## Dataset-v2: 300 held-out parts

The expanded contour corpus was evaluated using a seeded, family-balanced selection
of **300 test parts across all 47 families** (seed 20260907). Saved actual contours
were projected into 1024×768 image coordinates and passed to the unchanged release-v2
checkpoint and this module. Synthetic-shapes rendered review images from the same
geometry. This run exercised contour inputs, not PAE image extraction.

The first run completed 297 parts and exposed a straightness degeneracy on exactly
horizontal/vertical lines at nonzero image offsets. Rounded mean-centering could
pass a numerical rank check before Qhull rejected the flat input. Straightness now
uses a translated GEOS convex hull with explicit line/point degeneracy handling;
no artificial point jitter is introduced. Three regression tests were added,
including preservation of a nonzero 1e-9 px form width. **All 31 tests pass.**

The corrected run recomputed all measurements/reference fits, reusing the original
297 saved predictions and recomputing the three previously interrupted predictions:

| Result | Count |
| --- | ---: |
| Completed samples | 300 / 300 |
| Fitted feature records, including overlapping ancestors | 20,587 |
| Preserved model abstentions | 89 |
| Computed measurements | 62,246 |
| Unavailable measurements due to abstentions | 356 |
| Sample crashes / feature fit failures / measurement failures | 0 / 0 / 0 |
| Fitted reference feature records | 20,613 |

The batch used per-type defaults and selected line/center relationships (20
operators), not every supported operator or standard-specific dimensions. No
nominal tolerances were supplied, so these counts are numeric completion, not
part acceptance. Measurements alone took median 47.8 ms/part and p95 619.5 ms/part
in the contour-model CPU environment; rendering, inference and reference fitting
are excluded from those timings.

Comparison with the 297 previously completed samples covered 61,128 measurements.
Only straightness values changed, by at most 1.83e-13 px; all other results were
identical. The three previous crashes were resolved. Both runs are retained.

Same-semantic, arc-length-weighted one-to-one reference matching found 20,470
matches among 20,507 accepted model leaves and 20,613 reference leaves at IoU >=0.5
(99.82% precision, 99.31% recall). These are **selection** metrics, not dimensional
accuracy. Most matched selections were identical and therefore gave identical fits.
There were meaningful outliers: one heat-sink selection included an adjacent top
edge, giving 16.0503 px straightness versus 0.0367423 px on reference support.
A grouped thread contour differed by 571.944 px of observed polyline length;
this operator is not thread pitch or axial threaded length.

Artifacts (generated and git-ignored):

* Corrected batch report (`outputs/dataset-v2-test-300-fixed/REPORT.md`)
* All 300 annotated previews (`outputs/dataset-v2-test-300-fixed/index.html`)
* Selection diagnostics and paired outliers (`outputs/dataset-v2-test-300-fixed/DIAGNOSTICS.md`)
* Fix comparison (`outputs/dataset-v2-test-300-fixed/fix-regression.json`)
* Original batch report (`outputs/dataset-v2-test-300/REPORT.md`)

The batch scripts and reproduction commands are documented in README.md.

## Independent construction-truth evaluation

The selector-comparison batch above is not an accuracy oracle. A separate benchmark
now generates **300 deterministic analytic pixel parts** (25 × 12 families) and
compares both selection lanes directly with independently prescribed measurements.
The generator imports no measurement code or fitting libraries. All truth values,
formulas, construction parameters, seeds, error bounds and source hashes were saved
before inference/evaluation. All 300 saved truth and contour hashes were verified
unchanged after the run.

The metadata audit of the old 300-part corpus subset found 22,212 feature records,
9,261 with empty nominal parameter dictionaries, and no full primitive tables.
Although seeds/nominal geometry are available, nominal sizes are not post-warp
associated-measurement truth. New fixtures therefore provide explicit realized
measurands instead of assuming nominal equals actual.

| Selection lane | Requested checks | Within bound | Outside bound | Unavailable | Rejected measurement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference | 8,279 | 8,279 | 0 | 0 | 0 |
| Model-predicted | 8,279 | 7,292 | 165 | 818 | 4 |

A check is an expected measurement field; one operator call can have multiple
checks. The suite exercised 34 operators. Predicted assignments matched 1,518 of
1,684 requested feature targets using accepted, same-semantic, one-to-one support
IoU >=0.5; missing targets remained in the denominator. The geometry types come
from the fixed inspection plan; model output supplies edge selections. The model
was not given reference labels or dimensions. These are new analytic fixtures,
not the earlier held-out dataset-v2 distribution.

Reference error maxima by truth category:

| Reference check category | Checks | Largest absolute error |
| --- | ---: | ---: |
| Exact, px-valued | 6,395 | 1.5545e-8 px |
| Exact, degree-valued | 1,275 | 1.2075e-6 deg |
| Exact, polygon area | 484 | 2.0373e-10 px² |
| Exact, dimensionless slope | 25 | 6.6614e-16 |
| Exact integer extrema count | 50 | 0 |
| Explicitly bounded discretization | 50 | 0.21164 px; all within their individual analytical/grid bounds |

The default exact acceptance bound is 1e-5 in the stated unit; count checks require
exact agreement. Bounded checks cover polygon-circle clearance versus smooth
circle gap, and resampled linear-profile pitch. These are numerical benchmark
bounds, not production tolerances or physical measurement uncertainty.

This includes nonzero, independently known form error. For example:

* Bowed edge `bowed_plate-20350908`: prescribed straightness 1.1955162782218427 px;
  measured 1.1955162782218587 px. Prescribed TLS RMS 0.37818873873229925 px;
  measured 0.37818873873230624 px. The model had no accepted matching selection.
* Lobed bore `lobed_bore-20360908`: prescribed radial variation 0.536184506615998 px;
  both lanes measured 0.5361845067528321 px. Prescribed diameter 117.90904093067415 px;
  both measured 117.90904093382545 px.

The four rejected predicted measurements were step evaluations whose selected
lands were not parallel within the required limit. Other predicted errors included
incomplete or incorrect line/arc support and repeated-profile selections. All
reference checks passed, so this run found no additional measurement implementation
failure on the defined fixture domain. It does not establish correctness on every
possible contour or noisy image.

Reference evaluation latency was median 7.48 ms/part, p95 18.42 ms/part on the CPU
environment used by contour-model. The full run took about 108 seconds including
inference, rendering, scoring and artifact writing. PAE/image extraction and camera
calibration were excluded. Synthetic-shapes rendered review images only.

**All 44 tests pass**, including 12 construction-truth fixture regressions and a
check that fixture generation stays independent of fitting libraries.

See [benchmark design and formulas](benchmarks/README.md),
full report (`outputs/known-truth-v1/REPORT.md`),
300 truth/reference/model previews (`outputs/known-truth-v1/index.html`), and
per-operator results (`outputs/known-truth-v1/summary.json`).
