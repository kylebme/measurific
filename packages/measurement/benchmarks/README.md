# Independently known measurement truth

This benchmark addresses measurement correctness, separately from agreement between
two feature selectors. Both reference-selected and model-selected measurements are
compared to expected values derived from construction parameters.

## What is preserved

`known_geometry.py` imports only NumPy and Python's standard library. It does not
call the measurement package, fitters, Shapely, or an optimizer. Every case records:

* Generator version, source SHA-256, NumPy PCG64 seed, and all construction parameters.
* Exact contour points, connectivity, and reference edge selections.
* A fixed inspection plan specifying geometry types and measurement operators.
* Expected values, formulas, units, and absolute numerical/discretization bounds.

The runner saves **all fixture truth before loading the model or evaluating any
measurement**. `generator_snapshot.py` preserves the actual generator source;
`manifest.json` hashes the truth files and contour NPZs. Truth is never fitted from
reference-selected points or adjusted to make results pass.

The default corpus contains 300 cases: 25 each of 12 parameterized families, with
seeded dimensions, rotations, positions, sampling densities, and repeat counts.
All coordinates are constructed directly in pixels. There is no calibration.

## Why a new dataset

The existing expanded corpus preserves actual/nominal contours, seeds, nominal
parameters and imperfection metadata. The audit of the earlier 300-part subset
found 22,212 feature records; 9,261 had empty nominal parameter dictionaries and
none of the 300 metadata records included the full primitive tables. More
fundamentally, combined shear, bilinear deformation, harmonic displacement,
primitive displacement, translation, pitch warp and defects alter associated
sizes and form metrics. A preserved nominal radius is not the expected LS radius
after those transformations.

Reconstructing a contour deterministically does not independently establish every
measurand on it. The new data deliberately uses geometry with independently known
measurement values. `existing-corpus-audit.json` records the audit in the first
full run directory. Existing datasets, generators and checkpoints are unchanged.

## Fixture families and truth definitions

| Family | Representative independent truth |
| --- | --- |
| Rectangle with holes | W, H, WH, 2(W+H), corners, center spacings, parallel/perpendicular lines, section widths |
| Slotted plate | Stadium width 2r, length 2(a+r), end radius r, center separation 2a |
| Eccentric washer | Circle radii/centers; concentricity offset from prescribed center vector; wall gap with a polygon error bound |
| Rounded plate | W/H/corner radius; quarter-arc radius, 90° span, pi*r/2 length, sqrt(2)*r chord, r*(1-cos(pi/4)) sagitta |
| Stepped/notched plate | Prescribed land separation, groove depth and opening width |
| Tapered plate | Width H+s*x at specified stations and derivative s |
| Polar hole pattern | Center coordinates, pitch-circle radius, 360/N angular pitch |
| Linear repeated rack | Prescribed period and number of extrema; 2A profile height |
| Angular repeated profile | 360/N angular pitch, N extrema including the seam, 2A radial height |
| Bowed plate | Analytically known nonzero straightness and TLS residual statistics |
| Lobed bore | Analytically known nonzero radial form range and LS residual RMS |
| Elliptical bore | Prescribed center, full axes 2a/2b and major-axis orientation |

This is not an exhaustive test of every implemented operator. It exercises exact
associated geometry, multiple relationships, compounds, patterns/repetitions,
finite-polygon dimensions, and controlled nonzero form error. It does not establish
accuracy on arbitrary noisy contours, profile registration, PAE extraction, camera
images or physical measurements.

### Controlled nonzero form errors

For the bowed line, points are sampled uniformly at odd N locations on
`y = A*(1-(2x/W)^2)`, `x in [-W/2,W/2]`, before rigid transformation. Both endpoints
and the apex are present. Its convex hull contains the endpoint/apex triangle,
whose minimum width is A for the chosen A << W, and all points lie in the strip
`0 <= y <= A`. Therefore minimum-zone straightness is **exactly A**.

Symmetry makes the TLS direction horizontal. With `z=2x/W`, its expected RMS is
`A*sqrt(E[z^4]-E[z^2]^2)`, where the discrete moments are calculated in closed form:

```
E[z^2] = (N+1)/(3*(N-1))
E[z^4] = (N+1)*(3*N^2-7)/(15*(N-1)^3)
```

For the lobed bore, `r(theta)=R+A*cos(k*theta)` with k=3..6 and 64 samples per lobe.
The small-amplitude, balanced k-fold distribution has associated LS center at the
prescribed center and mean radius R. The sampled extrema are included explicitly.
Expected radial range is **2A**, and RMS is **A/sqrt(2)**. This is the implemented
LS-center radial form metric, not a minimum-zone circularity claim.

### Smooth geometry versus finite sampling

The diameter operator fits the smooth circle passing through its samples, so its
truth is 2R. The perimeter operator measures the observed polygon, so its truth is
`2*N*R*sin(pi/N)`, not `2*pi*R`. Polygon area similarly uses
`N*R^2*sin(2*pi/N)/2`. These distinctions avoid labeling expected chord effects as
implementation errors.

The finite-polygon minimum gap of nested circular contours is compared to the
smooth-circle gap `R-r-offset`, with a predefined error bound equal to the sum of
the two chord sagittae. Linear repetition pitch allows at most two predefined
resampling-grid intervals. Every check records whether it is an exact expression
or a bounded-discretization comparison.

Exact checks use a default absolute bound of 1e-5 in their stated units; extrema
counts require exact integer agreement. These are benchmark acceptance bounds,
not manufacturing tolerances, confidence intervals or an uncertainty budget.

## Two evaluation lanes

**Reference lane:** use the fixture's reference edge labels with the fixed
inspection plan. Compare every measured field to its independent expected value.
This isolates correctness of measurement on the specified geometry/support.

**Predicted lane:** run the unchanged release-v2 checkpoint on only the observed
contour coordinates/connectivity. Then match accepted predicted selections to
inspection targets by geometry identity and one-to-one arc-length IoU >=0.5.
For `primitives-compounds-v1`, base and compound namespaces are separate: line,
arc and circle targets match base classes; slots and tooth profiles match compound
classes; ellipse plans may use arbitrary-profile support. Legacy construction IDs
are never compared numerically to the new class IDs. Whole polygons and rounded
rectangles have no single corresponding model class and remain unmatched in this
fixed-plan benchmark.
Replace each target's reference edge indices with the matched predicted indices,
and execute the same fixed inspection plan. Compare directly to the same truth.

No reference points are substituted if prediction/matching fails. Missing and
abstained targets stay unavailable and remain in the total requested-check count.
A compound becomes unavailable when required constituents are missing. A datum
opening built from two lands is composed from those two predicted selections,
not from reference points.

Post-hoc matching is evaluation-only correspondence, not a claimed automatic
production feature-routing solution. Types such as `ellipse` are specified by
the inspection plan; the model supplies semantic selections, not independently
inferred ellipse parameters. Some generated targets are outside the model's
training distribution. Predicted-lane failures therefore need to be interpreted
separately from reference-lane correctness.

A check is one expected output field. One measurement may have several checks,
such as a repetition's pitch, count and profile height. Vector checks use maximum
absolute component error. Unoriented angle checks wrap modulo 180°. Numeric errors
are not averaged together across different units.

## Reproduce

From the repository root:

```bash
PYTHONPATH=packages/measurement/src:packages/contour-model/src:research/synthetic-shapes/src \
  packages/contour-model/.venv/bin/python packages/measurement/benchmarks/evaluate_known_truth.py \
  --per-family 25 --seed 20260908 \
  --checkpoint packages/contour-model/release-v2/model.pt \
  --output packages/measurement/outputs/known-truth-v1
```

Use a fresh output directory. `--reuse /path/prior-run` reuses matching model
predictions and rendered inputs while recomputing measurements against the same
frozen construction formulas. Generator, seed and checkpoint identities must match.

The runner uses synthetic-shapes only to render review images. Measurements and
model inference use the exact stored contours; **PAE is not run**. Rendered raster
values therefore cannot contaminate the contour-level correctness assessment.

Artifacts include `REPORT.md`, `summary.json`, `index.html`, and per-case truth,
requests, predictions, measured reports, scored comparisons, NPZ contours and
annotated previews. Full pass/fail/unavailable counts include all requested checks.
`packages/measurement/tests/test_known_truth.py` locks down the fixture expectations and
checks that the generator remains independent of fitting libraries.

Each case also saves `handoff-report.json.gz`, evaluating the new model document
itself through the same adapter used by the GUI. See
[PRIMITIVES_VALIDATION.md](PRIMITIVES_VALIDATION.md) for the current release run.

## Image extraction and optical experiments

The image-extraction investigation is separate from the contour-only measurement
benchmark above. See [PAE investigation](../../pae/INVESTIGATION.md).

```bash
# New images with continuous-area rasterization and separately stored truth:
packages/measurement/.venv/bin/python packages/measurement/benchmarks/evaluate_extraction_degradation.py \
  --output /tmp/extraction-sweep
# Re-evaluate immutable images; no truth is passed to the extractor:
packages/measurement/.venv/bin/python packages/measurement/benchmarks/evaluate_existing_extraction.py \
  --input app/demo-data/extraction-refined-20260910 --output /tmp/current-pae
# Unlabelled images test validity, not absolute coordinate accuracy:
packages/measurement/.venv/bin/python packages/measurement/benchmarks/evaluate_unlabelled_extraction.py \
  --input app/demo-data/extras --output /tmp/current-extras
```

`--method pae0|pae1|pae10` on the existing-image evaluator selects explicit full
PAE detection. Default uses the current measurement adapter.

`experimental_optical_fit.py` and `experimental_corner_fit.py` retain the offline
optical-model investigation. **They are not production extractors and do not
preserve original PAE sample identities.** The global fit estimates a Gaussian
PSF, image intensity fields, and a polygon. Two-edge corner proposals optionally
fit the original local pixels. Their covariance is conditional on the estimated
PSF and assumed model; it is not a calibrated error guarantee. No truth,
primitive labels, or CAD data participate in fitting. Heuristic initialization,
convergence, nearby edges, form errors, and non-Gaussian PSFs remain limitations.

Use `--method experimental-optical` or `--method experimental-corners` on either
image evaluator to run them explicitly. A full image can take minutes. The
current optical experiment fails on several extras, so it is deliberately absent
from `contours_from_image` and the GUI default. To reuse the expensive saved
optical fits while investigating corners:

```bash
packages/measurement/.venv/bin/python packages/measurement/benchmarks/evaluate_corner_recovery.py \
  --input app/demo-data/extraction-refined-20260910 \
  --optical-cache app/demo-data/extraction-validation-20260910/experimental-optical \
  --output /tmp/experimental-corners
PYTHONPATH=packages/measurement/benchmarks:packages/measurement/src:packages/pae/src \
  packages/measurement/.venv/bin/python -m pytest packages/measurement/benchmarks/test_experimental_optical.py -q
```

The optical tests use independent separable integrals for a blurred
rectangle, finite-difference shape/PSF derivatives, and noisy corner localization.
These establish local model arithmetic, not universal extraction reliability.

The newer local corner experiment and diverse validation (`corner-recovery-20260910/README.md`)
fits directly around PAE corners, estimates local blur, and challenges the sharp
corner assumption with a competing bevel. It avoids the global optical fit and
remains offline. The 72-image benchmark includes 48 held-out images; accepted
corner accuracy and whole-outline failures are reported separately. Use
`evaluate_diverse_corners.py --help` or `experimental-local-corners` on the
unlabelled evaluator. No GUI or production default is changed.
