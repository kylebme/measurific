# Pixel contour measurement

Standalone, CPU-only Python module. The core accepts **contours + a JSON
measurement request** and never runs a model; the image wrapper extracts contours
with the learned sub-pixel edge model ([edge-model](../edge-model/README.md), via
ONNX Runtime) and also produces an annotated image. Distances are in pixels, angles in degrees, areas in px², and
curvature in 1/px. Coordinates are image coordinates: x right, y down, integer
pixel centers. No calibration is applied. Existing project folders are unchanged.

## Install and run

From the repository root:

```bash
python3 -m venv packages/measurement/.venv
packages/measurement/.venv/bin/python -m pip install -e './packages/edge-model[onnx]' -e ./packages/pae -e './packages/measurement[image,test]'
packages/measurement/.venv/bin/python -m inspection_measurement input.png request.json packages/measurement/outputs/run
```

With uv, use `uv venv packages/measurement/.venv` followed by
`uv pip install --python packages/measurement/.venv/bin/python -e './packages/edge-model[onnx]' -e ./packages/pae -e './packages/measurement[image,test]'`.
Install only `./packages/measurement` for precomputed contours; the edge model, PAE, ONNX Runtime
and OpenCV are optional image dependencies. Nothing here imports Torch.

The CLI writes `results.json`, `contours.json`, and `annotated.png`. Use
`--contours contours.json` to reuse exactly the same ordered observations, and
`--polarity light-on-dark`, `--min-object-area`, `--min-hole-area` and
`--point-spacing` as needed.
Exit status is 1 if any measurement is unavailable/failed or a tolerance fails;
otherwise 0. Output files are overwritten on a repeated run.

## Python API

```python
import json
import numpy as np
from PIL import Image
from inspection_measurement import inspect_image, evaluate

image = np.asarray(Image.open('input.png').convert('L'))
request = json.load(open('request.json'))
report, annotated, contours = inspect_image(
    image, request, extraction={'polarity': 'dark-on-light'},
)
annotated.save('annotated.png')
with open('results.json', 'w') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)

# Reuse contours when changing only the measurement request.
report = evaluate(contours, request)
```

`inspect_image(image, request, contours=contours)` skips extraction.
`annotate(image, report, level=1..4)` filters displayed feature reporting levels;
all numeric results stay in JSON. The annotation adds a report panel beside the
image, colors selected contours, marks fitted centers, and draws distance witness
segments. It preserves the source image size and pixel coordinate frame.

## Contour contract

```json
{
  "units": "px", "coordinate_system": "image",
  "xy": [[10,10], [50,10], [50,30], [10,30]],
  "offsets": [0,4], "closed": [true]
}
```

`Contours(xy, offsets, closed=None)` accepts finite float64 `[N,2]` coordinates
and integer path offsets. Closure defaults to true. Rings omit a repeated closing
vertex and must be simple with nonzero area. Open paths need at least two vertices;
the final vertex has no outgoing edge. Consecutive duplicate vertices are rejected.
Different paths and disconnected selections are never implicitly connected.
`contours.to_dict()` and `Contours.from_dict(...)` round-trip this format.

### Image extraction (edge model)

`contours_from_image(image, ...)` returns contours and diagnostics. The image must
be grayscale uint8 or uint16, a dark part on a bright backlight (or
`polarity='light-on-dark'`, which inverts it). The edge model's classical seed
(background fit, Otsu threshold, gradient band) selects 16×16 px cells near the
silhouette. Its fine network then predicts 16× sub-pixel occupancy there, and
marching squares traces that occupancy into ordered closed rings. See
[edge-model](../edge-model/README.md) for the model, its accuracy and its limits.

Every output vertex is a crossing of the predicted boundary with the 1/16 px
lattice. `point_spacing` (default 1.1 px, the density of the synthetic generator
contours the recognizer was trained on) keeps the crossing that starts each interval of arc length. Crossings are
never moved or interpolated, and `point_spacing=0` keeps them all. Where thinning
would make a ring non-simple, that ring's spacing is halved until it is simple.
Coordinates are shifted by −0.5 px to this package's integer pixel centers. Outer
paths have positive shoelace area in image coordinates, holes negative. Rings are
ordered largest first, each starting at its top-left vertex, and extraction is
deterministic.

Parts cut by the image border or by `valid_mask` close along that boundary. Before
inference, excluded pixels and a 16 px apron around the frame are filled with the
continued local backlight. This is a normalized convolution of nearby background
pixels, so the model sees an ordinary backlit edge there and no step between two
background levels. A backlight brighter than 0.95 of full scale, e.g. one clipped
at 255, is scaled down to 0.95, the model's training maximum (`backlight_gain` in
the diagnostics). The gain acts like a shorter exposure. ROI-exterior pixel values never affect the result. The model's
prediction counts only within 2 px of valid pixels. Each ring must agree with the
model's classical seed mask over at least half of its own region. This drops the
model's occasional objects on faint fibers, vignetting ramps or the fill
(`unsupported_rings` in the diagnostics). Regions
below `min_object_area` / `min_hole_area` (defaults 50 / 8 px², e.g. dust specks)
are dropped together with everything inside them. `max_value` gives the sensor
full scale for data stored in a wider type, e.g. 4095 for 12-bit data in uint16.
`model=` accepts a loaded `edgesr.runtime.EdgeModel`; by default
`load_edge_model()` loads and caches the checkout's
`packages/edge-model/models/cfi_B_seed-onnx`.

Several objects (parts, dust, islands in holes) are supported; each ring's
diagnostics give its `depth` and containing `parent` path. Diagnostics report ring nesting, depth, per-ring lattice-crossing counts and final
spacing, dropped rings, the apron, the model's providers and stage timings. A
failure raises `ContourExtractionError` (a `ValueError`) with a compact overlay
diagnostic, for example when there is no silhouette or thinned rings would cross.
Multiple parts and nested islands are supported by the adapter, although the
existing transformer has narrower topology requirements.

The model was trained on synthetic renders and has not been validated on real
images yet. On blurred edges it sits 0.07–0.17 px outward (σ 0.6–1.5 px). These
checks establish extraction behaviour, not calibrated dimensional accuracy.

### Image extraction (coarse model + PAE)

`load_pae_edge_model(path)` returns a `PAEEdgeModel` ([pae_edge.py](src/inspection_measurement/pae_edge.py))
that `contours_from_image(..., model=...)` accepts in place of the edge model. `path` is a
coarse ONNX export or an edge model export whose `model.json` names its `coarse_model`, so
PAE shares the SDF model's coarse seed. The coarse model (512 × 512 frame) decides which rings
exist and roughly where; PAE ([pae](../pae/README.md), noise-compensated `smoothing=1`) then
locates the edge from the original intensities in a band of about 1.5 coarse pixels
(`max(H, W)/512` px) around each coarse ring. Earlier PAE adapters seeded PAE with a
threshold, which merged, split or lost rings under uneven backlight, glare and blur; only that
seed is replaced. The apron/ROI fill, trust margin and minimum areas are those of the edge
model path.

The PAE gradient threshold follows the measured part/backlight contrast and noise. Steps with
less than 35 % of that contrast (reflections, lips, texture) are ignored. Each observation is
assigned to the nearest coarse ring whose background side agrees with its PAE normal, ordered
by arc length, and dropped if its offset from the ring departs more than 1 px from its
neighbours' running median. Up to four observations within 3 px whose links cross, at sharp
corners, are reordered; nothing is moved or interpolated. A coarse ring with fewer than 0.2 PAE
observations per px is not an edge and is dropped with what it encloses
(`unsupported_rings`). A ring with a gap over 5 px or that still self-intersects raises
`ContourExtractionError` located at the problem, with the coarse ring for the overlay.
Diagnostics add `method: 'coarse_model_pae_rings'` and a `pae` block (threshold, contrast,
noise, search radius, observation counts).

On the GUI demo images the result matches the SDF model's topology and error against the
generator contours at about 0.2 s instead of 6–10 s on CPU. On an area-sampled quadrilateral,
straight-edge observations lie within 0.007 px of the truth (median 0.002 px); PAE's local
model does not cover sharp corners or wide blur (see [the PAE investigation](../pae/INVESTIGATION.md)),
where the SDF model is the better choice.

## Request format

```json
{
  "schema_version": 1, "units": "px",
  "features": [
    {"id":"edge", "type":"line", "selection":{"region":{"box":[20,30,180,40]}}},
    {"id":"hole", "type":"circle", "selection":{"path":1}}
  ],
  "measurements": [
    {"id":"hole_to_line", "operator":"distance", "feature_refs":["edge","hole"],
     "nominal_value":20, "tolerance":{"lower":19.8,"upper":20.2}},
    {"id":"hole_size", "operator":"least_squares_diameter", "feature_refs":["hole"]}
  ]
}
```

[examples/manual_request.json](examples/manual_request.json) shows two lines, a
circle and a compound width. Its regions must be adjusted to the actual image;
it is an illustrative request, not an automatic part detector.

Feature and measurement IDs must be unique nonempty strings. Each primitive
requires `type` and exactly one selection form:

| Selection | Meaning |
| --- | --- |
| `{"path":0}` | All observed edges of a path |
| `{"edge_indices":[5,6,7]}` | Global outgoing-edge indices; endpoints included in fitting |
| `{"spans":[{"ring_index":0,"start_edge":90,"edge_count":20}]}` | Connected spans; wrapping allowed on closed paths |
| `{"region":{"box":[xmin,ymin,xmax,ymax]}}` | Both edge endpoints inside box |
| `{"region":{"circle":[cx,cy,r]}}` | Both edge endpoints inside disk |

Disconnected runs remain separate. Lines can fit pooled support; arcs, ellipses,
slots and single-profile operators require connected support. A region does not
automatically distinguish nearby edges: use tight regions or explicit spans.
`max_rms_residual` optionally rejects a fit beyond a caller-defined limit. All
observations are retained; there is no silent outlier removal.

Primitive/profile types: `line`, `circle`, `arc`, `ellipse`, `elliptical_arc`,
`freeform`, `contour`, `polygon`, `slot`, `rounded_rectangle`, `repeated_profile`.
Freeform geometry is an observed polyline without implicit smoothing. Slot fits
use a stadium; rounded rectangles use four equal corner radii. Other rounded
profiles and D profiles use a generic contour, nominal comparison, or constituents.

Compound types: `width`, `angle`, `step`, `taper`, `groove`, `notch`, `pattern`.
Their `members` reference existing feature IDs. Feature-level `params` supply
`compound` measurement defaults; measurement parameters override them. Metadata
`semantic`, `parent`, `confidence`, and `level` is preserved. Semantics never gate
user-defined geometry or determine dimensions.

Absent `measurements`, conservative per-type defaults are generated. An explicit
empty list requests fitting only. Parameter-dependent compound/repetition operators
need explicit parameters. Invalid request structure raises `ValueError`; local
fit/measurement failures become result records, leaving unrelated results usable.

## Transformer handoff and user overrides

Release-v2 and release-v3 prediction JSON is accepted directly when `units="px"`.
The `primitives-compounds-v2` schema keeps `features` (Line, Arc, Circle)
and `compound_features` (Full round slot, Triangular thread,
Trapezoidal thread, Tooth Profile) as individually measurable selections.
`measurement_route`, `raw_edge_indices`, and `contour_spans` are mapped to feature
descriptions. Reports retain `label_level`, `base_instance_ids`, uncalibrated
confidence metadata, and adjacency/membership `relations`. Model compounds must
reference unique existing base instances and select exactly the union of their
supports; invalid compounds fail locally without discarding base measurements.
Compound acceptance is independent of constituent base abstention. Changing a
base selection requires updating affected compound selections as well.

Abstained and failed-fit selections retain valid observed support for GUI review.
They remain unavailable for measurement. Thread/tooth defaults report observed
support length only. Use the `thread` operator below for semantic thread measurements;
generic pitch uses an explicit `repetition` request and parameters,
with a single connected side selected for linear repetition. When both indices and spans exist they must agree. A supplied
`coordinate_hash` must match the original float64 coordinates and int64 offsets.
Stale selections are rejected. Hashing covers coordinates/offsets; native path
closure is carried separately.

Circle selections must originally contain one closed contour. If corner exclusion
splits that contour into open runs, the circle fit uses all retained observations
without closing or filling the gaps. Such fits require at least 180 degrees of
retained angular support about the fitted center, in addition to the usual fit
conditioning checks. `retained_angular_support` reports this coverage in degrees;
the associated circle's `angular_span` remains 360 degrees. Originally open or
disconnected circle selections remain invalid.

Use the same pixel-valued arrays for both modules:

```python
raw = contours.model_arrays()
# In the orchestration process where the existing model is loaded:
prediction, _ = predict(model, raw, units='px')
report = evaluate(contours, prediction)
```

The model's legacy input key is `actual_xy_mm`, but `model_arrays()` places pixel
values in it; `units='px'` during inference is essential. There is no scale change,
y flip or reindexing. Recompute model predictions from these arrays if an earlier
prediction used y-up coordinates. The model handles its own traversal normalization.

Semantic groove/notch/rounded-profile routes map to generic contours: member roles
are not invented from labels. Repeated profiles may span disconnected sides;
select one monotone side and explicitly choose direction/prominence for repetition.
Do not include the old model's optional `--measure` results as this module's request
list. Replace that `measurements` field or omit it to generate new defaults.

Merge user overrides into the prediction document:

```json
{"overrides":[
  {"id":"feature_0001", "source":"user", "accept":true,
   "type":"line", "selection":{"region":{"box":[10,20,110,25]}}}
]}
```

Overrides replace specified fields. Replacing a selection removes old indices and
spans. Both `source="user"` and `accept=true` are required to use an abstained model
feature. Add new user features directly to `features` with unique IDs. Add explicit
measurement requests referring to either source or their combination.

## Measurement definitions

Each request has `operator`, ordered `feature_refs`, and optional `params`.
Results report the precise `method`. The registry is
`inspection_measurement.operators.OPERATORS`.

| Operators | Definition / parameters |
| --- | --- |
| `length`, `orientation` | Fitted line projected span and angle modulo 180°; generic length sums observed segments. Line length uses the original selected observations, so corner exclusion does not shorten it |
| `radius`, `least_squares_diameter` | Orthogonal least-squares circle/arc associated size |
| `slot`, `tooth_profile` | Compound dimension tables; see [slot and tooth profile measurements](#slot-and-tooth-profile-measurements) |
| `arc_length`, `angular_span`, `chord_length`, `sagitta` | Associated circular-arc quantities, ordered span |
| `major_axis`, `minor_axis` | Full ellipse axis lengths from orthogonal fitting; ellipse orientation modulo 180° |
| `width`, `overall_length`, `end_radius`, `end_center_spacing` | Stadium fit |
| `width`, `overall_length`, `corner_radius` | Equal-radius rounded rectangle fit |
| `rms_residual`, `peak_to_valley_residual`, `max_abs_residual` | Fit diagnostics; ellipse residuals are unsigned distances |
| `straightness` | Minimum enclosing parallel strip, from convex-hull orientations |
| `circularity_ls` | Closed-circle radial range about LS center, **not** minimum-zone circularity |
| `area`, `perimeter`, `centroid` | One closed observed polygon; no automatic subtraction of other holes/paths |
| `extent` | Observed projection range along required `direction` |
| `center`, `x`, `y` | Associated center or closed-polygon area centroid |
| `center_distance`, `center_offset` | Center distance plus offset vector |
| `distance` | Line to circle center, or near-parallel line separation; `mode="segments"` uses observed finite segments |
| `minimum_separation` | Minimum observed segment distance, including segment interiors, with witness points |
| `angle` | Acute unoriented fitted-line angle, 0..90° |
| `parallelism`, `perpendicularity`, `angularity` | Angular deviations in degrees; angularity needs `nominal_angle`; not dimensional GD&T zones |
| `intersection` | Infinite fitted-line intersection / virtual sharp |
| `tangency` | Supporting line/circle or circle/circle gap; circle pairs use `mode="external"` or `"internal"`; not observed arc contact verification |
| `symmetry` | Two centers reflected about third reference, a datum line |
| `common_axis` | First reference datum line; remaining centers; max perpendicular offset |
| `true_position_2d` | Twice center offset from `nominal_center` in explicit `datum_origin`, `datum_x_direction` frame |
| `profile` | Observed vertices to `nominal_points` segments; optional `closed`; RMS, max, worst point, sampled Hausdorff |
| `curvature` | Signed three-point circumcircle curvature, one connected contour; noise-sensitive |
| `local_size` | Polygon sections normal to `direction` at `stations`, optional `origin`; rejects multiple material intervals |
| `wall_thickness` | Directed first-support vertex to second-support segment distance distribution; not medial-axis thickness |

Distance operators optionally accept `direction` for signed projection of their
witness vector. Two-line distance/step default to a 0.1° parallelism gate, adjustable
with `parallel_tolerance_degrees`. Nonparallel sides require width at explicit
stations or finite-segment distance.

Profile comparison uses supplied coordinates, with no automatic registration.
Closed nominal profiles report positive error outside and negative inside the
polygon, independent of winding. For a hole this is hole-region inward/outward,
not material excess/missing material. Open profiles return unsigned distances.
Hausdorff is sampled at supplied vertices in both directions, not a certified
continuous-curve extremum. Normal-to-CAD and registered minimum-zone comparison
are not implemented.

### Compounds

Use `operator="compound"` with one compound reference:

* `width`: two lines; `stations` are distances along the first line from its fitted
  center, default `[0]`. Return normal-to-first-line widths, min/max and flank angle.
* `taper`: same with at least two distinct stations; also signed width slope.
* `step`: separation of two near-parallel fitted lands.
* `angle`: acute included angle and virtual sharp of two lines.
* `groove` / `notch`: `roles` maps `opening`, `left`, `right`, optional `bottom` to
  line members. Return opening/bottom width, depth and flank angles. Without bottom,
  the flank intersection defines the V apex. Curved bottoms/transitions use separate
  arc constituents and primitive relationships.

Station and intersection evaluations use supporting lines and can extrapolate;
choose stations inside the observed region when extrapolation is inappropriate.
Member detection and design-intent inference are left to the caller/model/GUI.

### Semantic thread measurements

Use `operator="thread"` with one `repeated_profile` reference selecting **two
connected open opposing sides** of an external thread. Triangular, trapezoidal,
and square profiles use the same observed-geometry method; no nominal template
or standard dimensions are substituted. At least two complete visible crests and
one root on each side are required. Fragmented or ambiguous selections fail locally.

```json
{"id":"thread_dimensions", "operator":"thread", "feature_refs":["thread_1"]}
```

The result's `thread_metrics` contains `minor_diameter`, `major_diameter`,
`root_radius`, `crest_radius`, `pitch`, and `number_of_threads`. Each contains
`minimum`, `maximum`, `average`, `units`, observation `count`, `values`, and min/max
witness pairs. Unresolved metrics have `status="unavailable"`, null statistics,
and a reason. The primary scalar `value` is average pitch in pixels.

Diameters are transverse separations of piecewise-linear opposing root/crest
envelopes evaluated at their combined stations within common support. Pitch is
successive axial spacing between **corresponding flanks facing the same direction**,
intersected at one fixed radial section per side (midway between the 10th and 90th
height percentiles). Rising and falling flank sequences are measured separately
and pooled across both sides. Opposite-flank tooth/gap widths and moving high
points on noisy flat crests are never pitch witnesses. Hysteresis prevents
sampling jitter from creating extra section crossings. Thread count statistics
are the number of resolved crests **per side**, excluding partial end crests;
they do not sum both silhouettes. Root and crest radii are reported **only for arc
primitives** among the thread compound's base members (`base_instance_ids`). A
concave arc reaching the root band is a root radius; a convex arc reaching the
crest band is a crest radius. Arcs on flanks are ignored, and sharp or rounded
polyline turns without a reported arc are never fitted, so their radii remain
unavailable. User-defined thread compounds keep arc segments as members like
any other segment. A `repeated_profile` given only as an edge selection has no
members, so both radii are unavailable.

The axis is estimated from within-side variation. Optional `params.direction`
sets an explicit axis; `prominence` sets crest/root detection height in pixels
(default 25% of side depth), and `plateau_tolerance` controls flat-land centering
(default 0.2% of depth, smaller than prominence). Axial sampling jitter is
accepted up to two median observation spacings. The total reverse excursion is
bounded, so sustained backtracking is rejected even when its individual steps
are small. Resolved crest and root stations must remain strictly ordered. Results are visible 2D
silhouette dimensions, not pitch diameter, lead, helix, or standard conformity.

Unlike primitive line fits, thread measurements use the original selected corner
observations regardless of `corner_exclusion_px`. This policy is recorded in
`support_policy`; no excluded gap is bridged. `annotate()` draws extremum witnesses
and a boxed statistics list with a leader to the selected thread.

### Slot and tooth profile measurements

Like `thread`, these return a `compound_metrics` table whose entries have `minimum`,
`maximum`, `average`, `units`, `count`, `values`, min/max witness pairs and a `method`;
unresolved entries have `status="unavailable"` and a reason.

`operator="slot"` takes one fitted full round slot (`type="slot"`). `width` is the
observed normal gap between the two flats when its base members include two lines
within 2° of parallel (`parallel_tolerance_degrees`), otherwise the stadium fit.
`end_radius` comes from the end arc members, otherwise the stadium fit.
`overall_length` and `end_center_spacing` come from the stadium fit. The primary
`value` is the average width.

`operator="tooth_profile"` takes one `repeated_profile` with the original selected
observations, including corners, as for threads. The profile is **angular** (gears,
splines, sprockets) when its support sweeps at least 60° about a fitted center with
radial depth below 0.6 of the radius, otherwise **linear** along its principal axis
(racks, serrations); `params.domain`, `center` and `direction` override this. Tips
are protrusions of the part material, found from the even-odd interior of all closed
rings (`material_side` overrides), so internal gears and holes work. Metrics:

* `number_of_teeth`: resolved tips per connected run, excluding partial end teeth.
  A closed ring is counted around its full circumference.
* `pitch`: spacing of corresponding same-direction flanks at mid height, in px
  (linear) or degrees (angular), with the hysteresis used for threads.
* `tooth_height`: tip height above the mean of its adjacent roots.
* `tip_diameter`, `root_diameter`: twice the tip/root distance from the center;
  angular profiles only.
* `tip_radius`, `root_radius`: reported arc members only, convex at the tip band or
  concave at the root band. Arcs larger than the tooth depth are crown or root lands
  (for example concentric with a gear) and are ignored.

`prominence` sets tip/root detection height (default 25% of the run's depth). The
primary `value` is the average pitch. Module, pressure angle, pitch diameter and
standard conformity are not inferred.

### Patterns and repetitions

`pattern` accepts member feature references or one pattern compound:

* `domain="linear"`: required `direction`; sorted center spacing, mean/local pitch,
  local/cumulative errors and transverse range. Optional `nominal_pitch`; otherwise
  measured mean is the explicitly labeled reference.
* `domain="polar"`: optional center, otherwise fit from >=3 members; angular spacing
  including seam, indexing and radial/tangential errors. Optional
  `nominal_pitch_degrees`, `phase_degrees`, `nominal_radius`. Assumes a complete
  polar pattern; missing members appear as enlarged gaps.
* `domain="grid"`: required `direction` and integer `cells: [[column,row],...]`
  matching member order. Fit lattice origin/axes, pitches, orthogonality and residuals.
  Requires independently supported rows and columns.
* `domain="arbitrary"`: required `nominal_centers` matching member order; report
  error vectors and max position error, without matching/registration.

`repetition` uses one connected contour and required `prominence` in pixels:

* Linear: `domain="linear"`, required `direction`, one strictly monotone side.
  Uniform resampling and linear detrending precede extrema detection. Vertical,
  overhanging and pooled opposite-side profiles are rejected.
* Angular: `domain="angular"`, required `center`, single-valued radial profile.
  Closed profiles include the wrap gap and count seam extrema once.

Return resolved-extrema count, mean/local pitch, errors, positions, observed span,
profile height and grid resolution. Optional `extrema` is `maxima` or `minima`;
optional `nominal_pitch` defines pitch errors. At least three extrema are required.
Resolved extrema are not necessarily the intended tooth count. Thread pitch and
tooth pitch are generic visible-profile metrics; lead/helix, pitch diameter, module,
pressure angle and standards are not inferred.

## Results, tolerance checks, limits

Feature statuses: `fitted`, `fit_failed`, `abstain`. Measurement statuses:
`measured`, `failed`, `unavailable`. Failed results have no numeric value and cannot
pass. Reports retain observed support, fitted geometry, references, methods, units,
nominal/deviation, tolerance/pass-fail and error reasons.

Tolerance bounds are inclusive **absolute value limits**:
`{"lower":19.8,"upper":20.2}`. `nominal_value` separately defines deviation.
One-sided bounds are allowed. Vector/distribution results cannot be implicitly
reduced for tolerance evaluation. No guard band is applied. `uncertainty=null`
and `uncertainty_status="not_evaluated"` are deliberate; fitting residuals are
not a measurement uncertainty budget.

This is generic 2D metrology, not standardized GD&T certification. Deferred:
minimum-zone circularity, maximum-inscribed/minimum-circumscribed sizes, spline
fitting, CAD import/registration, standard thread/gear templates, automatic anomaly
classification, calibration and uncertainty estimation. No 3D quantity is claimed.

## Validation and reproducible demo

```bash
# From repository root; renderer is needed only for demo/integration tests.
uv pip install --python packages/measurement/.venv/bin/python -e ./research/synthetic-shapes
cd packages/measurement
.venv/bin/python -m pytest -q
```

Run the demo from the repository root:

```bash
PYTHONPATH=packages/measurement/src:packages/edge-model:research/synthetic-shapes/src \
  packages/measurement/.venv/bin/python packages/measurement/examples/synthetic_demo.py
```

`packages/measurement/outputs/washer/` contains the input, request, contours, results,
annotation and projected analytic diameter references. Tests combine analytic
geometry expectations with synthetic-shapes image integration, stale-hash and
abstention checks, topology failures and missing-support rejection. The renderer's
polygon rasterization has dimensional bias: image tests establish integration,
not subpixel accuracy certification.

## CPU performance

For CPU timing results and the reproducible performance benchmark, see
[PERFORMANCE.md](PERFORMANCE.md). Optimized contour workloads measured 1.5–5.8×
faster using Python and the existing dependencies, with unchanged accuracy bounds.

## Dataset-v2 batch evaluation

The batch runner uses the existing contour-model Python environment (Torch) and
imports the measurement module directly. It selects 300 held-out parts with seeded,
family-balanced sampling, projects the stored actual contours to synthetic pixels,
runs release-v2, and evaluates default feature measurements plus a small set of
line/center relationships. This is a contour-input benchmark, not image extraction.

```bash
PYTHONPATH=packages/measurement/src:packages/contour-model/src:research/synthetic-shapes/src \
  packages/contour-model/.venv/bin/python packages/measurement/scripts/evaluate_dataset_v2.py \
  --source /media/T7Shield/inspection-contour-training/v2-expanded \
  --checkpoint packages/contour-model/artifacts/release-v2/model.pt \
  --count 300 --split test --seed 20260907 \
  --output packages/measurement/outputs/dataset-v2-test-300
```

Use a fresh output directory for each run. `selection.json` records IDs and source
hashes before inference. `REPORT.md`, `summary.json`, and `samples.jsonl` contain
coverage/failure statistics and matched reference-selection comparisons.
`index.html` links annotated previews for every sample. Each sample saves compressed
prediction/request/result/reference JSON, a lossless pixel contour NPZ, projection
parameters, rendered image, and an annotated preview of up to 12 measurements.

Reference comparisons apply the existing observable-surface grouping and match
accepted model leaf features to same-semantic references by arc-length IoU. Both
sets of dimensions are fitted to actual observed contours. Their differences
quantify selection sensitivity, not error against nominal dimensions. Nominal
coordinates and semantic truth never enter model inference or measurement fits.

`--reuse /path/prior-run` reuses matching saved model predictions and rendered images
while recomputing all measurements and reference fits. Missing predictions are
recomputed. The new manifest records reuse and current measurement-source hashes;
use this to verify a measurement fix without changing model selections. Run
`packages/measurement/scripts/review_dataset_batch.py /path/run` with the same Python
source paths to add paired outlier images and `DIAGNOSTICS.md`.

## Independent ground-truth correctness evaluation

The dataset-v2 comparison above measures selector sensitivity. For measurement
correctness against independently known values, use the separate
[analytic benchmark](benchmarks/README.md). It constructs deterministic pixel parts
with recorded seeds, parameters, closed-form expected dimensions and nonzero form
errors. Both reference and model selections are scored against the same independent
truth, with missing measurements retained in the denominator. It also distinguishes
finite-polyline dimensions from smooth associated geometry.

### Construction-based distances

The desktop **Distance** tool uses `distance_construction`. The former
`distance_statistics` operator has been removed; requests using it are rejected.
Use an explicit distance construction below.

```json
{"id":"gap","operator":"distance_construction","feature_refs":["edge_a","edge_b"],
 "params":{"construction":"normal_gap","basis":"observed",
           "direction_mode":"normal_a","branches":["auto","auto"],"statistic":"average"}}
```

- `normal_gap` pairs intersections on the same parallel section. It reports signed
  minimum, maximum and interval-weighted average across the valid support. Gaps
  and excluded corners are retained. The maximum is never an endpoint diagonal.
- `section` measures one section at `station` (a scalar position in pixels).
- `closest_clearance` is the continuous shortest distance between bounded observed
  contours. It is a single scalar, with no artificial min/max/average family.
- `center_distance` uses two associated centers; lines require a different construction.
- `center_to_line` and `tangent_clearance` require line A and circle/arc B. The
  perpendicular foot must fall within the supported line. The tangent point must
  be within the supported arc; clearance is signed if the circle crosses the line.

`basis` is `observed` or `fitted` for section constructions. Fitted line support
is bounded per retained run; fitted arcs retain angular support. Extensions need
both `extend: true` and an explicit finite `station_range: [start,end]` and are
shown as virtual geometry. Fitted closest-clearance searches are not supported.
Center/tangent constructions always use associated geometry.

`direction_mode` is `normal_a`, `normal_b`, `x`, `y`, or `custom` (with a nonzero
`direction` vector). Normal modes orient from A toward B; explicit axes retain
sign. The engine also resolves `datum_feature_ref` against a fitted line's normal.
Branches are specified per feature: `auto` requires a unique intersection;
`near`/`far` select the extreme toward/away from the other feature center;
`negative`/`positive` select the corresponding extreme along the direction.
Ambiguous branch choices and absent overlap fail explicitly.

Sections are `p = s*n + t*d`, with `n = [d_y,-d_x]`. Request positions are absolute
image projections on `n`. The GUI expresses them relative to the beginning of A
and marks that origin with `0`. A gap is `t_B-t_A`; negative values are preserved.
The default requested span is A's full projected support. `coverage` is valid
section length divided by the requested span, including missing intervals in the
denominator. Results retain the complete `definition`, `requested_span`, valid
and excluded intervals, direction, coverage and field geometry. Extrema have true
witnesses; averages have contributing bands, not a fabricated witness segment.

Polyline results use endpoint/intersection events and exact linear integration;
fitted circular branches use analytic intersections and stationary points with
adaptive quadrature (absolute tolerance 1e-8 px², relative 1e-9). Display sampling
of circular bands does not determine the reported measurements. Thread diameters
use this same engine between explicitly displayed virtual crest/root envelopes,
perpendicular to the thread axis. Their averages are weighted by axial span.
