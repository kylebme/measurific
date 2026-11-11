# Training-free contour recognition

The optional `geometric-v1` backend needs no checkpoint, training corpus,
PyTorch import, GPU, or MPS. The learned release remains available for comparison.
The synthetic generators and their datasets have not been modified.
The package still lists PyTorch as an installation dependency for its retained
training/learned commands; the geometric execution path does not import it.

## Research and implementation review

This is an original implementation informed by the following sources, **not** a
port of a published implementation or a claim to reproduce its guarantees.

| Source inspected | Relevance and decision |
| --- | --- |
| Gribov, [Optimal Compression of a Polyline with Segments and Arcs (2016)](https://arxiv.org/abs/1604.07476), and [efficient combinatorial algorithm (2018)](https://arxiv.org/html/1811.05659v1) | Tolerance-constrained optimization; distinct line/arc penalties; avoid unnecessary fits. We use penalties 2 and 3 with error as the tie-breaker. Our bounded candidate graph and local refinement are **not** his exact global algorithm or maximum-jump implementation. |
| Rosin–West, [Segmentation of edges into lines and arcs (1989)](https://www.sciencedirect.com/science/article/pii/0262885689900048) | Publisher abstract describes recursive polygonal approximation followed by arc fitting over connected line groups. Supports coarse proposals plus merging. Our explicit physical tolerance differs from their threshold-free selection. The author's software listing was located, but its linked download was inaccessible. |
| Chen–Ventura–Wu, [Segmentation of planar curves into circular arcs and line segments (1996)](https://scholars.ncu.edu.tw/en/publications/segmentation-of-planar-curves-into-circular-arcs-and-line-segment/) | Abstract describes initial breakpoints followed by fitting-based boundary adjustment. Motivates refinement rather than treating proposal locations as exact geometric joins. Their specified entity-count formulation is not our objective. |
| [A dynamic programming approach for fitting digital planar curves with line segments and circular arcs](https://www.sciencedirect.com/science/article/pii/S0167865500001045) | Publisher summary explicitly favors lines when line/arc errors are comparable, and notes candidate/start-direction sensitivity. We make line preference explicit and test seam, orientation, and winding changes. |
| Abdul-Rahman–Chernov, [Fast and numerically stable circle fit](https://arxiv.org/abs/1505.03795), and [circle-fitting error analysis](https://arxiv.org/abs/0907.0421) | Numerical conditioning and algebraic-versus-geometric fitting matter, especially on shallow arcs. We use local centering/scaling, an algebraic initializer, geometric refinement, and support guards. We do not implement their specialized solver. |
| OpenCV [EdgeDrawing API](https://docs.opencv.org/4.13.0/d1/d1c/classcv_1_1ximgproc_1_1EdgeDrawing.html) and [DetectArcs implementation](https://github.com/opencv/opencv_contrib/blob/4.x/modules/ximgproc/src/edge_drawing.cpp) | Reviewed line-group arc detection, angular/coverage checks, and cyclic handling. Not adopted as the backend: it begins with image-derived edge chains, whereas we already have ordered precision observations and require exhaustive, nonoverlapping ownership. |
| MIT Media Lab [arc-line vectorizer](https://github.com/mitmedialab/arc_line_vectorization_suede), including [fitting source](https://raw.githubusercontent.com/mitmedialab/arc_line_vectorization_suede/release/vectorize/low_geometry/fitting.py) | Reviewed its line/circle fits, geometric refinement, inflection handling, and documented DP/top-down alternatives. Its sketch/robot-motion objective, approximate closure, and geometry beautification differ from metrology. No code or additional dependencies were copied. |

## Implemented algorithm

1. Validate the observed closed-ring topology. Ignore every other input array:
   no family, primitive IDs, nominal coordinates, synthetic labels, or checkpoint.
2. Test a complete circle geometrically, including edge-interior distances; a
   sparse polygon with cocircular vertices is not automatically a circle.
3. Pick a geometry-defined cyclic seam and generate fit-aware recursive boundary
   proposals. Stop splitting an interval as soon as a line or arc fits; otherwise
   split at its greatest deviation from the endpoint chord. This avoids an
   expensive graph over every tessellation point of a valid arc. No original
   coordinates are replaced, discarded, or smoothed for primitive fitting.
4. Search a bounded candidate graph with dynamic programming. Minimize total
   primitive penalty (line=2, arc=3), then integrated squared residual. A valid
   line wins over an arc on the same support. Test up to 48 preceding candidate
   intervals; skip an arc fit if its complexity penalty already exceeds the
   best path. Do not assume a failed circular fit rules out every longer span.
5. Use arc-length-weighted orthogonal line fits and circular fits initialized
   algebraically and refined by geometric Gauss–Newton iterations in a local,
   normalized coordinate frame. Validate against **all** original vertices and
   chord interiors, not only a reduced candidate polyline. Guard angular support,
   reversals, and degenerate circles.
6. Refine neighboring boundaries to lower residual, holding their model kinds
   fixed, then merge compatible neighbors, including across the cyclic seam.
7. Return disjoint base selections covering every raw edge exactly once, along
   with adjacency, diagnostic fits, and conservative compound relationships.

This is a candidate-graph/local-search approximation. It has neither a global
minimum guarantee over every raw breakpoint nor exact G1 continuity between
independently fitted primitives. The selected *observations* remain connected;
diagnostic ideal fits can have small join gaps. Measurement separately fits the
original observations, with the existing user-configurable corner exclusion.

## Tolerance and labels

Default maximum deviation: **0.75 px** for camera contours, **0.01 mm** for mm
inputs. The pixel default accommodates the roughly 0.53 px worst radial residual
observed in the independent raster-circle/PAE smoke test. It is a starting
recognition setting, not a validated camera uncertainty or a manufacturing
tolerance. It does not scale with the overall part size. Explicitly scale the
tolerance when converting units.

`min_arc_angle` defaults to 2 degrees and minimum circular support is six
vertices. These guards can leave unresolved short/shallow arcs represented by
lines. If both line and arc fit within tolerance, the line is selected. A closed
noncircular curve is split into pieces, not assigned the whole-circle class.

The tolerance is a strict approximation bound: a notch, burr, or form error
outside it can produce extra primitives. The algorithm does **not** infer that
a damaged surface was nominally one line/circle. Increasing tolerance trades
detail for simplicity and can erase meaningful fillets from the segmentation.
The raw data is retained in either case. This distinction is particularly
important on the old, deliberately deformed synthetic demo contours.

There are no invented neural confidence scores. `source=geometry`,
`recognition_backend=geometric-v1`, `recognition_tolerance`, and `fit_evidence`
identify provenance and the actual acceptance evidence. Fit evidence is
diagnostic; it is not a replacement for final metrology.

## Compound rules and deliberate omissions

- Full-round slots: closed hole with two parallel lines and two approximately
  equal semicircles; verify relative centers, line offsets, and opposing sides.
- Radial repeats: a near-star-shaped closed ring with at least six regularly
  spaced, sufficiently prominent radial peaks and high periodic correlation.
  `tooth_profile` means observed radial repetition, not a verified gear standard.
- Thread-like regions: at least four regular crests per side, compatible pitch,
  depth, flanks, overlap, separation, and approximately half-pitch phase between
  opposite sides. Membership always consists of complete base instances and
  exactly two connected open runs. Triangular versus trapezoidal is currently
  a flank-angle heuristic, explicitly reported in `rule_evidence`; it is not
  standard identification.

Aligned annular grooves are not called single-start threads. The old mirrored
synthetic threads can therefore remain ungrouped, intentionally. Multi-start
threads, strongly tapered/occluded threads, asymmetric radial patterns, and
linear racks are not comprehensively recognized by these conservative rules.
Their line/arc primitives remain available. A 2D silhouette cannot prove a 3D
helix or uniquely distinguish all manufactured compound types.

## Run and compare

From the repository root:

```sh
packages/contour-model/.venv/bin/python -m contour_model segment \
  /absolute/path/part.contours.npz /absolute/path/geometric.json \
  --units mm --tolerance 0.01
```

Input contract: finite `actual_xy_mm[N,2]` (historical name also used for px),
integer `ring_offsets[R+1]`, one outer ring and disjoint nonnested holes. Implicit
closure without a repeated vertex; arbitrary ring order and winding. Use
`--no-compounds` for base-only output. CLI does not require a checkpoint.

GUI environment settings (restart its Python worker/app after changing them):

```sh
INSPECTION_RECOGNITION=geometry npm run dev
INSPECTION_GEOMETRY_TOLERANCE_PX=0.5 npm run dev
INSPECTION_RECOGNITION=learned npm run dev
```

The last command restores release-v3 recognition; metrology remains identical.

## Validation and inspection

Independent analytic tests live in `tests/test_geometric.py`: sharp/rounded hex
nuts, mixed and oblique polygon geometry, convex and concave fillets, slots,
circles versus lobes/notches, shallow deviations, rounded phased thread-like
profiles versus aligned grooves, radial repeats, scale/sampling changes,
rotation/winding/seams, exact edge ownership, immutability, input allowlisting,
invalid inputs, JSON serialization, measurement handoff, and no PyTorch import.
The GUI also has an actual raster-image → PAE → recognition → measurement test.
Legacy learned-backend integration tests remain in place separately.

The corpus audit reads one existing held-out design per family (46 families):

```sh
packages/contour-model/.venv/bin/python packages/contour-model/scripts/benchmark_geometric.py \
  '/Volumes/T7 Shield/inspection-contour-training/primitives-three-base-v2' \
  packages/contour-model/artifacts/geometric-v1-review --workers 2
```

`index.html` contains zoomable previews and per-design evidence; `summary.json`
records coverage, approximation checks, and runtime. These are **not** semantic
accuracy numbers against the flawed labels, and are not a real-part validation.
The busy examples can contain over 100,000 observations and hundreds of holes;
this initial Python implementation is slower than the neural model on complex
parts. Record timing context and do not interpret concurrent audit timings as
single-image GUI latency. Real nut/bolt captures remain the next acceptance test.

### Recorded validation, 2026-09-20

- 219 Python tests passed across contour recognition, metrology, and GUI worker
  integration, including 25 independent geometric tests.
- 37 renderer tests and 8 Electron tests passed; TypeScript/Vite build passed.
- All 46 family samples partitioned every raw edge exactly once. Independent
  recomputation from the returned fit parameters confirmed the 0.01 mm radial/
  perpendicular residual bound, including edge interiors.
- Recognition-only concurrent audit: median **0.545 s**, p95 **10.183 s**,
  maximum **11.935 s**, using two CPU workers. These include unusually dense
  repeated profiles, not just small ordinary parts.
- The 109,674-point / 225-ring slotted grille took **3.92 s** in that audit,
  versus 36.26 s before fit-aware proposal optimization. No semantic accuracy
  or real-world generalization claim follows from these timing/error checks.
- CLI smoke, JSON serialization, raw-coordinate immutability, no-checkpoint
  worker startup, and no-PyTorch geometric import were verified.

Open the generated review (`artifacts/geometric-v1-review/index.html`) or
the machine-readable audit (`artifacts/geometric-v1-review/summary.json`).
