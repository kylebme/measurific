# Primitive/compound release measurement validation — 2026-09-09

The current `packages/contour-model/release-v2/model.pt` was evaluated on 300 independently
constructed analytic pixel parts (25 seeds in each of 12 families). All 8,279
reference-selection checks passed their predeclared numerical/discretization bounds.
Truth was frozen before inference and measurement; it comes from construction
formulas, including nonzero bowed-line and lobed-circle form errors.

| Selection source | Passed | Outside bound | Unavailable | Failed | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Known geometry selections | 8,279 | 0 | 0 | 0 | 8,279 |
| Current model selections | 7,238 | 277 | 755 | 9 | 8,279 |

The model lane uses the same explicit inspection plan, replacing support only
after taxonomy-aware one-to-one matching at arc-length IoU >= 0.5. New base and
compound class IDs occupy separate namespaces. Unmatched features never receive
reference support. Whole polygon and rounded-rectangle plans have no single new
model class; their checks remain unavailable in this benchmark. Ellipse plans
need a matching arbitrary-profile instance, which may not cover the whole opening.

All 125 bowed-line checks exceeded bounds with model support, while all passed
with the known support. Rounded plates account for another 106 out-of-bound checks.
These are selection-sensitive measurements; passing independent measurement tests
does not establish accurate automatic feature boundaries. The nine failed checks
are three linear-rack repetition requests (three fields each) whose selected
support is disconnected; the measurement module explicitly rejects joining it.
The checkpoint was not retrained or changed.

Every prediction was also evaluated directly through the GUI's schema-2 adapter,
producing 10,968 successful default measurements across 300 handoffs. This count
is a handoff coverage measure, not an independent accuracy score.

Additional regression coverage tests exact rotated/translated stadiums in both
windings and with three contour seams, retaining constituent line lengths and arc
radii plus compound width, overall length and end radius (absolute bound 1e-6 px).
Malformed membership/support fails locally; abstained base support remains visible
without preventing independently accepted compound fits; disconnected thread sides
retain separate runs and do not generate an inferred pitch.

Reproduce from the repository root:

```bash
PYTHONPATH=packages/measurement/src:packages/contour-model/src:research/synthetic-shapes/src \
  OMP_NUM_THREADS=4 packages/contour-model/.venv/bin/python \
  packages/measurement/benchmarks/evaluate_known_truth.py \
  --checkpoint packages/contour-model/release-v2/model.pt \
  --per-family 25 --seed 20260908 \
  --output packages/measurement/outputs/primitives-known-truth-20260909
```

Choose a fresh output directory on reruns. The local run contains
the full report (`../outputs/primitives-known-truth-20260909/REPORT.md`),
all preview images (`../outputs/primitives-known-truth-20260909/index.html`), and
per-case frozen formulas, contours, requests, predictions, results and scores.
`manifest.json` records generator, measurement source and checkpoint hashes.

These results cover contour measurements, not PAE extraction accuracy, camera
calibration, physical instrument uncertainty, or arbitrary noisy captured parts.

Software checks passed: 82 measurement tests, eight Python worker/matching tests,
five GUI state tests, and the TypeScript/Vite production build. A browser review
using the real slotted-bracket worker result verified switching layers, slot
measurements, constituent links and navigation back to primitive support. Electron's
preload API was supplied by the preview harness; the actual Python bridge was
validated separately with all four demo inputs.
