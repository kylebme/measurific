# Contour primitive recognition and measurement handoff

The GUI defaults to the **learned recognizer**, with an optional training-free
geometric backend for Line, Arc, and Circle. See [GEOMETRIC.md](GEOMETRIC.md) for the research review, algorithm,
tolerances, compound rules, validation, and comparison commands. It runs on CPU
without loading a checkpoint or PyTorch. See [INTENT_TRAINING.md](INTENT_TRAINING.md)
for the current intent-robustness training experiment.

The retained learned model recognizes **Line, Arc, and Circle**. A secondary
classifier identifies **Full round slot, Triangular thread, Trapezoidal thread
(including square), and Tooth Profile**. Compound instances reference their
constituent base instances, preserving individual measurement selections.

The released checkpoint is `release-v3/model.pt`. Experiments, test reports and
review images belong in `artifacts`; large datasets live on `/media/T7Shield`.
See [PRIMITIVES.md](PRIMITIVES.md) for the taxonomy, data contracts, generator,
training commands and limitations, and [the release validation report](release-v3/VALIDATION.md)
for results.

From the repository root:

```bash
packages/contour-model/.venv/bin/python -m contour_model predict \
  packages/contour-model/release-v3/model.pt \
  /absolute/path/part.contours.npz \
  /absolute/path/prediction.json --device cpu --measure
```

Inputs are finite `actual_xy_mm` coordinates `[N,2]` and integer `ring_offsets`
`[R+1]`. Rings close implicitly: do not repeat the first point. Supply one outer
ring and disjoint, non-nested through openings. Either winding and ring ordering
are accepted. Use a Cartesian y-up frame and `--units px` for pixel units.
Calibration and distortion correction belong upstream of measurement.

Sampling depends only on observed geometry. Omit `--tokens` for the adaptive
release. Predictions return raw-edge support, contour spans, boundary search
regions, adjacency and compound membership. Dimensions come from classical
measurement of the original observations, not model-regressed geometry.

The approximately 780k-parameter model supports CPU inference. Training supports
CUDA and Apple MPS; synthetic generation is CPU-only. Confidence scores remain uncalibrated.
Compound classification always chooses the most likely class for each detected
instance; base acceptance uses the configured threshold.

```bash
packages/contour-model/.venv/bin/python -m pytest packages/contour-model/tests -q
```

The [legacy documentation](LEGACY_README.md) describes the superseded 16-class
models and their datasets. Do not use its class IDs for the primitive release.
