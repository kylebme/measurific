# Contour primitives and compounds release

This release replaces the 16-class model with Line, Arc, Circle, and Arbitrary
Profile plus a learned secondary classifier for Full round slot, Triangular
thread, Trapezoidal thread (including square), and Tooth Profile. Compound
instances retain references to their constituent base instances. Keyhole panels,
buttress threads, and round threads are removed from the contour generator.

The 780,567-parameter checkpoint is `model.pt`. See [VALIDATION.md](VALIDATION.md)
for final scores, [TAXONOMY.md](TAXONOMY.md) for class and generator contracts, and
[DATASET.md](DATASET.md) for provenance. [The review](review/reference-and-prediction.png)
compares fresh reference and prediction examples, including the corrected labyrinth.

From the repository root:

```bash
packages/contour-model/.venv/bin/python -m contour_model predict \
  packages/contour-model/release-v2/model.pt \
  packages/contour-model/release-v2/examples/slotted_bracket.contours.npz \
  packages/contour-model/artifacts/my-prediction.json --device cpu --measure
```

For a separate installation, extract `source.tar.gz` into an empty directory,
create a Python 3.11+ environment, install a suitable PyTorch build, and run
`python -m pip install ./packages/contour-model`. Generation/training also needs
`python -m pip install ./research/synthetic-shapes`; review images require matplotlib.
The separate pixel measurement adapter is in `measurement` and can be installed
with `python -m pip install ./packages/measurement`.

Input NPZ arrays are `actual_xy_mm` (finite float64 `[N,2]`) and `ring_offsets`
(integer `[R+1]`, from 0 to N). Closure is implicit; do not repeat the first point.
Supply one outer ring with disjoint, non-nested openings. Winding and ring order
may vary. Use `--units px` for pixels. Omit `--tokens`: this checkpoint uses
`adaptive-turn-v1` sampling, matching its training cache. Model inputs never use
nominal geometry or reference labels.

Prediction JSON schema 2 returns `features` (base instances), `compound_features`,
and adjacency/membership `relations`. Compound support equals the union of its
referenced base supports. The probability NPZ has base `primary` `[N,5]`, base
`semantic` `[N,4]`, secondary `compound` `[N,5]`, defect `[N,3]`, and boundary `[N]`.
ID 0 is unlabeled/no-compound; base and compound IDs use separate namespaces.

Every detected compound receives its most likely class. Detection can return no
compound, and thread candidates require enough spans to support repeated flanks.
Base instances retain the configurable acceptance threshold (default 0.7).
Confidence scores are uncalibrated. Measurement uses original observations;
unsupported diagnostic measurement routes return an explicit unsupported status.
Real captured contours have not been validated.
