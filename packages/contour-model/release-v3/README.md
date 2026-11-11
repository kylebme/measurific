# Three-primitive contour model

This release recognizes only **Line**, **Arc**, and **Circle** base primitives.
There is no arbitrary-profile output class or checkpoint logit. Compound outputs
remain full-round slot, triangular thread, trapezoidal thread, and tooth profile.

The training corpus is
`/Volumes/T7 Shield/inspection-contour-training/primitives-three-base-v2`.
It contains the same 12,000 observations and deterministic splits as the audited
corrected v1 corpus. Former reference-free partial curves are supervised as arcs;
former reference-free complete rings are supervised as circles. The independent
1,000-design evaluation cohort is `primitives-three-base-independent-v2`.

`model.pt` is the compact CPU/MPS/CUDA inference checkpoint. It was warm-started
from release-v2 and trained for 4,000 updates on Apple M3 MPS. Checkpoint selection
used 400 validation designs and retained step 4,000.

```bash
packages/contour-model/.venv/bin/python -m contour_model predict \
  packages/contour-model/release-v3/model.pt \
  /absolute/path/part.contours.npz \
  /absolute/path/prediction.json --device cpu --measure
```

See `reports/test-report.json`, `reports/ood-report.json`,
`reports/independent-report.json`, and `reports/raw-test-report.json` for results.
