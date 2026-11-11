# release-v7-candidate

**Candidate only — not the GUI default** (the GUI still bundles `release-v6/onnx`).

Corpus v3 run `scratch` ([DATASET_V3.md](../DATASET_V3.md)): trained from scratch,
48k steps, lr 3e-4, E007 architecture/recipe plus `--mirror`, checkpoint selected on
PAE validation (best = final step). Mix: v3 clean ×1, PAE ×2, edge-model ×2
(`/media/kyle/T7 Shield/inspection-contour-training/v3-20261018/`). Labels use the
observable-v1 policy: sub-pixel arcs are lines and tiny edge breaks belong to the corner,
so expect fewer arcs on fine tooth crowns and chamfer-like corner details than v6.

ONNX export (`onnx/`, same three-graph layout as v6): parity max |Δlogit| 3.4e-5 over
24 contours (v3 PAE/edge test and holdout, v3 clean test, GUI demo contours). The ONNX
path reproduces the torch bench exactly (v3 PAE test .8482, v3 PAE holdout .8815). GUI
worker loads it and analyzes the demo images with both PAE and deep-learning edges.

| Base instance F1 | release-v6 | this candidate |
|---|---:|---:|
| v3 PAE test | .662 | .848 |
| v3 edge test | .672 | .848 |
| v3 holdout families (PAE / edge) | .544 / .544 | .882 / .878 |
| v3 stress (PAE) | .610 | .774 |
| old clean test, observable labels | .810 | .923 |
| old edge test, old labels | .911 | .755 |

The last row uses the old labelling policy; see DATASET_V3.md for why it drops.
Synthetic only: real-part evaluation is the purpose of this candidate.

## Try it in the GUI (development run)

```bash
cd app
INSPECTION_CHECKPOINT=../contour-model/release-v7-candidate/onnx npm run dev
```

Unset the variable to return to release-v6. Promoting it means changing
`app/python/worker.py` `DEFAULT_MODEL`, `app/package.json` and `app/electron/main.cjs`,
as was done for v6.
