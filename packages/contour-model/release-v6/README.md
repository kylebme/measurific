# release-v6

Autoresearch run `E007-full-edge3-32k` (see `../RESEARCH.md`): release-v5-pae-candidate
weights + 32k steps on clean intent-v1 contours ×1 and edge-model-extracted renders
(`primitives-edge-v1-train-full`, 6,927 renders) ×3. Same architecture and ONNX layout
as v5 (`onnx/`, parity max |Δlogit| 4.5e-4). Use with the default post-processing
(`merge_tol=1`, `inference_primitives.DEFAULT_POST`).

Benchmark (`scripts/bench.py`, edge-model renders of held-out designs; ONNX path):

| | v5 (GUI) no merge | v5 + merge | **v6 + merge** |
|---|---|---|---|
| edge test instance F1 | .8638 | .8966 | **.9108** |
| edge test precision | .856 | .939 | **.956** |
| edge val instance F1 | .8790 | .9098 | **.9215** |
| edge val resolvable F1 (≥4 px, exact families) | .9465 | .9784 | **.9882** |
| clean contours F1 | .9067 | .9188 | **.9215** |

Selected 2026-09-27 as the best of six final candidates (E007, E004, F003, F004,
F005, E009): top on every edge-val metric (micro .9215, exact .9643, resolvable
.9882, family macro .9129) and every edge-test metric. Seed-repeat noise is about
±.006 val micro, so E009 (.9209) is a statistical tie.

Synthetic only: real-capture validation is still missing.

The GUI bundles `release-v6/onnx` (`app/python/worker.py` `DEFAULT_MODEL`,
`app/package.json`, `app/electron/main.cjs`).
