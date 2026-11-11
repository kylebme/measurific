# CPU measurement performance

The optimization uses Python and the existing NumPy, SciPy, and Shapely
dependencies. It adds no GPU path, JIT, native extension, or dependency.

## Measured results

Median of seven warmed runs on macOS arm64, Python 3.12.8, NumPy 2.5.3,
SciPy 1.18.1, Shapely 2.1.2. Baseline is commit
`cfc919a3150b9d3cb18657d591f4487276efaa14`. Both versions used the same
interpreter, fixtures, benchmark script, and dependency versions. Timings include
selection, fitting, measurement operators, and JSON-safe report construction;
imports and fixture construction are excluded. Each workload succeeds completely.

| Workload | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| 12 known-truth families, complete requests | 45.51 ms | 29.63 ms | 1.54× |
| Noisy ellipse, 1,024 points | 32.97 ms | 8.23 ms | 4.01× |
| Closed nominal profile, 1,024 points | 56.49 ms | 9.70 ms | 5.82× |
| Wall thickness, 1,024 points per boundary | 17.44 ms | 9.74 ms | 1.79× |
| Distance statistics, 1,024-point contours / 128 samples | 8.51 ms | 2.64 ms | 3.22× |
| Bowed line and straightness, 8,193 points | 75.82 ms | 15.30 ms | 4.95× |

[Raw results](benchmarks/performance-results.json) record exact timings and
environment versions. These are contour-input measurements; image extraction,
annotation, file I/O, and model inference are outside their scope. Actual gains
depend on geometry, point count, and requested operators.

## Changes

- Ellipse fitting supplies an analytic sparse Jacobian to the existing joint
  orthogonal least-squares optimizer. It retains every observation, fitting
  bounds, convergence tolerances, and geometric validation.
- Straightness uses rotating calipers after the convex hull. The strip search
  takes O(H) time and storage for H hull vertices, replacing an O(NH) projection
  matrix. Collinear and very thin supports retain their existing treatment.
- Point-to-profile distances use batched geometry operations and an exact segment
  spatial index for larger workloads. Reverse profile comparison constructs the
  measured shape once. Disconnected components never acquire connecting edges.
- Distance statistics locates all uniform arc-length samples in one search of
  cumulative segment lengths, reusing shapes for the continuous minimum.
- Selection assembles connected runs from mask transitions instead of walking
  each edge in Python. Numeric report arrays are validated and converted in bulk.

## Reproduce and validate

From the repository root, after installing the package and test dependencies:

```bash
packages/measurement/.venv/bin/python -m pytest -q packages/measurement/tests
packages/measurement/.venv/bin/python packages/measurement/benchmarks/benchmark_performance.py \
  --repeat 7 --verify-truth --output /tmp/measurement-performance.json
```

For a before/after comparison, use this same benchmark script and interpreter
with `PYTHONPATH` pointing at the baseline package's `packages/measurement/src`, then
at the updated package's `packages/measurement/src`. Run each sequentially without other
CPU-intensive work. `--only profile` (or another workload substring) limits timing.

Validation: **69 tests passed**, including dense-reference noisy ellipse fits,
analytic derivative checks, exhaustive small edge-selection masks, brute-force
straightness comparisons, disconnected support, profile sign/winding, and
nonfinite serialization rejection. The independent construction benchmark also
passed **8,279 truth checks across 300 cases**, with its original accuracy bounds.
Timing thresholds are intentionally absent from unit tests.
