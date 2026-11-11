# PAE subpixel edges

`inspection-pae` is a standalone, CPU-only Python implementation of the
Partial Area Effect (PAE) subpixel edge detector for two-dimensional grayscale
images. It returns the subpixel position, normal, curvature, and intensity on
both sides of every detected edge.

Accuracy and reference fidelity take precedence over approximate shortcuts.
Calculations therefore use `float64`, and the variable-size per-edge windows
remain Numba-compiled translations of the established implementation. A
threshold-seeded fast path is provided specifically for controlled telecentric
part silhouettes.

## Installation

From this directory:

```bash
python -m pip install -e .
```

The runtime dependencies are NumPy, SciPy, Numba, and headless OpenCV.
scikit-image is not required.

## Use

```python
import numpy as np
from pae import detect, warmup

# Optional: compile kernels once at process startup.
warmup()

gray = np.load("gray_image.npy")  # shape (height, width)
edges = detect(gray, threshold=25, smoothing=0, order=2)

xy = np.column_stack((edges.x, edges.y))
normals = np.column_stack((edges.nx, edges.ny))
```

`threshold` is a gradient magnitude in the image's intensity units. A value of
25 is the reference example for 8-bit imagery; it is not universally optimal.
The image must already be grayscale.

### Telecentric silhouette fast path

For a dark part against a bright field, thresholding can restrict PAE fitting
to a narrow band around the silhouette. Thresholding supplies seeds only; all
reported features still come from the original grayscale intensities and the
reference PAE equations.

```python
from pae import SilhouetteWorkspace, detect_silhouette, warmup

workspace = SilhouetteWorkspace(gray.shape)  # reuse for every same-size frame
warmup(include_silhouette=True)

edges = detect_silhouette(
    gray,
    pae_threshold=25,
    silhouette_threshold=120,
    polarity="dark-on-light",
    search_radius=3,
    min_area=100,
    smoothing=1,  # compensated low-noise equations
    workspace=workspace,
)
```

Use `polarity="light-on-dark"` for a bright part. Exterior boundaries and
holes are retained. `min_area` and `max_area` apply independently to each
binary contour and can reject dust or unrelated objects.

`search_radius` is the maximum distance, in pixels, searched around each
threshold contour. It must be large enough to include the maximum-gradient
candidate. Radius 1 is sufficient for sharply focused, well-centered
thresholds; the default radius 3 covered Gaussian blur through sigma 4 pixels
in the included parity tests. Validate this parameter against the production
optics and expected threshold drift. `smoothing=0` (the library default) uses
the optimized seed-band implementation. `smoothing=1` and larger values run
the full compensated detector and then filter candidates to the seed band;
these modes do not have the same speedup. `detect_seeded` also accepts external
integer `(x, y)` seed pixels, such as an ordered Canny contour.

The output fields are one-dimensional NumPy arrays in deterministic reference
order:

| Field | Meaning |
| --- | --- |
| `x`, `y` | zero-based subpixel coordinates (column, row) |
| `nx`, `ny` | unit normal pointing toward greater intensity |
| `curv` | signed curvature in inverse pixels |
| `i0`, `i1` | low and high side intensity (`i0 <= i1`) |
| `position` | zero-based, column-major linear candidate-pixel index |

### Detector modes

- `smoothing=0`: clean/noiseless images; the fastest and most faithful path.
- `smoothing=1`: low-noise images; 3x3 averaging with analytically compensated
  PAE equations and close-edge handling. Blurring an image and calling mode 0
  is a different estimator.
- `smoothing>1`: iterative reconstruction with analytic circle/pixel areas;
  substantially slower. More iterations can worsen sharp corners.
- `order=1`: locally straight edges; `order=2`: locally curved edges.

The first call includes Numba compilation. Use `warmup()` or
`warmup(include_silhouette=True)` and measure only steady-state calls in
latency-sensitive systems.

PAE's local edge assumptions do not cover arbitrary defocus or sharp corners.
Passing topology checks does not establish subpixel coordinate accuracy. See
[the blur/noise investigation](INVESTIGATION.md) for implementation fixes,
independent optical tests, and the remaining full-outline accuracy gap.

## Performance target

A complete 12 MP detection in 1 ms remains unrealistic on the benchmark CPU,
but threshold seeding reduces a representative 12 MP silhouette from about
172 ms to 9.9 ms with radius 1 or 14.1 ms with the safer radius 3. The included
benchmark reports steady-state wall time:

```bash
python benchmarks/benchmark.py --height 3000 --width 4000 --repeats 5
python benchmarks/benchmark.py --height 3000 --width 4000 --repeats 5 \
  --seeded --search-radius 3
```

Seeded performance scales with total pixels during threshold/contour extraction
and with contour length and search-band width during refinement. Reuse a
`SilhouetteWorkspace` to avoid allocating the 12 MP binary mask every frame.
See `VALIDATION.md` for measured results from this checkout.

## Reference fidelity and license

This is a modernized fork, not a clean-room reimplementation. The numerical
code derives from [`gravi-toni/subpixel-edges`](https://github.com/gravi-toni/subpixel-edges)
commit `a4f8f1a4160f5405d85036b68c7a427568b6cd9d`, which was validated against
Agustin Trujillo-Pino's MATLAB implementation of the method. The algorithm is
described in [“Accurate Subpixel Edge Location Based on Partial Area Effect”](https://doi.org/10.1016/j.imavis.2012.10.005).

The upstream Python code is Copyright (c) 2020 Giuseppe Vitucci and Nicola
Vitucci and licensed under MIT. The copyright notice and full license are in
`LICENSE`; exact provenance and the local change boundary are in `NOTICE.md`.
The original MATLAB-generated parity fixtures for smoothing modes 0, 1, and 2 are
retained as base64 test data with upstream attribution.

## Verification

```bash
python -m pytest
```

The suite checks the original MATLAB golden results, analytically generated
partial-area steps, API contracts, and all detector modes. The encoded parity
fixtures under `tests/data` come from the upstream MIT-licensed repository at
the commit named above.
