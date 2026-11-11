# Validation and performance

The tables below preserve measurements from 7 September 2026, before the
noise-path corrections. They are historical evidence, not results for every
subsequent revision or guarantees for other machines or images. Current
accuracy checks and intentional departures from the iterative MATLAB fixture
are documented in [INVESTIGATION.md](INVESTIGATION.md).

## Environment

- AMD Ryzen 7 3700X (8 physical cores / 16 logical CPUs)
- Linux x86-64
- Python 3.13.13
- NumPy 2.5.3
- SciPy 1.18.1
- Numba 0.67.0, 16 worker threads
- CPU only; Numba kernels warmed before timing

## Correctness

`python -m pytest -W error` passes all 52 tests. The suite includes analytic
vertical and horizontal partial-area steps, all combinations of edge order 1/2
and smoothing 0/1/2, input-contract checks, and the upstream MATLAB golden
ring fixtures. The two-iteration reconstruction matches every MATLAB field and
candidate to the upstream 1e-3 tolerance.

The threshold-seeded silhouette path is bit-exact with the full unsmoothed
detector for:

- `uint8`, `uint16`, `float32`, and `float64` inputs;
- PAE edge orders 1 and 2;
- dark-on-light and light-on-dark polarity;
- an exterior contour with an interior hole;
- Gaussian-blurred silhouettes from sigma 0.5 through 4 pixels;
- all 2,376 candidates in the upstream MATLAB ring image.

The tests also cover reusable workspaces, empty masks, component-area
filtering, and invalid configurations. Bit-exact means candidate order and all
eight output arrays are identical, not merely within a tolerance.

Against the MATLAB fixture (`threshold=25`, `order=2`, 2,376 edges), maximum
absolute errors were:

| Feature | smoothing=0 | smoothing=1* |
| --- | ---: | ---: |
| x | 7.11e-15 px | 5.68e-14 px |
| y | 7.11e-15 px | 5.68e-14 px |
| nx | 0 | 1.52e-15 |
| ny | 0 | 1.51e-15 |
| curvature | 1.73e-18 px^-1 | 6.03e-15 px^-1 |
| low-side intensity | 0 | 7.45e-7 |
| high-side intensity | 0 | 1.49e-6 |

Candidate indices were identical. As documented by the upstream project, the
smoothing=1 comparison excludes one known differing candidate (array index
258); the other 2,375 candidates produce the errors above. The tiny intensity
difference is from the reference float32 3x3 convolution mask; geometric
features remain near machine precision.

The optimized smoothing=0 implementation was also compared directly with the
unmodified upstream algorithm on ten seeded 64x80 random images, both edge
orders, at threshold 80. Candidate indices were exactly equal and every
floating feature agreed to <=1e-12.

## Steady-state benchmark

The benchmark input is a `uint8` image containing one filled rectangular
object. Full-detector timings include conversion to the reference float64
representation. Seeded timings include OpenCV thresholding, full-resolution
contour extraction, seed-band search, PAE fitting from the original `uint8`
samples, and output creation using a reusable workspace. All timings exclude
Numba compilation. Sparse contours are favorable; complex silhouettes cost
more in proportion to contour length.

| Image | smoothing | Edges | Median wall time |
| --- | ---: | ---: | ---: |
| 256x256 (0.066 MP) | 0 | 510 | 0.771 ms (11 runs) |
| 1000x1000 (1 MP) | 0 | 1,998 | 7.082 ms (9 runs) |
| 2000x2500 (5 MP) | 0 | 4,498 | 77.250 ms (7 runs) |
| 3000x4000 (12 MP) | 0 | 6,998 | 169.107 ms (7 runs) |
| 3000x4000 (12 MP) | 1 | 6,990 | 697.217 ms (3 runs) |

Threshold-seeded 12 MP results from the same machine:

| Search radius | Edges | Median wall time | Speedup vs 172.085 ms full run |
| ---: | ---: | ---: | ---: |
| 1 px | 6,998 | 9.934 ms (9 runs) | 17.3x |
| 3 px | 6,998 | 14.092 ms (9 runs) | 12.2x |

Fusing derivatives and non-maximum suppression, preserving column-major
working storage, and removing full-image coordinate grids reduced the 12 MP
smoothing=0 path from an initial 573.917 ms to 169.107 ms on this host (3.39x).
No `fastmath`, float32 conversion, or altered PAE equations were used.

## The 1 ms requirement

The recommended radius-3 seeded result is about 14x over a 1 ms budget. A 12 MP
`uint8` input alone represents 12 GB/s if read once in 1 ms, before threshold
comparison, contour following, seed refinement, or PAE fitting. Therefore 12
MP in approximately 1 ms is still not achievable here without changing the
requirement or computation model.

To approach 1 ms, crop a pre-known region of interest, use tracked contour
bands from the preceding frame, reduce sensor resolution, or move threshold
and contour extraction to camera/FPGA hardware. Those choices change system
architecture and are intentionally not hidden behind an inaccurate “fast”
mode.
