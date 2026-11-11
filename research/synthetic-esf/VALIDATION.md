# Validation performed

## Version 0.2: field optics and benchmark extension

`python -m pytest -q`: **47 passed**. The extended automated tests cover distortion inversion/Jacobians, mapped contour normals, a direct-discrete-convolution reference for spatially varying blur, tile/ROI consistency, imported PSF field nodes/provenance, artifact split isolation, replay, held-out-truth independence of fitted coefficients, mesh tolerance, and suite failure reporting. The installed version 0.2.0 CLI passed both noiseless and RAW benchmark evaluations.

Generated `output/benchmark_v2/`: **52 optical frames / 208 RAW captures**, divided into 36 calibration and 16 held-out optical samples. No physical artifact signature crosses splits. Both noiseless-irradiance and RAW baseline evaluation completed.

`output/convergence_suite_v2.json`: **10 of 10 final case comparisons passed the 0.01-pixel diagnostic tolerance**. This covers curved diffuse silhouettes, subpixel phase, positive/negative Z displacement, independent source seeds, mesh tessellation, and an off-center distorted/field-dependent PSF case.

| Comparison | Largest final edge change among supplied cases |
|---|---:|
| Spatial 32× → 64× | 0.00164751 px |
| Angular 32 → 128 directions | 0.00472267 px |
| Field PSF support 4 → 6 px | 0.00041498 px |
| 128-segment mesh versus analytic cylinder | 0.00224287 px |

The mesh study retained its coarse failures: a 16-segment cylinder differed from the analytic render by up to 0.09998 px and a 32-segment cylinder by 0.02733 px. The 128-segment ideal sagitta was 0.00210827 px. All levels used identical analytic contour probes.

The deliberately simple 50%-crossing/cubic-map baseline **does not achieve calibrated 0.01-pixel accuracy**. On noiseless irradiance, calibration RMS normal error improved from 0.21293 to 0.13559 px, while held-out RMS worsened from 0.17730 to 0.19021 px. RAW results show the same failure. Final reports are `output/benchmark_v2_evaluation_final.json` and `output/benchmark_v2_raw_evaluation_final.json`; they retain group-wise and tagged-condition errors, including matched optics and difficult mismatch cases. This demonstrates a working failure-sensitive benchmark; it is not a successful calibrated ESF algorithm. The baseline uses supplied contour neighborhoods/normals and known point correspondence, and includes corner-adjacent errors.

These convergence studies qualify their selected local settings, not every 16× preview benchmark frame. They also do not validate empirical transfer or physical wave/scattering accuracy. See `BENCHMARKS.md` for exact conventions and remaining limitations.

## Version 0.1: original baseline checks

Executed on 2026-09-05 with Python 3.13.13 on Linux. Dependency versions are recorded in `requirements-tested.txt` and the generated dataset manifests.

`python -m pytest -q`: **31 passed**.

Generated and visually inspected:

- `output/full_frame/`: one actual 1920×1080 pin frame.
- `output/calibration_scene/`: sphere, horizontal pin and rotated gauge block under diffuse illumination.
- `output/mesh_scene/`: imported washer with through-hole and tilted chamfered block, rendered using Embree.
- `output/edge_patch/`: three 32×32 patches from a 1080p sensor; six captures each (three exposures, two repeats).

All four manifests reached `status: complete`. The previews are display-only; use NPZ linear/RAW arrays for measurements.

## Numerical study

Using the unrandomized shape and configuration in `examples/edge_patch.json`, six of the 128 sampled contour locations have usable normal profiles fully inside the ROI. All six remained paired throughout these studies.

| Change | Maximum change in diagnostic edge position |
|---|---:|
| Spatial supersampling 64× → 128× | 0.00077651 px |
| Angular quadrature 128 → 256 directions, at 64× | 0.00012984 px |
| Gaussian PSF support radius 4 → 6 pixels, at 64× | 0 px at saved float32 image precision |

Detailed output is in `output/edge_convergence.json` and `output/joint_convergence.json`. Commands:

```bash
PYTHONPATH=src .venv/bin/python -m telecentric converge examples/edge_patch.json \
  --levels 16 32 64 128 --output output/edge_convergence.json

PYTHONPATH=src .venv/bin/python -m telecentric converge examples/edge_patch.json \
  --levels 32 64 --angular-levels 32 64 128 256 --support-levels 3 4 6 \
  --output output/joint_convergence.json
```

Use fresh output paths when rerunning. This is evidence of numerical stability for one straight-edge case, not a guarantee for all parts or optical settings. The bilinear 50% detector itself reports approximately −0.015 px bias from the analytic contour in this case; stability must not be mistaken for detector accuracy. A separate test compares rendered straight edges against a closed-form, pixel-integrated Gaussian reference at four subpixel phases, with equivalent local edge errors below 0.005 px at 128×.

Diffuse curved-geometry examples and mesh imports passed functional tests and rendered successfully, but have not received an exhaustive 0.01-pixel convergence sweep. Real-hardware comparison, coherent diffraction and reflective-sidewall validation remain outside this implementation's verified scope.
