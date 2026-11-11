# What does the 50% detector cost in accuracy?

The answer depends strongly on the feature being measured. The previous pooled edge RMS combined lens distortion, corner behavior, optical response and interpolation; it was not a measure of the error introduced solely by choosing a 50% detector.

The current detector uses known dark/bright levels, supplied contour neighborhoods/normals, bilinear image interpolation and a 50% crossing. This analysis is conditional on correct contour/ROI selection, not a blind detector benchmark.

## Separate the components

For each true point, the analysis retains:

- Ideal geometric coordinate `q`.
- Distorted geometric coordinate `D(q)`.
- Observed 50% coordinate `m`.

In a common normal direction, `m-q = (D(q)-q) + (m-D(q))`. These **signed errors add; their RMS values do not generally add or subtract** because they can be correlated or cancel. To measure the residual independently of geometric distortion, the analysis also evaluates `D^-1(m)-q` using the simulator's exact inverse map. This is an oracle ablation, not evidence that an actual fitted calibration has zero residual.

The residual after inverse distortion includes optical/illumination bias, pixel-area integration, detector interpolation, and finite rendering error. It cannot all be attributed to the 50% choice alone. Independent analytic controls below isolate interpolation and curvature without binary rasterization.

## Point accuracy versus dimensional accuracy

On the existing 52-sample benchmark, noiseless **individual edge** RMS errors were:

| Geometry/selection | Distortion-only normal RMS | Remaining RMS after exact inverse distortion |
|---|---:|---:|
| Calibration block, small, stable side regions | 0.16679 px | 0.02802 px |
| Calibration block, large, stable side regions | 0.16558 px | 0.03138 px |
| Held-out rotated block, stable sides | 0.03615 px | 0.02494 px |
| Held-out circular pin, radius 7.3 px | 0.03450 px | 0.04529 px |
| Held-out asymmetric block, stable sides | 0.03545 px | 0.10977 px |

Stable sides exclude points within 3 ideal pixels of a projected corner. This is a geometry-defined analysis selection, not a learned rejection rule. Including corner-adjacent block points gives roughly 0.19–0.25 px residual RMS and individual errors around a pixel. Those are not representative of a properly placed straight-edge measurement ROI.

**A dimension is not one edge coordinate.** Common lateral shifts can cancel between opposite edges; radial shrinkage adds on opposing sides of a diameter. Averaging many edge points reduces some interpolation/noise effects, especially when a tilted contour spans many pixel phases. An aligned edge can retain the same phase bias along its entire length.

The following are representative fixed samples re-rendered at **128× spatial sampling**, with the original optical settings and exact inverse distortion. Rectangular dimensions use means of opposite stable sides at the supplied orientation. Circular diameters use a nonlinear circle fit with free center and radius.

| Measurement | Error from distortion alone | Residual dimension error after exact inverse distortion |
|---|---:|---:|
| Rotated block width | +0.03559 px | +0.00200 px |
| Rotated block height | +0.01333 px | +0.00326 px |
| Side-view pin diameter | +0.03260 px | −0.00240 px |
| Same pin's axial length | +0.01329 px | +0.13583 px |
| End-view circular pin diameter, 14.6 px wide | +0.02313 px | −0.07630 px |
| Asymmetric block width, away from corners | +0.03460 px | −0.00120 px |
| Asymmetric block height, away from corners | +0.01469 px | −0.00505 px |

The columns are separate diagnostic measurements, not numbers to sum as an exact nonlinear dimensional decomposition. For example, distortion expansion and threshold-induced circle shrinkage partly canceled before correction. Removing distortion can therefore expose a larger residual error even when the geometric correction is perfect.

The 64×→128× dimension changes were below 0.00027 px for the rotated block, circular pin and horizontal pin. The asymmetric block's height changed by 0.00290 px, so its last digits are not established to finer precision. These are spatial refinements; source quadrature and PSF support remained fixed.

A selected axis-aligned calibration block is less favorable: its width error was +0.0468 px at 16×, +0.0461 px at 64×, and +0.0384 px at 128×. Its last spatial change was 0.0077 px, so describe this as **about 0.04 px**, not a high-precision 0.0384-pixel result. The existence of good rotated-width examples does not establish a universal 0.01-pixel bound for aligned widths.

Other preview benchmark residuals after exact inverse distortion include approximately −0.074 px for the diffuse sphere's diameter, −0.111 px for the tabulated-kernel circular diameter, and +0.022 to +0.081 px for narrow-bar widths separated by a small gap. These were analyzed at the original 16× settings and are not independently qualified to the last digit.

## Interpolation alone: independent analytic control

For an isolated straight edge and a symmetric Gaussian PSF, the continuous, unit-pixel-integrated 50% point is exactly the true edge. There is no intrinsic midpoint bias. However, interpolating between discrete sensor samples introduces phase-dependent error.

The control uses the exact Gaussian-CDF pixel integral, 201 evenly spaced subpixel phases, known normalized plateaus, no noise and no distortion:

| Optical Gaussian sigma | Linear 50% RMS | Linear 50% worst phase | Cubic-spline 50% worst phase |
|---|---:|---:|---:|
| 0.4 px | 0.04013 px | 0.05599 px | 0.04211 px |
| 0.6 px | 0.02426 px | 0.03383 px | 0.01863 px |
| 0.8 px | 0.01537 px | 0.02143 px | 0.00771 px |
| 1.0 px | 0.01040 px | 0.01451 px | 0.00330 px |
| 1.5 px | 0.00489 px | 0.00682 px | 0.00059 px |

PCHIP gave little improvement over linear interpolation in this control. A pixel-integrated ERF fit recovers the constructed Gaussian position to numerical precision, as expected for matched-model data; this is a numerical check, not proof of robustness on arbitrary parts.

Width errors also depend on the relative phases of two edges. For identical isolated straight Gaussian edges at sigma 0.6 px, a noise-free linear-threshold calculation gives worst-phase width errors of essentially zero at exactly 17 pixels width, 0.03141 px at 17.1 pixels, 0.05889 px at 17.25 pixels, and 0.06537 px at 17.5 pixels. An integer-width example can benefit from cancellation that does not transfer to other dimensions or local magnifications.

Higher-order interpolation can improve a simple 50% detector without a calibrated ESF library, but the table does not establish its performance on noise, asymmetry, corners or arbitrary orientation. Increasing blur also changes curvature bias and noise sensitivity; it is not a free accuracy improvement.

## Optical curvature bias exists even with perfect crossing interpolation

For an opaque disk blurred by an isotropic Gaussian, the optical transmission can be evaluated as a noncentral-chi-square survival probability. Integrating that response over the unit square pixel with 8×8 and 12×12 Gauss-Legendre quadrature gives an independent continuous pixel-response reference, with no mesh or binary supersampling.

At sigma **0.6 pixel**, the axis-to-axis continuous 50% diameter errors are:

| Circle radius | Continuous diameter bias, including pixel integration but excluding interpolation/noise |
|---|---:|
| 7 px | −0.06358 px |
| 20 px | −0.02218 px |
| 50 px | −0.00887 px |
| 100 px | −0.00443 px |
| 250 px | −0.00177 px |

The two quadrature orders agreed within 3e-14 px in these controls. This checks quadrature stability, not floating-point statistical-library accuracy to that many digits. Curvature bias roughly scales with blur variance divided by radius. At sigma 1.0 pixel, a 100-pixel radius still gives about −0.01083 pixel diameter bias.

The benchmark's circular pins have radii only 6–9 pixels. Their errors should not be extrapolated directly to pins hundreds of pixels across. Also distinguish a pin's circular end-view silhouette from the nearly straight sides of its side-view silhouette.

## Does an ordinary ERF fit help?

An additional local fit uses native pixel values rather than an interpolated profile, with free edge position, Gaussian sigma, dark level and contrast. It integrates a straight Gaussian step over each square pixel using 8×8 Gauss-Legendre quadrature. The fitter does not receive the renderer's PSF parameters. It is evaluated only on straight regions at least 4 pixels from corners, using the same supplied normals/neighborhoods as the threshold diagnostic.

| Matched comparison | Linear threshold RMS | Uncalibrated integrated ERF RMS |
|---|---:|---:|
| Rotated block: individual stable edges | 0.02475 px | 0.00484 px |
| Rotated block: width/height errors | 0.00315 px | 0.00263 px |
| Large calibration block: individual stable edges | 0.03072 px | 0.01475 px |
| Small calibration block: width/height errors | 0.01491 px | 0.01887 px |

A better point estimator therefore does not automatically give a better dimension. The fits used the original 16× benchmark images; marginal improvements below that dataset's numerical uncertainty are not established. In the narrow-gap group, 72 of 144 attempted fits were rejected by convergence/bound checks, so successful-fit scores do not describe all its edges. All other compared groups accepted all attempted fits. Fit coverage is recorded explicitly.

No fully calibrated ESF model was implemented or evaluated in this analysis. Its necessity or benefit cannot be inferred from the success of a matched Gaussian control.

## Noise, physical units and practical decision

The dimension table above describes noiseless systematic error. Paired RAW repeats gave rough single-capture dimensional scatter of order 0.003–0.01 px. The dataset has only two repeats per exposure, with as few as two paired diameter measurements in a held-out group, so these are indicative rather than a repeatability qualification. PRNU is persistent; paired repeats do not reveal its static bias.

This benchmark uses 10 µm/object pixel: 0.01 px is 0.10 µm and 0.08 px is 0.80 µm. For the default 3.45 µm sensor at 0.5×, object pitch is 6.9 µm and those conversions would be 0.069 µm and 0.552 µm. Changing magnification also changes geometry/blur in pixel units; these conversions alone do not predict another setup's error.

The evidence supports keeping the simple detector as a serious baseline for stable width measurements, with geometric calibration and appropriate global line/circle fitting. It does **not** support claiming universal 0.01-pixel accuracy from a linearly interpolated 50% crossing. Before building a large calibrated ESF system, compare modest interpolation or uncalibrated integrated-fit improvements against feature-level errors across the actual part sizes, subpixel phases, orientations and Z range. Reserve calibrated optical/geometry corrections for residual systematic errors that exceed that application's dimensional budget.

## Reproduction and artifacts

```bash
cd research/synthetic-esf
PYTHONPATH=src .venv/bin/python -m telecentric.accuracy output/benchmark_v2 --output output/new_threshold_accuracy.json
PYTHONPATH=src .venv/bin/python -m telecentric.accuracy output/benchmark_v2 --refine --output output/new_threshold_refinement.json
.venv/bin/python -m pytest -q
```

- `output/threshold_accuracy_final.json`: 34,560 edge observations, 954 dimensional measurements, successful/rejected fit counts, paired-repeat estimates and independent analytic controls.
- `output/threshold_refinement.json`: fixed representative samples at 16×, 64× and 128×.
- `output/threshold_axis_aligned_refinement.json`: selected aligned-block case; reproduced by `refine_representatives(..., groups=['cal_block_large'], sample_indices={'cal_block_large': 3})`.
- `src/telecentric/accuracy.py`: reusable analysis implementation.

**52 tests passed**, including exact straight-step recovery, phase bias without distortion, cancellation of common shifts in dimensions, doubling of radial edge bias in diameter, and geometry-defined corner exclusion. Existing renderer/benchmark tests remain passing. Real measured ESFs, scattering and coherent propagation remain outside this simulation's verified scope.

Numerical library references: [SciPy PCHIP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html), [SciPy noncentral chi-square distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncx2.html).
