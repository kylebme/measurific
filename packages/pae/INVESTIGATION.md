# PAE blur and noise investigation — 10 September 2026

The default image adapter now completes all 36 original sweep images, all 36 regenerated area-corrected images, and all 16 extras in both polarities. The **0.2 px maximum full-outline goal remains unmet**. On the corrected synthetic sweep, the worst sampled error is 1.161 px, concentrated at sharp corners. Valid topology and accurate coordinates are separate acceptance criteria.

## Literature and the supplied feedback

The feedback correctly distinguishes noise compensation from generic prefiltering. The PAE paper derives a compensated 3×3 averaging mode and iterative restoration. Its local low-curvature assumptions exclude sharp corners, and its defocus discussion requires the transition to fit inside the local window. The reported averaging-filter blur experiment should not be interpreted as a guarantee for arbitrary Gaussian sigma 1–2 px. [Primary PAE paper, §§3.5, 4, 6.3, 7.2–7.3](https://accedacris.ulpgc.es/bitstream/10553/43474/1/accurate_subpixel_edge_location.pdf).

Explicit optical modelling is a distinct route: the blurred-edge model of Ye et al. includes Gaussian blur and finite pixel integration. That supports testing an optical model when PAE assumptions fail, rather than assuming extra PAE iterations repair all defocus. [Ye et al., High-accuracy edge detection with Blurred Edge Model (2005)](https://www.sciencedirect.com/science/article/abs/pii/S0262885604001908).

## Root causes fixed

1. **Iterative close-edge tests used intensity instead of derivatives.** Adding a constant to an image could change geometry despite unchanged gradients. The tests now use Gx/Gy. Constant-brightness-offset and transpose regressions expose this defect.
2. **Reconstruction sampled circles on an asymmetric grid.** Excluding the positive endpoint shifted the sample grid; the corner-only containment shortcut also missed tangent crescents. Analytic disk/rectangle area replaces that raster approximation. Stable definite integrals handle the near-straight radius-10000 case. Independent quadrature tests check areas to 2e-8.
3. **The low-noise vertical close-edge replacement started one cell late.** Its zero-based interval now agrees with the transposed horizontal case. The formerly excluded MATLAB observation has matching geometry; a candidate-index tie at fixture index 258 remains documented.
4. **Invalid low-noise fits could misalign output arrays.** All arrays are filtered together, including the all-invalid case, and zero-contrast fits are rejected.
5. **The adapter selected the ideal-image noise mode.** It now uses compensated PAE smoothing=1. The seeded API exposes smoothing without pretending that generic Gaussian prefiltering is equivalent. Noise modes currently evaluate the full detector before seed filtering and therefore cost more than the optimized ideal-image mode.
6. **Polarity could change automatic Canny thresholds.** The dark-background median admitted noise as edges on inverted images. Automatic thresholds now use a bright-background convention for either requested polarity. All 16 inverted extras pass; a noisy inversion regression compares thresholds and geometry.
7. **Raster arclength could reverse two adjacent corner observations.** A narrowly scoped swap resolves strictly crossed outer links for observations within one diagonal raster step. Coordinates and source identities are unchanged. Nonlocal crossings, gaps, and nesting changes still fail.

The original MATLAB fixtures remain. Ideal-image results retain their tight tolerances. Analytic iterative reconstruction intentionally departs from the sampled two-iteration fixture (geometry tolerance 0.01), so independent accuracy and invariance tests establish correctness rather than fixture matching alone.

## Ground-truth integrity

The previous synthetic material raster snapped continuous vertices to Pillow coordinates and used inclusive fills. This creates a subpixel coordinate bias even with nominally exact labels. Boundary cells now use exact continuous polygon/cell areas, with the pixel-center convention shared by projection. Tests cover fractional translations and supersampling factors 1–4. Discrete optical convolution still has a supersampling approximation. Existing images were preserved; the corrected sweep uses 4× supersampling.

The scoring metric now measures distances to continuous polyline segments in both directions. Arclength samples are at most 0.1 px apart, so the reported maximum can underestimate the continuous maximum by up to 0.05 px. RMS/p95 are arclength-sampled statistics. Ground truth is used only for evaluation and isolated estimator diagnostics, never by the production extractor.

## Isolated estimator evidence

`benchmarks/diagnose_edge_regimes.py` integrates analytic half-plane irradiance over square pixels. Sharp edges use an exact area formula; Gaussian edges use quadrature. It varies five angles, four subpixel phases, six blur levels, and five noise levels. Noise is Gaussian standard deviation as a fraction of the 216 DN contrast. Fixed-candidate controls separate localization from candidate selection.

The before/after table uses identical 60-trial random streams. Files: `edge-regimes-before.json` and `edge-regimes-final-paired.json`. The larger final run, `edge-regimes-final.json`, has 200 trials per condition (6,000 input images). These are controlled estimator tests, not whole-outline accuracy claims.

| Gaussian sigma | Noise / contrast | Mode | Before RMS (px) | After RMS (px) | After max (px) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 10 | 0.003975 | 0.000013 | 0.000026 |
| 0 | 0.001 | 10 | 0.017974 | 0.001161 | 0.002659 |
| 0 | 0.01 | 10 | 0.025193 | 0.010032 | 0.023998 |
| 2 | 0 | 1 | 0.119994 | 0.119994 | 0.185947 |
| 2 | 0 | 10 | 0.076403 | 0.048479 | 0.153781 |
| 2 | 0.01 | 1 | 0.126019 | 0.126019 | 0.251099 |
| 2 | 0.01 | 10 | 0.078841 | 0.051523 | 0.180488 |

Wide-blur error remains even without noise. At sigma 2 and 1% noise, the 200-trial ideal-image path has four candidate failures and 0.476 px RMS among accepted results. Compensated mode 1 has no candidate failures and 0.130 px RMS; ten iterations have no failures and 0.049 px RMS. This separates candidate instability from finite-window/model bias.

## End-to-end evidence

| Dataset / mode | Successful | Worst sampled outline error |
| --- | ---: | ---: |
| Original sweep, current default | 36/36 | 1.339 px |
| Corrected raster, current default | 36/36 | 1.161 px |
| Corrected raster, ten iterations | 36/36 | 1.719 px |
| Extras, current default (either polarity) | 16/16 | No ground truth |
| Corrected raster, experimental optical fit | 36/36 | 0.779 px |
| Corrected raster, experimental corner recovery | 36/36 | 0.312 px |

The previous report recorded 34/36 original-image successes and a 2.82 px worst outlier. Its nearest-sample metric differs slightly, so that number is historical context rather than an exactly paired geometric statistic. The regenerated raster is a separate dataset; its lower bias must not be credited entirely to extraction.

Ten iterations improve the corrected flange family to a 0.188 px sampled maximum, but worsen the sharp rack/bracket corners. Including the sampling bound, even that flange maximum does not certify a strict 0.2 px continuous bound. Raising the default iteration count is not justified.

Artifacts are under `app/demo-data/extraction-validation-20260910/`: `legacy`, `calibrated`, `iterative`, `extras`, and the explicitly experimental directories. Each contains results, extracted contours, and a contact sheet. Images remain separate from extracted contour files so GUI loading cannot bypass extraction. The new image/truth source is `app/demo-data/extraction-refined-20260910/`; its root scores now reflect the corrected default adapter.

## Remaining corner work

A global Gaussian polygon-fit experiment was removed from the default path and retained only in `packages/measurement/benchmarks/experimental_optical_fit.py`. Low image residuals did not guarantee geometric accuracy: almost zero-area spikes could be poorly observable. A total-turn penalty alone did not reliably fix convergence or the worst outliers. A corner requires evidence for both incident edges, a suitable local model, and a geometric uncertainty check. Neither interpolating missing PAE points nor accepting every small image residual meets that requirement.

The corner experiment now proposes two incident lines, then fits a Gaussian wedge
against original image pixels, integrating finite pixel area. Proposals require
straight support, an acceptable image residual, and a small conditional position
covariance. On the corrected 36-image sweep the worst sampled error falls from
0.779 to 0.312 px; the target is still missed. A geometric-only proposal achieved
0.165 px on the rack family, but did not supply a calibrated image-based
uncertainty check, and the bracket still reached 0.212 px. These are distinct
experiments, not interchangeable validation results.

The optical model succeeded on only one of the first six completed extras;
five failed convergence or image-support checks. The seventh was interrupted
once those failures established that the model was unsuitable as the default.
No result is claimed for the uncompleted inputs. Eight independent optical
arithmetic/localization tests pass, but broader initialization and model-selection
work is still required. Increasing patch size or geometric regularization alone
did not establish universal 0.2 px accuracy.

The next implementation should select among single smooth edges and two-edge
corners using original-pixel evidence, retain uncertainty/rejection reasons, and
avoid excessive tangential degrees of freedom. It must be validated on randomized
geometry, form errors, PSFs, poses, and contrast before replacing the adapter.
Current experimental scripts and cached optical fits make that work reproducible.

## Reproduction

Run from the repository root with the local packages installed in `packages/measurement/.venv`:

```bash
PYTHONPATH=packages/measurement/src:packages/measurement/benchmarks:packages/pae/src:research/synthetic-shapes/src packages/measurement/.venv/bin/python -m pytest packages/pae/tests packages/measurement/tests research/synthetic-shapes/tests packages/measurement/benchmarks/test_experimental_optical.py -q -W error
packages/measurement/.venv/bin/python packages/pae/benchmarks/diagnose_edge_regimes.py --trials 200 --output /tmp/edge-regimes.json
packages/measurement/.venv/bin/python packages/measurement/benchmarks/evaluate_existing_extraction.py --input app/demo-data/extraction-refined-20260910 --output /tmp/pae-calibrated
packages/measurement/.venv/bin/python packages/measurement/benchmarks/evaluate_existing_extraction.py --input app/demo-data/extraction-ground-truth-20260909 --output /tmp/pae-legacy
packages/measurement/.venv/bin/python packages/measurement/benchmarks/evaluate_unlabelled_extraction.py --input app/demo-data/extras --output /tmp/pae-extras
```

Validation: **181 tests passed with warnings treated as errors** (173 package tests and eight offline optical tests) on Python 3.12 (macOS). This includes all three packages, MATLAB fixtures, analytic areas, image-offset invariance, transpose consistency, seeded-mode parity, topology/source-index preservation, and renderer coverage.
