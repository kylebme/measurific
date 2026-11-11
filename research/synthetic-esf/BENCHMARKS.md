# Calibration, held-out evaluation and broader validation

## Generate and evaluate

```bash
cd research/synthetic-esf
PYTHONPATH=src .venv/bin/python -m telecentric benchmark examples/calibration_benchmark.json --output output/new_benchmark
PYTHONPATH=src .venv/bin/python -m telecentric evaluate output/new_benchmark --output output/new_evaluation.json
PYTHONPATH=src .venv/bin/python -m telecentric evaluate output/new_benchmark --channel raw_00 --output output/new_raw_evaluation.json
PYTHONPATH=src .venv/bin/python -m telecentric validation-suite examples/convergence_suite.json --output output/new_suite.json
```

Outputs must be new paths. The checked-in example specs and synthetic tabulated kernel can be rebuilt with `python examples/make_benchmark_specs.py`.

The supplied benchmark has **52 optical samples and 208 RAW captures**:

| Split | Samples | Contents |
|---|---:|---|
| Calibration | 36 | Two axial pin dimensions and two block dimensions at a 3×3 field grid |
| Held out | 16 | Eight groups at two unseen positions: new pin/block dimensions, diffuse sphere, horizontal pin at height, narrow gap, asymmetric corner, Airy proxy and synthetic tabulated PSF |

Each optical sample has two exposures and two repeats. Every group has one explicit split; all its poses, subpixel phases, heights and captures inherit that split. Group names are not the only check: a canonical geometry signature ignoring pose/name rejects the same exact artifact geometry in both splits. Mesh identity uses file bytes plus scale; alternative tessellations of the same CAD model must still be assigned the same group by the author. The software cannot infer common CAD provenance from unrelated files. Assemblies are fingerprinted by their constituent geometry identities as a collection.

The manifest contains resolved group membership, signatures, file paths, tags and a copy of the input spec. Samples live under `calibration/<group>/` or `held_out/<group>/`. `status` changes to `complete` only after all samples finish. Reordering groups does not change their sensor seeds. Reordering placements within a group does change sample indices and seeds; the complete spec is retained.

The example uses a **256×192 virtual sensor with 10 µm pixels** to keep execution inexpensive while demonstrating a full field. This is not the default 1080p camera. Positions in the spec are ideal full-sensor pixel coordinates, and geometry is specified in millimeters relative to each placement. Field distortion can move its geometric image away from the placement. ROI offsets and sizes are in sensor pixels. A magnification or sensor change should be treated as a separate calibrated setup with appropriate geometry and placement settings.

## What the baseline measures

`evaluate` is a **calibration harness with an intentionally simple detector**, not the requested future calibrated ESF fitter:

1. Use supplied contour neighborhoods and normals to sample a profile. This is an oracle-assisted local benchmark, not blind contour discovery.
2. Locate a bilinear 50% crossing; reject unusable, out-of-ROI or nonfinite profiles. With RAW input, saturated/clipped sampled neighborhoods are rejected and nominal radiometric scaling is applied. Fixed-pattern response is not corrected by this baseline.
3. On **calibration samples only**, fit a cubic vector-displacement field through its normal projections. Training groups have equal total least-squares weight regardless of their edge counts. The feature scale depends only on the known sensor size. No held-out labels or measurements enter fitting.
4. Apply the frozen map to both splits, saving observed coordinates, ideal and distorted geometric coordinates, corrections and normal residuals. Report bias, RMS, maximum and 95th-percentile errors by split and group, along with valid edge counts and calibration matrix rank.

The 20 features are the ten bivariate polynomial terms through degree three, multiplied by each normal component. The coefficients describe a smooth normal-displacement correction. The model does not estimate geometry-dependent ESF shape, uncertainty or a physical lens prescription; it does not measure tangential correspondence. It also assumes known calibration point correspondences, which is stronger than knowing only a pin diameter or block width at an unknown placement. A production calibration procedure must handle pose/shape estimation explicitly.

The baseline deliberately retains corner-adjacent contours and difficult held-out geometries. Their failures are useful evidence, not grounds to remove them from evaluation. Use stable regions, nuisance-parameter fitting and uncertainty/rejection logic in the actual ESF estimator. Do not tune those choices on the held-out split; create a separate development set for iterative selection.

The generated example uses 16× spatial sampling for throughput. It is a **functional benchmark**, not a numerically certified 0.01-pixel dataset. Increase sampling and qualify representative conditions before interpreting differences at that scale. Some test PSFs are deliberately outside the calibration optical distribution. Report matched-condition and mismatch groups separately; a global correction is not expected to remove every such error.

## Field variation and truth conventions

The renderer now includes:

- Brown-style radial/tangential distortion after ideal telecentric mapping. Each high-resolution sensor sample is inverse-mapped before geometry evaluation; there is no resampling warp of a finished RAW frame.
- Anisotropic Gaussian cores and asymmetric tails at a full-field PSF grid. Normalized kernels are interpolated at each output sample, before pixel integration. Patches and tiles share the same field.
- Optional tabulated PSF files at individual grid nodes. Paths are resolved relative to the job/spec, and hashes are recorded with the rendered PSF diagnostics.

Labels distinguish `xy_mm`, `ideal_xy_px` and `distorted_xy_px`; `xy_px` remains the distorted geometric location. Physical and mapped normals are separate. The evaluator's `observed`, `corrected_xy`, `error_before_px` and `error_after_px` are measured outputs, never replacements for the geometric labels.

Distortion and PSF origin are independently specified. Moving a PSF centroid and changing a geometric displacement can have indistinguishable local image effects. Calibration does not automatically identify a unique physical decomposition. Use registered kernel origins and inspect both geometric and total observed errors. The example's field model is a parametric proxy, not a traceable model of a particular C-mount lens.

## Model mismatch and empirical inputs

The supplied cases include curvature, finite-depth diffuse illumination, neighboring edges, corners, spatial anisotropy, asymmetric tails, an Airy intensity proxy and an imported radial-exponential kernel. The latter is **synthetic**, generated by a different functional family from the Gaussian core. It is not presented as a measured machine response.

For an actual measured pre-pixel 2D PSF, add a group override such as:

```json
{
  "optics": {
    "psf": "measured",
    "field_grid": [1, 1],
    "field_nodes": [],
    "measured_path": "machine_psfs/location_07.npy",
    "measured_pitch_um": 0.5,
    "support_px": 8
  },
  "kernel_provenance": {
    "kind": "empirical",
    "description": "Machine and acquisition/deconvolution procedure, field position, Z, wavelength, and calibration registration",
    "pre_pixel": true,
    "registered_origin": true
  }
}
```

This is an override fragment within a complete group; give it its own id, shapes, placements and split. The file must actually exist. The provenance declarations are required for imported benchmark kernels, but they cannot verify the measurement procedure. Arrays must be finite, nonnegative, odd-sized 2D kernels; sample pitch is object-space µm. A measured ESF/LSF containing pixel integration is not a substitute. No empirical hardware data has been supplied or validated in this workspace.

## Broader convergence studies

`convergence_suite.json` exercises ten combinations:

- Sphere under diffuse illumination: two phases × two heights.
- Horizontal pin under diffuse illumination: two heights × two independent source seeds.
- Axial pin: mesh tessellations at 16, 32, 64 and 128 segments against an analytic reference.
- Off-center curved silhouette with distortion, field anisotropy and asymmetry, including PSF-support convergence.

Each case has its own spatial/angular/support levels, contour density and ROI. All mesh levels use **the analytic contour's identical probe points**, so changing mesh contour sampling cannot masquerade as edge convergence. The mesh report also contains the independent ideal-space sagitta `R*(1-cos(pi/N))/pixel_pitch`, which catches errors even if the threshold detector happens to be insensitive. With distortion, a sampled Jacobian scale estimates its image-space magnitude.

Reports preserve every completed case, source seed, phase, height, configuration, valid/paired count and maximum/RMS edge change. A suite completes with exit code 2 if any final comparison fails; an unmeasurable ROI cannot count as a pass. This tests successive-level stability, not an absolute physical-error bound. Broader ranges, stronger defocus, different NA/magnification, source seeds and imported-mesh tolerances can need more samples than the examples.

The suite qualifies its listed local conditions. It does not automatically certify every benchmark sample, all full-frame locations, empirical interpolation, or any future estimator. Full coherent propagation, reflective sidewalls, prescription-derived field curvature and real-hardware validation remain outside this renderer's current scope.
