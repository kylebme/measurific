# Synthetic telecentric silhouettes

A Python renderer for developing and validating subpixel silhouette measurements. It supports analytic calibration solids, imported triangle meshes, 1080p frames, sensor-aligned patches, finite-angle or diffuse illumination, fixed focus, optical blur proxies, and a linear monochrome sensor. Results include independent geometry labels, linear irradiance, integer RAW frames, and reproducibility metadata.

Version 0.2 adds geometric distortion, smoothly varying field PSFs, an artifact-grouped calibration/held-out benchmark, and convergence suites spanning phase, height, source seeds and mesh tolerance. See [BENCHMARKS.md](BENCHMARKS.md) for the new workflow and interpretation.

[THRESHOLD_ACCURACY.md](THRESHOLD_ACCURACY.md) separates distortion from optical/sampling/threshold errors, compares edge versus dimension accuracy, and reports independent phase/curvature controls and a simple ERF-fit comparison.

The target is studying **0.01-pixel errors**. This is a numerical accuracy target to check for each setup, **not a blanket physical-accuracy guarantee**. Supersampling, angular quadrature, PSF truncation, and mesh tessellation each need a separate error budget.

## Run

Python 3.11 or newer:

```bash
cd research/synthetic-esf
uv venv .venv
uv pip install --python .venv/bin/python '.[test,fast]'
.venv/bin/telecentric generate examples/edge_patch.json --output output/my_patches
.venv/bin/telecentric generate examples/calibration_scene.json --output output/my_scene
.venv/bin/telecentric generate examples/full_frame.json --output output/my_full_frame
.venv/bin/python -m pytest -q
```

Calibration and validation workflows:

```bash
.venv/bin/telecentric benchmark examples/calibration_benchmark.json --output output/my_benchmark
.venv/bin/telecentric evaluate output/my_benchmark --output output/my_evaluation.json
.venv/bin/telecentric validation-suite examples/convergence_suite.json --output output/my_suite.json
```

The project virtual environment is already installed in this workspace. For running directly from edited source, use `PYTHONPATH=src .venv/bin/python -m telecentric ...`. The `fast` extra installs Embree; without it, Trimesh uses its slower triangle/rtree backend. No GPU, Blender, or lens prescription is required. Outputs must go to an empty directory to avoid overwriting existing data.

`requirements-tested.txt` records the tested Linux/Python 3.13 environment. Other supported Python versions may need different dependency versions from the portable ranges in `pyproject.toml`.

`examples/calibration_scene.json` is a small whole-scene preview. `examples/full_frame.json` actually renders 1920×1080, with low sampling for a quick smoke test. `examples/edge_patch.json` retains the 1080p camera and renders a 32×32 ROI near a gauge-block edge. Preview settings are deliberately inexpensive; increase sampling and run convergence before using them to rank close measurement algorithms.

Mesh example, including a through-hole and a tilted chamfered block:

```bash
.venv/bin/python examples/make_mesh_job.py
.venv/bin/telecentric generate examples/generated_mesh/job.json --output output/my_meshes
```

The Python API works without a job file:

```python
from telecentric import Camera, Illumination, RenderConfig, Primitive, Scene, render

camera = Camera(width=1920, height=1080, magnification=0.5)
config = RenderConfig(camera=camera, illumination=Illumination(kind="diffuse"))
scene = Scene([Primitive("sphere", radius_mm=0.5, center_mm=(0, 0, 0.5))])
result = render(scene, config, roi=(1016, 524, 32, 32))
linear_pixels = result.irradiance
truth = scene.labels(camera)
```

## Physical model

The camera is object-space telecentric, in air, with optional smooth image distortion. Each output sample is inverse-mapped through distortion to `(x, y, focus_z_mm)` in object space. For each accepted angular sample, the renderer traces a parallel line through that point and checks opaque-part visibility:

```text
analytic solid / triangle mesh
             ↓
visibility through fixed-focus telecentric pupil
             ↓  weighted angular radiance integration
geometric transmission
             ↓  optional field-dependent pre-pixel intensity PSF
optical irradiance on a supersampled grid
             ↓  pixel-area quadrature
linear sensor irradiance
             ↓  exposure, PRNU/DSNU, shot/read noise, full well, ADC
RAW uint16 + independent geometry labels
```

Rays are anchored at the **focus plane**, not an arbitrary shadow screen. Out-of-focus geometry therefore changes which accepted rays are blocked. No focus/aperture movement is simulated. A single depth-dependent Gaussian is not applied afterward: that would double-count the geometrical defocus already present in the pupil integration.

For `diffuse`, the source has constant radiance over the forward hemisphere. The accepted transverse direction cosines are uniformly sampled in the circular object-space pupil using scrambled Sobol quadrature. This implements the radiometric measure `cos(theta) dOmega`. `cone` restricts the source angle; `gaussian` weights angular radiance within the specified cone; `collimated` uses the axial direction. All sources are coaxial and spatially uniform before the optional field gradient. The model assumes a backlight large enough to fill these directions across the field.

**Angular weights are normalized after pupil acceptance.** `bright_e_per_ms` is the unobstructed *detected* electron rate, so widening the source does not automatically change exposure brightness. This separates edge-shape studies from throughput. Changing NA or magnification requires setting the physically appropriate detected rate yourself.

### Optics choices

| `optics.psf` | Purpose |
|---|---|
| `none` | Pure geometric pupil/visibility reference |
| `gaussian` | Controlled symmetric blur baseline |
| `mixture` | Gaussian core plus shifted Gaussian tail for estimator mismatch |
| `airy` | Monochromatic incoherent circular-pupil intensity PSF proxy |
| `measured` | Imported nonnegative, pre-pixel, odd-sized 2D `.npy` intensity kernel |

The wavelength default is 0.63 µm (red). Only `airy` directly uses wavelength; Gaussian/mixture/measured PSFs have their own supplied scale. For a measured PSF, supply `measured_path`, `measured_pitch_um` in **object space**, and enough `support_px`. The center array element defines zero displacement; the renderer preserves asymmetry and reports the kernel centroid. Linear interpolation resamples the kernel, so its sampling also needs validation. Metadata records the source hash and estimated retained energy before finite-support renormalization.

The intensity PSF can be constant or an **output-position-dependent local approximation**. A narrowly collimated source can have appreciable spatial coherence; correct diffraction then involves amplitude propagation and source-dependent imaging, not simply convolution by an incoherent PSF. This renderer does not claim to solve that wave-optics problem. It also omits reflective sidewalls, scattering, translucency, polarization, and prescription-derived pupil aberrations/vignetting. Field-dependent blur and asymmetry are now configurable proxies, not derived from a lens prescription.

A measured ESF/LSF is **not** automatically a 2D optical PSF. It usually already contains pixel integration and possibly geometrical defocus. Feeding it back as a pre-pixel PSF would double-count those effects; only import a kernel with a known measurement/deconvolution convention. Optiland/HCIPy-generated intensity PSFs can use the same file interface, but there is no lens-prescription adapter or full coherent propagation backend in this version.

## Camera and coordinates

Default active sensor: **1920×1080 at 3.45 µm**, or 6.624×3.726 mm. This is an explicit plausible small-sensor choice, not a conversion of the ambiguous “1/2-inch” optical-format label. C-mount alone does not imply telecentricity; use these settings to represent a telecentric lens with that mount.

Object-space pitch is `pixel_pitch_um / magnification`:

| Magnification | Object pitch | Horizontal FOV at 1920 px |
|---|---|---|
| 0.05× | 69 µm | 132.48 mm |
| 0.5× | 6.9 µm | 13.248 mm |
| 2× | 1.725 µm | 3.312 mm |

`object_na` is independent and must match the actual lens configuration. A sweep of magnifications represents separate setups, not a physically modeled zoom lens.

- All geometry distances are **mm**. Sensor pitch and wavelength have explicit µm fields.
- Pixel boundaries are integers; pixel centers are `(column+0.5, row+0.5)`.
- World `x` points right, `y` down, and ray `z` forward. The sensor center maps to `center_xy_mm`.
- `roi` is `[x, y, width, height]` in full-sensor pixels. Labels remain in full-sensor coordinates; subtract the ROI origin for patch coordinates.
- Tiles include the complete PSF halo and evaluate geometry outside the sensor field. This prevents dark padding or tile seams from corrupting border edges.

`camera.distortion_radial = [k1,k2,k3]` and `distortion_tangential = [p1,p2]` add a Brown-style polynomial displacement to the ideal telecentric image. Coordinates are normalized by **half the larger sensor dimension**, centered at the sensor midpoint; these are not coefficients normalized by an OpenCV focal length. Defaults are zero. The analytic Jacobian transforms contour normals; Newton inversion checks residuals to 1e-10 pixels and rejects singular/folded samples. A field grid check also rejects obviously folded configurations. The model is intended for mild lens distortion, not arbitrary warps.

`optics.field_grid = [columns,rows]` and `field_nodes` specify local PSF overrides on a uniform grid spanning the full sensor, in row-major order. For example:

```json
{
  "psf": "gaussian",
  "field_grid": [2, 2],
  "field_nodes": [
    {"sigma_px": 0.5},
    {"sigma_px": 0.9, "sigma_y_px": 0.5, "angle_deg": 30},
    {"psf": "mixture", "tail_offset_px": [-0.8, 0.2]},
    {"psf": "mixture", "tail_offset_px": [0.8, 0.2]}
  ]
}
```

The renderer blends **normalized kernels**, not fitted parameter values, using bilinear weights at each high-resolution output sample. The operator is `I(u)=sum_k w_k(u)*(T*h_k)(u)`. Weights sum to one, preserving a constant background, and are anchored to the full sensor rather than tile/ROI boundaries. This defines an output-indexed local blur model; it does not assert energy-conserving point-source propagation. Kernel derivatives can change at interpolation grid lines. Grid density is a model choice, and empirical interpolation should be validated at held-out locations. `sigma_y_px` and `angle_deg` support anisotropy; tail offsets are expressed in sensor axes. Measured kernels can also be supplied at field nodes, with independent paths and sample pitches.

## Geometry and ground truth

`sphere`, `box`, and finite `cylinder` use float64 analytic line intersections. Cylinders can lie along `y` or `z`; `angle_deg` rotates shapes about `z`. Boxes and horizontal cylinders have exact rectangular axial silhouettes, while spheres and axial cylinders have exact circular ones. These primitives avoid tessellation error for calibration references.

`mesh` uses Trimesh with Embree when available, otherwise the triangle backend. It accepts mesh formats supported by Trimesh, including STL/OBJ/PLY. Explicit `scale_to_mm`, `translation_mm`, and Euler `rotation_deg` are supported. Euler rotations are extrinsic x/y/z (`sxyz`). STEP/CAD should first be tessellated with a controlled tolerance. Mesh hashes, pose, backend, triangle count and watertightness are saved. Open surfaces are treated as opaque triangle sheets; the renderer does not repair malformed CAD.

Labels contain exact samples on each primitive's axial silhouette or samples along the polygonal union of projected mesh triangles, including holes. `xy_mm` is physical geometry, `ideal_xy_px` is its undistorted telecentric projection, and `distorted_xy_px` (also `xy_px` for compatibility) is the geometric image before blur. `normal_world_xy` is the physical normal, while `normal_xy` is the Jacobian-transformed image normal. `curvature_per_mm` remains physical projected curvature; it is not curvature of the distorted image. Labels also record in-frame status and whether another shape occludes that boundary. Per-object geometry remains available when shapes overlap. Curvature is zero on straight mesh segments; corners are excluded because normal/curvature is undefined there. Circular contours are *sampled* at exact positions; connecting those samples with straight lines introduces an additional approximation.

Mesh labels are exact relative to the floating-point input mesh, **not the original CAD curve**. For a tessellated radius `R` with `N` segments, radial sagitta is `R*(1-cos(pi/N))`. Budget this in object-space pixel units; a visually smooth mesh is not sufficient evidence of 0.01-pixel fidelity. Embree's internal precision is another reason to use analytic primitives for reference tests.

`center_mask` is distorted axial geometric occupancy at pixel centers. `geometric_coverage` is numerically integrated distorted axial occupancy; it is not an exact continuous ground-truth contour. Optical thresholds never define the geometry labels. Observed edge positions belong to estimator output; the benchmark evaluator saves them separately from both geometric coordinate systems. The renderer does not assign a unique “tangent Z” to every boundary: box sides and finite-cylinder end rims can have multiple contributing depths. Physical shape parameters and z bounds are retained instead.

## Datasets and sensor

A job JSON has `config`, `shapes`, optional `roi`, and optional `dataset`. Dataclass fields in `src/telecentric/config.py` define all configurable parameters. Unknown keys are rejected.

Dataset options:

```json
{
  "count": 20,
  "seed": 2026,
  "random_phase": true,
  "z_range_mm": [-0.5, 0.5],
  "magnifications": [0.05, 0.5, 2.0],
  "exposures_ms": [0.25, 1.0, 4.0],
  "repeat_captures": 3
}
```

One subpixel translation in `[-0.5,0.5)` per axis and one Z displacement are applied to the entire scene per sample. Magnifications cycle deterministically. Geometry and capture random streams are separated; increasing the requested count preserves earlier samples. PRNU/DSNU remain fixed on the full sensor across the dataset and exposure repeats. Exposure changes only the sensor capture, not the noiseless optical frame.

The sensor draws Poisson counts **before** full-well saturation, clips accumulated charge, adds Gaussian read noise, converts using `electrons_per_adu`, adds black level, clips and rounds to the configured ADC range. `noise=false` disables shot/read randomness, while retaining fixed-pattern response and ADC quantization. Use `irradiance` for truly sensor-free data. Noise draws are repeatable for the same ROI and seed; RAW noise is not coordinate-addressed across independently rendered ROIs.

Each sample writes:

- `.npz`: `irradiance`, `geometric_coverage`, `center_mask`; `raw_00`, `expected_e_00`, `saturated_00`, `clipped_low_00`, etc. for each capture. RAW arrays are uint16 with values restricted to the configured bit depth.
- `.json`: resolved configuration, seeds/exposures, ROI, phase/Z translations, PSF diagnostics, geometry labels and asset hashes.
- `.png`: an 8-bit **display preview**, unsuitable as the metrology input.

`manifest.json` records the input job, library versions and completed samples. Its status is `incomplete` until all samples finish. Files are listed only after each sample finishes; interrupted runs are inspectable but automatic resume is not implemented.

## Verify the 0.01-pixel target

Run numerical studies on small ROIs spanning the relevant edges:

```bash
.venv/bin/telecentric converge examples/edge_patch.json \
  --levels 16 32 64 128 \
  --output output/my_spatial_convergence.json

.venv/bin/telecentric converge examples/edge_patch.json \
  --levels 32 64 --angular-levels 32 64 128 256 --support-levels 3 4 6 \
  --output output/my_joint_convergence.json
```

The diagnostic samples along independently known contour normals, locates a bilinearly interpolated 50% crossing, and compares successive numerical resolutions. It reports maximum and RMS displacement, paired/valid counts, changes in crossing validity, image changes, and baseline threshold bias relative to the geometric contour. A nonconverged or unmeasurable last comparison exits with code 2.

Passing means **stability of this diagnostic on the sampled visible contours**, not proof that every edge, phase or estimator is within tolerance. Baseline threshold bias includes real modeled optical bias plus detector interpolation error. Repeat across phases, Z, curvature, source seeds, angular sample counts, PSF support and mesh tolerances. Different numerical errors may consume the same budget, so use tighter component tolerances than 0.01 px for final validation. Blank ROIs cannot pass by having zero detected edges.

Spatial block averaging is midpoint quadrature, not exact area integration. A straight binary edge on an S× grid can retain phase errors of order `1/(2S)` pixels. 16×/32× previews therefore cannot guarantee 0.01-pixel accuracy; 64×/128× patch studies and convergence checks are provided for that reason. Full-resolution high-sampling diffuse scenes can be expensive on a CPU: work scales roughly with pixels × S² × angular samples. Memory is bounded by tile size and ray chunks rather than the full supersampled frame.

Tests include an independent closed-form pixel-integrated Gaussian edge reference across phases; analytic angled-cylinder support; pupil/radiance integration; defocus behavior; full-frame/ROI/tile equality; analytic-versus-mesh visibility; holes; PSF normalization; Poisson/read-noise statistics; saturation order; and reproducible multi-exposure output.

The additional suite in `examples/convergence_suite.json` crosses curved silhouettes and diffuse illumination with multiple phases/heights, tests independent source seeds, compares mesh tessellations to an analytic cylinder using identical contour probes, and exercises distortion/field asymmetry. Reports save each completed case incrementally, retain failures, and exit with code 2 if any final comparison fails. Suite settings are editable; passing the supplied cases does not qualify every configuration in the benchmark or all of parameter space.

## Design references

- [Trimesh ray API](https://trimesh.org/ray.html) and [Embree adapter](https://trimesh.org/trimesh.ray.ray_pyembree.html): mesh visibility.
- [SciPy FFT convolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html): finite-support optical convolution.
- [Edmund Optics telecentric illumination](https://www.edmundoptics.com/knowledge-center/application-notes/imaging/telecentric-illumination-why-you-need-it-in-machine-vision-applications/): why source angular distribution and NA matching matter in silhouette measurement.
- [Optical image formation](https://bigwww.epfl.ch/preprints/vonesch0601p.pdf): illumination/collection apertures and wave-optical image formation.
- [OpenCV distortion model](https://docs.opencv.org/5.0.0-alpha/dc/dbb/tutorial_py_calibration.html): radial/tangential polynomial convention; this package applies it after a telecentric projection with explicit sensor normalization.

The starter conversation is retained unchanged in `starter-conversation-subpixel.md`.
