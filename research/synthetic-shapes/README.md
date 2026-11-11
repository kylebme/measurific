# Synthetic inspection shapes

Current contour-model training uses four base geometry classes and four compound
classes. See [the primitive training contract](../../packages/contour-model/PRIMITIVES.md).
Keyhole panels are removed; thread templates are triangular, trapezoidal, and square.
The rendered-image pipeline documented below retains its legacy construction labels.

Corpus v3 ([DATASET_V3.md](../../packages/contour-model/DATASET_V3.md)) adds `realism.py`
(tangent fillets, tool radii, edge breaks on any line/arc corner), `turned.py`
(turned silhouettes with half-pitch thread phase, runout, grooves, one-sided
keyways) and `csg.py` (grammar-free CSG/sketch-style parts with exact carrier
labels, tangent-arc cams, lobed profiles and round-groove pulleys).


A Python pipeline for training **semantic feature recognition** on planar backlit
silhouettes. Geometry stays in millimetres until rendering. It retains ideal and
imperfect contours, edge/primitive/feature ownership, defects, relationships, and
both contour and image segmentation targets. Default image size: **1920 × 1080**.

The output location for this project is
`/media/T7Shield/inspection-synthetic-shapes`. See `validation.json`, `manifest.json`
and `contact-sheet.png` there for the generated dataset and its validation results.

## Install and run

```bash
cd research/synthetic-shapes
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/inspection-shapes generate \
  --output /media/T7Shield/inspection-synthetic-shapes/cpu-run \
  --count 1200 --shard-size 100 --workers 4
.venv/bin/inspection-shapes validate /media/T7Shield/inspection-synthetic-shapes/cpu-run
.venv/bin/inspection-shapes preview /media/T7Shield/inspection-synthetic-shapes/cpu-run \
  /media/T7Shield/inspection-synthetic-shapes/cpu-run/contact-sheet.png --limit 24
.venv/bin/python -m pytest -q
```

All implementation is Python; numerical/raster operations use NumPy, SciPy,
Shapely and Pillow. No Blender, CAD application, compiled custom extension, model,
or network service is needed. The generator is CPU-only, with no CUDA code or dependencies. It uses a process
pool for parallel rendering. The initial delivered run finished with the earlier
renderer before GPU support was removed at the user's request; its metadata records
that provenance. Use a new output directory (such as `cpu-run` above) for future
CPU datasets because the source/configuration checksum intentionally prevents
mixing generator versions in an existing run. Existing contours and masks remain
fully readable. Writing outside a sandbox's allowed filesystem roots may still
require filesystem authorization, independently of rendering.

## Scale and reproducibility

`--count 1000000` creates one million samples without collecting the dataset in
memory. Tar shards contain at most `--shard-size` samples, and at most twice the
worker count is submitted to the process pool. The pilot used approximately
1.54 MB/sample (~1.54 TB per million; content/noise changes compression). Allow
additional space for the last temporary shard and filesystem overhead. Tar itself
is uncompressed; PNG and NPZ entries are compressed and individually readable.

Resume by repeating the same command. Completed shards are SHA-256 checked and
skipped; an unfinished shard is regenerated deterministically. Increase `--count`
to extend the dataset **at a complete shard boundary**. Changing seed, dimensions,
views, shard size, or generator source requires a new dataset directory;
worker count can change. Reducing count is rejected. Do not modify generator code
while a job is running. A `.writer.lock` prevents concurrent writers; after a hard
process kill, confirm no generator is active before removing a stale lock.

Each design has an independent geometry seed. Appearance seeds depend on design
and view. `--views-per-design 4` creates four views/optical realizations of the same
imperfect part. All views of a design receive the same split, selected by a stable
80/10/10 hash. This prevents same-design augmentation leakage. Families are evenly
cycled at the design level; splits are probabilistic and are **not** a held-out
part-family benchmark. With the default one view, 1,200 images are 1,200 designs.

`--contours-only` skips rasterization and produces compact contour datasets with
all geometric labels. Camera coordinates are still supplied. A run is reproducible
for fixed source, dependencies and seed; cross-version bitwise
identity is not promised. The config records a generator source hash, and the
delivery includes the current CPU generator source and exact environment versions.
The preserved initial-run configuration hash identifies the earlier renderer; the
CPU source snapshot is a successor, not a byte-identical copy of that renderer.

## Part grammar and taxonomy

See [taxonomy.md](taxonomy.md) for the hierarchy above the original
[starter taxonomy](starter-shape-taxonomy.md). Twelve implemented part families:

- Turned side views: stepped shafts, threaded shafts, grooved pins, tapered pins.
- End views: washers and drilled flanges.
- Prismatic face views: drilled plates, slotted/tabbed/notched brackets, polygon spacers.
- Cut profiles: toothed disks, racks, harmonic cams.

Continuous dimensions, orientation, corner style, hole count/layout, slot size,
groove shape, thread/tooth count/profile, form severity and optical parameters vary.
Generic thread templates are V, trapezoidal and square. Grooves
can be rectangular, V, trapezoidal or dovetail. Thread and tooth templates are
**generic planar profiles**, with no standards designation or conformance claim.
These twelve grammars are an initial coverage set, not exhaustive manufacturing
coverage. They do not generate hidden pockets, certified involute gears, internal
threads, certified thread templates, U-groove arcs, or arbitrary imported CAD yet.
Adding a family means supplying a closed, oriented `Chain` with labeled features.

## Geometry pipeline

1. Sample a part grammar with analytic line/arc definitions and nominal dimensions.
2. Sample primitive chains at ~0.055 mm arc-length spacing (dense sampled freeform
   profiles for cams and teeth). Preserve primitive IDs and junction positions.
3. Apply affine shear, a bilinear nonparallelism field, low-frequency normal form harmonics, per-primitive bow and
   skew, bore/pattern offsets, and smooth longitudinal pitch warp. Circles become
   noncircular, nominally paired lines lose their ideal relationship, and repeated
   profiles acquire nonuniform pitch. Features keep their nominal semantic class.
4. Add correlated edge roughness and finite-support burr/nick normal excursions.
   Positive excursions always add material, including at hole boundaries. Local
   normal perturbations fade over 0.18 mm around sharp junctions to prevent folded
   corners; global shear and pitch fields still move those junctions.
5. Reject invalid polygons/holes rather than repairing geometry and losing labels.
   Rejections are counted; failure after 50 attempts aborts the sample/run visibly.
6. Apply pose and forward radial/tangential distortion to the imperfect contours.
7. Rasterize material at 2× linear resolution, apply a symmetric Gaussian PSF,
   integrate square pixels, then apply illumination gradient/vignetting, Poisson
   shot noise, PRNU, read noise, clipping and 8-bit quantization.

Material rasterization now integrates the continuous polygon's area in each
subpixel cell. Interior filling remains fast; boundary cells use exact geometric
clipping with the same integer-pixel-center convention as projection. This
removes Pillow's integer vertex snapping and inclusive-fill bias, which were
significant when evaluating errors below one pixel. The discrete Gaussian PSF
still converges with supersampling; the extraction accuracy sweep uses 4×.
Previously generated images retain their original renderer and must not be
treated as an unbiased subpixel calibration reference. Source hashes prevent
mixing old and new renderers when resuming a dataset.

Nominal/actual vectors share indices exactly. Analytic primitives describe the
**nominal** geometry; imperfect geometry is a polyline, never incorrectly relabeled
as an exact line or circle. Harmonic amplitudes, edge errors, offsets and anomaly
parameters are recorded. Burr heights are requested and realized normal offsets,
not computed GD&T measurements. Roughness is a contour perturbation, not Ra/Rz.
Severity is sampled as 0 (8%), 0.4 (27%), 1 (45%), or 2.5 (20%); these are domain
randomization settings, not calibrated manufacturing process distributions.

## Files and labels

Each tar member begins `train/000000000.`, `val/...` or `test/...`.
No pickle is used. Label 0 means unlabeled/background. Global class IDs are in
`taxonomy.json`; feature/primitive/anomaly instance IDs are local to a sample.

| Suffix | Meaning |
|---|---|
| `json` | Part family/view, feature hierarchy, analytic nominal primitives, camera, seeds, imperfections, patterns and relationships |
| `contours.npz` | Ideal/actual vertices in mm and pixels; ring offsets/roles; edge feature, primitive, semantic bits, anomaly class/instance |
| `image.png` | 8-bit grayscale optical simulation |
| `material.png` | Binary material occupancy; holes and outside are 0 |
| `region_semantic.png` / `region_instance.png` | Filled outer part and hole/slot regions, including the empty space in a hole/slot |
| `boundary_semantic.png` / `boundary_instance.png` | Nearest-feature band extending 3 pixels on either side of the raster contour |
| `boundary_primitive.png` | Primitive instance IDs in the same boundary band |
| `anomaly.png` / `anomaly_instance.png` | Separate burr/nick class and instance labels in the boundary band |
| `semantic_bits.npz` | `bits`: uint32 multilabel image combining region, boundary and all feature ancestors |

**Contour convention:** vertices are not duplicated to close rings. Edge `i` runs
from vertex `i` to the next vertex *within that ring*, wrapping at its end. Ring
ranges are `[ring_offsets[j], ring_offsets[j+1])`. Outer rings are CCW in millimetres,
holes are CW; camera y is flipped. mm coordinates have x right/y up; image
coordinates have x right/y down and integer pixel centers. Use `actual_xy_mm` for
physical contour training and `actual_xy_px` for distorted image-domain contours.
Do not compute physical measurements from distorted pixel coordinates directly.

**Overlapping semantics:** a thread edge belongs to its thread and external-profile
ancestor. `edge_feature_id` plus `features[].parent` stores full instance ancestry;
`edge_semantic_bits` stores all semantic classes. In images, bit `1 << class_id`
indicates membership. Region and boundary masks are separate so a thread can coexist
with the external region. Open features such as chamfers, shoulders and thread
flanks have boundary bands, not arbitrary filled wedges. Filled circles/slots
include the hole interior even though it is background in the intensity image.
Boundary conflicts choose the nearest rasterized edge (SciPy EDT tie ordering);
they are not full overlapping instance masks. Adjacent very small features may
vanish in raster labels; their vector labels remain. The exact hierarchy is always
available from contour edges. This is explicit label policy, not optical ground
truth uncertainty.

The binary material target uses supersampled coverage threshold 0.5; filled region
labels use native polygon filling, and boundary labels use a native one-pixel
polyline dilated by nearest-edge distance. Those conventions can differ at an edge
by about a pixel. Masks describe latent geometry and are not blurred/noised. This
renderer targets semantic training, not subpixel measurement certification.

## Reading for training

```python
from inspection_shapes.dataset import iter_samples

for meta, arrays in iter_samples('/media/T7Shield/inspection-synthetic-shapes', split='train'):
    contour = arrays['contours']
    points = contour['actual_xy_mm']
    edge_targets = contour['edge_feature_id']
    if 'image' in arrays:
        image = arrays['image']
        bits = arrays['semantic_bits']['bits']
        thread_target = (bits & (1 << 10)) != 0
```

The reader streams one sample at a time. A training application can shuffle shard
order and use a bounded shuffle buffer; large-scale distributed training should
assign different shards to each rank/worker to avoid duplicate samples. NumPy
point arrays retain variable length. Resample by arc length *within labeled edges*
if a model needs fixed-size point inputs; do not interpolate integer instance IDs.

## Validation and limits

Tests exercise all family grammars, topology/winding, zero-imperfection identity,
nonzero circle/line form error, image polarity and hole labels, label hierarchy,
projection consistency, repeated views, split isolation, resume and corrupt shards.
`validate` checks every stored sample, shard checksum, contour topology, projection,
image dimensions, ID consistency, semantic bits and foreground/background intensity.
Visual checks include a 24-sample contact sheet and magnified thread/defect/mask
panels. Validation does not establish that a model trained only on synthetic data
will transfer to a particular inspection station. Real captures should be used
for a separate evaluation set and to tune these distributions.

No dimensional fitting/tolerance evaluator or training model is included: the
pipeline supplies semantic targets and geometric truth for those next stages.
Measurements are suggestions for classical downstream processing, not evaluated
values or pass/fail GD&T labels. In particular, circularity applies to a visible
closed circular profile, not to the full roundness of a shaft inferred from its
side silhouette; lead, helix, runout and other 3D characteristics are not inferred.

## Render saved contours later

```python
from PIL import Image
from inspection_shapes.dataset import iter_samples, part_from_record
from inspection_shapes.render import Camera, render

meta, arrays = next(iter_samples('/media/T7Shield/inspection-synthetic-shapes'))
part = part_from_record(meta, arrays['contours'])
camera = Camera(**meta['camera'])
camera.sigma_px = 1.2  # change optics without changing physical geometry
images, projected_contours, timings = render(part, camera, seed=meta['appearance_seed'])
Image.fromarray(images['image']).save('rerendered.png')
```

This works with contour-only datasets as well as records with existing images.
Use the returned masks together with the new image if camera geometry changes.
