# Agent 1 Plan: Blender/Cycles Synthetic Source-Image Generator

## Review gate agreed 2026-09-23

Before the full Cycles dataset, create a **300-part, 2048 × 2048** geometry catalog: 60 base parts in each of the five families. Each part mixes local feature types and scales, with individual feature spans no greater than 48 source pixels. Save the realized geometry parameters for reuse in later image-formation randomization. For this review gate, render **one 1× binary Workbench object mask per part**, plus contact sheets and enlarged local feature crops. Generate the full photometric dataset and its final mask specification after the parts have been reviewed. The original production plan below remains the target for that later stage; its 16× mask contract does not apply to these review masks.

The turned-family review parts now sample **whole-body diameter** across the full range: minimum section diameters from 1 to 500 pixels, with stepped large sections retained. The smallest sizes are complete slender parts rather than fine pins attached to otherwise large shafts.

## 1. Mission

Build a robust, reproducible synthetic-data generator in Blender using Cycles. The generator will create large grayscale source images of synthetic manufactured parts together with much higher-resolution binary geometric masks.

The purpose of this dataset is **not** to reproduce one particular camera, lens, material, or machine-vision installation. The purpose is to span a broad and physically plausible range of image formation so that behaviors encountered in real systems are represented somewhere in the synthetic distribution.

The goal should be to create on the order of 100GB of images and masks.

This plan covers **Agent 1 only**. Patch extraction, 48×48 training-sample generation, runtime augmentation, and model training belong to a separate downstream project.

### Primary outputs

For every generated source scene, produce:

- A high-resolution **single-channel 16-bit grayscale intensity image** rendered from an orthographic camera.
- A corresponding **binary object mask at 16× the linear image resolution** (256× as many pixels), aligned to the image coordinate system.
- Metadata sufficient to understand and reproduce the scene and its sampled domain-randomization parameters.

---

## 2. Core requirements and non-goals

### Requirements

1. Use **Blender with Cycles** for the photometric source-image render.
2. Use an **orthographic camera** for this dataset generation run.
3. Do not render the expensive Cycles image at the 16× mask resolution.
4. Ground truth must represent **projected object geometry**, not a thresholded version of the rendered grayscale image.
5. Include broad diversity in:
   - part geometry;
   - feature type, scale, density, and curvature;
   - material response;
   - backlight properties, including degree of collimation;
   - weak spill and ambient illumination;
   - backlight nonuniformity and contamination;
   - camera depth of field and focus conditions;
   - pose and framing where useful.
6. Support deterministic reproduction of individual generated scenes.
7. Validate dataset coverage and edge behavior rather than relying on visual variety alone.
8. Keep the output contract independent of Blender so downstream consumers do not need Blender installed.

### Non-goals

- Diffraction, interference, or coherent wave optics.
- A digital twin of a specific physical camera or lens.
- Perspective-camera data in this first run.
- Training-patch generation.
- Heavy per-patch sensor augmentation; that belongs downstream.
- Maximizing photorealistic appearance for its own sake.

---

## 3. Technical baseline

Use a maintained Blender LTS release and pin the exact patch version used for production once the pipeline has been validated. As of September 2026, Blender 5.2 LTS is the current maintained LTS line and 5.2.2 is the current patch release.

Relevant Blender 5.2 capabilities:

- Cycles is Blender's physically based path tracer.
- Blender supports orthographic cameras.
- Blender supports background/headless rendering from the command line.
- Cycles supports camera depth of field, including for orthographic cameras. In this project, intentional optical blur should be produced by the Cycles camera model rather than by a compositor or post-render blur stage.
- Workbench supports final rendering with **No Anti-Aliasing**, making it a reasonable first choice for a cheap high-resolution geometric silhouette pass.
- Cycles' Object Index pass is non-antialiased, but it is Cycles-only in Blender 5.2 and therefore is not the preferred way to create a 16× mask when the objective is to avoid a high-resolution Cycles render.

The implementation should not depend unnecessarily on undocumented Blender behavior. Any Blender-specific assumption that affects ground-truth geometry or image/mask registration should have a regression test.

---

## 4. Design principles

### 4.1 Favor physically plausible diversity over arbitrary randomness

Domain randomization should represent families of plausible imaging systems and parts. Avoid treating every renderer parameter as an independent random variable simply because it is available.

Correlated parameters are appropriate when the correlation reflects a plausible physical system. The distribution should include both ordinary and deliberately difficult conditions without becoming dominated by implausible combinations.

### 4.2 Parameterize geometry in image-space terms where practical

The downstream model ultimately sees pixels. Feature generation should therefore make it easy to specify and stratify feature scale in projected source-image units in addition to Blender world units.

The dataset should span:

- subpixel and near-pixel features;
- fine features several pixels wide;
- features on the order of the eventual 48-pixel patch size;
- very-low-spatial-frequency geometry whose characteristic scale is much larger than 48 pixels.

Large features do **not** need to fit inside a 48×48 patch. A small crop from a very large-radius curve or slowly varying surface is exactly the desired representation of low-spatial-frequency geometry.

### 4.3 Separate geometry ground truth from image formation

The binary mask should describe the ideal projected occupancy of the target object under the reference orthographic projection. It should not include the Cycles camera's depth-of-field blur, lighting gradients, material highlights, noise, or thresholding artifacts.

This separation allows the rendered image to have ambiguous or displaced apparent edges while retaining an accurate geometric target.

### 4.4 Prefer coverage-aware sampling

Pure independent uniform randomization is unlikely to provide good coverage of important rare features. The generator should explicitly track and test coverage over feature classes and meaningful parameter ranges.

The exact sampling strategy is an implementation choice. Stratification, quotas, low-discrepancy sampling, targeted rare-case generation, or mixtures of these are all acceptable if validation shows good coverage.

### 4.5 Reproducibility should survive code evolution where practical

Use deterministic randomization, but do not rely on a single seed as the only reproduction record. Store the realized scene parameters and relevant generator/version information in metadata.

---

## 5. Output contract

All output files should be put in /media/kyle/T7\ Shield/inspection-synthetic-renders

The exact on-disk formats should be selected after a small benchmark for compression ratio, decode cost, random access, and implementation complexity. Do not bake format choices into the architecture prematurely. Do not use lossy compression at all.

Every source scene must nevertheless expose the following logical data.

### 5.1 Grayscale source image

- One channel.
- 16-bit unsigned intensity for production output.
- Fixed image dimensions for a dataset configuration, but configurable between experiments.
- Generated by Cycles with the scene's camera depth of field included in the render, followed only by the explicitly controlled intensity-response and quantization stage required for storage.
- Lossless storage.

The source-image resolution should be chosen by experiment. It should be large enough to contain many diverse local regions and large-scale geometry, while keeping dataset generation practical.

### 5.2 Binary mask

- Exactly 16× the source image's width and 16× its height.
- Strictly binary after generation.
- No anti-aliased edge values.
- Same reference projection and framing as the source image.
- Represents target-object geometry only.

Because these masks are very large, the implementation should support efficient generation and storage without assuming that an entire mask must always reside in memory uncompressed. Tiled, chunked, bit-packed, or otherwise region-readable representations are reasonable options and should be benchmarked rather than assumed.

### 5.3 Scene metadata

Record enough information to reproduce and analyze each source image. At minimum include:

- scene identifier;
- generator version / source revision;
- Blender version;
- master seed and/or deterministic scene identifier;
- realized camera and projection parameters;
- part family;
- realized geometric parameters;
- feature instances and relevant feature classifications;
- material parameters;
- backlight parameters;
- spill / ambient parameters;
- camera depth-of-field parameters, including focus distance and aperture settings;
- intensity-response / quantization parameters;
- source-image and mask dimensions;
- paths or identifiers for associated data.

Metadata should be schema-versioned.

---

## 6. Part-family architecture

Keep the number of base part generators small. Diversity should primarily come from procedural geometry, feature composition, and reusable modifiers/decorators rather than a large catalog of hand-authored parts.

A good initial decomposition is five base families.

### 6.1 Plate

One plate family should cover both external-outline and internal-feature diversity rather than splitting every feature type into a new base part.

Support a broad range of plate thicknesses, including stepped or locally varying thickness where appropriate.

#### Internal features

Include, in varied combinations and densities:

- round and elliptical holes;
- polygonal holes;
- filleted and unfilleted polygons;
- spline/freeform openings;
- tooth-like internal profiles;
- slots and capsule-like slots;
- straight and curved slots;
- closely spaced features.

#### Outline features

Include:

- acute, right, and obtuse corners;
- sharp and filleted corners;
- straight segments;
- circular and arbitrary spline curves;
- large-radius / low-frequency curves;
- repeating triangular teeth;
- repeating square teeth;
- irregular or jittered repeating patterns;
- notches and protrusions;
- local edge damage;
- burrs, fibers, or hair-like attached geometry;
- near-single-pixel and subpixel projected features.

Feature combinations should range from extremely sparse to extremely dense and should sometimes mix several spatial scales on the same part.

### 6.2 Turned / revolved cylindrical part

Cover primarily axisymmetric manufactured geometry, including:

- plain cylinders;
- stepped diameters;
- shoulders and necks;
- grooves;
- chamfers;
- sharp and radiused transitions;
- threaded regions.

Thread profiles should include at least:

- triangular / V;
- round;
- trapezoidal;
- square.

Vary pitch, depth, crest/root geometry, phase, number of visible turns, and local damage or imperfection. Clean unthreaded regions are equally important and should remain well represented.

### 6.3 Machined cylindrical part

Use a second cylindrical family for non-axisymmetric features that are awkward to express as a simple revolved profile, such as:

- side holes and cross-drilled holes;
- oblique holes;
- slots;
- flats;
- keyway-like features;
- local cutouts;
- combinations with steps, grooves, or threading.

This family is intended to create genuinely three-dimensional projected-edge behavior rather than merely another texture variation of the turned-cylinder family.

### 6.4 Spherical / rounded part

Include:

- complete or partial spherical forms;
- through-holes or bores where geometrically meaningful;
- off-axis holes;
- flats;
- dimples or local cuts;
- transitions between spherical and simpler surfaces.

### 6.5 Freeform 3-D part

Provide geometry that is not easily classified as a plate, cylinder, or sphere. Useful forms include:

- smooth arbitrary surfaces;
- compound curvature;
- saddles;
- ridges and valleys;
- blended primitives;
- swept profiles;
- faceted surfaces;
- mixtures of smooth and sharp transitions.

The purpose of this family is not to mimic one manufacturing process. It is to prevent the training distribution from encoding overly strong assumptions about canonical part families.

---

## 7. Cross-cutting geometry features

Implement reusable feature systems where that reduces duplication across base part families.

Keep **object-attached debris** distinct from **backlight contamination**:

- A burr or hair physically attached to the target part is object geometry and belongs in the binary mask.
- Dust, fibers, stains, or defects on the backlight/background affect the grayscale image but do not belong in the object mask.

---

## 8. Geometry quality and tessellation

The 16× mask makes projected geometric precision important. Procedural curves and curved surfaces must be tessellated finely enough that mesh faceting is not mistaken for intentional edge structure.

Do not hard-code an arbitrary subdivision count for every object. Establish a projected-image-space error criterion and validate it on representative curves, circles, cylinders, spheres, and freeform surfaces.

Geometry generators should also be stress-tested for invalid booleans, self-intersections, non-manifold results where they are unintended, degenerate faces, and unstable behavior at extreme but supported parameter values.

---

## 9. Camera and framing

Use an orthographic camera for the entire first-generation dataset.

The camera system should provide a well-tested conversion among:

- Blender/world coordinates;
- source-image coordinates;
- 16× mask coordinates.

The projected scale should be explicit and recorded in metadata.

The photometric Cycles camera should use Blender's native depth-of-field controls. Focus distance and aperture are part of the domain-randomized camera state. Orthographic projection keeps geometric magnification independent of object depth, while Cycles DOF provides the desired depth-dependent loss of sharpness within the render.

Do not randomize part placement in the frame. The part should be centered with 47px of backlight boundary surrounding it. The consumer of these outputs will create random patches from these master images.

---

## 10. Backlight and scene illumination

The intended visual regime is predominantly backlit, with strongly but variably collimated illumination, plus weaker spill and ambient illumination.

### 10.1 Visible backlight

Use a luminous background whose intensity can vary spatially. Randomization should include small, plausible amounts of:

- broad gradients;
- slow nonuniformity;
- subtle texture;
- sparse dust/debris/contamination;
- localized imperfections.

Backlight texture should usually be weak enough that it remains a machine-vision backlight rather than becoming a dominant texture-generation mechanism.

Lighting should not be so bright that it causes clipping. Many parts of the image will be pure black or nearly pure black, but no areas should be pure white (except individual pixels).

### 10.2 Collimation

The dataset should span a useful range from very strongly directional illumination to moderately less-collimated backlighting.

In Blender, directional/Sun illumination is a reasonable building block because the Sun source is directional and has an angular-diameter control. More complex angular profiles may be approximated with multiple weighted directional contributions or another validated emitter construction.

Do not assume that the collimation parameter by itself will broaden the direct silhouette of an opaque object under an ideal orthographic camera. In this model it primarily changes illumination of visible surfaces, sidewalls, reflections, and spill. The generator should be designed around the actual effects produced by the chosen geometric-optics model rather than forcing every randomized parameter to visibly alter every edge.

### 10.3 Spill and ambient light

Add weaker illumination components capable of revealing front surfaces, sidewalls, curved surfaces, and reflective features. Randomize their strength and direction over a realistic range.

Some scenes should remain very close to pure silhouette imaging; others should contain enough spill/ambient light to produce useful geometry-dependent edge behavior.

---

## 11. Materials

Use a modest number of physically interpretable material families with randomized parameters rather than unconstrained random node graphs.

Useful coverage includes:

- dark matte materials;
- rough metals;
- smoother or polished metals;
- coated or semi-gloss surfaces;
- dark polymers;
- lighter / more reflective surfaces.

Vary surface roughness, reflectivity, metallic response, and small-scale appearance within plausible ranges. Where useful, allow exposed cut edges or sidewalls to differ from primary faces.

The dataset should include both highly absorbing silhouettes and objects whose surfaces meaningfully return spill or ambient light to the camera.

---

## 12. Cycles rendering and image formation

### 12.1 Render in a controlled linear pipeline

Treat the Cycles result, including camera depth of field, as the rendered image rather than as a display-ready image. Do not use a compositor or external image-space blur to create the optical edge spread.

Avoid uncontrolled view transforms or artistic post-processing in the dataset path. After rendering, limit processing to the explicitly defined intensity response and conversion needed for the stored 16-bit grayscale representation.

### 12.2 Sampling quality

Do not prescribe one sample count for all scenes. Establish a representative convergence test and choose settings that make Monte Carlo render noise small relative to the intentional image variations. Include strongly defocused and reflective validation scenes, since aperture sampling and difficult light paths may converge differently from simple silhouettes.

Denoising should be disabled by default for the source dataset unless experiments show that a particular denoising configuration improves throughput without introducing unacceptable changes to edge profiles or fine features.

### 12.3 Pixel filtering

Cycles' pixel reconstruction filter also affects edge softness. Keep it explicit and fixed within a dataset configuration, and choose it through validation so that it does not become an uncontrolled source of edge variation. A minimal-filter configuration is worth evaluating, but the plan does not require a particular filter or width.

### 12.4 Camera depth of field

Intentional optical blur must be generated as part of the Cycles render using the camera's depth-of-field model, not with Blender's compositor and not with a post-render blur kernel. This keeps blur coupled to actual scene depth and geometry, which is particularly important for thick plates, cylinders, spheres, stepped surfaces, and arbitrary 3-D parts.

Randomize focus distance and aperture over ranges that produce useful variation from nearly sharp to meaningfully defocused images. Aperture shape parameters may also be varied when they provide useful coverage, but do not add complexity solely for visual variety.

Because the camera is orthographic in this run, treat Blender's orthographic DOF controls as parameters of the synthetic imaging model rather than assuming that their numerical f-stop values correspond directly to a particular real telecentric lens. Validate the resulting edge behavior empirically.

Depth placement and focus should be sampled coherently. The generator should include cases where the nominal object boundary is close to the focus plane, cases where the object is modestly out of focus, and cases where different depths within the same 3-D part experience different amounts of defocus.

Record the realized focus distance and aperture settings in metadata.

### 12.5 Intensity response and 16-bit quantization

Apply a controlled mapping from scene-linear rendered intensity to the stored 16-bit grayscale image. Include enough variation to represent plausible camera response while avoiding aggressive per-scene processing that obscures geometric information.

The mapping must be deterministic for a generated scene and recorded sufficiently for analysis.

Heavy stochastic sensor noise is not a focus of Agent 1; the downstream patch pipeline can generate many noise realizations from the same source scene more efficiently.

---

## 13. High-resolution binary mask generation

The 16× mask should be generated through a cheap geometry-only path, independent of the Cycles source-image render.

### Recommended first implementation

Use a flat Workbench silhouette render with:

- No Anti-Aliasing;
- a deterministic foreground/background separation;
- no lighting-dependent effects;
- no shadows, cavity, outlines, or similar display effects that can alter the silhouette;
- camera depth of field disabled for the mask pass.

Convert the result to a strict binary representation before storage. The mask is geometric ground truth and must not inherit the photometric render's defocus.

This is a recommendation, not an immutable architectural requirement. If Workbench introduces platform-specific problems, coordinate inconsistencies, or undesirable limits at the required mask sizes, replace it with another geometry-only projection/rasterization method while preserving the same output contract.

---

## 14. Domain-randomization strategy

Randomization should operate on several largely independent conceptual axes while preserving plausible relationships within each axis. Here are some suggested but not required approaches to the domain randomization activity.

### Geometry

Vary:

- base part family;
- dimensions and thickness;
- feature classes;
- feature scale;
- curvature;
- corner angles;
- feature density;
- feature repetition and irregularity;
- damage/debris level;
- pose and framing.

### Appearance

Vary:

- material family;
- reflectivity / roughness;
- sidewall behavior;
- subtle surface variation.

### Illumination

Vary:

- backlight intensity;
- backlight nonuniformity;
- collimation / angular spread;
- alignment;
- spill;
- ambient/front/side illumination;
- background contamination.

### Image formation

Vary:

- focus distance and depth-of-field strength;
- aperture shape/aspect parameters where useful;
- intensity response;
- exposure / level mapping within reasonable bounds.

### Distribution design

Ensure the generated corpus includes:

- sparse and dense geometry;
- simple and compound boundaries;
- mixed spatial scales;
- extremely low-frequency edges;
- near-pixel and subpixel features;
- very clean imaging;
- difficult low-contrast or reflective cases;
- cases where different part families produce locally similar edge evidence.

Allow the same or closely related geometry to be rendered under multiple appearance/illumination configurations. This reduces the risk that a geometric feature becomes spuriously associated with one synthetic material or lighting pattern.


## Definition of done

This project is complete when all of the following are true:

1. A clean environment can run the generator headlessly with the pinned production Blender version.
2. Cycles produces the intended orthographic 16-bit grayscale source images.
3. Every production image has a correctly aligned 16× binary geometric mask produced without a 16× Cycles render.
4. The five base part families are implemented with the required feature diversity.
5. Feature scale covers subpixel through very-low-spatial-frequency geometry substantially larger than the eventual 48×48 training patch.
6. Materials, backlight texture, backlight collimation, spill, ambient light, camera focus, and depth of field are all represented in the domain-randomization system.
7. The randomization system demonstrably covers important feature and imaging strata rather than merely producing visually varied images.
8. Scene metadata is schema-versioned and sufficient for reproducibility and analysis.
9. Production resolution, Cycles quality settings, mask storage strategy, and other performance-sensitive choices have been selected from benchmarks rather than arbitrary defaults.
10. Downstream code can consume images, masks, and metadata without importing Blender or depending on Blender-specific scene structures.
