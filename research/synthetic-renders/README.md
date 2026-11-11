# Synthetic inspection source images

This project generates centered orthographic part images with Blender Cycles and independent, 16× linear-resolution geometric masks. It implements the source-image stage in `PLAN.md`; patch extraction and training are outside this project.

## Base-part review catalog

The review workflow creates **300 reusable base geometries**, 60 in each part family, at **2048 × 2048** pixels. Each part mixes many local features: plate holes and edge details, short axial and thread regions, machined cuts, or rim details. Sampled local feature width or span is at most 48 pixels. Geometry is saved in `catalog.json` so later appearance, focus, lighting, backlight, and response randomization can reuse these exact parts. This review pass uses only Blender Workbench to create native-resolution 1-bit object masks; it does not render Cycles images or start the full dataset run.

The **turned family** spans whole-body minimum diameters from **1 to 500 pixels**, with stepped diameters reaching the original large scale. This range is sampled across the 60 turned parts; fine diameters are not just tips on large shafts. Axial and thread depths scale with body diameter so small parts remain valid solids. The gallery uses visibility-enhanced thumbnails for the thinnest bodies; opening a mask always shows the unaltered 2048 × 2048 binary image.

```bash
python3 -m synth.review generate '/media/kyle/T7 Shield/inspection-synthetic-renders/review-300-2048' --count 300 --resolution 2048
```

The command writes `masks/part_….png` (1 means object, 0 means background), `results/` per-part measurements, `details/` enlarged 96 × 96 pixel feature crops, `contact_sheets/` for each family, `index.html` for browsing and filtering, and `review_report.json`. Open `index.html` locally to inspect masks and details. The mask PNGs themselves are full-resolution binary data; contact sheets and details are only previews. An interrupted render can be resumed with `python3 -m synth.review render OUTPUT` followed by `python3 -m synth.review gallery OUTPUT`. The gallery verifies all 300 masks, including their dimensions, 1-bit format, nonempty object, foreground count, 47-pixel framing margin, and plate interior continuity.

## Runtime

- Blender **5.2.2 LTS** is pinned. The local direct binary is `/snap/blender/current/blender`; pass `--blender` for another installation of the same version.
- Python 3.12 and `Pillow==10.2.0` are needed for the driver, TIFF compression, validation, and benchmarks. Install the host dependency with `python3 -m pip install -r requirements.txt`.
- NumPy is supplied by Blender's bundled Python and is used only by the Blender worker. Data readers do not need Blender.

## Generate

Run from this directory:

```bash
python3 -m synth.cli plan --seed 20260922 --count 60 --width 512 --height 512 --output /tmp/synthetic-manifests
python3 -m synth.cli coverage --require-complete /tmp/synthetic-manifests/*.json
python3 -m synth.cli generate --seed 20260922 --count 60 --width 512 --height 512
```

The default output is `/media/kyle/T7 Shield/inspection-synthetic-renders`. Supply `--output` to use another directory. `--start` selects a deterministic scene-index range. `--samples` and `--tile-size` override the manifest's render settings; omit them to preserve a saved manifest exactly. Replay one scene with:

```bash
python3 -m synth.cli generate --manifest /path/to/scene_manifest.json --output /path/to/output
```

The measured starting configuration is 512×512, 128 Cycles samples, and 1024×1024 mask tiles. To run toward the plan's 100 GB goal, use `--target-gb 100`; with that option, `--count` is an optional per-invocation scene limit. Restart with the next `--start` index to continue a bounded batch. Completed scenes in the output directory count toward the target.

To render the reviewed 300-part catalog with the full image-formation randomization:

```bash
python3 -m synth.parallel \
  --catalog '/media/kyle/T7 Shield/inspection-synthetic-renders/review-300-2048/catalog.json' \
  --seed 20260923 --target-gb 5 --samples 128 --tile-size 2048 \
  --workers 8 --threads-per-worker 2 \
  --output '/media/kyle/T7 Shield/inspection-synthetic-renders/full-catalog-5gb-2048'
```

This writes 2048 × 2048 source images and 32768 × 32768 masks. Each mask is stored as 256 independently readable 2048 × 2048 TIFF tiles. `base_catalog.json` freezes the approved geometry in the output directory. Scene indices cycle through the five families and traverse all 60 parts in each family before repeating them with new material, lighting, focus, backlight, and response settings. After the first 300 scenes, each new appearance reuses its part's validated 16× mask instead of rendering it again. The per-scene metadata records the mask source. The parallel driver keeps separate Blender processes active so the largely single-threaded mask pass overlaps other scenes' Cycles renders. The `--target-gb` limit counts completed scene files in decimal GB and finishes in-flight scenes even if they cross the target. Repeating the command skips completed scenes and resumes from the first missing scene index; `--start` can select another index range explicitly. `progress.json` records the latest completed byte count, and `worker_logs/` contains per-scene Blender output.

Scenes generated after the September 24 appearance update carry the `expanded_v2` appearance profile in metadata. Each part now cycles through five focus classes, four lighting classes, and six material families over successive appearances. The lighting profiles span near silhouettes, strong side/front illumination, and broad ambient illumination; backlight angular spread and direction vary with them. Camera depth of field was calibrated against actual 2048-pixel edge profiles, yielding sharp through visibly blurred Cycles renders. If bright highlights would exceed the clipping limit, the response stage lowers exposure deterministically and records `response.effective_gain` alongside the sampled gain. The 300 approved geometries and their mask projections are unchanged. Earlier completed scenes are retained with their original source hashes and appearance settings; a few lower indices were still in flight when the run stopped and therefore use the new profile when resumed.

The local 5 GB run has a watchdog at `scripts/watch_full_catalog.py`. It detects whether the scheduler is running, recounts completed scene bytes if it stopped, and resumes the same command only if the target has not been reached. `scripts/full_catalog.crontab` schedules that check every three hours; `full_catalog_watchdog.log` records each check. A failed scene is retried twice before the scheduler exits, and a later watchdog check can resume from any remaining missing scene. Place `.watchdog.paused` in the output directory to hold the run during an update; remove it and run the watchdog manually to resume.

## Physical appearance profile (`physical_v3`)

`--profile physical_v3` renders the same 300 catalog parts with a physically parameterized image-formation model and runs Cycles on the GPU (OptiX, else CUDA; `SYNTH_DEVICE=CPU` forces the CPU). A 2048² scene takes about 7 s of wall time per scene with 6 workers on an RTX 3070, against about 4 minutes on 2 CPU threads, and the GPU output matches the CPU render to Monte Carlo noise. The 30 GB run:

```bash
python3 -m synth.parallel --profile physical_v3 \
  --catalog '/media/kyle/T7 Shield/inspection-synthetic-renders/review-300-2048/catalog.json' \
  --seed 20260927 --target-gb 30 --tile-size 2048 --workers 6 --threads-per-worker 3 \
  --base-masks '/media/kyle/T7 Shield/inspection-synthetic-renders/full-catalog-5gb-2048' \
  --output '/media/kyle/T7 Shield/inspection-synthetic-renders/physical-v3-30gb-2048'
```

What each scene samples (`synth/appearance_v3.py`; realized values are written back into `metadata.json`):

- **Sensor:** Cycles box pixel filter, i.e. each pixel integrates its area. 10% of scenes keep the older point sampling. 512 samples with adaptive sampling (threshold 0.003); edge-pixel Monte Carlo error is about 0.6% of the edge contrast.
- **Optics:** an object-space NA (0.004–0.06, raised so the diffraction core stays ≤ 1.2 px), a pixel size of 3.5–40 µm and a wavelength. Focus is on the part's mid-depth (the axis or equator of turned parts and spheres) within ±20 px. Then a defocus class: in focus 40%, mild 25% (0.5–2 px blur circle), medium 22% (2–5 px) or heavy 13% (5–12 px). Cycles renders the defocus with `fstop = 1 / (2 NA D)`. The in-focus lens PSF (a Gaussian diffraction core plus aberration, sometimes astigmatic), veiling glare (a kernel with tails of 15–400 px, 0.3–15% of the light) and vignetting are applied to the linear image before the camera response. All are shift-invariant, so this equals convolving the scene irradiance.
- **Pose:** 30% of parts are tilted 0.6–20° about a random in-plane axis. The tilt is reduced if the part would leave the 47 px framing margin.
- **Backlight:** a diffuse (Lambertian) or collimated panel, with angular radiance cos^n about a slightly tilted axis. It sits 150–3000 px behind the part, so Cycles defocuses its texture. Sampled texture terms, mostly weak: gradient, low-frequency noise, diffuser grain, an LED hotspot grid, smudges, and dust on the diffuser (transmissive, not in the mask).
- **Lighting:** silhouette 40%, balanced 25%, reflective 15%, ambient 20%, using the expanded_v2 ranges. Half the materials get machining-mark bump texture.
- **Contaminants:** 60% of scenes are clean, 30% have 1–4 items and 10% have 5–25. Items are particles, lint fibers, hairs and burrs, sized in µm and converted with the pixel size (sub-pixel to ~20 px wide). Attached items sit on the silhouette edge. Loose items lie near the part at the depth of the nearby edge. **All contaminants are real geometry and are part of the mask.**

Masks are exact geometry, as before. The driver chooses a mask plan per scene (`mask_plan` in metadata):

- `base`: untilted and clean. The part's mask from `--base-masks` is reused.
- `base_plus_contaminants`: only the tiles containing contaminants are rendered, then ORed into the base mask.
- `full`: tilted. Tiles outside every object's projected bounds are empty, tiles deep inside the 1× silhouette are full, and only the rest are rendered.

The drive is exFAT with 128 KiB clusters, so identical tiles are stored once. `base_masks/<part_id>/` and `base_masks/_uniform/{empty,full}.tif` live in the output directory, and a scene's `mask/index.json` may name a tile as `../../base_masks/...`. Every reader that joins `scene/mask/<file>` works unchanged; do not copy a scene directory on its own. `metadata["mask_key"]` is the part id when the mask is an untouched base mask, otherwise the scene id, so consumers can deduplicate. **Masks are no longer one per part:** tilted and contaminated scenes have their own.

`scripts/review_v3.py OUTPUT` (needs numpy and OpenCV, e.g. the edge-model venv) writes a contact sheet. Each row is a scene with zoomed crops of edges and contaminants and the 16× ground-truth outline in red. The run is watched by `scripts/watch_full_catalog.py physical_v3` (hourly in `scripts/full_catalog.crontab`). Up to `SYNTH_GPU_SLOTS` (default 3) Cycles renders share the GPU at once. A render that still runs out of GPU memory falls back to the CPU.

## Scene writing and replay

The generator writes to a temporary scene directory, validates its contents, and renames it to the final scene ID. An existing valid scene is skipped. A failed temporary directory is retained for diagnosis. The manifest contains every sampled parameter, both seeds, the exact generator source digest, and the pinned Blender version. Replay requires that generator source digest and Blender version to match.

## Output contract

Each completed scene directory contains:

| Path | Content |
| --- | --- |
| `image.png` | One-channel, lossless 16-bit unsigned grayscale Cycles image; top-left origin |
| `mask/index.json` | Full mask dimensions and independent tile coordinates |
| `mask/x….._y…..tif` | Lossless 1-bit Group 4 TIFF mask tile; 1 means target geometry |
| `metadata.json` | Versioned scene manifest, realized parameters, file names, Blender version, and render statistics |

The mask is exactly 16 times the source width and height. It excludes backlight contamination and optical blur. Tiling keeps peak mask memory bounded by the tile size; consumers can read only the regions they need. The source image has a controlled linear-to-16-bit response recorded in metadata. The camera is centered, and the target is kept at least 47 source pixels from the frame boundary; `validate` checks that bound.

The five families are plate, turned cylinder, machined cylinder, sphere, and freeform solid. The original uncataloged sampler groups adjacent scene indices around a shared geometry. The catalog sampler uses each approved part with independently sampled imaging parameters. Feature-scale, density, material, focus, and illumination strata are deliberately cycled; the coverage command checks a representative batch.

## Checks and measurements

```bash
python3 -m unittest discover -s tests -v
python3 -m synth.cli validate /path/to/output/scene_id
python3 -m synth.benchmark /path/to/output/scene_id
python3 -m synth.calibrate --indices 6 14 --size 256 --samples 16 64 256
```

The integration test checks a Cycles silhouette against the Workbench mask for a circular plate with a hole. The benchmark measures actual scene formats, sizes, and decode times. The calibration command compares sample counts against its highest-count reference on defocused and reflective cases. Measurements should be repeated on the production hardware and resolution before a large run.

See `BENCHMARKS.md` for the local measurements used to set the defaults.

## Coordinate mapping

For a square source image of height `H`, the vertical camera extent is `H` world units. The corresponding full mask dimensions are `16W × 16H`. Pixel centers use a top-left origin, and `synth.projection.pixel_center_world` defines the exact orthographic mapping. Each Workbench tile receives a shifted square camera window with the same world-space pixel centers as a hypothetical full-size mask render. Edge tiles are cropped after rendering so non-square partial tiles retain the same projection. The mask pass uses flat object color, a white background, no anti-aliasing, no shadows, and no camera depth of field.
