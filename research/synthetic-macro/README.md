# synthetic-macro: macro-scale multi-part scenes for coarse segmentation

Backlit scenes of 1–10 parts at **512 × 512**, 1/4 of the 2048² source renders. Masks are
at the same resolution. They are for training the coarse model that finds parts and edge
regions for the fine edge model: instance segmentation, e.g. fine-tuning RF-DETR nano or
YOLO-seg. The images are semantically simple, so the set is modest: 6,000 scenes, split
80/10/10 by scene hash, with ~5 parts per scene on average.

Output: `/media/kyle/T7 Shield/inspection-synthetic-renders/macro-v1-512`.

## Run

```bash
cd research/synthetic-macro
# render (resumable; rerun the same command to fill in missing scenes)
../synthetic-shapes/.venv/bin/python -m macro.driver \
  --output '/media/kyle/T7 Shield/inspection-synthetic-renders/macro-v1-512' --count 6000 --workers 6
# training export: RF-DETR (COCO) + YOLO-seg
../../packages/edge-model/.venv/bin/python -m macro.export '/media/kyle/T7 Shield/inspection-synthetic-renders/macro-v1-512'
# contact sheet: image | instance overlay | loose debris
../../packages/edge-model/.venv/bin/python scripts/review.py DATASET --count 12 --out sheet.jpg
# tests
../synthetic-shapes/.venv/bin/python -m unittest discover -s tests -p 'test_sample.py'
../../packages/edge-model/.venv/bin/python -m unittest tests.test_export
```

The sampler needs numpy, shapely and synthetic-shapes (its venv). Rendering uses Blender
5.2.2 and the synthetic-renders `physical_v3` code (GPU Cycles, shared GPU slots). The
export needs OpenCV. A scene takes about 3 s of wall time with 6 workers.

## Geometry

The part grammars come from `../synthetic-shapes` (`taxonomy.md`,
`expanded-part-taxonomy.md`), with its mild v2 form-error mixture. Each is made 3-D
using the synthetic-renders families:

| group | share | source | 3-D form |
| --- | --- | --- | --- |
| face_simple | 26% | washer, flange, drilled plate, slotted bracket, polygon spacer, toothed disk, cam, rack | extruded plate, 0.6–10 mm thick |
| face_expanded | 16% | 27 expanded planar grammars (perforated and aperture panels, links, brackets, combs, pulleys, …) | extruded plate |
| turned_simple | 16% | stepped, threaded, grooved and tapered shafts/pins | revolved about the axis, lying on its side |
| turned_expanded | 6% | multi-step, groove-stack, multi-thread shafts, stepped-taper arbor | revolved |
| plain | 12% | disc, ring, rectangle, polygon, triangle, rod, pin | extruded or revolved |
| machined | 8% | simple turned parts plus 1–2 cross holes, through slots or flats | revolved + booleans |
| sphere | 7% | balls, 30% with a bore | synthetic-renders `sphere` |
| freeform | 9% | lobed and saddled solids | synthetic-renders `freeform` |

Simple grammars dominate on purpose: most parts carry only a few feature types, unlike
the full-resolution renders, which pack maximum edge diversity into each part.

**Composition.**
- **Part count:** 1–10, with fewer parts more likely (15% have 1, 7% have 10).
- **Size and fill:** each part gets a size multiplier of 0.5–1.6×. One scene scale sets
  8–55% of the frame as parts, with every part at least 10 px.
- **Layout:** separated (70%), touching (20%) or stacked (10%, one part resting on
  another). Rotations are random.
- **Frame cuts:** in 15% of scenes, 1–2 parts cross the frame edge, with 30–95% of each
  inside.
- **Tilt:** 25% of free parts are tilted 1–20° out of the stage plane.

### Thin wires (scenes 6000–7199)

A thin turned part only ~2 source px thick (0.5 macro px) was invisible to the coarse
model: its majority-label mask is nearly empty, and at this scale it looks like a hair.
`--wires P` adds 1–3 wires to a fraction P of the scenes. The second batch used 0.8:

```bash
../synthetic-shapes/.venv/bin/python -m macro.driver --output DATASET --start 6000 --count 1200 --wires 0.8
```

- **Shape:** straight revolved rods lying on the stage.
  - Diameter 0.25–4 macro px (log-uniform), i.e. 1–16 source px.
  - Length 0.3–0.95 of the frame. Loose hairs are ≤ 200 px and curved.
  - 60% plain, 25% with a pointed tip, 15% with a small head.
- **Placement:** after the other parts, into free space. A wire that does not fit is
  shortened or dropped. Half may cross the frame edge.
- **The batch:** 923 wire scenes (751 train, 90 val, 82 test) and 1,513 wires, spread
  evenly over < 0.5, 0.5–1, 1–2 and ≥ 2 px.
- **Labels:** wires thinner than 2 px also label every pixel they cover by at least
  min(1/4, max(1/8, d/2)), so a sub-pixel wire is a continuous 1–2 px line. Other parts
  keep majority labels, so their edges stay unbiased.
- **Mask supersampling:** 8× in wire scenes. At 4× a 0.25 px wire is one subsample wide.
- **Unchanged:** `--wires 0` (the default) reproduces scenes 0–5999 exactly.

## Appearance

The same physical model as `synthetic-renders` `physical_v3`, with every pixel-unit
quantity divided by 4:
- **Pixel size:** the macro pixel size (µm) follows from the scene scale.
- **Optics:** NA and diffraction core, depth of field and defocus classes, lens PSF,
  glare and vignetting.
- **Backlight:** diffuse or collimated panel with gradient, noise, grain, LED grid,
  smudges and diffuser dust.
- **Lighting:** silhouette, balanced, reflective or ambient.
- **Materials and sampling:** six material families, shared by all parts in 60% of
  scenes, with bump texture. Box pixel filter (10% point-sampled), 512 adaptive samples.

## Debris

Kept rare so the model does not confuse debris with small parts. Sizes are physical
(particles 20–400 µm, lint 8–30 µm × 0.5–8 mm, hair 40–120 µm × 2–15 mm), so most debris
is only a few pixels wide.

- **Attached** (8% of untilted, unstacked parts; 1–2 particles, burrs or fibers on the
  silhouette edge): **part of that part's mask**.
- **Loose** (30% of scenes; 1–3 items on the stage in free space, may cross parts):
  **in no part mask**. Stored separately in `debris.png`.

## Files per scene (`scenes/<scene_id>/`)

| file | content |
| --- | --- |
| `image.png` | 16-bit linear-response render (the physical_v3 response curve), 512² |
| `instances.png` | uint16 instance ids, 0 background, `k` = `parts[k-1]`. Each pixel takes the majority label of a 4×-supersampled flat render (modal: visible pixels only) |
| `debris.png` | 8-bit coverage of loose debris (0–255) |
| `metadata.json` | every sampled and realized parameter: parts (family, group, kind, pose, tilt, stacking, frame cut, material, attached debris, mask area and bbox), loose debris, optics, backlight, lighting, response, render timing |

`macro.export` writes 8-bit camera-simulated images and labels for training:
- **Camera simulation:** seeded exposure, Poisson-Gaussian noise, and 10% JPEG.
- **Labels:** one class, `part` (`--classes group` gives the 8 geometry groups instead).
  Holes are bridged into the polygon.
- **Layout:** `export/images/{train,valid,test}` with `_annotations.coco.json` (RF-DETR
  `dataset_dir=export/images`), plus `export/labels` and `export/data.yaml` for YOLO-seg.
- **Check:** re-rasterized polygons match the instance masks at IoU 0.9998 mean.
