# Coarse model: macro-scale part segmentation

The coarse model finds parts and their edge regions in a whole frame so the fine model
(`cfi_B_seed`, README.md) knows where to work. It replaces the classical seed's object
mask (adapter.py-style background fit + Otsu), which fails when parts are about as bright
as the backlight (ambient and reflective lighting, light or polished parts).

## Role and interface (final model)

- **Input:** the whole frame, area-resampled to **512 × 512**, whatever its size and
  aspect ratio. A 2048² frame is downsampled 4×; a 2448 × 2048 or 1920 × 1080 frame is
  squashed. Model pixel (j, k) covers source pixels [k·W/512, (k+1)·W/512) ×
  [j·H/512, (j+1)·H/512).
- **Input channels** (`coarse.coarse_inputs`, `in_mode="raw"`, `aux_inputs=True`):
  - the reading ×2 − 1 (in [0, 1] of full scale);
  - the **region of interest** (+1 inside, −1 outside; all +1 when there is none). The
    reading is zeroed outside a given region.
  - log2(W / H) of the original frame.
- **Aspect ratio and ROI are known inputs** (user, 2026-09-29).
  - Circular backlights make the region of interest a circle in the frame, which is an
    ellipse in the squashed model frame.
  - Training squashes frames from aspect 1.25–1.9 either way (50% of samples; upsampling
    capped at 2× per axis).
  - In 35% of samples it adds a circular ROI: inscribed, or larger and cut by the frame.
    Outside is zero (software crop) or a dark unlit backlight with a soft edge. In 20% of
    those the mask is withheld, so an unlit border alone is not taken for a part.
  - Parts are cut at the ROI boundary: nothing outside it is a part.
- **Outputs** (at 512 × 512):
  - `part` logit: any part, including attached debris; loose debris is excluded. This is
    the outline the fine model needs.
  - `sdf`: signed distance to the part boundary, clipped to ±6 model px.
  - `contact` logit: part pixels within 2 px of another part, for instance splitting.
    That is secondary; touching parts may merge.
- **For the fine model** (`coarse_infer.CoarseModel`):
  - `full_res_logit`: the part logit bilinearly resized back to the frame.
  - `band(width_px)`: frame pixels within width_px of the coarse boundary, or wherever
    the upsampled logit is uncertain.
  - A boundary error of 1 model px is W/512 source px, 4 px for a 2048-px-wide frame.
  - `coarse_maps` precomputes the part logit and sdf for every scene of the fine-model
    datasets.
- **Rejected input variant:** normalizing by the classical backlight field (`in_mode="field"`)
  fails on large parts (Findings 3). The field is fitted to tile p90 levels.

Code:
- `edgesr/coarse.py`: model, inputs, targets, instances.
- `edgesr/coarse_infer.py`: inference API, export.
- `edgesr/coarse_maps.py`: fine-model data.
- `edgesr/macro.py`: data cache.
- `edgesr/train_coarse.py`, `edgesr/bench_coarse.py`.
- `edgesr/review_coarse.py`, `edgesr/sweep_coarse.py`.
- `edgesr/coarse_onnx.py`: ONNX export, CPU latency.
- `edgesr/yolo_baseline.py`, `edgesr/compare_coarse.py`.

Tests: `tests/test_coarse.py`.

## Data: macro-v1-512

`/media/kyle/T7 Shield/inspection-synthetic-renders/macro-v1-512` (`../../research/synthetic-macro`):
6,000 scenes at 512², 1–10 parts each, 29k part instances. The split is the dataset's
own, by scene hash: train 4,806, val 599, test 595. Parts are sampled per scene, so no
geometry is shared across splits. `python -m edgesr.macro build` packs images, instance
maps and loose-debris maps into `work/cache/macro_v1` (6.3 GB memmaps).

Training draws from the 16-bit renders with on-the-fly augmentation:
- geometry: dihedral flips, scale 0.75–1.4 (area/bilinear resampling, center-aligned
  nearest for labels);
- photometric: backlight shading, gamma, exposure 0.35–0.95 of full scale;
- sensor: Poisson-Gaussian noise with full well 2k–1M e⁻ per macro pixel, 8/10/12-bit
  quantization, 10% JPEG at 8-bit.

The field is fitted on the full augmented frame, then a 384² crop is taken.

## Benchmark: macro_val_export

Every model sees the same pixels: the 8-bit images from `macro.export` (the images the
YOLO baseline trains on). These carry Poisson-Gaussian noise at 6k–60k e⁻ per macro
pixel and 10% JPEG. That is noisier than the pipeline's input, which is a 4× average of a
full-resolution reading, so the benchmark is conservative. All 599 val scenes are scored;
test is untouched until a final confirmation. Distances are in macro px (×4 for source
px).

- **Boundary:** sampled densely on a 4× grid.
  - GT: 0.5 crossings of the majority-label mask after Gaussian σ 0.5 px.
  - Prediction: zero crossings of the bilinearly upsampled part logit.
  - `bscore` = ½(mean min(d_gt→pred, 2) + mean min(d_pred→gt, 2)), lower is better.
  - Floor: ~0.04 for an exact copy of the majority labels, ~0.15 for a perfect sub-pixel
    edge. The GT staircase is ±0.5 px.
  - `recall@t` is band coverage: `recall@1` corresponds to a ±4 source-px band.
- **Parts:**
  - Missed: a GT part with less than 50% of its pixels predicted.
  - False: a predicted component of at least 16 px² (256 source px²) that is less than
    50% GT part.
- **Instances:** matched at IoU > 0.5; F1, SQ, PQ, merges.
  - Pieces of a matched part whose own GT mask is disconnected are `gt_fragments`:
    thin bridges below 512 px resolution, or occlusion (532 of 575 "false positives"
    in the first model were these).
  - `inst_pq_strict` counts those pieces as false positives.
- **Slices:** per image (illumination, blur, layout, backlight, part count, loose
  debris) and per part (group, material, size, tilt, stacking, frame cut, attached debris).

## Results (val, 599 images, macro_val_export)

| model | bscore | recall@1 | recall@2 | precision@1 | IoU | parts missed | false parts/img | inst F1 | PQ | PQ strict | merges/img |
|---|---|---|---|---|---|---|---|---|---|---|---|
| classical seed (Otsu) | 0.404 | 0.871 | 0.908 | 0.965 | 0.936 | 3.83% | 0.052 | 0.847 | 0.813 | 0.793 | 0.224 |
| coarse_ds_40m (label bug) | 0.433 | 0.970 | 0.992 | 0.962 | 0.976 | 0.20% | 0.030 | 0.910 | 0.870 | 0.841 | 0.130 |
| c1_ds_40m (0.3M, ds blocks) | 0.215 | 0.988 | 0.993 | 0.971 | 0.990 | 0.20% | 0.065 | 0.909 | 0.891 | 0.867 | 0.134 |
| c1_dense_40m (0.9M, dense 3x3 blocks) | 0.208 | 0.987 | 0.993 | 0.976 | 0.990 | 0.13% | 0.047 | 0.901 | 0.883 | 0.860 | 0.119 |
| c1_raw_40m (0.3M, ds, raw input) | 0.210 | 0.990 | 0.994 | 0.981 | 0.991 | 0.17% | 0.058 | 0.917 | 0.900 | 0.873 | 0.125 |
| **c3_final_180m = models/coarse_v1** (0.9M, dense, raw + ROI/aspect inputs, 3 h) | **0.206** | **0.990** | 0.994 | **0.984** | 0.990 | **0.13%** | **0.047** | 0.907 | 0.888 | 0.861 | 0.125 |
| YOLO11n-seg, 45 min (2.9M, AGPL) | 0.639 | 0.732 | 0.863 | 0.861 | 0.962 | 0.30% | 0.033 | 0.993 | 0.935 | 0.934 | 0.030 |

### coarse_v1: `models/coarse_v1.pt` (run `c3_final_180m`; superseded by coarse_v2 below)

**Configuration.** Dense 3×3 blocks, widths 16–96 at six levels (0.88M parameters),
raw reading input plus the ROI and log2(aspect) channels. Trained 180 min (127k steps, 2.0M samples,
batch 16, 384² crops) with the aspect-ratio squash and circular ROI augmentation.

**Standard benchmarks:**

| benchmark | model | bscore | recall@1 | precision@1 | parts missed | false parts/img |
|---|---|---|---|---|---|---|
| val | coarse_v1 | 0.206 | 0.990 | 0.984 | 0.13% | 0.047 |
| test (595 images, first use) | coarse_v1 | 0.210 | 0.987 | 0.984 | 0.13% | 0.022 |
| test | classical | 0.397 | 0.872 | 0.971 | 4.24% | 0.030 |

Val and test agree.

**Aspect ratio and circular ROI** (`--variant aspect_roi`; val, then test for coarse_v1):

| model | bscore | recall@1 | precision@1 | parts missed | false parts/img |
|---|---|---|---|---|---|
| classical | 0.466 | 0.852 | 0.944 | 5.33% | 0.063 |
| c1_raw_40m (not trained for it) | 0.272 | 0.983 | 0.960 | 0.64% | 0.260 |
| **coarse_v1**, val | **0.239** | **0.989** | **0.981** | **0.38%** | **0.023** |
| coarse_v1, test | 0.246 | 0.983 | 0.979 | 0.69% | 0.039 |

By ROI kind (bscore / false parts per image), coarse_v1 against c1_raw_40m:
- none: 0.251 / 0.037, against 0.277 / 0.207;
- software crop: 0.224 / 0.013, against 0.245 / 0.280;
- unlit border: 0.223 / 0.007, against 0.284 / 0.347.

Every aspect from 0.56 to 1.9 stays at bscore 0.20–0.28 (1.0: 0.201). The variant's
squashed crops upsample one axis up to 1.9×, which blurs edges, so its absolute numbers
are a little worse than the standard benchmark.

**CPU latency** (ONNX Runtime, idle 16-core desktop, whole coarse stage: resample to
512², network, instance post-processing):
- 69 ms at 4–8 threads for 2048², 2328 × 1552 and 2448 × 2048 frames (network ~58 ms);
- ~100 ms at 16 threads (hyperthreads compete).

`models/coarse_v1-onnx/coarse.onnx` matches torch to 3e-5.

**Hysteresis (`coarse.suppress_weak`, default in `postprocess` and `CoarseModel`).** A
predicted part component survives only if one of its pixels reaches logit 3.
- **Real image:** faint backlight mottling in the real pin-header JPEG came out as blobs
  peaking at logit ≤ 1.9, while the part peaks at 15.6. A 2.43:1 frame, beyond the
  trained 1.9:1.
- **Val sweep** (false parts per image / parts missed / bscore):
  - threshold 0: 0.047 / 0.13% / 0.2057;
  - threshold 2: 0.013 / 0.13% / 0.2052;
  - threshold 3: **0.008 / 0.17% / 0.2051**. The extra miss is 1 part in 2,975.
  - threshold 6: 0.005 / 0.17% / 0.2050. Not chosen: it would put faint real parts at
    risk.
  - The aspect_roi variant at threshold 3: 0.023 → 0.010 false parts per image, with
    no extra misses.
- The benchmark tables above were scored without it; with it, false parts drop about 6×.

**Fine-model renders (physical_v3, 725 val scenes of held-out parts; `coarse_maps`
with a nominal camera and hysteresis):**
- bscore 0.241, recall@1 0.977, recall@2 0.985, precision@1 0.974, IoU 0.997, false
  parts 0.006 per image.
- By lighting, bscore runs 0.205 (reflective) to 0.312 (ambient).
- By family: plate 0.202, freeform 0.211, sphere 0.225, turned 0.276, machined 0.322.
- Heavy contamination: recall@1 0.912. Loose contaminants are object in physical_v3 but
  not in the coarse training data.
- 24 of 725 scenes count as a missed part (contact sheet:
  `work/coarse/review/physical_v3_missed_val.jpg`).
  - **22 are `part_0011_turned`,** a straight wire about 2 source px thick (0.5 model px;
    only ~5 macro pixels are half covered), in every lighting and blur class. It is
    visible in the 512 input as a faint line, but the model leaves it out.
  - **Likely reason:** at macro scale a 2 px wire looks exactly like a loose hair or
    fiber. The macro training data teaches the model to exclude those, and it has no
    parts under ~10 px.
  - One of the 22, scene 86, carries attached contaminants and is 47% covered.
  - The other two misses are `part_0081` under heavy blur (49% covered) and one wire
    scene at 38%.
- **Conclusion for the fine model:** use the coarse outline for parts and bands, and
  take the union with the classical gradient band (`seed.band_from`). That catches
  sub-resolution parts and loose contaminants.

**Older catalog renders (full_catalog, val parts, same nominal camera):**
- bscore 0.243, recall@1 0.989, precision@1 0.969, false parts 0.048 per image.
- Parts missed: 3.9%, again the thinnest turned geometries below the coarse resolution.

## coarse_v2: thin wires (user, 2026-09-29)

coarse_v1 missed a 2-source-px wire, physical_v3 `part_0011_turned`, in 22 of 24
missed val scenes. At macro scale it looks like a loose hair, which the macro data
labels as not-part, and its majority-vote mask is nearly empty.

**Data.** `synthetic-macro --wires 0.8` rendered 1,200 new scenes (6000–7199, all
`validate`d, none failed):
- 923 scenes with 1–3 straight wires, 1,513 wires in all;
- 0.25–4 macro px thick, spread evenly across < 0.5, 0.5–1, 1–2 and ≥ 2 px;
- half of them cut by the frame.

Wires under 2 px are labelled by coverage (≥ 1/8–1/4 of a pixel) instead of majority,
so they become continuous 1–2 px lines. The export and cache now cover 7,200 scenes:
train 5,788, val 718, test 694. The original scenes are unchanged; `--subset v1` scores
exactly the old benchmark.

**Runs** (fine-tunes of coarse_v1, same architecture):
- `c4_wires_ft60` (lr 5e-4, 60 min, wire scenes 2×):
  - Wires are learned: val wires missed 67% → 12%.
  - Loose hairs and fibers become parts too: 53% of loose-debris pixels predicted part,
    against 3.8%. The model learned "thin line = part".
- **`c5_wires_debris_ft45` = `models/coarse_v2.pt`:** from c4, lr 3e-4, 45 min.
  - Loose-debris pixels (≥ 1/8 covered, 1 px dilated) get BCE weight ×9 as negatives.
  - Scenes with debris and scenes with wires are each sampled 2×.
  - Debris confusion falls back to 15% of debris pixels, while wires stay found.
- A from-scratch 3 h retrain was started as a comparison and stopped once the debris
  problem showed, since its recipe was outdated.

| val | coarse_v1 | c4 (wires) | **coarse_v2** |
|---|---|---|---|
| original scenes: bscore | 0.2051 | 0.2072 | **0.2025** |
| original scenes: false parts per image | 0.008 | 0.025 | 0.010 |
| original scenes: loose-debris pixels predicted part | 3.8% | 53% | 15% |
| wire scenes: bscore | 0.340 | 0.229 | 0.234 |
| wire scenes: wires missed | 67% | 12% | 18% |
| aspect_roi variant (original scenes): bscore | 0.2383 | 0.2361 | **0.2345** |
| physical_v3 val: parts missed | 3.3% | 0.3% | **0.14%** |
| physical_v3 val: thin-part scenes found (≥ 50% of pixels) | 0% | 87% | 87% |
| physical_v3 val: detections off the majority GT per image | 0.006 | 0.175 | 0.073 |

The physical_v3 detections off the majority GT are sub-pixel contaminants: all 127
checked for c4 sit on GT coverage. physical_v3 labels them as object.

Val wires found, by thickness (macro px; share with ≥ 50% of the wire predicted):

| thickness | < 0.5 (38) | 0.5–1 (33) | 1–2 (35) | ≥ 2 (42) |
|---|---|---|---|---|
| coarse_v1 | 0% | 3% | 26% | 98% |
| c4 | 89% | 94% | 100% | 100% |
| coarse_v2 | 76% | 94% | 97% | 100% |

coarse_v2 gives up some of the thinnest wires, under 2 source px and hair-like, to keep
loose debris out.

**Test** (first use for the wire scenes; hysteresis on for both models):

| test | coarse_v1 | coarse_v2 |
|---|---|---|
| original scenes (595): bscore | 0.2094 | **0.2061** |
| original scenes: false parts | 3 | 9 |
| of those, on loose debris | 0 | 7 |
| wire scenes: bscore | 0.335 | **0.236** |
| wire scenes: wires missed | 64% | **17%** |
| wire scenes: wire recall@1 | 0.37 | **0.84** |
| aspect_roi variant (original scenes): bscore | 0.2458 | **0.2442** |

**Fine-model datasets, rebuilt with coarse_v2** (`coarse_maps`, nominal camera):
- **physical_v3 val:** bscore 0.243, recall@1 0.984, parts missed 0.14% (coarse_v1
  3.3%). 19 of 23 thin-part scenes found (coarse_v1: 0).
- **full_catalog val:** bscore 0.251, recall@1 0.990, parts missed 0.44% (coarse_v1
  3.9%). 8 of 8 thin-part scenes found.
- "False parts" rose to 0.07–0.13 per image, but all of them sit on GT coverage (139
  of 139 on full_catalog, 127 of 127 on physical_v3 for c4). They are real thin
  geometry or contaminants below the majority-label threshold, not background.

**Residual debris confusion.** Across val and test, coarse_v2 marks loose debris as a
part in 8 components: 6 distinct items (one fiber is in 3 pieces) in 76 scenes with
visible debris, about 1% of all images. coarse_v1 did so once. All are straight or
gently curved hairs and fibers, 1–2 px wide and 30–80 px long, which look like short
wires at this scale (`work/coarse/review/coarse_v2_debris_false_parts.jpg`).
- Pushing further toward "not a part" (a higher debris weight) would lose more of the
  thinnest wires, as c4 → c5 showed.
- Cues that could separate the two: wires in the data are ≥ 150 px long (≥ 52 px after
  shortening) and straight; loose hairs are ≤ 200 px and curved. Sheet: `work/coarse/review/physical_v3_missed_before_after.jpg`
(coarse_v1 | coarse_v2 | full-res crop). Two of the 24 stay weak: an ambient scene on
a mottled backlight (26% found) and a heavily blurred, noisy one (15%).

## Hand-off to fine-model training

Everything is in place to train the fine model with coarse inputs:
- `models/coarse_v2.pt` (the default) and `edgesr.coarse_infer.CoarseModel`:
  - `full_res_logit` gives the frame-resolution part logit;
  - `band(width_px)` gives the band.
  - The raw maps are at 512 × 512.
- `work/cache/coarse_maps/physical_v3` (7,275 scenes) and `.../full_catalog` (2,299 scenes),
  rebuilt with coarse_v2:
  - `part_logit`: raw, so apply `coarse.suppress_weak`;
  - `sdf`: model px;
  - `gt_cover`: GT coverage at 512²;
  - an index with split, part_index, mask_key and classes.
  - All are computed on square 2048² frames, i.e. exactly 4 source px per model pixel.
  - The camera is nominal: backlight at 0.75 of full scale, 20k e⁻, 12-bit.
- **Suggested fine-model inputs:**
  - the upsampled coarse part logit (clipped, e.g. tanh(logit / 4)) in place of the
    Otsu mask channel;
  - band = coarse band (±8 source px) ∪ classical gradient band;
  - keep the reading and the field channels.
  - At inference the maps come from the same `CoarseModel` on the actual frame.
- **Open policy question:** loose contaminants near a part are object in physical_v3
  (and the user wanted debris attached to the edge in the outline). The coarse model
  leaves loose debris out, so the fine model must see them through the gradient band.

**Priority (user, 2026-09-28):** separating touching parts is not important. What
matters is the overall outline of the compound part, i.e. the union mask: boundary
bscore and recall, missed parts, false parts. The center/offset instance head is
implemented (`model.instance_head="center"`, tested) but was not trained. The instance
columns above are secondary.

**Dense vs inverted-residual blocks:** at the same ONNX CPU latency (~40 ms per 512²
before any tuning), dense 3×3 blocks (0.9M parameters) are slightly better on the
outline metrics:
- bscore 0.208 against 0.215;
- parts missed 0.13% against 0.20%;
- false parts 0.047 per image against 0.065;
- ambient bscore 0.304 against 0.313.

Dense blocks are the final choice.

**YOLO11n-seg** (`yolo11n_seg_45m`: COCO-pretrained, fine-tuned on the export at imgsz
512 for 98 epochs):
- **Instances:** best of all three. PQ is 0.935, touching-parts PQ 0.936 (c1_ds 0.797),
  merges 0.03 per image. Ambient false parts per image: 0 (c1_ds 0.24).
- **Boundaries:** worst of all three. Recall@1 is 0.73 and perforated panels 0.56,
  because its masks are assembled from 128² prototypes at 1/4 of the input and it fills
  holes.
- **Verdict:** it cannot seed the fine model's band: at recall@1 0.73, about a quarter
  of the true edge would fall outside a ±4 source-px band. It shows that object-level
  grouping is the weak point of the contact-band instance split. That led to the
  center/offset head (`c2_center_*`).

c1_ds_40m against classical, by illumination (bscore / recall@1 / missed / false per
image / PQ):
- ambient: 0.313 / 0.963 / 0.9% / 0.24 / 0.83, against 0.596 / 0.757 / 20.5% / 0.28 / 0.60
- balanced: 0.196 / 0.997 / 0% / 0.02 / 0.86, against 0.382 / 0.884 / 0.4% / 0 / 0.81
- reflective: 0.197 / 0.987 / 0.2% / 0.01 / 0.84, against 0.409 / 0.853 / 0.5% / 0 / 0.79
- silhouette: 0.187 / 0.994 / 0% / 0.04 / 0.97, against 0.327 / 0.924 / 0% / 0 / 0.94

It is better on every image and part slice. What remains: ambient false parts,
stacked-layout false parts (0.19 per image), heavy blur (recall@1 0.964), touching parts
(PQ 0.80, merges 0.13 per image) and spheres (recall@1 0.971).

## Findings

1. **Label misalignment in the scale augmentation (fixed).** `cv2.INTER_NEAREST` maps
   destination pixel x to source floor(x / s), not to the pixel center. At the training
   scales the labels sat 0.29–0.50 source px off the image, and nearly every sample is
   rescaled. `INTER_NEAREST_EXACT` is center-aligned (checked against the ideal mapping).
   `coarse_ds_40m` was trained with the bug. With the fix, `c1_ds_40m` reached bscore 0.240
   after 10 min, against 0.454 at the same point before.
2. **Classical seed at macro scale** (Otsu on the field-normalized, σ 1.2 image):
   - Under ambient lighting it misses 20.5% of parts, and light parts flip its polarity so
     the background becomes the object.
   - It fills perforations and fine features: face_expanded recall@1 is 0.79.
   - Touching parts merge (touching PQ 0.56).
3. **Transfer to the fine-model renders (physical_v3).** Tested on 60 scenes with
   `coarse_maps` (nominal camera, macro GT from the 16× mask tiles).
   - **Field-normalized input fails on large parts.** With `c1_dense_40m`, spheres and
     plates filling ~85% of the frame dissolve into nested rings: the quadratic field is
     fitted to tile p90 levels, and those are the part when it covers most tiles. The
     macro scenes fill 8–55%.
   - **Raw input avoids this.** The `c1_raw_40m` checkpoint at 30 min scored:

     | on 60 physical_v3 scenes | field (`c1_dense_40m`) | raw (`c1_raw_40m`) |
     |---|---|---|
     | bscore | 0.43 | 0.28 |
     | recall@1 | 0.961 | 0.977 |
     | precision@1 | 0.77 | 0.91 |
     | IoU | 0.945 | 0.988 |

     The spheres and plate become clean. Raw input was chosen for the final model.
   - **Remaining on physical_v3:**
     - Turned parts only 1–4 source px thick are below macro resolution. They are
       found but can drag a false region along with them under heavy blur.
     - Bright parts under ambient light split along their grooves.
     - physical_v3 labels loose contaminants as object, while the coarse model leaves
       loose debris out. The fine model's band should therefore also include the
       classical gradient band (`seed.band_from`).
4. **False parts of the learned model** (first model, 599 val images): 50 components,
   median 11 px². By cause: 19 on background texture/smudges, 10 next to parts
   (reflections), 9 on loose debris, 6 inside holes of bright parts, 6 at the frame. A
   16 px² minimum area removes about two thirds (0.083 → 0.030 per image) for +0.03 pt
   missed parts.
