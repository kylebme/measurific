# Training on the physical_v3 renders: guidance

The new dataset is `/media/kyle/T7 Shield/inspection-synthetic-renders/physical-v3-30gb-2048`:
30 GB, about 7,400 scenes, same 300 catalog parts and the same geometry split. How it is made:
`../../research/synthetic-renders/README.md` ("Physical appearance profile"). Per the human, it
**replaces** `full-catalog-5gb-2048` for training. The old set stays on disk. `val_v3_full`
and `test_v3_full` remain useful as regression checks on the old distribution.

## Why: what the shipped model gets wrong on the new renders

`../../research/synthetic-renders/scripts/score_edge_model.py` ran `models/cfi_B_seed-onnx` on 104 pilot
scenes with the fullimage sensor model (20k e⁻ full well, 5e-4 read noise, 12-bit).
Numbers are per-scene medians; the distance floor of this metric is ~1/32 px.

| slice | loc (px) | extra | missed | contaminant boundary missed |
| --- | --- | --- | --- | --- |
| all | 0.156 | 4.4% | 4.4% | 16% |
| in focus / mild / medium / heavy blur | 0.136 / 0.130 / 0.171 / 0.218 | 3.9 / 2.8 / 1.3 / 15.4% | 3.6 / 2.6 / 2.5 / 17.0% | 28 / 4 / 10 / 51% |
| tilted / untilted | 0.161 / 0.151 | 5.2 / 3.9% | 5.4 / 3.6% | |
| ambient lighting | 0.223 | 68% | 24% | 62% |

- **Outward bias.** On clean silhouette and balanced scenes the matched contour sits
  **+0.11–0.13 px outside** the geometry for in-focus, mild and medium blur. The renders
  themselves are unbiased: a `fit_shift` of each image against its exact coverage gives
  −0.002 to +0.018 px per blur class. This is the bias README.md documents under Gaussian
  blur (+0.07–0.17 px), and it matches the human's real pin-header image, where the contour
  rides the bright outer skirt of ~4 px wide edges. The old renders were point-sampled
  with no lens PSF, so almost every "sharp" training edge was a binary step. The new
  in-focus edges have σ ≈ 0.7 px, and mild, medium and heavy edges have σ ≈ 1.0, 1.4
  and 2.8 px.
- **Contaminants** (fibers, particles, burrs; all labelled as object) are missed 16% of the
  time, and half the time under heavy blur. Part of this is the model; part is policy
  (below).
- On the real JPEG, most of the side-wall wiggle came from JPEG blocks. A 0.8 px blur cut the
  straight-edge residual from 0.42 to 0.14 px. The human's production images are lossless,
  so JPEG is a robustness concern, not the main target.

## Pipeline changes the new data needs

1. **Masks are per scene, not per part.** `cache.py` stores `cover.u16` and `block_index`
   per `part_index` and assumes every appearance shares them. That no longer holds for
   tilted (30%) or contaminated (~28%) scenes.
   - Key masks by `metadata["mask_key"]`: the part id for untouched base masks, otherwise
     the scene id. Expect ~300 + ~4.3k unique masks.
   - A dense `cover.u16` for each of them would be ~38 GB. Store coverage block-sparse,
     like `bits.u8`, or compute candidates at build time and keep only what the sampler
     needs.
   - Tile `file` entries can be relative paths (`../../base_masks/...`). The existing
     `os.path.join(scene, "mask", file)` already handles them.
2. **Split** stays by `catalog.part_index` with the same seed. Contaminants and tilt are
   appearance, not geometry.
3. **Image cache size.** 7.4k × 2048² × 2 B ≈ 62 GB, about all of the RAM. The internal
   disk has ~1 GB free, and random reads from the USB T7 will starve the GPU. Stream shards
   instead: load N scenes sequentially into RAM, draw many crops, rotate. Or free internal
   disk first.
4. **`backlight_level`** (median over mask-background pixels) is still right. It now
   includes gradient and vignetting, which is what a calibration would measure too.
5. **Sampler.** Add a `contaminant` crop category, about 5–10%, centred on boundaries where
   the scene mask differs from the base mask. Rare items are otherwise seldom sampled.
   Keep `uniform` and flat crops: they are what suppress false contours (v13_hard).
6. **Seed support and region filter.** Arm B keeps only rings the classical seed supports,
   then drops objects under 50 px². Faint or small contaminants can be learned correctly
   and still be removed after the model. Decide the policy explicitly: the renders
   label everything, as asked. Score contaminant boundaries with and without both
   guards.

## Training-side camera effects (augmentation)

Keep the existing exposure, offset, bit-depth and backlight-value jitter. Changes:

- **Noise:** the human judged the old noise too aggressive. Move `full_well_e` from
  [2k, 40k] to about [6k, 60k] log-uniform and keep `read_noise`. The renders now carry
  their own backlight texture, so less synthetic noise is needed for robustness.
- **ISP sharpening** (p ≈ 0.2): unsharp mask, σ 0.5–1.5 px, amount 0–0.6. The real image
  has the typical overshoot (211 → 216 DN just outside the edge). A symmetric unsharp mask
  does not move a symmetric edge's 50% crossing, so the labels stay valid.
- **JPEG** (p ≈ 0.1, quality 75–95, 8-bit): only for robustness to shared or recompressed
  images, which is how the example image arrived.
- **Downscale** (p ≈ 0.1, factor 1–2): area-average, not bicubic point sampling, so pixel
  integration stays physical. Put the scale into the patch matrix `A` so the mask is
  resampled consistently.
- **Do not** add input-only blur or geometric distortion. Blur is now physical in the
  renders and consistent with the 3-D geometry. A 2-D blur of a 3-D edge is what taught
  the outward prior in the first place.

## Recipe and acceptance

- The old model was not data-limited: the train/val gap was ~3% and the time budget
  dominated. The new distribution is harder and 3× larger. Start from the `cfi_B_seed`
  recipe at 60 min for comparability, then 3–4 h. Re-run the trainfit gap to see whether
  the harder data changes the data-limited verdict.
- Build a new benchmark from val parts of the new set, e.g. `val_v4_full`. Slice it by blur
  class, tilt, contamination (contaminant-boundary missed rate), lighting, backlight type
  and glare. Report the **signed** matched offset per slice as well: the outward bias is
  the headline defect.
- Also score the old `val_v3_full` and `test_v3_full` so the sharp-silhouette accuracy
  (~0.035 px) is not traded away silently.
- Real image check: the human's pin-header example (a JPEG shared in chat; not in the repo). The straight-edge
  residual on the lower pins was 0.42 px as is and 0.14 px after a 0.8 px blur. A lossless
  capture from the real system would make this a proper sim-to-real check.
