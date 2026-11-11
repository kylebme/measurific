# PLAN v2: full-image, CPU-fast subpixel boundary model

Status: approved 2026-09-26; phases 0–2 done, with arm B (classical seed) selected; paused at human request after the A/B. See RESEARCH.md "FINAL STATE".

## Why a new direction

The patch model works (bscore ≈ 0.137 on val_v2 at 60 min), but its compute is spent
in the wrong places:

| fact (measured 2026-09-26) | consequence |
| --- | --- |
| 4.67 GMAC per 48 px tile; 12.6 TMAC per 2048² image (stride 40) | ~118 s on this 16-thread CPU (~107 GMAC/s effective) |
| Tile seams need blending (stride 32: 1.6× tiles) | ~190 s CPU per image |
| 58% of per-tile compute runs at ≥ 96² output resolution, everywhere | dense 16× output for pixels that are nearly all uniform |
| Median 5.4k boundary pixels per image (0.13%); 1.4% of pixels (max 2.2%) are within 6 px of a boundary | >98% of the image needs no subpixel work |
| Trained RF spans the whole patch, so predictions depend on tile position | seams, and ~1 px grid-shift jitter without blending |

## Targets

- **Latency:** ~1 s single-image latency for a ~2k² image on a CPU of this class (16
  threads). This is a ballpark, not a hard gate: deployments vary, and some have GPUs.
  Budget ≈ 100 GMAC in PyTorch fp32 at the measured ~107 GMAC/s, more with an
  optimized runtime. Export must be portable (ONNX), and the GPU path must be kept.
- **Accuracy, tied to the information present in the image.** Blur, sharp corners,
  noise and saturation are lossy, so the bar differs by slice (see Evaluation):
  - **Protected (maintain or improve vs today's model):** sparse sharp edges, and
    sparse medium-blur edges, under all lighting classes including challenging
    lighting.
  - **Minor degradation allowed:** heavy blur; very small or dense features; corners;
    heavy noise; bright specular highlights at an edge.
- **Output:** contour polylines with no functional downside vs a dense 16× mask (see
  Contours). The 1 Gpx dense mask is never materialized.
- **Forward compatibility with brightfield imaging** (multi-angle lighting, semantic
  edges rather than one silhouette). This shapes the output head now (see below).

## Architecture: Coarse–Fine–Implicit (CFI), fully convolutional, no tiles

One network over the whole image, built from translation-equivariant convolutions.
Sparse evaluation on gathered windows is then *exactly* equal to dense evaluation, so
there are no seams by construction, and the only overlap cost is the fine branch's
receptive-field margin.

1. **Coarse context branch (1/4 res, whole image).** A strided stem, then a light U-Net
   over 1/4, 1/8 and 1/16 (32–64 ch, depthwise-separable blocks, channel LayerNorm), with
   RF ≫ 100 px. It decides region class and polarity (the ambient/specular/bright-part
   cases), and carries **local plateau levels and edge localization** that the fine
   branch uses to normalize and center the ESF. The backlight value is an input.
   - Outputs: context features, coarse occupancy, and a band map ("an edge lies within
     R px").
   - ~3–6 GMAC.
2. **Fine geometry branch (source res, band only).** A CNN on the image, the upsampled
   coarse features and an optional 1/2-res mid-scale feature.
   - **Its context radius R is a primary research axis, not a fixed choice.** It must
     see enough of the edge-spread function (both plateaus and the full ramp) for
     medium blur, whether or not the edge is centered in its window. Candidates:
     R ∈ {±5, ±8, ±12, ±16} px, via dilation or a 1/2-res mid branch so cost grows
     slowly.
   - Hypothesis: medium blur needs R ≳ 2–3× the local ESF width, unless the coarse
     features already supply plateau levels.
   - Cost scales with (band + margin R) area. With ~1.4% of pixels in the band, R
     dominates the fine cost, which is why the latency skeleton sweeps it first.
3. **Implicit field decoder (continuous queries).** It interpolates fine features at a
   query point q (LIIF-style local ensemble), appends Fourier-encoded sub-pixel
   offsets, and runs a small MLP. It has **two heads**:
   - **occupancy** logit: the backlit silhouette; the proven target and loss.
   - **edge field:** unsigned distance to the nearest edge (clipped, ~1 px) plus the
     unit direction to it. Trained from the same 16× GT (exact distance transform of
     the hi-res boundary). It does not assume a closed silhouette, so it is the
     brightfield path (below).

**Estimated total ≈ 10–20 GMAC per 2048² image** across the R sweep, ≈ 0.1–0.3 s at
today's measured efficiency.

### Arm B (2026-09-26, human-directed): classical seed instead of a learned coarse network

Motivation (human): the synthetic parts are unrealistic when zoomed out, so a learned
global model risks overfitting at that scale. Stay focused on the fine model.
- **Seed** (`edgesr/seed.py`) mirrors `_background_mask` in
  `packages/measurement/src/inspection_measurement/adapter.py` (the reference; its README may
  be stale):
  - 12×12 tile 90th percentiles; quadratic field with two-sided trimming
  - normalize ×230; Gaussian 1.2; Otsu gives the object mask
  - noise from bright pixels
- **Band** = threshold contour ± `search_radius` (4 px) ∪ (gradient magnitude >
  clamp(4·noise, 4, 25) DN) ± 1 px. **PAE is not used** (human: unstable on noisy and
  dense geometry).
- **Fine model inputs:** [normalized image, threshold mask, local field level]. No
  coarse network; the threshold mask fills outside the band.
- **Training:** per-scene field coefficients and Otsu level are cached from the raw
  render and mapped through each crop's photometric gain/offset. This agrees with
  full-image seeding (mask agreement 99.98–99.997%).
- **A/B at the front of the queue:** arm A (learned coarse) vs arm B (seed), 60 min,
  2 seeds each, scored on val_v3_full. The PAE-feature arm was dropped.

## Contours: no functional downside vs the dense mask

Requirements the contour output must meet, each tested against contours extracted
from a dense 16× rendering of the same field on benchmark regions:
- nested contours and holes; open contours at the image border;
- sub-pixel features, and multiple edges crossing within one source pixel;
- topology identical to the dense extraction, points within 0.01 px.

Extraction: evaluate the field on the **full 1/16 px lattice inside the ±1 px edge band**
(≈ 14k px × 256 ≈ 3.6M queries, ~5–8 GMAC), then marching squares with interpolated
crossings. This is exactly what the dense mask would give there. Adaptive (coarse
lattice then refine) is only an optimization, and must pass a recall test, because
features narrower than the coarse lattice step would otherwise vanish.

## Brightfield readiness (future; constrains design now)

- **Inputs:** N lighting channels plus per-channel lighting metadata. The stem is
  configurable from day one; backlit is N = 1 with the backlight value.
- **Outputs:** occupancy is silhouette-specific. The **edge-field head** (distance +
  direction to the nearest edge) is the representation for semantic edges: open
  curves, T-junctions, creases. Curves come from q + d·dir samples linked into
  polylines. The point-to-curve metrics already work for open curves.
- **v2 de-risk step:** train both heads on backlit and compare contour accuracy from
  occupancy vs from the edge field. If they match, brightfield needs new labels, not
  a new model.
- **Label interface (for synthetic-renders, later):** 16× semantic-edge maps (face-ID
  or normal-crease / depth-discontinuity boundaries) alongside images from multiple
  light angles. Out of scope here; noted as a requirement.

## Training (reuses the data, split, registration and metrics)

- **Crops:** 192–256 px source crops with continuous rotation/translation, bicubic
  input, and photometric + backlight augmentation. The whole CFI is trained densely on
  the crop; sparsity is inference-only.
- **Point supervision:** GT at query points from the block store at floor(16·s),
  exact and cheap. Mix: ~70% within 1 px of an edge, 20% in the ±R band, 10% uniform.
  Samples/s was the dominant lever, and this removes the 768² masks entirely.
- **Losses:**
  - point BCE (occupancy)
  - edge-field L1 on distance and direction near edges
  - coarse occupancy (soft BCE vs cover/256)
  - band map with a high recall weight
- **Keep what was learned:**
  - backlight input
  - channel-LN stabilization
  - uniform/sparse regions in the sampling mix (they suppress false contours)
  - exposure ranges ⊇ deployment
  - 2-seed confirmations

## Evaluation (new benchmark version, recorded as an infrastructure change)

- **`val_v3_full`:** full 2048² val images (≈ 2 appearances × 30 val parts, stratified
  over focus/illumination/bit depth), with fixed seeded photometrics and whole-image
  contour metrics.
- **Information-aware slices, labeled per GT boundary point:**
  - **blur:** local ESF 10–90% width measured on the noise-free render along the GT
    normal. Sharp < 1 px, medium 1–3 px, heavy > 3 px. This replaces the scene-level
    focus class, which mixes depths.
  - **sparsity / feature size:** local fg and bg thickness and the distance to the
    nearest other edge (sparse ≥ 4 px on both sides).
  - **corner:** local GT curvature.
  - **lighting:** illumination class, plus a **specular-at-edge** flag (render brighter
    than the backlight within 2 px of the edge).
  - **noise:** SNR from the photometric draw (full well, read noise, bit depth).
- **Acceptance gate vs reference** (today's best patch model, stride-32 blend, scored
  on the same benchmark):
  - protected slices (sparse×sharp, sparse×medium, each lighting class) must be
    non-inferior: loc error within +0.005 px and missed/extra within +0.5 pt.
  - other slices: ≤ ~15% relative bscore degradation, reported.
- **Optional:** an information-floor reference for straight sparse edges (a
  Cramér–Rao-style bound from ESF width, contrast and noise). It shows how far each
  slice is from what the image allows.
- **Also tracked:**
  - CPU latency (end-to-end, median of 5, 16 threads, 2048²) and GPU latency
  - band recall (≥ 99.9% of GT edge points)
  - grid/crop-shift jitter, which should be ~0 by construction

## Phases and gates

0. **Latency skeleton.** CFI with random weights, timed on CPU per stage on a real
   image's band, sweeping fine R and coarse width.
   - Gate: a mid config (R = ±8, full 1/16 queries in the ±1 px band) ≤ 0.5 s
     end-to-end in PyTorch. Also build the queue pause/hold mechanism so CPU timing runs
     on a quiet machine.
1. **Pipeline:**
   - crop + point-query sampler and edge-field GT
   - registration tests (point GT vs mask windows, rotated crops, distance-field GT vs
     brute force)
   - val_v3_full with slice labels
   - reference scoring of the current model
2. **Dense training of CFI on crops, both heads.**
   - Gate: protected slices non-inferior, band recall ≥ 99.9%.
   - Compare occupancy vs edge-field contours.
3. **Sparse inference:**
   - band gather, 1/16 queries in the band, contour extraction
   - tests: sparse == dense to float tolerance; contours == dense-rendering contours
     (topology and 0.01 px)
   - end-to-end latency
4. **Autoresearch on CFI** under the latency target. Axes, first:
   - **fine R**, coarse resolution/width, mid-scale branch
   - decoder (implicit MLP vs a per-pixel 16×16 block head)
   - query/point sampling mix, crop size, band threshold
5. **Deployment:** ONNX Runtime / OpenVINO, thread tuning, and an int8 test (the fine
   branch and decoder must keep subpixel accuracy).

## Alternatives considered

- **Patch CNN, skip uniform tiles and blend:** ~12 s CPU, 12× over, still seam-prone,
  no global context.
- **Transformer / hybrid:** the coarse CNN supplies global context for ~1% of the
  budget, and attention over 4 Mpx is CPU-unfriendly. Revisit as a small attention block
  at 1/16 if polarity failures persist. For brightfield, cross-attention over lighting
  channels may become worthwhile.
- **Direct contour regression:** it cannot represent sub-pixel topology. The edge-field
  head covers the useful part (distance + direction) inside the implicit framework.

## Main risks

- **Crop vs full-image context mismatch:** use large crops; exclude the loss near crop
  borders.
- **Band misses on faint or defocused edges:** conservative dilation; recall is
  gated.
- **Implicit decoders round sharp corners:** Fourier offsets; watch the corner slice.
  Some rounding is lossy in the image anyway.
- **Fine R too small for medium blur:** it is the first research axis, and the
  protected slice catches it.
- **Sparse gather/scatter overhead:** fixed-size windows as batched dense ops; export to
  an optimized runtime.
