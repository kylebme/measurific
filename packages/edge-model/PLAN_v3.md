# PLAN v3: a coherent edge model (large GPU model first)

Status: drafted 2026-09-30 after the human's decision (RESEARCH.md, "Fine model v4"). It
supersedes PLAN_v2 for new work. The CFI v4 model (`models/cfi_v4*`) is kept for GUI
integration and testing.

## 1. Goal (the human's words, condensed)

- **A foundation model.** It learns the edge spread function (ESF) and finds where the true
  geometric edge is, across the optics and parts in the dataset. It is not tuned to one
  lens or part type.
- **Accuracy:**
  - Sharp edges: excellent subpixel accuracy.
  - Degraded edges or lighting: accuracy may degrade, but the result must stay a
    reasonable estimate.
  - Never a winding contour or hallucinated extra edges.
  - Where the inverse problem has several solutions, give close to the best estimate the
    data allows.
- **Diagnosis:** "the model only provides an estimate for one point at a time." Option B:
  regress where the edge is, coherently, with topology decided upstream.
- **Order of work:**
  1. Start with a large model that needs a GPU to be fast, and is fast to train and learn
     from.
  2. Move toward the 1–2 s CPU goal later, if that is possible.
- **Hardware:** one RTX 3070 (8 GB), shared with scoring jobs. 16-thread CPU for data.

## 2. Why CFI plateaued (what the new design must fix)

CFI (PLAN_v2) evaluates an occupancy logit independently at each of 16×16 sub-pixel
positions per pixel. Each logit comes from a 3×3 neighbourhood of fine features, and those
see about ±13 px.

1. **Point-wise decisions.**
   - The contour is the 0-level of millions of independent classifications. Only points
     at the edge carry its position, and only through a sign flip.
   - Under ambiguity (depth-of-field ramps, heavy blur, ambient light), neighbouring
     decisions disagree. The level set then wanders, fragments and closes small loops:
     the "winding garbage".
2. **Little evidence pooling.**
   - An edge's ESF is shared along the edge (same depth, same blur). A ±13 px window sees
     ~26 px of it, and a 3×3 decoder pools almost nothing.
   - R25, w32 and more training on these edges each bought about 1 pt on the shifted-edge
     slice.
3. **Topology and geometry are coupled.** One local classifier decides both whether an
   edge exists and where it is, so local ambiguity creates edges.
4. **No notion of uncertainty.** There is nothing to say "this edge is ±1.5 px" and keep the
   estimate smooth.

Measured (RESEARCH.md):
- ~91% of missed part-boundary points are depth-shifted edges.
- An oracle coarse channel does not fix them.
- Misses concentrate where the focal plane cuts through the part, which is genuinely
  ambiguous. There the target is the best estimate, not the impossible exact answer.

## 3. Design axes and candidates

### 3.1 Output representation (the core decision)

| candidate | how an edge is estimated | coherence / hallucination | sharp-edge subpixel | ambiguity behaviour | train speed | risk |
|---|---|---|---|---|---|---|
| R0 occupancy (CFI) | sign flip of per-point logits | poor (above) | good (0.05–0.07 px) | wanders | ok (sparse queries) | known plateau |
| **R1 truncated signed distance (TSDF)** | every pixel within T px regresses its signed distance to the edge; contour = 0-level | strong: ~2T pixels across the edge and all along it predict the same edge; the field must be 1-Lipschitz, so a wiggle needs inconsistent distances everywhere in the band | expected good: linear interpolation of a smooth field is exact for straight edges; needs precise targets | L1 → median edge position; Laplace NLL adds per-pixel uncertainty | **dense** supervision: every pixel of every crop | low |
| R2 TSDF + normal votes | as R1 plus direction; edge points = p − d·n, pooled over pixels (Hough-like) | stronger (explicit consensus) | as R1 | as R1 | as R1 | post-processing only |
| R3 explicit contour (snake) | proposal polygon; a 1-D sequence model over normal profiles predicts offsets per vertex (DeepSnake / E2EC style) | strongest; topology fixed by the proposal | very good on straight edges (1-D ESF) | smooth by construction | slower: correspondences, resampling | high: proposals must catch small features; corners; merges |
| R4 edge heat-map + subpixel offsets | classic edge detection with NMS and linking | open curves, threshold-dependent extras | good | poor | dense | linking/topology |
| R5 parametric ESF fit (analysis by synthesis) | network predicts local ESF parameters; least-squares edge fit | good on simple edges | excellent on straight sharp edges | explicit, but only within its model | n/a | fails on corners, depth ramps, mixed edges |
| R6 polygon / set transformers | global decoder emits contours | coherent | weak subpixel | ok | slow, data-hungry | high |
| R7 mixture of offsets | K hypotheses + weights per pixel | as R1 | as R1 | represents several inverse solutions explicitly | as R1 | complexity |

**Choice: R1 (TSDF)**, with R2-style consensus available at contour extraction.
- R1 answers the human's diagnosis directly: an edge's position becomes a quantity that many
  pixels estimate jointly, not a sign flip at one point.
- It trains on dense targets, the fastest learner here, and reuses the data pipeline and
  benchmarks.
- Two candidates stay open as later stages:
  - **R3 as a refinement stage**, on contours extracted from R1. It is also the cheapest
    CPU path: 1-D work along ~5k boundary px per image.
  - **R7** if ambiguous edges need explicit alternatives.

### 3.2 Backbone

| candidate | context | full-res precision | memory on 8 GB | notes |
|---|---|---|---|---|
| **B1 U-Net** (residual blocks, GroupNorm, stride-2 down, full-res skip) | RF ≈ 250 px at 5 levels | full-res encoder and decoder stages | fits at 256² crops, batch ~16 | simple, fast, translation-equivariant except the downsampling |
| B2 U-Net + self-attention at 1/16 | whole crop | as B1 | +small | cheap: 16×16 tokens per 256² crop |
| B3 HRNet (parallel resolutions) | large | best | heavy at full res | later, if precision lags |
| B4 ViT / Swin | global | patchify hurts subpixel equivariance | heavy | rejected |
| B5 CFI sparse fine branch | ±13–25 px | good | light | the plateau we are leaving |

**Choice: B1, deliberately local** (revised 2026-09-30 after the human's question "do we
really need the model to see the entire image?").

**Locality principle:**
- The ESF is local. Macro-scale content (part layout, global shape) has no physical bearing
  on where an edge is. A network that sees it can overfit to the generator's layouts, or
  "complete" shapes from priors. That is a metrology hazard, and same-generator val/test
  sets cannot detect it.
- So the geometry network sees only what the physics needs: the ESF, depth-of-field ramps
  of 5–30 px with their plateaus, and enough of the edge's length to pool evidence.
- The only global input is the topology hint (the coarse prior). It is heavily perturbed,
  so it can say which side is part but not where the edge is.

**Design and measurements:**
- **3-level U-Net:** widths 48 / 96 / 192 at strides 1 / 2 / 4, encoder blocks 2 / 2 / 3,
  decoder 1 / 1.
  - Theoretical receptive field 101 px (±50).
  - About 95% of the influence lies within r ≈ 17 px at init; training spreads it.
  - 3.0M params, 31 GMAC per 256² crop.
- **Normalization is per-pixel channel LayerNorm, never GroupNorm or BatchNorm-style
  spatial statistics.** GroupNorm made the theoretical receptive field the entire input:
  every output depended on the whole image's statistics. That leaks exactly the global
  information we want to exclude, and differs between training crops and full frames.
  The 1/16 self-attention option is likewise global and is off.
- **The 5-level variant** (RF 361 px) stays an ablation. It is to be used only if a specific,
  physically motivated failure needs more context.
- **Consequence for data:** the training images must be realistic at the 100 px scale, not
  globally. physical_v3 already is, except for local part-to-part interactions: gaps,
  touching or stacked parts, debris at an edge. Cheap lattice-aligned compositions of
  physical_v3 renders (multiplied transmittances, mask union) can cover those. A full
  resolution re-render of the macro scenes is not required for this model.

### 3.3 Topology and global context

- **C1 (chosen): the frozen coarse_v2 model's part probability as an input channel**, as now.
  Training perturbs it much harder:
  - shift ±3 px;
  - erode or dilate the logit by up to 3 px;
  - scale the logit ×0.5–2;
  - 15% of crops get an uninformative (zero) channel.
  The coarse-oracle test showed the fine model follows a coarse boundary that is 3 px off.
  The new model must treat the channel as a topology hint (which side is part), not as
  geometry.
- C2: a learned global branch (512² whole-image features into the bottleneck). It is
  end-to-end, but costs a second forward per crop. Later.
- C3: crop only. This loses part-level context under ambient light. Rejected for v1.

### 3.4 Loss under ambiguity

- **SDF:** L1 on the truncated signed distance (median estimate), weighted toward the edge
  (w = 1 + 4·exp(−|d|)). T = 8 px for v1, so the network must place edges inside ramps up to
  ~16 px wide while far pixels only need the sign.
- **Uncertainty:** a separate head trained with Laplace NLL on the detached error
  (|e|/b + log b). It never biases the estimate, and gives a per-pixel ±b for contour
  smoothing, the GUI and hallucination suppression.
- **Direction** (unit normal toward the edge): cosine loss near the edge, small weight. It
  feeds R2 consensus and corners.
- **Deep supervision:** TSDF/4 at 1/4 resolution. It trains faster, and later becomes the
  band proposal of a CPU cascade (section 7).

### 3.5 Contour extraction

- **v1:** marching squares on the 1× SDF with linear interpolation, densified to 1/16 px
  points for the benchmark. Rings for the GUI come from the existing tracer machinery.
  - Straight edges: exact.
  - Corners: rounded by ≲0.2–0.3 px. That is a minor degradation the plan allows.
  - Features under ~1 px wide (ultra-thin wires): not represented. Not a priority.
- **Later:** a 2× SDF head (pixel shuffle) if corners or thin features matter. R2 vote
  pooling. Uncertainty-weighted smoothing along the contour, widening with b, so heavy
  blur gives a smooth estimate while sharp edges keep every detail.

## 4. The v5 model ("SDFNet") in numbers

- **Inputs (4 channels):** reading normalized by the coarse-background field, coarse prior
  2p − 1, field level, and the raw reading. These are the CFI v4 seed channels plus the raw
  level, used for saturation and polarity.
- **U-Net:**
  - widths 32 / 64 / 128 / 192 / 256 at strides 1 / 2 / 4 / 8 / 16;
  - 2 residual blocks per encoder level and 1–2 per decoder level;
  - optional attention at 1/16;
  - heads at full resolution: SDF, log b and direction (fp32);
  - TSDF/4 head at 1/4.
  - Budget: ~5–8M params and ~30–40 GMAC per 256² crop. Full 2048² frame on the GPU in one
    pass, ~0.1–0.3 s, tiled if memory is short.
- **Training:**
  - 256² crops from the cache_v4 ScenePool, with the same sampler categories, augmentation
    (DATA_V3) and anywhere crops as CFI v4.
  - 16 px loss border.
  - AdamW, cosine schedule, bf16 autocast with fp32 heads, ~16 crops per batch.
  - Dense targets per crop pixel: signed distance to the GT boundary as point-to-segment
    distance to the two nearest exact GT crossings (exact for straight edges, not
    ±1/32 px), sign from the 16× mask, direction, scaled by the crop matrix (rotation and
    downscale).
  - Target: ~100 crops/s. The data pipeline may be the limit; measure first.

## 5. Metrics (match the goal, not just the 0.5 px hit rate)

- On val_v4_full (unchanged spec), all existing slices and `edge_shift`, plus:
  - **Best-estimate error:** mean and median |GT → prediction| (clipped at 3 px) per
    `edge_shift` bin, alongside the 0.5 px missed rate.
  - **Winding:** the ratio of predicted to GT boundary points near the GT (within 2 px).
    Above 1 means doubled or wiggly contours.
  - **Hallucination:** far-extra points per image (existing), split by lighting.
- **Acceptance against CFI v4 (w32):**
  1. Aligned (sharp) edges: missed ≤ 0.8% and |offset| ≤ 0.095 px. On val_v3: sparse sharp
     loc ≤ 0.074 px.
  2. Shifted edges: lower mean |offset| and absent rate; signed bias ≈ 0.
  3. Far extras < 6.9k per image; winding ≈ 1.
  4. Heavy blur: plausible, with fewer absent and false edges. It need not be precise.

## 6. Experiments (phase 1: about a day of GPU time)

1. **Smoke test (5 min):** data throughput, memory, loss decreasing.
2. **`v5_sdf_60m`:** the base model.
3. **`v5_occ_60m`:** the same backbone with an occupancy (BCE) head. This ablation separates
   representation from context. If occupancy matches the SDF, the gain came from the
   backbone.
4. **`v5_sdf_180m`** if (2) is promising.
5. Then one change at a time:
   - crop 384 (more context);
   - the coarse prior's dropout and perturbation strength;
   - attention at 1/16;
   - 2× head;
   - T = 16.

## 7. Path to the 1–2 s CPU goal (later)

- **Cascade:** run the 1/4-resolution TSDF everywhere (cheap), then the full-resolution
  decoder only where |TSDF/4| is small. Convolutional, so sparse evaluation is exact, like
  CFI's windows.
- **Distillation:** train a small student on the large model's outputs.
- **R3 refinement:** a cheap 1-D contour stage on top of a lower-resolution SDF.

## 8. Risks

- **Subpixel precision of SDF regression at 1×.** Mitigations: exact segment-distance
  targets, fp32 heads, near-edge weighting, and the 2× head fallback.
- **Data throughput** at 256² with dense targets. Bound the k-d tree query
  (distance_upper_bound = T + 1) and profile.
- **GPU memory** while scoring shares the GPU: scoring is capped at 40%, training is
  budgeted at ~4.5 GB, and inference falls back to tiles.
- **Crop-border context** differs from full-frame inference. Mitigations: the loss border,
  full-frame inference, and testing the border effect on val.
