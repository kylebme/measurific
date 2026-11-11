# RESEARCH.md: durable research state

> **2026-09-26: direction change.** The human set a ~1 s CPU single-image latency target
> for ~2k² images. The patch model (~118–190 s CPU) is now the **accuracy reference**,
> not the product. Work follows `PLAN_v2.md`: a full-image Coarse–Fine–Implicit model
> with information-aware acceptance slices, fine-context radius as the first research
> axis, and an edge-field head for future brightfield. No new patch-model experiments;
> the queued ones finish as references.

Last updated: 2026-09-26 23:55 (FINAL: stopped at human request after the A/B)

## Edge model v5 (PLAN_v3): coherent edges with a TSDF U-Net (2026-09-30 → 10-01)

**Selected (2026-10-01, the human: "stop the run and use the best model so far"):
`v5_sdf_ft3_180m` → `models/sdf_v5.pt` and `models/sdf_v5-onnx/`.**
- Training: v5_sdf_180m plus three 180-min warm starts (lr 4e-4, 3e-4, 2.5e-4), 720 min and
  ~4M crops in all.
- ONNX Runtime matches torch within 4.8e-4 px.

| val_v4_full | cfi_v4 | v5_sdf_180m | v5_sdf_ft180 | v5_sdf_ft2_180m | **sdf_v5 (ft3)** |
|---|---|---|---|---|---|
| training (min) | 180 | 180 | 360 | 540 | 720 |
| bscore | 0.1686 | 0.1635 | 0.1537 | 0.1456 | **0.1421** |
| missed / extra | 7.0 / 9.7% | 8.4 / 6.5% | 7.2 / 5.4% | 6.3 / 4.65% | **5.9 / 4.3%** |
| loc (px) | 0.1085 | 0.112 | 0.112 | 0.110 | **0.1087** |
| far extras per image | 6.9k | 0.94k | 0.84k | 0.67k | **0.57k** |
| aligned missed | **0.8%** | 1.3% | 0.8% | 0.9% | **0.8%** |
| shift 0.5–2 missed | 16.9% | 18.6% | 16.7% | 14.6% | **14.0%** |
| shift 2–8 missed | 19.5% | 25.2% | 21.3% | 16.4% | **14.5%** |
| no visible edge missed | 23.5% | 30.1% | 25.7% | 23.7% | **20.0%** |

- The remaining queue was cancelled: `v5_nearw_ft2_180m`, `v5_sdf_w64_180m` and
  `v5_sdf_shift_180m`.
- **Speed** (`edgesr.latency_sdf`, models/sdf_v5, 8 val_v4_full frames of 2048² × 3 repeats,
  medians; RTX 3070 8 GB, 16-thread CPU, with the human's watchdog using ~1–2 cores):

  | path | total | coarse | inputs | network | post | missed / loc on the 8 frames |
  |---|---|---|---|---|---|---|
  | **GPU bf16, compiled** (channels-last, cuDNN autotune, torch.compile) | **0.46 s** | 0.025 | 0.11 | 0.23 | 0.08 | 3.55% / 0.100 |
  | GPU fp32, compiled | 0.68 s | 0.024 | 0.11 | 0.45 | 0.08 | 3.53% / 0.100 |
  | GPU bf16, eager | 2.52 s | 0.023 | 0.10 | 2.30 | 0.07 | 3.46% / 0.100 |
  | GPU fp32, eager | 3.22 s | 0.023 | 0.10 | 3.00 | 0.08 | 3.52% / 0.100 |
  | **CPU, ONNX Runtime fp32** (torch-free `sdf_runtime`) | **26.5 s** | 0.13 | 0.12 | 26.2 | 0.09 | 3.53% / 0.100 |
  | CPU, PyTorch fp32 | 41.2 s | 1.06 | 0.10 | 40.0 | 0.06 | 3.53% / 0.100 |

  - Compilation costs ~1 min once per frame size. Accuracy is identical across paths (bf16
    loses nothing).
  - The network is 2.1 TMAC per 2048² frame (31 GMAC per 256²), dense over every pixel.
  - GPU: 0.46 s against CFI v4's 0.49 s (PyTorch GPU). About 0.2 s of it is CPU-side
    (input field fit, contour extraction) and could move to the GPU.
  - CPU: 26.5 s against CFI v4's 1.45 s (ONNX) and the 1–2 s goal, as expected for a
    dense model.
  - CPU levers for later, in rough order of payoff: evaluate only near edges (cascade:
    1/4-res TSDF everywhere, full-res decoder in the band); fewer full-res channels or a
    distilled student; int8. The ±50 px receptive field makes band windows large, so the
    cascade must reuse the low-res features.
- The near-weight recipe (`v5_nearw_ft180`, 0.1475 at 360 min) and the capacity and shift
  probes are open leads for a later round.

- **Human decision (2026-09-30):**
  - The CFI model "only provides an estimate for one point at a time".
  - Go with option B: regress the edge position coherently, with topology upstream.
  - Start with a large GPU model that is fast to train and learn from. Move toward the
    1–2 s CPU goal later.
  - Think through the architectural options first.
  - Keep the best CFI model for GUI integration and testing.
- **Checkpoint for the GUI:** `models/cfi_v4.pt` and `models/cfi_v4-onnx` (= v4_C_w32_180m:
  w32, R13, coarse seed). It is the best val_v4 bscore at an acceptable CPU cost (1.45 s
  median). Held-out scores, first and only test use:

  | test set | model | bscore | missed | extra | loc (px) | far pts/img | sparse sharp loc / missed |
  |---|---|---|---|---|---|---|---|
  | test_v4_full | cfi_v4 | **0.167** | 8.65% | 9.2% | 0.102 | 7.9k | |
  | test_v4_full | shipped cfi_B_seed | 0.245 | 13.7% | 13.4% | 0.151 | 125k | |
  | test_v3_full | cfi_v4 | **0.179** | 3.9% | 18.3% | 0.084 | 17k | 0.072 / 0.39% |
  | test_v3_full | shipped (c3_B_s0) | 0.322 | 5.0% | 51.2% | 0.062 | 92k | **0.041** / 0.66% |

  - Test agrees with val: bscore 0.167 / 0.169.
  - The old-render sharp-edge regression (0.041 → 0.072 px on point-sampled binary steps)
    persists, as flagged in the v4 section.
- **Design** (`PLAN_v3.md`): the options study (seven output representations, four
  backbones, three topology sources) chose a truncated-signed-distance U-Net, **SDFNet**.
  - 7.0M params, 17.9 GMAC per 256² crop, widths 32–256 at strides 1–16.
  - Heads: TSDF (T = 8 px), Laplace uncertainty, direction, and TSDF/4 deep supervision.
  - Inputs: the coarse prior, heavily perturbed, plus the seed channels and the raw
    reading.
  - Dense targets come from the exact GT crossings. Their 0-level is within 0.02 px of
    GT (median; the 1/16 lattice floor).
  - Contours are the bilinear 0-level, densified to 1/16 px. The test's 12 px radius
    disk is within 0.01 px, and straight edges are exact.
  - bench_v4 gained `winding` (predicted / GT crossings within 2 px) and far extras by
    lighting.
- **Locality (human, 2026-09-30):** "do we really need the model to see the entire image? …
  the ESF is very local."
  - GroupNorm made the theoretical receptive field the whole input. SDFNet now uses
    per-pixel channel LayerNorm and 3 levels (48 / 96 / 192, strides 1–4).
  - That gives an RF of ≤ 101 px (±50) and about 95% of influence within ~17 px at init.
    3.0M params, 31 GMAC per 256².
  - The human then set the next ~12 h: physical_v3 data, max ~100 px context, touching
    parts later.
- **Smoke run (12 min, batch 8, eager, permute LayerNorm):**
  - 52 crops/s, GPU-bound at 97%, 4.1 GB.
  - Losses fall smoothly. Crop-eval bscore 0.32 → 0.25 → 0.214 at 12 min, missed 13.8%,
    absent 3.4%.
- **Speed:** a permute-free channel LayerNorm (1-D params) and `torch.compile` give 93–96
  crops/s at batch 8–16 (1.6–3.5 GB), up from 53. The config now compiles at batch 16, and
  the 60-min runs were restarted with it.
- **`v5_sdf_60m`** (local SDFNet, 60 min, 340k crops) on val_v4_full, against CFI v4
  (`v4_C_w32_180m`, 3× the training):

  | | v5_sdf_60m | cfi_v4 |
  |---|---|---|
  | bscore | 0.190 | 0.169 |
  | missed / extra | 11.4% / 8.8% | 7.0% / 9.7% |
  | loc mean / p95 (px) | 0.118 / 0.340 | 0.109 / 0.317 |
  | **far extras per image** | **1.4k** | 6.9k |
  | winding | 0.99 | |
  | aligned missed / \|off\| | 3.0% / 0.127 | 0.8% / 0.095 |
  | shift 0.5–2 missed / absent | 23.6% / 1.8% | 16.9% / 0.9% |
  | shift 2–8 missed / absent | 35.8% / 13.0% | 19.5% / 7.0% |
  | medium blur missed | 12.3% | 3.7% |
  | sphere missed | 12.5% | 3.4% |

  - Far extras by lighting (v5): ambient 1.1k, balanced 1.6k, reflective 0.26k, silhouette
    1.7k per image.
  - **Hallucination and winding goals are met already** (5× fewer false contour points, no
    doubling). Precision is not, yet.
  - **Diagnosis** (val 3 and 13, spheres, medium blur, "defocused" camera focus, part
    plateau 0.2–0.27 of backlight):
    - The edge is an ordinary symmetric blur ramp. Its 50% point is on GT (±0.05 px), and
      the coarse prior agrees.
    - v5 puts the 0-level at the ramp's foot, where intensity first leaves the grey
      plateau: 1.6 px inside on val 3 and 0.9 px on val 13. That is the depth-shoulder rule
      applied to a defocus ramp.
    - With the prior heavily perturbed, the model must tell the two ramp shapes apart
      itself, and at 60 min it has not. CFI v4 (lightly perturbed prior, ±13 px) gets both
      right.
    - Crop evals were still improving fast (bscore 0.276 → 0.235 → 0.203 at 15 / 30 / 45
      min).
  - **`v5_occ_60m`, the representation ablation** (the same local backbone, occupancy BCE
    head, contour from the 1× logit's 0-level):

    | | v5_sdf_60m | v5_occ_60m |
    |---|---|---|
    | bscore | **0.190** | 0.225 |
    | missed | **11.4%** | 13.9% |
    | loc (px) | **0.118** | 0.146 |
    | far extras per image | **1.4k** | 3.0k |
    | winding | 0.99 | 0.99 |
    | aligned missed / signed | 3.0% / −0.012 | 2.4% / −0.048 |
    | shift 0.5–2 missed | **23.6%** | 29.0% |
    | shift 2–8 missed | **35.8%** | 55.3% |
    | none missed | **26.6%** | 30.9% |

    - The TSDF beats occupancy on everything except the aligned hit rate. That includes
      half the hallucinations and far fewer misses on depth ramps, which supports the
      human's diagnosis (point-wise decisions).
    - Part of the occupancy deficit is subpixel decoding: a saturating 1× logit is not
      linear in distance. CFI's 16× block decoder avoids that, and the TSDF gets it free.
  - **`v5_sdf_180m`** (base recipe, 180 min, ~1M crops): **bscore 0.1635, better than CFI
    v4 (0.1686)**.

    | | cfi_v4 | v5_sdf_60m | v5_sdf_180m |
    |---|---|---|---|
    | missed / extra | **7.0%** / 9.7% | 11.4% / 8.8% | 8.4% / **6.5%** |
    | loc mean / p95 (px) | **0.109** / **0.317** | 0.118 / 0.340 | 0.112 / 0.323 |
    | far extras per image | 6.9k | 1.4k | **0.9k** |
    | winding | | 0.99 | 0.99 |
    | aligned missed / \|off\| | **0.8% / 0.095** | 3.0% / 0.127 | 1.3% / 0.105 |
    | shift 0.5–2 | **16.9 / 0.9** | 23.6 / 1.8 | 18.6 / 1.0 |
    | shift 2–8 | **19.5 / 7.0** | 35.8 / 13.0 | 25.2 / 8.0 |
    | no visible edge | **23.5 / 17.2** | 26.6 / 17.5 | 30.1 / 13.9 |
    | medium blur missed | **3.7%** | 12.3% | 6.0% |
    | sphere missed | **3.4%** | 12.5% | 7.5% |
    | machined missed | **14.5%** | 22.3% | 17.1% |
    | turned missed | 14.2% | 16.8% | **13.6%** |
    | contaminant boundary missed | **24.6%** | 35.4% | 31.6% |

    Shift rows are missed / absent rates (%).

    - Training time is the strongest lever so far: 60 → 180 min took bscore from 0.190 to
      0.164, and crop evals were still falling at the end.
    - Remaining gaps against CFI: sharp-edge precision (0.105 against 0.095 px), medium-blur
      spheres (val 3 still 22% missed, inward), the ambient sphere val 1 (35% against 10%),
      and turned end faces (val 72).
    - Wins: the thin pin val 4 (1% against 60%) and val 6 (19% against 34%), and far fewer
      false contours everywhere.
    - `train.init` (warm start, each run ≤ 3 h, as for the coarse fine-tunes) added. Queued
      `v5_sdf_ft180`: v5_sdf_180m plus 180 min at peak lr 4e-4, ahead of `v5_sdf_nearw_180m`.
  - **`v5_sdf_prior_light_180m`** (coarse prior perturbed as for CFI: jitter 1 px, morph 1,
    drop 5%, scale 0.7–1.4):
    - bscore 0.1623 (base 0.1635), missed 8.2%, extra 6.1%, loc 0.113, far 0.97k/img.
    - Shift bins (missed rate): aligned **1.7%** (base 1.3%), 0.5–2 18.0%, 2–8 24.8%, none
      **24.8%** (base 30.1%).
    - Per image it is mixed: it fixes val 13, 1, 111 and 72, but val 3 goes from 22% to 75%
      missed.
    - A wash overall, with no fix for the blur/shoulder confusion. Relying more on the
      prior also means leaning on the coarse model for geometry, against the foundation
      model aim. The base keeps the heavy perturbation.
  - Queued `v5_sdf_w64_180m` (capacity probe, widths 64/128/192, same RF) after
    `v5_sdf_nearw_180m`.
  - **`v5_sdf_ft180`** (warm start from v5_sdf_180m, +180 min at lr 4e-4; 360 min, ~2M
    crops in all): **bscore 0.1537, the best model so far.**

    | | cfi_v4 | v5_sdf_180m | v5_sdf_ft180 |
    |---|---|---|---|
    | missed / extra | 7.0% / 9.7% | 8.4% / 6.5% | 7.2% / **5.4%** |
    | loc (px) | **0.109** | 0.112 | 0.112 |
    | far extras per image | 6.9k | 0.9k | **0.84k** |
    | winding | | 0.99 | 0.99 |
    | aligned missed / \|off\| | 0.8% / **0.095** | 1.3% / 0.105 | **0.8%** / 0.098 |
    | shift 0.5–2 | 16.9 / 0.9 | 18.6 / 1.0 | **16.7** / 0.9 |
    | shift 2–8 | **19.5 / 7.0** | 25.2 / 8.0 | 21.3 / 7.5 |
    | no visible edge | **23.5** / 17.2 | 30.1 / 13.9 | 25.7 / **12.7** |
    | heavy / medium blur missed | 15.7 / **3.7%** | 17.7 / 6.0% | 16.4 / 4.1% |
    | machined / turned / sphere missed | **14.5** / 14.2 / **3.4%** | 17.1 / 13.6 / 7.5% | 15.6 / **11.3** / 4.8% |

    Shift rows are missed / absent rates (%).

    - Misses are now at CFI's level, with 8× fewer false contours and 45% fewer extras.
      The model is still training-limited.
    - Queued `v5_sdf_ft2_180m`: a second warm start at lr 3e-4, ahead of the capacity
      probe.
  - **`v5_sdf_ft2_180m`** (second warm start, +180 min at lr 3e-4; 540 min in all):
    **bscore 0.1456. It beats CFI v4 on nearly every slice.**

    | | cfi_v4 | v5_sdf_ft180 | v5_sdf_ft2_180m |
    |---|---|---|---|
    | missed / extra | 7.0 / 9.7% | 7.2 / 5.4% | **6.3 / 4.65%** |
    | loc mean / p95 (px) | **0.1085** / 0.317 | 0.112 / 0.321 | 0.110 / **0.316** |
    | far extras per image | 6.9k | 0.84k | **0.67k** |
    | winding | | 0.99 | 0.99 |
    | aligned missed / \|off\| | **0.8** / **0.095** | 0.8 / 0.098 | 0.9 / 0.096 |
    | shift 0.5–2 missed / \|off\| | 16.9 / 0.267 | 16.7 / 0.265 | **14.6 / 0.247** |
    | shift 2–8 missed / absent / \|off\| | 19.5 / 7.0 / 0.275 | 21.3 / 7.5 / 0.286 | **16.4 / 5.4 / 0.251** |
    | no visible edge missed / absent | **23.5** / 17.2 | 25.7 / 12.7 | 23.7 / **12.9** |
    | heavy / medium blur missed | 15.7 / 3.7 | 16.4 / 4.1 | 15.8 / **3.5** |
    | machined / turned / sphere missed | 14.5 / 14.2 / **3.4** | 15.6 / 11.3 / 4.8 | **12.4 / 9.8** / 4.5 |
    | ambient / balanced / silhouette missed | 8.5 / 6.7 / 7.1 | 8.6 / 5.8 / 8.0 | **7.7 / 4.7 / 6.9** |
    | contaminant boundary missed | 24.6 | 27.8 | **24.5** |

    - Sharp edges are at parity with CFI.
    - Per image: val 3 is fixed (2% missed). Val 1, the ambient sphere, is 24% against CFI's
      10%, and val 25, the heavy-blur sphere, is 36% against 22%.
    - Still training-limited: 0.1635 → 0.1537 → 0.1456 per +180 min.
    - Queued `v5_sdf_ft3_180m` (lr 2.5e-4) after `v5_nearw_ft180`.
  - **`v5_sdf_nearw_180m`** (near-edge loss weight 1 + 16·exp(−|d|) instead of 1 + 4·…;
    180 min, compared with v5_sdf_180m):

    | | base 180m | nearw 180m |
    |---|---|---|
    | bscore | 0.1635 | **0.1602** |
    | missed / extra | 8.4 / 6.5% | **7.7 / 6.0%** |
    | loc | 0.112 | 0.113 |
    | far extras per image | **0.94k** | 1.80k |
    | aligned missed / \|off\| | 1.3% / 0.105 | 1.1% / 0.102 |
    | shift 0.5–2 missed / absent | **18.6 / 1.0** | 19.3 / 1.6 |
    | shift 2–8 missed / absent | 25.2 / 8.0 | **18.4 / 5.2** |
    | none missed / absent | 30.1 / 13.9 | **21.6 / 12.2** |

    - **Depth ramps improve a lot.** 2–8 px misses are below CFI's 19.5% for the first
      time, and "no visible edge" (bores) is below CFI's 23.5%.
    - Per-image missed rates (base → nearw): val 1 34.5 → 14.2%, val 72 67 → 42%, val 3
      22 → 17%, val 13 5.7 → 1.9%.
    - Cost: false contours double (balanced and silhouette machined parts: val 47, 56, 45,
      8), though they stay 4× below CFI.
    - Queued `v5_nearw_ft180`: this recipe +180 min, to compare with v5_sdf_ft180 at 360
      min each.
    - **`v5_nearw_ft180`** (360 min):
      - bscore **0.1475**, against v5_sdf_ft180's 0.1537 at the same budget and
        v5_sdf_ft2_180m's 0.1456 at 540 min. Near weighting is worth roughly +180 min of
        base-recipe training.
      - Missed 6.3%, extra 4.7%, loc 0.111. Far extras 1.1k interior plus 110 near the
        frame per image (ft2: 0.67k + 33).
      - Shift bins, missed rate: aligned **0.79%** (best yet), 0.5–2 15.2%, 2–8 **15.8%**,
        none **22.2%** (both best yet).
      - Queued `v5_nearw_ft2_180m` (this chain +180 min) after `v5_sdf_ft3_180m`, to
        compare the two chains at 540 min.
  - **Tractability probes** (v5_sdf_ft2_180m, no retraining, 40 or 30 val images; asked
    by the human: "is this just asking the model to guess?"):
    - **Noise floor.** The same scenes read noise-free (full well 1e9, no read noise, 16
      bit) barely change the error: GT → prediction mean (clipped 3 px) / missed:

      | bin | noisy | noise-free |
      |---|---|---|
      | aligned | 0.111 / 1.0% | 0.107 / 0.5% |
      | shift 0.5–2 | 0.261 / 14.4% | 0.257 / 13.9% |
      | shift 2–8 | 0.514 / 23.5% | 0.483 / 20.9% |
      | none | 0.533 / 17.1% | 0.512 / 16.0% |

      Sensor noise is not the limit.
    - **What else could limit simple edges?** The renderer's lens model is minimal
      (`synth/worker_v3.apply_optics`). A shift-invariant Gaussian PSF: diffraction σ =
      0.21λ/NA, combined with an "aberration" σ of 0–0.6 px that is only extra Gaussian
      width, elliptical (aspect ≤ 1.35) in 30% of scenes. Plus a broad veiling-glare
      halo, and vignetting. No distortion, flare or chromatic effects.
      - A symmetric PSF keeps the 50% point on a straight edge. Glare only adds a smooth
        pedestal that the model can see and correct.
      - The remaining render-side terms are Cycles Monte Carlo noise (~0.6% of edge
        contrast), a tiny curvature bias, and the metric's ~0.015 px floor.
      - So on simple edges the remaining error is mostly the model's, which leaves real
        headroom. CFI reached 0.041 px on the old renders' sharp edges.
    - **The simplest edges** (the clean render's naive 50% edge within 0.1 px of GT, 234k
      points):
      - v5 mean / median / p90 0.072 / 0.058 / 0.146 px; CFI v4 0.068 / 0.054 / 0.136.
      - The point-set metric's floor is ~0.015 (the GT crossings are discrete).
    - **Self-knowledge.** The predicted Laplace scale b at GT points rises with ambiguity
      (median / p90): aligned 0.069 / 0.17, shift 0.5–2 0.10 / 0.38, shift 2–8
      **0.37 / 0.94**, none 0.17 / 0.35.
      - Actual error grows monotonically with b: b < 0.08 → mean 0.11, 0.9% missed;
        0.17–0.30 → 0.25, 9.9%; 0.44–0.93 → 0.54, 34%; > 0.93 → 1.36, 69%. Spearman 0.35.
      - Roughly calibrated above b ≈ 0.3, overconfident on easy edges (b ≈ 0.07 against
        error ≈ 0.11). The uncertainty head is a usable "this edge is a guess" flag.
  - **Contour smoothing (no retraining) is negative.** Gaussian smoothing of the
    v5_sdf_180m field before contouring:
    - σ 0.7 → bscore 0.174; σ 1.0 → 0.188; σ 1.5 → 0.208; smoothing scaled by the
      uncertainty → 0.197; raw → 0.1635.
    - It hurts most on 2–8 px ramps (25% → 33–50% missed), and even aligned |off| goes
      0.105 → 0.108.
    - The predicted field is not symmetric-linear around the edge (the slopes differ on
      the two sides of ramps), so symmetric smoothing moves the 0-level. Keep the raw
      field.
  - **Failure review** (`failures_v4`, v5_sdf_180m, `work/review/v5_sdf_180m_failures/`):
    - False edge length 47 px/image against CFI's 343. Images with a false run ≥ 5 px: 30
      against 57 of 120.
    - The absent rate is the same (1.4% against 1.3%). Offsets 0.5–2 px: 7.0% against 5.7%
      of edge length.
    - The remaining offset runs are the same as CFI's: a thin grey strip outside sharp
      in-focus edges of machined and turned parts, with the prediction on the strip's
      outer side, ~1 px out.
    - A defocused thin lip at another depth looks much like that optical shoulder, so this
      is plausibly a genuine multiple-solution case. Contours stay smooth (no winding).
  - Next:
    - `v5_occ_60m` (representation ablation, done above);
    - `v5_sdf_180m` (time);
    - `v5_sdf_prior_light_180m`: CFI-level prior perturbation, testing whether distrusting
      the prior causes the bias.
- **Queued:**
  - `v5_sdf_smoke` (12 min);
  - `v5_sdf_60m`;
  - `v5_occ_60m`: the same backbone with an occupancy head, to separate representation
    from context;
  - `v5_sdf_180m` (fallback).
  - The CFI combination run `v4_C_w32_focusw_shift_180m` was cancelled.
- **Last CFI result, `v4_C_r25_shift_180m`** (R25 + shift crops):
  - bscore 0.1746, missed 6.92%, far 5.4k/img (best).
  - Shift bins, missed rate: aligned 0.9%, 0.5–2 **14.7%** (best; the two levers stack),
    2–8 23.2% (worse), none 21.9%.
  - Narrow shoulders improve; wide ramps do not. This confirms the plateau.

## Fine model v4: physical_v3 data + coarse seed (2026-09-29, in progress)

Human request: train the fine model on the new data (`physical-v3-30gb-2048`,
DATA_V3.md) with the new coarse model (`models/coarse_v2.pt`, COARSE.md).

**Pipeline (infrastructure change; new benchmark name):**
- **`cache_v4`** (`work/cache_v4`, 112 GB on T7): 7,275 scenes and 4,522 unique masks,
  keyed by `mask_key`, since tilted and contaminated scenes have their own. Same layout
  as the v3 cache; `sampler.mask_of` picks the mask. Built in 43 min.
- **Coarse seed** (`seed.coarse_seed_full` / `coarse_crop_seed`). The fine model is
  unchanged (3 inputs, seed mode), but:
  - the mask channel is 2p − 1 of the upsampled coarse_v2 part probability, after
    hysteresis;
  - the backlight field is fitted only to the coarse model's confident background
    (`fit_field_masked`), so large or bright parts cannot bias it;
  - band = coarse band (±8 px) ∪ the classical gradient band. The gradient band covers
    sub-resolution parts and loose contaminants the coarse model leaves out.
  - Training crops use per-scene precomputed coarse maps (`work/cache/coarse_maps/physical_v3`)
    with ±1 px shift and ×0.7–1.4 logit jitter.
- **Data stream:** `crops.ScenePool`, 16 scenes per worker in RAM, one swapped every 48
  crops. The 61 GB of images exceed the page cache, so random crop reads over USB are
  avoided.
  - Throughput fixes: pooled mask arrays and binned boundary points, float32 flat-gather
    bicubic, no `np.linalg` per crop, and one OpenCV thread set in the parent, since
    setting it in forked workers deadlocks.
  - Result: 250–280 crops/s with ~1% data wait, at 14 workers.
- **Augmentation (DATA_V3):**
  - full well 6k–60k e⁻;
  - ISP unsharp mask (p 0.2, σ 0.5–1.5, amount 0–0.6);
  - JPEG (p 0.1, quality 75–95, 8-bit);
  - area downscale (p 0.1, ×1–2, folded into the crop matrix so GT stays exact);
  - `contaminant` crop category, 8% (boundaries where the scene mask differs from the
    base mask).
- **`val_v4_full`** (`benchmarks/val_v4_full.json`): 4 scenes for each of the 30 val
  parts, 120 images. Slices: blur class, tilt, contamination, lighting, backlight, glare,
  family, and contaminant vs part boundary. Reports the **signed offset** (+ = outside)
  of matched points.
- Config: `configs/cfi_C_coarse.json` (the cfi_B_seed recipe plus the above).

**Results (val_v4_full, 120 images):**

| model | missed | extra | loc (px) | bscore | signed (px) | far pts/img frame / interior |
|---|---|---|---|---|---|---|
| cfi_B_seed (shipped; classical seed) | 15.8% | 13.3% | 0.148 | 0.250 | +0.079 | 1958 / 69,766 |
| v4_C_coarse_60m | **7.9%** | **10.9%** | **0.111** | **0.181** | **+0.002** | **1 / 7,919** |

By slice, v4_C_coarse_60m against the shipped model:
- heavy blur, missed: 19.0% against 34.7%;
- ambient light, bscore: 0.190 against 0.308;
- machined parts, missed: 17.5% against 28.7%;
- contaminant boundaries, missed: 32% against 47%.

The outward bias, the headline defect in DATA_V3, is gone: signed +0.079 → +0.002 px.
In-focus −0.006 / +0.006 px; mild −0.027 / −0.009; heavy +0.005 / +0.018
(median / mean).

**Remaining far extras** (`edgesr.review_v4`, `work/review/v4_C_coarse_60m_v4_far/`):
- doubled contours 2–5 px outside the blurred end faces of turned and machined parts;
- internal groove edges on grey parts under balanced or ambient light, even though the
  coarse channel says "inside".

**Regression on the old renders** (val_v3_full, 60 images, coarse seed at inference):

| model | missed | extra | loc (px) | bscore | sparse sharp loc / missed | sparse medium loc / missed | far pts/img interior |
|---|---|---|---|---|---|---|---|
| c3_B_s0_60m (shipped) | 3.9% | 32.8% | 0.068 | 0.231 | **0.049** / 0.77% | 0.093 / 5.6% | 35,033 |
| v4_C_coarse_60m | 4.7% | **11.3%** | 0.090 | **0.153** | 0.078 / 0.88% | 0.101 / 7.8% | **6,660** |

- The old renders are point-sampled binary steps with no lens PSF. The v4 model learned
  physically blurred edges (box pixel integration plus PSF, σ ≈ 0.7 px in focus), which
  DATA_V3 considers closer to the real camera.
- So sharp-edge localization on the **old** data regresses: 0.049 → 0.078 px. DATA_V3
  asked that this not be traded away silently: it is flagged here and for the human.
- False contours drop 5× (far extras 35k → 6.7k per image), and bscore improves overall.

**v4_C_coarse_180m** (same recipe, 180 min):

| benchmark | metric | 60 min | 180 min |
|---|---|---|---|
| val_v4_full | bscore | 0.181 | **0.172** |
| val_v4_full | missed | 7.9% | 7.5% |
| val_v4_full | extra | 10.9% | 9.8% |
| val_v4_full | loc (px) | 0.111 | 0.109 |
| val_v4_full | signed mean / median (px) | | +0.004 / −0.010 |
| val_v4_full | far interior pts/img | 7.9k | 6.8k |
| val_v3_full | bscore | 0.153 | 0.143 |
| val_v3_full | sparse sharp loc (px) | 0.078 | 0.076 |
| val_v3_full | far interior pts/img | 6.7k | 4.7k |

- The 3× longer schedule gains ~5%, spread over all slices: diminishing returns.
- The weakest val_v4 slices at 180 min:
  - machined: missed 17.1% / extra 22.5%;
  - turned: missed 14.5%;
  - heavy blur: missed 18.7%;
  - contaminant boundaries: missed 30%.
- **The "doubled contours" at turned/machined end faces are depth defocus, not a label
  error** (val_v4 image 8, machined, balanced, "mild").
  - The end-face rim of a part lying on its side spans the full part height in depth, so
    its silhouette is a sum of different blurs: a one-sided ramp more than 10 px wide.
  - The geometric edge, the GT, sits at the dark foot of the ramp (reading 0.25, rising
    to 0.42 over +17 px).
  - The model puts a wavering contour near the ramp's 50% point, 7 px outside. The
    cues for the true edge are in the ramp's shape. This is inherently hard; more
    training or capacity may help.
- **`v4_C_anyw_180m`** (seed 1; anywhere crops 0.25, uniform weight ×2):
  - val_v4: bscore 0.177, missed 7.35%, extra 11.1%, far interior 7.5k per image.
  - val_v3: bscore 0.140, sparse sharp loc 0.073.
  - Not better than the base 180-min run on val_v4, and the far extras did not drop. The
    far extras sit next to parts (end faces, grooves), not in open background.
  - `v4_C_coarse_180m_s2` (base recipe, seed 2) is queued to measure the seed variance
    of these ~0.005 bscore differences.
- **`v4_C_w32_180m`** (fine width 32, else the base recipe):
  - val_v4 bscore 0.169 against 0.172 (missed 6.96% / 7.52%, extra 9.70% / 9.77%, loc
    0.1085 / 0.1093, far 6.9k / 6.8k); val_v3 bscore 0.138 against 0.143.
  - The gains sit in the slices the new data made hard (missed, w24 → w32):
    - heavy blur 18.7 → 15.7%;
    - machined 17.1 → 14.5%;
    - heavy contamination 13.2 → 10.7%;
    - contaminant boundaries 30.2 → 24.6%.
  - In-focus and ambient are about unchanged. This looks like a real capacity limit on
    the harder data.
  - Cost: about 1.8× the fine-stage FLOPs on CPU. To be measured on an idle machine
    against the ~1 s budget.
- Next (earlier plan): `v4_C_anyw_180m` (more anywhere/uniform crops against far extras), then
  `v4_C_w32_180m` (fine width 32: is the harder data capacity-limited? A CPU latency
  cost).

**Seed variance and width (2026-09-30).** `v4_C_coarse_180m_s2` (w24, seed 2):
- val_v4 bscore 0.1738, missed 7.50%, extra 10.3%; val_v3 bscore 0.139.
- It ran about 7% fewer steps than seed 0 (168k), because of CPU contention from review jobs.
- The w24 seed pair is 0.172 / 0.174 bscore and 7.52 / 7.50% missed. w32 (0.169, 6.96%)
  is clearly above that noise on missed rate, so `v4_C_w32_180m_s2` was stopped at 8 min.

**CPU latency (`edgesr.latency_v4`, idle machine, 16 threads, ONNX, 8 val_v4 frames of
2048², median):**

| model | points path | seed / fine / decode / canvas (s) | p90 (s) | rings (s) |
|---|---|---|---|---|
| cfi_B_seed (shipped, classical seed) | 1.42 | 0.09 / 0.71 / 0.06 / 0.37 | 21.0 | 3.0 |
| v4 w24 (coarse seed) | 1.23 | 0.19 / 0.55 / 0.05 / 0.30 | 2.7 | 3.0 |
| v4 w32 (coarse seed) | 1.45 | 0.19 / 0.73 / 0.06 / 0.30 | 3.2 | 3.5 |

- The coarse seed's smaller band makes v4 w24 faster than the shipped model and removes
  its 21 s tail, which came from speckle bands.
- `rings` (the GUI path) is dominated by tracing. See the runtime fixes below.

**Failure review (`edgesr.failures_v4`, w32, val_v4_full; sheets and summary.md in
`work/review/v4_C_w32_180m_failures/`).** bench_v4's "missed" counts everything beyond
0.5 px. Split by distance:

| | typical (72 images) | hard (48 images: heavy blur, ambient or heavy contamination) |
|---|---|---|
| GT 0.5–2 px from the prediction | 4.2% of edge | 7.8% |
| GT > 2 px (absent) | 0.9% | 1.8% |
| false (> 2 px from GT) | 272 px/image | 450 px/image |

- **Typical scenes** are dominated by specific geometry, not by the sensor or lighting:
  - **Depth-of-field edges of thick parts:** machined and turned parts on their side are
    2–3 mm deep, and at NA 0.057 the "depth blur" reaches 5–60 px. The image shows a ramp or
    a grey shoulder outside the geometric edge. The model follows the visible transition,
    0.7–1.5 px (shoulders) to 3–6 px (ramps) outside it.
  - **Deep narrow bores in spheres:** a dim spot smaller than the geometric hole. The hole is
    missed, or traced about 3 px inside.
  - **Other:** a 1.5 px pin, and thread valleys under blur.
  - Real hallucinations are rare (a reflective highlight arc, one internal groove).
- **Hard scenes:**
  - False edges come mostly from ambient light (front-lit surfaces read as background).
  - Absent edges are mostly contaminant fibers, several below the 50 px² filter.
- **Human's priorities (2026-09-30):**
  1. **Depth-of-field edges of thick parts.** The parts are deep relative to the DOF, and
     the renders and GT are realistic.
  2. **Deep narrow holes** (lower).
  - Small fibers do not matter (area filter). Ultra-thin out-of-focus wires are partly
    unrealistic and not a priority.

**New slice `edge_shift` in bench_v4** (additive; existing metrics unchanged):
- GT part-boundary points are binned by their distance to the naive visible edge
  (`visible_shift`: the 0.5 level of the clean render between the dark level and the
  backlight field): aligned ≤ 0.5 px, 0.5–2, 2–8, none within 8.
- Reports missed, absent (> 2 px) and the signed GT → prediction offset of found edges.
- Check on 10 sheet images (w32):
  - aligned points: missed 0–0.4% on 8 of 10 images, but 29.6% on val 78;
  - thick parts: 9–84% of points shifted, 17–76% of those missed.
- This is the metric for the priority work.
- Baselines on val_v4_full (part boundary; absent = > 2 px; |off| and signed are over found
  edges, ≤ 2 px):

  | bin (share of GT) | w24 s0 missed / absent | w24 s2 | w32 | w32 \|off\| / signed (px) |
  |---|---|---|---|---|
  | aligned (68%) | 0.9 / 0.1% | 1.0 / 0.1% | 0.8 / 0.0% | 0.095 / +0.001 |
  | shift 0.5–2 (21.5%) | 17.7 / 0.9% | 17.3 / 1.0% | 16.9 / 0.9% | 0.267 / +0.062 |
  | shift 2–8 (7.3%) | 21.1 / 7.9% | 21.9 / 8.7% | 19.5 / 7.0% | 0.275 / −0.027 |
  | no visible edge (2.7%) | 24.4 / 18.0% | 23.0 / 17.9% | 23.5 / 17.2% | 0.213 / +0.096 |

  - Shipped cfi_B_seed (classical seed), missed by bin: aligned 4.4%, shift 0.5–2 28.5%,
    shift 2–8 54.6%, no visible edge 55.3%. v4 roughly halves the shifted misses already.
  - **Shifted edges account for ~91% of all missed part-boundary points** (w32: 657k of
    723k).
  - The model already beats the naive detector there. On 2–8 px shifted edges, ~80% of GT
    is found within 0.5 px, where the naive visible edge is 2–8 px away by construction.
  - The remaining misses are the target. Seed noise on shift 0.5–2 is ~0.4 pt.
- `candidates_scene.npz`: 6,839 of 7,275 scenes have shift candidates. The shifted share of
  the part boundary is 17.7% on average, median 2.8%, p90 67%. Smoke test: the category
  draws 13.7% of crops (expected 14.2%) at normal speed.
- R25 costs training speed: 147–220 crops/s against ~260 for R13, since the GPU is the
  bottleneck. In the same 180 min it sees ~60–80% of the crops.

**`v4_C_r25_180m` result** (w24, R25; 158k steps against 168k for R13):

| run | bscore | missed | aligned | shift 0.5–2 | shift 2–8 | no visible edge | machined / turned / heavy blur |
|---|---|---|---|---|---|---|---|
| w24 R13 s0 / s2 | 0.172 / 0.174 | 7.52 / 7.50% | 0.9 / 1.0% | 17.7 / 17.3% | 21.1 / 21.9% | 24.4 / 23.0% | 17.1 / 14.5 / 18.7%, 16.1 / 15.3 / 18.0% |
| w32 R13 | 0.169 | 6.96% | 0.8% | 16.9% | 19.5% | 23.5% | 14.5 / 14.2 / 15.7% |
| **w24 R25** | 0.170 | 6.93% | 0.8% | **16.1%** | 21.4% | 23.1% | 14.5 / 13.7 / 16.1% |

The shift columns are missed rates.

- val_v3 (old renders): bscore 0.138, sparse sharp loc 0.0736. That is about w32 and
  slightly better than w24 (0.143 / 0.076).
- The wider receptive field helps the **narrow shoulders** (shift 0.5–2: −1.4 pt against the w24
  mean; seed noise ~0.4). It does **not** help the wide ramps (2–8 px) or the edges with no
  visible transition.
- So the 2–8 px misses are not a plain context-size limit. Next: the shift crop category
  (running), then R25 + shift (`v4_C_r25_shift_180m`, fallback, cancel if shift does not
  help).
- Latency: R25 widens the gathered window from 48 to 72 px, about 2.25× fine-stage compute.
  That would be roughly 1.8 s per 2048² image on the CPU, against 1.23 s.
- Where the shifted misses are (w32, val_v4_full):
  - By family: machined 47%, turned 28%, freeform 11%.
  - By lighting: silhouette 43%, ambient 34%, balanced 17%.
  - By blur class: in-focus 49%, mild 22%, heavy 21%.
  - They are concentrated: 12 of 120 images hold 50% and 24 hold 80%. The worst are val
    65 (machined, ambient), 78 (freeform, silhouette), 4 (the 1.5 px pin, not a priority),
    72 (turned, silhouette), and several machined silhouette scenes (11, 16, 17, 46).
  - With only 6 val parts per family, per-part geometry drives the slice. Read changes of
    about 1 pt with care.

**`v4_C_shift_180m`** (w24, shift crop category):
- bscore 0.1728, missed 7.29%, far 8.1k/img.
- Shift bins, missed rate: aligned 0.9%, 0.5–2 16.4%, 2–8 22.6%, none 24.4%.
- Like R25, it helps the narrow shoulders (−1.1 pt) and not the 2–8 px ramps. The
  signed bias on 0.5–2 drops from +0.063 to +0.048.

**Is the fine model anchored to the coarse boundary?** w32 on 14 shifted-heavy val images,
coarse channel replaced:

| coarse channel | aligned | shift 0.5–2 | shift 2–8 | none |
|---|---|---|---|---|
| real | 3.8% | 36.4% | 38.0% | 59.1% |
| oracle (GT coverage at 512) | 3.8% | 34.7% | 37.7% | 54.8% |
| boundary 3 px out | 6.1% | 38.8% | 54.4% | 78.4% |
| boundary 3 px in | 10.0% | 36.3% | 39.4% | 59.1% |

- On full val, fine misses on 2–8 px edges are 1% where the coarse 0.5 contour is within
  1 px of GT and 33% where it is not.
- That is mostly common cause: an oracle coarse channel barely helps. The fine model does
  follow a badly placed coarse boundary (3 px out: +16 pt), but its residual misses are
  intrinsic. Some images stay at 90–100% missed whatever the coarse channel says.

**Shifted misses by the renderer's camera focus class** (w32; the cache's `focus_class`
is the blur class, so `camera_focus_class` was added to scenes.json and cache_v4):

| camera focus class | shift 0.5–2 missed | shift 2–8 missed | part missed |
|---|---|---|---|
| sharp | 3.6% | 0.8% | 1.4% |
| defocused | 7.5% | 6.4% | 3.3% |
| moderate | 22.9% | 11.7% | 5.0% |
| strong_defocus | 17.9% | 32.2% | 12.3% |
| **depth_split** | 21.0% | **36.2%** | 8.5% |

- `depth_split`, where the focal plane cuts through the part's depth, holds the wide-ramp
  misses. That is the human's case: parts deep relative to the DOF.
- The image then mixes an in-focus step, where the focal plane meets the wall, with a
  defocused shoulder from the rest of the wall. GT is at the step, which can sit below the
  50% level.
- In training, depth_split is 1,887 of 5,821 scenes (32%).
- Queued `v4_C_focusw_180m`: scene weights depth_split ×3 and strong_defocus ×2
  (`focus_weights` with `_key: camera_focus_class`). The smoke test raises depth_split
  from 23% to 50% of crops.
- Fixed `sampler._scene_index`: its cache key was the `id()` of the pool's `eligible`
  dict, which ScenePool rebuilds every swap, and a freed id can be reused. Focus weights
  were unused in the v4 runs, so no earlier result is affected.

**Depth-edge experiments (queued 2026-09-30):**
- `v4_C_r25_180m`: fine receptive field R25 (dilations 2,4,8,5; crop border 28). Ramps are
  5–30 px wide, beyond R13. On val_v3, R13 → R19 was flat, but the old renders had ≤ 3 px
  of blur. Cost: window 72 against 48, about 2.25× fine FLOPs.
- `v4_C_shift_180m`: a `shift` crop category, 0.2/1.2 of crops. Centers come from GT
  boundary points 0.75–8 px from the visible edge, excluding contaminant boundaries.
  `edgesr.shift_candidates` writes them to `work/cache_v4/candidates_scene.npz`, keyed by
  scene, because the shift depends on each scene's optics. `sampler.SCENE_CATEGORIES` is
  the new mechanism.
- Both use w24, to compare with the w24 seed pair.

## FINAL STATE (2026-09-26): stopped at human request

**Selected:** arm B, the classical seed (adapter.py background + Otsu + gradient band,
no learned global model) plus the CFI fine model (dense blocks, R13, width 24, block
decoder, edge head), trained 60 min with 15% anywhere crops.
- Config: `configs/cfi_B_seed.json`
- Checkpoints: `work/runs/c3_B_s{0,1}_60m/final.pt`
- Inference: `edgesr.cfi_infer.infer`; region filter defaults object 50 px², hole 8 px²
- Human verdict: "good enough and very similar to A". Arm A (learned coarse context):
  `configs/cfi_A_coarse.json`, `work/runs/c1_r13_any_60m`, `work/runs/c3_A_s1_60m`.

**Latency** (finished B model, 2048² image, 16 CPU threads, PyTorch, compiled fine
branch): **1.31 s** median with the exact min-area filter (fine 0.70 s, contours+filter
0.43 s, seed 0.09 s); **0.91 s** without the filter. GPU: 0.49 s per image. Patch
reference: 118–190 s CPU, 10.8 s GPU.

**Accuracy**, val_v3_full and test_v3_full (60 full images each; test parts untouched
until this final evaluation). Missed % / mean localization (px) on GT points:

| slice | val ref | val A | val B (s0, s1) | test ref | test A | test B (s0, s1) |
| --- | --- | --- | --- | --- | --- | --- |
| sparse/sharp/all | 0.50 / .0435 | 0.54 / .0431 | 0.89, 0.86 / .049, .048 | 0.18 / .0384 | 0.48 / .0394 | 0.66, 0.70 / .0414, .0412 |
| sparse/sharp/silhouette | 0.07 / .0355 | 0.09 / .0363 | 0.11, 0.13 / .039 | 0.02 / .0317 | 0.07 / .0330 | 0.06, 0.05 / .0356, .0349 |
| sparse/medium/all | 5.62 / .0903 | 5.66 / .0866 | 6.04, 6.23 / .092 | 6.35 / .0918 | 6.53 / .0852 | 6.93, 7.65 / .088, .092 |
| sparse/sharp/ambient | 0.48 / .068 | 0.54 / .062 | 3.4, 3.1 / .092, .085 | 1.92 / .071 | 5.07 / .073 | 6.6, 7.1 / .074, .078 |
| heavy blur | 14.9 / .148 | 9.3 / .124 | 10.3, 12.9 / .14, .15 | 4.6 / .131 | 4.2 / .124 | 5.7, 5.3 / .126, .124 |
| specular | 0.30 / .072 | 1.2 / .066 | 5.6, 3.7 / .09 | 11.6 / .103 | 14.0 / .108 | 22.7, 24.1 / .10, .11 |

- Seed noise is small (A: ~0.0005 px on protected slices). Sharp-edge localization
  generalizes (test B–A gap +0.002 px vs +0.005 on val).
- **Strict gates on test** (loc ≤ +0.005 px; missed/near-extra ≤ +0.5 pt vs the patch
  reference):
  - B passes sparse/sharp/silhouette, sparse/sharp/balanced, sparse/medium/silhouette,
    sparse/medium/reflective.
  - B is marginal on sparse/sharp/all (missed +0.5 pt).
  - B fails on reflective near-extra, medium/all (+0.6–1.3 pt), and the ambient/balanced
    medium slices. A also fails the ambient and balanced-medium slices. The patch
    reference keeps lower missed rates there at ~100× the compute.
- Far hallucination per image (test): ref 132k frame + 265k interior; A 24k + 13k;
  **B 1.8k + 86–92k** (mostly bright-part scenes).

**Known limitations of B:**
- Parts as bright as or brighter than the backlight (ambient-lit light matte,
  shaded spheres): the dark-on-light seed splits or inverts. Whole-part inversion
  (val #45) is outside the adapter's assumption.
- Specular and ambient slices; corners (information-limited).
- Backlight dust ≥ 50 px² is reported as contours (by design).
- The rendered noise is more aggressive than reality (human).

**Packaged for inference** (2026-09-27): `models/cfi_B_seed.pt` (1.07 MB; exported from
c3_B_s0_60m), `python -m edgesr.predict`, README.md "Run the model",
tests/test_predict.py (runs with no dataset or cache).
- **Synthetic check:** on an exact area-coverage disk, the model is unbiased when sharp
  (0.023 px mean |error|). Under symmetric Gaussian blur it shows an **outward bias of
  +0.10 px (σ 0.6), +0.07 (σ 1.0), +0.17 (σ 1.5)**, and the same with point sampling.
  Not the render gamma (symmetric around 1; mean 1.002). Likely the renders' defocus
  of 3-D parts (outermost silhouette from different depths) teaches an inward-shifted
  50% crossing prior, which over-corrects on a 2-D Gaussian blur. **Must be checked on
  real blurred images** before trusting absolute dimensions there.

**If resumed, next steps:**
1. Ordered sub-pixel contour rings (polyline output), with area filtering by
   shoelace, O(contour length). This replaces the 0.43 s exact region filter and gives
   the deliverable format.
2. Latency: ONNX Runtime / OpenVINO, merging adjacent windows, int8 test.
3. B fine model: R19 and width 32 (queued, cancelled). B lacks A's coarse context, so
   wider fine context may matter more for B than it did for A.
4. Real-image validation set.
5. Polarity-robust seeding for bright parts.
6. Edge-field head evaluation (brightfield path).

## Current best (val_v2)

| field | value |
| --- | --- |
| config | `configs/best.json` = baseline_v2 + backlight_input + dilations [1,2,4,8]×2 + dec_blocks [1,1,0,0] + **channel LayerNorm trunk** (pre-norm blocks + trunk output) |
| runs | v10_cn_60m_s0 **0.1354**, v10_cn_60m_s1 0.1392 (seed 2 pending) |
| bscore (val_v2, 60 min) | **0.1373 (2-seed mean)** vs previous best (no norm) 0.1410 ± 0.0073 (3 seeds) |
| missed / extra | ~6.2% / ~10.3% |
| loc error (matched) | mean 0.0735 px, p95 0.278 px |
| component miss / IoU | ~0.37% / 0.990 |
| system | 0.88M params, 400 patches/s, 5.2 GB; activations bounded (trunk ≤ 6) |

Adopted because accuracy is equal or better and training is structurally stable
(survives const lr 2e-3, unlike the unnormalized model, which diverged at 120 min and
with a 96×12 trunk) at −2% speed. Best single run so far: 0.1326 (unnormalized seed 1).

## Budgets

- Calibration: at 20 min, seed 0 scored 0.3254 against 0.2768 at 60 min, so 20 min
  still separates well from 60 min. Seed noise is unknown because seed 1 diverged;
  re-measure on the stable recipe.
- Screening budget: **20 min**. Seed spread on best ≈ 0.007 bscore (2 seeds; a 3rd
  is queued).
- Confirmation budget: **TBD** (candidate: 60 min).
- Seed noise on bscore at the screening budget: **TBD**. Differences smaller than
  about 2× this noise are not evidence.

## Infrastructure change v2 (2026-09-25): backlight value, `val_v2`

- **Human decision:** parts can be brighter than the backlight (speculars, e.g. scene
  2259, and whole polished plates rendering near-white on a dark backlight). The model
  may receive a **backlight brightness value** as an input. How that value is obtained
  (calibration or some estimator) is deliberately outside this project. Do not build
  or bake in an estimator.
- Training/benchmark value: `backlight_level` in scenes.json, the median image value
  over mask-background pixels (ground truth), transformed with the patch's exposure
  gain and offset. Training multiplies it by exp(0.03·N(0,1)) (`aug.backlight_jitter`)
  so the model tolerates an imprecise supplied value. The benchmark uses the exact
  value.
- Bug fixed: exposure augmentation previously normalized by the image 99.5th
  percentile, which exceeds the backlight by > 10% on 5% of scenes (up to 14.7× on
  specular parts). It now normalizes by `backlight_level` (`aug.exposure_ref`).
- `val_v2` = the same 2100 patches as `val_v1`, regenerated with the fixed exposure
  plus per-patch backlight values. `val_v1` still reproduces its hash (its spec lacks
  `exposure_ref`, so it keeps the p99.5 behaviour). **v1 and v2 scores are not
  comparable.** All runs from here use `val_v2` and `configs/baseline_v2.json`.
- Model option `model.backlight_input`: the input becomes [2(x/b − 0.5), 4(b − 0.65)].

## Infra note: input context (`model.context_px`, 2026-09-25)

The model can take (48+2c)² inputs around the same center. It still predicts the
768² mask of the central 48 px, and scoring is unchanged. The trunk runs on the full
input; features are cropped to 48+2·min(c,3) before the decoder, and the output is
cropped to 768². `Benchmark(name, context=c)` uses the same patches, masks, and
backlight values, with inputs extended by c px. Only the noise realization differs
from c=0, because the larger array draws different normals. A unit test checks that
the central 48 px of an extended input equals the c=0 input exactly.

## Pipeline milestones (all passed 2026-09-24)

- Registration. Analytic unit tests pass: lattice patches are exact copies, continuous
  transforms are unbiased to < 5e-4 px, and the test fails on a deliberate 1/32 px
  error. On real sharp silhouette renders, the fitted image↔mask shift is
  median ≤ 0.003 px for lattice and rotated patches, with 0.04 px per-patch MAD
  (`python -m edgesr.registration`).
- Block-sparse mask cache reproduces the raw G4 TIFF tiles bit-for-bit (unit test).
- Benchmark regeneration reproduces the stored content hash.
- Overfit 16 fixed patches in 4 min: bscore 0.019 px, 0 missed, IoU 0.999
  (`work/runs/m1_overfit16`).
- Checkpoint save, hard kill, and resume: the run continued from its last checkpoint
  and completed (`work/runs/m2_resume`).
- GPU feed: after replacing the NumPy mask gather with a cv2 nearest warp
  (1.2 ms/patch, identical except ~1e-7 fixed-point ties), the loader supplies ~1500
  patches/s. The GPU consumes ~205/s, so data wait is ~0.

## Important decisions

1. **Split by geometry, not only by render.** Masks are shared by all appearances of a
   part, so a render-level split would leak geometry. A separate 30-part test set is
   held back.
2. **Validation uses lattice patches only** (no resampling), matching deployment on
   native sensor pixels. Training mixes lattice (p=0.3) and continuous
   rotation/translation (bicubic input, nearest mask).
3. Nearest mask resampling on rotated patches adds ≤ 1/32 px zero-mean jitter to the
   targets, near-axis-aligned in the worst case. Accepted. If the model is ever that
   precise, revisit by supersampling the mask resample.
4. Photometric aug (input only): backlight level U(0.35, 0.95) of full scale, black
   offset U(0, 0.03), Poisson-Gaussian noise (full well 2k–40k e⁻ log-U, read noise
   1e-4–2e-3 log-U), quantization to 8/10/12/16 bits uniformly.
5. GPU: torch.compile + bf16 + channels_last, batch 32. Compile is +33% throughput
   and 2.2× less memory than eager. Throughput is flat for batch ≥ 8, so the model is
   bandwidth-bound in the 384²/768² stages.
6. Primary score `bscore`: a clipped symmetric mean boundary distance, in px.

## Baseline diagnosis (c0_base_60m; `work/runs/c0_base_60m/breakdown.json`)

- Region (topology) errors dominate, not edge placement. Matched points are
  localized to 0.07–0.09 px in every category, yet extra-boundary rates are 24–46% and
  predicted contour points outnumber GT points by ~50%. The overlays show noisy blob
  contours inside flat regions.
- Mechanism 1, receptive field: the trained baseline's effective RF is ~20×20 px
  (gradient > 1e-3 of max), so pixels far from an edge must guess their class from
  absolute intensity. Queued test: dilated trunk (RF 48, same params and FLOPs).
- Mechanism 2, polarity ambiguity: under `ambient` illumination (12% of val
  patches), objects can be brighter than the backlight (e.g. polymer 0.79 vs bg 0.60,
  coated 0.57 vs 0.51). The edge is visible, but which side is the object is not
  determined locally. Ambient bscore 0.587, missed 30%, extra 80%; reflective 0.114;
  silhouette 0.178.
- Strong defocus: bscore 0.560, missed 45%. Blurred edges are often not detected.
- Sharp focus (point-sampled, aliased) is not worse than moderate blur: 0.194 vs
  moderate 0.340, depth_split 0.188.
- Bit depth barely matters (8-bit 0.293 vs 16-bit 0.266).

## Results log (val_v2, 20 min unless noted)

- v2b_base_20m 0.2721: v2 reference, stable recipe.
- v2b_bl_20m **0.1780**: backlight input. Clear win: extra 34→15%, IoU 0.934→0.979,
  ambient 0.587(v1)→0.367. ADOPTED.
- v2b_dil_20m 0.2672: dilated trunk without backlight. Within likely noise of the
  reference, though train loss is lower (0.051 vs 0.082). Re-testing on the best
  config and at 60 min.
- v3_bl_s1_20m 0.1854: best, seed 1. **Seed spread at 20 min ≈ 0.007** (0.178 vs
  0.185); treat differences < ~0.015 as noise.
- v3_bl_dil_20m 0.1785: dilated trunk on best. No effect; RF is not the bottleneck
  once the backlight is given (effective RF result superseded). IoU 0.983 vs 0.979,
  component miss 0.96% vs 1.06%, both within noise.
- v3_bl_w96b12_20m 0.2007: 96×12 trunk (2.4M params). Worse at 20 min; 15% slower
  (178/s), 6.4k vs 7.6k steps. Capacity is not the limiter at this budget.
- v3_bl_resize_20m: DIVERGED at step 1200 (loss 0.085 → 0.697; the guard aborted it).
  Trunk activations stayed ~200–340, so the blow-up was in the decoder. Also 40% slower
  (127/s) and 7.2 GB. Not pursued; pixelshuffle stays. The unnormalized decoder is
  fragile. If a best-config run ever spikes, test decoder normalization or residual
  scaling.
- v3_bl_inskip_20m: crashed (input-skip channel bug with backlight input; fixed in
  90b107f). Re-queued as v3_bl_inskip2_20m.
- v3_bl_lattice_20m 0.1796: lattice-only sampling. No difference, so bicubic
  rotation augmentation neither helps nor hurts on the lattice benchmark.
- v3_bl_s2_20m 0.1810: seed 2. **3-seed best (bl) at 20 min: 0.1815 ± 0.0035.**
- v3_bl_60m 0.1597: bl at 60 min (−0.022 vs 20 min). Training budget matters a lot.
- v3_bl_dil_60m **0.1464**: bl+dil at 60 min, −0.013 vs bl at 60 min (extra
  13.1→10.8%, comp miss 0.73→0.54%). Dilation helps only with the longer budget.
  ADOPTED (1 seed; a 60-min seed repeat is queued as a fallback).
- v3_bl_inskip2_20m 0.1644: input skip on bl. −0.017 vs the 3-seed mean (~5σ) despite
  26% slower training (155/s, 5.6k steps). Extra 11.4% vs ~15.5%, missed 9.6% vs ~8.7%.
  Promising; being tested on bl+dil.
- v4_bd_inskip_20m 0.1716: input skip on bl+dil. −0.007 vs bl+dil 20 min (0.1785),
  ~2σ, a smaller gain than on bl alone. 30% slower. Pending the 60-min confirmation.
- v4_bd_bw4_20m 0.1855: boundary-weighted BCE ×4, r=0.5 px. Primary worse (+0.007), but
  missed 7.4% (vs ~8.7%) and loc 0.077/p95 0.287 are better, while extra 19% (vs ~15.7%)
  is worse. The weighting trades false boundaries for recall and localization. 13%
  slower (max-pool band). Not adopted; a milder ×2, r=0.25 px is queued.
- v4_bd_s1_20m 0.1731: bl+dil seed 1, so the bl+dil 20-min mean is 0.1758.
- v4_bd_focus_20m 0.1751: oversample defocused ×1.5, strong ×3. Primary neutral.
  strong_defocus improves 0.4925→0.4511 (missed 43→39%), but every other focus
  class gets slightly worse. Not adopted.
- v4_bd_inskip_60m 0.1533: input skip at 60 min is WORSE than best (0.1464). 34% slower
  (153/s → 16.8k vs 22.8k steps); IoU 0.990 and comp miss 0.47% are marginally better.
  The 20-min gain did not survive a time-matched longer budget. Rejected.
- v4_bd_decwide_20m 0.1868: decoder 64/48/24/12 is much slower (111/s, 4.0k steps).
  Rejected. **Pattern: under a fixed time budget, throughput dominates; extra
  high-resolution compute costs more steps than it buys.** Next: cheaper final stage
  (v5_bd_nob768).
- v4_bd_bw2r4_20m 0.1926: mild boundary weight ×2, r=0.25 px. Worse than ×4 on the
  primary score (extra 19.5%). Boundary weighting consistently raises false contours.
  **Dropped as a direction** unless combined with something that suppresses extra
  contours.
- v5_bd_ctx8_20m 0.2009 and v5_bd_ctx16_20m 0.2086: input context 8/16 px. Worse (slower
  160/151 per s, extra 20–21%). ctx8 strong_defocus 0.521 vs 0.49, so context does not
  help its target either. Rejected. (Both runs crashed in overlay drawing after training;
  fixed, and recovered with `python -m edgesr.train --run-id ID --finalize`.)
- v5_bd_nob768_20m **0.1675**: drop the 768² residual block. −0.008 vs bl+dil 20-min
  mean (~2.5σ), and +45% throughput (302/s, 10.7k steps), 5.5 GB. Best 20-min result;
  60-min confirmation queued.
- v5_bd_b16_20m 0.1780: batch 16 (2× steps, same samples/s). Neutral, so **samples
  seen, not optimizer steps, is what the budget buys.** Keep batch 32.
- v6_nob768_60m 0.1538: nob768 at 60 min is **worse** than best (0.1464), despite 58%
  more samples (36k steps). nob768 seed 1 at 20 min is 0.1633 (20-min mean 0.1654 vs
  bl+dil 0.1758). The 768² res block costs speed early and pays off later. Not
  adopted. **Lesson: 20-min screens can mislead on speed/quality trade-offs; confirm
  them at 60 min.**
- v6_nob384_20m **0.1563**: drop both the 384² and 768² res blocks. 410/s (+96%),
  14.7k steps. Best 20-min result by far, but nob768 reversed at 60 min, so a 60-min
  confirmation is queued.
- v7_gn_20m 0.1746: full GroupNorm at the normal recipe. Neutral vs 0.1758 despite
  139/s (5.0k steps), i.e. more learning per sample. Needs a 60-min read.
- v13_hard_20m 0.2479: oversample dense/narrow/corner and drop uniform. Much worse; extra
  32% vs 14%. **Uniform and sparse patches teach the model not to draw contours in flat
  regions.** Rejected; the opposite direction (more flat) is queued as v14_moreflat.
- v13_mildaug_20m 0.2886: milder noise and background level [0.5, 0.95]. Much worse (extra
  39%, IoU 0.92), mostly a train/benchmark distribution mismatch: val_v2 has background
  levels down to 0.35. So the model does not extrapolate across exposure. Aug-range
  experiments must keep training ⊇ benchmark ranges (or use a separate benchmark).
- v12_w96b12_s2_60m 0.1341, so the 96×12 3-seed mean is 0.1325 (sd 0.004) vs base cn
  0.1373 (2 seeds): −0.0048, ~0.9 SE. Held pending base seed 2; at 25% slower and
  2.7× params, 96×12 needs a clear win to adopt.
- v13_ref_20m 0.1654: 20-min reference for the cn best (for curriculum/aug screens).
- v12_w128b12_60m 0.1366: 128×12 (4.06M params, 266/s) is no better than base 0.1373
  and worse than the 96×12 mean. Capacity saturates at the 60-min budget: beyond
  ~96×12, lost samples outweigh the capacity.
- v12_w96b12_s1_60m 0.1356, so 96×12 2-seed mean is 0.1317 vs base cn 0.1373 (2 seeds):
  −0.0056, about the SE of the difference; not decisive. Costs 25% throughput and 2.7×
  params. Seed 2 of 96×12 (main) and base (fallback) are queued before deciding.
- v11_lr2e3_60m 0.1366: lr 2e-3 on the channel-LN best is neutral vs 0.1373. Stable
  but not better; keep 1e-3.
- v10_cn_w96b12_60m **0.1278**: 96×12 dilated trunk + channel LN (2.39M params),
  60 min. Best single run so far; −0.0095 vs the cn mean 0.1373 (> 1 sd, 1 seed). Extra
  8.0% (lowest), IoU 0.991, 301/s, 5.6 GB, stable (the same trunk diverged without the
  norm). Capacity is now a lever. Seed 1 and a 128×12 run are queued.
- v10_cn_60m_s1 0.1392, so the cn 2-seed mean is 0.1373 vs 0.1410 (3 seeds, no norm).
  ADOPTED (see Current best).
- v10_cn_60m_s0 0.1354: channel-LN trunk at 60 min, seed 0. vs best 3-seed mean
  0.1410 ± 0.0073: within noise, favourable. Missed 6.0% (lowest yet), loc 0.073.
  Activations bounded (trunk ≤ 6, stages ≤ 382). 401/s. Seeds 1 and 2 pending.
- v4_bd_s1_60m **0.1326**: best config (nob384) seed 1 at 60 min. vs seed 0 0.1439, so
  60-min seed noise is ~0.008 sd. The earlier adoption decisions at 60 min (bl+dil
  0.1464 vs nob384 0.1439) were within noise; nob384 stands on its 2× throughput.
- **Stress2 (const 2e-3, 10 min, nob384 base):** weight decay 0.05 survived, but
  activations still grew (trunk 71k, stage 179k), so it is not a fix. **Channel
  LayerNorm (per-pixel over channels; pre-norm trunk blocks + trunk output) survived
  with everything bounded (trunk ≤ 9, stages ≤ 253), lowest loss 0.043, at 400/s
  (−2%).** Trunk-output-only channel LN was also bounded (loss 0.050, 407/s) but had a
  0.25 loss spike. Channel LN is the leading stabilizer: nearly free, and it does not
  mix spatial statistics (unlike GroupNorm). 60-min ×2 seed confirmations are queued,
  plus capacity (96×12) and 120-min scaling on top of it.
- st2_ctrl_10m: the nob384 control DIVERGED at step ~1500 under stress (loss 269).
- v7_gnlow_60m 0.1506: GN-low at 60 min loses to both bests (185/s, 20k steps).
- v8_gnlow_nob384_20m 0.1799: GN-low on the nob384 base is much worse than nob384
  alone (0.1563). **GroupNorm costs quality.** Hypothesis: per-sample spatial statistics
  erase the patch's absolute intensity and contrast, which the backlight conditioning
  provides. GN rejected.
- v4_bd_w96b12_60m (on the nob384 base): DIVERGED at step 3800 (unnormalized, bigger).
  Capacity work is blocked on a stabilizer.
- v7_nob384_60m **0.1439**: drop the 384² and 768² res blocks, 60 min. Beats 0.1464 at 2×
  throughput and 4.8 GB; loc 0.0744/p95 0.276. ADOPTED. Unlike nob768, the gain held at
  60 min.
- v7_gnlow_20m **0.1660**: GN on the trunk + 96/192 stages at the normal recipe. ~3σ
  better than 0.1758. Error mix: extra 10.6% (vs ~15.7%), missed 10.4% (vs ~8.4%),
  loc 0.086 (vs 0.080). 185/s. Both a stability fix and a gain; 60-min confirmation
  queued.
- v7_gn_lr2e3_20m 0.1883: full GN at lr 2e-3 is worse. Stability does not buy a usable
  higher LR; keep 1e-3.
- Strong-defocus diagnosis (`work/scratch/strong_defocus.png`): with heavy,
  asymmetric blur (thin features, reflective rims) the GT edge is not at mid-intensity,
  and predictions wobble, creating extra points. This slice (10% of patches) is ~27%
  of the primary score and may be partly irreducible from 48 px. Next test: input
  context (see infra note).
- Stability probes (lr 1e-3, 5 min, 4 seeds): no divergence; trunk absmax peaks
  0.6k–2.6k and does not run away. 5-min bscores span 0.31–0.47, so short runs are
  very noisy.

## Data-scaling diagnostic (2026-09-25)

`benchmarks/trainfit_v2.json` is a diagnostic benchmark: val_v2's construction applied
to the **train** split. It is not for model selection. Best model (v3_bl_dil_60m):
train scenes 0.1413 vs val scenes 0.1463, a ~3% gap. Missed 6.4 vs 7.4%, extra 10.3 vs
10.8%, loc 0.075 vs 0.077. **The model is not data-limited**; it is compute/optimization
limited (20→60 min is worth ~0.02–0.03), plus possibly information-limited on strong
defocus. More renders of the same distribution are low priority. The highest-value
data addition would be a small real-image validation set to measure the sim-to-real
gap (asked the human, 2026-09-25). Re-check this gap whenever capacity increases.

## Error vs distance from patch edge (best model, val_v2 full patch, 2026-09-25)

| edge dist px | bscore | miss% | extra% | loc mean/p95 |
| --- | --- | --- | --- | --- |
| 0–1 | 0.224 | 12.0 | 19.5 | 0.099/0.335 |
| 1–2 | 0.190 | 8.7 | 17.8 | 0.084/0.297 |
| 2–3 | 0.165 | 7.6 | 14.8 | 0.075/0.279 |
| 3–4 | 0.156 | 6.7 | 13.6 | 0.075/0.281 |
| 4–8 | ~0.155 | ~7 | ~13 | 0.075/0.28 |
| 8–24 | ~0.143 | ~7.5 | ~10 | 0.077/0.29 |

Degradation is confined to the outer ~3 px. The 4 px scoring border already excludes
it. Beyond that only the extra rate improves toward the center, likely a
patch-sampling composition effect. **Stitching guidance:** keep the central 40×40
(stride 40, 1.44× compute); 6–8 px margins buy a little more for 1.8–2.25×.
Deployment overlap supplies context for free, unlike training-time input context
(rejected: it is slower).

## PLAN_v2 phase 0: CFI latency skeleton (2026-09-26)

Code: `edgesr/cfi.py` (model), `edgesr/cfi_infer.py` (sparse inference),
`edgesr/latency.py` (sweep); results in work/cfi/latency_*.json. Tests
(`tests/test_cfi.py`): sparse window inference == dense whole-image evaluation to
1e-4 (seam-free by construction), and point decoder == lattice decoder.

Findings (random weights, real val images, 16 CPU threads, PyTorch):
- **CPU efficiency is dominated by op type, not MACs.** Depthwise-separable + LayerNorm
  blocks run at ~16 GMAC/s eager (memory-bound). Dense 3×3 blocks compiled with
  channels_last run at ~100 GMAC/s, so the model uses dense blocks. Budget ≈ 70–90
  dense GMAC per image for ~0.8–1 s.
- **Decoder:** the implicit MLP over the full 1/16 lattice is ~1 s (memory-bound
  intermediates). The **block head** (3×3 conv → 16×16 logits, band pixels only) is
  0.06–0.09 s, so it is the default. Implicit remains an option with adaptive queries.
- **Band:** decoding whole 4×4 coarse cells is 4× more pixels than the ±1 px band.
  Added a per-pixel band head on the fine features (decode only near edges).
- **Coarse:** pixel_unshuffle stem + 1×1 at 1/4, work at 1/8 and 1/16: 0.14–0.17 s.
- End-to-end, compiled, ideal ±1 px band (~22k px), block decoder, cell 16/32 (best of):

| fine R (px) | total s | fine s |
| --- | --- | --- |
| 6 | 0.62 | 0.19 |
| 10 | 0.99 | 0.50 |
| 15 | 1.09 | 0.58 |
| 19 | 1.27 | 0.73 |
| 25 | 1.67 | 1.09 |

- Fine cost is margin-dominated (window (cell+2M)² vs core cell²). **Gate verdict:** the
  plan's "R≈8 ≤ 0.5 s" is not met in plain PyTorch (≈0.8 s at R≈8–10). Accepted as
  within the ~1 s ballpark given a concrete optimization backlog:
  - sparse contour extraction (0.2 → ~0.05 s)
  - merging adjacent cells into larger windows
  - a half-res mid branch for wide context (4× cheaper per px of radius)
  - ONNX Runtime / OpenVINO
  **Decision:** proceed to phase 1. R stays an accuracy axis (6–25), and latency is
  tracked per config.

## PLAN_v2 phase 1: pipeline + val_v3_full (2026-09-26)

- **Boundary-point cache** (`cache.py boundary-points`): the exact marching-squares
  crossings of every part's 32768² mask. 29.3M points, median 93k per part, 75 s to
  build. Tested equal to direct extraction. `store.mask_points` gives exact
  point-wise GT from the block store (tested equal to mask windows).
- **Crop/point sampler** (`crops.py`): 192 px rotated crops, 3072 near-edge + 1536
  band + 512 uniform queries. GT is occupancy (exact), plus distance and direction to
  the nearest GT crossing, coarse occupancy/band, and per-pixel band. Checks:
  occupancy flips across q + d·dir for 99.94% of near-edge queries; the image is dark
  where occupancy is 1. 19 ms/crop.
- **CFI training** (`train_cfi.py`): **300 crops/s** (≈12× the patch model's pixel
  throughput), 93% GPU, 1.35 GB. The quick eval (`valcrops_v1`) mirrors inference:
  sub-pixel logits only where the pixel band (dilated 1 px) fires, coarse occupancy
  elsewhere. A 4-min smoke run reached pixel-band recall 99.8–99.96%.
- **val_v3_full** (`bench_v3.py`, `benchmarks/val_v3_full.json`): 60 full val images
  (2 per val part, seeded photometrics), scored on whole-image contours.
  Per-GT-point labels: blur (10–90% ESF width on the raw render along the normal),
  sparse (no other edge within 4 px), corner, specular-at-edge, lighting, and SNR.
  **Blur calibration:** rendered sharp/moderate/defocused/depth_split edges all
  measure a median 1.37–1.5 px (the point-sampling floor through bicubic); only
  strong_defocus is heavy (3–7 px). Bins: sharp < 1.6, medium 1.6–3, heavy > 3 px.
  **Genuine medium blur is a minority of the current renders** (defocused and
  depth_split tails), so the protected medium slice has thin coverage.
- **Finding: full images expose background and frame hallucination.** The patch model
  on a full image has 0 missed and loc 0.032 px on the part, but 74% of predicted
  points are spurious: 50% along the image frame, 25% on flat background and dust
  far from the part. Neither was in the patch training distribution (uniform samples
  came from within ~64 px of the part). The benchmark now reports far extras (> 2 px
  from any GT) separately, split frame/interior, and does not charge them to GT
  slices. CFI fixes:
  - `crop.p_anywhere`: crops anywhere, overhanging the frame with edge-replicate
    sampling
  - inference pads the frame by 32 px (edge replicate), the same convention
- Queue: c1 radius sweep (R = 13, 6, 19, 25; 60 min, p_anywhere 0 so the sweep is
  clean), the patch reference on val_v3, and c1_r13_any (p_anywhere 0.15). Each run is
  followed by its val_v3 scoring.

### val_v3_full acceptance targets: patch reference (v10_cn_60m_s0, stride-32 blend)

`work/bench_v3/v10_cn_60m_s0.json`; 10.8 s/image on GPU. GT-side metrics per protected
slice (CFI must be non-inferior: loc within +0.005 px, missed/near-extra within +0.5 pt):

| slice | GT pts | missed | loc mean / p95 | near-extra |
| --- | --- | --- | --- | --- |
| sparse/sharp/all | 3.9M | 0.50% | 0.0435 / 0.123 | 0.50% |
| sparse/sharp/silhouette | 1.44M | 0.07% | 0.036 / 0.089 | 0.05% |
| sparse/sharp/reflective | 1.17M | 0.00% | 0.037 / 0.090 | 0.00% |
| sparse/sharp/balanced | 0.82M | 1.99% | 0.053 / 0.181 | 1.96% |
| sparse/sharp/ambient | 0.48M | 0.48% | 0.068 / 0.222 | 0.53% |
| sparse/medium/all | 0.79M | 5.62% | 0.090 / 0.279 | 6.24% |
| sparse/medium/balanced | 0.39M | 11.2% | 0.109 / 0.330 | 12.2% |

Non-protected: heavy blur 14.9% missed (loc 0.148), corners 10.5%, dense 8.4%.
**Far extras dominate the reference's overall extra rate (77%):** ~134k frame + ~172k
interior background points per image.

### Coarse band calibration (c1_r13_60m on full v3 images)

The recall-weighted band4 head is not selective on full images. At logit > 0 it
selects 53% of 16 px cells (ideal ≈ 3–4%). Threshold 3: GT recall 99.98% mean (min
99.84%), 20% of cells. Threshold 4: 99.90% (min 98.8%), 8% of cells. The likely cause
is that the run never saw far background or the frame (p_anywhere = 0); c1_r13_any
tests this. The band threshold is treated as an operating point (bench `--band-thresh`,
default 3). Cell fraction drives CPU latency, so selectivity matters. Also fixed:
the full-image scoring OOM (chunked fine/decode; occupancy-only decode at inference,
4× less decode work).

### Arm B: classical seed (human-directed, 2026-09-26)

See PLAN_v2 "Arm B". Implementation notes and checks:
- `seed.py` follows adapter.py (not its README). Seed band on 20 val_v3 images vs
  search radius:
  - r3: recall 99.91% mean (min 99.16%), 486 cells
  - **r4: 99.92% (min 99.16%), 508 cells**
  - r6: 99.92%, 582 cells
  - r8: 99.92%, 656 cells
  Compare 1,338 cells for the learned coarse band at threshold 3. **~2.6× less
  fine-branch work.** The worst-image miss (0.8%) is faint edges neither threshold nor
  gradient catches; check which slices.
- seed_full: 148 ms on the full 2048² image (was 665 ms before separable field
  evaluation, strided noise/percentiles, and window-level model inputs). Measured
  with training running concurrently.
- Tests: seed-mode sparse == dense (tests/test_cfi.py); crop seed vs full-image seed
  mask agreement 99.98–99.997%, field within 0.7–2%. Crop sampling cost unchanged
  (~25 ms/crop).
- Queue: c3_B_s0, c3_A_s1, c3_B_s1 (60 min, R13, anywhere 0.15) at the front, each
  followed by val_v3 scoring. A seed 0 = c1_r13_any_60m.

### A vs B, seed 0, val_v3 with filter 50/8 (`*_f50.json`)

| slice | ref miss / loc | A s0 miss / loc | B s0 miss / loc |
| --- | --- | --- | --- |
| sparse/sharp/all | 0.50 / 0.0435 | 0.54 / 0.0431 | 0.89 / **0.0491** |
| sparse/sharp/silhouette | 0.07 / 0.0355 | 0.09 / 0.0363 | 0.11 / 0.0392 |
| sparse/sharp/ambient | 0.48 / 0.0683 | 0.54 / 0.0619 | **3.37 / 0.0923** |
| sparse/medium/all | 5.62 / 0.0903 | 5.66 / 0.0866 | 6.04 / 0.0920 |
| sparse/medium/silhouette | 0.21 / 0.0577 | 0.19 / 0.0633 | 0.63 / **0.0669** |
| specular | 0.30 / 0.072 | 1.22 / 0.066 | **5.56 / 0.093** |
| heavy blur | 14.9 / 0.148 | 9.3 / 0.124 | 10.3 / 0.144 |

- **B fails several protected gates** (sharp/all loc +0.0056, near-extra +1.0 pt;
  ambient; medium/silhouette loc +0.009) and **is 0.003–0.004 px worse even on
  silhouette/reflective sharp**, where the threshold seed is ideal. Hypothesis: B's
  fine model lost the coarse context features (plateau levels/centering) that A
  had. A's flat R-sweep may not transfer to B. Queued c3_B_r19 and c3_B_w32 after B s1.
- **B's advantages:** 512 vs 1,338 cells per image; 0.83 vs 2.22 s per image (GPU,
  with the CPU filter); frame far extras 888 vs 23.7k per image. B interior far
  extras are 19.4k per image, mostly the bright-part failures.
- R25 (A-style, no anywhere): sharp loc 0.0435, heavy miss 7.6%, the same as R13/R19,
  so the A plateau is confirmed.
- **A seed noise on val_v3 is small:** A s0 vs A s1 at sharp/all loc 0.0431 / 0.0432,
  silhouette 0.0363 / 0.0369, medium/all 0.0866 / 0.0873, missed within ±0.06 pt.
  sd ≈ 0.0005 px. B's +0.006 px gap on sharp/all is ~10× that, so it is real unless B
  is far noisier (B s1 pending). A s1: 1,447 cells per image; far extras 37k frame +
  18k interior per image.

### Filtered review (defaults: object 50 px², hole 8 px², human decision)

`work/review/<run>_extras_filtered/`: contact_overview.png (whole images, GT green,
far extras red), contact_{specular,bright_object,background}.png, overview/ and
per-category crops.
- **Arm B:** 13 of 20 images have 0 far extras (dust removed); others ≤ 1k points. The
  remaining failures are all parts as bright as or brighter than the backlight, where
  Otsu splits inside the part:
  - #45: whole bright plate, inversion (835k)
  - #30: bright plate, contour just inside the outer edge plus holes (50k)
  - #54: shaded light-matte sphere, threshold ring through the bright center (99k)
  - #27: bright shaft (31k)
  - #48: gray polished plate (16k)
- **Arm A:** 6k–61k far points in 19 of 20 images, almost all along the image frame.
  Frame-connected, so the area filter keeps them. Otherwise clean.
- Human: the noise level of these renders is more aggressive than reality; the
  concerning failures are the polished-metal ones (both arms).

### Min-area region filter (human request, 2026-09-26)

`infer(min_area=...)`, bench `--min-area` (default 8 px², like the adapter) and
`--min-object-area`. Closed object specks and holes smaller than the threshold are
flipped before contour extraction. Exact hi-res version: per-cell canvases labelled
in one cv2 mosaic call per class, merged across cell borders with scipy csgraph
union-find, and kept if they touch the fill/frame. Tested (tests/test_cfi.py).
- Cost: exact ≈ 0.4–0.7 s CPU per image (35M hi-res px labelled). A fast quarter-pixel
  variant (~0.1–0.2 s) misses loops within 1/4 px of edges (image 30: 309k vs 223k
  points). **Exact is the default; making it fast is a phase-3 item.** The right end
  state is shoelace area on ordered sub-pixel rings, O(contour length).
- **Dust is 40–100 px²** (removed only at ≥ 64–128 px²), not "a few px²".
- **GT small regions (val parts):** 24% of GT regions are < 8 px² (0.19% of boundary),
  40% < 32, 56% < 128 (1.5% of boundary), mostly tiny through-holes. So object specks
  (dust) and holes get separate thresholds: `min_area` is scalar or
  (min_object_area, min_hole_area).
- **Defaults set by the human: object specks 50 px², holes 8 px²** ("true specks are
  fine to report; the user can see what they are"). Re-scores of A s0 / B s0 at the
  default are queued (v3_f50_*).

### Far-extra review: arm A vs arm B (`python -m edgesr.review_extras`, 20 val_v3 images)

Crops: `work/review/<run>_extras/{specular,bright_object,background}/`, plus
`contact_*.png` and `failure_images.png`. Far extras = predicted contour points > 2 px
from any GT edge.

- **Arm B (seed), normal images:** 0–1.4k far points per image (~1% of predictions).
  They are mostly **backlight dust**: the renderer's dust specks get outlined (some
  defocused). This is a policy question (the adapter rejects dust via `min_area`).
- **Arm A (learned coarse):** 17k–97k far points in almost every image, ~all "background".
  Consistent with its val_v3 far extras (frame ~40k + interior ~10k per image).
- **Arm B failures are concentrated in ambient-lit bright-part images:**
  - **#45 ambient light_matte: the entire part is brighter than the backlight.** The
    dark-on-light seed inverts completely: the field fit takes the bright plate as
    backlight, and Otsu labels the backlight ring as object, giving 1.35M far
    points. Not specular highlights; it is whole-part polarity inversion, outside the
    adapter's assumption. Arm A: 108k on this image.
  - #27/#30/#48/#54 (ambient/reflective, bright parts): 45–150k each. A mix of bright
    part interiors and small loops just inside dark-part edges in noisy images.
  - Genuine specular-highlight extras are few. The classifier's "bright_object" also
    catches near-edge loops, because its ±2 px max reaches the backlight.
- By point totals: B 1.81M (1.35M from #45 alone) vs A 1.03M. **Excluding #45, B ≈ 0.46M
  vs A ≈ 0.92M.**

### Arm B seed 0: training curve and a warning sign

Quick eval (same 64 val crops), A = c1_r13_any vs B = c3_B_s0:

| min | A bscore / miss / extra | B bscore / miss / extra |
| --- | --- | --- |
| 20 | 0.097 / 1.4% / 3.9% | 0.227 / 1.8% / 31.5% |
| 50 | **0.086 / 1.1% / 2.3%** | 0.139 / 0.9% / 12.8% |

B misses slightly less but has far more extra contours. Diagnostic on 48 crops (B at
50 min): mask-fill + gated 42% extra, ungated 61.5%. The extras are mostly > 4 px
from GT, i.e. region errors. Hypothesis: where Otsu mislabels a region (bright or
specular object areas, ambient-lit parts), the local fine model (R13 + mask channel)
follows the mask. This is the polarity risk. val_v3 per-slice/lighting scoring will
quantify it; valcrops oversample hard categories. Config fix: `crop.seed_inputs` is now
recorded in config.json.

### MILESTONE: CFI c1_r13_any_60m vs patch reference on val_v3_full

(`work/bench_v3/c1_r13_any_60m.json`; band_thresh 3, pixel_band −1.) 60 min of
training, R = 13, 15% anywhere crops. **0.20 s per full image on GPU vs 10.8 s**
(reference, stride-32 blend).

| slice | ref missed / loc / p95 | CFI missed / loc / p95 |
| --- | --- | --- |
| sparse/sharp/all | 0.50% / 0.0435 / 0.123 | 0.54% / 0.0431 / 0.116 |
| sparse/sharp/silhouette | 0.07 / 0.0355 / 0.089 | 0.09 / 0.0363 / 0.091 |
| sparse/sharp/reflective | 0.00 / 0.0366 / 0.090 | 0.05 / 0.0371 / 0.089 |
| sparse/sharp/balanced | 1.99 / 0.0534 / 0.181 | 2.03 / 0.0528 / 0.164 |
| sparse/sharp/ambient | 0.48 / 0.0683 / 0.222 | 0.54 / 0.0619 / 0.192 |
| sparse/medium/all | 5.62 / 0.0903 / 0.279 | 5.53 / 0.0868 / 0.259 |
| sparse/medium/silhouette | 0.21 / 0.0577 / 0.136 | 0.19 / **0.0633** / 0.141 |
| blur:heavy | 14.9 / 0.148 | **8.4 / 0.126** |
| corner | 10.5 / 0.093 | 10.7 / 0.098 |
| specular | 0.30 / 0.072 | 0.84 / 0.066 |

- **Gate:** every protected slice passes (loc ≤ +0.005 px, missed/near-extra ≤ +0.5 pt)
  **except sparse/medium/silhouette localization: +0.0056 px**, marginally over.
  Heavy blur is much better; specular missed is slightly worse (non-protected).
- **Far extras:** 305k → 51k points per image (frame 40k, interior 10.5k). The frame
  remains the main hallucination source; to investigate.
- **Latency concern:** this operating point selects a median 1,338 16-px cells per
  image, vs ~550 assumed in phase 0. Fine-branch CPU cost is ~2.4× that estimate, so
  ≈ 2 s CPU. **Coarse band selectivity is now the latency bottleneck.**
- Earlier OOMs: c1_r6_60m and the c1_r13 scoring failed from concurrent GPU use.
  Eval now caps GPU memory at 40% and falls back to CPU per image. Both re-queued.

### Fine context radius sweep (val_v3; no-anywhere runs, 1 seed each)

| R | sparse/sharp loc | sparse/medium miss / loc | heavy miss | specular miss | valcrops |
| --- | --- | --- | --- | --- | --- |
| 6 | 0.0451 | 5.78% / 0.0884 | 10.6% | 1.06% | 0.128 |
| 13 | 0.0428 | 5.44% / 0.0871 | 8.4% | 0.71% | 0.126 |
| 19 | 0.0429 | 5.55% / 0.0877 | 8.7% | 0.49% | 0.120 |
| ref | 0.0435 | 5.62% / 0.0903 | 14.9% | 0.30% | — |

R 6 → 13 helps sharp localization and heavy blur, supporting the ESF-context
intuition. **R 13 → 19 is flat.** R = 13 stays the default (latency). The
sparse/medium/silhouette localization spread across runs (0.058–0.063) looks like
part-level noise; the 2-seed A/B will calibrate it. R25 pending.

### Fine context radius, first read (val_v3; no-anywhere runs, 1 seed each)

| run | sparse/sharp miss / loc | sparse/medium miss / loc | heavy miss | valcrops bscore |
| --- | --- | --- | --- | --- |
| c1_r13 | 0.52% / 0.0428 | 5.44% / 0.0871 | 8.4% | 0.126 |
| c1_r19 | 0.50% / 0.0429 | 5.55% / 0.0877 | 8.7% | 0.120 |
| c1_r13_any | 0.54% / 0.0431 | 5.53% / 0.0868 | 8.4% | **0.108** |

- **R 13 → 19 does not improve medium blur** (or anything else beyond noise). Either the
  coarse features already supply the plateau/centering context, or rendered medium
  blur (≤ 3 px) is small relative to R ≥ 13. R6 and R25 pending.
- **Anywhere crops help broadly**, not only against far hallucination: valcrops 0.108
  vs 0.126. Default from now on.
- **Consistent small gap on sparse/medium/silhouette localization:** all CFI variants
  0.061–0.063 vs reference 0.0577. Hypothesis: fine/decoder capacity (width-24
  features into the block head vs the reference's 48/32-channel decoder stages).
  Queued c2_r13_any_w32 (fine width 32).
- Far hallucination without anywhere crops: r13 2.29M, r19 1.10M points per image;
  with anywhere 51k.

### Infra: queue lanes (2026-09-26 16:24)

val_v3 scoring is CPU-bound (KD-trees over up to millions of predicted points per
image) and was idling the GPU between training runs. The runner now has lanes:
`queue run --lane train` and `--lane eval` run concurrently. Queue edits are
file-locked, and scoring entries carry `after: <run_id>`. KD queries are
multi-threaded. CFI GPU inference is 0.25–0.3 s per full image.

Early v3 signal for c1_r13_60m (trained without background/frame crops): 0.4–2.3M
predicted points per image, 80–97% extra, i.e. massive background hallucination as
predicted. GT-side metrics per slice are pending.

### First CFI run (c1_r13_60m) finished

valcrops_v1 (400 val crops, band-gated): **bscore 0.126**, missed 4.8%, extra 9.1%,
loc 0.073 / p95 0.256, pixel-band recall 99.96%. 64.6k steps at 294 crops/s.
val_v3 scoring is queued.

Quick eval (64 val crops, band-gated), by minute: 10: bscore 0.258; 20: 0.211; 30: 0.205.
At 30 min: missed 7.1%, loc 0.068 / p95 0.23 px, pixel-band recall 99.94% (−1 logit,
1 px dilation). Extra ~19–25%, but the **median crop has 0 extras**. They concentrate
in hard crops (`work/scratch/cfi_extras.png`):
- heavy blur: contour ~1 px off and wiggly, plus blobs
- textured bright object interiors (reflective/ambient): spurious loops where the pixel
  band fires on texture
Gating vs ungated decode gives identical extras, so the coarse fill at the band
border is not the cause. Possible lever if interior loops persist: condition the
decoder/band more strongly on the coarse region class.

## Full-image benchmark (2026-09-26, `python -m edgesr.fullimage`, results in work/fullimage/)

Throughput for 2048² → 32768² mask, 48 px tiles keeping the central 40 (stride 40,
2704 tiles):

| model | GPU 3070 bf16 compiled | CPU 16 threads fp32 |
| --- | --- | --- |
| best cn 0.88M | **1.74 s/image** (1550 tiles/s; model-only 1660) | ~118 s/image (23 tiles/s) |
| 96×12 2.39M | 2.5 s/image (1083 tiles/s) | ~136 s/image (20 tiles/s) |

Tile seams (6 val regions of 200² px on dense geometry, same fixed 12-bit rendering):
- **Plain tiling (keep central 40) has real seam artifacts.** Within 0.25 px of a seam,
  extra contour points are 59% (vs 16% far from seams): adjacent tile cores disagree,
  and the step in probability across the seam creates spurious 0.5 crossings along
  it. Shifting the tile grid by half a tile moves contours by a median 0.96 px at
  seams (0.01 px away from them; p95 ~0.75 px even far away).
- **Overlap blending removes them.** Averaging logits with a tapered window (zero in the
  4 px discard margin) gives uniform error vs seam distance, and grid-shift jitter
  median ~0.005 px everywhere:
  - stride 32 (1.61× plain tiles): bscore 0.152, jitter p95 0.54 px
  - stride 24 (2.86×): bscore 0.158, jitter p95 0.49 px
  - stride 16 (6.35×): bscore 0.161, jitter p95 0.22 px
  **Recommendation: blend at stride 32** (≈2.8 s/image on GPU, ≈190 s on CPU). The p95
  jitter of ~0.5 px that remains at any stride ≥ 24 is the model's own position
  dependence in ambiguous regions (typical median jitter is ~0.005 px).
- 96×12 under plain tiling looked worse on these regions, but it is within blend
  noise. Only 6 regions; use a larger region set for model comparisons.
- **Only 6.5% of 48 px tiles (stride 32) contain any boundary** (1.9–11% per val part).
  Skipping uniform tiles with a conservative intensity-range test would cut
  full-image cost ~15× (GPU ~0.2 s, CPU ~12 s per image, blended). The test must
  not skip faint or defocused edges.

## Failed ideas worth remembering

- **LR 2e-3 (the original recipe) is unstable.** c0_base_20m_s1 diverged at step ~200,
  just as warmup reached peak LR. Loss jumped to 0.87, then stayed pinned at ln 2 for
  the rest of the run. Trunk and decoder activations had exploded to 1e5–1e7 in the
  unnormalized residual stream, and the last SiLU saturated to 0, leaving a dead
  network (bscore undefined, IoU 0.53). Seed 0 also spiked (0.66 at step 400).
  Fix (recipe change, applies to all v2b runs): LR 1e-3, warmup 5%, a divergence
  guard (abort if loss > 0.68 after step 1000), and `trunk_absmax` logged every 100
  steps. If the probes still show blow-up, add normalization or residual scaling.

- A NumPy per-pixel mask gather in DataLoader workers collapsed to ~80 patches/s total,
  from memory-bandwidth and allocation contention across workers, despite 13 ms/patch
  in-process. Use cv2 warps for large per-sample arrays.

## Unresolved questions

- (Resolved 2026-09-25) Objects can be brighter than the backlight; a backlight value
  may be given to the model (see infra v2).
- How close to the 1/32 px GT quantization floor can a CNN get? What is the error floor
  from sensor noise and 8-bit quantization?
- Does training on bicubic-resampled rotated patches hurt lattice (deployment)
  accuracy? `sampler.p_lattice` is the knob.
- Sharp-focus renders are point-sampled (pixel filter 0.01), so edges are aliased
  steps. How much subpixel information is recoverable there, compared with blurred
  renders? Check the per-focus-class breakdown.

## Next experiments

1. (queued v4, on bl+dil) input skip; boundary-weighted BCE; focus oversampling; seed;
   inskip at 60 min. Fallbacks: 120 min, 96×12 at 60 min, seed at 60 min.
2. Then: patch/context size (the plan's axis) for strong defocus; aux low-res
   supervision; decoder channel widths at 384/768. Then combine the winners and run the 96×12 capacity test.
2. If RF helps: a proper multi-scale (U-Net style 48→24→12) encoder for global context,
   or a global pooling / context branch.
3. Capacity beyond e2: more compute before upsampling.
4. Decoder: pixelshuffle vs resize-conv, decoder channels at 384/768, input skip.
5. Loss: boundary-weighted BCE, auxiliary low-res supervision.
6. Sampling: p_lattice, category weights.

## Queue / cycle log

- Cycle 0 (2026-09-24 ~22:45): queued c0_base_60m, c0_base_20m_s0, c0_base_20m_s1.
- Cycle 1 (2026-09-24 ~23:50): baseline 0.2768 analysed (see diagnosis). Queued
  e1_dilated_20m and e2_w96b12_20m after the calibration pair.
- Cycle 2 (2026-09-25 ~00:15): e1 and e2 cancelled before running (v1 exposure bug).
  Infra v2 built; queued v2_base_20m, v2_bl_20m, v2_dil_20m.
- Cycle 26 (12:30): hard curriculum and mild aug rejected (extra contours). Queued
  moreflat, and 96×12 at 120 min as a fallback. The cn 120-min fallback is running.
- 2026-09-26 11:38–11:46: queue paused at the human's request for the full-image
  throughput and seam benchmark (see section). v13_hard_20m was interrupted and
  resumes from last.pt.
- Cycle 25 (11:30): 96×12 mean 0.1325 (held). cn 20-min reference 0.1654. Curriculum
  screen running.
- Cycle 24 (10:30): 128×12 no gain. Queued the untested plan axes as 20-min screens:
  ref, hard-geometry curriculum, and milder augmentation. (Human question on split
  methodology answered; offered val_v3 with more val parts, awaiting reply.)
- Cycle 23 (09:30): 96×12 2-seed mean 0.1317 (not yet decisive). Queued 96×12 seed 2.
  128×12 is running.
- Cycle 22 (08:30): lr 2e-3 neutral. 96×12 seed 1 running. Queue ~4.6 h.
- Cycle 21 (07:30): 96×12 + cn = 0.1278. Queued its seed 1 and 128×12 (both 60 min).
- Cycle 20 (06:30): channel LN adopted into best.json. Queued lr 2e-3 at 60 min on the
  stable base. w96b12+cn is running.
- Cycle 19 (05:30): cn s0 0.1354. Added a cn seed-2 fallback.
- Cycle 18 (04:30): best seed 2 = 0.1464, so the best 3-seed mean is 0.1410 ± 0.0073.
  cn 60-min s0 is running.
- Cycle 17 (03:30): channel LN wins stress2. Queued cn 60-min s0/s1, cn+96×12 60 min,
  and a cn 120-min fallback; cancelled the unnormalized 120 min. Queue ~5.5 h.
- Cycle 16 (02:30): best seed 1 = 0.1326, so 60-min noise sd ~0.008. Stress2 control
  diverged; wd/channel-LN stress runs in progress.
- Cycle 15 (01:30): GN rejected (quality cost); w96b12 diverged. Added the
  channel-LayerNorm option (per-pixel, no spatial mixing) and norm_trunk_out. Queued
  stress2 (control / wd 0.05 / channel-LN trunk / channel-LN trunk-out only) on the
  nob384 base, plus a 60-min seed-2 fallback.
- Cycle 14 (00:30): nob384 adopted (0.1439 @60). best.json updated. The gnlow_60m
  confirmation runs on the old base; the gnlow+nob384 combo screen is next.
- Cycle 13 (23:30): gnlow 0.1660 @20 and gn_lr2e3 worse. Queued the gnlow+nob384
  combo screen. Queue ~6 h.
- Cycle 12 (22:30): nob384 (0.1563 @20) and GN-low stress pass. Queued 60-min
  confirmations for nob384 and gnlow. Queue ~6.5 h.
- Cycle 11 (21:30): stress results (GN fixes it; β2 does not). nob768 lost at 60 min.
  Added `norm_dec_stages`. Queued the GN-low stress, then v7 GN / GN-low / GN lr 2e-3
  screens, with a 120-min GN-low fallback. Cancelled nob768_t12.
- Cycle 10 (20:30): v4_bd_120m diverged. Added the norm option (GroupNorm), betas,
  constant schedule, and stage diagnostics. Stress tests are queued at the front.
- Cycle 9 (19:30): nob768 promising and b16 neutral. The runner hit fallback v4_bd_120m
  (no idle). Queued nob768 at 60 min, a nob768 seed, nob384, and nob768 + 12-block trunk.
- Cycle 8 (18:30): bw2r4 and context 8/16 rejected. Fixed the overlay crash for context
  inputs and added `--finalize` recovery. Queued batch-16 (throughput). Queue ~4.5 h.
- Cycle 7 (17:30): inskip60 and decwide rejected (throughput). Added per-stage
  dec_blocks; queued nob768. Queue ~5.5 h.
- Cycle 6 (16:30): seed 1 and focus reviewed; strong defocus diagnosed. Implemented
  input context and queued ctx8 and ctx16 screens. Queue ~6 h.
- Cycle 5 (15:30, hourly cron): inskip and bw4 screens reviewed; nothing adopted.
  Queued decwide and bw2r4 screens. ~6 h of work queued.
- Cycle 4 (14:43): the queue emptied at 13:24 and the check-in notification was lost
  again (80 min idle). The runner now supports `--fallback` entries that run only when
  the main queue is empty; long runs and seeds go there. Queued v4: on bl+dil, input
  skip, boundary-weighted BCE ×4, focus oversampling, a seed repeat, and inskip at
  60 min. Fallbacks: 120-min best, 96×12 at 60 min, and seed 1 at 60 min.
- Cycle 3 (09:10): the queue had emptied at 01:51 and the wakeup notification was
  lost, so the GPU idled ~7 h. Lesson: always keep hours of follow-on work queued.
  Adopted backlight input and queued the 9-run v3 bolus (~4.5 h).
- Cycle 2b (00:26): c0 seed 1 diverged, so the v2 runs were stopped and cancelled.
  Queued 4 stability probes (lr 1e-3, 5 min each), then v2b_base/bl/dil at 20 min.
