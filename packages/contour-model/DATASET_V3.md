# Contour corpus v3 — realistic parts, observable labels, PAE + edge extraction

Status (2026-10-19): training-scale build `v3-20261018` and two training runs complete
(see "Results"). No model has been promoted to the GUI. Pilot data on the T7 drive:

```
/media/kyle/T7 Shield/inspection-contour-training/v3-pilot-20261018/
  corpus/           600 clean designs (sources/, cache npz, index.json, config.json)
  extract/          PAE renders of 300 designs; edge-model extraction of 150 of them (same images)
  review-corpus/    contact sheets of the clean labels (index.html)
  review-extract/   render + extracted contour + transferred labels (index.html)
  stats-clean.json  label/geometry statistics vs. the intent-v1 corpus
```

## What changed and why

| Problem in intent-v1 / three-base-v2 | v3 change |
|---|---|
| 20 of 46 families had only sharp corners; 12% tangent junctions | `realism.add_corner_details`: per-part tool radius, fillets, edge breaks (arc or chamfer), rounded thread/tooth roots. Analytic tangent construction, exact labels |
| 55% of arcs swept <5° (sub-pixel tooth crowns) | `labels_v3.observable_labels`: arcs with sagitta <0.25 px → line; instances <2.5 px dissolve into neighbours; unresolvable line junctions merge. Applied at 1000 px across the part for clean data, at the actual scale for renders |
| Dense showcase parts dominated (median 84 instances/part) | Weighted groups: CSG 30%, common parts 25%, turned 20%, repeat 18%, curved 7%; dense families use `moderate` caps |
| Turned-part silhouettes mirrored exactly (threads in phase) | `turned.turned_part`: lower thread flank offset by P/2 (10% in phase), sampled flank angles/truncations, diameter-relative pitches, start chamfers, runout or relief grooves, U/V/rect/trapezoid grooves, one-sided keyways/flats, cross holes |
| Fixed family proportions → shortcuts | `csg.csg_part`: grammar-free outlines (rectilinear, union, link, radial) with exact line/arc labels from carrier snapping; sketch-style openings (standard hole sizes, grid/line/bolt-circle/scatter layouts, slots, curved slots, rounded rects, polygons, D/double-D) |
| Freeform cams/lobes/pulleys had arbitrary arc breakpoints | Replaced by tangent-arc `lobed_profile`, `tangent_cam` (base+nose circles, tangent flanks, keyed bore), `round_groove_pulley` |
| Ellipses and other unsupported curves had no rule | A few ellipse openings with `unsupported=True`: zero loss weight |
| Renders always filled a 640–1920 px frame, axis-aligned, flat noise | `render_v3`: part spans 350–3200 px (fixed magnification), random rotation, subpixel raster, defocus, glare halo, backlight gradient/blotches, dust and fibres (some on the edge), shot/read noise/PRNU, 8 or 12 bit |
| Strict transfer discarded 30–42% of renders (survivorship bias) | `extract_v3.transfer_keep`: keep the observation; zero-weight edges far from the source (>1.5 px), backtracking, or around vanished instances; reject only a lost outer ring or >25% ignored outline |
| Only the edge model was used for renders | Each render is extracted with the GUI's PAE (sdf_v5 coarse + PAE) and, for a subset, the sdf_v5 edge model (torch on GPU, identical to the ONNX export within 0.09 px). Paired observations of the same image |
| No reflection augmentation | `train_primitives --mirror` (random reflection, canonical winding kept) |
| No unseen-family test | `holdout` split: every `t_bracket`, `dogbone_link`, `ratchet_wheel`, `stepped_taper_arbor` and CSG `radial` design; never trained on |

### Observability calibration

The pilot was labelled with first-guess thresholds (0.25 px sagitta, 2.5 px minimum
span). `scripts/calibrate_observability.py` then measured, on the 145 paired PAE/edge
observations (`v3-pilot-20261018/observability-calibration.json`):

* Arc curvature (line-vs-circle fit of extracted points inside each construction
  arc): 0% visible below 0.35 px sagitta, ~35–65% at 0.5–0.75 px, ≥85% from 0.75 px.
  Extraction noise is 0.05 (edge) / 0.06 (PAE) px RMS.
* Sharp corners are already rounded by the extractors: the 95th-percentile deviation
  from the ideal corner is 0.34–0.41 px (edge) and 0.48–0.83 px (PAE).

The training-scale build therefore uses `SAGITTA_PX = 0.5` and dissolves an instance
shorter than 8 px into its neighbours unless it deviates more than 0.5 px from the
sharp corner they would form (plus anything under 2.5 px). Turned parts also carry
more threads with a larger trapezoidal/square share (trapezoidal threads in 4% of
parts instead of 2%).

Label schema is unchanged (`primitives-compounds-v2`: line/arc/circle +
slot/threads/tooth). The labelling policy is recorded as `label_policy:
observable-v1`. Zero-weight edges are removed from token `weights`, which the
existing losses and evaluation already use, so no loss code changed.

## Pilot numbers (500 train designs each, `scripts/corpus_stats_v3.py`)

| | intent-v1 | v3 pilot |
|---|---:|---:|
| Instances per part, median / p95 | 84 / 677 | 26 / 191 |
| Rings per part, p95 | 197 | 29 |
| Arcs sweeping <5° | 55% | 3% |
| Arc share of contour length | 10.5% | 22.7% |
| Tangent junctions (<3°) | 12% | 29% |
| Zero-weight length | 0 | 1.1% |

Extraction (`extract/summary.json`, `extract/audit.json`): PAE kept 268 of 300
renders, the edge model 145 of 150 (strict transfer previously kept ~58–70%).
PAE rejections are mostly genuine PAE failures ("missing PAE edge support"),
which the GUI would also report. Accepted observations ignore a median 0% and a
90th percentile ≈0.3–1% of the outline (mostly fibres touching the edge). The
edge model also returns dust specks as separate objects (67 dropped rings in
145 renders); they are removed rather than treated as openings.

## Commands

```bash
cd packages/contour-model
R="/media/kyle/T7 Shield/inspection-contour-training"
.venv/bin/python -m contour_model.corpus_v3 "$R/v3-pilot-20261018/corpus" --count 600 --workers 10
EXTRACT_THREADS=2 .venv/bin/python -m contour_model.extract_v3 "$R/v3-pilot-20261018/corpus" \
  "$R/v3-pilot-20261018/extract" --pae-limit 300 --edge-limit 150 --workers 10
.venv/bin/python -m contour_model.review_v3 "$R/v3-pilot-20261018/extract" "$R/v3-pilot-20261018/review-extract"
.venv/bin/python scripts/corpus_stats_v3.py OUT.json old=$R/primitives-three-base-v2:500 v3=$R/v3-pilot-20261018/corpus:500
```

PAE runs in CPU workers; edge-model extraction then runs in the parent process
on one GPU context (several GPU workers exhausted the 8 GB card).

## Training-scale build

Kept close to the current corpus size rather than larger:

Output: `/media/kyle/T7 Shield/inspection-contour-training/v3-20261018/{corpus,extract}`.

* 8,000 clean designs (≈5,800 train) — current: 12,000.
* Extraction plan (designs; edge ⊂ PAE, same images):
  `train:4000:2000,val:500:500,test:500:500,holdout:500:500,stress:250:250`.
* Training mix (`autoresearch/make_mix.py`): clean ×1, PAE ×2, edge ×2, `--mirror`,
  checkpoints selected on PAE validation (`--val-extractor pae`; edge/clean logged).
  Two runs with the E007 recipe: warm start from release-v6, and from scratch.
* Benchmark: `scripts/bench.py` sets `v3_{pae,edge,clean}[_val|_holdout|_stress]`
  ignore zero-weight edges. Old `edge*` sets remain for regression context; their
  labels use the old policy, so drops there on tooth crowns/edge breaks are expected.

Estimated cost: clean corpus ~5 min (10 workers); PAE ~1 s/render; edge ~1–6 s/render
on the GPU, about 2–3 h for 2,500 renders.

## Results (2026-10-19)

Build: 8,000 designs (5,769 train). Extraction kept 5,155/5,750 PAE observations (90%)
and 3,655/3,750 edge-model observations (97%) in 2 h 40 min. Pipeline:
`scripts/pipeline_v3.sh`; outputs, logs, runs and bench JSON under `v3-20261018/`.

Runs (E007 recipe, `--mirror`, selected on PAE validation): **warm** from release-v6,
32k steps, lr 1e-4 (41 min); **scratch**, 48k steps, lr 3e-4 (60 min).

Base instance F1 (IoU ≥ .5; `scripts/bench.py`, ONNX-equivalent torch path, merge_tol=1):

| Set | release-v6 | warm | scratch |
|---|---:|---:|---:|
| v3 PAE test | .662 | .835 | **.848** |
| v3 edge test | .672 | .834 | **.848** |
| v3 PAE val | .665 | .841 | **.855** |
| v3 PAE holdout families | .544 | .850 | **.882** |
| v3 edge holdout families | .544 | .848 | **.878** |
| v3 PAE stress | .610 | **.780** | .774 |
| v3 clean test | .694 | **.879** | .877 |
| old edge test (old labels) | **.911** | .739 | .755 |
| old clean test (old labels) | **.922** | .793 | .799 |
| old clean test, observable labels | .810 | .908 | **.923** |

The old benchmarks' drop is the labelling-policy change, not a regression: rescored
with observable labels (`scripts/rescore_observable.py`), the new models lead. The swing
is concentrated in tooth families (toothed disk, spline ring, serrated washer, ratchet):
v6 ≈1.0 on the old labels and ≈0.5 on observable ones, the new models the reverse.
v6's arc precision under observable labels is .31 — it calls sub-pixel crowns arcs.

Remaining differences worth checking on real parts: `d_profile_spacer` (old clean,
observable labels: v6 .80, scratch .73) and `tapered_pin` (1.0 → .91). Cams, lobed
couplings and the old timing pulley stay low for every model (conventional freeform
breakpoints). All numbers are synthetic; real captures remain the deciding test.

## Known limits

* Still synthetic: real captures remain the only real test.
* `labyrinth_plate` and involute/linear `toothed_disk` flanks still rely on numeric
  decomposition / the intent policy.
* Real CAD sketches (DeepCAD, SketchGraphs) are not imported; their construction
  statistics only informed the CSG generator.
* Observability thresholds (0.25 px sagitta, 2.5 px minimum feature) are choices,
  not measured limits of the extractors; check against real captures.
