# Research specification (compact, stable)

Task: 48×48 backlit sensor patch → 768×768 (16×) binary object mask, one logit per
output pixel. Contours are the 0.5 iso-line of the predicted mask. Full brief: `PLAN.md`.

## Fixed infrastructure (changes must be recorded in RESEARCH.md as infra changes)

- **Source data**: `/media/kyle/T7 Shield/inspection-synthetic-renders/full-catalog-5gb-2048`,
  2299 Cycles scenes (2048², uint16) over 300 catalog geometries; 32768² masks.
- **Cache**: `work/cache` (T7). Built by `python -m edgesr.cache build`. Source renders
  only, never patches.
- **Split**: by geometry (catalog part), seed 20260924, 6 parts per family for each
  held-out set: train 240 parts / 1840 scenes, val 30 / 229, test 30 / 230.
  **Test is untouched** until a final confirmation of the selected model.
- **Coordinates**: source pixel i covers [i, i+1). Hi-res mask pixel j covers
  [j/16, (j+1)/16). Patch map s = c + A (P − 24). Image: Keys bicubic at s.
  Mask: hi-res pixel containing s. Verified by `tests/test_registration.py` and by
  `python -m edgesr.registration` (real data median shift ≤ 0.003 px).
- **Current benchmark: `val_v2`** (same patches as v1; exposure normalized by the
  scene backlight; stores each patch's backlight value). Models may take a backlight
  brightness value input. Its source at deployment (calibration or an estimator) is
  out of scope.
- **Benchmark `val_v1`** (superseded; p99.5 exposure normalization bug): `benchmarks/val_v1.json`, 2100 lattice patches (integer
  center, dihedral orientation, no resampling) from val scenes. Category counts
  edge 600, sparse 300, dense 300, narrow 400, corner 400, uniform 100. The
  photometric augmentation (default aug) uses fixed per-patch seeds. Content hash is
  checked at load.
- **Metrics** (`edgesr/metrics.py`): contour points are marching-squares crossings of
  0.5, in source px. Scoring excludes the 4 px patch border.
  - **Primary: `bscore`** = ½·(mean min(d_gt→pred, 1 px) + mean min(d_pred→gt, 1 px)),
    pooled over all boundary points. Lower is better.
  - Secondary: missed_rate / extra_rate (d > 0.5 px), loc_error_mean/p95 (matched
    points), p90/p99, accurate_rate (≤ 0.1 px), component_miss_rate (GT fg/bg
    components reproduced < 50%), IoU, BCE, per-category breakdowns.
- **Run budget**: time-based (training wall seconds, excluding eval); cosine LR over
  the budget. Screening and confirmation budgets: see RESEARCH.md.

## Commands

```
.venv/bin/python -m unittest discover -s tests          # registration tests
.venv/bin/python -m edgesr.queue add ID --set k=v ... --desc "hypothesis"
nohup setsid .venv/bin/python -m edgesr.queue run >/dev/null 2>&1 &
.venv/bin/python -m edgesr.queue status
.venv/bin/python -m edgesr.report                       # ledger summary table
.venv/bin/python -m edgesr.benchmark eval work/runs/ID/final.pt --overlays out.png
.venv/bin/python -m edgesr.gpubench --set model.width=96 --batches 16 32
```

Per-run artifacts are in `work/runs/<id>/`: config.json, log.jsonl (curves and quick
evals), metrics.json, overlays(.png, _zoom.png), final.pt, last.pt, code.diff.
`ledger.jsonl` is append-only, with one line per completed run.
