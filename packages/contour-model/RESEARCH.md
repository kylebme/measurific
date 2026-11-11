# Contour-model autoresearch

Goal (user, 2026-09-27): the contour recognizer creates many extra segments and
mislabels often in the GUI. Improve it; speed is already fine.

Durable state: this file, `autoresearch/spec.json` (compact spec),
`autoresearch/ledger.jsonl` (append-only, one line per benchmarked run).
Queue: `autoresearch/gpuqueue.py` (lanes run via `setsid nohup ... run --lane N`),
per-run wrapper `autoresearch/ar_run.py` (resumable train → bench → ledger line).
Large outputs: `/media/kyle/T7 Shield/inspection-contour-training/autoresearch/`
(`runs/<id>`, `logs/<id>.log`, `bench/`).

## Status: FINAL (2026-09-27) — iteration stopped at the user's request

Shipped: **`release-v6`** (= E007), bundled by the GUI (`app/python/worker.py`
`DEFAULT_MODEL`, `app/package.json`, `app/electron/main.cjs`, smoke script, CI
workflow), with `merge_tol=1` post-processing (`DEFAULT_POST`). GUI worker tests
(18) pass; the worker loads release-v6/onnx and analyzes the demo images.

| Metric | v5 (before) | v5 + merge | **v6 + merge** |
|---|---|---|---|
| Edge test instance F1 | .8638 | .8966 | **.9108** |
| Edge test precision | .856 | .939 | **.956** |
| Edge val F1 / exact / resolvable / fam_macro | .8790/.9288/.9465/.8726 | .9098/.9559/.9784/.9005 | **.9215/.9643/.9882/.9129** |
| Hi-res (2.5–4k px) test F1 / precision / resolvable | — | .8376/.8643/.9456 | **.9000/.9460/.9796** |
| Clean contours F1 | .9067 | .9188 | **.9215** |

Final selection (edge val micro/exact/resolv/fam | edge test micro): E007 .9215/.9643/
.9882/.9129 | .9108; E009 (full ×6) .9209/.9633/.9870/.9082 | .9106; E004 .9199/
.9635/.9871/.9099 | .9094; F003 .9188/.9646/.9881/.9092 | .9058; F004 (EMA)
.9151/.9604/.9846/.9050 | .9046; F005 (E004 seed repeat) .9137/.9589/.9827/.8999 |
.9069 → seed noise ≈ ±.006 val micro.

Note: from 2026-09-27 02:30 every benchmark uses `merge_tol=1` by default, so
compare new models against P005, not B001.

## Benchmark (fixed)

- Primary: base-instance F1 (IoU ≥ .5, length-weighted raw-edge IoU, Hungarian per
  class) on every accepted **edge-model** test render in
  `primitives-edge-v1-bench` (split test). `edge_val` (split val) is for tuning
  decisions; report test only for confirmed changes. `orig` = first 300 clean
  intent-v1 test contours (secondary). `scripts/bench.py` also reports fragment /
  mislabel / merge false-positive counts, edge accuracy, compound instance F1.
- Corpus: `contour_model.extract_corpus` renders the actual imperfect source
  contour (sides 640/960/1280/1920 px, 3× AA, blur .35–1.1, noise ≤1.2), runs the
  production extractor `inspection_measurement.contours_from_image` (edge-model
  ONNX, 1.1 px point spacing) and transfers labels with the strict PAE-era
  transfer (reject topology change, lost/reordered instances, >1.5 px).
  ~58% of renders are accepted; rejections are mostly tiny features lost at low
  resolution, so the benchmark under-represents the hardest small features.
- Source designs: `primitives-intent-v1-20260920` (12,000 designs, intent labels).
  Test/val designs are never used for weight training.

## Decisions

- 2026-09-27: GUI extraction switched from PAE to the learned edge model (user,
  commit 2ddce60). PAE benchmark numbers are kept only as history; new benchmark
  and training renders use the edge model.
- Decisions use `edge_val`; `edge` test is the reported number.
- Training runs are batch-1 and CPU-bound (~12 steps/s, GPU ~20% busy), so
  several queue lanes can share the GPU.

## Findings

1. (PAE bench, v5) Token metrics are ~.98 but instance F1 is .857: token
   scores hide instance errors. False positives are dominated by **tiny
   fragments** (median 2 raw edges; 5,252 fragments vs 1,991 mislabels on 438
   renders). ~60% of fragments sit between two same-class neighbours: the
   boundary head fires on adjacent tokens around one corner and both peaks
   become cuts. Mislabels are mostly whole-instance line↔arc swaps.
2. (PAE val, v5) Post-processing `min_piece` (drop cuts that would leave a
   piece shorter than k token spacings): k=2 → F1 .8748, k=3 → .8756
   (arcP .73→.885) but line recall falls (.861→.814). `cut_merge` (NMS over cuts)
   hurts line recall; `min_run` (class smoothing) ≈ no effect; peak height .6
   slightly better than .5. Plateau ≈ .877: the remaining errors need model or
   segmentation changes. Must be re-verified on the edge-model bench.
3. (PAE val, v5) **Fit-aware merge** (`merge_tol`: merge adjacent same-class
   pieces if one line / algebraic circle fits the union within k median point
   spacings): k=1 → F1 **.8938** (P .835→.933, R .864→.858), k=.5 → .888,
   k=2 → .879 (over-merges). Adding min_piece=2 on top lowers recall (.8866).
   Geometric arc→line relabel (`relabel_tol` .5) is disastrous (arc recall .37):
   true arcs in this corpus are often shallow; do not relabel by line fit.

4. (Edge bench, 652 test / 723 val renders; build accepted ~70%)
   v5 default .8638 test / .8790 val; **merge_tol=1 → .8966 / .9098** (fragments
   7,759 → 1,386 on test) — transfers from PAE val and also helps clean contours
   (.9067 → .9188). Adopted as default. Latency within noise (CPU was
   saturated during measurement; pair fits cached). release-v3 (clean-trained)
   is much worse (.8207): rendered-image training transfers to edge-model
   contours. E001 (16k more PAE-mix steps) only +.004 on PAE test.
   After merge, **mislabels dominate** (test: 2,643 mislabel vs 1,386 fragment
   vs 910 merge FPs); arc precision .868 and line recall .839 are the weak spots.

5. (Edge val, 200 renders, v5+merge) **80% of mislabels are timing_pulley**
   (972/1,213; 56% of its instances). Truth-arc→line swaps: median 14 px long,
   sagitta .46 px; truth-line→arc: median 3 px long. These tooth details are
   near/below render resolution, so instance-micro F1 over-weights one family's
   unobservable features. Added `fam_macro` (mean over families of per-part F1)
   to bench headlines: B001 .8751 → P005 **.901** (test). Weakest families
   (P005 test): eccentric_cam .24, cam .25, lobed_coupling .38, timing_pulley
   .46, d_profile_spacer .76, labyrinth_plate .83. Cams are freeform curves whose
   arc/line decomposition label is conventional (intent policy), so low scores
   there may be label ambiguity rather than model error — review visually.

6. E001b (v5 + 16k PAE-mix steps, merge on): edge test **.9064** / val
   **.9162** / fam_macro .902 vs P005 .8966 / .9098 / .901. More rendered-image
   training helps micro F1 (arcP .868→.931, fewer fragments and mislabels) but
   not the family macro. Edge train corpus (4,000 designs): 2,889 accepted.

7. Visual review (E001, edge val; truth vs prediction): lobed_coupling truth cuts
   smooth tangent-continuous lobes into ~130 arcs at arbitrary breakpoints; the
   model predicts ~57 longer pieces (fewer segments — arguably what the user
   wants) and is scored as wrong. timing_pulley: smooth wavy profiles with
   arbitrary truth breakpoints, and one val part has ~1,045 sub-resolution
   instances. toothed_disk and d_profile_spacer are essentially right. Added
   bench `exact_f1`: micro F1 excluding AMBIGUOUS = {timing_pulley,
   lobed_coupling, cam, eccentric_cam}. Decisions should weigh edge_val micro
   F1, exact_f1 and fam_macro together; do not chase breakpoints on smooth
   freeform curves.

8. Matched 16k continuations from v5 (merge on), edge val micro / exact / fam_macro:
   v5 .9098/.9559/.9005; E001 PAE-mix .9162/.9626/.9023; E002 clean-only
   .9148/.9576/.8942; E005 edge-snapshot×4 .9133/**.9652**/**.9073**. Any extra
   training helps ~+.005 micro; edge renders help the exact/family metrics most,
   PAE the micro metric. Single-seed differences (~.005) are near noise. Token
   selection scores swing .963–.979 between evals (batch 1): checkpoint
   selection is noisy → testing EMA weights (E006 vs control F002).
   Test (edge) exact F1: v5 no merge .9207, v5+merge .9511, E001 .9585, E005 .9596.

9. E004 (edge ×6 of 2,889 renders + clean, 24k): val .9199/.9635/.9099, best
   checkpoint = final step (not converged). Beats E005 (snapshot ×4, 16k) and
   E001/E002 on micro and fam_macro. Heavier edge share + more steps helps.
   Full edge train corpus: 6,927 accepted of 9,586 designs.

10. E003 (edge ×2 + clean, 24k): val .9161/.9623/.9075, test .9030/.9589/.9061 —
    below E004 (edge ×6). Edge share ordering holds: ×6 > ×2 ≈ PAE/clean.

11. Plateau: E004 (edge×6, 24k) .9199, F002 (edge×3, 24k) .9203, F001 (edge×3,
    **48k**) .9191, E007 (full corpus ×3, 32k) **.9215** edge val micro. Doubling
    steps gives nothing; seed/recipe noise ≈ ±.003.
12. Remaining misses in exact families (E004, 300 val renders): 3,223 missed
    refs, median length **2.2 px** (90% < 5.4 px), mostly short lines merged into
    neighbours — sub-resolution at 1.1 px contour spacing. Added bench
    `resolv_f1` (exact families, refs ≥ 4 px, predictions mostly on ignored refs
    dropped): v5 .9465 → v5+merge .9784 → E004 .9871 → E007 .9882 (edge val).
    Resolvable error rate 5.4% → 1.2%. The synthetic benchmark is close to
    saturated; real captures are the main missing evidence.

13. EMA .999 (E006) vs matched control F002 (clean ×1 + edge ×3, 24k): val
    .9169 vs .9203, test .9083 vs .9041 — no consistent gain (noise). EMA narrows
    the validation-score swing (.964–.974 vs .959–.977) but not final quality.
    Not adopted.

14. E008 (E004 recipe + boundary pos 8 / weight .6): val .9134/.9600/.9839/.9058
    vs E004 .9199/.9635/.9871/.9099 — worse. Keep the lighter boundary loss.
    High-res bench cut to 120 test designs (renders at 2.5–4k px are ~0.7/min).

15. F003 (full corpus ×2, 32k): val .9188/.9646/.9881/.9092 < E007 (full ×3)
    .9215. Edge share ordering holds on the full corpus too → E009 tests ×6.
    High-res bench build OOM-crashed at 10 workers (4096 px renders); restarted
    at 3 workers. H001/H002 set to 'held' (they blocked GPU lanes while waiting).

## Failed / not worth repeating

- `cut_merge` ≥ 1.5 token spacings: loses true short lines (PAE val).
- `min_run` class smoothing alone: negligible (PAE val).
- `relabel_tol` (arc→line when a line fits): catastrophic for shallow arcs.
- `merge_tol` 1.5+ over-merges; `peak_height` .6 with merge: no gain.
- EMA weights (.999): neutral. 48k vs 24k steps: neutral.
- Stronger boundary loss (pos 8 / weight .6) with merge post: worse (E008).
- Infra: `autoresearch/queue.py` shadowed stdlib `queue` inside ar_run (torch
  import failed after bench). A sys.path filter was not enough (module cached);
  renamed to `gpuqueue.py`.

## Open questions

- Real captures are not available here; synthetic renders are the only proxy.
  How close are edge-model renders to real GUI contours?
- Does PAE-render training (E001) transfer to edge-model contours vs clean-only (E002)?
- Would a fit-aware segmentation (DP over model boundary candidates with
  line/circle residuals, like `geometric.py`) beat heuristic cutting?

## If work resumes

1. Real captures: the synthetic benchmark is near saturation (remaining misses
   are ~2 px features and conventional breakpoints on smooth freeform curves).
2. High-resolution renders in training (v6 already improves most at 2.5–4k px,
   but was trained on 640–1920 px renders only).
3. Fit-aware splitting (pieces whose primitive fit is poor) to recover missed
   boundaries; imaging robustness (defocus, vignetting, dust) in the renderer.
