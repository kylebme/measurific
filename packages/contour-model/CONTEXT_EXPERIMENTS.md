# Whole-part context comparison and PAE follow-up

Requested 2026-09-20. Current GUI candidate stays unchanged.

## Stage 1: matched context pilot

Run `scripts/run_context_training.py --launch` outside the sandbox for MPS/T7.
It is a single-use, detached, finite pipeline with an idle-sleep inhibitor.

Outputs: `/Volumes/T7 Shield/inspection-contour-training/experiments-context-v1-20260920`.
Dataset overlay: `primitives-context-pilot-v1-20260920` on the same drive.

Three variants, each 4,000 additional steps, seed 73, starting from the exact
same active step-8,000 intent candidate:

1. 256 coarse tokens: continued-training control.
2. 1,024 coarse tokens: more spatial detail in whole-part context.
3. Full-resolution context: every fine token attends to every fine token in one
   context-attention layer. The existing 256-token coarse encoder is retained;
   this is not a replacement of all encoder layers with dense attention. Keys
   contain local features plus interpolated globally encoded context. Query
   chunking and activation recomputation reduce peak memory, not pair count.

Same data, split assignments, update counts, optimizer, learning-rate schedule,
augmentation, and validation IDs. Dropout paths differ, so not identical random
draws inside attention. Actual fine-token count is unchanged by the 1,024 option.
Adaptive sampling stays label-blind. Full attention means all adaptive tokens,
not every original raw contour vertex.

For a feasible first dense-attention test, ALL THREE variants use the same set
of complete designs with at most 8,192 tokens. No part is cropped or truncated.
The complete corpus ranges from 768 to 37,632 tokens (median 1,536; p95 13,376).
The pilot config records exclusions. Results cannot establish performance on
excluded high-complexity parts; test those separately before deployment.

Validation every 500 updates on 100 fixed designs; retain best/last checkpoints.
After all training, evaluate the active baseline and three variants on the same
200 test designs, original/wavy/stronger-wavy, plus raw-edge instance metrics on
100 designs. Report instance precision/recall, not only token scores. Compare
training duration and inference latency too. Do not automatically activate.

## Stage 2: render → production PAE → training

Implemented after stage 1 completed. Keeping it separate avoids confounding
context changes with input-domain changes. The original plan was:

- Review results briefly; use validation, not repeated test optimization, to
  select a practical architecture for the extraction experiment.
- Render actual imperfect contours with subpixel shifts, multiple resolutions,
  blur/contrast/noise variation. Retain their original nominal-derived base
  instance labels and compound membership as supervision.
- Call `inspection_measurement.contours_from_image` directly: the GUI wrapper
  may bypass PAE if a synthetic sidecar exists. The old `bridge.extract` is an
  OpenCV-only reference extractor and is NOT the requested production PAE.
- Transfer labels using ring-matched, cyclic-order-preserving geometric
  correspondence to pre-extraction observed edges. Never run handwritten
  segmentation on the PAE output to create new labels. Handle winding, segment
  endpoints, same-class joins and disconnected compound membership explicitly.
- Reject topology changes/ambiguous correspondence and log acceptance by family,
  feature size and resolution. Do not assign unreliable regions to background
  or silently remove difficult families. Verify boundary placement visually on
  a small pilot before a larger cache build.
- Keep every render of one design in its original split. Mix original and PAE
  training inputs. Compare equal-update original-only and mixed-input controls,
  on matched untouched-design original and PAE holdouts.
- Store all new data on T7. Preserve current caches/checkpoints. Start the finite
  generation/training job, schedule another infrequent follow-up and stop
  thinking while it runs. No automatic GUI promotion.

## Monitoring

Read only `status.json` and the log tail when a scheduled check fires. If work
is advancing, return immediately without analysis or notification. Diagnose
failures conservatively; never overwrite or blindly restart experiment folders.
User requested low active reasoning during long jobs. Keep Mac plugged in,
lid open, T7 attached, and the app available for the scheduled follow-up.

## Stage 1 results and stage 2 execution

Stage 1 completed. Base raw-instance F1: active baseline .8910; continued 256
control .9034; 1024 .8933; full .9056. Arc-instance F1: .8252 / .8370 / .8168 /
.8438 respectively. Full-context p95 inference .239s versus .136s for the
continued control. Token macro F1 did not improve over the active baseline.
These small single-seed differences are not evidence of broad superiority.
Validation selection scores: 256 .992884, 1024 .992806, full .993105. Stage 2
uses full-context based on this validation comparison, with matched controls.

`pae_corpus.py` renders actual imperfect contours at 640/960/1280-pixel longest
dimensions with 3x antialiasing, random subpixel phase, contrast, blur and noise.
It calls production `contours_from_image` directly (no synthetic sidecar bypass).
Ring matching is role-constrained. Dense source-edge search handles long edges;
projected cyclic progress must cover exactly one loop without significant
backtracking. All original base identities and their cyclic order must survive;
maximum vertex/edge-midpoint correspondence error is 1.5 px. No labels are
inferred from PAE curvature. Same-class boundaries and compounds are retained.
PAE geometry is converted back to source physical coordinates without smoothing.

A 30-render pilot accepted 19 and rejected 11. Endpoint-marked contact sheet was
visually inspected. This is a strict acceptance policy, with selection bias
toward resolved features. Audit records include family, split, minimum source
feature length in pixels, rendering settings, errors and rejection reasons.
The pilot is not proof of correspondence accuracy at every corner.

Stage 2 launcher: `scripts/run_pae_training.py --launch`.
Outputs: `/Volumes/T7 Shield/inspection-contour-training/experiments-pae-v1-20260921`.
PAE corpus: `primitives-pae-v1-20260921` on the same drive.
2,400 attempted designs, balanced round-robin across family/split cells. One
render per design in this first experiment; resolution varies between designs.
All source designs come from the same <=8192-token context-pilot subset.
All renders keep their original split/design_id; render sample IDs are offset
by 100,000,000. No original source or active GUI checkpoint is overwritten.

After generation, require at least 300 train, 40 val and 40 test accepted designs.
Train two 4,000-update variants from the same stage-1 full-context checkpoint:
original-only versus 50/50 original+PAE. Both use precisely the same accepted
design pool; both have identical original+PAE validation records. Best-checkpoint
selection is the existing clean/wavy validation mixture, not test feedback.
Compare both variants and their initializer on matched original/PAE test designs,
including raw feature-instance metrics. Rejected renders are NOT part of these
scores; report acceptance and rejected-family/feature-size coverage alongside
them. Original-only control is intentionally restricted to the same accepted
design pool to avoid attributing selection effects to extraction augmentation.

At the next timer, read stage-2 status and log tail only. If progressing, stop
immediately. On completion report paired scores, precision/recall regressions,
rejection coverage and real-part limitations; do not auto-promote. Delete the
follow-up after final reporting. If failed, diagnose safely without overwriting
or blindly restarting existing outputs.
