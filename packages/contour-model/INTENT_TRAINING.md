# Intent-aware learned model experiment — 2026-09-20

Status: training/evaluation complete. At the user's request, the step-8,000 best
candidate is now the GUI default for real-part evaluation, copied locally to
`release-v4-candidate/model.pt`. Restart the app to load it. Release-v3 is retained
unchanged; `INSPECTION_CHECKPOINT` can select it for rollback. This activation
does not imply real-part validation has passed.

## Focus

A nominal flat can be wavy without becoming several arcs. Pure bounded-error
fitting cannot recover that intent reliably. Context can provide evidence, not
certainty: indistinguishable shallow arcs and bowed flats remain ambiguous.

- Existing encoder: two shifted local-attention blocks, a three-layer global
  encoder over 128 pooled tokens, interpolated coarse context, local refinement.
- Candidate: 256 pooled tokens plus fine-to-global cross-attention, chunked into
  512-query blocks. A zero-initialized residual gate permits a release-v3 warm
  start without immediately injecting random context. Global/context dropout .1.
- Training augmentation: 75% of samples receive two smooth spatial displacement
  waves shared across every ring, independent of labels and feature boundaries.
  Wavelengths .04–.8 normalized part radii, maximum amplitude .003 per wave,
  also bounded by wavelength to keep the total displacement-field Lipschitz
  constant below .15. Supervision and nominal boundaries remain unchanged.
  Rotation, cyclic shifts, ring-order changes and the existing small noise remain.
- Boundary positive weight reduced 8 → 4 and loss multiplier .6 → .3. These are
  experimental settings, not a measured optimum; check missed small fillets too.
- Retain all explicit nominal line/arc/circle identities despite form error.
  Correct only the old reference-free fallback: nominal open spans prefer lines
  within max(.01 mm, .5% chord length), otherwise require an actual circular fit
  or subdivision. Complete freeform rings require a circular fit within .01 mm;
  lobed cams are no longer blanket circles. This heuristic convention still
  needs visual review. It never fits the imperfect observation to choose labels.

Synthetic generators were not changed in this focused experiment. Sharp-only
families, incomplete thread modeling, camera/extraction domain gap, and family
shortcuts remain open work. This is not the complete generator overhaul from the
earlier review. Waviness is applied after tokenization, so it does not reproduce
all changes in adaptive sampling or camera extraction. Physical-part evaluation
is still necessary; better synthetic scores alone are not acceptance evidence.

## Storage and execution

All new corpus/training outputs are on the user's T7 disk:

```
/Volumes/T7 Shield/inspection-contour-training/primitives-intent-v1-20260920
/Volumes/T7 Shield/inspection-contour-training/experiments-intent-v1-20260920
```

The new corpus is an immutable overlay: unchanged entries reference the original
corpus by absolute path; changed entries contain rebuilt targets and retained raw
provenance. Do not delete or move the parent `primitives-three-base-v2` corpus.
All original split/design assignments and observed coordinates are preserved.
All 12,000 records are included. No parent files are overwritten.

Launcher: `scripts/run_intent_training.py --launch` (must run outside the sandbox
for MPS and disk writes). It is single-use for these dated output directories.
It detaches, logs stdout/stderr, and uses `caffeinate -i` to prevent idle sleep
while running. Closing the lid or disconnecting the disk can still interrupt it.
Keep the MacBook plugged in, lid open, and disk connected.

Training: width 64, 805,334 parameters, MPS, 12,000 updates, warm start release-v3,
AdamW learning rate .0001, cosine schedule, query weight 1.5, validation every
500 steps on 200 fixed validation designs. Best/last checkpoints include optimizer
and RNG state. Existing trainer does not expose a resume CLI; interrupted work
requires explicit recovery, not blindly rerunning this launcher.

`status.json` identifies building_cache / training / evaluating / complete / failed.
`pipeline.log` reports dataset progress and training progress every 100 updates.
`candidate/history.jsonl` reports clean and deterministic wavy validation scores.
Checkpoint selection is 70% clean + 30% wavy validation selection score; each
component uses the existing base/compound/repeat-instance metric mixture.

After training, the same 400 held-out test designs are evaluated for release-v3
and the candidate on original, .003 waviness, and stronger .006 waviness inputs.
Raw-edge base and compound instance precision/recall are also compared on 200
held-out designs, using corrected reference labels. Test scores are not used to
choose training checkpoints. These are still same-generator holdouts.

The pipeline deliberately does not replace the GUI checkpoint automatically.
Inspect `comparison.json`, `release-v3-raw-test.json`, `candidate-raw-test.json`,
and representative real captures before promotion.

## Launch-time verification

- 223 Python tests passed (contour, measurement, GUI worker).
- New tests: label-blind deformation with unchanged nominal targets, backward
  gradients through cross-attention and zero-gate warm-start equivalence,
  nominal-only freeform relabeling, lobed closed freeform not labeled circle.
- 50-record cache smoke test passed, with six corrected records and unchanged
  encoded observations verified against the source cache.
- Two-step MPS forward/backward/evaluation smoke test completed successfully.
  Smoke scores are only a wiring check, not evidence of improvement.
