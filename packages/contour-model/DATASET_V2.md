# Expanded contour corpus

The v2 corpus contains 50,000 new parts from 47 procedural families. See
[the expanded taxonomy](../../research/synthetic-shapes/expanded-part-taxonomy.md) for family
coverage, feature semantics and the milder physical imperfection distribution.
The original datasets and v1 release remain separate.

| Artifact | Location on this machine |
|---|---|
| Raw observations, nominal references, metadata, uniform adaptive cache | `/media/T7Shield/inspection-contour-training/v2-expanded` |
| Corner-focused adaptive cache, referencing the same raw observations | `/media/T7Shield/inspection-contour-training/v2-local-cache` |
| All 50,000 reference contours, 23–24 per contact sheet | `/media/T7Shield/inspection-contour-training/v2-contact-sheets-reference/index.html` |

The 40,011 training, 3,979 validation, 3,923 test and 2,087 challenge parts are
partitioned by deterministic design ID. There are no alternate images of one design
across these splits. Challenge cases use the same family vocabulary: report severe
form errors separately from extreme parameter settings, which can overlap the
training parameter range. This is not evidence of unseen-family generalization.

## Files and labels

Each corpus shard holds 500 compact training NPZs. Its `sources` directory retains
`ID.contours.npz` and `ID.json`. Coordinates remain in millimeters; each ring closes
implicitly. The original contour NPZ contains:

- `actual_xy_mm`, `nominal_xy_mm`, `ring_offsets`;
- `edge_feature_id`, `edge_semantic_bits`, `edge_anomaly_class`.

Metadata gives part family, nominal feature definitions, emitted patterns and
relationships, split, random seed, severity, stress/complexity stratum, applied
imperfections, defects and realized form-displacement statistics. The model receives
only actual coordinates and ring connectivity. Ground truth and nominal geometry
are used to construct targets and evaluate predictions.

The cached NPZ contains geometry encoding `x`, ring `offsets`, physical interval
`weights`, primary and multilabel semantic targets, instance masks/classes, boundary
events and defect targets. The label adapter groups contiguous tab/notch walls and
adds consistent external-profile ancestry. A whole internal polygon or rounded
opening is a leaf compound; external polygon/rounded ancestry can overlap leaf
surface labels. `observed-compounds-v1` is the adapter version.

`v2-expanded/index.json` supplies relative cache paths and counts. The local cache
index also supplies absolute `source_contours` and `source_metadata` paths back to
the raw corpus. Keep both directories together, or update those two path fields
when relocating them. The absolute paths are not model inputs. `config.json`
identifies the sampling protocol and source/index hashes; mismatched model/cache
protocols are rejected during training and evaluation.

## Build another corpus

From the workspace root, with both Python packages installed:

```bash
python -m contour_model.corpus_v2 /path/new-v2-corpus --count 50000 --seed 9237 --workers 6
python -m contour_model.recache /path/new-v2-corpus /path/new-v2-local-cache --workers 4
python -m contour_model.contact_sheets /path/new-v2-corpus /path/new-review --workers 4
```

Use a fresh destination for a changed generator or sampling implementation.
Source/config fingerprints prevent accidental mixing. Interrupted builds retain
completed records and use an exclusive lock; confirm the original process has
exited before removing a stale lock and resuming with the same frozen sources.

The delivered raw corpus's `generation-source.tar.gz` preserves its exact
fingerprinted generator version. The local cache's `source.tar.gz` preserves its
three fingerprinted preprocessing files. The U-channel family-name correction is
recorded in `taxonomy-name-correction.json`; numerical arrays were verified unchanged.
These snapshots explain why current source may have a different hash after later
validation or inference fixes. Use the frozen sources for exact reproduction of
an existing artifact, and current source for a newly versioned corpus.

`observable-feature-groups.json` supplies a measurement-reference layer for the
frozen corpus. It aliases adjacent nominally collinear flat IDs in 4,254 parts;
the early comb/fin generator subdivided one continuous land at a pattern-cell seam.
The same issue occurred at the trailing flat of threaded-zone compositions.
No observed coordinates, semantic classes, nominal coordinates or training caches
were changed. `validation.observable_reference_labels` applies these physical-surface
groups for handoff evaluation; nominal geometry is never used by prediction or fits.
The current comb/fin generator shares feature IDs across those cell seams.

## Train and evaluate

```bash
python -m contour_model train /path/new-v2-local-cache /path/new-run \
  --adaptive --width 64 --queries 64 --repeat-queries --instances --relative-noise \
  --steps 60000 --device cuda --val-limit 500 --eval-every 3000
python -m contour_model evaluate /path/new-run/best.pt /path/new-v2-local-cache \
  --split val --device cuda --output /path/full-validation.json
python -m contour_model measure-evaluate /path/new-run/best.pt /path/new-v2-corpus \
  /path/feature-handoff.json --split val --limit 300
python -m contour_model export /path/new-run/best.pt /path/model.pt
```

This is a supported fresh run, not an exact reproduction of the delivered staged
initializations. Saved run configurations and validation documentation record the
actual schedules. A step samples one part with replacement; it is not an epoch.
Semantic-only controls omit `--instances`; their learned compound associations
must not be treated as trained. Repeat-query metrics cover thread/tooth instances;
raw handoff metrics include all leaf features and are the relevant measurement
selection check. Always select checkpoints on validation before evaluating test.
Current repeat-only runs select `best.pt` using 90% semantic macro F1 and 10%
repeat-instance F1; the initial v2 experiments used the older equal-weight score
and are compared again on full validation and raw selections. A changed selection
policy requires `--init` rather than silently resuming with incompatible scores.
For a staged alternative, initialize from a trained semantic checkpoint and add
`--instances --freeze-backbone`: only query/association parameters then update.

Adaptive models choose geometry-only per-ring resolution, with a total limit of
65,536 intervals. There is no one-query-per-hole limit. Existing v1 checkpoints
retain their fixed 1,024-interval contract and should not use these adaptive caches.
The 47-family grammar does not remove the single-part, closed-contour, non-touching
ring input requirements. Fits and measurements always use the original observations.
