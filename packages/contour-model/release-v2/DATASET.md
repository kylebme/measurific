# Data and training provenance

Large datasets remain in `/media/T7Shield/inspection-contour-training`.
`primitives-corrected-v1` contains 12,000 designs across 46 families:
9,586 train, 992 validation, 923 test, and 499 generator stress/OOD designs.
Only train designs participate in gradient updates; validation chooses checkpoints.
The OOD designation is a generator stress split, not proof of real-world OOD coverage.

The initial `primitives-v2` observations were relabeled into `primitives-whole-v1`
to make every compound own complete observable base instances, then into
`primitives-corrected-v1` to repair labyrinth primitive decomposition. All actual
and nominal observations, ring offsets, anomaly labels, and cached input features
remain bitwise identical for the 260 repaired labyrinth parts (2,026 opening rings).
The repair audit records hashes per part. A bad three-point circle fallback had
absorbed a long straight segment into a tiny rounded join; unequal-chord triples
now preserve the straight line and the true small-radius arc separately.

`primitives-independent-v1` contains 1,000 additional designs, seed 89237, generated
after the labeling correction. Its generator-origin train/validation labels do
not grant training eligibility: none of this cohort was used for weight training
or checkpoint selection. An evaluation-only index in
`packages/contour-model/artifacts/primitives-independent-cache` marks all 1,000 as test.
Isolated thread fragments found in its raw review informed the final repeat-support
detection guard; this cohort is not an untouched test of that guard. The guard
was additionally checked on 300 validation designs and the complete 923-design test.

Twelve experiments completed 76,000 optimizer updates, with two or three GPU runs
in parallel on the RTX 3070. They compare transformer warm starts, transformer
training from scratch, a convolution control, query-loss weights, and learning
rates. `reports/experiments.json` and each run's config/history record the schedules
and initialization chain. Label versions were repaired during the work, so early
scores must not be compared directly with corrected scores. The earlier staged
candidate in artifacts was withdrawn before release.

The final weights come from `primitives-corrected-final-b/best.pt`, step 2,000 of
its 3,000-step correction schedule (learning rate 0.00005, query weight 1.5).
Selection uses 70% validation base macro F1, 20% compound macro F1, and 10% query
instance F1; raw instance checks compare the finalists. This schedule starts from
previous weights; it is not the total training duration. `--init` does not restore
optimizer/RNG state. The final inference export omits optimizer state.

`reports` contains exact dataset configs and indices, integrity/repair audits,
selection reports, runtime versions, experiment histories, and final evaluations.
`source.tar.gz` freezes the delivered model, generator, tests, measurement adapter,
and GUI integration source. `source-files.json` hashes individual source files;
`MANIFEST.json` hashes release files. Earlier generation snapshots remain in artifacts.
