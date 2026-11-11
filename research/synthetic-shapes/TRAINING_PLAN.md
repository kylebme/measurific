# Plan: contour-only semantic transformer

Status: the contour training and measurement-handoff implementation is in
[`../../packages/contour-model/README.md`](../../packages/contour-model/README.md), with results and current
limits in [`../../packages/contour-model/VALIDATION.md`](../../packages/contour-model/VALIDATION.md).
The original proposal below remains the longer-term experimental roadmap. The
initial release uses approximately 1,024 intervals and 50,000 synthetic parts;
real-data adaptation, confidence calibration, comprehensive ablations and production
measurement qualification remain future work. Synthetic rendering is CPU-only;
model training supports the RTX 3070 and inference supports CPU.

## 1. Task and output contract

Train a small hierarchical transformer on ordered contours. For every observed
contour edge, predict semantic membership, feature boundaries, confidence, and
ultimately feature-instance membership. Pass the selected original contour spans
to classical measurement routines. Do not replace imperfect geometry with predicted
ideal geometry before measurement.

Inputs at deployment: extracted closed contours for one part, ring connectivity,
and optionally calibration. Neither grayscale images nor nominal CAD are model
inputs. Start with complete silhouettes; fragmented/occluded contours are a later
explicitly trained operating mode.

Output should contain:

- Per-edge semantic probabilities, including overlapping ancestor classes.
- Feature instances with one or more contour spans, class and confidence.
- Boundary probabilities and a separate optional burr/nick probability head.
- References back to original observations and explicit low-confidence/unsupported
  status, so downstream fitting can decline uncertain selections.

Circle diameter, profile residuals, slot width, thread pitch and similar values
come from the downstream geometry, not from transformer regression. Semantic
recognition does not establish compliance with a thread standard or a 3D tolerance.

## 2. Audit and version the training labels first

The current 1,200-sample dataset is useful for plumbing and debugging. It has only
100 threaded parts and 79 rounded-profile feature instances, despite many labeled
edges. Its 12 grammars are not enough evidence of broad transfer.

Before expanding training, create a versioned training-label adapter and review
colored contour examples for every class. Do not modify the delivered dataset.

1. Make hierarchy consistent. Some outer rings have an external-profile ancestor;
   polygon spacers, cams and toothed disks can instead start at their specialized
   class. Define a consistent external-profile superclass on outer boundaries.
2. Define instance scope. Slots are whole instances, a thread already spans two
   disconnected sides, while bracket tabs/notches currently have separate IDs per
   surface. Group the three surfaces of a tab/notch into a compound instance in
   the training adapter. Decide and document whether opposite annular-groove
   silhouettes are one compound instance or two observed subfeatures.
3. Retain nominal semantic identity under form error. Lobed holes remain holes,
   and bowed surfaces remain their surface feature. Defects are a separate task.
4. Keep geometry and intended use distinct. Some grooves/notches and thread/rack
   profiles can be indistinguishable from contours alone. Allow a generic geometric
   superclass and uncertain semantic interpretation rather than enforcing an
   unobservable distinction. Audit examples and report both geometric and specific
   semantic performance.
5. Do not teach the model that every primitive junction is a feature boundary.
   A thread contains many primitives but is one semantic feature. Boundary targets
   come from the agreed feature-instance partition, including same-class neighbors.

Freeze label version and a small manually reviewed gold set before large runs.

## 3. Contour representation and preprocessing

Use `actual_xy_mm`, ring offsets and ring roles. Use `edge_feature_id`,
`edge_semantic_bits`, feature ancestry and anomaly arrays only as supervision.
Nominal coordinates, primitive IDs/kinds, family names, generator parameters,
pattern membership and seeds must never enter the model input.

**Resample without consulting labels.** Uniformly resample each entire ring by
observed arc length, using the exact same algorithm in training and deployment.
Do not split the sampling grid at known feature/primitive boundaries. The current
primitive-based sampling can otherwise reveal artificial seams or point density.
Assign targets afterward through source-edge overlap; use length-weighted soft
membership on intervals crossing a semantic boundary. Keep the mapping to source
arc length for reconstruction of predictions.

Use a variable token count with a starting budget of 2,048–4,096 edge-interval
tokens per part, minimum allocation per ring, and masks for padding. Choose final
spacing by a resolution sweep: the smallest feature we intend to identify must
have several observed samples. Do not silently drop small holes or force an overly
complex part into a budget that destroys its chamfers; raise its budget or reject
it as outside the initial operating envelope. Uniform resampling creates a model
representation, not new independent measurement information.

Each token initially contains normalized edge midpoint, local displacement/tangent,
normalized interval length, and hole/outer role. Add curvature/turning estimates
at multiple physical neighborhood scales as an ablation; compute them from the
observed contour, not nominal primitive fits. Stabilize their estimates on noisy
contours and retain the unsmoothed observations for measurement.

Center and uniformly scale the **whole part**, preserving all inter-ring sizes and
positions. Use a rotation-invariant scale such as centered maximum radius. Keep
the inverse transform outside the model for measurement. Do not normalize holes
independently. Default semantics should not depend on physical size; add known
calibration as a separately tested input only if it serves a specific requirement.

Canonicalize winding using containment, randomize cyclic start positions and ring
order, and apply random in-plane rotations. Reversal is an optional augmentation
with tangents/curvature recomputed consistently. Numerical ring IDs are routing
information, not learned categorical features. Remove dependence on generator axes
or a particular first contour point.

## 4. Model: local contour attention plus global part context

Recommended first transformer: a compact 1D encoder-decoder over multiple closed
loops, with high-resolution local attention and global attention after pooling.
The use of curve connectivity is motivated by CurveCloudNet, which processes
point clouds as polylines. That paper supports the structural idea, not a claim
that this proposed 2D architecture is already validated for inspection.
[CurveCloudNet, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Stearns_CurveCloudNet_Processing_Point_Clouds_with_1D_Structure_CVPR_2024_paper.html)

Proposed starting configuration, subject to profiling:

| Component | Initial design |
|---|---|
| Input embedding | MLP to 96 channels |
| Encoder | Three resolutions, widths 96 / 144 / 192; two blocks per level |
| Fine attention | Overlapping/shifted windows of about 64 tokens along each ring |
| Coarse attention | Full attention over roughly 128–256 pooled part tokens |
| Position information | Relative displacement and cyclic arc separation; no privileged start index |
| Decoder | Within-ring interpolation and skip connections to edge resolution |
| Main head | 16 independent semantic logits after the label audit |
| Boundary head | Feature-boundary probability at each interval endpoint |
| Optional defect head | Separate none/burr/nick prediction |

Local windows wrap on closed rings and never continue into another ring. Pooling
is within rings, retains small rings, and includes their positions relative to the
part. Global tokens attend across rings and distant portions of the outer contour:
this enables reasoning about opposite slot walls, shaft sides and hole context.
Pass that global context back to fine tokens through the decoder.

Implement actual windowed attention, not a dense N-by-N attention matrix with a
local mask. Use native PyTorch attention and ordinary Python modules; start without
custom CUDA extensions. PyTorch provides scaled dot-product attention with backend
dispatch, but kernel selection and memory must be measured for the actual shapes.
[PyTorch attention documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

Keep a small circular 1D convolutional encoder-decoder as a control experiment.
Compare it with the transformer under identical data and evaluation. A generic
point-cloud transformer is a later comparison, not a requirement to import a large
3D stack. PTv3 is a useful scaling reference, but its published implementation is
built around point-cloud processing rather than the explicit ring structure here.
[Official Point Transformer V3 implementation](https://github.com/Pointcept/PointTransformerV3)

## 5. Losses and instance grouping

Start with semantic and boundary supervision:

- Class-balanced binary cross-entropy plus per-class Dice for overlapping semantic
  memberships. Normalize by represented contour length, with bounded class weights
  and feature-balanced sampling so long flats do not dominate short chamfers.
- Boundary focal/BCE loss with a small arc-length tolerance around true transitions.
- Low-weight optional defect loss. Do not let abundant background/none examples
  dominate or let the defect task suppress recognition of the underlying feature.
- Optional hierarchy-consistency penalty once the adapted hierarchy is reliable.

Derive a mutually exclusive primary semantic evaluation from primary feature IDs;
keep it distinct from multilabel ancestor performance. Do not train a single
softmax that forces a thread edge to choose between thread and external profile.

First group adjacent predictions using class and boundary probabilities, including
wraparound at a ring seam. This gives contiguous observed feature spans, not final
compound instances. A feature may span disconnected runs or multiple rings.

After semantic performance is sound, add a small query decoder (start around 64
queries, after measuring target-instance counts) that predicts a class and a mask
over contour tokens per query. Train with bipartite matching and class/mask/Dice
losses. This can group both thread sides or separated compound surfaces without
assuming contiguity. Preserve hierarchy as a separate relation/ancestor layer;
nested features must not compete through exclusive per-token instance assignment.
Mask2Former's mask-query approach is the inspiration, adapted here to contour
embeddings rather than images.
[Mask2Former, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html)

## 6. Data expansion and synthetic-to-real transfer

Use the delivered 1,200 parts for smoke tests and visualization. Create a separate
CPU-generated contour-only training corpus: start with roughly 50,000 unique
designs, plus independently seeded fixed validation/test sets of roughly 5,000
each. Scale toward 200,000–500,000 only after the learning curves and failure cases
show what additional data is useful. These are planning sizes, not required minima.

Before scaling, extend the grammar to break family shortcuts: threaded shafts with
grooves/shoulders, holes on different outer shapes, slots on different plate
outlines, rounded and polygonal variants, and repeated profiles in new contexts.
Vary dimensions and styles without assigning each semantic class a distinctive
size range or fixed position. Include generic contours, near-circles, unusual
profiles and true negative examples to measure false positive behavior.

Separate manufacturing deviations from contour-extraction errors. The generator
already supplies the former. Add observation augmentations for subpixel jitter,
correlated localization error, uneven sampling, smoothing bias, quantization,
small outliers and residual calibration/distortion errors. Tune amplitudes to the
intended contour extractor; avoid arbitrary large independent jitter that changes
the identity of small features. Start with complete closed loops and handle gaps
only when fragment connectivity/validity is represented explicitly.

Build a smaller bridge set by rendering parts on CPU and extracting contours with
the actual intended classical image-processing pipeline. The model still receives
only extracted contours. Transfer labels through ring matching and monotonic/local
arc-length correspondence; ignore uncertain matches and topology-changing cases
until they have an explicit policy. Global nearest-neighbor label transfer alone
can jump between adjacent thread flanks or nearby hole/outer boundaries.

Acquire an initial 100–300 manually reviewed real contour examples spanning the
expected part types and capture conditions, then expand around failures. Keep a
locked real test set containing different physical parts/designs from tuning.
Unlabeled real contours can later support consistency/self-supervised training,
but a supervised synthetic baseline and real evaluation come first.

## 7. Training schedule and hardware

Create a separate `contour-model` project with a pinned PyTorch environment.
Leave `inspection-shapes` and its dependencies CPU-only. For model training, the
RTX 3070 is the proposed accelerator; CPU is sufficient for preprocessing,
unit tests and early pipeline debugging. Do not assume all 8 GB is free.

Starting optimizer settings: AdamW, learning rate 3e-4, weight decay 1e-2,
5% warm-up, cosine decay, gradient clipping at 1.0. Use these as an initial trial,
not fixed optimal values. Start with 2–4 parts per microbatch and gradient
accumulation toward 16–32 effective parts, reducing the microbatch if necessary.
Use mixed precision with loss scaling where supported, and profile a real training
step before allocating a long run. Keep a full-precision reference for stability.
[PyTorch mixed-precision recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

Train in stages:

1. Overfit 16–32 mixed examples with no augmentation. Inspect predictions and
   boundaries; failure here blocks larger runs.
2. Train the convolutional control and semantic transformer on the expanded corpus.
   Compare held-out geometry, fine-feature recall and extraction robustness.
3. Add the instance decoder after semantic errors are understood.
4. Mix bridge/real examples and fine-tune at a lower learning rate, keeping synthetic
   examples to limit forgetting. Tune confidence thresholds on validation only.
5. Repeat the selected configuration with three seeds and evaluate locked tests.

Measure throughput, peak memory and a short learning curve before estimating wall
clock time. Use step-based budgets with validation-based checkpoint selection and
early stopping; do not launch a million-shape run simply because generation is cheap.

## 8. Evaluation and acceptance gates

Preserve the existing same-design split rule, but create additional tests for new
feature combinations, held-out profile styles, dimension extrapolation, severe
form deviations and unseen extraction conditions. Holding out an entire family is
only a valid supervised test for classes also present in training; otherwise mark
it explicitly as an open-set test. Random seeds alone do not test grammar transfer.

Report:

- Arc-length-weighted per-class IoU/F1, macro averages, and primary vs ancestor
  metrics separately. Do not let the external-profile class hide poor chamfer recall.
- Recall and precision per feature instance; token-mask IoU weighted by arc length.
- Boundary precision/recall at a declared physical or pixel-equivalent tolerance.
- Fragmentation/merging errors, including both sides of threads and small holes.
- Results by part family, feature size, severity, profile style and extraction mode.
- Invariance/equivariance checks under start-index shifts, ring permutations,
  rotations, density changes and reversal when supported.
- Confidence calibration, risk-versus-coverage under abstention, and false positives
  on generic/unseen contours.
- End-to-end measurement success and failure rate: run the same fitter on model-
  selected spans and oracle-selected spans from the **same imperfect observations**.
  Report selection-induced measurement differences rather than attributing actual
  manufacturing deviations to model error. Keep all original points available;
  defect exclusion is a declared fitting policy, not hidden contour smoothing.

Provisional engineering gates: near-perfect mixed-example overfit; then at least
0.90 macro primary semantic F1 and 0.90 feature recall on the expanded in-distribution
synthetic validation set, with every small-feature class reported separately.
These are debugging targets, not evidence of production readiness. Final acceptance
requires application-specific false-positive/false-negative and measurement-error
limits on the locked real set. Define those limits with the inspection requirements.

## 9. Implementation order and deliverables

| Milestone | Deliverable / decision |
|---|---|
| A. Labels and data | Versioned label adapter, reviewed contour overlays, leakage-free resampler, immutable split lists |
| B. Baseline | Circular convolutional control, transformer encoder-decoder, small-set overfit report |
| C. Generalization | Expanded CPU contour corpus, bridge extraction set, class/size/robustness ablations |
| D. Instances | Disconnected-span grouping, hierarchy reconstruction, feature-level evaluation |
| E. Measurement | Classical fitter integration with raw contour references and abstention |
| F. Transfer | Real-data tuning, locked evaluation, CPU inference benchmark, reproducible checkpoints |

The first implementation milestone should be **A plus a 32-part overfit experiment**.
It will establish whether the representation and labels are learnable before
committing to a large corpus or a more complex instance transformer.
