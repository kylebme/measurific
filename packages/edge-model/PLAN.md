# Subpixel Mask Super-Resolution — Autonomous Research Plan

## Objective

Build and autonomously improve a model that maps a small backlit inspection-image patch to a **16× higher-resolution binary object mask**.

Start with a **48×48 input patch → 768×768 binary mask**. The eventual contour is extracted from this predicted mask; do not introduce explicit contour regression, physics-informed losses, SDFs, or transformers unless experiments later justify them.

The goal is subpixel boundary accuracy and reliable recovery of difficult geometry: closely spaced edges, acute features, smooth curves, sharp corners, variable contrast, blur, noise, and illumination.

## Inputs

The upstream dataset consists of rendered part images and corresponding binary masks rendered by Blender at 16× the image resolution.

Do not generate or store a patch dataset on disk. Load/cache source renders and masks as practical, then generate training patches dynamically.

For each sample, choose a source image and jointly sample the image and its 16× mask using the same random crop geometry, including arbitrary rotation and subpixel translation. Image/mask registration is critical; validate the sampling implementation carefully before training.

Apply modest input-only postprocessing randomization after geometric sampling. Start with sensor-like noise plus small exposure/gain and offset changes. The Blender renders already contain the major appearance randomization including blur, so avoid an unnecessarily elaborate augmentation stack initially. The Blender renders are 16bit depth, so one key postprocessing randomization to add is to squash the bit depth. Real use of the model will sometimes be with 8bit, 10bit, 12bit, or 16bit data, so that should be represented in the training.

Training and validation source images must be separated at the **source-render level**, not merely by patch.

## Baseline model

Implement a compact convolutional image-to-image model.

Use a native-resolution convolutional/residual feature extractor followed by progressive learned upsampling from 48×48 to 768×768. Keep channel counts low in the highest-resolution stages so the model fits comfortably on the RTX 3070.

Output one logit per high-resolution mask pixel.

Start with a straightforward binary segmentation loss such as `BCEWithLogitsLoss`. Because patches should normally contain useful boundary geometry, avoid adding complicated losses until the simple baseline is measured.

Do not start with a transformer. The initial research question is how far a well-trained CNN can take this direct super-resolution formulation.

## Patch sampling

Bias sampling toward regions containing boundaries, while retaining some easy foreground/background and lower-density geometry.

The sampler should expose enough controls that subsequent experiments can alter the mixture of dense geometry, sparse geometry, corners, curves, narrow gaps, and other difficult regions without changing the source dataset.

Random rotation and translation should be continuous rather than restricted to integer pixels.

Before serious training, create visual and numerical registration tests showing that transformed input patches correspond to the correct transformed 16× masks. A target error of one high-resolution mask pixel is already 1/16 of a sensor pixel, so silent transform misalignment will dominate the experiment.

## Evaluation

Create a deterministic held-out validation benchmark and treat it as fixed research infrastructure.

Training loss is not the primary research metric. Extract the predicted 0.5 mask contour and compare it with the 16× ground-truth contour in original sensor-pixel units.

Track at minimum:

* mean and high-percentile boundary localization error;
* missed/extra boundary rate, especially for closely spaced features;
* ordinary mask loss/IoU as secondary diagnostics;
* throughput, peak VRAM, and GPU utilization.

Choose one stable boundary-centric scalar as the primary experiment score, but retain the full metrics so an apparent improvement cannot hide a topology or recall regression.

Save representative prediction overlays for each serious experiment.

## Initial validation milestones

First prove the pipeline rather than tuning the architecture.

The system should be able to overfit a tiny sample set, accurately reproduce a few individual patches, demonstrate image/mask registration under random transforms, run deterministic validation, save/resume checkpoints, and keep the GPU fed continuously.

Then establish one untouched baseline configuration and record its accuracy, throughput, memory use, and learning curves.

Only after that baseline is trustworthy should autonomous architecture research begin.

## Autonomous research loop

Treat this as an autoresearch project, not a predetermined hyperparameter sweep.

Maintain a compact persistent research specification plus a human-readable `RESEARCH.md`. `RESEARCH.md` must survive context compaction and always contain the current best configuration, best score, important decisions, failed ideas worth remembering, unresolved questions, and the next few plausible experiments.

Also keep an append-only machine-readable experiment ledger containing run ID/commit, configuration, training budget, primary and secondary validation metrics, peak VRAM, runtime, and a short description/result.

Do not put the entire research history into `AGENTS.md`; keep repository-wide agent instructions minimal.

For each research iteration:

1. Inspect the current best result and recent failures.
2. Form one useful hypothesis.
3. Make the smallest change needed to test it.
4. Train under a comparable compute budget.
5. Evaluate on the fixed benchmark.
6. Keep, revert, or branch from the result based on evidence.
7. Record the conclusion in the durable research state before moving on.

Avoid blind Cartesian hyperparameter searches. Use previous results to choose the next experiment.

## Research priorities

Investigate roughly in this order, changing the order when evidence warrants it:

**GPU/data pipeline first.** Maximize useful RTX 3070 utilization. Use CUDA mixed precision and benchmark appropriate PyTorch/CUDA optimizations. Tune batch size close to the available 8 GB VRAM without making runs fragile. Ensure dynamic patch generation and mask transforms do not leave the GPU waiting.

**Model capacity.** Explore feature width, residual depth, receptive field, and the amount of computation performed before upsampling.

**Upsampling architecture.** Compare sensible progressive decoder choices such as resize/convolution and pixel-shuffle-style learned upsampling. Watch for checkerboard or periodic boundary artifacts.

**Loss and supervision.** Once plain BCE is understood, test boundary weighting, auxiliary lower-resolution supervision, or other simple losses that directly improve contour accuracy.

**Patch/context size.** Start at 48×48. Test smaller or larger context only when the baseline makes this worthwhile.

**Sampling curriculum.** Vary how strongly training emphasizes dense features, narrow gaps, corners, and other difficult geometry.

**Post-render augmentation.** Determine how much noise/exposure randomization helps held-out generalization without damaging precision.

**Larger architectural changes.** Only after the CNN baseline appears to plateau should attention blocks, transformer components, or substantially different model families enter the experiment queue.

Do not change multiple major axes simultaneously unless the result of the experiment specifically requires it.

## Compute strategy

The RTX 3070 should spend as much wall time training as practical while the agent spends as little wall time idling on an active context as possible.

Early in the project, calibrate a **fixed research-run budget** that gives meaningful separation between ideas and fits roughly within a fraction of an hour to about an hour. Use shorter runs to screen ideas and longer confirmation runs for configurations that look promising.

Work in approximately hourly autonomous cycles. At the beginning of a cycle, review results and prepare enough training/evaluation work to keep the GPU occupied for most of the cycle. Launch it using durable/background execution, then use the coding environment's native scheduling or long-running-task mechanism to resume research approximately an hour later rather than busy-waiting.

When resuming, first read `RESEARCH.md` and the experiment ledger, inspect the completed runs, update the durable state, and choose the next bolus of work.

If the environment cannot self-schedule agent turns, leave the training/evaluation runner resumable and checkpointed rather than holding an interactive agent process open solely to wait.

## Reproducibility

Every meaningful run must be reproducible from repository state plus its recorded configuration.

Keep the evaluation harness stable during architecture comparisons. Changes to validation data or metrics are research-infrastructure changes and should be explicitly recorded rather than mixed into model experiments.

Save checkpoints for the current best model and for experiments whose behavior is scientifically useful even if their scalar score is worse.

## Scope boundaries

The synthetic Blender scene-generation system is out of scope except for defining the image/mask interface.

Do not replace the 16× mask target with hand-generated analytic contours.

Do not introduce explicit optical models or physics-informed objectives.

Do not optimize for conventional segmentation benchmarks at the expense of boundary localization.

Do not prematurely build the production full-image contour-stitching system. Establish that patch-level mask prediction is accurate first.

## Desired outcome

Produce:

* a fast, reproducible dynamic patch-training pipeline;
* a strong 48×48 → 16× mask baseline on the RTX 3070;
* a fixed contour-centric validation suite;
* a durable autonomous experiment loop with clear human-readable research history;
* a sequence of evidence-backed architecture/training improvements;
* the best model and configuration found, together with enough experiment history to understand why it was selected.

There is no arbitrary accuracy threshold at this stage. Continue autonomous research until the experiment space clearly plateaus or the human stops the run.
