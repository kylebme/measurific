# Delivery validation

Delivered dataset: `/media/T7Shield/inspection-synthetic-shapes`.

- 1,200 independent designs, 100 in each of 12 part families.
- 1,200 grayscale 1920 × 1080 images with vector and raster segmentation targets.
- 12 tar shards, 1,837,844,480 bytes total (1.84 GB decimal).
- Split counts: 945 train, 134 validation, 121 test.
- All 16 non-background semantic classes represented.
- 2,221 labeled burr/nick instances.
- Severity distribution: 96 ideal, 321 mild, 557 moderate, 226 exaggerated.

Every delivered sample passed shard checksum, ring topology, projection, ID/shape,
semantic-bit and image-polarity checks. See the dataset's `validation.json` for full
class counts. All 1,200 geometry seeds were also checked before rendering; the
largest number of rejected candidate parts for a seed was three.

The current CPU-only implementation passes **20 automated tests**, including
contour-only reconstruction, independent surface nonparallelism, circle form error,
zero-imperfection identity, image/label consistency, repeated-view split isolation,
round-trip reading, resume, partial-shard overlap and corruption detection.
A separate 12-sample 1080p run after removal of GPU support also passed validation.

Visual inspection covered the first 24 delivered samples and detailed panels for
thread edges, a bore burr, and a defective slot. Those panels show grayscale,
filled semantic regions, boundary labels, and magnified nominal/actual contours.
Artifacts: `contact-sheet.png` and `diagnostics.png` in the dataset folder.

The initial dataset finished using the earlier renderer, as requested after its
run had started. GPU support and CUDA dependencies were then removed from the
reusable generator. The supplied `cpu-generator-source.tar.gz` contains the current
CPU implementation; it is not the exact historical source identified by the
initial dataset's config hash. That hash and original environment are preserved
for provenance. Use a fresh output directory for CPU runs; existing records can
be reconstructed and rendered directly using `part_from_record`.

This validation establishes internal geometry/annotation/rendering consistency.
It does not validate manufacturing standards conformance, metrology accuracy, or
real-world model performance. Templates and deviations are domain randomization
models; real silhouettes remain necessary for transfer evaluation.
