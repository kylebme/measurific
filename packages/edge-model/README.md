# edge-model: sub-pixel silhouette edges for backlit inspection

A small full-image model (265k parameters) that finds object boundaries in backlit
(dark-on-light) grayscale images to a few hundredths of a pixel. It is a classical seed
(background fit, Otsu threshold and gradient band, as in
`packages/measurement/src/inspection_measurement/adapter.py`) plus a learned fine CNN that decodes
16×16 sub-pixel occupancy near edges. Research history: `RESEARCH.md` ("FINAL STATE"),
`PLAN_v2.md`.

## Run the model (inference only)

Requirements: Python 3.13 and the packages in `requirements.txt` (torch, numpy,
opencv-python-headless, scipy, pillow). No GPU, dataset or cache is needed; the model is
`models/cfi_B_seed.pt` (1.1 MB).

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt

# contour points (.npy or .json) and an overlay image
.venv/bin/python -m edgesr.predict part.png --out points.npy --overlay overlay.png
# 12-bit data stored in 16-bit files: give the full scale
.venv/bin/python -m edgesr.predict part.tif --max-value 4095 --out points.json
# faster repeated CPU use after warm-up
.venv/bin/python -m edgesr.predict part.png --compile --threads 16
```

```python
from edgesr.predict import load, read_image
model = load()                                   # CPU; load(device="cuda") for GPU
result = model(read_image("part.png"))           # or any 2-D uint8/uint16 array
points = result["points"]                        # (N, 2) float64, x/y in source pixels
```

- **Input:** a 2-D grayscale image, dark part on a bright backlight, 8- to 16-bit. Any
  size; the frame is padded internally.
- **Output:** sub-pixel contour points in source-pixel coordinates (x right, y down;
  pixel i covers [i, i+1), so pixel centers are at i + 0.5). These are the
  crossings of the predicted boundary with the 1/16 px lattice, **unordered**. For
  ordered closed rings use `edgesr.runtime.EdgeModel.rings` (below).
- **Region filter:** closed regions smaller than `--min-object-area` (default 50 px²,
  e.g. dust specks) or `--min-hole-area` (default 8 px²) are removed.
- **Speed** (2048² image, 16 CPU threads): ~1.3 s with the region filter, ~0.9 s without
  (`--min-object-area 0 --min-hole-area 0`), with `--compile`. About 0.5 s on a GPU.

## Torch-free runtime (ONNX) and ordered rings

The inspection app and `inspection_measurement.contours_from_image` run the model
without torch. `models/cfi_B_seed-onnx/` holds two graphs: `fine.onnx` (gathered
seed windows → fine features and band logits) and `decode.onnx` (3×3 feature
neighbourhoods → 16×16 occupancy logits). `edgesr.runtime` does the rest in numpy:
seed, cell selection, canvases, region filter and contour tracing.

```bash
uv pip install -e './packages/edge-model[onnx]'       # numpy, scipy, shapely, opencv, onnxruntime
```

```python
from edgesr.runtime import EdgeModel
model = EdgeModel()                     # models/cfi_B_seed-onnx; EdgeModel(device="gpu") for CUDA/DirectML
points = model(image)["points"]         # same unordered crossings as edgesr.predict
out = model.rings(image, valid_mask=None)
out["rings"], out["is_hole"], out["depth"], out["parent"]    # ordered closed rings, largest first
```

`rings` traces the 0 level of the hi-res occupancy with marching squares across
cell borders. Objects are 8-connected, holes 4-connected. Each vertex is a
lattice crossing and consecutive vertices are ≤ 1/16 px apart. Object outlines have
negative shoelace area (x right, y down), holes positive. Invalid pixels and a
16 px apron around the frame are filled with the continued local backlight, a
normalized convolution of nearby background pixels, so parts cut by the frame or a
ROI close along it. A constant fill makes the model see an edge where two
background levels meet. A backlight brighter than 0.95 of full scale, e.g. clipped
at 255 in a JPEG, is scaled down to 0.95. That is the training maximum, and next to
a saturated backlight the model hallucinates extra edges.

The model sometimes predicts objects that are not there: faint fibers on the
backlight, vignetting ramps at the lens's image-circle edge, and the synthetic fill.
Two guards keep these out of the rings:
- **Trust margin:** predicted occupancy counts only within 2 px of valid pixels.
  Beyond that (fill, apron, frame padding) it is background, so boundaries always
  close.
- **Seed support:** the classical seed decides which rings exist and the model where
  they are. A ring whose own region (inside it, minus the rings directly inside it)
  is less than 50% seed object (outlines) or seed background (holes), within 1 px,
  is dropped with everything inside it. `unsupported_rings` counts these. Rings below the minimum areas are dropped with everything
inside them.

After changing the model, re-export and check parity with torch. The export needs
torch, onnx and onnxscript. The current export matches torch within 1.5e-4 logit and
1e-4 px:

```bash
.venv/bin/python -m edgesr.onnx_export --overwrite --parity ../../app/demo-data/drilled_plate.png
```

CPU timing on an M-series Mac is 0.7–1.8 s per 1920×1080 image for rings, mostly
the fine graph. The CoreML provider is about 6× slower, so the runtime does not
use it.

## SDF edge model v5 (PLAN_v3; GPU model)

`models/sdf_v5.pt` and `models/sdf_v5-onnx/` hold a local U-Net (3.0M params, receptive
field ≤ ±50 px). It regresses a truncated signed distance to the part boundary, with a
per-pixel uncertainty. The contour is the distance field's 0-level, so every edge is
estimated jointly by the pixels around it.

On val_v4_full it scores bscore 0.142 against CFI v4's 0.169 (`models/cfi_v4`):
- missed 5.9% against 7.0%, extra 4.3% against 9.7%;
- 12× fewer false contours;
- the same localization on sharp edges.

It needs a GPU to be fast: about 0.46 s per 2048² frame on an RTX 3070 (bf16, compiled),
against 26.5 s on a 16-thread CPU with ONNX Runtime. It is not yet the GUI model.

```python
from edgesr.sdf_infer import SDFPredictor          # PyTorch; GPU
points, seconds = SDFPredictor("models/sdf_v5.pt", fast=True)(reading)
from edgesr.sdf_runtime import SDFRuntime          # torch-free; ONNX Runtime
points, seconds = SDFRuntime("models/sdf_v5-onnx")(reading)
```

Training: `python -m edgesr.train_sdf configs/sdf_v5.json --run-id NAME` (warm starts
via `train.init`). Export: `python -m edgesr.sdf_export RUN/final.pt NAME`. Speed:
`python -m edgesr.latency_sdf models/NAME.pt`.

## Coarse model (part outlines, the fine model's input)

`models/coarse_v2.pt` (0.9M parameters) segments whole frames into parts at 512 × 512.
v2 adds thin wires (down to ~0.5 model px, 2 source px) to v1 (`models/coarse_v1.pt`),
while still leaving loose hairs and fibers out.
It finds parts the classical seed misses (light or shiny parts under ambient light) and
gives the fine model its mask and band. Details, benchmarks and the training data
(`synthetic-macro`, macro-v1-512): `COARSE.md`.

```python
from edgesr.coarse_infer import CoarseModel, inscribed_circle
cm = CoarseModel()                                  # device="cuda" for GPU
out = cm(image, valid_mask=inscribed_circle(image.shape))   # ROI optional
logit = cm.full_res_logit(out, image.shape)         # part logit at frame resolution (> 0 = part)
band = cm.band(out, image.shape, 8.0)               # pixels within ~8 source px of the outline
```

- Any frame size or aspect ratio: frames are resampled to 512 × 512, and the model is
  told the frame aspect ratio.
- An optional region of interest, e.g. the inscribed circle of a circular backlight;
  nothing outside it is a part.
- Precomputed maps for fine-model training: `python -m edgesr.coarse_maps build` writes the
  part logit, sdf and GT coverage of every physical_v3 scene to `work/cache/coarse_maps`.

## Accuracy and known limitations

The model was trained only on synthetic Blender renders (`../../research/synthetic-renders`). On 60
held-out full test images, sparse sharp edges localize to ~0.04 px mean (≈0.035 px on
silhouette-lit parts). See `RESEARCH.md` "FINAL STATE" for all slices and comparisons.
- Parts as bright as or brighter than the backlight (strong front or ambient light,
  light-colored or shaded parts) can be mis-segmented: the classical seed assumes
  dark-on-light.
- Strong specular highlights, heavy blur and sharp corners are less accurate. The
  information is lossy there.
- Under symmetric Gaussian blur of a flat 2-D edge, contours sit **+0.07–0.17 px
  outward** (σ 0.6–1.5 px; `tests/test_predict.py`). This is likely a prior learned
  from the renders' defocus of 3-D parts. Sharp edges are unbiased (0.02 px). Check it
  on real blurred images before relying on absolute dimensions there.
- Not yet validated on real images.

## Tests

```bash
.venv/bin/python -m unittest tests.test_predict tests.test_cfi      # no dataset needed
.venv/bin/python -m unittest discover -s tests                      # full (needs the cache)
```
