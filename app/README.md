# Measurific desktop app

Electron + Vite + React desktop prototype for the inspection pipeline. The app
opens maximized (ordinary resizable window, not macOS full-screen), loads a fresh
synthetic test image, runs the learned contour recognizer automatically, and keeps
all geometric measurement in the Python `inspection-measurement` package.

## Run

```bash
cd app
npm install
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python \
  -e '../packages/edge-model[onnx]' -e '../packages/pae' -e '../packages/measurement[test,image]' \
  -e '../packages/contour-model[test,bridge]' -e '../research/synthetic-shapes[test]'
npm run demo-data
npm run dev
```

Use `npm start` after `npm run build`. Set `INSPECTION_PYTHON` to select another
compatible interpreter instead of `app/.venv/bin/python`. The development venv
above still installs torch for training work; the app itself only needs
`onnxruntime` (installed by `contour-model[onnx]`).

Recognition runs `packages/contour-model/release-v6` (v5 fine-tuned on clean contours plus
edge-model-extracted renders; see `packages/contour-model/RESEARCH.md`) through ONNX Runtime,
from the exported graphs in its `onnx/` folder. Post-processing merges adjacent
same-class pieces that one line or circle fits (`merge_tol=1`).
The app always uses this bundled model. The **Settings** dialog (gear, top
right) selects CPU or GPU inference; changes apply to the next analyzed image.

GPU inference uses CoreML on macOS and DirectML on Windows; Linux builds run on
the CPU. CPU is the default. For the current contour model, CoreML only covers
part of the graph and is slower than the CPU.

Settings → Analysis → **Edges** selects how contours are extracted; changing it
restarts the worker and applies to the next analyzed image:

- **PAE** (default): the coarse edge model (`coarse_v2-onnx`) finds the rings and
  PAE ([`pae`](../packages/pae/README.md)) locates each edge to subpixel precision near them
  ([details](../packages/measurement/README.md#image-extraction-coarse-model--pae)). Much
  faster (about 0.2 s per demo image on CPU) and good for sharp edges. Rings PAE
  cannot support, e.g. strongly blurred edges, fail with a located diagnostic.
  The first start on a computer compiles PAE's Numba kernels (a few seconds; cached).
- **Deep Learning**: the SDF v5 edge model
  [`packages/edge-model/models/sdf_v5-onnx`](../packages/edge-model/README.md), exported from
  `sdf_v5.pt`. ONNX Runtime runs both its SDF network and the required
  `coarse_v2-onnx` model. The bilinear zero level is traced into closed rings on a
  1/16 px lattice. Works best for blurry edges and very fine detail.

Both retain holes, nested objects, image-border closure, circular camera ROIs and
native 8/16-bit grayscale input. Settings → Runtime shows the active edge model
and provider.

The edge model uses CUDA or DirectML when available and CPU otherwise (including
macOS; the contour recognizer can still use CoreML). SDF v5 is a dense network:
large images can take substantially longer on CPU than the previous CFI model.
After retraining it, re-export and check parity against torch (this exporter
uses the local `val_v4_full` cache for its parity check):

```bash
cd packages/edge-model && .venv/bin/python -m edgesr.sdf_export models/sdf_v5.pt sdf_v5
```

After retraining, export the checkpoint and check it against the torch model
before switching the app to it:

```bash
PYTHONPATH=packages/contour-model/src .venv/bin/python -m contour_model.onnx_export \
  packages/contour-model/<release>/model.pt packages/contour-model/<release>/onnx \
  --parity app/demo-data/*.contours.npz
```

`INSPECTION_DEVICE` (`cpu`/`gpu`) and `INSPECTION_EDGE_METHOD`
(`pae`/`deep-learning`) override the saved settings and show as locked in the dialog. In development runs only, `INSPECTION_CHECKPOINT` points
the worker at another exported ONNX folder (or a `.pt` checkpoint) before it is
bundled; packaged apps ignore it. `INSPECTION_EDGE_MODEL` similarly overrides
the development edge model with an ONNX directory or the `sdf_v5.pt` checkpoint.
The checkpoint requires torch plus `coarse_v2.pt` beside it; ONNX is the default
and packaged apps remain torch-free.

```bash
cd app
INSPECTION_EDGE_MODEL=../packages/edge-model/models/sdf_v5.pt npm run dev
```

## Settings and calibrations

State is stored as plain JSON by `electron-store` ([state.cjs](electron/state.cjs)):

| File | Scope | Location |
|---|---|---|
| `settings.json`, `preferences.json` | this user | Electron `userData`: `~/Library/Application Support/Measurific`, `%APPDATA%\Measurific`, `~/.config/Measurific` |
| `calibrations.json` | this computer | `/Users/Shared/Measurific`, `%ProgramData%\Measurific`, `/var/lib/measurific` |

Calibrations are stored under `calibration.<id>`, where the id is the save time
as an ISO timestamp with `:` and `.` replaced by `-`. Each record also carries
`savedAt`, `savedBy` and `appVersion`. The machine folder is created without
administrator rights on macOS and Windows and made writable for every account.
If this account cannot write it, calibrations are saved in the user folder,
and the machine folder is still read. On Linux, an administrator enables shared
calibrations once:

```bash
sudo install -d -m 0777 /var/lib/measurific
```

A damaged settings file is set aside and replaced with defaults. A damaged
calibration file is never reset: it is reported in Settings and saving is
disabled until it is fixed. **Export…/Import…** move settings and all
calibrations as one JSON file. Importing never overwrites an existing
calibration.

Development runs use a separate `Measurific Dev` folder.
`INSPECTION_DATA_DIR` and `INSPECTION_MACHINE_DATA_DIR` redirect the two
locations.

## Packaging and releases

Installers are per-user and need no administrator rights: a DMG on macOS
(arm64), a one-click NSIS installer on Windows (x64, installs to
`%LOCALAPPDATA%\Programs`) and an AppImage on Linux (x64). The macOS zip
exists only for future auto-updates and is not meant for users. Uninstalling
keeps settings and calibrations.

The app bundles a relocatable CPython (python-build-standalone via `uv`) with
the locked packages in [python/runtime](python/runtime) and the pipeline
packages installed without their dependencies, so torch is never shipped.
Also bundled: the ONNX recognition, SDF edge and coarse models, the worker, the demo
images and the camera bridge.
All of these live under the app's `resources` folder, outside `app.asar`.

```bash
npm run python:lock     # after editing python/runtime/requirements.in
npm run dist:mac        # or dist:win (MSYS2 needed for the bridge) / dist:linux
node scripts/smoke-worker.mjs --resources "release/mac-arm64/Measurific.app/Contents/Resources"
```

[desktop.yml](../.github/workflows/desktop.yml) runs the tests and builds, and
smoke-tests all three platforms on pull requests and manual runs. It uploads
the installers as workflow artifacts. Pushing a tag `vX.Y.Z` that matches
`package.json` also attaches them to a draft GitHub release.

Builds are not yet signed with a certificate. macOS builds are ad-hoc signed
with hardened runtime ([entitlements](build-resources/entitlements.mac.plist)),
so the downloaded app opens after **Open Anyway** in System Settings → Privacy &
Security. Windows shows SmartScreen's **More info → Run anyway**. To sign later,
replace `"identity": "-"` with a Developer ID and add notarization.

## Interaction

- Recognized compounds are outlined on the image. Hold Ctrl to inspect their
  constituent primitives. Dashed outlines indicate features that need review.
- Hover a colored contour for feature identity. Only fitted features can enter
  measurements; scores are labeled uncalibrated.
- Click in the pointer tool to pin/unpin the feature card. Escape clears every
  pinned selection and any in-progress relationship selection; with nothing
  selected, Escape returns to the pointer tool.
- Corners where segments meet are selectable vertices (hover to reveal). When
  corner exclusion trims the segments at a corner, each segment keeps its own
  vertex at its retained endpoint, so two vertices may sit close together;
  zoom in to pick a specific one.
- Tools that need one selection apply on click: datum, circularity, straightness,
  size tools, thread/slot/tooth profile measurements, circle redefinition, and compound removal. Tools with
  several selections wait for Enter or Apply.
- Automatic measurements remain available to the analysis pipeline but are not
  displayed in the GUI.
- Choose distance or angle in the top toolbar, click two features, configure
  the measurement definition, and add the previewed result. Distance offers only
  the constructions that fit the selection, with a one-sentence description of
  exactly what is measured:
  - Point to line: perpendicular from a line to a center, corner, or the near/far
    side of a circle or arc; optionally to the line's fitted extension.
  - Point to point: between centers, corners, or circle sides, as a straight-line
    distance or a component (ΔX, ΔY, an angle, or parallel/perpendicular to a datum).
  - Gap along edges / at one position: smallest, largest, and average gap over the
    supported area, or at one station.
  - Shortest distance: closest points on the actual edges.
  Saved distances can be edited without reselecting features. Angle accepts fitted
  line features only.
- Concentricity, perpendicularity, parallelism, circularity, and straightness
  results create compact drawing-style annotations outside the part. Their leader
  lines terminate on each measured feature, and the eye controls on the annotation
  and result card hide or restore the overlay. A measurement type can only be
  added once for the same feature set.
- Choose **Thread measurements** (≋) in the top toolbar, then click a recognized
  triangular or trapezoidal/square thread compound. This immediately adds minor
  diameter, major diameter, root radius, crest radius, pitch, and visible thread
  count, with min/max/average in one boxed callout and matching extrema dimensions.
  Equal extrema share a min/max label. The callout and all dimensions hide/remove
  together, and the full table remains in Results. Radii are shown only for arcs
  at roots or crests; otherwise they show “—”.
  Thread count is resolved crests per side; dimensions use the original selected
  corners, independent of primitive corner exclusion. See the measurement module's
  semantic thread definitions for selection and resolution limits.
- **Size** tools apply on click: **Ø** diameter of a circle, **R** radius of an arc
  or circle, **↔** length of a line (corner to corner, unaffected by corner
  exclusion), and **⌒** arc angle. Diameter, radius and length are drawn as
  dimensions; arc angle as a callout.
- **Slot measurements** and **Tooth profile measurements** (beside Thread
  measurements) apply on click to a full round slot or tooth profile compound and
  add a min/max/average table with matching extrema dimensions, like threads.
  Slots report width between flats, overall length, end radii and end center
  spacing. Tooth profiles report tooth count, pitch, tooth height, and for round
  profiles (gears, splines) tip and root diameters; pitch is in degrees for round
  profiles and stays in the table. Tip and root radii are shown only for rounding
  arcs at tips or roots. See the measurement module's
  [slot and tooth profile definitions](../packages/measurement/README.md#slot-and-tooth-profile-measurements).
- The Redefine toolbar section also defines compounds. For a triangular or
  trapezoidal thread, click where the thread starts and ends along one side, then
  along the opposite side; it is applied on the fourth click. For a full round slot
  or tooth profile, click the constituent segments (shown individually while the
  tool is active), then press Enter or Apply. Short unselected segments between
  selected ones are included, and compounds sharing a member are replaced.
- **Remove compound** (the ungroup button at the end of the Redefine section)
  undoes a compound definition: hover a thread, slot, or tooth profile until it
  is highlighted, then click it to ungroup it back into its primitives. Only
  compounds respond while this tool is active. Measurements of that compound
  are removed; its primitives are unchanged.
- Redefining contour geometry removes measurements that reference the edited
  feature or an invalidated compound while preserving unrelated results.
- Distance extrema use aligned dimension lines, perpendicular extensions, and
  exact witnesses; short spans use outward arrows and an offset value. Angle arcs
  use the real fitted-line intersection, with feature-linked callouts for distant
  or parallel lines. Distances, angles, and GD&T share collision-aware placement
  and visibility controls. A notice reports annotations that cannot fit the view.
  See [annotation placement](docs/annotation-placement.md) for the Draftwright
  reference, geometry rules, limitations, and visual regression fixtures.
- Open image supports ordinary grayscale/color files. Synthetic demo images use
  their saved exact generated contour observations; arbitrary files run the
  learned sub-pixel edge model first.
- On a color camera, the camera panel's **Image** setting chooses red, green or
  blue sites only (one color site of the sensor at half resolution, no demosaicing
  or color processing; red is the default) or the camera's own grayscale image. The
  choice is saved per user (`INSPECTION_CAMERA_CHANNEL` overrides it), applies
  immediately, and automatic exposure meters the chosen channel. Camera snapshots
  record `sampleStep` and `sampleOffsetX/Y` so image pixels map back to the sensor.
  A pixel-size calibration must be made in the same mode it is used with.
- Several objects in one image (parts, a large piece of dust, a part inside another
  part's hole) are each recognized on their own, because the recognizer takes one
  outline with its holes. The results are merged into one report: each feature's
  `parent` names its object (`object_1`, largest first), and measurements can relate
  features on different objects.

## Architecture

The sandboxed renderer talks only to a narrow preload API. Electron owns a
persistent JSON-lines Python worker, started with the saved compute and edge
settings and restarted when they change. That worker extracts contours (PAE or the edge model), runs the
learned recognizer through ONNX Runtime, and passes all primitive and relationship requests to
`inspection_measurement.evaluate`. The renderer only draws returned support,
measurement witnesses, and report values.

`npm run demo-data` regenerates four 1600×900 renderings from `synthetic-shapes`:
drilled plate, stepped shaft, grooved pin, and slotted bracket. The files and a
small provenance manifest are written to `app/demo-data`.

## Validation

```bash
# Repository root, using an environment with contour-model and measurement dependencies:
OMP_NUM_THREADS=4 packages/contour-model/.venv/bin/python -m pytest -q app/python
cd app
npm test
npm run build
```

The Python tests load the exported release model, analyze all four saved demos, and
exercise compound/constituent relationships and failed-image state reset. The
[measurement validation](../packages/measurement/benchmarks/PRIMITIVES_VALIDATION.md)
records the independent analytic ground-truth benchmark.

## Camera exposure

Automatic exposure is controlled by the app using full-resolution statistics
inside the selected rectangle/circle (the full frame if no ROI is selected).
It targets the 99.9th percentile near 90% of sensor range, caps the peak with
headroom, and reduces exposure whenever any metered pixel clips. Increases are
damped and a deadband prevents hunting. Vendor auto exposure is disabled, as is
auto gain when enabling this controller. Manual gain is preserved. Pause stops
adjustments before taking the measurement snapshot; Live resumes metering.
Changing ROI discards in-flight statistics for the old ROI. The rebuilt native
bridge and a restarted Electron process are required for this version.

## Live focus assistance

The live camera panel displays **Sharpness**, a relative score (higher is sharper),
a bar relative to the best score seen, and **Reset peak**. Turn focus slowly to
maximize it. Select a rectangle/circle ROI containing the edge of interest; keep
the part still and reset the reference after moving or replacing it. The peak
also resets when resuming live view or changing ROI, channel, exposure, or gain.
Blank/noisy/tiny regions show guidance instead of a misleading score; missing
frames make the reading stale. The indicator disappears when the image is paused
or a file is opened.

Measurement uses smoothed, noise-corrected, variance-normalized Tenengrad on
native delivered pixels, before preview resizing or 8-bit conversion, at up to
five updates/sec. It does not run the learned measurement pipeline. The score is
not an absolute optical-resolution or measurement-quality threshold. See the
[research, comparisons, formula, and hardware validation limits](docs/live-focus.md).
