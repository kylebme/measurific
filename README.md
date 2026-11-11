# Measurific

Sub-pixel dimensional inspection for backlit, telecentric machine-vision images.

Measurific is a desktop app for macOS, Windows and Linux. It takes a
silhouette image from a file or a live camera, extracts sub-pixel contours,
recognizes the geometry (lines, arcs, circles, slots, threads and tooth
profiles), and measures it: distances, diameters, angles, radii, pitch, thread
and profile parameters, with tolerance checks and an annotated report.

Everything runs locally on the CPU (GPU optional). The app ships with its own
Python runtime and ONNX models, so it needs no separate install.

## Download

Installers for each version are on the
[**Releases**](https://github.com/kylebme/measurific/releases/latest) page:

| Platform | File | Notes |
|---|---|---|
| macOS (Apple silicon) | `Measurific-<version>-mac-arm64.dmg` | Drag to Applications. |
| Windows (x64) | `Measurific-<version>-win-x64.exe` | Per-user install, no admin rights needed. |
| Linux (x64) | `Measurific-<version>-linux-x86_64.AppImage` | `chmod +x`, then run. |

Builds are not yet signed with a paid certificate:

- **macOS**: the first launch is blocked. Open **System Settings → Privacy &
  Security** and click **Open Anyway**.
- **Windows**: SmartScreen shows a warning. Click **More info → Run anyway**.

The app checks GitHub Releases for updates. Uninstalling keeps your settings and
calibrations.

### Cameras

The bundled camera bridge uses [Aravis](https://github.com/AravisProject/aravis)
and works with USB3 Vision and GigE Vision cameras without vendor SDKs. See
[camera-bridge/README.md](camera-bridge/README.md).

## Repository layout

| Path | What it is |
|---|---|
| [`app/`](app/README.md) | Electron + React desktop app and its Python worker. Packaging and release configuration. |
| [`camera-bridge/`](camera-bridge/README.md) | Native Aravis client (C) that the app runs as a child process. |
| [`packages/measurement/`](packages/measurement/README.md) | `inspection-measurement`: CPU-only geometric measurement from contours and a JSON request. |
| [`packages/edge-model/`](packages/edge-model/README.md) | `inspection-edge-model` (`edgesr`): learned sub-pixel edge and coarse segmentation models. |
| [`packages/pae/`](packages/pae/README.md) | `inspection-pae`: Partial Area Effect sub-pixel edge detector (MIT fork of `subpixel-edges`). |
| [`packages/contour-model/`](packages/contour-model/README.md) | `inspection-contour-model`: learned contour-primitive recognizer, training code, and released checkpoints. |
| [`research/`](research/README.md) | Synthetic data generators and renderers used for training and validation. |
| [`docs/`](docs/) | Release process. |

How the pieces fit:

```
image ─▶ edge extraction ─────────▶ contour recognition ─▶ measurement ─▶ report
         (coarse model + PAE,        (contour-model,        (measurement
          or SDF edge model)          ONNX Runtime)          package)
```

The app bundles `packages/contour-model/release-v6/onnx`,
`packages/edge-model/models/sdf_v5-onnx` and
`packages/edge-model/models/coarse_v2-onnx`.

## Development

Requirements: Node.js 22, [uv](https://docs.astral.sh/uv/) and Python 3.13.
Building the camera bridge also needs CMake, Ninja and pkg-config (see
[camera-bridge/README.md](camera-bridge/README.md)).

```bash
cd app
npm install
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python \
  -e '../packages/edge-model[onnx]' -e ../packages/pae -e '../packages/measurement[test,image]' \
  -e '../packages/contour-model[test,bridge]' -e '../research/synthetic-shapes[test]'
npm run dev
```

The development environment installs PyTorch for the training code. The
packaged app uses ONNX Runtime only.

Run the tests:

```bash
cd app && npm test && npx tsc -b
cd app && .venv/bin/python -m pytest -q python
cd packages/measurement && ../../app/.venv/bin/python -m pytest -q
```

Each package's own tests run the same way from its folder. See
[CONTRIBUTING.md](CONTRIBUTING.md) for details and
[docs/RELEASING.md](docs/RELEASING.md) for building installers and publishing
a release.

## License

[MIT](LICENSE) © Kyle Bartholomew. `packages/pae` is derived from
[gravi-toni/subpixel-edges](https://github.com/gravi-toni/subpixel-edges) (MIT).
Released installers bundle third-party components under their own licenses,
including LGPL libraries in the camera bridge. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
