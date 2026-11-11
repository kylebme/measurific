# Inspection camera bridge

`inspection-camera-bridge` is a small native Aravis 0.8 client kept outside the
GUI. Electron starts it as a child process and is the only code allowed to talk
to it. Commands travel as tab-separated, URI-escaped lines on stdin. Responses
and frames use the versioned `ICAM` protocol documented in
[`PROTOCOL.md`](PROTOCOL.md).

The bridge discovers any transport Aravis exposes. It has explicit paths for
USB3 Vision and GigE Vision, and does not contain Basler- or Allied-specific SDK code. In
`camera` output it asks for the highest supported unpacked monochrome format. A
color-only camera falls back to RGB/BGR or Bayer and is converted to grayscale
before any bytes cross the process boundary.

On a color camera the app instead selects one raw Bayer color site (red by
default) at half resolution; see `set channel` in [PROTOCOL.md](PROTOCOL.md).
A camera's own grayscale on a color sensor is demosaiced, which leaves zipper
artifacts at edges. It is often also color-processed: a Basler ace 2 applies its
sRGB curve, which moves edges by more than half a pixel. A single site avoids
both, and it avoids mixing colors that lateral chromatic aberration offsets from
each other. The only camera-specific names the bridge touches are Basler's
optional `BslLightSourcePreset` and `BslColorSpace`, which it skips when absent.

Live previews are full-frame, aspect-preserving grayscale frames limited to
2048 pixels on the long edge and 20 fps. This avoids wasting UI/IPC bandwidth.
The bridge retains the corresponding native-resolution frame. `pause` converts
and sends that complete frame, retaining 16-bit storage for sensors above 8
bits. The GUI writes it as a lossless grayscale PNG for measurement.

## Build on Linux and Windows

`scripts/build-linux.sh` builds the same private Aravis and copies every
non-glibc shared library into `dist/linux-<arch>/lib`. Install `cmake
pkg-config ninja-build libglib2.0-dev libxml2-dev libusb-1.0-0-dev
zlib1g-dev` first. `scripts/build-windows.sh` runs in an MSYS2 UCRT64 shell
against MSYS2's `aravis` package, and copies the DLLs it loads next to
`dist/win32-x64/bin/inspection-camera-bridge.exe`. USB3 Vision cameras on
Windows still need the WinUSB driver bound once per machine (an administrator
step), and on Linux they need a udev rule.

## Build for development on macOS

The build needs CMake, pkg-config, libusb, and GLib development files. Aravis
itself is downloaded from its official release, checksum-verified, and built
privately under `camera-bridge/build`; it is not installed into the OS. Meson
may be installed already or supplied ephemerally by `uvx`. None of these are
required on a machine running the packaged app.

```bash
brew install cmake pkg-config glib libusb
./scripts/build-macos.sh
```

The script installs the executable under `dist/macos-<arch>` and copies every
non-system dylib it uses into the same self-contained directory. Electron
Builder includes that directory as an app resource.

With or without a camera attached, discovery and protocol startup can be checked
with:

```bash
node tests/smoke.cjs dist/macos-arm64/bin/inspection-camera-bridge
node tests/smoke.cjs dist/macos-arm64/bin/inspection-camera-bridge --fake
node tests/meter.cjs dist/macos-arm64/bin/inspection-camera-bridge
node tests/smoke.cjs dist/macos-arm64/bin/inspection-camera-bridge --camera
```

The `--camera` form opens the first discovered physical camera, receives a
preview, pauses on a full-resolution frame, and then closes the device.

Aravis itself is portable to Linux and Windows and the bridge avoids macOS-only
APIs. The current release packaging script is intentionally macOS-only. Linux
will need the equivalent ELF dependency collection/rpath pass; Windows will
need an Aravis/GLib runtime DLL collection pass plus signed USB driver guidance.
