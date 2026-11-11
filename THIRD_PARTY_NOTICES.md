# Third-party notices

The source in this repository is MIT licensed (see [LICENSE](LICENSE)), except
where noted below. Released installers also bundle third-party software, each
under its own license.

## Source included in this repository

| Component | License | Notes |
|---|---|---|
| [`packages/pae`](packages/pae) | MIT | Derived from [gravi-toni/subpixel-edges](https://github.com/gravi-toni/subpixel-edges), Copyright (c) 2020 Giuseppe Vitucci and Nicola Vitucci. See [packages/pae/LICENSE](packages/pae/LICENSE) and [packages/pae/NOTICE.md](packages/pae/NOTICE.md). |

## Bundled in release installers

### Camera bridge native libraries (LGPL)

The camera bridge (`resources/camera-bridge/lib`, or the DLLs next to the
executable on Windows) links dynamically to these libraries. They are shipped as
separate, replaceable shared libraries, unmodified from upstream:

| Library | License | Source |
|---|---|---|
| Aravis 0.8 | LGPL-2.1-or-later | https://github.com/AravisProject/aravis |
| GLib / GIO / GObject / GModule | LGPL-2.1-or-later | https://gitlab.gnome.org/GNOME/glib |
| libusb 1.0 | LGPL-2.1-or-later | https://github.com/libusb/libusb |
| GNU gettext (libintl) | LGPL-2.1-or-later | https://www.gnu.org/software/gettext/ |
| PCRE2 | BSD-3-Clause | https://github.com/PCRE2Project/pcre2 |
| libxml2 (Linux, Windows) | MIT | https://gitlab.gnome.org/GNOME/libxml2 |
| zlib (Linux, Windows) | Zlib | https://zlib.net |

The Linux and Windows builds also copy the bridge's other non-system
dependencies from the build machine (for example libffi, libiconv or the MinGW
runtime), each under its own license. The exact versions and how they are built
are in [camera-bridge/scripts](camera-bridge/scripts). You may replace any of these
libraries with your own build of a compatible version. The full text of the GNU
LGPL v2.1 is at https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.

### Python runtime

| Component | License |
|---|---|
| CPython ([python-build-standalone](https://github.com/astral-sh/python-build-standalone)) | PSF-2.0, plus its bundled libraries' licenses |
| NumPy, SciPy, Shapely, Numba, llvmlite | BSD-3-Clause / BSD-2-Clause (llvmlite also includes LLVM under Apache-2.0 with LLVM exceptions) |
| OpenCV (`opencv-python-headless`) | Apache-2.0 (bundles third-party libraries under their own licenses) |
| ONNX Runtime | MIT |
| Pillow | MIT-CMU |

The exact pinned set is in
[app/python/runtime/requirements.lock](app/python/runtime/requirements.lock).
Each installed package's license files are included in the bundled runtime's
`site-packages`.

### Electron app

| Component | License |
|---|---|
| Electron and Chromium | MIT, with Chromium's licenses in `LICENSES.chromium.html` inside the app |
| React, React DOM, Zustand, Radix UI, Floating UI | MIT |
| Lucide icons | ISC |
| Geist Sans and Geist Mono fonts | SIL Open Font License 1.1 |
