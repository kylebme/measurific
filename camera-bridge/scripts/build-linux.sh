#!/usr/bin/env bash
# Builds the bridge with a private Aravis and bundles every non-core shared
# library into dist/linux-<arch>/lib (the app sets LD_LIBRARY_PATH to it).
# Needs: cmake, pkg-config, ninja, meson (or uvx), libglib2.0-dev, libxml2-dev,
# libusb-1.0-0-dev, zlib1g-dev.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
bridge_root="$(cd "$script_dir/.." && pwd)"
arch="$(uname -m)"
[[ "$arch" == "x86_64" ]] && arch="x64"
[[ "$arch" == "aarch64" ]] && arch="arm64"
build_dir="$bridge_root/build/linux-$arch"
dist_dir="$bridge_root/dist/linux-$arch"
deps_dir="$bridge_root/build/deps-linux-$arch"
aravis_version="0.8.36"
aravis_sha256="246deaa0042a387ff1bc00332d0fb80537ce14abde2c28d1a54c91f17adc51bf"
source_archive="$bridge_root/build/aravis-$aravis_version.tar.xz"
source_dir="$bridge_root/build/aravis-$aravis_version"
aravis_build_dir="$bridge_root/build/aravis-build-linux-$arch"

run_meson() {
  if command -v meson >/dev/null; then meson "$@"
  elif command -v uvx >/dev/null; then uvx --from meson meson "$@"
  else echo "meson or uvx is required to build the bundled Aravis library" >&2; exit 1
  fi
}

if [[ ! -f "$deps_dir/lib/pkgconfig/aravis-0.8.pc" ]]; then
  mkdir -p "$bridge_root/build"
  [[ -f "$source_archive" ]] || curl --fail --location --output "$source_archive" \
    "https://github.com/AravisProject/aravis/releases/download/$aravis_version/aravis-$aravis_version.tar.xz"
  echo "$aravis_sha256  $source_archive" | sha256sum --check --status || { echo "Aravis source checksum mismatch" >&2; exit 1; }
  [[ -d "$source_dir" ]] || tar -xf "$source_archive" -C "$bridge_root/build"
  run_meson setup "$aravis_build_dir" "$source_dir" \
    --prefix "$deps_dir" --libdir lib --buildtype release \
    -Dviewer=disabled -Dgst-plugin=disabled -Dintrospection=disabled \
    -Ddocumentation=disabled -Dtests=false -Dusb=enabled
  run_meson compile -C "$aravis_build_dir"
  run_meson install -C "$aravis_build_dir"
fi

rm -rf "$dist_dir"
PKG_CONFIG_PATH="$deps_dir/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
  cmake -S "$bridge_root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$dist_dir"
cmake --build "$build_dir" --config Release --parallel
cmake --install "$build_dir" --config Release
mkdir -p "$dist_dir/lib"

# glibc and the compiler runtime must come from the host; everything else ships.
LD_LIBRARY_PATH="$deps_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$dist_dir/bin/inspection-camera-bridge" |
  awk '/=> \// {print $3}' |
  grep -Ev '/(ld-linux[^/]*|libc|libm|libdl|libpthread|librt|libresolv|libgcc_s|libstdc\+\+)\.so' |
  while read -r library; do cp -L "$library" "$dist_dir/lib/"; done

echo "Bundled camera bridge: $dist_dir"
