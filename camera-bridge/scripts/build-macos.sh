#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
bridge_root="$(cd "$script_dir/.." && pwd)"
machine_arch="$(uname -m)"
arch="$machine_arch"
[[ "$machine_arch" == "x86_64" ]] && arch="x64"
build_dir="$bridge_root/build/macos-$arch"
dist_dir="$bridge_root/dist/macos-$arch"
deps_dir="$bridge_root/build/deps-macos-$arch"
aravis_version="0.8.36"
aravis_sha256="246deaa0042a387ff1bc00332d0fb80537ce14abde2c28d1a54c91f17adc51bf"
source_archive="$bridge_root/build/aravis-$aravis_version.tar.xz"
source_dir="$bridge_root/build/aravis-$aravis_version"
aravis_build_dir="$bridge_root/build/aravis-build-macos-$arch"
export CCACHE_DIR="$bridge_root/build/ccache"
export CCACHE_TEMPDIR="$bridge_root/build/ccache-tmp"
mkdir -p "$CCACHE_DIR" "$CCACHE_TEMPDIR"

run_meson() {
  if command -v meson >/dev/null; then
    meson "$@"
  elif command -v uvx >/dev/null; then
    UV_CACHE_DIR="$bridge_root/build/uv-cache" \
      UV_TOOL_DIR="$bridge_root/build/uv-tools" \
      UV_TOOL_BIN_DIR="$bridge_root/build/uv-bin" \
      uvx --from meson meson "$@"
  else
    echo "meson or uvx is required to build the bundled Aravis library" >&2
    exit 1
  fi
}

if [[ ! -f "$deps_dir/lib/pkgconfig/aravis-0.8.pc" ]]; then
  mkdir -p "$bridge_root/build"
  if [[ ! -f "$source_archive" ]]; then
    curl --fail --location --output "$source_archive" \
      "https://github.com/AravisProject/aravis/releases/download/$aravis_version/aravis-$aravis_version.tar.xz"
  fi
  actual_sha256="$(shasum -a 256 "$source_archive" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$aravis_sha256" ]]; then
    echo "Aravis source checksum mismatch" >&2
    exit 1
  fi
  if [[ ! -d "$source_dir" ]]; then
    tar -xf "$source_archive" -C "$bridge_root/build"
  fi
  run_meson setup "$aravis_build_dir" "$source_dir" \
    --prefix "$deps_dir" --buildtype release \
    -Dviewer=disabled -Dgst-plugin=disabled -Dintrospection=disabled \
    -Ddocumentation=disabled -Dtests=false -Dusb=enabled
  run_meson compile -C "$aravis_build_dir"
  run_meson install -C "$aravis_build_dir"
fi

PKG_CONFIG_PATH="$deps_dir/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
  cmake -S "$bridge_root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$dist_dir"
cmake --build "$build_dir" --config Release --parallel
cmake --install "$build_dir" --config Release

mkdir -p "$dist_dir/lib"

copy_dependencies() {
  local target="$1"
  local dependency
  local name
  while IFS= read -r dependency; do
    case "$dependency" in
      /System/*|/usr/lib/*|@loader_path/*) continue ;;
      @rpath/*)
        echo "Unresolved non-system @rpath dependency in $target: $dependency" >&2
        exit 1
        ;;
    esac
    name="$(basename "$dependency")"
    if [[ ! -f "$dist_dir/lib/$name" ]]; then
      cp -L "$dependency" "$dist_dir/lib/$name"
      chmod u+w "$dist_dir/lib/$name"
      copy_dependencies "$dist_dir/lib/$name"
    fi
    install_name_tool -change "$dependency" "@loader_path/../lib/$name" "$target"
  done < <(otool -L "$target" | tail -n +2 | awk '{print $1}')
}

copy_dependencies "$dist_dir/bin/inspection-camera-bridge"
for library in "$dist_dir"/lib/*.dylib; do
  [[ -e "$library" ]] || continue
  install_name_tool -id "@loader_path/$(basename "$library")" "$library"
  while IFS= read -r dependency; do
    name="$(basename "$dependency")"
    [[ -f "$dist_dir/lib/$name" ]] && install_name_tool -change "$dependency" "@loader_path/$name" "$library"
  done < <(otool -L "$library" | tail -n +2 | awk '{print $1}')
done

codesign --force --sign - "$dist_dir/bin/inspection-camera-bridge"
for library in "$dist_dir"/lib/*.dylib; do [[ -e "$library" ]] && codesign --force --sign - "$library"; done

echo "Bundled camera bridge: $dist_dir"
