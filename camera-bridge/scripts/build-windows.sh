#!/usr/bin/env bash
# Builds the bridge in an MSYS2 UCRT64 shell against MSYS2's Aravis package and
# copies every DLL it loads from the MSYS2 prefix next to the executable.
#   pacman -S --needed mingw-w64-ucrt-x86_64-{toolchain,cmake,ninja,pkgconf,aravis}
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
bridge_root="$(cd "$script_dir/.." && pwd)"
build_dir="$bridge_root/build/win32-x64"
dist_dir="$bridge_root/dist/win32-x64"
prefix="${MINGW_PREFIX:?run from an MSYS2 UCRT64 shell}"

rm -rf "$dist_dir"
cmake -S "$bridge_root" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$dist_dir"
cmake --build "$build_dir" --config Release --parallel
cmake --install "$build_dir" --config Release

# Windows searches the executable's folder first, so DLLs live beside it.
ldd "$dist_dir/bin/inspection-camera-bridge.exe" |
  awk -v prefix="$prefix/bin/" 'index($3, prefix) == 1 {print $3}' |
  while read -r library; do cp "$library" "$dist_dir/bin/"; done
# GIO loads its modules at runtime; the bridge does not need any.
echo "Bundled camera bridge: $dist_dir"
