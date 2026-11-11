"""Compare lossless, region-readable storage using an actual rendered scene."""

import argparse
import json
import os
import statistics
import tempfile
import time

from PIL import Image


def _decode_seconds(path, repetitions=5):
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        with Image.open(path) as image:
            image.load()
        times.append(time.perf_counter() - start)
    return round(statistics.median(times), 6)


def benchmark(scene_dir):
    with open(os.path.join(scene_dir, "metadata.json")) as stream:
        meta = json.load(stream)
    with open(os.path.join(scene_dir, meta["files"]["mask_index"])) as stream:
        index = json.load(stream)
    image_path = os.path.join(scene_dir, meta["files"]["image"])
    candidates = [t for t in index["tiles"] if 0 < t["foreground_pixels"] < t["width"] * t["height"]]
    if not candidates:
        raise ValueError("no mixed-occupancy mask tile to benchmark")
    tile = min(candidates, key=lambda t: abs(t["foreground_pixels"] / (t["width"] * t["height"]) - 0.5))
    mask_path = os.path.join(scene_dir, meta["files"]["mask_dir"], tile["file"])
    report = {"scene_id": meta["scene_id"], "mask_tile": tile["file"], "variants": {}}
    with tempfile.TemporaryDirectory() as directory:
        for kind, original, variants in (
            ("image", image_path, (("png_16", None), ("tiff_deflate_16", "tiff_adobe_deflate"),
                                     ("tiff_lzw_16", "tiff_lzw"))),
            ("mask_tile", mask_path, (("png_1", "PNG"), ("tiff_group4_1", "group4"),
                                       ("tiff_deflate_1", "tiff_adobe_deflate")))):
            with Image.open(original) as source:
                source.load()
                for name, compression in variants:
                    if compression is None:
                        path = original
                    elif compression == "PNG":
                        path = os.path.join(directory, name + ".png")
                        source.save(path, format="PNG")
                    else:
                        path = os.path.join(directory, name + ".tif")
                        source.save(path, format="TIFF", compression=compression)
                        with Image.open(path) as decoded:
                            assert list(source.getdata()) == list(decoded.getdata()), name
                    report["variants"][name] = {"bytes": os.path.getsize(path),
                                                 "median_decode_seconds": _decode_seconds(path)}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir")
    args = parser.parse_args()
    print(json.dumps(benchmark(args.scene_dir), indent=2))
