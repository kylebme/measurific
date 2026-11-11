"""Dataset validation and coverage accounting; runs without Blender."""

import json
import os
from collections import Counter

from PIL import Image


def validate_scene(path):
    with open(os.path.join(path, "metadata.json")) as stream:
        meta = json.load(stream)
    with open(os.path.join(path, meta["files"]["mask_index"])) as stream:
        index = json.load(stream)
    p = meta["projection"]
    w, h, scale = p["source_width"], p["source_height"], p["mask_scale"]
    assert meta["schema_version"] == 1 and p["type"] == "orthographic"
    assert scale == 16 and index["width"] == w * scale and index["height"] == h * scale
    image_path = os.path.join(path, meta["files"]["image"])
    with Image.open(image_path) as image:
        with open(image_path, "rb") as stream:
            header = stream.read(26)
        assert header[:8] == b"\x89PNG\r\n\x1a\n" and header[24:26] == b"\x10\x00"
        assert image.size == (w, h) and image.mode in ("I", "I;16", "I;16B"), (image.size, image.mode)
        lo, hi = image.getextrema()
        assert 0 <= lo <= hi < 65535, (lo, hi)
    clipped = meta["render_result"]["source"].get("response_clipped_pixels", 0)
    assert clipped <= max(4, int(w * h * 0.001)), f"too many response-clipped pixels: {clipped}"
    expected = {(x, y): (min(index["tile_size"], w * scale - x),
                         min(index["tile_size"], h * scale - y))
                for y in range(0, h * scale, index["tile_size"])
                for x in range(0, w * scale, index["tile_size"])}
    total = 0
    margin = p["margin_px"] * scale
    min_x, min_y = w * scale, h * scale
    max_x = max_y = -1
    for tile in index["tiles"]:
        xy = (tile["x"], tile["y"])
        assert xy in expected and expected.pop(xy) == (tile["width"], tile["height"])
        with Image.open(os.path.join(path, meta["files"]["mask_dir"], tile["file"])) as image:
            assert image.mode == "1" and image.size == (tile["width"], tile["height"])
            assert image.getextrema() in ((0, 0), (0, 255), (255, 255))
            count = image.histogram()[255]
            assert count == tile["foreground_pixels"], (xy, count, tile["foreground_pixels"])
            total += count
            if count:
                box = image.getbbox()
                min_x = min(min_x, tile["x"] + box[0])
                min_y = min(min_y, tile["y"] + box[1])
                max_x = max(max_x, tile["x"] + box[2] - 1)
                max_y = max(max_y, tile["y"] + box[3] - 1)
    assert not expected, f"missing {len(expected)} tiles"
    assert total == index["foreground_pixels"] and total > 0
    assert min_x >= margin and min_y >= margin and max_x < w * scale - margin and max_y < h * scale - margin, (min_x, min_y, max_x, max_y, margin)
    return {"scene_id": meta["scene_id"], "image_range": (lo, hi), "mask_pixels": total,
            "mask_bounds": (min_x, min_y, max_x, max_y)}


def coverage(manifests):
    result = {key: Counter() for key in ("family", "scale", "density", "material", "focus", "lighting", "outline", "thread", "feature")}
    for m in manifests:
        g = m["geometry"]
        result["family"][g["family"]] += 1
        result["scale"][g["scale_class"]] += 1
        result["density"][g["density_class"]] += 1
        result["material"][m["material"]["family"]] += 1
        result["focus"][m["camera"]["focus_class"]] += 1
        result["lighting"][m["illumination"]["class"]] += 1
        result["outline"][g["outline"]] += 1
        if "thread_profile" in g:
            result["thread"][g["thread_profile"]] += 1
        for feature in g.get("features", []):
            result["feature"][feature["kind"]] += 1
    return {key: dict(sorted(value.items())) for key, value in result.items()}


def require_complete_coverage(manifests):
    """Gate for a 60-scene-or-larger representative production batch."""
    report = coverage(manifests)
    required = {
        "family": ("plate", "turned", "machined", "sphere", "freeform"),
        "scale": ("subpixel", "near_pixel", "fine", "patch", "large", "very_large"),
        "density": ("sparse", "medium", "dense"),
        "material": ("dark_matte", "rough_metal", "polished_metal", "coated", "polymer", "light_matte"),
        "focus": ("sharp", "moderate", "defocused", "strong_defocus", "depth_split"),
        "lighting": ("silhouette", "balanced", "reflective", "ambient"),
        "thread": ("none", "v", "round", "trapezoid", "square"),
        "feature": ("round", "ellipse", "polygon", "filleted_polygon", "freeform", "tooth", "slot", "curved_slot"),
    }
    missing = {axis: [value for value in values if report[axis].get(value, 0) == 0]
               for axis, values in required.items()}
    missing = {axis: values for axis, values in missing.items() if values}
    if missing:
        raise ValueError(f"coverage gate failed: {missing}")
    return report
