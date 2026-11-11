"""Contact sheet for physical_v3 scenes: whole image plus zoomed crops with the GT outline.

Needs numpy and opencv (e.g. the edge-model venv):
    ../../packages/edge-model/.venv/bin/python scripts/review_v3.py OUTPUT_DIR [--count N] [--out sheet.jpg]

Each row is one scene: a downsampled whole image (auto-contrast to the backlight),
then up to five 64 px crops at 8x zoom around placed contaminants or random edges.
Red is the ground-truth mask outline from the 16x mask. The label gives the lighting
class, blur class, material, tilt, contamination level and backlight type.
"""

import argparse
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synth.reader import SceneReader  # noqa: E402

CROP, ZOOM, THUMB = 64, 6, 384


def _coverage_1x(reader, shape):
    """Source-resolution object coverage, read tile by tile (a full 16x mask is ~1 GB)."""
    from PIL import Image
    cov = np.zeros(shape, np.float32)
    mdir = os.path.join(reader.scene_dir, reader.metadata["files"]["mask_dir"])
    for t in reader.mask_index["tiles"]:
        x, y, w, h = t["x"] // 16, t["y"] // 16, t["width"] // 16, t["height"] // 16
        if t["foreground_pixels"] in (0, t["width"] * t["height"]):
            cov[y:y + h, x:x + w] = float(t["foreground_pixels"] > 0)
            continue
        with Image.open(os.path.join(mdir, t["file"])) as im:
            a = np.array(im.convert("L"), np.float32) / 255
        cov[y:y + h, x:x + w] = a.reshape(h, 16, w, 16).mean((1, 3))
    return cov


def _norm(img, ref):
    return np.clip(img.astype(np.float32) / max(ref, 1) * 0.9, 0, 1)


def _crop_panel(reader, img, ref, cx, cy):
    h, w = img.shape
    x0 = int(np.clip(cx - CROP // 2, 0, w - CROP))
    y0 = int(np.clip(cy - CROP // 2, 0, h - CROP))
    crop = _norm(img[y0:y0 + CROP, x0:x0 + CROP], ref)
    big = cv2.resize(crop, (CROP * ZOOM, CROP * ZOOM), interpolation=cv2.INTER_NEAREST)
    big = cv2.cvtColor((big * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    m = np.array(reader.mask_region((x0 * 16, y0 * 16, (x0 + CROP) * 16, (y0 + CROP) * 16)), np.float32)
    m = cv2.resize(m, (CROP * ZOOM, CROP * ZOOM), interpolation=cv2.INTER_AREA)
    edge = cv2.morphologyEx((m > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((2, 2), np.uint8)) > 0
    big[edge] = (0, 0, 255)
    return big


def _label(panel, text, y=16):
    cv2.putText(panel, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(panel, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)


def scene_row(scene_dir, rng):
    reader = SceneReader(scene_dir)
    m = reader.metadata
    img = cv2.imread(os.path.join(scene_dir, "image.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    ref = float(np.percentile(img, 97))
    thumb = cv2.resize(_norm(img, ref), (THUMB, THUMB), interpolation=cv2.INTER_AREA)
    thumb = cv2.cvtColor((thumb * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    thumb = cv2.copyMakeBorder(thumb, 0, CROP * ZOOM - THUMB, 0, 0, cv2.BORDER_CONSTANT)
    o, c = m["optics"], m["contamination"]
    _label(thumb, f"{m['scene_index']} {m['geometry']['family']} {m['illumination']['class']} "
                  f"{o['blur_class']} {m['material']['family']}")
    _label(thumb, f"tilt {np.degrees(m['pose'].get('realized_tilt_rad', 0)):.1f}deg  "
                  f"dirt {c['level']}  bl {m['backlight']['angular']['type']} "
                  f"{m['backlight']['angular']['half_angle_deg']:.0f}deg", 34)
    _label(thumb, f"NA {o['na']:.3f} px {o['pixel_size_um']:.1f}um coc {o['target_coc_px']:.1f} "
                  f"psf {np.hypot(o['lens_psf']['diffraction_sigma_px'], o['lens_psf']['aberration_sigma_px']):.2f} "
                  f"glare {o['glare']['fraction']:.3f}", 52)
    _label(thumb, f"filter {m['render']['pixel_filter']}  mask {m['mask_plan']}", 70)
    centers = []
    for r in c.get("realized", []):
        if r.get("placed"):
            item = c["items"][r["index"]]
            pt = r.get("center_px") or r.get("base_px") or r.get("start_px")
            centers.append((pt, item["kind"]))
    # Fill with random GT edge points from the 1x coverage of the mask.
    sil = _coverage_1x(reader, img.shape) > 0.5 if len(centers) < 5 else None
    if sil is not None:
        edge = sil ^ cv2.erode(sil.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ys, xs = np.nonzero(edge)
        for i in rng.sample(range(len(xs)), min(5 - len(centers), len(xs))):
            centers.append(((xs[i], ys[i]), "edge"))
    rng.shuffle(centers)
    panels = [thumb]
    for (x, y), kind in centers[:5]:
        p = _crop_panel(reader, img, ref, x, y)
        _label(p, kind)
        panels.append(p)
    while len(panels) < 6:
        panels.append(np.zeros_like(panels[1] if len(panels) > 1 else _crop_panel(reader, img, ref, 100, 100)))
    return np.hstack([cv2.copyMakeBorder(p, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for p in panels])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="review_v3.jpg")
    args = ap.parse_args()
    scenes = sorted(glob.glob(os.path.join(args.output, "scene_*", "metadata.json")))
    scenes = [os.path.dirname(s) for s in scenes][args.start:args.start + args.count]
    rng = random.Random(args.seed)
    rows = [scene_row(s, rng) for s in scenes]
    cv2.imwrite(args.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(args.out, len(rows), "scenes")


if __name__ == "__main__":
    main()
