"""Export macro scenes for instance-segmentation training (RF-DETR COCO and YOLO-seg).

    ../../packages/edge-model/.venv/bin/python -m macro.export DATASET [--out DATASET/export] [--classes part|group]

Layout (one copy of each image serves both trainers):
    export/images/{train,valid,test}/<scene>.png|.jpg
    export/images/{split}/_annotations.coco.json   RF-DETR: dataset_dir=export/images
    export/labels/{train,valid,test}/<scene>.txt  YOLO-seg: data.yaml (labels mirror images)
    export/data.yaml, export/stats.json

Images: the linear 16-bit render becomes an 8-bit camera reading with a per-scene
seeded exposure (backlight at 55-92% of full scale), Poisson-Gaussian sensor noise
(6k-60k e- full well, read noise 1e-4-2e-3 of full scale) and, for 10% of scenes, JPEG
(quality 75-95). The clean 16-bit render stays in the dataset for other pipelines.

Masks: each part's visible (modal) pixels from instances.png. Polygons follow boundary
pixel centres; holes are bridged into the outer ring with a zero-width cut because
neither COCO polygons nor YOLO-seg lines can carry holes. The export re-rasterizes every
polygon (even-odd; cv2.fillPoly as used by Ultralytics agrees) and reports its IoU
against the instance mask: 0.9998 mean on the pilot.
"""

import argparse
import glob
import hashlib
import json
import os

import cv2
import numpy as np

SPLITS = {"train": "train", "val": "valid", "test": "test"}
GROUP_CLASSES = ("face_simple", "face_expanded", "turned_simple", "turned_expanded", "plain",
                 "machined", "sphere", "freeform")


def camera_image(raw, background, rng):
    """16-bit linear render -> 8-bit sensor reading."""
    x = raw.astype(np.float64) / 65535.0
    level = float(np.median(x[background])) if background.any() else float(np.percentile(x, 95))
    x = x * (rng.uniform(0.55, 0.92) / max(level, 1e-6)) + rng.uniform(0, 0.02)
    full_well = np.exp(rng.uniform(np.log(6000), np.log(60000)))
    read = np.exp(rng.uniform(np.log(1e-4), np.log(2e-3)))
    x = rng.poisson(np.clip(x, 0, None) * full_well) / full_well + rng.normal(0, read, x.shape)
    return np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)


def _bridge(outer, hole):
    """Join a hole into an outer ring along the shortest connection (zero-width cut)."""
    d = ((outer[:, None, :] - hole[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return np.concatenate([outer[:i + 1], hole[j:], hole[:j + 1], outer[i:]])


def mask_polygons(mask):
    """List of polygons (N, 2) float, one per connected component, holes bridged in."""
    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    polys = []
    for k, c in enumerate(contours):
        if hier[0][k][3] != -1 or len(c) < 3:     # holes are handled with their parent
            continue
        poly = c[:, 0, :].astype(np.float64)
        child = hier[0][k][2]
        while child != -1:
            hole = contours[child][:, 0, :].astype(np.float64)
            if len(hole) >= 3:
                poly = _bridge(poly, hole)
            child = hier[0][child][0]
        if len(poly) >= 3:
            polys.append(poly)
    return polys


def merge_components(polys):
    """YOLO-seg allows one polygon per instance: bridge components together."""
    out = polys[0]
    for p in polys[1:]:
        out = _bridge(out, p)
    return out


def rasterize_evenodd(poly, shape):
    """Even-odd raster of one bridged polygon (holes empty), matching how trainers fill."""
    from PIL import Image, ImageDraw
    img = Image.new("1", (shape[1], shape[0]), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in poly], fill=1, outline=1)
    return np.array(img, np.uint8)


def export(dataset, out, classes="part"):
    scenes = sorted(glob.glob(os.path.join(dataset, "scenes", "macro_*", "metadata.json")))
    cats = ([{"id": 1, "name": "part", "supercategory": "part"}] if classes == "part" else
            [{"id": i + 1, "name": g, "supercategory": "part"} for i, g in enumerate(GROUP_CLASSES)])
    coco = {s: {"images": [], "annotations": [], "categories": cats,
                "info": {"description": "synthetic-macro macro_v1", "classes": classes}}
            for s in SPLITS.values()}
    stats = {"scenes": 0, "instances": 0, "iou": [], "jpeg": 0, "per_split": {}}
    for s in SPLITS.values():
        os.makedirs(os.path.join(out, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", s), exist_ok=True)
    ann_id = 1
    for path in scenes:
        scene = os.path.dirname(path)
        meta = json.load(open(path))
        split = SPLITS[meta["split"]]
        name = meta["scene_id"]
        seed = int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "big")
        rng = np.random.default_rng(seed)
        raw = cv2.imread(os.path.join(scene, "image.png"), cv2.IMREAD_UNCHANGED)
        inst = cv2.imread(os.path.join(scene, "instances.png"), cv2.IMREAD_UNCHANGED)
        debris = cv2.imread(os.path.join(scene, "debris.png"), cv2.IMREAD_UNCHANGED)
        img = camera_image(raw, (inst == 0) & (debris < 10), rng)
        if rng.random() < 0.10:
            fname = name + ".jpg"
            cv2.imwrite(os.path.join(out, "images", split, fname), img,
                        [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(75, 96))])
            stats["jpeg"] += 1
        else:
            fname = name + ".png"
            cv2.imwrite(os.path.join(out, "images", split, fname), img)
        h, w = img.shape
        image_id = len(coco[split]["images"]) + 1
        coco[split]["images"].append({"id": image_id, "file_name": fname, "width": w, "height": h})
        lines = []
        for part in meta["parts"]:
            m = inst == part["instance_id"]
            if not m.any():
                continue
            polys = mask_polygons(m)
            if not polys:
                continue
            cat = 1 if classes == "part" else GROUP_CLASSES.index(part["group"]) + 1
            ys, xs = np.nonzero(m)
            coco[split]["annotations"].append({
                "id": ann_id, "image_id": image_id, "category_id": cat, "iscrowd": 0,
                "area": float(m.sum()),
                "bbox": [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)],
                "segmentation": [p.ravel().round(2).tolist() for p in polys],
                "attributes": {"family": part["family"], "group": part["group"], "kind": part["kind"],
                               "tilted": part["pose"]["tilt_rad"] > 0, "stacked": part["pose"]["stack_on"] is not None,
                               "cut_by_frame": part["cut_by_frame"]}})
            ann_id += 1
            merged = merge_components(polys)
            lines.append(f"{cat - 1} " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in merged))
            back = rasterize_evenodd(merged, m.shape).astype(bool)
            stats["iou"].append(float((back & m).sum() / max((back | m).sum(), 1)))
            stats["instances"] += 1
        with open(os.path.join(out, "labels", split, name + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        stats["scenes"] += 1
        stats["per_split"][split] = stats["per_split"].get(split, 0) + 1
    for s, c in coco.items():
        with open(os.path.join(out, "images", s, "_annotations.coco.json"), "w") as f:
            json.dump(c, f)
    names = [c["name"] for c in cats]
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/valid\ntest: images/test\n"
                f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    iou = np.array(stats.pop("iou"))
    stats["polygon_iou"] = {"mean": float(iou.mean()) if len(iou) else None,
                            "p01": float(np.percentile(iou, 1)) if len(iou) else None}
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset")
    ap.add_argument("--out")
    ap.add_argument("--classes", choices=("part", "group"), default="part")
    args = ap.parse_args()
    print(json.dumps(export(args.dataset, args.out or os.path.join(args.dataset, "export"), args.classes), indent=2))


if __name__ == "__main__":
    main()
