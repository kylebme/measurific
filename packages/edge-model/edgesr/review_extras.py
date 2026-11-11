"""Classify and export far-extra contour detections (predicted contour points > FAR px
from any GT edge) for human review.

Categories (from the raw render and GT coverage at the detection):
  specular       inside the part, render max within 2 px >= backlight level
  bright_object  inside the part, 0.5 x backlight <= render max < backlight
  dark_object    inside the part, render max < 0.5 x backlight
  background     outside the part (dust, noise, backlight texture, frame)
Blobs = connected groups of extra points (4 px grid, 8-connectivity).

Usage: python -m edgesr.review_extras CKPT [--images 20] [--out work/review/NAME]
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

from . import WORK_DIR
from . import bench_v3 as B3
from . import sampler as S

FAR = 2.0
CATS = ("specular", "bright_object", "dark_object", "background")


def classify(store, sc, pts):
    meta = store.scenes[sc]
    raw = np.asarray(store.images[sc], np.float32)
    cov = np.asarray(store.cover[meta["part_index"]])
    bl = meta["backlight_level"]
    mx = cv2.dilate(raw, np.ones((5, 5), np.uint8))          # max within 2 px
    ix = np.clip(np.floor(pts).astype(int), 0, 2047)
    inside = cov[ix[:, 1], ix[:, 0]] > 0
    rel = mx[ix[:, 1], ix[:, 0]] / bl
    cat = np.where(~inside, "background",
                   np.where(rel >= 1.0, "specular", np.where(rel >= 0.5, "bright_object", "dark_object")))
    return cat, rel


def blobs(pts, cell=4.0):
    if not len(pts):
        return np.zeros(0, int), 0
    g = np.floor(pts / cell).astype(int)
    g -= g.min(0)
    grid = np.zeros(g.max(0)[::-1] + 1, bool)
    grid[g[:, 1], g[:, 0]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
    return lab[g[:, 1], g[:, 0]], n


def render(img, gt, pred_far, pred_ok, cx, cy, half=24, zoom=8):
    x0, y0 = int(cx) - half, int(cy) - half
    H, W = img.shape
    x0, y0 = int(np.clip(x0, 0, W - 2 * half)), int(np.clip(y0, 0, H - 2 * half))
    c = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
    lo, hi = float(c.min()), float(c.max())
    z = np.kron((c - lo) / max(hi - lo, 1e-6), np.ones((zoom, zoom)))
    vis = (np.stack([z] * 3, -1) * 200 + 30).astype(np.uint8)
    for pts, col in ((gt, (0, 255, 0)), (pred_ok, (0, 255, 255)), (pred_far, (0, 0, 255))):
        q = (pts - [x0, y0]) * zoom
        q = q[np.all((q >= 0) & (q < 2 * half * zoom), 1)].astype(int)
        vis[q[:, 1], q[:, 0]] = col
    return vis, (x0, y0, lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--images", type=int, default=20)
    ap.add_argument("--out")
    ap.add_argument("--per-cat", type=int, default=40)
    ap.add_argument("--min-object-area", type=float, default=50.0)
    ap.add_argument("--min-hole-area", type=float, default=8.0)
    a = ap.parse_args()
    name = os.path.basename(os.path.dirname(a.ckpt))
    out = a.out or os.path.join(WORK_DIR, "review", f"{name}_extras")
    os.makedirs(out, exist_ok=True)
    store = S.SourceStore()
    spec = B3.load_spec()
    by_id = {s["scene_id"]: i for i, s in enumerate(store.scenes)}
    pred_fn = B3.cfi_predictor(a.ckpt, min_area=(a.min_object_area, a.min_hole_area))
    os.makedirs(os.path.join(out, "overview"), exist_ok=True)
    step = max(1, len(spec["images"]) // a.images)
    stats = {"points": {c: 0 for c in CATS}, "blobs": {c: 0 for c in CATS},
             "by_lighting": {}, "n_pred": 0, "n_far": 0}
    samples = {c: [] for c in CATS}
    rng = np.random.default_rng(0)
    for k in range(0, len(spec["images"]), step)[:a.images]:
        im = spec["images"][k]
        sc = by_id[im["scene_id"]]
        meta = store.scenes[sc]
        img, bl, info = B3.reading(store, sc, im["aug_seed"])
        pred, _ = pred_fn(img, bl)
        gt = store.boundary_points(meta["part_index"]).astype(np.float64)
        d = cKDTree(gt).query(pred, workers=8)[0]
        far = d > FAR
        fp = pred[far]
        stats["n_pred"] += len(pred)
        stats["n_far"] += int(far.sum())
        cat, rel = classify(store, sc, fp)
        lab, nb = blobs(fp)
        light = meta["illumination_class"]
        L = stats["by_lighting"].setdefault(light, {"n_pred": 0, **{c: 0 for c in CATS}})
        L["n_pred"] += len(pred)
        for c in CATS:
            stats["points"][c] += int((cat == c).sum())
            L[c] += int((cat == c).sum())
        for b in range(1, nb + 1):
            m = lab == b
            if m.sum() < 3:
                continue
            vals, cnt = np.unique(cat[m], return_counts=True)
            c = vals[np.argmax(cnt)]
            stats["blobs"][c] += 1
            samples[c].append((int(k), sc, fp[m].mean(0), int(m.sum())))
        # keep the image data only for crops we may render
        for c in CATS:
            for s_ in samples[c]:
                if s_[0] == k and len(s_) == 4:
                    pass
        # render immediately (avoid holding images): up to per-cat crops per image
        for c in CATS:
            mine = [s_ for s_ in samples[c] if s_[0] == k]
            for j, (_, _, (cx, cy), npts) in enumerate(mine[:max(1, a.per_cat // a.images + 1)]):
                vis, _ = render(img, gt, fp, pred[~far], cx, cy)
                cv2.putText(vis, f"{c} {meta['illumination_class']} {meta['material_family']} n={npts}",
                            (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 160, 0), 1)
                os.makedirs(os.path.join(out, c), exist_ok=True)
                cv2.imwrite(os.path.join(out, c, f"img{k:02d}_{j:02d}_x{int(cx)}_y{int(cy)}.png"), vis)
        # whole-image overview: GT green, far extras red (enlarged), 1024 px
        ov = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        ov = (np.stack([np.clip(ov / max(ov.max(), 1e-6), 0, 1)] * 3, -1) * 200 + 30).astype(np.uint8)
        for (px, py) in gt[::8] / 2:
            ov[int(min(py, 1023)), int(min(px, 1023))] = (0, 200, 0)
        for (px, py) in fp[::4] / 2:
            cv2.circle(ov, (int(px), int(py)), 2, (0, 0, 255), -1)
        cv2.putText(ov, f"img {k} {light} {meta['material_family']} far={int(far.sum())} of {len(pred)}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 160, 0), 2)
        cv2.imwrite(os.path.join(out, "overview", f"img{k:02d}_{light}_{meta['material_family']}.png"), ov)
        print(f"image {k} ({light}, {meta['material_family']}): {len(pred)} pts, far {int(far.sum())} "
              + " ".join(f"{c}={int((cat == c).sum())}" for c in CATS), flush=True)
    # contact sheets per category (up to 24 crops) and an overview sheet
    import glob
    for c in CATS + ("overview",):
        fs = sorted(glob.glob(os.path.join(out, c, "*.png")))[:24]
        if not fs:
            continue
        size = 340 if c == "overview" else 256
        tiles = [cv2.resize(cv2.imread(f), (size, size), interpolation=cv2.INTER_AREA) for f in fs]
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.concatenate([np.concatenate(tiles[i:i + 6], 1) for i in range(0, len(tiles), 6)], 0)
        cv2.imwrite(os.path.join(out, f"contact_{c}.png"), sheet)
    tot = max(stats["n_far"], 1)
    stats["point_share"] = {c: stats["points"][c] / tot for c in CATS}
    stats["filter"] = {"min_object_area": a.min_object_area, "min_hole_area": a.min_hole_area}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: stats[k] for k in ("n_pred", "n_far", "point_share", "blobs", "by_lighting")},
                     indent=1))


if __name__ == "__main__":
    main()
