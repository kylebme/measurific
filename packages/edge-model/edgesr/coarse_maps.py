"""Precomputed coarse-model maps for fine-model training data (physical_v3 or the older
full-catalog renders: 2048^2 scenes with 16x tiled masks).

  python -m edgesr.coarse_maps build [--model models/coarse_v2.pt] [--source DIR] [--name NAME]
  python -m edgesr.coarse_maps score [--name NAME] [--split val]

Per scene, the 16-bit render passes a nominal camera (backlight median at 0.75 of full
scale, 20k e- full well, 5e-4 read noise, 12-bit; as synthetic-renders
scripts/score_edge_model.py), is area-downsampled 4x and run through the coarse model.
Stored under work/cache/coarse_maps/NAME/ (T7):
  part_logit.npy  (N, 512, 512) float16   raw coarse part logit (> 0 = part; apply
                                          coarse.suppress_weak for the model's final mask)
  sdf.npy         (N, 512, 512) float16   signed distance to the coarse boundary, macro px
  gt_cover.npy    (N, 512, 512) uint8     GT part coverage of each macro pixel (0-255), from
                                          the 16x mask tiles (64 x 64 hi-res px each)
  index.json      scenes: scene_id, dir, part_index, split (edgesr.cache.part_split),
                  mask_key, lighting / blur / tilt / contamination classes
Macro pixel j covers source pixels [4j, 4j + 4); see coarse_infer for upsampling.

Policy difference to keep in mind for the fine model: physical_v3 labels every contaminant
(loose items near the part too) as object, while the coarse model was trained to leave
loose debris out of parts. `score` reports the boundary recall of the coarse band on
this data, so the fine model's band can be widened (or unioned with the classical
gradient band) where needed.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image

from . import CACHE_DIR, REPO_DIR

SOURCES = {"physical_v3": "/media/kyle/T7 Shield/inspection-synthetic-renders/physical-v3-30gb-2048",
           "full_catalog": "/media/kyle/T7 Shield/inspection-synthetic-renders/full-catalog-5gb-2048"}
OUT_DIR = os.path.join(CACHE_DIR, "coarse_maps")
HI = 64                      # hi-res (16x) pixels per macro pixel edge


def scene_dirs(root):
    names = sorted(n for n in os.listdir(root) if n.startswith("scene_"))
    return [os.path.join(root, n) for n in names if os.path.exists(os.path.join(root, n, "metadata.json"))]


def macro_cover(scene_dir):
    """GT part coverage per macro pixel from the 16x mask tiles (float32 in [0, 1])."""
    index = json.load(open(os.path.join(scene_dir, "mask", "index.json")))
    tiles = index["tiles"]
    W = max(t["x"] + t["width"] for t in tiles)
    H = max(t["y"] + t["height"] for t in tiles)
    cover = np.zeros((H // HI, W // HI), np.float32)
    for t in tiles:
        fg = t["foreground_pixels"]
        if fg == 0:
            continue
        y0, x0, h, w = t["y"] // HI, t["x"] // HI, t["height"] // HI, t["width"] // HI
        if fg == t["width"] * t["height"]:
            cover[y0:y0 + h, x0:x0 + w] = 1.0
            continue
        with Image.open(os.path.join(scene_dir, "mask", t["file"])) as im:
            m = np.array(im.convert("L")) > 127
        cover[y0:y0 + h, x0:x0 + w] = m.reshape(h, HI, w, HI).mean((1, 3))
    return cover


def nominal_reading(img16, seed):
    rng = np.random.default_rng(seed)
    x = img16.astype(np.float64) / 65535
    x = x * (0.75 / max(np.percentile(x[::4, ::4], 90), 1e-6))
    x = rng.poisson(np.clip(x, 0, None) * 20000) / 20000 + rng.normal(0, 5e-4, x.shape)
    return (np.round(np.clip(x, 0, 1) * 4095) / 4095).astype(np.float32)


def _load(args):
    k, d = args
    from .coarse import model_reading
    meta = json.load(open(os.path.join(d, "metadata.json")))
    img = cv2.imread(os.path.join(d, "image.png"), cv2.IMREAD_UNCHANGED)
    r = model_reading(nominal_reading(img, 1000 + k))
    cov = macro_cover(d)
    return k, r, cov, meta


def _record(d, meta, split):
    g = lambda *ks: _get(meta, ks)                                      # noqa: E731
    return {"scene_id": meta["scene_id"], "dir": d, "part_index": int(meta["catalog"]["part_index"]),
            "split": split.get(int(meta["catalog"]["part_index"])), "mask_key": meta.get("mask_key"),
            "lighting": g("illumination", "class"), "blur": g("optics", "blur_class") or g("camera", "focus_class"),
            "tilted": bool((g("pose", "realized_tilt_rad") or 0) > 0),
            "contamination": g("contamination", "level"), "family": g("geometry", "family")}


def _get(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def build(model_path, source, name, workers=6, device="cuda", limit=None, split=None):
    from .cache import part_split
    from .coarse_infer import CoarseModel
    dirs = scene_dirs(source)
    if split:
        from .cache import part_split
        ps = part_split()
        dirs = [d for d in dirs
                if ps.get(int(json.load(open(os.path.join(d, "metadata.json")))["catalog"]["part_index"])) == split]
    dirs = dirs[:limit]
    n = len(dirs)
    out = os.path.join(OUT_DIR, name)
    os.makedirs(out, exist_ok=True)
    cm = CoarseModel(model_path, device)
    lg = np.lib.format.open_memmap(os.path.join(out, "part_logit.npy"), "w+", np.float16, (n, 512, 512))
    sd = np.lib.format.open_memmap(os.path.join(out, "sdf.npy"), "w+", np.float16, (n, 512, 512))
    gc = np.lib.format.open_memmap(os.path.join(out, "gt_cover.npy"), "w+", np.uint8, (n, 512, 512))
    split = part_split()
    recs = [None] * n
    with ProcessPoolExecutor(workers) as ex:
        for k, r, cov, meta in ex.map(_load, enumerate(dirs), chunksize=4):
            o = cm.maps(r)
            lg[k] = np.clip(o["part_logit"], -60000, 60000)
            sd[k] = o["sdf"]
            gc[k] = np.round(cov * 255).astype(np.uint8)
            recs[k] = _record(dirs[k], meta, split)
            if k % 500 == 0:
                print(f"{k}/{n}", flush=True)
    for a in (lg, sd, gc):
        a.flush()
    with open(os.path.join(out, "index.json"), "w") as f:
        json.dump({"source": source, "model": os.path.relpath(model_path, REPO_DIR), "scenes": recs,
                   "camera": "backlight 0.75 FS, 20k e-, read 5e-4, 12-bit; seed 1000 + index"}, f)
    print(f"wrote {out} ({n} scenes)")


def score(name, split="val", n=None):
    """Coarse model vs macro GT on the fine-model dataset (bench_coarse metrics; one
    instance per scene: the part with its contaminants)."""
    from . import bench_coarse as BC
    from .coarse import suppress_weak
    out = os.path.join(OUT_DIR, name)
    idx = json.load(open(os.path.join(out, "index.json")))["scenes"]
    lg = np.load(os.path.join(out, "part_logit.npy"), mmap_mode="r")
    gc = np.load(os.path.join(out, "gt_cover.npy"), mmap_mode="r")
    sel = [k for k, r in enumerate(idx) if split in (None, "all") or r["split"] == split][:n]
    recs, groups, thin = [], {}, []
    for k in sel:
        cover = np.asarray(gc[k])
        inst = (cover >= 128).astype(np.uint8)
        pred = {"part_logit": suppress_weak(np.asarray(lg[k], np.float32))}
        pred["instances"] = (pred["part_logit"] > 0).astype(np.int32)
        r = BC.score_image(pred, inst, np.zeros_like(inst), {"parts": [{"instance_id": 1}]})
        recs.append(r)
        for key in ("lighting", "blur", "family", "contamination", "tilted"):
            groups.setdefault(key, {}).setdefault(str(idx[k][key]), []).append(r)
        # Thin scenes: the majority GT holds < half of the >= 25%-covered pixels (a part
        # thinner than ~1 model px). Recall: share of those pixels within 1 px of the
        # predicted part.
        q = cover >= 64
        if q.sum() and inst.sum() < 0.5 * q.sum():
            near = cv2.dilate((pred["part_logit"] > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            thin.append(float((near & q).sum() / q.sum()))
    res = BC.summarize(recs)
    res["slices"] = {g: {v: BC.summarize(rs) for v, rs in sorted(d.items())} for g, d in groups.items()}
    res["thin_scenes"] = len(thin)
    res["thin_recall_mean"] = float(np.mean(thin)) if thin else None
    res["thin_detected"] = float(np.mean([t >= 0.5 for t in thin])) if thin else None
    keys = ("bscore", "recall@1.0", "recall@2.0", "precision@1.0", "iou", "part_miss", "false_parts_per_img")
    print(f"{name} {split}: " + " ".join(f"{k}={res[k]:.4f}" for k in keys))
    if thin:
        print(f"  thin parts: {len(thin)} scenes, mean recall of >=25%-covered pixels "
              f"{res['thin_recall_mean']:.3f}, detected (>= 50%) {res['thin_detected']:.3f}")
    for g, d in res["slices"].items():
        for v, s in d.items():
            print(f"  {g}={v:<14s} n={s['images']:<4d} " + " ".join(f"{k}={s[k]:.4f}" for k in keys[:3]))
    with open(os.path.join(out, f"score_{split}.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score"])
    ap.add_argument("--model", default=os.path.join(REPO_DIR, "models", "coarse_v2.pt"))
    ap.add_argument("--source", default="physical_v3")
    ap.add_argument("--name")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--split", help="build: only this split (default all); score: default val")
    ap.add_argument("--n", type=int)
    a = ap.parse_args()
    name = a.name or a.source
    if a.cmd == "build":
        build(a.model, SOURCES.get(a.source, a.source), name, a.workers, limit=a.limit,
              split=None if a.split in (None, "all") else a.split)
    else:
        score(name, a.split or "val", a.n)


if __name__ == "__main__":
    main()
