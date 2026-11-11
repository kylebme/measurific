"""Read-optimized cache of the physical_v3 renders (fine model v4, DATA_V3.md).

Same file layout as cache.py (so SourceStore(cache_dir=CACHE_V4) reads it), with one
difference: masks are per unique `metadata["mask_key"]` (the part's base mask, or a
scene's own mask when it is tilted or contaminated), not per catalog part. The "part"
axis of cover.u16 / block_index.i32 / bpoints / candidates is the mask index, and each
scene record carries `mask_index` (sampler.mask_of) next to `part_index` (the split
stays by catalog part, cache.part_split).

Under CACHE_V4 (T7):
  scenes.json      meta + per-scene records: split, classes (blur, lighting, tilt,
                   contamination, material, backlight, glare, pixel filter), levels,
                   backlight_level (median over mask background), coarse map index and
                   the coarse-background field (seed.fit_field_masked on the raw render)
  images.u16       (n_scenes, 2048, 2048) raw renders
  cover.u16        (n_masks, 2048, 2048) 16x mask coverage per source pixel (0..256)
  block_index.i32, bits.u8   block-sparse 16x masks (64 x 64 blocks)
  bpoints.npy, bpoints_offsets.npy   marching-squares GT boundary crossings per mask
  candidates.npz   crop-center candidates per mask and category; categories as cache.py
                   plus "contaminant" (boundary pixels where a contaminated mask differs
                   from its part's base mask)

Usage: python -m edgesr.cache_v4 build [--workers 10]
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image

from . import SCALE, WORK_DIR
from . import cache as C

SOURCE_V4 = os.environ.get(
    "EDGESR_SOURCE_V4", "/media/kyle/T7 Shield/inspection-synthetic-renders/physical-v3-30gb-2048")
CACHE_V4 = os.path.join(WORK_DIR, "cache_v4")
COARSE_MAPS = os.path.join(WORK_DIR, "cache", "coarse_maps", "physical_v3")
H = W = 2048
BLOCK = C.BLOCK
NB = H * SCALE // BLOCK
CATEGORIES = C.CATEGORIES + ("contaminant",)
OUT = CACHE_V4          # set by build(); worker processes inherit it (fork)


def _get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _scene_record(path):
    with open(os.path.join(path, "metadata.json")) as f:
        m = json.load(f)
    assert m["projection"]["mask_scale"] == SCALE
    assert m["projection"]["source_width"] == W and m["projection"]["source_height"] == H
    blur = _get(m, "optics", "blur_class")
    return {
        "scene_id": m["scene_id"], "dir": path,
        "part_index": int(m["catalog"]["part_index"]), "part_id": m["catalog"]["part_id"],
        "mask_key": m["mask_key"], "mask_plan": m.get("mask_plan"),
        "family": m["geometry"]["family"],
        "blur_class": blur, "focus_class": blur,
        "camera_focus_class": _get(m, "camera", "focus_class"),
        "illumination_class": m["illumination"]["class"],
        "material_family": m["material"]["family"],
        "contamination": _get(m, "contamination", "level", default="clean"),
        "n_contaminants": len(_get(m, "contamination", "items", default=[]) or []),
        "tilted": bool((_get(m, "pose", "realized_tilt_rad", default=0) or 0) > 0),
        "backlight_type": _get(m, "backlight", "angular", "type"),
        "glare": float(_get(m, "optics", "glare", "fraction", default=0.0) or 0.0),
        "pixel_filter": _get(m, "render", "pixel_filter"),
    }


def _load_image(args):
    i, path, images_path, n = args
    a = cv2.imread(os.path.join(path, "image.png"), cv2.IMREAD_UNCHANGED)
    assert a.dtype == np.uint16 and a.shape == (H, W), (path, a.dtype, a.shape)
    mm = np.memmap(images_path, np.uint16, "r+", shape=(n, H, W))
    mm[i] = a
    mm.flush()
    lo, hi = np.percentile(a[::4, ::4], [1.0, 99.5])
    return i, float(lo), float(hi)


def _load_mask(args):
    """Decode one mask's tiles: block index, packed mixed blocks, coverage."""
    m, scene_path, cover_path, n_masks = args
    with open(os.path.join(scene_path, "mask", "index.json")) as f:
        index = json.load(f)
    assert index["width"] == W * SCALE and index["height"] == H * SCALE
    block_index = np.full((NB, NB), -1, np.int32)
    bits = []
    cover = np.memmap(cover_path, np.uint16, "r+", shape=(n_masks, H, W))
    cov = np.zeros((H, W), np.uint16)
    for tile in index["tiles"]:
        tx, ty, tw, th = tile["x"], tile["y"], tile["width"], tile["height"]
        sx, sy = tx // SCALE, ty // SCALE
        fg = tile["foreground_pixels"]
        bx0, by0 = tx // BLOCK, ty // BLOCK
        if fg == 0:
            continue
        if fg == tw * th:
            cov[sy:sy + th // SCALE, sx:sx + tw // SCALE] = SCALE * SCALE
            block_index[by0:by0 + th // BLOCK, bx0:bx0 + tw // BLOCK] = -2
            continue
        with Image.open(os.path.join(scene_path, "mask", tile["file"])) as im:
            t = np.array(im.convert("1"), dtype=bool)
        assert t.shape == (th, tw)
        cov[sy:sy + th // SCALE, sx:sx + tw // SCALE] = \
            t.reshape(th // SCALE, SCALE, tw // SCALE, SCALE).sum((1, 3), dtype=np.uint16)
        blocks = t.reshape(th // BLOCK, BLOCK, tw // BLOCK, BLOCK).transpose(0, 2, 1, 3)
        s = blocks.sum((2, 3))
        bi = block_index[by0:by0 + th // BLOCK, bx0:bx0 + tw // BLOCK]
        bi[s == BLOCK * BLOCK] = -2
        ys, xs = np.nonzero((s > 0) & (s < BLOCK * BLOCK))
        if len(ys):
            bi[ys, xs] = -1000
            bits.append((by0 + ys, bx0 + xs, np.packbits(blocks[ys, xs], axis=-1)))
    cover[m] = cov
    cover.flush()
    total = int(cov.sum(dtype=np.int64))
    assert total == index["foreground_pixels"], (m, scene_path, total, index["foreground_pixels"])
    if bits:
        ys = np.concatenate([b[0] for b in bits])
        xs = np.concatenate([b[1] for b in bits])
        packed = np.concatenate([b[2] for b in bits])
    else:
        ys = xs = np.zeros(0, np.int64)
        packed = np.zeros((0, BLOCK, BLOCK // 8), np.uint8)
    return m, block_index, ys, xs, packed


def _mask_boundary_points(m):
    """As cache._part_boundary_points, for mask m of the v4 cache."""
    from .metrics import contour_points
    from .sampler import SourceStore
    store = SourceStore(OUT)
    bi = np.asarray(store._open()["block_index"][m])
    T = 2048
    nbt = T // BLOCK
    pts = []
    n = H * SCALE
    for ty in range(0, n, T):
        for tx in range(0, n, T):
            by, bx = ty // BLOCK, tx // BLOCK
            sub = bi[by:by + nbt + 1, bx:bx + nbt + 1]
            if not ((sub >= 0).any() or ((sub == -1).any() and (sub == -2).any())):
                continue
            w = store.mask_window(m, tx, ty, min(T + 1, n - tx), min(T + 1, n - ty))
            p = contour_points(w.astype(np.float32))
            if len(p):
                pts.append(p + np.array([tx, ty]) / SCALE)
    if not pts:
        return m, np.zeros((0, 2), np.float32)
    q = np.unique(np.round(np.concatenate(pts) * 64).astype(np.int64), axis=0)
    return m, (q / 64.0).astype(np.float32)


def _categorize(args):
    """cache._categorize for mask m, plus the "contaminant" category."""
    m, base_m, cover_path, n_masks, seed = args
    covers = np.memmap(cover_path, np.uint16, "r", shape=(n_masks, H, W))
    cover = np.asarray(covers[m])
    cov = cover.astype(np.float32) / (SCALE * SCALE)
    boundary = ((cover > 0) & (cover < SCALE * SCALE)).astype(np.uint8)
    win = cv2.boxFilter(boundary.astype(np.float32), -1, (48, 48), normalize=False,
                        borderType=cv2.BORDER_CONSTANT)
    fg = (cover >= SCALE * SCALE // 2).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thin_fg = fg & (1 - cv2.morphologyEx(fg, cv2.MORPH_OPEN, k))
    thin_bg = (1 - fg) & cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    near_b = cv2.dilate(boundary, np.ones((3, 3), np.uint8)).astype(bool)
    narrow = (thin_fg | thin_bg).astype(bool) & near_b
    harris = cv2.cornerHarris(cov, 3, 3, 0.04)
    corner = (harris > 0.02) & near_b
    near_obj = cv2.dilate(boundary, np.ones((129, 129), np.uint8)).astype(bool)
    valid = np.zeros((H, W), bool)
    valid[C.CENTER_MARGIN:H - C.CENTER_MARGIN, C.CENTER_MARGIN:W - C.CENTER_MARGIN] = True
    masks = {"edge": boundary.astype(bool), "sparse": (win > 0) & (win <= 60) & boundary.astype(bool),
             "dense": (win >= 100) & boundary.astype(bool), "narrow": narrow, "corner": corner,
             "uniform": (win == 0) & near_obj}
    if base_m is not None and base_m != m:
        diff = np.asarray(covers[base_m]) != cover
        masks["contaminant"] = boundary.astype(bool) & cv2.dilate(
            diff.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    else:
        masks["contaminant"] = np.zeros((H, W), bool)
    rng = np.random.default_rng(seed + m)
    out = {}
    for name in CATEGORIES:
        ys, xs = np.nonzero(masks[name] & valid)
        if len(ys) > C.MAX_CANDIDATES:
            sel = rng.choice(len(ys), C.MAX_CANDIDATES, replace=False)
            ys, xs = ys[sel], xs[sel]
        out[name] = np.stack([xs, ys], 1).astype(np.int16)
    return m, out


def _levels(args):
    """backlight_level (median image over mask-background pixels) and the
    coarse-background field of the raw render (8-bit DN of raw / 65535)."""
    i, m, n, n_masks, k_coarse = args
    from .coarse import suppress_weak
    from .seed import coarse_background, fit_field_masked
    img = np.asarray(np.memmap(os.path.join(OUT, "images.u16"), np.uint16, "r", shape=(n, H, W))[i])
    cov = np.asarray(np.memmap(os.path.join(OUT, "cover.u16"), np.uint16, "r",
                               shape=(n_masks, H, W))[m])
    bl = float(np.median(img[cov == 0]))
    lg = np.load(os.path.join(COARSE_MAPS, "part_logit.npy"), mmap_mode="r")[k_coarse]
    lg = suppress_weak(np.asarray(lg, np.float32))
    coef, lvl = fit_field_masked(img.astype(np.float32) * (255.0 / 65535.0), coarse_background(lg, (H, W)))
    return i, bl, [float(c) for c in coef], float(lvl)


def build(workers=10, source=SOURCE_V4, out=CACHE_V4, limit=None):
    global OUT
    OUT = out
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    names = sorted(n for n in os.listdir(source) if n.startswith("scene_"))
    dirs = [os.path.join(source, n) for n in names if os.path.exists(os.path.join(source, n, "metadata.json"))]
    dirs = dirs[:limit]
    with ProcessPoolExecutor(workers) as ex:
        scenes = list(ex.map(_scene_record, dirs, chunksize=16))
    split = C.part_split()
    keys = sorted({s["mask_key"] for s in scenes})
    mask_of = {k: i for i, k in enumerate(keys)}
    base_key = {}          # part_index -> its base mask key (the part id)
    for s in scenes:
        s["split"] = split[s["part_index"]]
        s["mask_index"] = mask_of[s["mask_key"]]
        if s["mask_key"] == s["part_id"]:
            base_key[s["part_index"]] = s["mask_key"]
    coarse_idx = json.load(open(os.path.join(COARSE_MAPS, "index.json")))["scenes"]
    k_of = {r["scene_id"]: k for k, r in enumerate(coarse_idx)}
    for s in scenes:
        s["coarse_index"] = k_of[s["scene_id"]]
    n, n_masks = len(scenes), len(keys)
    print(f"{n} scenes, {len({s['part_index'] for s in scenes})} parts, {n_masks} masks", flush=True)

    images_path = os.path.join(out, "images.u16")
    np.memmap(images_path, np.uint16, "w+", shape=(n, H, W)).flush()
    with ProcessPoolExecutor(workers) as ex:
        for done, (i, lo, hi) in enumerate(ex.map(
                _load_image, [(i, s["dir"], images_path, n) for i, s in enumerate(scenes)], chunksize=4)):
            scenes[i]["object_level"], scenes[i]["background_level"] = lo, hi
            if done % 500 == 0:
                print(f"images {done}/{n} {time.time() - t0:.0f}s", flush=True)

    rep = {}
    for s in scenes:                         # a scene that holds each mask
        rep.setdefault(s["mask_key"], s["dir"])
    cover_path = os.path.join(out, "cover.u16")
    np.memmap(cover_path, np.uint16, "w+", shape=(n_masks, H, W)).flush()
    block_index = np.memmap(os.path.join(out, "block_index.i32"), np.int32, "w+", shape=(n_masks, NB, NB))
    packed_all, offset = [], 0
    with ProcessPoolExecutor(workers) as ex:
        jobs = [(m, rep[k], cover_path, n_masks) for k, m in mask_of.items()]
        for done, (m, bi, ys, xs, packed) in enumerate(ex.map(_load_mask, jobs, chunksize=2)):
            bi[ys, xs] = offset + np.arange(len(ys), dtype=np.int32)
            assert not (bi == -1000).any()
            block_index[m] = bi
            packed_all.append(packed)
            offset += len(ys)
            if done % 250 == 0:
                print(f"masks {done}/{n_masks} mixed_blocks={offset} {time.time() - t0:.0f}s", flush=True)
    block_index.flush()
    del block_index
    bits = np.concatenate(packed_all)
    bits.tofile(os.path.join(out, "bits.u8"))
    del packed_all

    meta = {"source_dir": source, "n_scenes": n, "n_parts": n_masks, "n_masks": n_masks,
            "mask_axis": "mask_index (per unique mask_key)", "height": H, "width": W,
            "scale": SCALE, "block": BLOCK, "n_mixed_blocks": int(bits.shape[0]),
            "split_seed": C.SPLIT_SEED, "categories": list(CATEGORIES),
            "center_margin": C.CENTER_MARGIN, "coarse_maps": COARSE_MAPS,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out, "scenes.json"), "w") as f:
        json.dump({"meta": meta, "scenes": scenes}, f)

    res, offsets, total = {}, [0], 0
    with ProcessPoolExecutor(workers) as ex:
        for done, (m, p) in enumerate(ex.map(_mask_boundary_points, range(n_masks), chunksize=4)):
            res[m] = p
            if done % 500 == 0:
                print(f"boundary points {done}/{n_masks} {time.time() - t0:.0f}s", flush=True)
    for m in range(n_masks):
        total += len(res[m])
        offsets.append(total)
    np.save(os.path.join(out, "bpoints.npy"), np.concatenate([res[m] for m in range(n_masks)]))
    np.save(os.path.join(out, "bpoints_offsets.npy"), np.array(offsets, np.int64))
    del res

    base_of_mask = {}
    for s in scenes:
        bk = base_key.get(s["part_index"])
        base_of_mask[s["mask_index"]] = mask_of[bk] if bk is not None and s["mask_plan"] == "base_plus_contaminants" else None
    cands = {}
    with ProcessPoolExecutor(workers) as ex:
        for m, o in ex.map(_categorize, [(m, base_of_mask.get(m), cover_path, n_masks, C.SPLIT_SEED)
                                         for m in range(n_masks)], chunksize=4):
            for name, arr in o.items():
                cands[f"{m}/{name}"] = arr
    np.savez(os.path.join(out, "candidates.npz"), **cands)
    print(f"candidates {time.time() - t0:.0f}s", flush=True)

    with ProcessPoolExecutor(workers) as ex:
        for i, bl, coef, lvl in ex.map(_levels, [(i, s["mask_index"], n, n_masks, s["coarse_index"])
                                                 for i, s in enumerate(scenes)], chunksize=8):
            scenes[i].update(backlight_level=bl, field_coef=coef, field_level=lvl)
    meta["backlight_level"] = "median image value over mask-background pixels"
    meta["field"] = "seed.fit_field_masked on the raw render (8-bit DN), coarse_v2 background"
    with open(os.path.join(out, "scenes.json"), "w") as f:
        json.dump({"meta": meta, "scenes": scenes}, f)
    print(f"done in {time.time() - t0:.0f}s: {n} scenes, {n_masks} masks, {total} boundary points", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=CACHE_V4)
    ap.add_argument("--limit", type=int, help="first N scenes only (smoke test)")
    a = ap.parse_args(argv)
    build(a.workers, out=a.out, limit=a.limit)


if __name__ == "__main__":
    main(sys.argv[1:])
