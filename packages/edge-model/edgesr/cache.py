"""Build a read-optimized cache of the source renders (not a patch dataset).

Layout under CACHE_DIR:
  scenes.json        per-scene table: id, part, split, appearance classes, levels
  images.u16         (n_scenes, H, W) raw uint16 source images, memmap
  cover.u16          (n_parts, H, W) mask coverage per source pixel, 0..256
  block_index.i32    (n_parts, H*16/64, W*16/64) -1 bg, -2 fg, >=0 index into bits
  bits.u8            (n_mixed, 64, 8) packed 64x64 hi-res mask blocks
  candidates.npz     per-part patch-center candidates by geometry category

Usage: python -m edgesr.cache build [--workers N]
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

from . import CACHE_DIR, SCALE, SOURCE_DIR

H = W = 2048
BLOCK = 64                      # hi-res pixels per mask block edge
NB = H * SCALE // BLOCK         # blocks per mask edge (512)
N_PARTS = 300
FAMILIES = ("plate", "turned", "machined", "sphere", "freeform")
PARTS_PER_FAMILY_HELDOUT = 6    # per family, for each of val and test
SPLIT_SEED = 20260924
CENTER_MARGIN = 48              # patch centers stay this far from the frame
MAX_CANDIDATES = 6000

CATEGORIES = ("edge", "sparse", "dense", "narrow", "corner", "uniform")


def scene_dirs():
    names = sorted(n for n in os.listdir(SOURCE_DIR) if n.startswith("scene_"))
    return [os.path.join(SOURCE_DIR, n) for n in names
            if os.path.exists(os.path.join(SOURCE_DIR, n, "metadata.json"))]


def part_split():
    """Deterministic geometry-level split; every appearance of a part shares it."""
    rng = np.random.default_rng(SPLIT_SEED)
    split = {}
    for fam_i, fam in enumerate(FAMILIES):
        parts = [p for p in range(N_PARTS) if p % 5 == fam_i]
        order = rng.permutation(parts)
        k = PARTS_PER_FAMILY_HELDOUT
        for p in order[:k]:
            split[int(p)] = "val"
        for p in order[k:2 * k]:
            split[int(p)] = "test"
        for p in order[2 * k:]:
            split[int(p)] = "train"
    return split


def _scene_record(path):
    with open(os.path.join(path, "metadata.json")) as f:
        m = json.load(f)
    assert m["projection"]["mask_scale"] == SCALE
    assert m["projection"]["source_width"] == W and m["projection"]["source_height"] == H
    return {
        "scene_id": m["scene_id"],
        "dir": path,
        "part_index": int(m["catalog"]["part_index"]),
        "part_id": m["catalog"]["part_id"],
        "family": m["geometry"]["family"],
        "focus_class": m["camera"]["focus_class"],
        "illumination_class": m["illumination"]["class"],
        "material_family": m["material"]["family"],
        "mask_reused_from": m["render_result"].get("mask_reused_from_scene_id"),
    }


def _load_image(args):
    i, path, images_path, n = args
    a = np.array(Image.open(os.path.join(path, "image.png")))
    assert a.dtype == np.uint16 and a.shape == (H, W), (path, a.dtype, a.shape)
    mm = np.memmap(images_path, np.uint16, "r+", shape=(n, H, W))
    mm[i] = a
    mm.flush()
    # Background (backlight) and object levels for exposure normalization.
    lo, hi = np.percentile(a[::4, ::4], [1.0, 99.5])
    return i, float(lo), float(hi)


def _load_part(args):
    part, scene_path, cover_path = args
    with open(os.path.join(scene_path, "mask", "index.json")) as f:
        index = json.load(f)
    assert index["width"] == W * SCALE and index["height"] == H * SCALE
    block_index = np.full((NB, NB), -1, np.int32)
    bits = []
    cover = np.memmap(cover_path, np.uint16, "r+", shape=(N_PARTS, H, W))
    for tile in index["tiles"]:
        tx, ty, tw, th = tile["x"], tile["y"], tile["width"], tile["height"]
        assert tw % BLOCK == 0 and th % BLOCK == 0 and tx % BLOCK == 0 and ty % BLOCK == 0
        sx, sy = tx // SCALE, ty // SCALE
        if tile["foreground_pixels"] == 0:
            cover[part, sy:sy + th // SCALE, sx:sx + tw // SCALE] = 0
            continue
        with Image.open(os.path.join(scene_path, "mask", tile["file"])) as im:
            t = np.array(im.convert("1"), dtype=bool)
        assert t.shape == (th, tw)
        assert int(t.sum()) == tile["foreground_pixels"], tile["file"]
        c = t.reshape(th // SCALE, SCALE, tw // SCALE, SCALE).sum((1, 3), dtype=np.uint16)
        cover[part, sy:sy + th // SCALE, sx:sx + tw // SCALE] = c
        blocks = t.reshape(th // BLOCK, BLOCK, tw // BLOCK, BLOCK).transpose(0, 2, 1, 3)
        s = blocks.sum((2, 3))
        bx0, by0 = tx // BLOCK, ty // BLOCK
        bi = block_index[by0:by0 + th // BLOCK, bx0:bx0 + tw // BLOCK]
        bi[s == BLOCK * BLOCK] = -2
        ys, xs = np.nonzero((s > 0) & (s < BLOCK * BLOCK))
        if len(ys):
            bi[ys, xs] = -1000  # placeholder, assigned after concatenation
            bits.append((by0 + ys, bx0 + xs, np.packbits(blocks[ys, xs], axis=-1)))
    cover.flush()
    total = int(np.asarray(cover[part], np.int64).sum())
    assert total == index["foreground_pixels"], (part, total, index["foreground_pixels"])
    if bits:
        ys = np.concatenate([b[0] for b in bits])
        xs = np.concatenate([b[1] for b in bits])
        packed = np.concatenate([b[2] for b in bits])
    else:
        ys = xs = np.zeros(0, np.int64)
        packed = np.zeros((0, BLOCK, BLOCK // 8), np.uint8)
    return part, block_index, ys, xs, packed


def _categorize(args):
    """Per-part patch-center candidates, at source-pixel resolution."""
    part, cover_path, seed = args
    cover = np.memmap(cover_path, np.uint16, "r", shape=(N_PARTS, H, W))[part]
    cov = np.asarray(cover, np.float32) / (SCALE * SCALE)
    boundary = ((cover > 0) & (cover < SCALE * SCALE)).astype(np.uint8)
    win = cv2.boxFilter(boundary.astype(np.float32), -1, (48, 48), normalize=False,
                        borderType=cv2.BORDER_CONSTANT)
    fg = (cover >= SCALE * SCALE // 2).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thin_fg = fg & (1 - cv2.morphologyEx(fg, cv2.MORPH_OPEN, k))
    thin_bg = (1 - fg) & cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    narrow = (thin_fg | thin_bg).astype(bool)
    narrow &= cv2.dilate(boundary, np.ones((3, 3), np.uint8)).astype(bool)
    harris = cv2.cornerHarris(cov, 3, 3, 0.04)
    corner = (harris > 0.02) & cv2.dilate(boundary, np.ones((3, 3), np.uint8)).astype(bool)
    near_obj = cv2.dilate(boundary, np.ones((129, 129), np.uint8)).astype(bool)
    valid = np.zeros((H, W), bool)
    valid[CENTER_MARGIN:H - CENTER_MARGIN, CENTER_MARGIN:W - CENTER_MARGIN] = True
    masks = {
        "edge": boundary.astype(bool),
        "sparse": (win > 0) & (win <= 60) & boundary.astype(bool),
        "dense": (win >= 100) & boundary.astype(bool),
        "narrow": narrow,
        "corner": corner,
        "uniform": (win == 0) & near_obj,
    }
    rng = np.random.default_rng(seed + part)
    out = {}
    for name in CATEGORIES:
        ys, xs = np.nonzero(masks[name] & valid)
        if len(ys) > MAX_CANDIDATES:
            sel = rng.choice(len(ys), MAX_CANDIDATES, replace=False)
            ys, xs = ys[sel], xs[sel]
        out[name] = np.stack([xs, ys], 1).astype(np.int16)
    return part, out


def build(workers):
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    dirs = scene_dirs()
    with ProcessPoolExecutor(workers) as ex:
        scenes = list(ex.map(_scene_record, dirs, chunksize=16))
    split = part_split()
    for s in scenes:
        s["split"] = split[s["part_index"]]
    n = len(scenes)
    print(f"{n} scenes, {len({s['part_index'] for s in scenes})} parts", flush=True)

    images_path = os.path.join(CACHE_DIR, "images.u16")
    np.memmap(images_path, np.uint16, "w+", shape=(n, H, W)).flush()
    with ProcessPoolExecutor(workers) as ex:
        for done, (i, lo, hi) in enumerate(ex.map(
                _load_image, [(i, s["dir"], images_path, n) for i, s in enumerate(scenes)],
                chunksize=4)):
            scenes[i]["object_level"] = lo
            scenes[i]["background_level"] = hi
            if done % 200 == 0:
                print(f"images {done}/{n} {time.time() - t0:.0f}s", flush=True)

    # One mask per part: prefer a scene that rendered it rather than reused it.
    part_scene = {}
    for s in scenes:
        p = s["part_index"]
        if p not in part_scene or (part_scene[p]["mask_reused_from"] and not s["mask_reused_from"]):
            part_scene[p] = s
    assert sorted(part_scene) == list(range(N_PARTS))
    cover_path = os.path.join(CACHE_DIR, "cover.u16")
    np.memmap(cover_path, np.uint16, "w+", shape=(N_PARTS, H, W)).flush()
    block_index = np.full((N_PARTS, NB, NB), -1, np.int32)
    packed_all, offset = [], 0
    with ProcessPoolExecutor(workers) as ex:
        jobs = [(p, part_scene[p]["dir"], cover_path) for p in range(N_PARTS)]
        for done, (p, bi, ys, xs, packed) in enumerate(ex.map(_load_part, jobs)):
            bi[ys, xs] = offset + np.arange(len(ys), dtype=np.int32)
            assert not (bi == -1000).any()
            block_index[p] = bi
            packed_all.append(packed)
            offset += len(ys)
            if done % 20 == 0:
                print(f"masks {done}/{N_PARTS} mixed_blocks={offset} {time.time() - t0:.0f}s",
                      flush=True)
    block_index.tofile(os.path.join(CACHE_DIR, "block_index.i32"))
    bits = np.concatenate(packed_all)
    bits.tofile(os.path.join(CACHE_DIR, "bits.u8"))

    cands = {}
    with ProcessPoolExecutor(workers) as ex:
        for p, out in ex.map(_categorize, [(p, cover_path, SPLIT_SEED) for p in range(N_PARTS)]):
            for name, arr in out.items():
                cands[f"{p}/{name}"] = arr
    np.savez(os.path.join(CACHE_DIR, "candidates.npz"), **cands)

    meta = {
        "source_dir": SOURCE_DIR, "n_scenes": n, "n_parts": N_PARTS, "height": H, "width": W,
        "scale": SCALE, "block": BLOCK, "n_mixed_blocks": int(bits.shape[0]),
        "split_seed": SPLIT_SEED, "categories": list(CATEGORIES),
        "center_margin": CENTER_MARGIN, "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(CACHE_DIR, "scenes.json"), "w") as f:
        json.dump({"meta": meta, "scenes": scenes}, f, indent=1)
    add_backlight(workers)
    print(f"done in {time.time() - t0:.0f}s: {meta}", flush=True)


def _backlight(args):
    i, part, n = args
    img = np.memmap(os.path.join(CACHE_DIR, "images.u16"), np.uint16, "r", shape=(n, H, W))[i]
    cov = np.memmap(os.path.join(CACHE_DIR, "cover.u16"), np.uint16, "r",
                    shape=(N_PARTS, H, W))[part]
    return float(np.median(np.asarray(img)[np.asarray(cov) == 0]))


def add_backlight(workers):
    """Add backlight_level to scenes.json: the scene's backlight brightness, defined
    from ground truth as the median image value over background (coverage 0) pixels.
    It is the value supplied to backlight-conditioned models."""
    path = os.path.join(CACHE_DIR, "scenes.json")
    with open(path) as f:
        d = json.load(f)
    n = len(d["scenes"])
    with ProcessPoolExecutor(workers) as ex:
        levels = list(ex.map(_backlight, [(i, s["part_index"], n)
                                          for i, s in enumerate(d["scenes"])], chunksize=16))
    for s, bl in zip(d["scenes"], levels):
        s["backlight_level"] = bl
    d["meta"].pop("backlight_estimator", None)
    d["meta"]["backlight_level"] = "median image value over mask-background pixels"
    with open(path + ".tmp", "w") as f:
        json.dump(d, f, indent=1)
    os.replace(path + ".tmp", path)


def _part_boundary_points(args):
    """All marching-squares crossings of one part's 16x mask, in source px (exact to the
    1/32 px quantization of the mask). Processes 2048^2 hi-res tiles with a 1-px
    overlap; duplicates on the overlap are removed."""
    from .metrics import contour_points
    from .sampler import SourceStore
    part = args
    store = SourceStore()
    bi = np.asarray(store._open()["block_index"][part])
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
            w = store.mask_window(part, tx, ty, min(T + 1, n - tx), min(T + 1, n - ty))
            p = contour_points(w.astype(np.float32))
            if len(p):
                pts.append(p + np.array([tx, ty]) / SCALE)
    if not pts:
        return part, np.zeros((0, 2), np.float32)
    p = np.concatenate(pts)
    q = np.unique(np.round(p * 64).astype(np.int64), axis=0)
    return part, (q / 64.0).astype(np.float32)


def build_boundary_points(workers):
    out, offsets, total = [], [0], 0
    with ProcessPoolExecutor(workers) as ex:
        res = dict(ex.map(_part_boundary_points, range(N_PARTS)))
    for p in range(N_PARTS):
        out.append(res[p])
        total += len(res[p])
        offsets.append(total)
    np.save(os.path.join(CACHE_DIR, "bpoints.npy"), np.concatenate(out))
    np.save(os.path.join(CACHE_DIR, "bpoints_offsets.npy"), np.array(offsets, np.int64))
    print(f"boundary points: {total} total, median/part {int(np.median(np.diff(offsets)))}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "add-backlight", "boundary-points"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.workers)
        build_boundary_points(a.workers)
    elif a.cmd == "boundary-points":
        build_boundary_points(a.workers)
    else:
        add_backlight(a.workers)


if __name__ == "__main__":
    main(sys.argv[1:])
