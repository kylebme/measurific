"""Macro-scale multi-part scenes (synthetic-macro, macro-v1-512) for the coarse model.

Scenes are 512 x 512 renders at 1/4 of the 2048 source resolution: `image.png` (16-bit
linear-response render), `instances.png` (0 background, k = parts[k-1]; majority label of a
4x supersampled render), `debris.png` (8-bit coverage of loose debris, which is in no part
mask). Attached debris is part of its part's mask. See ../../research/synthetic-macro/README.md.

The cache packs every scene into memmaps (images.u16, inst.u8, debris.u8) plus an index
with the attributes used for benchmark slices. The split is the dataset's own (80/10/10
by scene hash; parts are sampled per scene, so there is no shared geometry to leak).

Usage: python -m edgesr.macro build [--workers 8]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from . import CACHE_DIR

MACRO_DIR = os.environ.get(
    "EDGESR_MACRO", "/media/kyle/T7 Shield/inspection-synthetic-renders/macro-v1-512")
MACRO_CACHE = os.path.join(CACHE_DIR, "macro_v1")
SIZE = 512


def _scene_dirs(root=MACRO_DIR):
    d = os.path.join(root, "scenes")
    return sorted(os.path.join(d, s) for s in os.listdir(d) if s.startswith("macro_"))


def _load(path):
    meta = json.load(open(os.path.join(path, "metadata.json")))
    img = cv2.imread(os.path.join(path, "image.png"), cv2.IMREAD_UNCHANGED)
    inst = cv2.imread(os.path.join(path, "instances.png"), cv2.IMREAD_UNCHANGED)
    deb = cv2.imread(os.path.join(path, "debris.png"), cv2.IMREAD_UNCHANGED)
    assert img.shape == (SIZE, SIZE) and img.dtype == np.uint16
    assert inst.max() < 256
    bg = (inst == 0) & (deb < 10)
    parts = [{"instance_id": p["instance_id"], "group": p["group"], "family": p["family"],
              "kind": p["kind"], "tilted": p["pose"]["tilt_rad"] > 0,
              "stacked": p["pose"]["stack_on"] is not None, "cut_by_frame": p["cut_by_frame"],
              "material": p["material"]["family"], "attached_debris": len(p["debris"]),
              "mask_area_px": int((inst == p["instance_id"]).sum()),
              "holes": int(sum(p.get("holes", [])))}
             for p in meta["parts"]]
    rec = {"scene_id": meta["scene_id"], "split": meta["split"], "layout": meta["layout"],
           "illumination": meta["illumination"]["class"], "blur": meta["optics"]["blur_class"],
           "backlight_type": meta["backlight"]["angular"]["type"],
           "px_per_mm": meta["px_per_mm"], "n_parts": len(parts), "parts": parts,
           "loose_debris": len(meta["loose_debris"]), "debris_px": int((deb >= 128).sum()),
           "backlight_level": float(np.median(img[bg])) / 65535.0 if bg.any() else None,
           "materials": sorted({p["material"] for p in parts})}
    return rec, img, inst.astype(np.uint8), deb


def build(workers=8, root=MACRO_DIR, out=MACRO_CACHE):
    dirs = _scene_dirs(root)
    n = len(dirs)
    os.makedirs(out, exist_ok=True)
    imgs = np.lib.format.open_memmap(os.path.join(out, "images.npy"), "w+", np.uint16, (n, SIZE, SIZE))
    insts = np.lib.format.open_memmap(os.path.join(out, "inst.npy"), "w+", np.uint8, (n, SIZE, SIZE))
    debs = np.lib.format.open_memmap(os.path.join(out, "debris.npy"), "w+", np.uint8, (n, SIZE, SIZE))
    index = []
    with ProcessPoolExecutor(workers) as ex:
        for i, (rec, img, inst, deb) in enumerate(ex.map(_load, dirs, chunksize=16)):
            imgs[i], insts[i], debs[i] = img, inst, deb
            index.append(rec)
            if i % 500 == 0:
                print(f"{i}/{n}", flush=True)
    for a in (imgs, insts, debs):
        a.flush()
    with open(os.path.join(out, "index.json"), "w") as f:
        json.dump({"source": root, "scenes": index}, f)
    from collections import Counter
    print(f"{n} scenes", Counter(r["split"] for r in index))


class MacroStore:
    def __init__(self, root=MACRO_CACHE, in_memory=False):
        mode = None if in_memory else "r"
        self.images = np.load(os.path.join(root, "images.npy"), mmap_mode=mode)
        self.inst = np.load(os.path.join(root, "inst.npy"), mmap_mode=mode)
        self.debris = np.load(os.path.join(root, "debris.npy"), mmap_mode=mode)
        self.scenes = json.load(open(os.path.join(root, "index.json")))["scenes"]

    def split(self, name):
        return [i for i, s in enumerate(self.scenes) if s["split"] == name]

    def debris_max(self):
        """Per-scene maximum loose-debris coverage (0-255); computed once, cached."""
        path = os.path.join(os.path.dirname(os.path.abspath(self.debris.filename)), "debris_max.npy")
        if os.path.exists(path):
            m = np.load(path)
            if len(m) == len(self.scenes):
                return m
        m = np.array([int(np.asarray(self.debris[i]).max()) for i in range(len(self.scenes))], np.uint8)
        np.save(path, m)
        return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    build(a.workers)


if __name__ == "__main__":
    main()
