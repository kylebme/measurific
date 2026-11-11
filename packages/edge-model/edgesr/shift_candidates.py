"""Per-scene "shift" crop candidates for the v4 cache: GT part boundary where the naive
visible edge (bench_v4.visible_shift: the 0.5 level of the clean render between the dark
level and the backlight field) is 0.75-8 px away. These are the depth-of-field ramps and
grey shoulders of thick parts and deep narrow bores, the human's priority failure mode
(RESEARCH.md, 2026-09-30). The sampler's "shift" category draws crop centers from them.

  python -m edgesr.shift_candidates [--workers 4] [--limit N]
Writes work/cache_v4/candidates_scene.npz (keys "<scene>/shift", int16 pixel coords).
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import sampler as S

LO_PX, HI_PX = 0.75, 8.0
MAX_CANDIDATES = 4000


_STORE = None


def _scene(sc):
    global _STORE
    from . import bench_v4 as B4
    if _STORE is None:
        _STORE = S.open_store({"cache": "v4"})
    store = _STORE
    rec = store.scenes[sc]
    gt = store.boundary_points(rec["mask_index"]).astype(np.float64)
    if not len(gt):
        return sc, np.zeros((0, 2), np.int16), 0.0
    dv = B4.visible_shift(store, sc, gt)
    sel = (dv > LO_PX) & (dv <= HI_PX) & ~B4.contaminant_points(store, sc, gt)
    pts = gt[sel]
    if len(pts) > MAX_CANDIDATES:
        pts = pts[np.random.default_rng(sc).choice(len(pts), MAX_CANDIDATES, replace=False)]
    return sc, np.floor(pts).astype(np.int16), float(sel.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    store = S.open_store({"cache": "v4"})
    scenes = range(len(store.scenes))[:a.limit] if a.limit else range(len(store.scenes))
    out, fracs, t0 = {}, [], time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for done, (sc, pts, frac) in enumerate(ex.map(_scene, scenes, chunksize=8)):
            if len(pts):
                out[f"{sc}/shift"] = pts
            fracs.append(frac)
            if done % 500 == 0:
                print(f"{done}/{len(scenes)} {time.time() - t0:.0f}s", flush=True)
    path = os.path.join(store.cache_dir, "candidates_scene.npz" if not a.limit else "candidates_scene_test.npz")
    np.savez(path, **out)
    f = np.array(fracs)
    print(f"wrote {path}: {len(out)}/{len(fracs)} scenes with shift candidates; shifted fraction of part "
          f"boundary mean {f.mean():.3f}, median {np.median(f):.3f}, p90 {np.percentile(f, 90):.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
