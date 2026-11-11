"""Crop + point-query sampling for CFI training (PLAN_v2).

A crop is an S x S source-pixel window at a continuous rotation/translation (same
conventions as sampler.py: crop pixel (x, y) centered at P = (x+0.5, y+0.5), source
s = c + A (P - S/2)). Supervision is at query points, never dense hi-res masks:

  occupancy   GT mask value at hi-res index floor(16 s)         (exact)
  edge field  distance + direction to the nearest cached GT boundary crossing
  coarse      per 4x4 cell occupancy fraction and "edge within Rb px" (from cover map)
  pixel band  per crop pixel "edge within 1 px" (from cover map)
"""

import math

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import IterableDataset, get_worker_info

from . import SCALE
from . import sampler as S

DEFAULT_CROP = {
    "size": 192,
    "border": 24,             # no point / fine losses within this many px of the crop edge
    "n_near": 3072,           # queries within 1 px of a GT edge
    "n_band": 1536,           # queries within band_px of a GT edge
    "n_uniform": 512,
    "band_px": 8.0,
    "coarse_band_px": 8,      # band4 target: 4x4 cell within this many px of an edge
    "dist_clip": 2.0,
    "seed_inputs": False,     # also produce classical seed inputs (context="seed" models)
    "p_anywhere": 0.0,        # fraction of crops centered uniformly over the whole image,
                              # overhanging the frame (edge-replicate), any scene
    # fine model v4 (DATA_V3.md)
    "p_downscale": 0.0,       # crops read at 1..2x the size and area-downsampled (the
    "downscale": [1.0, 2.0],  #   scale goes into the crop matrix A, so GT stays exact)
    "coarse_jitter_px": 0.0,  # coarse-seed inputs: random shift of the coarse map (source px)
    "coarse_logit_scale": [1.0, 1.0],   # and random scale of its logit
}


def crop_geometry(store, rng, eligible, sampler_cfg, size, p_anywhere=0.0):
    """Center/rotation via the patch proposal machinery, re-clipped for the crop size.
    With probability p_anywhere: any scene, center uniform over the image, allowed to
    overhang the frame."""
    if p_anywhere and rng.random() < p_anywhere:
        pool = eligible["edge"]
        scene = pool[rng.integers(len(pool))]
        flip = bool(rng.integers(2))
        if rng.random() < sampler_cfg["p_lattice"]:
            A = S.patch_matrix(flip=flip, quarter_turns=int(rng.integers(4)))
            c = (float(rng.integers(0, store.W + 1)), float(rng.integers(0, store.H + 1)))
        else:
            A = S.patch_matrix(angle=rng.uniform(0, 2 * math.pi), flip=flip)
            c = (rng.uniform(0, store.W), rng.uniform(0, store.H))
        return scene, c, A, "anywhere"
    scene, center, A, cat = S.propose(store, rng, eligible, sampler_cfg)
    rotated = abs(A[0, 1]) > 1e-9 and abs(A[0, 0]) > 1e-9
    half = size / 2 * (math.sqrt(2) if rotated else 1.0) + 4
    cx = float(np.clip(center[0], half, store.W - half))
    cy = float(np.clip(center[1], half, store.H - half))
    if not rotated:
        cx, cy = float(round(cx)), float(round(cy))
    return scene, (cx, cy), A, cat


BIN = 64          # px, spatial bins of pooled boundary points


def bin_points(bp, W=2048, H=2048):
    """Boundary points sorted into BIN x BIN px cells: (sorted points, cell offsets)."""
    nx, ny = (W + BIN - 1) // BIN, (H + BIN - 1) // BIN
    cx = np.clip((bp[:, 0] // BIN).astype(np.int64), 0, nx - 1)
    cy = np.clip((bp[:, 1] // BIN).astype(np.int64), 0, ny - 1)
    key = cy * nx + cx
    order = np.argsort(key, kind="stable")
    return bp[order], np.searchsorted(key[order], np.arange(nx * ny + 1)), nx, ny


def boundary_points_near(store, part, center, r):
    """GT boundary crossings within the square of half-size r around center."""
    binned = getattr(store, "binned_points", None)
    if binned is None:
        bp = store.boundary_points(part)
    else:
        pts, offs, nx, ny = binned(part)
        x0, x1 = max(int((center[0] - r) // BIN), 0), min(int((center[0] + r) // BIN), nx - 1)
        y0, y1 = max(int((center[1] - r) // BIN), 0), min(int((center[1] + r) // BIN), ny - 1)
        if x0 > x1 or y0 > y1:
            return np.zeros((0, 2), np.float32)
        bp = np.concatenate([pts[offs[y * nx + x0]:offs[y * nx + x1 + 1]] for y in range(y0, y1 + 1)])
    return bp[np.all(np.abs(bp - np.array(center)) <= r, 1)]


def crop_to_source(P, center, A, size):
    d = P - size / 2
    return np.stack([center[0] + A[0, 0] * d[:, 0] + A[0, 1] * d[:, 1],
                     center[1] + A[1, 0] * d[:, 0] + A[1, 1] * d[:, 1]], 1)


def source_to_crop(s, center, A, size):
    d = s - np.array(center)
    # A = f R (rotation, flip, downscale f). Closed-form 2x2 inverse: np.linalg.inv goes
    # through OpenBLAS, whose spinning thread pool in every data worker capped the whole
    # loader at ~160 crops/s.
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    return np.stack([(A[1, 1] * d[:, 0] - A[0, 1] * d[:, 1]) / det,
                     (-A[1, 0] * d[:, 0] + A[0, 0] * d[:, 1]) / det], 1) + size / 2


def sample_crop(store, rng, eligible, sampler_cfg, aug_cfg, cfg):
    size, border = cfg["size"], cfg["border"]
    scene, center, A, cat = crop_geometry(store, rng, eligible, sampler_cfg, size,
                                          cfg.get("p_anywhere", 0.0))
    part = S.mask_of(store.scenes[scene])
    f = 1.0
    if cfg.get("p_downscale") and rng.random() < cfg["p_downscale"]:
        big = 2 * int(round(size * rng.uniform(*cfg["downscale"]) / 2))
        f = big / size
    if f > 1:
        # read f x larger at native resolution, then area-average down (physical pixel
        # integration); crop pixel P maps to source c + (f A)(P - size/2)
        raw = cv2.resize(S.sample_image(store, scene, center, A, context=(big - S.PATCH) // 2),
                         (size, size), interpolation=cv2.INTER_AREA)
        A = A * f
    else:
        raw = S.sample_image(store, scene, center, A, context=(size - S.PATCH) // 2)
    x, info = S.photometric(raw, store.scenes[scene], rng, aug_cfg)

    # GT boundary crossings inside (and slightly around) the crop, in crop coordinates.
    r = (size / 2 * math.sqrt(2) + cfg["dist_clip"] + cfg["band_px"] + 2) * f
    bps = boundary_points_near(store, part, center, r).astype(np.float64)
    bpc = source_to_crop(bps, center, A, size)

    lo, hi = border, size - border
    inside = np.all((bpc >= lo) & (bpc <= hi), 1)
    cand = bpc[inside]
    qs = []
    if len(cand):
        for n, rad in ((cfg["n_near"], 1.0), (cfg["n_band"], cfg["band_px"])):
            base = cand[rng.integers(len(cand), size=n)]
            ang = rng.uniform(0, 2 * math.pi, n)
            rr = rad * np.sqrt(rng.uniform(0, 1, n))
            qs.append(base + np.stack([rr * np.cos(ang), rr * np.sin(ang)], 1))
    n_uni = cfg["n_uniform"] + (0 if len(cand) else cfg["n_near"] + cfg["n_band"])
    qs.append(rng.uniform(lo, hi, (n_uni, 2)))
    q = np.concatenate(qs)
    q = q[np.all((q >= lo) & (q <= hi), 1)]

    qs_src = crop_to_source(q, center, A, size)
    occ = store.mask_points(part, np.floor(qs_src[:, 0] * SCALE), np.floor(qs_src[:, 1] * SCALE))
    if len(bps):
        d, j = cKDTree(bps).query(qs_src)
        vec = (bps[j] - qs_src) / np.maximum(d, 1e-9)[:, None]
        dirc = np.stack([A[0, 0] * vec[:, 0] + A[1, 0] * vec[:, 1],
                         A[0, 1] * vec[:, 0] + A[1, 1] * vec[:, 1]], 1)
        if f > 1:           # distances in crop px; directions stay unit vectors
            d = d / f
            dirc = dirc / np.maximum(np.linalg.norm(dirc, axis=1, keepdims=True), 1e-9)
    else:
        d = np.full(len(q), 1e3)
        dirc = np.zeros((len(q), 2))

    # Dense low-res targets from the source-res coverage map (nearest sample per crop px;
    # downscaled crops: sampled at the read resolution and area-averaged).
    g = (np.arange(size) + 0.5)
    gx, gy = np.meshgrid(g, g)
    src = crop_to_source(np.stack([gx.ravel(), gy.ravel()], 1), center, A, size)
    cover_part = store.cover[part]
    if f > 1:
        gb = (np.arange(big) + 0.5) / f
        bx, by = np.meshgrid(gb, gb)
        sb = crop_to_source(np.stack([bx.ravel(), by.ravel()], 1), center, A, size)
        ix = np.clip(np.floor(sb[:, 0]).astype(int), 0, store.W - 1)
        iy = np.clip(np.floor(sb[:, 1]).astype(int), 0, store.H - 1)
        cov = cv2.resize(np.asarray(cover_part)[iy, ix].reshape(big, big).astype(np.float32) / 256,
                         (size, size), interpolation=cv2.INTER_AREA)
    else:
        ix = np.clip(np.floor(src[:, 0]).astype(int), 0, store.W - 1)
        iy = np.clip(np.floor(src[:, 1]).astype(int), 0, store.H - 1)
        cov = np.asarray(cover_part)[iy, ix].reshape(size, size).astype(np.float32) / 256
    edge = ((cov > 0) & (cov < 1)).astype(np.uint8)
    pband = cv2.dilate(edge, np.ones((3, 3), np.uint8))
    k = 2 * cfg["coarse_band_px"] + 1
    band4 = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    band4 = band4.reshape(size // 4, 4, size // 4, 4).max((1, 3))
    occ4 = cov.reshape(size // 4, 4, size // 4, 4).mean((1, 3))
    extra = {}
    if cfg.get("seed_inputs") == "coarse":
        from .seed import coarse_crop_seed
        rec = store.scenes[scene]
        sin, smask = coarse_crop_seed(x, src, store.coarse_logit(scene), np.asarray(rec["field_coef"]),
                                      rec["field_level"], info["gain_raw"], info["offset"], store.W, store.H,
                                      rng, cfg.get("coarse_jitter_px", 0.0),
                                      tuple(cfg.get("coarse_logit_scale", (1.0, 1.0))))
        extra = {"seed": sin, "seed_mask": smask.astype(np.float32)}
    elif cfg.get("seed_inputs"):
        from .seed import crop_seed
        sin, smask = crop_seed(x, src, scene, info["gain_raw"], info["offset"], store.W, store.H)
        extra = {"seed": sin, "seed_mask": smask.astype(np.float32)}
    return {**extra,
        "x": x.astype(np.float32), "backlight": np.float32(info["backlight"]),
        "q": q.astype(np.float32), "occ": occ.astype(np.float32),
        "dist": d.astype(np.float32), "dir": dirc.astype(np.float32),
        "occ4": occ4.astype(np.float32), "band4": band4.astype(np.float32),
        "pband": pband.astype(np.float32), "scene": scene, "category": cat,
        "gt_pts": bpc.astype(np.float64),       # exact GT crossings, crop coords
    }


class ScenePool:
    """A worker's pool of scenes held in RAM (image, mask coverage, coarse map). Crops
    come only from pooled scenes; one scene is swapped for a random one every
    `swap_every` crops. On the v4 cache (61 GB of images on the USB drive, more than the
    page cache) this turns random crop reads into sequential whole-scene reads."""

    def __init__(self, store, scenes, size, swap_every, weights, rng):
        self.store, self.all, self.size, self.swap_every = store, list(scenes), size, swap_every
        self.weights, self.rng, self.n = weights, rng, 0
        self.images, self.cover, self._coarse, self.bpoints, self.members = {}, {}, {}, {}, []
        self.masks = {}
        for i in rng.choice(self.all, min(size, len(self.all)), replace=False):
            self._add(int(i))
        self._refresh()

    def _add(self, i):
        st = self.store
        self.images[i] = np.array(st.images[i])
        m = S.mask_of(st.scenes[i])
        if m not in self.cover:
            self.cover[m] = np.array(st.cover[m])
            self.bpoints[m] = bin_points(st.boundary_points(m), st.W, st.H)
            self.masks[m] = st.mask_arrays(m)      # point lookups from RAM, not random USB reads
        if "coarse_index" in st.scenes[i]:
            self._coarse[i] = st.coarse_logit(i)
        self.members.append(i)

    def _drop(self, i):
        self.members.remove(i)
        self.images.pop(i)
        self._coarse.pop(i, None)
        m = S.mask_of(self.store.scenes[i])
        if all(S.mask_of(self.store.scenes[j]) != m for j in self.members):
            self.cover.pop(m)
            self.bpoints.pop(m)
            self.masks.pop(m)

    def _refresh(self):
        self.eligible = S.eligible_scenes(self.store, self.members, self.weights)

    def tick(self):
        self.n += 1
        if self.n % self.swap_every == 0:
            self._drop(self.members[int(self.rng.integers(len(self.members)))])
            while True:
                i = int(self.rng.choice(self.all))
                if i not in self.images:
                    break
            self._add(i)
            self._refresh()


class PooledStore:
    """SourceStore view whose images / cover / coarse maps come from a ScenePool."""

    def __init__(self, store, pool):
        self._store, self.pool = store, pool
        self.images, self.cover = pool.images, pool.cover

    def coarse_logit(self, scene):
        return self.pool._coarse[scene]

    def binned_points(self, part):
        return self.pool.bpoints[part]

    def boundary_points(self, part):
        return self.pool.bpoints[part][0]

    def mask_points(self, part, mx, my):
        bi, bits = self.pool.masks[part]
        return S.lookup_mask(bi, bits, self._store.block, mx, my)

    def __getattr__(self, name):
        return getattr(self._store, name)


class CropStream(IterableDataset):
    """Yields collated batches (variable query counts are concatenated with a batch
    index), so the DataLoader should use batch_size=None. data: config "data" section
    ({"cache": "v3" | "v4", "pool": K, "swap_every": M}); pool > 0 streams from a
    per-worker ScenePool."""

    def __init__(self, split, batch, sampler_cfg, aug_cfg, crop_cfg, seed, data=None):
        self.split, self.batch = split, batch
        self.sampler_cfg, self.aug_cfg, self.crop_cfg, self.seed = sampler_cfg, aug_cfg, crop_cfg, seed
        self.data = data or {}

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng([*self.seed, info.id if info else 0])
        store = S.open_store(self.data)
        scenes = store.scene_indices(self.split)
        pool = None
        if self.data.get("pool"):
            pool = ScenePool(store, scenes, self.data["pool"], self.data.get("swap_every", 16),
                             self.sampler_cfg["weights"], rng)
            view = PooledStore(store, pool)
        else:
            view = store
            eligible = S.eligible_scenes(store, scenes, self.sampler_cfg["weights"])
        while True:
            items = []
            for _ in range(self.batch):
                if pool is not None:
                    eligible = pool.eligible
                items.append(sample_crop(view, rng, eligible, self.sampler_cfg, self.aug_cfg, self.crop_cfg))
                if pool is not None:
                    pool.tick()
            yield collate(items)


def collate(items):
    keys = ["x", "backlight", "occ4", "band4", "pband"] + (["seed", "seed_mask"] if "seed" in items[0] else [])
    out = {k: torch.from_numpy(np.stack([it[k] for it in items])) for k in keys}
    out["x"] = out["x"][:, None]
    for k in ("q", "occ", "dist", "dir"):
        out[k] = torch.from_numpy(np.concatenate([it[k] for it in items]))
    out["qb"] = torch.from_numpy(np.concatenate(
        [np.full(len(it["q"]), i, np.int64) for i, it in enumerate(items)]))
    return out
