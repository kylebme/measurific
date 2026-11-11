"""Dense training crops for the TSDF edge model (PLAN_v3).

A crop is the same continuous rotation/translation/downscale window as crops.sample_crop
(crop pixel (x, y) centred at P = (x+0.5, y+0.5), source s = c + A (P - S/2)), with dense
per-pixel targets instead of point queries:

  sdf   truncated signed distance to the GT boundary, crop px, + inside the object;
        point-to-line distance through the two nearest exact GT crossings (exact for
        straight edges), sign from the 16x mask
  dir   unit vector (crop coordinates) from the pixel toward its nearest edge point

Inputs (4 channels): the coarse-seed channels of CFI v4 (field-normalized reading,
coarse prior 2p - 1, field level) and the raw reading. The coarse prior is perturbed far
more than for CFI (shift, erosion/dilation, logit scale, dropout) so the model treats it as
a topology hint, not as the edge position.
"""

import math

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import IterableDataset, get_worker_info

from . import SCALE
from . import crops as CR
from . import sampler as S

DEFAULT_SDF_CROP = {
    "size": 256,
    "border": 16,              # no loss within this many px of the crop edge
    "trunc_px": 8.0,           # TSDF truncation T (crop px)
    "p_anywhere": 0.15,
    "p_downscale": 0.1,
    "downscale": [1.0, 2.0],
    "coarse_jitter_px": 3.0,   # random shift of the coarse map (source px)
    "coarse_logit_scale": [0.5, 2.0],
    "coarse_morph_px": 3,      # random erosion / dilation of the coarse prior, up to this (crop px)
    "p_coarse_drop": 0.15,     # coarse prior replaced by 0 (uninformative)
}


def dense_targets(store, part, center, A, size, f, T):
    """Truncated signed distance (crop px, + inside) and unit direction to the nearest edge
    for every crop pixel centre."""
    g = np.arange(size) + 0.5
    gx, gy = np.meshgrid(g, g)
    src = CR.crop_to_source(np.stack([gx.ravel(), gy.ravel()], 1), center, A, size)
    inside = store.mask_points(part, np.floor(src[:, 0] * SCALE), np.floor(src[:, 1] * SCALE)) > 0
    r = (size / 2 * math.sqrt(2) + (T + 2) * f) * 1.0
    bps = CR.boundary_points_near(store, part, center, r).astype(np.float64)
    d = np.full(len(src), np.inf)
    vec = np.zeros((len(src), 2))
    if len(bps) >= 2:
        dd, jj = cKDTree(bps).query(src, k=2, distance_upper_bound=(T + 1) * f)
        ok = np.isfinite(dd[:, 0])
        s, j1 = src[ok], jj[ok, 0]
        b1 = bps[j1]
        foot = b1.copy()
        two = np.isfinite(dd[ok, 1])
        b2 = bps[np.where(two, jj[ok, 1], j1)]
        seg = b2 - b1
        L2 = (seg ** 2).sum(1)
        # line through the two nearest crossings when they are neighbours on one edge
        use = two & (L2 > 1e-12) & (L2 < 0.25 ** 2)
        t = np.where(use, ((s - b1) * seg).sum(1) / np.maximum(L2, 1e-12), 0.0)
        use &= (t > -1.5) & (t < 2.5)
        foot[use] = b1[use] + t[use, None] * seg[use]
        v = foot - s
        n = np.linalg.norm(v, axis=1)
        d[ok] = n
        vec[ok] = v / np.maximum(n, 1e-9)[:, None]
    d = np.minimum(d / f, T)
    sdf = np.where(inside, d, -d).reshape(size, size).astype(np.float32)
    # directions into crop coordinates: crop delta = A^-1 source delta; A = f R with R
    # orthogonal (rotation, flip), so A^-1 v is A^T v up to the scale f
    dirc = np.stack([A[0, 0] * vec[:, 0] + A[1, 0] * vec[:, 1], A[0, 1] * vec[:, 0] + A[1, 1] * vec[:, 1]], 1)
    dirc /= np.maximum(np.linalg.norm(dirc, axis=1, keepdims=True), 1e-9)
    return sdf, dirc.reshape(size, size, 2).transpose(2, 0, 1).astype(np.float32), src


def perturb_prior(prior, rng, cfg):
    """Coarse prior channel (2p - 1): random erosion/dilation and dropout."""
    k = int(cfg.get("coarse_morph_px", 0))
    if k > 0:
        r = int(rng.integers(-k, k + 1))
        if r:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(r) + 1, 2 * abs(r) + 1))
            prior = (cv2.dilate if r > 0 else cv2.erode)(prior, ker)
    if rng.random() < cfg.get("p_coarse_drop", 0.0):
        prior = np.zeros_like(prior)
    return prior


def sample_sdf_crop(store, rng, eligible, sampler_cfg, aug_cfg, cfg):
    size = cfg["size"]
    scene, center, A, cat = CR.crop_geometry(store, rng, eligible, sampler_cfg, size, cfg.get("p_anywhere", 0.0))
    rec = store.scenes[scene]
    part = S.mask_of(rec)
    f = 1.0
    if cfg.get("p_downscale") and rng.random() < cfg["p_downscale"]:
        big = 2 * int(round(size * rng.uniform(*cfg["downscale"]) / 2))
        f = big / size
    if f > 1:
        raw = cv2.resize(S.sample_image(store, scene, center, A, context=(big - S.PATCH) // 2),
                         (size, size), interpolation=cv2.INTER_AREA)
        A = A * f
    else:
        raw = S.sample_image(store, scene, center, A, context=(size - S.PATCH) // 2)
    x, info = S.photometric(raw, rec, rng, aug_cfg)
    sdf, dirc, src = dense_targets(store, part, center, A, size, f, cfg["trunc_px"])
    from .seed import coarse_crop_seed
    seed, _ = coarse_crop_seed(x, src, store.coarse_logit(scene), np.asarray(rec["field_coef"]),
                               rec["field_level"], info["gain_raw"], info["offset"], store.W, store.H,
                               rng, cfg.get("coarse_jitter_px", 0.0), tuple(cfg.get("coarse_logit_scale", (1, 1))))
    seed[1] = perturb_prior(seed[1], rng, cfg)
    inp = np.concatenate([seed, ((x.astype(np.float32) - 0.5) * 2.0)[None]], 0)
    return {"x": inp.astype(np.float32), "sdf": sdf, "dir": dirc, "scene": scene, "category": cat,
            "gt_pts": CR.source_to_crop(CR.boundary_points_near(store, part, center, size * f), center, A, size)}


from .sdf_runtime import model_input  # noqa: E402,F401  (full-frame inputs; torch-free)


class SDFCropStream(IterableDataset):
    """Collated dense batches from per-worker ScenePools (data["pool"] scenes each)."""

    def __init__(self, split, batch, sampler_cfg, aug_cfg, crop_cfg, seed, data):
        self.split, self.batch = split, batch
        self.sampler_cfg, self.aug_cfg, self.crop_cfg, self.seed, self.data = sampler_cfg, aug_cfg, crop_cfg, seed, data

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng([*self.seed, info.id if info else 0])
        store = S.open_store(self.data)
        scenes = store.scene_indices(self.split)
        pool = CR.ScenePool(store, scenes, self.data.get("pool", 16), self.data.get("swap_every", 48),
                            self.sampler_cfg["weights"], rng)
        view = CR.PooledStore(store, pool)
        while True:
            items = []
            for _ in range(self.batch):
                items.append(sample_sdf_crop(view, rng, pool.eligible, self.sampler_cfg, self.aug_cfg, self.crop_cfg))
                pool.tick()
            yield {k: torch.from_numpy(np.stack([it[k] for it in items])) for k in ("x", "sdf", "dir")}
