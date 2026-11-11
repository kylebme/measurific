"""Dynamic joint sampling of source-image patches and their 16x mask patches.

Coordinate convention (all continuous, x right, y down, top-left origin):
  * source pixel i covers [i, i+1); its center is i + 0.5
  * hi-res mask pixel j covers [j/16, (j+1)/16) in source units
  * patch coordinates P are in input-pixel units over [0, 48)^2; input pixel
    (x, y) is centered at P = (x + 0.5, y + 0.5) and output mask pixel (a, b) at
    P = ((a + 0.5) / 16, (b + 0.5) / 16)
  * a patch is the map s = c + A (P - 24) from patch to source coordinates,
    where c is the patch center and A a rotation (optionally with a flip)

Input pixels are Keys-bicubic interpolated at s. Mask pixels take the value of
the hi-res mask pixel that contains s, i.e. index floor(16 s). For lattice
patches (integer c and A in the dihedral group) both are exact copies.
"""

import json
import math
import os

import cv2
import numpy as np

from . import CACHE_DIR, SCALE

PATCH = 48
OUT = PATCH * SCALE


def mask_of(rec):
    """Mask index of a scene record: per unique mask in the v4 cache (tilted and
    contaminated scenes have their own), the catalog part in the original cache."""
    return rec.get("mask_index", rec["part_index"])


def open_store(data=None):
    """SourceStore for a config's "data" section: {"cache": "v3" (default) | "v4"}."""
    if (data or {}).get("cache", "v3") == "v4":
        from .cache_v4 import CACHE_V4
        return SourceStore(CACHE_V4)
    return SourceStore()


def lookup_mask(block_index, bits, B, mx, my):
    """Mask values at integer hi-res indices from one mask's block index (-1 empty,
    -2 full, >= 0 row of `bits`) and packed 64 x 64 blocks; outside -> 0."""
    nb = block_index.shape[0]
    mx, my = np.asarray(mx, np.int64), np.asarray(my, np.int64)
    inside = (mx >= 0) & (my >= 0) & (mx < nb * B) & (my < nb * B)
    bx, by = np.clip(mx // B, 0, nb - 1), np.clip(my // B, 0, nb - 1)
    idx = np.asarray(block_index)[by, bx]
    out = (idx == -2).astype(np.uint8)
    mixed = (idx >= 0) & inside
    if mixed.any():
        wy, wx = my[mixed] % B, mx[mixed] % B
        byte = bits[idx[mixed], wy, wx >> 3]
        out[mixed] = (byte >> (7 - (wx & 7))) & 1
    out[~inside] = 0
    return out


class SourceStore:
    """Read-only view of the cache. Opens memmaps lazily so it is fork-safe."""

    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        with open(os.path.join(cache_dir, "scenes.json")) as f:
            d = json.load(f)
        self.meta, self.scenes = d["meta"], d["scenes"]
        self.H, self.W = self.meta["height"], self.meta["width"]
        self.block = self.meta["block"]
        self._maps = None
        self._cands = None

    def _open(self):
        if self._maps is None:
            m, c = self.meta, self.cache_dir
            nb = self.H * SCALE // self.block
            self._maps = dict(
                images=np.memmap(os.path.join(c, "images.u16"), np.uint16, "r",
                                 shape=(m["n_scenes"], self.H, self.W)),
                cover=np.memmap(os.path.join(c, "cover.u16"), np.uint16, "r",
                                shape=(m["n_parts"], self.H, self.W)),
                block_index=np.memmap(os.path.join(c, "block_index.i32"), np.int32, "r",
                                      shape=(m["n_parts"], nb, nb)),
                bits=np.memmap(os.path.join(c, "bits.u8"), np.uint8, "r",
                               shape=(m["n_mixed_blocks"], self.block, self.block // 8)),
            )
        return self._maps

    @property
    def images(self):
        return self._open()["images"]

    @property
    def cover(self):
        return self._open()["cover"]

    def candidates(self, part, category):
        """Candidate crop centers of a mask (part) and category; read lazily per key (the
        v4 cache has ~4.6k masks x 7 categories)."""
        if self._cands is None:
            self._npz = np.load(os.path.join(self.cache_dir, "candidates.npz"))
            self._cand_keys = set(self._npz.files)
            self._cands = {}
        key = f"{part}/{category}"
        if key not in self._cands:
            self._cands[key] = self._npz[key] if key in self._cand_keys else np.zeros((0, 2), np.int16)
        return self._cands[key]

    def scene_candidates(self, scene, category):
        """Candidate crop centers that depend on the image, not only the mask (SCENE_CATEGORIES),
        from candidates_scene.npz (edgesr.shift_candidates); none if it is absent."""
        if not hasattr(self, "_scands"):
            path = os.path.join(self.cache_dir, "candidates_scene.npz")
            self._snpz = np.load(path) if os.path.exists(path) else None
            self._skeys = set(self._snpz.files) if self._snpz is not None else set()
            self._scands = {}
        key = f"{scene}/{category}"
        if key not in self._scands:
            self._scands[key] = self._snpz[key] if key in self._skeys else np.zeros((0, 2), np.int16)
        return self._scands[key]

    def coarse_logit(self, scene):
        """The scene's 512 x 512 coarse part logit (hysteresis applied), v4 cache only."""
        if not hasattr(self, "_coarse"):
            self._coarse = np.load(os.path.join(self.meta["coarse_maps"], "part_logit.npy"), mmap_mode="r")
        from .coarse import suppress_weak
        return suppress_weak(np.asarray(self._coarse[self.scenes[scene]["coarse_index"]], np.float32))

    def scene_indices(self, split):
        return [i for i, s in enumerate(self.scenes) if s["split"] == split]

    def mask_points(self, part, mx, my):
        """GT mask values (uint8) at integer hi-res indices (mx, my); outside -> 0."""
        maps = self._open()
        return lookup_mask(maps["block_index"][part], maps["bits"], self.block, mx, my)

    def mask_arrays(self, part):
        """(block_index, bits) of one mask as RAM arrays, block_index re-based onto the
        mask's own contiguous run of mixed blocks (one sequential read)."""
        maps = self._open()
        bi = np.array(maps["block_index"][part])
        mixed = bi >= 0
        if not mixed.any():
            return bi, np.zeros((0, self.block, self.block // 8), np.uint8)
        lo, hi = int(bi[mixed].min()), int(bi[mixed].max())
        bits = np.array(maps["bits"][lo:hi + 1])
        bi[mixed] -= lo
        return bi, bits

    def boundary_points(self, part):
        """(N, 2) float32 hi-res GT boundary crossings of a part, source px."""
        if not hasattr(self, "_bp"):
            self._bp = np.load(os.path.join(self.cache_dir, "bpoints.npy"), mmap_mode="r")
            self._bpo = np.load(os.path.join(self.cache_dir, "bpoints_offsets.npy"))
        return np.asarray(self._bp[self._bpo[part]:self._bpo[part + 1]])

    def mask_window(self, part, x0, y0, w, h):
        """Hi-res mask pixels [x0, x0+w) x [y0, y0+h) as uint8; outside is 0."""
        maps = self._open()
        B = self.block
        nb = maps["block_index"].shape[1]
        bx0, by0 = x0 // B, y0 // B
        bx1, by1 = (x0 + w - 1) // B + 1, (y0 + h - 1) // B + 1
        grid = np.full((by1 - by0, bx1 - bx0), -1, np.int32)
        cx0, cy0, cx1, cy1 = max(bx0, 0), max(by0, 0), min(bx1, nb), min(by1, nb)
        if cx0 < cx1 and cy0 < cy1:
            grid[cy0 - by0:cy1 - by0, cx0 - bx0:cx1 - bx0] = \
                maps["block_index"][part, cy0:cy1, cx0:cx1]
        blocks = np.zeros(grid.shape + (B, B), np.uint8)
        blocks[grid == -2] = 1
        mixed = grid >= 0
        if mixed.any():
            blocks[mixed] = np.unpackbits(maps["bits"][grid[mixed]], axis=-1)
        full = blocks.transpose(0, 2, 1, 3).reshape(grid.shape[0] * B, grid.shape[1] * B)
        ox, oy = x0 - bx0 * B, y0 - by0 * B
        return full[oy:oy + h, ox:ox + w]


def patch_matrix(angle=0.0, flip=False, quarter_turns=None):
    """2x2 patch-to-source linear map. quarter_turns gives an exact lattice map."""
    if quarter_turns is not None:
        c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][quarter_turns % 4]
    else:
        c, s = math.cos(angle), math.sin(angle)
    A = np.array([[c, -s], [s, c]], np.float64)
    if flip:
        A = A @ np.diag([-1.0, 1.0])
    return A


def _patch_points(n, step):
    """Grid of n points spaced `step`, centered on the patch center (offsets from it)."""
    g = (np.arange(n, dtype=np.float64) + 0.5) * step - n * step / 2
    return np.meshgrid(g, g)  # (x, y), each (n, n) in patch coords minus center


def _to_source(center, A, n, step):
    px, py = _patch_points(n, step)
    sx = center[0] + A[0, 0] * px + A[0, 1] * py
    sy = center[1] + A[1, 0] * px + A[1, 1] * py
    return sx, sy


def _keys_weights(t):
    a = -0.5
    t1, t2 = 1 + t, 1 - t
    w0 = ((a * t1 - 5 * a) * t1 + 8 * a) * t1 - 4 * a
    w1 = ((a + 2) * t - (a + 3)) * t * t + 1
    w2 = ((a + 2) * t2 - (a + 3)) * t2 * t2 + 1
    w3 = 1 - w0 - w1 - w2
    return np.stack([w0, w1, w2, w3], -1)


def bicubic(image, sx, sy):
    """Sample a 2D array at continuous source coords (pixel centers at i+0.5)."""
    u, v = sx - 0.5, sy - 0.5
    iu, iv = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    wu, wv = _keys_weights(u - iu), _keys_weights(v - iv)
    x0, y0 = iu.min() - 1, iv.min() - 1
    x1, y1 = iu.max() + 3, iv.max() + 3
    H, W = image.shape
    if x0 >= 0 and y0 >= 0 and x1 <= W and y1 <= H:
        win = np.asarray(image[y0:y1, x0:x1], np.float64)
    else:   # outside the frame: edge-replicate (the inference padding convention)
        ys = np.clip(np.arange(y0, y1), 0, H - 1)
        xs = np.clip(np.arange(x0, x1), 0, W - 1)
        win = np.asarray(image[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1], np.float64)[
            ys - ys[0]][:, xs - xs[0]]
    # float32 with flat np.take gathers (4x less memory traffic than float64 2-D fancy
    # indexing; the data workers were memory-bandwidth bound). Lattice samples stay exact.
    win = np.ascontiguousarray(win, np.float32)
    Ww = win.shape[1]
    flat = win.ravel()
    base = (iv - y0 - 1) * Ww + (iu - x0 - 1)
    wu, wv = wu.astype(np.float32), wv.astype(np.float32)
    out = np.zeros(sx.shape, np.float32)
    for j in range(4):
        row = np.zeros(sx.shape, np.float32)
        for i in range(4):
            row += wu[..., i] * np.take(flat, base + (j * Ww + i))
        out += wv[..., j] * row
    return out


def sample_image(store, scene, center, A, context=0):
    """Input pixels for the 48 px patch plus `context` px on each side."""
    sx, sy = _to_source(center, A, PATCH + 2 * context, 1.0)
    return bicubic(store.images[scene], sx, sy).astype(np.float32)


def sample_mask_reference(store, part, center, A):
    """Direct per-pixel implementation of the mask convention (slow; for tests)."""
    sx, sy = _to_source(center, A, OUT, 1.0 / SCALE)
    mx = np.floor(sx * SCALE).astype(np.int64)
    my = np.floor(sy * SCALE).astype(np.int64)
    x0, y0 = int(mx.min()), int(my.min())
    win = store.mask_window(part, x0, y0, int(mx.max()) - x0 + 1, int(my.max()) - y0 + 1)
    return win[my - y0, mx - x0]


def sample_mask(store, part, center, A):
    """Same convention as sample_mask_reference, via an OpenCV nearest warp.

    Output pixel (a, b) samples hi-res index floor(16 s) = round(16 s - 0.5), and
    16 s - 0.5 is affine in (a, b); OpenCV's nearest warp rounds with 1/1024
    pixel fixed-point precision.
    """
    cx, cy = center[0] * SCALE, center[1] * SCALE
    h = OUT / 2 - 0.5
    ox = cx - 0.5 - (A[0, 0] + A[0, 1]) * h        # 16 s_x - 0.5 at (a, b) = (0, 0)
    oy = cy - 0.5 - (A[1, 0] + A[1, 1]) * h
    ext = (abs(A[0, 0]) + abs(A[0, 1])) * (OUT - 1), (abs(A[1, 0]) + abs(A[1, 1])) * (OUT - 1)
    x0 = int(math.floor(min(ox, ox + A[0, 0] * (OUT - 1), ox + A[0, 1] * (OUT - 1),
                            ox + (A[0, 0] + A[0, 1]) * (OUT - 1)))) - 2
    y0 = int(math.floor(min(oy, oy + A[1, 0] * (OUT - 1), oy + A[1, 1] * (OUT - 1),
                            oy + (A[1, 0] + A[1, 1]) * (OUT - 1)))) - 2
    w, hh = int(math.ceil(ext[0])) + 5, int(math.ceil(ext[1])) + 5
    win = np.ascontiguousarray(store.mask_window(part, x0, y0, w, hh))
    M = np.array([[A[0, 0], A[0, 1], ox - x0], [A[1, 0], A[1, 1], oy - y0]], np.float64)
    return cv2.warpAffine(win, M, (OUT, OUT), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def sample_patch(store, scene, center, A, context=0):
    """Raw (float32 16-bit units) (48+2c)^2 input and uint8 768x768 mask."""
    part = mask_of(store.scenes[scene])
    return sample_image(store, scene, center, A, context), sample_mask(store, part, center, A)


# ---------------------------------------------------------------------------
# Patch-geometry proposals

DEFAULT_SAMPLER = {
    # Mixture over per-part candidate categories (see cache.py)
    "weights": {"edge": 0.35, "dense": 0.2, "narrow": 0.15, "corner": 0.15,
                "sparse": 0.1, "uniform": 0.05},
    "jitter_px": 16.0,        # uniform offset of the center from the candidate pixel
    "focus_weights": None,    # optional {focus_class: relative scene weight}
    "p_lattice": 0.3,         # probability of an exact integer/dihedral patch
}


# Categories whose candidates depend on the scene's image, keyed by scene: "shift" = GT
# boundary where the naive visible edge is 0.75-8 px away (depth-of-field ramps and grey
# shoulders of thick parts, deep narrow bores; bench_v4.visible_shift).
SCENE_CATEGORIES = ("shift",)


def candidate_centers(store, scene, category):
    if category in SCENE_CATEGORIES:
        return store.scene_candidates(scene, category)
    return store.candidates(mask_of(store.scenes[scene]), category)


def eligible_scenes(store, scenes, categories):
    """Per category, the scenes whose part has at least one candidate center."""
    return {c: [i for i in scenes if len(candidate_centers(store, i, c))] for c in categories}


_SCENE_P = {}


def _scene_index(store, rng, eligible, cat, focus_weights):
    if not focus_weights:
        return rng.integers(len(eligible[cat]))
    # keyed by the eligible scenes themselves: ScenePool rebuilds `eligible` on every swap,
    # and a freed dict's id() can be reused by the next one
    key = (cat, tuple(eligible[cat]))
    if key not in _SCENE_P:
        if len(_SCENE_P) > 4096:
            _SCENE_P.clear()
        # focus_weights may name the scene field it keys on ("_key"; default focus_class,
        # which in the v4 cache is the blur class; camera_focus_class is the renderer's
        # focus class: sharp / moderate / depth_split / defocused / strong_defocus)
        field = focus_weights.get("_key", "focus_class")
        w = np.array([focus_weights.get(store.scenes[i].get(field), 1.0)
                      for i in eligible[cat]], np.float64)
        _SCENE_P[key] = np.cumsum(w / w.sum())
    return min(int(np.searchsorted(_SCENE_P[key], rng.random())), len(eligible[cat]) - 1)


def propose(store, rng, eligible, cfg):
    """Draw (scene, center, A, category) for one training patch.

    eligible: output of eligible_scenes(); the category is drawn first so the
    mixture weights hold even though some parts lack some categories.
    """
    names = [n for n in cfg["weights"] if cfg["weights"][n] > 0 and eligible.get(n)]
    p = np.array([cfg["weights"][n] for n in names], np.float64)
    cat = names[rng.choice(len(names), p=p / p.sum())]
    scene = eligible[cat][_scene_index(store, rng, eligible, cat, cfg.get("focus_weights"))]
    cands = candidate_centers(store, scene, cat)
    cx, cy = cands[rng.integers(len(cands))].astype(np.float64) + 0.5
    j = cfg["jitter_px"]
    cx += rng.uniform(-j, j)
    cy += rng.uniform(-j, j)
    m = store.meta["center_margin"] + cfg.get("context_px", 0)
    cx = float(np.clip(cx, m, store.W - m))
    cy = float(np.clip(cy, m, store.H - m))
    flip = bool(rng.integers(2))
    if rng.random() < cfg["p_lattice"]:
        center = (float(round(cx)), float(round(cy)))
        A = patch_matrix(flip=flip, quarter_turns=int(rng.integers(4)))
    else:
        center = (cx, cy)
        A = patch_matrix(angle=rng.uniform(0, 2 * math.pi), flip=flip)
    return scene, center, A, cat


# ---------------------------------------------------------------------------
# Input-only photometric postprocessing

DEFAULT_AUG = {
    "background_level": [0.35, 0.95],   # backlight level as a fraction of full scale
    "offset": [0.0, 0.03],              # added black offset, fraction of full scale
    "full_well_e": [2000.0, 40000.0],   # log-uniform electrons at full scale (shot noise)
    "read_noise": [0.0001, 0.002],      # log-uniform read noise std, fraction of full scale
    "bit_depths": [8, 10, 12, 16],
    # Exposure reference: "backlight" (the scene's backlight brightness, see
    # cache.backlight_level) or "p99.5" (image 99.5th percentile; reproduces val_v1).
    "exposure_ref": "backlight",
    "backlight_jitter": 0.03,           # log-normal sigma of error on the provided value
}

def photometric(raw, scene, rng, cfg):
    """Map raw 16-bit render values to a quantized [0, 1] sensor reading.

    scene: the scene record (needs background_level, backlight_level).
    Returns (reading, info), where info["backlight"] is the backlight brightness
    value given to the model, in the same transformed [0, 1] units as the reading.
    How that value is obtained at deployment (calibration, an estimator) is outside
    the model; backlight_jitter simulates its error during training.
    """
    backlight = scene.get("backlight_level", float("nan"))
    if cfg is None:
        return (raw / 65535.0).astype(np.float32), {"backlight": backlight / 65535.0}
    ref = scene["background_level"] if cfg.get("exposure_ref", "p99.5") == "p99.5" else backlight
    target = rng.uniform(*cfg["background_level"])
    gain = target / ref
    v = raw.astype(np.float64) * gain
    offset = rng.uniform(*cfg["offset"])
    v += offset
    fw = math.exp(rng.uniform(math.log(cfg["full_well_e"][0]), math.log(cfg["full_well_e"][1])))
    rn = math.exp(rng.uniform(math.log(cfg["read_noise"][0]), math.log(cfg["read_noise"][1])))
    sigma = np.sqrt(np.clip(v, 0, None) / fw + rn * rn)
    v = v + sigma * rng.standard_normal(v.shape)
    if cfg.get("p_sharpen") and rng.random() < cfg["p_sharpen"]:
        # ISP unsharp mask; symmetric, so a symmetric edge's 50% crossing does not move
        sg = rng.uniform(*cfg["sharpen_sigma"])
        v = v + rng.uniform(*cfg["sharpen_amount"]) * (v - cv2.GaussianBlur(v, (0, 0), sg))
    bits = int(cfg["bit_depths"][rng.integers(len(cfg["bit_depths"]))])
    q = (1 << bits) - 1
    out = (np.round(np.clip(v, 0.0, 1.0) * q) / q).astype(np.float32)
    if bits == 8 and cfg.get("p_jpeg") and rng.random() < cfg["p_jpeg"]:
        qual = int(rng.integers(cfg["jpeg_quality"][0], cfg["jpeg_quality"][1] + 1))
        _, enc = cv2.imencode(".jpg", (out * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, qual])
        out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    b_hat = backlight * gain + offset
    jitter = cfg.get("backlight_jitter", 0.0)
    if jitter:
        b_hat *= math.exp(jitter * rng.standard_normal())
    return out, {"backlight": float(b_hat), "exposure": target, "full_well_e": fw,
                 "read_noise": rn, "bits": bits, "gain_raw": float(gain * 65535.0),
                 "offset": float(offset)}


def warm_page_cache(cache_dir=CACHE_DIR, names=("block_index.i32", "bits.u8", "images.u16"),
                    chunk=64 << 20):
    """Sequentially read cache files so random patch reads hit RAM, not USB."""
    for name in names:
        with open(os.path.join(cache_dir, name), "rb", buffering=0) as f:
            buf = bytearray(chunk)
            while f.readinto(buf):
                pass
