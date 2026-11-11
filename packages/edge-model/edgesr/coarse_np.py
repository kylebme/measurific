"""Torch-free parts of the coarse model (edgesr.coarse): configuration, input
preparation, resizing, hysteresis, training targets and instance post-processing.
Used by the ONNX runtime (coarse_runtime) as well as by coarse.py, which re-exports
everything here."""

import cv2
import numpy as np

from . import seed as SD

DEFAULT_COARSE = {
    "widths": [16, 24, 32, 48, 64, 96],     # levels 1, 1/2, 1/4, 1/8, 1/16, 1/32 (macro px)
    "blocks": [0, 1, 2, 2, 2, 2],           # encoder blocks per level (after the downsample)
    "dec_blocks": [1, 1, 1, 1, 1],          # decoder blocks at levels 0..4
    "block": "ds",                          # "ds" inverted residual | "dense" 3x3+3x3
    "expand": 2,
    "dilations": [1, 2, 4],                 # cycled in the two deepest encoder levels
    "in_mode": "field",                     # "field": [reading/field, reading] | "raw"
    "contact_px": 2,
    "sdf_clip": 6.0,
    "aux_inputs": False,                    # + valid-ROI channel and log2(frame aspect) channel
    "instance_head": "contact",             # "contact" | "center" (adds heatmap + offsets)
    "center_sigma": 5.0,
    "offset_scale": 64.0,
}


def n_outputs(cfg):
    return 6 if cfg.get("instance_head") == "center" else 3


def in_channels(cfg):
    return (2 if cfg["in_mode"] == "field" else 1) + (2 if cfg.get("aux_inputs") else 0)


def resize_axes(a, w, h, nearest=False):
    """Resize (H, W) to (h, w) one axis at a time: area when shrinking an axis, bilinear
    when enlarging it (half-pixel centers either way); labels use center-aligned nearest."""
    if nearest:
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST_EXACT)
    H, W = a.shape
    if w != W:
        a = cv2.resize(a, (w, H), interpolation=cv2.INTER_AREA if w < W else cv2.INTER_LINEAR)
    if h != H:
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA if h < H else cv2.INTER_LINEAR)
    return a


def model_reading(img, size=512):
    """Full-resolution reading (H, W) in [0, 1] of any aspect ratio -> the (size, size)
    model input by area resampling (non-square frames are squashed). Model pixel (j, k)
    covers source pixels [k W/size, (k+1) W/size) x [j H/size, (j+1) H/size)."""
    return resize_axes(np.asarray(img, np.float32), size, size)


def macro_reading(img, factor=4):
    """Full-resolution reading (H, W) in [0, 1] -> area-downsampled macro reading."""
    h, w = img.shape
    a = img[:h // factor * factor, :w // factor * factor].astype(np.float32)
    # INTER_AREA at an integer factor is the exact block mean (and ~10x faster than numpy)
    return cv2.resize(a, (w // factor, h // factor), interpolation=cv2.INTER_AREA)


def coarse_inputs(reading, mode="field", aux=False, valid=None, aspect=1.0):
    """Macro reading (H, W) in [0, 1] -> network input (C, H, W) float32. With aux: a
    valid-region channel (+1 inside the region of interest, -1 outside; all +1 when there
    is none) and a constant log2(aspect) channel (aspect = width / height of the original
    frame before it was squashed to the square model input)."""
    r = reading.astype(np.float32)
    if mode == "raw":
        ch = [r * 2.0 - 1.0]
    else:
        src = r * 255.0
        h, w = src.shape
        coef, p90 = SD.fit_field(src)
        field = SD.eval_field_grid(coef, p90, w, h)
        n = np.clip(src / field, 0.0, 2.0)
        ch = [(n - 0.5) * 2.0, r * 2.0 - 1.0]
    if aux:
        v = np.ones_like(r) if valid is None else np.asarray(valid, np.float32)
        ch += [v * 2.0 - 1.0, np.full_like(r, np.log2(aspect))]
    return np.stack(ch).astype(np.float32)


def maps_from_output(out, cfg):
    m = {"part_logit": out[0], "part": 1 / (1 + np.exp(-out[0])),
         "contact": 1 / (1 + np.exp(-out[1])), "sdf": out[2] * cfg["sdf_clip"]}
    if out.shape[0] == 6:
        m["heat"] = out[3]
        m["offset"] = out[4:6] * cfg["offset_scale"]
    return m


MIN_PEAK_LOGIT = 3.0        # hysteresis: a part component needs one pixel at logit >= this


def suppress_weak(part_logit, min_peak=MIN_PEAK_LOGIT):
    """Hysteresis on the part map: connected components (logit > 0) whose peak logit is
    below min_peak are made background (their logit is negated). Faint backlight mottling
    on real images comes out as such low-confidence blobs, real parts peak far higher."""
    if not min_peak:
        return part_logit
    fg = (part_logit > 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(fg, connectivity=8)
    if n <= 1:
        return part_logit
    peak = np.full(n, -np.inf, np.float32)
    np.maximum.at(peak, lab.ravel(), part_logit.ravel().astype(np.float32))
    weak = peak < min_peak
    weak[0] = False
    return np.where(weak[lab], -np.abs(part_logit) - 0.01, part_logit).astype(np.float32)


def postprocess(m, min_peak=MIN_PEAK_LOGIT):
    """Maps -> dict for scoring/consumers: part logit (weak components suppressed),
    contact and instance labels."""
    m = dict(m)
    m["part_logit"] = suppress_weak(m["part_logit"], min_peak)
    m["part"] = 1 / (1 + np.exp(-m["part_logit"]))
    if "heat" in m:
        lab = instances_center(m["part"], m["heat"], m["offset"])
    else:
        lab = instances(m["part"], m["contact"])
    return {"part_logit": m["part_logit"], "contact": m["contact"], "instances": lab}


# ---------------------------------------------------------------------------
# targets and post-processing

def part_centers(inst):
    """Mass center (x, y) in pixel-index coordinates of every instance id > 0."""
    h, w = inst.shape
    ids = inst.ravel().astype(np.int64)
    n = int(ids.max()) + 1
    cnt = np.bincount(ids, minlength=n).astype(np.float64)
    sx = np.bincount(ids, weights=np.tile(np.arange(w, dtype=np.float64), h), minlength=n)
    sy = np.bincount(ids, weights=np.repeat(np.arange(h, dtype=np.float64), w), minlength=n)
    return {k: (sx[k] / cnt[k], sy[k] / cnt[k]) for k in range(1, n) if cnt[k] > 0}


def targets(inst, contact_px=2, sdf_clip=6.0, centers=None, sigma=5.0, offset_scale=64.0):
    """inst: (H, W) uint8 instance labels (0 background) -> part, contact, sdf (float32)
    and, if centers {id: (x, y)} (crop coordinates, possibly outside the crop) are given,
    center heat and offsets / offset_scale."""
    import cv2
    part = inst > 0
    k = np.ones((2 * contact_px + 1,) * 2, np.uint8)
    hi = cv2.dilate(inst, k)
    lo = cv2.erode(np.where(part, inst, 255).astype(np.uint8), k)
    contact = part & ((hi != inst) | (lo != inst))
    p8 = part.astype(np.uint8)
    d_in = cv2.distanceTransform(p8, cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform(1 - p8, cv2.DIST_L2, 5)
    sdf = np.where(part, d_in - 0.5, 0.5 - d_out)
    out = [part.astype(np.float32), contact.astype(np.float32),
           np.clip(sdf, -sdf_clip, sdf_clip).astype(np.float32)]
    if centers is not None:
        h, w = inst.shape
        heat = np.zeros((h, w), np.float32)
        cx = np.zeros(256, np.float32)
        cy = np.zeros(256, np.float32)
        r = int(np.ceil(3 * sigma))
        for pid, (x, y) in centers.items():
            cx[pid], cy[pid] = x, y
            x0, x1 = max(int(x) - r, 0), min(int(x) + r + 2, w)
            y0, y1 = max(int(y) - r, 0), min(int(y) + r + 2, h)
            if x0 >= x1 or y0 >= y1:
                continue
            gx = np.exp(-(np.arange(x0, x1) - x) ** 2 / (2 * sigma ** 2))
            gy = np.exp(-(np.arange(y0, y1) - y) ** 2 / (2 * sigma ** 2))
            np.maximum(heat[y0:y1, x0:x1], np.outer(gy, gx).astype(np.float32), out=heat[y0:y1, x0:x1])
        xs = np.arange(w, dtype=np.float32)[None, :]
        ys = np.arange(h, dtype=np.float32)[:, None]
        offx = np.where(part, cx[inst] - xs, 0) / offset_scale
        offy = np.where(part, cy[inst] - ys, 0) / offset_scale
        out += [heat, offx.astype(np.float32), offy.astype(np.float32)]
    return out


def instances_center(part_p, heat, offset, thresh=0.5, heat_thresh=0.1, nms=7, top_k=64):
    """Part pixels grouped by the nearest detected center (heatmap peaks after max-pool
    NMS), using each pixel's predicted offset. Returns int32 labels (0 background)."""
    import cv2
    fg = part_p > thresh
    peaks = (heat >= cv2.dilate(heat, np.ones((nms, nms), np.uint8))) & (heat > heat_thresh)
    py, px = np.nonzero(peaks)
    lab = np.zeros(fg.shape, np.int32)
    if not len(py):
        n, cc = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
        return cc.astype(np.int32)
    order = np.argsort(-heat[py, px])[:top_k]
    py, px = py[order].astype(np.float32), px[order].astype(np.float32)
    ys, xs = np.nonzero(fg)
    tx = xs + offset[0][ys, xs]
    ty = ys + offset[1][ys, xs]
    best = np.full(len(ys), np.inf, np.float32)
    arg = np.zeros(len(ys), np.int32)
    for k in range(len(py)):
        d = (tx - px[k]) ** 2 + (ty - py[k]) ** 2
        better = d < best
        best[better] = d[better]
        arg[better] = k + 1
    lab[ys, xs] = arg
    ids = np.unique(lab)
    ids = ids[ids > 0]
    lut = np.zeros(lab.max() + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[lab]


def instances(part_p, contact_p, thresh=0.5, min_core=4, max_iter=64):
    """Part mask split into instances: cores = part & ~contact, connected (8-conn);
    remaining part pixels join the geodesically nearest core; part components without a
    core are instances of their own. Returns int32 labels (0 background)."""
    import cv2
    fg = (part_p > thresh).astype(np.uint8)
    core = (fg & (contact_p <= thresh)).astype(np.uint8)
    n, lab = cv2.connectedComponents(core, connectivity=8)
    lab = lab.astype(np.int32)
    if n > 1:
        sizes = np.bincount(lab.ravel(), minlength=n)
        small = sizes < min_core
        small[0] = False
        lab[small[lab]] = 0
    k = np.ones((3, 3), np.uint8)
    for _ in range(max_iter):
        todo = (fg > 0) & (lab == 0)
        if not todo.any():
            break
        grown = cv2.dilate(lab.astype(np.float32), k).astype(np.int32)
        new = todo & (grown > 0)
        if not new.any():
            break
        lab[new] = grown[new]
    rest = ((fg > 0) & (lab == 0)).astype(np.uint8)
    if rest.any():
        m, rl = cv2.connectedComponents(rest, connectivity=8)
        off = lab.max()
        lab[rl > 0] = rl[rl > 0] + off
    # relabel consecutively
    ids = np.unique(lab)
    ids = ids[ids > 0]
    lut = np.zeros(lab.max() + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[lab]
