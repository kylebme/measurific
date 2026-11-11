"""Classical seed for the fine model (PLAN_v2 arm B).

Mirrors `_background_mask` in measurement/src/inspection_measurement/adapter.py
(the reference implementation; its README may be out of date), dark-on-light:

  1. 12 x 12 tile grid, 90th-percentile levels; quadratic field fitted to tiles at or
     above the 40th percentile, then 4 rounds of two-sided 3-sigma trimming (MAD scale,
     floor 2 DN) -> field; clipped to [max(1 DN, 0.25 * p90), full scale]
  2. normalized = source / field * 230, Gaussian sigma 1.2 -> clean
  3. Otsu on uint8(clean) -> foreground = clean <= threshold (object mask)
  4. noise = MAD of (source - Gaussian_1(source)) over pixels with clean > threshold+15
Band (replacing PAE's candidate test, since PAE is not used): threshold contour
dilated by `search_radius`, union gradient magnitude of clean above
clamp(4 * noise, 4, 25) DN (the adapter's PAE threshold) dilated by 1 px.

Units: 8-bit DN equivalents as in the adapter (reading in [0, 1] x 255).
The per-scene field coefficients and Otsu level of the raw render are cached so
training crops reproduce, under the photometric gain/offset, what `seed_full`
computes on a full augmented image.

Usage: python -m edgesr.seed build-cache
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from . import CACHE_DIR

SEED_CACHE = os.path.join(CACHE_DIR, "seed_params.npz")
NORM = 230.0


def _design(x, y):
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)


def fit_field(source):
    """source: (H, W) in 8-bit DN units. Returns (coefficients over x/w, y/h, p90)."""
    h, w = source.shape
    samples, levels = [], []
    for ys in np.array_split(np.arange(h), min(12, h)):
        for xs in np.array_split(np.arange(w), min(12, w)):
            block = source[ys[0]:ys[-1] + 1:2, xs[0]:xs[-1] + 1:2]      # 2x-strided sample
            samples.append([float(xs.mean()) / w, float(ys.mean()) / h])
            levels.append(np.percentile(block, 90))
    xy, levels = np.asarray(samples), np.asarray(levels)
    design = _design(xy[:, 0], xy[:, 1])
    p90 = float(np.percentile(source[::4, ::4], 90))
    coef = np.array([p90, 0, 0, 0, 0, 0], np.float64)
    if len(levels) >= 12:
        keep = levels >= np.percentile(levels, 40)
        for _ in range(4):
            if keep.sum() < 6:
                break
            coef = np.linalg.lstsq(design[keep], levels[keep], rcond=None)[0]
            residual = levels - design @ coef
            scale = max(2., 1.4826 * np.median(np.abs(residual[keep] - np.median(residual[keep]))))
            keep = (residual > -3 * scale) & (residual < 3 * scale)
    return coef, p90


def eval_field(coef, p90, xs, ys, w, h):
    """Field at pixel-index coordinates (xs, ys) (adapter: x/w with integer pixel index)."""
    f = _design(xs / w, ys / h) @ coef
    return np.clip(f, max(1., p90 * .25), 255.)


def segment(source, field):
    """Adapter steps 2-4 given source and field (both 8-bit DN)."""
    normalized = np.clip(source / field * NORM, 0, 255)
    clean = cv2.GaussianBlur(normalized.astype(np.float32), (0, 0), 1.2)
    sample = np.rint(clean).astype(np.uint8).reshape(-1, 1)
    threshold, _ = cv2.threshold(sample, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return clean, float(threshold)


def noise_estimate(source, clean, threshold):
    residual = (source - cv2.GaussianBlur(source.astype(np.float32), (0, 0), 1.))[::4, ::4]
    bright = clean[::4, ::4] > threshold + 15
    r = residual[bright]
    return 1.4826 * float(np.median(np.abs(r - np.median(r)))) if len(r) else 0.


def band_from(clean, foreground, threshold, noise, search_radius=4, grad_radius=1):
    contour = foreground ^ cv2.erode(foreground, np.ones((3, 3), np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * search_radius + 1,) * 2)
    gx = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3)
    g_thresh = max(4., min(25., 4 * noise)) * 8          # Sobel(3) = 8 x per-pixel gradient
    gband = (cv2.magnitude(gx, gy) > g_thresh).astype(np.uint8)
    band = cv2.dilate(contour, k) | cv2.dilate(gband, np.ones((2 * grad_radius + 1,) * 2, np.uint8))
    return band.astype(bool), float(gband.mean())


def model_inputs(source, field, foreground):
    """3 fine-model input channels (float32, (3, H, W)) from 8-bit-DN arrays."""
    n = source / field                                   # ~1 on backlight, ~0 dark object
    return np.stack([(n - 0.5) * 2.0, foreground.astype(np.float32) * 2.0 - 1.0,
                     (field / 255.0 - 0.65) * 4.0]).astype(np.float32)


def eval_field_grid(coef, p90, w, h):
    """Field over the full pixel grid, evaluated separably (float32 broadcasting)."""
    x = (np.arange(w, dtype=np.float32) / w)[None, :]
    y = (np.arange(h, dtype=np.float32) / h)[:, None]
    c = coef.astype(np.float32)
    f = (c[0] + c[2] * y + c[5] * y * y) + (c[1] + c[4] * y) * x + c[3] * x * x
    return np.clip(f, max(1., p90 * .25), 255.).astype(np.float32)


def seed_full(img, search_radius=4, inputs=True):
    """Seed for a full sensor reading img (H, W) in [0, 1]. With inputs=False the
    3-channel model inputs are not materialized (sparse inference computes them per
    window from the returned source/field/mask via window_inputs)."""
    source = img.astype(np.float32) * 255.0
    h, w = source.shape
    coef, p90 = fit_field(source)
    field = eval_field_grid(coef, p90, w, h)
    clean, threshold = segment(source, field)
    foreground = (clean <= threshold).astype(np.uint8)
    noise = noise_estimate(source, clean, threshold)
    band, gfrac = band_from(clean, foreground, threshold, noise, search_radius)
    out = {"source": source, "field": field, "mask": foreground, "band": band,
           "threshold": threshold, "noise": noise, "grad_band_frac": gfrac, "coef": coef,
           "p90": p90}
    if inputs:
        out["inputs"] = model_inputs(source, field, foreground)
    return out


def window_inputs(source, field, mask):
    """Torch version of model_inputs for gathered windows (N, h, w) -> (N, 3, h, w)."""
    import torch
    n = source / field
    return torch.stack([(n - 0.5) * 2.0, mask.float() * 2.0 - 1.0,
                        (field / 255.0 - 0.65) * 4.0], 1)


# ---------------------------------------------------------------------------
# Coarse seed (fine model v4): the coarse model's part probability replaces the Otsu
# mask, the backlight field is fitted to the coarse model's confident background, and
# the band is the coarse band plus the gradient band (sub-resolution parts and loose
# contaminants that the coarse model leaves out). Same 3 model inputs as model_inputs,
# with the mask channel = 2 p - 1.

def fit_field_masked(source, background, min_frac=0.2):
    """Quadratic field fitted to background pixels only: 12 x 12 tiles, median of each
    tile's background pixels (tiles with >= min_frac background), then the adapter's
    two-sided 3-sigma trimming. Parts, however large or bright, do not bias it. Falls
    back to fit_field when fewer than 12 tiles have enough background."""
    h, w = source.shape
    samples, levels = [], []
    for ys in np.array_split(np.arange(h), min(12, h)):
        for xs in np.array_split(np.arange(w), min(12, w)):
            blk = source[ys[0]:ys[-1] + 1:2, xs[0]:xs[-1] + 1:2]
            bg = background[ys[0]:ys[-1] + 1:2, xs[0]:xs[-1] + 1:2]
            if bg.mean() < min_frac:
                continue
            samples.append([float(xs.mean()) / w, float(ys.mean()) / h])
            levels.append(float(np.median(blk[bg])))
    if len(levels) < 12:
        return fit_field(source)
    xy, levels = np.asarray(samples), np.asarray(levels)
    design = _design(xy[:, 0], xy[:, 1])
    keep = np.ones(len(levels), bool)
    coef = np.array([np.median(levels), 0, 0, 0, 0, 0], np.float64)
    for _ in range(4):
        if keep.sum() < 6:
            break
        coef = np.linalg.lstsq(design[keep], levels[keep], rcond=None)[0]
        residual = levels - design @ coef
        scale = max(2., 1.4826 * np.median(np.abs(residual[keep] - np.median(residual[keep]))))
        keep = (residual > -3 * scale) & (residual < 3 * scale)
    return coef, float(np.median(levels))


def coarse_background(part_logit512, shape, margin=2):
    """Confident background (logit < -2) at least `margin` model px from any coarse part,
    resized (nearest) from the 512 x 512 coarse frame to `shape`."""
    bg = (part_logit512 < -2).astype(np.uint8)
    if margin:
        bg = cv2.erode(bg, np.ones((2 * margin + 1,) * 2, np.uint8))
    H, W = shape
    return cv2.resize(bg, (W, H), interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)


def coarse_seed_full(img, coarse_model, valid=None, band_px=8.0, grad_band=True):
    """Coarse seed for a full reading img (H, W) in [0, 1]. Returns the same keys the
    classical seed_full uses for inference: source, field (8-bit DN units), mask (part
    probability, float32), fill (bool), band (bool), plus the coarse output."""
    source = img.astype(np.float32) * 255.0
    h, w = source.shape
    out = coarse_model(img, valid_mask=valid)
    logit = coarse_model.full_res_logit(out, img.shape)
    prob = (1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))).astype(np.float32)
    coef, lvl = fit_field_masked(source, coarse_background(out["part_logit"], img.shape))
    field = eval_field_grid(coef, lvl, w, h)
    band = coarse_model.band(out, img.shape, band_px)
    gfrac = 0.0
    if grad_band:
        clean, t = segment(source, field)
        noise = noise_estimate(source, clean, t)
        gx = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3)
        gband = (cv2.magnitude(gx, gy) > max(4., min(25., 4 * noise)) * 8).astype(np.uint8)
        gfrac = float(gband.mean())
        band = band | cv2.dilate(gband, np.ones((3, 3), np.uint8)).astype(bool)
    if valid is not None:
        band &= np.asarray(valid, bool)
    return {"source": source, "field": field, "mask": prob, "fill": prob > 0.5, "band": band,
            "coef": coef, "level": lvl, "grad_band_frac": gfrac, "coarse": out}


def coarse_crop_seed(x, src_xy, coarse_logit512, coef, level, gain_raw, offset, W, H, rng=None,
                     jitter_px=0.0, logit_scale=(1.0, 1.0)):
    """Training-crop inputs for the coarse seed. x: augmented crop reading (S, S) in
    [0, 1]; src_xy: (S*S, 2) continuous source coordinates of its pixel centers;
    coarse_logit512: the scene's 512 x 512 coarse part logit (hysteresis applied); coef,
    level: the scene's coarse-background field fitted on the raw render (fit_field_masked,
    8-bit DN of raw / 65535). reading = gain_raw * raw01 + offset. jitter_px shifts the
    coarse map (source px) and logit_scale scales its logit, both at random, for the
    difference between precomputed and inference-time coarse maps."""
    S = x.shape[0]
    f_raw = eval_field(coef, level, src_xy[:, 0] - 0.5, src_xy[:, 1] - 0.5, W, H).reshape(S, S)
    field = np.clip(gain_raw * f_raw + offset * 255.0, 1.0, 255.0).astype(np.float32)
    sx, sy = src_xy[:, 0], src_xy[:, 1]
    if rng is not None and jitter_px:
        sx = sx + rng.uniform(-jitter_px, jitter_px)
        sy = sy + rng.uniform(-jitter_px, jitter_px)
    # source px -> 512 model px (half-pixel centers, the cv2 INTER_LINEAR convention)
    mx = (sx * (coarse_logit512.shape[1] / W) - 0.5).astype(np.float32).reshape(S, S)
    my = (sy * (coarse_logit512.shape[0] / H) - 0.5).astype(np.float32).reshape(S, S)
    lg = cv2.remap(np.asarray(coarse_logit512, np.float32), mx, my, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    if rng is not None and logit_scale != (1.0, 1.0):
        lg = lg * float(np.exp(rng.uniform(np.log(logit_scale[0]), np.log(logit_scale[1]))))
    prob = 1.0 / (1.0 + np.exp(-np.clip(lg, -30, 30)))
    source = x.astype(np.float32) * 255.0
    return model_inputs(source, field, prob.astype(np.float32)), (prob > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-scene cache (raw renders) and crop reconstruction

def _scene_params(i):
    from .sampler import SourceStore
    store = SourceStore()
    source = np.asarray(store.images[i], np.float32) * (255.0 / 65535.0)
    h, w = source.shape
    coef, p90 = fit_field(source)
    field = eval_field_grid(coef, p90, w, h)
    clean, threshold = segment(source, field)
    return i, coef, p90, threshold


def build_cache(workers=8):
    from .sampler import SourceStore
    n = len(SourceStore().scenes)
    coefs, p90s, ts = np.zeros((n, 6)), np.zeros(n), np.zeros(n)
    with ProcessPoolExecutor(workers) as ex:
        for i, c, p, t in ex.map(_scene_params, range(n), chunksize=8):
            coefs[i], p90s[i], ts[i] = c, p, t
    np.savez(SEED_CACHE, coef=coefs, p90=p90s, t=ts)
    print(f"seed params for {n} scenes; Otsu level (DN of normalized*230) median "
          f"{np.median(ts):.1f} range {ts.min():.1f}-{ts.max():.1f}")


_PARAMS = None


def scene_params(scene):
    global _PARAMS
    if _PARAMS is None:
        z = np.load(SEED_CACHE)
        _PARAMS = (z["coef"], z["p90"], z["t"])
    return _PARAMS[0][scene], float(_PARAMS[1][scene]), float(_PARAMS[2][scene])


def crop_seed(x, src_xy, scene, gain_raw, offset, W, H):
    """Seed inputs for a training crop. x: augmented crop reading (S, S) in [0, 1];
    src_xy: (S*S, 2) continuous source coords of its pixel centers; the reading
    relates to the raw render as reading = gain_raw * raw01 + offset. The raw field
    and Otsu level (cached) are mapped through that affine change."""
    coef, p90, t_raw = scene_params(scene)
    S = x.shape[0]
    f_raw = eval_field(coef, p90, src_xy[:, 0] - 0.5, src_xy[:, 1] - 0.5, W, H).reshape(S, S)
    field = np.clip(gain_raw * f_raw + offset * 255.0, 1.0, 255.0).astype(np.float32)
    source = x.astype(np.float32) * 255.0
    normalized = np.clip(source / field * NORM, 0, 255)
    clean = cv2.GaussianBlur(normalized.astype(np.float32), (0, 0), 1.2)
    # A raw-normalized level t maps to (g * t/NORM * f + o) / (g * f + o) * NORM.
    fm = float(np.median(f_raw))
    t = (gain_raw * t_raw / NORM * fm + offset * 255.0) / (gain_raw * fm + offset * 255.0) * NORM
    foreground = (clean <= t).astype(np.uint8)
    return model_inputs(source, field, foreground), foreground


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-cache"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    build_cache(a.workers)


if __name__ == "__main__":
    main(sys.argv[1:])
