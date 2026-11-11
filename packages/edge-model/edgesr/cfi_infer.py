"""Sparse full-image inference for CFI, with per-stage timing.

1. coarse branch on the whole image (1/4 res): context, coarse occupancy, band map
2. active cells (cell x cell source px) = cells containing band pixels
3. gather windows = cell + margin (>= fine receptive field), batched fine branch
4. decode the 16x16 sub-pixel lattice of band pixels
5. contours: marching-squares crossings on per-cell hi-res canvases (outside the band,
   occupancy comes from the coarse map)

Because every stage is convolutional with margin >= its receptive field, the result on
band pixels equals dense evaluation over the whole image.
"""

import math
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from .cfi import SCALE
from .regions import filter_small_regions_exact  # noqa: F401  (torch-free; re-exported)


@contextmanager
def _timed(times, key, device):
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    yield
    if device == "cuda":
        torch.cuda.synchronize()
    times[key] = times.get(key, 0.0) + time.perf_counter() - t


def band_from_coarse(band4, thresh=1.4, dilate_cells=1):
    """Source-res bool band from the 1/4-res band logits (conservative: whole cells)."""
    b = (band4[0, 0] > thresh).float()[None, None]
    if dilate_cells:
        k = 2 * dilate_cells + 1
        b = F.max_pool2d(b, k, 1, dilate_cells)
    return b[0, 0].bool().repeat_interleave(4, 0).repeat_interleave(4, 1)


FRAME_PAD = 32


@torch.no_grad()
def infer(model, img, backlight, band=None, cell=16, device="cpu", contours=True,
          pixel_band=None, compiled=None, frame_pad=FRAME_PAD, chunk=256, band_thresh=1.4,
          min_area=(50.0, 8.0), region_filter="exact", coarse_model=None, valid=None):
    """img: (H, W) float32 reading in [0,1]; backlight: float in the same units (learned
    coarse models only; seed-mode models ignore it and accept None).
    band: optional (H, W) bool pixels to decode (else from the coarse band head).
    pixel_band: None to decode every band pixel, or a threshold on the fine branch's
    per-pixel band logit (decode only pixels predicted near an edge).
    compiled: optional dict of torch.compile'd callables {"coarse", "fine"}.
    min_area: closed regions smaller than this many source px^2 are removed before contour
    extraction (the adapter's per-contour min_area). Scalar, or (min_object_area,
    min_hole_area): object specks (e.g. backlight dust) and holes are thresholded
    separately. Default (50, 8) px^2 (human decision 2026-09-26). 0 = off. region_filter: "exact" (hi-res components) | "fast"
    (quarter-pixel components + sub-pixel speck removal; misses loops within 1/4 px).
    coarse_model: a coarse_infer.CoarseModel for seed-mode models trained with the coarse
    seed (crop seed_inputs "coarse"): its part probability, coarse-background field and
    coarse + gradient band replace the classical seed; valid: optional ROI for it.
    Returns dict with timings (s) and outputs."""
    times = {}
    H0, W0 = img.shape
    csd = None
    if coarse_model is not None and getattr(model, "seed_mode", False):
        from .seed import coarse_seed_full
        with _timed(times, "coarse_seed", device):
            csd = coarse_seed_full(img, coarse_model, valid)       # on the unpadded frame
    if frame_pad:
        # Edge-replicate frame padding (the training convention for crops overhanging
        # the image); pad further to a multiple of 16. Points are shifted back and
        # clipped to the original image at the end.
        ph = (-(H0 + 2 * frame_pad)) % 16
        pw = (-(W0 + 2 * frame_pad)) % 16
        img = np.pad(img, ((frame_pad, frame_pad + ph), (frame_pad, frame_pad + pw)), mode="edge")
        if band is not None:
            band = np.pad(band, ((frame_pad, frame_pad + ph), (frame_pad, frame_pad + pw)))
    H, W = img.shape
    assert H % 16 == 0 and W % 16 == 0
    seed_mode = getattr(model, "seed_mode", False)
    if seed_mode:
        from .seed import seed_full
        with _timed(times, "seed", device):
            if csd is not None:                      # coarse seed, padded like the image
                pads = ((frame_pad, frame_pad + ph), (frame_pad, frame_pad + pw)) if frame_pad else ((0, 0), (0, 0))
                sd = {"source": np.pad(csd["source"], pads, mode="edge"),
                      "field": np.pad(csd["field"], pads, mode="edge"),
                      "mask": np.pad(csd["mask"], pads, mode="edge"),
                      "band": np.pad(csd["band"], pads)}
            else:
                sd = seed_full(img, inputs=False)    # classical seed on the padded image
            # (source, field, mask) stacked; model inputs are formed per window below
            xin = torch.from_numpy(np.stack([sd["source"], sd["field"],
                                             sd["mask"].astype(np.float32)]))[None].to(device)
            fill = torch.from_numpy((sd["mask"] > 0.5).astype(np.uint8)).to(device)    # object = 1
            if band is None:
                band = sd["band"]
        occ4 = band4 = ctx = None
    with _timed(times, "coarse", device):
        if seed_mode:        # the classical seed replaces the coarse branch (no backlight needed)
            pass
        elif compiled:
            from .cfi import _prep
            x = torch.from_numpy(img)[None, None].to(device)
            b = torch.tensor([backlight], device=device, dtype=torch.float32)
            xin = _prep(x, b)
            ctx, occ4, band4 = compiled["coarse"](xin)
        else:
            x = torch.from_numpy(img)[None, None].to(device)
            b = torch.tensor([backlight], device=device, dtype=torch.float32)
            xin, ctx, occ4, band4 = model.context(x, b)
    with _timed(times, "band", device):
        if band is None:
            bandt = band_from_coarse(band4, band_thresh)
        else:
            bandt = torch.from_numpy(band).to(device)
        cells = bandt.view(H // cell, cell, W // cell, cell).any(3).any(1)
        cy, cx = torch.nonzero(cells, as_tuple=True)
        n_cells = len(cy)
    R = model.radius
    M = int(math.ceil(R / 4) * 4)              # margin, multiple of 4 for ctx alignment
    win = cell + 2 * M
    with _timed(times, "gather", device):
        xin_p = F.pad(xin, (M, M, M, M), mode="replicate")[0]           # (C, H+2M, W+2M)
        ar = torch.arange(win, device=device)
        ys = (cy * cell)[:, None] + ar[None]                             # padded coords
        xs = (cx * cell)[:, None] + ar[None]
        xw = xin_p[:, ys[:, :, None], xs[:, None, :]].permute(1, 0, 2, 3)  # (N,C,win,win)
        if seed_mode:
            from .seed import window_inputs
            xw = window_inputs(xw[:, 0], xw[:, 1], xw[:, 2])
    with _timed(times, "gather", device):
        if not seed_mode:     # upsample the 1/4-res context for each window
            m4, w4 = M // 4 + 1, win // 4 + 2
            ctx_p = F.pad(ctx, (m4, m4, m4, m4), mode="replicate")[0]
            ar4 = torch.arange(w4, device=device)
            y4 = (cy * cell // 4)[:, None] + ar4[None]
            x4 = (cx * cell // 4)[:, None] + ar4[None]
            cw = ctx_p[:, y4[:, :, None], x4[:, None, :]].permute(1, 0, 2, 3)
            cw = F.interpolate(cw, scale_factor=4, mode="bilinear", align_corners=False)
            cw = cw[:, :, 4:4 + win, 4:4 + win]
    feats, pbands = [], []
    with _timed(times, "fine", device):
        for c0 in range(0, n_cells, chunk):
            fin = xw[c0:c0 + chunk] if seed_mode else torch.cat([xw[c0:c0 + chunk], cw[c0:c0 + chunk]], 1)
            if compiled:
                fin = fin.contiguous(memory_format=torch.channels_last)
                f, pb = compiled["fine"](fin)
            else:
                f, pb = model.fine(fin)
            feats.append(f[:, :, M - 1:M + cell + 1, M - 1:M + cell + 1].contiguous())
            pb = pb[:, :, M - 1:M + cell + 1, M - 1:M + cell + 1].float()
            pbands.append(F.max_pool2d(pb, 3, 1, 0)[:, 0])     # 1-px dilation -> (n, cell, cell)
        feat = torch.cat(feats) if feats else torch.zeros(0, model.cfg["fine_width"], cell + 2,
                                                            cell + 2, device=device)
        pband = torch.cat(pbands) if pbands else torch.zeros(0, cell, cell, device=device)
    with _timed(times, "decode", device):
        bw = bandt.view(H // cell, cell, W // cell, cell).permute(0, 2, 1, 3)[cy, cx]
        if pixel_band is not None:
            bw = bw & (pband > pixel_band)
        n, py, px = torch.nonzero(bw, as_tuple=True)
        logits = []
        P = model.project(feat) if model.cfg["decoder"] == "implicit" else None
        for s0 in range(0, len(n), 16384):
            sl = slice(s0, s0 + 16384)
            logits.append(model.decode_pixels(feat, n[sl], py[sl] + 1, px[sl] + 1, P,
                                              occupancy_only=True)[..., 0])
        logits = torch.cat(logits) if logits else torch.zeros(0, SCALE * SCALE, device=device)
    out = {"n_cells": n_cells, "n_band_px": int(len(n)), "window": win, "margin": M,
           "radius": R}
    if contours:
        with _timed(times, "contours", device):
            # canvases per active cell; outside-band pixels take the coarse occupancy sign
            # (seed mode: the threshold mask)
            if seed_mode:
                ar_c = torch.arange(cell, device=device)
                occ_c = fill[(cy * cell)[:, None, None] + ar_c[None, :, None],
                             (cx * cell)[:, None, None] + ar_c[None, None, :]].float() * 2 - 1
                canvas = occ_c.repeat_interleave(SCALE, 1).repeat_interleave(SCALE, 2) * 8.0
            else:
                c4 = occ4[0, 0]
                occ_c = c4[(cy * cell // 4)[:, None, None] + torch.arange(cell // 4, device=device)[None, :, None],
                           (cx * cell // 4)[:, None, None] + torch.arange(cell // 4, device=device)[None, None, :]]
                canvas = torch.sign(occ_c).repeat_interleave(4 * SCALE, 1).repeat_interleave(4 * SCALE, 2) * 8.0
            blocks = logits.view(-1, SCALE, SCALE)
            cv = canvas.view(n_cells, cell, SCALE, cell, SCALE).permute(0, 1, 3, 2, 4)
            cv[n, py, px] = blocks
            canvas = cv.permute(0, 1, 3, 2, 4).reshape(n_cells, cell * SCALE, cell * SCALE)
            ins = canvas >= 0
            if (np.any(np.asarray(min_area) > 0)) and n_cells:
                if region_filter == "exact":
                    filt = torch.from_numpy(filter_small_regions_exact(
                        ins.cpu().numpy(), cy.cpu().numpy(), cx.cpu().numpy(), cell, min_area)).to(device)
                else:
                    filt = filter_small_regions(ins, cy.cpu().numpy(), cx.cpu().numpy(), cell,
                                                min_area if np.isscalar(min_area) else min(min_area))
                changed = filt != ins
                # flipped hi-res pixels get a clear logit of the new sign
                canvas = torch.where(changed, torch.where(filt, 8.0, -8.0), canvas)
                ins = filt
            hx = torch.nonzero(ins[:, :, :-1] != ins[:, :, 1:])
            vx = torch.nonzero(ins[:, :-1, :] != ins[:, 1:, :])
            a = canvas[hx[:, 0], hx[:, 1], hx[:, 2]]
            bb = canvas[hx[:, 0], hx[:, 1], hx[:, 2] + 1]
            t = (-a / (bb - a)).clamp(0, 1)
            px_h = (hx[:, 2] + 0.5 + t) / SCALE + (cx[hx[:, 0]] * cell)
            py_h = (hx[:, 1] + 0.5) / SCALE + (cy[hx[:, 0]] * cell)
            a = canvas[vx[:, 0], vx[:, 1], vx[:, 2]]
            bb = canvas[vx[:, 0], vx[:, 1] + 1, vx[:, 2]]
            t = (-a / (bb - a)).clamp(0, 1)
            px_v = (vx[:, 2] + 0.5) / SCALE + (cx[vx[:, 0]] * cell)
            py_v = (vx[:, 1] + 0.5 + t) / SCALE + (cy[vx[:, 0]] * cell)
            pts = torch.cat([torch.stack([px_h, py_h], 1), torch.stack([px_v, py_v], 1)])
            if frame_pad:
                pts = pts - frame_pad
                keep = (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] <= W0) & (pts[:, 1] <= H0)
                pts = pts[keep]
        out["n_contour_pts"] = int(len(pts))
        out["points"] = pts
    out["logits"], out["band_index"] = logits, (n, py, px, cy, cx)
    out["times"] = times
    out["total_s"] = sum(times.values())
    return out


def filter_small_regions(ins, cy, cx, cell, min_area, scale=SCALE, level=4):
    """Fast min-area filter (used by infer). ins: (N, cell*s, cell*s) bool torch or numpy.
    1. Sub-pixel specks: a source pixel's s x s block whose border ring is uniform but whose
       interior contains the other class holds an isolated speck (< 1 px^2): flip it.
    2. Regions >= 1/level^2 px: label at `level`x resolution (area = summed class
       fraction, i.e. sub-pixel area) with the exact cross-cell merging of
       filter_small_regions_exact; flip removed regions' level-blocks at hi-res."""
    t = torch.as_tensor(ins)
    N, S, _ = t.shape
    c = S // scale
    blk = t.view(N, c, scale, c, scale).permute(0, 1, 3, 2, 4)            # (N,c,c,s,s)
    ring = torch.cat([blk[..., 0, :], blk[..., -1, :], blk[..., :, 0], blk[..., :, -1]], -1)
    ring_all = ring.all(-1)
    ring_none = ~ring.any(-1)
    uniform_ring = ring_all | ring_none
    mixed = blk.flatten(-2).any(-1) & ~blk.flatten(-2).all(-1)
    speck = uniform_ring & mixed
    if speck.any():
        blk = blk.clone()
        blk[speck] = ring_all[speck][:, None, None].expand(-1, scale, scale)
        t = blk.permute(0, 1, 3, 2, 4).reshape(N, S, S)
    f = scale // level
    frac = t.float().view(N, S // f, f, S // f, f).mean((2, 4))           # (N, S/f, S/f)
    coarse = (frac >= 0.5).cpu().numpy()
    fr = frac.cpu().numpy()
    keep = _filter_levels(coarse, fr, cy, cx, min_area * level * level)
    changed = torch.from_numpy(keep != coarse).to(t.device)
    if changed.any():
        up = changed.repeat_interleave(f, 1).repeat_interleave(f, 2)
        newv = torch.from_numpy(keep).to(t.device).repeat_interleave(f, 1).repeat_interleave(f, 2)
        t = torch.where(up, newv, t)
    return t


def _filter_levels(ins, frac, cy, cx, min_units):
    """Mosaic-labelled component filter on (N, S, S) bool with per-pixel class fraction
    `frac` (area of the True class); areas in level-px units. Returns filtered bool."""
    import cv2
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    N, S, _ = ins.shape
    cy, cx = np.asarray(cy, np.int64), np.asarray(cx, np.int64)
    gh, gw = int(cy.max()) + 2, int(cx.max()) + 2
    grid = np.full((gh + 1, gw + 1), -1, np.int64)
    grid[cy + 1, cx + 1] = np.arange(N)
    right, left = grid[cy + 1, cx + 2], grid[cy + 1, cx]
    down, up = grid[cy + 2, cx + 1], grid[cy, cx + 1]
    G = int(np.ceil(np.sqrt(N)))
    R = int(np.ceil(N / G))
    T = S + 1
    out = ins.copy()
    for value, conn in ((True, 8), (False, 4)):
        tiles = np.zeros((R * G, T, T), np.uint8)
        tiles[:N, :S, :S] = ins == value
        mosaic = tiles.reshape(R, G, T, T).transpose(0, 2, 1, 3).reshape(R * T, G * T)
        n, lab = cv2.connectedComponents(mosaic, connectivity=conn, ltype=cv2.CV_32S)
        if n <= 1:
            continue
        labels = lab.reshape(R, T, G, T).transpose(0, 2, 1, 3).reshape(R * G, T, T)[:N, :S, :S] - 1
        w = frac if value else 1.0 - frac
        sel = labels >= 0
        area = np.bincount(labels[sel], weights=w[sel], minlength=n - 1)
        a_idx, b_idx = [], []
        for nbr, mine, theirs in ((right, (slice(None), slice(None), -1), (slice(None), slice(None), 0)),
                                  (down, (slice(None), -1, slice(None)), (slice(None), 0, slice(None)))):
            has = nbr >= 0
            if not has.any():
                continue
            la, lb = labels[mine][has], labels[theirs][nbr[has]]
            pairs = [(la, lb)]
            if conn == 8:
                pairs += [(la[:, :-1], lb[:, 1:]), (la[:, 1:], lb[:, :-1])]
            for x, y in pairs:
                m = (x >= 0) & (y >= 0)
                a_idx.append(x[m])
                b_idx.append(y[m])
        open_ = np.zeros(n - 1, bool)
        for nbr, edge in ((left, (slice(None), 0)), (right, (slice(None), -1)),
                          (up, (0, slice(None))), (down, (-1, slice(None)))):
            e = labels[(slice(None),) + edge][nbr < 0].ravel()
            open_[e[e >= 0]] = True
        a_all = np.concatenate(a_idx) if a_idx else np.zeros(0, np.int64)
        b_all = np.concatenate(b_idx) if b_idx else np.zeros(0, np.int64)
        g = coo_matrix((np.ones(len(a_all)), (a_all, b_all)), shape=(n - 1, n - 1))
        _, comp = connected_components(g, directed=False)
        comp_area = np.bincount(comp, weights=area)
        comp_open = np.bincount(comp, weights=open_.astype(float)) > 0
        kill = ((comp_area < min_units) & ~comp_open)[comp]
        if kill.any():
            flip = np.zeros(labels.shape, bool)
            flip[sel] = kill[labels[sel]]
            out[flip] = not value
    return out
