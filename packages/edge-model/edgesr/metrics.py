"""Contour-centric patch metrics, in source-pixel units.

The 0.5 iso-contour is represented by its marching-squares vertices: every
crossing of the 0.5 level along a horizontal or vertical segment between two
adjacent hi-res pixel centers, linearly interpolated. For a binary ground-truth
mask those are the midpoints between differing neighbors, i.e. the true
16x-mask boundary to within 1/32 source pixel. Boundary errors are nearest-point
distances between predicted and ground-truth contour points.
"""

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import SCALE

CLIP_PX = 1.0          # distance clip for the primary score
MISS_PX = 0.5          # a boundary point farther than this is missed / extra
FINE_PX = 0.1          # "accurate" threshold, reported as a secondary rate
BORDER_PX = 4          # source pixels excluded from each patch edge for scoring


def contour_points(p, level=0.5):
    """(N, 2) x,y contour points in source-pixel units for a (H, W) map."""
    p = np.asarray(p, np.float32)
    inside = p >= level
    pts = []
    # horizontal neighbor pairs
    a, b = p[:, :-1], p[:, 1:]
    ys, xs = np.nonzero(inside[:, :-1] != inside[:, 1:])
    if len(ys):
        pa, pb = a[ys, xs], b[ys, xs]
        t = np.clip((level - pa) / (pb - pa), 0.0, 1.0)
        pts.append(np.stack([xs + 0.5 + t, ys + 0.5], 1))
    a, b = p[:-1, :], p[1:, :]
    ys, xs = np.nonzero(inside[:-1, :] != inside[1:, :])
    if len(ys):
        pa, pb = a[ys, xs], b[ys, xs]
        t = np.clip((level - pa) / (pb - pa), 0.0, 1.0)
        pts.append(np.stack([xs + 0.5, ys + 0.5 + t], 1))
    if not pts:
        return np.zeros((0, 2), np.float64)
    return np.concatenate(pts).astype(np.float64) / SCALE


def _interior(pts, size_px, border):
    return np.all((pts >= border) & (pts <= size_px - border), axis=1)


def _component_misses(gt, pred, border_hr):
    """Count GT fg and bg connected components (touching the scored interior)
    that the prediction fails to reproduce (less than half of their area)."""
    out = {"components": 0, "missed_components": 0}
    for g, pr in ((gt, pred), (1 - gt, 1 - pred)):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8), connectivity=4)
        if n <= 1:
            continue
        inner = np.zeros_like(g, bool)
        inner[border_hr:-border_hr, border_hr:-border_hr] = True
        hit = np.bincount(lab[inner], minlength=n) > 0
        covered = np.bincount(lab.ravel(), weights=pr.ravel().astype(np.float64), minlength=n)
        for k in range(1, n):
            if not hit[k]:
                continue
            out["components"] += 1
            if covered[k] < 0.5 * stats[k, cv2.CC_STAT_AREA]:
                out["missed_components"] += 1
    return out


def patch_metrics(prob, gt, border=BORDER_PX):
    """Metrics for one patch. prob: (768, 768) float probabilities; gt: uint8 mask."""
    size = gt.shape[0] / SCALE
    gpts = contour_points(gt.astype(np.float32))
    ppts = contour_points(prob)
    gin, pin = gpts[_interior(gpts, size, border)], ppts[_interior(ppts, size, border)]
    if len(gin):
        d_gp = cKDTree(ppts).query(gin)[0] if len(ppts) else np.full(len(gin), np.inf)
    else:
        d_gp = np.zeros(0)
    if len(pin):
        d_pg = cKDTree(gpts).query(pin)[0] if len(gpts) else np.full(len(pin), np.inf)
    else:
        d_pg = np.zeros(0)
    b = int(border * SCALE)
    pb = prob[b:-b, b:-b]
    gb = gt[b:-b, b:-b].astype(bool)
    predb = pb >= 0.5
    eps = 1e-6
    bce = -np.mean(np.where(gb, np.log(np.clip(pb, eps, 1)), np.log(np.clip(1 - pb, eps, 1))))
    comp = _component_misses(gt.astype(np.uint8), (prob >= 0.5).astype(np.uint8), b)
    return {
        "d_gp": d_gp.astype(np.float32), "d_pg": d_pg.astype(np.float32),
        "inter": int((predb & gb).sum()), "union": int((predb | gb).sum()),
        "bce": float(bce), **comp,
    }


def aggregate(results, categories=None):
    """Pool per-patch metrics into a summary dict. categories: list aligned with results."""
    def summarize(rs):
        d_gp = np.concatenate([r["d_gp"] for r in rs]) if rs else np.zeros(0)
        d_pg = np.concatenate([r["d_pg"] for r in rs]) if rs else np.zeros(0)
        s = {"n_patches": len(rs), "n_gt_points": int(len(d_gp)), "n_pred_points": int(len(d_pg))}
        if len(d_gp):
            g = np.minimum(d_gp, CLIP_PX)
            s["gt_to_pred_mean_clipped"] = float(g.mean())
            finite = d_gp[np.isfinite(d_gp)]
            s["gt_to_pred_median"] = float(np.median(np.minimum(d_gp, 1e9)))
            s["gt_to_pred_p90"] = float(np.percentile(np.minimum(d_gp, 1e9), 90))
            s["gt_to_pred_p99"] = float(np.percentile(np.minimum(d_gp, 1e9), 99))
            s["missed_rate"] = float((d_gp > MISS_PX).mean())
            s["accurate_rate"] = float((d_gp <= FINE_PX).mean())
            inl = finite[finite <= MISS_PX]
            s["loc_error_mean"] = float(inl.mean()) if len(inl) else float("nan")
            s["loc_error_p95"] = float(np.percentile(inl, 95)) if len(inl) else float("nan")
        if len(d_pg):
            s["pred_to_gt_mean_clipped"] = float(np.minimum(d_pg, CLIP_PX).mean())
            s["extra_rate"] = float((d_pg > MISS_PX).mean())
        if len(d_gp) and len(d_pg):
            s["bscore"] = 0.5 * (s["gt_to_pred_mean_clipped"] + s["pred_to_gt_mean_clipped"])
        inter = sum(r["inter"] for r in rs)
        union = sum(r["union"] for r in rs)
        s["iou"] = inter / union if union else 1.0
        s["bce"] = float(np.mean([r["bce"] for r in rs])) if rs else float("nan")
        comps = sum(r["components"] for r in rs)
        s["component_miss_rate"] = sum(r["missed_components"] for r in rs) / comps if comps else 0.0
        return s

    out = summarize(results)
    if categories is not None:
        out["by_category"] = {
            c: summarize([r for r, k in zip(results, categories) if k == c])
            for c in sorted(set(categories))}
    return out
