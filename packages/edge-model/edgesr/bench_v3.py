"""val_v3_full: full-image benchmark with information-aware slices (PLAN_v2).

Spec (benchmarks/val_v3_full.json): 2 scenes per val part, each with a photometric
seed (default aug ranges, exposure normalized by the scene backlight, exact backlight
value supplied). GT = the part's exact hi-res boundary crossings.

Per-GT-point slice labels (cached in work/cache/bench_val_v3_full_labels.npz):
  blur       10-90% edge-spread width along the GT normal on the raw render:
             sharp < 1.6 px (sampling floor), medium 1.6-3 px, heavy > 3 px,
             undetermined (no flat plateaus within +-6 px)
  sparse     no other edge within 4 px (crossing count in a 4 px disk vs straight edge)
  corner     high local turning within 1.5 px
  specular   render within 2 px along the normal exceeds 1.05 x backlight
  lighting   scene illumination class;  snr  image-level low / mid / high

Usage:
  python -m edgesr.bench_v3 make
  python -m edgesr.bench_v3 labels
  python -m edgesr.bench_v3 eval-cfi CKPT        [--limit N]
  python -m edgesr.bench_v3 eval-patch CKPT      [--limit N]   (stride-32 blend reference)
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import CACHE_DIR, REPO_DIR, SCALE, WORK_DIR
from . import metrics as M
from . import sampler as S

NAME = os.environ.get("EDGESR_BENCH", "val_v3_full")      # or test_v3_full (--name)


def spec_path(name=None):
    return os.path.join(REPO_DIR, "benchmarks", (name or NAME) + ".json")


def labels_path(name=None):
    return os.path.join(CACHE_DIR, f"bench_{name or NAME}_labels.npz")


SPEC = spec_path()
LABELS = labels_path()
AUG = {**S.DEFAULT_AUG, "exposure_ref": "backlight", "backlight_jitter": 0.0}
# Calibrated 2026-09-26: rendered in-focus edges measure 1.1-1.6 px 10-90% (the pixel
# sampling floor of point-sampled steps through bicubic interpolation).
BLUR_BINS = [("sharp", 0, 1.6), ("medium", 1.6, 3.0), ("heavy", 3.0, 1e9)]


def make_spec(seed=31337, name=None, split="val"):
    store = S.SourceStore()
    rng = np.random.default_rng(seed)
    by_part = {}
    for i in store.scene_indices(split):
        by_part.setdefault(store.scenes[i]["part_index"], []).append(i)
    images = []
    for part in sorted(by_part):
        for sc in rng.choice(by_part[part], 2, replace=False):
            images.append({"scene_id": store.scenes[int(sc)]["scene_id"],
                           "aug_seed": int(rng.integers(2 ** 31))})
    spec = {"name": name or NAME, "split": split, "aug": AUG, "images": images}
    path = spec_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assert not os.path.exists(path), "benchmarks are immutable"
    with open(path, "w") as f:
        json.dump(spec, f, indent=1)
    print(f"{len(images)} images")


def load_spec(name=None):
    with open(spec_path(name)) as f:
        return json.load(f)


def reading(store, scene_idx, seed):
    raw = np.asarray(store.images[scene_idx], np.float32)
    x, info = S.photometric(raw, store.scenes[scene_idx], np.random.default_rng(seed), AUG)
    return x.astype(np.float32), float(info["backlight"]), info


# ---------------------------------------------------------------------------
# Slice labels

def _bicubic_points(img, px, py):
    return S.bicubic(img, px, py)


def _labels_one(args):
    k, scene_id, seed = args
    store = S.SourceStore()
    sc = {s["scene_id"]: i for i, s in enumerate(store.scenes)}[scene_id]
    meta = store.scenes[sc]
    part = meta["part_index"]
    gt = store.boundary_points(part).astype(np.float64)
    tree = cKDTree(gt)
    # local geometry: PCA of crossings within 0.75 px (tangent), and within 1.5 px (turning)
    n1 = tree.query_ball_point(gt, 0.75, return_length=True)
    n4 = tree.query_ball_point(gt, 4.0, return_length=True)
    sparse = n4 <= 1.25 * (4.0 / 0.75) * n1      # straight edge: chord 8 px vs 1.5 px
    # tangent/normal and turning from neighbours within 1.5 px (vectorized via kNN)
    kk = 48
    d, j = tree.query(gt, kk)
    nbr = gt[j]
    w = (d <= 1.5).astype(np.float64)
    wsum = np.maximum(w.sum(1, keepdims=True), 1)
    mu = (nbr * w[..., None]).sum(1) / wsum
    c = nbr - mu[:, None]
    cxx = (w * c[..., 0] ** 2).sum(1) / wsum[:, 0]
    cyy = (w * c[..., 1] ** 2).sum(1) / wsum[:, 0]
    cxy = (w * c[..., 0] * c[..., 1]).sum(1) / wsum[:, 0]
    tr, det = cxx + cyy, cxx * cyy - cxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    corner = (l2 / np.maximum(l1, 1e-12)) > 0.04
    ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy)                  # tangent direction
    nx, ny = -np.sin(ang), np.cos(ang)
    # ESF along the normal on the raw (noise-free-ish) render
    raw = np.asarray(store.images[sc], np.float32)
    t = np.arange(-6, 6.001, 0.25)
    blur = np.full(len(gt), np.nan)
    spec = np.zeros(len(gt), bool)
    bl = meta["backlight_level"]
    for s0 in range(0, len(gt), 20000):
        g = gt[s0:s0 + 20000]
        px = g[:, 0:1] + nx[s0:s0 + 20000, None] * t[None]
        py = g[:, 1:2] + ny[s0:s0 + 20000, None] * t[None]
        ok = (px.min(1) > 3) & (py.min(1) > 3) & (px.max(1) < store.W - 4) & (py.max(1) < store.H - 4)
        prof = np.full(px.shape, np.nan)
        prof[ok] = _bicubic_points(raw, px[ok], py[ok])
        lo = np.nanmedian(prof[:, :9], 1)                        # t in [-6, -4]
        hi = np.nanmedian(prof[:, -9:], 1)                       # t in [4, 6]
        contrast = hi - lo
        f = (prof - lo[:, None]) / np.where(np.abs(contrast) > 1e-6, contrast, np.nan)[:, None]
        # plateaus must be flat, else the profile crosses other structure: undetermined
        flat = (np.nanstd(prof[:, :9], 1) < 0.05 * np.abs(contrast)) & \
               (np.nanstd(prof[:, -9:], 1) < 0.05 * np.abs(contrast)) & \
               (np.abs(contrast) > 0.1 * bl)
        mono = np.clip(f, 0, 1)
        c10 = np.array([np.interp(0.1, np.maximum.accumulate(r), t) if np.isfinite(r).all() else np.nan
                        for r in mono])
        c90 = np.array([np.interp(0.9, np.maximum.accumulate(r), t) if np.isfinite(r).all() else np.nan
                        for r in mono])
        wdt = np.abs(c90 - c10)
        wdt[~flat] = np.nan
        blur[s0:s0 + 20000] = wdt
        near = np.abs(t) <= 2
        spec[s0:s0 + 20000] = np.nanmax(prof[:, near], 1) > 1.05 * bl
    return k, {"blur": blur.astype(np.float32), "sparse": sparse, "corner": corner,
               "specular": spec}


def build_labels(workers=6, name=None):
    spec = load_spec(name)
    out = {}
    with ProcessPoolExecutor(workers) as ex:
        for k, lab in ex.map(_labels_one, [(k, im["scene_id"], im["aug_seed"])
                                          for k, im in enumerate(spec["images"])]):
            for key, arr in lab.items():
                out[f"{k}/{key}"] = arr
            print(f"labels {k}: blur sharp/med/heavy/undet "
                  f"{np.mean(lab['blur'] < 1):.2f}/{np.mean((lab['blur'] >= 1) & (lab['blur'] < 3)):.2f}/"
                  f"{np.mean(lab['blur'] >= 3):.2f}/{np.mean(np.isnan(lab['blur'])):.2f} "
                  f"sparse {lab['sparse'].mean():.2f} corner {lab['corner'].mean():.3f} "
                  f"specular {lab['specular'].mean():.3f}", flush=True)
    np.savez_compressed(labels_path(name), **out)


def snr_class(info, backlight):
    shot = math.sqrt(max(backlight, 1e-6) / info["full_well_e"])
    sigma = math.sqrt(shot ** 2 + info["read_noise"] ** 2 + (1 / ((1 << info["bits"]) - 1)) ** 2 / 12)
    snr = backlight / sigma
    return "low" if snr < 60 else ("mid" if snr < 200 else "high")


# ---------------------------------------------------------------------------
# Scoring

def point_slices(labels, k, meta, info, backlight):
    blur = labels[f"{k}/blur"]
    b = np.full(len(blur), "undet", dtype=object)
    for name, lo, hi in BLUR_BINS:
        b[(blur >= lo) & (blur < hi)] = name
    return {
        "blur": b,
        "geom": np.where(labels[f"{k}/sparse"], "sparse", "dense"),
        "corner": np.where(labels[f"{k}/corner"], "corner", "smooth"),
        "specular": np.where(labels[f"{k}/specular"], "specular", "normal"),
        "lighting": np.full(len(blur), meta["illumination_class"], dtype=object),
        "snr": np.full(len(blur), snr_class(info, backlight), dtype=object),
    }


def score_image(gt, pred, workers=8):
    d_gp = cKDTree(pred).query(gt, workers=workers)[0] if len(pred) else np.full(len(gt), np.inf)
    if len(pred):
        d_pg, j = cKDTree(gt).query(pred, workers=workers)
    else:
        d_pg, j = np.zeros(0), np.zeros(0, int)
    return d_gp, d_pg, j


def summarize(d_gp, d_pg):
    s = {"n_gt": int(len(d_gp)), "n_pred": int(len(d_pg))}
    if len(d_gp):
        s["missed_rate"] = float((d_gp > M.MISS_PX).mean())
        inl = d_gp[d_gp <= M.MISS_PX]
        s["loc_error_mean"] = float(inl.mean()) if len(inl) else float("nan")
        s["loc_error_p95"] = float(np.percentile(inl, 95)) if len(inl) else float("nan")
        s["gt_to_pred_mean_clipped"] = float(np.minimum(d_gp, M.CLIP_PX).mean())
    if len(d_pg):
        s["extra_rate"] = float((d_pg > M.MISS_PX).mean())
        s["pred_to_gt_mean_clipped"] = float(np.minimum(d_pg, M.CLIP_PX).mean())
    if len(d_gp) and len(d_pg):
        s["bscore"] = 0.5 * (s["gt_to_pred_mean_clipped"] + s["pred_to_gt_mean_clipped"])
    return s


FAR_PX = 2.0
FRAME_PX = 8.0


def evaluate(predict_fn, tag, limit=None, out_path=None, name=None):
    """predict_fn(img, backlight) -> (pred_points (N,2) source px, seconds)."""
    store = S.SourceStore()
    spec = load_spec(name)
    labels = np.load(labels_path(name), allow_pickle=True)
    by_id = {s["scene_id"]: i for i, s in enumerate(store.scenes)}
    acc = {"gp": [], "pg": [], "sl_gp": {}, "sl_pg": {}, "far_frame": 0, "far_interior": 0,
           "n_pred": 0}
    times = []
    images = spec["images"][:limit] if limit else spec["images"]
    for k, im in enumerate(images):
        sc = by_id[im["scene_id"]]
        meta = store.scenes[sc]
        img, bl, info = reading(store, sc, im["aug_seed"])
        gt = store.boundary_points(meta["part_index"]).astype(np.float64)
        pred, sec = predict_fn(img, bl)
        times.append(sec)
        d_gp, d_pg, j = score_image(gt, pred)
        acc["gp"].append(d_gp)
        acc["pg"].append(d_pg)
        sl = point_slices(labels, k, meta, info, bl)
        # predicted points > FAR_PX from any GT edge are background hallucinations: they
        # are reported separately, not charged to the nearest GT point's slice
        far = d_pg > FAR_PX
        if len(pred):
            fr = np.minimum.reduce([pred[:, 0], pred[:, 1], img.shape[1] - pred[:, 0],
                                    img.shape[0] - pred[:, 1]]) <= FRAME_PX
            acc["far_frame"] += int((far & fr).sum())
            acc["far_interior"] += int((far & ~fr).sum())
        acc["n_pred"] += len(pred)
        for name, lab in sl.items():
            acc["sl_gp"].setdefault(name, []).append((lab, d_gp))
            lp = lab[j][~far] if len(j) else lab[:0]
            acc["sl_pg"].setdefault(name, []).append((lp, d_pg[~far]))
        print(f"{tag} image {k}: {len(pred)} pts, {sec:.2f}s, "
              f"missed {np.mean(d_gp > 0.5):.3f} extra {np.mean(d_pg > 0.5) if len(d_pg) else 0:.3f}",
              flush=True)
    res = {"tag": tag, "benchmark": name or NAME, "images": len(images),
           "seconds_per_image_median": float(np.median(times)),
           "overall": summarize(np.concatenate(acc["gp"]), np.concatenate(acc["pg"])),
           "far_extra": {"frame_frac_of_pred": acc["far_frame"] / max(acc["n_pred"], 1),
                         "interior_frac_of_pred": acc["far_interior"] / max(acc["n_pred"], 1),
                         "frame_points_per_image": acc["far_frame"] / len(images),
                         "interior_points_per_image": acc["far_interior"] / len(images)}}
    slices = {}
    for name in acc["sl_gp"]:
        lab_g = np.concatenate([l for l, _ in acc["sl_gp"][name]])
        dg = np.concatenate([d for _, d in acc["sl_gp"][name]])
        lab_p = np.concatenate([l for l, _ in acc["sl_pg"][name]])
        dp = np.concatenate([d for _, d in acc["sl_pg"][name]])
        slices[name] = {str(v): summarize(dg[lab_g == v], dp[lab_p == v]) for v in np.unique(lab_g)}
    # protected slices: sparse x {sharp, medium} x lighting
    lab = {n: np.concatenate([l for l, _ in acc["sl_gp"][n]]) for n in ("blur", "geom", "lighting")}
    labp = {n: np.concatenate([l for l, _ in acc["sl_pg"][n]]) for n in ("blur", "geom", "lighting")}
    dg = np.concatenate(acc["gp"])
    dp = np.concatenate([d[d <= FAR_PX] for d in acc["pg"]])
    prot = {}
    for blur in ("sharp", "medium"):
        for light in ["all"] + sorted(np.unique(lab["lighting"])):
            mg = (lab["geom"] == "sparse") & (lab["blur"] == blur)
            mp = (labp["geom"] == "sparse") & (labp["blur"] == blur)
            if light != "all":
                mg &= lab["lighting"] == light
                mp &= labp["lighting"] == light
            prot[f"sparse/{blur}/{light}"] = summarize(dg[mg], dp[mp])
    res["slices"] = slices
    res["protected"] = prot
    if out_path:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
    return res


def cfi_predictor(ckpt, device="cuda", pixel_band=-1.0, band_thresh=3.0, stats=None,
                  gpu_fraction=0.4, min_area=(50.0, 8.0), coarse=None):
    """GPU inference capped to a fraction of GPU memory (training may share the GPU);
    an image that still runs out of memory is redone on CPU. coarse: coarse model path
    for coarse-seed fine models (v4)."""
    from .cfi_infer import infer
    from .train_cfi import load_cfi
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_fraction)
    model = load_cfi(ckpt).to(device).eval()
    cpu_model = None
    cm = cm_cpu = None
    if coarse:
        from .coarse_infer import CoarseModel
        cm, cm_cpu = CoarseModel(coarse, device), CoarseModel(coarse, "cpu")

    def fn(img, bl):
        nonlocal cpu_model
        t = time.perf_counter()
        try:
            out = infer(model, img, bl, band=None, cell=16, device=device, contours=True,
                        pixel_band=pixel_band, band_thresh=band_thresh, min_area=min_area,
                        coarse_model=cm)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if cpu_model is None:
                cpu_model = load_cfi(ckpt).eval()
            out = infer(cpu_model, img, bl, band=None, cell=16, device="cpu", contours=True,
                        pixel_band=pixel_band, band_thresh=band_thresh, min_area=min_area,
                        coarse_model=cm_cpu)
        if stats is not None:
            stats.append({"cells": out["n_cells"], "band_px": out["n_band_px"]})
        return out["points"].double().cpu().numpy(), time.perf_counter() - t
    return fn


def patch_predictor(ckpt, device="cuda", chunk=256, stride=32):
    from .benchmark import torch_contour_points
    from .fullimage import predict_region
    from .train import load_model
    model = load_model(ckpt).to(device).eval().to(memory_format=torch.channels_last)

    def fn(img, bl):
        t = time.perf_counter()
        H, W = img.shape
        pts = []
        for y0 in range(0, H, chunk):
            for x0 in range(0, W, chunk):
                # 1 px overlap so crossings on chunk seams are found; keep points owned by
                # this chunk: x in [x0, x0+chunk), y in [y0, y0+chunk)
                w, h = min(chunk + 1, W - x0), min(chunk + 1, H - y0)
                lg = predict_region(model, img, bl, x0, y0, w, h, blend_stride=stride)
                p = torch_contour_points(torch.sigmoid(lg)) + np.array([x0, y0])
                own = (p[:, 0] < x0 + chunk) & (p[:, 1] < y0 + chunk)
                pts.append(p[own])
        return np.concatenate(pts), time.perf_counter() - t
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "labels", "eval-cfi", "eval-patch"])
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    ap.add_argument("--name", default=NAME, help="benchmark name (val_v3_full | test_v3_full)")
    ap.add_argument("--split", default="val", help="split for `make`")
    ap.add_argument("--band-thresh", type=float, default=3.0)
    ap.add_argument("--pixel-band", type=float, default=-1.0)
    ap.add_argument("--min-area", type=float, default=8.0,
                    help="min hole area (px^2); also object specks unless --min-object-area")
    ap.add_argument("--min-object-area", type=float, default=50.0,
                    help="min object-speck area (px^2), e.g. backlight dust; default 50")
    ap.add_argument("--coarse", help="coarse model for coarse-seed (v4) fine models")
    ap.add_argument("--tag", help="output name (default: the run directory name)")
    a = ap.parse_args()
    if a.cmd == "make":
        make_spec(name=a.name, split=a.split)
    elif a.cmd == "labels":
        build_labels(name=a.name)
    else:
        stats = []
        ma = a.min_area if a.min_object_area is None else (a.min_object_area, a.min_area)
        fn = (cfi_predictor(a.ckpt, pixel_band=a.pixel_band, band_thresh=a.band_thresh,
                            stats=stats, min_area=ma, coarse=a.coarse)
              if a.cmd == "eval-cfi" else patch_predictor(a.ckpt))
        name = a.tag or os.path.basename(os.path.dirname(a.ckpt))
        os.makedirs(os.path.join(WORK_DIR, "bench_v3"), exist_ok=True)
        suffix = "" if a.name == "val_v3_full" else f"_{a.name}"
        out = a.out or os.path.join(WORK_DIR, "bench_v3", f"{name}{suffix}.json")
        r = evaluate(fn, name, a.limit, None, name=a.name)
        if stats:
            r["operating_point"] = {"band_thresh": a.band_thresh, "pixel_band": a.pixel_band,
                                    "min_area": a.min_area, "min_object_area": a.min_object_area,
                                    "cells_per_image_median": float(np.median([x["cells"] for x in stats])),
                                    "band_px_per_image_median": float(np.median([x["band_px"] for x in stats]))}
        with open(out, "w") as f:
            json.dump(r, f, indent=1)
        print(json.dumps({"overall": r["overall"], "far_extra": r["far_extra"],
                          "protected": r["protected"]}, indent=1))


if __name__ == "__main__":
    main()
