"""val_v4_full: full-image fine-model benchmark on the physical_v3 renders (DATA_V3.md).

Spec (benchmarks/val_v4_full.json, immutable): 4 scenes per val part (120 images), each
with a photometric seed. Sensor: DEFAULT_AUG with full well 6k-60k e- (DATA_V3), no
sharpening or JPEG, exact backlight value. GT = the scene's exact 16x boundary crossings
(tilted and contaminated scenes have their own masks).

Metrics (M.MISS_PX = 0.5 px): missed / extra / loc / bscore as bench_v3, and the
**signed offset** of matched predicted points (+ = outside the object, i.e. the outward
bias DATA_V3 calls the headline defect), mean and median. Far extras (> 2 px from any GT
edge) are split frame / interior and not charged to slices.
Image slices: blur class, tilt, contamination level, lighting, backlight type, glare
(none / < 2% / >= 2%), family. Point slice: contaminant boundary (GT crossings within
4 px of where the scene mask differs from its part's base mask) vs part boundary.
Point slice `edge_shift` (added 2026-09-30, part boundaries only): GT crossings binned by
their distance to the naive visible edge (`visible_shift`): aligned <= 0.5 px, 0.5-2,
2-8, none within 8 px. Shifted edges are depth-of-field ramps / grey shoulders of thick
parts and deep narrow bores, the human's priority failure mode. Reports missed, absent
(> 2 px) and the signed GT -> prediction offset of found edges (+ = outside).
`winding` (added with PLAN_v3): predicted / GT crossings within 2 px of each other; > 1
means doubled or wiggly contours. Far extras are also split by lighting class.
SDFNet checkpoints (config model.kind "sdf") are scored through sdf_infer.SDFPredictor.

Usage:
  python -m edgesr.bench_v4 make [--name val_v4_full --split val]
  python -m edgesr.bench_v4 score RUN_DIR/final.pt [--coarse models/coarse_v2.pt] [--limit N]
  python -m edgesr.bench_v4 score RUN_DIR/final.pt --classical      (classical-seed models)
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import REPO_DIR, SCALE, WORK_DIR
from . import metrics as M
from . import sampler as S
from .bench_v3 import FAR_PX, FRAME_PX, summarize

AUG = {**S.DEFAULT_AUG, "full_well_e": [6000.0, 60000.0], "exposure_ref": "backlight",
       "backlight_jitter": 0.0}
OUT_DIR = os.path.join(WORK_DIR, "bench_v4")


def spec_path(name):
    return os.path.join(REPO_DIR, "benchmarks", name + ".json")


def make_spec(name="val_v4_full", split="val", per_part=4, seed=40404):
    store = S.open_store({"cache": "v4"})
    rng = np.random.default_rng(seed)
    by_part = {}
    for i in store.scene_indices(split):
        by_part.setdefault(store.scenes[i]["part_index"], []).append(i)
    images = []
    for part in sorted(by_part):
        for sc in rng.choice(by_part[part], min(per_part, len(by_part[part])), replace=False):
            images.append({"scene_id": store.scenes[int(sc)]["scene_id"], "aug_seed": int(rng.integers(2 ** 31))})
    path = spec_path(name)
    assert not os.path.exists(path), "benchmarks are immutable"
    with open(path, "w") as f:
        json.dump({"name": name, "split": split, "cache": "v4", "aug": AUG, "images": images}, f, indent=1)
    print(f"{name}: {len(images)} images from {len(by_part)} parts")


def reading(store, sc, seed):
    raw = np.asarray(store.images[sc], np.float32)
    x, info = S.photometric(raw, store.scenes[sc], np.random.default_rng(seed), AUG)
    return x.astype(np.float32), float(info["backlight"]), info


def glare_bin(g):
    return "none" if g <= 0 else "<2%" if g < 0.02 else ">=2%"


IMAGE_SLICES = {"blur": lambda r: r["blur_class"], "tilt": lambda r: "tilted" if r["tilted"] else "flat",
                "contamination": lambda r: r["contamination"], "lighting": lambda r: r["illumination_class"],
                "backlight": lambda r: r["backlight_type"], "glare": lambda r: glare_bin(r["glare"]),
                "family": lambda r: r["family"]}


def contaminant_points(store, sc, gt):
    """GT crossings within 4 px of where the scene mask differs from its base mask."""
    rec = store.scenes[sc]
    if rec.get("mask_plan") != "base_plus_contaminants":
        return np.zeros(len(gt), bool)
    if not hasattr(store, "_base_mask"):
        store._base_mask = {s["mask_key"]: s["mask_index"] for s in store.scenes}
    base = store._base_mask.get(rec["part_id"])
    if base is None:
        return np.zeros(len(gt), bool)
    import cv2
    diff = (np.asarray(store.cover[rec["mask_index"]]) != np.asarray(store.cover[base])).astype(np.uint8)
    diff = cv2.dilate(diff, np.ones((9, 9), np.uint8)) > 0
    ix = np.clip(gt.astype(int), 0, [store.W - 1, store.H - 1])
    return diff[ix[:, 1], ix[:, 0]]


SHIFT_BINS = ((0.0, 0.5, "aligned"), (0.5, 2.0, "shift_0.5-2"), (2.0, 8.0, "shift_2-8"),
              (8.0, np.inf, "no_visible_edge"))


def visible_shift(store, sc, gt):
    """Distance from each GT crossing to the naive visible edge: the 0.5 level of the clean
    render normalized between the scene's dark level and its fitted backlight field
    (linear crossings between pixel centres, render lightly smoothed). Depth-of-field
    edges of thick parts (ramps, grey shoulders) and deep narrow bores put the geometric
    edge away from it; a fine model must beat this naive detector exactly there."""
    import cv2
    from .seed import eval_field_grid
    rec = store.scenes[sc]
    raw = cv2.GaussianBlur(np.asarray(store.images[sc], np.float32) / 257.0, (0, 0), 0.6)
    field = eval_field_grid(np.asarray(rec["field_coef"]), rec["field_level"], store.W, store.H)
    lo = rec["object_level"] / 257.0
    s = (raw - lo) / np.maximum(field - lo, 1.0) - 0.5
    pts = []
    for axis in (1, 0):
        a, b = (s[:, :-1], s[:, 1:]) if axis == 1 else (s[:-1], s[1:])
        i, j = np.nonzero((a > 0) != (b > 0))
        t = a[i, j] / (a[i, j] - b[i, j])
        pts.append(np.stack([j + 0.5 + t, i + 0.5], 1) if axis == 1 else np.stack([j + 0.5, i + 0.5 + t], 1))
    vis = np.concatenate(pts)
    return cKDTree(vis).query(gt, workers=8, distance_upper_bound=64.0)[0] if len(vis) else np.full(len(gt), np.inf)


def shift_summary(d_gp, signed_gp):
    """Per-GT-point view of a shift bin: missed (> 0.5 px), absent (> 2 px), and the signed
    GT -> prediction offset of found edges (<= 2 px; + = prediction outside the object)."""
    s = {"n_gt": int(len(d_gp))}
    if len(d_gp):
        found = d_gp <= 2.0
        s.update(missed_rate=float((d_gp > M.MISS_PX).mean()), absent_rate=float((~found).mean()),
                 gt_to_pred_mean_clipped=float(np.minimum(d_gp, M.CLIP_PX).mean()))
        if found.any():
            s.update(signed_mean=float(signed_gp[found].mean()), signed_median=float(np.median(signed_gp[found])),
                     abs_mean=float(np.abs(signed_gp[found]).mean()))
    return s


def signed_offsets(store, mask, pred, d_pg):
    """Signed distance of matched predicted points: + outside the object."""
    sel = d_pg <= M.MISS_PX
    p = pred[sel]
    inside = store.mask_points(mask, np.floor(p[:, 0] * SCALE), np.floor(p[:, 1] * SCALE)) > 0
    return np.where(inside, -d_pg[sel], d_pg[sel])


def with_offsets(s, signed):
    if len(signed):
        s["signed_mean"] = float(np.mean(signed))
        s["signed_median"] = float(np.median(signed))
    return s


def evaluate(predict_fn, tag, name="val_v4_full", limit=None):
    """predict_fn(img, backlight) -> (pred_points (N, 2) source px, seconds)."""
    store = S.open_store({"cache": "v4"})
    spec = json.load(open(spec_path(name)))
    by_id = {s["scene_id"]: i for i, s in enumerate(store.scenes)}
    images = spec["images"][:limit] if limit else spec["images"]
    acc = {"gp": [], "pg": [], "sg": [], "far_frame": 0, "far_interior": 0, "n_pred": 0}
    groups = {k: {} for k in IMAGE_SLICES}
    cont = {"contaminant": ([], []), "part": ([], [])}
    shift = {name: ([], []) for _, _, name in SHIFT_BINS}
    times = []
    for k, im in enumerate(images):
        sc = by_id[im["scene_id"]]
        rec = store.scenes[sc]
        img, bl, info = reading(store, sc, im["aug_seed"])
        gt = store.boundary_points(rec["mask_index"]).astype(np.float64)
        pred, sec = predict_fn(img, bl)
        times.append(sec)
        d_gp, j_gp = cKDTree(pred).query(gt, workers=8) if len(pred) else (np.full(len(gt), np.inf), None)
        d_pg, j = (cKDTree(gt).query(pred, workers=8) if len(pred) else (np.zeros(0), np.zeros(0, int)))
        far = d_pg > FAR_PX
        if len(pred):
            fr = np.minimum.reduce([pred[:, 0], pred[:, 1], img.shape[1] - pred[:, 0],
                                    img.shape[0] - pred[:, 1]]) <= FRAME_PX
            acc["far_frame"] += int((far & fr).sum())
            acc["far_interior"] += int((far & ~fr).sum())
            lt = acc.setdefault("far_by_lighting", {}).setdefault(rec["illumination_class"], [0, 0])
            lt[0] += int((far & ~fr).sum())
            lt[1] += 1
        acc["n_pred"] += len(pred)
        # winding: predicted vs GT crossings near each other (> 1 = doubled or wiggly contours)
        acc["pred_near"] = acc.get("pred_near", 0) + int((d_pg <= FAR_PX).sum())
        acc["gt_near"] = acc.get("gt_near", 0) + int((d_gp <= FAR_PX).sum())
        sg = signed_offsets(store, rec["mask_index"], pred, d_pg) if len(pred) else np.zeros(0)
        acc["gp"].append(d_gp)
        acc["pg"].append(d_pg[~far])
        acc["sg"].append(sg)
        for g, f in IMAGE_SLICES.items():
            b = groups[g].setdefault(str(f(rec)), ([], [], []))
            b[0].append(d_gp)
            b[1].append(d_pg[~far])
            b[2].append(sg)
        cm = contaminant_points(store, sc, gt)
        cont["contaminant"][0].append(d_gp[cm])
        cont["part"][0].append(d_gp[~cm])
        # depth-shift slice (part boundaries only): GT -> prediction, signed + outside
        signed_gp = d_gp.copy()
        if len(pred):
            q = pred[j_gp]
            signed_gp[store.mask_points(rec["mask_index"], np.floor(q[:, 0] * SCALE), np.floor(q[:, 1] * SCALE)) > 0] *= -1
        dv = visible_shift(store, sc, gt)
        for lo, hi, name in SHIFT_BINS:
            sel = ~cm & (dv > lo) & (dv <= hi) if lo > 0 else ~cm & (dv <= hi)
            shift[name][0].append(d_gp[sel])
            shift[name][1].append(signed_gp[sel])
        if len(pred):
            pc = cm[j]
            cont["contaminant"][1].append(d_pg[pc & ~far])
            cont["part"][1].append(d_pg[~pc & ~far])
        print(f"{tag} {k}: {len(pred)} pts {sec:.2f}s missed {np.mean(d_gp > 0.5):.3f} "
              f"extra {np.mean(d_pg > 0.5) if len(d_pg) else 0:.3f} "
              f"signed {np.mean(sg) if len(sg) else float('nan'):+.3f}", flush=True)
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)      # noqa: E731
    res = {"tag": tag, "benchmark": name, "images": len(images),
           "seconds_per_image_median": float(np.median(times)),
           "overall": with_offsets(summarize(cat(acc["gp"]), cat(acc["pg"])), cat(acc["sg"])),
           "far_extra": {"frame_points_per_image": acc["far_frame"] / len(images),
                         "interior_points_per_image": acc["far_interior"] / len(images),
                         "frame_frac_of_pred": acc["far_frame"] / max(acc["n_pred"], 1),
                         "interior_frac_of_pred": acc["far_interior"] / max(acc["n_pred"], 1),
                         "interior_per_image_by_lighting": {k: v[0] / max(v[1], 1) for k, v in
                                                            sorted(acc.get("far_by_lighting", {}).items())}},
           "winding": acc.get("pred_near", 0) / max(acc.get("gt_near", 0), 1)}
    res["slices"] = {g: {v: with_offsets(summarize(cat(b[0]), cat(b[1])), cat(b[2])) for v, b in sorted(d.items())}
                     for g, d in groups.items()}
    res["slices"]["boundary"] = {k: summarize(cat(v[0]), cat(v[1])) for k, v in cont.items()}
    res["slices"]["edge_shift"] = {k: shift_summary(cat(v[0]), cat(v[1])) for k, v in shift.items()}
    return res


def cfi_predictor(ckpt, coarse=None, device="cuda", pixel_band=-1.0, gpu_fraction=0.4, min_area=(50.0, 8.0)):
    """Seed-mode CFI model: classical seed, or the coarse seed when `coarse` (a coarse
    model path) is given."""
    from .cfi_infer import infer
    from .train_cfi import load_cfi
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_fraction)
    model = load_cfi(ckpt).to(device).eval()
    cm = cpu = None
    if coarse:
        from .coarse_infer import CoarseModel
        cm = CoarseModel(coarse, device)

    def fn(img, bl):
        nonlocal cpu
        t = time.perf_counter()
        try:
            out = infer(model, img, bl, band=None, cell=16, device=device, contours=True,
                        pixel_band=pixel_band, min_area=min_area, coarse_model=cm)
        except torch.OutOfMemoryError:        # large bands while training shares the GPU
            torch.cuda.empty_cache()
            if cpu is None:
                from .coarse_infer import CoarseModel
                cpu = (load_cfi(ckpt).eval(), CoarseModel(coarse, "cpu") if coarse else None)
            out = infer(cpu[0], img, bl, band=None, cell=16, device="cpu", contours=True,
                        pixel_band=pixel_band, min_area=min_area, coarse_model=cpu[1])
        return out["points"].double().cpu().numpy(), time.perf_counter() - t
    return fn


HEAD = ("missed_rate", "extra_rate", "loc_error_mean", "bscore", "signed_mean", "signed_median")


def print_summary(r):
    o = r["overall"]
    print(r["tag"], " ".join(f"{k}={o.get(k, float('nan')):.4f}" for k in HEAD),
          "far/img frame %.1f interior %.1f" % (r["far_extra"]["frame_points_per_image"],
                                                r["far_extra"]["interior_points_per_image"]),
          "winding %.3f" % r.get("winding", float("nan")))
    for g, d in r["slices"].items():
        for v, s in d.items():
            print(f"  {g}={v:<16s} n_gt={s.get('n_gt', 0):<8d} " + " ".join(
                f"{k}={s.get(k, float('nan')):.4f}" for k in HEAD))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "score"])
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--name", default="val_v4_full")
    ap.add_argument("--split", default="val")
    ap.add_argument("--coarse", help="coarse model for coarse-seed fine models")
    ap.add_argument("--classical", action="store_true", help="classical-seed model (no coarse)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "make":
        make_spec(a.name, a.split)
        return
    coarse = None if a.classical else (a.coarse or os.path.join(REPO_DIR, "models", "coarse_v2.pt"))
    tag = a.tag or os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    kind = torch.load(a.ckpt, map_location="cpu", weights_only=False)["config"]["model"].get("kind")
    if kind == "sdf":           # PLAN_v3 SDFNet: contours from the predicted signed distance
        from .sdf_infer import SDFPredictor
        predictor = SDFPredictor(a.ckpt, coarse, gpu_fraction=0.4)
    else:
        predictor = cfi_predictor(a.ckpt, coarse)
    r = evaluate(predictor, tag, a.name, a.limit)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = a.out or os.path.join(OUT_DIR, f"{tag}_{a.name}.json")
    with open(out, "w") as f:
        json.dump(r, f, indent=1)
    print_summary(r)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
