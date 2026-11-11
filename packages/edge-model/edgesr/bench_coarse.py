"""Coarse-model benchmark on the macro-v1-512 val/test scenes.

Inputs are the 8-bit exported camera images (`macro.export`: seeded exposure,
Poisson-Gaussian noise at 6k-60k e- per macro pixel, 10% JPEG), so every model, including
off-the-shelf instance segmenters trained on the export, sees identical pixels. That is
noisier than the pipeline's input (a 4x area average of a full-resolution reading), so
it is conservative. GT: instances.png (parts), debris.png (loose debris, not a part).

Metrics (macro px; 1 macro px = 4 source px):
- Boundary, sampled densely (4x upsampled grid, <= 0.25 px apart, so point spacing does not
  inflate distances): GT points are the 0.5 crossings of the GT part mask (nearest 4x,
  Gaussian sigma 0.5 macro px, which straightens the majority-label staircase); predicted
  points are the zero crossings of the bilinearly 4x-upsampled part logit.
  d = nearest-point distance.
  bscore = 1/2 (mean min(d_gt->pred, 2) + mean min(d_pred->gt, 2)), lower is better
  (primary). recall@t = GT points within t of a predicted point (the fine model's band
  must cover the true edge: recall@1 ~ a +-4 source px band); precision@t likewise.
- IoU of the part mask (pooled).
- Parts: a GT part is missed if < 50% of its pixels are predicted part. A predicted
  component (8-conn, >= min_area px) is false if < 50% of it is GT part; it is a debris
  false part if it overlaps loose debris.
- Instances: predicted instances vs GT at IoU > 0.5: F1, SQ (mean matched IoU), PQ;
  merges (a predicted instance matched to >= 2 GT parts at > 50% of their area). Extra
  pieces of a matched GT part whose own mask is disconnected (thin bridges below the 512 px
  resolution, occlusion) are counted as gt_fragments, not FP (inst_pq_strict counts them).
Slices: per image (illumination, blur, layout, backlight, part count, loose debris) and
per GT boundary point / part (group, material, tilted, stacked, cut by frame, attached
debris).

Variant `aspect_roi` (--variant aspect_roi): the same images squashed from other frame
aspect ratios and, for half of them, with a circular region of interest (see
aspect_roi_transform); the region mask and the aspect are given to the model.

Usage:
  python -m edgesr.bench_coarse score RUN_DIR/final.pt [--split val] [--variant aspect_roi]
  python -m edgesr.bench_coarse classical [--split val] [--variant aspect_roi]
"""

import argparse
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import WORK_DIR
from .coarse import instances as split_instances
from .macro import MACRO_DIR, MacroStore

EXPORT_DIR = os.path.join(MACRO_DIR, "export", "images")
SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}
BENCH_DIR = os.path.join(WORK_DIR, "coarse", "bench")
CAP = 2.0
TOLS = (0.5, 1.0, 1.5, 2.0)
MIN_AREA = 16         # macro px^2 (= 256 source px^2): smaller predicted components are dropped


def export_reading(scene_id, split):
    d = os.path.join(EXPORT_DIR, SPLIT_DIR[split])
    for ext in (".png", ".jpg"):
        p = os.path.join(d, scene_id + ext)
        if os.path.exists(p):
            return cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    raise FileNotFoundError(scene_id)


def has_wire(scene):
    return any(p["group"] == "wire" for p in scene["parts"])


def is_v1(scene):
    return int(scene["scene_id"][-7:]) < 6000


SUBSETS = {"v1": is_v1, "wires": has_wire}


def eval_set(store, split="val", n=None, subset=None):
    """subset: "v1" (the original 6,000 scenes, as before the wire batch) or "wires"."""
    idx = store.split(split)
    if subset:
        idx = [i for i in idx if SUBSETS[subset](store.scenes[i])]
    return idx[:n] if n else idx


# ---------------------------------------------------------------------------
# boundary points

UP = 4               # boundary sampling: 4x upsampled grid


def crossings(f, factor=1):
    """Zero crossings of a map along horizontal/vertical pixel-center links, linearly
    interpolated. f is on a grid `factor` x finer than macro px. Returns (N, 2) x/y in macro
    px (pixel i covers [i, i+1)) and the flat index of each link's positive endpoint."""
    f = f.astype(np.float64)
    W = f.shape[1]
    pts, pos = [], []
    a, b = f[:, :-1], f[:, 1:]
    dy, dx = np.nonzero((a > 0) != (b > 0))
    t = a[dy, dx] / (a[dy, dx] - b[dy, dx])
    pts.append(np.stack([dx + 0.5 + t, dy + 0.5], 1))
    pos.append(dy * W + np.where(a[dy, dx] > 0, dx, dx + 1))
    a, b = f[:-1, :], f[1:, :]
    dy, dx = np.nonzero((a > 0) != (b > 0))
    t = a[dy, dx] / (a[dy, dx] - b[dy, dx])
    pts.append(np.stack([dx + 0.5, dy + 0.5 + t], 1))
    pos.append(np.where(a[dy, dx] > 0, dy, dy + 1) * W + dx)
    return np.concatenate(pts) / factor, np.concatenate(pos)


def gt_points(inst):
    """Dense GT boundary points and the instance id on their inside."""
    h, w = inst.shape
    up = cv2.resize(inst, (w * UP, h * UP), interpolation=cv2.INTER_NEAREST)
    sm = cv2.GaussianBlur((up > 0).astype(np.float32), (0, 0), 0.5 * UP)
    pts, pos = crossings(sm - 0.5, UP)
    lab = up.ravel()[pos]
    if (lab == 0).any():          # smoothing moved the crossing past a 1-px feature
        near = cv2.dilate(up, np.ones((2 * UP + 1,) * 2, np.uint8)).ravel()[pos]
        lab = np.where(lab == 0, near, lab)
    return pts, lab


def pred_points(logit):
    h, w = logit.shape
    up = cv2.resize(logit.astype(np.float32), (w * UP, h * UP), interpolation=cv2.INTER_LINEAR)
    return crossings(up, UP)[0]


def remove_small(mask, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[lab]


# ---------------------------------------------------------------------------
# per-image scoring

def part_attrs(scene):
    return {p["instance_id"]: p for p in scene["parts"]}


def score_image(pred, inst, debris, scene, min_area=MIN_AREA):
    """pred: dict with 'part_logit' (H, W) (> 0 = part) and optionally 'contact' (prob) or
    'instances' (int labels). Returns a record of sums for aggregation."""
    gt = inst > 0
    logit = pred["part_logit"]
    pm = logit > 0
    if min_area:
        keep = remove_small(pm, min_area)
        logit = np.where(pm & ~keep, -1.0, logit)     # drop small predicted specks
        pm = pm & keep
    # boundary
    gpts, glab = gt_points(inst)
    ppts = pred_points(logit)
    if len(ppts) and len(gpts):
        d_gp = cKDTree(ppts).query(gpts, workers=4)[0]
        d_pg = cKDTree(gpts).query(ppts, workers=4)[0]
    else:
        d_gp = np.full(len(gpts), np.inf)
        d_pg = np.full(len(ppts), np.inf)
    rec = {"n_gpts": len(gpts), "n_ppts": len(ppts),
           "sum_gp": float(np.minimum(d_gp, CAP).sum()), "sum_pg": float(np.minimum(d_pg, CAP).sum()),
           "inter": int((gt & pm).sum()), "union": int((gt | pm).sum())}
    for t in TOLS:
        rec[f"rec@{t}"] = int((d_gp <= t).sum())
        rec[f"prec@{t}"] = int((d_pg <= t).sum())
    # per-part boundary recall (attributed by the inside pixel's instance)
    attrs = part_attrs(scene)
    per_part = []
    for pid, p in attrs.items():
        m = inst == pid
        area = int(m.sum())
        if not area:
            continue
        sel = glab == pid
        per_part.append({"pid": pid, "area": area, "covered": float((pm & m).sum() / area),
                         "n_gpts": int(sel.sum()), "rec@1.0": int((d_gp[sel] <= 1.0).sum()),
                         "sum_gp": float(np.minimum(d_gp[sel], CAP).sum())})
    # false parts
    n, lab, st, _ = cv2.connectedComponentsWithStats(pm.astype(np.uint8), connectivity=8)
    false, false_debris = 0, 0
    deb = debris >= 128
    for k in range(1, n):
        comp = lab == k
        if st[k, cv2.CC_STAT_AREA] < min_area:
            continue
        if (comp & gt).sum() < 0.5 * st[k, cv2.CC_STAT_AREA]:
            false += 1
            if (comp & deb).any():
                false_debris += 1
    rec.update({"false_parts": false, "false_debris": false_debris,
                "debris_px": int(deb.sum()), "debris_pred_px": int((deb & ~gt & pm).sum())})
    # instances
    if "instances" in pred:
        pin = pred["instances"]
    else:
        contact = pred.get("contact")
        pin = split_instances(1 / (1 + np.exp(-logit)) if contact is not None else pm.astype(np.float32),
                              contact if contact is not None else np.zeros_like(logit))
    rec.update(match_instances(pin, inst, per_part))
    rec["per_part"] = per_part
    return rec


def match_instances(pin, inst, per_part):
    gids = [p["pid"] for p in per_part]
    npred = int(pin.max())
    if not gids or not npred:
        return {"tp": 0, "fp": npred, "fn": len(gids), "sum_iou": 0.0, "merges": 0, "gt_fragments": 0}
    # contingency table
    ng = int(inst.max()) + 1
    both = inst.astype(np.int64) * (npred + 1) + pin
    cnt = np.bincount(both.ravel(), minlength=ng * (npred + 1)).reshape(ng, npred + 1)
    garea = cnt.sum(1)
    parea = cnt.sum(0)
    tp, s_iou, matched_p = 0, 0.0, set()
    for p in per_part:
        g = p["pid"]
        inter = cnt[g, 1:]
        iou = inter / (garea[g] + parea[1:] - inter)
        j = int(np.argmax(iou))
        p["iou"] = float(iou[j])
        if iou[j] > 0.5:
            tp += 1
            s_iou += float(iou[j])
            matched_p.add(j + 1)
    # predicted instances covering > 50% of two or more GT parts
    frac = cnt[1:, 1:] / np.maximum(garea[1:, None], 1)
    merges = int(((frac > 0.5).sum(0) >= 2).sum())
    # An unmatched prediction lying mostly inside a matched GT part whose own mask is split
    # into disconnected pieces (thin bridges lost at 512 px, occlusion) is a fragment of it:
    # no local method can reattach it. Reported separately; the strict score counts it as FP.
    matched_g = {g for g in gids if cnt[g, 1:].max() > 0 and
                 (cnt[g, 1:] / (garea[g] + parea[1:] - cnt[g, 1:])).max() > 0.5}
    fp = frag = 0
    for j in range(1, npred + 1):
        if j in matched_p or parea[j] < MIN_AREA:
            continue
        g = int(np.argmax(cnt[1:, j])) + 1
        if (g in matched_g and cnt[g, j] > 0.5 * parea[j]
                and cv2.connectedComponents((inst == g).astype(np.uint8), connectivity=8)[0] > 2):
            frag += 1
        else:
            fp += 1
    return {"tp": tp, "fp": fp, "fn": len(gids) - tp, "sum_iou": s_iou, "merges": merges,
            "gt_fragments": frag}


# ---------------------------------------------------------------------------
# aggregation

def summarize(recs):
    s = defaultdict(float)
    for r in recs:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                s[k] += v
    g, p = max(s["n_gpts"], 1), max(s["n_ppts"], 1)
    out = {"images": len(recs),
           "bscore": 0.5 * (s["sum_gp"] / g + s["sum_pg"] / p),
           "gt_dist_mean": s["sum_gp"] / g, "pred_dist_mean": s["sum_pg"] / p,
           "iou": s["inter"] / max(s["union"], 1)}
    for t in TOLS:
        out[f"recall@{t}"] = s[f"rec@{t}"] / g
        out[f"precision@{t}"] = s[f"prec@{t}"] / p
    r, pr = out["recall@1.0"], out["precision@1.0"]
    out["bf1@1"] = 2 * r * pr / max(r + pr, 1e-9)
    parts = [pp for rr in recs for pp in rr["per_part"]]
    out["parts"] = len(parts)
    out["part_miss"] = float(np.mean([pp["covered"] < 0.5 for pp in parts])) if parts else 0.0
    out["false_parts_per_img"] = s["false_parts"] / max(len(recs), 1)
    out["false_debris_per_img"] = s["false_debris"] / max(len(recs), 1)
    out["debris_px_pred_frac"] = s["debris_pred_px"] / max(s["debris_px"], 1)
    tp, fp, fn, fr = s["tp"], s["fp"], s["fn"], s["gt_fragments"]
    out["inst_f1"] = 2 * tp / max(2 * tp + fp + fn, 1)          # fragments of split GT parts ignored
    out["inst_sq"] = s["sum_iou"] / max(tp, 1)
    out["inst_pq"] = s["sum_iou"] / max(tp + 0.5 * fp + 0.5 * fn, 1)
    out["inst_pq_strict"] = s["sum_iou"] / max(tp + 0.5 * (fp + fr) + 0.5 * fn, 1)
    out["gt_fragments_per_img"] = fr / max(len(recs), 1)
    out["merges_per_img"] = s["merges"] / max(len(recs), 1)
    return out


def part_summary(parts):
    g = sum(p["n_gpts"] for p in parts)
    return {"parts": len(parts), "part_miss": float(np.mean([p["covered"] < 0.5 for p in parts])),
            "recall@1.0": sum(p["rec@1.0"] for p in parts) / max(g, 1),
            "gt_dist_mean": sum(p["sum_gp"] for p in parts) / max(g, 1),
            "inst_iou_mean": float(np.mean([p.get("iou", 0.0) for p in parts]))}


def n_bucket(n):
    return "1" if n == 1 else "2-4" if n <= 4 else "5-10"


def slices(recs, scenes):
    out = {}
    img_keys = {"illumination": lambda s: s["illumination"], "blur": lambda s: s["blur"],
                "layout": lambda s: s["layout"], "backlight": lambda s: s["backlight_type"],
                "n_parts": lambda s: n_bucket(s["n_parts"]),
                "loose_debris": lambda s: "yes" if s["loose_debris"] else "no"}
    if scenes and "roi" in scenes[0]:
        img_keys.update({"aspect": lambda s: s["aspect"], "roi": lambda s: s["roi"]})
    for name, f in img_keys.items():
        groups = defaultdict(list)
        for r, sc in zip(recs, scenes):
            groups[f(sc)].append(r)
        out[name] = {k: summarize(v) for k, v in sorted(groups.items())}
    part_keys = {"group": lambda p: p["group"], "material": lambda p: p["material"],
                 "tilted": lambda p: str(p["tilted"]), "stacked": lambda p: str(p["stacked"]),
                 "cut_by_frame": lambda p: str(p["cut_by_frame"]),
                 "attached_debris": lambda p: str(p["attached_debris"] > 0),
                 "size": lambda p: ("<256" if p["mask_area_px"] < 256 else
                                    "<4096" if p["mask_area_px"] < 4096 else ">=4096")}
    for name, f in part_keys.items():
        groups = defaultdict(list)
        for r, sc in zip(recs, scenes):
            attrs = part_attrs(sc)
            for pp in r["per_part"]:
                groups[f(attrs[pp["pid"]])].append(pp)
        out["part_" + name] = {k: part_summary(v) for k, v in sorted(groups.items())}
    return out


ASPECTS = (1.0, 4 / 3, 16 / 9, 0.75, 1.5, 9 / 16, 1.25, 1.9)


def aspect_roi_transform(k, x, inst, debris):
    """Variant `aspect_roi`, deterministic per image k: the centered region of original
    aspect ASPECTS[k % 8] squashed to the square frame, and for even k a circular region of
    interest (in original-frame coordinates, so an ellipse here): the inscribed circle or
    one of 0.95 x half the long side (cut by the frame). Outside it the reading is zero
    (software crop) or a dark unlit backlight at 0.12 of the background with a 2 px edge,
    and GT parts are cut. The region mask is given to the model (it is known)."""
    from .coarse import resize_axes
    H, W = x.shape
    r = ASPECTS[k % len(ASPECTS)]
    w0, h0 = (W, int(round(H / r))) if r >= 1 else (int(round(W * r)), H)
    y0, x0 = (H - h0) // 2, (W - w0) // 2
    sl = (slice(y0, y0 + h0), slice(x0, x0 + w0))
    x = resize_axes(x[sl], W, H)
    inst = resize_axes(np.ascontiguousarray(inst[sl]), W, H, nearest=True)
    debris = resize_axes(np.ascontiguousarray(debris[sl]), W, H, nearest=True)
    valid, roi = None, "none"
    if k % 2 == 0:
        R = 0.5 * min(w0, h0) if (k // 2) % 2 == 0 else 0.475 * max(w0, h0)
        u = (np.arange(W) + 0.5) * (w0 / W) - w0 / 2
        v = (np.arange(H) + 0.5) * (h0 / H) - h0 / 2
        d = (R - np.sqrt(u[None, :] ** 2 + v[:, None] ** 2)) * (W / np.sqrt(w0 * h0))
        valid = d >= 0
        bg = float(np.median(x[(inst == 0) & valid])) if ((inst == 0) & valid).any() else 0.8
        inst = np.where(valid, inst, 0).astype(inst.dtype)
        debris = np.where(valid, debris, 0).astype(debris.dtype)
        if (k // 4) % 2 == 0:
            x = np.where(valid, x, 0.0).astype(np.float32)
            roi = "software"
        else:
            m = np.clip(d / 2.0 + 0.5, 0, 1)
            noise = np.random.default_rng(k).normal(0, 0.004, x.shape)
            x = np.clip(x * m + (0.12 * bg + noise) * (1 - m), 0, 1).astype(np.float32)
            roi = "unlit"
    return x, inst, debris, valid, r, roi


def evaluate(predict, store, split="val", n=None, min_area=MIN_AREA, log_every=0, variant=None,
             subset=None):
    """predict(reading (H, W) float32 in [0, 1], valid=None, aspect=1.0) -> dict (see
    score_image). variant "aspect_roi": aspect_roi_transform."""
    idx = eval_set(store, split, n, subset)
    recs, scenes, t_pred = [], [], 0.0
    for k, i in enumerate(idx):
        sc = store.scenes[i]
        x = export_reading(sc["scene_id"], split)
        inst, debris = np.asarray(store.inst[i]), np.asarray(store.debris[i])
        kw = {}
        if variant == "aspect_roi":
            x, inst, debris, valid, r, roi = aspect_roi_transform(k, x, inst, debris)
            kw = {"valid": valid, "aspect": r}
            sc = dict(sc, aspect=f"{r:.2f}", roi=roi)
        t = time.perf_counter()
        pred = predict(x, **kw)
        t_pred += time.perf_counter() - t
        recs.append(score_image(pred, inst, debris, sc, min_area))
        scenes.append(sc)
        if log_every and (k + 1) % log_every == 0:
            print(f"  {k + 1}/{len(idx)}", flush=True)
    out = summarize(recs)
    out["predict_ms"] = 1e3 * t_pred / max(len(idx), 1)
    return out, recs, scenes


# ---------------------------------------------------------------------------
# predictors

def classical_predict(x, valid=None, aspect=1.0):
    """adapter.py-style seed at macro scale: field fit, normalize, Gaussian 1.2, Otsu.
    Logit = Otsu level - clean (DN), so the boundary is the threshold's sub-pixel crossing."""
    from . import seed as SD
    src = x.astype(np.float32) * 255.0
    h, w = src.shape
    coef, p90 = SD.fit_field(src)
    field = SD.eval_field_grid(coef, p90, w, h)
    clean, t = SD.segment(src, field)
    logit = (t - clean).astype(np.float32)
    if valid is not None:
        logit = np.where(valid, logit, -8.0).astype(np.float32)
    return {"part_logit": logit}


def model_predictor(ckpt, device="cuda", min_peak=None):
    import torch
    from .coarse import CoarseNet, postprocess, predict_maps
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    slim = "state_dict" in ck                      # models/coarse_v*.pt (coarse_infer.export)
    model = CoarseNet(ck["model_config"] if slim else ck["config"]["model"])
    model.load_state_dict(ck["state_dict"] if slim else ck["model"])
    model = model.to(device).eval()

    def predict(x, valid=None, aspect=1.0):
        m = predict_maps(model, x, device, valid, aspect)
        return postprocess(m) if min_peak is None else postprocess(m, min_peak)
    return predict


def run(name, predict, split, n, out_path=None, min_area=MIN_AREA, variant=None, subset=None):
    store = MacroStore()
    res, recs, scenes = evaluate(predict, store, split, n, min_area, log_every=100, variant=variant,
                                 subset=subset)
    res["slices"] = slices(recs, scenes)
    res.update({"name": name, "split": split, "min_area": min_area, "variant": variant,
                "subset": subset, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    suffix = (f"_{variant}" if variant else "") + (f"_{subset}" if subset else "")
    out_path = out_path or os.path.join(BENCH_DIR, f"{name}_{split}{suffix}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print_summary(res)
    print(f"wrote {out_path}")
    return res


HEAD = ("bscore", "bf1@1", "recall@1.0", "recall@2.0", "precision@1.0", "iou", "part_miss",
        "false_parts_per_img", "inst_f1", "inst_pq", "inst_pq_strict", "merges_per_img")


def print_summary(res):
    print(" ".join(f"{k}={res[k]:.4f}" for k in HEAD if k in res))
    sl = res.get("slices", {})
    for name in ("illumination", "blur", "layout", "aspect", "roi"):
        for k, v in sl.get(name, {}).items():
            print(f"  {name}={k:<12} n={v['images']:<4} bscore={v['bscore']:.4f} "
                  f"recall@1={v['recall@1.0']:.4f} miss={v['part_miss']:.4f} "
                  f"false/img={v['false_parts_per_img']:.3f} pq={v['inst_pq']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "eval", "classical"])      # score == eval
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int)
    ap.add_argument("--name")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-area", type=float, default=MIN_AREA)
    ap.add_argument("--variant", choices=["aspect_roi"])
    ap.add_argument("--min-peak", type=float, help="hysteresis (default coarse.MIN_PEAK_LOGIT)")
    ap.add_argument("--subset", choices=sorted(SUBSETS), help="v1: original scenes; wires: wire scenes")
    a = ap.parse_args()
    if a.cmd == "classical":
        run(a.name or "classical", classical_predict, a.split, a.n, a.out, a.min_area, a.variant, a.subset)
    else:
        name = a.name or os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
        run(name, model_predictor(a.ckpt, a.device, a.min_peak), a.split, a.n, a.out, a.min_area,
            a.variant, a.subset)


if __name__ == "__main__":
    main()
