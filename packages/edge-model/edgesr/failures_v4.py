"""Failure review of a fine model on val_v4_full: what the missed and extra edges are,
how often, under which scene conditions, with contact sheets.

Kinds (per boundary point; bench_v4 counts everything beyond 0.5 px as missed/extra):
  offset - GT point whose nearest prediction is 0.5-2 px away (edge found, mislocated)
  absent - GT point with no prediction within 2 px (edge not found)
  false  - predicted point more than 2 px from every GT edge (edge that is not there)
Absent and false points connected at 1 px form failure clusters. Lengths are point counts
/ POINTS_PER_PX: a boundary crosses the 1/16 px lattice ~20 times per px of length.
Absent GT points are also tagged "contaminant" (within 4 px of where the scene mask differs
from the part's base mask: fibers, particles, burrs) and "small" (on an object below
50 px^2 or a hole below 8 px^2, which the region filter removes by design).
A scene is "hard" with heavy blur, ambient lighting or heavy contamination.

  python -m edgesr.failures_v4 predict RUN/final.pt [--coarse models/coarse_v2.pt]
  python -m edgesr.failures_v4 review RUN_TAG
"""

import argparse
import json
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import REPO_DIR, WORK_DIR
from . import bench_v4 as B4
from . import sampler as S

POINTS_PER_PX = 20.0
OFFSET_PX, GROSS_PX = 0.5, 2.0
SMALL_AREA = (50.0, 8.0)
CONDITIONS = ("family", "blur_class", "illumination_class", "contamination", "backlight_type", "tilted")
GT_C, PRED_C, OFFSET_C, ABSENT_C, FALSE_C = (0, 190, 0), (255, 200, 0), (0, 150, 255), (0, 0, 255), (255, 0, 255)


def out_dir(tag):
    return os.path.join(WORK_DIR, "review", f"{tag}_failures")


def is_hard(rec):
    return rec["blur_class"] == "heavy" or rec["illumination_class"] == "ambient" or rec["contamination"] == "heavy"


def depth_coc(rec):
    """Blur circle (px) across the part's depth: 2 NA x the depth range from the focus plane.
    The blur class describes the focus plane only; silhouette edges formed by geometry at
    other depths (end faces, flats, chamfers of thick parts) are blurred by up to this."""
    meta = json.load(open(os.path.join(rec["dir"], "metadata.json")))
    return float(meta["optics"]["realized"]["depth_coc_px"])


def coc_bin(v):
    return "<5" if v < 5 else "5-15" if v < 15 else "15-30" if v < 30 else ">=30"


def predict(a):
    tag = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    out = out_dir(tag)
    os.makedirs(out, exist_ok=True)
    import torch
    if torch.load(a.ckpt, map_location="cpu", weights_only=False)["config"]["model"].get("kind") == "sdf":
        from .sdf_infer import SDFPredictor
        fn = SDFPredictor(a.ckpt, a.coarse, gpu_fraction=0.35)
    else:
        fn = B4.cfi_predictor(a.ckpt, a.coarse)
    store = S.open_store({"cache": "v4"})
    spec = json.load(open(B4.spec_path(a.name)))
    by_id = {s["scene_id"]: i for i, s in enumerate(store.scenes)}
    for k, im in enumerate(spec["images"]):
        path = os.path.join(out, f"pred_{k:03d}.npy")
        if os.path.exists(path):
            continue
        img, bl, _ = B4.reading(store, by_id[im["scene_id"]], im["aug_seed"])
        pred, sec = fn(img, bl)
        np.save(path, pred.astype(np.float32))
        print(k, len(pred), f"{sec:.1f}s", flush=True)


def small_regions(cover):
    """Pixels of objects below SMALL_AREA[0] px^2 and holes below SMALL_AREA[1], dilated 2 px."""
    obj = (cover >= 128).astype(np.uint8)
    small = np.zeros(obj.shape, bool)
    for m, conn, limit in ((obj, 8, SMALL_AREA[0]), (1 - obj, 4, SMALL_AREA[1])):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=conn)
        tiny = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < limit)
        tiny = tiny[tiny > 0]
        if len(tiny):
            small |= np.isin(lab, tiny)
    return cv2.dilate(small.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def clusters(points, shape):
    """Connected groups (1 px grid, 8-connected after a 3x3 dilation) of points."""
    if not len(points):
        return []
    H, W = shape
    ix = np.clip(points.astype(int), 0, [W - 1, H - 1])
    grid = np.zeros(shape, np.uint8)
    grid[ix[:, 1], ix[:, 0]] = 1
    _, lab = cv2.connectedComponents(cv2.dilate(grid, np.ones((3, 3), np.uint8)), connectivity=8)
    which = lab[ix[:, 1], ix[:, 0]]
    order = np.argsort(which, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(which[order])) + 1)


def analyse(store, spec, tag):
    by_id = {s["scene_id"]: i for i, s in enumerate(store.scenes)}
    images, found = [], []
    for k, im in enumerate(spec["images"]):
        sc = by_id[im["scene_id"]]
        rec = store.scenes[sc]
        gt = store.boundary_points(rec["mask_index"]).astype(np.float64)
        pred = np.load(os.path.join(out_dir(tag), f"pred_{k:03d}.npy")).astype(np.float64)
        d_gp, j_gp = cKDTree(pred).query(gt, workers=8) if len(pred) else (np.full(len(gt), np.inf), None)
        d_pg = cKDTree(gt).query(pred, workers=8)[0] if len(pred) else np.zeros(0)
        # signed GT -> prediction distance: + where the nearest prediction lies outside the object
        signed = d_gp.copy()
        if len(pred):
            q = pred[j_gp]
            signed[store.mask_points(rec["mask_index"], np.floor(q[:, 0] * 16), np.floor(q[:, 1] * 16)) > 0] *= -1
        cont = B4.contaminant_points(store, sc, gt)
        small_px = small_regions(np.asarray(store.cover[rec["mask_index"]]))
        ix = np.clip(gt.astype(int), 0, [store.W - 1, store.H - 1])
        small = small_px[ix[:, 1], ix[:, 0]]
        absent = d_gp > GROSS_PX
        offset = (d_gp > OFFSET_PX) & ~absent
        false = d_pg > GROSS_PX
        coc = depth_coc(rec)
        row = {"k": k, "scene_id": rec["scene_id"], "hard": is_hard(rec), "glare": rec["glare"],
               **{c: rec[c] for c in CONDITIONS}, "depth_coc_px": coc, "depth_coc": coc_bin(coc),
               "gt_px": len(gt) / POINTS_PER_PX, "offset_px": offset.sum() / POINTS_PER_PX,
               "absent_px": absent.sum() / POINTS_PER_PX, "false_px": false.sum() / POINTS_PER_PX,
               "absent_contaminant_px": (absent & cont).sum() / POINTS_PER_PX,
               "absent_small_px": (absent & small & ~cont).sum() / POINTS_PER_PX,
               "offset_contaminant_px": (offset & cont).sum() / POINTS_PER_PX}
        for kind, pts, sel in (("absent", gt, absent), ("false", pred, false), ("offset", gt, offset)):
            idx = np.flatnonzero(sel)
            for c in clusters(pts[idx], (store.H, store.W)):
                members = idx[c]
                length = len(members) / POINTS_PER_PX
                if length < 1.0:
                    continue
                p = pts[members]
                f = {"k": k, "kind": kind, "length_px": length, "center": p.mean(0).tolist(),
                     "anchor": p[np.argmin(((p - p.mean(0)) ** 2).sum(1))].tolist(),
                     "bbox": [*p.min(0).tolist(), *p.max(0).tolist()], "hard": row["hard"], "depth_coc_px": coc}
                if kind != "false":
                    f["contaminant"] = float(cont[members].mean())
                    f["small"] = float(small[members].mean())
                    f["gap_px"] = float(np.median(d_gp[members]))
                    f["signed_px"] = float(np.median(signed[members]))
                else:
                    f["gap_px"] = float(np.median(d_pg[members]))
                found.append(f)
        row["absent_clusters_5px"] = sum(1 for f in found if f["k"] == k and f["kind"] == "absent" and f["length_px"] >= 5)
        row["false_clusters_5px"] = sum(1 for f in found if f["k"] == k and f["kind"] == "false" and f["length_px"] >= 5)
        images.append(row)
        print(k, rec["family"], rec["blur_class"], rec["illumination_class"], rec["contamination"],
              "offset %.0f absent %.0f false %.0f px of %.0f" % (row["offset_px"], row["absent_px"], row["false_px"],
                                                                 row["gt_px"]), flush=True)
    return images, found


def table(images):
    """Markdown summary by condition."""
    def line(name, rows):
        gt = sum(r["gt_px"] for r in rows)
        n = len(rows)
        return (f"| {name} | {n} | {100 * sum(r['offset_px'] for r in rows) / gt:.1f}% | "
                f"{100 * sum(r['absent_px'] for r in rows) / gt:.2f}% | "
                f"{sum(r['absent_px'] - r['absent_contaminant_px'] - r['absent_small_px'] for r in rows) / n:.0f} | "
                f"{sum(r['absent_contaminant_px'] + r['absent_small_px'] for r in rows) / n:.0f} | "
                f"{sum(r['false_px'] for r in rows) / n:.0f} | "
                f"{sum(r['absent_clusters_5px'] > 0 for r in rows)}/{n} | {sum(r['false_clusters_5px'] > 0 for r in rows)}/{n} |")
    out = ["| scenes | images | offset 0.5-2 px (% of edge) | absent > 2 px (% of edge) | absent part edge px/img | "
           "absent contaminant+small px/img | false edge px/img | images with an absent run >= 5 px | "
           "images with a false run >= 5 px |", "|---|" + "---|" * 8,
           line("all", images), line("typical", [r for r in images if not r["hard"]]),
           line("hard", [r for r in images if r["hard"]])]
    for c in (*CONDITIONS, "depth_coc"):
        for v in sorted({str(r[c]) for r in images}):
            out.append(line(f"{c}={v}", [r for r in images if str(r[c]) == v]))
    return "\n".join(out)


def stretch(a, lo=None, hi=None):
    lo = np.percentile(a, 0.5) if lo is None else lo
    hi = np.percentile(a, 99.5) if hi is None else hi
    return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 235).astype(np.uint8)


def draw(canvas, pts, x0, y0, zoom, color, size=1):
    p = ((pts - [x0, y0]) * zoom).astype(int)
    h, w = canvas.shape[:2]
    p = p[(p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] < w) & (p[:, 1] < h)]
    for dy in range(size):
        for dx in range(size):
            canvas[np.minimum(p[:, 1] + dy, h - 1), np.minimum(p[:, 0] + dx, w - 1)] = color


def cell(store, spec, by_id, f, cache, panel=320):
    k = f["k"]
    if k not in cache:
        im = spec["images"][k]
        sc = by_id[im["scene_id"]]
        rec = store.scenes[sc]
        img, bl, _ = B4.reading(store, sc, im["aug_seed"])
        gt = store.boundary_points(rec["mask_index"]).astype(np.float64)
        pred = np.load(os.path.join(out_dir(f["tag"]), f"pred_{k:03d}.npy")).astype(np.float64)
        d_gp = cKDTree(pred).query(gt, workers=8)[0]
        d_pg = cKDTree(gt).query(pred, workers=8)[0]
        cache.clear()
        cache[k] = (rec, img, bl, gt, pred, d_gp, d_pg)
    rec, img, bl, gt, pred, d_gp, d_pg = cache[k]
    H, W = img.shape
    # centre on the member point nearest the mean (a long run's or a ring's mean can lie off it)
    cx, cy = f["anchor"]
    x0b, y0b, x1b, y1b = f["bbox"]
    # context: 256 px window at the scene's real brightness (reading / backlight)
    cw = 256
    X0, Y0 = int(np.clip(cx - cw / 2, 0, W - cw)), int(np.clip(cy - cw / 2, 0, H - cw))
    ctx = cv2.resize(stretch(img[Y0:Y0 + cw, X0:X0 + cw], 0.0, bl * 1.05), (panel, panel),
                     interpolation=cv2.INTER_AREA)
    ctx = cv2.cvtColor(ctx, cv2.COLOR_GRAY2BGR)
    z = panel / cw
    draw(ctx, gt[::2], X0, Y0, z, GT_C)
    draw(ctx, pred[::2], X0, Y0, z, PRED_C)
    # zoom: the cluster's box plus margin, 48-128 px (offsets: 32 px), contrast stretched
    zw = 32 if f["kind"] == "offset" else int(np.clip(max(x1b - x0b, y1b - y0b) + 24, 48, 128))
    x0, y0 = int(np.clip(cx - zw / 2, 0, W - zw)), int(np.clip(cy - zw / 2, 0, H - zw))
    cv2.rectangle(ctx, (int((x0 - X0) * z), int((y0 - Y0) * z)), (int((x0 + zw - X0) * z), int((y0 + zw - Y0) * z)),
                  (0, 220, 255), 1)
    zm = panel / zw
    zoom = cv2.cvtColor(cv2.resize(stretch(img[y0:y0 + zw, x0:x0 + zw]), (panel, panel),
                                   interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    draw(zoom, gt, x0, y0, zm, GT_C)
    draw(zoom, pred, x0, y0, zm, PRED_C)
    draw(zoom, gt[(d_gp > OFFSET_PX) & (d_gp <= GROSS_PX)], x0, y0, zm, OFFSET_C, 2)
    draw(zoom, gt[d_gp > GROSS_PX], x0, y0, zm, ABSENT_C, 2)
    draw(zoom, pred[d_pg > GROSS_PX], x0, y0, zm, FALSE_C, 2)
    # caption
    cap = np.full((44, 2 * panel + 4, 3), 32, np.uint8)
    l1 = (f"val {k}  {rec['family']}  blur {rec['blur_class']}  light {rec['illumination_class']}  "
          f"contam {rec['contamination']}  {rec['backlight_type']}{'  tilted' if rec['tilted'] else ''}  "
          f"depth blur {f['depth_coc_px']:.0f} px")
    l2 = f"{f['kind']} run {f['length_px']:.0f} px, gap {f['gap_px']:.1f} px"
    if f["kind"] != "false":
        l2 += (f" ({'outside' if f['signed_px'] > 0 else 'inside'}), contaminant {100 * f['contaminant']:.0f}%, "
               f"small {100 * f['small']:.0f}%")
    l2 += f"  |  zoom {zw} px wide"
    cv2.putText(cap, l1, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(cap, l2, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 255), 1, cv2.LINE_AA)
    body = np.concatenate([ctx, np.full((panel, 4, 3), 32, np.uint8), zoom], 1)
    return np.concatenate([cap, body], 0)


def sheet(store, spec, picks, title, path, per_row=2):
    by_id = {s["scene_id"]: i for i, s in enumerate(store.scenes)}
    cache = {}
    cells = [cell(store, spec, by_id, f, cache) for f in sorted(picks, key=lambda f: f["k"])]
    h, w = cells[0].shape[:2]
    while len(cells) % per_row:
        cells.append(np.full((h, w, 3), 20, np.uint8))
    rows = [np.concatenate(sum([[c, np.full((h, 6, 3), 20, np.uint8)] for c in cells[i:i + per_row]], [])[:-1], 1)
            for i in range(0, len(cells), per_row)]
    grid = np.concatenate(sum([[r, np.full((6, r.shape[1], 3), 20, np.uint8)] for r in rows], [])[:-1], 0)
    head = np.full((58, grid.shape[1], 3), 20, np.uint8)
    cv2.putText(head, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    x = 8
    for name, col in (("GT edge", GT_C), ("prediction", PRED_C), ("offset 0.5-2 px", OFFSET_C),
                      ("absent (no prediction within 2 px)", ABSENT_C), ("false (> 2 px from any GT)", FALSE_C)):
        cv2.rectangle(head, (x, 36), (x + 14, 50), col, -1)
        cv2.putText(head, name, (x + 20, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        x += 30 + 9 * len(name)
    cv2.putText(head, "left: 256 px context at true brightness   right: zoom, contrast stretched",
                (grid.shape[1] - 560, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.imwrite(path, np.concatenate([head, grid], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])


def pick(found, kind, hard, n, per_image=1, min_px=2.0):
    """Largest clusters of a kind, at most per_image per image."""
    got, count = [], {}
    for f in sorted(found, key=lambda f: -f["length_px"]):
        if f["kind"] != kind or f["hard"] != hard or f["length_px"] < min_px or count.get(f["k"], 0) >= per_image:
            continue
        count[f["k"]] = count.get(f["k"], 0) + 1
        got.append(f)
        if len(got) == n:
            break
    return got


def review(a):
    store = S.open_store({"cache": "v4"})
    spec = json.load(open(B4.spec_path(a.name)))
    out = out_dir(a.tag)
    images, found = analyse(store, spec, a.tag)
    for f in found:
        f["tag"] = a.tag
    json.dump({"images": images, "clusters": found}, open(os.path.join(out, "failures.json"), "w"))
    md = table(images)
    open(os.path.join(out, "summary.md"), "w").write(md + "\n")
    print(md)
    for kind in ("absent", "false", "offset"):
        for hard in (False, True):
            picks = pick(found, kind, hard, a.per_sheet)
            if not picks:
                continue
            name = f"{kind}_{'hard' if hard else 'typical'}"
            n_img = sum(1 for r in images if r["hard"] == hard)
            title = (f"{a.tag}: largest {kind} edge runs, {'hard' if hard else 'typical'} scenes "
                     f"({n_img} val images; hard = heavy blur, ambient light or heavy contamination)")
            sheet(store, spec, picks, title, os.path.join(out, f"{name}.jpg"))
            print("wrote", name, len(picks), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("ckpt")
    p.add_argument("--coarse", default=os.path.join(REPO_DIR, "models", "coarse_v2.pt"))
    p.add_argument("--name", default="val_v4_full")
    r = sub.add_parser("review")
    r.add_argument("tag")
    r.add_argument("--name", default="val_v4_full")
    r.add_argument("--per-sheet", type=int, default=12)
    a = ap.parse_args()
    predict(a) if a.cmd == "predict" else review(a)


if __name__ == "__main__":
    main()
