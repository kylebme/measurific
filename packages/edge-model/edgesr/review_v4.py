"""Review far-extra contours (> 2 px from any GT edge) of a fine model on val_v4_full.

For the first N benchmark images: count far points, then render the worst ones as
overview (GT green, far predictions red) plus zoomed crops of the largest far clusters.

  python -m edgesr.review_v4 RUN/final.pt [--n 40] [--show 8] [--classical]
"""

import argparse
import json
import os

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import REPO_DIR, WORK_DIR
from . import bench_v4 as B4
from . import sampler as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--classical", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    tag = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    out = a.out or os.path.join(WORK_DIR, "review", f"{tag}_v4_far")
    os.makedirs(out, exist_ok=True)
    coarse = None if a.classical else os.path.join(REPO_DIR, "models", "coarse_v2.pt")
    fn = B4.cfi_predictor(a.ckpt, coarse)
    store = S.open_store({"cache": "v4"})
    spec = json.load(open(B4.spec_path("val_v4_full")))
    by_id = {s["scene_id"]: i for i, s in enumerate(store.scenes)}
    rows = []
    for k, im in enumerate(spec["images"][:a.n]):
        sc = by_id[im["scene_id"]]
        img, bl, _ = B4.reading(store, sc, im["aug_seed"])
        gt = store.boundary_points(store.scenes[sc]["mask_index"]).astype(np.float64)
        pred, _ = fn(img, bl)
        d = cKDTree(gt).query(pred, workers=8)[0] if len(pred) else np.zeros(0)
        far = pred[d > B4.FAR_PX]
        rows.append((len(far), k, sc, img, gt, pred, far))
        print(k, len(pred), len(far), flush=True)
    rows.sort(key=lambda r: -r[0])
    sheet = []
    for n_far, k, sc, img, gt, pred, far in rows[:a.show]:
        rec = store.scenes[sc]
        ov = cv2.resize(img, (768, 768), interpolation=cv2.INTER_AREA)
        ov = (np.stack([np.clip(ov / max(ov.max(), 1e-6), 0, 1)] * 3, -1) * 220).astype(np.uint8)
        s = 768 / img.shape[0]
        for px, py in gt[::6] * s:
            ov[min(int(py), 767), min(int(px), 767)] = (0, 200, 0)
        for px, py in far[::3] * s:
            ov[min(int(py), 767), min(int(px), 767)] = (0, 0, 255)
        cv2.putText(ov, f"img {k} {rec['illumination_class']} {rec['blur_class']} {rec['contamination']} "
                    f"{rec['family']} far {n_far}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 1)
        # largest far cluster: zoom 96 px crop x 4
        crops = []
        if len(far):
            g = np.zeros((512, 512), np.uint8)
            q = np.clip((far / 4).astype(int), 0, 511)
            g[q[:, 1], q[:, 0]] = 1
            lab, n = ndimage.label(cv2.dilate(g, np.ones((5, 5), np.uint8)))
            sizes = ndimage.sum(g, lab, range(1, n + 1))
            for c in np.argsort(-sizes)[:2]:
                ys, xs = np.nonzero(lab == c + 1)
                cx, cy = int(xs.mean() * 4), int(ys.mean() * 4)
                x0, y0 = int(np.clip(cx - 48, 0, img.shape[1] - 96)), int(np.clip(cy - 48, 0, img.shape[0] - 96))
                cr = img[y0:y0 + 96, x0:x0 + 96]
                cr = (np.clip((cr - cr.min()) / max(cr.max() - cr.min(), 1e-6), 0, 1) * 220).astype(np.uint8)
                z = cv2.cvtColor(cv2.resize(cr, (384, 384), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
                for pts, col in ((gt, (0, 200, 0)), (pred, (0, 200, 255)), (far, (0, 0, 255))):
                    p = (pts - [x0, y0]) * 4
                    p = p[np.all((p >= 0) & (p < 384), 1)].astype(int)
                    z[p[:, 1], p[:, 0]] = col
                crops.append(z)
        while len(crops) < 2:
            crops.append(np.zeros((384, 384, 3), np.uint8))
        sheet.append(np.concatenate([ov, np.concatenate(crops, 0)], 1))
    cv2.imwrite(os.path.join(out, "far_extras.jpg"), np.concatenate(sheet, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("far points per image (first n):", float(np.mean([r[0] for r in rows])), "->", out)


if __name__ == "__main__":
    main()
