"""Visual review of a coarse model (or the classical seed) on macro val scenes.

Each panel: input | prediction (instances coloured, part boundary red, contact yellow) |
GT (instances coloured, loose debris magenta). Writes the worst images by bscore and a
random sample, contact sheets, and predictions on real images given with --real.

  python -m edgesr.review_coarse work/runs/ID/final.pt [--n 12] [--real img.jpg ...]
  python -m edgesr.review_coarse classical
"""

import argparse
import os

import cv2
import numpy as np

from . import WORK_DIR
from . import bench_coarse as BC
from .macro import MacroStore

PALETTE = np.array([[0, 0, 0]] + [[int(c) for c in cv2.cvtColor(
    np.uint8([[[h, 200, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]] for h in range(0, 180, 17)], np.uint8)


def colour(lab):
    return PALETTE[np.where(lab > 0, (lab - 1) % (len(PALETTE) - 1) + 1, 0)]


def panel(x, pred, inst=None, debris=None, title=""):
    g = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    base = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    lab = pred.get("instances")
    if lab is None:
        from .coarse import instances
        p = 1 / (1 + np.exp(-pred["part_logit"]))
        lab = instances(p, pred.get("contact", np.zeros_like(p)))
    pv = cv2.addWeighted(base, 0.5, colour(lab), 0.5, 0)
    pv[lab == 0] = base[lab == 0]
    if "contact" in pred:
        pv[pred["contact"] > 0.5] = (0, 220, 255)
    edge = (pred["part_logit"] > 0).astype(np.uint8)
    edge = edge ^ cv2.erode(edge, np.ones((3, 3), np.uint8))
    pv[edge > 0] = (0, 0, 255)
    tiles = [base, pv]
    if inst is not None:
        gv = cv2.addWeighted(base, 0.5, colour(inst.astype(np.int32)), 0.5, 0)
        gv[inst == 0] = base[inst == 0]
        if debris is not None:
            gv[debris >= 64] = (255, 0, 255)
        tiles.append(gv)
    out = np.concatenate(tiles, 1)
    cv2.putText(out, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1, cv2.LINE_AA)
    return out


def sheet(paths, out, cols=2, width=1200):
    tiles = []
    for p in paths:
        im = cv2.imread(p)
        tiles.append(cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1])),
                                interpolation=cv2.INTER_AREA))
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out")
    ap.add_argument("--real", nargs="*", default=[])
    ap.add_argument("--variant", choices=["aspect_roi"])
    a = ap.parse_args()
    if a.ckpt == "classical":
        name, predict = "classical", BC.classical_predict
    else:
        name = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
        predict = BC.model_predictor(a.ckpt)
    out = a.out or os.path.join(WORK_DIR, "coarse", "review", name + (f"_{a.variant}" if a.variant else ""))
    os.makedirs(out, exist_ok=True)
    store = MacroStore()
    idx = store.split(a.split)

    def load(k, i):
        sc = store.scenes[i]
        x = BC.export_reading(sc["scene_id"], a.split)
        inst, debris = np.asarray(store.inst[i]), np.asarray(store.debris[i])
        kw = {}
        if a.variant:
            x, inst, debris, valid, r, roi = BC.aspect_roi_transform(k, x, inst, debris)
            kw = {"valid": valid, "aspect": r}
            sc = dict(sc, illumination=f"{sc['illumination']} aspect {r:.2f} roi {roi}")
        return sc, x, inst, debris, kw

    scores = []
    for k, i in enumerate(idx):
        sc, x, inst, debris, kw = load(k, i)
        pred = predict(x, **kw)
        r = BC.score_image(pred, inst, debris, sc)
        b = 0.5 * (r["sum_gp"] / max(r["n_gpts"], 1) + r["sum_pg"] / max(r["n_ppts"], 1))
        miss = sum(p["covered"] < 0.5 for p in r["per_part"])
        scores.append((b + 0.5 * miss + 0.5 * r["false_parts"], k, i, b, miss, r["false_parts"]))
    scores.sort(reverse=True)
    rng = np.random.default_rng(0)
    pick = {"worst": scores[:a.n],
            "random": [scores[k] for k in rng.choice(len(scores), a.n, replace=False)]}
    for kind, items in pick.items():
        paths = []
        for _, k, i, b, miss, false in items:
            sc, x, inst, debris, kw = load(k, i)
            title = (f"{sc['scene_id'][-7:]} {sc['illumination']} {sc['blur']} {sc['layout']} "
                     f"bscore {b:.2f} missed {miss} false {false}")
            p = os.path.join(out, f"{kind}_{sc['scene_id']}.png")
            cv2.imwrite(p, panel(x, predict(x, **kw), inst, debris, title))
            paths.append(p)
        sheet(paths, os.path.join(out, f"contact_{kind}.jpg"))
    from .coarse import model_reading
    for rp in a.real:
        img = cv2.imread(rp, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        H, W = img.shape
        pred = predict(model_reading(img), aspect=W / H)       # the frame squashed to 512 x 512
        pred = {"part_logit": cv2.resize(pred["part_logit"], (W, H), interpolation=cv2.INTER_LINEAR),
                "contact": cv2.resize(pred["contact"], (W, H), interpolation=cv2.INTER_LINEAR),
                "instances": cv2.resize(pred["instances"].astype(np.float32), (W, H),
                                        interpolation=cv2.INTER_NEAREST).astype(np.int32)}
        p = os.path.join(out, "real_" + os.path.splitext(os.path.basename(rp))[0] + ".png")
        cv2.imwrite(p, panel(img, pred, title=os.path.basename(rp)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
