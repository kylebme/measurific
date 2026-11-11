"""Contact sheet of macro scenes: image | instance outlines (colour per part) | debris.

    ../../packages/edge-model/.venv/bin/python scripts/review.py OUTPUT [--count 12] [--out sheet.jpg]
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

S = 384


def _label(img, text, y):
    cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
    cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1)


def row(scene):
    meta = json.load(open(os.path.join(scene, "metadata.json")))
    img = cv2.imread(os.path.join(scene, "image.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
    img = np.clip(img / max(np.percentile(img, 97), 1) * 0.9, 0, 1)
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    inst = cv2.imread(os.path.join(scene, "instances.png"), cv2.IMREAD_UNCHANGED)
    over = gray.copy()
    rng = np.random.default_rng(1)
    for k in np.unique(inst)[1:]:
        colour = tuple(int(c) for c in rng.integers(60, 256, 3))
        m = (inst == k).astype(np.uint8)
        tint = over.copy()
        tint[m > 0] = colour
        over = cv2.addWeighted(over, 0.55, tint, 0.45, 0)
        cs, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(over, cs, -1, colour, 1)
    deb = cv2.imread(os.path.join(scene, "debris.png"), cv2.IMREAD_UNCHANGED)
    dview = gray.copy()
    dview[deb > 20] = (0, 0, 255)
    panels = [cv2.resize(p, (S, S), interpolation=cv2.INTER_AREA if p is gray else cv2.INTER_NEAREST)
              for p in (gray, over, dview)]
    fams = ", ".join(sorted({p["family"] for p in meta["parts"]}))[:60]
    o = meta["optics"]
    _label(panels[0], f"{meta['scene_index']} {meta['split']} n={len(meta['parts'])} {meta['layout']} "
                      f"{meta['illumination']['class']}", 14)
    _label(panels[0], f"{o['blur_class']} coc {o['target_coc_px']:.2f} px/mm {meta['px_per_mm']:.2f}", 30)
    _label(panels[1], fams, 14)
    tilted = sum(p["pose"]["tilt_rad"] > 0 for p in meta["parts"])
    att = sum(len(p["debris"]) for p in meta["parts"])
    _label(panels[2], f"tilted {tilted} attached debris {att} loose {len(meta['loose_debris'])}", 14)
    return np.hstack([cv2.copyMakeBorder(p, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for p in panels])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output")
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="macro_review.jpg")
    args = ap.parse_args()
    scenes = sorted(glob.glob(os.path.join(args.output, "scenes", "macro_*", "metadata.json")))
    scenes = [os.path.dirname(s) for s in scenes[args.start:args.start + args.count]]
    rows = [row(s) for s in scenes]
    # two scenes per sheet row
    pairs = [np.hstack(rows[i:i + 2]) if i + 1 < len(rows) else
             np.hstack([rows[i], np.zeros_like(rows[i])]) for i in range(0, len(rows), 2)]
    cv2.imwrite(args.out, np.vstack(pairs), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(args.out, len(rows))


if __name__ == "__main__":
    main()
