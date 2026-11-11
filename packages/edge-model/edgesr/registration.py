"""Numerical and visual image/mask registration check on real renders.

For sharp, silhouette-lit scenes the backlit image darkness is a blurred copy
of the mask coverage. A least-squares subpixel shift between the two, measured
per patch and mapped to the source frame, should be ~0 for both lattice and
rotated patches. Misregistration from the sampler (or the interface) appears as
a consistent offset.

Usage: python -m edgesr.registration [--n 400] [--out work/registration]
"""

import argparse
import json
import math
import os

import cv2
import numpy as np

from . import SCALE, WORK_DIR
from . import sampler as S


def fit_shift(dark, cov):
    """Shift delta (patch px) and gain/offset with dark(x) ~= a*cov(x - delta) + b."""
    gy, gx = np.gradient(cov)
    m = np.zeros_like(cov, bool)
    m[3:-3, 3:-3] = True
    X = np.stack([cov[m], np.ones(m.sum()), -gx[m], -gy[m]], 1)
    coef, *_ = np.linalg.lstsq(X, dark[m], rcond=None)
    a, b, ax, ay = coef
    return np.array([ax / a, ay / a]), a


def draw_overlay(img, mask):
    """Nearest-upsampled input with the hi-res mask boundary in red."""
    lo, hi = np.percentile(img, [0.5, 99.5])
    g = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    up = np.kron(g, np.ones((SCALE, SCALE)))
    rgb = (np.stack([up] * 3, -1) * 255).astype(np.uint8)
    edge = mask ^ cv2.erode(mask, np.ones((3, 3), np.uint8))
    rgb[edge.astype(bool)] = (255, 0, 0)
    return rgb


def run(n, out_dir, seed=0):
    store = S.SourceStore()
    os.makedirs(out_dir, exist_ok=True)
    scenes = [i for i, s in enumerate(store.scenes)
              if s["focus_class"] == "sharp" and s["illumination_class"] == "silhouette"]
    rng = np.random.default_rng(seed)
    results = {}
    tiles = []
    for mode in ("lattice", "rotated"):
        shifts, gains = [], []
        for k in range(n):
            scene = scenes[rng.integers(len(scenes))]
            sc = store.scenes[scene]
            cands = store.candidates(sc["part_index"], "sparse")
            cx, cy = cands[rng.integers(len(cands))].astype(float) + 0.5
            if mode == "lattice":
                center = (float(round(cx)), float(round(cy)))
                A = S.patch_matrix(flip=bool(rng.integers(2)), quarter_turns=int(rng.integers(4)))
            else:
                center = (cx + rng.uniform(-4, 4), cy + rng.uniform(-4, 4))
                A = S.patch_matrix(angle=rng.uniform(0, 2 * math.pi), flip=bool(rng.integers(2)))
            img, mask = S.sample_patch(store, scene, center, A)
            cov = mask.reshape(S.PATCH, SCALE, S.PATCH, SCALE).mean((1, 3))
            dark = (sc["backlight_level"] - img) / (sc["backlight_level"] - sc["object_level"])
            if cov[3:-3, 3:-3].std() < 0.05:
                continue
            d, a = fit_shift(dark, cov)
            shifts.append(A @ d)      # into the source frame
            gains.append(a)
            if k < 6:
                tiles.append(draw_overlay(img, mask))
        s = np.array(shifts)
        results[mode] = {
            "patches": len(s),
            "median_shift_px": np.median(s, 0).round(4).tolist(),
            "mean_shift_px": s.mean(0).round(4).tolist(),
            "sem_px": (s.std(0) / math.sqrt(len(s))).round(4).tolist(),
            "mad_px": np.median(np.abs(s - np.median(s, 0)), 0).round(4).tolist(),
            "median_gain": float(np.median(gains)),
        }
    rows = [np.concatenate(tiles[i:i + 6], 1) for i in range(0, len(tiles), 6)]
    cv2.imwrite(os.path.join(out_dir, "overlays.png"),
                cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    with open(os.path.join(out_dir, "registration.json"), "w") as f:
        json.dump(results, f, indent=1)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(WORK_DIR, "registration"))
    a = ap.parse_args()
    print(json.dumps(run(a.n, a.out), indent=1))


if __name__ == "__main__":
    main()
