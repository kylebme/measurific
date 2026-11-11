#!/usr/bin/env python3
"""Separate PAE candidate selection, localization, noise and optical blur.

Analytic Gaussian half-plane irradiance is integrated over square pixels by
Gauss-Legendre quadrature (sharp edges use exact polygon area). No shape raster
or contour-ordering code participates. Fixed-candidate diagnostics deliberately
use ground truth; they are tests of the estimator, never production extraction.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import convolve
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pae import detect
from pae.silhouette import _fit_selected_edges
from pae.edges_iter1 import v_edges


def edge_image(sigma, angle, phase, size=32):
    yy, xx = np.mgrid[:size, :size]
    n = np.array([np.cos(angle), np.sin(angle)])
    center = np.array([size//2+phase, size//2])
    d = (xx-center[0])*n[0]+(yy-center[1])*n[1]
    if sigma == 0:
        # Exact area below a line: CDF of the sum of two uniforms.
        a, b = abs(n[0]), abs(n[1])
        if b < 1e-12:
            coverage = np.clip(d/a+.5, 0, 1)
        else:
            z = d+(a+b)/2
            coverage = (np.maximum(z, 0)**2-np.maximum(z-a, 0)**2-
                        np.maximum(z-b, 0)**2+np.maximum(z-a-b, 0)**2)/(2*a*b)
        coverage = np.clip(coverage, 0, 1)
    else:
        nodes, weights = np.polynomial.legendre.leggauss(12)
        coverage = np.zeros_like(d)
        for u, wu in zip(nodes/2, weights/2):
            for v, wv in zip(nodes/2, weights/2):
                coverage += wu*wv*ndtr((d+u*n[0]+v*n[1])/sigma)
    return 16.+216.*coverage, center, n


def fixed_low(image, position):
    rows, cols = image.shape
    g = convolve(image, np.ones((3, 3), np.float32)/9, mode='constant')
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:]-g[:, :-2])/2
    gy[1:-1] = (g[2:]-g[:-2])/2
    y, x = np.mgrid[:rows, :cols]
    result = v_edges(x.ravel('F'), y.ravel('F'), image, g, rows,
                     gx.ravel('F'), gy.ravel('F'), .75, 1.,
                     np.array([position]), 2, 10., image.ravel('F'), g.ravel('F'))
    return np.c_[result[1], result[2]]


def stats(errors, rejected, changed=0):
    e = np.asarray(errors)
    return dict(count=len(e), rejected=rejected, changed_candidate=changed,
                bias_px=float(e.mean()) if len(e) else None,
                std_px=float(e.std()) if len(e) else None,
                rms_px=float(np.sqrt(np.mean(e*e))) if len(e) else None,
                max_abs_px=float(np.max(abs(e))) if len(e) else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, nargs='+', default=[0, 1, 10])
    args = parser.parse_args()
    rng = np.random.default_rng(20260910)
    records = []
    for blur in [0., .25, .5, 1., 1.5, 2.]:
        for noise in [0., .001, .003, .01, .03]:
            modes = {str(mode): [[], 0, 0] for mode in args.iterations}
            modes.update({'fixed_0': [[], 0, 0], 'fixed_1': [[], 0, 0]})
            for trial in range(args.trials):
                angle = [-.63, -.27, 0., .27, .63][trial % 5]
                phase = [-.4, -.15, .1, .35][(trial//5) % 4]
                clean, center, normal = edge_image(blur, angle, phase)
                image = clean+rng.normal(0, noise*216, clean.shape)
                row = image.shape[0]//2
                oracle = int(np.floor(center[0]+.5))*len(image)+row
                for mode in args.iterations:
                    e = detect(image, 10., smoothing=mode)
                    ids = np.flatnonzero((e.y == row) & (abs(e.x-center[0]) < 5.))
                    if len(ids) != 1:
                        modes[str(mode)][1] += 1
                    else:
                        j = ids[0]
                        modes[str(mode)][0].append(float((np.array([e.x[j], e.y[j]])-center)@normal))
                        modes[str(mode)][2] += int(e.position[j] != oracle)
                e = _fit_selected_edges(image, np.array([], dtype=np.int64), np.array([oracle]), 2)
                fixed = {'fixed_0': np.c_[e.x, e.y], 'fixed_1': fixed_low(image, oracle)}
                for key, q in fixed.items():
                    if len(q) != 1 or not np.isfinite(q).all(): modes[key][1] += 1
                    else: modes[key][0].append(float((q[0]-center)@normal))
            r = dict(blur_sigma_px=blur, noise_fraction=noise, trials=args.trials,
                     modes={k: stats(*v) for k, v in modes.items()})
            records.append(r)
            print(blur, noise, {k: (round(v['rms_px'], 4) if v['rms_px'] is not None else None, v['rejected']) for k,v in r['modes'].items()}, flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(records, indent=2)+'\n')

if __name__ == '__main__': main()
