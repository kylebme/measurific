"""Backlit silhouette renderer for contour-extraction training data (v3).

Compared with ``pae_corpus.render`` (part always scaled to fill a 640–1920 px
frame, axis-aligned, flat background, Gaussian noise):

* Scale follows a fixed-magnification station: the part's larger extent spans a
  log-uniform 350–3200 px (frame capped at 4096 px), with random margins, so small
  features land at realistic pixel sizes instead of always filling the frame.
* Random in-plane rotation and subpixel placement; edges are not grid aligned.
* Subpixel polygon rasterization (OpenCV fixed-point vertices, area-averaged
  supersampling) instead of integer-snapped Pillow polygons.
* Optics: defocus PSF, a weak wide glare halo, backlight gradient/vignetting and
  low-frequency blotches.
* Contamination on the backlight: dust specks and curved fibres, sometimes on
  the part boundary, blurred as out-of-focus.
* Sensor: shot noise, read noise, PRNU, 8-bit or 12-bit (uint16) output.

Telecentric optics are assumed: no perspective growth of thick parts.
"""
import cv2
import numpy as np
from .data import ring_geometry

SHIFT = 6  # 1/64 px fixed-point vertices


def _loguniform(rng, a, b):
    return float(np.exp(rng.uniform(np.log(a), np.log(b))))


def _fill(canvas, rings, value, ss):
    pts = [np.round((r*ss+(ss-1)/2)*(1 << SHIFT)).astype(np.int32).reshape(-1, 1, 2) for r in rings]
    cv2.fillPoly(canvas, pts, value, lineType=cv2.LINE_8, shift=SHIFT)


def _smooth_field(rng, shape, scale, amplitude):
    small = rng.normal(0, 1, (max(2, shape[0]//scale), max(2, shape[1]//scale))).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), 1.5)
    small /= max(float(np.abs(small).max()), 1e-6)
    return 1+amplitude*cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def _contamination(rng, shape, truth, ss):
    """Opacity map of dust and fibres on the backlight (supersampled)."""
    h, w = shape
    layer = np.zeros((h*ss, w*ss), np.float32)
    def anchor():
        if rng.random() < .35:
            # On or near the part boundary, where it corrupts the extracted contour.
            p = truth[int(rng.integers(len(truth)))]
            return p+rng.normal(0, 2.5, 2)
        return np.array([rng.uniform(0, w), rng.uniform(0, h)])
    items = []
    for _ in range(int(rng.poisson(2.5))):
        c = anchor(); r = _loguniform(rng, .6, 6.); a = float(rng.uniform(.25, 1.))
        cv2.circle(layer, tuple(np.round((c*ss+(ss-1)/2)*(1 << SHIFT)).astype(int)), int(round(r*ss*(1 << SHIFT))),
                   a, -1, lineType=cv2.LINE_AA, shift=SHIFT)
        items.append(dict(kind='dust', radius_px=r, opacity=a))
    for _ in range(int(rng.poisson(.6))):
        c = anchor(); length = _loguniform(rng, 20, 350); angle = rng.uniform(0, np.pi)
        d = np.array([np.cos(angle), np.sin(angle)]); n = np.array([-d[1], d[0]])
        t = np.linspace(-.5, .5, 40)[:, None]
        bend = rng.normal(0, .15)*length*(1-4*t**2)
        curve = c+t*length*d+bend*n
        width = rng.uniform(.8, 3.); a = float(rng.uniform(.3, 1.))
        pts = np.round((curve*ss+(ss-1)/2)*(1 << SHIFT)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(layer, [pts], False, a, thickness=max(1, int(round(width*ss))), lineType=cv2.LINE_AA, shift=SHIFT)
        items.append(dict(kind='fibre', length_px=length, width_px=width, opacity=a))
    return np.clip(layer, 0, 1), items


def render(raw, seed, extent_px=None, max_side=4096):
    """Return (image, truth_px, settings); truth_px are the source contour vertices in pixels.

    ``settings['to_mm']`` maps pixel coordinates back to the source millimetre frame.
    """
    rng = np.random.default_rng(seed)
    xy, offsets, holes = ring_geometry(raw)
    angle = float(rng.uniform(-np.pi, np.pi))
    c, s = np.cos(angle), np.sin(angle)
    # mm (y up) -> px (y down), rotated.
    rot = np.array([[c, -s], [-s, -c]])
    q = xy@rot.T
    extent = np.ptp(q, axis=0)
    target = float(extent_px or _loguniform(rng, 350, 3200))
    ppm = target/extent.max()
    margins = rng.integers(16, 140, 4)
    size = np.ceil(extent*ppm).astype(int)+[margins[0]+margins[1], margins[2]+margins[3]]
    if size.max() > max_side:
        ppm *= (max_side-margins.max()*2)/(extent*ppm).max()
        size = np.ceil(extent*ppm).astype(int)+[margins[0]+margins[1], margins[2]+margins[3]]
    w, h = int(size[0]), int(size[1])
    origin = np.array([margins[0], margins[2]])+rng.random(2)
    offset = origin-q.min(0)*ppm
    truth = q*ppm+offset
    ss = 4 if max(w, h) <= 1400 else 3 if max(w, h) <= 2400 else 2
    canvas = np.zeros((h*ss, w*ss), np.uint8)
    _fill(canvas, [truth[a:b] for a, b, hole in zip(offsets[:-1], offsets[1:], holes) if not hole], 255, ss)
    _fill(canvas, [truth[a:b] for a, b, hole in zip(offsets[:-1], offsets[1:], holes) if hole], 0, ss)
    coverage = canvas.reshape(h, ss, w, ss).mean(axis=(1, 3), dtype=np.float32)/255
    dirt, items = _contamination(rng, (h, w), truth, ss)
    dirt = dirt.reshape(h, ss, w, ss).mean(axis=(1, 3))
    sigma = _loguniform(rng, .4, 1.6)
    blurred = cv2.GaussianBlur(coverage, (0, 0), sigma)
    dirt = cv2.GaussianBlur(dirt, (0, 0), sigma+rng.uniform(0, 2.5)) if dirt.any() else dirt
    glare = float(rng.uniform(0, .05)); halo_sigma = _loguniform(rng, 3, 25)
    halo = cv2.GaussianBlur(coverage, (0, 0), halo_sigma) if glare else blurred
    bright = float(rng.uniform(150, 250)); dark = float(rng.uniform(3, 60))
    yy, xx = np.mgrid[-1:1:h*1j, -1:1:w*1j].astype(np.float32)
    background = bright*(1+rng.uniform(-.08, .08)*xx+rng.uniform(-.08, .08)*yy-rng.uniform(0, .12)*(xx*xx+yy*yy))
    background *= _smooth_field(rng, (h, w), int(rng.integers(48, 256)), float(rng.uniform(0, .025)))
    transmitted = (1-blurred)*(1-glare)+glare*(1-halo)
    expected = background*transmitted*(1-dirt*(1-dark/bright))+dark*blurred*(1-glare)
    expected *= rng.normal(1, rng.uniform(0, .006), (h, w)).astype(np.float32)
    bits = 12 if rng.random() < .4 else 8
    full = 4095 if bits == 12 else 255; gain = full/255
    electrons = float(rng.uniform(10, 150))/gain
    signal = rng.poisson(np.maximum(expected*gain, 0)*electrons)/electrons
    signal = signal+rng.normal(0, rng.uniform(.3, 2.5)*gain, (h, w))
    image = np.clip(np.rint(signal), 0, full).astype(np.uint16 if bits == 12 else np.uint8)
    inverse = np.linalg.inv(rot*ppm)
    settings = dict(seed=int(seed), ppm=float(ppm), extent_px=float(extent.max()*ppm), width=w, height=h, angle=angle,
                    supersample=ss, blur_sigma=sigma, glare=glare, halo_sigma=halo_sigma, bright=bright, dark=dark,
                    bits=bits, max_value=full, electrons_per_dn=electrons*gain, contamination=items,
                    to_mm=dict(matrix=inverse.tolist(), offset=offset.tolist()))
    return image, truth, settings


def to_mm(points, settings):
    m = np.asarray(settings['to_mm']['matrix']); o = np.asarray(settings['to_mm']['offset'])
    return (np.asarray(points)-o)@m.T
