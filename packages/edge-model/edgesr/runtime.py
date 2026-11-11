"""Torch-free inference of the exported edge model (ONNX Runtime + numpy), with ordered rings.

Same computation as `cfi_infer.infer` for the shipped seed-mode model
(models/cfi_B_seed.pt), split between host numpy and two exported graphs:

- fine.onnx: gathered (source, field, mask) windows -> fine features around each active
  cell and the 1-px dilated per-pixel band logit
- decode.onnx: 3x3 feature neighbourhoods of band pixels -> 16x16 sub-pixel occupancy
  logits (the block decoder's occupancy rows)

The host runs the classical seed (`seed.seed_full`), gathers windows, selects band pixels,
assembles the per-cell hi-res occupancy canvases and applies the minimum-area filter.
`EdgeModel.__call__` returns the same unordered lattice crossings as `predict.Predictor`;
`EdgeModel.rings` traces the occupancy into ordered closed rings.

Coordinates: source pixel i covers [i, i+1) (pixel centers at i + 0.5), x right, y down.

Export (needs torch): python -m edgesr.onnx_export
"""

import json
import os
import time
from pathlib import Path

import numpy as np

from . import REPO_DIR
from .regions import filter_small_regions_exact
from .seed import seed_full

FORMAT = "edgesr-cfi-onnx-v1"
DEFAULT_MODEL = os.path.join(REPO_DIR, "models", "cfi_B_seed-onnx")
# CoreML is left out: it ran the fine graph ~6x slower than the CPU (M-series, 2026-09).
GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")
LOGIT = 8.0            # canvas logit outside decoded band pixels (sign from the seed mask)
APRON = 16             # backlight apron (px) that closes silhouettes cut by the frame
MAX_BACKLIGHT = 0.95   # brighter backlights are scaled down to the training maximum
TRUST_MARGIN = 2       # px beyond valid pixels where predicted occupancy is kept
MIN_SEED_SUPPORT = 0.5 # fraction of a ring's region the seed mask must agree with


def select_providers(device="cpu", available=None):
    """Execution providers for "cpu" or "gpu"; "gpu" falls back to CPU when none is available."""
    if device not in ("cpu", "gpu"):
        raise ValueError("Compute device must be cpu or gpu")
    if available is None:
        import onnxruntime
        available = onnxruntime.get_available_providers()
    gpu = [name for name in GPU_PROVIDERS if name in available]
    if device == "cpu" or not gpu:
        return ["CPUExecutionProvider"]
    return [gpu[0], "CPUExecutionProvider"]


def reading(image, max_value=None):
    """2-D uint8/uint16 (or float in [0, max_value or 1]) -> float32 reading in [0, 1].
    max_value: sensor full scale (default: dtype max; e.g. 4095 for 12-bit data in uint16)."""
    a = np.asarray(image)
    if a.ndim != 2:
        raise ValueError("expected a 2-D grayscale image")
    full = float(max_value or (np.iinfo(a.dtype).max if a.dtype.kind in "ui" else 1.0))
    return np.clip(a.astype(np.float32) / full, 0, 1)


def backlit_frame(img, valid=None, apron=APRON):
    """Replace invalid pixels and surround the frame with the continued local backlight.

    A part cut by the image border or a shaped ROI then has an ordinary backlit edge
    there, so every silhouette closes inside the analysed frame. The fill must continue
    the local backlight: the model reads a step between two background levels as an
    edge. It is a normalized convolution of the valid background pixels (Otsu, eroded
    away from edges) at growing scales, falling back to their median where no
    background is near.

    A backlight above MAX_BACKLIGHT of full scale (typically clipped at full scale, as
    in many JPEGs) is scaled down to it: the model was trained with backlights of
    0.35-0.95, reads their absolute level, and hallucinates edges beside a saturated
    one. Backlights inside the training range are left unchanged. A linear gain is a shorter exposure and does not move edges.
    Returns ((H + 2 apron, W + 2 apron) float32, backlight level, gain)."""
    import cv2
    h, w = img.shape
    if valid is None:
        valid = np.ones(img.shape, bool)
    framed = np.pad(img.astype(np.float32), apron)
    known = np.pad(valid, apron)
    level8 = np.round(framed[known] * 255).astype(np.uint8).reshape(-1, 1)
    threshold = cv2.threshold(level8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0] / 255
    background = cv2.erode((known & (framed > threshold)).astype(np.uint8), np.ones((5, 5), np.uint8))
    weight = background.astype(np.float32)
    level = float(np.median(framed[background > 0]) if background.any() else np.percentile(framed[known], 99))
    gain = min(1.0, MAX_BACKLIGHT / max(level, 1e-6))
    fill = np.full(framed.shape, level, np.float32)
    done = known.copy()
    H, W = framed.shape
    for sigma in (4, 16, 64, 256):
        f = max(1, sigma // 4)          # blur at sigma 4 on an f-times smaller grid
        small = (cv2.resize(framed * weight, (max(1, W // f), max(1, H // f)), interpolation=cv2.INTER_AREA),
                 cv2.resize(weight, (max(1, W // f), max(1, H // f)), interpolation=cv2.INTER_AREA))
        num, den = (cv2.resize(cv2.GaussianBlur(a, (0, 0), 4), (W, H), interpolation=cv2.INTER_LINEAR)
                    for a in small)
        ok = ~done & (den > 0.05)
        fill[ok] = num[ok] / den[ok]
        done |= ok
        if done.all():
            break
    return (np.where(known, framed, fill) * np.float32(gain)).astype(np.float32), level, gain


class EdgeModel:
    def __init__(self, directory=DEFAULT_MODEL, device="cpu", threads=None):
        import onnxruntime

        self.directory = Path(directory)
        metadata = json.loads((self.directory / "model.json").read_text())
        if metadata.get("format") != FORMAT:
            raise ValueError(f"Unsupported edge model format: {metadata.get('format')!r}")
        self.metadata = metadata
        geometry = metadata["geometry"]
        self.cell, self.scale = int(geometry["cell"]), int(geometry["scale"])
        self.margin, self.frame_pad = int(geometry["margin"]), int(geometry["frame_pad"])
        self.defaults = metadata["inference_defaults"]
        options = onnxruntime.SessionOptions()
        if threads:
            options.intra_op_num_threads = int(threads)
        requested = select_providers(device)
        self.sessions = {name: onnxruntime.InferenceSession(str(self.directory / f"{name}.onnx"),
                                                            options, providers=requested)
                         for name in ("fine", "decode")}
        self.providers = self.sessions["fine"].get_providers()
        self.coarse = None
        if metadata.get("seed", "classical") == "coarse":
            # fine model v4: the coarse model's part probability, coarse-background field and
            # coarse + gradient band replace the classical seed (seed.coarse_seed_full)
            from .coarse_runtime import CoarseRuntime
            self.coarse = CoarseRuntime(self.directory / metadata["coarse_model"], requested, threads)

    def describe(self):
        return {"format": FORMAT, "directory": str(self.directory), "providers": self.providers,
                "source_checkpoint_sha256": self.metadata.get("provenance", {}).get("source_checkpoint_sha256")}

    def min_area(self, min_object_area=None, min_hole_area=None):
        return (float(self.defaults["min_object_area"] if min_object_area is None else min_object_area),
                float(self.defaults["min_hole_area"] if min_hole_area is None else min_hole_area))

    def occupancy(self, img, min_area=(50.0, 8.0), chunk=256):
        """Hi-res occupancy logit canvases of the active cells of img, (H, W) float32 reading
        in [0, 1] of a dark part on a bright backlight. Frame coordinates are padded by
        `frame_pad` (edge replicate) and to a multiple of the cell size."""
        times = {}
        c, s, M, fp = self.cell, self.scale, self.margin, self.frame_pad
        H0, W0 = img.shape
        pads = ((fp, fp + (-(H0 + 2 * fp)) % c), (fp, fp + (-(W0 + 2 * fp)) % c))
        t = time.perf_counter()
        if self.coarse is not None:          # coarse seed on the unpadded frame, then padded
            from .seed import coarse_seed_full
            cs = coarse_seed_full(img, self.coarse)
            sd = {"source": np.pad(cs["source"], pads, mode="edge"),
                  "field": np.pad(cs["field"], pads, mode="edge")}
            prob = np.pad(cs["mask"], pads, mode="edge")
            mask, band = prob > 0.5, np.pad(cs["band"], pads)
        else:
            img = np.pad(img, pads, mode="edge")
            sd = seed_full(img, inputs=False)
            mask, band = sd["mask"], sd["band"]
            prob = mask.astype(np.float32)
        H, W = band.shape
        cy, cx = np.nonzero(band.reshape(H // c, c, W // c, c).any(axis=(1, 3)))
        n_cells = len(cy)
        times["seed"] = time.perf_counter() - t

        t = time.perf_counter()
        win = c + 2 * M
        planes = np.pad(np.stack([sd["source"], sd["field"], prob.astype(np.float32)]),
                        ((0, 0), (M, M), (M, M)), mode="edge")
        ar = np.arange(win)
        feats, pbands = [], []
        for c0 in range(0, n_cells, chunk):
            ys = (cy[c0:c0 + chunk] * c)[:, None] + ar
            xs = (cx[c0:c0 + chunk] * c)[:, None] + ar
            windows = planes[:, ys[:, :, None], xs[:, None, :]].transpose(1, 0, 2, 3)
            f, pb = self.sessions["fine"].run(["features", "band"],
                                              {"windows": np.ascontiguousarray(windows, np.float32)})
            feats.append(f)
            pbands.append(pb)
        width = self.metadata["model_config"]["fine_width"]
        feat = np.concatenate(feats) if feats else np.zeros((0, width, c + 2, c + 2), np.float32)
        pband = np.concatenate(pbands) if pbands else np.zeros((0, c, c), np.float32)
        times["fine"] = time.perf_counter() - t

        t = time.perf_counter()
        bw = band.reshape(H // c, c, W // c, c).transpose(0, 2, 1, 3)[cy, cx]
        if self.defaults.get("pixel_band") is not None:
            bw = bw & (pband > self.defaults["pixel_band"])
        n, py, px = np.nonzero(bw)
        neighbours = np.stack([feat[n, :, py + 1 + dy, px + 1 + dx]
                               for dy in (-1, 0, 1) for dx in (-1, 0, 1)], -1).reshape(len(n), feat.shape[1] * 9)
        logits = [self.sessions["decode"].run(["logits"], {"neighbours": neighbours[k:k + 65536]})[0]
                  for k in range(0, len(n), 65536)]
        logits = np.concatenate(logits) if logits else np.zeros((0, s * s), np.float32)
        times["decode"] = time.perf_counter() - t

        t = time.perf_counter()
        ar_c = np.arange(c)
        occ = mask[(cy * c)[:, None, None] + ar_c[None, :, None], (cx * c)[:, None, None] + ar_c[None, None, :]]
        canvas = np.repeat(np.repeat((occ.astype(np.float32) * 2 - 1) * LOGIT, s, 1), s, 2)
        canvas.reshape(n_cells, c, s, c, s).transpose(0, 1, 3, 2, 4)[n, py, px] = logits.reshape(-1, s, s)
        if np.any(np.asarray(min_area) > 0) and n_cells:
            ins = canvas >= 0
            changed = filter_small_regions_exact(ins, cy, cx, c, min_area) != ins
            canvas[changed] = np.where(ins[changed], -LOGIT, LOGIT)
        times["canvas"] = time.perf_counter() - t
        return {"canvas": canvas, "cy": cy, "cx": cx, "mask": mask, "frame_pad": fp,
                "image_shape": (H0, W0), "n_cells": n_cells, "n_band_px": int(len(n)), "times": times}

    def __call__(self, image, max_value=None, min_object_area=None, min_hole_area=None):
        """Unordered sub-pixel contour points, as `predict.Predictor` returns them."""
        t = time.perf_counter()
        ma = self.min_area(min_object_area, min_hole_area)
        occ = self.occupancy(reading(image, max_value), ma)
        points = lattice_crossings(occ, self.cell, self.scale)
        fp = occ["frame_pad"]
        H0, W0 = occ["image_shape"]
        points -= fp
        points = points[(points[:, 0] >= 0) & (points[:, 1] >= 0) & (points[:, 0] <= W0) & (points[:, 1] <= H0)]
        return {"points": points, "seconds": time.perf_counter() - t, "stage_seconds": occ["times"],
                "n_cells": occ["n_cells"], "n_band_px": occ["n_band_px"], "min_area": ma}

    def rings(self, image, max_value=None, valid_mask=None, min_object_area=None, min_hole_area=None,
              apron=APRON):
        """Ordered closed silhouette rings of a dark-on-light image.

        Invalid pixels (valid_mask False) and an apron around the frame are replaced by the
        fitted backlight, so parts cut by the frame or the ROI close along its boundary.
        Predicted occupancy counts only within TRUST_MARGIN px of valid pixels; beyond that
        (fill, apron, frame padding) it is background, so nothing the model predicts there
        becomes a ring. The classical seed decides which rings exist and the model where
        they are: a ring whose own region (inside it, minus the rings directly inside it)
        the seed mask supports by less than MIN_SEED_SUPPORT is dropped with everything
        inside it. This rejects the model's occasional objects on faint fibers, vignetting
        ramps or the fill.
        Every vertex is a crossing of the predicted boundary with the 1/16 px lattice.
        Returns {"rings": [(n, 2) float64 x/y], "is_hole", "depth", "parent", ...}, largest
        first; object outlines and hole outlines have opposite orientation (`trace_rings`)."""
        t = time.perf_counter()
        img = reading(image, max_value)
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, bool)
            if valid_mask.shape != img.shape or not valid_mask.any():
                raise ValueError("valid_mask must match the image and contain pixels")
        framed, backlight, gain = backlit_frame(img, valid_mask, apron)
        ma = self.min_area(min_object_area, min_hole_area)
        # nest_rings applies the minimum areas to whole rings (and what they enclose)
        occ = self.occupancy(framed, (0.0, 0.0))
        t2 = time.perf_counter()
        offset = occ["frame_pad"] + apron
        restrict_occupancy(occ, valid_mask, img.shape, offset, self.cell, self.scale)
        traced = trace_rings(occ, self.cell, self.scale)
        seed = occ["mask"][offset:offset + img.shape[0], offset:offset + img.shape[1]].astype(bool)
        nested = nest_rings([q - offset for q in traced["rings"]], traced["is_hole"], ma, seed=seed)
        occ["times"]["trace"] = time.perf_counter() - t2
        return {**nested, "lattice_crossings": traced["crossings"], "apron": apron, "min_area": ma,
                "backlight": backlight, "gain": gain,
                "n_cells": occ["n_cells"], "n_band_px": occ["n_band_px"],
                "stage_seconds": occ["times"], "seconds": time.perf_counter() - t}


def restrict_occupancy(occ, valid, shape, offset, cell, scale, margin=TRUST_MARGIN):
    """Background, in the canvases and the seed mask, beyond `margin` px of the valid
    pixels (the image sits at `offset` in the padded frame). In place."""
    import cv2
    mask, canvas, cy, cx = occ["mask"], occ["canvas"], occ["cy"], occ["cx"]
    trusted = np.zeros(mask.shape, np.uint8)
    trusted[offset:offset + shape[0], offset:offset + shape[1]] = 1 if valid is None else valid
    trusted = cv2.dilate(trusted, np.ones((2 * margin + 1,) * 2, np.uint8)).astype(bool)
    mask &= trusted
    if len(cy):
        ar = np.arange(cell)
        cells = trusted[(cy * cell)[:, None, None] + ar[None, :, None], (cx * cell)[:, None, None] + ar[None, None, :]]
        canvas[~np.repeat(np.repeat(cells, scale, 1), scale, 2)] = -LOGIT


def lattice_crossings(occ, cell, scale):
    """Within-cell crossings of the 0 level with the hi-res lattice (as cfi_infer.infer),
    in padded-frame coordinates."""
    canvas, cy, cx = occ["canvas"], occ["cy"], occ["cx"]
    ins = canvas >= 0
    out = []
    for axis in (2, 1):
        k, i, j = np.nonzero(np.diff(ins, axis=axis))
        a = canvas[k, i, j]
        b = canvas[k, i + (axis == 1), j + (axis == 2)]
        t = np.clip(-a / (b - a), 0, 1)
        x = (j + 0.5 + t * (axis == 2)) / scale + cx[k] * cell
        y = (i + 0.5 + t * (axis == 1)) / scale + cy[k] * cell
        out.append(np.stack([x, y], 1))
    return np.concatenate(out).astype(np.float64)


# Marching squares on the hi-res lattice. A square's corners a, b, c, d are the samples
# (Y, X), (Y, X+1), (Y+1, X+1), (Y+1, X); edge k joins corner k and k+1 (0 top, 1 right,
# 2 bottom, 3 left). A segment enters through an edge whose corners go outside -> inside
# in that cyclic order and leaves through the next inside -> outside edge. Neighbouring
# squares traverse a shared edge in opposite directions, so segments chain consistently.
# Saddles join the two object corners: objects 8-connected, holes 4-connected, as in the
# region filter.
def _segment_table():
    pairs = np.zeros((16, 2, 2), np.int64)
    counts = np.zeros(16, np.int64)
    for code in range(16):
        inside = [(code >> k) & 1 for k in range(4)]
        entries = [k for k in range(4) if not inside[k] and inside[(k + 1) % 4]]
        exits = [k for k in range(4) if inside[k] and not inside[(k + 1) % 4]]
        counts[code] = len(entries)
        if len(entries) == 1:
            pairs[code, 0] = entries[0], exits[0]
        elif len(entries) == 2:
            for m, e in enumerate(entries):
                pairs[code, m] = e, (e - 1) % 4
    return pairs, counts


_PAIRS, _COUNTS = _segment_table()
_EDGE_DY = np.array([0, 0, 1, 0])
_EDGE_DX = np.array([0, 1, 0, 0])
_EDGE_VERTICAL = np.array([0, 1, 0, 1])
_EDGE_FROM = np.array([0, 1, 3, 0])      # corner index of the edge's lower sample
_EDGE_TO = np.array([1, 2, 2, 3])


def _mask_edge_squares(mask, cy, cx, cell, scale):
    """Top-left samples (Y, X) of the lattice squares on seed-mask boundaries that do not
    lie between pixels of active cells. The band normally covers the seed's boundary, but
    restrict_occupancy cuts the seed at the trusted margin, and a seed that reaches the
    lattice border ends there (beyond is background); both boundaries run through inactive
    cells. Each pixel edge contributes the scale + 1 squares along it."""
    H, W = mask.shape
    active = np.zeros((H // cell, W // cell), bool)
    active[cy, cx] = True
    active = np.pad(np.repeat(np.repeat(active, cell, 0), cell, 1), 1)
    m = np.pad(mask.astype(bool), 1)
    k = np.arange(scale + 1)
    # between pixels (a-1, b-1) and (a-1, b): the squares X = b*scale - 1 along pixel row a-1
    a, b = np.nonzero((m[:, :-1] != m[:, 1:]) & ~(active[:, :-1] & active[:, 1:]))
    Yv, Xv = ((a - 1) * scale - 1)[:, None] + k, np.repeat((b * scale - 1)[:, None], scale + 1, 1)
    # between pixels (a-1, b-1) and (a, b-1): the squares Y = a*scale - 1 along pixel column b-1
    a, b = np.nonzero((m[:-1] != m[1:]) & ~(active[:-1] & active[1:]))
    Yh, Xh = np.repeat((a * scale - 1)[:, None], scale + 1, 1), ((b - 1) * scale - 1)[:, None] + k
    return np.concatenate([Yv.ravel(), Yh.ravel()]), np.concatenate([Xv.ravel(), Xh.ravel()])


def trace_rings(occ, cell, scale):
    """Chain the occupancy boundary into closed rings (padded-frame coordinates).

    Squares inside a cell come straight from its canvas. Squares on cell borders read
    neighbouring canvases, or the seed mask where the neighbour is inactive, and are
    de-duplicated by their global lattice position. Object outlines come out with
    negative shoelace area (x_i y_{i+1} - x_{i+1} y_i, y down), hole outlines positive."""
    canvas, cy, cx, mask = occ["canvas"], occ["cy"], occ["cx"], occ["mask"]
    N = len(cy)
    if not N:
        return {"rings": [], "is_hole": [], "crossings": 0}
    S = cell * scale
    H, W = mask.shape
    Wg = W * scale + 3
    ins = canvas >= 0

    # squares inside each cell
    code = (ins[:, :-1, :-1].astype(np.uint8) | ins[:, :-1, 1:] << 1 | ins[:, 1:, 1:] << 2
            | ins[:, 1:, :-1] << 3)
    k, i, j = np.nonzero((code != 0) & (code != 15))
    inner_code = code[k, i, j]
    inner_vals = np.stack([canvas[k, i, j], canvas[k, i, j + 1], canvas[k, i + 1, j + 1],
                           canvas[k, i + 1, j]], 1)
    inner_Y, inner_X = cy[k] * S + i, cx[k] * S + j
    del code

    # squares on cell borders (top-left sample in local row/col -1 or S-1)
    grid = np.full((H // cell, W // cell), -1, np.int64)
    grid[cy, cx] = np.arange(N)
    fill = (mask.astype(np.float32) * 2 - 1) * LOGIT

    def sample(Y, X):
        # Beyond the lattice is background, so every boundary closes inside it.
        inside = (Y >= 0) & (X >= 0) & (Y < H * scale) & (X < W * scale)
        Y = np.clip(Y, 0, H * scale - 1)
        X = np.clip(X, 0, W * scale - 1)
        owner = np.where(inside, grid[Y // S, X // S], -1)
        values = np.where(inside, fill[Y // scale, X // scale], -LOGIT)
        active = owner >= 0
        values[active] = canvas[owner[active], Y[active] % S, X[active] % S]
        return values

    r = np.arange(-1, S)
    I, J = np.meshgrid(r, r, indexing="ij")
    edge = (I == -1) | (J == -1) | (I == S - 1) | (J == S - 1)
    Y = (cy[:, None] * S + I[edge][None]).ravel()
    X = (cx[:, None] * S + J[edge][None]).ravel()
    eY, eX = _mask_edge_squares(mask, cy, cx, cell, scale)
    Y, X = np.concatenate([Y, eY]), np.concatenate([X, eX])
    _, first = np.unique((Y + 1) * Wg + (X + 1), return_index=True)
    Y, X = Y[first], X[first]
    vals = np.stack([sample(Y, X), sample(Y, X + 1), sample(Y + 1, X + 1), sample(Y + 1, X)], 1)
    border_code = ((vals >= 0) * (1 << np.arange(4))).sum(1)
    mixed = (border_code != 0) & (border_code != 15)

    Y = np.concatenate([inner_Y, Y[mixed]])
    X = np.concatenate([inner_X, X[mixed]])
    code = np.concatenate([inner_code, border_code[mixed]]).astype(np.int64)
    vals = np.concatenate([inner_vals, vals[mixed]])

    counts = _COUNTS[code]
    square = np.repeat(np.arange(len(code)), counts)
    slot = np.arange(len(square)) - np.repeat(np.cumsum(counts) - counts, counts)
    e_in, e_out = _PAIRS[code[square], slot, 0], _PAIRS[code[square], slot, 1]
    Ys, Xs = Y[square], X[square]

    def edge_id(e):
        return ((Ys + _EDGE_DY[e] + 1) * Wg + (Xs + _EDGE_DX[e] + 1)) * 2 + _EDGE_VERTICAL[e]

    id_in, id_out = edge_id(e_in), edge_id(e_out)
    v0 = vals[square, _EDGE_FROM[e_out]]
    v1 = vals[square, _EDGE_TO[e_out]]
    t = np.clip(-v0 / (v1 - v0), 0, 1)
    vertical = _EDGE_VERTICAL[e_out].astype(bool)
    px = (Xs + _EDGE_DX[e_out] + 0.5 + np.where(vertical, 0, t)) / scale
    py = (Ys + _EDGE_DY[e_out] + 0.5 + np.where(vertical, t, 0)) / scale
    points = np.stack([px, py], 1)
    # Orientation comes from the lattice-edge midpoints (exact, |area| >= 1/2 lattice
    # square): interpolated vertices can collapse onto one sample (a lone sample just
    # below 0 inside an object), leaving a ring of ~0 area whose sign is rounding noise.
    mid = np.stack([Xs + _EDGE_DX[e_out] + np.where(vertical, 0.5, 1.0),
                    Ys + _EDGE_DY[e_out] + np.where(vertical, 1.0, 0.5)], 1)

    order = np.argsort(id_in, kind="stable")
    sorted_in = id_in[order]
    if len(sorted_in) > 1 and (np.diff(sorted_in) == 0).any():
        raise ValueError("inconsistent occupancy boundary (an edge is entered twice)")
    at = np.minimum(np.searchsorted(sorted_in, id_out), len(sorted_in) - 1)
    if not (sorted_in[at] == id_out).all():
        bad = points[sorted_in[at] != id_out]
        raise ValueError(f"open occupancy boundary at {len(bad)} lattice edges, e.g. "
                         f"x={bad[0, 0]:.2f}, y={bad[0, 1]:.2f} (padded frame)")
    if not len(points):
        return {"rings": [], "is_hole": [], "crossings": 0}
    order, starts = _cycles(order[at])
    rings = np.split(points[order], starts[1:])
    is_hole = (_shoelace(mid[order], starts) > 0).tolist()
    return {"rings": rings, "is_hole": is_hole, "crossings": len(points)}


def _cycles(successor):
    """Cycles of a permutation: (order, starts) such that order[starts[r]:starts[r + 1]]
    walks cycle r from its smallest index along `successor`, cycles in order of that
    index. A plain walk: pointer jumping in numpy is ~6x slower (random gathers)."""
    succ = successor.tolist()
    visited = bytearray(len(succ))
    order, starts = [], []
    for start in range(len(succ)):
        if visited[start]:
            continue
        starts.append(len(order))
        s = start
        while not visited[s]:
            visited[s] = 1
            order.append(s)
            s = succ[s]
    return np.array(order, np.int64), np.array(starts, np.int64)


def _shoelace(points, starts):
    """Twice the signed area of each ring of concatenated `points` (rings begin at `starts`)."""
    n = len(points)
    nxt = np.arange(1, n + 1)
    nxt[np.r_[starts[1:], n] - 1] = starts
    return np.add.reduceat(points[:, 0] * points[nxt, 1] - points[nxt, 0] * points[:, 1], starts)


def nest_rings(rings, is_hole, min_area, seed=None, min_support=MIN_SEED_SUPPORT):
    """Nesting of closed, mutually disjoint rings, largest first; drops rings below the
    minimum areas ((object, hole) px^2) together with everything inside them.

    With `seed` (bool (H, W) object mask in the rings' pixel frame), a ring is also
    dropped, with everything inside it, when less than `min_support` of its own region
    (its interior minus the rings directly inside it) agrees with the seed: object for
    outlines, background for holes, each within 1 px (the seed is blurred)."""
    import shapely

    R = len(rings)
    lengths = np.array([len(q) for q in rings], int)
    flat = np.concatenate(rings) if R else np.zeros((0, 2))
    area = np.abs(_shoelace(flat, np.cumsum(lengths) - lengths)) / 2 if R else np.zeros(0)
    order = np.argsort(-area, kind="stable")
    rings = [rings[i] for i in order]
    is_hole = np.asarray(is_hole, bool).reshape(-1)[order]
    area = area[order]
    first = np.array([q[0] for q in rings]).reshape(-1, 2)
    parent = np.full(R, -1)
    if R > 1:
        # containers are larger (earlier), so the innermost container of a ring's first
        # vertex is the latest one that contains it
        polygons = shapely.polygons(shapely.linearrings(flat, indices=np.repeat(np.arange(R), lengths)))[order]
        container, inner = shapely.STRtree(shapely.points(first)).query(polygons, predicate="contains")
        ok = container < inner
        np.maximum.at(parent, inner[ok], container[ok])
    depth = np.zeros(R, int)
    small = area < np.where(is_hole, min_area[1], min_area[0])
    unsupported = np.zeros(R, bool)
    if seed is not None:
        unsupported = ~small & (seed_support(rings, is_hole, parent, seed, ~small) < min_support)
    drop = small | unsupported
    for i in range(R):
        if parent[i] >= 0:
            depth[i] = depth[parent[i]] + 1
            drop[i] |= drop[parent[i]]
        if is_hole[i] != bool(depth[i] % 2):
            x, y = first[i]
            raise ValueError(f"ring nesting disagrees with its orientation at x={x:.2f}, y={y:.2f}")
    keep = np.flatnonzero(~drop)
    index = np.full(R + 1, -1)
    index[keep] = np.arange(len(keep))
    return {"rings": [rings[i] for i in keep], "is_hole": is_hole[keep].tolist(),
            "depth": depth[keep].tolist(), "parent": index[parent[keep]].tolist(),
            "dropped_rings": int(R - len(keep)), "unsupported_rings": int(unsupported.sum())}


def seed_support(rings, is_hole, parent, seed, which):
    """Fraction of each selected ring's own region (pixel centers inside it and outside
    its direct children) that the 1-px-dilated seed object (outlines) or seed background
    (holes) covers; 1 where the region holds no pixel center."""
    import cv2
    H, W = seed.shape
    kernel = np.ones((3, 3), np.uint8)
    near = {False: cv2.dilate(seed.astype(np.uint8), kernel).astype(bool),
            True: cv2.dilate((~seed).astype(np.uint8), kernel).astype(bool)}
    support = np.ones(len(rings))
    for i in np.flatnonzero(which):
        q = rings[i]
        x0, y0 = np.maximum(np.floor(q.min(0)).astype(int) - 1, 0)
        x1, y1 = np.minimum(np.ceil(q.max(0)).astype(int) + 1, (W, H))
        if x1 <= x0 or y1 <= y0:
            continue
        region = np.zeros((y1 - y0, x1 - x0), np.uint8)
        # pixel i covers [i, i+1); cv2 samples at integer (center) positions
        to_cv = lambda p: np.round((p - 0.5 - (x0, y0)) * 16).astype(np.int32)
        cv2.fillPoly(region, [to_cv(q)], 1, shift=4)
        children = [to_cv(rings[j]) for j in np.flatnonzero(parent == i)]
        if children:
            cv2.fillPoly(region, children, 0, shift=4)
        own = region.astype(bool)
        if own.any():
            support[i] = near[bool(is_hole[i])][y0:y1, x0:x1][own].mean()
    return support
