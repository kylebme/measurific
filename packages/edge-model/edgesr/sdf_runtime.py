"""Torch-free SDFNet inference (ONNX Runtime + numpy/OpenCV) and the shared pieces of the
SDFNet pipeline (PLAN_v3): input channels, frame padding, small-region filter and
contour extraction.

  from edgesr.sdf_runtime import SDFRuntime
  rt = SDFRuntime("models/sdf_v5-onnx")          # needs ../coarse_v2-onnx
  points, seconds = rt(reading)                   # reading (H, W) in [0, 1]

The contour is the 0-level of the predicted truncated signed distance (+ inside), read at
pixel centres. Inside every 2x2 block of pixel centres the field is interpolated
bilinearly; its crossings with a 1/`sub` px lattice give densified boundary points (the
density of the exact GT crossings the benchmarks use). Small regions (object < 50 px^2,
hole < 8 px^2 by default, as for CFI) are removed by flipping their sign first.

Export (needs torch): python -m edgesr.sdf_export RUN/final.pt sdf_v5
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from . import REPO_DIR

FORMAT = "edgesr-sdf-onnx-v1"


def model_input(source, field, prob, img):
    """The 4 SDFNet input channels: seed.model_inputs (field-normalized reading, coarse
    prior 2p - 1, field level) and the raw reading."""
    from .seed import model_inputs
    return np.concatenate([model_inputs(source, field, prob), ((img.astype(np.float32) - 0.5) * 2.0)[None]], 0)


def contour_points(sdf, sub=16, offset=(0.0, 0.0)):
    """Boundary points (N, 2) x/y in source px of the 0-level of `sdf` sampled at pixel
    centres (pixel i covers [i, i+1)), plus `offset`."""
    s = np.asarray(sdf, np.float32)
    v00, v01, v10, v11 = s[:-1, :-1], s[:-1, 1:], s[1:, :-1], s[1:, 1:]
    mn = np.minimum(np.minimum(v00, v01), np.minimum(v10, v11))
    mx = np.maximum(np.maximum(v00, v01), np.maximum(v10, v11))
    qi, qj = np.nonzero((mn < 0) & (mx >= 0))
    if not len(qi):
        return np.zeros((0, 2))
    a, b, c, d = (v[qi, qj][:, None] for v in (v00, v01, v10, v11))
    k = (np.arange(sub, dtype=np.float32) / sub)[None]
    out = []
    # horizontal lattice lines y = k/sub inside the quad: f(u) linear in u
    left, right = a + (c - a) * k, b + (d - b) * k                # (Q, sub)
    m = (left < 0) != (right < 0)
    u = left / np.where(m, left - right, 1.0)
    qq, kk = np.nonzero(m)
    out.append(np.stack([qj[qq] + 0.5 + u[qq, kk], qi[qq] + 0.5 + k[0, kk]], 1))
    # vertical lattice lines x = k/sub
    top, bot = a + (b - a) * k, c + (d - c) * k
    m = (top < 0) != (bot < 0)
    v = top / np.where(m, top - bot, 1.0)
    qq, kk = np.nonzero(m)
    out.append(np.stack([qj[qq] + 0.5 + k[0, kk], qi[qq] + 0.5 + v[qq, kk]], 1))
    return np.concatenate(out).astype(np.float64) + np.asarray(offset)


def filter_small(sdf, min_area=(50.0, 8.0)):
    """Flip the sign of objects (8-connected, + regions) below min_area[0] px^2 and holes
    (4-connected, - regions) below min_area[1]."""
    s = sdf.copy()
    for sign, conn, limit in ((1, 8, min_area[0]), (-1, 4, min_area[1])):
        if limit <= 0:
            continue
        m = (s >= 0).astype(np.uint8) if sign > 0 else (s < 0).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=conn)
        small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < limit)
        small = small[small > 0]
        if len(small):
            sel = np.isin(lab, small)
            s[sel] = -sign * np.maximum(np.abs(s[sel]), 1e-3)
    return s


def frame_inputs(img, coarse):
    """The 4 SDFNet input channels for a full reading img (H, W) in [0, 1], with stage
    times: the coarse model (torch CoarseModel or ONNX CoarseRuntime) gives the part
    probability, and the backlight field is fitted on its confident background (as
    seed.coarse_seed_full, without the band, which SDFNet does not use)."""
    from .seed import coarse_background, eval_field_grid, fit_field_masked
    t0 = time.perf_counter()
    out = coarse(img)
    t1 = time.perf_counter()
    logit = coarse.full_res_logit(out, img.shape)
    prob = (1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))).astype(np.float32)
    source = img.astype(np.float32) * 255.0
    coef, lvl = fit_field_masked(source, coarse_background(out["part_logit"], img.shape))
    field = eval_field_grid(coef, lvl, img.shape[1], img.shape[0])
    x = model_input(source, field, prob, img)
    return x, {"coarse": t1 - t0, "inputs": time.perf_counter() - t1}


def pad_frame(x, multiple, pad=32):
    """Edge-pad (C, H, W) by `pad` and up to a multiple of `multiple`; returns the array
    and the crop slices back to the frame."""
    _, H, W = x.shape
    ph, pw = pad + (-(H + 2 * pad)) % multiple, pad + (-(W + 2 * pad)) % multiple
    return np.pad(x, ((0, 0), (pad, ph), (pad, pw)), mode="edge"), (slice(pad, pad + H), slice(pad, pad + W))


class SDFRuntime:
    """ONNX Runtime SDFNet + ONNX coarse model. `times` holds the last call's stage times."""

    def __init__(self, directory=os.path.join(REPO_DIR, "models", "sdf_v5-onnx"), threads=None, providers=None):
        import onnxruntime
        from .coarse_runtime import CoarseRuntime
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "model.json").read_text())
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"unsupported SDF model format {self.meta.get('format')!r}")
        options = onnxruntime.SessionOptions()
        if threads:
            options.intra_op_num_threads = int(threads)
        providers = providers or ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(str(self.directory / "sdfnet.onnx"), options, providers=providers)
        self.coarse = CoarseRuntime(self.directory / self.meta["coarse_model"], providers, threads)
        d = self.meta["inference_defaults"]
        self.multiple, self.pad = int(self.meta["multiple"]), int(d["pad"])
        self.min_area = (float(d["min_object_area"]), float(d["min_hole_area"]))
        self.times = {}

    def field(self, img):
        x, self.times = frame_inputs(np.asarray(img, np.float32), self.coarse)
        xp, (ys, xs) = pad_frame(x, self.multiple, self.pad)
        t = time.perf_counter()
        sdf, log_b = self.session.run(["sdf", "log_b"], {"x": np.ascontiguousarray(xp[None], np.float32)})
        self.times["network"] = time.perf_counter() - t
        return {"sdf": sdf[0][ys, xs], "log_b": log_b[0][ys, xs]}

    def __call__(self, img, backlight=None):
        t = time.perf_counter()
        f = self.field(img)
        t2 = time.perf_counter()
        pts = contour_points(filter_small(f["sdf"], self.min_area) if self.min_area else f["sdf"])
        self.times["post"] = time.perf_counter() - t2
        return pts, time.perf_counter() - t
