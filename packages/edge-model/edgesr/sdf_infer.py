"""Full-image inference for SDFNet with PyTorch (PLAN_v3); the shared numpy pieces
(inputs, padding, small-region filter, contour extraction) live in the torch-free
`sdf_runtime` and are re-exported here.

  pts, seconds = SDFPredictor("work/runs/RUN/final.pt")(img)   # reading in [0, 1], (H, W)

`times` holds the last call's stage times (coarse, inputs, network, post).
"""

import os
import time

import numpy as np
import torch

from . import REPO_DIR
from .sdf_runtime import contour_points, filter_small, frame_inputs, pad_frame  # noqa: F401 (re-exported)
from .sdfnet import load_sdfnet


class SDFPredictor:
    """img (H, W) reading in [0, 1] -> boundary points. Full frame in one pass (edge-padded
    to a multiple of the network's stride plus `pad`), tiled with overlap when `tile` is
    set or on OOM. `amp`: bf16 autocast on CUDA."""

    def __init__(self, ckpt, coarse=None, device="cuda", min_area=(50.0, 8.0), pad=32, tile=None,
                 gpu_fraction=None, amp=True, fast=False):
        """fast (CUDA): channels-last, cuDNN autotuning and torch.compile of the network
        (compiled per frame size; the first call of a new size is slow)."""
        if device == "cuda" and gpu_fraction:
            torch.cuda.set_per_process_memory_fraction(gpu_fraction)
        self.model = load_sdfnet(ckpt).to(device).eval()
        self.device, self.min_area, self.pad, self.tile, self.amp = device, min_area, pad, tile, amp
        self.net = self.model
        self.fast = fast and device == "cuda"
        if self.fast:
            torch.backends.cudnn.benchmark = True
            self.model = self.model.to(memory_format=torch.channels_last)
            self.net = torch.compile(self.model)
        from .coarse_infer import CoarseModel
        self.coarse = CoarseModel(coarse or os.path.join(REPO_DIR, "models", "coarse_v2.pt"), device)
        self.times = {}

    @torch.no_grad()
    def field(self, img):
        """Predicted maps (dict of (H, W) float32 numpy) for a full reading."""
        x, self.times = frame_inputs(img, self.coarse)
        xp, (ys, xs) = pad_frame(x, self.model.multiple, self.pad)
        t = time.perf_counter()
        try:
            out = self._run(xp) if not self.tile else self._tiled(xp)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            out = self._tiled(xp, self.tile or 768)
        self.times["network"] = time.perf_counter() - t
        return {k: v[ys, xs] for k, v in out.items()}

    def _run(self, xp):
        t = torch.from_numpy(xp)[None].to(self.device)
        if self.fast:
            t = t.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda" and self.amp):
            o = self.net(t)
        return {k: o[k][0].float().cpu().numpy() for k in ("sdf", "log_b")}

    def _tiled(self, xp, tile=768, margin=128):
        _, H, W = xp.shape
        m = self.model.multiple
        out = {k: np.zeros((H, W), np.float32) for k in ("sdf", "log_b")}
        step = tile - 2 * margin
        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                ya, xa = max(y0 - margin, 0), max(x0 - margin, 0)
                yb, xb = min(y0 + step + margin, H), min(x0 + step + margin, W)
                yb, xb = ya + (yb - ya) // m * m, xa + (xb - xa) // m * m
                o = self._run(xp[:, ya:yb, xa:xb])
                y1, x1 = min(y0 + step, yb), min(x0 + step, xb)
                for k in out:
                    out[k][y0:y1, x0:x1] = o[k][y0 - ya:y1 - ya, x0 - xa:x1 - xa]
        return out

    def __call__(self, img, backlight=None):
        t = time.perf_counter()
        f = self.field(np.asarray(img, np.float32))
        t2 = time.perf_counter()
        pts = contour_points(filter_small(f["sdf"], self.min_area) if self.min_area else f["sdf"])
        self.times["post"] = time.perf_counter() - t2
        return pts, time.perf_counter() - t
