"""Torch-free coarse model (ONNX Runtime + numpy), interchangeable with
coarse_infer.CoarseModel: same call, full_res_logit and band, so seed.coarse_seed_full
works with either.

  from edgesr.coarse_runtime import CoarseRuntime
  cm = CoarseRuntime()                     # models/coarse_v2-onnx
  out = cm(image_reading, valid_mask=roi)  # 512 x 512 part_logit (hysteresis applied), sdf

Export (needs torch): python -m edgesr.coarse_onnx export models/coarse_v2.pt models/coarse_v2-onnx
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from . import REPO_DIR
from .coarse_np import (coarse_inputs, maps_from_output, model_reading, postprocess,
                        resize_axes)

DEFAULT_MODEL = os.path.join(REPO_DIR, "models", "coarse_v2-onnx")
SIZE = 512


class CoarseRuntime:
    def __init__(self, directory=DEFAULT_MODEL, providers=None, threads=None):
        import onnxruntime
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "coarse.json").read_text())
        self.cfg = self.meta["model_config"]
        options = onnxruntime.SessionOptions()
        if threads:
            options.intra_op_num_threads = int(threads)
        self.session = onnxruntime.InferenceSession(str(self.directory / "coarse.onnx"), options,
                                                    providers=providers or ["CPUExecutionProvider"])
        self.multiple = 2 ** (len(self.cfg["widths"]) - 1)

    def maps(self, reading512, valid=None, aspect=1.0):
        cfg = self.cfg
        r = np.asarray(reading512, np.float32)
        if valid is not None:
            r = np.where(valid, r, 0.0).astype(np.float32)
        x = coarse_inputs(r, cfg["in_mode"], cfg.get("aux_inputs", False), valid, aspect)
        h, w = r.shape
        xp = np.pad(x, ((0, 0), (0, (-h) % self.multiple), (0, (-w) % self.multiple)), mode="edge")[None]
        out = self.session.run(None, {"x": xp})[0][0, :, :h, :w]
        m = maps_from_output(out, cfg)
        if valid is not None:
            m["part_logit"] = np.where(valid, m["part_logit"], np.minimum(m["part_logit"], -8.0))
            m["part"] = 1 / (1 + np.exp(-m["part_logit"]))
        res = postprocess(m)
        res["sdf"] = m["sdf"]
        return res

    def __call__(self, image, max_value=None, valid_mask=None):
        """image: 2-D reading in [0, 1] (float) or uint8/uint16 (scaled by max_value/dtype)."""
        t = time.perf_counter()
        a = np.asarray(image)
        if a.dtype.kind in "ui":
            a = a.astype(np.float32) / float(max_value or np.iinfo(a.dtype).max)
        H, W = a.shape
        valid = None
        if valid_mask is not None:
            valid = resize_axes(np.asarray(valid_mask, np.float32), SIZE, SIZE) >= 0.5
        out = self.maps(model_reading(a, SIZE), valid, W / H)
        out["seconds"] = time.perf_counter() - t
        out["frame_shape"] = (H, W)
        return out

    @staticmethod
    def _up(a, shape):
        H, W = shape
        return cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    def full_res_logit(self, out, shape):
        return self._up(out["part_logit"], shape)

    def band(self, out, shape, width_px=8.0):
        H, W = shape
        sdf = self._up(out["sdf"], shape) * (max(H, W) / SIZE)
        return (np.abs(sdf) <= width_px) | (np.abs(self.full_res_logit(out, shape)) < 2.0)
