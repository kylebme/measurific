"""Coarse model inference on full-resolution frames (the input side of the fine model).

  from edgesr.coarse_infer import CoarseModel
  cm = CoarseModel()                       # models/coarse_v2.pt, CPU (device="cuda" for GPU)
  out = cm(image, valid_mask=roi)          # 2-D uint8/uint16 (or float in [0, 1]) full frame;
                                           # roi: optional full-res bool region of interest
  out["part_logit"]  (512, 512) float32    model-frame part logit (> 0 = part)
  out["sdf"]         (512, 512) float32    signed distance to the part boundary, model px
  out["instances"]   (512, 512) int32      part instances (touching parts may merge)
  logit = cm.full_res_logit(out, image.shape)     # bilinear back to the frame
  band = cm.band(out, image.shape, 8.0)           # frame pixels within ~8 source px of the
                                                  # coarse boundary (bool)

Every frame is resampled to 512 x 512 (area), so non-square frames are squashed; the
model was trained on squashed frames and is given log2(width / height) and the region of
interest (zero reading and "invalid" outside it). Model pixel (j, k) covers source pixels
[k W/512, (k+1) W/512) x [j H/512, (j+1) H/512); cv2 INTER_LINEAR resizing back uses the
same half-pixel-center convention.

CLI:
  python -m edgesr.coarse_infer IMAGE [--overlay out.png] [--max-value 4095] [--roi-circle]
  python -m edgesr.coarse_infer --export work/runs/ID/final.pt models/coarse_v2.pt
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch

from . import REPO_DIR
from .coarse import CoarseNet, model_reading, postprocess, predict_maps, resize_axes

DEFAULT_MODEL = os.path.join(REPO_DIR, "models", "coarse_v2.pt")
FORMAT = "edgesr-coarse-v1"
SIZE = 512


def export(src, dst, note=""):
    """Slim inference file (config + weights) from a training checkpoint."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    torch.save({"format": FORMAT, "model_config": ck["config"]["model"], "input_size": SIZE,
                "state_dict": {k: v.float().contiguous() for k, v in ck["model"].items()},
                "provenance": {"training_checkpoint": os.path.relpath(src, REPO_DIR),
                               "train_steps": ck["state"]["step"],
                               "train_seconds": ck["state"]["train_seconds"], "note": note}}, dst)


def read_reading(image, max_value=None):
    """2-D uint8/uint16/float image -> float32 reading in [0, 1]."""
    a = np.asarray(image)
    if a.ndim != 2:
        raise ValueError("expected a 2-D grayscale image")
    if a.dtype.kind in "ui":
        return a.astype(np.float32) / float(max_value or np.iinfo(a.dtype).max)
    return np.clip(a.astype(np.float32) / float(max_value or 1.0), 0, 1)


def inscribed_circle(shape):
    """Full-res bool mask of the circle inscribed in the frame (a common circular ROI)."""
    H, W = shape
    yy, xx = np.ogrid[:H, :W]
    r = min(H, W) / 2
    return (xx + 0.5 - W / 2) ** 2 + (yy + 0.5 - H / 2) ** 2 <= r * r


class CoarseModel:
    def __init__(self, path=DEFAULT_MODEL, device="cpu", threads=None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("format") == FORMAT:
            cfg, sd = ck["model_config"], ck["state_dict"]
        else:                                   # a training checkpoint
            cfg, sd = ck["config"]["model"], ck["model"]
        if threads:
            torch.set_num_threads(threads)
        self.model = CoarseNet(cfg)
        self.model.load_state_dict(sd)
        self.model = self.model.to(device).eval()
        self.device = device
        self.cfg = self.model.cfg

    def maps(self, reading, valid=None, aspect=1.0):
        """Model-frame reading (512, 512) in [0, 1] -> maps dict (see module docstring)."""
        m = predict_maps(self.model, reading, self.device, valid, aspect)
        out = postprocess(m)
        out["sdf"] = m["sdf"]
        return out

    def __call__(self, image, max_value=None, valid_mask=None):
        t = time.perf_counter()
        x = read_reading(image, max_value)
        H, W = x.shape
        valid = None
        if valid_mask is not None:
            valid = resize_axes(np.asarray(valid_mask, np.float32), SIZE, SIZE) >= 0.5
        out = self.maps(model_reading(x, SIZE), valid, W / H)
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
        """Frame pixels within width_px source px of the coarse part boundary (the sdf in
        model px scaled by the larger source/model pixel ratio, so the band is at least as
        wide as asked on both axes), or wherever the upsampled logit is uncertain."""
        H, W = shape
        sdf = self._up(out["sdf"], shape) * (max(H, W) / SIZE)
        return (np.abs(sdf) <= width_px) | (np.abs(self.full_res_logit(out, shape)) < 2.0)


def overlay(image, out, path, valid_mask=None):
    a = read_reading(image)
    g = (np.clip(a / max(float(a.max()), 1e-6), 0, 1) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    edge = (CoarseModel._up(out["part_logit"], a.shape) > 0).astype(np.uint8)
    edge = cv2.dilate(edge ^ cv2.erode(edge, np.ones((3, 3), np.uint8)), np.ones((2, 2), np.uint8))
    rgb[edge > 0] = (0, 0, 255)
    if valid_mask is not None:
        v = valid_mask.astype(np.uint8)
        rgb[(v ^ cv2.erode(v, np.ones((3, 3), np.uint8))) > 0] = (255, 160, 0)
    cv2.imwrite(path, rgb)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-value", type=float)
    ap.add_argument("--roi-circle", action="store_true", help="use the inscribed circle as the ROI")
    ap.add_argument("--overlay")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--export", nargs=2, metavar=("TRAINING_CKPT", "DST"))
    a = ap.parse_args()
    if a.export:
        export(*a.export)
        print(f"wrote {a.export[1]}")
        return
    from .predict import read_image
    img = read_image(a.image)
    roi = inscribed_circle(img.shape) if a.roi_circle else None
    cm = CoarseModel(a.model, a.device)
    out = cm(img, a.max_value, roi)
    print(f"{int(out['instances'].max())} parts, {out['seconds'] * 1e3:.0f} ms")
    if a.overlay:
        overlay(img, out, a.overlay, roi)


if __name__ == "__main__":
    main()
