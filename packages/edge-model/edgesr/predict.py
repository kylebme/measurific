"""Run the selected edge model on an image (inference only; no dataset, cache or GPU needed).

Model: models/cfi_B_seed.pt, the PLAN_v2 "arm B" model: classical seed (background
fit + Otsu + gradient band, as in measurement/.../adapter.py) + the CFI fine model.
Dark-on-light backlit images, 8- or 16-bit grayscale.

CLI:
  python -m edgesr.predict IMAGE [--out points.npy|.json] [--overlay overlay.png]
         [--max-value 4095] [--min-object-area 50] [--min-hole-area 8]
         [--device cpu|cuda] [--threads 16] [--compile]

Python:
  from edgesr.predict import load, read_image
  model = load()                          # models/cfi_B_seed.pt on CPU
  result = model(read_image("part.png"))  # dict: points (N, 2), timings, ...

Output: sub-pixel contour points in source-pixel coordinates (x right, y down; pixel i
covers [i, i+1), so pixel centers are at i + 0.5). These are the 0.5 iso-contour
crossings of the predicted 16x occupancy, i.e. every point where the object boundary
crosses the 1/16 px lattice, unordered. Ordered closed rings: edgesr.runtime.EdgeModel.rings
(torch-free, ONNX).
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from . import REPO_DIR

DEFAULT_MODEL = os.path.join(REPO_DIR, "models", "cfi_B_seed.pt")
FORMAT = "edgesr-cfi-inference-v1"


def export(src, dst, note="", coarse_model="coarse_v2.pt"):
    """Write a slim inference file (config + weights, no optimizer state) from a
    training checkpoint. Models trained with the coarse seed (crop seed_inputs "coarse")
    record it, with the coarse model file (relative to models/) they need."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    seed = {"seed": "coarse", "coarse_model": coarse_model} \
        if ck["config"].get("crop", {}).get("seed_inputs") == "coarse" else {"seed": "classical"}
    torch.save({"format": FORMAT, "model_config": ck["config"]["model"],
                "state_dict": {k: v.float().contiguous() for k, v in ck["model"].items()},
                "inference_defaults": {"min_object_area": 50.0, "min_hole_area": 8.0,
                                       "pixel_band": -1.0, "cell": 16, **seed},
                "provenance": {"training_checkpoint": os.path.relpath(src, REPO_DIR),
                               "train_steps": ck["state"]["step"],
                               "train_seconds": ck["state"]["train_seconds"], "note": note}},
               dst)


def read_image(path):
    """Grayscale uint8/uint16 array from PNG/TIFF/etc. (colour images are converted)."""
    from PIL import Image
    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            a = np.array(im)
            return a.astype(np.uint16) if a.dtype != np.uint16 else a
        if im.mode != "L":
            im = im.convert("L")
        return np.array(im)


class Predictor:
    def __init__(self, path=DEFAULT_MODEL, device="cpu", compile=False, threads=None):
        from .cfi import CFI
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("format") != FORMAT:
            raise ValueError(f"{path} is not an {FORMAT} file")
        if threads:
            torch.set_num_threads(threads)
        self.model = CFI(ck["model_config"])
        self.model.load_state_dict(ck["state_dict"])
        self.model = self.model.to(device).eval().to(memory_format=torch.channels_last)
        self.device = device
        self.defaults = ck["inference_defaults"]
        self.provenance = ck.get("provenance", {})
        self.compiled = {"fine": torch.compile(self.model.fine, dynamic=True)} if compile else None
        self.coarse = None
        if self.defaults.get("seed") == "coarse":
            from .coarse_infer import CoarseModel
            self.coarse = CoarseModel(os.path.join(os.path.dirname(os.path.abspath(path)),
                                                   self.defaults["coarse_model"]), device)

    def __call__(self, image, max_value=None, min_object_area=None, min_hole_area=None):
        """image: 2-D uint8/uint16 (or float in [0, 1]). max_value: full-scale value of the
        sensor (default: dtype max; e.g. 4095 for 12-bit data stored in uint16)."""
        from .cfi_infer import infer
        a = np.asarray(image)
        if a.ndim != 2:
            raise ValueError("expected a 2-D grayscale image")
        if a.dtype.kind in "ui":
            full = float(max_value or np.iinfo(a.dtype).max)
            x = a.astype(np.float32) / full
        else:
            x = a.astype(np.float32) / float(max_value or 1.0)
        ma = (self.defaults["min_object_area"] if min_object_area is None else min_object_area,
              self.defaults["min_hole_area"] if min_hole_area is None else min_hole_area)
        t = time.perf_counter()
        out = infer(self.model, np.clip(x, 0, 1), None, device=self.device, contours=True,
                    pixel_band=self.defaults["pixel_band"], cell=self.defaults["cell"],
                    compiled=self.compiled, min_area=ma, coarse_model=self.coarse)
        return {"points": out["points"].double().cpu().numpy(), "seconds": time.perf_counter() - t,
                "stage_seconds": out["times"], "n_cells": out["n_cells"],
                "n_band_px": out["n_band_px"], "min_area": ma}


def load(path=DEFAULT_MODEL, device="cpu", compile=False, threads=None):
    return Predictor(path, device, compile, threads)


def overlay(image, points, path, stride=1):
    import cv2
    a = np.asarray(image).astype(np.float32)
    a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6)
    rgb = (np.stack([a] * 3, -1) * 255).astype(np.uint8)
    p = np.floor(points[::stride]).astype(int)
    p = p[(p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] < a.shape[1]) & (p[:, 1] < a.shape[0])]
    rgb[p[:, 1], p[:, 0]] = (0, 0, 255)          # BGR red
    cv2.imwrite(path, rgb)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", help="write points to .npy or .json")
    ap.add_argument("--overlay", help="write an overlay PNG (contour points in red)")
    ap.add_argument("--max-value", type=float, help="sensor full scale (default: dtype max)")
    ap.add_argument("--min-object-area", type=float, help="px^2, default 50")
    ap.add_argument("--min-hole-area", type=float, help="px^2, default 8")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--compile", action="store_true", help="torch.compile the fine branch (CPU speedup after warm-up)")
    ap.add_argument("--export", nargs=2, metavar=("TRAINING_CKPT", "DST"),
                    help="maintainers: write a slim inference model file")
    a = ap.parse_args()
    if a.export:
        export(*a.export)
        print(f"wrote {a.export[1]}")
        return
    if not a.image:
        ap.error("IMAGE is required")
    img = read_image(a.image)
    model = load(a.model, a.device, a.compile, a.threads)
    if a.compile:
        model(img, a.max_value, a.min_object_area, a.min_hole_area)      # warm-up/compile
    r = model(img, a.max_value, a.min_object_area, a.min_hole_area)
    print(f"{len(r['points'])} contour points in {r['seconds']:.2f} s "
          f"({', '.join(f'{k} {v:.3f}' for k, v in r['stage_seconds'].items())})")
    if a.out:
        if a.out.endswith(".json"):
            with open(a.out, "w") as f:
                json.dump({"points": r["points"].round(5).tolist(),
                           "coordinates": "source px; x right, y down; pixel i covers [i, i+1)",
                           "min_area": r["min_area"], "model": os.path.basename(a.model)}, f)
        else:
            np.save(a.out, r["points"])
    if a.overlay:
        overlay(img, r["points"], a.overlay)


if __name__ == "__main__":
    main()
