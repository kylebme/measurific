"""Export the inference model to the two-graph ONNX layout of `runtime` and check parity.

    python -m edgesr.onnx_export [--model models/cfi_B_seed.pt] [--out models/cfi_B_seed-onnx]
                                 [--parity IMAGE ...] [--overwrite]

Parity runs `cfi_infer.infer` (torch) and `runtime.EdgeModel` on a synthetic disk plus the
given images and fails if a decoded occupancy logit differs by more than --tolerance or
if the lattice crossings disagree.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import REPO_DIR
from .cfi import SCALE
from .predict import DEFAULT_MODEL as DEFAULT_CHECKPOINT
from .predict import FORMAT as CHECKPOINT_FORMAT
from .predict import Predictor
from .runtime import DEFAULT_MODEL, FORMAT, EdgeModel
from .seed import window_inputs

OPSET = 18
CELL = 16            # active-cell size (source px), as cfi_infer.infer
FRAME_PAD = 32       # cfi_infer.FRAME_PAD


class FineGraph(nn.Module):
    """(N, 3, win, win) gathered source/field/mask windows (8-bit DN, mask 0/1) ->
    features around the cell (N, C, cell+2, cell+2) and dilated band logits (N, cell, cell)."""

    def __init__(self, model, margin):
        super().__init__()
        self.fine, self.margin = model.fine, margin

    def forward(self, windows):
        M = self.margin
        f, pb = self.fine(window_inputs(windows[:, 0], windows[:, 1], windows[:, 2]))
        crop = slice(M - 1, M + CELL + 1)
        return f[:, :, crop, crop], F.max_pool2d(pb[:, :, crop, crop], 3, 1, 0)[:, 0]


class DecodeGraph(nn.Module):
    """(K, C*9) c-major 3x3 feature neighbourhoods -> (K, 256) occupancy logits (sy*16+sx)."""

    def __init__(self, model):
        super().__init__()
        w = model.block.weight.reshape(model.block.out_channels, -1)[:SCALE * SCALE]
        self.weight = nn.Parameter(w.detach().clone())
        self.bias = nn.Parameter(model.block.bias[:SCALE * SCALE].detach().clone())

    def forward(self, neighbours):
        return neighbours @ self.weight.t() + self.bias


def _export(module, example, names, outputs, dynamic, path):
    program = torch.onnx.export(module, (example,), dynamo=True, opset_version=OPSET,
                                input_names=names, output_names=outputs, dynamic_shapes=dynamic,
                                external_data=False, optimize=True)
    program.save(str(path))


def export_onnx(checkpoint=DEFAULT_CHECKPOINT, output=DEFAULT_MODEL, overwrite=False):
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    predictor = Predictor(str(checkpoint))
    model = predictor.model.to(memory_format=torch.contiguous_format)
    if not model.seed_mode or model.cfg["decoder"] != "block":
        raise ValueError("only seed-mode models with the block decoder are supported")
    margin = int(np.ceil(model.radius / 4) * 4)          # as cfi_infer.infer
    win = CELL + 2 * margin
    staging = output.with_name(output.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    rng = np.random.default_rng(0)
    windows = np.stack([rng.uniform(0, 255, (5, win, win)), rng.uniform(150, 255, (5, win, win)),
                        rng.integers(0, 2, (5, win, win))], 1).astype(np.float32)
    cells = torch.export.Dim("cells", min=1)
    _export(FineGraph(model, margin).eval(), torch.from_numpy(windows), ["windows"],
            ["features", "band"], {"windows": {0: cells}}, staging / "fine.onnx")
    width = model.cfg["fine_width"]
    _export(DecodeGraph(model).eval(), torch.zeros(7, width * 9), ["neighbours"], ["logits"],
            {"neighbours": {0: torch.export.Dim("pixels", min=1)}}, staging / "decode.onnx")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert ck["format"] == CHECKPOINT_FORMAT
    metadata = {
        "format": FORMAT,
        "model_config": ck["model_config"],
        "geometry": {"cell": CELL, "scale": SCALE, "margin": margin, "window": win,
                     "radius": model.radius, "frame_pad": FRAME_PAD},
        "inference_defaults": {"min_object_area": ck["inference_defaults"]["min_object_area"],
                               "min_hole_area": ck["inference_defaults"]["min_hole_area"],
                               "pixel_band": ck["inference_defaults"]["pixel_band"]},
        # coarse seed: runtime.EdgeModel loads this ONNX coarse model (path relative to the
        # model directory) and uses seed.coarse_seed_full instead of the classical seed
        "seed": ck["inference_defaults"].get("seed", "classical"),
        **({"coarse_model": "../" + ck["inference_defaults"]["coarse_model"].replace(".pt", "-onnx")}
           if ck["inference_defaults"].get("seed") == "coarse" else {}),
        "provenance": {
            "source_checkpoint": os.path.relpath(checkpoint, REPO_DIR),
            "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "training": ck.get("provenance", {}),
            "torch_version": torch.__version__,
            "opset": OPSET,
        },
        "graphs": {name: hashlib.sha256((staging / f"{name}.onnx").read_bytes()).hexdigest()
                   for name in ("fine", "decode")},
    }
    (staging / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    staging.replace(output)
    return output, predictor


def disk(size=192, cx=96.3, cy=95.7, r=40.25, ss=8):
    """A dark area-sampled disk on a backlit field (uint16), for parity checks."""
    g = (np.arange(size * ss) + 0.5) / ss
    cover = ((g[None, :] - cx) ** 2 + (g[:, None] - cy) ** 2 < r * r).reshape(size, ss, size, ss).mean((1, 3))
    noise = np.random.default_rng(0).normal(0, 150, cover.shape)
    return np.clip(np.round(52000 - 49000 * cover + noise), 0, 65535).astype(np.uint16)


def parity(predictor, onnx_dir, images, tolerance=2e-3):
    """Largest decoded |logit| difference vs torch, and the 99.9% quantile of the distance
    (px) from each torch crossing to the nearest runtime crossing."""
    from scipy.spatial import cKDTree

    from .cfi_infer import infer
    from .predict import read_image
    from .runtime import lattice_crossings, reading

    runtime = EdgeModel(onnx_dir)
    ma = runtime.min_area()
    worst_logit, worst_point = 0.0, 0.0
    for image in images:
        name = str(image) if isinstance(image, (str, Path)) else "synthetic disk"
        img = reading(read_image(image) if isinstance(image, (str, Path)) else image)
        out = infer(predictor.model, img, None, contours=True, cell=CELL,
                    pixel_band=predictor.defaults["pixel_band"], min_area=ma,
                    coarse_model=getattr(predictor, "coarse", None))
        occ = runtime.occupancy(img, ma)
        n, py, px, _, _ = (v.numpy() for v in out["band_index"])
        expected = out["logits"].numpy().reshape(-1, SCALE, SCALE)
        actual = occ["canvas"].reshape(-1, CELL, SCALE, CELL, SCALE).transpose(0, 1, 3, 2, 4)[n, py, px]
        # blocks touched by the min-area filter are compared through the crossings below
        decoded = (np.abs(np.abs(actual) - 8.0) > 1e-6).reshape(len(n), -1).all(1)
        diff = float(np.abs(expected[decoded] - actual[decoded]).max()) if decoded.any() else 0.0
        ref = out["points"].double().numpy()
        pts = lattice_crossings(occ, runtime.cell, runtime.scale) - occ["frame_pad"]
        H0, W0 = img.shape
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] <= W0) & (pts[:, 1] <= H0)]
        dist = cKDTree(pts).query(ref)[0] if len(pts) and len(ref) else np.zeros(0)
        worst_logit = max(worst_logit, diff)
        # a sign flip of a near-zero logit may move a single crossing by < 1/16 px
        worst_point = max(worst_point, float(np.quantile(dist, 0.999)) if len(dist) else 0.0)
        if diff > tolerance or abs(len(pts) - len(ref)) > 1e-3 * len(ref) + 2:
            raise AssertionError(f"{name}: logits differ by {diff:.3g}, crossings {len(pts)} vs {len(ref)}")
    return worst_logit, worst_point, runtime.providers


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--out", default=DEFAULT_MODEL)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--parity", nargs="*", default=[], help="images to compare against torch")
    ap.add_argument("--tolerance", type=float, default=2e-3)
    a = ap.parse_args(argv)
    output, predictor = export_onnx(a.model, a.out, a.overwrite)
    print(f"wrote {output}")
    images = [disk(), *a.parity]
    logit, point, providers = parity(predictor, output, images, a.tolerance)
    print(f"parity on {providers[0]}: max |logit difference| {logit:.3g}, "
          f"99.9% of crossings within {point:.2g} px, over {len(images)} images")


if __name__ == "__main__":
    main()
