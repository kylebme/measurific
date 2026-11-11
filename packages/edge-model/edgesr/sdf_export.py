"""Export an SDFNet training checkpoint (PLAN_v3) for inference:
  models/NAME.pt        slim torch file (config + fp32 weights + inference defaults)
  models/NAME-onnx/     sdfnet.onnx (input x (1, 4, H, W), H and W multiples of the
                        network stride; outputs sdf and log_b (1, H, W)) + model.json,
                        using ../coarse_v2-onnx for the coarse prior
and check ONNX Runtime against torch on a real val frame.

  python -m edgesr.sdf_export work/runs/RUN/final.pt sdf_v5
"""

import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn

from . import REPO_DIR
from .sdf_runtime import FORMAT
from .sdfnet import SDFNet

SLIM_FORMAT = "edgesr-sdf-v1"
DEFAULTS = {"min_object_area": 50.0, "min_hole_area": 8.0, "pad": 32}


class Graph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        o = self.model(x)
        return o["sdf"], o["log_b"]


def export(src, name, opset=17):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    cfg = ck["config"]["model"]
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()
    prov = {"source_checkpoint": os.path.relpath(src, REPO_DIR), "source_checkpoint_sha256": sha,
            "train_state": ck.get("state")}
    models = os.path.join(REPO_DIR, "models")
    torch.save({"format": SLIM_FORMAT, "model_config": cfg,
                "state_dict": {k: v.float().contiguous() for k, v in ck["model"].items()},
                "inference_defaults": DEFAULTS, "coarse_model": "coarse_v2.pt", "provenance": prov},
               os.path.join(models, name + ".pt"))
    model = SDFNet(cfg).eval()
    model.load_state_dict(ck["model"])
    out_dir = os.path.join(models, name + "-onnx")
    os.makedirs(out_dir, exist_ok=True)
    x = torch.zeros(1, cfg.get("in_ch", 4), 256, 256)
    path = os.path.join(out_dir, "sdfnet.onnx")
    torch.onnx.export(Graph(model), (x,), path, input_names=["x"], output_names=["sdf", "log_b"],
                      dynamic_axes={"x": {2: "h", 3: "w"}, "sdf": {1: "h", 2: "w"}, "log_b": {1: "h", 2: "w"}},
                      opset_version=opset, dynamo=False)
    meta = {"format": FORMAT, "model_config": cfg, "multiple": model.multiple, "inference_defaults": DEFAULTS,
            "coarse_model": "../coarse_v2-onnx", "inputs": "sdf_runtime.model_input (4 channels)",
            "provenance": prov}
    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return model, out_dir


def parity(model, out_dir):
    """ORT vs torch (fp32 CPU) network outputs on a real val_v4_full frame's inputs."""
    import onnxruntime
    from . import bench_v4 as B4
    from . import sampler as S
    from .coarse_runtime import CoarseRuntime
    from .sdf_runtime import frame_inputs, pad_frame
    store = S.open_store({"cache": "v4"})
    spec = json.load(open(B4.spec_path("val_v4_full")))
    by_id = {s["scene_id"]: i for i, s in enumerate(store.scenes)}
    img = B4.reading(store, by_id[spec["images"][0]["scene_id"]], spec["images"][0]["aug_seed"])[0]
    xin, _ = frame_inputs(img[512:1536, 256:1792], CoarseRuntime())
    xp, _ = pad_frame(xin, model.multiple, 32)
    with torch.no_grad():
        ref = model(torch.from_numpy(xp)[None])["sdf"][0].numpy()
    sess = onnxruntime.InferenceSession(os.path.join(out_dir, "sdfnet.onnx"), providers=["CPUExecutionProvider"])
    got = sess.run(["sdf"], {"x": xp[None].astype(np.float32)})[0][0]
    near = np.abs(ref) < 2
    return float(np.abs(got - ref).max()), float(np.abs(got - ref)[near].max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("name")
    a = ap.parse_args()
    model, out_dir = export(a.ckpt, a.name)
    mx, near = parity(model, out_dir)
    print(f"wrote models/{a.name}.pt and {os.path.relpath(out_dir, REPO_DIR)}; ORT vs torch sdf: max |diff| "
          f"{mx:.2e} px, near edges (|sdf| < 2) {near:.2e} px")


if __name__ == "__main__":
    main()
