"""ONNX export and CPU latency of the coarse model.

  python -m edgesr.coarse_onnx export work/runs/ID/final.pt OUT_DIR     # coarse.onnx + coarse.json
  python -m edgesr.coarse_onnx latency work/runs/ID/final.pt [--threads 4 8 16]

Latency covers the whole coarse stage on a full-resolution frame: area resampling to
512 x 512 and input normalization (numpy), the network (onnxruntime, CPU) and
the instance split (numpy/OpenCV).
"""

import argparse
import io
import json
import os
import time

import numpy as np
import torch

from .coarse import CoarseNet, coarse_inputs, in_channels, instances, model_reading


def load(ckpt):
    """Training checkpoint or slim coarse_infer file -> (model, provenance dict)."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "state_dict" in ck:
        cfg, sd, prov = ck["model_config"], ck["state_dict"], ck.get("provenance", {})
    else:
        cfg, sd, prov = ck["config"]["model"], ck["model"], {"train_steps": ck["state"]["step"]}
    model = CoarseNet(cfg)
    model.load_state_dict(sd)
    return model.eval(), prov


def parity(model, sess, seed=0):
    """Max |torch - onnxruntime| of the outputs on a synthetic squashed frame with a ROI."""
    rng = np.random.default_rng(seed)
    img = np.full((1552, 2328), 0.8, np.float32)
    yy, xx = np.mgrid[:1552, :2328]
    img[(yy - 700) ** 2 + (xx - 1100) ** 2 < 400 ** 2] = 0.1
    img += rng.normal(0, 0.01, img.shape).astype(np.float32)
    r = model_reading(np.clip(img, 0, 1))
    valid = ((np.arange(512)[None, :] - 256) ** 2 + (np.arange(512)[:, None] - 256) ** 2) < 250 ** 2
    x = coarse_inputs(np.where(valid, r, 0), model.cfg["in_mode"], model.cfg.get("aux_inputs", False),
                      valid, 2328 / 1552)[None]
    with torch.no_grad():
        a = model(torch.from_numpy(x)).numpy()
    b = sess.run(None, {"x": x})[0]
    return float(np.abs(a - b).max())


def onnx_bytes(model):
    x = torch.randn(1, in_channels(model.cfg), 512, 512)
    f = io.BytesIO()
    torch.onnx.export(model, x, f, input_names=["x"], output_names=["maps"], dynamo=False,
                      opset_version=17, dynamic_axes={"x": {2: "h", 3: "w"}, "maps": {2: "h", 3: "w"}})
    return f.getvalue()


def export(ckpt, out_dir):
    model, prov = load(ckpt)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "coarse.onnx"), "wb") as f:
        f.write(onnx_bytes(model))
    diff = parity(model, session(model, 8))
    print(f"torch vs onnxruntime: max |diff| {diff:.2e}")
    meta = {"format": "edgesr-coarse-onnx-v1", "model_config": model.cfg, "parity_max_abs": diff,
            "input": "full frame area-resampled to 512 x 512 (edgesr.coarse.model_reading), "
                     "channels from edgesr.coarse.coarse_inputs (reading, [valid ROI, log2 aspect])",
            "outputs": ["part logit", "contact logit", "sdf / sdf_clip"],
            "source": os.path.basename(ckpt), "provenance": prov}
    with open(os.path.join(out_dir, "coarse.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {out_dir}")


def session(model, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_bytes(model), so, providers=["CPUExecutionProvider"])


def coarse_stage(sess, img, cfg, multiple):
    t = [time.perf_counter()]
    r = model_reading(img)
    x = coarse_inputs(r, cfg["in_mode"], cfg.get("aux_inputs", False), None, img.shape[1] / img.shape[0])
    t.append(time.perf_counter())
    h, w = r.shape
    xp = np.pad(x, ((0, 0), (0, (-h) % multiple), (0, (-w) % multiple)), mode="edge")[None]
    out = sess.run(None, {"x": xp})[0][0, :, :h, :w]
    t.append(time.perf_counter())
    part = 1 / (1 + np.exp(-out[0]))
    contact = 1 / (1 + np.exp(-out[1]))
    lab = instances(part, contact)
    t.append(time.perf_counter())
    return lab, np.diff(t)


def latency(ckpt, threads=(4, 8, 16), shapes=((2048, 2048), (1552, 2328), (2048, 2448)), reps=10):
    model, _ = load(ckpt)
    rng = np.random.default_rng(0)
    res = []
    for th in threads:
        sess = session(model, th)
        for shp in shapes:
            img = np.clip(0.8 + rng.normal(0, 0.01, shp), 0, 1).astype(np.float32)
            yy, xx = np.mgrid[:shp[0], :shp[1]]
            img[(yy - shp[0] / 2) ** 2 + (xx - shp[1] / 2) ** 2 < (shp[0] / 4) ** 2] = 0.1
            for _ in range(3):
                coarse_stage(sess, img, model.cfg, model.multiple)
            ts = np.array([coarse_stage(sess, img, model.cfg, model.multiple)[1]
                           for _ in range(reps)])
            med = np.median(ts, 0) * 1e3
            rec = {"threads": th, "shape": list(shp), "prep_ms": med[0], "net_ms": med[1],
                   "post_ms": med[2], "total_ms": float(med.sum())}
            res.append(rec)
            print(f"threads {th:2d} {shp[1]}x{shp[0]}: prep {med[0]:.1f} ms, net {med[1]:.1f} ms, "
                  f"post {med[2]:.1f} ms, total {med.sum():.1f} ms", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "latency"])
    ap.add_argument("ckpt")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--threads", type=int, nargs="*", default=[4, 8, 16])
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.ckpt, a.out)
    else:
        res = latency(a.ckpt, a.threads)
        if a.out:
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
