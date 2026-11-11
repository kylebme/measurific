"""Speed of SDFNet inference on 2048^2 val_v4_full frames, per stage (coarse model, input
channels, network, post-processing = small-region filter + contour points), on GPU and
CPU, with the accuracy of each path on the same frames as a sanity check.

Paths: gpu_bf16 (PyTorch CUDA, bf16 autocast, eager), gpu_fp32, gpu_bf16_fast and
gpu_fp32_fast (channels-last, cuDNN autotuning, torch.compile), cpu_torch (PyTorch fp32),
cpu_onnx (ONNX Runtime fp32 + ONNX coarse model; torch-free). Waits for an idle machine
(no other edgesr processes) like latency_v4.

  python -m edgesr.latency_sdf models/sdf_v5.pt [--paths gpu_bf16 cpu_onnx] [--images 8]
"""

import argparse
import json
import os
import statistics
import time

import numpy as np
from scipy.spatial import cKDTree

from . import REPO_DIR, WORK_DIR
from . import bench_v4 as B4
from . import sampler as S
from .latency_v4 import busy, wait_idle

STAGES = ("coarse", "inputs", "network", "post")


def make(path, ckpt, threads):
    import torch
    if path.startswith("gpu"):
        from .sdf_infer import SDFPredictor
        return SDFPredictor(ckpt, device="cuda", amp=path in ("gpu_bf16", "gpu_bf16_fast"), fast=path.endswith("_fast"))
    if path == "cpu_torch":
        torch.set_num_threads(threads)
        from .sdf_infer import SDFPredictor
        return SDFPredictor(ckpt, device="cpu", amp=False)
    from .sdf_runtime import SDFRuntime
    onnx_dir = os.path.splitext(ckpt)[0] + "-onnx"
    return SDFRuntime(onnx_dir, threads=threads)


def measure(path, ckpt, frames, gts, threads, repeats):
    p = make(path, ckpt, threads)
    p(frames[0][0])                                     # warm-up (cudnn autotune, allocations)
    if path.startswith("gpu"):
        p(frames[0][0])
    rows, gp = [], []
    for (img, _), gt in zip(frames, gts):
        for r in range(repeats):
            pts, sec = p(img)
            rows.append({"seconds": sec, **{s: p.times.get(s, 0.0) for s in STAGES}})
        gp.append(cKDTree(pts).query(gt, workers=8)[0] if len(pts) else np.full(len(gt), np.inf))
    d = np.concatenate(gp)
    med = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
    return {"path": path, "median": med, "p90_seconds": float(np.percentile([r["seconds"] for r in rows], 90)),
            "missed_rate": float((d > 0.5).mean()), "loc_mean": float(d[d <= 0.5].mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--paths", nargs="+", default=["gpu_bf16", "gpu_fp32", "cpu_onnx", "cpu_torch"])
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--wait-minutes", type=float, default=60)
    ap.add_argument("--out")
    a = ap.parse_args()
    idle = wait_idle(a.wait_minutes)
    store = S.open_store({"cache": "v4"})
    spec = json.load(open(B4.spec_path("val_v4_full")))
    by_id = {s["scene_id"]: i for i, s in enumerate(store.scenes)}
    picks = spec["images"][::max(1, len(spec["images"]) // a.images)][:a.images]
    frames = [B4.reading(store, by_id[im["scene_id"]], im["aug_seed"])[:2] for im in picks]
    gts = [store.boundary_points(store.scenes[by_id[im["scene_id"]]]["mask_index"]).astype(np.float64) for im in picks]
    res = {"ckpt": a.ckpt, "idle": idle, "busy": busy(), "threads": a.threads, "images": len(frames),
           "frame_shape": list(frames[0][0].shape), "repeats": a.repeats, "results": []}
    for path in a.paths:
        r = measure(path, a.ckpt, frames, gts, a.threads, a.repeats)
        res["results"].append(r)
        m = r["median"]
        print(f"{path:9s} median {m['seconds']:.3f} s (p90 {r['p90_seconds']:.3f}) | " +
              " ".join(f"{s} {m[s]:.3f}" for s in STAGES) +
              f" | missed {100 * r['missed_rate']:.2f}% loc {r['loc_mean']:.4f}", flush=True)
    out = a.out or os.path.join(WORK_DIR, "latency_sdf_" + os.path.splitext(os.path.basename(a.ckpt))[0] + ".json")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
