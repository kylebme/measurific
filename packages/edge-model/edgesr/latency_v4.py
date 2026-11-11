"""CPU latency of exported edge models (runtime.EdgeModel, ONNX) on val_v4_full frames.

Waits until no other edgesr training or benchmark process is running, so it can sit in
the train lane of the queue between two runs and measure an idle machine. Writes
work/runs/<run-id>/metrics.json (the queue's completion marker).

  python -m edgesr.latency_v4 --run-id lat_v4 --models models/cfi_B_seed-onnx models/cfi_v4_w32-onnx
"""

import argparse
import glob
import json
import os
import statistics
import time

import numpy as np

from . import REPO_DIR, WORK_DIR
from . import bench_v4 as B4
from . import sampler as S
from .runtime import EdgeModel


def busy():
    """Other edgesr processes (training, benchmarks, cache builds), excluding the queue
    runner and this process's ancestors."""
    mine = {os.getpid(), os.getppid()}
    found = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) in mine:
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if "edgesr." in cmd and "edgesr.queue" not in cmd and "edgesr.latency_v4" not in cmd:
            found.append(cmd[:120])
    return found


def wait_idle(max_minutes):
    t = time.time()
    while time.time() - t < max_minutes * 60:
        if not busy():
            time.sleep(20)          # let page cache writeback and stragglers settle
            if not busy():
                return True
        time.sleep(30)
    return False


def measure(directory, frames, threads, repeats):
    model = EdgeModel(directory, threads=threads)
    model(frames[0])                                   # warm-up (session init, allocations)
    calls, rings, failures = [], [], []
    for k, img in enumerate(frames):
        for _ in range(repeats):
            r = model(img)
            calls.append({"seconds": r["seconds"], **r["stage_seconds"],
                          "n_cells": r["n_cells"], "n_band_px": r["n_band_px"]})
        t = time.perf_counter()
        try:
            model.rings(img)
            rings.append(time.perf_counter() - t)
        except ValueError as e:       # ring tracing/nesting failures are reported, not fatal
            failures.append(f"image {k}: {e}")
    med = {k: statistics.median(c[k] for c in calls) for k in calls[0]}
    return {"model": str(directory), "fine_width": model.metadata["model_config"]["fine_width"],
            "seed": model.metadata.get("seed", "classical"), "median": med,
            "p90_seconds": float(np.percentile([c["seconds"] for c in calls], 90)),
            "rings_median_seconds": statistics.median(rings) if rings else None,
            "rings_failures": failures}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--models", nargs="+", help="ONNX model directories (default: models/cfi_*-onnx)",
                    default=sorted(glob.glob(os.path.join(REPO_DIR, "models", "cfi_*-onnx"))))
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--wait-minutes", type=float, default=90)
    a = ap.parse_args()
    idle = wait_idle(a.wait_minutes)
    load = os.getloadavg()
    store = S.open_store({"cache": "v4"})
    spec = json.load(open(B4.spec_path("val_v4_full")))
    by_id = {s["scene_id"]: i for i, s in enumerate(store.scenes)}
    frames = [B4.reading(store, by_id[im["scene_id"]], im["aug_seed"])[0]
              for im in spec["images"][:a.images]]
    results = [measure(d, frames, a.threads, a.repeats) for d in a.models]
    out = {"run_id": a.run_id, "desc": a.desc, "idle": idle, "busy": busy(), "loadavg_at_start": load,
           "threads": a.threads, "frame_shape": list(frames[0].shape), "images": len(frames),
           "repeats": a.repeats, "results": results}
    run_dir = os.path.join(WORK_DIR, "runs", a.run_id)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(out, open(os.path.join(run_dir, "metrics.json"), "w"), indent=1)
    for r in results:
        m = r["median"]
        rs = r["rings_median_seconds"]
        print(f"{r['model']}: median {m['seconds']:.3f} s (p90 {r['p90_seconds']:.3f}), rings "
              f"{'n/a' if rs is None else f'{rs:.3f} s'} ({len(r['rings_failures'])} failed) | " +
              " ".join(f"{k} {m[k]:.3f}" for k in ("seed", "fine", "decode", "canvas")) +
              f" | cells {m['n_cells']:.0f} band px {m['n_band_px']:.0f}", flush=True)


if __name__ == "__main__":
    main()
