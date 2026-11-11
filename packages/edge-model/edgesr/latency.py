"""Phase-0 latency sweep for CFI on a real 2048^2 val image (random weights).

Band variants:
  px1   - pixels within 1 px of a GT boundary (ideal per-pixel band)
  cell4 - whole 4x4 cells containing a GT boundary, dilated by one cell (what the
          1/4-res band head can select without a per-pixel refinement)

Usage: python -m edgesr.latency [--threads 16] [--device cpu] [--scenes 3]
"""

import argparse
import json
import os
import statistics
import time

import cv2
import numpy as np
import torch

from . import WORK_DIR
from . import sampler as S
from .cfi import CFI
from .cfi_infer import infer
from .fullimage import sensor_image

CONFIGS = {   # dense blocks: radius = 1 + sum(dilation + 1)
    "R5":   {"fine_dilations": [1, 1]},
    "R9":   {"fine_dilations": [1, 1, 1, 1]},
    "R13":  {"fine_dilations": [1, 2, 3, 3]},
    "R17":  {"fine_dilations": [1, 2, 4, 6]},
    "R23":  {"fine_dilations": [2, 4, 8, 5]},
    "R17w32": {"fine_dilations": [1, 2, 4, 6], "fine_width": 32},
}


def gt_bands(store, scene):
    cov = np.asarray(store.cover[store.scenes[scene]["part_index"]])
    edge = ((cov > 0) & (cov < 256)).astype(np.uint8)
    px1 = cv2.dilate(edge, np.ones((3, 3), np.uint8)).astype(bool)
    e4 = edge.reshape(512, 4, 512, 4).max((1, 3))
    e4 = cv2.dilate(e4, np.ones((3, 3), np.uint8))
    cell4 = np.kron(e4, np.ones((4, 4), np.uint8)).astype(bool)
    return {"px1": px1, "cell4": cell4}


def run(args):
    torch.set_num_threads(args.threads)
    store = S.SourceStore()
    val = store.scene_indices("val")
    rng = np.random.default_rng(0)
    scenes = [int(s) for s in rng.choice(val, args.scenes, replace=False)]
    data = []
    for sc in scenes:
        img, bl = sensor_image(store, sc, seed=sc)
        data.append((sc, img.astype(np.float32), bl, gt_bands(store, sc)))
    results = []
    for name in args.configs:
        for dec in args.decoders:
            torch.manual_seed(0)
            model = CFI({**CONFIGS[name], "decoder": dec, "edge_head": True}).eval().to(args.device)
            compiled = None
            if args.compile:
                model = model.to(memory_format=torch.channels_last)
                compiled = {"coarse": torch.compile(model.coarse),
                            "fine": torch.compile(model.fine, dynamic=True)}
            for band_kind in args.bands:
                for cell in args.cells:
                    runs = []
                    for sc, img, bl, bands in data:
                        for _ in range(2):   # warm-up / compile
                            infer(model, img, bl, bands[band_kind], cell=cell, device=args.device,
                                  compiled=compiled)
                        ts = [infer(model, img, bl, bands[band_kind], cell=cell, device=args.device,
                                    compiled=compiled) for _ in range(args.repeats)]
                        runs.append(min(ts, key=lambda r: r["total_s"]))
                    tot = statistics.median(r["total_s"] for r in runs)
                    stages = {k: round(statistics.median(r["times"][k] for r in runs), 3)
                              for k in runs[0]["times"]}
                    rec = {"config": name, "decoder": dec, "band": band_kind, "cell": cell,
                           "compiled": bool(args.compile),
                           "radius": runs[0]["radius"], "margin": runs[0]["margin"],
                           "total_s": round(tot, 3), "stages_s": stages,
                           "band_px": int(statistics.median(r["n_band_px"] for r in runs)),
                           "cells": int(statistics.median(r["n_cells"] for r in runs)),
                           "params": sum(p.numel() for p in model.parameters()),
                           "threads": args.threads, "device": args.device}
                    print(json.dumps(rec), flush=True)
                    results.append(rec)
    os.makedirs(os.path.join(WORK_DIR, "cfi"), exist_ok=True)
    tag = "c" if args.compile else "e"
    with open(os.path.join(WORK_DIR, "cfi", f"latency_{args.device}_{args.threads}t_{tag}.json"), "w") as f:
        json.dump({"scenes": scenes, "results": results}, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--cells", nargs="*", type=int, default=[16, 32])
    ap.add_argument("--decoders", nargs="*", default=["block"])
    ap.add_argument("--bands", nargs="*", default=["px1"])
    ap.add_argument("--compile", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
