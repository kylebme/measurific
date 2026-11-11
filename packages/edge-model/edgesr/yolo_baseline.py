"""Off-the-shelf baseline for the coarse model: Ultralytics YOLO11n-seg fine-tuned on the
macro export (export/data.yaml, 8-bit camera images, one `part` class).

Ultralytics is AGPL-3.0 (commercial licence needed to ship it in closed software); it is
used here only as a comparison. It is not in requirements.txt: it lives in
work/pydeps/yolo (pip --target), put on sys.path by this module.

Usage (also as a queue module entry: python -m edgesr.queue add ID --module edgesr.yolo_baseline):
  python -m edgesr.yolo_baseline --run-id yolo11n_seg_45m [--set hours=0.75 imgsz=512 batch=32]
Writes RUNS_DIR/ID/{weights, metrics.json} and scores the macro val benchmark
(bench_coarse, same images and metrics as the coarse model).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from . import RUNS_DIR, WORK_DIR

PYDEPS = os.path.join(WORK_DIR, "pydeps", "yolo")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(WORK_DIR, "pydeps", "yolo_cfg"))
if PYDEPS not in sys.path:
    sys.path.insert(0, PYDEPS)

DEFAULTS = {"model": "yolo11n-seg.pt", "hours": 0.75, "imgsz": 512, "batch": 32, "workers": 10,
            "conf": 0.25, "iou": 0.6}


def weights_path(name):
    d = os.path.join(WORK_DIR, "pydeps", "weights")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def predictor(weights, cfg, device=0):
    from ultralytics import YOLO
    model = YOLO(weights)

    def predict(x, valid=None, aspect=1.0):
        img = np.repeat((np.clip(x, 0, 1) * 255).astype(np.uint8)[..., None], 3, 2)
        r = model.predict(img, imgsz=cfg["imgsz"], conf=cfg["conf"], iou=cfg["iou"], device=device,
                          retina_masks=True, verbose=False)[0]
        h, w = x.shape
        lab = np.zeros((h, w), np.int32)
        if r.masks is not None:
            m = r.masks.data.cpu().numpy() > 0.5            # (N, h, w), sorted by confidence
            for k in range(len(m) - 1, -1, -1):             # most confident drawn last (on top)
                lab[m[k]] = k + 1
        if valid is not None:
            lab = np.where(valid, lab, 0).astype(np.int32)
        return {"part_logit": np.where(lab > 0, 1.0, -1.0).astype(np.float32), "instances": lab}
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    for s in a.set:
        k, v = s.split("=", 1)
        cfg[k] = json.loads(v) if v[:1] in "0123456789[{-\"tfn" else v
    run_dir = os.path.join(RUNS_DIR, a.run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)                  # stray downloads (AMP check weights, fonts) land here
    from ultralytics import YOLO, settings
    settings.update({"weights_dir": os.path.dirname(weights_path(cfg["model"])),
                     "runs_dir": run_dir, "datasets_dir": os.path.dirname(run_dir)})

    from .bench_coarse import run as bench_run
    from .macro import MACRO_DIR
    best = os.path.join(run_dir, "train", "weights", "best.pt")
    t0 = time.time()
    if not a.eval_only:
        model = YOLO(weights_path(cfg["model"]))
        model.train(data=os.path.join(MACRO_DIR, "export", "data.yaml"), imgsz=cfg["imgsz"],
                    time=cfg["hours"], epochs=1000, batch=cfg["batch"], workers=cfg["workers"],
                    device=0, project=run_dir, name="train", exist_ok=True, single_cls=True,
                    resume=a.resume and os.path.exists(os.path.join(run_dir, "train", "weights", "last.pt")),
                    plots=False, verbose=False)
    train_s = time.time() - t0
    res = bench_run(a.run_id, predictor(best, cfg), "val", None)
    result = {"run_id": a.run_id, "description": a.desc, "kind": "yolo_baseline",
              "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config": cfg,
              "budget": {"train_seconds": round(train_s, 1)},
              "benchmark": {"name": "macro_val_export", "images": res["images"]},
              "primary": {"bscore": res["bscore"]},
              "metrics": {k: v for k, v in res.items() if k != "slices"}, "run_dir": run_dir}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=1)
    if not a.no_ledger:
        from .train_coarse import LEDGER
        with open(LEDGER, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
