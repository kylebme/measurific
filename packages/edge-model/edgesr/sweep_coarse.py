"""Post-processing sweep for a trained coarse model on the macro val benchmark: part
threshold, contact threshold (instance split) and minimum component area. Network
outputs are computed once; each setting re-runs only the post-processing and scoring.

  python -m edgesr.sweep_coarse work/runs/ID/final.pt [--contact 0.2 0.3 0.5] [--min-area 4 16]
"""

import argparse
import json
import os

import numpy as np

from . import bench_coarse as BC
from .coarse import instances
from .macro import MacroStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--part", type=float, nargs="*", default=[0.5])
    ap.add_argument("--contact", type=float, nargs="*", default=[0.2, 0.3, 0.5])
    ap.add_argument("--min-area", type=float, nargs="*", default=[4, 16])
    a = ap.parse_args()
    store = MacroStore()
    predict = BC.model_predictor(a.ckpt)
    idx = BC.eval_set(store, a.split)
    cache = []
    for i in idx:
        sc = store.scenes[i]
        p = predict(BC.export_reading(sc["scene_id"], a.split))
        cache.append((i, p["part_logit"].astype(np.float32), p["contact"].astype(np.float16)))
    results = []
    for pt in a.part:
        shift = float(np.log(pt / (1 - pt)))
        for ct in a.contact:
            for ma in a.min_area:
                recs = []
                for i, logit, contact in cache:
                    lg = logit - shift
                    lab = instances(1 / (1 + np.exp(-lg)), contact.astype(np.float32) + (0.5 - ct))
                    recs.append(BC.score_image({"part_logit": lg, "instances": lab},
                                               np.asarray(store.inst[i]), np.asarray(store.debris[i]),
                                               store.scenes[i], ma))
                s = BC.summarize(recs)
                s.update({"part_thresh": pt, "contact_thresh": ct, "min_area": ma})
                results.append(s)
                print(f"part {pt:.2f} contact {ct:.2f} min_area {ma:4.0f}: " +
                      " ".join(f"{k}={s[k]:.4f}" for k in BC.HEAD), flush=True)
    name = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    out = os.path.join(BC.BENCH_DIR, f"sweep_{name}_{a.split}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
