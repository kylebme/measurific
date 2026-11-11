"""Break benchmark metrics down by scene appearance and sensor parameters.

Usage:
  python -m edgesr.analyze work/runs/ID/final.pt [--device cuda]
  python -m edgesr.analyze --classical            # bicubic + mid-level threshold
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import CACHE_DIR, SCALE
from . import benchmark as B
from . import metrics as M


class Classical(nn.Module):
    """Non-learned reference: bicubic 16x upsample, threshold halfway between the
    patch's 2nd and 98th percentile levels (object is dark on the backlight)."""

    def forward(self, x, backlight=None):
        up = F.interpolate(x, scale_factor=SCALE, mode="bicubic", align_corners=False)
        flat = x.flatten(1)
        lo = torch.quantile(flat, 0.02, dim=1)[:, None, None, None]
        hi = torch.quantile(flat, 0.98, dim=1)[:, None, None, None]
        p = (hi - up) / (hi - lo).clamp_min(1e-3)
        return torch.logit(p.clamp(1e-4, 1 - 1e-4))


def breakdown(bench, results, params):
    with open(os.path.join(CACHE_DIR, "scenes.json")) as f:
        scenes = {s["scene_id"]: s for s in json.load(f)["scenes"]}
    keys = {
        "category": lambda i: bench.categories[i],
        "focus": lambda i: scenes[bench.patches[i]["scene_id"]]["focus_class"],
        "illumination": lambda i: scenes[bench.patches[i]["scene_id"]]["illumination_class"],
        "material": lambda i: scenes[bench.patches[i]["scene_id"]]["material_family"],
        "family": lambda i: scenes[bench.patches[i]["scene_id"]]["family"],
        "bits": lambda i: str(params[i]["bits"]),
    }
    out = {}
    for name, fn in keys.items():
        labels = [fn(i) for i in range(len(results))]
        out[name] = M.aggregate(results, labels)["by_category"]
    return out


def print_breakdown(bd):
    cols = ("bscore", "missed_rate", "extra_rate", "loc_error_mean", "loc_error_p95",
            "component_miss_rate")
    for name, groups in bd.items():
        print(f"\n== {name}")
        print(f"{'group':16s} {'n':>5s} " + " ".join(f"{c[:12]:>12s}" for c in cols))
        for g, s in groups.items():
            print(f"{g:16s} {s['n_patches']:5d} " +
                  " ".join(f"{s.get(c, float('nan')):12.4f}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--classical", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", default="val_v1")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.classical:
        model = Classical()
        context = 0
    else:
        from .train import load_model
        model = load_model(a.checkpoint)
        context = model.cfg.get("context_px", 0)
    bench = B.Benchmark(a.name, context=context)
    params = B.Benchmark(a.name).photometric_params()
    model = model.to(a.device)
    summary, results = B.evaluate(model, bench, device=a.device, return_patches=True)
    bd = breakdown(bench, results, params)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, indent=1))
    print_breakdown(bd)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": summary, "breakdown": bd}, f, indent=1)


if __name__ == "__main__":
    main()
