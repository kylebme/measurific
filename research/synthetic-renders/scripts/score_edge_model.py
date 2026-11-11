"""Score the shipped edge model (ONNX runtime) on physical_v3 scenes against the 16x masks.

Per scene: sensor noise like edge-model fullimage (20k e- full well, 5e-4 read, 12 bit,
backlight at 0.75 of full scale), rings from EdgeModel, then per mask tile
(ignoring 1 px at tile borders):
  loc     mean distance of predicted points within 0.5 px of the GT boundary
  extra   share of predicted points > 0.5 px from GT boundary
  missed  share of GT boundary (hi-res edge pixels) > 0.5 px from any prediction
  cmissed missed restricted to contaminant boundaries (scene mask minus base mask)
  signed  median signed distance of matched points, + = contour outside the object

Needs onnxruntime (e.g. the contour-model venv):
    ../../packages/contour-model/.venv/bin/python scripts/score_edge_model.py OUT.jsonl OUTPUT/scene_* ...
Distances are measured to hi-res boundary pixel centers, so ~1/32 px is a floor.
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "packages", "edge-model"))
from edgesr.runtime import EdgeModel

S, B = 16, 16          # scale, border (hi px) ignored at tile edges


def tile_bool(scene_dir, t):
    if t["foreground_pixels"] == 0:
        return None
    if t["foreground_pixels"] == t["width"] * t["height"]:
        return True
    with Image.open(os.path.join(scene_dir, "mask", t["file"])) as im:
        return np.array(im.convert("L")) > 127


def score(scene_dir, model, rng):
    meta = json.load(open(os.path.join(scene_dir, "metadata.json")))
    index = json.load(open(os.path.join(scene_dir, "mask", "index.json")))
    img = cv2.imread(os.path.join(scene_dir, "image.png"), -1).astype(np.float64) / 65535
    bg = np.percentile(img, 90)
    x = img * (0.75 / bg)
    x = rng.poisson(np.clip(x, 0, None) * 20000) / 20000 + rng.normal(0, 5e-4, x.shape)
    x = np.round(np.clip(x, 0, 1) * 4095) / 4095
    pts = model(x.astype(np.float32))["points"]
    hi = np.floor(pts * S).astype(np.int64)
    acc = dict(dp=[], dg=[], dc=[])
    for t in index["tiles"]:
        sel = (hi[:, 0] >= t["x"] + B) & (hi[:, 0] < t["x"] + t["width"] - B) & \
              (hi[:, 1] >= t["y"] + B) & (hi[:, 1] < t["y"] + t["height"] - B)
        m = tile_bool(scene_dir, t)
        if m is None or m is True:
            # uniform tile: every prediction here is > 1 px from GT unless near a border
            acc["dp"].append(np.full(int(sel.sum()), 9.0))
            continue
        edge = (m ^ cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))
        inner = np.zeros_like(edge)
        inner[B:-B, B:-B] = True
        dt_gt = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5) / S
        p = hi[sel] - [t["x"], t["y"]]
        dd = dt_gt[p[:, 1], p[:, 0]]
        acc["dp"].append(dd)
        # signed: + when the predicted point lies outside the object (contour pushed outward)
        acc.setdefault("sg", []).append(np.where(m[p[:, 1], p[:, 0]], -dd, dd)[dd <= 0.5])
        praster = np.ones_like(edge, np.uint8)
        praster[p[:, 1], p[:, 0]] = 0
        dt_p = cv2.distanceTransform(praster, cv2.DIST_L2, 5) / S
        ge = edge & inner
        acc["dg"].append(dt_p[ge])
        # contaminant boundaries: where the scene mask differs from the part's base mask
        if t["file"] and not t["file"].startswith(".."):
            bt = None
            if meta["mask_plan"] == "base_plus_contaminants":
                part = meta["catalog"]["part_id"]
                bpath = os.path.join(os.path.dirname(scene_dir), "base_masks", part, t["file"])
                if os.path.exists(bpath):
                    with Image.open(bpath) as im:
                        bt = np.array(im.convert("L")) > 127
            if bt is not None:
                diff = cv2.dilate((m & ~bt).astype(np.uint8), np.ones((49, 49), np.uint8)).astype(bool)
                acc["dc"].append(dt_p[ge & diff])
    dp = np.concatenate(acc["dp"]) if acc["dp"] else np.zeros(0)
    dg = np.concatenate(acc["dg"]) if acc["dg"] else np.zeros(0)
    dc = np.concatenate(acc["dc"]) if acc["dc"] else np.zeros(0)
    sg = np.concatenate(acc["sg"]) if acc.get("sg") else np.zeros(0)
    return {"signed": float(np.median(sg)) if len(sg) else float("nan"), "scene": os.path.basename(scene_dir), "n_pred": int(len(dp)), "n_gt": int(len(dg)),
            "loc": float(dp[dp <= 0.5].mean()) if (dp <= 0.5).any() else float("nan"),
            "extra": float((dp > 0.5).mean()) if len(dp) else float("nan"),
            "far": float((dp > 2).mean()) if len(dp) else float("nan"),
            "missed": float((dg > 0.5).mean()) if len(dg) else float("nan"),
            "n_cont": int(len(dc)), "cmissed": float((dc > 0.5).mean()) if len(dc) else float("nan"),
            "blur": meta["optics"]["blur_class"], "tilt": meta["pose"].get("realized_tilt_rad", 0) > 0,
            "dirt": meta["contamination"]["level"], "light": meta["illumination"]["class"],
            "material": meta["material"]["family"], "filter": meta["render"]["pixel_filter"],
            "backlight": meta["backlight"]["angular"]["type"], "glare": meta["optics"]["glare"]["fraction"],
            "family": meta["geometry"]["family"]}


if __name__ == "__main__":
    out = sys.argv[1]
    model = EdgeModel()
    rng = np.random.default_rng(0)
    with open(out, "a") as f:
        for d in sys.argv[2:]:
            r = score(d, model, rng)
            f.write(json.dumps(r) + "\n")
            print(r["scene"][-3:], r["blur"], r["dirt"], "loc %.3f extra %.3f missed %.3f cmissed %.3f" %
                  (r["loc"], r["extra"], r["missed"], r["cmissed"]), flush=True)
