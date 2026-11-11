"""Fixed, deterministic held-out validation benchmark and evaluator.

The benchmark is a spec (benchmarks/<name>.json, committed): a list of lattice
patches (integer center, dihedral orientation, no resampling, as in deployment)
from validation-split scenes, with a per-patch photometric seed. Patches are
regenerated from the source cache and checked against a stored content hash, so
the benchmark cannot drift silently.

Usage:
  python -m edgesr.benchmark make  [--name val_v1]
  python -m edgesr.benchmark eval  CHECKPOINT [--name val_v1] [--overlays DIR]
"""

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from . import CACHE_DIR, REPO_DIR, SCALE
from . import metrics as M
from . import sampler as S

BENCH_DIR = os.path.join(REPO_DIR, "benchmarks")

VAL_V1 = {
    "split": "val",
    "seed": 7919,
    "counts": {"edge": 600, "sparse": 300, "dense": 300, "narrow": 400, "corner": 400,
               "uniform": 100},
    "jitter_px": 16,
    "aug": {k: v for k, v in S.DEFAULT_AUG.items()
            if k not in ("exposure_ref", "backlight_jitter")},   # implies p99.5 exposure
}

# v2 (infra change 2026-09-25): same patch list as v1. Exposure is normalized by the
# scene's backlight brightness (cache backlight_level, from mask background) instead of
# the image 99.5th percentile, which specular parts inflate by up to 14x. Each patch
# also stores its exact backlight value (no jitter) for models that condition on it.
VAL_V2 = {**VAL_V1, "aug": {**VAL_V1["aug"], "exposure_ref": "backlight",
                             "backlight_jitter": 0.0}}

BENCHMARKS = {"val_v1": VAL_V1, "val_v2": VAL_V2}


def make_spec(store, cfg):
    rng = np.random.default_rng(cfg["seed"])
    scenes = store.scene_indices(cfg["split"])
    eligible = S.eligible_scenes(store, scenes, cfg["counts"])
    patches = []
    m = store.meta["center_margin"]
    for cat, n in cfg["counts"].items():
        for _ in range(n):
            scene = eligible[cat][rng.integers(len(eligible[cat]))]
            cands = store.candidates(store.scenes[scene]["part_index"], cat)
            x, y = cands[rng.integers(len(cands))].astype(int)
            j = cfg["jitter_px"]
            x = int(np.clip(x + rng.integers(-j, j + 1), m, store.W - m))
            y = int(np.clip(y + rng.integers(-j, j + 1), m, store.H - m))
            patches.append({
                "scene_id": store.scenes[scene]["scene_id"], "category": cat,
                "center": [x, y], "quarter_turns": int(rng.integers(4)),
                "flip": bool(rng.integers(2)), "aug_seed": int(rng.integers(2 ** 31)),
            })
    return {"config": cfg, "cache_meta": store.meta, "patches": patches}


def _gen(store, scene_by_id, aug, p, context=0):
    scene = scene_by_id[p["scene_id"]]
    A = S.patch_matrix(flip=p["flip"], quarter_turns=p["quarter_turns"])
    raw, mask = S.sample_patch(store, scene, tuple(float(v) for v in p["center"]), A, context)
    x, info = S.photometric(raw, store.scenes[scene], np.random.default_rng(p["aug_seed"]), aug)
    return x, mask, info["backlight"]


class Benchmark:
    def __init__(self, name="val_v1", store=None, workers=12, context=0):
        """context > 0: same patches, targets and scoring, with inputs extended by
        `context` px per side. The noise realization differs from context 0 (the
        augmentation draws noise for the larger array), so the input hash differs."""
        with open(os.path.join(BENCH_DIR, name + ".json")) as f:
            self.spec = json.load(f)
        self.name = name
        self.patches = self.spec["patches"]
        self.categories = [p["category"] for p in self.patches]
        self.context = context
        cache = os.path.join(CACHE_DIR, f"bench_{name}" + (f"_c{context}" if context else "") + ".npz")
        if os.path.exists(cache):
            z = np.load(cache)
            self.inputs, packed = z["inputs"], z["masks"]
            self.backlight = z["backlight"] if "backlight" in z.files else None
        else:
            store = store or S.SourceStore()
            by_id = {s["scene_id"]: i for i, s in enumerate(store.scenes)}
            with ThreadPoolExecutor(workers) as ex:
                out = list(ex.map(lambda p: _gen(store, by_id, self.spec["config"]["aug"], p,
                                                 context),
                                  self.patches))
            self.inputs = np.stack([o[0] for o in out])
            packed = np.packbits(np.stack([o[1] for o in out]), axis=-1)
            self.backlight = np.array([o[2] for o in out], np.float32)
            if self.spec["config"]["aug"].get("exposure_ref", "p99.5") == "p99.5":
                self.backlight = None   # val_v1 predates backlight conditioning
                np.savez(cache, inputs=self.inputs, masks=packed)
            else:
                np.savez(cache, inputs=self.inputs, masks=packed, backlight=self.backlight)
        self.masks = packed
        extra = self.backlight.tobytes() if self.backlight is not None else b""
        h = hashlib.sha256(self.inputs.tobytes() + packed.tobytes() + extra).hexdigest()
        if "content_sha256" in self.spec and not context:
            assert h == self.spec["content_sha256"], "benchmark content drifted"
        self.content_sha256 = h
        self._gt_pts = None

    def mask(self, i):
        return np.unpackbits(self.masks[i], axis=-1)

    def gt_points(self):
        if self._gt_pts is None:
            with ThreadPoolExecutor(12) as ex:
                self._gt_pts = list(ex.map(
                    lambda i: M.contour_points(self.mask(i).astype(np.float32)),
                    range(len(self.patches))))
        return self._gt_pts

    def photometric_params(self):
        """Replay each patch's augmentation draw to recover its parameters (analysis)."""
        store = S.SourceStore()
        by_id = {s["scene_id"]: i for i, s in enumerate(store.scenes)}
        out = []
        for p in self.patches:
            scene = by_id[p["scene_id"]]
            A = S.patch_matrix(flip=p["flip"], quarter_turns=p["quarter_turns"])
            raw = S.sample_image(store, scene, tuple(float(v) for v in p["center"]), A)
            _, prm = S.photometric(raw, store.scenes[scene],
                                   np.random.default_rng(p["aug_seed"]), self.spec["config"]["aug"])
            out.append(prm)
        return out

    def subset(self, n, seed=0):
        """A fixed stratified subset for cheap in-training monitoring."""
        idx = np.random.default_rng(seed).permutation(len(self.patches))[:n]
        return np.sort(idx)


# ---------------------------------------------------------------------------
# Evaluation

def torch_contour_points(p, level=0.5):
    """GPU version of metrics.contour_points for one (H, W) probability map."""
    inside = p >= level
    out = []
    a, b = p[:, :-1], p[:, 1:]
    ys, xs = torch.nonzero(inside[:, :-1] != inside[:, 1:], as_tuple=True)
    pa, pb = a[ys, xs], b[ys, xs]
    t = ((level - pa) / (pb - pa)).clamp(0, 1)
    out.append(torch.stack([xs + 0.5 + t, ys + 0.5], 1))
    a, b = p[:-1, :], p[1:, :]
    ys, xs = torch.nonzero(inside[:-1, :] != inside[1:, :], as_tuple=True)
    pa, pb = a[ys, xs], b[ys, xs]
    t = ((level - pa) / (pb - pa)).clamp(0, 1)
    out.append(torch.stack([xs + 0.5, ys + 0.5 + t], 1))
    return (torch.cat(out).double() / SCALE).cpu().numpy()


def _score_patch(gpts, ppts, gt, pred_bin, border):
    size = gt.shape[0] / SCALE
    gin = gpts[M._interior(gpts, size, border)]
    pin = ppts[M._interior(ppts, size, border)]
    d_gp = (cKDTree(ppts).query(gin)[0] if len(ppts) else np.full(len(gin), np.inf)) \
        if len(gin) else np.zeros(0)
    d_pg = (cKDTree(gpts).query(pin)[0] if len(gpts) else np.full(len(pin), np.inf)) \
        if len(pin) else np.zeros(0)
    comp = M._component_misses(gt, pred_bin, int(border * SCALE))
    return {"d_gp": d_gp.astype(np.float32), "d_pg": d_pg.astype(np.float32), **comp}


@torch.no_grad()
def evaluate(model, bench, device="cuda", indices=None, batch=16, amp=torch.bfloat16,
             overlay_path=None, overlay_indices=None, border=M.BORDER_PX, return_patches=False):
    model.eval()
    idx = np.arange(len(bench.patches)) if indices is None else np.asarray(indices)
    gts = bench.gt_points()
    b = int(border * SCALE)
    results = [None] * len(idx)
    overlays = {}
    want = set(overlay_indices or [])
    futures = []
    ex = ThreadPoolExecutor(8)
    t_model = 0.0
    for s in range(0, len(idx), batch):
        ii = idx[s:s + batch]
        x = torch.from_numpy(bench.inputs[ii][:, None]).to(device)
        bl = getattr(bench, "backlight", None)
        bl = torch.from_numpy(bl[ii]).to(device) if bl is not None else None
        y = torch.from_numpy(np.stack([bench.mask(i) for i in ii])[:, None]).to(device)
        sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
        sync()
        t0 = time.time()
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None and device == "cuda"):
            logits = model(x, bl)
        logits = logits.float()
        sync()
        t_model += time.time() - t0
        prob = torch.sigmoid(logits)
        yi, li = y[..., b:-b, b:-b].float(), logits[..., b:-b, b:-b]
        bce = F.binary_cross_entropy_with_logits(li, yi, reduction="none").mean((1, 2, 3))
        pb, gb = prob[..., b:-b, b:-b] >= 0.5, yi >= 0.5
        inter, union = (pb & gb).sum((1, 2, 3)), (pb | gb).sum((1, 2, 3))
        pred_bin = (prob >= 0.5).to(torch.uint8).cpu().numpy()
        for k, i in enumerate(ii):
            ppts = torch_contour_points(prob[k, 0])
            r = {"bce": float(bce[k]), "inter": int(inter[k]), "union": int(union[k])}
            futures.append((s + k, r, ex.submit(_score_patch, gts[i], ppts,
                                                y[k, 0].cpu().numpy(), pred_bin[k, 0], border)))
            if i in want:
                overlays[i] = (bench.inputs[i], y[k, 0].cpu().numpy(), pred_bin[k, 0])
    for pos, r, f in futures:
        r.update(f.result())
        results[pos] = r
    ex.shutdown()
    summary = M.aggregate(results, [bench.categories[i] for i in idx])
    summary["model_seconds"] = t_model
    summary["patches_per_second_eval"] = len(idx) / t_model
    if overlay_path and overlays:
        save_overlays(overlay_path, [overlays[i] for i in sorted(overlays)],
                      [bench.categories[i] for i in sorted(overlays)])
    model.train()
    if return_patches:
        return summary, results
    return summary


def save_overlays(path, items, labels, zoom_border=0):
    """Input (nearest 16x) with GT boundary green, prediction red, agreement yellow."""
    tiles, zooms = [], []
    for (x, gt, pred), lab in zip(items, labels):
        c = (x.shape[0] - S.PATCH) // 2
        if c:
            x = x[c:-c, c:-c]           # context inputs: show the scored 48 px region
        lo, hi = float(x.min()), float(x.max())
        g = np.kron((x - lo) / max(hi - lo, 1e-6), np.ones((SCALE, SCALE)))
        rgb = (np.stack([g] * 3, -1) * 160 + 40).astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        ge = (gt ^ cv2.erode(gt, k)).astype(bool)
        pe = (pred ^ cv2.erode(pred, k)).astype(bool)
        rgb[ge] = (0, 255, 0)
        rgb[pe] = (255, 0, 0)
        rgb[ge & pe] = (255, 255, 0)
        # Full-resolution 256x256 crop (16x16 input px) around the boundary nearest the center.
        ys, xs = np.nonzero(ge)
        c = 384
        if len(ys):
            j = np.argmin((ys - c) ** 2 + (xs - c) ** 2)
            cy, cx = (int(np.clip(v, 128, 640)) for v in (ys[j], xs[j]))
        else:
            cy = cx = c
        zooms.append(rgb[cy - 128:cy + 128, cx - 128:cx + 128].copy())
        cv2.putText(rgb, lab, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 255), 2)
        tiles.append(cv2.resize(rgb, (384, 384), interpolation=cv2.INTER_AREA))
    for name, ts in ((path, tiles), (path.replace(".png", "_zoom.png"), zooms)):
        cols = 6
        ts = ts + [np.zeros_like(ts[0])] * (-len(ts) % cols)
        rows = [np.concatenate(ts[i:i + cols], 1) for i in range(0, len(ts), cols)]
        cv2.imwrite(name, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))


def overlay_indices(bench, per_category=3, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for c in sorted(set(bench.categories)):
        idx = [i for i, k in enumerate(bench.categories) if k == c]
        out += list(rng.choice(idx, per_category, replace=False))
    return sorted(int(i) for i in out)


@torch.no_grad()
def measure_throughput(model, batch, device="cuda", iters=30, amp=torch.bfloat16,
                       channels_last=True):
    model.eval()
    n = S.PATCH + 2 * getattr(model, "cfg", {}).get("context_px", 0)
    x = torch.rand(batch, 1, n, n, device=device)
    bl = torch.full((batch,), 0.7, device=device)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    for _ in range(5):
        with torch.autocast("cuda", dtype=amp):
            model(x, bl)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        with torch.autocast("cuda", dtype=amp):
            model(x, bl)
    torch.cuda.synchronize()
    model.train()
    return batch * iters / (time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "eval"])
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--name", default="val_v1")
    ap.add_argument("--overlays")
    a = ap.parse_args()
    if a.cmd == "make":
        os.makedirs(BENCH_DIR, exist_ok=True)
        path = os.path.join(BENCH_DIR, a.name + ".json")
        assert not os.path.exists(path), "benchmarks are immutable; make a new name"
        store = S.SourceStore()
        spec = make_spec(store, BENCHMARKS[a.name])
        with open(path, "w") as f:
            json.dump(spec, f)
        bench = Benchmark(a.name, store)
        spec["content_sha256"] = bench.content_sha256
        with open(path, "w") as f:
            json.dump(spec, f)
        print(f"{len(spec['patches'])} patches, sha256 {bench.content_sha256}")
    else:
        from .train import load_model
        model = load_model(a.checkpoint).cuda().to(memory_format=torch.channels_last)
        bench = Benchmark(a.name)
        s = evaluate(model, bench, overlay_path=a.overlays,
                     overlay_indices=overlay_indices(bench) if a.overlays else None)
        print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
