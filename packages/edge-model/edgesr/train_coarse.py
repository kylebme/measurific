"""Train the coarse model (edgesr.coarse) on macro-v1-512 scenes.

Usage:
  python -m edgesr.train_coarse [CONFIG.json] --run-id NAME --desc "..." [--resume]
         [--set train.budget_minutes=60 model.block="dense" ...] [--no-ledger]
  (or through the queue: python -m edgesr.queue add NAME --module edgesr.train_coarse ...)

Samples are drawn from the 16-bit renders with on-the-fly augmentation: dihedral,
scale 0.75-1.4 (area / bilinear resampling, nearest for labels), backlight shading,
gamma, exposure, Poisson-Gaussian noise over a wide full-well range (the pipeline input
is a 4x area average of a full-resolution reading, so its noise is ~4x lower than the
export's), bit depth 8-12 and JPEG. The backlight field used for the input
normalization is fitted on the whole augmented frame, then a crop is taken.

Per-run artifacts in RUNS_DIR/NAME: config.json, log.jsonl, last.pt, final.pt,
metrics.json (val benchmark, bench_coarse), code/. One line per completed run is appended
to ledger_coarse.jsonl.
"""

import argparse
import copy
import hashlib
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from . import REPO_DIR, RUNS_DIR
from . import bench_coarse as BC
from .coarse import (DEFAULT_COARSE, CoarseNet, coarse_inputs, part_centers, postprocess,
                     predict_maps, resize_axes, targets)
from .macro import MacroStore
from .train import GpuMonitor, apply_sets, git_state, lr_at, merge, snapshot_code

LEDGER = os.path.join(REPO_DIR, "ledger_coarse.jsonl")

DEFAULT_AUG = {
    "scale": [0.75, 1.4],
    "exposure": [0.35, 0.95],       # backlight level as a fraction of full scale
    "offset": [0.0, 0.03],
    "full_well": [2000, 1000000],   # e- per macro pixel, log-uniform
    "read_noise": [1e-4, 3e-3],     # fraction of full scale, log-uniform
    "bits": [8, 10, 12],
    "p_jpeg": 0.1, "jpeg_quality": [70, 95],
    "p_gamma": 0.3, "gamma": [0.8, 1.25],
    "p_shade": 0.3, "shade": 0.15,
    "p_blur": 0.1, "blur_sigma": [0.3, 1.0],
    # original frame aspect ratio (squashed into the square model frame)
    "p_aspect": 0.5, "aspects": [1.25, 1.3333, 1.5, 1.6, 1.7778, 1.9, 1.1953, 1.5],
    "p_portrait": 0.25, "max_upsample": 2.0,
    # circular region of interest (circular backlight): an ellipse after squashing
    "p_roi": 0.35, "roi_radius": [0.4, 0.55],     # x min(w, h) .. x max(w, h) of the frame
    "roi_center_jitter": 0.1, "p_roi_software": 0.5, "roi_dark": [0.0, 0.3],
    "p_roi_hidden": 0.2,            # ROI present but not given to the model (aux_inputs)
    "roi_soft_px": [0.0, 4.0],
}

DEFAULT_TRAIN = {
    "batch": 16,
    "crop": 384,
    "lr": 2e-3,
    "weight_decay": 0.02,
    "warmup_frac": 0.03,
    "schedule": "cosine",
    "budget_minutes": 40.0,
    "amp": "bf16",
    "compile": True,
    "workers": 12,
    "seed": 0,
    "quick_eval_minutes": 10.0,
    "quick_eval_images": 150,
    "checkpoint_minutes": 10.0,
    "grad_clip": 1.0,
    "loss_weights": {"part": 1.0, "dice": 0.5, "contact": 0.5, "sdf": 0.2, "heat": 100.0,
                     "offset": 1.0},
    "edge_weight": 2.0,             # extra BCE weight within 1.5 px of the part boundary
    "contact_pos_weight": 4.0,
    "init": None,                   # fine-tune: start from this checkpoint's weights
    "wire_weight": 1.0,             # sampling weight of scenes that contain wire parts
    "debris_weight": 0.0,           # extra BCE weight on loose-debris pixels (not part)
    "debris_scene_weight": 1.0,     # sampling weight of scenes with visible loose debris
}


def default_config():
    return {"model": dict(DEFAULT_COARSE), "aug": copy.deepcopy(DEFAULT_AUG),
            "train": copy.deepcopy(DEFAULT_TRAIN)}


# ---------------------------------------------------------------------------
# augmentation

def dihedral(a, k):
    a = np.rot90(a, k % 4)
    return np.ascontiguousarray(a[:, ::-1] if k >= 4 else a)


def loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def camera(raw, level, rng, aug):
    """Linear render in [0, 1] (backlight median `level`) -> sensor reading in [0, 1]."""
    x = raw
    h, w = x.shape
    if rng.random() < aug["p_blur"]:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(*aug["blur_sigma"]))
    if rng.random() < aug["p_shade"]:
        u = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        v = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        c = rng.uniform(-1, 1, 5) * aug["shade"]
        x = x * np.clip(1 + c[0] * u + c[1] * v + c[2] * u * u + c[3] * v * v + c[4] * u * v, 0.5, 1.5)
    if rng.random() < aug["p_gamma"]:
        g = loguniform(rng, *aug["gamma"])
        x = np.clip(x, 0, None) ** g
        level = level ** g
    x = x * (rng.uniform(*aug["exposure"]) / max(level, 1e-4)) + rng.uniform(*aug["offset"])
    fw = loguniform(rng, *aug["full_well"])
    rd = loguniform(rng, *aug["read_noise"])
    noise = rng.standard_normal(x.shape, dtype=np.float32)
    x = x + noise * np.sqrt(np.clip(x, 0, None) / fw + rd * rd)
    bits = int(rng.choice(aug["bits"]))
    q = float(2 ** bits - 1)
    x = np.round(np.clip(x, 0, 1) * q) / q
    if bits == 8 and rng.random() < aug["p_jpeg"]:
        qual = int(rng.integers(aug["jpeg_quality"][0], aug["jpeg_quality"][1] + 1))
        ok, enc = cv2.imencode(".jpg", (x * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, qual])
        x = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return x.astype(np.float32)


def sample_aspect(rng, aug):
    if rng.random() >= aug.get("p_aspect", 0.0):
        return 1.0
    r = float(rng.choice(aug["aspects"])) * loguniform(rng, 0.97, 1.03)
    return 1.0 / r if rng.random() < aug["p_portrait"] else r


def frame_geometry(raw, inst, rng, aug, crop, extra=None):
    """Render (isotropic, square) -> model frame S x S: a region with the sampled original
    aspect ratio r, squashed to a square at a random scale. Returns raw, inst and the
    original-frame size (w0, h0) in render px."""
    H, W = raw.shape
    r = sample_aspect(rng, aug)
    w0, h0 = (W, int(round(H / r))) if r >= 1 else (int(round(W * r)), H)
    y = int(rng.integers(0, H - h0 + 1))
    x = int(rng.integers(0, W - w0 + 1))
    raw, inst = raw[y:y + h0, x:x + w0], inst[y:y + h0, x:x + w0]
    if extra is not None:
        extra = extra[y:y + h0, x:x + w0]
    S = int(round(max(W, H) * loguniform(rng, *aug["scale"])))
    S = min(S, int(aug.get("max_upsample", 2.0) * min(w0, h0)))
    S = max(S, crop)
    if (S, S) != (w0, h0):
        raw = resize_axes(raw, S, S)
        # nearest-exact samples at pixel centers like AREA/LINEAR; plain NEAREST is offset
        # by up to half a source pixel (labels misaligned 0.2-0.4 px; coarse_*_40m runs)
        inst = resize_axes(inst, S, S, nearest=True)
        if extra is not None:
            extra = resize_axes(np.ascontiguousarray(extra), S, S, nearest=True)
    if extra is not None:
        return np.ascontiguousarray(raw), np.ascontiguousarray(inst), (w0, h0), r, np.ascontiguousarray(extra)
    return np.ascontiguousarray(raw), np.ascontiguousarray(inst), (w0, h0), r


def roi_ellipse(S, w0, h0, rng, aug):
    """Circle in original-frame coordinates (w0 x h0 render px) -> float ellipse mask in
    the S x S model frame, and its soft edge width in model px."""
    R = rng.uniform(aug["roi_radius"][0] * min(w0, h0), aug["roi_radius"][1] * max(w0, h0))
    j = aug["roi_center_jitter"]
    cx = w0 * (0.5 + rng.uniform(-j, j))
    cy = h0 * (0.5 + rng.uniform(-j, j))
    u = (np.arange(S, dtype=np.float32) + 0.5) * (w0 / S) - cx
    v = (np.arange(S, dtype=np.float32) + 0.5) * (h0 / S) - cy
    # signed distance to the circle in original px, converted to model px (approx.)
    d = (R - np.sqrt(u[None, :] ** 2 + v[:, None] ** 2)) * (S / np.sqrt(w0 * h0))
    return d


def apply_roi(raw, inst, level, d, rng, aug):
    """Dark outside the ROI: either an unlit backlight (dark level, soft edge; before the
    camera) or a software crop (exact zeros after the camera, see make_sample)."""
    inside = d >= 0
    inst = np.where(inside, inst, 0).astype(inst.dtype)
    software = rng.random() < aug["p_roi_software"]
    if not software:
        soft = rng.uniform(*aug["roi_soft_px"])
        m = np.clip(d / max(soft, 1e-3) + 0.5, 0, 1) if soft > 0.3 else inside.astype(np.float32)
        dark = rng.uniform(*aug["roi_dark"]) * level
        raw = raw * m + dark * (1 - m)
    return raw.astype(np.float32), inst, inside, software


def make_sample(store, i, rng, cfg, crop):
    aug, mc = cfg["aug"], cfg["model"]
    raw = np.asarray(store.images[i], np.float32) / 65535.0
    inst = np.asarray(store.inst[i])
    deb = np.asarray(store.debris[i])
    k = int(rng.integers(8))
    raw, inst, deb = dihedral(raw, k), dihedral(inst, k), dihedral(deb, k)
    raw, inst, (w0, h0), aspect, deb = frame_geometry(raw, inst, rng, aug, crop, deb)
    level = store.scenes[i]["backlight_level"] or float(np.percentile(raw, 95))
    inside, software = None, False
    if rng.random() < aug.get("p_roi", 0.0):
        d = roi_ellipse(raw.shape[0], w0, h0, rng, aug)
        raw, inst, inside, software = apply_roi(raw, inst, level, d, rng, aug)
    x = camera(raw, level, rng, aug)
    if software:                # software crop: exactly zero outside
        x = np.where(inside, x, 0.0).astype(np.float32)
    valid = inside if inside is not None and rng.random() >= aug.get("p_roi_hidden", 0.0) else None
    if valid is not None:       # as predict_maps: the reading is zeroed outside a given ROI
        x = np.where(valid, x, 0.0).astype(np.float32)
    inp = coarse_inputs(x, mc["in_mode"], mc.get("aux_inputs", False), valid, aspect)
    H, W = x.shape
    y0 = int(rng.integers(0, H - crop + 1))
    x0 = int(rng.integers(0, W - crop + 1))
    inp = inp[:, y0:y0 + crop, x0:x0 + crop]
    centers = None
    if mc.get("instance_head") == "center":     # centers of whole parts, in crop coordinates
        centers = {k: (x - x0, y - y0) for k, (x, y) in part_centers(inst).items()}
    ic = np.ascontiguousarray(inst[y0:y0 + crop, x0:x0 + crop])
    ys = targets(ic, mc["contact_px"], mc["sdf_clip"], centers, mc.get("center_sigma", 5.0),
                 mc.get("offset_scale", 64.0))
    # last channel: loose debris (>= 1/8 covered, 1 px dilated, never part), a hard negative
    dm = cv2.dilate((deb[y0:y0 + crop, x0:x0 + crop] >= 32).astype(np.uint8), np.ones((3, 3), np.uint8))
    ys.append(((dm > 0) & (ic == 0)).astype(np.float32))
    return np.ascontiguousarray(inp), np.stack(ys)


class MacroStream(IterableDataset):
    def __init__(self, cfg, seed):
        self.cfg, self.seed = cfg, seed

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([*self.seed, wid])
        store = MacroStore()
        idx = np.array(store.split("train"))
        dmax = store.debris_max()
        tc = self.cfg["train"]
        w = np.array([(tc.get("wire_weight", 1.0) if any(p["group"] == "wire" for p in store.scenes[i]["parts"])
                       else 1.0) * (tc.get("debris_scene_weight", 1.0) if dmax[i] >= 32 else 1.0)
                      for i in idx])
        w /= w.sum()
        while True:
            xs, ys = zip(*[make_sample(store, int(rng.choice(idx, p=w)), rng, self.cfg, tc["crop"])
                           for _ in range(tc["batch"])])
            yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


# ---------------------------------------------------------------------------
# loss

def losses(out, y, tc, sdf_clip):
    lp, lc, ls = out[:, 0], out[:, 1], out[:, 2]
    part, contact, sdf = y[:, 0], y[:, 1], y[:, 2]
    w = 1.0 + tc["edge_weight"] * (sdf.abs() <= 1.5).float() + tc.get("debris_weight", 0.0) * y[:, -1]
    bce = (F.binary_cross_entropy_with_logits(lp, part, reduction="none") * w).mean()
    p = torch.sigmoid(lp)
    inter = (p * part).sum((1, 2))
    dice = (1 - (2 * inter + 1) / (p.sum((1, 2)) + part.sum((1, 2)) + 1)).mean()
    con = F.binary_cross_entropy_with_logits(
        lc, contact, pos_weight=torch.tensor(tc["contact_pos_weight"], device=out.device))
    sdl = F.smooth_l1_loss(ls, sdf / sdf_clip, beta=0.1)
    lw = tc["loss_weights"]
    total = lw["part"] * bce + lw["dice"] * dice + lw["contact"] * con + lw["sdf"] * sdl
    parts = {"part": bce.item(), "dice": dice.item(), "contact": con.item(), "sdf": sdl.item()}
    if out.shape[1] == 6:
        heat = F.mse_loss(out[:, 3], y[:, 3])
        m = part.unsqueeze(1)
        off = ((out[:, 4:6] - y[:, 4:6]).abs() * m).sum() / (2 * m.sum() + 1)
        total = total + lw["heat"] * heat + lw["offset"] * off
        parts.update({"heat": heat.item(), "offset": off.item()})
    return total, parts


def quick_eval(model, n, device="cuda"):
    model.eval()
    store = MacroStore()

    def predict(x, valid=None, aspect=1.0):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m = predict_maps(model, x, device, valid, aspect)
        return postprocess(m)
    res, _, _ = BC.evaluate(predict, store, "val", n)
    model.train()
    return res


# ---------------------------------------------------------------------------

def train(cfg, run_id, desc, resume=False, ledger=True):
    tc = cfg["train"]
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    last = os.path.join(run_dir, "last.pt")
    if os.path.exists(os.path.join(run_dir, "metrics.json")) and not resume:
        raise SystemExit(f"run {run_id} already completed")
    if os.path.exists(last) and not resume:
        raise SystemExit(f"run {run_id} exists; pass --resume")
    git = git_state()
    git["code_sha256"] = snapshot_code(run_dir)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    torch.manual_seed(tc["seed"])
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, None: None}[tc["amp"]]
    model = CoarseNet(cfg["model"]).to(dev).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    state = {"step": 0, "train_seconds": 0.0, "samples": 0, "resumes": 0}
    if tc.get("init") and not (resume and os.path.exists(last)):
        init = torch.load(tc["init"], map_location=dev, weights_only=False)
        model.load_state_dict(init.get("model") or init["state_dict"])
        print(f"initialized from {tc['init']}", flush=True)
    if resume and os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        state["resumes"] += 1
    fwd = torch.compile(model) if tc["compile"] else model

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                    "config": cfg}, path + ".tmp")
        os.replace(path + ".tmp", path)

    loader = DataLoader(MacroStream(cfg, [tc["seed"], state["resumes"]]), batch_size=None,
                        num_workers=tc["workers"], pin_memory=True, persistent_workers=True,
                        prefetch_factor=4)
    it = iter(loader)
    log = open(os.path.join(run_dir, "log.jsonl"), "a")
    mon = GpuMonitor()
    mon.start()
    budget = tc["budget_minutes"] * 60
    last_q = last_ck = state["train_seconds"]
    acc, n_acc, wait_acc, win_t0, win_samples = {}, 0, 0.0, time.time(), 0
    torch.cuda.reset_peak_memory_stats()
    mon.active = True
    print(f"run {run_id}: {n_params / 1e6:.3f}M params, budget {budget / 60:.1f} min", flush=True)
    model.train()
    t_seg = time.time()
    while True:
        elapsed = state["train_seconds"] + (time.time() - t_seg)
        progress = elapsed / budget
        if progress >= 1.0:
            break
        tw = time.time()
        x, y = next(it)
        wait_acc += time.time() - tw
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(dev, non_blocking=True)
        lr = lr_at(progress, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            out = fwd(x)
        loss, parts = losses(out.float(), y, tc, cfg["model"]["sdf_clip"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if tc["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        state["step"] += 1
        state["samples"] += x.shape[0]
        win_samples += x.shape[0]
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + v
        n_acc += 1
        if state["step"] % 50 == 0:
            dt = time.time() - win_t0
            gpu = mon.take()
            rec = {"step": state["step"], "train_seconds": round(elapsed, 1), "lr": lr,
                   "loss": {k: round(v / n_acc, 5) for k, v in acc.items()},
                   "samples_per_s": win_samples / dt, "data_wait_frac": wait_acc / dt,
                   "gpu_util": float(np.mean([g[0] for g in gpu])) if gpu else None,
                   "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            if not np.isfinite(rec["loss"]["part"]) or (state["step"] >= 1000 and rec["loss"]["part"] > 0.6):
                log.write(json.dumps({"step": state["step"], "diverged": True}) + "\n")
                log.close()
                mon.stop()
                raise SystemExit(f"run {run_id} diverged")
            acc, n_acc, wait_acc, win_t0, win_samples = {}, 0, 0.0, time.time(), 0
        elapsed = state["train_seconds"] + (time.time() - t_seg)
        q_due = elapsed - last_q >= tc["quick_eval_minutes"] * 60
        c_due = elapsed - last_ck >= tc["checkpoint_minutes"] * 60
        if q_due or c_due:
            state["train_seconds"] = elapsed
            mon.active = False
            if q_due:
                last_q = elapsed
                s = quick_eval(model, tc["quick_eval_images"])
                rec = {"step": state["step"], "train_seconds": round(elapsed, 1),
                       "quick_eval": {k: s.get(k) for k in BC.HEAD}}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            if c_due:
                last_ck = elapsed
                save(last)
            mon.active = True
            t_seg = time.time()
            win_t0, wait_acc, win_samples = time.time(), 0.0, 0
    state["train_seconds"] += time.time() - t_seg
    mon.active = False
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    save(last)
    save(os.path.join(run_dir, "final.pt"))
    mon.stop()
    del it, loader
    return finalize(cfg, run_id, desc, model, state, n_params, git, peak, ledger)


def finalize(cfg, run_id, desc, model, state, n_params, git, peak, ledger=True):
    run_dir = os.path.join(RUNS_DIR, run_id)
    t0 = time.time()
    model.eval()
    predict = BC.model_predictor(os.path.join(run_dir, "final.pt"))
    res = BC.run(run_id, predict, "val", None)
    logs = [json.loads(l) for l in open(os.path.join(run_dir, "log.jsonl"))]
    tl = [r for r in logs if "samples_per_s" in r]
    result = {
        "run_id": run_id, "description": desc, "kind": "coarse",
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git["commit"], "dirty": git["dirty"], "code_sha256": git["code_sha256"],
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16],
        "config": cfg, "params": n_params,
        "budget": {"train_seconds": round(state["train_seconds"], 1), "steps": state["step"],
                   "samples": state["samples"]},
        "benchmark": {"name": "macro_val_export", "images": res["images"]},
        "primary": {"bscore": res["bscore"]},
        "metrics": {k: v for k, v in res.items() if k != "slices"},
        "system": {"peak_vram_mb": round(peak),
                   "train_samples_per_s": float(np.mean([r["samples_per_s"] for r in tl[1:]] or [0])),
                   "gpu_util_mean": float(np.mean([r["gpu_util"] for r in tl if r.get("gpu_util")] or [0])),
                   "data_wait_frac": float(np.mean([r["data_wait_frac"] for r in tl[1:]] or [0])),
                   "eval_seconds": round(time.time() - t0, 1)},
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=1)
    if ledger:
        with open(LEDGER, "a") as f:
            f.write(json.dumps(result) + "\n")
    print(json.dumps({k: result[k] for k in ("run_id", "primary", "budget", "system")}, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = default_config()
    if a.config:
        with open(a.config) as f:
            cfg = merge(cfg, json.load(f))
    if a.resume and os.path.exists(os.path.join(RUNS_DIR, a.run_id, "config.json")):
        with open(os.path.join(RUNS_DIR, a.run_id, "config.json")) as f:
            cfg = json.load(f)
    else:
        cfg = apply_sets(cfg, a.set)
    train(cfg, a.run_id, a.desc, resume=a.resume, ledger=not a.no_ledger)


if __name__ == "__main__":
    main()
