"""Train a CFI model on crops with point supervision (PLAN_v2).

Usage:
  python -m edgesr.train_cfi CONFIG.json --run-id NAME --desc "..." [--resume]
         [--set train.budget_minutes=60 model.fine_dilations=[1,2,4,6] ...] [--no-ledger]

Per-run artifacts in RUNS_DIR/NAME as for edgesr.train. The ledger line uses the
`valcrops_v1` crop benchmark (fixed val crops, contour metrics on crop interiors) until
the full-image val_v3 benchmark scores the checkpoint separately.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

from . import RUNS_DIR, SCALE
from . import benchmark as B
from . import crops as CR
from . import metrics as M
from . import sampler as S
from .cfi import CFI, DEFAULT_CFI
from .train import (LEDGER, GpuMonitor, apply_sets, git_state, lr_at, merge,
                    snapshot_code)

DEFAULT_TRAIN = {
    "batch": 16,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "warmup_frac": 0.05,
    "schedule": "cosine",
    "budget_minutes": 30.0,
    "amp": "bf16",
    "compile": True,
    "workers": 8,
    "seed": 0,
    "quick_eval_minutes": 10.0,
    "quick_eval_crops": 64,
    "final_eval_crops": 400,
    "checkpoint_minutes": 10.0,
    "grad_clip": 1.0,
    "loss_weights": {"occ": 1.0, "dist": 0.5, "dir": 0.2, "occ4": 0.2, "band4": 0.2,
                     "pband": 0.2},
    "band_pos_weight": 4.0,
}


def default_config():
    return {"model": dict(DEFAULT_CFI), "sampler": copy.deepcopy(S.DEFAULT_SAMPLER),
            "aug": copy.deepcopy(S.DEFAULT_AUG), "crop": dict(CR.DEFAULT_CROP),
            "data": {"cache": "v3", "pool": 0, "swap_every": 16},
            "train": copy.deepcopy(DEFAULT_TRAIN)}


def load_cfi(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    m = CFI(ck["config"]["model"])
    m.load_state_dict(ck["model"])
    return m


def losses(model, batch, out, tc, crop):
    feat, pband_l, occ4_l, band4_l = out
    q, qb = batch["q"], batch["qb"]
    logits = model.decode_points(feat, q[:, 0], q[:, 1], qb).float()
    lw = tc["loss_weights"]
    L = {"occ": F.binary_cross_entropy_with_logits(logits[:, 0], batch["occ"])}
    if model.k_out > 1:
        clip = crop["dist_clip"]
        d_t = batch["dist"].clamp(max=clip)
        L["dist"] = F.l1_loss(F.softplus(logits[:, 1]), d_t)
        m = (batch["dist"] > 0.02) & (batch["dist"] < clip)
        if m.any():
            pd = F.normalize(logits[m][:, 2:4], dim=1)
            L["dir"] = (1 - (pd * batch["dir"][m]).sum(1)).mean()
    b = crop["border"]
    b4 = b // 4
    pw = torch.tensor(tc["band_pos_weight"], device=q.device)
    if occ4_l is not None:
        L["occ4"] = F.binary_cross_entropy_with_logits(
            occ4_l[:, 0, b4:-b4, b4:-b4].float(), batch["occ4"][:, b4:-b4, b4:-b4])
        L["band4"] = F.binary_cross_entropy_with_logits(
            band4_l[:, 0, b4:-b4, b4:-b4].float(), batch["band4"][:, b4:-b4, b4:-b4], pos_weight=pw)
    L["pband"] = F.binary_cross_entropy_with_logits(
        pband_l[:, 0, b:-b, b:-b].float(), batch["pband"][:, b:-b, b:-b], pos_weight=pw)
    total = sum(lw[k] * v for k, v in L.items())
    return total, {k: float(v) for k, v in L.items()}


def eval_crops(split, n, cfg, seed=1234):
    """A fixed, deterministic list of crops with their exact GT boundary points."""
    store = S.open_store(cfg.get("data"))
    rng = np.random.default_rng(seed)
    eligible = S.eligible_scenes(store, store.scene_indices(split), cfg["sampler"]["weights"])
    aug = {**cfg["aug"], "backlight_jitter": 0.0}
    items = []
    for _ in range(n):
        r = np.random.default_rng(rng.integers(2 ** 31))
        it = CR.sample_crop(store, r, eligible, cfg["sampler"], aug, cfg["crop"])
        items.append(it)
    return items


@torch.no_grad()
def evaluate_crops(model, items, device, amp, border, band_thresh=-1.0):
    """Contour metrics on crop interiors, mirroring sparse inference: sub-pixel logits
    only for pixels whose band logit > band_thresh (dilated by 1 px); elsewhere the
    coarse occupancy decides (as a constant +-10 logit)."""
    model.eval()
    res = []
    rec = {t: [0, 0] for t in (-2.0, -1.0, 0.0)}
    for it in items:
        x = torch.from_numpy(it["x"])[None, None].to(device)
        b = torch.tensor([float(it["backlight"])], device=device)
        sd = torch.from_numpy(it["seed"])[None].to(device) if "seed" in it else None
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            feat, pband, occ4, band4 = model.dense_features(x, b, sd)
        feat, pband = feat.float(), pband.float()
        H = feat.shape[-1]
        if model.cfg["decoder"] == "block":
            lg = F.pixel_shuffle(F.conv2d(feat, model.block.weight, model.block.bias, padding=1),
                                 SCALE)[0, 0]
        else:
            yy, xx = torch.meshgrid(torch.arange(1, H - 1, device=device),
                                    torch.arange(1, H - 1, device=device), indexing="ij")
            lg = torch.full((H * SCALE, H * SCALE), -10.0, device=device)
            out = model.decode_pixels(feat, torch.zeros_like(yy.ravel()), yy.ravel(), xx.ravel())[..., 0]
            lg[SCALE:-SCALE, SCALE:-SCALE] = out.view(H - 2, H - 2, SCALE, SCALE).permute(
                0, 2, 1, 3).reshape((H - 2) * SCALE, -1)
        gate = F.max_pool2d((pband > band_thresh).float(), 3, 1, 1)[0, 0] > 0
        if occ4 is not None:
            coarse = torch.where(occ4.float()[0, 0] > 0, 10.0, -10.0).repeat_interleave(4 * SCALE, 0) \
                .repeat_interleave(4 * SCALE, 1)
        else:   # seed mode: outside the band, the threshold mask decides
            sm = torch.from_numpy(it["seed_mask"]).to(device)
            coarse = torch.where(sm > 0.5, 10.0, -10.0).repeat_interleave(SCALE, 0).repeat_interleave(SCALE, 1)
        g16 = gate.repeat_interleave(SCALE, 0).repeat_interleave(SCALE, 1)
        lg = torch.where(g16, lg, coarse)
        ppts = B.torch_contour_points(torch.sigmoid(lg))
        gpts = it["gt_pts"]
        inner = lambda p: p[np.all((p >= border) & (p <= H - border), 1)]
        gi, pi = inner(gpts), inner(ppts)
        d_gp = cKDTree(ppts).query(gi)[0] if len(ppts) and len(gi) else np.full(len(gi), np.inf)
        d_pg = cKDTree(gpts).query(pi)[0] if len(gpts) and len(pi) else np.full(len(pi), np.inf)
        res.append({"d_gp": d_gp.astype(np.float32), "d_pg": d_pg.astype(np.float32),
                    "inter": 0, "union": 0, "bce": 0.0, "components": 0, "missed_components": 0})
        t = torch.from_numpy(it["pband"] > 0).to(device)[border:-border, border:-border]
        for th in rec:
            pb = (F.max_pool2d((pband > th).float(), 3, 1, 1)[0, 0] > 0)[border:-border, border:-border]
            rec[th][0] += int((pb & t).sum())
            rec[th][1] += int(t.sum())
    model.train()
    out = M.aggregate(res, [it["category"] for it in items])
    for k in ("iou", "bce", "component_miss_rate"):
        out.pop(k, None)
    out["pixel_band_recall"] = {str(th): v[0] / max(v[1], 1) for th, v in rec.items()}
    out["band_thresh"] = band_thresh
    return out


def train(cfg, run_id, desc, resume=False, ledger=True):
    tc = cfg["train"]
    # before config.json: seed-mode models need seed inputs ("coarse" = coarse seed, v4)
    cfg["crop"]["seed_inputs"] = (cfg["crop"].get("seed_inputs") or True) if cfg["model"].get("context") == "seed" else False
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, None: None}[tc["amp"]]
    model = CFI(cfg["model"]).to(dev).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    state = {"step": 0, "train_seconds": 0.0, "samples": 0, "resumes": 0}
    if resume and os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        state["resumes"] += 1
    dense = torch.compile(model.dense_features) if tc["compile"] else model.dense_features

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                    "config": cfg}, path + ".tmp")
        os.replace(path + ".tmp", path)

    if cfg.get("data", {}).get("cache", "v3") == "v3":
        S.warm_page_cache()
    else:
        # One OpenCV thread per process, set before any OpenCV call so forked data workers
        # inherit it: 12 workers x all-core OpenCV calls oversubscribed the CPU (data-bound
        # at ~155 crops/s), and cv2.setNumThreads inside a forked worker can deadlock.
        import cv2
        cv2.setNumThreads(1)
    qitems = eval_crops("val", tc["quick_eval_crops"], cfg)
    ds = CR.CropStream("train", tc["batch"], cfg["sampler"], cfg["aug"], cfg["crop"],
                       [tc["seed"], state["resumes"]], cfg.get("data"))
    loader = DataLoader(ds, batch_size=None, num_workers=tc["workers"], pin_memory=True,
                        persistent_workers=True, prefetch_factor=4)
    it = iter(loader)
    log = open(os.path.join(run_dir, "log.jsonl"), "a")
    mon = GpuMonitor()
    mon.start()
    budget = tc["budget_minutes"] * 60
    last_q = last_ck = state["train_seconds"]
    acc, n_acc, wait_acc, win_t0, win_samples = {}, 0, 0.0, time.time(), 0
    torch.cuda.reset_peak_memory_stats()
    mon.active = True
    print(f"run {run_id}: {n_params / 1e6:.3f}M params, radius {model.radius}, "
          f"budget {budget / 60:.1f} min", flush=True)
    t_seg = time.time()
    while True:
        elapsed = state["train_seconds"] + (time.time() - t_seg)
        progress = elapsed / budget
        if progress >= 1.0:
            break
        tw = time.time()
        batch = next(it)
        wait_acc += time.time() - tw
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        x = batch["x"].to(memory_format=torch.channels_last)
        lr = lr_at(progress, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            out = dense(x, batch["backlight"], batch.get("seed"))
        out = tuple(o.float() if o is not None else None for o in out)
        loss, parts = losses(model, batch, out, tc, cfg["crop"])
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
                   "crops_per_s": win_samples / dt, "data_wait_frac": wait_acc / dt,
                   "gpu_util": float(np.mean([g[0] for g in gpu])) if gpu else None,
                   "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            if state["step"] >= 500 and rec["loss"]["occ"] > 0.68:
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
                s = evaluate_crops(model, qitems, dev, amp, cfg["crop"]["border"])
                rec = {"step": state["step"], "train_seconds": round(elapsed, 1),
                       "quick_eval": {k: s.get(k) for k in (
                           "bscore", "missed_rate", "extra_rate", "loc_error_mean",
                           "loc_error_p95", "pixel_band_recall")}}
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
    tc = cfg["train"]
    run_dir = os.path.join(RUNS_DIR, run_id)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, None: None}[tc["amp"]]
    t0 = time.time()
    items = eval_crops("val", tc["final_eval_crops"], cfg, seed=4321)
    summary = evaluate_crops(model, items, "cuda", amp, cfg["crop"]["border"])
    logs = [json.loads(l) for l in open(os.path.join(run_dir, "log.jsonl"))]
    tl = [r for r in logs if "crops_per_s" in r]
    result = {
        "run_id": run_id, "description": desc, "kind": "cfi",
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git["commit"], "dirty": git["dirty"], "code_sha256": git["code_sha256"],
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16],
        "config": cfg, "params": n_params, "radius": model.radius,
        "budget": {"train_seconds": round(state["train_seconds"], 1), "steps": state["step"],
                   "samples": state["samples"]},
        "benchmark": {"name": "valcrops_v4" if cfg.get("data", {}).get("cache") == "v4" else "valcrops_v1", "crops": tc["final_eval_crops"], "seed": 4321},
        "primary": {"bscore": summary.get("bscore")},
        "metrics": summary,
        "system": {"peak_vram_mb": round(peak),
                   "train_samples_per_s": float(np.mean([r["crops_per_s"] for r in tl[1:]] or [0])),
                   "gpu_util_mean": float(np.mean([r["gpu_util"] for r in tl if r.get("gpu_util")] or [0])),
                   "data_wait_frac": float(np.mean([r["data_wait_frac"] for r in tl[1:]] or [0])),
                   "eval_seconds": round(time.time() - t0, 1)},
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=1)
    if ledger:
        slim = {k: v for k, v in result.items() if k != "metrics"}
        slim["metrics"] = {k: v for k, v in summary.items() if k != "by_category"}
        slim["metrics"]["by_category"] = {c: {k: v.get(k) for k in ("bscore", "missed_rate",
                                                                    "extra_rate", "loc_error_mean")}
                                          for c, v in summary["by_category"].items()}
        with open(LEDGER, "a") as f:
            f.write(json.dumps(slim) + "\n")
    print(json.dumps({k: result[k] for k in ("run_id", "primary", "budget", "system")}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, indent=1))
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
