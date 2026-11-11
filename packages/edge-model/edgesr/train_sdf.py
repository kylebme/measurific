"""Train SDFNet (PLAN_v3) on dense 256^2 crops from the v4 cache.

Usage:
  python -m edgesr.train_sdf CONFIG.json --run-id NAME --desc "..." [--resume]
         [--set train.budget_minutes=60 model.attention=true ...] [--no-ledger]

Per-run artifacts in RUNS_DIR/NAME as for edgesr.train_cfi. Quick and final evaluation on
fixed val crops (`valcrops_sdf_v1`): contour points of the predicted 0-level vs the exact GT
crossings on crop interiors, plus the SDF error near edges.
"""

import argparse
import copy
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

from . import RUNS_DIR
from . import sampler as S
from . import sdf_data as D
from .bench_v3 import summarize
from .sdf_infer import contour_points
from .sdfnet import DEFAULT_SDFNET, SDFNet
from .train import LEDGER, GpuMonitor, apply_sets, git_state, lr_at, merge, snapshot_code

DEFAULT_TRAIN = {
    "batch": 16,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "warmup_frac": 0.03,
    "schedule": "cosine",
    "budget_minutes": 60.0,
    "amp": "bf16",
    "compile": False,
    "workers": 14,
    "seed": 0,
    "quick_eval_minutes": 15.0,
    "quick_eval_crops": 64,
    "final_eval_crops": 300,
    "checkpoint_minutes": 10.0,
    "grad_clip": 1.0,
    "init": None,                # warm start: model weights from another run's checkpoint
                                 # (a new run with its own schedule; each run stays <= 3 h)
    "loss_weights": {"sdf": 1.0, "unc": 0.1, "dir": 0.1, "sdf4": 0.25},
    "near_weight": 4.0,          # sdf L1 weight 1 + near_weight * exp(-|d|)
}


def default_config():
    return {"model": dict(DEFAULT_SDFNET), "sampler": copy.deepcopy(S.DEFAULT_SAMPLER),
            "aug": copy.deepcopy(S.DEFAULT_AUG), "crop": dict(D.DEFAULT_SDF_CROP),
            "data": {"cache": "v4", "pool": 16, "swap_every": 48},
            "train": copy.deepcopy(DEFAULT_TRAIN)}


def losses(out, batch, cfg):
    tc, b = cfg["train"], cfg["crop"]["border"]
    T = cfg["crop"]["trunc_px"]
    lw = tc["loss_weights"]
    t = batch["sdf"][:, b:-b, b:-b]
    L = {}
    if cfg["model"]["target"] == "occ":
        L["occ"] = F.binary_cross_entropy_with_logits(out["occ"][:, b:-b, b:-b], (t > 0).float())
    else:
        p = out["sdf"][:, b:-b, b:-b]
        e = p - t
        w = 1.0 + tc["near_weight"] * torch.exp(-t.abs())
        L["sdf"] = (w * e.abs()).sum() / w.sum()
        near = t.abs() < T - 1e-3
        if near.any():
            lb = out["log_b"][:, b:-b, b:-b][near]
            L["unc"] = (e.detach().abs()[near] * torch.exp(-lb) + lb).mean()
    dmask = (t.abs() > 0.05) & (t.abs() < T - 1e-3)
    if dmask.any():
        pd = F.normalize(out["dir"][:, :, b:-b, b:-b].permute(0, 2, 3, 1)[dmask], dim=1)
        td = batch["dir"][:, :, b:-b, b:-b].permute(0, 2, 3, 1)[dmask]
        L["dir"] = (1 - (pd * td).sum(1)).mean()
    if "sdf4" in out and cfg["model"]["target"] != "occ":
        t4 = F.avg_pool2d(batch["sdf"][:, None], 4)[:, 0] / 4
        b4 = b // 4
        L["sdf4"] = (out["sdf4"][:, b4:-b4, b4:-b4] - t4[:, b4:-b4, b4:-b4]).abs().mean()
    total = sum(lw.get(k, 1.0) * v for k, v in L.items())
    return total, {k: float(v.detach()) for k, v in L.items()}


def eval_crops(split, n, cfg, seed=1234):
    store = S.open_store(cfg.get("data"))
    rng = np.random.default_rng(seed)
    eligible = S.eligible_scenes(store, store.scene_indices(split), cfg["sampler"]["weights"])
    aug = {**cfg["aug"], "backlight_jitter": 0.0}
    crop = {**cfg["crop"], "p_coarse_drop": 0.0, "coarse_morph_px": 0, "coarse_jitter_px": 0.0,
            "coarse_logit_scale": [1.0, 1.0]}
    return [D.sample_sdf_crop(store, np.random.default_rng(rng.integers(2 ** 31)), eligible, cfg["sampler"], aug, crop)
            for _ in range(n)]


@torch.no_grad()
def evaluate_crops(model, items, device, amp, border, T):
    model.eval()
    gp, pg, sd_err = [], [], []
    for i in range(0, len(items), 8):
        chunk = items[i:i + 8]
        x = torch.from_numpy(np.stack([it["x"] for it in chunk])).to(device)
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            out = model(x)
        sdf = out["sdf"].float().cpu().numpy()
        for it, s in zip(chunk, sdf):
            H = s.shape[-1]
            inner = lambda p: p[np.all((p >= border) & (p <= H - border), 1)]  # noqa: E731
            pp, gg = contour_points(s), it["gt_pts"]
            gi, pi = inner(gg), inner(pp)
            gp.append(cKDTree(pp).query(gi)[0] if len(pp) and len(gi) else np.full(len(gi), np.inf))
            pg.append(cKDTree(gg).query(pi)[0] if len(gg) and len(pi) else np.full(len(pi), np.inf))
            t = it["sdf"][border:-border, border:-border]
            near = np.abs(t) < 2
            sd_err.append(np.abs(s[border:-border, border:-border] - t)[near])
    model.train()
    d_gp, d_pg = np.concatenate(gp), np.concatenate(pg)
    far = d_pg > 2.0
    res = summarize(d_gp, d_pg[~far])
    res["absent_rate"] = float((d_gp > 2.0).mean()) if len(d_gp) else float("nan")
    res["far_extra_frac"] = float(far.mean()) if len(d_pg) else 0.0
    e = np.concatenate(sd_err) if sd_err else np.zeros(1)
    res["sdf_l1_near"] = float(e.mean())
    return res


def train(cfg, run_id, desc, resume=False, ledger=True):
    tc = cfg["train"]
    cfg["model"]["trunc_px"] = cfg["crop"]["trunc_px"]
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
    import cv2
    cv2.setNumThreads(1)      # before forking workers (see train_cfi)
    torch.manual_seed(tc["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, None: None}[tc["amp"]]
    model = SDFNet(cfg["model"]).to(dev).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    state = {"step": 0, "train_seconds": 0.0, "samples": 0, "resumes": 0}
    if tc.get("init") and not (resume and os.path.exists(last)):
        src = tc["init"] if os.path.isabs(tc["init"]) else os.path.join(RUNS_DIR, tc["init"], "final.pt")
        init = torch.load(src, map_location=dev, weights_only=False)
        model.load_state_dict(init["model"])
        print(f"warm start from {src}", flush=True)
    if resume and os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        state["resumes"] += 1
    fwd = torch.compile(model) if tc["compile"] else model

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state, "config": cfg}, path + ".tmp")
        os.replace(path + ".tmp", path)

    qitems = eval_crops("val", tc["quick_eval_crops"], cfg)
    ds = D.SDFCropStream("train", tc["batch"], cfg["sampler"], cfg["aug"], cfg["crop"],
                         [tc["seed"], state["resumes"]], cfg["data"])
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
    print(f"run {run_id}: SDFNet {n_params / 1e6:.2f}M params, budget {budget / 60:.1f} min", flush=True)
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
            out = fwd(x)
        loss, parts = losses(out, batch, cfg)
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
            if not np.isfinite(loss.item()):
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
                s = evaluate_crops(model, qitems, dev, amp, cfg["crop"]["border"], cfg["crop"]["trunc_px"])
                rec = {"step": state["step"], "train_seconds": round(elapsed, 1), "quick_eval": s}
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
    summary = evaluate_crops(model, items, "cuda", amp, cfg["crop"]["border"], cfg["crop"]["trunc_px"])
    logs = [json.loads(l) for l in open(os.path.join(run_dir, "log.jsonl"))]
    tl = [r for r in logs if "crops_per_s" in r]
    result = {
        "run_id": run_id, "description": desc, "kind": "sdf",
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git["commit"], "dirty": git["dirty"], "code_sha256": git["code_sha256"],
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16],
        "config": cfg, "params": n_params,
        "budget": {"train_seconds": round(state["train_seconds"], 1), "steps": state["step"],
                   "samples": state["samples"]},
        "benchmark": {"name": "valcrops_sdf_v1", "crops": tc["final_eval_crops"], "seed": 4321},
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
        with open(LEDGER, "a") as f:
            f.write(json.dumps(result) + "\n")
    print(json.dumps({k: result[k] for k in ("run_id", "primary", "budget", "system")}, indent=1))
    print(json.dumps(summary, indent=1))
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
