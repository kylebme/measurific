"""Train one configuration under a fixed wall-clock (or step) budget.

Usage:
  python -m edgesr.train CONFIG.json --run-id NAME --desc "hypothesis" [--resume]
         [--set train.batch=32 model.width=96 ...] [--no-ledger]

Everything large goes to RUNS_DIR/NAME (checkpoints, logs, overlays). A summary
line is appended to ledger.jsonl in the repository when the run completes.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from . import REPO_DIR, RUNS_DIR
from . import benchmark as B
from . import sampler as S
from .model import DEFAULT_MODEL, MaskSR

LEDGER = os.path.join(REPO_DIR, "ledger.jsonl")

DEFAULT_TRAIN = {
    "batch": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "warmup_frac": 0.05,
    "budget_minutes": 30.0,      # training wall time, excluding evaluation
    "max_steps": None,           # optional step budget instead of time
    "amp": "bf16",
    "channels_last": True,
    "compile": True,
    "workers": 6,
    "seed": 0,
    "quick_eval_minutes": 5.0,
    "quick_eval_patches": 300,
    "checkpoint_minutes": 10.0,
    "grad_clip": 1.0,
    "overfit_n": 0,              # >0: train on a fixed tiny patch set (pipeline check)
    "schedule": "cosine",        # "cosine" | "constant" (after warmup; stress tests)
    "betas": [0.9, 0.999],
    "boundary_weight": 1.0,      # BCE weight on pixels within boundary_radius of a GT edge
    "boundary_radius": 8,        # hi-res pixels (8 = half a source pixel)
}


def default_config():
    return {"model": dict(DEFAULT_MODEL), "sampler": copy.deepcopy(S.DEFAULT_SAMPLER),
            "aug": copy.deepcopy(S.DEFAULT_AUG), "train": dict(DEFAULT_TRAIN),
            "benchmark": "val_v2"}


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_sets(cfg, sets):
    for s in sets or []:
        key, val = s.split("=", 1)
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = json.loads(val)
    return cfg


class PatchStream(IterableDataset):
    def __init__(self, split, sampler_cfg, aug_cfg, seed):
        self.split, self.sampler_cfg, self.aug_cfg, self.seed = split, sampler_cfg, aug_cfg, seed

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([*self.seed, wid])
        store = S.SourceStore()
        eligible = S.eligible_scenes(store, store.scene_indices(self.split),
                                     self.sampler_cfg["weights"])
        while True:
            scene, center, A, _ = S.propose(store, rng, eligible, self.sampler_cfg)
            raw, mask = S.sample_patch(store, scene, center, A, self.sampler_cfg.get("context_px", 0))
            x, info = S.photometric(raw, store.scenes[scene], rng, self.aug_cfg)
            yield (torch.from_numpy(x)[None], torch.tensor(info["backlight"], dtype=torch.float32),
                   torch.from_numpy(mask)[None])


class FixedSet:
    """A tiny fixed patch set, used both to train and to score overfitting."""

    def __init__(self, n, sampler_cfg, aug_cfg, seed):
        store = S.SourceStore()
        rng = np.random.default_rng(seed)
        eligible = S.eligible_scenes(store, store.scene_indices("train"), sampler_cfg["weights"])
        xs, ms, bls, self.categories = [], [], [], []
        for _ in range(n):
            scene, center, A, cat = S.propose(store, rng, eligible, sampler_cfg)
            raw, mask = S.sample_patch(store, scene, center, A)
            x, info = S.photometric(raw, store.scenes[scene], rng, aug_cfg)
            xs.append(x)
            bls.append(info["backlight"])
            ms.append(mask)
            self.categories.append(cat)
        self.inputs = np.stack(xs)
        self.backlight = np.array(bls, np.float32)
        self._masks = np.stack(ms)
        self.patches = list(range(n))
        self._gt = None

    def mask(self, i):
        return self._masks[i]

    def gt_points(self):
        if self._gt is None:
            from .metrics import contour_points
            self._gt = [contour_points(m.astype(np.float32)) for m in self._masks]
        return self._gt


class GpuMonitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.active = False
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-lms", "1000"],
            stdout=subprocess.PIPE, text=True)

    def run(self):
        for line in self.proc.stdout:
            if self.active:
                try:
                    u, m = (float(v) for v in line.split(","))
                    self.samples.append((u, m))
                except ValueError:
                    pass

    def take(self):
        s, self.samples = self.samples, []
        return s

    def stop(self):
        self.proc.terminate()


def git_state():
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO_DIR, capture_output=True, text=True).stdout
    return {"commit": run("rev-parse", "HEAD").strip(),
            "dirty": bool(run("status", "--porcelain", "--", ".").strip()),
            "diff": run("diff", "HEAD", "--", ".")}


def snapshot_code(run_dir):
    """Copy the package source into the run dir; return a hash of it."""
    src = os.path.join(REPO_DIR, "edgesr")
    dst = os.path.join(run_dir, "code")
    os.makedirs(dst, exist_ok=True)
    h = hashlib.sha256()
    for name in sorted(os.listdir(src)):
        if name.endswith(".py"):
            with open(os.path.join(src, name), "rb") as f:
                data = f.read()
            h.update(name.encode() + data)
            with open(os.path.join(dst, name), "wb") as f:
                f.write(data)
    return h.hexdigest()[:16]


def load_model(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = MaskSR(ck["config"]["model"])
    model.load_state_dict(ck["model"])
    return model


def mask_loss(logits, y, tc):
    yf = y.float()
    bw = tc.get("boundary_weight", 1.0)
    if bw == 1.0:
        return F.binary_cross_entropy_with_logits(logits, yf)
    r = tc.get("boundary_radius", 8)
    with torch.no_grad():
        dil = F.max_pool2d(yf, 2 * r + 1, 1, r)
        ero = -F.max_pool2d(-yf, 2 * r + 1, 1, r)
        w = 1.0 + (bw - 1.0) * (dil != ero).float()
    bce = F.binary_cross_entropy_with_logits(logits, yf, reduction="none")
    return (bce * w).sum() / w.sum()


def lr_at(progress, cfg):
    w = cfg["warmup_frac"]
    if progress < w:
        return cfg["lr"] * progress / w
    if cfg.get("schedule", "cosine") == "constant":
        return cfg["lr"]
    p = (progress - w) / max(1 - w, 1e-9)
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


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
    if git["diff"]:
        with open(os.path.join(run_dir, "code.diff"), "w") as f:
            f.write(git["diff"])

    torch.manual_seed(tc["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, None: None}[tc["amp"]]
    fmt = torch.channels_last if tc["channels_last"] else torch.contiguous_format
    model = MaskSR(cfg["model"]).to(dev).to(memory_format=fmt)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"],
                            betas=tuple(tc.get("betas", (0.9, 0.999))))
    scaler = torch.amp.GradScaler(enabled=tc["amp"] == "fp16")
    state = {"step": 0, "train_seconds": 0.0, "samples": 0, "resumes": 0}
    if resume and os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        state["resumes"] += 1
        print(f"resumed at step {state['step']} ({state['train_seconds']:.0f}s)", flush=True)
    fwd = torch.compile(model) if tc["compile"] else model

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                    "config": cfg}, path + ".tmp")
        os.replace(path + ".tmp", path)

    if tc["overfit_n"]:
        fixed = FixedSet(tc["overfit_n"], cfg["sampler"], cfg["aug"], tc["seed"])
        bench, quick_idx = fixed, None

        def batches():
            g = torch.Generator().manual_seed(tc["seed"])
            X = torch.from_numpy(fixed.inputs)[:, None]
            BL = torch.from_numpy(fixed.backlight)
            Y = torch.from_numpy(fixed._masks)[:, None]
            while True:
                i = torch.randint(len(X), (tc["batch"],), generator=g)
                yield X[i], BL[i], Y[i]
        it = batches()
        loader = None
    else:
        S.warm_page_cache()
        bench = B.Benchmark(cfg["benchmark"], context=cfg["model"].get("context_px", 0))
        quick_idx = bench.subset(tc["quick_eval_patches"])
        ds = PatchStream("train", {**cfg["sampler"], "context_px": cfg["model"].get("context_px", 0)},
                         cfg["aug"], [tc["seed"], state["resumes"]])
        loader = DataLoader(ds, batch_size=tc["batch"], num_workers=tc["workers"],
                            pin_memory=True, persistent_workers=True, prefetch_factor=4)
        it = iter(loader)

    log = open(os.path.join(run_dir, "log.jsonl"), "a")
    mon = GpuMonitor()
    mon.start()
    budget = tc["budget_minutes"] * 60
    max_steps = tc["max_steps"]
    last_q = last_ck = state["train_seconds"]
    loss_acc, n_acc, wait_acc, win_t0, win_samples = 0.0, 0, 0.0, time.time(), 0
    torch.cuda.reset_peak_memory_stats()
    mon.active = True
    print(f"run {run_id}: {n_params / 1e6:.3f}M params, budget {budget / 60:.1f} min", flush=True)
    t_seg = time.time()
    while True:
        elapsed = state["train_seconds"] + (time.time() - t_seg)
        progress = state["step"] / max_steps if max_steps else elapsed / budget
        if progress >= 1.0:
            break
        tw = time.time()
        x, bl, y = next(it)
        wait_acc += time.time() - tw
        x = x.to(dev, non_blocking=True).to(memory_format=fmt)
        bl = bl.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True)
        lr = lr_at(progress, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            logits = fwd(x, bl)
        loss = mask_loss(logits.float(), y, tc)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if tc["grad_clip"]:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        scaler.step(opt)
        scaler.update()
        state["step"] += 1
        state["samples"] += x.shape[0]
        win_samples += x.shape[0]
        loss_acc += loss.detach()
        n_acc += 1
        if state["step"] % 100 == 0:
            dt = time.time() - win_t0
            gpu = mon.take()
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                act = model.stage_absmax(x[:4], bl[:4])
            rec = {"step": state["step"], "train_seconds": round(elapsed, 1),
                   "loss": float(loss_acc) / n_acc, "lr": lr, "trunk_absmax": act[0],
                   "stage_absmax": [round(a, 1) for a in act[1:]],
                   "samples_per_s": win_samples / dt, "data_wait_frac": wait_acc / dt,
                   "gpu_util": float(np.mean([g[0] for g in gpu])) if gpu else None,
                   "peak_vram_mb": torch.cuda.max_memory_allocated() / 2 ** 20}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            loss_acc, n_acc, wait_acc, win_t0, win_samples = 0.0, 0, 0.0, time.time(), 0
            # A dead network outputs constant logits: loss pinned at ~ln 2.
            if state["step"] >= 1000 and rec["loss"] > 0.68 and not tc["overfit_n"]:
                log.write(json.dumps({"step": state["step"], "diverged": True}) + "\n")
                log.close()
                mon.stop()
                raise SystemExit(f"run {run_id} diverged (loss {rec['loss']:.3f} at "
                                 f"step {state['step']})")
        elapsed = state["train_seconds"] + (time.time() - t_seg)
        q_due = elapsed - last_q >= tc["quick_eval_minutes"] * 60
        c_due = elapsed - last_ck >= tc["checkpoint_minutes"] * 60
        if q_due or c_due:
            state["train_seconds"] = elapsed
            mon.active = False
            if q_due:
                last_q = elapsed
                s = B.evaluate(model, bench, indices=quick_idx)
                rec = {"step": state["step"], "train_seconds": round(elapsed, 1),
                       "quick_eval": {k: s.get(k) for k in (
                           "bscore", "gt_to_pred_mean_clipped", "pred_to_gt_mean_clipped",
                           "missed_rate", "extra_rate", "loc_error_mean", "iou", "bce",
                           "component_miss_rate")}}
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
    peak_vram = torch.cuda.max_memory_allocated() / 2 ** 20
    save(last)
    save(os.path.join(run_dir, "final.pt"))
    mon.stop()
    if loader is not None:
        del it, loader
    return finalize(cfg, run_id, desc, model, bench, state, n_params, git, peak_vram, ledger)


def finalize(cfg, run_id, desc, model, bench, state, n_params, git, peak_vram, ledger=True):
    """Full benchmark evaluation, overlays, metrics.json and ledger line for a run."""
    tc = cfg["train"]
    run_dir = os.path.join(RUNS_DIR, run_id)
    gpu_all = [json.loads(l) for l in open(os.path.join(run_dir, "log.jsonl"))
               if '"gpu_util"' in l]
    # Final evaluation on the full fixed benchmark.
    ov = B.overlay_indices(bench) if not tc["overfit_n"] else list(range(min(12, tc["overfit_n"])))
    t0 = time.time()
    summary = B.evaluate(model, bench, overlay_path=os.path.join(run_dir, "overlays.png"),
                         overlay_indices=ov)
    eval_seconds = time.time() - t0
    thr = B.measure_throughput(model, 64, channels_last=tc["channels_last"])
    util = [r["gpu_util"] for r in gpu_all if r.get("gpu_util") is not None]
    sps = [r["samples_per_s"] for r in gpu_all]
    waits = [r["data_wait_frac"] for r in gpu_all]
    result = {
        "run_id": run_id, "description": desc,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git["commit"], "dirty": git["dirty"], "code_sha256": git["code_sha256"],
        "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16],
        "config": cfg, "params": n_params,
        "budget": {"train_seconds": round(state["train_seconds"], 1), "steps": state["step"],
                   "samples": state["samples"]},
        "benchmark": {"name": cfg["benchmark"] if not tc["overfit_n"] else "overfit",
                      "sha256": getattr(bench, "content_sha256", None)},
        "primary": {"bscore": summary.get("bscore")},
        "metrics": summary,
        "system": {"peak_vram_mb": round(peak_vram), "train_samples_per_s": float(np.mean(sps[1:] or sps)),
                   "gpu_util_mean": float(np.mean(util)) if util else None,
                   "data_wait_frac": float(np.mean(waits[1:] or waits)),
                   "infer_patches_per_s_b64": thr, "eval_seconds": round(eval_seconds, 1)},
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=1)
    if ledger and not tc["overfit_n"]:
        slim = {k: v for k, v in result.items() if k != "metrics"}
        slim["metrics"] = {k: v for k, v in summary.items() if k != "by_category"}
        slim["metrics"]["by_category"] = {
            c: {k: v.get(k) for k in ("bscore", "missed_rate", "extra_rate", "loc_error_mean",
                                      "component_miss_rate")}
            for c, v in summary["by_category"].items()}
        with open(LEDGER, "a") as f:
            f.write(json.dumps(slim) + "\n")
    print(json.dumps({k: result[k] for k in ("run_id", "primary", "budget", "system")}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, indent=1))
    return result



def finalize_from_checkpoint(run_id, desc, ledger=True):
    """Re-run finalization for a run whose training finished but whose final
    evaluation crashed (uses final.pt; VRAM peak and throughput come from log.jsonl)."""
    run_dir = os.path.join(RUNS_DIR, run_id)
    ck = torch.load(os.path.join(run_dir, "final.pt"), map_location="cuda", weights_only=False)
    cfg = ck["config"]
    fmt = torch.channels_last if cfg["train"]["channels_last"] else torch.contiguous_format
    model = MaskSR(cfg["model"]).cuda().to(memory_format=fmt)
    model.load_state_dict(ck["model"])
    bench = B.Benchmark(cfg["benchmark"], context=cfg["model"].get("context_px", 0))
    logs = [json.loads(l) for l in open(os.path.join(run_dir, "log.jsonl"))]
    peak = max(r.get("peak_vram_mb", 0) for r in logs)
    h = hashlib.sha256()
    code = os.path.join(run_dir, "code")
    for name in sorted(os.listdir(code)):
        with open(os.path.join(code, name), "rb") as f:
            h.update(name.encode() + f.read())
    git = git_state()
    git["code_sha256"] = h.hexdigest()[:16]
    n_params = sum(p.numel() for p in model.parameters())
    return finalize(cfg, run_id, desc + " [finalized from checkpoint]", model, bench,
                    ck["state"], n_params, git, peak, ledger)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--finalize", action="store_true",
                    help="evaluate final.pt of a finished run and write metrics/ledger")
    a = ap.parse_args()
    if a.finalize:
        finalize_from_checkpoint(a.run_id, a.desc, ledger=not a.no_ledger)
        return
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
