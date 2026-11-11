"""Full-image tiled inference: throughput (GPU/CPU) and tile-seam quality.

Tiling: 48 px input tiles; each keeps its central (48 - 2*margin)^2 output core, so the
stride equals the core size (default 40, margin 4). Optional overlap blending uses a
smaller stride and averages logits with a tapered window.

Usage:
  python -m edgesr.fullimage throughput CKPT [CKPT ...]
  python -m edgesr.fullimage seams CKPT [CKPT ...] [--regions 6]
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from . import SCALE, WORK_DIR
from . import benchmark as B
from . import metrics as M
from . import sampler as S
from .train import load_model

P = S.PATCH                      # 48
FIXED_AUG = {"background_level": [0.75, 0.75], "offset": [0.01, 0.01],
             "full_well_e": [20000.0, 20000.0], "read_noise": [5e-4, 5e-4],
             "bit_depths": [12], "exposure_ref": "backlight", "backlight_jitter": 0.0}


def sensor_image(store, scene, seed=0):
    """A full source image as a fixed 12-bit sensor reading, plus its backlight value."""
    raw = np.asarray(store.images[scene], np.float32)
    x, info = S.photometric(raw, store.scenes[scene], np.random.default_rng(seed), FIXED_AUG)
    return x, info["backlight"]


def tile_origins(x0, y0, w, h, core, margin, offset=(0, 0)):
    """Top-left input-pixel origins of tiles whose cores cover [x0,x0+w) x [y0,y0+h)."""
    ox, oy = offset
    xs = np.arange(x0 - ox - core, x0 + w, core)
    ys = np.arange(y0 - oy - core, y0 + h, core)
    xs = xs[xs + core > x0]
    ys = ys[ys + core > y0]
    return [(int(x - margin), int(y - margin)) for y in ys for x in xs]


def run_tiles(model, img, bl, origins, device, batch, amp=True):
    """Yield (origin, logits (768,768) tensor) for each tile."""
    H, W = img.shape
    pad = P
    padded = np.pad(img, pad, mode="edge")
    for s in range(0, len(origins), batch):
        chunk = origins[s:s + batch]
        xb = np.stack([padded[y + pad:y + pad + P, x + pad:x + pad + P] for x, y in chunk])
        n = len(chunk)
        if n < batch:            # keep shapes static for compiled models
            xb = np.concatenate([xb, np.repeat(xb[-1:], batch - n, 0)])
        x = torch.from_numpy(xb[:, None]).to(device)
        if device == "cuda":
            x = x.to(memory_format=torch.channels_last)
        b = torch.full((batch,), bl, device=device)
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16,
                                             enabled=amp and device == "cuda"):
            out = model(x, b).float()
        for k in range(n):
            yield chunk[k], out[k, 0]


def predict_region(model, img, bl, x0, y0, w, h, device="cuda", core=40, offset=(0, 0),
                   blend_stride=None, batch=128):
    """Hi-res logits for source region [x0,x0+w) x [y0,y0+h).

    Plain tiling: each tile writes its central core. Blend: stride blend_stride, logits
    averaged with a separable triangular window that is zero at the tile edges."""
    margin = (P - core) // 2
    out = torch.zeros(h * SCALE, w * SCALE, device=device)
    if blend_stride is None:
        origins = tile_origins(x0, y0, w, h, core, margin, offset)
        for (tx, ty), lg in run_tiles(model, img, bl, origins, device, batch):
            cx, cy = tx + margin, ty + margin            # core origin (source px)
            ax0, ay0 = max(cx, x0), max(cy, y0)
            ax1, ay1 = min(cx + core, x0 + w), min(cy + core, y0 + h)
            if ax0 >= ax1 or ay0 >= ay1:
                continue
            out[(ay0 - y0) * SCALE:(ay1 - y0) * SCALE, (ax0 - x0) * SCALE:(ax1 - x0) * SCALE] = \
                lg[(ay0 - ty) * SCALE:(ay1 - ty) * SCALE, (ax0 - tx) * SCALE:(ax1 - tx) * SCALE]
        return out
    s = blend_stride
    wsum = torch.zeros_like(out)
    t = (torch.arange(P * SCALE, device=device, dtype=torch.float32) + 0.5) / SCALE
    e = torch.clamp(torch.minimum(t, P - t) - margin, min=0) + 1e-3   # 0 in the discard margin
    win = e[:, None] * e[None, :]
    xs = np.arange(x0 - P, x0 + w, s)
    ys = np.arange(y0 - P, y0 + h, s)
    origins = [(int(x), int(y)) for y in ys for x in xs]
    for (tx, ty), lg in run_tiles(model, img, bl, origins, device, batch):
        ax0, ay0 = max(tx, x0), max(ty, y0)
        ax1, ay1 = min(tx + P, x0 + w), min(ty + P, y0 + h)
        if ax0 >= ax1 or ay0 >= ay1:
            continue
        ro = (slice((ay0 - y0) * SCALE, (ay1 - y0) * SCALE), slice((ax0 - x0) * SCALE, (ax1 - x0) * SCALE))
        ri = (slice((ay0 - ty) * SCALE, (ay1 - ty) * SCALE), slice((ax0 - tx) * SCALE, (ax1 - tx) * SCALE))
        out[ro] += lg[ri] * win[ri]
        wsum[ro] += win[ri]
    return out / wsum.clamp_min(1e-6)


# ---------------------------------------------------------------------------
# Throughput

def throughput(ckpts, out_path):
    store = S.SourceStore()
    scene = store.scene_indices("val")[0]
    img, bl = sensor_image(store, scene)
    H, W = img.shape
    core, margin = 40, 4
    origins = tile_origins(0, 0, W, H, core, margin)
    results = []
    for ck in ckpts:
        m0 = load_model(ck)
        name = os.path.basename(os.path.dirname(ck))
        n_params = sum(p.numel() for p in m0.parameters())
        # GPU: compiled bf16, full image, cores assembled into a full 32768^2 uint8 mask.
        m = m0.cuda().eval().to(memory_format=torch.channels_last)
        mc = torch.compile(m)
        full = torch.zeros(H * SCALE, W * SCALE, dtype=torch.uint8, device="cuda")
        batch = 64
        for _ in run_tiles(mc, img, bl, origins[:batch * 2], "cuda", batch):   # compile + warm
            pass
        torch.cuda.synchronize()
        t0 = time.time()
        c0 = margin * SCALE
        for (tx, ty), lg in run_tiles(mc, img, bl, origins, "cuda", batch):
            cx, cy = tx + margin, ty + margin
            if cx >= W or cy >= H:
                continue
            ex, ey = min(core, W - cx), min(core, H - cy)
            full[cy * SCALE:(cy + ey) * SCALE, cx * SCALE:(cx + ex) * SCALE] = \
                (lg[c0:c0 + ey * SCALE, c0:c0 + ex * SCALE] > 0)
        torch.cuda.synchronize()
        gpu_s = time.time() - t0
        # Model-only GPU rate at the same batch.
        x = torch.rand(batch, 1, P, P, device="cuda").to(memory_format=torch.channels_last)
        b = torch.full((batch,), bl, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                mc(x, b)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                mc(x, b)
            torch.cuda.synchronize()
        gpu_model_tps = 10 * batch / (time.time() - t0)
        del full, mc, m
        torch.cuda.empty_cache()
        # CPU: fp32, all threads, a subset of tiles, extrapolated to the full image.
        torch.set_num_threads(os.cpu_count())
        mcpu = m0.cpu().eval()
        cpu = {}
        for fmt in ("contiguous", "channels_last"):
            mm = mcpu.to(memory_format=torch.channels_last) if fmt == "channels_last" else mcpu
            sub = origins[:64]
            for _ in run_tiles(mm, img, bl, sub[:16], "cpu", 16, amp=False):
                pass
            t0 = time.time()
            for _ in run_tiles(mm, img, bl, sub, "cpu", 16, amp=False):
                pass
            cpu[fmt] = len(sub) / (time.time() - t0)
        cpu_tps = max(cpu.values())
        r = {"model": name, "params": n_params, "tiles_per_image": len(origins),
             "image": f"{W}x{H} -> {W * SCALE}x{H * SCALE} mask",
             "gpu_image_seconds": round(gpu_s, 2), "gpu_tiles_per_s": round(len(origins) / gpu_s, 1),
             "gpu_model_only_tiles_per_s": round(gpu_model_tps, 1),
             "gpu_mpix_out_per_s": round(len(origins) * (core * SCALE) ** 2 / gpu_s / 1e6, 1),
             "cpu_tiles_per_s": {k: round(v, 2) for k, v in cpu.items()},
             "cpu_image_seconds_est": round(len(origins) / cpu_tps, 1),
             "cpu_threads": os.cpu_count()}
        print(json.dumps(r), flush=True)
        results.append(r)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    return results


# ---------------------------------------------------------------------------
# Seam quality

SEAM_BINS = [0, 0.25, 0.5, 1, 2, 4, 20.01]


def seam_distance(pts, period):
    """Distance (source px) from region-relative points to the nearest tile boundary,
    for a tile grid aligned with the region origin with the given period."""
    gx, gy = pts[:, 0] % period, pts[:, 1] % period
    return np.minimum(np.minimum(gx, period - gx), np.minimum(gy, period - gy))


def pick_regions(store, n, size, seed=0):
    """Val-scene regions centered on dense/edge candidates, spread over scenes."""
    rng = np.random.default_rng(seed)
    scenes = store.scene_indices("val")
    picks = []
    for sc in rng.choice(scenes, n, replace=False):
        part = store.scenes[sc]["part_index"]
        c = store.candidates(part, "dense")
        if not len(c):
            c = store.candidates(part, "edge")
        x, y = c[rng.integers(len(c))]
        x0 = int(np.clip(x - size // 2, 64, store.W - size - 64))
        y0 = int(np.clip(y - size // 2, 64, store.H - size - 64))
        picks.append((int(sc), x0, y0))
    return picks


def seams(ckpts, out_path, n_regions, size):
    store = S.SourceStore()
    regions = pick_regions(store, n_regions, size)
    core = 40
    results = []
    for ck in ckpts:
        name = os.path.basename(os.path.dirname(ck))
        m = load_model(ck).cuda().eval().to(memory_format=torch.channels_last)
        acc = {k: {"d_gp": [], "e_gp": [], "d_pg": [], "e_pg": []} for k in ("plain", "blend24")}
        jit = {"d": [], "e": []}
        jit_blend = {"d": [], "e": []}
        for sc, x0, y0 in regions:
            img, bl = sensor_image(store, sc, seed=sc)
            part = store.scenes[sc]["part_index"]
            gt = store.mask_window(part, x0 * SCALE, y0 * SCALE, size * SCALE, size * SCALE)
            gpts = M.contour_points(gt.astype(np.float32))
            preds = {
                "plain": predict_region(m, img, bl, x0, y0, size, size, core=core),
                "shifted": predict_region(m, img, bl, x0, y0, size, size, core=core,
                                          offset=(core // 2, core // 2)),
                "blend24": predict_region(m, img, bl, x0, y0, size, size, blend_stride=24),
            }
            pts = {k: B.torch_contour_points(torch.sigmoid(v)) for k, v in preds.items()}
            inner = lambda p: p[np.all((p >= 2) & (p <= size - 2), 1)]
            g_in = inner(gpts)
            for k, period in (("plain", core), ("blend24", 24)):
                pp = pts[k]
                p_in = inner(pp)
                acc[k]["d_gp"].append(cKDTree(pp).query(g_in)[0])
                acc[k]["e_gp"].append(seam_distance(g_in, period))
                acc[k]["d_pg"].append(cKDTree(gpts).query(p_in)[0])
                acc[k]["e_pg"].append(seam_distance(p_in, period))
            # Jitter: plain tiling vs the same tiling shifted by half a tile. Any
            # disagreement is caused purely by where the seams fall.
            a, b2 = inner(pts["plain"]), pts["shifted"]
            jit["d"].append(cKDTree(b2).query(a)[0])
            jit["e"].append(seam_distance(a, core))
            # Same measurement for blended tiling (stride 24, shifted by 12).
            pb = preds["blend24"]
            pbs = B.torch_contour_points(torch.sigmoid(
                predict_region(m, img, bl, x0 + 12, y0 + 12, size - 24, size - 24, blend_stride=24)))
            pb_pts = inner(B.torch_contour_points(torch.sigmoid(pb)))
            pb_pts = pb_pts[np.all((pb_pts >= 14) & (pb_pts <= size - 14), 1)]
            jit_blend["d"].append(cKDTree(pbs + 12).query(pb_pts)[0])
            jit_blend["e"].append(seam_distance(pb_pts, 24))
        r = {"model": name, "regions": regions, "region_size": size, "core": core}
        for k, v in acc.items():
            dg, eg = np.concatenate(v["d_gp"]), np.concatenate(v["e_gp"])
            dp, ep = np.concatenate(v["d_pg"]), np.concatenate(v["e_pg"])
            r[k] = bin_table(dg, eg, dp, ep)
        dj, ej = np.concatenate(jit["d"]), np.concatenate(jit["e"])
        r["jitter_plain_vs_shifted"] = jitter_table(dj, ej)
        dj, ej = np.concatenate(jit_blend["d"]), np.concatenate(jit_blend["e"])
        r["jitter_blend24"] = jitter_table(dj, ej)
        print(json.dumps(r), flush=True)
        results.append(r)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    return results


def bin_table(dg, eg, dp, ep):
    rows = []
    for lo, hi in zip(SEAM_BINS[:-1], SEAM_BINS[1:]):
        g = dg[(eg >= lo) & (eg < hi)]
        p = dp[(ep >= lo) & (ep < hi)]
        inl = g[g <= M.MISS_PX]
        rows.append({"seam_dist": f"[{lo},{hi})", "n_gt": int(len(g)),
                     "bscore": float(0.5 * (np.minimum(g, 1).mean() + np.minimum(p, 1).mean())),
                     "missed": float((g > M.MISS_PX).mean()), "extra": float((p > M.MISS_PX).mean()),
                     "loc_mean": float(inl.mean()), "loc_p95": float(np.percentile(inl, 95))})
    return rows


def jitter_table(d, e):
    rows = []
    for lo, hi in zip(SEAM_BINS[:-1], SEAM_BINS[1:]):
        x = d[(e >= lo) & (e < hi)]
        rows.append({"seam_dist": f"[{lo},{hi})", "n": int(len(x)),
                     "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
                     "p99": float(np.percentile(x, 99)), "frac_gt_0.1px": float((x > 0.1).mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["throughput", "seams"])
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--regions", type=int, default=6)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(WORK_DIR, "fullimage"), exist_ok=True)
    out = a.out or os.path.join(WORK_DIR, "fullimage", f"{a.cmd}.json")
    if a.cmd == "throughput":
        throughput(a.ckpts, out)
    else:
        seams(a.ckpts, out, a.regions, a.size)


if __name__ == "__main__":
    main()
