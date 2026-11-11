"""Train-step throughput / memory microbenchmark on synthetic tensors.

Usage: python -m edgesr.gpubench [--set model.width=96 ...] --batches 8 16 32
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F

from .model import MaskSR
from .train import apply_sets, default_config


def bench(cfg, batch, channels_last, compile_, amp, steps=20):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fmt = torch.channels_last if channels_last else torch.contiguous_format
    model = MaskSR(cfg["model"]).cuda().to(memory_format=fmt)
    fwd = torch.compile(model) if compile_ else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.rand(batch, 1, 48, 48, device="cuda").to(memory_format=fmt)
    bl = torch.full((batch,), 0.7, device="cuda")
    y = (torch.rand(batch, 1, 768, 768, device="cuda") > 0.5).to(torch.uint8)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[amp]
    scaler = torch.amp.GradScaler(enabled=amp == "fp16")

    def step():
        with torch.autocast("cuda", dtype=dtype):
            out = fwd(x, bl)
        loss = F.binary_cross_entropy_with_logits(out.float(), y.float())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    try:
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        sps = batch * steps / (time.time() - t)
    except torch.OutOfMemoryError:
        return None, None
    finally:
        del model, fwd, opt
    return sps, torch.cuda.max_memory_allocated() / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--batches", nargs="*", type=int, default=[8, 16, 32])
    ap.add_argument("--variants", nargs="*", default=["cl", "nocl", "cl+compile", "cl+fp16"])
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    cfg = apply_sets(default_config(), a.set)
    for v in a.variants:
        for b in a.batches:
            sps, mem = bench(cfg, b, "nocl" not in v, "compile" in v, "fp16" if "fp16" in v else "bf16")
            print(json.dumps({"variant": v, "batch": b, "samples_per_s": sps and round(sps, 1),
                              "peak_mb": mem and round(mem)}), flush=True)


if __name__ == "__main__":
    main()
