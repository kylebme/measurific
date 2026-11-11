"""Render macro scenes in parallel; resumable.

    ../synthetic-shapes/.venv/bin/python -m macro.driver --output DIR --count 6000 --workers 6

Each worker process samples a scene (synthetic-shapes grammars, ~3 s of CPU), runs one
Blender process, validates the result and renames the temporary directory into
``DIR/scenes/<scene_id>``. Completed scenes are skipped. Processes rather than threads:
threads that start Blender subprocesses next to other file work produced sporadic EBADF
failures in the synthetic-renders run.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

from . import ROOT
from .sample import sample_macro_scene

WORKER = os.path.join(ROOT, "macro", "worker.py")


def validate(scene_dir):
    with open(os.path.join(scene_dir, "metadata.json")) as f:
        meta = json.load(f)
    size = meta["size_px"]
    with Image.open(os.path.join(scene_dir, "image.png")) as im:
        assert im.size == (size, size) and im.mode.startswith("I"), (im.size, im.mode)
    inst = np.array(Image.open(os.path.join(scene_dir, "instances.png")))
    assert inst.shape == (size, size)
    ids = set(np.unique(inst).tolist()) - {0}
    expected = {p["instance_id"] for p in meta["parts"]}
    assert ids <= expected, (ids, expected)
    missing = [p["instance_id"] for p in meta["parts"] if p["mask_area_px"] == 0]
    assert not missing, f"parts without any mask pixel: {missing}"
    with Image.open(os.path.join(scene_dir, "debris.png")) as im:
        assert im.size == (size, size)
    return {"scene_id": meta["scene_id"], "parts": len(meta["parts"]), "split": meta["split"]}


def run_scene(seed, index, output, blender, threads, wires=0.0):
    manifest, arrays = sample_macro_scene(seed, index, wires)
    target = os.path.join(output, "scenes", manifest["scene_id"])
    if os.path.isfile(os.path.join(target, "metadata.json")):
        return {"skipped": True, "scene_id": manifest["scene_id"]}
    temp = tempfile.mkdtemp(prefix=f".{manifest['scene_id']}.", dir=os.path.join(output, "scenes"))
    mpath, apath = os.path.join(temp, "input.json"), os.path.join(temp, "input.npz")
    with open(mpath, "w") as f:
        json.dump(manifest, f)
    np.savez(apath, **arrays)
    log = os.path.join(output, "worker_logs", manifest["scene_id"] + ".log")
    cmd = [blender, "-b", "-noaudio", "--factory-startup", "-t", str(threads),
           "--python", WORKER, "--", mpath, apath, temp]
    with open(log, "w") as stream:
        subprocess.run(cmd, check=True, stdout=stream, stderr=subprocess.STDOUT)
    os.remove(mpath)
    os.remove(apath)
    result = validate(temp)
    os.rename(temp, target)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads-per-worker", type=int, default=2)
    ap.add_argument("--blender", default="/snap/blender/current/blender")
    ap.add_argument("--wires", type=float, default=0.0,
                    help="fraction of scenes that also get 1-3 thin wires (0 = original sampler)")
    args = ap.parse_args(argv)
    os.makedirs(os.path.join(args.output, "scenes"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "worker_logs"), exist_ok=True)
    with open(os.path.join(args.output, "run_config.json"), "w") as f:
        json.dump({**vars(args), "argv": sys.argv}, f, indent=2)
    done = failed = 0
    t0 = time.time()
    indices = range(args.start, args.start + args.count)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_scene, args.seed, i, args.output, args.blender,
                               args.threads_per_worker, args.wires): i for i in indices}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                r = fut.result()
                done += 1
                print("DONE", i, json.dumps(r), f"{(time.time() - t0) / done:.2f} s/scene", flush=True)
            except Exception as error:   # keep going; a rerun retries missing scenes
                failed += 1
                print("FAILED", i, repr(error), flush=True)
    print(f"FINISHED done={done} failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
