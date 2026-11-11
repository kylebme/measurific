"""Keep several independent catalog scenes in flight until a byte target is met."""

import argparse
import errno
import json
import os
import time
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone

from .appearance_v3 import sample_v3_scene
from .catalog import sample_catalog_scene
from .cli import (_completed_bytes, _directory_bytes, _freeze_catalog,
                  _load_catalog, _run_one)


def _sample_with_retry(seed, index, catalog, profile="expanded_v2"):
    sample = sample_v3_scene if profile == "physical_v3" else sample_catalog_scene
    for attempt in range(3):
        try:
            return sample(seed, index, catalog["parts"], catalog["catalog_sha256"])
        except OSError as error:
            if error.errno != errno.EBADF or attempt == 2:
                raise
            print(f"SAMPLE_RETRY {index}: {error!r}", flush=True)
            time.sleep(attempt + 1)


def _existing_indices(output, scene_prefix):
    indices = set()
    for name in os.listdir(output):
        if not name.startswith(scene_prefix):
            continue
        path = os.path.join(output, name)
        if os.path.isfile(os.path.join(path, "metadata.json")):
            indices.add(int(name.rsplit("_", 1)[1]))
    return indices


def _write_progress(output, total, target, completed, next_index, workers):
    path = os.path.join(output, "progress.json")
    temp = path + ".tmp"
    with open(temp, "w") as stream:
        json.dump({"completed_bytes": total, "target_bytes": target,
                   "completed_scenes": completed, "next_scene_index": next_index,
                   "workers": workers, "updated_at_utc": datetime.now(timezone.utc).isoformat()},
                  stream, indent=2)
        stream.write("\n")
    os.replace(temp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-gb", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, help="maximum number of new scenes to start")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--samples", type=int,
                        help="Cycles samples (expanded_v2 default 128; physical_v3 keeps its manifest value)")
    parser.add_argument("--profile", choices=("expanded_v2", "physical_v3"), default="expanded_v2")
    parser.add_argument("--base-masks", help="physical_v3: previous dataset whose untilted part masks are reused")
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--blender", default="/snap/blender/current/blender")
    args = parser.parse_args(argv)
    if (args.target_gb <= 0 or args.start < 0 or args.workers < 1 or
            args.threads_per_worker < 1 or (args.samples is not None and args.samples < 1) or
            args.tile_size < 1 or (args.count is not None and args.count < 1)):
        parser.error("target, workers, threads, samples, tile size, and count must be positive")

    catalog = _load_catalog(args.catalog)
    _freeze_catalog(args.output, catalog)
    logs = os.path.join(args.output, "worker_logs")
    os.makedirs(logs, exist_ok=True)
    target = int(args.target_gb * 1_000_000_000)
    total = _completed_bytes(args.output)
    if args.samples is None and args.profile == "expanded_v2":
        args.samples = 128
    prefix = _sample_with_retry(args.seed, 0, catalog, args.profile)["scene_id"].rsplit("_", 1)[0] + "_"
    existing = _existing_indices(args.output, prefix)
    next_index = args.start
    launched = 0
    failures = []
    pending = deque()
    attempts = defaultdict(int)

    def schedule(pool, active):
        nonlocal next_index, launched
        if total >= target or failures or (args.count is not None and launched >= args.count and not pending):
            return False
        retrying = bool(pending)
        if retrying:
            index = pending.popleft()
        else:
            while next_index in existing:
                next_index += 1
            index = next_index
        manifest = _sample_with_retry(args.seed, index, catalog, args.profile)
        if args.samples is not None:
            manifest["render"]["cycles_samples"] = args.samples
        manifest["render"]["mask_tile_size"] = args.tile_size
        log_path = os.path.join(logs, manifest["scene_id"] + ".log")
        future = pool.submit(_run_one, manifest, args.output, args.blender,
                             args.threads_per_worker, False, log_path, args.base_masks)
        active[future] = (index, manifest["scene_id"])
        print(f"{'RETRY_SCENE' if retrying else 'START_SCENE'} {index} {manifest['scene_id']}", flush=True)
        if not retrying:
            next_index += 1
            launched += 1
        return True

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active = {}
        for _ in range(args.workers):
            if not schedule(pool, active):
                break
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                index, scene_id = active.pop(future)
                try:
                    created = future.result()
                    if created:
                        total += _directory_bytes(os.path.join(args.output, scene_id))
                    existing.add(index)
                    print(f"DATASET_BYTES {total} / {target}", flush=True)
                except Exception as error:
                    attempts[index] += 1
                    if attempts[index] <= 2:
                        pending.append(index)
                        print(f"SCENE_ERROR {scene_id} {error!r}; retry {attempts[index]}/2", flush=True)
                    else:
                        failures.append((scene_id, repr(error)))
                        print(f"SCENE_ERROR {scene_id} {error!r}; retries exhausted", flush=True)
                _write_progress(args.output, total, target, len(existing), next_index, args.workers)
            for _ in range(args.workers - len(active)):
                if not schedule(pool, active):
                    break
    if failures:
        raise RuntimeError(f"{len(failures)} scene(s) failed: {failures}")
    print(f"DATASET_TOTAL {total} / {target} bytes; next scene index {next_index}", flush=True)


if __name__ == "__main__":
    main()
