"""Headless driver. The data format and validator do not import Blender."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from PIL import Image

from .catalog import catalog_sha256, sample_catalog_scene
from .sampling import sample_scene
from .validation import coverage, require_complete_coverage, validate_scene


DEFAULT_OUTPUT = "/media/kyle/T7 Shield/inspection-synthetic-renders"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(PROJECT_ROOT, "synth", "blender_worker.py")
_POSTPROCESS_LOCK = threading.Lock()


def _write_json(path, data):
    with open(path, "w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)


def _load(path):
    with open(path) as stream:
        return json.load(stream)


def _load_catalog(path):
    document = _load(path)
    parts = document["parts"]
    if catalog_sha256(parts) != document["catalog_sha256"]:
        raise ValueError("catalog SHA-256 does not match its parts")
    if len(parts) != document["count"] or any(p["resolution"] != document["resolution"] for p in parts):
        raise ValueError("catalog count or resolution is inconsistent")
    return document


def _sample_manifest(args, index, catalog):
    if catalog:
        resolution = catalog["resolution"]
        if (args.width is not None and args.width != resolution) or (args.height is not None and args.height != resolution):
            raise ValueError("--width and --height must match the catalog resolution")
        return sample_catalog_scene(args.seed, index, catalog["parts"], catalog["catalog_sha256"])
    return sample_scene(args.seed, index, args.width or 512, args.height or 512)


def _freeze_catalog(output, catalog):
    os.makedirs(output, exist_ok=True)
    path = os.path.join(output, "base_catalog.json")
    if os.path.exists(path):
        existing = _load_catalog(path)
        if existing["catalog_sha256"] != catalog["catalog_sha256"]:
            raise ValueError("output already contains a different base catalog")
    else:
        _write_json(path, catalog)


def _compress_mask_tiles(output_dir, manifest):
    index_path = os.path.join(output_dir, manifest["files"]["mask_index"])
    index = _load(index_path)
    mask_dir = os.path.join(output_dir, manifest["files"]["mask_dir"])
    for tile in index["tiles"]:
        source = os.path.join(mask_dir, tile["file"])
        target_name = os.path.splitext(tile["file"])[0] + ".tif"
        target = os.path.join(mask_dir, target_name)
        if not os.path.exists(source):
            if not os.path.exists(target):
                raise FileNotFoundError(source)
        else:
            for attempt in range(3):
                try:
                    with Image.open(source) as image:
                        assert image.mode == "1"
                        image.save(target, format="TIFF", compression="group4")
                    with Image.open(target) as image:
                        assert image.mode == "1" and image.size == (tile["width"], tile["height"])
                    os.remove(source)
                    break
                except (OSError, RuntimeError):
                    if attempt == 2:
                        raise
                    if os.path.exists(target):
                        os.remove(target)
                    time.sleep(attempt + 1)
        tile["file"] = target_name
    index["format"] = "TIFF Group 4, 1-bit grayscale"
    index["bit_order"] = "0 background, 1 object"
    _write_json(index_path, index)


def _reusable_catalog_mask(manifest, output):
    catalog = manifest.get("catalog")
    if not catalog or catalog["appearance_variant_index"] == 0:
        return None
    catalog_count = _load(os.path.join(output, "base_catalog.json"))["count"]
    base_index = manifest["scene_index"] % catalog_count
    base_scene_id = manifest["scene_id"].rsplit("_", 1)[0] + f"_{base_index:08d}"
    base_dir = os.path.join(output, base_scene_id)
    metadata_path = os.path.join(base_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        return None
    base = _load(metadata_path)
    if (base["catalog"]["part_id"] != catalog["part_id"] or
            base["geometry"] != manifest["geometry"] or
            base["projection"] != manifest["projection"] or
            base["blender_version"] != manifest["expected_blender_version"]):
        raise ValueError(f"base mask is incompatible with {manifest['scene_id']}")
    return base_dir, base


def _run_one(manifest, output, blender, threads, overwrite=False, worker_log=None, base_masks=None):
    if manifest.get("appearance_profile") == "physical_v3":
        if overwrite:
            raise ValueError("--overwrite is not supported for physical_v3 scenes")
        from .host_v3 import run_one_v3
        return run_one_v3(manifest, output, blender, threads, base_masks, worker_log, WORKER,
                          PROJECT_ROOT, validate_scene)
    scene_id = manifest["scene_id"]
    target = os.path.join(output, scene_id)
    if os.path.exists(target):
        if not overwrite:
            validate_scene(target)
            print(f"SKIP {scene_id}: {target} exists", flush=True)
            return False
    os.makedirs(output, exist_ok=True)
    temp = tempfile.mkdtemp(prefix=f".{scene_id}.", dir=output)
    manifest_path = os.path.join(temp, "input_manifest.json")
    _write_json(manifest_path, manifest)
    reusable = _reusable_catalog_mask(manifest, output)
    cmd = [blender, "-b", "-noaudio", "--factory-startup", "-t", str(threads),
           "--python", WORKER, "--", manifest_path, temp]
    if reusable:
        cmd.append("--source-only")
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    try:
        if worker_log is None:
            subprocess.run(cmd, check=True, env=env)
        else:
            with open(worker_log, "w") as stream:
                subprocess.run(cmd, check=True, env=env, stdout=stream, stderr=subprocess.STDOUT)
        # Separate Blender workers can finish together. Serialize Pillow TIFF
        # work to avoid concurrent encoder failures observed in this run.
        with _POSTPROCESS_LOCK:
            os.remove(manifest_path)
            if reusable:
                base_dir, base = reusable
                shutil.copytree(os.path.join(base_dir, base["files"]["mask_dir"]),
                                os.path.join(temp, manifest["files"]["mask_dir"]))
                produced = _load(os.path.join(temp, "metadata.json"))
                produced["render_result"]["mask"] = base["render_result"]["mask"]
                produced["render_result"]["mask_reused_from_scene_id"] = base["scene_id"]
                _write_json(os.path.join(temp, "metadata.json"), produced)
            else:
                _compress_mask_tiles(temp, manifest)
            result = validate_scene(temp)
            if overwrite and os.path.exists(target):
                backup = tempfile.mkdtemp(prefix=f".{scene_id}.backup.", dir=output)
                os.rmdir(backup)
                os.rename(target, backup)
                try:
                    os.rename(temp, target)
                except Exception:
                    os.rename(backup, target)
                    raise
                shutil.rmtree(backup)
            else:
                os.rename(temp, target)
        if reusable:
            print(f"MASK_REUSED {scene_id} from {reusable[1]['scene_id']}", flush=True)
        print("VALIDATED", json.dumps(result), flush=True)
        return True
    except Exception:
        print(f"FAILED {scene_id}; files retained in {temp}", file=sys.stderr, flush=True)
        raise


def _cmd_generate(args):
    if args.catalog and args.manifest:
        raise ValueError("--catalog cannot be combined with --manifest")
    catalog = _load_catalog(args.catalog) if args.catalog else None
    if catalog:
        _freeze_catalog(args.output, catalog)
    if args.target_gb is not None and args.manifest:
        raise ValueError("--target-gb cannot be combined with --manifest")
    if args.target_gb is not None:
        if args.target_gb <= 0:
            raise ValueError("--target-gb must be positive")
        if args.overwrite:
            raise ValueError("--target-gb cannot be combined with --overwrite")
        os.makedirs(args.output, exist_ok=True)
        target_bytes = int(args.target_gb * 1_000_000_000)
        total = _completed_bytes(args.output)
        index = args.start
        processed = 0
        while total < target_bytes and (args.count is None or processed < args.count):
            manifest = _sample_manifest(args, index, catalog)
            _apply_render_overrides(manifest, args)
            created = _run_one(manifest, args.output, args.blender, args.threads, args.overwrite)
            if created:
                total += _directory_bytes(os.path.join(args.output, manifest["scene_id"]))
            print(f"DATASET_BYTES {total} / {target_bytes}", flush=True)
            index += 1
            processed += 1
        print(f"DATASET_TOTAL {total} / {target_bytes} bytes; next scene index {index}", flush=True)
        return
    if args.manifest:
        manifests = [_load(path) for path in args.manifest]
    else:
        manifests = [_sample_manifest(args, i, catalog)
                     for i in range(args.start, args.start + (args.count or 1))]
    for manifest in manifests:
        _apply_render_overrides(manifest, args)
        _run_one(manifest, args.output, args.blender, args.threads, args.overwrite)


def _apply_render_overrides(manifest, args):
    if args.samples is not None:
        manifest["render"]["cycles_samples"] = args.samples
    if args.tile_size is not None:
        manifest["render"]["mask_tile_size"] = args.tile_size


def _directory_bytes(path):
    return sum(os.path.getsize(os.path.join(root, name))
               for root, _, names in os.walk(path) for name in names)


def _completed_bytes(path):
    total = 0
    for name in os.listdir(path):
        scene_path = os.path.join(path, name)
        if name.startswith("scene_") and os.path.isfile(os.path.join(scene_path, "metadata.json")):
            total += _directory_bytes(scene_path)
    return total


def _cmd_plan(args):
    catalog = _load_catalog(args.catalog) if args.catalog else None
    manifests = [_sample_manifest(args, i, catalog)
                 for i in range(args.start, args.start + (args.count or 1))]
    os.makedirs(args.output, exist_ok=True)
    for manifest in manifests:
        _write_json(os.path.join(args.output, manifest["scene_id"] + ".json"), manifest)
    print(json.dumps(coverage(manifests), indent=2))


def _cmd_validate(args):
    for path in args.paths:
        print(json.dumps(validate_scene(path), indent=2))


def _cmd_coverage(args):
    manifests = []
    for path in args.paths:
        if os.path.isdir(path):
            path = os.path.join(path, "metadata.json")
        manifests.append(_load(path))
    report = require_complete_coverage(manifests) if args.require_complete else coverage(manifests)
    print(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=20260922)
    common.add_argument("--start", type=int, default=0)
    common.add_argument("--count", type=int)
    common.add_argument("--width", type=int)
    common.add_argument("--height", type=int)
    common.add_argument("--catalog", help="reuse approved base geometry from a review catalog")
    plan = commands.add_parser("plan", parents=[common], help="write deterministic manifests")
    plan.add_argument("--output", required=True)
    plan.set_defaults(func=_cmd_plan)
    gen = commands.add_parser("generate", parents=[common], help="render and validate scenes")
    gen.add_argument("--output", default=DEFAULT_OUTPUT)
    gen.add_argument("--manifest", nargs="+")
    gen.add_argument("--blender", default="/snap/blender/current/blender")
    gen.add_argument("--threads", type=int, default=4)
    gen.add_argument("--samples", type=int)
    gen.add_argument("--tile-size", type=int)
    gen.add_argument("--target-gb", type=float, help="continue until completed scene files reach this decimal GB size")
    gen.add_argument("--overwrite", action="store_true")
    gen.set_defaults(func=_cmd_generate)
    val = commands.add_parser("validate", help="validate completed scene directories")
    val.add_argument("paths", nargs="+")
    val.set_defaults(func=_cmd_validate)
    cov = commands.add_parser("coverage", help="summarize scene manifest coverage")
    cov.add_argument("paths", nargs="+")
    cov.add_argument("--require-complete", action="store_true")
    cov.set_defaults(func=_cmd_coverage)
    args = parser.parse_args(argv)
    if hasattr(args, "count") and args.count is not None and args.count < 1:
        parser.error("--count must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
