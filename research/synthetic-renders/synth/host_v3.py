"""Host-side driver for physical_v3 scenes: mask plans and shared mask storage.

The output drive is exFAT with 128 KiB clusters, so every tile file costs at least
128 KiB of disk. Masks are therefore shared where they are identical:

- ``base_masks/<part_id>/`` holds each catalog part's untilted mask, copied once from a
  previous dataset (``--base-masks``) whose masks were rendered with the same geometry,
  projection and Blender version.
- ``base_masks/_uniform/`` holds one all-background and one all-object tile.
- A scene's ``mask/index.json`` lists every tile as usual. A tile's ``file`` is either a
  file inside the scene's own ``mask/`` or a relative path such as
  ``../../base_masks/part_0080_plate/x00000_y00000.tif``. Readers that join
  ``scene/mask/<file>`` (validation, SceneReader, edge-model cache) work unchanged.

``metadata["mask_key"]`` names the mask identity: the part id for an untouched base
mask, otherwise the scene id, so consumers can deduplicate identical masks.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading

from PIL import Image, ImageChops

_TIFF_LOCK = threading.Lock()
_BASE_INDEX = {}
TILE = 2048


def _load(path):
    with open(path) as stream:
        return json.load(stream)


def _write_json(path, data):
    with open(path, "w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)


def base_mask_sources(base_root):
    """part_id -> scene directory of a previous dataset that rendered that part's mask."""
    if base_root in _BASE_INDEX:
        return _BASE_INDEX[base_root]
    found = {}
    for name in sorted(os.listdir(base_root)):
        meta_path = os.path.join(base_root, name, "metadata.json")
        if not name.startswith("scene_") or not os.path.isfile(meta_path):
            continue
        meta = _load(meta_path)
        part = meta.get("catalog", {}).get("part_id")
        if part and part not in found and "mask_reused_from_scene_id" not in meta["render_result"] \
                and meta.get("appearance_profile") != "physical_v3":
            found[part] = os.path.join(base_root, name)
        if len(found) >= 300:
            break
    _BASE_INDEX[base_root] = found
    return found


def _uniform_dir(output):
    path = os.path.join(output, "base_masks", "_uniform")
    if not os.path.isfile(os.path.join(path, "full.tif")):
        os.makedirs(path, exist_ok=True)
        for name, value in (("empty.tif", 0), ("full.tif", 1)):
            temp = os.path.join(path, "." + name)
            Image.new("1", (TILE, TILE), value).save(temp, format="TIFF", compression="group4")
            os.replace(temp, os.path.join(path, name))
    return path


def ensure_base_mask(manifest, output, base_root):
    """Copy the part's base mask into output/base_masks once; return (dir, base metadata)."""
    part = manifest["catalog"]["part_id"]
    source = base_mask_sources(base_root).get(part) if base_root else None
    if source is None:
        return None
    base = _load(os.path.join(source, "metadata.json"))
    if (base["geometry"] != manifest["geometry"] or base["projection"] != manifest["projection"] or
            base["blender_version"] != manifest["expected_blender_version"]):
        raise ValueError(f"base mask {source} is incompatible with {manifest['scene_id']}")
    target = os.path.join(output, "base_masks", part)
    with _TIFF_LOCK:
        if not os.path.isfile(os.path.join(target, "index.json")):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            temp = target + ".partial"
            if os.path.exists(temp):
                shutil.rmtree(temp)
            shutil.copytree(os.path.join(source, base["files"]["mask_dir"]), temp)
            _write_json(os.path.join(temp, "source.json"),
                        {"scene_id": base["scene_id"], "part_id": part,
                         "render_result_mask": base["render_result"]["mask"]})
            os.rename(temp, target)
    return target, base


def _shared_index(base_dir, part):
    index = _load(os.path.join(base_dir, "index.json"))
    for tile in index["tiles"]:
        tile["file"] = f"../../base_masks/{part}/{tile['file']}"
    return index


def _to_tiff(source, target):
    with Image.open(source) as image:
        image = image.convert("1")
        image.save(target, format="TIFF", compression="group4")
    os.remove(source)


def _finish_full_mask(temp, output):
    """Convert rendered PNG tiles to TIFF and point uniform tiles at shared files."""
    mask_dir = os.path.join(temp, "mask")
    index = _load(os.path.join(mask_dir, "index.json"))
    uniform = None
    for tile in index["tiles"]:
        full = tile["foreground_pixels"] == tile["width"] * tile["height"]
        if tile["file"] is None:
            # Uniform tile the worker did not write.
            if tile["width"] == tile["height"] == TILE:
                uniform = uniform or _uniform_dir(output)
                tile["file"] = "../../base_masks/_uniform/" + ("full.tif" if full else "empty.tif")
            else:
                name = f"x{tile['x']:05d}_y{tile['y']:05d}.tif"
                Image.new("1", (tile["width"], tile["height"]), int(full)).save(
                    os.path.join(mask_dir, name), format="TIFF", compression="group4")
                tile["file"] = name
        else:
            name = os.path.splitext(tile["file"])[0] + ".tif"
            _to_tiff(os.path.join(mask_dir, tile["file"]), os.path.join(mask_dir, name))
            tile["file"] = name
    index["format"] = "TIFF Group 4, 1-bit grayscale"
    index["bit_order"] = "0 background, 1 object"
    _write_json(os.path.join(mask_dir, "index.json"), index)
    return index


def _merge_extra(temp, base_dir, part):
    """OR the worker's contaminant-only tiles into the shared base mask."""
    index = _shared_index(base_dir, part)
    extra_dir = os.path.join(temp, "mask_extra")
    extra = _load(os.path.join(extra_dir, "index.json"))
    by_xy = {(t["x"], t["y"]): t for t in index["tiles"]}
    mask_dir = os.path.join(temp, "mask")
    os.makedirs(mask_dir, exist_ok=True)
    changed = 0
    for t in extra["tiles"]:
        if t["foreground_pixels"] == 0:
            continue
        tile = by_xy[(t["x"], t["y"])]
        base_name = os.path.basename(tile["file"])
        with Image.open(os.path.join(base_dir, base_name)) as b, \
                Image.open(os.path.join(extra_dir, t["file"])) as e:
            merged = ImageChops.logical_or(b.convert("1"), e.convert("1"))
        count = merged.histogram()[255]
        if count == tile["foreground_pixels"]:
            continue
        merged.save(os.path.join(mask_dir, base_name), format="TIFF", compression="group4")
        tile["file"] = base_name
        tile["foreground_pixels"] = count
        changed += 1
    index["foreground_pixels"] = sum(t["foreground_pixels"] for t in index["tiles"])
    _write_json(os.path.join(mask_dir, "index.json"), index)
    shutil.rmtree(extra_dir)
    return index, changed


def run_one_v3(manifest, output, blender, threads, base_root, worker_log, worker, project_root,
               validate_scene):
    scene_id = manifest["scene_id"]
    target = os.path.join(output, scene_id)
    if os.path.exists(target):
        with _TIFF_LOCK:
            validate_scene(target)
        print(f"SKIP {scene_id}: {target} exists", flush=True)
        return False
    plan = manifest["mask_plan"]
    base = ensure_base_mask(manifest, output, base_root) if plan != "full" else None
    if plan != "full" and base is None:
        manifest["mask_plan_requested"] = plan
        plan = manifest["mask_plan"] = "full"
    temp = tempfile.mkdtemp(prefix=f".{scene_id}.", dir=output)
    manifest_path = os.path.join(temp, "input_manifest.json")
    _write_json(manifest_path, manifest)
    cmd = [blender, "-b", "-noaudio", "--factory-startup", "-t", str(threads),
           "--python", worker, "--", manifest_path, temp]
    cmd += {"base": ["--source-only"], "base_plus_contaminants": ["--contaminant-mask"],
            "full": []}[plan]
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    try:
        with open(worker_log or os.devnull, "w") as stream:
            subprocess.run(cmd, check=True, env=env, stdout=stream, stderr=subprocess.STDOUT)
        os.remove(manifest_path)
        # Pillow's libtiff codec is not thread-safe (concurrent use fails with EBADF /
        # "Cannot read TIFF header"), so all TIFF work and validation is serialized.
        with _TIFF_LOCK:
            return _postprocess(manifest, output, temp, target, plan, base, scene_id, validate_scene)
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"FAILED {scene_id}; files retained in {temp}", file=sys.stderr, flush=True)
        raise


def _postprocess(manifest, output, temp, target, plan, base, scene_id, validate_scene):
    meta_path = os.path.join(temp, "metadata.json")
    meta = _load(meta_path)
    part = manifest["catalog"]["part_id"]
    if plan == "full":
        index = _finish_full_mask(temp, output)
        meta["mask_key"] = scene_id
    else:
        base_dir, base_meta = base
        if os.path.isdir(os.path.join(temp, "mask_extra")):
            index, changed = _merge_extra(temp, base_dir, part)
        else:
            index, changed = _shared_index(base_dir, part), 0
            os.makedirs(os.path.join(temp, "mask"), exist_ok=True)
            _write_json(os.path.join(temp, "mask", "index.json"), index)
        worker_mask = meta["render_result"]["mask"]
        meta["render_result"]["mask"] = {
            "base_scene_id": base_meta["scene_id"], "tiles_changed": changed,
            "contaminant_tiles_rendered": worker_mask.get("tiles_rendered", 0),
            "seconds": worker_mask.get("seconds", 0.0),
            "foreground_pixels": index["foreground_pixels"], "tile_count": len(index["tiles"])}
        meta["render_result"]["mask_reused_from_scene_id"] = base_meta["scene_id"]
        meta["mask_key"] = scene_id if changed else part
    _write_json(meta_path, meta)
    result = validate_scene(temp)
    os.rename(temp, target)
    print("VALIDATED", json.dumps(result), flush=True)
    return True
