"""Blender worker: render catalog geometry as native-resolution binary masks."""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bpy
import numpy as np

from synth import geometry
from synth.png import write_binary


def _clear():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for blocks in (bpy.data.meshes, bpy.data.curves, bpy.data.cameras):
        for block in list(blocks):
            if block.users == 0:
                blocks.remove(block)


def render_part(part, root):
    _clear()
    resolution = part["resolution"]
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.filter_size = 0.01
    scene.display.render_aa = "OFF"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "OBJECT"
    scene.display.shading.show_shadows = False
    scene.display.shading.show_cavity = False
    scene.display.shading.show_object_outline = False
    scene.display.shading.background_type = "WORLD"
    scene.display.shading.background_color = (1, 1, 1)
    scene.world.color = (1, 1, 1)
    scene.view_settings.view_transform = "Raw"
    camera_data = bpy.data.cameras.new("review_camera")
    camera = bpy.data.objects.new("review_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = resolution
    camera_data.clip_end = resolution * 8
    camera.location = (0, 0, resolution * 2)
    scene.camera = camera
    targets = geometry.build(part["geometry"])
    for target in targets:
        target.color = (0, 0, 0, 1)
    temp = os.path.join(root, ".review_render.png")
    scene.render.filepath = temp
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(temp, check_existing=False)
    pixels = np.empty(resolution * resolution * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    os.remove(temp)
    mask = pixels.reshape(resolution, resolution, 4)[::-1, :, 0] < 0.5
    foreground = int(mask.sum())
    ys, xs = np.where(mask)
    if foreground == 0:
        raise ValueError("empty mask")
    bounds = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    path = os.path.join(root, "masks", part["part_id"] + ".png")
    write_binary(path, mask)
    return {"part_id": part["part_id"], "resolution": resolution,
            "foreground_pixels": foreground, "foreground_fraction": round(foreground / mask.size, 6),
            "bounds_xyxy": bounds, "mask_file": "masks/" + part["part_id"] + ".png"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog")
    parser.add_argument("output")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    with open(args.catalog) as stream:
        parts = json.load(stream)["parts"]
    os.makedirs(os.path.join(args.output, "masks"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "results"), exist_ok=True)
    for part in parts[args.start:args.stop]:
        result_file = os.path.join(args.output, "results", part["part_id"] + ".json")
        if not args.force and os.path.isfile(result_file) and os.path.isfile(os.path.join(args.output, "masks", part["part_id"] + ".png")):
            print("REVIEW_MASK_SKIP", part["part_id"], flush=True)
            continue
        started = time.monotonic()
        try:
            result = render_part(part, args.output)
            result["elapsed_seconds"] = round(time.monotonic() - started, 3)
            with open(result_file + ".tmp", "w") as stream:
                json.dump(result, stream, indent=2)
            os.replace(result_file + ".tmp", result_file)
            print("REVIEW_MASK_DONE", part["part_id"], result["foreground_pixels"],
                  result["bounds_xyxy"], result["elapsed_seconds"], flush=True)
        except Exception as error:
            print("REVIEW_MASK_ERROR", part["part_id"], repr(error), flush=True)
            raise


if __name__ == "__main__":
    main()
    os._exit(0)
