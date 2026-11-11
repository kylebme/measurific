"""Invoked inside Blender: build one manifest and render source plus tiled masks."""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bpy
import numpy as np
from mathutils import Vector

from synth import geometry
from synth.png import write_binary, write_gray16
from synth.projection import tile_camera
from synth.sampling import _source_digest


def _render_pixels(scene, width, height, path, file_format):
    scene.render.filepath = path
    scene.render.image_settings.file_format = file_format
    scene.render.image_settings.color_mode = "RGBA"
    if file_format == "OPEN_EXR":
        scene.render.image_settings.color_depth = "32"
    else:
        scene.render.image_settings.color_depth = "8"
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(path, check_existing=False)
    array = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(array)
    bpy.data.images.remove(image)
    os.remove(path)
    return array.reshape(height, width, 4)[::-1].copy()


def _material(m):
    mat = bpy.data.materials.new("target_" + m["family"])
    mat.diffuse_color = (m["albedo"],) * 3 + (1,)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (m["albedo"],) * 3 + (1,)
    bsdf.inputs["Roughness"].default_value = m["roughness"]
    bsdf.inputs["Metallic"].default_value = m["metallic"]
    bsdf.inputs["Coat Weight"].default_value = m["coat_weight"]
    return mat


def _background_material(b):
    mat = bpy.data.materials.new("visible_backlight")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    texcoord = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 2.0
    noise.inputs["Roughness"].default_value = 0.4
    gradient = nodes.new("ShaderNodeMath")
    gradient.operation = "MULTIPLY_ADD"
    gradient.inputs[1].default_value = b["gradient"]
    gradient.inputs[2].default_value = b["intensity"] - b["gradient"] * 0.5
    texture = nodes.new("ShaderNodeMath")
    texture.operation = "MULTIPLY_ADD"
    texture.inputs[1].default_value = b["texture_strength"]
    texture.inputs[2].default_value = -b["texture_strength"] * 0.5
    total = nodes.new("ShaderNodeMath")
    total.operation = "ADD"
    links.new(texcoord.outputs["Generated"], separate.inputs[0])
    links.new(texcoord.outputs["Generated"], noise.inputs["Vector"])
    links.new(separate.outputs["X"], gradient.inputs[0])
    links.new(noise.outputs["Fac"], texture.inputs[0])
    links.new(gradient.outputs[0], total.inputs[0])
    links.new(texture.outputs[0], total.inputs[1])
    links.new(total.outputs[0], emission.inputs["Color"])
    links.new(emission.outputs[0], out.inputs["Surface"])
    return mat


def _setup_scene(manifest):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    p = manifest["projection"]
    w, h = p["source_width"], p["source_height"]
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = manifest["render"]["cycles_samples"]
    scene.cycles.use_denoising = manifest["render"]["denoise"]
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.filter_size = manifest["render"]["pixel_filter_width"]
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Raw"
    scene.render.image_settings.color_mode = "RGBA"
    scene.world.color = (0, 0, 0)
    scene.world.use_nodes = True
    world_bg = scene.world.node_tree.nodes.get("Background")
    world_bg.inputs["Color"].default_value = (1, 1, 1, 1)
    world_bg.inputs["Strength"].default_value = manifest["illumination"]["ambient_strength"]

    camera_data = bpy.data.cameras.new("reference_ortho")
    camera = bpy.data.objects.new("reference_ortho", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = tuple(p["camera_position_world"])
    camera.rotation_euler = tuple(p["camera_rotation_rad"])
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = p["ortho_scale_world"]
    camera_data.clip_end = max(1000, p["camera_position_world"][2] * 4)
    camera_data.dof.use_dof = True
    camera_data.dof.focus_distance = manifest["camera"]["focus_distance_world"]
    camera_data.dof.aperture_fstop = manifest["camera"]["fstop"]
    camera_data.dof.aperture_blades = manifest["camera"]["aperture_blades"]
    scene.camera = camera

    material = _material(manifest["material"])
    targets = geometry.build(manifest["geometry"])
    for obj in targets:
        obj.data.materials.append(material)
        obj.color = (0, 0, 0, 1)

    catalog_scene = "catalog" in manifest
    backlight_z = -p["camera_position_world"][2] * 0.5 if catalog_scene else -35
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, backlight_z))
    backlight = bpy.context.object
    backlight.name = "visible_backlight"
    backlight.dimensions = (w + 100, h + 100, 0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    backlight.data.materials.append(_background_material(manifest["backlight"]))
    # The visible luminous plane should not block directional illumination.
    if hasattr(backlight, "visible_shadow"):
        backlight.visible_shadow = False

    b = manifest["backlight"]
    for i, dust in enumerate(b["dust"]):
        bpy.ops.mesh.primitive_circle_add(vertices=24, radius=dust["radius_px"], fill_type="NGON",
                                          location=(*dust["center_px"], backlight_z + 1))
        obj = bpy.context.object
        obj.name = f"background_dust_{i}"
        mat = bpy.data.materials.new(obj.name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        level = b["intensity"] * (1 - dust["opacity"])
        bsdf.inputs["Base Color"].default_value = (level, level, level, 1)
        bsdf.inputs["Roughness"].default_value = 1
        obj.data.materials.append(mat)

    sun_data = bpy.data.lights.new("collimated_backlight", "SUN")
    sun = bpy.data.objects.new("collimated_backlight", sun_data)
    bpy.context.collection.objects.link(sun)
    tilt = b.get("tilt_angle_rad", 0)
    azimuth = b.get("tilt_azimuth_rad", 0)
    ray = Vector((math.sin(tilt) * math.cos(azimuth),
                  math.sin(tilt) * math.sin(azimuth), math.cos(tilt)))
    sun.rotation_euler = (-ray).to_track_quat("-Z", "Y").to_euler()
    sun_data.energy = b["intensity"] * b.get("directional_strength", 0.5)
    sun_data.angle = b["collimation_angle_rad"]

    spill = manifest["illumination"]
    area_data = bpy.data.lights.new("front_spill", "AREA")
    area = bpy.data.objects.new("front_spill", area_data)
    bpy.context.collection.objects.link(area)
    a = spill["spill_angle_rad"]
    area_z = (p["camera_position_world"][2] * spill.get("spill_height_fraction", 0.5)
              if catalog_scene else 330)
    radius = h * spill.get("spill_radius_fraction", 0.35)
    area.location = (radius * math.cos(a), radius * math.sin(a), area_z)
    direction = -area.location
    area.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    area_data.shape = "DISK"
    area_data.size = h * spill.get("spill_size_fraction", 0.7)
    area_data.energy = spill["spill_strength"] * h * h * 0.02
    return scene, camera, targets, backlight


def _render_source(manifest, output_dir, scene, post=None, scratch=None):
    w = manifest["projection"]["source_width"]
    h = manifest["projection"]["source_height"]
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.camera.data.dof.use_dof = True
    rgba = _render_pixels(scene, w, h, os.path.join(scratch or output_dir, ".source.exr"), "OPEN_EXR")
    rgb = np.maximum(0, rgba[:, :, :3])
    linear = rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722
    if post is not None:
        linear = post(linear)
    response = manifest["response"]
    effective_gain = response["gain"]
    intensity = np.power(np.maximum(0, linear * effective_gain + response["black_level"]),
                         response["gamma"])
    clipped = int(np.count_nonzero(intensity > response["clip_max"]))
    allowed_clipped = max(4, int(w * h * 0.001))
    if clipped > allowed_clipped:
        # Preserve sampled exposure unless bright specular patches would exceed
        # the output contract. Limit gain at the exact allowed pixel quantile.
        cutoff = float(np.partition(linear.ravel(), linear.size - allowed_clipped)
                       [linear.size - allowed_clipped])
        preclip_limit = response["clip_max"] ** (1 / response["gamma"])
        effective_gain = min(effective_gain,
                             0.999 * (preclip_limit - response["black_level"]) / cutoff)
        intensity = np.power(np.maximum(0, linear * effective_gain + response["black_level"]),
                             response["gamma"])
        clipped = int(np.count_nonzero(intensity > response["clip_max"]))
    response["effective_gain"] = float(effective_gain)
    intensity = np.clip(intensity, 0, response["clip_max"])
    image = np.rint(intensity * 65535).astype(np.uint16)
    write_gray16(os.path.join(output_dir, manifest["files"]["image"]), image)
    return {"minimum": int(image.min()), "maximum": int(image.max()),
            "mean": float(image.mean()), "response_clipped_pixels": clipped,
            "effective_gain": float(effective_gain)}


def _render_mask(manifest, output_dir, scene, camera, targets, backlight):
    p = manifest["projection"]
    scale = p["mask_scale"]
    full_w, full_h = p["source_width"] * scale, p["source_height"] * scale
    tile_size = manifest["render"]["mask_tile_size"]
    mask_dir = os.path.join(output_dir, manifest["files"]["mask_dir"])
    os.makedirs(mask_dir, exist_ok=True)
    camera.data.dof.use_dof = False
    backlight.hide_render = True
    for obj in bpy.data.objects:
        if obj.name.startswith("background_dust_") or obj.type == "LIGHT":
            obj.hide_render = True
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.world.color = (1, 1, 1)
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "OBJECT"
    scene.display.shading.show_shadows = False
    scene.display.shading.show_cavity = False
    scene.display.shading.show_object_outline = False
    scene.display.shading.background_type = "WORLD"
    scene.display.shading.background_color = (1, 1, 1)
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.display.render_aa = "OFF"
    scene.render.filter_size = 0.01
    full_scale = p["ortho_scale_world"]
    tiles = []
    foreground = 0
    for y in range(0, full_h, tile_size):
        for x in range(0, full_w, tile_size):
            tw = min(tile_size, full_w - x)
            th = min(tile_size, full_h - y)
            # Always render a square camera window. Blender changes ortho framing
            # for a non-square render aspect; crop the excess edge-tile pixels.
            cx, cy, tile_scale = tile_camera(full_w, full_h, full_scale, x, y, tile_size, tile_size)
            camera.data.ortho_scale = tile_scale
            camera.location.x = cx
            camera.location.y = cy
            scene.render.resolution_x = tile_size
            scene.render.resolution_y = tile_size
            rgb = _render_pixels(scene, tile_size, tile_size, os.path.join(output_dir, ".mask.png"), "PNG")[:th, :tw, :3]
            binary = rgb[:, :, 0] < 0.5
            count = int(binary.sum())
            foreground += count
            name = f"x{x:05d}_y{y:05d}.png"
            write_binary(os.path.join(mask_dir, name), binary)
            tiles.append({"x": x, "y": y, "width": tw, "height": th,
                          "file": name, "foreground_pixels": count})
            print(f"MASK_TILE {x} {y} {count}", flush=True)
    camera.data.ortho_scale = full_scale
    camera.location = tuple(p["camera_position_world"])
    index = {"schema_version": 1, "width": full_w, "height": full_h, "tile_size": tile_size,
             "bit_order": "PNG grayscale 1-bit, 0 background, 1 object", "tiles": tiles,
             "foreground_pixels": foreground}
    with open(os.path.join(output_dir, manifest["files"]["mask_index"]), "w") as stream:
        json.dump(index, stream, indent=2)
    return {"foreground_pixels": foreground, "tile_count": len(tiles)}


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    manifest_path, output_dir = args[:2]
    source_only = "--source-only" in args[2:]
    with open(manifest_path) as stream:
        manifest = json.load(stream)
    if manifest["schema_version"] != 1 or manifest["projection"]["mask_scale"] != 16:
        raise ValueError("unsupported manifest schema or mask scale")
    if manifest.get("expected_blender_version") != bpy.app.version_string:
        raise ValueError(f"Blender {manifest.get('expected_blender_version')} required; running {bpy.app.version_string}")
    if manifest.get("generator_source_sha256") != _source_digest():
        raise ValueError("manifest generator source digest differs from this checkout")
    os.makedirs(output_dir, exist_ok=True)
    start = time.monotonic()
    if manifest.get("appearance_profile") == "physical_v3":
        from synth import worker_v3
        contaminants_only = "--contaminant-mask" in args[2:]
        scratch = worker_v3.scratch_dir()
        scene, camera, targets, contaminants = worker_v3.setup_scene(manifest, scratch)
        t0 = time.monotonic()
        post = lambda linear: worker_v3.apply_optics(linear, manifest["optics"])  # noqa: E731
        try:
            with worker_v3.gpu_slot(scene):
                source = _render_source(manifest, output_dir, scene, post=post, scratch=scratch)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            # Same sampler and scene on the CPU; only the noise realization differs.
            print("GPU_OOM_FALLBACK_CPU", flush=True)
            scene.cycles.device = "CPU"
            manifest["render"]["device"] = "CPU (GPU out of memory)"
            source = _render_source(manifest, output_dir, scene, post=post, scratch=scratch)
        source["render_seconds"] = round(time.monotonic() - t0, 3)
        if source_only or (contaminants_only and not contaminants):
            mask = {"skipped": True}
        else:
            t0 = time.monotonic()
            mask = worker_v3.render_masks(manifest, output_dir, scene, camera, targets, contaminants,
                                          contaminants_only, scratch)
            mask["seconds"] = round(time.monotonic() - t0, 3)
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
    else:
        scene, camera, targets, backlight = _setup_scene(manifest)
        source = _render_source(manifest, output_dir, scene)
        mask = {"skipped": True} if source_only else _render_mask(manifest, output_dir, scene, camera, targets, backlight)
    manifest["blender_version"] = bpy.app.version_string
    manifest["render_result"] = {"source": source, "mask": mask,
                                 "duration_seconds": round(time.monotonic() - start, 3)}
    with open(os.path.join(output_dir, "metadata.json"), "w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    print("SCENE_COMPLETE", manifest["scene_id"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
