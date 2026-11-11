"""Blender side of macro scenes: build parts and debris, render image and masks.

    blender -b --python macro/worker.py -- scene.json scene.npz OUTPUT_DIR

Writes image.png (16-bit, same response as synthetic-renders), instances.png (uint16,
0 background, k = part index + 1; majority label of a 4x supersampled flat render, so a
pixel belongs to the part covering most of it; wires thinner than 2 px also label the
pixels they cover by >= 1/8-1/4, see main), debris.png (8-bit coverage of loose debris,
which is in no part mask) and metadata.json.
"""

import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "synthetic-renders"))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

from synth import geometry  # noqa: E402
from synth import worker_v3 as W  # noqa: E402
from synth.blender_worker import _render_pixels, _render_source  # noqa: E402
from synth.png import write_gray16  # noqa: E402


# ------------------------------------------------------------------------ parts

def _extrude(part, arrays):
    loops = []
    for key, hole in zip(part["rings"], part["holes"]):
        ring = [tuple(p) for p in arrays[key]]
        area = geometry._signed_area(ring)
        if (area < 0) != hole:
            ring.reverse()
        loops.append(ring)
    return [geometry._curve_object("part_extrude", loops, part["thickness_px"])]


def _lathe(profile, err=0.03):
    """Solid of revolution of an (x, r) polyline about the x axis, closed at both ends."""
    prof = [(float(x), max(0.0, float(r))) for x, r in profile]
    prof = [(prof[0][0], 0.0)] + prof + [(prof[-1][0], 0.0)]
    rmax = max(r for _, r in prof)
    n = geometry._segments(max(rmax, 0.5), error=err, minimum=24, maximum=384)
    verts, rings = [], []
    for x, r in prof:
        if r <= 1e-9:
            rings.append([len(verts)])
            verts.append((x, 0.0, 0.0))
        else:
            ids = []
            for k in range(n):
                t = math.tau * k / n
                ids.append(len(verts))
                verts.append((x, r * math.cos(t), r * math.sin(t)))
            rings.append(ids)
    faces = []
    for a, b in zip(rings[:-1], rings[1:]):
        if len(a) == 1 and len(b) == 1:
            continue
        for k in range(n):
            k1 = (k + 1) % n
            if len(a) == 1:
                faces.append((a[0], b[k], b[k1]))
            elif len(b) == 1:
                faces.append((a[k], b[0], a[k1]))
            else:
                faces.append((a[k], b[k], b[k1], a[k1]))
    obj = geometry._mesh("part_revolve", verts, faces)
    bpy.context.view_layer.objects.active = obj
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    return obj


def _machine(obj, cuts, rmax):
    for c in cuts:
        w = c["width_px"]
        if c["kind"] == "cross_hole":
            cutter = geometry._cylinder("cutter_cross", w / 2, 6 * rmax + 10)   # along z (view axis)
            cutter.location = (c["x_px"], 0, 0)
        else:
            bpy.ops.mesh.primitive_cube_add(size=1)
            cutter = bpy.context.object
            if c["kind"] == "through_slot":
                cutter.dimensions = (c["length_px"], w * 0.6, 6 * rmax + 10)
                cutter.location = (c["x_px"], 0, 0)
            else:   # flat on the side facing the camera
                cutter.dimensions = (c["length_px"], 4 * rmax, 2 * rmax)
                cutter.location = (c["x_px"], 0, rmax * (2 - 2 * c["depth_frac"]))
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        try:
            geometry._boolean(obj, cutter)
        except (RuntimeError, ValueError) as error:
            bpy.data.objects.remove(cutter, do_unlink=True)
            c["failed"] = str(error)


def build_part(part, arrays):
    kind = part["kind"]
    if kind == "extrude":
        objs = _extrude(part, arrays)
    elif kind == "revolve":
        prof = arrays[part["profile"]]
        obj = _lathe(prof)
        if part.get("cuts"):
            _machine(obj, part["cuts"], float(prof[:, 1].max()))
        geometry._smooth(obj, True)
        objs = [obj]
    elif kind == "sphere":
        r = part["radius_px"]
        cuts = [{"kind": "bore", "width_px": 2 * r * part["bore_frac"], "offset_px": 0.0}] if part["bore_frac"] else []
        objs = [geometry.sphere({"radius_px": r, "cuts": cuts, "partial": False,
                                 "tessellation_error_px": 0.02})]
    else:
        objs = [geometry.freeform({"radius_px": part["radius_px"], "lobes": part["lobes"],
                                   "ridge_amplitude_px": part["ridge_amplitude_px"],
                                   "saddle_amplitude_px": part["saddle_amplitude_px"],
                                   "faceted": part["faceted"], "tessellation_error_px": 0.02})]
    return objs


def pose_part(objs, pose, z_base):
    empty = bpy.data.objects.new("part_pose", None)
    bpy.context.collection.objects.link(empty)
    for o in objs:
        o.parent = empty
    axis = Vector((math.cos(pose["tilt_axis_rad"]), math.sin(pose["tilt_axis_rad"]), 0))
    rot = Matrix.Rotation(pose["tilt_rad"], 4, axis) @ Matrix.Rotation(pose["rot_rad"], 4, "Z")
    empty.matrix_world = rot
    v = W.world_vertices(objs)
    empty.matrix_world = Matrix.Translation((pose["x"], pose["y"], z_base - float(v[:, 2].min()))) @ rot
    v = W.world_vertices(objs)
    return empty, float(v[:, 2].min()), float(v[:, 2].max())


# ----------------------------------------------------------------------- debris

class _PartPlacer(W._Placer):
    """synthetic-renders placer driven by host boundary samples of one part."""

    def __init__(self, scene, size, targets, boundary=None, loose=None, zmin=0.0):
        self.scene, self.p = scene, {"source_width": size, "source_height": size}
        self.top, self.zmin = 10 * size, zmin
        self.targets = {o.name for o in targets}
        self.w = self.h = size
        self.lo, self.hi_x, self.hi_y = 2, size - 2, size - 2
        b = np.zeros((0, 4)) if boundary is None else np.asarray(boundary)
        self.bx, self.by, self.nx, self.ny = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        self._loose = loose

    def loose_point(self, rng, clearance):
        return None if self._loose is None else (*self._loose, self.zmin)


def place_debris(scene, manifest, arrays, part_objs, material_of):
    size = manifest["size_px"]
    all_targets = [o for objs in part_objs.values() for o in objs]
    attached, loose = {}, []
    for part in manifest["parts"]:
        if not part["debris"]:
            continue
        pl = _PartPlacer(scene, size, part_objs[part["index"]], arrays[part["boundary"]])
        for j, item in enumerate(part["debris"]):
            rng, np_rng = random.Random(item["seed"]), np.random.default_rng(item["seed"])
            tag = 1000 * (part["index"] + 1) + j
            if item["kind"] == "particle":
                made = W._particle(pl, item, rng, np_rng, tag)
            elif item["kind"] in ("fiber", "hair"):
                made = W._fiber(pl, item, rng, tag)
            else:
                made = W._burr(pl, item, rng, tag, material_of[part["index"]])
            item["placed"] = made is not None
            if made:
                attached.setdefault(part["index"], []).append(made[0])
    for j, item in enumerate(manifest["loose_debris"]):
        pl = _PartPlacer(scene, size, all_targets, loose=item["position_px"])
        rng, np_rng = random.Random(item["seed"]), np.random.default_rng(item["seed"])
        if item["kind"] == "particle":
            made = W._particle(pl, item, rng, np_rng, 90000 + j)
        else:
            made = W._fiber(pl, item, rng, 90000 + j)
        item["placed"] = made is not None
        if made:
            loose.append(made[0])
    return attached, loose


# ------------------------------------------------------------------------ masks

def _label_render(scene, camera, visible_groups, size, ss, scratch):
    """Flat-colour render at ss x size; returns uint8 ids (0 = none) of the visible groups."""
    W._workbench(scene)
    scene.render.image_settings.color_mode = "RGB"
    scene.render.dither_intensity = 0.0          # ids are exact colour codes
    names = {}
    for gid, objs in visible_groups.items():
        for o in objs:
            names[o.name] = gid
    for o in bpy.data.objects:
        if o.type == "CAMERA":
            continue
        o.hide_render = o.name not in names and o.type != "EMPTY"
        if o.name in names:
            o.color = (names[o.name] / 255.0, 0, 0, 1)
    scene.world.color = (1, 1, 1)
    camera.data.dof.use_dof = False
    scene.render.resolution_x = scene.render.resolution_y = size * ss
    path = os.path.join(scratch, ".labels.png")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path, check_existing=False)
    a = np.empty(size * ss * size * ss * 4, np.float32)
    img.pixels.foreach_get(a)
    bpy.data.images.remove(img)
    os.remove(path)
    a = a.reshape(size * ss, size * ss, 4)[::-1]
    ids = np.rint(a[:, :, 0] * 255).astype(np.int32)
    white = (a[:, :, 1] > 0.5) & (a[:, :, 2] > 0.5)
    ids[white] = 0
    return ids


def _majority(ids, size, ss, n_labels):
    blocks = ids.reshape(size, ss, size, ss)
    counts = np.stack([(blocks == k).sum((1, 3)) for k in range(n_labels + 1)])
    return counts.argmax(0).astype(np.uint16), counts


# ------------------------------------------------------------------------ main

def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    manifest_path, arrays_path, out = args[:3]
    with open(manifest_path) as f:
        manifest = json.load(f)
    arrays = dict(np.load(arrays_path))
    os.makedirs(out, exist_ok=True)
    scratch = W.scratch_dir()
    t0 = time.monotonic()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    size, r = manifest["size_px"], manifest["render"]
    scene.render.engine = "CYCLES"
    device = W.use_gpu(scene)
    scene.cycles.samples = r["cycles_samples"]
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = r["adaptive_threshold"]
    scene.cycles.adaptive_min_samples = r["adaptive_min_samples"]
    scene.cycles.use_denoising = r["denoise"]
    scene.cycles.pixel_filter_type = "BOX" if r["pixel_filter"] == "box" else "BLACKMAN_HARRIS"
    scene.render.filter_size = r["pixel_filter_width"]
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Raw"
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = manifest["illumination"]["ambient_strength"]
    cam_z = 4.0 * size
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    bpy.context.collection.objects.link(camera)
    camera.location = (0, 0, cam_z)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = size
    camera.data.clip_end = cam_z * 4
    scene.camera = camera

    # Parts, bottom-up so stacked parts rest on the part below.
    part_objs, material_of, z_range = {}, {}, {}
    by_index = {p["index"]: p for p in manifest["parts"]}
    order = sorted(manifest["parts"], key=lambda p: p["pose"]["stack_on"] is not None)
    for part in order:
        objs = build_part(part, arrays)
        mat = W._material_v3(part["material"], f"part_{part['index']}")
        for o in objs:
            o.data.materials.append(mat)
        below = part["pose"]["stack_on"]
        z_base = z_range[below][1] if below is not None else 0.0
        _, zmin, zmax = pose_part(objs, part["pose"], z_base)
        part_objs[part["index"]], material_of[part["index"]] = objs, mat
        z_range[part["index"]] = (zmin, zmax)
        part["realized_z_px"] = [round(zmin, 4), round(zmax, 4)]
    attached, loose = place_debris(scene, manifest, arrays, part_objs, material_of)

    # Focus on the parts' silhouette plane (median mid-depth) as in physical_v3.
    o = manifest["optics"]
    mids = sorted((a + b) / 2 for a, b in z_range.values())
    spans = [b - a for a, b in z_range.values()]
    z_ref = mids[len(mids) // 2] + (o["focus_depth_fraction"] - 0.5) * min(max(spans), 10.0)
    z_focus = z_ref + o["defocus_sign"] * o["target_coc_px"] / (2 * o["na"])
    focus_distance = cam_z - z_focus
    camera.data.dof.use_dof = True
    camera.data.dof.focus_distance = focus_distance
    camera.data.dof.aperture_fstop = 1 / (2 * o["na"] * focus_distance)
    zmin = min(a for a, _ in z_range.values())
    zmax = max(b for _, b in z_range.values())
    o["realized"] = {"z_focus": round(z_focus, 4), "focus_distance": round(focus_distance, 4),
                     "fstop": camera.data.dof.aperture_fstop,
                     "depth_coc_px": round(2 * o["na"] * max(zmax - z_ref, z_ref - zmin), 4)}

    b = manifest["backlight"]
    reach = o["na"] * (focus_distance + b["distance_px"]) * 1.5 + 50
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, min(0.0, zmin) - b["distance_px"]))
    backlight = bpy.context.object
    backlight.name = "visible_backlight"
    backlight.dimensions = (size + 2 * reach, size + 2 * reach, 0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    backlight.data.materials.append(W._backlight_material(b, size / 2))
    backlight.visible_shadow = False
    spill = manifest["illumination"]
    area_data = bpy.data.lights.new("front_spill", "AREA")
    area = bpy.data.objects.new("front_spill", area_data)
    bpy.context.collection.objects.link(area)
    ang = spill["spill_angle_rad"]
    rad = size * spill["spill_radius_fraction"]
    area.location = (rad * math.cos(ang), rad * math.sin(ang), cam_z * spill["spill_height_fraction"])
    area.rotation_euler = (-area.location).to_track_quat("-Z", "Y").to_euler()
    area_data.shape = "DISK"
    area_data.size = size * spill["spill_size_fraction"]
    area_data.energy = spill["spill_strength"] * size * size * 0.02

    # Source image (same response pipeline as synthetic-renders; its "files" contract).
    manifest["files"] = {"image": "image.png"}
    manifest["projection"] = {"source_width": size, "source_height": size}
    t1 = time.monotonic()
    post = lambda lin: W.apply_optics(lin, o)  # noqa: E731
    try:
        with W.gpu_slot(scene):
            source = _render_source(manifest, out, scene, post=post, scratch=scratch)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        scene.cycles.device = "CPU"
        device = "CPU (GPU out of memory)"
        source = _render_source(manifest, out, scene, post=post, scratch=scratch)
    render_seconds = time.monotonic() - t1

    # Instance masks: parts plus their attached debris; loose debris excluded.
    ss = r["mask_supersample"]
    groups = {p["index"] + 1: part_objs[p["index"]] + attached.get(p["index"], []) for p in manifest["parts"]}
    ids = _label_render(scene, camera, groups, size, ss, scratch)
    inst, counts = _majority(ids, size, ss, max(groups))
    # Thin wires (< 2 px): majority labels would leave a sub-pixel wire almost unlabeled,
    # so a background pixel is wire where the wire covers >= min(1/4, max(1/8, d/2)) of it.
    # That gives a continuous 1-2 px line. Normal parts keep majority labels (unbiased edges).
    for p in manifest["parts"]:
        d = p.get("wire_diameter_px")
        if p.get("group") == "wire" and d is not None and d < 2:
            k = p["index"] + 1
            need = max(1, int(round(ss * ss * min(0.25, max(0.125, d / 2)))))
            inst[(counts[k] >= need) & (inst == 0)] = k
            p["label_rule"] = f"coverage >= {need}/{ss * ss}"
    write_gray16(os.path.join(out, "instances.png"), inst)
    stats = []
    for p in manifest["parts"]:
        k = p["index"] + 1
        area_px = int((inst == k).sum())
        ys, xs = np.nonzero(inst == k)
        p["instance_id"] = k
        p["mask_area_px"] = area_px
        p["coverage_area_px"] = round(float(counts[k].sum()) / (ss * ss), 3)
        p["bbox_xywh"] = ([int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
                          if area_px else None)
        stats.append(area_px)
    if loose:
        dids = _label_render(scene, camera, {1: loose}, size, ss, scratch)
        cov = (dids.reshape(size, ss, size, ss) > 0).mean((1, 3))
    else:
        cov = np.zeros((size, size))
    from synth.png import write_png
    deb = np.rint(cov * 255).astype(np.uint8)
    write_png(os.path.join(out, "debris.png"), size, size, 8, 0, (row.tobytes() for row in deb))
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    manifest["render_result"] = {"source": source, "device": device, "render_seconds": round(render_seconds, 3),
                                 "duration_seconds": round(time.monotonic() - t0, 3),
                                 "blender_version": bpy.app.version_string,
                                 "instances": len(groups), "debris_pixels": int((cov >= 0.5).sum())}
    manifest["files"] = {"image": "image.png", "instances": "instances.png", "debris": "debris.png"}
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print("MACRO_SCENE_COMPLETE", manifest["scene_id"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
