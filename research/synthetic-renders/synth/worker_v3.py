"""physical_v3 scene construction, rendering and masks. Runs inside Blender.

See appearance_v3.py for the sampled parameters. Everything placed here is derived
deterministically from the manifest; realized values are written back into it.
"""

import contextlib
import math
import os
import random

import bpy
import numpy as np
from mathutils import Matrix, Vector

from synth import geometry
from synth.png import write_binary
from synth.projection import tile_camera

MARGIN_PX = 47
MASK_EDGE_PAD_PX = 3          # source px of slack around projected bounds and uniform tests


# ----------------------------------------------------------------------------- setup

def scratch_dir():
    """Per-process directory for Blender's write-then-read files (EXR, tile PNGs).

    These are written uncompressed and read straight back; on the USB exFAT output drive
    that took ~3 s per mask tile, so they go to RAM (SYNTH_SCRATCH, default /dev/shm)."""
    root = os.environ.get("SYNTH_SCRATCH", "/dev/shm")
    path = os.path.join(root, f"synth_{os.getpid()}")
    os.makedirs(path, exist_ok=True)
    return path


@contextlib.contextmanager
def gpu_slot(scene):
    """Hold one of SYNTH_GPU_SLOTS (default 3) inter-process GPU render slots.

    Parallel workers overlap their CPU stages; only a few Cycles renders share the GPU
    at a time so an 8 GB card does not run out of memory."""
    if scene.cycles.device != "GPU":
        yield
        return
    import fcntl
    import time
    slots = int(os.environ.get("SYNTH_GPU_SLOTS", "3"))
    lock_dir = os.environ.get("SYNTH_LOCK_DIR", "/tmp")
    while True:
        for i in range(slots):
            handle = open(os.path.join(lock_dir, f".synth_gpu_slot_{i}.lock"), "w")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()
            return
        time.sleep(0.2)


def use_gpu(scene):
    """Select an OptiX/CUDA device when available (SYNTH_DEVICE=CPU forces the CPU)."""
    wanted = os.environ.get("SYNTH_DEVICE", "GPU").upper()
    if wanted != "CPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
            except TypeError:
                continue
            prefs.get_devices()
            if any(d.type == kind for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type == kind
                scene.cycles.device = "GPU"
                return kind
    scene.cycles.device = "CPU"
    return "CPU"


def _px_to_world(p, x, y):
    return x - p["source_width"] / 2, p["source_height"] / 2 - y


def _world_to_px(p, x, y):
    return x + p["source_width"] / 2, p["source_height"] / 2 - y


def world_vertices(objs):
    """Evaluated world-space vertices of mesh and curve objects, (N, 3)."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for o in objs:
        if o.type not in ("MESH", "CURVE"):
            continue
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        n = len(me.vertices)
        if n:
            co = np.empty(n * 3, np.float64)
            me.vertices.foreach_get("co", co)
            m = np.array(oe.matrix_world)
            out.append(co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3])
        oe.to_mesh_clear()
    return np.concatenate(out) if out else np.zeros((0, 3))


def _material_v3(m, name="target_v3"):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (m["albedo"],) * 3 + (1,)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (m["albedo"],) * 3 + (1,)
    bsdf.inputs["Roughness"].default_value = m["roughness"]
    bsdf.inputs["Metallic"].default_value = m.get("metallic", 0.0)
    bsdf.inputs["Coat Weight"].default_value = m.get("coat_weight", 0.0)
    bump = m.get("bump")
    if bump:
        # Machining/brushing marks: anisotropic noise height in object coordinates.
        tc = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (1 / bump["scale_px"],
                                                 bump["anisotropy"] / bump["scale_px"],
                                                 1 / bump["scale_px"])
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Detail"].default_value = 4.0
        b = nodes.new("ShaderNodeBump")
        b.inputs["Strength"].default_value = bump["strength"]
        links.new(tc.outputs["Object"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
        links.new(noise.outputs["Fac"], b.inputs["Height"])
        links.new(b.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


class _Nodes:
    """Tiny helper for building scalar math node graphs."""

    def __init__(self, tree):
        self.nodes, self.links = tree.nodes, tree.links

    def _in(self, node, i, v):
        if isinstance(v, (int, float)):
            node.inputs[i].default_value = float(v)
        else:
            self.links.new(v, node.inputs[i])

    def math(self, op, a, b=0.0, c=None):
        n = self.nodes.new("ShaderNodeMath")
        n.operation = op
        self._in(n, 0, a)
        self._in(n, 1, b)
        if c is not None:
            self._in(n, 2, c)
        return n.outputs[0]

    def noise(self, x, y, scale_px, detail, w=0.0):
        comb = self.nodes.new("ShaderNodeCombineXYZ")
        self._in(comb, 0, self.math("DIVIDE", x, scale_px))
        self._in(comb, 1, self.math("DIVIDE", y, scale_px))
        comb.inputs[2].default_value = w
        n = self.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = 1.0
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = 0.5
        self.links.new(comb.outputs[0], n.inputs["Vector"])
        return n.outputs["Fac"]


def _backlight_material(b, half_width):
    mat = bpy.data.materials.new("visible_backlight_v3")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    N = _Nodes(tree)
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    emission = tree.nodes.new("ShaderNodeEmission")
    tc = tree.nodes.new("ShaderNodeTexCoord")
    sep = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(tc.outputs["Object"], sep.inputs[0])
    x, y = sep.outputs["X"], sep.outputs["Y"]
    total = 1.0

    def add(term):
        nonlocal total
        total = N.math("ADD", total, term)

    g = b["gradient"]
    if g["strength"]:
        proj = N.math("ADD", N.math("MULTIPLY", x, math.cos(g["angle_rad"])),
                      N.math("MULTIPLY", y, math.sin(g["angle_rad"])))
        add(N.math("MULTIPLY", proj, g["strength"] / (2 * half_width)))
    for key, detail in (("noise", 2.0), ("grain", 0.0)):
        c = b[key]
        if c["strength"]:
            fac = N.noise(x, y, c["scale_px"], detail, w=3.7 if key == "grain" else 0.0)
            add(N.math("MULTIPLY", N.math("SUBTRACT", fac, 0.5), 2 * c["strength"]))
    led = b["led_grid"]
    if led["strength"]:
        ca, sa = math.cos(led["angle_rad"]), math.sin(led["angle_rad"])
        u = N.math("ADD", N.math("MULTIPLY", x, ca), N.math("MULTIPLY", y, sa))
        v = N.math("SUBTRACT", N.math("MULTIPLY", y, ca), N.math("MULTIPLY", x, sa))
        k = math.tau / led["pitch_px"]
        cu = N.math("COSINE", N.math("MULTIPLY", u, k))
        cv = N.math("COSINE", N.math("MULTIPLY", v, k))
        add(N.math("MULTIPLY", N.math("MULTIPLY", cu, cv), led["strength"]))
    sm = b["smudges"]
    if sm["strength"]:
        fac = N.noise(x, y, sm["scale_px"], 3.0, w=11.3)
        # Noise Fac is ~N(0.5, 0.08); threshold for the requested area coverage.
        t = 0.5 + 0.08 * _probit(1 - sm["coverage"])
        mr = tree.nodes.new("ShaderNodeMapRange")
        mr.interpolation_type = "SMOOTHSTEP"
        tree.links.new(fac, mr.inputs["Value"])
        mr.inputs["From Min"].default_value = t
        mr.inputs["From Max"].default_value = t + 0.06
        add(N.math("MULTIPLY", mr.outputs["Result"], -sm["strength"]))
    for d in b["dust"]:
        # Diffuser dust transmits (1 - opacity); soft edge of ~1 px.
        dx = N.math("SUBTRACT", x, d["center_px"][0])
        dy = N.math("SUBTRACT", y, d["center_px"][1])
        r = N.math("SQRT", N.math("ADD", N.math("MULTIPLY", dx, dx), N.math("MULTIPLY", dy, dy)))
        inside = N.math("SUBTRACT", 1.0, N.math("SMOOTH_MAX", N.math("SUBTRACT", r, d["radius_px"]), 0.0, 0.5))
        inside = N.math("MAXIMUM", N.math("MINIMUM", inside, 1.0), 0.0)
        add(N.math("MULTIPLY", inside, -d["opacity"]))
    # Angular radiance profile: cos^n about the (slightly tilted) illuminator axis.
    ang = b["angular"]
    tilt, az = math.radians(ang["axis_tilt_deg"]), ang["axis_azimuth_rad"]
    axis = (math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), math.cos(tilt))
    half = math.radians(ang["half_angle_deg"])
    exponent = 0.0 if half >= math.pi / 2 - 1e-6 else math.log(0.5) / math.log(math.cos(half))
    geo = tree.nodes.new("ShaderNodeNewGeometry")
    dot = tree.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    tree.links.new(geo.outputs["Incoming"], dot.inputs[0])
    dot.inputs[1].default_value = axis
    profile = N.math("POWER", N.math("MAXIMUM", dot.outputs["Value"], 0.0), exponent)
    b["angular"]["exponent"] = round(exponent, 6)
    level = N.math("MULTIPLY", N.math("MULTIPLY", N.math("MAXIMUM", total, 0.0), profile), b["intensity"])
    tree.links.new(level, emission.inputs["Strength"])
    emission.inputs["Color"].default_value = (1, 1, 1, 1)
    tree.links.new(emission.outputs[0], out.inputs["Surface"])
    return mat


def _probit(p):
    # Acklam-free rational approximation is overkill here; bisection on erf is exact enough.
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ----------------------------------------------------------------------- binary views

def _workbench(scene):
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.world.color = (1, 1, 1)
    sh = scene.display.shading
    sh.light = "FLAT"
    sh.color_type = "OBJECT"
    sh.show_shadows = False
    sh.show_cavity = False
    sh.show_object_outline = False
    sh.background_type = "WORLD"
    sh.background_color = (1, 1, 1)
    scene.render.film_transparent = False
    scene.display.render_aa = "OFF"
    scene.render.filter_size = 0.01
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 0


def _only_visible(visible):
    names = {o.name for o in visible}
    for o in bpy.data.objects:
        if o.type == "CAMERA":
            continue
        o.hide_render = o.name not in names and o.type != "EMPTY"
    for o in visible:
        o.color = (0, 0, 0, 1)


def render_binary(scene, camera, full_w, full_h, full_scale, x, y, size, path):
    """Non-antialiased silhouette of the visible objects for one square tile."""
    cx, cy, tile_scale = tile_camera(full_w, full_h, full_scale, x, y, size, size)
    camera.data.ortho_scale = tile_scale
    camera.location.x, camera.location.y = cx, cy
    camera.data.dof.use_dof = False
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(path, check_existing=False)
    a = np.empty(size * size * 4, np.float32)
    image.pixels.foreach_get(a)
    bpy.data.images.remove(image)
    os.remove(path)
    return a.reshape(size, size, 4)[::-1, :, 0] < 0.5


# ----------------------------------------------------------------------- contaminants

def _ray_z(scene, x, y, top, targets):
    """World z of the first part surface below (x, y) seen from the camera, or None.

    Only the part (target objects) counts; the backlight and contaminants are skipped."""
    dg = bpy.context.evaluated_depsgraph_get()
    origin = Vector((x, y, top))
    for _ in range(16):
        hit, loc, _n, _i, obj, _m = scene.ray_cast(dg, origin, Vector((0, 0, -1)))
        if not hit:
            return None
        if obj.original.name in targets:
            return loc.z
        origin = Vector((x, y, loc.z - 1e-3))
    return None


def _box_blur(a, r):
    for axis in (0, 1):
        c = np.cumsum(np.pad(a, [(r + 1, r) if i == axis else (0, 0) for i in range(2)], mode="edge"),
                      axis=axis, dtype=np.float64)
        a = (np.take(c, range(2 * r + 1, c.shape[axis]), axis=axis) -
             np.take(c, range(0, c.shape[axis] - 2 * r - 1), axis=axis)) / (2 * r + 1)
    return a


def _boundary_samples(sil, margin):
    """Boundary pixel coordinates (x, y) of a 1x silhouette with outward unit normals."""
    inner = sil.copy()
    inner[1:] &= sil[:-1]
    inner[:-1] &= sil[1:]
    inner[:, 1:] &= sil[:, :-1]
    inner[:, :-1] &= sil[:, 1:]
    edge = sil & ~inner
    h, w = sil.shape
    edge[:margin] = edge[-margin:] = False
    edge[:, :margin] = edge[:, -margin:] = False
    ys, xs = np.nonzero(edge)
    smooth = sil.astype(np.float64)
    for _ in range(3):
        smooth = _box_blur(smooth, 2)
    gy, gx = np.gradient(smooth)
    nx, ny = -gx[ys, xs], -gy[ys, xs]
    norm = np.hypot(nx, ny)
    ok = norm > 1e-6
    return xs[ok] + 0.5, ys[ok] + 0.5, nx[ok] / norm[ok], ny[ok] / norm[ok]


class _Placer:
    def __init__(self, scene, manifest, sil, zmin, top, targets):
        self.scene, self.p, self.top, self.zmin = scene, manifest["projection"], top, zmin
        self.targets = {o.name for o in targets}
        self.w, self.h = self.p["source_width"], self.p["source_height"]
        self.lo, self.hi_x, self.hi_y = MARGIN_PX + 3, self.w - MARGIN_PX - 3, self.h - MARGIN_PX - 3
        self.bx, self.by, self.nx, self.ny = _boundary_samples(sil, MARGIN_PX + 40)

    def inside_frame(self, x, y):
        return self.lo <= x <= self.hi_x and self.lo <= y <= self.hi_y

    def surface_z(self, x, y):
        return _ray_z(self.scene, *_px_to_world(self.p, x, y), self.top, self.targets)

    def edge_point(self, rng):
        """A boundary point refined to ~1/16 px along its normal, plus normal and surface z."""
        for _ in range(20):
            i = rng.randrange(len(self.bx))
            x, y, nx, ny = self.bx[i], self.by[i], self.nx[i], self.ny[i]
            lo, hi = -3.0, 3.0            # s: offset along the outward normal
            if self.surface_z(x + lo * nx, y + lo * ny) is None or \
                    self.surface_z(x + hi * nx, y + hi * ny) is not None:
                continue
            for _ in range(7):
                mid = (lo + hi) / 2
                if self.surface_z(x + mid * nx, y + mid * ny) is None:
                    hi = mid
                else:
                    lo = mid
            ex, ey = x + lo * nx, y + lo * ny
            z = self.surface_z(ex - 0.75 * nx, ey - 0.75 * ny)
            return ex, ey, nx, ny, (z if z is not None else self.surface_z(ex, ey))
        return None

    def loose_point(self, rng, clearance):
        for _ in range(30):
            e = self.edge_point(rng)
            if e is None:
                continue
            d = rng.uniform(clearance + 3, clearance + 250)
            x, y = e[0] + e[2] * d, e[1] + e[3] * d
            if self.inside_frame(x, y) and self.surface_z(x, y) is None:
                # Debris rests in the object plane at the depth of the nearby silhouette,
                # so it is imaged like the part edge rather than far out of focus.
                return x, y, (e[4] if e[4] is not None else self.zmin)
        return None


def _mesh_object(name, verts, faces, material):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def _plain_material(name, albedo, roughness):
    return _material_v3({"albedo": albedo, "roughness": roughness}, name)


def _particle(pl, item, rng, np_rng, i):
    r = item["diameter_px"] / 2
    if item["attach"] == "edge":
        e = pl.edge_point(rng)
        if e is None:
            return None
        s = rng.uniform(-0.3, 0.8) * r
        cx, cy, z0 = e[0] + e[2] * s, e[1] + e[3] * s, e[4]
    else:
        pt = pl.loose_point(rng, r)
        if pt is None:
            return None
        cx, cy, z0 = pt
    if not pl.inside_frame(cx - r, cy - r) or not pl.inside_frame(cx + r, cy + r):
        return None
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3 if r > 4 else 2, radius=1.0)
    obj = bpy.context.object
    obj.name = f"contaminant_particle_{i}"
    n = len(obj.data.vertices)
    co = np.empty(n * 3)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    bump = np.zeros(n)
    for _ in range(4):
        d = np_rng.normal(size=3)
        bump += np_rng.uniform(0.05, 0.14) * np.sin(co @ (d * np_rng.uniform(1.5, 4)) + np_rng.uniform(0, 6.3))
    co *= (1 + bump)[:, None]
    co *= np.array([1.0, np_rng.uniform(0.55, 1.0), np_rng.uniform(0.4, 1.0)]) * r
    obj.data.vertices.foreach_set("co", co.ravel())
    obj.data.update()
    wx, wy = _px_to_world(pl.p, cx, cy)
    obj.location = (wx, wy, (z0 if z0 is not None else pl.zmin) + r * 0.5)
    obj.rotation_euler = tuple(np_rng.uniform(0, math.tau, 3))
    obj.data.materials.append(_plain_material(obj.name, item["albedo"], item["roughness"]))
    return obj, {"center_px": [round(cx, 3), round(cy, 3)], "z": round(obj.location.z, 3)}


def _walk(x, y, heading, length, step, radius_c, rng, pl):
    pts = []
    kappa = 0.0
    travelled = 0.0
    while travelled < length:
        kappa = 0.85 * kappa + rng.gauss(0, 0.6 / radius_c)
        heading += kappa * step
        x, y = x + math.cos(heading) * step, y + math.sin(heading) * step
        if not pl.inside_frame(x, y):
            break
        pts.append((x, y))
        travelled += step
    return pts


def _fiber(pl, item, rng, i):
    r = item["diameter_px"] / 2
    step = min(8.0, max(0.5, 2 * r))
    L = item["length_px"]
    if item["attach"] == "edge":
        e = pl.edge_point(rng)
        if e is None:
            return None
        ex, ey, nx, ny, z_edge = e
        h = math.atan2(ny, nx) + rng.uniform(-1.1, 1.1)
        outside = _walk(ex, ey, h, L * item["outside_fraction"], step, item["curvature_radius_px"], rng, pl)
        inside = _walk(ex, ey, h + math.pi, L * (1 - item["outside_fraction"]), step,
                       item["curvature_radius_px"], rng, pl)
        pts = inside[::-1] + [(ex, ey)] + outside
        z_default = z_edge if z_edge is not None else pl.zmin
    else:
        pt = pl.loose_point(rng, r)
        if pt is None:
            return None
        pts = [pt[:2]] + _walk(pt[0], pt[1], rng.uniform(0, math.tau), L, step,
                               item["curvature_radius_px"], rng, pl)
        z_default = pt[2]
    if len(pts) < 2:
        return None
    curve = bpy.data.curves.new(f"contaminant_{item['kind']}_{i}", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = r
    curve.bevel_resolution = 2 if r < 2 else 4
    curve.use_fill_caps = True
    spline = curve.splines.new("POLY")
    spline.points.add(len(pts) - 1)
    for sp, (x, y) in zip(spline.points, pts):
        z = pl.surface_z(x, y)
        wx, wy = _px_to_world(pl.p, x, y)
        sp.co = (wx, wy, (z if z is not None else z_default) + r, 1)
    obj = bpy.data.objects.new(curve.name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(_plain_material(obj.name, item["albedo"], item["roughness"]))
    return obj, {"points": len(pts), "start_px": [round(pts[0][0], 3), round(pts[0][1], 3)],
                 "end_px": [round(pts[-1][0], 3), round(pts[-1][1], 3)]}


def _burr(pl, item, rng, i, material):
    e = pl.edge_point(rng)
    if e is None:
        return None
    ex, ey, nx, ny, z_edge = e
    w0, L, t = item["base_width_px"], item["length_px"], item["thickness_px"]
    inward = 1.0 + 0.3 * w0            # overlap into the part so the burr is attached
    heading = math.atan2(ny, nx) + item["angle_rad"]
    stations = 7
    total = L + inward
    x, y = ex - math.cos(heading) * inward, ey - math.sin(heading) * inward
    z_top = (z_edge if z_edge is not None else pl.zmin)
    verts, rings = [], []
    for k in range(stations):
        s = k / (stations - 1)
        width = w0 * (1 - 0.85 * max(0.0, (s * total - inward) / L))
        h = heading + item["curl"] * s * s * (1 if item["seed"] & 1 else -1)
        px, py = -math.sin(h), math.cos(h)
        ring = []
        for side in (-1, 1):
            vx, vy = _px_to_world(pl.p, x + px * side * width / 2, y + py * side * width / 2)
            for zz in (z_top, z_top - t):
                ring.append(len(verts))
                verts.append((vx, vy, zz))
        rings.append(ring)
        x, y = x + math.cos(h) * total / (stations - 1), y + math.sin(h) * total / (stations - 1)
    faces = [tuple(rings[0][j] for j in (0, 1, 3, 2)), tuple(rings[-1][j] for j in (0, 2, 3, 1))]
    for a, b in zip(rings[:-1], rings[1:]):
        for j0, j1 in ((0, 2), (2, 3), (3, 1), (1, 0)):
            faces.append((a[j0], a[j1], b[j1], b[j0]))
    obj = _mesh_object(f"contaminant_burr_{i}", verts, faces, material)
    return obj, {"base_px": [round(ex, 3), round(ey, 3)], "heading_rad": round(heading, 5)}


def place_contaminants(scene, camera, manifest, targets, part_material, zmin, zmax, scratch):
    items = manifest["contamination"]["items"]
    realized = []
    objs = []
    if not items:
        manifest["contamination"]["realized"] = realized
        return objs
    p = manifest["projection"]
    w, h = p["source_width"], p["source_height"]
    saved = (scene.render.engine, scene.render.filter_size, scene.world.color[:],
             scene.render.resolution_x, scene.render.resolution_y)
    _workbench(scene)
    _only_visible(targets)
    sil = render_binary(scene, camera, w, h, p["ortho_scale_world"], 0, 0, w,
                        os.path.join(scratch, ".silhouette.png"))
    camera.location = tuple(p["camera_position_world"])
    camera.data.ortho_scale = p["ortho_scale_world"]
    camera.data.dof.use_dof = True
    for o in bpy.data.objects:
        o.hide_render = False
    (scene.render.engine, scene.render.filter_size, scene.world.color,
     scene.render.resolution_x, scene.render.resolution_y) = saved
    pl = _Placer(scene, manifest, sil, zmin, p["camera_position_world"][2] - 1, targets)
    if not len(pl.bx):
        manifest["contamination"]["realized"] = realized
        return objs
    for i, item in enumerate(items):
        rng = random.Random(item["seed"])
        np_rng = np.random.default_rng(item["seed"])
        if item["kind"] == "particle":
            made = _particle(pl, item, rng, np_rng, i)
        elif item["kind"] in ("fiber", "hair"):
            made = _fiber(pl, item, rng, i)
        else:
            made = _burr(pl, item, rng, i, part_material)
        if made is None:
            realized.append({"index": i, "placed": False})
            continue
        obj, info = made
        objs.append(obj)
        realized.append({"index": i, "placed": True, "object": obj.name, **info})
    # Anything that ended up inside the framing margin is removed, not clipped.
    for obj, rec in list(zip(objs, [r for r in realized if r["placed"]])):
        v = world_vertices([obj])
        xs, ys = v[:, 0] + w / 2, h / 2 - v[:, 1]
        if xs.min() < MARGIN_PX + 1 or ys.min() < MARGIN_PX + 1 or \
                xs.max() > w - MARGIN_PX - 1 or ys.max() > h - MARGIN_PX - 1:
            bpy.data.objects.remove(obj)
            objs.remove(obj)
            rec["placed"] = False
            rec["dropped"] = "frame_margin"
    manifest["contamination"]["realized"] = realized
    return objs


# ------------------------------------------------------------------------------ scene

def setup_scene(manifest, scratch):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    p = manifest["projection"]
    w, h = p["source_width"], p["source_height"]
    r = manifest["render"]
    scene.render.engine = "CYCLES"
    device = use_gpu(scene)
    scene.cycles.samples = r["cycles_samples"]
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = r["adaptive_threshold"]
    scene.cycles.adaptive_min_samples = r["adaptive_min_samples"]
    scene.cycles.use_denoising = r["denoise"]
    scene.cycles.pixel_filter_type = "BOX" if r["pixel_filter"] == "box" else "BLACKMAN_HARRIS"
    scene.render.filter_size = r["pixel_filter_width"]
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Raw"
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = manifest["illumination"]["ambient_strength"]

    camera_data = bpy.data.cameras.new("reference_ortho")
    camera = bpy.data.objects.new("reference_ortho", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = tuple(p["camera_position_world"])
    camera.rotation_euler = tuple(p["camera_rotation_rad"])
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = p["ortho_scale_world"]
    camera_data.clip_end = p["camera_position_world"][2] * 4
    scene.camera = camera

    material = _material_v3(manifest["material"])
    targets = geometry.build(manifest["geometry"])
    for obj in targets:
        obj.data.materials.append(material)
    pose = bpy.data.objects["target_pose"]

    # Tilt out of the image plane about an in-plane axis through the part center.
    g = manifest["geometry"]
    tilt = manifest["pose"]["tilt_rad"]
    axis = Vector((math.cos(manifest["pose"]["tilt_axis_rad"]),
                   math.sin(manifest["pose"]["tilt_axis_rad"]), 0))
    base = Matrix.Rotation(g["pose_angle_rad"], 4, "Z")
    reductions = 0
    while True:
        pose.matrix_world = Matrix.Rotation(tilt, 4, axis) @ base
        v = world_vertices(targets)
        xs, ys = v[:, 0] + w / 2, h / 2 - v[:, 1]
        if (xs.min() >= MARGIN_PX + 1 and ys.min() >= MARGIN_PX + 1 and
                xs.max() <= w - MARGIN_PX - 1 and ys.max() <= h - MARGIN_PX - 1) or tilt == 0:
            break
        reductions += 1
        tilt = 0.0 if reductions >= 12 else tilt * 0.8
    manifest["pose"]["realized_tilt_rad"] = round(tilt, 6)
    manifest["pose"]["tilt_reductions"] = reductions
    zmin, zmax = float(v[:, 2].min()), float(v[:, 2].max())

    # Focus: a reference plane inside the part's depth, then a sampled defocus.
    o = manifest["optics"]
    # Operators focus on the silhouette: the part's mid-depth (the axis/equator of turned
    # parts and spheres) with a focus error of at most +-20 px; depth blur elsewhere is natural.
    focus_window = min(zmax - zmin, 40.0)
    z_ref = (zmin + zmax) / 2 + (o["focus_depth_fraction"] - 0.5) * focus_window
    dz = o["defocus_sign"] * o["target_coc_px"] / (2 * o["na"])
    z_focus = z_ref + dz
    focus_distance = p["camera_position_world"][2] - z_focus
    fstop = 1 / (2 * o["na"] * focus_distance)
    camera_data.dof.use_dof = True
    camera_data.dof.focus_distance = focus_distance
    camera_data.dof.aperture_fstop = fstop
    camera_data.dof.aperture_blades = 0
    depth_coc = 2 * o["na"] * max(zmax - z_ref, z_ref - zmin)
    focus_class = {"mild": "moderate", "medium": "defocused", "heavy": "strong_defocus"}.get(
        o["blur_class"], "sharp" if depth_coc < 1.5 else "depth_split")
    manifest["camera"].update({"focus_class": focus_class, "focus_distance_world": round(focus_distance, 5),
                               "fstop": fstop, "aperture_blades": 0})
    o["realized"] = {"z_range_px": [round(zmin, 4), round(zmax, 4)], "z_focus": round(z_focus, 4),
                     "depth_coc_px": round(depth_coc, 4)}

    # Backlight panel behind the part, large enough for every aperture ray.
    b = manifest["backlight"]
    backlight_z = zmin - b["distance_px"]
    reach = o["na"] * (focus_distance + b["distance_px"]) * 1.5 + 200
    half_width = w / 2 + reach
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, backlight_z))
    backlight = bpy.context.object
    backlight.name = "visible_backlight"
    backlight.dimensions = (2 * half_width, h + 2 * reach, 0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    backlight.data.materials.append(_backlight_material(b, w / 2))
    if hasattr(backlight, "visible_shadow"):
        backlight.visible_shadow = False

    spill = manifest["illumination"]
    area_data = bpy.data.lights.new("front_spill", "AREA")
    area = bpy.data.objects.new("front_spill", area_data)
    bpy.context.collection.objects.link(area)
    a = spill["spill_angle_rad"]
    radius = h * spill["spill_radius_fraction"]
    area.location = (radius * math.cos(a), radius * math.sin(a),
                     p["camera_position_world"][2] * spill["spill_height_fraction"])
    area.rotation_euler = (-area.location).to_track_quat("-Z", "Y").to_euler()
    area_data.shape = "DISK"
    area_data.size = h * spill["spill_size_fraction"]
    area_data.energy = spill["spill_strength"] * h * h * 0.02

    contaminants = place_contaminants(scene, camera, manifest, targets, material, zmin, zmax, scratch)
    manifest["render"]["device"] = device
    return scene, camera, targets, contaminants


# ---------------------------------------------------------------------- post optics

def apply_optics(linear, optics):
    """Lens PSF core, veiling glare and vignetting on the linear image."""
    psf = optics["lens_psf"]
    sigma = math.hypot(psf["diffraction_sigma_px"], psf["aberration_sigma_px"])
    glare = optics["glare"]
    h, w = linear.shape
    if sigma > 0 or glare["fraction"] > 0:
        pad = 512
        img = np.pad(linear.astype(np.float64), pad, mode="reflect")
        fy = np.fft.fftfreq(img.shape[0])[:, None]
        fx = np.fft.rfftfreq(img.shape[1])[None, :]
        c, s = math.cos(psf["angle_rad"]), math.sin(psf["angle_rad"])
        u, v = fx * c + fy * s, -fx * s + fy * c
        sa, sb = sigma * math.sqrt(psf["aspect"]), sigma / math.sqrt(psf["aspect"])
        transfer = np.exp(-2 * math.pi ** 2 * (sa * sa * u * u + sb * sb * v * v))
        if glare["fraction"] > 0:
            # 2-D FT of the kernel (1 + r^2/s^2)^(-3/2) (normalized) is exp(-2 pi s |f|).
            rho = np.hypot(fx, fy)
            transfer = transfer * ((1 - glare["fraction"]) +
                                   glare["fraction"] * np.exp(-2 * math.pi * glare["scale_px"] * rho))
        img = np.fft.irfft2(np.fft.rfft2(img) * transfer, s=img.shape)
        linear = np.maximum(0, img[pad:pad + h, pad:pad + w])
    vig = optics["vignetting"]
    if vig["strength"] > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2 + vig["center_px"][0], h / 2 + vig["center_px"][1]
        r2 = ((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2) / (0.5 * math.hypot(w, h)) ** 2
        linear = linear * (1 - vig["strength"] * r2)
    return linear


# ---------------------------------------------------------------------------- masks

def _bounds_hi(objs, p):
    """Per-object mask-pixel bounds (x0, y0, x1, y1), padded conservatively."""
    s = p["mask_scale"]
    w, h = p["source_width"], p["source_height"]
    out = []
    for o in objs:
        v = world_vertices([o])
        if not len(v):
            continue
        xs, ys = (v[:, 0] + w / 2) * s, (h / 2 - v[:, 1]) * s
        pad = MASK_EDGE_PAD_PX * s
        out.append((xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad))
    return out


def _intersects(bounds, x, y, tw, th):
    return any(x0 < x + tw and x1 > x and y0 < y + th and y1 > y for x0, y0, x1, y1 in bounds)


def render_masks(manifest, output_dir, scene, camera, targets, contaminants, contaminants_only,
                 scratch=None):
    """Write 1-bit PNG mask tiles and the index.

    contaminants_only: render only the contaminant objects, only where they are, into
    mask_extra/ (the host ORs them into a reused base mask). Otherwise render the full
    mask; tiles outside all projected bounds are empty, and tiles whose 1x silhouette
    neighbourhood is uniformly object are full, without rendering.
    """
    p = manifest["projection"]
    s = p["mask_scale"]
    w, h = p["source_width"], p["source_height"]
    full_w, full_h, tile = w * s, h * s, manifest["render"]["mask_tile_size"]
    visible = contaminants if contaminants_only else targets + contaminants
    _workbench(scene)
    _only_visible(visible)
    bounds = _bounds_hi(visible, p)
    full_tiles = set()
    if not contaminants_only:
        sil = render_binary(scene, camera, w, h, p["ortho_scale_world"], 0, 0, w,
                            os.path.join(scratch or output_dir, ".sil1x.png"))
        step, pad = tile // s, MASK_EDGE_PAD_PX
        for ty in range(0, h, step):
            for tx in range(0, w, step):
                block = sil[max(0, ty - pad):ty + step + pad, max(0, tx - pad):tx + step + pad]
                if block.all():
                    full_tiles.add((tx * s, ty * s))
    subdir = manifest["files"]["mask_dir"] + ("_extra" if contaminants_only else "")
    mask_dir = os.path.join(output_dir, subdir)
    os.makedirs(mask_dir, exist_ok=True)
    tiles, foreground, rendered = [], 0, 0
    for y in range(0, full_h, tile):
        for x in range(0, full_w, tile):
            tw, th = min(tile, full_w - x), min(tile, full_h - y)
            needed = _intersects(bounds, x, y, tw, th)
            if contaminants_only and not needed:
                continue
            name = None                      # uniform tiles are not written (host shares them)
            if (x, y) in full_tiles:
                count = tw * th
            elif needed:
                binary = render_binary(scene, camera, full_w, full_h, p["ortho_scale_world"], x, y, tile,
                                       os.path.join(scratch or output_dir, ".mask.png"))[:th, :tw]
                rendered += 1
                count = int(binary.sum())
                if 0 < count < tw * th or contaminants_only:
                    name = f"x{x:05d}_y{y:05d}.png"
                    write_binary(os.path.join(mask_dir, name), binary)
            else:
                count = 0
            foreground += count
            tiles.append({"x": x, "y": y, "width": tw, "height": th, "file": name,
                          "foreground_pixels": count})
    camera.data.ortho_scale = p["ortho_scale_world"]
    camera.location = tuple(p["camera_position_world"])
    index = {"schema_version": 1, "width": full_w, "height": full_h, "tile_size": tile,
             "bit_order": "PNG grayscale 1-bit, 0 background, 1 object", "tiles": tiles,
             "foreground_pixels": foreground}
    import json
    with open(os.path.join(mask_dir, "index.json"), "w") as stream:
        json.dump(index, stream, indent=2)
    return {"foreground_pixels": foreground, "tile_count": len(tiles), "tiles_rendered": rendered,
            "tiles_full_uniform": len(full_tiles), "contaminants_only": contaminants_only}
