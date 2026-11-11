"""Parameterized mesh and curve generators. Coordinates are source-image pixels."""

import math

import bpy


def _mesh(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _smooth(obj, enabled=True):
    for poly in obj.data.polygons:
        poly.use_smooth = enabled


def _segments(radius, error=0.02, minimum=32, maximum=768):
    """Circle sagitta <= 0.02 source pixels, i.e. < 0.32 mask pixels."""
    if radius <= error:
        return minimum
    n = math.ceil(math.pi / math.acos(max(-1, 1 - error / radius)))
    return min(maximum, max(minimum, n))


def _curve_object(name, loops, thickness):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "2D"
    curve.fill_mode = "BOTH"
    curve.resolution_u = 1
    curve.extrude = thickness / 2
    for loop in loops:
        spline = curve.splines.new("POLY")
        spline.points.add(len(loop) - 1)
        for p, (x, y) in zip(spline.points, loop):
            p.co = (x, y, 0, 1)
        spline.use_cyclic_u = True
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    return obj


def _signed_area(points):
    return sum(points[i][0] * points[(i + 1) % len(points)][1] -
               points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points)))


def _angle_distance(a, b):
    return (a - b + math.pi) % math.tau - math.pi


def _local_edge_delta(radius, angle, features):
    delta = 0.0
    for feature in features:
        half = feature["span_px"] / 2
        distance = abs(_angle_distance(angle, feature["angle_rad"])) * radius
        if distance >= half:
            continue
        t = distance / half
        depth = feature["depth_px"]
        kind = feature["kind"]
        if kind in ("notch", "chip"):
            delta -= depth * ((1 - t) ** (2 if kind == "chip" else 1))
        elif kind == "square_notch":
            delta -= depth * min(1, (1 - t) * 8)
        elif kind in ("tooth", "hair"):
            delta += depth * (1 - t)
        elif kind == "square_tooth":
            delta += depth * min(1, (1 - t) * 8)
        elif kind == "wave":
            delta += depth * math.cos(math.pi * t) * (1 - t)
    return delta


def _outline(g):
    radius = g["radius_px"]
    kind = g["outline"]
    phase = g["outline_phase_rad"]
    n = _segments(radius, error=g.get("tessellation_error_px", 0.02))
    if g.get("edge_segments"):
        n = max(n, math.ceil(math.tau * radius / g.get("edge_sample_px", 0.75)))
    elif kind == "polygon":
        n = g["outline_sides"]
    elif kind == "fillet":
        n = max(72, n)
    else:
        n = max(n, g["outline_repeats"] * 12)
    result = []
    for i in range(n):
        a = math.tau * i / n + phase
        t = ((a / math.tau) * g["outline_repeats"]) % 1
        if kind == "teeth":
            delta = g["outline_amplitude_px"] * (1 - 2 * abs(t - 0.5))
        elif kind == "square_teeth":
            delta = g["outline_amplitude_px"] * (1 if t < 0.5 else -1)
        elif kind == "notched":
            delta = -g["outline_amplitude_px"] * (1 if t < 0.12 else 0)
        elif kind == "spline":
            delta = g["outline_amplitude_px"] * (0.6 * math.sin(3 * a) + 0.4 * math.sin(7 * a))
        elif kind == "fillet":
            delta = g["outline_amplitude_px"] * math.cos(g["outline_sides"] * a) ** 8
        else:
            delta = 0
        if kind == "polygon" and g.get("edge_segments"):
            sides = g["outline_sides"]
            sector = math.tau / sides
            q = ((a - phase + sector / 2) % sector) - sector / 2
            r = radius * math.cos(math.pi / sides) / math.cos(q)
        else:
            r = radius - g["outline_amplitude_px"] + delta
        r += _local_edge_delta(radius, a, g.get("edge_segments", ()))
        result.append((r * math.cos(a), r * math.sin(a)))
    return result


def _feature_loop(feature):
    kind = feature["kind"]
    x0, y0 = feature["center_px"]
    w, length = feature["width_px"], feature["length_px"]
    angle = feature["angle_rad"]
    points = []
    n = max(20, _segments(max(w / 2, 0.1), error=0.02, minimum=20))
    if kind in ("tooth", "freeform", "filleted_polygon"):
        # Coarse polygonal approximations of small radial profiles can confuse
        # Blender's 2-D curve fill and drop an entire plate region.
        n = max(n, 80)
    if kind in ("round", "ellipse", "polygon", "filleted_polygon", "freeform", "tooth"):
        if kind == "polygon":
            n = feature["sides"]
        for i in range(n):
            a = math.tau * i / n
            r = w / 2
            if kind == "freeform":
                r *= 1 + 0.16 * math.sin(3 * a + feature["phase_rad"])
                if w >= 9:
                    r *= 1 + 0.09 * math.sin(7 * a)
            elif kind == "tooth":
                r *= 1 + 0.17 * math.cos(10 * a)
            elif kind == "filleted_polygon":
                r *= 1 + 0.10 * math.cos(feature["sides"] * a) ** 6
            sx = r * math.cos(a) * (min(2, length / max(w, 0.001)) if kind == "ellipse" else 1)
            sy = r * math.sin(a)
            points.append((sx, sy))
    elif kind == "slot":
        half = max(0, (length - w) / 2)
        for i in range(n // 2 + 1):
            a = -math.pi / 2 + math.pi * i / (n // 2)
            points.append((half + w / 2 * math.cos(a), w / 2 * math.sin(a)))
        for i in range(n // 2 + 1):
            a = math.pi / 2 + math.pi * i / (n // 2)
            points.append((-half + w / 2 * math.cos(a), w / 2 * math.sin(a)))
    else:  # curved slot, a ribbon around a gently curved centerline
        steps = max(24, n)
        for sign in (1, -1):
            iterator = range(steps + 1) if sign == 1 else range(steps, -1, -1)
            for i in iterator:
                t = i / steps
                x = length * (t - 0.5)
                y = 0.16 * length * math.sin(math.pi * t) + sign * w / 2
                points.append((x, y))
    c, s = math.cos(angle), math.sin(angle)
    return [(x0 + x * c - y * s, y0 + x * s + y * c) for x, y in points]


def plate(g):
    outer = _outline(g)
    if _signed_area(outer) < 0:
        outer.reverse()
    loops = [outer]
    for f in g["features"]:
        loop = _feature_loop(f)
        if _signed_area(loop) > 0:
            loop.reverse()
        loops.append(loop)
    return _curve_object("target_plate", loops, g["thickness_px"])


def _thread_shape(kind, phase):
    t = phase % 1
    if kind == "v":
        return 1 - 2 * abs(t - 0.5)
    if kind == "round":
        return (1 - math.cos(math.tau * t)) / 2
    if kind == "trapezoid":
        return min(1, max(0, 4 * min(t, 1 - t)))
    if kind == "square":
        return 1 if 0.25 < t < 0.75 else 0
    return 0


def turned(g):
    length = g["length_px"]
    radius = g["base_radius_px"]
    pitch = g["thread_pitch_px"]
    threaded = g["thread_profile"] != "none"
    if "thread_regions" in g or "axial_features" in g:
        positions = {-length / 2, length / 2}

        def add_interval(left, right, increment):
            left = max(-length / 2, left)
            right = min(length / 2, right)
            if right <= left:
                return
            count = max(1, math.ceil((right - left) / increment))
            positions.update(round(left + (right - left) * i / count, 4) for i in range(count + 1))

        add_interval(-length / 2, length / 2, g.get("base_axial_step_px", 2.0))
        for boundary, transition in enumerate(g["transitions"], start=1):
            center = -length / 2 + length * boundary / len(g["steps"])
            half = transition["width_px"] / 2
            add_interval(center - half, center + half, min(0.5, transition["width_px"] / 12))
        for feature in g.get("axial_features", ()):
            half = feature["span_px"] / 2
            add_interval(feature["x_px"] - half, feature["x_px"] + half,
                         min(0.5, feature["span_px"] / 16))
        for region in g.get("thread_regions", ()):
            half = region["span_px"] / 2
            add_interval(region["x_px"] - half, region["x_px"] + half,
                         min(0.5, region["pitch_px"] / 12))
        xs = sorted(positions)
    else:
        nx = max(64, math.ceil(length / (min(0.5, pitch / 12) if threaded else 0.5)))
        xs = [-length / 2 + length * ix / nx for ix in range(nx + 1)]
    na = _segments(radius * max(g["steps"]), error=g.get("tessellation_error_px", 0.02), minimum=96)
    verts, faces = [], []
    for x in xs:
        u = (x + length / 2) / length
        step = min(len(g["steps"]) - 1, int(u * len(g["steps"])))
        base_factor = g["steps"][step]
        for boundary, transition in enumerate(g["transitions"], start=1):
            center = -length / 2 + length * boundary / len(g["steps"])
            half_width = transition["width_px"] / 2
            if center - half_width <= x <= center + half_width:
                t = (x - center + half_width) / transition["width_px"]
                if transition["kind"] == "sharp":
                    t = float(t >= 0.5)
                elif transition["kind"] == "radius":
                    t = t * t * (3 - 2 * t)
                base_factor = g["steps"][boundary - 1] * (1 - t) + g["steps"][boundary] * t
                break
        base = radius * base_factor
        for feature in g.get("axial_features", ()):
            half = feature["span_px"] / 2
            t = abs(x - feature["x_px"]) / half
            if t >= 1:
                continue
            kind = feature["kind"]
            depth = feature["depth_px"]
            if kind == "groove":
                base -= depth * (1 - t)
            elif kind == "neck":
                base -= depth * min(1, (1 - t) * 5)
            elif kind == "ridge":
                base += depth * (1 - t)
            elif kind == "shoulder":
                base += depth * min(1, (1 - t) * 5)
            elif kind == "wave":
                base += depth * math.cos(math.pi * t) * (1 - t)
        for ia in range(na):
            a = math.tau * ia / na
            if "thread_regions" in g:
                thread_delta = 0
                for region in g["thread_regions"]:
                    half = region["span_px"] / 2
                    distance = abs(x - region["x_px"])
                    if distance < half:
                        envelope = min(1, (half - distance) / region["pitch_px"])
                        phase = x / region["pitch_px"] - a / math.tau + region["phase_rad"] / math.tau
                        thread_delta += region["depth_px"] * envelope * _thread_shape(region["profile"], phase)
                r = base - thread_delta
            else:
                profile = _thread_shape(g["thread_profile"], x / pitch - a / math.tau + g["thread_phase_rad"] / math.tau)
                r = base - (g["thread_depth_px"] * profile if threaded and 0.18 < u < 0.8 else 0)
            verts.append((x, r * math.cos(a), r * math.sin(a)))
    for ix in range(len(xs) - 1):
        for ia in range(na):
            j = ix * na + ia
            k = ix * na + (ia + 1) % na
            faces.append((j, k, k + na, j + na))
    faces.extend([tuple(reversed(tuple(range(na)))), tuple((len(xs) - 1) * na + i for i in range(na))])
    obj = _mesh("target_turned", verts, faces)
    _smooth(obj)
    return obj


def _boolean(target, cutter):
    import bmesh

    def quality(mesh):
        bm = bmesh.new()
        bm.from_mesh(mesh)
        volume = abs(bm.calc_volume(signed=True))
        nonmanifold = sum(not edge.is_manifold for edge in bm.edges)
        bm.free()
        return volume, nonmanifold

    original = target.data.copy()
    initial_volume, initial_nonmanifold = quality(original)
    if initial_nonmanifold or initial_volume <= 0:
        raise ValueError("Boolean target is not a closed solid")
    for attempt, solver in enumerate(("MANIFOLD", "EXACT")):
        if attempt:
            failed = target.data
            target.data = original.copy()
            bpy.data.meshes.remove(failed)
        bpy.context.view_layer.objects.active = target
        modifier = target.modifiers.new("machined_cut", "BOOLEAN")
        modifier.operation = "DIFFERENCE"
        modifier.solver = solver
        modifier.object = cutter
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        volume, nonmanifold = quality(target.data)
        if nonmanifold == 0 and volume > initial_volume * 0.02:
            bpy.data.meshes.remove(original)
            bpy.data.objects.remove(cutter, do_unlink=True)
            return
    target.data = original
    raise RuntimeError("both Boolean solvers produced invalid target geometry")


def _cylinder(name, radius, depth, vertices=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices or _segments(radius), radius=radius, depth=depth)
    obj = bpy.context.object
    obj.name = name
    return obj


def machined(g):
    obj = turned(g)
    r = g["base_radius_px"] * max(g["steps"])
    for cut in g["cuts"]:
        kind, w, x = cut["kind"], cut["width_px"], cut["x_px"]
        if kind in ("cross_hole", "oblique_hole"):
            cutter = _cylinder("cutter_hole", w / 2, 4 * r)
            cutter.location = (x, 0, 0)
            cutter.rotation_euler[1] = cut["angle_rad"] if kind == "oblique_hole" else 0
        else:
            bpy.ops.mesh.primitive_cube_add(size=1)
            cutter = bpy.context.object
            cutter.name = "cutter_slot"
            if kind == "flat":
                cutter.dimensions = (cut.get("length_px", g["length_px"] * 0.55), 3 * r, 3 * r)
                cutter.location = (x, r * 1.8, 0)
            elif kind == "keyway":
                cutter.dimensions = (cut.get("length_px", g["length_px"] * 0.65), w, r * 0.7)
                cutter.location = (x, 0, r * 0.95)
            else:
                cutter.dimensions = (cut.get("length_px", w * 2.5), w, 4 * r)
                cutter.location = (x, 0, 0)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        _boolean(obj, cutter)
    return obj


def sphere(g):
    r = g["radius_px"]
    seg = _segments(r, error=g.get("tessellation_error_px", 0.025), minimum=96)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=max(48, seg // 2), radius=r)
    obj = bpy.context.object
    obj.name = "target_sphere"
    if g.get("rim_features"):
        for vertex in obj.data.vertices:
            x, y, z = vertex.co
            angle = math.atan2(y, x)
            taper = max(0, 1 - abs(z) / r) ** 4
            radial = math.hypot(x, y)
            if radial > 0:
                scale = 1 + taper * _local_edge_delta(r, angle, g["rim_features"]) / radial
                vertex.co.x *= scale
                vertex.co.y *= scale
    _smooth(obj)
    for cut in g["cuts"]:
        kind, w, offset = cut["kind"], cut["width_px"], cut["offset_px"]
        if kind in ("bore", "off_axis_bore"):
            cutter = _cylinder("cutter_bore", w / 2, 3 * r)
            cutter.location = ((offset if kind == "off_axis_bore" else 0), 0, 0)
        elif kind == "dimple":
            bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=w)
            cutter = bpy.context.object
            cutter.location = (offset, 0, r + w * 0.5)
        elif kind == "rim_notch":
            bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=w / 2)
            cutter = bpy.context.object
            a = cut["angle_rad"]
            cutter.location = ((r + w / 2 - cut["depth_px"]) * math.cos(a),
                               (r + w / 2 - cut["depth_px"]) * math.sin(a), 0)
        else:
            bpy.ops.mesh.primitive_cube_add(size=1)
            cutter = bpy.context.object
            cutter.dimensions = (3 * r, 3 * r, r)
            cutter.location = (0, 0, r * 1.25)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        _boolean(obj, cutter)
    if g["partial"]:
        bpy.ops.mesh.primitive_cube_add(size=1)
        cutter = bpy.context.object
        cutter.dimensions = (3 * r, 3 * r, 2 * r)
        cutter.location = (0, -1.7 * r, 0)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        _boolean(obj, cutter)
    return obj


def freeform(g):
    r = g["radius_px"]
    nu = max(96, _segments(r, error=g.get("tessellation_error_px", 0.02), minimum=96))
    if g.get("rim_features"):
        nu = max(nu, math.ceil(math.tau * r / g.get("rim_sample_px", 2.5)))
    nv = max(48, min(128, nu // 10 if g.get("rim_features") else nu // 2))
    verts, faces = [(0, 0, 0.38 * r)], []
    for j in range(1, nv):
        v = math.pi * j / nv
        for i in range(nu):
            a = math.tau * i / nu
            radial = r * (1 + 0.16 * math.sin(g["lobes"] * a) * math.sin(v) ** 2)
            radial += _local_edge_delta(r, a, g.get("rim_features", ())) * math.sin(v) ** 4
            x = radial * math.sin(v) * math.cos(a)
            y = radial * math.sin(v) * math.sin(a)
            z = 0.38 * r * math.cos(v) + g["ridge_amplitude_px"] * math.sin(3 * a) * math.sin(v) ** 2
            z += g["saddle_amplitude_px"] * (x * x - y * y) / (r * r)
            verts.append((x, y, z))
    bottom = len(verts)
    verts.append((0, 0, -0.38 * r))
    for i in range(nu):
        faces.append((0, 1 + i, 1 + (i + 1) % nu))
    for j in range(nv - 2):
        for i in range(nu):
            a = 1 + j * nu + i
            b = 1 + j * nu + (i + 1) % nu
            faces.append((a, a + nu, b + nu, b))
    last = 1 + (nv - 2) * nu
    for i in range(nu):
        faces.append((last + i, bottom, last + (i + 1) % nu))
    obj = _mesh("target_freeform", verts, faces)
    _smooth(obj, not g["faceted"])
    return obj


def add_burrs(g):
    result = []
    r = g["radius_px"] if g["family"] in ("plate", "sphere", "freeform") else g["length_px"] / 2
    for b in g["burrs"]:
        a = b["angle_rad"]
        if g["family"] in ("plate", "sphere", "freeform"):
            start = (r * 0.96 * math.cos(a), r * 0.96 * math.sin(a), 0)
            end = ((r + b["length_px"]) * math.cos(a), (r + b["length_px"]) * math.sin(a), 0)
        else:
            start = (r * (1 if math.cos(a) >= 0 else -1), g["base_radius_px"] * math.sin(a), 0)
            end = (start[0] + b["length_px"] * (1 if start[0] > 0 else -1), start[1], 0)
        direction = tuple(end[i] - start[i] for i in range(3))
        length = math.sqrt(sum(v * v for v in direction))
        if length < 0.01:
            continue
        obj = _cylinder("target_burr", b["width_px"] / 2, length, vertices=8)
        obj.location = tuple((start[i] + end[i]) / 2 for i in range(3))
        import mathutils
        obj.rotation_euler = mathutils.Vector(direction).to_track_quat("Z", "Y").to_euler()
        result.append(obj)
    return result


def build(g):
    obj = {"plate": plate, "turned": turned, "machined": machined,
           "sphere": sphere, "freeform": freeform}[g["family"]](g)
    objects = [obj] + add_burrs(g)
    pose = bpy.data.objects.new("target_pose", None)
    bpy.context.collection.objects.link(pose)
    for item in objects:
        item.parent = pose
    pose.rotation_euler.z = g["pose_angle_rad"]
    return objects
