"""Deterministic macro scene sampling (host side; needs numpy + shapely + synthetic-shapes).

A scene is 1-10 parts lying on a backlit stage, seen by an orthographic camera at 1/4 of
the source resolution (512 px). Parts come from the synthetic-shapes planar grammars,
made 3-D: face-view grammars are extruded into plates, turned side-view grammars are
revolved about their axis. Machined turned parts, spheres, freeform solids and plain
primitives add 3-D and simple shapes. Appearance reuses the physical_v3 model of
synthetic-renders with every pixel-unit quantity divided by 4.

All in-scene coordinates here are world units = macro pixels, x right, y up, origin at
the image centre. Image coordinates (x right, y down) appear only in debris boundary
samples, which the Blender side consumes through synthetic-renders' placer helpers.
"""

import hashlib
import math
import random

import numpy as np
from shapely import affinity
from shapely.geometry import LinearRing, Point, Polygon, box
from shapely.ops import unary_union

from . import FULL_SCALE, PROFILE, SIZE
from inspection_shapes import geometry as sg
from inspection_shapes.expanded import NEW_FAMILIES, expanded_nominal
from synth.appearance_v3 import BLUR_CLASSES, LIGHT_WEIGHTS, _backlight, _lighting, _logu, _u, _weighted
from synth.sampling import _seed

BASE_FACE = ("washer", "flange", "drilled_plate", "slotted_bracket", "polygon_spacer",
             "toothed_disk", "cam", "rack")
BASE_TURNED = ("stepped_shaft", "threaded_shaft", "grooved_pin", "tapered_pin")
EXP_TURNED = ("multi_step_shaft", "groove_stack_pin", "multi_thread_shaft", "stepped_taper_arbor")
EXP_FACE = tuple(f for f in NEW_FAMILIES if f not in EXP_TURNED)
PLAIN_FACE = ("plain_disc", "plain_ring", "plain_rect", "plain_polygon", "plain_triangle")
PLAIN_TURNED = ("plain_rod", "plain_pin")
# Simple grammars dominate: the coarse model mostly sees parts with few feature types.
GROUPS = ((0.26, "face_simple"), (0.16, "face_expanded"), (0.16, "turned_simple"),
          (0.06, "turned_expanded"), (0.12, "plain"), (0.08, "machined"),
          (0.07, "sphere"), (0.09, "freeform"))
PART_COUNTS = (0.15, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07)   # 1..10
LAYOUTS = ((0.70, "separated"), (0.20, "touching"), (0.10, "stacked"))
MATERIALS = {   # (roughness, albedo) ranges, as synthetic-renders sampling.py
    "dark_matte": ((0.65, 0.95), (0.012, 0.09)), "rough_metal": ((0.38, 0.85), (0.12, 0.62)),
    "polished_metal": ((0.08, 0.34), (0.22, 0.78)), "coated": ((0.18, 0.62), (0.08, 0.55)),
    "polymer": ((0.32, 0.82), (0.03, 0.28)), "light_matte": ((0.55, 0.92), (0.34, 0.82)),
}
WIRE_COUNTS = (0.55, 0.30, 0.15)     # 1..3 wires in a wire scene
WIRE_DIAMETER_PX = (0.25, 4.0)       # macro px, log-uniform (1-16 source px)
WIRE_LENGTH_FRAC = (0.3, 0.95)       # of the frame side; loose hairs are <= 200 px and curved
TILT_FRACTION = 0.25
CUT_FRACTION = 0.15
LOOSE_DEBRIS_FRACTION = 0.30
ATTACHED_DEBRIS_FRACTION = 0.08
EDGE_PAD = 3.0          # px kept clear of the frame unless a part is deliberately cut


class _DefectBudget:
    """Per-part Poisson defect budget (as contour-model corpus_v2), not per ring."""

    def __init__(self, rng, rings, count):
        self.rng, self.i = rng, 0
        self.hits = np.bincount(rng.integers(rings, size=count), minlength=rings)

    def poisson(self, *args, **kwargs):
        n = int(self.hits[self.i])
        self.i += 1
        return n

    def __getattr__(self, name):
        return getattr(self.rng, name)


def _plain(family, rng):
    """Plain primitives as (rings_mm, holes, kind)."""
    def circle(r, n=720):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.c_[r * np.cos(t), r * np.sin(t)]
    if family == "plain_disc":
        return [circle(rng.uniform(6, 30))], [False], "extrude"
    if family == "plain_ring":
        r = rng.uniform(8, 30)
        return [circle(r), circle(r * rng.uniform(0.3, 0.8))[::-1]], [False, True], "extrude"
    if family == "plain_rect":
        w, h = rng.uniform(10, 70), rng.uniform(5, 40)
        g = box(-w / 2, -h / 2, w / 2, h / 2)
        if rng.random() < 0.6:
            rr = min(w, h) * rng.uniform(0.02, 0.25)
            g = g.buffer(-rr).buffer(rr, quad_segs=24)
        return [np.asarray(g.exterior.coords)[:-1]], [False], "extrude"
    if family in ("plain_polygon", "plain_triangle"):
        n = 3 if family == "plain_triangle" else int(rng.choice([4, 5, 6, 8]))
        r = rng.uniform(8, 30)
        t = np.arange(n) * 2 * np.pi / n + rng.uniform(0, 2 * np.pi)
        return [np.c_[r * np.cos(t), r * np.sin(t)]], [False], "extrude"
    # Turned: side profile of a plain rod or pin with optional end chamfers.
    L, r = rng.uniform(20, 80), rng.uniform(1.5, 10)
    c = min(r * 0.4, rng.uniform(0, 1.2)) if family == "plain_rod" else r * rng.uniform(0.3, 0.9)
    top = [[-L / 2, r - c], [-L / 2 + c, r], [L / 2 - c, r], [L / 2, r - c]]
    ring = np.array(top + [[x, -y] for x, y in top[::-1]])[::-1]
    return [ring], [False], "revolve"


def _grammar(family, rng):
    """synthetic-shapes grammar with mild form errors -> (rings_mm, holes)."""
    severity = float(rng.choice([0., .03, .08, .16, .30], p=[.05, .20, .40, .25, .10]))
    for _ in range(50):
        p = expanded_nominal(family, rng) if family in NEW_FAMILIES else sg.make_nominal(family, rng)
        if not sg.valid(p, False):
            continue
        extent = max(np.ptp(np.concatenate([r.nominal for r in p.rings]), axis=0))
        effective = severity * min(1., 80 / extent)
        defects = int(rng.poisson(.35)) if effective else 0
        sg.imperfect(p, _DefectBudget(rng, len(p.rings), defects), effective)
        if sg.valid(p):
            return [np.asarray(r.actual, float) for r in p.rings], [bool(r.hole) for r in p.rings]
    raise RuntimeError(f"could not generate {family}")


def _half_profile(outer):
    """Upper half (x increasing, r >= 0) of a turned side-view ring symmetric about y = 0."""
    up = outer[:, 1] > 1e-9
    n = len(outer)
    # longest cyclic run of upper points
    best, start = (0, 0), None
    for k in range(2 * n):
        if up[k % n]:
            start = k if start is None else start
            if k - start + 1 > best[1] - best[0]:
                best = (start, k + 1)
        else:
            start = None
    idx = [i % n for i in range(best[0], min(best[1], best[0] + n))]
    top = outer[idx]
    if top[0, 0] > top[-1, 0]:
        top = top[::-1]
    return top


def _simplify(ring, tol):
    q = np.asarray(LinearRing(ring).simplify(tol).coords)[:-1]
    return q if len(q) >= 3 else ring


def _sample_part(rng, a):
    _, group = _weighted(a, GROUPS)
    part = {"group": group}
    if group in ("face_simple", "face_expanded", "turned_simple", "turned_expanded", "plain", "machined"):
        family = {"face_simple": BASE_FACE, "face_expanded": EXP_FACE, "turned_simple": BASE_TURNED,
                  "turned_expanded": EXP_TURNED, "machined": BASE_TURNED + PLAIN_TURNED,
                  "plain": PLAIN_FACE + PLAIN_TURNED}[group]
        family = family[int(rng.integers(len(family)))]
        if family.startswith("plain_"):
            rings, holes, kind = _plain(family, rng)
        else:
            rings, holes = _grammar(family, rng)
            kind = "revolve" if family in BASE_TURNED + EXP_TURNED else "extrude"
        part.update(family=family, kind=kind, rings_mm=rings, holes=holes)
        outer = rings[0]
        footprint = Polygon(outer, [r for r, h in zip(rings[1:], holes[1:]) if h])
        if not footprint.is_valid:
            footprint = footprint.buffer(0)
        part["footprint_mm"] = footprint
        if kind == "extrude":
            ext = max(np.ptp(outer, axis=0))
            part["thickness_mm"] = float(min(ext * 0.35, _logu(a, 0.6, 10)))
        if group == "machined":
            top = _half_profile(outer)
            cuts = []
            for _ in range(int(rng.integers(1, 3))):
                x = float(rng.uniform(top[0, 0] * 0.7, top[-1, 0] * 0.7))
                r_here = float(np.interp(x, top[:, 0], top[:, 1]))
                kind_c = str(rng.choice(["cross_hole", "cross_hole", "through_slot", "flat"]))
                cuts.append({"kind": kind_c, "x_mm": x, "width_mm": float(r_here * rng.uniform(0.35, 1.1)),
                             "length_mm": float(r_here * rng.uniform(1.5, 4)),
                             "depth_frac": float(rng.uniform(0.15, 0.45))})
            part["cuts_mm"] = cuts
    elif group == "sphere":
        r = float(_logu(a, 2, 25))
        part.update(family="sphere", kind="sphere", radius_mm=r,
                    bore_frac=float(rng.uniform(0.15, 0.55)) if rng.random() < 0.3 else 0.0,
                    footprint_mm=Point(0, 0).buffer(r, quad_segs=32))
    else:
        r = float(_logu(a, 5, 35))
        part.update(family="freeform", kind="freeform", radius_mm=r, lobes=int(rng.integers(2, 7)),
                    ridge_frac=float(rng.uniform(0.02, 0.12)), saddle_frac=float(rng.uniform(-0.12, 0.12)),
                    faceted=bool(rng.random() < 0.25),
                    footprint_mm=Point(0, 0).buffer(r * 1.18, quad_segs=32))
    part["size_mult"] = float(_logu(a, 0.5, 1.6))
    return part


def _wire(rng, a, s):
    """A thin straight wire or pin lying on its side, sized in macro px (then converted to
    mm with the scene scale s px/mm): a plain rod, sometimes with a pointed tip or a small
    head. Returns a part dict like _sample_part (group "wire", kind "revolve")."""
    d = _logu(a, *WIRE_DIAMETER_PX)
    L = SIZE * _u(a, *WIRE_LENGTH_FRAC)
    r = d / 2
    style = _weighted(a, ((0.60, "rod"), (0.25, "point"), (0.15, "head")))[1]
    if style == "rod":
        top = [[-L / 2, r], [L / 2, r]]
    elif style == "point":
        taper = min(L * 0.3, d * _u(a, 2, 10))
        top = [[-L / 2, r], [L / 2 - taper, r], [L / 2, r * 0.05]]
    else:
        hr, hl = r * _u(a, 1.5, 3.0), min(L * 0.1, d * _u(a, 1, 4))
        top = [[-L / 2, hr], [-L / 2 + hl, hr], [-L / 2 + hl, r], [L / 2, r]]
    top = np.asarray(top, float)
    ring = np.concatenate([top, top[::-1] * [1, -1]])[::-1] / s
    return {"group": "wire", "family": f"wire_{style}", "kind": "revolve", "rings_mm": [ring],
            "holes": [False], "footprint_mm": Polygon(ring), "size_mult": 1.0,
            "wire_diameter_px": round(d, 5)}


def _add_wires(parts, placed, s, w):
    """Place 1-3 wires into the free space around the already placed parts (rejection
    sampling; a wire that does not fit is shortened, then dropped). Half of them may
    cross the frame edge like a long pin running out of view."""
    from shapely.prepared import prep
    frame = box(-SIZE / 2 + EDGE_PAD, -SIZE / 2 + EDGE_PAD, SIZE / 2 - EDGE_PAD, SIZE / 2 - EDGE_PAD)
    inner = prep(frame)
    n_w = 1 + int(np.searchsorted(np.cumsum(WIRE_COUNTS), w.random() * sum(WIRE_COUNTS)))
    for _ in range(min(n_w, len(WIRE_COUNTS))):
        occupied = prep(unary_union([r["shape"] for r in placed]).buffer(2.0))
        wire = _wire(None, w, s)
        cut = w.random() < 0.5
        for shrink in range(4):
            shape0 = affinity.scale(wire["footprint_mm"], s, s, origin=(0, 0))
            found = None
            for attempt in range(300):
                rot = w.uniform(0, math.tau)
                shape = affinity.rotate(shape0, rot, origin=(0, 0), use_radians=True)
                x, y = w.uniform(-SIZE / 2, SIZE / 2), w.uniform(-SIZE / 2, SIZE / 2)
                cand = affinity.translate(shape, x, y)
                if occupied.intersects(cand):
                    continue
                if not cut:
                    if inner.contains(cand):
                        found = (cand, x, y, rot)
                        break
                    continue
                inside = cand.intersection(frame).area / max(cand.area, 1e-9)
                if 0.3 < inside < 0.95:
                    found = (cand, x, y, rot)
                    break
            if found:
                break
            wire = _shorten(wire, 0.7)
        if not found:
            continue
        i = len(parts)
        parts.append(wire)
        placed.append({"i": i, "shape": found[0], "x": found[1], "y": found[2], "rot": found[3],
                       "stack_on": None, "cut": cut})


def _shorten(wire, f):
    ring = wire["rings_mm"][0] * [f, 1]
    return {**wire, "rings_mm": [ring], "footprint_mm": Polygon(ring)}


def _material(a):
    fam = random.Random(a.random()).choice(list(MATERIALS))
    rough, albedo = MATERIALS[fam]
    return {"family": fam, "roughness": _u(a, *rough), "albedo": _u(a, *albedo),
            "metallic": 1.0 if "metal" in fam else 0.0,
            "coat_weight": _u(a, 0.1, 0.4) if fam == "coated" else 0.0,
            "bump": ({"strength": _logu(a, 0.02, 0.4), "scale_px": _logu(a, 0.5, 15),
                      "anisotropy": _u(a, 1, 12)} if a.random() < 0.5 else None)}


def _optics(a, macro_pixel_um):
    """physical_v3 optics for a macro pixel of the given size; px values are macro px."""
    pixel_um = macro_pixel_um / FULL_SCALE             # full-resolution sensor pixel
    wavelength = _u(a, 0.45, 0.65)
    na_min = max(0.004, 0.21 * wavelength / (1.2 * pixel_um))
    na = _logu(a, na_min, max(0.06, na_min * 1.5))
    _, blur_class, coc_range = _weighted(a, BLUR_CLASSES)
    coc_full = _u(a, *coc_range) if coc_range[1] > 0 else 0.0
    return {"na": na, "pixel_size_um": round(macro_pixel_um, 5), "full_pixel_size_um": round(pixel_um, 5),
            "wavelength_um": wavelength, "blur_class": blur_class,
            "target_coc_px": round(coc_full / FULL_SCALE, 6), "defocus_sign": a.choice((-1, 1)),
            "focus_depth_fraction": _u(a, 0.0, 1.0),
            "lens_psf": {"diffraction_sigma_px": round(0.21 * wavelength / na / pixel_um / FULL_SCALE, 6),
                         "aberration_sigma_px": (_u(a, 0.0, 0.6) if a.random() < 0.7 else 0.0) / FULL_SCALE,
                         "aspect": _u(a, 1.0, 1.35) if a.random() < 0.3 else 1.0,
                         "angle_rad": _u(a, 0, math.pi)},
            "glare": {"fraction": _logu(a, 0.003, 0.15) if a.random() < 0.75 else 0.0,
                      "scale_px": _logu(a, 15, 400) / FULL_SCALE},
            "vignetting": {"strength": _u(a, 0, 0.18) if a.random() < 0.5 else 0.0,
                           "center_px": [_u(a, -75, 75), _u(a, -75, 75)]}}


def _scale_backlight(b):
    k = 1 / FULL_SCALE
    b["distance_px"] *= k
    for key in ("noise", "grain", "smudges"):
        b[key]["scale_px"] *= k
    b["led_grid"]["pitch_px"] *= k
    for d in b["dust"]:
        d["center_px"] = [c * k for c in d["center_px"]]
        d["radius_px"] = max(0.3, d["radius_px"] * k)
    return b


def _debris_item(a, kind, pixel_um, attach):
    item = {"kind": kind, "attach": attach, "seed": a.getrandbits(48)}
    if kind == "particle":
        d = _logu(a, 20, 400)
        item.update(diameter_um=d, diameter_px=round(min(12, max(0.3, d / pixel_um)), 5),
                    albedo=_u(a, 0.03, 0.7), roughness=_u(a, 0.3, 1.0))
    elif kind in ("fiber", "hair"):
        d = _logu(a, 40, 120) if kind == "hair" else _logu(a, 8, 30)
        L = _logu(a, 2000, 15000) if kind == "hair" else _logu(a, 500, 8000)
        item.update(diameter_um=d, length_um=L, diameter_px=round(max(0.2, d / pixel_um), 5),
                    length_px=round(min(200, max(4, L / pixel_um)), 4),
                    curvature_radius_px=_logu(a, 80, 1500) if kind == "hair" else _logu(a, 20, 400),
                    outside_fraction=_u(a, 0.3, 0.85),
                    albedo=_u(a, 0.02, 0.12) if kind == "hair" else _u(a, 0.25, 0.85),
                    roughness=_u(a, 0.35, 0.95))
    else:
        L = _logu(a, 30, 600)
        item.update(length_um=L, length_px=round(min(10, max(0.3, L / pixel_um)), 5),
                    angle_rad=_u(a, -0.6, 0.6), curl=_u(a, 0, 0.8))
        item["base_width_px"] = round(max(0.25, item["length_px"] * _u(a, 0.3, 1.2)), 5)
        item["thickness_px"] = round(max(0.25, item["base_width_px"] * _u(a, 0.2, 0.8)), 5)
    return item


def _boundary_samples(poly, n=240):
    """(x, y, nx, ny) samples on a polygon's boundary, image coordinates (y down).

    The normal points away from the material (out of the part, or into a hole)."""
    out = []
    for ring in [poly.exterior] + list(poly.interiors):
        L = ring.length
        for t in np.linspace(0, L, max(8, int(n * L / max(poly.length, 1e-9))), endpoint=False):
            p, q = ring.interpolate(t), ring.interpolate((t + min(0.5, L / 1000)) % L)
            tx, ty = q.x - p.x, q.y - p.y
            norm = math.hypot(tx, ty) or 1.0
            nx, ny = ty / norm, -tx / norm
            if poly.contains(Point(p.x + 0.5 * nx, p.y + 0.5 * ny)):
                nx, ny = -nx, -ny
            out.append((p.x + SIZE / 2, SIZE / 2 - p.y, nx, -ny))
    return out


def _place(parts, order, s, layout, cut_parts, stack_part, a):
    """Rejection placement of rotated footprints; None if the scale is too large."""
    frame = box(-SIZE / 2 + EDGE_PAD, -SIZE / 2 + EDGE_PAD, SIZE / 2 - EDGE_PAD, SIZE / 2 - EDGE_PAD)
    placed = []
    for i in order:
        p = parts[i]
        k = s * p["size_mult"]
        base = affinity.scale(p["footprint_mm"], k, k, origin=(0, 0))
        gap = {"separated": _u(a, 3, 30), "touching": _u(a, 0, 1.5), "stacked": _u(a, 3, 20)}[layout]
        want_stack = i == stack_part and placed
        found = None
        for attempt in range(600):
            rot = a.uniform(0, math.tau)
            shape = affinity.rotate(base, rot, origin=(0, 0), use_radians=True)
            bx0, by0, bx1, by1 = shape.bounds
            near = placed and (want_stack and attempt < 400 or layout == "touching" and attempt < 400)
            if near:
                # propose around an existing part so contacts and stacks actually occur
                q = a.choice(placed)["shape"].bounds
                x = a.uniform(q[0] - bx1, q[2] - bx0)
                y = a.uniform(q[1] - by1, q[3] - by0)
            elif i in cut_parts:
                x, y = a.uniform(-SIZE / 2, SIZE / 2), a.uniform(-SIZE / 2, SIZE / 2)
            else:
                lo_x, hi_x = -SIZE / 2 + EDGE_PAD - bx0, SIZE / 2 - EDGE_PAD - bx1
                lo_y, hi_y = -SIZE / 2 + EDGE_PAD - by0, SIZE / 2 - EDGE_PAD - by1
                if lo_x > hi_x or lo_y > hi_y:
                    return None
                x, y = a.uniform(lo_x, hi_x), a.uniform(lo_y, hi_y)
            cand = affinity.translate(shape, x, y)
            inside = cand.intersection(frame).area / max(cand.area, 1e-9)
            if i in cut_parts:
                if not 0.3 < inside < 0.95:
                    continue
            elif inside < 0.999:
                continue
            if want_stack and attempt < 400:
                below = [r for r in placed if r["shape"].intersects(cand)]
                if len(below) == 1 and cand.intersection(below[0]["shape"]).area > 0.15 * cand.area:
                    found = (cand, x, y, rot, below[0]["i"])
                    break
                continue
            dists = [cand.distance(r["shape"]) for r in placed]
            if all(d > gap for d in dists) and (not near or min(dists) < gap + 3):
                found = (cand, x, y, rot, None)
                break
        if found is None:
            return None
        cand, x, y, rot, below = found
        placed.append({"i": i, "shape": cand, "x": x, "y": y, "rot": rot, "stack_on": below})
    return placed


def sample_macro_scene(master_seed, index, wire_fraction=0.0):
    """wire_fraction > 0 (a later batch; 0 keeps the original scenes bit-identical): with
    that probability the scene also gets 1-3 thin wires (see _wire)."""
    a = random.Random(_seed(master_seed, index, "macro-scene"))
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, index, 7717]))
    split_hash = int.from_bytes(hashlib.blake2b(f"{master_seed}:{index}:split".encode(),
                                                digest_size=8).digest(), "big") % 100
    split = "train" if split_hash < 80 else "val" if split_hash < 90 else "test"
    n = int(rng.choice(np.arange(1, 11), p=np.array(PART_COUNTS) / sum(PART_COUNTS)))
    parts = [_sample_part(rng, a) for _ in range(n)]
    _, layout = _weighted(a, LAYOUTS if n > 1 else LAYOUTS[:1])
    cut = a.random() < CUT_FRACTION
    fill = _u(a, 0.10, 0.55) if n == 1 else _u(a, 0.08, 0.45)
    area = sum(p["footprint_mm"].area * p["size_mult"] ** 2 for p in parts)
    s = math.sqrt(fill * SIZE * SIZE / area)             # px per mm
    ext = [max(np.ptp(np.asarray(p["footprint_mm"].exterior.coords), axis=0)) * p["size_mult"] for p in parts]
    s = min(s, 0.92 * SIZE / max(ext))
    # Tiny parts are grown rather than dropped: the smallest spans at least 10 px.
    for p, e in zip(parts, ext):
        if e * s < 10:
            p["size_mult"] *= 10 / (e * s)
    want_wires = bool(wire_fraction) and a.random() < wire_fraction
    order = sorted(range(n), key=lambda i: -parts[i]["footprint_mm"].area * parts[i]["size_mult"] ** 2)
    cut_parts = set(a.sample(range(n), min(n, a.randint(1, 2)))) if cut else set()
    stack_part = order[-1] if layout == "stacked" else None
    for attempt in range(15):
        placed = _place(parts, order, s, layout, cut_parts, stack_part, a)
        if placed is not None:
            break
        s *= 0.9
    else:
        raise RuntimeError(f"could not place scene {index}")
    if want_wires:
        _add_wires(parts, placed, s, random.Random(_seed(master_seed, index, "macro-wires")))
    macro_pixel_um = 1000 / s
    # geometry in px, part-local (centred), simplified to 0.02 px
    arrays, out_parts = {}, []
    mat_shared = _material(a) if a.random() < 0.6 else None
    for rec in sorted(placed, key=lambda r: r["i"]):
        p = parts[rec["i"]]
        k = s * p["size_mult"]
        entry = {"index": rec["i"], "group": p["group"], "family": p["family"], "kind": p["kind"],
                 **({"wire_diameter_px": p["wire_diameter_px"]} if p["group"] == "wire" else {}),
                 "pose": {"x": rec["x"], "y": rec["y"], "rot_rad": rec["rot"], "stack_on": rec["stack_on"],
                          "tilt_rad": (_u(a, math.radians(1), math.radians(20))
                                       if rec["stack_on"] is None and a.random() < TILT_FRACTION else 0.0),
                          "tilt_axis_rad": _u(a, 0, math.pi)},
                 "cut_by_frame": rec.get("cut", rec["i"] in cut_parts), "px_per_mm": k,
                 "material": mat_shared or _material(a)}
        if p["kind"] in ("extrude", "revolve"):
            rings = [_simplify(r * k, 0.02) for r in p["rings_mm"]]
            if p["kind"] == "extrude":
                entry["thickness_px"] = p["thickness_mm"] * k
                for j, r in enumerate(rings):
                    arrays[f"p{rec['i']}_ring{j}"] = r.astype(np.float64)
                entry["rings"] = [f"p{rec['i']}_ring{j}" for j in range(len(rings))]
                entry["holes"] = p["holes"]
            else:
                prof = _half_profile(rings[0])
                arrays[f"p{rec['i']}_profile"] = prof.astype(np.float64)
                entry["profile"] = f"p{rec['i']}_profile"
                entry["cuts"] = [{**c, "x_px": c["x_mm"] * k, "width_px": c["width_mm"] * k,
                                  "length_px": c["length_mm"] * k} for c in p.get("cuts_mm", [])]
        elif p["kind"] == "sphere":
            entry.update(radius_px=p["radius_mm"] * k, bore_frac=p["bore_frac"])
        else:
            r = p["radius_mm"] * k
            entry.update(radius_px=r, lobes=p["lobes"], ridge_amplitude_px=p["ridge_frac"] * r,
                         saddle_amplitude_px=p["saddle_frac"] * r, faceted=p["faceted"])
        entry["footprint_area_px"] = rec["shape"].area
        entry["debris"] = []
        if (entry["pose"]["tilt_rad"] == 0 and rec["stack_on"] is None and not entry["cut_by_frame"]
                and a.random() < ATTACHED_DEBRIS_FRACTION):
            for _ in range(a.randint(1, 2)):
                kind = _weighted(a, ((0.4, "particle"), (0.4, "burr"), (0.2, "fiber")))[1]
                entry["debris"].append(_debris_item(a, kind, macro_pixel_um, "edge"))
            samples = _boundary_samples(rec["shape"])
            arrays[f"p{rec['i']}_boundary"] = np.asarray(samples, np.float64)
            entry["boundary"] = f"p{rec['i']}_boundary"
        out_parts.append(entry)
    loose = []
    if a.random() < LOOSE_DEBRIS_FRACTION:
        occupied = unary_union([r["shape"] for r in placed]).buffer(4)
        for _ in range(a.randint(1, 3)):
            kind = _weighted(a, ((0.5, "particle"), (0.35, "fiber"), (0.15, "hair")))[1]
            item = _debris_item(a, kind, macro_pixel_um, "loose")
            for _ in range(100):
                x, y = a.uniform(-SIZE / 2 + 8, SIZE / 2 - 8), a.uniform(-SIZE / 2 + 8, SIZE / 2 - 8)
                if not occupied.contains(Point(x, y)):
                    item["position_px"] = [x + SIZE / 2, SIZE / 2 - y]      # image coordinates
                    loose.append(item)
                    break
    optics = _optics(a, macro_pixel_um)
    light = _lighting(a, _weighted(a, LIGHT_WEIGHTS)[1])
    point = a.random() < 0.10
    manifest = {
        "schema_version": 1, "profile": PROFILE, "scene_index": index, "master_seed": master_seed,
        "scene_id": f"macro_{_seed(master_seed, 'macro'):012x}_{index:07d}", "split": split,
        "size_px": SIZE, "full_scale": FULL_SCALE, "px_per_mm": s, "layout": layout,
        "parts": out_parts, "loose_debris": loose,
        "optics": optics, "illumination": light,
        "backlight": _scale_backlight(_backlight(a, 1954, optics["na"])),
        "response": {"gain": _u(a, 0.8, 1.12), "gamma": _u(a, 0.88, 1.12),
                     "black_level": _u(a, 0, 0.006), "clip_max": 0.98},
        "render": {"cycles_samples": 512, "adaptive_threshold": 0.003, "adaptive_min_samples": 64,
                   "denoise": False, "pixel_filter": "point" if point else "box",
                   "pixel_filter_width": 0.01 if point else 1.0,
                   # 8x when there are wires: a 0.25 px wire is one subsample wide at 4x
                   "mask_supersample": 8 if any(p["group"] == "wire" for p in out_parts) else 4},
    }
    return manifest, arrays
