"""Deterministic base-part catalog for high-resolution geometry review.

All dimensions are in final 1x image pixels. The saved geometry dictionaries can
be passed directly to geometry.build when photometric variants are made later.
"""

import hashlib
import math
import random

from .sampling import FAMILIES, FOCUS, LIGHTING, MATERIALS, _seed, sample_scene

CATALOG_SCHEMA = 1
LOCAL_FEATURE_LIMIT_PX = 48


def _f(rng, lo, hi):
    return round(rng.uniform(lo, hi), 4)


def _edge_features(rng, count, kinds):
    features = []
    for i in range(count):
        span = _f(rng, 3, LOCAL_FEATURE_LIMIT_PX)
        features.append({"kind": kinds[i % len(kinds)],
                         "angle_rad": round((i + rng.uniform(-0.26, 0.26)) * math.tau / count, 6),
                         "span_px": span, "depth_px": _f(rng, 1.2, min(13, span * 0.65))})
    rng.shuffle(features)
    return features


def _plate_features(rng, radius, index):
    kinds = ("round", "ellipse", "polygon", "filleted_polygon", "freeform",
             "tooth", "slot", "curved_slot")
    scales = ((1.2, 2.5), (2.5, 7), (7, 17), (17, 48))
    features = []
    placed = []
    count = 32 + index % 17
    for i in range(count):
        lo, hi = scales[i % len(scales)]
        width = _f(rng, lo, hi)
        length = _f(rng, width, min(48, width * rng.uniform(1.2, 2.7)))
        # Reject overlaps so the catalog actually exposes the sampled details.
        for _ in range(80):
            a = rng.uniform(0, math.tau)
            rr = rng.uniform(0.1, 0.76) * radius
            x, y = rr * math.cos(a), rr * math.sin(a)
            if all(math.hypot(x - px, y - py) > (length + plen) / 2 + 5
                   for px, py, plen in placed):
                break
        placed.append((x, y, length))
        features.append({"kind": kinds[(i + index) % len(kinds)],
                         "center_px": [round(x, 4), round(y, 4)],
                         "width_px": width, "length_px": length,
                         "angle_rad": _f(rng, 0, math.tau),
                         "scale_class": ("near_pixel", "fine", "fine", "patch")[i % 4],
                         "sides": rng.randint(3, 9), "phase_rad": _f(rng, 0, math.tau)})
    return features


def _axial_features(rng, length, count):
    kinds = ("groove", "neck", "ridge", "shoulder", "wave")
    features = []
    for i in range(count):
        span = _f(rng, 4, LOCAL_FEATURE_LIMIT_PX)
        x = -length * 0.43 + (i + rng.uniform(0.15, 0.85)) * length * 0.86 / count
        features.append({"kind": kinds[i % len(kinds)], "x_px": round(x, 4),
                         "span_px": span, "depth_px": _f(rng, 1.5, 10)})
    rng.shuffle(features)
    return features


def _thread_regions(rng, length, count, index):
    profiles = ("v", "round", "trapezoid", "square")
    regions = []
    for i in range(count):
        span = _f(rng, 25, 48)
        x = -length * 0.40 + (i + 0.5) * length * 0.80 / count
        regions.append({"profile": profiles[(i + index) % len(profiles)],
                        "x_px": round(x, 4), "span_px": span,
                        "pitch_px": _f(rng, 3.2, 11.5), "depth_px": _f(rng, 1.5, 7),
                        "phase_rad": _f(rng, 0, math.tau)})
    return regions


def _turned_min_diameter_px(index):
    """Stratify 60 turned bodies from 1 px to the existing large-diameter range."""
    ordinal = index // len(FAMILIES)
    if ordinal < 30:
        value = 80 ** (ordinal / 29)
    else:
        value = 85 * (500 / 85) ** ((ordinal - 30) / 29)
    return round(value, 4)


def sample_part(master_seed, index, resolution=2048):
    """Return one reproducible base part; every fifth index revisits a family."""
    if resolution < 256 or index < 0:
        raise ValueError("resolution must be >= 256 and index must be nonnegative")
    rng = random.Random(_seed(master_seed, index, "catalog-geometry-v1"))
    family = FAMILIES[index % len(FAMILIES)]
    usable = resolution - 94  # At least 47 pixels of nominal framing margin.
    radius = usable * _f(rng, 0.34, 0.45)
    if family == "freeform":
        radius /= 1.22
    g = {"family": family, "radius_px": round(radius, 4),
         "thickness_px": _f(rng, 4, 38),
         "scale_class": "mixed", "density_class": "dense",
         "pose_angle_rad": _f(rng, -0.045, 0.045),
         "outline": rng.choice(("round", "polygon", "fillet", "teeth", "notched", "spline")),
         "outline_sides": rng.randint(4, 10), "outline_phase_rad": _f(rng, 0, math.tau),
         "outline_amplitude_px": _f(rng, 2, 12),
         "outline_repeats": rng.randint(5, 21), "edge_damage": True,
         "burrs": [], "tessellation_error_px": 0.10}
    if family == "plate":
        g["features"] = _plate_features(rng, radius, index)
        g["edge_segments"] = _edge_features(rng, 18 + index % 9,
            ("notch", "chip", "square_notch", "tooth", "hair", "square_tooth", "wave"))
        g["edge_sample_px"] = 0.8
    elif family in ("turned", "machined"):
        length = round(usable * _f(rng, 0.72, 0.88), 4)
        nsteps = rng.randint(3, 6)
        g.update({"length_px": length,
                  "base_radius_px": round(usable * _f(rng, 0.09, 0.15), 4),
                  "steps": [_f(rng, 0.72, 1.23) for _ in range(nsteps)],
                  "transitions": [{"kind": ("sharp", "chamfer", "radius")[i % 3],
                                   "width_px": _f(rng, 3, 15)} for i in range(nsteps - 1)],
                  "thread_profile": "none", "thread_pitch_px": 8,
                  "thread_depth_px": 0, "thread_phase_rad": 0,
                  "thread_regions": _thread_regions(rng, length, 4 + index % 3, index),
                  "axial_features": _axial_features(rng, length, 14 + index % 8),
                  "base_axial_step_px": 3.0})
        if family == "machined":
            kinds = ("cross_hole", "oblique_hole", "slot", "flat", "keyway")
            g["cuts"] = [{"kind": kinds[(index + i) % len(kinds)],
                           "x_px": round((-0.32 + i * 0.21) * length, 4),
                           "width_px": _f(rng, 9, 40), "length_px": _f(rng, 20, 48),
                           "angle_rad": _f(rng, -0.42, 0.42)} for i in range(3)]
        if family == "turned":
            minimum_diameter = _turned_min_diameter_px(index)
            base_radius = minimum_diameter / (2 * min(g["steps"]))
            g["minimum_diameter_px"] = minimum_diameter
            g["base_radius_px"] = round(base_radius, 6)
            g["radius_px"] = round(base_radius * max(g["steps"]), 6)
            depth_scale = min(1, minimum_diameter / 40)
            for feature in g["axial_features"]:
                feature["depth_px"] = round(feature["depth_px"] * depth_scale, 6)
            for region in g["thread_regions"]:
                region["depth_px"] = round(region["depth_px"] * depth_scale, 6)
    elif family == "sphere":
        g["partial"] = False
        g["rim_features"] = _edge_features(rng, 16 + index % 9,
            ("notch", "chip", "square_notch", "tooth", "hair", "wave"))
        # A small bore supplies interior negative space without obscuring the rim.
        g["cuts"] = [{"kind": "off_axis_bore", "width_px": _f(rng, 8, 35),
                      "offset_px": _f(rng, -radius * 0.38, radius * 0.38)}]
    else:
        g.update({"lobes": rng.randint(2, 7),
                  "ridge_amplitude_px": _f(rng, 3, 26),
                  "saddle_amplitude_px": _f(rng, -20, 20),
                  "faceted": rng.random() < 0.25,
                  "rim_features": _edge_features(rng, 18 + index % 9,
                      ("notch", "chip", "square_notch", "tooth", "hair", "wave")),
                  "rim_sample_px": 3.5})
    part_id = f"part_{index:04d}_{family}"
    return {"catalog_schema_version": CATALOG_SCHEMA, "part_id": part_id,
            "index": index, "master_seed": master_seed,
            "geometry_seed": _seed(master_seed, index, "catalog-geometry-v1"),
            "resolution": resolution, "geometry": g}


def sample_catalog(master_seed=731029, count=300, resolution=2048):
    if count < 1:
        raise ValueError("count must be positive")
    return [sample_part(master_seed, i, resolution) for i in range(count)]


def catalog_sha256(parts):
    import json
    return hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sample_catalog_scene(master_seed, scene_index, parts, catalog_digest):
    """Pair one approved base geometry with independently sampled image formation."""
    if not parts or len(parts) % len(FAMILIES):
        raise ValueError("catalog must contain complete five-family groups")
    groups = len(parts) // len(FAMILIES)
    family_slot = scene_index % len(FAMILIES)
    # Traverse each family with a coprime stride. The first pilot scenes see
    # small, medium, and large turned diameters rather than only the 1px end.
    stride = 17 if groups == 60 else (1 if groups == 1 else next(
        (step for step in range(2, groups + 1) if math.gcd(step, groups) == 1), 1))
    ordinal = ((scene_index // len(FAMILIES)) * stride + 13) % groups
    part = parts[ordinal * len(FAMILIES) + family_slot]
    if part["geometry"]["family"] != FAMILIES[family_slot]:
        raise ValueError("catalog family ordering is inconsistent")
    resolution = part["resolution"]
    variant = scene_index // len(parts)
    # Each approved part must see every broad imaging class as variants accrue.
    # Coprime strides and independent offsets prevent the 300-part cycle from
    # locking a part to one material, focus, and lighting combination.
    classes = {
        "focus": FOCUS[(_seed(master_seed, part["part_id"], "focus-v2") + variant) % len(FOCUS)],
        "material": MATERIALS[(_seed(master_seed, part["part_id"], "material-v2") + 5 * variant) % len(MATERIALS)],
        "lighting": LIGHTING[(_seed(master_seed, part["part_id"], "lighting-v2") + 3 * variant) % len(LIGHTING)],
    }
    manifest = sample_scene(master_seed, scene_index, resolution, resolution,
                            mask_scale=16, appearance_classes=classes)
    import copy
    g = copy.deepcopy(part["geometry"])
    manifest["geometry"] = g
    manifest["geometry_seed"] = part["geometry_seed"]
    manifest["scene_id"] = f"scene_{_seed(master_seed, catalog_digest, 'group'):016x}_{scene_index:08d}"
    manifest["catalog"] = {"part_id": part["part_id"], "part_index": part["index"],
                           "catalog_sha256": catalog_digest, "appearance_variant_index": variant}
    manifest["appearance_profile"] = "expanded_v2"
    camera_z = float(resolution * 2)
    manifest["projection"]["camera_position_world"] = [0, 0, camera_z]
    manifest["render"]["mask_tile_size"] = 2048
    if g["family"] == "plate":
        front_z = g["thickness_px"] / 2
    elif g["family"] in ("turned", "machined", "sphere"):
        front_z = g["radius_px"] if g["family"] == "sphere" else g["base_radius_px"] * max(g["steps"])
    else:
        front_z = 0.38 * g["radius_px"] + abs(g["ridge_amplitude_px"]) + abs(g["saddle_amplitude_px"])
    focus_offset = manifest["camera"]["focus_distance_world"] - 700
    if classes["focus"] == "depth_split":
        focus_rng = random.Random(_seed(master_seed, scene_index, "depth-focus-v2"))
        depth = max(g["thickness_px"], front_z * 1.5)
        focus_offset = round(max(45, min(650, depth * focus_rng.uniform(0.18, 0.55))), 5)
    manifest["camera"]["focus_distance_world"] = round(camera_z - front_z + focus_offset, 5)
    return manifest
