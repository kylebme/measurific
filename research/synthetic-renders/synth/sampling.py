"""Deterministic, coverage-aware scene parameter sampling. No Blender dependency."""

import hashlib
import math
import os
import random
from functools import lru_cache

from . import BLENDER_VERSION, GENERATOR_VERSION, SCHEMA_VERSION


FAMILIES = ("plate", "turned", "machined", "sphere", "freeform")
MATERIALS = ("dark_matte", "rough_metal", "polished_metal", "coated", "polymer", "light_matte")
SCALES = ("subpixel", "near_pixel", "fine", "patch", "large", "very_large")
DENSITIES = ("sparse", "medium", "dense")
FOCUS = ("sharp", "moderate", "defocused", "strong_defocus", "depth_split")
LIGHTING = ("silhouette", "balanced", "reflective", "ambient")


def _seed(*parts):
    data = ":".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


@lru_cache(maxsize=1)
def _source_digest():
    digest = hashlib.sha256()
    root = os.path.dirname(__file__)
    for name in ("__init__.py", "sampling.py", "catalog.py", "geometry.py", "blender_worker.py", "png.py",
                 "projection.py", "appearance_v3.py", "worker_v3.py"):
        digest.update(name.encode("utf-8"))
        with open(os.path.join(root, name), "rb") as source:
            digest.update(source.read())
    return digest.hexdigest()


def _choice(rng, seq):
    return seq[rng.randrange(len(seq))]


def _jitter(rng, low, high):
    return round(rng.uniform(low, high), 6)


def _point(rng, radius):
    a = rng.uniform(0, math.tau)
    r = radius * math.sqrt(rng.random())
    return [round(r * math.cos(a), 5), round(r * math.sin(a), 5)]


def _plate_features(rng, radius, density, scale):
    count = {"sparse": 1, "medium": 4, "dense": 10}[density]
    shapes = ("round", "ellipse", "polygon", "filleted_polygon", "freeform", "tooth", "slot", "curved_slot")
    features = []
    widths = {"subpixel": (0.25, 0.85), "near_pixel": (0.85, 2.0), "fine": (2.0, 9.0),
              "patch": (9.0, 48.0), "large": (48.0, 100.0), "very_large": (100.0, 190.0)}
    for i in range(count):
        kind = shapes[i % len(shapes)] if density == "dense" else _choice(rng, shapes)
        local_scale = scale if i == 0 else _choice(rng, SCALES)
        w = _jitter(rng, *widths[local_scale])
        w = min(w, radius * 0.60)
        center = _point(rng, max(0, radius * 0.55 - w / 2))
        features.append({"kind": kind, "center_px": center, "width_px": w,
                         "length_px": round(min(radius * 0.7, w * rng.uniform(1.2, 4)), 5),
                         "angle_rad": _jitter(rng, 0, math.tau), "scale_class": local_scale,
                         "sides": rng.randint(3, 8), "phase_rad": _jitter(rng, 0, math.tau)})
    return features


def sample_scene(master_seed, scene_index, width=512, height=512, mask_scale=16,
                 appearance_classes=None):
    if width < 192 or height < 192 or width < height:
        raise ValueError("width and height must be >=192 and width must be >= height")
    if mask_scale != 16:
        raise ValueError("this dataset contract requires mask_scale=16")
    if width * mask_scale >= 2**31 or height * mask_scale >= 2**31:
        raise ValueError("mask exceeds supported image dimension limit")
    geometry_seed = _seed(master_seed, scene_index // 2, "geometry")
    appearance_seed = _seed(master_seed, scene_index, "appearance")
    g = random.Random(geometry_seed)
    a = random.Random(appearance_seed)
    family = FAMILIES[(scene_index // 2) % len(FAMILIES)]
    scale_class = SCALES[(scene_index // 2) % len(SCALES)]
    density = DENSITIES[(scene_index // 2 // len(SCALES)) % len(DENSITIES)]
    focus_class = (appearance_classes["focus"] if appearance_classes else
                   FOCUS[scene_index % len(FOCUS)])
    light_class = (appearance_classes["lighting"] if appearance_classes else
                   LIGHTING[(scene_index // 2) % len(LIGHTING)])
    usable = min(width, height) - 94
    radius = min(usable * _jitter(g, 0.37, 0.49), usable / 2 - 8)
    if family == "freeform":
        radius /= 1.17
    thickness = _jitter(g, 2, min(35, radius / 5))
    geometry = {"family": family, "radius_px": round(radius, 5), "thickness_px": thickness,
                "scale_class": scale_class, "density_class": density,
                "pose_angle_rad": _jitter(g, -0.08, 0.08),
                "outline": _choice(g, ("round", "polygon", "fillet", "teeth", "square_teeth", "notched", "spline")),
                "outline_sides": g.randint(3, 9), "outline_phase_rad": _jitter(g, 0, math.tau),
                "outline_amplitude_px": _jitter(g, 0.3, min(12, radius / 12)),
                "outline_repeats": g.randint(5, 32), "edge_damage": g.random() < 0.32,
                "burrs": []}
    if family == "plate":
        geometry["features"] = _plate_features(g, radius, density, scale_class)
    elif family in ("turned", "machined"):
        length = round(usable * _jitter(g, 0.55, 0.85), 5)
        steps = [round(_jitter(g, 0.65, 1.25), 5) for _ in range(g.randint(3, 7))]
        transition_limit = min(10, length / len(steps) * 0.45)
        geometry.update({"length_px": length,
                         "base_radius_px": round(usable * _jitter(g, 0.09, 0.18), 5),
                         "steps": steps,
                         "transitions": [{"kind": ("sharp", "chamfer", "radius")[(scene_index // 2 + i) % 3],
                                          "width_px": _jitter(g, 1, transition_limit)}
                                         for i in range(len(steps) - 1)],
                         "thread_profile": ("none", "v", "round", "trapezoid", "square", "none")[(scene_index // 10) % 6],
                         "thread_pitch_px": _jitter(g, 2, 14), "thread_depth_px": _jitter(g, 0.25, 5),
                         "thread_phase_rad": _jitter(g, 0, math.tau)})
        if family == "machined":
            geometry["cuts"] = [{"kind": k, "x_px": round(g.uniform(-0.25, 0.25) * geometry["length_px"], 5),
                                 "width_px": _jitter(g, 3, min(40, usable / 8)),
                                 "angle_rad": _jitter(g, -0.55, 0.55)}
                                for k in g.sample(("cross_hole", "oblique_hole", "slot", "flat", "keyway"),
                                                  {"sparse": 1, "medium": 2, "dense": 4}[density])]
    elif family == "sphere":
        geometry.update({"partial": g.random() < 0.35,
                         "cuts": [{"kind": k, "width_px": _jitter(g, 4, radius * 0.5),
                                   "offset_px": _jitter(g, -radius * 0.3, radius * 0.3)}
                                  for k in g.sample(("bore", "off_axis_bore", "flat", "dimple"),
                                                    {"sparse": 1, "medium": 2, "dense": 3}[density])]})
    else:
        geometry.update({"lobes": g.randint(2, 6), "ridge_amplitude_px": _jitter(g, 4, 28),
                         "saddle_amplitude_px": _jitter(g, -25, 25), "faceted": g.random() < 0.28})
    if geometry["edge_damage"]:
        for _ in range(g.randint(1, 3)):
            angle = _jitter(g, 0, math.tau)
            geometry["burrs"].append({"angle_rad": angle, "length_px": _jitter(g, 0.3, 7),
                                       "width_px": _jitter(g, 0.18, 1.3)})
    material_family = (appearance_classes["material"] if appearance_classes else
                       MATERIALS[scene_index % len(MATERIALS)])
    material_ranges = {
        "dark_matte": ((0.65, 0.95), (0.012, 0.09)),
        "rough_metal": ((0.38, 0.85), (0.12, 0.62)),
        "polished_metal": ((0.08, 0.34), (0.22, 0.78)),
        "coated": ((0.18, 0.62), (0.08, 0.55)),
        "polymer": ((0.32, 0.82), (0.03, 0.28)),
        "light_matte": ((0.55, 0.92), (0.34, 0.82)),
    }
    roughness_range, albedo_range = material_ranges[material_family]
    mat = {"family": material_family,
           "roughness": _jitter(a, *roughness_range),
           "metallic": 1.0 if "metal" in material_family else 0.0,
           "albedo": _jitter(a, *albedo_range),
           "coat_weight": _jitter(a, 0.1, 0.4) if material_family == "coated" else 0.0}
    # The scene uses one world unit per source pixel. Blender's orthographic
    # f-stop needs small numeric values for visible blur at this scale.
    aperture = {"sharp": (8, 32), "moderate": (0.07, 0.16),
                "defocused": (0.025, 0.07), "strong_defocus": (0.008, 0.025),
                "depth_split": (0.015, 0.05)}[focus_class]
    focus_offset = {"sharp": (-5, 5), "moderate": (300, 700),
                    "defocused": (400, 850), "strong_defocus": (550, 1000),
                    "depth_split": (-thickness * 0.5, thickness * 0.5)}[focus_class]
    focus_distance = 700 + _jitter(a, *focus_offset)
    light_ranges = {
        "silhouette": ((0.0, 0.04), (0.0, 0.018), (0.002, 0.08)),
        "balanced": ((0.3, 1.1), (0.08, 0.38), (0.05, 0.35)),
        "reflective": ((0.9, 2.3), (0.04, 0.3), (0.1, 0.55)),
        "ambient": ((0.08, 0.6), (0.35, 0.9), (0.3, 0.8)),
    }
    spill_range, ambient_range, collimation_range = light_ranges[light_class]
    spill = _jitter(a, *spill_range)
    directional_ranges = {"silhouette": (0.02, 0.22), "balanced": (0.25, 0.9),
                          "reflective": (0.7, 1.6), "ambient": (0.18, 0.75)}
    tilt_ranges = {"silhouette": (0, 0.05), "balanced": (0, 0.25),
                   "reflective": (0.08, 0.42), "ambient": (0.1, 0.5)}
    dust = []
    for _ in range(a.randrange(0, 5)):
        dust.append({"center_px": _point(a, usable * 0.55), "radius_px": _jitter(a, 0.3, 4.0),
                     "opacity": _jitter(a, 0.02, 0.15)})
    sid = f"scene_{_seed(master_seed, 'group'):016x}_{scene_index:08d}"
    return {"schema_version": SCHEMA_VERSION, "generator_version": GENERATOR_VERSION,
            "generator_source_sha256": _source_digest(), "expected_blender_version": BLENDER_VERSION,
            "scene_id": sid, "scene_index": scene_index, "master_seed": master_seed,
            "geometry_seed": geometry_seed, "appearance_seed": appearance_seed,
            "projection": {"type": "orthographic", "source_width": width, "source_height": height,
                           "mask_scale": mask_scale, "ortho_scale_world": float(height),
                           "camera_position_world": [0, 0, 700], "camera_rotation_rad": [0, 0, 0],
                           "margin_px": 47},
            "geometry": geometry, "material": mat,
            "backlight": {"intensity": _jitter(a, 0.38, 0.75), "gradient": _jitter(a, -0.08, 0.08),
                          "texture_strength": _jitter(a, 0, 0.045), "dust": dust,
                          "collimation_angle_rad": _jitter(a, *collimation_range),
                          "tilt_angle_rad": _jitter(a, *tilt_ranges[light_class]),
                          "tilt_azimuth_rad": _jitter(a, 0, math.tau),
                          "directional_strength": _jitter(a, *directional_ranges[light_class])},
            "illumination": {"class": light_class, "spill_strength": spill,
                             "spill_angle_rad": _jitter(a, 0, math.tau),
                             "spill_radius_fraction": _jitter(a, 0.22, 0.62),
                             "spill_height_fraction": _jitter(a, 0.25, 0.65),
                             "spill_size_fraction": _jitter(a, 0.35, 0.85),
                             "ambient_strength": _jitter(a, *ambient_range)},
            "camera": {"focus_class": focus_class, "focus_distance_world": focus_distance,
                       "fstop": _jitter(a, *aperture), "aperture_blades": _choice(a, (0, 0, 0, 6, 8))},
            "response": {"gain": _jitter(a, 0.8, 1.12), "gamma": _jitter(a, 0.88, 1.12),
                         "black_level": _jitter(a, 0, 0.006), "clip_max": 0.98},
            "render": {"cycles_samples": 128, "pixel_filter_width": 0.01, "denoise": False,
                       "mask_tile_size": 1024},
            "files": {"image": "image.png", "mask_dir": "mask", "mask_index": "mask/index.json"}}
