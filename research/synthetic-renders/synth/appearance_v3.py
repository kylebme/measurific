"""Physically parameterized appearance profile ("physical_v3"). No Blender dependency.

The profile reuses the approved catalog geometry and the expanded_v2 material and
response sampling, and replaces the image-formation model:

- Optics are a telecentric-style system described by a numerical aperture (NA), an
  object-space pixel size and a focus position. Defocus is rendered by the Cycles
  orthographic camera; blur-circle diameter = 2 NA |dz| (calibrated in BENCHMARKS.md).
  The in-focus lens point-spread function (diffraction + aberration) and veiling glare
  are shift-invariant and are applied to the linear render before the camera response.
- Pixels integrate over their area (Cycles box filter) except for a small point-sampled
  fraction kept for robustness.
- A minority of parts is tilted out of the image plane (mask rendered per scene).
- Backlights range from Lambertian diffuse panels to collimated illuminators and carry
  sampled low-level nonuniformity: gradient, low-frequency noise, diffuser grain, LED
  hotspot grid and smudges.
- Contaminants (burrs, particles, fibers, hairs) are real geometry and are part of the
  ground-truth mask. Sizes are sampled in micrometres and converted with the pixel size.
"""

import math
import random

from .catalog import sample_catalog_scene
from .sampling import _seed

PROFILE = "physical_v3"

# (weight, name) tables. Weights need not sum to one.
LIGHT_WEIGHTS = ((0.40, "silhouette"), (0.25, "balanced"), (0.15, "reflective"), (0.20, "ambient"))
# Defocus classes by blur-circle diameter (px) at the focus reference plane.
BLUR_CLASSES = ((0.40, "in_focus", (0.0, 0.0)), (0.25, "mild", (0.5, 2.0)),
                (0.22, "medium", (2.0, 5.0)), (0.13, "heavy", (5.0, 12.0)))
CONTAMINATION = ((0.60, "clean", (0, 0)), (0.30, "light", (1, 4)), (0.10, "heavy", (5, 25)))
CONTAMINANT_KINDS = ((0.45, "particle"), (0.20, "fiber"), (0.10, "hair"), (0.25, "burr"))
TILT_FRACTION = 0.30
TILT_MAX_RAD = math.radians(20)
POINT_SAMPLED_FRACTION = 0.10


def _weighted(rng, table):
    total = sum(row[0] for row in table)
    x = rng.uniform(0, total)
    for row in table:
        x -= row[0]
        if x <= 0:
            return row
    return table[-1]


def _u(rng, lo, hi):
    return round(rng.uniform(lo, hi), 6)


def _logu(rng, lo, hi):
    return round(math.exp(rng.uniform(math.log(lo), math.log(hi))), 6)


def _optional(rng, probability, make):
    return make() if rng.random() < probability else None


def _lighting(a, light_class):
    # Same per-class ranges as expanded_v2 (sampling.py), resampled for the new class mix.
    spill_range, ambient_range = {
        "silhouette": ((0.0, 0.04), (0.0, 0.018)),
        "balanced": ((0.3, 1.1), (0.08, 0.38)),
        "reflective": ((0.9, 2.3), (0.04, 0.3)),
        "ambient": ((0.08, 0.6), (0.35, 0.9)),
    }[light_class]
    return {"class": light_class, "spill_strength": _u(a, *spill_range),
            "spill_angle_rad": _u(a, 0, math.tau),
            "spill_radius_fraction": _u(a, 0.22, 0.62),
            "spill_height_fraction": _u(a, 0.25, 0.65),
            "spill_size_fraction": _u(a, 0.35, 0.85),
            "ambient_strength": _u(a, *ambient_range)}


def _backlight(a, usable, na):
    # Mostly clean panels: each texture component is present with some probability
    # and log-distributed amplitude, so the sum is usually below ~1% rms.
    diffuse = a.random() < 0.45
    # A collimated illuminator narrower than the lens acceptance cone would dim the
    # visible backlight itself; real systems match or exceed the lens NA.
    min_half = 1.5 * math.degrees(math.asin(na))
    half_angle = (_u(a, 50, 90) if diffuse else _logu(a, max(1.5, min_half), max(25, min_half * 2)))
    return {
        "intensity": _u(a, 0.38, 0.75),
        "distance_px": _logu(a, 150, 3000),
        "angular": {"type": "diffuse" if diffuse else "collimated",
                    "half_angle_deg": half_angle,
                    "axis_tilt_deg": _u(a, 0, 1.0 if diffuse else 3.0),
                    "axis_azimuth_rad": _u(a, 0, math.tau)},
        "gradient": {"strength": _u(a, 0, 0.08) if a.random() < 0.6 else 0.0,
                     "angle_rad": _u(a, 0, math.tau)},
        "noise": {"strength": _logu(a, 0.001, 0.04) if a.random() < 0.7 else 0.0,
                  "scale_px": _logu(a, 60, 1500)},
        "grain": {"strength": _logu(a, 0.0005, 0.012) if a.random() < 0.5 else 0.0,
                  "scale_px": _logu(a, 1.5, 12)},
        "led_grid": {"strength": _logu(a, 0.002, 0.03) if a.random() < 0.2 else 0.0,
                     "pitch_px": _logu(a, 40, 400), "angle_rad": _u(a, 0, math.pi / 2)},
        "smudges": {"strength": _logu(a, 0.003, 0.06) if a.random() < 0.25 else 0.0,
                    "scale_px": _logu(a, 40, 400), "coverage": _u(a, 0.05, 0.4)},
        # Diffuser-plane dust: far behind the focal plane, not part of the mask.
        "dust": [{"center_px": [_u(a, -usable * 0.55, usable * 0.55),
                                _u(a, -usable * 0.55, usable * 0.55)],
                  "radius_px": _logu(a, 0.5, 8.0), "opacity": _u(a, 0.02, 0.2)}
                 for _ in range(a.choice((0, 0, 0, 1, 2, 3, 6)))],
    }


def _optics(a):
    pixel_um = _logu(a, 3.5, 40)
    wavelength_um = _u(a, 0.45, 0.65)
    # Keep the diffraction core below ~1.2 px sigma: lower NA paired with small pixels
    # is not a design anyone uses.
    na_min = max(0.004, 0.21 * wavelength_um / (1.2 * pixel_um))
    na = _logu(a, na_min, max(0.06, na_min * 1.5))
    _, blur_class, coc_range = _weighted(a, BLUR_CLASSES)
    # Gaussian approximation of the Airy core: sigma = 0.21 lambda / NA (object space).
    diffraction_sigma = 0.21 * wavelength_um / na / pixel_um
    aberration_sigma = _u(a, 0.0, 0.6) if a.random() < 0.7 else 0.0
    return {
        "na": na, "pixel_size_um": pixel_um, "wavelength_um": wavelength_um,
        "blur_class": blur_class,
        "target_coc_px": _u(a, *coc_range) if coc_range[1] > 0 else 0.0,
        "defocus_sign": a.choice((-1, 1)),
        "focus_depth_fraction": _u(a, 0.0, 1.0),
        "lens_psf": {"diffraction_sigma_px": round(diffraction_sigma, 6),
                     "aberration_sigma_px": aberration_sigma,
                     "aspect": _u(a, 1.0, 1.35) if a.random() < 0.3 else 1.0,
                     "angle_rad": _u(a, 0, math.pi)},
        "glare": {"fraction": _logu(a, 0.003, 0.15) if a.random() < 0.75 else 0.0,
                  "scale_px": _logu(a, 15, 400)},
        "vignetting": {"strength": _u(a, 0, 0.18) if a.random() < 0.5 else 0.0,
                       "center_px": [_u(a, -300, 300), _u(a, -300, 300)]},
    }


def _contaminants(a, pixel_um):
    _, level, count_range = _weighted(a, CONTAMINATION)
    items = []
    for _ in range(a.randint(*count_range)):
        _, kind = _weighted(a, CONTAMINANT_KINDS)
        attach = "edge" if kind == "burr" or a.random() < 0.7 else "loose"
        item = {"kind": kind, "attach": attach, "seed": a.getrandbits(48)}
        if kind == "particle":
            d_um = _logu(a, 5, 200)
            item.update({"diameter_um": d_um, "diameter_px": round(min(40, max(0.3, d_um / pixel_um)), 5),
                         "albedo": _u(a, 0.03, 0.7), "roughness": _u(a, 0.3, 1.0)})
        elif kind in ("fiber", "hair"):
            d_um = _logu(a, 40, 120) if kind == "hair" else _logu(a, 8, 30)
            l_um = _logu(a, 500, 10000) if kind == "hair" else _logu(a, 200, 5000)
            item.update({"diameter_um": d_um, "length_um": l_um,
                         "diameter_px": round(min(24, max(0.25, d_um / pixel_um)), 5),
                         "length_px": round(min(900, max(6, l_um / pixel_um)), 4),
                         "curvature_radius_px": _logu(a, 150, 3000) if kind == "hair" else _logu(a, 30, 800),
                         "outside_fraction": _u(a, 0.2, 0.85),
                         "albedo": _u(a, 0.02, 0.12) if kind == "hair" else _u(a, 0.25, 0.85),
                         "roughness": _u(a, 0.35, 0.95)})
        else:
            l_um = _logu(a, 10, 300)
            item.update({"length_um": l_um,
                         "length_px": round(min(40, max(0.3, l_um / pixel_um)), 5),
                         "base_width_px": None, "thickness_px": None,
                         "angle_rad": _u(a, -0.6, 0.6), "curl": _u(a, 0, 0.8)})
            item["base_width_px"] = round(max(0.25, item["length_px"] * _u(a, 0.3, 1.2)), 5)
            item["thickness_px"] = round(max(0.25, item["base_width_px"] * _u(a, 0.2, 0.8)), 5)
        items.append(item)
    return {"level": level, "items": items}


def sample_v3_scene(master_seed, scene_index, parts, catalog_digest):
    """Pair one approved base geometry with a physical_v3 image-formation sample."""
    manifest = sample_catalog_scene(master_seed, scene_index, parts, catalog_digest)
    a = random.Random(_seed(master_seed, scene_index, "appearance-v3"))
    usable = manifest["projection"]["source_height"] - 94
    manifest["appearance_profile"] = PROFILE
    manifest["illumination"] = _lighting(a, _weighted(a, LIGHT_WEIGHTS)[1])
    optics = _optics(a)
    manifest["optics"] = optics
    manifest["backlight"] = _backlight(a, usable, optics["na"])
    tilted = a.random() < TILT_FRACTION
    manifest["pose"] = {"tilt_rad": _u(a, 0.01, TILT_MAX_RAD) if tilted else 0.0,
                        "tilt_axis_rad": _u(a, 0, math.pi)}
    material = manifest["material"]
    material["bump"] = ({"strength": _logu(a, 0.02, 0.4), "scale_px": _logu(a, 2, 60),
                         "anisotropy": _u(a, 1, 12)} if a.random() < 0.5 else None)
    manifest["contamination"] = _contaminants(a, optics["pixel_size_um"])
    point = a.random() < POINT_SAMPLED_FRACTION
    manifest["render"] = {"cycles_samples": 512, "adaptive_threshold": 0.003,
                          "adaptive_min_samples": 64, "denoise": False,
                          "pixel_filter": "point" if point else "box",
                          "pixel_filter_width": 0.01 if point else 1.0,
                          "mask_tile_size": 2048}
    # Filled in by the worker once the posed part's depth is known.
    manifest["camera"] = {"focus_class": None, "focus_distance_world": None, "fstop": None,
                          "aperture_blades": 0}
    if manifest["pose"]["tilt_rad"] > 0:
        manifest["mask_plan"] = "full"
    elif manifest["contamination"]["items"]:
        manifest["mask_plan"] = "base_plus_contaminants"
    else:
        manifest["mask_plan"] = "base"
    return manifest
