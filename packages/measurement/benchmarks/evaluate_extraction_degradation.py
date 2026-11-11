#!/usr/bin/env python3
"""Render a controlled blur/noise sweep and score image contour extraction.

The source geometry, camera pose, illumination, and appearance RNG are fixed for
each family.  Only PSF blur and sensor-noise parameters vary.  Exact projected
contours are kept in a separate ``truth`` directory so opening an image in the
GUI exercises image extraction instead of the GUI's colocated-sidecar shortcut.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely import distance, points

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "measurement" / "src"),
    str(ROOT / "edge-model"),
    str(ROOT.parent / "research" / "synthetic-shapes" / "src"),
]

from inspection_measurement import contours_from_image  # noqa: E402
from inspection_shapes.geometry import generate  # noqa: E402
from inspection_shapes.render import Camera, render  # noqa: E402
from inspection_shapes.taxonomy import PARTS  # noqa: E402


BLURS = (0.35, 0.80, 1.30, 1.80)
NOISES = {
    # Shot noise remains in the physical renderer.  The very high conversion
    # gain makes it sub-quantization here, yielding a practical clean control.
    "clean": {"read_noise_dn": 0.0, "electrons_per_dn": 100_000_000.0, "prnu": 0.0},
    "low": {"read_noise_dn": 0.15, "electrons_per_dn": 1_000.0, "prnu": 0.0005},
    "nominal": {"read_noise_dn": 1.5, "electrons_per_dn": 40.0, "prnu": 0.003},
}


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=json_default) + "\n")


def fixed_camera(part, *, width: int, height: int, supersample: int) -> Camera:
    """Fit a deterministic, slightly rotated part with a generous border."""
    angle = 0.213
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([[c, -s], [-s, -c]])
    transformed = np.concatenate([ring.actual for ring in part.rings]) @ matrix.T
    extent = np.ptp(transformed, axis=0)
    middle = (transformed.min(axis=0) + transformed.max(axis=0)) / 2
    margin = 72.0
    pixels_per_mm = float(min((width - 2 * margin) / extent[0],
                              (height - 2 * margin) / extent[1]))
    return Camera(
        width=width,
        height=height,
        supersample=supersample,
        rotation=angle,
        pixels_per_mm=pixels_per_mm,
        translation_x=float((width - 1) / 2 - middle[0] * pixels_per_mm),
        translation_y=float((height - 1) / 2 - middle[1] * pixels_per_mm),
        sigma_px=0.8,
        dark=12.0,
        bright=235.0,
        read_noise_dn=0.0,
        electrons_per_dn=100_000_000.0,
        prnu=0.0,
    )


def paths(xy: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [xy[lo:hi] for lo, hi in zip(offsets[:-1], offsets[1:])]


def densify(path: np.ndarray, spacing: float = 0.25) -> np.ndarray:
    end = np.roll(path, -1, axis=0)
    length = np.linalg.norm(end - path, axis=1)
    pieces = []
    for start, stop, distance in zip(path, end, length):
        count = max(1, int(np.ceil(distance / spacing)))
        pieces.append(start + np.arange(count)[:, None] / count * (stop - start))
    return np.concatenate(pieces)


def geometry(path: np.ndarray) -> tuple[np.ndarray, float]:
    polygon = Polygon(path)
    return np.array([polygon.centroid.x, polygon.centroid.y]), float(polygon.area)


def score_extraction(truth_paths, truth_holes, extracted, diagnostics) -> dict:
    observed_paths = [extracted.path(i) for i in range(len(extracted.closed))]
    observed_holes = [bool(item["is_hole"]) for item in diagnostics["rings"]]
    truth_geometry = [geometry(q) for q in truth_paths]
    observed_geometry = [geometry(q) for q in observed_paths]
    cost = np.empty((len(truth_paths), len(observed_paths)))
    diagonal = np.linalg.norm(np.ptp(np.concatenate(truth_paths), axis=0))
    for i, (truth_center, truth_area) in enumerate(truth_geometry):
        for j, (observed_center, observed_area) in enumerate(observed_geometry):
            role_penalty = 10.0 if bool(truth_holes[i]) != observed_holes[j] else 0.0
            cost[i, j] = (np.linalg.norm(truth_center - observed_center) / max(diagonal, 1.0)
                          + abs(np.log(max(observed_area, 1e-12) / truth_area))
                          + role_penalty)
    rows, columns = linear_sum_assignment(cost)
    distances = []
    ring_scores = []
    for i, j in zip(rows, columns):
        truth_dense = densify(truth_paths[i], .1)
        observed_dense = densify(observed_paths[j], .1)
        # Distances to continuous segments have no nearest-sample error floor.
        # Sampling both directions at <=.1 px bounds missed maxima by .05 px.
        truth_line = LineString(np.vstack([truth_paths[i], truth_paths[i][0]])).simplify(1e-10)
        observed_line = LineString(np.vstack([observed_paths[j], observed_paths[j][0]])).simplify(1e-10)
        observed_to_truth = distance(points(observed_dense), truth_line)
        truth_to_observed = distance(points(truth_dense), observed_line)
        both = np.r_[observed_to_truth, truth_to_observed]
        distances.append(both)
        truth_center, truth_area = truth_geometry[i]
        observed_center, observed_area = observed_geometry[j]
        ring_scores.append({
            "truth_ring": int(i),
            "observed_ring": int(j),
            "is_hole": bool(truth_holes[i]),
            "hausdorff_px": float(both.max()),
            "p95_distance_px": float(np.quantile(both, 0.95)),
            "rms_distance_px": float(np.sqrt(np.mean(both ** 2))),
            "centroid_error_px": float(np.linalg.norm(truth_center - observed_center)),
            "relative_area_error": float(abs(observed_area - truth_area) / truth_area),
            "observed_points": int(len(observed_paths[j])),
        })
    combined = np.concatenate(distances) if distances else np.array([], dtype=float)
    topology_ok = (len(truth_paths) == len(observed_paths)
                   and all(bool(truth_holes[i]) == observed_holes[j]
                           for i, j in zip(rows, columns)))
    return {
        "status": "ok" if topology_ok else "topology_mismatch",
        "truth_rings": len(truth_paths),
        "observed_rings": len(observed_paths),
        "matched_rings": len(rows),
        "hausdorff_px": float(combined.max()) if len(combined) else None,
        "p95_distance_px": float(np.quantile(combined, 0.95)) if len(combined) else None,
        "rms_distance_px": float(np.sqrt(np.mean(combined ** 2))) if len(combined) else None,
        "maximum_relative_area_error": max((r["relative_area_error"] for r in ring_scores), default=None),
        "maximum_centroid_error_px": max((r["centroid_error_px"] for r in ring_scores), default=None),
        "canny_thresholds": diagnostics.get("canny_thresholds"),
        "canny_topology_close_size": diagnostics.get("canny_topology_close_size"),
        "extraction_diagnostics": diagnostics,
        "rings": ring_scores,
    }


def preview_tile(image, truth_paths, extracted, result, title: str, size=(300, 220)):
    source = Image.fromarray(image).convert("RGB")
    source.thumbnail(size)
    canvas = Image.new("RGB", (size[0], size[1] + 42), "#141b22")
    left = (size[0] - source.width) // 2
    canvas.paste(source, (left, 42))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((7, 5), title, fill="white", font=font)
    status = result["status"]
    if status == "ok":
        detail = f'p95 {result["p95_distance_px"]:.2f}px | max {result["hausdorff_px"]:.2f}px'
        color = "#8fe3a8"
    else:
        detail = result.get("error", status)
        color = "#ff7e8a"
    draw.text((7, 22), detail[:52], fill=color, font=font)
    scale_x, scale_y = source.width / image.shape[1], source.height / image.shape[0]

    def line(points, color):
        q = [(left + float(x) * scale_x, 42 + float(y) * scale_y) for x, y in points]
        if q:
            draw.line(q + [q[0]], fill=color, width=1)

    for q in truth_paths:
        line(q, "#21e6de")
    if extracted is not None:
        for i in range(len(extracted.closed)):
            line(extracted.path(i), "#ff4fc8")
    return canvas


def make_contact_sheet(tiles, destination: Path, families: list[str]) -> None:
    tile_width, tile_height = tiles[0][2].size
    columns = len(BLURS)
    family_height = len(NOISES) * tile_height + 34
    sheet = Image.new("RGB", (columns * tile_width, len(families) * family_height), "#0d1319")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for family_index, family in enumerate(families):
        top = family_index * family_height
        draw.text((8, top + 9), f"{family} — cyan truth, magenta extracted", fill="white", font=font)
        family_tiles = [tile for tile in tiles if tile[0] == family]
        for _, condition_index, tile in family_tiles:
            row, column = divmod(condition_index, columns)
            sheet.paste(tile, (column * tile_width, top + 34 + row * tile_height))
    sheet.save(destination)


def report_markdown(results: list[dict], families: list[str], *, source=None, method='default') -> str:
    lines = [
        "# Controlled image-extraction degradation sweep",
        "",
        "Every family uses one fixed, perfect synthetic geometry, camera pose, contrast, and appearance seed. Only Gaussian PSF blur and sensor noise change. Exact projected contours are stored separately from the images so GUI loading cannot bypass extraction.",
        "",
        f"Extractor: `{method}`. " + (f"Image/truth source: `{source}`." if source is not None else "Image/truth source: this directory."),
        "",
        "Errors are distances to continuous segments, sampled in both directions at <=0.1 px arclength intervals. The sampled maximum can underestimate the continuous maximum by up to 0.05 px.",
        "",
        "| Noise | Blur (px) | Successful | Topology mismatch | Extraction error | Median p95 error (px) | Worst error (px) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for noise in NOISES:
        for blur in BLURS:
            subset = [r for r in results if r["noise"] == noise and r["blur_sigma_px"] == blur]
            ok = [r for r in subset if r["status"] == "ok"]
            values = [r["p95_distance_px"] for r in ok]
            worst = [r["hausdorff_px"] for r in ok]
            median_text = f"{np.median(values):.3f}" if values else "—"
            worst_text = f"{max(worst):.3f}" if worst else "—"
            lines.append(
                f"| {noise} | {blur:.2f} | {len(ok)}/{len(families)} | "
                f"{sum(r['status'] == 'topology_mismatch' for r in subset)} | "
                f"{sum(r['status'] == 'extraction_error' for r in subset)} | "
                f"{median_text} | {worst_text} |"
            )
    lines += [
        "",
        "## Layout",
        "",
        "- Source `images/`: GUI-testable PNGs, without adjacent contour sidecars.",
        "- Source `truth/`: exact projected contours, topology, feature/primitive IDs, and millimetre geometry.",
        "- `results.json`: extraction status and geometric error against truth (generation also writes `results.csv`).",
        "- `contact-sheet.png`: cyan truth and magenta extracted contours for fast visual review.",
        "- `manifest.json`: experiment method/source; the source manifest records seeds, camera, and renderer settings.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=["rack", "slotted_bracket", "flange"])
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--supersample", type=int, default=4)
    args = parser.parse_args()
    unknown = set(args.families) - set(PARTS)
    if unknown:
        raise ValueError(f"Unknown part families: {sorted(unknown)}")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{args.output.name}-", dir=args.output.parent) as temporary:
        output = Path(temporary)
        (output / "images").mkdir()
        (output / "truth").mkdir()
        results = []
        tiles = []
        family_records = []
        for family_index, family in enumerate(args.families):
            geometry_seed = args.seed + family_index * 10_000
            appearance_seed = args.seed + family_index * 10_000 + 1
            # Severity zero intentionally removes dimensional/form variation. The
            # observed error is therefore attributable to imaging/extraction.
            part = generate(family, geometry_seed, severity=0.0)
            camera = fixed_camera(part, width=args.width, height=args.height,
                                  supersample=args.supersample)
            truth_paths = [camera.project(ring.actual) for ring in part.rings]
            offsets = np.r_[0, np.cumsum([len(q) for q in truth_paths])].astype(np.int32)
            np.savez_compressed(
                output / "truth" / f"{family}.npz",
                actual_xy_px=np.concatenate(truth_paths),
                actual_xy_mm=np.concatenate([ring.actual for ring in part.rings]),
                nominal_xy_mm=np.concatenate([ring.nominal for ring in part.rings]),
                ring_offsets=offsets,
                ring_is_hole=np.array([ring.hole for ring in part.rings]),
                edge_feature_id=np.concatenate([ring.feature for ring in part.rings]),
                edge_primitive_id=np.concatenate([ring.primitive for ring in part.rings]),
            )
            family_records.append({
                "family": family,
                "geometry_seed": geometry_seed,
                "appearance_seed": appearance_seed,
                "truth_file": f"truth/{family}.npz",
                "camera": asdict(camera),
                "features": part.features,
                "primitives": part.primitives,
                "relationships": part.relationships,
                "patterns": part.patterns,
            })
            condition_index = 0
            for noise, noise_parameters in NOISES.items():
                for blur in BLURS:
                    condition_camera = Camera(**{
                        **asdict(camera),
                        "sigma_px": blur,
                        **noise_parameters,
                    })
                    rendered, _, _ = render(part, condition_camera, appearance_seed)
                    image = rendered["image"]
                    stem = f"{family}__blur-{blur:.2f}__noise-{noise}"
                    image_file = output / "images" / f"{stem}.png"
                    Image.fromarray(image).save(image_file)
                    extracted = None
                    try:
                        extracted, diagnostics = contours_from_image(image)
                        result = score_extraction(
                            truth_paths,
                            [ring.hole for ring in part.rings],
                            extracted,
                            diagnostics,
                        )
                    except Exception as error:  # Each failed condition remains evidence.
                        result = {
                            "status": "extraction_error",
                            "error": f"{type(error).__name__}: {error}",
                            "truth_rings": len(truth_paths),
                            "observed_rings": None,
                            "matched_rings": 0,
                            "hausdorff_px": None,
                            "p95_distance_px": None,
                            "rms_distance_px": None,
                            "maximum_relative_area_error": None,
                            "maximum_centroid_error_px": None,
                        }
                    result.update({
                        "family": family,
                        "image": f"images/{image_file.name}",
                        "truth": f"truth/{family}.npz",
                        "blur_sigma_px": blur,
                        "noise": noise,
                        **noise_parameters,
                    })
                    results.append(result)
                    tiles.append((family, condition_index, preview_tile(
                        image, truth_paths, extracted, result,
                        f"blur {blur:.2f}px | {noise}",
                    )))
                    condition_index += 1
                    print(f"{family:18s} blur={blur:.2f} noise={noise:7s} {result['status']}", flush=True)

        write_json(output / "results.json", results)
        scalar_fields = [
            "family", "image", "truth", "blur_sigma_px", "noise", "read_noise_dn",
            "electrons_per_dn", "prnu", "status", "error", "truth_rings",
            "observed_rings", "matched_rings", "hausdorff_px", "p95_distance_px",
            "rms_distance_px", "maximum_relative_area_error", "maximum_centroid_error_px",
            "canny_topology_close_size",
        ]
        with (output / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=scalar_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        manifest = {
            "schema_version": 2,
            "rasterization": "exact polygon area per subpixel",
            "distance_metric": "bidirectional sample-to-continuous-segment, <=0.1 px sampling; max underestimation <=0.05 px",
            "purpose": "Controlled synthetic-shapes image-extraction degradation sweep",
            "base_seed": args.seed,
            "geometry_severity": 0.0,
            "fixed_across_conditions": [
                "geometry", "pose", "scale", "contrast", "illumination", "appearance RNG seed",
            ],
            "varied": ["Gaussian PSF sigma", "shot noise", "read noise", "PRNU"],
            "image_size": [args.width, args.height],
            "supersample": args.supersample,
            "blur_sigma_px": BLURS,
            "noise_presets": NOISES,
            "families": family_records,
            "gui_extraction_note": "Truth files are intentionally not beside images under matching stems; opening images in the GUI exercises edge-model extraction.",
        }
        write_json(output / "manifest.json", manifest)
        make_contact_sheet(tiles, output / "contact-sheet.png", args.families)
        (output / "README.md").write_text(report_markdown(results, args.families))
        os.replace(output, args.output)


if __name__ == "__main__":
    main()
