"""Create, render, validate, and browse the 1x base-part review catalog."""

import argparse
import hashlib
import html
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

from . import BLENDER_VERSION
from .catalog import LOCAL_FEATURE_LIMIT_PX, catalog_sha256, sample_catalog
from .sampling import FAMILIES

BLENDER = "/snap/blender/current/blender"


def source_sha256():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("catalog.py", "geometry.py", "review_worker.py", "png.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def plan(root, seed=731029, count=300, resolution=2048):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    parts = sample_catalog(seed, count, resolution)
    document = {"schema_version": 1, "count": count, "resolution": resolution,
                "master_seed": seed, "local_feature_limit_px": LOCAL_FEATURE_LIMIT_PX,
                "source_sha256": source_sha256(), "catalog_sha256": catalog_sha256(parts),
                "parts": parts}
    path = root / "catalog.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != document:
            raise ValueError("catalog.json exists with different content; select a new output directory")
    else:
        path.write_text(json.dumps(document, indent=2) + "\n")
    return path


def render(root, blender=BLENDER, start=0, stop=None, force=False):
    root = Path(root)
    command = [blender, "-b", "-t", "4", "--python", str(Path(__file__).parent / "review_worker.py"),
               "--", str(root / "catalog.json"), str(root), "--start", str(start)]
    if stop is not None:
        command += ["--stop", str(stop)]
    if force:
        command.append("--force")
    subprocess.run(command, check=True)


def _feature_points(part, mask):
    g = part["geometry"]
    n = part["resolution"]
    pose = g["pose_angle_rad"]
    points = []

    def project(x, y, kind):
        xx = x * math.cos(pose) - y * math.sin(pose)
        yy = x * math.sin(pose) + y * math.cos(pose)
        points.append((round(n / 2 + xx), round(n / 2 - yy), kind))

    def project_edge(angle, radius, kind):
        a = angle + pose
        final = None
        for rr in range(max(1, round(radius * 0.62)), min(n // 2 - 47, round(radius * 1.32))):
            x = round(n / 2 + rr * math.cos(a))
            y = round(n / 2 - rr * math.sin(a))
            if 0 <= x < n and 0 <= y < n and mask.getpixel((x, y)):
                final = (x, y, kind)
        if final:
            points.append(final)

    family = g["family"]
    if family == "plate":
        for f in g["features"]:
            project(*f["center_px"], f["kind"])
        for f in g["edge_segments"]:
            project_edge(f["angle_rad"], g["radius_px"], f["kind"])
    elif family in ("turned", "machined"):
        if family == "turned" and g.get("minimum_diameter_px", 1000) <= 48:
            project(0, 0, f"body min Ø {g['minimum_diameter_px']:g} px")
        for f in g["thread_regions"] + g["axial_features"]:
            x = round(n / 2 + f["x_px"] * math.cos(pose))
            x = min(n - 1, max(0, x))
            # Find the actual silhouette at this axial position after all steps.
            column = mask.crop((x, 0, x + 1, n))
            box = column.getbbox()
            if box:
                points.append((x, box[1], f.get("profile", f.get("kind", "feature"))))
        for f in g.get("cuts", []):
            project(f["x_px"], 0, f["kind"])
    elif family == "sphere":
        for f in g["cuts"]:
            project(f["offset_px"], 0, f["kind"])
        for f in g["rim_features"]:
            project_edge(f["angle_rad"], g["radius_px"], f["kind"])
    else:
        for f in g["rim_features"]:
            project_edge(f["angle_rad"], g["radius_px"], f["kind"])
    return points


def _details(part, mask, path):
    from PIL import Image, ImageDraw, ImageOps

    candidates = _feature_points(part, mask)
    if not candidates:
        return
    # Spread selected regions across the part, while retaining different kinds.
    selected = []
    used_kinds = set()
    for x, y, kind in candidates:
        if kind in used_kinds or any(math.hypot(x - xx, y - yy) < 90 for xx, yy, _ in selected):
            continue
        selected.append((x, y, kind))
        used_kinds.add(kind)
        if len(selected) == 8:
            break
    for item in candidates:
        if len(selected) == 8:
            break
        if item not in selected and all(math.hypot(item[0] - x, item[1] - y) >= 70 for x, y, _ in selected):
            selected.append(item)
    tile = 174
    out = Image.new("RGB", (tile * 4, tile * 2), "#e9edf2")
    draw = ImageDraw.Draw(out)
    for i, (x, y, kind) in enumerate(selected[:8]):
        crop = mask.crop((x - 48, y - 48, x + 48, y + 48))
        visible = ImageOps.invert(crop.convert("L")).resize((150, 150), Image.Resampling.NEAREST)
        left, top = (i % 4) * tile + 12, (i // 4) * tile + 5
        out.paste(visible.convert("RGB"), (left, top))
        draw.text((left, top + 151), kind[:21], fill="#182333")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, optimize=True)


def _thin_preview(mask, size):
    """Make a visible thumbnail of a 1-pixel body without altering its mask."""
    from PIL import Image

    coverage = mask.convert("L").resize((size, size), Image.Resampling.BOX)
    return coverage.point(lambda value: 0 if value else 255)


def build_gallery(root, require_complete=True):
    from PIL import Image, ImageDraw, ImageOps

    root = Path(root)
    document = json.loads((root / "catalog.json").read_text())
    parts = document["parts"]
    if catalog_sha256(parts) != document["catalog_sha256"]:
        raise ValueError("catalog parts do not match the recorded SHA-256")
    records = []
    by_family = {family: [] for family in FAMILIES}
    for part in parts:
        pid = part["part_id"]
        mask_path = root / "masks" / (pid + ".png")
        result_path = root / "results" / (pid + ".json")
        if not mask_path.exists() or not result_path.exists():
            if require_complete:
                raise ValueError(f"missing mask or result for {pid}")
            continue
        with Image.open(mask_path) as original:
            if original.size != (part["resolution"],) * 2 or original.mode != "1":
                raise ValueError(f"invalid 1x binary mask: {pid}")
            mask = original.copy()
        bounds = mask.getbbox()
        if bounds is None:
            raise ValueError(f"empty mask: {pid}")
        margin = min(bounds[0], bounds[1], part["resolution"] - bounds[2], part["resolution"] - bounds[3])
        if margin < 47:
            raise ValueError(f"mask enters nominal 47-pixel margin: {pid} ({margin}px)")
        result = json.loads(result_path.read_text())
        if result["bounds_xyxy"] != list(bounds):
            raise ValueError(f"mask/result bounds mismatch: {pid}")
        if result["foreground_pixels"] != mask.histogram()[255]:
            raise ValueError(f"mask/result pixel count mismatch: {pid}")
        if part["geometry"]["family"] == "plate":
            # Every configured plate outline contains these inner rings. A
            # handful of tiny holes may hit probes; a missing sector means the
            # curve fill dropped a large region despite a valid PNG.
            missing = 0
            for fraction in (0.4, 0.55, 0.62):
                radius = fraction * part["geometry"]["radius_px"]
                for i in range(24):
                    angle = math.tau * i / 24
                    x = round(part["resolution"] / 2 + radius * math.cos(angle))
                    y = round(part["resolution"] / 2 - radius * math.sin(angle))
                    missing += not mask.getpixel((x, y))
            if missing > 3:
                raise ValueError(f"plate interior fill failed: {pid} ({missing} of 72 probes missing)")
        if part["geometry"]["family"] == "turned" and part["geometry"].get("minimum_diameter_px", 1000) <= 3:
            g = part["geometry"]
            pose = g["pose_angle_rad"]
            hits = 0
            for i in range(128):
                x_world = g["length_px"] * (-0.45 + 0.9 * i / 127)
                x = round(part["resolution"] / 2 + x_world * math.cos(pose))
                y = round(part["resolution"] / 2 - x_world * math.sin(pose))
                hits += any(mask.getpixel((xx, yy)) for xx in range(x - 1, x + 2)
                            for yy in range(y - 1, y + 2))
            if hits < 115:
                raise ValueError(f"thin turned body is too fragmented: {pid} ({hits}/128 axis samples)")
        _details(part, mask, root / "details" / (pid + ".png"))
        family = part["geometry"]["family"]
        if family == "turned" and part["geometry"].get("minimum_diameter_px", 1000) < 16:
            path = root / "previews" / (pid + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            _thin_preview(mask, 256).save(path, optimize=True)
        by_family[family].append((part, mask))
        records.append((part, result))
    for family, entries in by_family.items():
        if not entries:
            continue
        cols, size, label = 6, 200, 27
        rows = math.ceil(len(entries) / cols)
        sheet = Image.new("RGB", (cols * size, rows * (size + label)), "#e7ebef")
        draw = ImageDraw.Draw(sheet)
        for i, (part, mask) in enumerate(entries):
            if family == "turned" and part["geometry"].get("minimum_diameter_px", 1000) < 16:
                image = _thin_preview(mask, size - 12)
            else:
                image = ImageOps.invert(mask.convert("L"))
                image.thumbnail((size - 12, size - 12), Image.Resampling.LANCZOS)
            x, y = (i % cols) * size, (i // cols) * (size + label)
            sheet.paste(image.convert("RGB"), (x + (size - image.width) // 2, y + (size - image.height) // 2))
            draw.text((x + 8, y + size), part["part_id"], fill="#202934")
            if family == "turned":
                draw.text((x + 8, y + size + 12),
                          f"min diameter {part['geometry']['minimum_diameter_px']:g} px", fill="#52606d")
        path = root / "contact_sheets" / (family + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path, optimize=True)
    cards = []
    feature_counts = Counter()
    for part, result in records:
        pid = part["part_id"]
        g = part["geometry"]
        family = g["family"]
        counts = Counter()
        for key in ("features", "edge_segments", "thread_regions", "axial_features", "cuts", "rim_features"):
            counts[key] = len(g.get(key, []))
            for feature in g.get(key, []):
                feature_counts[f"{key}:{feature.get('kind', feature.get('profile', 'unknown'))}"] += 1
        summary = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items() if v)
        if family == "turned":
            summary = f"minimum diameter {g['minimum_diameter_px']:g} px · " + summary
        geometry_text = html.escape(json.dumps(g, indent=2))
        thin = family == "turned" and g["minimum_diameter_px"] < 16
        whole_src = f"previews/{pid}.png" if thin else f"masks/{pid}.png"
        whole_class = "whole enhanced" if thin else "whole"
        cards.append(f'<article class="card" data-family="{family}" data-id="{pid}">'
                     f'<a href="masks/{pid}.png" target="_blank"><img class="{whole_class}" loading="lazy" src="{whole_src}" alt="{pid} full object mask"></a>'
                     f'<div class="name">{pid}</div><div class="meta">{html.escape(summary)}</div>'
                     f'<a href="details/{pid}.png" target="_blank"><img class="details" loading="lazy" src="details/{pid}.png" alt="{pid} local feature crops"></a>'
                     f'<div class="stats">{result["foreground_fraction"]:.1%} foreground · {result["elapsed_seconds"]:.1f}s</div>'
                     f'<details><summary>Geometry parameters</summary><pre>{geometry_text}</pre></details></article>')
    sheet_links = " ".join(f'<a href="contact_sheets/{family}.png">{family} sheet</a>'
                           for family, entries in by_family.items() if entries)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Base part review</title>
<style>body{{font:15px system-ui,sans-serif;margin:0;background:#f3f5f7;color:#1d2630}}header{{position:sticky;top:0;background:#fff;z-index:2;padding:18px 24px;box-shadow:0 2px 10px #0002}}h1{{margin:0 0 8px;font-size:22px}}nav{{display:flex;gap:10px;flex-wrap:wrap;align-items:center}}button,input{{font:inherit;padding:6px 11px;border:1px solid #aab4bf;border-radius:5px;background:white}}button.active{{background:#1c334b;color:white}}input{{min-width:180px}}.sheets{{margin-top:8px;display:flex;gap:14px;flex-wrap:wrap}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px;padding:20px}}.card{{background:white;border-radius:8px;padding:12px;box-shadow:0 1px 5px #0002}}.whole{{display:block;width:100%;height:245px;object-fit:contain;filter:invert(1);background:#fff}}.whole.enhanced{{filter:none;image-rendering:pixelated}}.name{{font-weight:700;margin:8px 0 4px}}.meta,.stats{{font-size:12px;color:#52606d}}.details{{width:100%;height:auto;margin-top:10px;image-rendering:pixelated}}details{{margin-top:9px}}pre{{max-height:320px;overflow:auto;padding:8px;background:#f4f6f8;font-size:11px}}a{{color:#23558b}}</style></head><body><header><h1>{len(parts)} base parts · {document['resolution']} × {document['resolution']} binary object masks</h1><nav><button class="active" data-filter="all">All</button>'''
    page += "".join(f'<button data-filter="{family}">{family}</button>' for family in FAMILIES)
    page += f'<input id="search" placeholder="Search part ID"><span id="visible">{len(records)} shown</span></nav><div class="sheets">{sheet_links} <a href="catalog.json">geometry catalog JSON</a></div></header><main class="grid">' + "".join(cards) + '''</main><script>const cards=[...document.querySelectorAll('.card')],buttons=[...document.querySelectorAll('button[data-filter]')],search=document.querySelector('#search');let filter='all';function update(){let n=0;for(const card of cards){const show=(filter==='all'||card.dataset.family===filter)&&card.dataset.id.includes(search.value.trim().toLowerCase());card.hidden=!show;if(show)n++}document.querySelector('#visible').textContent=n+' shown'}for(const button of buttons)button.onclick=()=>{filter=button.dataset.filter;buttons.forEach(b=>b.classList.toggle('active',b===button));update()};search.oninput=update;</script></body></html>'''
    (root / "index.html").write_text(page)
    turned_diameters = [part["geometry"]["minimum_diameter_px"] for part, _ in records
                        if part["geometry"]["family"] == "turned"]
    diameter_ranges = ((0, 2), (2, 5), (5, 10), (10, 25),
                       (25, 50), (50, 100), (100, 250), (250, float("inf")))
    diameter_counts = {f"{low:g}-{high:g}px" if math.isfinite(high) else f"{low:g}px+":
                       sum(low <= value < high for value in turned_diameters)
                       for low, high in diameter_ranges}
    report = {"expected_parts": len(parts), "rendered_parts": len(records),
              "family_counts": {family: len(entries) for family, entries in by_family.items()},
              "resolution": document["resolution"], "blender_version": BLENDER_VERSION,
              "catalog_sha256": document["catalog_sha256"],
              "feature_counts": dict(sorted(feature_counts.items())),
              "turned_minimum_diameter_px_range": [min(turned_diameters), max(turned_diameters)] if turned_diameters else [],
              "turned_diameter_bucket_counts": diameter_counts,
              "render_seconds_total": round(sum(result["elapsed_seconds"] for _, result in records), 2),
              "foreground_fraction_range": [min((r["foreground_fraction"] for _, r in records), default=0),
                                            max((r["foreground_fraction"] for _, r in records), default=0)],
              "total_mask_bytes": sum((root / "masks" / (p["part_id"] + ".png")).stat().st_size for p, _ in records)}
    (root / "review_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("plan", "render", "gallery", "generate"):
        p = sub.add_parser(name)
        p.add_argument("output")
        if name in ("plan", "generate"):
            p.add_argument("--seed", type=int, default=731029)
            p.add_argument("--count", type=int, default=300)
            p.add_argument("--resolution", type=int, default=2048)
        if name in ("render", "generate"):
            p.add_argument("--blender", default=BLENDER)
            p.add_argument("--start", type=int, default=0)
            p.add_argument("--stop", type=int)
            p.add_argument("--force", action="store_true")
        if name == "gallery":
            p.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    if args.action in ("plan", "generate"):
        plan(args.output, args.seed, args.count, args.resolution)
    if args.action in ("render", "generate"):
        render(args.output, args.blender, args.start, args.stop, args.force)
    if args.action in ("gallery", "generate"):
        print(json.dumps(build_gallery(args.output, require_complete=not getattr(args, "allow_partial", False)), indent=2))


if __name__ == "__main__":
    main()
