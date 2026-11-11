#!/usr/bin/env python3
"""Locate bidirectional worst errors and render raw/truth/before/after crops."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely import distance, points
from shapely.geometry import LineString, Point, box as geometry_box
from shapely.ops import nearest_points, unary_union

from evaluate_extraction_degradation import densify, paths

CASES = ['rack-d2-focused', 'threaded_shaft-d1-blurred',
         'polygon_spacer-d0-blurred', 'threaded_shaft-d2-focused',
         'grooved_pin-d0-blurred', 'rack-d0-blurred',
         'slotted_bracket-d0-blurred', 'stepped_shaft-d0-blurred',
         'flange-d2-focused', 'toothed_disk-d1-blurred']


def render(source, development, heldout, output):
    output.mkdir(parents=True, exist_ok=True)
    cases = {x['id']: x for x in json.loads((source/'manifest.json').read_text())['cases']}
    records = {}
    for run in [development, heldout]:
        records.update({x['id']: (x, run) for x in json.loads((run/'results.json').read_text())})
    font = ImageFont.load_default(size=16)
    small = ImageFont.load_default(size=13)
    entries, sections = [], []
    for stem in CASES:
        record, run = records[stem]
        case = cases[stem]
        image = Image.open(source/case['image']).convert('RGB')
        with np.load(source/case['truth']) as t:
            truth = paths(t['actual_xy_px'], t['ring_offsets'])
        def load(suffix):
            obj = json.loads((run/(stem+suffix)).read_text())
            return paths(np.asarray(obj['xy']), obj['offsets'])
        before, after = load('.original.json'), load('.contours.json')
        worst = (-1., None, None, None)
        for ring in record['result']['rings']:
            actual, expected = after[ring['observed_ring']], truth[ring['truth_ring']]
            for a, b, direction in [(actual, expected, 'recovered-to-truth'), (expected, actual, 'truth-to-recovered')]:
                dense = densify(a, .025)
                line = LineString(np.vstack([b, b[:1]]))
                errors = distance(points(dense), line)
                index = errors.argmax()
                if errors[index] > worst[0]:
                    query = dense[index]
                    nearest = np.array(nearest_points(Point(query), line)[1].coords[0])
                    worst = (float(errors[index]), query, nearest, direction)
        error, query, nearest, direction = worst
        center = (query+nearest)/2
        x, y = np.rint(center).astype(int)
        box = (x-16, y-16, x+16, y+16)
        def clipped(rings):
            return unary_union([LineString(np.vstack([q, q[:1]])) for q in rings]).intersection(geometry_box(*box))
        unchanged = clipped(before).equals(clipped(after))
        size, zoom = 288, 9
        raw = image.crop(box).resize((size, size), Image.Resampling.NEAREST)
        sheet = Image.new('RGB', (1200, 384), '#141b22')
        d = ImageDraw.Draw(sheet)
        d.text((12, 8), stem, font=font, fill='white')
        d.text((12, 31), f"Full-outline maximum: {record['baseline']['hausdorff_px']:.3f} -> {record['result']['hausdorff_px']:.3f} px | sigma {case['camera']['sigma_px']:.2f} px",font=small,fill='#dddddd')
        labels = ['Raw pixels', 'Ground truth', 'PAE + truth', 'Recovery: unchanged' if unchanged else 'Recovery + truth']
        def mapped(p):
            return ((float(p[0])-box[0]+.5)*zoom-.5, (float(p[1])-box[1]+.5)*zoom-.5)
        for col, label in enumerate(labels):
            panel = raw.copy()
            draw = ImageDraw.Draw(panel)
            layers = [] if col == 0 else [(truth, '#21e6de')]
            if col == 2: layers.append((before, '#ffad46'))
            if col == 3: layers.append((after, '#ff4fc8'))
            for rings, color in layers:
                for q in rings:
                    p = [mapped(v) for v in q]
                    draw.line(p+p[:1], fill=color, width=2)
            if col == 3:
                draw.line([mapped(query), mapped(nearest)], fill='#ffff66', width=3)
                for p in [query, nearest]:
                    xx, yy = mapped(p)
                    draw.ellipse((xx-3, yy-3, xx+3, yy+3), outline='#ffff66', width=1)
            left = 12+col*298
            d.text((left, 58), label, font=small, fill='white')
            sheet.paste(panel, (left, 80))
        d.text((12, 369), '32 x 32 source pixels, enlarged 9x; cyan truth / orange PAE / magenta recovery / yellow worst-distance segment', font=small, fill='#dddddd')
        sheet.save(output/(stem+'.png'))
        context = image.copy()
        draw = ImageDraw.Draw(context)
        draw.rectangle(box, outline='#ff3030', width=3)
        context.thumbnail((480, 360))
        context.save(output/(stem+'-context.png'))
        recovery = record.get('recovery') or {}
        candidates = recovery.get('accepted', [])+recovery.get('rejected', [])
        nearest_candidate = min(candidates, key=lambda a: np.linalg.norm(np.array(a.get('original_xy', a.get('xy')))-center)) if candidates else None
        entry = dict(id=stem, crop_exactly_unchanged=unchanged, baseline_max_px=record['baseline']['hausdorff_px'],
                     recovered_max_px=record['result']['hausdorff_px'],
                     refined_max_px=error, direction=direction,
                     query_xy=query.tolist(), nearest_xy=nearest.tolist(), crop_box=[int(v) for v in box],
                     sigma_px=case['camera']['sigma_px'], nearest_candidate=nearest_candidate)
        entries.append(entry)
        reason = nearest_candidate['reason'] if nearest_candidate else 'no corner candidate'
        crop_note = 'The recovered contour in this crop exactly equals PAE.' if unchanged else 'Recovery changes part of this crop; the maximum-error feature remains unrecovered.'
        sections.append(f'<section><h2>{html.escape(stem)}</h2><p>{crop_note} Maximum {entry["baseline_max_px"]:.3f} → {entry["recovered_max_px"]:.3f} px. Nearest corner proposal: <code>{html.escape(reason)}</code> (not necessarily the source of the maximum).</p><a href="{stem}.png"><img src="{stem}.png" alt="Raw, truth, baseline and recovered crop"></a><details><summary>Show location in full image</summary><img class="context" src="{stem}-context.png"></details></section>')
    (output/'locations.json').write_text(json.dumps(entries, indent=2)+'\n')
    page = '''<!doctype html><meta charset="utf-8"><title>Problematic silhouette crops</title>
<style>body{background:#10151b;color:#eee;font:16px system-ui;max-width:1240px;margin:32px auto;padding:0 16px}h1{font-size:28px}h2{font-size:20px}p{line-height:1.5;color:#d1d5db}section{margin:40px 0}img{width:100%;height:auto}.context{max-width:480px}summary{cursor:pointer;margin:12px 0}a{color:#5ddddd}code{color:#ffdb99}</style>
<h1>Where the full-outline maximum comes from</h1>
<p>Ten problematic cases selected from the existing frozen benchmark, including the worst development and held-out cases. Crops are centered on the largest bidirectional distance, not a hand-selected attractive corner. Cyan: actual truth; orange: original PAE; magenta: recovery. The yellow segment marks the worst-distance pair. Raw crops use nearest-neighbor enlargement so no display smoothing hides pixels.</p>
<p>Each crop spans 32 × 32 source pixels. Numbers are <b>pixels, not millimeters</b>. Before/after values use the original benchmark; crop localization uses denser 0.025 px samples (maximum undershoot bound 0.0125 px). Truth includes generated form errors. A true small notch can therefore dominate the maximum even when most of the contour agrees well. No algorithm changes or rerun of fitting were made.</p>
'''+'\n'.join(sections)
    (output/'index.html').write_text(page)
    print(json.dumps([{k:e[k] for k in ['id','refined_max_px','query_xy']} for e in entries], indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--development', type=Path, required=True)
    p.add_argument('--heldout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    render(a.input,a.development,a.heldout,a.output)
