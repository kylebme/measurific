"""Label-independent coverage/error/runtime audit and browsable raw-edge previews.

Reads the existing corpus; never generates or edits synthetic examples. No F1 is
computed against construction labels, which are not an approximation oracle.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from html import escape
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from contour_model.geometric import predict, GeometricConfig

COLORS = {'line': '#168aad', 'arc': '#ef8354', 'circle': '#8c5ec0'}


def run(job):
    source, entry, output, tolerance = job
    path = Path(source)/Path(entry['file']).parent/'sources'/(entry['id']+'.contours.npz')
    with np.load(path, allow_pickle=False) as a:
        raw = {k: a[k] for k in ('actual_xy_mm', 'ring_offsets')}
    q, offsets = raw['actual_xy_mm'], raw['ring_offsets']
    started = time.perf_counter()
    result, _ = predict(raw, units='mm', config=GeometricConfig(tolerance=tolerance))
    elapsed = time.perf_counter()-started
    selected = [i for f in result['features'] for i in f['raw_edge_indices']]
    if sorted(selected) != list(range(len(q))):
        raise AssertionError('Missing or overlapping edge ownership')
    maximum = max(f['fit_evidence']['max_error'] for f in result['features'])
    if maximum > tolerance*1.000001:
        raise AssertionError('Accepted approximation exceeds tolerance')
    # Recompute residual bounds from the returned diagnostic geometry, rather
    # than trusting only the recognizer's own max_error field.
    for f in result['features']:
        ids = np.asarray(f['raw_edge_indices'])
        ri = f['contour_spans'][0]['ring_index']
        lo, hi = offsets[ri:ri+2]
        end = lo+(ids[-1]-lo+1) % (hi-lo)
        points = q[np.r_[ids, end]]
        evidence = f['fit_evidence']
        if f['semantic'] == 'line':
            axis = np.asarray(evidence['direction'])
            error = np.abs((points-evidence['origin'])@np.array([-axis[1], axis[0]])).max()
        else:
            z = points-evidence['center']
            edge = np.diff(z, axis=0)
            nearest = np.clip(-np.einsum('ij,ij->i', z[:-1], edge)/np.einsum('ij,ij->i', edge, edge), 0, 1)
            radial = np.linalg.norm(np.vstack([z, z[:-1]+nearest[:, None]*edge]), axis=1)
            error = np.abs(radial-evidence['radius']).max()
        if error > tolerance*1.00001:
            raise AssertionError('Independent residual check failed')
    row = dict(id=entry['id'], family=entry['family'], split=entry['split'],
               points=len(q), rings=len(offsets)-1, seconds=elapsed,
               primitives=dict(Counter(f['semantic'] for f in result['features'])),
               compounds=dict(Counter(f['semantic'] for f in result['compound_features'])),
               max_error=maximum, exact_edge_partition=True)
    output = Path(output)
    (output/(entry['id']+'.json')).write_text(json.dumps(result, indent=2, allow_nan=False))
    extent = np.ptp(q, axis=0)
    factor = min(560/max(extent[0], 1e-9), 260/max(extent[1], 1e-9))
    screen = (q-(q.min(0)+q.max(0))/2)*[factor, -factor]+[300, 178]
    im = Image.new('RGB', (600, 360), 'white')
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 15)
    except OSError:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 15)
    draw.text((15, 10), entry['id']+'  '+entry['family'], fill='#1e293b', font=font)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 360">', '<rect width="600" height="360" fill="white"/>']
    for f in result['features']:
        ids = np.asarray(f['raw_edge_indices'])
        ri = f['contour_spans'][0]['ring_index']
        lo, hi = offsets[ri:ri+2]
        end = lo+(ids[-1]-lo+1) % (hi-lo)
        support = screen[np.r_[ids, end]]
        points = ' '.join(f'{x:.3f},{y:.3f}' for x, y in support)
        color = COLORS[f['semantic']]
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"><title>{f["id"]}: {f["semantic"]}; max error {f["fit_evidence"]["max_error"]:.5g} mm</title></polyline>')
        draw.line([tuple(v) for v in support], fill=color, width=2)
        if f['semantic'] != 'circle':
            x, y = support[0]
            svg.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="2" fill="#1e293b"/>')
            draw.ellipse((x-2, y-2, x+2, y+2), fill='#1e293b')
    summary = ' / '.join(f'{n} {kind}' for kind, n in row['primitives'].items())
    draw.text((15, 326), summary+f'  |  {elapsed:.2f}s', fill='#475569', font=font)
    im.save(output/(entry['id']+'.png'))
    svg.append('</svg>')
    (output/(entry['id']+'.svg')).write_text(''.join(svg))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--tolerance', type=float, default=.01)
    args = parser.parse_args()
    index = json.loads((args.source/'index.json').read_text())
    families = {}
    for e in index:
        if e['split'] == 'test':
            families.setdefault(e['family'], e)
    entries = list(families.values())[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run, [(str(args.source), e, str(args.output), args.tolerance) for e in entries]):
            rows.append(row)
            print(f'{len(rows)}/{len(entries)} {row["family"]} {row["seconds"]:.2f}s', flush=True)
    times = [r['seconds'] for r in rows]
    summary = dict(samples=len(rows), families=len(rows), tolerance_mm=args.tolerance,
                   all_edges_partitioned=True, all_fits_within_tolerance=True,
                   timing=dict(median_seconds=float(np.median(times)), p95_seconds=float(np.percentile(times, 95)),
                               max_seconds=max(times), parallel_workers=args.workers),
                   note='One existing test design per family; no construction-label accuracy claim. '
                        'Recognition-only timing, with concurrent CPU jobs; not a real-camera generalization benchmark.',
                   rows=rows)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    cards = []
    for row in rows:
        ident = row['id']
        counts = escape(str(row['primitives']))
        cards.append(f'<article><h3>{escape(row["family"])} · {ident}</h3><a href="{ident}.svg"><img src="{ident}.png"></a>'
                     f'<p>{counts} · {row["seconds"]:.2f}s · max error {row["max_error"]:.5f} mm</p>'
                     f'<p>Compounds: {escape(str(row["compounds"]))}</p><a href="{ident}.json">Fit evidence / raw-edge selections</a></article>')
    html = '<!doctype html><meta charset="utf-8"><title>Geometric contour review</title><style>body{font:15px system-ui;background:#f1f5f9;color:#1e293b;margin:30px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:20px}article{background:white;padding:16px;border-radius:10px}img{width:100%}h3{margin:0}p{color:#475569}</style>'
    html += f'<h1>Training-free contour recognition</h1><p>{len(rows)} existing test designs, one per family. Tolerance {args.tolerance} mm. '
    html += 'Blue = line; orange = arc; purple = circle; dots = primitive boundaries. Curves shown are the original observations, colored by prediction.</p>'
    html += '<p>No generator changes or training labels used for recognition. Strict error checks do not establish correct manufacturing intent. '
    html += 'Thread rules deliberately reject aligned opposite-side repeats; damaged surfaces may split. <a href="summary.json">Audit summary</a></p><main>'+''.join(cards)+'</main>'
    (args.output/'index.html').write_text(html)
    sheet = Image.new('RGB', (1800, 360*min(4, (len(rows)+2)//3)), 'white')
    for i, row in enumerate(rows[:12]):
        with Image.open(args.output/(row['id']+'.png')) as im:
            sheet.paste(im, ((i % 3)*600, (i//3)*360))
    sheet.save(args.output/'overview.png')
    print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
