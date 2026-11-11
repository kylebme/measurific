"""Add selection-error diagnostics and paired support images to a completed batch."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def paired_preview(root, entry, output):
    directory = root/'samples'/entry['sample_id']
    predicted = load(directory/'results.json.gz')
    reference = load(directory/'reference-results.json.gz')
    a = next(f for f in predicted['features'] if f['id'] == entry['feature_id'])
    b = next(f for f in reference['features'] if f['id'] == entry['reference_id'])
    points = np.concatenate([np.asarray(q) for f in (a, b) for q in f['support']])
    image = Image.open(directory/'input.png').convert('RGB')
    low, high = points.min(0), points.max(0)
    extent = np.maximum(high-low, 80)
    center = (low+high)/2
    low = np.maximum(np.floor(center-extent*.65), 0)
    high = np.minimum(np.ceil(center+extent*.65), image.size)
    crop = image.crop((*low.astype(int), *high.astype(int)))
    scale = min(520/crop.width, 480/crop.height)
    crop = crop.resize((round(crop.width*scale), round(crop.height*scale)), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (1120, 620), '#111820')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    draw.text((16, 12), f"{entry['sample_id']}  {entry['family']}  {entry['operator']}", font=font, fill='white')
    draw.text((16, 38), f"Support IoU {entry['iou']:.5f} | absolute measurement difference {entry['absolute_difference']:.6g} {entry['units']}", font=font, fill='white')
    for i, (feature, label, value, color) in enumerate([(a, 'Model selection', entry['predicted_value'], '#00c2ff'),
                                                       (b, 'Reference selection', entry['reference_value'], '#ffb347')]):
        origin = np.array([16+i*560, 104])
        canvas.paste(crop, tuple(origin))
        draw.text((origin[0], 76), f"{label}: {value:.6g} {entry['units']}", font=font, fill=color)
        for q, closed in zip(feature['support'], feature['closed']):
            q = np.asarray(q)
            if closed:
                q = np.vstack([q, q[0]])
            transformed = (q-low)*scale+origin
            draw.line([tuple(p) for p in transformed], width=2, fill=color)
        geometry = feature['geometry']
        if 'endpoints' in geometry:
            q = (np.asarray(geometry['endpoints'])-low)*scale+origin
            draw.line([tuple(p) for p in q], width=1, fill='#ec5252')
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.output
    records = [json.loads(x) for x in (root/'samples.jsonl').read_text().splitlines()]
    comparisons = []
    oracle = Counter()
    timings = defaultdict(list)
    for r in records:
        if r['status'] != 'completed':
            continue
        oracle.update(r['reference']['oracle_feature_status'])
        for key in ('inference_seconds', 'measurement_seconds', 'elapsed_seconds'):
            stage = 'prediction_load_seconds' if key == 'inference_seconds' and r.get('prediction_reused') else key
            timings[stage].append(r[key])
        for c in r['reference']['comparisons']:
            if 'absolute_difference' in c:
                comparisons.append({**c, 'sample_id': r['id'], 'family': r['family']})
    stats = {}
    for threshold in (.5, .9, 1-1e-9):
        by_operator = defaultdict(list)
        for c in comparisons:
            if c['iou'] >= threshold:
                by_operator[c['operator']].append(c['absolute_difference'])
        stats[str(threshold)] = {op: {'count': len(v), 'median': float(np.median(v)),
                                     'p95': float(np.quantile(v, .95)), 'maximum': float(max(v))}
                                for op, v in sorted(by_operator.items())}
    worst = {}
    for c in comparisons:
        if c['operator'] not in worst or c['absolute_difference'] > worst[c['operator']]['absolute_difference']:
            worst[c['operator']] = c
    directory = root/'outliers'
    directory.mkdir(exist_ok=True)
    for op, c in worst.items():
        paired_preview(root, c, directory/f'{op}.png')
    result = {'oracle_feature_status': dict(oracle), 'comparison_by_minimum_iou': stats,
              'worst_by_operator': worst,
              'timings': {k: {'median': float(np.median(v)), 'p95': float(np.quantile(v, .95)),
                              'total': float(sum(v))} for k, v in timings.items()}}
    (root/'diagnostics.json').write_text(json.dumps(result, indent=2)+'\n')
    text = ['# Selection and fitting diagnostics', '',
            'These are differences between fits on model-selected and reference-selected actual points, not errors against calibrated ground truth.', '',
            f'Reference-selection feature statuses: {dict(oracle)}', '',
            '## Largest differences', '', '| Operator | Sample | Family | IoU | Model value | Reference value | Absolute difference |',
            '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for op, c in sorted(worst.items()):
        text.append(f"| [{op}](outliers/{op}.png) | {c['sample_id']} | {c['family']} | {c['iou']:.5f} | {c['predicted_value']:.6g} | {c['reference_value']:.6g} | {c['absolute_difference']:.6g} {c['units']} |")
    text += ['', 'A `fitted` status reports a successful numeric fit; it does not certify that the selection matches the intended physical surface. Line RMS acceptance limits are opt-in through `max_rms_residual`. IoU may remain high when a few erroneous points strongly change form measurements.', '',
             'For repeated profiles, default `length` is observed polyline length across selected runs; it is not thread pitch or axial threaded length. An incorrect association of multiple repeated regions can cause a large length difference.', '',
             '`diagnostics.json` separates comparisons at IoU >= 0.5, >= 0.9, and effectively identical support, and records per-stage timing. Paired outlier images use the same crop and scale; cyan is model support, orange reference support, red fitted line.']
    (root/'DIAGNOSTICS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'oracle_feature_status': dict(oracle), 'timings': result['timings']}))


if __name__ == '__main__':
    main()
