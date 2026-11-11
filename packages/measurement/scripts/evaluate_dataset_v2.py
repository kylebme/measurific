"""Reproducible held-out contour/model/measurement batch (no PAE image extraction).

Run with the contour-model Python environment and this repository's three src
folders on PYTHONPATH. All source data/checkpoint files are read-only.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import html
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch

from contour_model.train import load_model
from contour_model.inference import predict, describe_selection, ROUTES
from contour_model.validation import indexed_records, observable_reference_labels
from contour_model.data import NAMES
from inspection_measurement import Contours, evaluate, annotate
from inspection_measurement.engine import normalize
from inspection_shapes.geometry import Part, Ring
from inspection_shapes.render import Camera, render


def write_json(path, value):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'wt') as stream:
            json.dump(value, stream, allow_nan=False, separators=(',', ':'))
    else:
        path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def select_entries(index, split, count, seed):
    groups = defaultdict(list)
    for entry in index:
        if entry['split'] == split:
            groups[entry['family']].append(entry)
    rng = np.random.default_rng(seed)
    pools = {family: [items[i] for i in rng.permutation(len(items))]
             for family, items in sorted(groups.items())}
    chosen = []
    while len(chosen) < count:
        changed = False
        for family in pools:
            if pools[family]:
                chosen.append(pools[family].pop())
                changed = True
                if len(chosen) == count:
                    break
        if not changed:
            raise ValueError('Requested count exceeds split size')
    return sorted(chosen, key=lambda e: e['id'])


def projection(c):
    xy = c['actual_xy_mm']
    extent = np.ptp(xy, axis=0)
    scale = float(min(900/extent[0], 640/extent[1]))
    middle = (xy.min(0)+xy.max(0))/2
    return Camera(width=1024, height=768, supersample=2, pixels_per_mm=scale,
                  translation_x=float(512-middle[0]*scale),
                  translation_y=float(384+middle[1]*scale),
                  sigma_px=.8, read_noise_dn=.25, prnu=0, electrons_per_dn=1000)


def render_source(meta, c, camera):
    # V2 omits primitive IDs. Zero IDs are sufficient for unused renderer labels;
    # all intensity geometry and feature/ring identity come from the saved record.
    part = Part(meta['family'], features=deepcopy(meta['features']))
    for i, (lo, hi) in enumerate(zip(c['ring_offsets'][:-1], c['ring_offsets'][1:])):
        zero = np.zeros(hi-lo, dtype=int)
        part.rings.append(Ring(c['nominal_xy_mm'][lo:hi], c['edge_feature_id'][lo:hi],
                               zero, i > 0, c['actual_xy_mm'][lo:hi],
                               c['edge_anomaly_class'][lo:hi], zero))
    return render(part, camera, int(meta['seed']))[0]['image']


def extra_requests(prediction, contours):
    req = normalize(prediction, contours)
    accepted = [f for f in req['features'] if f.get('status') == 'accepted']
    lines = [f['id'] for f in accepted if f['type'] == 'line']
    circles = [f['id'] for f in accepted if f['type'] == 'circle']
    extras = []
    if len(lines) >= 2:
        extras.extend([('first_line_pair.angle', 'angle', lines[:2]),
                       ('first_line_pair.separation', 'minimum_separation', lines[:2])])
    if len(circles) >= 2:
        extras.append(('first_circle_pair.center_distance', 'center_distance', circles[:2]))
    if lines and circles:
        extras.append(('first_line_circle.distance', 'distance', [lines[0], circles[0]]))
    req['measurements'].extend({'id': ident, 'operator': op, 'feature_refs': refs}
                               for ident, op, refs in extras)
    return req


def reference_comparison(meta, c, contours, prediction, report):
    ids, classes, _ = observable_reference_labels(meta, c)
    reference = []
    for ident in np.unique(ids):
        edges = np.flatnonzero(ids == ident)
        cls = int(classes[edges[0]])
        reference.append({'id': f'ref_{ident}', 'semantic_id': cls, 'semantic': NAMES[cls],
                          'raw_edge_indices': edges.tolist(),
                          'contour_spans': describe_selection(edges, contours.model_arrays()),
                          'status': 'accepted', 'confidence': 1., 'units': 'px',
                          'measurement_route': ROUTES.get(cls, 'profile')})
    oracle = evaluate(contours, {'units': 'px', 'features': reference})
    predicted = [f for f in prediction['features'] if f['status'] == 'accepted' and f.get('grouping') != 'semantic_parent']
    weights = np.empty(len(ids))
    for lo, hi in zip(contours.offsets[:-1], contours.offsets[1:]):
        q = contours.xy[lo:hi]
        weights[lo:hi] = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
    owner = np.empty(len(ids), dtype=int)
    for i, f in enumerate(reference):
        owner[f['raw_edge_indices']] = i
    truth_length = np.bincount(owner, weights=weights, minlength=len(reference))
    intersection = np.zeros((len(predicted), len(reference)))
    length = np.zeros(len(predicted))
    for i, f in enumerate(predicted):
        edges = f['raw_edge_indices']
        intersection[i] = np.bincount(owner[edges], weights=weights[edges], minlength=len(reference))
        length[i] = weights[edges].sum()
    iou = intersection/np.maximum(length[:, None]+truth_length[None]-intersection, 1e-12)
    if len(predicted):
        iou *= np.array([f['semantic_id'] for f in predicted])[:, None] == np.array([f['semantic_id'] for f in reference])[None]
    rows, cols = linear_sum_assignment(-iou)
    pred_values = {(m['feature_refs'][0], m['operator']): m for m in report['measurements'] if len(m['feature_refs']) == 1}
    ref_values = {(m['feature_refs'][0], m['operator']): m for m in oracle['measurements'] if len(m['feature_refs']) == 1}
    comparisons = []
    supported = {'length', 'least_squares_diameter', 'radius', 'arc_length', 'width', 'overall_length',
                 'straightness', 'circularity_ls', 'rms_residual', 'orientation'}
    for row, col in zip(rows, cols):
        if iou[row, col] < .5:
            continue
        pf, rf = predicted[row], reference[col]
        for op in supported:
            a, b = pred_values.get((pf['id'], op)), ref_values.get((rf['id'], op))
            if not a or not b:
                continue
            comparison = {'feature_id': pf['id'], 'reference_id': rf['id'], 'semantic': pf['semantic'],
                          'operator': op, 'iou': float(iou[row, col]),
                          'predicted_status': a['status'], 'reference_status': b['status']}
            if a['status'] == b['status'] == 'measured':
                difference = abs(a['value']-b['value'])
                if op == 'orientation':
                    difference = min(difference % 180, 180-difference % 180)
                comparison.update(absolute_difference=float(difference), units=a['units'],
                                  predicted_value=a['value'], reference_value=b['value'])
            comparisons.append(comparison)
    return {'reference_features': len(reference), 'accepted_leaf_features': len(predicted),
            'matched_iou50': int(np.sum(iou[rows, cols] >= .5)),
            'matched_iou90': int(np.sum(iou[rows, cols] >= .9)),
            'oracle_feature_status': dict(Counter(f['status'] for f in oracle['features'])),
            'comparisons': comparisons}, oracle


def preview(image, report):
    # Full measurements are retained in reports; keep each preview readable.
    failed = [m for m in report['measurements'] if m['status'] != 'measured']
    candidates = failed[:3]+report['measurements']
    chosen, seen = [], set()
    for m in candidates:
        if m['id'] not in seen:
            chosen.append(m)
            seen.add(m['id'])
        if len(chosen) == 12:
            break
    refs = {r for m in chosen for r in m['feature_refs']}
    small = {**report, 'features': [f for f in report['features'] if f['id'] in refs], 'measurements': chosen}
    return annotate(image, small)


def aggregate(output, manifest):
    records = [json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
    feature_status, measurement_status, reasons, by_operator, families = Counter(), Counter(), Counter(), {}, {}
    differences = defaultdict(list)
    reference_features = accepted = matched50 = matched90 = 0
    for rec in records:
        family = families.setdefault(rec['family'], {'samples': 0, 'sample_failures': 0, 'measurements': 0, 'measured': 0, 'feature_failures': 0})
        family['samples'] += 1
        if rec['status'] != 'completed':
            family['sample_failures'] += 1
            reasons['sample: '+rec['reason']] += 1
            continue
        feature_status.update(rec['feature_status'])
        measurement_status.update(rec['measurement_status'])
        reasons.update(rec['feature_failure_reasons'])
        family['measurements'] += sum(rec['measurement_status'].values())
        family['measured'] += rec['measurement_status'].get('measured', 0)
        family['feature_failures'] += rec['feature_status'].get('fit_failed', 0)
        for op, counts in rec['by_operator'].items():
            by_operator.setdefault(op, Counter()).update(counts)
        r = rec['reference']
        reference_features += r['reference_features']
        accepted += r['accepted_leaf_features']
        matched50 += r['matched_iou50']
        matched90 += r['matched_iou90']
        for item in r['comparisons']:
            if 'absolute_difference' in item:
                differences[item['operator']].append(item['absolute_difference'])
    summary = {'protocol': manifest, 'samples_attempted': len(records),
               'samples_completed': sum(r['status'] == 'completed' for r in records),
               'samples_with_feature_failures': sum(r.get('feature_status', {}).get('fit_failed', 0) > 0 for r in records),
               'feature_status': dict(feature_status), 'measurement_status': dict(measurement_status),
               'feature_failure_reasons': dict(reasons),
               'by_operator': {k: dict(v) for k, v in sorted(by_operator.items())},
               'families': families,
               'selection_reference': {'accepted_leaves': accepted, 'reference_leaves': reference_features,
                                      'matched_iou50': matched50, 'matched_iou90': matched90,
                                      'precision_iou50': matched50/max(1, accepted),
                                      'recall_iou50': matched50/max(1, reference_features)},
               'matched_selection_differences': {k: {'count': len(v), 'median': float(np.median(v)),
                       'p95': float(np.quantile(v, .95)), 'maximum': float(max(v)),
                       'units': 'deg' if k == 'orientation' else 'px'} for k, v in sorted(differences.items())}}
    write_json(output/'summary.json', summary)
    text = ['# Dataset-v2 measurement batch', '',
            f"Completed {summary['samples_completed']} / {len(records)} selected held-out samples across {len(families)} families.", '',
            'This run projects saved observed contours into synthetic image pixels. It does not run PAE, estimate calibration, or infer dimensional tolerances.', '',
            f'Feature statuses: {dict(feature_status)}', '', f'Measurement statuses: {dict(measurement_status)}', '',
            '## Fit failure reasons', '', '| Reason | Count |', '| --- | ---: |']
    text += [f'| {k} | {v} |' for k, v in reasons.most_common()]
    text += ['', '## Measurement operators', '', '| Operator | Measured | Failed | Unavailable |', '| --- | ---: | ---: | ---: |']
    text += [f"| {k} | {v.get('measured',0)} | {v.get('failed',0)} | {v.get('unavailable',0)} |" for k, v in sorted(by_operator.items())]
    text += ['', '## Model-selection comparison', '',
             f'Same-semantic, arc-length-weighted, one-to-one IoU >= 0.5: {matched50} / {accepted} accepted model leaves matched {reference_features} reference leaves.', '',
             'Differences below compare this same measurement engine on model-selected versus reference-selected **actual** contours. They are selection sensitivity, not error against nominal manufactured dimensions.', '',
             '| Operator | Matched scalar pairs | Median difference | P95 | Maximum | Units |', '| --- | ---: | ---: | ---: | ---: | --- |']
    for op, v in summary['matched_selection_differences'].items():
        text.append(f"| {op} | {v['count']} | {v['median']:.6g} | {v['p95']:.6g} | {v['maximum']:.6g} | {v['units']} |")
    text += ['', '## Artifacts', '',
             '`index.html` contains all annotated previews (up to 12 measurements per preview); complete results are saved in each sample directory.', '',
             'Each sample contains compressed prediction/request/result/reference JSON, lossless pixel contours, rendered input and annotated preview. `selection.json` records all selected IDs before inference.', '',
             'Selection is family-balanced rather than frequency-weighted. No dimensions or nominal labels enter model inference or measured fits. Dataset references are used only after inference for comparisons. Default requests do not exercise every supported operator.']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n')
    cards = []
    for rec in records:
        name = f"{rec['id']} — {rec['family']}"
        if rec['status'] == 'completed':
            counts = rec['measurement_status']
            cards.append(f'<article><h2>{html.escape(name)}</h2><p>{html.escape(str(counts))}</p><a href="samples/{rec["id"]}/annotated.png"><img loading="lazy" src="samples/{rec["id"]}/annotated.png"></a></article>')
        else:
            cards.append(f'<article><h2>{html.escape(name)}</h2><p>{html.escape(rec["reason"])}</p></article>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Dataset-v2 measurement review</title><style>body{background:#111820;color:#ddd;font:16px sans-serif;margin:24px}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}img{width:100%}h2{font-size:16px}article{padding:12px;background:#1c2630}</style><h1>Dataset-v2: 300-sample measurement review</h1><p>Pixel projection of raw contours; up to 12 measurements shown per preview. Full results in compressed JSON.</p><main>'+''.join(cards)+'</main>')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/media/T7Shield/inspection-contour-training/v2-expanded'))
    parser.add_argument('--checkpoint', type=Path, default=Path('packages/contour-model/artifacts/release-v2/model.pt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=300)
    parser.add_argument('--split', default='test')
    parser.add_argument('--seed', type=int, default=20260907)
    parser.add_argument('--reuse', type=Path, help='Reuse predictions and input images from a prior matching run')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'samples.jsonl').exists():
        raise FileExistsError('Use a fresh output directory to avoid mixing batch results')
    entries = select_entries(json.loads((args.source/'index.json').read_text()), args.split, args.count, args.seed)
    manifest = {'source': str(args.source.resolve()), 'checkpoint': str(args.checkpoint.resolve()),
                'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                'split': args.split, 'seed': args.seed, 'count': args.count,
                'selection': 'seeded family-balanced round-robin, without replacement',
                'coordinate_policy': 'uniform Camera projection of actual mm observations into 1024x768 image pixels; no distortion',
                'inference_threshold': .7, 'pae_extraction': False,
                'measurement_source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path('packages/measurement/src/inspection_measurement').glob('*.py'))}}
    if args.reuse:
        previous = json.loads((args.reuse/'selection.json').read_text())
        if previous['entries'] != entries or previous['protocol']['checkpoint_sha256'] != manifest['checkpoint_sha256']:
            raise ValueError('Reuse run must have identical selected entries and checkpoint')
        manifest['reuse_predictions_from'] = str(args.reuse.resolve())
    write_json(args.output/'selection.json', {'protocol': manifest, 'entries': entries})
    torch.set_num_threads(4)
    model, checkpoint = load_model(args.checkpoint, 'cpu')
    begin = time.perf_counter()
    with (args.output/'samples.jsonl').open('w') as log:
        for index, (meta, c) in enumerate(indexed_records(args.source, args.split, {e['id'] for e in entries}), 1):
            start = time.perf_counter()
            ident = meta['sample_id']
            directory = args.output/'samples'/ident
            directory.mkdir(parents=True, exist_ok=True)
            rec = {'id': ident, 'family': meta['family'], 'status': 'failed'}
            try:
                camera = projection(c)
                contours = Contours(camera.project(c['actual_xy_mm']), c['ring_offsets'])
                np.savez_compressed(directory/'contours.npz', xy=contours.xy, offsets=contours.offsets, closed=contours.closed)
                write_json(directory/'projection.json', asdict(camera))
                raw = contours.model_arrays()
                t = time.perf_counter()
                old = args.reuse/'samples'/ident if args.reuse else None
                if old and (old/'prediction.json.gz').exists():
                    with gzip.open(old/'prediction.json.gz', 'rt') as stream:
                        prediction = json.load(stream)
                    rec['prediction_reused'] = True
                else:
                    prediction, _ = predict(model, raw, threshold=.7, instances=checkpoint['instances_trained'], units='px')
                    rec['prediction_reused'] = False
                rec['inference_seconds'] = time.perf_counter()-t
                request = extra_requests(prediction, contours)
                t = time.perf_counter()
                report = evaluate(contours, request)
                rec['measurement_seconds'] = time.perf_counter()-t
                write_json(directory/'prediction.json.gz', prediction)
                write_json(directory/'request.json.gz', request)
                write_json(directory/'results.json.gz', report)
                comparison, oracle = reference_comparison(meta, c, contours, prediction, report)
                write_json(directory/'reference-results.json.gz', oracle)
                rec['reference'] = comparison
                image = np.asarray(Image.open(old/'input.png')) if old and (old/'input.png').exists() else render_source(meta, c, camera)
                Image.fromarray(image).save(directory/'input.png')
                preview(image, report).save(directory/'annotated.png')
                rec.update(status='completed', feature_status=dict(Counter(f['status'] for f in report['features'])),
                           measurement_status=dict(Counter(m['status'] for m in report['measurements'])),
                           feature_failure_reasons=dict(Counter(f['reason'] for f in report['features'] if f['status'] == 'fit_failed')),
                           by_operator={op: dict(Counter(m['status'] for m in report['measurements'] if m['operator'] == op))
                                        for op in sorted({m['operator'] for m in report['measurements']})})
            except Exception as error:
                rec.update(reason=f'{type(error).__name__}: {error}')
                (directory/'error.txt').write_text(traceback.format_exc())
            rec['elapsed_seconds'] = time.perf_counter()-start
            log.write(json.dumps(rec, allow_nan=False)+'\n')
            log.flush()
            print(f"{index}/{len(entries)} {ident} {meta['family']} {rec['status']} {rec['elapsed_seconds']:.2f}s features={rec.get('feature_status', {})}", flush=True)
    summary = aggregate(args.output, manifest)
    print(json.dumps({'elapsed_seconds': time.perf_counter()-begin,
                      'samples_completed': summary['samples_completed'],
                      'feature_status': summary['feature_status'],
                      'measurement_status': summary['measurement_status']}), flush=True)


if __name__ == '__main__':
    main()
