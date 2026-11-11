#!/usr/bin/env python3
"""Summarize saved fits without rerunning or changing the image estimator."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point


def summarize(source, runs, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source/'manifest.json').read_text())
    cases = {c['id']: c for c in manifest['cases']}
    records, provenance = [], []
    for run in runs:
        provenance.append(json.loads((run/'manifest.json').read_text()))
        for r in json.loads((run/'results.json').read_text()):
            recovery = r.get('recovery') or {}
            compact = {k: v for k, v in r.items() if k not in ['recovery', 'corner_audit']}
            compact['rejections'] = dict(Counter(x['reason'] for x in recovery.get('rejected', [])))
            audit = []
            with np.load(source/cases[r['id']]['truth']) as truth:
                rings = [truth['actual_xy_px'][a:b] for a, b in zip(truth['ring_offsets'][:-1], truth['ring_offsets'][1:])]
            matches = {x['observed_ring']: x['truth_ring'] for x in r.get('baseline', {}).get('rings', [])}
            for item in recovery.get('accepted', []):
                if item['path'] not in matches:
                    audit.append(dict(xy=item['xy'], unmatched_ring=True))
                    continue
                q = rings[matches[item['path']]]
                u, v = q-np.roll(q, 1, axis=0), np.roll(q, -1, axis=0)-q
                angle = np.arctan2(u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0], (u*v).sum(axis=1))
                # Audit includes shallower true corners than the proposal cutoff:
                # simplification can increase a measured turn past 0.5 radians.
                vertices = q[abs(angle) > .1]
                distances = np.linalg.norm(vertices-item['xy'], axis=1)
                audit.append(dict(xy=item['xy'],
                    corner_error_px=float(distances.min()) if len(distances) else None,
                    boundary_error_px=LineString(np.vstack([q, q[:1]])).distance(Point(item['xy'])),
                    estimated_sigma_px=item['blur_sigma_px']))
            compact['accepted_corner_audit'] = audit
            records.append(compact)
    summaries = {}
    for split in ['development', 'heldout', 'all']:
        rows = [r for r in records if split == 'all' or r['split'] == split]
        good = [r for r in rows if r['status'] == 'ok']
        errors = [a['corner_error_px'] for r in rows for a in r['accepted_corner_audit'] if a.get('corner_error_px') is not None]
        changes = [r['result']['hausdorff_px']-r['baseline']['hausdorff_px'] for r in good]
        summaries[split] = dict(cases=len(rows), valid=len(good), failures=len(rows)-len(good),
            accepted_corners=sum(len(r['accepted_corner_audit']) for r in rows),
            unmatched_corners=sum(a.get('corner_error_px') is None for r in rows for a in r['accepted_corner_audit']),
            accepted_corner_max_px=max(errors, default=None),
            accepted_corner_p95_px=float(np.quantile(errors, .95)) if errors else None,
            accepted_corners_over_02px=sum(e > .2 for e in errors),
            images_max_improved=sum(d < -1e-6 for d in changes),
            images_max_worsened=sum(d > 1e-6 for d in changes),
            images_max_unchanged=sum(abs(d) <= 1e-6 for d in changes),
            baseline_max_px=max((r['baseline']['hausdorff_px'] for r in good), default=None),
            recovered_max_px=max((r['result']['hausdorff_px'] for r in good), default=None),
            seconds_total=sum(r['seconds'] for r in rows))
    payload = dict(summary=summaries, records=records, runs=provenance,
        audit='Nearest vertex with absolute turn >0.1 radians on the matched truth ring; boundary error reported separately. Accuracy conditional on acceptance; failures retained.',
        summarizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'results.json').write_text(json.dumps(payload, indent=2)+'\n')
    (output/'dataset-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    summarize(a.input, a.runs, a.output)
