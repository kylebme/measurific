#!/usr/bin/env python3
"""Controlled, exploratory ablation of only the sharp-versus-bevel rejection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from evaluate_diverse_corners import evaluate_dataset
from preview_corner_outliers import CASES
import experimental_corner_fit


def clipped_contour(path, bounds):
    obj = json.loads(path.read_text())
    xy = np.asarray(obj['xy'])
    rings = [xy[a:b] for a, b in zip(obj['offsets'][:-1], obj['offsets'][1:])]
    return unary_union([LineString(np.vstack([q, q[:1]])) for q in rings]).intersection(box(*bounds))


def investigate(source, development, heldout, crops, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source/'manifest.json').read_text())
    # Include the known false-sharp-chamfer development case as a negative control.
    selected = set(CASES+['threaded_shaft-d0-blurred'])
    manifest['cases'] = [c for c in manifest['cases'] if c['id'] in selected]
    subset = output/'source'
    subset.mkdir(exist_ok=True)
    (subset/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for name in ['images', 'truth']:
        link = subset/name
        if not link.exists(): link.symlink_to((source/name).resolve(), target_is_directory=True)
    original = experimental_corner_fit.fit_corner_pixels
    def without_bevel_challenge(*args, **kwargs):
        kwargs['check_chamfer'] = False
        return original(*args, **kwargs)
    experimental_corner_fit.fit_corner_pixels = without_bevel_challenge
    try:
        evaluate_dataset(subset, output/'no-bevel-check', 'all', 'local')
    finally:
        experimental_corner_fit.fit_corner_pixels = original
    run_manifest = output/'no-bevel-check'/'manifest.json'
    info = json.loads(run_manifest.read_text())
    info.update(ablation='Only check_chamfer=False; all other parameters unchanged.',
                exploratory=True, independent_validation=False,
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    run_manifest.write_text(json.dumps(info, indent=2)+'\n')
    locations = {c['id']: c for c in json.loads((crops/'locations.json').read_text())}
    baseline = {}
    for run in [development, heldout]:
        baseline.update({c['id']: (c, run) for c in json.loads((run/'results.json').read_text())})
    records = []
    for r in json.loads((output/'no-bevel-check'/'results.json').read_text()):
        old, run = baseline[r['id']]
        item = dict(id=r['id'], original_max_px=old['baseline']['hausdorff_px'],
                    guarded_max_px=old['result']['hausdorff_px'],
                    no_bevel_max_px=r['result']['hausdorff_px'],
                    guarded_accepted=len(old['recovery']['accepted']),
                    no_bevel_accepted=len(r['recovery']['accepted']))
        if r['id'] in locations:
            bounds = locations[r['id']]['crop_box']
            a = clipped_contour(run/(r['id']+'.original.json'), bounds)
            b = clipped_contour(run/(r['id']+'.contours.json'), bounds)
            item['original_vs_guarded_crop_exactly_equal'] = a.equals(b)
            item['original_vs_guarded_crop_hausdorff_px'] = a.hausdorff_distance(b)
            candidate = locations[r['id']]['nearest_candidate']
            item['nearest_proposal_reason'] = candidate['reason'] if candidate else None
        records.append(item)
    (output/'comparison.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['input','development','heldout','crops','output']:
        p.add_argument('--'+key, type=Path, required=True)
    a=p.parse_args()
    investigate(a.input,a.development,a.heldout,a.crops,a.output)
