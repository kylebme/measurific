"""Deterministic CPU timings; run with the same interpreter before/after edits.

    python measurement/benchmarks/benchmark_performance.py --output /tmp/before.json

Fixture construction and imports are outside timings. Every timed evaluate call
includes selection, fitting, operators, and JSON-safe report construction.
"""
import argparse
import json
import platform
from time import perf_counter

import numpy as np
import scipy
import shapely

from inspection_measurement import Contours, evaluate
from known_geometry import FAMILIES, generate


def workloads():
    cases = [generate(f, 20260908+i*10000) for i, f in enumerate(FAMILIES)]
    fixtures = [(Contours.from_dict(c['contours']), c['request']) for c in cases]
    yield 'known_truth_12_families', lambda: [evaluate(c, r) for c, r in fixtures]

    t = np.linspace(0, 2*np.pi, 1024, endpoint=False)
    unit = np.c_[np.cos(t), np.sin(t)]
    q = unit*[80, 30]+[100, 100]
    rng = np.random.default_rng(20260908)
    noisy = q+rng.normal(0, .03, q.shape)
    ellipse = Contours(noisy, [0, len(noisy)])
    request = {'features': [{'id': 'e', 'type': 'ellipse', 'selection': {'path': 0}}]}
    yield 'noisy_ellipse_1024', lambda: evaluate(ellipse, request)

    contours = Contours(np.concatenate([q, unit*20+[100, 100]]), [0, 1024, 2048])
    features = [{'id': str(i), 'type': 'contour', 'selection': {'path': i}} for i in range(2)]
    for name, operator, refs, params in [
        ('profile_1024', 'profile', ['0'], {'nominal_points': (q+[.1, .1]).tolist(), 'closed': True}),
        ('wall_thickness_1024', 'wall_thickness', ['1', '0'], {}),
        ('closest_clearance_1024', 'distance_construction', ['0', '1'], {'construction': 'closest_clearance'}),
    ]:
        req = {'features': features, 'measurements': [
            {'id': 'm', 'operator': operator, 'feature_refs': refs, 'params': params}]}
        yield name, lambda req=req: evaluate(contours, req)

    x = np.linspace(-100, 100, 8193)
    bowed = Contours(np.c_[x, 1-(x/100)**2], [0, len(x)], [False])
    req = {'features': [{'id': 'line', 'type': 'line', 'selection': {'path': 0}}]}
    yield 'bowed_line_8193', lambda: evaluate(bowed, req)


def verify_truth():
    """Check 300 fixtures against their independent construction formulas."""
    checks = 0
    for i, family in enumerate(FAMILIES):
        for j in range(25):
            case = generate(family, 20260908+i*10000+j)
            measured = {m['id']: m for m in evaluate(case['contours'], case['request'])['measurements']}
            for truth in case['truth']:
                actual = measured[truth['measurement_id']]
                assert actual['status'] == 'measured', (family, j, actual)
                difference = np.abs(np.asarray(actual[truth['field']])-truth['expected'])
                if 'period' in truth:
                    difference %= truth['period']
                    difference = np.minimum(difference, truth['period']-difference)
                assert np.max(difference) <= truth['absolute_tolerance'], (family, j, truth, actual)
                checks += 1
    return {'cases': len(FAMILIES)*25, 'truth_checks_passed': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--output')
    parser.add_argument('--only', help='Substring of workload name')
    parser.add_argument('--verify-truth', action='store_true', help='Also check 300 known-truth fixtures outside timing')
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    report = {'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'numpy': np.__version__, 'scipy': scipy.__version__, 'shapely': shapely.__version__},
              'repeat': args.repeat, 'workloads': {}}
    if args.verify_truth:
        report['correctness'] = verify_truth()
        print(report['correctness'], flush=True)
    for name, run in workloads():
        if args.only and args.only not in name:
            continue
        run()  # Warm imports and CPU dispatch before timing.
        times = []
        for _ in range(args.repeat):
            start = perf_counter()
            result = run()
            times.append(perf_counter()-start)
            for item in result if isinstance(result, list) else [result]:
                assert item['summary']['failed_or_unavailable'] == 0, item
                assert all(f['status'] == 'fitted' for f in item['features']), item
        report['workloads'][name] = {'median_ms': float(np.median(times)*1000),
                                     'min_ms': min(times)*1000}
        print(f'{name}: {np.median(times)*1000:.3f} ms', flush=True)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
