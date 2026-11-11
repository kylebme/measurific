import json
import numpy as np
import pytest
from inspection_measurement import Contours, evaluate
from inspection_measurement.geometry import Geometry, fit_geometry
from inspection_measurement.threads import thread_measurements
from inspection_measurement.interaction import prepare_interactive_report, evaluate_interactive_measurement


def sides(style='triangular', count=6, angle=0):
    fractions = {'triangular': [(0, 0), (.5, 1), (1, 0)],
                 'trapezoidal': [(0, 0), (.2, 0), (.4, 1), (.6, 1), (.8, 0), (1, 0)],
                 'square': [(0, 0), (.25, 0), (.25, 1), (.75, 1), (.75, 0), (1, 0)]}[style]
    points = []
    for n in range(count):
        for (a, b), (c, d) in zip(fractions[:-1], fractions[1:]):
            points.extend([[10*(n+a+t*(c-a)), 10+3*(b+t*(d-b))] for t in np.linspace(0, 1, 12, endpoint=False)])
    points.append([count*10, 10])
    q = np.asarray(points)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return [q@rotation.T+100, (q*[1, -1])[::-1]@rotation.T+100]


@pytest.mark.parametrize('style', ['triangular', 'trapezoidal', 'square'])
@pytest.mark.parametrize('angle', [0, .73, 1.57])
def test_known_thread_dimensions(style, angle):
    runs = sides(style, angle=angle)
    report = thread_measurements(Geometry('thread', 'repeated_profile', runs, [False, False]), {})
    metrics = report['thread_metrics']
    for name, expected in [('minor_diameter', 20), ('major_diameter', 26), ('pitch', 10), ('number_of_threads', 6)]:
        assert metrics[name]['status'] == 'measured', (name, metrics[name])
        for stat in ['minimum', 'maximum', 'average']:
            assert metrics[name][stat] == pytest.approx(expected, abs=1e-7), (name, metrics[name])
    # Sharp polyline turns are not arcs, so no radius is reported.
    for name in ['root_radius', 'crest_radius']:
        assert metrics[name]['status'] == 'unavailable', (name, metrics[name])
    json.dumps(report, allow_nan=False)


def test_interactive_thread_uses_original_corners():
    runs = sides('square')
    contours = Contours(np.concatenate(runs), [0, len(runs[0]), sum(map(len, runs))], [False, False])
    request = {'features': [{'id': 't', 'type': 'repeated_profile', 'semantic': 'square_thread',
                            'selection': {'edge_indices': list(range(len(runs[0])-1))+list(range(len(runs[0]),sum(map(len,runs))-1))}}],
               'measurements': [], 'corner_exclusion_px': 1}
    report, prepared = prepare_interactive_report(contours, request)
    result = evaluate_interactive_measurement(contours, prepared, report, {'operator': 'thread', 'featureRefs': ['t']})
    assert result['thread_metrics']['pitch']['average'] == pytest.approx(10)
    assert result['support_policy'] == 'original_selected_observations_including_corners'
    json.dumps(result, allow_nan=False)


def test_missing_side_and_non_thread_are_rejected():
    with pytest.raises(ValueError, match='both opposing sides'):
        thread_measurements(Geometry('t', 'repeated_profile', sides()[:1], [False]), {})
    runs = sides(count=1)
    with pytest.raises(ValueError, match='two resolved'):
        thread_measurements(Geometry('t', 'repeated_profile', runs, [False, False]), {})


def test_round_radii_come_only_from_reported_arcs():
    # Alternating semicircular crests and roots have independently known radius.
    q, spans = [], []
    for i in range(8):
        theta = np.linspace(np.pi, 0, 41)
        sign = 1 if i % 2 == 0 else -1
        spans.append((len(q), len(q)+41))
        q.extend(np.c_[i*4+2+2*np.cos(theta), 12+sign*2*np.sin(theta)][:-1])
    q.append([32, 12])
    q = np.array(q)
    runs = [q, q*[1,-1]]
    unreported = thread_measurements(Geometry('t', 'repeated_profile', runs, [False, False]), {'direction': [1,0]})
    for key in ['root_radius', 'crest_radius']:
        assert unreported['thread_metrics'][key]['status'] == 'unavailable'
    arcs = [fit_geometry(Geometry('arc', 'arc', [run[a:b]], [False])) for run in runs for a, b in spans]
    result = thread_measurements(Geometry('t', 'repeated_profile', runs, [False, False], members=arcs), {'direction': [1,0]})
    for key in ['root_radius', 'crest_radius']:
        assert result['thread_metrics'][key]['count'] == 8
        assert result['thread_metrics'][key]['average'] == pytest.approx(2, abs=1e-6)


def test_nonuniform_pitch_and_diameter_statistics_and_witnesses():
    runs = sides('trapezoidal')
    warped = []
    for q in runs:
        q = q.copy()
        x = q[:, 0]-100
        q[:, 0] = 100+x+.002*x*x
        q[:, 1] = 100+(q[:, 1]-100)*(1+.001*x)
        warped.append(q)
    result = thread_measurements(Geometry('t', 'repeated_profile', warped, [False, False]), {'direction': [1,0]})
    for name in ['pitch', 'major_diameter', 'minor_diameter']:
        metric = result['thread_metrics'][name]
        assert metric['minimum'] < metric['average'] < metric['maximum']
        assert metric['average'] == pytest.approx(np.average(metric['values'], weights=metric.get('sample_weights')))
        if name != 'pitch':
            for stat in ['minimum', 'maximum']:
                a, b = np.asarray(metric[stat+'_witness'])
                assert (b-a)@np.asarray(result['axis_direction']) == pytest.approx(0, abs=1e-8)
        for stat in ['minimum', 'maximum']:
            a, b = np.asarray(metric[stat+'_witness'])
            assert abs((b-a)[0 if name == 'pitch' else 1]) == pytest.approx(metric[stat])


def test_two_crests_and_invalid_parameters():
    g = Geometry('t', 'repeated_profile', sides(count=2), [False, False])
    assert thread_measurements(g, {})['thread_metrics']['number_of_threads']['average'] == 2
    for params in [{'direction':[0,0]}, {'prominence': float('nan')}, {'plateau_tolerance': -1}]:
        with pytest.raises(ValueError):
            thread_measurements(g, params)


def test_thread_annotation_export():
    from inspection_measurement import annotate
    g = Geometry('t', 'repeated_profile', sides(), [False, False])
    result = dict(id='thread', operator='thread', feature_refs=['t'], status='measured', **thread_measurements(g, {}))
    report = {'features': [dict(id='t', status='fitted', geometry={}, support=g.runs, closed=g.closed)], 'measurements': [result]}
    output = annotate(np.full((400,600), 240, np.uint8), report)
    assert output.size == (1180,400)
    assert np.any(np.asarray(output)[:,:600] != 240)


def test_released_model_thread_handoff():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    example = root/'contour-model/release-v2/examples/multi_thread_shaft'
    prediction = json.loads(example.with_suffix('.prediction.json').read_text())
    with np.load(example.with_suffix('.contours.npz')) as data:
        contours = Contours(data['actual_xy_mm']*20+1000, data['ring_offsets'])
    # Convert this millimeter fixture explicitly to synthetic pixel coordinates.
    prediction['units'] = 'px'
    prediction.pop('coordinate_hash', None)
    for feature in prediction['features']+prediction['compound_features']:
        feature['units'] = 'px'
    threads = [f for f in prediction['compound_features'] if 'thread' in f['semantic']]
    prediction['measurements'] = [{'id': f['id'], 'operator': 'thread', 'feature_refs': [f['id']]} for f in threads]
    report = evaluate(contours, prediction)
    assert len(report['measurements']) == 4
    for result in report['measurements']:
        assert result['status'] == 'measured', result
        assert all(result['thread_metrics'][key]['status'] == 'measured'
                   for key in ['minor_diameter', 'major_diameter', 'pitch', 'number_of_threads'])
    json.dumps(report, allow_nan=False)


def test_extras_image_25_extracted_thread():
    # Original PAE observations selected as compound_0001 by release-v2 from
    # app/demo-data/extras/000000025.image.png; no point sorting or smoothing.
    from pathlib import Path
    with np.load(Path(__file__).parent/'fixtures/thread25.npz') as data:
        runs = [data['side_a'], data['side_b']]
    contours = Contours(np.concatenate(runs), [0, len(runs[0]), sum(map(len, runs))], [False, False])
    ids = list(range(len(runs[0])-1))+list(range(len(runs[0]), sum(map(len, runs))-1))
    request = {'features': [{'id': 't', 'type': 'repeated_profile', 'semantic': 'trapezoidal_thread',
                            'selection': {'edge_indices': ids}}], 'measurements': []}
    report, prepared = prepare_interactive_report(contours, request)
    result = evaluate_interactive_measurement(contours, prepared, report, {'operator': 'thread', 'featureRefs': ['t']})
    assert result['status'] == 'measured'
    pitch = result['thread_metrics']['pitch']
    assert 24 < pitch['minimum'] < pitch['maximum'] < 26
    # A user-selected contour has no reported arcs, so no radius is reported.
    assert result['thread_metrics']['root_radius']['status'] == 'unavailable'
    assert result['thread_metrics']['crest_radius']['status'] == 'unavailable'
    for metric in result['thread_metrics'].values():
        if metric['status'] == 'measured':
            assert metric['minimum'] <= metric['average'] <= metric['maximum']
    json.dumps(result, allow_nan=False)


def test_cumulative_backtracking_is_rejected_even_with_small_steps():
    runs = sides('square')
    q = runs[0].copy()
    # Reverse a whole vertical flank axially in small individually legal steps.
    # Each reverse step is less than half of the median observation spacing.
    q[12:25, 0] -= np.linspace(0, 1.2, 13)
    runs[0] = q
    with pytest.raises(ValueError, match='backtracks'):
        thread_measurements(Geometry('t', 'repeated_profile', runs, [False, False]), {'direction': [1,0]})


def test_pitch_uses_corresponding_flanks_not_noisy_crest_extrema():
    runs = sides('trapezoidal')
    for q in runs:
        x = q[:, 0]-100
        sign = np.sign(q[:, 1]-100)
        land = np.isclose(abs(q[:, 1]-100), 13)
        # Alternate which end of the flat crest is highest. Flanks at mid-depth
        # remain exactly periodic; crest-extremum spacing alternates 8 / 12 px.
        q[land, 1] += sign[land]*.15*(x[land] % 10-5)*(-1.)**np.floor(x[land]/10)
    result = thread_measurements(Geometry('t', 'repeated_profile', runs, [False, False]), {'direction': [1,0]})
    metric = result['thread_metrics']['pitch']
    assert metric['values'] == pytest.approx([10]*20, abs=1e-7)
    for stat in ['minimum', 'maximum']:
        a, b = np.asarray(metric[stat+'_witness'])
        assert b-a == pytest.approx([10,0], abs=1e-7)


def rounded_trapezoid(root_radius=.25, crest_radius=.6):
    vertices = np.array([[x, y] for n in range(6) for x, y in
                         [(n*10,10),(n*10+2,10),(n*10+4,13),(n*10+6,13),(n*10+8,10)]]+[[60,10]], float)
    # Remove collinear vertices before constructing independent analytic fillets.
    retained = [vertices[0]]
    for a, b, c in zip(vertices[:-2],vertices[1:-1],vertices[2:]):
        u, v = b-a, c-b
        if abs(u[0]*v[1]-u[1]*v[0]) > 1e-8: retained.append(b)
    vertices = np.array(retained+[vertices[-1]])
    points, fillets = [vertices[0]], []
    def line_to(end):
        start = points[-1]
        steps = max(2, int(np.ceil(np.linalg.norm(end-start)/.04)))
        points.extend(start+(end-start)*t for t in np.linspace(0,1,steps+1)[1:])
    for a,b,c in zip(vertices[:-2], vertices[1:-1], vertices[2:]):
        u, v = (b-a)/np.linalg.norm(b-a), (c-b)/np.linalg.norm(c-b)
        cross = u[0]*v[1]-u[1]*v[0]
        turn = np.arctan2(cross, u@v)
        radius = root_radius if cross > 0 else crest_radius
        trim = radius*np.tan(abs(turn)/2)
        start, end = b-trim*u, b+trim*v
        center = start+np.sign(turn)*radius*np.array([-u[1],u[0]])
        line_to(start)
        angle = np.arctan2(*(start-center)[::-1])
        points.extend(center+radius*np.array([np.cos(t),np.sin(t)])
                      for t in np.linspace(angle,angle+turn,25)[1:])
        fillets.append((len(points)-25, len(points)))
    line_to(vertices[-1])
    return np.array(points), fillets


@pytest.mark.parametrize('rotation', [0, .72])
def test_trapezoidal_radii_are_individual_reported_fillet_arcs(rotation):
    q, fillets = rounded_trapezoid()
    matrix = np.array([[np.cos(rotation),-np.sin(rotation)],[np.sin(rotation),np.cos(rotation)]])
    runs = [q@matrix.T+100, (q*[1,-1])@matrix.T+100]
    arcs = [fit_geometry(Geometry('arc','arc',[run[a:b]],[False])) for run in runs for a, b in fillets]
    # An arc on a flank is neither a root nor a crest radius.
    theta = np.linspace(-.1, .1, 9)+np.arctan2(-2, 3)
    flank = np.c_[3+6*np.cos(theta), 11.5+6*np.sin(theta)]-6*np.array([np.cos(theta[4]), np.sin(theta[4])])
    arcs.append(fit_geometry(Geometry('flank','arc',[flank@matrix.T+100],[False])))
    g = Geometry('t','repeated_profile',runs,[False,False],members=arcs)
    result = thread_measurements(g, {'direction': matrix[:,0]})
    for key, expected in [('root_radius',.25), ('crest_radius',.6)]:
        metric = result['thread_metrics'][key]
        assert metric['count'] == len(fillets), metric
        assert metric['values'] == pytest.approx([expected]*metric['count'], abs=1e-6)
        for stat in ['minimum', 'maximum']:
            a, b = np.asarray(metric[stat+'_witness'])
            assert np.linalg.norm(b-a) == pytest.approx(metric[stat], abs=1e-8)
    assert result['thread_metrics']['pitch']['average'] == pytest.approx(10,abs=1e-6)


def test_model_thread_compound_reports_its_arc_members():
    q, fillets = rounded_trapezoid()
    runs = [q+100, (q*[1,-1])[::-1]+100]
    contours = Contours(np.concatenate(runs), [0, len(runs[0]), sum(map(len, runs))], [False, False])
    features, offset = [], 0
    for run in runs:
        n = len(run)
        spans = fillets if run is runs[0] else [(n-b, n-a) for a, b in fillets[::-1]]
        cuts = [0]
        for a, b in spans:
            cuts += [a, b-1]
        cuts.append(n-1)
        for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            if b > a:
                features.append({'id': f'base_{len(features):05d}', 'measurement_route': 'arc' if i % 2 else 'line',
                                 'raw_edge_indices': list(range(offset+a, offset+b))})
        offset += n
    edges = sorted(e for f in features for e in f['raw_edge_indices'])
    prediction = {'schema_version': 2, 'label_version': 'primitives-compounds-v2', 'units': 'px',
                  'features': features,
                  'compound_features': [{'id': 'thread', 'semantic': 'trapezoidal_thread', 'measurement_route': 'thread_pitch',
                                         'base_instance_ids': [f['id'] for f in features], 'raw_edge_indices': edges}],
                  'measurements': [{'id': 'm', 'operator': 'thread', 'feature_refs': ['thread']}]}
    result = evaluate(contours, prediction)['measurements'][0]
    assert result['status'] == 'measured', result
    for key, expected in [('root_radius',.25), ('crest_radius',.6)]:
        metric = result['thread_metrics'][key]
        assert metric['count'] == len(fillets), metric
        assert metric['average'] == pytest.approx(expected, abs=1e-6)



def test_user_defined_thread_keeps_arc_segments_as_radius_sources():
    # Both thread definition tools keep arc segments as members, so a thread
    # defined over arcs reports their radii.
    from inspection_measurement import redefine_compound, redefine_compound_sides
    q, fillets = rounded_trapezoid()
    runs = [q+100, (q*[1,-1])[::-1]+100]
    contours = Contours(np.concatenate(runs), [0, len(runs[0]), sum(map(len, runs))], [False, False])
    features, offset = [], 0
    for run in runs:
        n = len(run)
        spans = fillets if run is runs[0] else [(n-b, n-a) for a, b in fillets[::-1]]
        cuts = [0]
        for a, b in spans:
            cuts += [a, b-1]
        cuts.append(n-1)
        for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            if b > a:
                features.append({'id': f'base_{len(features):05d}', 'measurement_route': 'arc' if i % 2 else 'line',
                                 'raw_edge_indices': list(range(offset+a, offset+b))})
        offset += n
    prediction = {'schema_version': 2, 'label_version': 'primitives-compounds-v2', 'units': 'px',
                  'features': features, 'compound_features': [], 'relations': []}
    _, session = prepare_interactive_report(contours, prediction)
    ends = [runs[0][0], runs[0][-1], runs[1][0], runs[1][-1]]
    for result in [redefine_compound_sides(contours, session, 'trapezoidal_thread', [p.tolist() for p in ends]),
                   redefine_compound(contours, session, 'trapezoidal_thread', [f['id'] for f in features])]:
        measured = evaluate_interactive_measurement(contours, result['request'], result['report'],
                                                    {'operator': 'thread', 'featureRefs': [result['feature_id']]})
        for key, expected in [('root_radius', .25), ('crest_radius', .6)]:
            metric = measured['thread_metrics'][key]
            assert metric['count'] == len(fillets), metric
            assert metric['average'] == pytest.approx(expected, abs=1e-6)
