import json
import numpy as np
import pytest
from inspection_measurement import Contours, evaluate


def circle(c=(0, 0), r=5, n=128):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.c_[np.cos(t), np.sin(t)]*r+c


def bundle(paths, closed):
    return Contours(np.concatenate(paths), np.r_[0, np.cumsum([len(p) for p in paths])], np.array(closed))


def request(features, operators):
    # Exact operator fixtures opt out of image-data corner trimming.
    return {'corner_exclusion_px': 0, 'features': features, 'measurements': [
        {'id': str(i), 'operator': op, 'feature_refs': refs, 'params': params}
        for i, (op, refs, params) in enumerate(operators)]}


def feature(ident, kind, path):
    return {'id': ident, 'type': kind, 'selection': {'path': path}}


def values(report):
    assert all(m['status'] == 'measured' for m in report['measurements']), report
    return [m['value'] for m in report['measurements']]


def test_primitive_prediction_keeps_base_and_compound_measurement_support():
    c=bundle([circle()], [True])
    base={'id':'base_1','semantic':'circle','semantic_id':3,'label_level':'base',
          'measurement_route':'circle','raw_edge_indices':list(range(128)),'units':'px'}
    compound={'id':'compound_1','semantic':'tooth_profile','semantic_id':4,'label_level':'compound',
              'base_instance_ids':['base_1'],'measurement_route':'repeated_profile',
              'raw_edge_indices':list(range(128)),'units':'px'}
    report=evaluate(c,{'schema_version':2,'label_version':'primitives-compounds-v1','units':'px',
                       'features':[base],'compound_features':[compound]})
    assert {f['id'] for f in report['features']}=={'base_1','compound_1'}
    assert all(f['status']=='fitted' for f in report['features'])


def test_lines_circle_relationships_and_tolerances():
    x = np.linspace(-10, 10, 21)
    c = bundle([np.c_[x, 0*x], np.c_[x, 0*x+8], circle((3, 4), 2)], [False, False, True])
    req = request([feature('a', 'line', 0), feature('b', 'line', 1), feature('c', 'circle', 2)], [
        ('distance', ['a', 'b'], {}), ('distance', ['a', 'c'], {}),
        ('least_squares_diameter', ['c'], {}), ('parallelism', ['a', 'b'], {})])
    req['measurements'][0].update(nominal_value=8, tolerance={'lower': 7.9, 'upper': 8.1})
    report = evaluate(c, req)
    np.testing.assert_allclose(values(report), [8, 4, 4, 0], atol=1e-8)
    assert report['measurements'][0]['pass_fail'] == 'pass'
    assert report['measurements'][0]['deviation'] == 0
    assert report['measurements'][0]['uncertainty'] is None
    json.dumps(report, allow_nan=False)


def test_rotation_translation_invariance():
    q = np.c_[np.linspace(-4, 4, 30), np.zeros(30)]
    angle = .73
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    c = bundle([q@rotation.T+[35, 66], (q+[0, 7])@rotation.T+[35, 66]], [False]*2)
    report = evaluate(c, request([feature('a', 'line', 0), feature('b', 'line', 1)], [('distance', ['a', 'b'], {})]))
    np.testing.assert_allclose(values(report), [7], atol=1e-10)


def test_segment_minimum_distance_includes_segment_interiors():
    c = bundle([np.array([[-10, 0], [10, 0]]), np.array([[0, -10], [0, 10]])], [False]*2)
    req = request([feature('a', 'line', 0), feature('b', 'line', 1)], [
        ('minimum_separation', ['a', 'b'], {}), ('intersection', ['a', 'b'], {}),
        ('distance', ['a', 'b'], {})])
    report = evaluate(c, req)
    assert report['measurements'][0]['value'] == 0
    np.testing.assert_allclose(report['measurements'][1]['value'], [0, 0])
    assert report['measurements'][2]['status'] == 'failed'


def test_removed_pairwise_operator_is_rejected():
    from inspection_measurement.operators import OPERATORS
    from inspection_measurement.interaction import MEASUREMENT_SPECS
    assert 'distance_statistics' not in OPERATORS
    assert 'distance_statistics' not in MEASUREMENT_SPECS
    c = bundle([np.array([[0., 0.], [10., 0.]]), np.array([[0., 5.], [10., 5.]])], [False, False])
    report = evaluate(c, request([feature('a', 'line', 0), feature('b', 'line', 1)], [
        ('distance_statistics', ['a', 'b'], {})]))
    assert report['measurements'][0]['status'] == 'failed'
    assert report['measurements'][0]['reason'] == 'Unsupported operator: distance_statistics'


def test_wrapped_span_and_disconnected_support():
    q = circle()
    c = bundle([q], [True])
    req = request([{'id': 'arc', 'type': 'arc', 'selection': {'spans': [
        {'ring_index': 0, 'start_edge': 110, 'edge_count': 30}]}}], [
        ('radius', ['arc'], {}), ('angular_span', ['arc'], {}), ('circularity_ls', ['arc'], {})])
    report = evaluate(c, req)
    np.testing.assert_allclose(report['measurements'][0]['value'], 5, atol=1e-8)
    np.testing.assert_allclose(report['measurements'][1]['value'], 30*360/128, atol=1e-8)
    assert report['measurements'][2]['status'] == 'failed'
    req['features'][0]['selection']['spans'].append({'ring_index': 0, 'start_edge': 40, 'edge_count': 10})
    assert evaluate(c, req)['features'][0]['status'] == 'fit_failed'


def test_minimum_strip_and_observed_polygon():
    q = np.array([[-3, -1], [3, -1], [3, 1], [-3, 1.]])
    c = bundle([q], [True])
    report = evaluate(c, request([feature('p', 'polygon', 0), feature('l', 'line', 0)], [
        ('area', ['p'], {}), ('perimeter', ['p'], {}), ('straightness', ['l'], {}),
        ('local_size', ['p'], {'direction': [1, 0], 'stations': [-2, 0, 2]})]))
    np.testing.assert_allclose(values(report), [12, 16, 2, 2])


def test_profile_signed_errors_and_no_implicit_alignment():
    c = bundle([circle(r=6)], [True])
    report = evaluate(c, request([feature('c', 'circle', 0)], [('profile', ['c'], {'nominal_points': circle(r=5).tolist(), 'closed': True})]))
    m = report['measurements'][0]
    assert m['status'] == 'measured'
    np.testing.assert_allclose(m['outward'], 1, atol=1e-8)
    assert m['inward'] == 0


def test_model_hash_abstention_and_user_override():
    c = bundle([circle()], [True])
    f = {'id': 'hole', 'measurement_route': 'circle', 'units': 'px', 'status': 'abstain',
         'raw_edge_indices': list(range(128)), 'contour_spans': [{'ring_index': 0, 'start_edge': 0, 'edge_count': 128}]}
    req = {'units': 'px', 'coordinate_hash': c.coordinate_hash, 'features': [f]}
    assert evaluate(c, req)['features'][0]['status'] == 'abstain'
    req['overrides'] = [{'id': 'hole', 'source': 'user', 'accept': True}]
    assert evaluate(c, req)['features'][0]['status'] == 'fitted'
    req['coordinate_hash'] = 'stale'
    with pytest.raises(ValueError, match='hash'):
        evaluate(c, req)


def test_regions_and_isolation_of_failed_features():
    c = bundle([circle()], [True])
    req = request([feature('circle', 'circle', 0), {'id': 'bad', 'type': 'line', 'selection': {'region': {'box': [100, 100, 110, 110]}}}], [
        ('radius', ['circle'], {}), ('length', ['bad'], {})])
    report = evaluate(c, req)
    assert report['measurements'][0]['status'] == 'measured'
    assert report['measurements'][1]['status'] == 'unavailable'


def test_pattern_pitch_and_polar_seam():
    centers = np.array([[2, 3], [7, 3], [12, 3], [17, 3]])
    c = bundle([circle(center, 1) for center in centers], [True]*4)
    features = [feature(str(i), 'circle', i) for i in range(4)]
    req = request(features, [('pattern', [str(i) for i in range(4)], {'direction': [1, 0], 'nominal_pitch': 5})])
    np.testing.assert_allclose(values(evaluate(c, req)), [5])
    centers = circle(r=10, n=6)
    c = bundle([circle(center, 1) for center in centers], [True]*6)
    features = [feature(str(i), 'circle', i) for i in range(6)]
    req = request(features, [('pattern', [str(i) for i in range(6)], {'domain': 'polar', 'center': [0, 0]})])
    report = evaluate(c, req)
    np.testing.assert_allclose(values(report), [60])
    np.testing.assert_allclose(report['measurements'][0]['angular_pitch'], 60, atol=1e-9)


def test_repetition_and_compound_taper():
    x = np.linspace(0, 40, 1001)
    c = bundle([np.c_[x, np.cos(2*np.pi*x/5)]], [False])
    req = request([feature('r', 'repeated_profile', 0)], [('repetition', ['r'], {'direction': [1, 0], 'prominence': 1.})])
    np.testing.assert_allclose(values(evaluate(c, req)), [5], atol=.015)
    x = np.linspace(-10, 10, 50)
    c = bundle([np.c_[x, x*0], np.c_[x, 5+.1*x]], [False]*2)
    features = [feature('a', 'line', 0), feature('b', 'line', 1), {'id': 't', 'type': 'taper', 'members': ['a', 'b']}]
    report = evaluate(c, request(features, [('compound', ['t'], {'stations': [-10, 10]})]))
    np.testing.assert_allclose(report['measurements'][0]['widths'], [4, 6])


def test_ellipse_slot_and_rounded_rectangle():
    t = np.linspace(0, 2*np.pi, 80, endpoint=False)
    c = bundle([np.c_[8*np.cos(t), 3*np.sin(t)]], [True])
    report = evaluate(c, request([feature('e', 'ellipse', 0)], [('major_axis', ['e'], {}), ('minor_axis', ['e'], {})]))
    np.testing.assert_allclose(values(report), [16, 6], atol=1e-5)
    t = np.linspace(-np.pi/2, np.pi/2, 40)
    q = np.vstack([np.c_[5+2*np.cos(t), 2*np.sin(t)], np.c_[-5-2*np.cos(t), -2*np.sin(t)]])
    c = bundle([q], [True])
    report = evaluate(c, request([feature('s', 'slot', 0)], [('width', ['s'], {}), ('overall_length', ['s'], {})]))
    np.testing.assert_allclose(values(report), [4, 14], atol=1e-6)
    q = np.concatenate([np.c_[cx+np.cos(a), cy+np.sin(a)] for cx, cy, a in [
        (4, 2, np.linspace(0, np.pi/2, 20)), (-4, 2, np.linspace(np.pi/2, np.pi, 20)),
        (-4, -2, np.linspace(np.pi, 1.5*np.pi, 20)), (4, -2, np.linspace(1.5*np.pi, 2*np.pi, 20))]])
    c = bundle([q], [True])
    report = evaluate(c, request([feature('r', 'rounded_rectangle', 0)], [('corner_radius', ['r'], {}), ('overall_length', ['r'], {})]))
    np.testing.assert_allclose(values(report), [1, 10], atol=1e-5)


@pytest.mark.parametrize('change', ['units', 'duplicate_id', 'float_offsets', 'duplicate_vertex', 'self_intersection'])
def test_reject_invalid_input(change):
    c = bundle([circle()], [True])
    req = {'features': [feature('a', 'circle', 0)]}
    with pytest.raises(ValueError):
        if change == 'units':
            req['units'] = 'mm'
        elif change == 'duplicate_id':
            req['features'] *= 2
        elif change == 'float_offsets':
            c = Contours(c.xy, [0., 128.])
        elif change == 'duplicate_vertex':
            c = Contours(np.vstack([c.xy, c.xy[0]]), [0, 129])
        else:
            c = Contours([[0, 0], [2, 2], [0, 2], [2, 0]], [0, 4])
        evaluate(c, req)


def test_cycle_reports_failure_and_vector_tolerance_rejected():
    c = bundle([circle()], [True])
    req = {'features': [{'id': 'a', 'type': 'width', 'members': ['b']}, {'id': 'b', 'type': 'width', 'members': ['a']}]}
    report = evaluate(c, req)
    assert all(f['status'] == 'fit_failed' for f in report['features'])
    req = request([feature('c', 'circle', 0)], [('center', ['c'], {})])
    req['measurements'][0]['tolerance'] = {'upper': 3}
    assert evaluate(c, req)['measurements'][0]['status'] == 'failed'


def test_elliptical_arc_reverse_traversal_preserves_length():
    from scipy.integrate import quad
    t = np.linspace(.2, 2.2, 60)
    q = np.c_[8*np.cos(t), 3*np.sin(t)]
    expected = quad(lambda a: np.hypot(8*np.sin(a), 3*np.cos(a)), .2, 2.2)[0]
    for points in (q, q[::-1]):
        c = bundle([points], [False])
        req = request([feature('e', 'elliptical_arc', 0)], [('arc_length', ['e'], {})])
        np.testing.assert_allclose(values(evaluate(c, req)), [expected], atol=2e-4)


@pytest.mark.parametrize('vertical', [False, True])
def test_straightness_exact_translated_line_does_not_crash(vertical):
    # Mean-centering these constant coordinates can leave a nonzero constant
    # residual that passes a numerical rank test but makes Qhull reject the hull.
    q = np.c_[np.linspace(931.5, 938.364, 12), np.full(12, 315.1)]
    if vertical:
        q = q[:, ::-1]
    c = bundle([q], [False])
    req = request([feature('line', 'line', 0)], [('straightness', ['line'], {})])
    assert values(evaluate(c, req)) == [0.]


def test_straightness_retains_thin_nonzero_form_error():
    from inspection_measurement.geometry import minimum_zone_straightness
    q = np.array([[0., 0.], [10., 0.], [10., 1e-9], [0., 1e-9]])+[900., 315.1]
    np.testing.assert_allclose(minimum_zone_straightness(q), 1e-9, rtol=1e-4, atol=1e-13)
