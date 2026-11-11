import numpy as np
from inspection_measurement import evaluate
from test_measurement import bundle, circle, feature, request, values


def test_groove_step_angle_and_width():
    x = np.linspace(-6, 6, 25)
    y = np.linspace(0, 4, 20)
    c = bundle([np.c_[x, x*0], np.c_[y*0-3, y], np.c_[y*0+3, y], np.c_[x, x*0+4]], [False]*4)
    features = [feature('opening', 'line', 0), feature('left', 'line', 1), feature('right', 'line', 2), feature('bottom', 'line', 3),
                {'id': 'groove', 'type': 'groove', 'members': ['opening', 'left', 'right', 'bottom'],
                 'params': {'roles': {k: k for k in ('opening', 'left', 'right', 'bottom')}}},
                {'id': 'step', 'type': 'step', 'members': ['opening', 'bottom']},
                {'id': 'angle', 'type': 'angle', 'members': ['opening', 'left']},
                {'id': 'width', 'type': 'width', 'members': ['left', 'right']}]
    report = evaluate(c, request(features, [('compound', [k], {}) for k in ('groove', 'step', 'angle', 'width')]))
    np.testing.assert_allclose(values(report), [4, 4, 90, 6])
    assert report['measurements'][0]['opening_width'] == 6
    assert report['measurements'][0]['bottom_width'] == 6


def test_symmetry_datum_position_tangency_and_common_axis():
    c = bundle([circle((-5, 3), 2), circle((5, 3), 2), np.array([[0, -10.], [0, 10.]]),
                np.array([[-10., 1], [10., 1]])], [True, True, False, False])
    fs = [feature('a', 'circle', 0), feature('b', 'circle', 1), feature('axis', 'line', 2), feature('tangent', 'line', 3)]
    req = request(fs, [('symmetry', ['a', 'b', 'axis'], {}), ('tangency', ['tangent', 'a'], {}),
                       ('common_axis', ['axis', 'a', 'b'], {}), ('true_position_2d', ['b'],
                       {'nominal_center': [5, 3], 'datum_origin': [0, 0], 'datum_x_direction': [1, 0]})])
    np.testing.assert_allclose(values(evaluate(c, req)), [0, 0, 5, 0], atol=1e-8)


def test_grid_and_arbitrary_patterns():
    cells = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    centers = cells*[8, 5]+[2, 3]
    c = bundle([circle(p, 1) for p in centers], [True]*4)
    fs = [feature(str(i), 'circle', i) for i in range(4)]
    refs = [f['id'] for f in fs]
    req = request(fs, [('pattern', refs, {'domain': 'grid', 'direction': [1, 0], 'cells': cells.tolist()}),
                       ('pattern', refs, {'domain': 'arbitrary', 'nominal_centers': centers.tolist()})])
    report = evaluate(c, req)
    np.testing.assert_allclose(values(report), [0, 0], atol=1e-8)
    np.testing.assert_allclose(report['measurements'][0]['row_pitch'], 5)
    np.testing.assert_allclose(report['measurements'][0]['column_pitch'], 8)


def test_angular_repetition_counts_seam_peak_once():
    t = np.linspace(0, 2*np.pi, 720, endpoint=False)
    r = 20+2*np.cos(8*t)
    c = bundle([np.c_[r*np.cos(t), r*np.sin(t)]], [True])
    req = request([feature('teeth', 'repeated_profile', 0)], [
        ('repetition', ['teeth'], {'domain': 'angular', 'center': [0, 0], 'prominence': 2})])
    report = evaluate(c, req)
    np.testing.assert_allclose(values(report), [45], atol=.01)
    assert report['measurements'][0]['count'] == 8


def test_open_freeform_length_does_not_join_disconnected_spans():
    q = np.array([[0., 0], [1, 0], [2, 0], [100, 0], [101, 0], [102, 0]])
    c = bundle([q], [False])
    req = request([{'id': 'f', 'type': 'freeform', 'selection': {'edge_indices': [0, 1, 4]}}], [('length', ['f'], {})])
    assert values(evaluate(c, req)) == [3]


def test_curvature_and_directed_wall_distribution():
    c = bundle([circle(r=5), circle(r=8)], [True]*2)
    req = request([feature('a', 'circle', 0), feature('b', 'circle', 1)], [
        ('curvature', ['a'], {}), ('wall_thickness', ['a', 'b'], {})])
    report = evaluate(c, req)
    assert all(m['status'] == 'measured' for m in report['measurements'])
    np.testing.assert_allclose(report['measurements'][0]['value'], .2, atol=1e-10)
    np.testing.assert_allclose(report['measurements'][1]['value'], 3, atol=.003)
