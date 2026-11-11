import numpy as np
import pytest
from inspection_measurement.geometry import Geometry, fit_geometry
from inspection_measurement.operators import measure


def geom(kind, runs, closed=None):
    g = Geometry('test', kind, [np.array(q, float) for q in runs], closed or [False]*len(runs))
    return fit_geometry(g)


def line(a, b):
    return geom('line', [[a, b]])


def distance(a, b, **params):
    return measure('distance_construction', [a, b], dict(construction='normal_gap', **params))


def test_offset_parallel_segments_use_shared_sections_not_diagonal():
    r = distance(line([0, 0], [10, 0]), line([5, 4], [20, 4]))
    assert [r[k] for k in ('minimum', 'maximum', 'average')] == [4, 4, 4]
    assert r['coverage'] == .5
    np.testing.assert_allclose(np.diff(r['maximum_witness'], axis=0), [[0, 4]])


def test_no_overlap_does_not_fall_back_to_endpoints():
    a, b = line([0, 0], [5, 0]), line([10, 4], [15, 4])
    with pytest.raises(ValueError, match='No common supported'):
        distance(a, b)
    r = distance(a, b, basis='fitted', extend=True, station_range=[0, 15])
    assert r['minimum'] == r['maximum'] == 4
    assert r['definition']['extend']
    with pytest.raises(ValueError, match='explicit finite'):
        distance(a, b, basis='fitted', extend=True)


def test_interval_weighted_mean_preserves_gaps_and_density():
    a = geom('line', [[[0, 0], [2, 0]], [[6, 0], [10, 0]]])
    b = line([0, 1], [10, 11])
    r = distance(a, b)
    assert r['coverage'] == pytest.approx(.6)
    assert r['average'] == pytest.approx((2*2+4*9)/6)
    assert r['excluded_intervals'] == [[2, 6]]
    dense = geom('line', [[[x, x+1] for x in [0, .1, .2, .3, 1, 7, 10]]])
    assert distance(a, dense)['average'] == pytest.approx(r['average'])
    assert 'witness' not in distance(a, b, statistic='average')


def test_signed_crossing_and_custom_direction_construct_intersections():
    a, b = line([0, 0], [10, 0]), line([0, -2], [10, 2])
    r = distance(a, b, direction_mode='y')
    assert r['minimum'] == -2 and r['maximum'] == 2 and r['crosses']
    assert r['average'] == pytest.approx(0)
    r = distance(a, line([0, 4], [20, 4]), direction_mode='custom', direction=[1, 1])
    np.testing.assert_allclose(np.diff(r['witness'], axis=0), [[4, 4]])
    assert r['value'] == pytest.approx(np.sqrt(32))


def circle(cx=5, cy=5, r=2, arc=False):
    t = np.linspace(0, np.pi if arc else 2*np.pi, 101, endpoint=arc)
    return geom('arc' if arc else 'circle', [np.c_[cx+r*np.cos(t), cy+r*np.sin(t)]], [not arc])


def test_fitted_circle_exact_extrema_mean_and_branch():
    a, b = line([0, 0], [10, 0]), circle()
    with pytest.raises(ValueError, match='Multiple surface'):
        distance(a, b, basis='fitted')
    r = distance(a, b, basis='fitted', branches=['auto', 'near'])
    assert r['minimum'] == pytest.approx(3)
    assert r['maximum'] == pytest.approx(5)
    assert r['average'] == pytest.approx(5-np.pi/2)
    assert r['coverage'] == pytest.approx(.4)
    np.testing.assert_allclose(r['minimum_witness'], [[5, 0], [5, 3]], atol=1e-7)
    far = distance(a, b, basis='fitted', branches=['auto', 'far'])
    assert far['maximum'] == pytest.approx(7)


def test_arc_support_is_not_a_whole_circle():
    a, b = line([0, 0], [10, 0]), circle(arc=True)
    r = distance(a, b, basis='fitted')
    assert r['minimum'] == pytest.approx(5)
    assert r['maximum'] == pytest.approx(7)
    with pytest.raises(ValueError, match='outside the visible arc'):
        measure('distance_construction', [a, b], {'construction': 'tangent_clearance'})


def test_scalar_constructions_do_not_invent_statistics():
    a, b = line([0, 0], [10, 0]), circle()
    for construction, expected in [('center_to_line', 5), ('tangent_clearance', 3)]:
        r = measure('distance_construction', [a, b], {'construction': construction})
        assert r['value'] == pytest.approx(expected)
        assert 'average' not in r
    r = measure('distance_construction', [circle(), circle(15)], {})
    assert r['value'] == pytest.approx(10)
    assert r['definition']['construction'] == 'center_distance'
    r = measure('distance_construction', [line([0, 0], [1, 0]), line([4, 4], [5, 4])], {'construction': 'closest_clearance'})
    assert r['value'] == pytest.approx(5)


def test_section_single_station_and_rotation():
    a, b = line([0, 0], [10, 0]), line([0, 2], [10, 12])
    r = measure('distance_construction', [a, b], {'construction': 'section', 'station': 3})
    assert r['value'] == pytest.approx(5)
    rotation = np.array([[.6, -.8], [.8, .6]])
    a, b = [geom('line', [q@rotation.T for q in g.runs]) for g in (a, b)]
    r = distance(a, b, direction_mode='custom', direction=rotation@[0, 1])
    assert r['minimum'] == pytest.approx(2)
    assert r['maximum'] == pytest.approx(12)
    assert r['average'] == pytest.approx(7)


def test_branch_crossing_event_is_not_missed():
    a = line([0, 0], [10, 0])
    b = geom('contour', [[[0, 2], [10, 12]], [[0, 12], [10, 2]]])
    r = distance(a, b, branches=['auto', 'negative'])
    assert r['minimum'] == 2 and r['maximum'] == 7
    assert r['average'] == pytest.approx(4.5)


def test_single_section_at_touching_support_endpoints():
    r = measure('distance_construction', [line([0, 0], [5, 0]), line([5, 4], [10, 4])],
                {'construction': 'section', 'station': 5})
    assert r['value'] == 4
    with pytest.raises(ValueError, match='coincides'):
        measure('distance_construction', [line([0, 0], [5, 0]),
                geom('contour', [[[0, 4], [5, 4], [5, 8], [10, 8]]])], {'construction': 'section', 'station': 5})


def test_fitted_circle_pair_stationary_points_match_dense_oracle():
    a, b = circle(5, 5, 3), circle(6, 12, 2)
    for branches in [['positive', 'negative'], ['negative', 'negative'], ['positive', 'positive']]:
        r = distance(a, b, basis='fitted', direction_mode='y', branches=branches)
        s = np.linspace(4, 8, 200001)
        va = 5 + (1 if branches[0] == 'positive' else -1)*np.sqrt(np.maximum(0, 9-(s-5)**2))
        vb = 12 + (1 if branches[1] == 'positive' else -1)*np.sqrt(np.maximum(0, 4-(s-6)**2))
        assert r['minimum'] == pytest.approx(np.min(vb-va), abs=1e-7)
        assert r['maximum'] == pytest.approx(np.max(vb-va), abs=1e-7)


def point(x, y):
    return Geometry('p', 'point', values={'center': np.array([x, y], float)})


def test_point_to_line_accepts_any_center_feature_in_either_order():
    a = line([0, 0], [10, 0])
    for pair in ([a, point(3, 4)], [point(3, 4), a]):
        r = measure('distance_construction', pair, {'construction': 'center_to_line'})
        assert r['value'] == pytest.approx(4)
        np.testing.assert_allclose(r['witness'], [[3, 0], [3, 4]], atol=1e-9)
    ellipse = geom('ellipse', [np.c_[5+3*np.cos(np.linspace(0, 2*np.pi, 200, endpoint=False)),
                                     7+np.sin(np.linspace(0, 2*np.pi, 200, endpoint=False))]], [True])
    r = measure('distance_construction', [ellipse, a], {'construction': 'center_to_line'})
    assert r['value'] == pytest.approx(7, abs=1e-6)
    with pytest.raises(ValueError, match='one line and one point'):
        measure('distance_construction', [a, line([0, 5], [10, 5])], {'construction': 'center_to_line'})


def test_point_to_line_circle_edges_and_extension():
    a, b = line([0, 0], [10, 0]), circle()
    near, far = [measure('distance_construction', [b, a], {'construction': 'center_to_line', 'anchors': [anchor, 'center']})
                 for anchor in ('near', 'far')]
    assert near['value'] == pytest.approx(3) and near['signed']
    assert far['value'] == pytest.approx(7) and 'signed' not in far
    np.testing.assert_allclose(far['witness'], [[5, 0], [5, 7]], atol=1e-9)
    overlap = measure('distance_construction', [a, circle(5, 1, 2)], {'construction': 'center_to_line', 'anchors': ['center', 'near']})
    assert overlap['value'] == pytest.approx(-1)
    upper = circle(5, 5, 2, arc=True)
    assert measure('distance_construction', [a, upper], {'construction': 'center_to_line', 'anchors': ['center', 'far']})['value'] == pytest.approx(7)
    with pytest.raises(ValueError, match='outside the visible arc'):
        measure('distance_construction', [a, upper], {'construction': 'center_to_line', 'anchors': ['center', 'near']})
    with pytest.raises(ValueError, match='need a fitted circle'):
        measure('distance_construction', [a, point(3, 4)], {'construction': 'center_to_line', 'anchors': ['center', 'near']})
    beyond = point(15, 4)
    with pytest.raises(ValueError, match='outside supported line spans'):
        measure('distance_construction', [a, beyond], {'construction': 'center_to_line'})
    r = measure('distance_construction', [a, beyond], {'construction': 'center_to_line', 'extend': True})
    assert r['value'] == pytest.approx(4) and r['definition']['extend']
    np.testing.assert_allclose(r['virtual_segments'], [[[10, 0], [15, 0]]], atol=1e-9)


def test_point_to_point_axis_components_and_circle_edges():
    a, b = circle(0, 0, 2), circle(10, 5, 3)
    direct = measure('distance_construction', [a, b], {'construction': 'center_distance'})
    assert direct['value'] == pytest.approx(np.hypot(10, 5))
    assert direct['definition']['anchors'] == ['center', 'center']
    wall = measure('distance_construction', [a, b], {'construction': 'center_distance', 'anchors': ['near', 'near']})
    assert wall['value'] == pytest.approx(np.hypot(10, 5)-5) and wall['signed']
    span = measure('distance_construction', [a, b], {'construction': 'center_distance', 'anchors': ['far', 'far']})
    assert span['value'] == pytest.approx(np.hypot(10, 5)+5)
    dx = measure('distance_construction', [b, a], {'construction': 'center_distance', 'direction_mode': 'x'})
    assert dx['value'] == pytest.approx(10)
    np.testing.assert_allclose(dx['dimension_direction'], [-1, 0])
    dy = measure('distance_construction', [a, b], {'construction': 'center_distance', 'direction_mode': 'y', 'anchors': ['far', 'near']})
    assert dy['value'] == pytest.approx(5+2-3)
    np.testing.assert_allclose(dy['witness'], [[0, -2], [10, 2]], atol=1e-9)
    corner_to_edge = measure('distance_construction', [point(0, 0), circle(8, 6, 2)], {'construction': 'center_distance', 'anchors': ['center', 'near']})
    assert corner_to_edge['value'] == pytest.approx(8)
    with pytest.raises(ValueError, match='choose a measurement direction'):
        measure('distance_construction', [circle(), circle(5, 5, 1)], {'construction': 'center_distance', 'anchors': ['near', 'center']})
    with pytest.raises(ValueError, match='direct, x, y or custom'):
        measure('distance_construction', [a, b], {'construction': 'center_distance', 'direction_mode': 'normal_a'})
