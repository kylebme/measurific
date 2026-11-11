"""Independent analytic fixtures, not labels from the training generators."""
import json
import numpy as np
import pytest

from contour_model.geometric import GeometricConfig, predict


def polygon(vertices, spacing=.2):
    vertices = np.asarray(vertices, float)
    return np.concatenate([np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b-a)/spacing))), endpoint=False)
                           for a, b in zip(vertices, np.roll(vertices, -1, axis=0))])


def rounded_box(w=40., h=24., r=3., samples=35):
    points = []
    for center, angle in [([w-r, h-r], 0), ([r, h-r], np.pi/2),
                          ([r, r], np.pi), ([w-r, r], 3*np.pi/2)]:
        t = np.linspace(angle, angle+np.pi/2, samples)
        arc = center+r*np.c_[np.cos(t), np.sin(t)]
        if points:
            points.append(np.linspace(points[-1][-1], arc[0], 61)[1:-1])
        points.append(arc)
    points.append(np.linspace(points[-1][-1], points[0][0], 61)[1:-1])
    return np.concatenate(points)


def arrays(*rings):
    return dict(actual_xy_mm=np.concatenate(rings), ring_offsets=np.r_[0, np.cumsum(list(map(len, rings)))])


def recognize(q, tolerance=.03):
    return predict(arrays(q), units='mm', config=GeometricConfig(tolerance=tolerance), compounds=False)[0]


def kinds(result):
    return [f['semantic'] for f in result['features']]


def assert_partition(result, n):
    ids = [i for f in result['features'] for i in f['raw_edge_indices']]
    assert sorted(ids) == list(range(n))
    assert all(f['fit_evidence']['max_error'] <= f['recognition_tolerance']*1.000001 for f in result['features'])
    json.dumps(result, allow_nan=False)


def test_hexagon_is_six_lines():
    t = np.arange(6)*np.pi/3
    q = polygon(20*np.c_[np.cos(t), np.sin(t)])
    result = recognize(q)
    assert kinds(result) == ['line']*6
    assert_partition(result, len(q))


def test_rounded_corners_and_raw_observations():
    q = rounded_box()
    before = q.copy()
    result = recognize(q)
    assert kinds(result).count('line') == 4
    assert kinds(result).count('arc') == 4
    assert_partition(result, len(q))
    np.testing.assert_array_equal(q, before)
    for f in result['features']:
        if f['semantic'] == 'arc':
            assert f['fit_evidence']['radius'] == pytest.approx(3, abs=.08)


@pytest.mark.parametrize('shift,reverse,angle', [(79, False, .37), (110, True, 1.2), (3, False, 2.1)])
def test_seam_winding_rotation_translation(shift, reverse, angle):
    q = rounded_box()
    q = np.roll(q, shift, axis=0)
    if reverse:
        q = q[::-1]
    r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    q = q@r.T+[10000, -21000]
    result = recognize(q)
    assert sorted(kinds(result)) == ['arc']*4+['line']*4
    assert_partition(result, len(q))


def test_circle_and_lobed_profile_are_not_the_same():
    t = np.linspace(0, 2*np.pi, 600, endpoint=False)
    circle = 20*np.c_[np.cos(t), np.sin(t)]
    assert kinds(recognize(circle)) == ['circle']
    lobed = (20+2*np.cos(5*t))[:, None]*np.c_[np.cos(t), np.sin(t)]
    result = recognize(lobed)
    assert 'circle' not in kinds(result)
    assert_partition(result, len(lobed))


def test_near_straight_deviation_below_tolerance_stays_line():
    q = polygon([[0, 0], [40, 0], [40, 20], [0, 20]])
    bottom = q[:, 1] == 0
    q[bottom, 1] += .015*np.sin(np.pi*q[bottom, 0]/40)
    assert kinds(recognize(q)) == ['line']*4


def test_oblique_bracket_and_chamfers():
    q = polygon([[0, 0], [35, 0], [37, 2], [35, 5], [12, 5], [20, 25], [15, 29]])
    result = recognize(q)
    assert kinds(result) == ['line']*7
    assert_partition(result, len(q))


def test_noisy_circle_and_label_blindness():
    t = np.linspace(0, 2*np.pi, 500, endpoint=False)
    q = (20+.01*np.sin(31*t))[:, None]*np.c_[np.cos(t), np.sin(t)]
    raw = arrays(q)
    first, _ = predict(raw, config=GeometricConfig(tolerance=.03), compounds=False)
    raw.update(nominal_xy_mm=np.zeros_like(q), edge_primitive_id=np.zeros(len(q)), family='hexagon')
    second, _ = predict(raw, config=GeometricConfig(tolerance=.03), compounds=False)
    assert first == second
    assert kinds(first) == ['circle']


@pytest.mark.parametrize('tolerance', [0, -1, np.nan, np.inf])
def test_bad_tolerance(tolerance):
    with pytest.raises(ValueError):
        GeometricConfig(tolerance=tolerance)


def test_float_offsets_rejected():
    raw = arrays(polygon([[0, 0], [10, 0], [10, 10]]))
    raw['ring_offsets'] = raw['ring_offsets'].astype(float)
    with pytest.raises(ValueError, match='integer'):
        predict(raw, compounds=False)


def stadium():
    t = np.linspace(-np.pi/2, np.pi/2, 90)
    right = [10, 0]+4*np.c_[np.cos(t), np.sin(t)]
    left = [-10, 0]+4*np.c_[np.cos(t+np.pi), np.sin(t+np.pi)]
    return np.vstack([right, np.linspace(right[-1], left[0], 80)[1:-1],
                      left, np.linspace(left[-1], right[0], 80)[1:-1]])


def threaded_outline(phase=.5, rounded=True):
    from scipy.ndimage import gaussian_filter1d
    x = np.linspace(0, 60, 1801)
    def profile(offset):
        phase = (x/5+offset) % 1
        y = 8+3*np.maximum(0, 1-np.abs(phase-.5)/.35)
        if rounded:
            y = gaussian_filter1d(y, 8)
        return y
    return np.vstack([np.c_[x, profile(0)], np.c_[x[::-1], -profile(phase)[::-1]]])


def test_slot_compound_and_measurement_handoff():
    measurement = pytest.importorskip('inspection_measurement')
    raw = arrays(polygon([[-30, -20], [30, -20], [30, 20], [-30, 20]]), stadium())
    result, _ = predict(raw, config=GeometricConfig(tolerance=.03))
    assert [f['semantic'] for f in result['compound_features']] == ['full_round_slot']
    contours = measurement.Contours(raw['actual_xy_mm'], raw['ring_offsets'])
    report, _ = measurement.prepare_interactive_report(contours, result)
    slot = next(f for f in report['features'] if f['type'] == 'slot')
    assert slot['status'] == 'fitted'
    assert slot['source'] == 'geometry'
    assert_partition(result, len(contours.xy))


@pytest.mark.parametrize('phase,expected', [(.5, True), (0, False)])
def test_thread_phase_and_rounded_roots(phase, expected):
    q = threaded_outline(phase)
    result, _ = predict(arrays(q), config=GeometricConfig(tolerance=.06))
    threads = [f for f in result['compound_features'] if 'thread' in f['semantic']]
    assert bool(threads) == expected
    turns = [f['fit_evidence']['sweep'] for f in result['features'] if f['semantic'] == 'arc']
    assert any(a > 0 for a in turns) and any(a < 0 for a in turns)
    if threads:
        assert threads[0]['rule_evidence']['pitch'] == pytest.approx(5., abs=.08)
        base = {f['id']: f for f in result['features']}
        chosen = threads[0]['base_instance_ids']
        assert set(threads[0]['raw_edge_indices']) == {i for f in chosen for i in base[f]['raw_edge_indices']}
    assert_partition(result, len(q))


def test_radial_repeat_without_training_labels():
    t = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    q = (20+2*np.cos(16*t))[:, None]*np.c_[np.cos(t), np.sin(t)]
    result, _ = predict(arrays(q), config=GeometricConfig(tolerance=.05))
    assert [f['semantic'] for f in result['compound_features']] == ['tooth_profile']
    assert result['compound_features'][0]['rule_evidence']['count'] == 16


def test_rounded_hex_nut_with_circular_bore():
    from shapely.geometry import Polygon
    t = np.arange(6)*np.pi/3
    exterior = Polygon(15*np.c_[np.cos(t), np.sin(t)]).buffer(1, quad_segs=24)
    q = np.asarray(exterior.exterior.coords)[:-1]
    t = np.linspace(0, 2*np.pi, 160, endpoint=False)
    bore = 5*np.c_[np.cos(t), np.sin(t)]
    result, _ = predict(arrays(q, bore), config=GeometricConfig(tolerance=.015))
    assert sorted(kinds(result)) == ['arc']*6+['circle']+['line']*6
    assert_partition(result, len(q)+len(bore))


def test_arc_line_concave_fillets():
    from shapely.geometry import Point, box
    from shapely.ops import unary_union
    outline = unary_union([Point(-15, 0).buffer(7, quad_segs=64),
                           Point(15, 0).buffer(7, quad_segs=64), box(-15, -2, 15, 2)])
    outline = outline.buffer(1, quad_segs=32).buffer(-1, quad_segs=32)
    q = np.asarray(outline.exterior.coords)[:-1]
    result = recognize(q, tolerance=.02)
    arcs = [f['fit_evidence'] for f in result['features'] if f['semantic'] == 'arc']
    assert kinds(result).count('line') == 2
    assert any(a['sweep'] > 0 for a in arcs) and any(a['sweep'] < 0 for a in arcs)
    assert sum(.85 < a['radius'] < 1.15 for a in arcs) == 4
    assert_partition(result, len(q))


def test_physical_units_and_sampling_density():
    q = rounded_box(samples=65)
    original = recognize(q, .03)
    scaled = recognize(q*100, 3)
    sparse = recognize(rounded_box(samples=20), .03)
    assert sorted(kinds(original)) == sorted(kinds(scaled)) == sorted(kinds(sparse))


def test_strict_tolerance_does_not_hide_a_notch_in_a_circle():
    t = np.linspace(0, 2*np.pi, 500, endpoint=False)
    radius = 20-np.exp(-((t-np.pi)/.08)**2)
    q = radius[:, None]*np.c_[np.cos(t), np.sin(t)]
    result = recognize(q, .03)
    assert 'circle' not in kinds(result)
    assert_partition(result, len(q))


def test_geometry_import_does_not_load_torch():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c',
                             'import sys; import contour_model.geometric; assert "torch" not in sys.modules'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_ring_order_does_not_imply_hole_identity():
    t = np.linspace(0, 2*np.pi, 120, endpoint=False)
    hole = 3*np.c_[np.cos(t), np.sin(t)]+[15, 12]
    result, _ = predict(arrays(hole, rounded_box()), config=GeometricConfig(tolerance=.03))
    assert result['part']['ring_is_hole'] == [True, False]
    assert kinds(result).count('circle') == 1
    assert kinds(result).count('line') == 4
    assert_partition(result, len(hole)+len(rounded_box()))


def test_sparse_polygon_is_not_circle_and_tiny_loops_are_not_closed_lines():
    triangle = np.array([[0., 0.], [10., 0.], [5., 8.]])
    for tolerance in [.01, 100.]:
        result = recognize(triangle, tolerance)
        assert 'circle' not in kinds(result)
        assert all(not f['contour_spans'][0]['closed_ring'] for f in result['features'])
        assert_partition(result, len(triangle))
