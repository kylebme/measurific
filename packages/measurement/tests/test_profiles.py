import json
import numpy as np
import pytest
from inspection_measurement import Contours, evaluate
from inspection_measurement.interaction import (prepare_interactive_report, evaluate_interactive_measurement,
                                                redefine_compound)


def gear(teeth=12, root=40., tip=50., center=(200., 150.), samples=10):
    """Trapezoidal radial teeth in increasing polar angle, starting mid root land."""
    fractions = [(0, 0), (.2, 0), (.4, 1), (.6, 1), (.8, 0), (1, 0)]
    points = []
    for n in range(teeth):
        for (a, b), (c, d) in zip(fractions[:-1], fractions[1:]):
            for t in np.linspace(0, 1, samples, endpoint=False):
                angle = 2*np.pi*(n+a+t*(c-a))/teeth
                r = root+(tip-root)*(b+t*(d-b))
                points.append([center[0]+r*np.cos(angle), center[1]+r*np.sin(angle)])
    return np.asarray(points)


def square(lo, hi):
    return np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]], float)


def measured(contours, features, operator, ref):
    report, session = prepare_interactive_report(contours, {'features': features, 'measurements': []})
    result = evaluate_interactive_measurement(contours, session, report, {'operator': operator, 'featureRefs': [ref]})
    json.dumps(result, allow_nan=False)
    return result


def assert_metric(metrics, name, expected, abs=1e-6):
    assert metrics[name]['status'] == 'measured', (name, metrics[name])
    for stat in ('minimum', 'maximum', 'average'):
        assert metrics[name][stat] == pytest.approx(expected, abs=abs), (name, metrics[name])


@pytest.mark.parametrize('reverse', [False, True])
def test_external_gear_tooth_profile(reverse):
    ring = gear()
    contours = Contours(ring[::-1] if reverse else ring, [0, len(ring)], [True])
    features = [{'id': 'teeth', 'type': 'repeated_profile', 'semantic': 'tooth_profile', 'selection': {'path': 0}}]
    result = measured(contours, features, 'tooth_profile', 'teeth')
    metrics = result['compound_metrics']
    assert result['domain'] == 'angular' and result['units'] == 'deg'
    np.testing.assert_allclose(result['profile_center'], [200, 150], atol=1e-6)
    assert_metric(metrics, 'number_of_teeth', 12)
    assert_metric(metrics, 'pitch', 30)
    assert_metric(metrics, 'tooth_height', 10)
    assert_metric(metrics, 'tip_diameter', 100)
    assert_metric(metrics, 'root_diameter', 80)
    witness = np.asarray(metrics['tip_diameter']['maximum_witness'])
    assert np.linalg.norm(witness[1]-witness[0]) == pytest.approx(100)
    # Sharp polyline turns are not arcs, so no radius is reported.
    assert metrics['tip_radius']['status'] == metrics['root_radius']['status'] == 'unavailable'
    assert result['support_policy'] == 'original_selected_observations_including_corners'


def test_internal_gear_tips_point_toward_center():
    outer, hole = square(100, 300), gear(center=(200., 200.))
    contours = Contours(np.vstack([outer, hole[::-1]]), [0, 4, 4+len(hole)], [True, True])
    features = [{'id': 'teeth', 'type': 'repeated_profile', 'semantic': 'tooth_profile', 'selection': {'path': 1}}]
    metrics = measured(contours, features, 'tooth_profile', 'teeth')['compound_metrics']
    assert_metric(metrics, 'number_of_teeth', 12)
    assert_metric(metrics, 'tip_diameter', 80)
    assert_metric(metrics, 'root_diameter', 100)
    assert_metric(metrics, 'tooth_height', 10)


def test_rack_tooth_profile_on_one_edge():
    # Rectangle whose top edge (image y=20) carries eight triangular teeth rising to y=14.
    top = [[100-5*i, 20 if i % 2 == 0 else 14] for i in range(17)]
    edge = np.concatenate([np.linspace(a, b, 8, endpoint=False) for a, b in zip(top[:-1], top[1:])] + [[top[-1]]])
    ring = np.vstack([[[0, 60], [100, 60]], edge])
    contours = Contours(ring, [0, len(ring)], [True])
    teeth = list(range(2, 2+len(edge)-1))
    features = [{'id': 'teeth', 'type': 'repeated_profile', 'semantic': 'tooth_profile',
                 'selection': {'edge_indices': teeth}}]
    result = measured(contours, features, 'tooth_profile', 'teeth')
    metrics = result['compound_metrics']
    assert result['domain'] == 'linear' and result['units'] == 'px'
    assert_metric(metrics, 'number_of_teeth', 8)
    assert_metric(metrics, 'pitch', 10)
    assert_metric(metrics, 'tooth_height', 6)
    witness = np.asarray(metrics['tooth_height']['minimum_witness'])
    assert witness[0][1] == pytest.approx(14) and witness[1][1] == pytest.approx(20)
    assert metrics['tip_diameter']['status'] == 'unavailable'
    assert 'no center' in metrics['tip_diameter']['reason']


def test_tooth_profile_rejects_other_features():
    ring = gear()
    contours = Contours(ring, [0, len(ring)], [True])
    features = [{'id': 'profile', 'type': 'repeated_profile', 'semantic': 'thread', 'selection': {'path': 0}}]
    with pytest.raises(ValueError, match='tooth profile compound'):
        measured(contours, features, 'tooth_profile', 'profile')


def test_full_round_slot_measurements():
    t = np.linspace(-np.pi/2, np.pi/2, 24)
    right = np.c_[10+5*np.cos(t), 5*np.sin(t)]
    left = np.c_[-10-5*np.cos(t), -5*np.sin(t)]
    top = np.c_[np.linspace(10, -10, 12)[1:-1], np.full(10, 5.)]
    bottom = np.c_[np.linspace(-10, 10, 12)[1:-1], np.full(10, -5.)]
    ring = np.vstack([right, top, left, bottom])+100
    contours = Contours(ring, [0, len(ring)], [True])
    spans = [(0, 24), (24, 34), (34, 58), (58, 68)]
    request = {'schema_version': 2, 'label_version': 'primitives-compounds-v2', 'corner_exclusion_px': 0,
               'features': [{'id': f'base_{i}', 'type': kind, 'raw_edge_indices': list(range(lo, hi))}
                            for i, ((lo, hi), kind) in enumerate(zip(spans, ['arc', 'line', 'arc', 'line']))],
               'compound_features': [], 'relations': []}
    _, session = prepare_interactive_report(contours, request)
    grouped = redefine_compound(contours, session, 'full_round_slot', ['base_0', 'base_1', 'base_2', 'base_3'])
    result = evaluate_interactive_measurement(contours, grouped['request'], grouped['report'],
                                              {'operator': 'slot', 'featureRefs': [grouped['feature_id']]})
    json.dumps(result, allow_nan=False)
    metrics = result['compound_metrics']
    assert metrics['width']['method'] == 'observed_normal_gap_between_flats'
    assert_metric(metrics, 'width', 10, abs=1e-6)
    # Each end arc's span includes the first vertex of the next flat.
    assert_metric(metrics, 'end_radius', 5, abs=.1)
    assert metrics['end_radius']['count'] == 2
    assert_metric(metrics, 'overall_length', 30, abs=.1)
    assert_metric(metrics, 'end_center_spacing', 20, abs=.1)
    assert result['value'] == pytest.approx(10)
    with pytest.raises(ValueError, match='full round slot'):
        evaluate_interactive_measurement(contours, grouped['request'], grouped['report'],
                                         {'operator': 'slot', 'featureRefs': ['base_1']})


def test_size_measurements_report_dimension_witnesses():
    angle = np.linspace(0, 2*np.pi, 90, endpoint=False)
    circle = np.c_[50+10*np.cos(angle), 50+10*np.sin(angle)]
    rectangle = np.array([[100, 20], [120, 20], [140, 20], [160, 20], [160, 60], [100, 60]], float)
    sweep = np.linspace(0, np.pi/2, 30)
    arc = np.c_[200+8*np.cos(sweep), 50+8*np.sin(sweep)]
    contours = Contours(np.vstack([circle, rectangle, arc]), [0, 90, 96, 126], [True, True, False])
    features = [{'id': 'hole', 'type': 'circle', 'selection': {'path': 0}},
                {'id': 'edge', 'type': 'line', 'selection': {'edge_indices': [90, 91, 92]}},
                {'id': 'fillet', 'type': 'arc', 'selection': {'path': 2}}]
    diameter = measured(contours, features, 'least_squares_diameter', 'hole')
    assert diameter['value'] == pytest.approx(20)
    a, b = np.asarray(diameter['witness'])
    assert np.linalg.norm(b-a) == pytest.approx(20) and (a+b)/2 == pytest.approx([50, 50])
    radius = measured(contours, features, 'radius', 'fillet')
    assert radius['value'] == pytest.approx(8)
    a, b = np.asarray(radius['witness'])
    np.testing.assert_allclose(a, [200, 50], atol=1e-6)
    np.testing.assert_allclose(b, [200+8*np.cos(np.pi/4), 50+8*np.sin(np.pi/4)], atol=1e-6)
    length = measured(contours, features, 'length', 'edge')
    assert length['value'] == pytest.approx(60)
    np.testing.assert_allclose(sorted(np.asarray(length['witness'])[:, 0]), [100, 160])
    # Corner exclusion trims the fitted line's support but not its corner-to-corner length.
    assert length['support_policy'] == 'original_selected_observations_including_corners'
    trimmed = next(f for f in prepare_interactive_report(contours, {'features': features, 'measurements': []})[0]['features']
                   if f['id'] == 'edge')
    assert trimmed['geometry']['length'] < 60
    span = measured(contours, features, 'angular_span', 'fillet')
    assert span['value'] == pytest.approx(90) and span['units'] == 'deg'
    with pytest.raises(ValueError, match='circle or arc'):
        measured(contours, features, 'radius', 'edge')
    with pytest.raises(ValueError, match='must be a circle'):
        measured(contours, features, 'least_squares_diameter', 'fillet')


def test_batch_requests_accept_new_operators():
    ring = gear()
    contours = Contours(ring, [0, len(ring)], [True])
    report = evaluate(contours, {'features': [
        {'id': 'teeth', 'type': 'repeated_profile', 'semantic': 'tooth_profile', 'selection': {'path': 0}}],
        'measurements': [{'id': 'm', 'operator': 'tooth_profile', 'feature_refs': ['teeth'],
                          'params': {'domain': 'angular', 'center': [200, 150]}}]})
    assert report['measurements'][0]['status'] == 'measured'
    assert report['measurements'][0]['compound_metrics']['number_of_teeth']['average'] == 12


def test_tooth_radii_come_from_rounding_arcs_only():
    # Rack teeth with semicircular tips (R2) on a rectangle's top edge, pitch 10.
    points, arcs = [], []
    for n in range(6):
        x0 = 95-10*n
        points += [[x0-t, 20] for t in np.linspace(0, 2, 3, endpoint=False)]
        points += [[x0-2-t, 20-6*t/2] for t in np.linspace(0, 1, 4, endpoint=False)]
        start = len(points)
        sweep = np.linspace(0, np.pi, 13)
        points += [[x0-5+2*np.cos(a), 14-2*np.sin(a)+2] for a in sweep[:-1]]
        arcs.append((start, len(points)))
        points += [[x0-7-t, 14+6*t/2] for t in np.linspace(0, 1, 4, endpoint=False)]
        points += [[x0-8-t, 20] for t in np.linspace(0, 2, 3, endpoint=False)]
    edge = np.asarray(points)
    ring = np.vstack([[[0, 60], [100, 60], [100, 20]], edge])
    contours = Contours(ring, [0, len(ring)], [True])
    offset = 3
    request = {'schema_version': 2, 'label_version': 'primitives-compounds-v2', 'corner_exclusion_px': 0,
               'features': [{'id': f'tip_{i}', 'type': 'arc', 'raw_edge_indices': list(range(offset+lo, offset+hi-1))}
                            for i, (lo, hi) in enumerate(arcs)],
               'compound_features': [], 'relations': []}
    _, session = prepare_interactive_report(contours, request)
    grouped = redefine_compound(contours, session, 'tooth_profile', [f'tip_{i}' for i in range(6)])
    compound = next(f for f in grouped['request']['compound_features'] if f['id'] == grouped['feature_id'])
    compound['raw_edge_indices'] = list(range(offset, len(ring)-1))
    report, prepared = prepare_interactive_report(contours, grouped['request'])
    result = evaluate_interactive_measurement(contours, prepared, report,
                                              {'operator': 'tooth_profile', 'featureRefs': [grouped['feature_id']]})
    metrics = result['compound_metrics']
    assert_metric(metrics, 'tip_radius', 2, abs=1e-6)
    assert metrics['tip_radius']['count'] == 6
    assert metrics['root_radius']['status'] == 'unavailable'
    # The fixture's flanks meet the tip arcs with small steps.
    assert_metric(metrics, 'pitch', 10, abs=.02)
