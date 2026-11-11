"""Analytic dimensions through the release's base/compound JSON handoff."""
from copy import deepcopy
import numpy as np
import pytest
from inspection_measurement import Contours, evaluate


def stadium_prediction(reverse=False, shift=0):
    radius, spacing, n = 7., 40., 48
    t = np.linspace(np.pi/2, -np.pi/2, n, endpoint=False)
    q = np.vstack([
        np.c_[np.linspace(-spacing/2, spacing/2, n, endpoint=False), np.full(n, radius)],
        np.c_[spacing/2 + radius*np.cos(t), radius*np.sin(t)],
        np.c_[np.linspace(spacing/2, -spacing/2, n, endpoint=False), np.full(n, -radius)],
        np.c_[-spacing/2 - radius*np.cos(t), -radius*np.sin(t)],
    ])
    angle = .63
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    q = q @ rotation.T + [235., 172.]
    indices = np.arange(len(q))
    if reverse:
        q = q[::-1]
        indices = (len(q)-2-indices) % len(q)
    q = np.roll(q, shift, axis=0)
    indices = (indices+shift) % len(q)
    c = Contours(q, [0, len(q)])
    bases = []
    for i, kind in enumerate(['line', 'arc', 'line', 'arc']):
        ids = indices[i*n:(i+1)*n].tolist()
        start = ids[-1] if reverse else ids[0]
        bases.append({'id': f'base_{i}', 'semantic': kind, 'semantic_id': 1 if kind == 'line' else 2,
                      'measurement_route': kind, 'status': 'accepted', 'confidence': .91,
                      'confidence_calibrated': False, 'raw_edge_indices': ids,
                      'contour_spans': [{'ring_index': 0, 'start_edge': start, 'edge_count': n}]})
    group = {'id': 'compound_0', 'semantic': 'full_round_slot', 'semantic_id': 1,
             'measurement_route': 'slot', 'base_instance_ids': [f['id'] for f in bases],
             'status': 'accepted', 'confidence': .8, 'confidence_calibrated': False,
             'raw_edge_indices': list(range(len(q)))}
    prediction = {'schema_version': 2, 'label_version': 'primitives-compounds-v1', 'units': 'px',
                  'coordinate_hash': c.coordinate_hash, 'features': bases, 'compound_features': [group],
                  'relations': [{'type': 'compound_membership', 'feature_a': f['id'], 'feature_b': group['id']} for f in bases]}
    return c, prediction


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('shift', [0, 27, 131])
def test_stadium_dimensions_and_membership_survive_handoff(reverse, shift):
    c, prediction = stadium_prediction(reverse, shift)
    original = deepcopy(prediction)
    report = evaluate(c, prediction)
    assert prediction == original
    assert all(f['status'] == 'fitted' for f in report['features']), report['features']
    assert report['relations'] == prediction['relations']
    assert report['features'][-1]['base_instance_ids'] == [f'base_{i}' for i in range(4)]
    assert report['features'][-1]['label_level'] == 'compound'
    values = {m['id']: m['value'] for m in report['measurements']}
    for ident, expected in {'base_0.length': 40, 'base_2.length': 40, 'base_1.radius': 7,
                            'base_3.radius': 7, 'compound_0.width': 14,
                            'compound_0.overall_length': 54, 'compound_0.end_radius': 7}.items():
        assert values[ident] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize('fault', ['unknown', 'duplicate', 'union', 'span'])
def test_bad_compound_support_does_not_discard_base_measurements(fault):
    c, prediction = stadium_prediction()
    group = prediction['compound_features'][0]
    if fault == 'unknown': group['base_instance_ids'][0] = 'missing'
    elif fault == 'duplicate': group['base_instance_ids'].append('base_0')
    elif fault == 'union': group['raw_edge_indices'].pop()
    else: group['contour_spans'] = [{'ring_index': 0, 'start_edge': 0, 'edge_count': 12}]
    report = evaluate(c, prediction)
    assert all(f['status'] == 'fitted' for f in report['features'][:-1])
    assert report['features'][-1]['status'] == 'fit_failed'
    assert report['measurements'][0]['status'] == 'measured'
    assert all(m['status'] == 'unavailable' for m in report['measurements'] if m['feature_refs'] == ['compound_0'])


def test_compound_acceptance_is_independent_of_base_abstention():
    c, prediction = stadium_prediction()
    prediction['features'][0]['status'] = 'abstain'
    report = evaluate(c, prediction)
    assert report['features'][0]['status'] == 'abstain'
    assert report['features'][0]['support']
    assert report['features'][-1]['status'] == 'fitted'
    assert report['measurements'][0]['status'] == 'unavailable'
    prediction['overrides'] = [{'id': 'base_0', 'source': 'user', 'accept': True}]
    assert evaluate(c, prediction)['features'][0]['status'] == 'fitted'


def test_thread_support_does_not_bridge_opposite_sides_or_invent_pitch():
    c = Contours(np.array([[0, 0], [10, 0], [10, 8], [0, 8]], float), [0, 4])
    bases = [{'id': f'b{i}', 'measurement_route': 'line', 'raw_edge_indices': [i], 'status': 'accepted'} for i in [0, 2]]
    prediction = {'schema_version': 2, 'label_version': 'primitives-compounds-v1', 'units': 'px',
                  'corner_exclusion_px': 0,
                  'features': bases, 'compound_features': [{'id': 'thread', 'measurement_route': 'thread_pitch',
                    'semantic': 'trapezoidal_thread', 'base_instance_ids': ['b0', 'b2'], 'raw_edge_indices': [0, 2]}]}
    report = evaluate(c, prediction)
    thread = report['features'][-1]
    assert len(thread['support']) == 2 and thread['closed'] == [False, False]
    assert 'Pitch requires' in thread['measurement_note']
    m = next(m for m in report['measurements'] if m['feature_refs'] == ['thread'])
    assert m['operator'] == 'length' and m['value'] == pytest.approx(20.)
