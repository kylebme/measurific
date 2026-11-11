import numpy as np
import pytest

from inspection_measurement import (Contours, evaluate_interactive_measurement,
                                    prepare_interactive_report, redefine_compound, redefine_compound_sides,
                                    redefine_contour_geometry, remove_compound)


def test_interactive_report_adds_shared_segment_vertex_and_deformation_contract():
    first = np.array([[0., 0.], [5., .1], [10., 0.]])
    second = np.array([[10., 0.], [10.1, 5.], [10., 10.]])
    contours = Contours(np.vstack([first, second]), [0, 3, 6], [False, False])
    request = {'features': [
        {'id': 'horizontal', 'type': 'line', 'selection': {'path': 0}},
        {'id': 'vertical', 'type': 'line', 'selection': {'path': 1}},
    ]}
    report, session = prepare_interactive_report(contours, request)
    vertex = next(feature for feature in report['features'] if feature['type'] == 'point')
    assert vertex['adjacent_feature_ids'] == ['horizontal', 'vertical']
    np.testing.assert_allclose(vertex['geometry']['center'], [10, 0], atol=.2)
    assert report['interaction']['deformation']['max_factor'] > 1
    assert next(f for f in report['features'] if f['id'] == 'horizontal')['ideal_support']

    distance = evaluate_interactive_measurement(contours, session, report, {
        'operator': 'distance_construction', 'featureRefs': [vertex['id'], 'horizontal'],
        'params': {'construction': 'closest_clearance'},
    })
    assert distance['status'] == 'measured'
    assert distance['value'] < .2


def test_junction_vertex_stays_on_the_contour_where_a_long_line_bows():
    # A long edge that is not quite straight meets a short flank at a sharp
    # corner at (300, 0). Its whole-line fit is pixels off at that end; the
    # vertex must use the lines near the corner.
    x = np.arange(0., 301.)
    top = np.c_[x, 3*np.sin(np.pi*x/300)]
    flank = np.c_[300+np.arange(1., 11.), np.arange(1., 11.)]
    xy = np.vstack([top, flank])
    contours = Contours(xy, [0, len(xy)], [False])
    request = {'corner_exclusion_px': 0, 'features': [
        {'id': 'top', 'type': 'line', 'selection': {'edge_indices': list(range(300))}},
        {'id': 'flank', 'type': 'line', 'selection': {'edge_indices': list(range(300, len(xy)-1))}}]}
    report, _ = prepare_interactive_report(contours, request)
    [vertex] = _vertices(report)
    assert vertex['adjacent_feature_ids'] == ['flank', 'top']
    np.testing.assert_allclose(vertex['geometry']['center'], [300, 0], atol=.25)
    # The line feature itself is still fitted to its entire segment.
    top_line = next(f for f in report['features'] if f['id'] == 'top')
    assert len(top_line['support'][0]) == len(top)


def _l_corner(split):
    """Two lines meeting at a right-angle corner at (20, 0); the model split at point ``split``."""
    xy = np.vstack([np.c_[np.arange(21.), np.zeros(21)], np.c_[np.full(20, 20.), np.arange(1., 21.)]])
    contours = Contours(xy, [0, len(xy)], [False])
    features = [{'id': 'horizontal', 'type': 'line', 'selection': {'edge_indices': list(range(0, split))}},
                {'id': 'vertical', 'type': 'line', 'selection': {'edge_indices': list(range(split, len(xy)-1))}}]
    return contours, features


def _vertices(report):
    return [f for f in report['features'] if f.get('semantic') == 'vertex']


def test_corner_exclusion_keeps_a_vertex_at_each_trimmed_segment_end():
    contours, features = _l_corner(20)
    report, _ = prepare_interactive_report(contours, {'corner_exclusion_px': 0, 'features': features})
    [junction] = _vertices(report)
    assert junction['adjacent_feature_ids'] == ['horizontal', 'vertical']
    np.testing.assert_allclose(junction['geometry']['center'], [20, 0], atol=1e-6)

    report, _ = prepare_interactive_report(contours, {'corner_exclusion_px': 1, 'features': features})
    lines = {f['id']: f for f in report['features'] if f['type'] == 'line'}
    assert all(len(lines[name]['excluded_edge_indices']) for name in lines)
    vertices = _vertices(report)
    assert sorted(v['adjacent_feature_ids'][0] for v in vertices) == ['horizontal', 'vertical']
    for vertex in vertices:
        [member] = vertex['adjacent_feature_ids']
        support = lines[member]['support'][0]
        # At the retained end facing the corner, not at the far end of the line.
        corner_end = min((support[0], support[-1]), key=lambda end: np.linalg.norm(np.subtract(end, [20, 0])))
        np.testing.assert_allclose(vertex['geometry']['center'], corner_end)
        assert np.linalg.norm(np.subtract(corner_end, [20, 0])) <= 2


def test_untrimmed_segment_facing_a_trimmed_one_keeps_its_own_vertex():
    # The model split where the excluded edges end, so exclusion trims only the
    # horizontal line; the vertical line's end is observed but has no partner.
    contours, features = _l_corner(22)
    report, _ = prepare_interactive_report(contours, {'corner_exclusion_px': 1, 'features': features})
    lines = {f['id']: f for f in report['features'] if f['type'] == 'line'}
    assert len(lines['horizontal']['excluded_edge_indices']) and not len(lines['vertical']['excluded_edge_indices'])
    vertices = {v['adjacent_feature_ids'][0]: v for v in _vertices(report)}
    assert sorted(vertices) == ['horizontal', 'vertical']
    np.testing.assert_allclose(vertices['vertical']['geometry']['center'], [20, 2])


def test_planar_datum_and_concentricity_are_explicit_measurements():
    t = np.linspace(0, 2*np.pi, 64, endpoint=False)
    circles = [np.c_[np.cos(t), np.sin(t)]*radius + center
               for radius, center in [(2, [0, 0]), (3, [1, 0])]]
    line = np.array([[-4., 5.], [4., 5.]])
    contours = Contours(np.vstack([*circles, line]), [0, 64, 128, 130], [True, True, False])
    request = {'features': [
        {'id': 'a', 'type': 'circle', 'selection': {'path': 0}},
        {'id': 'b', 'type': 'circle', 'selection': {'path': 1}},
        {'id': 'datum_line', 'type': 'line', 'selection': {'path': 2}},
    ]}
    report, session = prepare_interactive_report(contours, request)
    datum = evaluate_interactive_measurement(contours, session, report,
                                             {'operator': 'datum', 'featureRefs': ['datum_line']})
    center_datum = evaluate_interactive_measurement(contours, session, report,
                                                    {'operator': 'datum', 'featureRefs': ['a']})
    concentricity = evaluate_interactive_measurement(contours, session, report,
        {'operator': 'concentricity', 'featureRefs': ['a', 'b']})
    assert datum['value'] == 0
    assert datum['datum']['type'] == 'line'
    assert center_datum['datum']['type'] == 'center'
    np.testing.assert_allclose(center_datum['datum']['origin'], [0, 0], atol=1e-9)
    np.testing.assert_allclose(concentricity['value'], 1, atol=1e-9)


def test_geometry_redefinition_snaps_and_uses_ordered_same_ring_interval():
    square = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])
    other = square+30
    contours = Contours(np.vstack([square, other]), [0, 4, 8])
    request = {'corner_exclusion_px': 0, 'features': [
        {'id': 'old', 'type': 'arc', 'selection': {'path': 0}},
        {'id': 'other', 'type': 'circle', 'selection': {'path': 1}},
    ]}
    _, session = prepare_interactive_report(contours, request)
    preview = redefine_contour_geometry(contours, session, 'line', [[4., -2.]])
    assert preview == {'complete': False, 'snapped_points': [[4., 0.]], 'ring_index': 0}

    result = redefine_contour_geometry(contours, session, 'line', [[4., -2.], [12., 7.]])
    assert result['complete'] is True
    assert result['snapped_points'] == [[4., 0.], [10., 7.]]
    feature = next(f for f in result['report']['features'] if f['id'] == result['feature_id'])
    assert feature['type'] == 'line' and feature['source'] == 'user'
    np.testing.assert_allclose(np.asarray(feature['support'][0])[[0, -1]], [[4, 0], [10, 7]])
    assert next(f for f in result['report']['features'] if f['id'] == 'other')['type'] == 'circle'


def test_circle_redefinition_uses_entire_nearest_closed_ring():
    t = np.linspace(0, 2*np.pi, 32, endpoint=False)
    contours = Contours(np.c_[8*np.cos(t), 8*np.sin(t)], [0, len(t)])
    request = {'features': [{'id': 'wrong', 'type': 'arc', 'selection': {'path': 0}}]}
    _, session = prepare_interactive_report(contours, request)
    result = redefine_contour_geometry(contours, session, 'circle', [[9., .3]])
    feature = next(f for f in result['report']['features'] if f['id'] == result['feature_id'])
    assert feature['type'] == 'circle' and feature['closed'] == [True]
    assert len(feature['edge_indices']) == len(t)


def test_cursor_path_selects_the_long_reverse_route_on_a_closed_ring():
    square = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])
    contours = Contours(square, [0, 4])
    request = {'corner_exclusion_px': 0, 'features': [{'id': 'old', 'type': 'arc', 'selection': {'path': 0}}]}
    _, session = prepare_interactive_report(contours, request)
    cursor_path = [[1., 1.], [0., 5.], [1., 9.], [5., 10.], [9., 9.]]
    result = redefine_contour_geometry(
        contours, session, 'arc', [[4., -2.], [12., 7.]], cursor_path)
    feature = next(f for f in result['report']['features'] if f['id'] == result['feature_id'])
    support = np.asarray(feature['support'][0])
    np.testing.assert_allclose(support[[0, -1]], [[4, 0], [10, 7]])
    assert np.min(np.linalg.norm(support-[0, 10], axis=1)) < 1e-9
    interval = next(f for f in result['request']['features'] if f['id'] == result['feature_id'])['selection']['contour_interval']
    assert interval['direction'] == -1


def test_repeated_overrides_allocate_unique_fragment_ids():
    t = np.linspace(0, 2*np.pi, 64, endpoint=False)
    q = np.c_[20*np.cos(t), 20*np.sin(t)]
    contours = Contours(q, [0, len(q)])
    request = {'features': [{'id': 'base_00008', 'type': 'arc', 'selection': {'path': 0}}]}
    _, session = prepare_interactive_report(contours, request)

    def halfway(index):
        return ((q[index]+q[(index+1) % len(q)])/2).tolist()

    for start, end in [(5, 15), (25, 30), (18, 21)]:
        result = redefine_contour_geometry(
            contours, session, 'arc', [halfway(start), halfway(end)],
            [halfway(index) for index in range(start, end+1)],
        )
        session = result['request']
        ids = [feature['id'] for feature in session['features']]
        assert len(ids) == len(set(ids))
    assert any(feature['id'] == 'base_00008_fragment_3' for feature in session['features'])


def test_distance_construction_roundtrip_and_datum_direction():
    runs = [np.array(q, float) for q in [[[0, 0], [10, 0]], [[5, 4], [15, 4]], [[0, 20], [10, 20]]]]
    contours = Contours(np.vstack(runs), [0, 2, 4, 6], [False]*3)
    request = {'corner_exclusion_px': 0, 'features': [
        {'id': name, 'type': 'line', 'selection': {'path': i}} for i, name in enumerate(['a', 'b', 'datum'])]}
    report, session = prepare_interactive_report(contours, request)
    assert 'distance_construction' in report['interaction']['measurement_specs']
    result = evaluate_interactive_measurement(contours, session, report, {
        'operator': 'distance_construction', 'featureRefs': ['a', 'b'],
        'params': {'construction': 'normal_gap', 'statistic': 'average', 'datum_feature_ref': 'datum'}})
    assert result['value'] == 4
    assert result['coverage'] == .5
    assert result['definition']['datum_feature_ref'] == 'datum'
    again = evaluate_interactive_measurement(contours, session, report, {
        'operator': 'distance_construction', 'featureRefs': result['feature_refs'], 'params': result['definition']})
    assert result == again


def test_distance_preview_reuses_current_report_and_rejects_stale_coordinates(monkeypatch):
    from inspection_measurement import interaction
    import pytest
    contours = Contours(np.array([[0., 0.], [10., 0.], [5., 4.], [15., 4.]]), [0, 2, 4], [False, False])
    request = {'corner_exclusion_px': 0, 'features': [
        {'id': name, 'type': 'line', 'selection': {'path': i}} for i, name in enumerate(['a', 'b'])]}
    report, session = prepare_interactive_report(contours, request)
    def refit_forbidden(*args, **kwargs):
        raise AssertionError('Interactive slider must not refit all features')
    monkeypatch.setattr(interaction, 'evaluate', refit_forbidden)
    req = {'operator': 'distance_construction', 'featureRefs': ['a', 'b']}
    for basis in ['observed', 'fitted']:
        req['params'] = {'basis': basis}
        assert evaluate_interactive_measurement(contours, session, report, req)['value'] == 4
    changed = Contours(contours.xy+1, [0, 2, 4], [False, False])
    with pytest.raises(ValueError, match='stale'):
        evaluate_interactive_measurement(changed, session, report, req)


def test_compound_redefinition_groups_bases_and_replaces_overlapping_compounds():
    # A full-round slot opening: two straight sides joined by semicircular ends.
    t = np.linspace(-np.pi/2, np.pi/2, 24)
    right = np.c_[10+5*np.cos(t), 5*np.sin(t)]
    left = np.c_[-10-5*np.cos(t), -5*np.sin(t)]
    top = np.c_[np.linspace(10, -10, 12)[1:-1], np.full(10, 5.)]
    bottom = np.c_[np.linspace(-10, 10, 12)[1:-1], np.full(10, -5.)]
    ring = np.vstack([right, top, left, bottom])
    contours = Contours(ring, [0, len(ring)], [True])
    spans = [(0, 24), (24, 34), (34, 58), (58, 68)]
    request = {'schema_version': 2, 'label_version': 'primitives-compounds-v2', 'corner_exclusion_px': 0,
               'features': [{'id': f'base_{i}', 'type': kind, 'raw_edge_indices': list(range(lo, hi))}
                            for i, ((lo, hi), kind) in enumerate(zip(spans, ['arc', 'line', 'arc', 'line']))],
               'compound_features': [{'id': 'model_group', 'semantic': 'tooth_profile', 'type': 'repeated_profile',
                                      'base_instance_ids': ['base_1', 'base_3'], 'source': 'user',
                                      'raw_edge_indices': [*range(24, 34), *range(58, 68)]}],
               'relations': []}
    _, session = prepare_interactive_report(contours, request)
    result = redefine_compound(contours, session, 'full_round_slot', ['base_0', 'base_1', 'base_2', 'base_3'])
    slot = next(f for f in result['report']['features'] if f['id'] == result['feature_id'])
    assert result['feature_id'] == 'user_full_round_slot_0001'
    assert result['invalidated_feature_ids'] == ['model_group']
    assert slot['type'] == 'slot' and slot['status'] == 'fitted'
    np.testing.assert_allclose(slot['geometry']['width'], 10, atol=.2)
    assert {r['feature_a'] for r in result['request']['relations'] if r['feature_b'] == slot['id']} == {
        'base_0', 'base_1', 'base_2', 'base_3'}
    try:
        redefine_compound(contours, result['request'], 'not_a_compound', ['base_0'])
    except ValueError as error:
        assert 'Compound type' in str(error)
    else:
        raise AssertionError('Unknown compound types must be rejected')


def test_compound_redefinition_bridges_short_unselected_gaps_only():
    # Two sides of one open contour joined by a long land; tiny segments on each
    # side are hard to click and must not split the side into several runs.
    left = np.c_[np.zeros(21), np.arange(21.)]
    land = np.c_[np.arange(1., 40.), np.full(39, 20.)]
    right = np.c_[np.full(21, 40.), np.arange(20., -1., -1.)]
    xy = np.vstack([left, land, right])
    contours = Contours(xy, [0, len(xy)], [False])
    pieces = {'l1': (0, 9), 'l_tiny': (9, 11), 'l2': (11, 20), 'land': (20, 60), 'r1': (60, 69),
              'r_tiny': (69, 71), 'r2': (71, 80)}
    request = {'schema_version': 2, 'label_version': 'primitives-compounds-v2', 'corner_exclusion_px': 0,
               'features': [{'id': name, 'type': 'line', 'raw_edge_indices': list(range(lo, hi))}
                            for name, (lo, hi) in pieces.items()],
               'compound_features': [], 'relations': []}
    _, session = prepare_interactive_report(contours, request)
    result = redefine_compound(contours, session, 'tooth_profile', ['l1', 'l2', 'r1', 'r2'])
    assert result['bridged_feature_ids'] == ['l_tiny', 'r_tiny']
    compound = next(f for f in result['request']['compound_features'] if f['id'] == result['feature_id'])
    assert sorted(compound['base_instance_ids']) == ['l1', 'l2', 'l_tiny', 'r1', 'r2', 'r_tiny']
    assert 'land' not in compound['base_instance_ids']
    assert len(next(f for f in result['report']['features'] if f['id'] == result['feature_id'])['support']) == 2


def test_two_sided_compound_from_start_and_end_clicks_per_side():
    left = np.c_[np.zeros(21), np.arange(21.)]
    land = np.c_[np.arange(1., 40.), np.full(39, 20.)]
    right = np.c_[np.full(21, 40.), np.arange(20., -1., -1.)]
    xy = np.vstack([left, land, right])
    contours = Contours(xy, [0, len(xy)], [False])
    pieces = {'l1': (0, 9), 'l_tiny': (9, 11), 'l2': (11, 20), 'land': (20, 60), 'r1': (60, 69),
              'r_tiny': (69, 71), 'r2': (71, 80)}
    request = {'schema_version': 2, 'label_version': 'primitives-compounds-v2', 'corner_exclusion_px': 0,
               'features': [{'id': name, 'type': 'line', 'raw_edge_indices': list(range(lo, hi))}
                            for name, (lo, hi) in pieces.items()],
               'compound_features': [], 'relations': []}
    _, session = prepare_interactive_report(contours, request)
    clicks = [[-1., .5], [1., 19.], [41., 19.5], [39., .2]]
    preview = redefine_compound_sides(contours, session, 'triangular_thread', clicks[:3])
    assert not preview['complete']
    np.testing.assert_allclose(preview['snapped_points'], [[0, .5], [0, 19], [40, 19.5]])
    result = redefine_compound_sides(contours, session, 'triangular_thread', clicks)
    compound = next(f for f in result['request']['compound_features'] if f['id'] == result['feature_id'])
    assert sorted(compound['base_instance_ids']) == ['l1', 'l2', 'l_tiny', 'r1', 'r2', 'r_tiny']
    assert len(result['snapped_points']) == 4
    try:
        redefine_compound_sides(contours, session, 'triangular_thread', [[0, 1], [0, 19], [0, 5], [0, 15]])
    except ValueError as error:
        assert 'overlap' in str(error)
    else:
        raise AssertionError('Overlapping sides must be rejected')


def test_compound_redefinition_never_bridges_a_bolt_end_face_between_thread_sides():
    # Zigzag flanks are long in arclength, so a short end face between them is
    # below a fraction of the selected length; it must still stay out.
    left = np.array([[i % 2, 20.-i] for i in range(21)], float)
    face = np.c_[np.arange(1., 6.), np.zeros(5)]
    right = np.array([[6.+i % 2, i] for i in range(21)], float)
    xy = np.vstack([left, face, right])
    contours = Contours(xy, [0, len(xy)], [False])
    pieces = {**{f'l{i}': (2*i, 2*i+2) for i in range(10)}, 'face': (20, 26),
              **{f'r{i}': (26+2*i, 28+2*i) for i in range(10)}}
    request = {'schema_version': 2, 'label_version': 'primitives-compounds-v2', 'corner_exclusion_px': 0,
               'features': [{'id': name, 'type': 'line', 'raw_edge_indices': list(range(lo, hi))}
                            for name, (lo, hi) in pieces.items()],
               'compound_features': [], 'relations': []}
    _, session = prepare_interactive_report(contours, request)
    picked = [name for name in pieces if name not in ('face', 'l4')]
    clicked = redefine_compound(contours, session, 'triangular_thread', picked)
    assert clicked['bridged_feature_ids'] == ['l4']
    sides = redefine_compound_sides(contours, session, 'triangular_thread',
                                    [[0, 20], [0, 0], [6, 0], [6, 20]])
    for result in (clicked, sides):
        compound = next(f for f in result['request']['compound_features'] if f['id'] == result['feature_id'])
        assert 'face' not in compound['base_instance_ids']


def test_remove_compound_ungroups_only_the_selected_compound():
    left = np.c_[np.zeros(21), np.arange(21.)]
    land = np.c_[np.arange(1., 40.), np.full(39, 20.)]
    right = np.c_[np.full(21, 40.), np.arange(20., -1., -1.)]
    xy = np.vstack([left, land, right])
    contours = Contours(xy, [0, len(xy)], [False])
    pieces = {'l1': (0, 9), 'l2': (9, 20), 'land': (20, 60), 'r1': (60, 69), 'r2': (69, 80)}
    request = {'schema_version': 2, 'label_version': 'primitives-compounds-v2', 'corner_exclusion_px': 0,
               'features': [{'id': name, 'type': 'line', 'raw_edge_indices': list(range(lo, hi))}
                            for name, (lo, hi) in pieces.items()],
               'compound_features': [], 'relations': []}
    _, session = prepare_interactive_report(contours, request)
    session = redefine_compound(contours, session, 'tooth_profile', ['l1', 'l2'])['request']
    grouped = redefine_compound(contours, session, 'tooth_profile', ['r1', 'r2'])
    session, right_id = grouped['request'], grouped['feature_id']
    left_id = next(f['id'] for f in session['compound_features'] if f['id'] != right_id)
    result = remove_compound(contours, session, right_id)
    assert result['feature_id'] == right_id
    assert result['invalidated_feature_ids'] == [right_id]
    assert sorted(result['restored_feature_ids']) == ['r1', 'r2']
    assert [f['id'] for f in result['request']['compound_features']] == [left_id]
    assert not any(right_id in (r['feature_a'], r['feature_b']) for r in result['request']['relations'])
    fitted = {f['id']: f['status'] for f in result['report']['features']}
    assert right_id not in fitted
    assert fitted['r1'] == fitted['r2'] == fitted[left_id] == 'fitted'
    # Base features and already-removed compounds are not compounds to remove.
    for ident in ['r1', right_id]:
        with pytest.raises(ValueError, match='Select a compound'):
            remove_compound(contours, result['request'], ident)


def test_point_distance_along_or_normal_to_datum():
    runs = [np.array(q, float) for q in [[[0, 0], [10, 0]], [[0, 5], [10, 15]]]]
    contours = Contours(np.vstack(runs), [0, 2, 4], [False]*2)
    request = {'corner_exclusion_px': 0, 'features': [
        {'id': 'datum', 'type': 'line', 'selection': {'path': 0}},
        {'id': 'edge', 'type': 'line', 'selection': {'path': 1}}]}
    report, session = prepare_interactive_report(contours, request)
    vertex = next(f for f in report['features'] if f['type'] == 'point')
    corner = next(f for f in report['features'] if f['id'] == 'datum')
    values = {}
    for axis in ('normal', 'along'):
        result = evaluate_interactive_measurement(contours, session, report, {
            'operator': 'distance_construction', 'featureRefs': ['datum', vertex['id']],
            'params': {'construction': 'center_distance', 'datum_feature_ref': 'datum', 'datum_axis': axis}})
        assert result['definition']['datum_axis'] == axis
        values[axis] = result['value']
    expected = np.asarray(vertex['geometry']['center'])-np.asarray(corner['geometry']['center'])
    assert values == {'normal': pytest.approx(abs(expected[1])), 'along': pytest.approx(abs(expected[0]))}
