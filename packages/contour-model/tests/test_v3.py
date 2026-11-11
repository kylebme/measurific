import numpy as np
import pytest
from inspection_shapes.geometry import Part, Chain, circle
from contour_model.corpus_v3 import create_record, split_for, HOLDOUT_FAMILIES
from contour_model.data import mirror, prepare
from contour_model.extract_v3 import transfer_keep
from contour_model.labels_v3 import observable_labels, weighted_targets
from contour_model.render_v3 import render, to_mm
from contour_model.sampling import prepare_adaptive


def record(part):
    c = {'nominal_xy_mm': np.concatenate([r.nominal for r in part.rings]),
         'actual_xy_mm': np.concatenate([r.nominal for r in part.rings]),
         'ring_offsets': np.r_[0, np.cumsum([len(r.nominal) for r in part.rings])],
         'edge_feature_id': np.concatenate([r.feature for r in part.rings]),
         'edge_primitive_id': np.concatenate([r.primitive for r in part.rings])}
    c['edge_anomaly_class'] = np.zeros(len(c['actual_xy_mm']), np.uint8)
    return dict(features=part.features, primitives=part.primitives), c


def notched_plate(break_radius=.02, shallow_arc=False):
    """40 x 20 plate: one corner with a tiny edge-break arc, one shallow arc edge."""
    p = Part('test'); f = p.feature('external_profile'); c = Chain(p)
    r = break_radius
    c.line([-20, -10], [20-r, -10], f)
    c.arc([20-r, -10+r], r, -np.pi/2, 0, f)
    if shallow_arc:
        # 19 mm chord with a 0.002 mm sagitta: invisible at 25 px/mm.
        h, sag, yc = (20-r)/2, .002, r/2
        R = (h*h+sag*sag)/(2*sag); a = np.arcsin(h/R)
        c.arc([20+sag-R, yc], R, -a, a, f)
        c.points[-1][0], c.points[-1][-1] = [20, -10+r], [20, 10]
    else:
        c.line([20, -10+r], [20, 10], f)
    c.line([20, 10], [-20, 10], f); c.line([-20, 10], [-20, -10], f); c.close()
    circle(p, [0, 0], 3, 'hole', True)
    return p


def test_tiny_edge_break_dissolves_into_corner():
    meta, c = record(notched_plate(.02))
    ann, groups, weight = observable_labels(meta, c, px_per_mm=25)
    outer = slice(c['ring_offsets'][0], c['ring_offsets'][1])
    assert set(ann['edge_base_class'][outer]) == {1}
    assert len(np.unique(ann['edge_base_instance'][outer])) == 4
    ann, _, _ = observable_labels(meta, c, px_per_mm=400)
    assert 2 in ann['edge_base_class'][outer]


def test_shallow_arc_becomes_line_and_merges():
    meta, c = record(notched_plate(1., shallow_arc=True))
    ann, _, _ = observable_labels(meta, c, px_per_mm=25)
    outer = slice(c['ring_offsets'][0], c['ring_offsets'][1])
    classes = ann['edge_base_class'][outer]
    assert (classes == 2).sum() > 0  # the 1 mm fillet is visible
    assert len(np.unique(ann['edge_base_instance'][outer])) == 5


def test_unsupported_feature_is_zero_weight():
    p = notched_plate(1.)
    p.features[-1]['nominal_parameters']['unsupported'] = True
    meta, c = record(p)
    ann, groups, weight = observable_labels(meta, c)
    hole = slice(c['ring_offsets'][1], c['ring_offsets'][2])
    assert (weight[hole] == 0).all() and (weight[:c['ring_offsets'][1]] == 1).all()
    prepared = prepare_adaptive(c)
    t, w = weighted_targets(meta, c, prepared, ann, groups, weight)
    ring = slice(prepared.offsets[1], prepared.offsets[2])
    assert (w[ring] == 0).all() and (w[:prepared.offsets[1]] > 0).all()


def test_mirror_is_involution_and_moves_boundaries():
    meta, c = record(notched_plate(1.))
    ann, groups, weight = observable_labels(meta, c)
    prepared = prepare(c, tokens=256)
    t, w = weighted_targets(meta, c, prepared, ann, groups, weight)
    sample = dict(x=prepared.x, offsets=prepared.offsets, weights=w, **t)
    twice = mirror(mirror(sample))
    np.testing.assert_allclose(twice['x'], sample['x'], atol=1e-5)
    for key in ('primary', 'boundary', 'weights'):
        np.testing.assert_array_equal(twice[key], sample[key])
    once = mirror(sample)
    lo, hi = once['offsets'][0], once['offsets'][1]
    cls = once['primary'][lo:hi].argmax(1)
    changes = np.flatnonzero(cls != np.roll(cls, 1))
    assert changes.size and (once['boundary'][lo:hi][changes] >= .5).all()
    # Mirrored coordinates are a reflection of the originals.
    assert np.allclose(np.sort(once['x'][:, 0]), np.sort(-sample['x'][:, 0]), atol=1e-4)


def test_render_round_trip_to_mm():
    meta, c = record(notched_plate(1.))
    image, truth, settings = render(c, 5, extent_px=600)
    np.testing.assert_allclose(to_mm(truth, settings), c['actual_xy_mm'], atol=1e-9)
    assert image.shape == (settings['height'], settings['width'])
    assert truth.min() > 0 and (truth.max(0) < [settings['width'], settings['height']]).all()


def test_transfer_keep_ignores_spikes_and_lost_instances():
    meta, c = record(notched_plate(1.))
    ann, groups, weight = observable_labels(meta, c, px_per_mm=10)
    raw = dict(c, **ann, edge_label_weight=weight)
    truth = c['actual_xy_mm']*[10, -10]
    off = c['ring_offsets']
    # Observed: thinned outer ring with a 5 px spike; the hole unchanged.
    outer = truth[off[0]:off[1]][::3].copy()
    outer[40] += [0, 5]
    observed = np.vstack([outer, truth[off[1]:off[2]][::3]])
    obs_off = np.array([0, len(outer), len(observed)])
    result, stats = transfer_keep(raw, truth, observed, obs_off)
    w = result['edge_label_weight'][:len(outer)]
    assert w[39] == 0 and w[40] == 0 and w[:30].all()
    assert stats['lost_rings'] == 0
    # Dropping the hole is tolerated, losing the outer ring is not.
    result, stats = transfer_keep(raw, truth, outer, np.array([0, len(outer)]))
    assert stats['lost_rings'] == 1
    with pytest.raises(ValueError):
        transfer_keep(raw, truth, truth[off[1]:off[2]][::3], np.array([0, len(truth[off[1]:off[2]][::3])]))


def test_corpus_v3_records_are_deterministic_and_holdout_isolated():
    for index in range(12):
        m1, c1 = create_record(index, 99)
        m2, c2 = create_record(index, 99)
        np.testing.assert_array_equal(c1['actual_xy_mm'], c2['actual_xy_mm'])
        assert (m1['split'] == 'holdout') == (m1['family'] in HOLDOUT_FAMILIES or m1['csg_style'] == 'radial')
    assert split_for(1, 2, 't_bracket', None) == 'holdout'
    assert split_for(1, 2, 'csg_part', 'radial') == 'holdout'
