"""Several objects in one contour set: each is recognized alone and merged back."""
import hashlib

import numpy as np

from contour_model.geometric import predict
from contour_model.objects import predict_objects, split_objects


def square(cx, cy, r, hole=False):
    t = np.linspace(0, 1, 4 * 20, endpoint=False)
    side = (t * 4).astype(int)
    u = (t * 4) % 1 * 2 - 1
    q = np.stack([np.choose(side, [u, np.ones_like(u), -u, -np.ones_like(u)]),
                  np.choose(side, [-np.ones_like(u), u, np.ones_like(u), -u])], 1) * r + (cx, cy)
    return q[::-1] if hole else q


def contours(*rings):
    return {'actual_xy_mm': np.concatenate(rings), 'ring_offsets': np.r_[0, np.cumsum([len(q) for q in rings])]}


def test_split_objects_groups_holes_with_their_outline():
    # small dust, a plate with a hole, and an island inside that hole
    c = contours(square(300, 0, 5), square(0, 0, 100), square(0, 0, 50, hole=True), square(0, 0, 20))
    assert split_objects(c) == [[1, 2], [3], [0]]


def test_single_object_is_passed_through():
    c = contours(square(0, 0, 100), square(0, 0, 40, hole=True))
    one = lambda part: predict(part, units='px')
    assert predict_objects(one, c)[0] == one(c)[0]


def test_objects_are_merged_onto_global_indices():
    rings = [square(300, 0, 30), square(0, 0, 100), square(0, 0, 40, hole=True)]
    c = contours(*rings)
    result, arrays = predict_objects(lambda part: predict(part, units='px'), c)
    offsets = c['ring_offsets']
    assert [o['ring_indices'] for o in result['objects']] == [[1, 2], [0]]
    assert result['part']['ring_is_hole'] == [False, False, True]
    assert result['coordinate_hash'] == hashlib.sha256(
        np.asarray(c['actual_xy_mm'], np.float64).tobytes() + np.asarray(offsets, np.int64).tobytes()).hexdigest()
    features = result['features'] + result['compound_features']
    ids = [f['id'] for f in features]
    assert len(ids) == len(set(ids))
    for f in features:
        members = result['objects'][f['object_index']]['ring_indices']
        assert f['parent'] == result['objects'][f['object_index']]['id']
        for span in f['contour_spans']:
            assert span['ring_index'] in members
            lo = offsets[span['ring_index']]
            assert lo + span['start_edge'] in f['raw_edge_indices']
    covered = sorted(i for f in result['features'] for i in f['raw_edge_indices'])
    assert covered == list(range(len(c['actual_xy_mm'])))
    known = set(ids)
    assert all(r['feature_a'] in known and r['feature_b'] in known for r in result['relations'])
    assert all(len(v) == len(c['actual_xy_mm']) for v in arrays.values())
