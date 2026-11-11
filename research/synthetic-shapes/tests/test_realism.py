import numpy as np
import pytest
from inspection_shapes.csg import csg_part, CURVED
from inspection_shapes.expanded import expanded_nominal
from inspection_shapes.geometry import valid, imperfect
from inspection_shapes.primitive_labels import labels
from inspection_shapes.realism import round_corners, add_corner_details
from inspection_shapes.turned import turned_part, plan


def contours(p):
    return dict(edge_primitive_id=np.concatenate([r.primitive for r in p.rings]),
                edge_feature_id=np.concatenate([r.feature for r in p.rings]),
                nominal_xy_mm=np.concatenate([r.nominal for r in p.rings]),
                ring_offsets=np.r_[0, np.cumsum([len(r.nominal) for r in p.rings])])


def junction_turns(p):
    """Turning angle at every primitive junction, from the sampled nominal ring."""
    turns = []
    for ring in p.rings:
        q, pid = ring.nominal, ring.primitive
        for j in np.flatnonzero(pid != np.roll(pid, 1)):
            a = q[j]-q[j-1]; b = q[(j+1) % len(q)]-q[j]
            turns.append(abs(np.arctan2(a[0]*b[1]-a[1]*b[0], a@b)))
    return np.array(turns)


def assert_exact_primitives(p, tol=1e-6):
    for ring in p.rings:
        for pid in np.unique(ring.primitive):
            prim = p.primitives[pid-1]; q = ring.nominal[ring.primitive == pid]
            params = prim['nominal_parameters']
            if prim['kind'] == 'circular_arc':
                r = np.linalg.norm(q-params['center'], axis=1)
                assert np.abs(r-params['radius']).max() < tol
            elif prim['kind'] == 'line':
                a, b = np.asarray(params['start']), np.asarray(params['end']); d = (b-a)/np.linalg.norm(b-a)
                w = q-a
                assert np.abs(w[:, 0]*d[1]-w[:, 1]*d[0]).max() < tol


def test_round_corners_tangent_and_exact():
    for seed in range(6):
        p = expanded_nominal('l_bracket', np.random.default_rng(seed))
        q = round_corners(p, lambda info: ('arc', .8))
        assert valid(q, False)
        arcs = [x for x in q.primitives if x['kind'] == 'circular_arc']
        assert len(arcs) >= 5
        # Fillets are tangent: no sampled junction turns by more than one sampling step.
        assert junction_turns(q).max() < np.deg2rad(8)
        assert_exact_primitives(q)


def test_round_corners_keeps_shared_feature_for_repeats():
    p = expanded_nominal('rack', np.random.default_rng(3), 'moderate')
    q = round_corners(p, lambda info: ('arc', .05) if info['same'] else None)
    annotations, groups = labels(dict(features=q.features, primitives=q.primitives), contours(q))
    assert len(groups) == 1 and groups[0]['class'] == 'tooth_profile'
    rounded = [x for x in q.primitives if x['kind'] == 'circular_arc']
    assert rounded and all(q.features[x['feature_id']-1]['semantic'] == 'tooth_profile' for x in rounded)


def test_corner_details_survive_imperfections():
    for seed in range(10):
        rng = np.random.default_rng(seed)
        p, info = add_corner_details(expanded_nominal('c_frame', rng), rng)
        assert info['status'] == 'ok'
        imperfect(p, rng, .2)
        assert valid(p)


def test_turned_thread_sides_are_half_pitch_offset():
    found = 0
    for seed in range(60):
        spec = plan(np.random.default_rng(seed))
        threads = [s for s in spec['segments'] if s['content'] == 'thread' and abs(s['bottom_shift']) > 0]
        if not threads:
            continue
        p = turned_part(np.random.default_rng(seed))
        assert valid(p, False)
        shifts = [t['bottom_shift']/t['pitch'] for t in spec['segments'] if t['content'] == 'thread']
        xy = np.concatenate([r.nominal for r in p.rings]); feat = np.concatenate([r.feature for r in p.rings])
        for f in p.features:
            if f['semantic'] != 'thread':
                continue
            pitch = f['nominal_parameters']['pitch']; sel = xy[feat == f['id']]
            def crest_phase(side):
                # Crest x positions (outermost points) on each side, modulo pitch.
                y = np.abs(side[:, 1]); crest = side[y > y.max()-1e-6, 0]
                return np.angle(np.exp(2j*np.pi*crest/pitch).mean())
            offset = (crest_phase(sel[sel[:, 1] < 0])-crest_phase(sel[sel[:, 1] > 0])) % (2*np.pi)/(2*np.pi)
            # About half a pitch, as designed (clipped end crests bias the estimate slightly).
            assert min(min(abs(offset-s) % 1, 1-abs(offset-s) % 1) for s in shifts) < .06
        found += 1
    assert found >= 3


@pytest.mark.parametrize('seed', range(25))
def test_csg_parts_valid_exact_and_deterministic(seed):
    p = csg_part(np.random.default_rng(seed))
    q = csg_part(np.random.default_rng(seed))
    assert valid(p, False)
    for a, b in zip(p.rings, q.rings):
        np.testing.assert_array_equal(a.nominal, b.nominal)
    assert_exact_primitives(p)
    annotations, groups = labels(dict(features=p.features, primitives=p.primitives), contours(p))
    assert (annotations['edge_base_class'] > 0).all()


@pytest.mark.parametrize('family', sorted(CURVED))
def test_curved_profiles_are_tangent_arcs(family):
    for seed in range(8):
        p = CURVED[family](np.random.default_rng(seed))
        assert valid(p, False)
        assert_exact_primitives(p)
        if family == 'lobed_profile':
            assert junction_turns(p).max() < np.deg2rad(5)
            assert all(x['kind'] == 'circular_arc' for x in p.primitives)
