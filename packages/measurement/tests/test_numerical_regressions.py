"""Numerical and topology checks for the accelerated measurement paths."""
import numpy as np
import pytest
import shapely
from scipy.optimize import least_squares
from shapely.geometry import LineString, MultiPoint, Point, Polygon

from inspection_measurement import Contours
from inspection_measurement.contours import select
from inspection_measurement.engine import jsonable
from inspection_measurement.geometry import Geometry, _ellipse_jacobian, fit_geometry, minimum_zone_straightness
from inspection_measurement.operators import _point_distances, nominal_profile


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64, np.uint64, np.bool_])
def test_numeric_array_serialization(dtype):
    q = np.array([[0, 1], [1, 0]], dtype=dtype)
    assert jsonable({'support': [q], 'scalar': q[0, 0]}) == {'support': [q.tolist()], 'scalar': q[0, 0].item()}
    assert jsonable(q[0, 0:1].reshape(())) == q[0, 0].item()


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_array_serialization_still_rejected(value):
    with pytest.raises(ValueError, match='Nonfinite'):
        jsonable({'support': [np.array([[1., value]])]})


def ellipse_points(z):
    ca, sa = np.cos(z[2]), np.sin(z[2])
    return np.c_[z[3]*np.cos(z[5:]), z[4]*np.sin(z[5:])]@np.array([[ca, sa], [-sa, ca]])+z[:2]


def test_ellipse_derivatives_against_central_differences():
    z = np.r_[[42., -17., .7, 8., 3.], np.linspace(-2., 4., 13)]
    numerical = np.empty((26, len(z)))
    for i in range(len(z)):
        step = np.zeros_like(z)
        step[i] = 1e-5
        numerical[:, i] = ((ellipse_points(z+step)-ellipse_points(z-step))/(2e-5)).ravel()
    np.testing.assert_allclose(_ellipse_jacobian(z).toarray(), numerical, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize('closed', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_noisy_ellipse_matches_dense_orthogonal_fit(closed, reverse):
    rng = np.random.default_rng(17)
    t = np.linspace(.2, 2*np.pi+.2 if closed else 3.7, 48, endpoint=not closed)
    initial = np.r_[[42., -17., .7, 8., 3.], t]
    q = ellipse_points(initial)+rng.normal(0, .005, (len(t), 2))
    reference = least_squares(lambda z: (ellipse_points(z)-q).ravel(), initial,
                              ftol=1e-12, xtol=1e-12, gtol=1e-12)
    g = fit_geometry(Geometry('e', 'ellipse' if closed else 'elliptical_arc',
                              [q[::-1] if reverse else q], [closed]))
    np.testing.assert_allclose(g.values['center'], reference.x[:2], atol=2e-5)
    np.testing.assert_allclose([g.values['major_axis'], g.values['minor_axis']],
                               2*reference.x[3:5], atol=3e-5)
    np.testing.assert_allclose(g.values['rms_residual'],
                               np.sqrt(np.mean(np.sum(reference.fun.reshape(-1, 2)**2, axis=1))), atol=1e-8)


def test_calipers_against_all_support_projections():
    rng = np.random.default_rng(18)
    for n in (3, 4, 20, 200):
        for scale in (1e-7, 1., 1000.):
            q = rng.normal(size=(n, 2))*[scale, 2]+[900., 315.1]
            local = q-q[0]
            hull = np.asarray(MultiPoint(local).convex_hull.exterior.coords)[:-1]
            edges = np.roll(hull, -1, axis=0)-hull
            normals = np.c_[-edges[:, 1], edges[:, 0]]/np.linalg.norm(edges, axis=1)[:, None]
            expected = np.ptp(local@normals.T, axis=0).min()
            np.testing.assert_allclose(minimum_zone_straightness(q), expected, rtol=1e-7, atol=1e-12)
    # Parallel opposite edges and tied support vertices.
    q = np.array([[0., 0.], [10., 0.], [10., 2.], [0., 2.]])
    assert minimum_zone_straightness(q) == 2.


@pytest.mark.parametrize('closed', [False, True])
def test_selection_exhaustive_small_masks(closed):
    t = np.linspace(0, 2*np.pi, 7, endpoint=False)
    q = np.c_[np.cos(t), np.sin(t)]
    contours = Contours(q, [0, len(q)], [closed])
    for bits in range(1, 2**(len(q)-(not closed))):
        mask = np.array([bool(bits & (1 << i)) for i in range(len(q))])
        ids = np.flatnonzero(mask)
        runs, closure, actual_ids = select(contours, {'edge_indices': ids})
        np.testing.assert_array_equal(actual_ids, ids)
        if mask.all():
            assert closure == [True]
            np.testing.assert_array_equal(runs[0], q)
            continue
        expected = []
        for start in np.flatnonzero(mask & ~np.roll(mask, 1)):
            vertices, j = [start], start
            while mask[j]:
                j = (j+1) % len(q)
                vertices.append(j)
            expected.append(q[vertices])
        assert closure == [False]*len(expected)
        assert len(runs) == len(expected)
        for actual, reference in zip(runs, expected):
            np.testing.assert_array_equal(actual, reference)


def test_indexed_distances_preserve_segment_interiors_and_disconnected_paths():
    x = np.linspace(-10., 10., 200)
    g = Geometry('g', 'contour', [np.c_[x, x*0-5], np.c_[x, x*0+5]], [False, False])
    rng = np.random.default_rng(19)
    q = np.vstack([rng.uniform(-8, 8, (1000, 2)), [10., 0.]])
    shape = g.shape()
    expected = np.array([Point(p).distance(shape) for p in q])
    actual = _point_distances(shapely.points(q), shape)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    assert actual[-1] == 5.  # A fictitious connector at x=10 would give zero.


@pytest.mark.parametrize('reverse', [False, True])
def test_profile_distribution_matches_scalar_reference(reverse):
    t = np.linspace(0, 2*np.pi, 300, endpoint=False)
    unit = np.c_[np.cos(t), np.sin(t)]
    nominal = unit*5
    if reverse:
        nominal = nominal[::-1]
    q = unit*(5+.2*np.cos(3*t))[:, None]
    g = Geometry('g', 'contour', [q], [True])
    actual = nominal_profile(g, {'nominal_points': nominal, 'closed': True})
    boundary = LineString(np.vstack([nominal, nominal[0]]))
    region = Polygon(nominal)
    distances = np.array([Point(p).distance(boundary) for p in q])
    signed = distances*np.array([-1 if region.contains(Point(p)) else 1 for p in q])
    np.testing.assert_allclose(actual['distances'], distances, atol=1e-13)
    np.testing.assert_allclose(actual['signed_distances'], signed, atol=1e-13)
    expected_reverse = max(Point(p).distance(g.shape()) for p in nominal)
    np.testing.assert_allclose(actual['sampled_hausdorff'], max(expected_reverse, distances.max()), atol=1e-13)
