"""Associated geometry fits. All residuals retain every selected observation."""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from shapely.geometry import LineString, Polygon, MultiPoint, Point
from shapely.ops import unary_union


@dataclass
class Geometry:
    id: str
    kind: str
    runs: list = field(default_factory=list)
    closed: list = field(default_factory=list)
    values: dict = field(default_factory=dict)
    members: list = field(default_factory=list)

    @property
    def points(self):
        if not self.runs:
            raise ValueError(f'{self.id} has no contour support')
        return np.concatenate(self.runs)

    def shape(self):
        if self.kind == 'point':
            return Point(center(self))
        if not self.runs:
            return unary_union([m.shape() for m in self.members])
        return unary_union([LineString(np.vstack([q, q[0]]) if closed else q)
                            for q, closed in zip(self.runs, self.closed)])


def direction(value):
    a = np.asarray(value, float)
    if a.shape != (2,) or not np.isfinite(a).all() or np.linalg.norm(a) <= 1e-12:
        raise ValueError('direction must be a finite nonzero 2-vector')
    return a/np.linalg.norm(a)


def single(g, closed=None):
    if len(g.runs) != 1 or (closed is not None and g.closed[0] != closed):
        raise ValueError('Operator requires one ' + ('closed' if closed else 'connected') + ' contour')
    return g.runs[0]


def residual_stats(residual):
    return {'rms_residual': float(np.sqrt(np.mean(residual**2))),
            'peak_to_valley_residual': float(np.ptp(residual)),
            'max_abs_residual': float(np.max(np.abs(residual)))}


def solve(fun, initial, bounds=(-np.inf, np.inf)):
    fit = least_squares(fun, initial, bounds=bounds, max_nfev=1500,
                        ftol=1e-11, xtol=1e-11, gtol=1e-11)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('Geometric fit did not converge')
    s = np.linalg.svd(fit.jac, compute_uv=False)
    if s[-1] <= s[0]*1e-10:
        raise ValueError('Geometric fit is underconstrained')
    return fit


def _ellipse_jacobian(z):
    """Exact sparse derivative of the joint ellipse/footpoint residuals.

    Each coordinate depends only on five shared ellipse parameters and its own
    footpoint angle. Construct CSR directly, with O(N) storage and no differencing.
    """
    ca, sa = np.cos(z[2]), np.sin(z[2])
    ct, st = np.cos(z[5:]), np.sin(z[5:])
    a, b = z[3:5]
    n = len(ct)
    data = np.empty((n, 2, 6))
    data[:, 0, :] = np.column_stack((np.ones(n), np.zeros(n),
                                    -a*ct*sa-b*st*ca, ct*ca, -st*sa,
                                    -a*st*ca-b*ct*sa))
    data[:, 1, :] = np.column_stack((np.zeros(n), np.ones(n),
                                    a*ct*ca-b*st*sa, ct*sa, st*ca,
                                    -a*st*sa+b*ct*ca))
    indices = np.empty((n, 2, 6), dtype=np.int32)
    indices[:, :, :5] = np.arange(5)
    indices[:, :, 5] = np.arange(5, n+5)[:, None]
    return csr_matrix((data.ravel(), indices.ravel(), np.arange(0, 12*n+1, 6)),
                      shape=(2*n, n+5))


def fit_geometry(g, *, allow_circle_gaps=False):
    q = g.points
    c = q.mean(0)
    if len(q) < 2 or np.linalg.norm(np.ptp(q, axis=0)) < 1e-10:
        raise ValueError('Insufficient distinct support')
    _, singular, vt = np.linalg.svd(q-c, full_matrices=False)
    axis = vt[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    normal = np.array([-axis[1], axis[0]])
    if g.kind == 'line':
        u, residual = (q-c)@axis, (q-c)@normal
        if singular[0] < 1.05*singular[1]:
            raise ValueError('Line direction is ambiguous')
        g.values = {'center': c, 'direction': axis, 'normal': normal,
                    'endpoints': np.array([c+u.min()*axis, c+u.max()*axis]),
                    'length': float(np.ptp(u)), 'orientation': float(np.degrees(np.arctan2(axis[1], axis[0])) % 180),
                    'method': 'orthogonal_least_squares_line', **residual_stats(residual)}
    elif g.kind in ('circle', 'arc'):
        if not (g.kind == 'circle' and allow_circle_gaps):
            single(g, True if g.kind == 'circle' else False)
        if len(q) < 3 or singular[1] < singular[0]*1e-6:
            raise ValueError('Circle/arc support is collinear or too short')
        local = q-c
        abc = np.linalg.lstsq(np.c_[2*local, np.ones(len(q))], (local*local).sum(1), rcond=None)[0]
        initial = np.r_[c+abc[:2], np.sqrt(max(abc[2]+abc[:2]@abc[:2], 1e-12))]
        fit = solve(lambda z: np.linalg.norm(q-z[:2], axis=1)-z[2], initial,
                    ([-np.inf, -np.inf, 1e-12], [np.inf]*3))
        center, r = fit.x[:2], fit.x[2]
        if g.kind == 'circle' and allow_circle_gaps:
            # Count only observed angular support, never the gaps between runs.
            # At least half a circumference must survive exclusion. The solver
            # above also rejects rank-deficient fits.
            coverage = sum(abs(np.diff(np.unwrap(np.arctan2(
                run[:, 1]-center[1], run[:, 0]-center[0])))).sum()
                for run in g.runs)
            if coverage < np.pi:
                raise ValueError('Corner exclusion leaves insufficient circle angular support (requires 180 degrees)')
        angles = np.unwrap(np.arctan2(q[:, 1]-center[1], q[:, 0]-center[0]))
        if g.kind == 'arc':
            da = np.diff(angles)
            if min(np.sum(da > 1e-6), np.sum(da < -1e-6)) > .05*len(da):
                raise ValueError('Arc support doubles back')
            span = abs(angles[-1]-angles[0])
            if span < np.radians(5) or span > 2*np.pi+1e-6:
                raise ValueError('Arc angular support is insufficient or wraps multiple times')
        else:
            span = 2*np.pi
        g.values = {'center': center, 'radius': float(r), 'least_squares_diameter': float(2*r),
                    'start_angle': float(np.degrees(angles[0])), 'angular_span': float(np.degrees(span)),
                    'arc_length': float(r*span), 'chord_length': float(2*r*abs(np.sin(span/2))) if g.kind == 'arc' else 0.,
                    'sagitta': float(r*(1-np.cos(span/2))),
                    'method': 'orthogonal_least_squares_circle', **residual_stats(fit.fun)}
        if g.kind == 'circle' and allow_circle_gaps:
            g.values['retained_angular_support'] = float(np.degrees(coverage))
    elif g.kind in ('ellipse', 'elliptical_arc'):
        single(g, g.kind == 'ellipse')
        if len(q) < 6 or singular[1] < singular[0]*1e-5:
            raise ValueError('Insufficient ellipse support')
        # Jointly solve ellipse parameters and per-observation footpoint angles.
        local = (q-c)@np.stack([axis, normal]).T
        radii = np.maximum(np.ptp(local, axis=0)/2, 1e-6)
        # Algebraic conic initialization avoids the biased bounding-box center
        # on partial arcs. The reported fit still minimizes orthogonal distance.
        scale = np.linalg.norm(np.ptp(q, axis=0))
        x, y = ((q-c)/scale).T
        design = np.c_[x*x, x*y, y*y, x, y, np.ones(len(q))]
        coefficients = np.linalg.svd(design, full_matrices=False)[2][-1]
        matrix = np.array([[coefficients[0], coefficients[1]/2],
                           [coefficients[1]/2, coefficients[2]]])
        if abs(np.linalg.det(matrix)) > 1e-12:
            conic_center = -.5*np.linalg.solve(matrix, coefficients[3:5])
            rhs = conic_center@matrix@conic_center-coefficients[5]
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            squared_radii = rhs/eigenvalues
            if np.isfinite(squared_radii).all() and (squared_radii > 0).all():
                radii = np.sqrt(squared_radii)*scale
                c = c+conic_center*scale
                axis = eigenvectors[:, 0]
                normal = np.array([-axis[1], axis[0]])
                local = (q-c)@np.stack([axis, normal]).T
        t = np.arctan2(local[:, 1]/radii[1], local[:, 0]/radii[0])
        initial = np.r_[c, np.arctan2(axis[1], axis[0]), radii, t]
        def residual(z):
            ca, sa = np.cos(z[2]), np.sin(z[2])
            curve = np.c_[z[3]*np.cos(z[5:]), z[4]*np.sin(z[5:])]@np.array([[ca, sa], [-sa, ca]])+z[:2]
            return (curve-q).ravel()
        fit = least_squares(residual, initial, bounds=(np.r_[[-np.inf]*3, [1e-9]*2, [-np.inf]*len(q)], np.full(len(initial), np.inf)),
                            jac=_ellipse_jacobian, max_nfev=1000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if not fit.success:
            raise ValueError('Ellipse fit did not converge')
        z = fit.x
        major, minor = max(z[3:5]), min(z[3:5])
        if major/minor < 1+1e-6:
            raise ValueError('Ellipse orientation is underconstrained for circular support; use circle')
        angle = z[2]+(np.pi/2 if z[4] > z[3] else 0)
        ts = np.unwrap(z[5:])
        span = 2*np.pi if g.kind == 'ellipse' else abs(ts[-1]-ts[0])
        if g.kind == 'elliptical_arc':
            differences = np.diff(ts)
            if span < np.radians(15) or span > 2*np.pi+1e-6:
                raise ValueError('Insufficient or multiply wrapped elliptical arc span')
            if min(np.sum(differences > 1e-6), np.sum(differences < -1e-6)) > .05*len(differences):
                raise ValueError('Elliptical arc support doubles back')
        from scipy.integrate import quad
        lo = ts[0] if g.kind == 'ellipse' else min(ts[0], ts[-1])
        length = quad(lambda t: np.hypot(z[3]*np.sin(t), z[4]*np.cos(t)), lo, lo+span)[0]
        g.values = {'center': z[:2], 'major_axis': float(2*major), 'minor_axis': float(2*minor),
                    'orientation': float(np.degrees(angle) % 180), 'angular_span': float(np.degrees(span)),
                    'arc_length': float(length), 'method': 'orthogonal_least_squares_ellipse',
                    **residual_stats(np.linalg.norm(fit.fun.reshape(-1, 2), axis=1))}
    elif g.kind in ('slot', 'rounded_rectangle'):
        single(g, True)
        if len(q) < 12:
            raise ValueError('Insufficient rounded profile support')
        local = (q-c)@np.stack([axis, normal]).T
        half = np.ptp(local, axis=0)/2
        angle = np.arctan2(axis[1], axis[0])
        def residual(z):
            u = (q-z[:2])@np.array([[np.cos(z[2]), -np.sin(z[2])], [np.sin(z[2]), np.cos(z[2])]])
            if g.kind == 'slot':
                return np.hypot(np.maximum(np.abs(u[:, 0])-z[3], 0), u[:, 1])-z[4]
            d = np.abs(u)-z[3:5]
            return np.linalg.norm(np.maximum(d, 0), axis=1)+np.minimum(np.max(d, axis=1), 0)-z[5]
        if g.kind == 'slot':
            z0 = [*c, angle, max(half[0]-half[1], 1e-3), half[1]]
        else:
            r = min(half)*.25
            z0 = [*c, angle, *(half-r), r]
        fit = solve(residual, z0, ([-np.inf]*3+[1e-9]*(len(z0)-3), [np.inf]*len(z0)))
        z = fit.x
        r = z[-1]
        g.values = {'center': z[:2], 'direction': np.array([np.cos(z[2]), np.sin(z[2])]),
                    'orientation': float(np.degrees(z[2]) % 180), 'end_radius' if g.kind == 'slot' else 'corner_radius': float(r),
                    'width': float(2*r if g.kind == 'slot' else 2*(z[4]+r)),
                    'overall_length': float(2*(z[3]+r)), 'method': 'least_squares_'+g.kind,
                    **residual_stats(fit.fun)}
        if g.kind == 'slot':
            g.values['end_center_spacing'] = float(2*z[3])
    elif g.kind in ('contour', 'polygon', 'freeform', 'repeated_profile'):
        if g.kind == 'polygon':
            single(g, True)
        g.values = {'method': 'observed_polyline', 'point_count': len(q)}
    else:
        raise ValueError(f'Unsupported geometry type: {g.kind}')
    g.values['point_count'] = len(q)
    return g


def idealized_support(g):
    """Return support projected onto the fitted ideal form for display only.

    The returned points are deliberately not used by any measurement operator.
    Keeping this projection beside fitting prevents renderers from duplicating
    contour math and makes deformation exaggeration a reversible visualization.
    """
    if not g.runs:
        return []
    out = []
    for q in g.runs:
        if g.kind == 'line':
            c, u, _ = line(g)
            ideal = c + ((q-c)@u)[:, None]*u
        elif g.kind in ('circle', 'arc'):
            c = center(g)
            delta = q-c
            length = np.linalg.norm(delta, axis=1)
            ideal = c + delta/length[:, None]*g.values['radius']
        elif g.kind in ('ellipse', 'elliptical_arc'):
            c = center(g)
            angle = np.radians(g.values['orientation'])
            u = np.array([np.cos(angle), np.sin(angle)])
            v = np.array([-u[1], u[0]])
            a, b = g.values['major_axis']/2, g.values['minor_axis']/2
            local = np.c_[(q-c)@u, (q-c)@v]
            t = np.arctan2(local[:, 1]/b, local[:, 0]/a)
            ideal = c + (a*np.cos(t))[:, None]*u + (b*np.sin(t))[:, None]*v
        else:
            ideal = q.copy()
        out.append(ideal)
    return out


def minimum_zone_straightness(q):
    # Subtract an observed point, not the rounded mean: identical coordinates
    # then stay exactly zero. GEOS handles lower-dimensional hulls explicitly,
    # unlike Qhull's full-dimensional simplex construction on near-flat input.
    local = q-q[0]
    hull_geometry = MultiPoint(local).convex_hull
    if hull_geometry.geom_type != 'Polygon':
        return 0.
    hull = np.asarray(hull_geometry.exterior.coords)[:-1]
    # A convex polygon's minimum strip has one supporting hull edge. As that
    # edge advances, its opposite support vertex advances monotonically too
    # (rotating calipers). Avoid an N-by-hull-size projection matrix.
    if np.sum(hull[:, 0]*np.roll(hull[:, 1], -1)-hull[:, 1]*np.roll(hull[:, 0], -1)) < 0:
        hull = hull[::-1]
    edges = np.roll(hull, -1, axis=0)-hull
    ex, ey = edges[0]
    opposite = int(np.argmax(ex*(hull[:, 1]-hull[0, 1])-ey*(hull[:, 0]-hull[0, 0])))
    size = len(hull)
    # Scalar lists avoid allocating tiny NumPy arrays inside the caliper loop.
    xs, ys = hull.T.tolist()
    width = np.inf
    for i, (ex, ey) in enumerate(edges.tolist()):
        x, y = xs[i], ys[i]
        area = ex*(ys[opposite]-y)-ey*(xs[opposite]-x)
        for _ in range(size):
            following = (opposite+1) % size
            next_area = ex*(ys[following]-y)-ey*(xs[following]-x)
            if next_area <= area:
                break
            opposite, area = following, next_area
        width = min(width, area/np.hypot(ex, ey))
    return float(width)


def polygon(g):
    return Polygon(single(g, True))


def center(g):
    if 'center' in g.values:
        return np.asarray(g.values['center'])
    if len(g.runs) == 1 and g.closed[0]:
        return np.array(polygon(g).centroid.coords[0])
    raise ValueError(f'{g.id} has no defined center')


def line(g):
    if g.kind != 'line':
        raise ValueError('Operator requires fitted line geometry')
    return g.values['center'], g.values['direction'], g.values['normal']
