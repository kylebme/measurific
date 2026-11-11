"""Full round slot and tooth profile metrology on observed silhouettes.

Like threads, each metric reports minimum, maximum and average with witnesses.
No nominal template, module, pressure angle or standard is inferred. Radii come
only from arc members reported at tips or roots; otherwise they are unavailable.
"""
import numpy as np
from scipy.signal import find_peaks
from .geometry import direction
from .threads import statistics


def _fitted(value, witness, method, units='px'):
    metric = statistics([(float(value), [np.asarray(p, float).tolist() for p in witness])], units)
    metric['method'] = method
    return metric


def slot_measurements(g, params):
    """Width, overall length, end radius and end center spacing of one full round slot.

    Width is the observed normal gap between the two flat members when both are
    near-parallel lines; otherwise it is the stadium fit. End radii come from the
    arc members, otherwise from the stadium fit.
    """
    if g.kind != 'slot' or 'end_center_spacing' not in g.values:
        raise ValueError('Slot measurements require a fitted full round slot')
    c, u = np.asarray(g.values['center'], float), np.asarray(g.values['direction'], float)
    n = np.array([-u[1], u[0]])
    r, half = float(g.values['end_radius']), float(g.values['end_center_spacing'])/2
    lines = [m for m in g.members if m.kind == 'line']
    arcs = [m for m in g.members if m.kind == 'arc']
    tolerance = np.cos(np.radians(float(params.get('parallel_tolerance_degrees', 2))))
    metrics = {}
    if len(lines) == 2 and abs(lines[0].values['direction']@lines[1].values['direction']) >= tolerance:
        from .distances import distance_construction
        try:
            field = distance_construction(lines[0], lines[1], {'construction': 'normal_gap'})
        except ValueError:
            field = None
        if field is not None:
            metrics['width'] = dict(
                status='measured', units='px', count=2,
                **{key: float(field[key]) for key in ('minimum', 'maximum', 'average')},
                **{key: np.asarray(field[key], float).tolist() for key in ('minimum_witness', 'maximum_witness')},
                method='observed_normal_gap_between_flats')
    if 'width' not in metrics:
        metrics['width'] = _fitted(2*r, [c-r*n, c+r*n], 'least_squares_stadium_width')
    metrics['overall_length'] = _fitted(2*(half+r), [c-(half+r)*u, c+(half+r)*u], 'least_squares_stadium_overall_length')
    if arcs:
        samples = []
        for arc in arcs:
            center, radius = np.asarray(arc.values['center'], float), float(arc.values['radius'])
            mid = _arc_midpoint(arc)
            samples.append((radius, [center.tolist(), (center+radius*mid).tolist()]))
        metrics['end_radius'] = statistics(samples)
        metrics['end_radius']['method'] = 'reported_end_arc_radius'
    else:
        samples = [(r, [(c+sign*half*u).tolist(), (c+sign*(half+r)*u).tolist()]) for sign in (-1, 1)]
        metrics['end_radius'] = statistics(samples)
        metrics['end_radius']['method'] = 'least_squares_stadium_end_radius'
    metrics['end_center_spacing'] = _fitted(2*half, [c-half*u, c+half*u], 'least_squares_stadium_end_center_spacing')
    return dict(value=metrics['width']['average'], units='px', method='full_round_slot_dimensions',
                compound_metrics=metrics, axis_direction=u.tolist(),
                scope='Width between observed flats when both are lines, otherwise the least-squares stadium. '
                      'Overall length and end center spacing come from the stadium fit; end radii from the end arcs.')


def _arc_midpoint(arc):
    """Unit direction from an arc's center to the middle of its observed sweep."""
    center = np.asarray(arc.values['center'], float)
    q = arc.points
    angles = np.unwrap(np.arctan2(q[:, 1]-center[1], q[:, 0]-center[0]))
    middle = (angles[0]+angles[-1])/2
    return np.array([np.cos(middle), np.sin(middle)])


def _outward(q, material_side):
    """Per-edge unit normals pointing away from the part material."""
    t = np.diff(q, axis=0)
    right = np.c_[t[:, 1], -t[:, 0]]
    right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return right if material_side == 'left' else -right


def _circle(q):
    c = q.mean(0)
    local = q-c
    abc = np.linalg.lstsq(np.c_[2*local, np.ones(len(q))], (local*local).sum(1), rcond=None)[0]
    return c+abc[:2], float(np.sqrt(max(abc[2]+abc[:2]@abc[:2], 1e-12)))


def _crossings(x, h, q):
    """Same-direction flank crossings of the mid-height section, with hysteresis."""
    low, high = np.percentile(h, [10, 90])
    level, band = (low+high)/2, (high-low)*.15
    out = []
    for polarity in [1, -1]:
        z = polarity*(h-level)
        armed = z[0] <= -band
        candidate = None
        found = []
        for i in range(1, len(h)):
            if z[i] <= -band:
                armed, candidate = True, None
            if armed and z[i-1] <= 0 < z[i]:
                t = -z[i-1]/(z[i]-z[i-1])
                candidate = (x[i-1]+t*(x[i]-x[i-1]), q[i-1]+t*(q[i]-q[i-1]))
            if armed and z[i] >= band and candidate is not None:
                found.append(candidate)
                armed, candidate = False, None
        out.append(found)
    return out, level


class _Profile:
    """One connected run as height over position: tips are height maxima.

    Linear profiles use axial position along ``u`` and height along the outward
    normal ``v``. Angular profiles use polar angle and radius about ``center``.
    """

    def __init__(self, q, closed, domain, material_side, u=None, center=None):
        self.domain, self.closed, self.center, self.u = domain, closed, center, u
        if closed:
            q = np.vstack([q, q[:1]])
        outward = _outward(q, material_side)
        mids = (q[1:]+q[:-1])/2
        weights = np.linalg.norm(np.diff(q, axis=0), axis=1)
        if domain == 'linear':
            self.v = np.array([-u[1], u[0]])
            self.sign = 1. if (weights*(outward@self.v)).sum() >= 0 else -1.
        else:
            radial = (mids-center)/np.maximum(np.linalg.norm(mids-center, axis=1, keepdims=True), 1e-12)
            self.sign = 1. if (weights*(outward*radial).sum(1)).sum() >= 0 else -1.
        if closed:
            # Start and end the ring at its deepest root so that every tooth is complete.
            ring = q[:-1]
            root = int(np.argmin(self.sign*self._height(ring)))
            ring = np.roll(ring, -root, axis=0)
            q = np.vstack([ring, ring[:1]])
        if self._position(q)[-1] < self._position(q)[0]:
            q = q[::-1]
        self.q = q
        self.x, self.h = self._position(q), self.sign*self._height(q)

    def _height(self, q):
        return q@self.v if self.domain == 'linear' else np.linalg.norm(q-self.center, axis=1)

    def _position(self, q):
        if self.domain == 'linear':
            return q@self.u
        return np.unwrap(np.arctan2(q[:, 1]-self.center[1], q[:, 0]-self.center[0]))

    def height_direction(self, p):
        """Image direction of increasing height at ``p``."""
        if self.domain == 'linear':
            return self.sign*self.v
        d = p-self.center
        return self.sign*d/np.linalg.norm(d)


def tooth_profile_measurements(g, params):
    """Visible tooth count, pitch, tooth height, tip/root diameters and radii.

    The profile is angular about a fitted center when its support sweeps at least
    60 degrees around that center with moderate radial depth (gears, splines,
    sprockets), otherwise linear along its principal axis (racks, serrations).
    ``params`` may set ``domain``, ``center`` or ``direction``, ``prominence``
    (tip/root detection height, default 25% of depth) and ``material_side``
    (``left`` or ``right`` of contour travel; the engine derives it).
    """
    if not g.runs:
        raise ValueError('Tooth profile has no observed support')
    runs = [np.asarray(q, float) for q in g.runs]
    if any(len(q) < 7 for q in runs):
        raise ValueError('Insufficient tooth profile support')
    material_side = params.get('material_side', 'left')
    if material_side not in ('left', 'right'):
        raise ValueError('material_side must be left or right')
    points = np.concatenate(runs)
    center, radius = (np.asarray(params['center'], float), None) if 'center' in params else _circle(points)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('Invalid tooth profile center')
    domain = params.get('domain')
    if domain is None:
        sweep = 0.
        for q, closed in zip(runs, g.closed):
            q = np.vstack([q, q[:1]]) if closed else q
            angles = np.unwrap(np.arctan2(q[:, 1]-center[1], q[:, 0]-center[0]))
            sweep += abs(angles[-1]-angles[0])
        depth = np.ptp(np.linalg.norm(points-center, axis=1))
        domain = 'angular' if radius is not None and np.degrees(sweep) >= 60 and depth < .6*radius else 'linear'
    if domain not in ('linear', 'angular'):
        raise ValueError('Tooth profile domain must be linear or angular')
    u = None
    if domain == 'linear':
        if 'direction' in params:
            u = direction(params['direction'])
        else:
            u = np.linalg.svd(np.concatenate([q-q.mean(0) for q in runs]), full_matrices=False)[2][0]
        if u[np.argmax(np.abs(u))] < 0:
            u = -u
    profiles = [_Profile(q, closed, domain, material_side, u, center) for q, closed in zip(runs, g.closed)]
    angular = domain == 'angular'
    pitch_units = 'deg' if angular else 'px'
    pitch, counts, heights, tips_d, roots_d = [], [], [], [], []
    for p in profiles:
        depth = float(np.ptp(p.h))
        prominence = float(params.get('prominence', depth*.25))
        if not np.isfinite(prominence) or prominence <= 0:
            raise ValueError('Tooth prominence must be finite and positive')
        h = np.round(p.h, 9)
        tips = find_peaks(h, prominence=prominence)[0]
        roots = find_peaks(-h, prominence=prominence)[0]
        if p.closed:
            roots = np.r_[0, roots, len(h)-1]
        if not len(tips):
            continue
        counts.append((len(tips), [p.q[tips[0]].tolist(), p.q[tips[-1]].tolist()]))
        for tip in tips:
            neighbours = [roots[roots < tip][-1:], roots[roots > tip][:1]]
            adjacent = np.concatenate(neighbours)
            if not len(adjacent):
                continue
            height = float(h[tip]-h[adjacent].mean())
            top = p.q[tip]
            heights.append((height, [top.tolist(), (top-height*p.height_direction(top)).tolist()]))
        if angular:
            for indices, target in [(tips, tips_d), (roots[:-1] if p.closed else roots, roots_d)]:
                for i in indices:
                    d = p.q[i]-center
                    r = float(np.linalg.norm(d))
                    target.append((2*r, [(center-d).tolist(), (center+d).tolist()]))
        flanks, level = _crossings(p.x, p.h, p.q)
        for found in flanks:
            xs = np.array([x for x, _ in found])
            if len(xs) > 1 and np.any(np.diff(xs) <= 0):
                raise ValueError('Corresponding tooth flanks must progress along the profile')
            pairs = list(zip(found[:-1], found[1:]))
            if p.closed and len(found) > 1:
                first = found[0]
                pairs.append((found[-1], (first[0]+2*np.pi, first[1])))
            for (xa, pa), (xb, pb) in pairs:
                if angular:
                    pitch.append((float(np.degrees(xb-xa)), [pa.tolist(), pb.tolist()]))
                else:
                    section = p.sign*level*p.v
                    pitch.append((float(xb-xa), [(xa*u+section).tolist(), (xb*u+section).tolist()]))
    if sum(count for count, _ in counts) < 2:
        raise ValueError('At least two resolved tooth tips are required')
    if not pitch:
        raise ValueError('Pitch requires at least two corresponding flanks at the mid-height section')
    metrics = {'number_of_teeth': statistics(counts, 'count')}
    metrics['number_of_teeth']['method'] = 'resolved_tips_per_connected_run'
    metrics['pitch'] = statistics(pitch, pitch_units)
    metrics['pitch']['method'] = ('angular' if angular else 'axial')+'_spacing_of_corresponding_same_direction_flanks_at_mid_height'
    metrics['tooth_height'] = statistics(heights, reason='No tip has an adjacent resolved root')
    metrics['tooth_height']['method'] = 'tip_height_above_mean_adjacent_root'
    reason = None if angular else 'Linear tooth profiles have no center'
    for key, samples in [('tip_diameter', tips_d), ('root_diameter', roots_d)]:
        metrics[key] = statistics(samples, reason=reason)
        metrics[key]['method'] = 'twice_tip_or_root_distance_from_fitted_center'
    for key, samples in _arc_radii([m for m in g.members if m.kind == 'arc'], profiles, material_side).items():
        metrics[key] = statistics(samples, reason='No arc reported at a tooth tip or root')
        metrics[key]['method'] = 'reported_tip_or_root_arc_radius'
    out = dict(value=metrics['pitch']['average'], units=pitch_units, method='visible_tooth_profile',
               compound_metrics=metrics, domain=domain,
               scope='2D silhouette; count is resolved tips per connected run, excluding partial end teeth. '
                     'Pitch is the spacing of corresponding same-direction flanks at mid height. '
                     'Tooth height is tip height above the mean of its adjacent roots. Radii are reported only for '
                     'arcs at a tip or root. No module, pressure angle, pitch diameter or standard is inferred.')
    if angular:
        out['profile_center'] = center.tolist()
    else:
        out['axis_direction'] = u.tolist()
    return out


def _arc_radii(arcs, profiles, material_side):
    """Tip/root radii from reported arcs: convex arcs reaching the tip band, concave the root band.

    Arcs larger than the tooth depth are crown or root lands (for example
    concentric with a gear), not roundings, and are ignored.
    """
    result = {'tip_radius': [], 'root_radius': []}
    for arc in arcs:
        center, radius = np.asarray(arc.values['center'], float), float(arc.values['radius'])
        support = arc.points
        if len(support) < 2:
            continue
        i = len(support)//2
        observed = support[i]
        normal = _outward(support[i-1:i+1], material_side)[0]
        profile = min(profiles, key=lambda p: np.min(np.linalg.norm(p.q-observed, axis=1)))
        if radius > np.ptp(profile.h):
            continue
        low, high = np.percentile(profile.h, [10, 90])
        band = (high-low)*.25
        height = profile.sign*profile._height(support)
        convex = (center-observed)@normal < 0
        if convex and height.max() >= high-band:
            kind = 'tip_radius'
        elif not convex and height.min() <= low+band:
            kind = 'root_radius'
        else:
            continue
        p = center+radius*_arc_midpoint(arc)
        result[kind].append((radius, [center.tolist(), p.tolist()]))
    return result
