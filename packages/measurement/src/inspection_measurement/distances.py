"""Bounded, directional distance constructions, independent of legacy pairwise statistics.

Sections are s*n + t*d. Only intersections at the SAME s are paired. The
integration domain is the union of supported intervals, never a point-count
average. Fitted arcs retain their observed angular support; extensions require
an explicit finite station range and are reported as virtual geometry.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import quad
from shapely.ops import nearest_points
from .geometry import center, direction, line


@dataclass
class Piece:
    lo: float
    hi: float
    # Linear t=m*s+b, or circular t=b+sign*sqrt(r*r-(s-c)**2).
    m: float = 0.
    b: float = 0.
    c: float = 0.
    r: float = 0.
    sign: int = 0

    def at(self, s):
        return self.b+self.sign*math.sqrt(max(0., self.r**2-(s-self.c)**2)) if self.sign else self.m*s+self.b


def _pieces(g, d, n, basis, extend, window):
    pieces = []
    if basis == 'fitted' and g.kind in ('circle', 'arc'):
        c, r = center(g), float(g.values['radius'])
        cs, ct = float(c@n), float(c@d)
        # Unwrap in the section frame, preserving angular direction and gaps.
        spans = []
        if g.kind == 'circle' or extend:
            spans = [(0., 2*math.pi)]
        else:
            for q in g.runs:
                angles = np.unwrap(np.arctan2((q-c)@d, (q-c)@n))
                spans.append((float(min(angles[0], angles[-1])), float(max(angles[0], angles[-1]))))
        for lo, hi in spans:
            breaks = sorted({lo, hi, *(k*math.pi for k in range(math.floor(lo/math.pi), math.ceil(hi/math.pi)+1) if lo < k*math.pi < hi)})
            for a, b in zip(breaks[:-1], breaks[1:]):
                s0, s1 = cs+r*math.cos(a), cs+r*math.cos(b)
                pieces.append(Piece(min(s0, s1), max(s0, s1), b=ct, c=cs, r=r,
                                    sign=1 if math.sin((a+b)/2) > 0 else -1))
    else:
        if basis == 'fitted' and g.kind != 'line':
            raise ValueError('Fitted sections require lines, circles or arcs')
        runs = g.runs
        if basis == 'fitted':
            c, u, _ = line(g)
            if extend:
                if abs(u@n) < 1e-10:
                    raise ValueError('Section direction is parallel to the fitted line')
                runs = [np.array([c+u*((s-c@n)/(u@n)) for s in window])]
            else:
                runs = [c+np.outer((q-c)@u, u) for q in runs]
        for q, closed in zip(runs, [False]*len(runs) if basis == 'fitted' else g.closed):
            q = np.vstack([q, q[0]]) if closed else q
            for a, b in zip(q[:-1], q[1:]):
                sa, sb, ta, tb = float(a@n), float(b@n), float(a@d), float(b@d)
                if abs(sa-sb) < 1e-10:
                    continue  # a coincident section has no unique surface point
                slope = (tb-ta)/(sb-sa)
                pieces.append(Piece(min(sa, sb), max(sa, sb), m=slope, b=ta-slope*sa))
    if not pieces:
        raise ValueError('No transverse support for this section direction')
    return pieces


ANCHORS = ('center', 'near', 'far')


def _axis(mode, params, a, b):
    if mode in ('normal_a', 'normal_b'):
        return line(a if mode == 'normal_a' else b)[2]
    if mode == 'x':
        return np.array([1., 0.])
    if mode == 'y':
        return np.array([0., 1.])
    if mode == 'custom':
        return direction(params.get('direction'))
    raise ValueError('Unknown section direction')


def _anchors(params, gs):
    """Per-feature reference point: the fitted center, or the near/far edge of a circle or arc."""
    anchors = params.get('anchors', ['center', 'center'])
    if not isinstance(anchors, (list, tuple)) or len(anchors) != 2 or any(v not in ANCHORS for v in anchors):
        raise ValueError('anchors requires two center/near/far choices')
    for g, anchor, name in zip(gs, anchors, 'AB'):
        if anchor != 'center' and g.kind not in ('circle', 'arc'):
            raise ValueError(f'Near/far edge points need a fitted circle or arc; {name} has only a center')
    return list(anchors)


def _arc_contains(g, p):
    """Whether point p on a fitted circle/arc lies inside the observed angular support."""
    if g.kind == 'circle':
        return True
    c = center(g)
    d = (p-c)/np.linalg.norm(p-c)
    n = np.array([d[1], -d[0]])
    return any(q.lo-1e-9 <= p@n <= q.hi+1e-9 and abs(q.at(p@n)-p@d) < 1e-7
               for q in _pieces(g, d, n, 'fitted', False, None))


def _edge_point(g, c, anchor, toward, name):
    """Center, or the circle point nearest (near) or farthest (far) along unit vector toward."""
    if anchor == 'center':
        return c
    p = c+(1 if anchor == 'near' else -1)*float(g.values['radius'])*toward
    if not _arc_contains(g, p):
        raise ValueError(f'The {anchor} edge point of {name} lies outside the visible arc')
    return p


def _point_to_point(a, b, anchors, params, out, definition):
    """Straight-line or axis-projected distance between centers/corners or circle edge points.

    Direct mode measures along the center line. Axis modes measure the
    component along x, y or a custom direction, oriented from A toward B, and
    choose circle edge points as the extremes along that axis.
    """
    mode = params.get('direction_mode', 'direct')
    ca, cb = center(a), center(b)
    if mode == 'direct':
        span = float(np.linalg.norm(cb-ca))
        if span < 1e-10:
            if anchors != ['center', 'center']:
                raise ValueError('Centers coincide, so near/far edge points are undefined; choose a measurement direction')
            return dict(out, value=0., witness=[ca, cb])
        d = (cb-ca)/span
    elif mode in ('x', 'y', 'custom'):
        d = _axis(mode, params, a, b)
        definition['direction_mode'] = mode
        if mode == 'custom':
            definition['direction'] = d.tolist()
        if (cb-ca)@d < 0:
            d = -d
    else:
        raise ValueError('Point-to-point direction must be direct, x, y or custom')
    pa, pb = _edge_point(a, ca, anchors[0], d, 'A'), _edge_point(b, cb, anchors[1], -d, 'B')
    result = dict(out, value=float((pb-pa)@d), witness=[pa, pb])
    if mode != 'direct':
        result['dimension_direction'] = d
    if 'near' in anchors:
        result['signed'] = True  # negative: the edges overlap along the measuring direction
    return result


def _point_to_line(a, b, anchors, extend, tangent, out):
    """Perpendicular distance from a line to a center/corner or circle edge point.

    Either feature may be the line. The legacy tangent construction is the
    near-edge anchor. Without extension the foot must lie on observed support.
    """
    if a.kind != 'line' and b.kind == 'line':
        a, b, anchor = b, a, anchors[0]
    else:
        anchor = anchors[1]
    if a.kind != 'line' or b.kind == 'line':
        raise ValueError('Point to line needs one line and one point, corner or centered feature')
    if tangent:
        if b.kind not in ('circle', 'arc'):
            raise ValueError('Select a line as Feature A and a circle/arc as Feature B')
        anchor = 'near'
    ca, u, n = line(a)
    cb = center(b)
    foot = cb-((cb-ca)@n)*n
    along = float((foot-ca)@u)
    support = [(float(np.min((q-ca)@u)), float(np.max((q-ca)@u))) for q in a.runs]
    inside = any(lo-1e-9 <= along <= hi+1e-9 for lo, hi in support)
    if not inside and not extend:
        raise ValueError('Center projection falls outside supported line spans')
    offset = float((cb-ca)@n)
    if anchor == 'center':
        result = dict(out, value=abs(offset), witness=[foot, cb])
    else:
        # An edge point needs a known side. An arc must contain it; fitting an
        # arc does not authorize a whole circle.
        if abs(offset) < 1e-10:
            raise ValueError('Tangent side is ambiguous when the center lies on the line')
        side = n*np.sign(offset)  # from the line toward the center
        p = _edge_point(b, cb, anchor, -side, 'the circle')
        result = dict(out, value=float((p-ca)@side), witness=[foot, p])
        if anchor == 'near':
            result['signed'] = True  # negative: the circle extends past the line
    if not inside:
        nearest = min((x for span in support for x in span), key=lambda x: abs(x-along))
        result['virtual_segments'] = [[ca+nearest*u, foot]]
    return result


def _branch(active, s, branch, toward):
    active = sorted(active, key=lambda p: p.at(s))
    unique = []
    for p in active:
        if not unique or abs(p.at(s)-unique[-1].at(s)) > 1e-8:
            unique.append(p)
    if len(unique) == 1:
        return unique[0]
    if branch == 'auto':
        raise ValueError('Multiple surface intersections: choose a near/far or negative/positive branch')
    if branch in ('near', 'far'):
        if abs(toward) < 1e-9:
            raise ValueError('Near/far is ambiguous at coincident projected centers; choose negative/positive')
        positive = (toward > 0) == (branch == 'near')
    else:
        positive = branch == 'positive'
    return unique[-1] if positive else unique[0]


def _critical(a, b):
    """All interior stationary points of differences of linear/circular branches."""
    if not a.sign and not b.sign:
        return []
    if a.sign and b.sign:
        den = b.r-a.r if a.sign == b.sign else b.r+a.r
        return [] if abs(den) < 1e-12 else [(a.c*b.r-b.c*a.r if a.sign == b.sign else a.c*b.r+b.c*a.r)/den]
    arc, straight = (a, b) if a.sign else (b, a)
    return [arc.c-arc.sign*straight.m*arc.r/math.sqrt(1+straight.m**2)]


def distance_construction(a, b, params):
    construction = params.get('construction', 'normal_gap' if a.kind == 'line' or b.kind == 'line' else 'center_distance')
    basis = params.get('basis', 'observed')
    if basis not in ('observed', 'fitted'):
        raise ValueError('basis must be observed or fitted')
    statistic = params.get('statistic', 'minimum')
    if statistic not in ('minimum', 'maximum', 'average'):
        raise ValueError('statistic must be minimum, maximum or average')
    extend = params.get('extend', False)
    if not isinstance(extend, bool):
        raise ValueError('extend must be boolean')
    definition = dict(construction=construction, basis=basis, extend=extend, statistic=statistic)
    out = dict(units='px', method='explicit_distance_construction', definition=definition)
    if construction == 'closest_clearance':
        if basis != 'observed' or extend:
            raise ValueError('Closest clearance uses bounded observed contours')
        pa, pb = (np.asarray(p.coords[0]) for p in nearest_points(a.shape(), b.shape()))
        return dict(out, value=float(np.linalg.norm(pb-pa)), witness=[pa, pb])
    if construction in ('center_distance', 'center_to_line', 'tangent_clearance'):
        definition['basis'] = 'fitted'
        anchors = _anchors(params, (a, b))
        definition['anchors'] = anchors
        if construction == 'center_distance':
            if extend:
                raise ValueError('Extension applies only to section and point-to-line constructions')
            return _point_to_point(a, b, anchors, params, out, definition)
        return _point_to_line(a, b, anchors, extend, construction == 'tangent_clearance', out)
    if construction not in ('normal_gap', 'section'):
        raise ValueError('Unknown distance construction')
    mode = params.get('direction_mode', 'normal_a' if a.kind == 'line' else 'normal_b')
    d = _axis(mode, params, a, b)
    # Deterministic sign: normal modes point from A toward B. Explicit axes keep
    # their sign, so a reversed pair produces the opposite signed field.
    separation = float((np.asarray(b.values.get('center', b.points.mean(0)))-np.asarray(a.values.get('center', a.points.mean(0))))@d)
    if mode.startswith('normal') and separation < 0:
        d = -d
    n = np.array([d[1], -d[0]])
    branches = params.get('branches', ['auto', 'auto'])
    if not isinstance(branches, (list, tuple)) or len(branches) != 2 or any(v not in ('auto', 'near', 'far', 'negative', 'positive') for v in branches):
        raise ValueError('branches requires two auto/near/far/negative/positive choices')
    window = params.get('station_range')
    if window is not None:
        window = np.asarray(window, float)
        if window.shape != (2,) or not np.isfinite(window).all() or window[1] <= window[0]:
            raise ValueError('station_range must be two increasing finite coordinates')
    if extend and (basis != 'fitted' or window is None):
        raise ValueError('Extensions require fitted geometry and an explicit finite station_range')
    aa, bb = [_pieces(g, d, n, basis, extend, window) for g in (a, b)]
    if window is None:
        window = np.array([min(p.lo for p in aa), max(p.hi for p in aa)])
    lo, hi = map(float, window)
    definition.update(direction_mode=mode, branches=list(branches))
    if mode == 'custom':
        definition['direction'] = d.tolist()
    if params.get('station_range') is not None:
        definition['station_range'] = [lo, hi]
    out.update(section_direction=d, station_direction=n, requested_span=[lo, hi], signed=True)
    # Event partition: endpoints, tangencies and intersections where the selected
    # polyline branch can change order. No fixed sample grid drives the result.
    events = {lo, hi}
    for pieces in (aa, bb):
        events.update(x for p in pieces for x in (p.lo, p.hi) if lo < x < hi)
        active = []
        for p in sorted(pieces, key=lambda p: p.lo):
            active = [q for q in active if q.hi > p.lo]
            for q in active:
                if not p.sign and not q.sign and abs(p.m-q.m) > 1e-12:
                    x = (q.b-p.b)/(p.m-q.m)
                    if max(lo, p.lo, q.lo) < x < min(hi, p.hi, q.hi):
                        events.add(x)
            active.append(p)
    events = sorted(events)
    intervals, missing, extrema, bands = [], [], [], []
    integral, length = 0., 0.
    toward = float((np.asarray(b.values.get('center', b.points.mean(0)))-np.asarray(a.values.get('center', a.points.mean(0))))@d)
    station = params.get('station')
    if construction == 'section':
        if isinstance(station, bool) or station is None or not np.isfinite(float(station)):
            raise ValueError('section requires a finite station coordinate')
        station = float(station)
        if not lo <= station <= hi:
            raise ValueError('Section is outside the requested station range')
        definition['station'] = station
    if construction == 'section':
        for g in (a, b):
            for q in g.runs:
                if basis == 'fitted':
                    if g.kind != 'line':
                        continue
                    c, u, _ = line(g)
                    q = c+np.outer((q-c)@u, u)
                for start, end in zip(q[:-1], q[1:]):
                    if abs(start@n-station) < 1e-9 and abs(end@n-station) < 1e-9 and abs((end-start)@d) > 1e-9:
                        raise ValueError('Section coincides with an edge; move the position or change the direction')
        active = [[p for p in pieces if p.lo-1e-9 <= station <= p.hi+1e-9] for pieces in (aa, bb)]
        if not all(active):
            raise ValueError('Section has no common supported surface intersections')
        pa, pb = [_branch(ps, station, br, sign*toward) for ps, br, sign in zip(active, branches, [1, -1])]
        return dict(out, value=pb.at(station)-pa.at(station),
                    witness=[station*n+pa.at(station)*d, station*n+pb.at(station)*d])
    pending = [sorted(pieces, key=lambda p: p.lo) for pieces in (aa, bb)]
    cursors, active = [0, 0], [[], []]
    for left, right in zip(events[:-1], events[1:]):
        if right-left < 1e-10:
            continue
        mid = (left+right)/2
        for i in range(2):
            while cursors[i] < len(pending[i]) and pending[i][cursors[i]].lo < mid:
                active[i].append(pending[i][cursors[i]])
                cursors[i] += 1
            active[i] = [p for p in active[i] if p.hi > mid]
        if not all(active):
            missing.append([left, right])
            continue
        pa, pb = [_branch(ps, mid, br, sign*toward) for ps, br, sign in zip(active, branches, [1, -1])]
        gap = lambda s: pb.at(s)-pa.at(s)
        witness = lambda s: [s*n+pa.at(s)*d, s*n+pb.at(s)*d]
        candidates = [left, right, *(s for s in _critical(pa, pb) if left < s < right)]
        extrema.extend((gap(s), witness(s)) for s in candidates)
        integral += (right-left)*(gap(left)+gap(right))/2 if not pa.sign and not pb.sign else quad(gap, left, right, epsabs=1e-8, epsrel=1e-9)[0]
        length += right-left
        intervals.append([left, right])
        stations = np.linspace(left, right, 17) if pa.sign or pb.sign else [left, right]
        band = [witness(s) for s in stations]
        if bands and np.allclose(bands[-1][-1], band[0], rtol=0, atol=1e-9):
            bands[-1].extend(band[1:])
        else:
            bands.append(band)
    if not extrema or length <= 1e-10:
        raise ValueError('No common supported station interval; choose a different direction or explicitly extend fitted geometry')
    low, high = min(extrema, key=lambda x: x[0]), max(extrema, key=lambda x: x[0])
    values = dict(minimum=low[0], maximum=high[0], average=integral/length)
    out.update(values, value=values[statistic], selected_statistic=statistic,
               minimum_witness=low[1], maximum_witness=high[1], valid_intervals=intervals,
               excluded_intervals=missing, supported_length=length, coverage=length/(hi-lo),
               coverage_definition='valid station length / requested station span (Feature A projection by default)',
               crosses=low[0] < 0 < high[0], field_bands=bands)
    if statistic != 'average':
        out['witness'] = low[1] if statistic == 'minimum' else high[1]
    return out
