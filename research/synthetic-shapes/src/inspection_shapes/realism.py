"""Manufacturing corner details on nominal parts: tool radii, fillets and edge breaks.

Real machined parts almost never have perfectly sharp corners: internal corners
carry the cutter radius, shoulders are filleted or undercut, and external edges
are broken by a small radius or chamfer. ``round_corners`` rebuilds every ring of
a nominal part from its analytic line/arc construction spans and replaces chosen
corners by an exactly tangent arc (fillet) or a chamfer line. Freeform spans are
kept unchanged and their corners are never modified.

Identity follows construction: the new span is its own primitive. It joins the
adjacent feature when both sides belong to the same feature (thread roots, groove
bottoms, tooth tips), otherwise it is a new ``fillet``/``chamfer`` feature.
"""
from copy import deepcopy
import numpy as np
from .geometry import Part, Chain, valid

MIN_TURN = np.deg2rad(2.)
MAX_TURN = np.deg2rad(178.)
MAX_TRIM = .45


def _cross(a, b):
    return a[0]*b[1]-a[1]*b[0]


def _unit(v):
    return v/max(np.linalg.norm(v), 1e-15)


def _left(v):
    return np.array([-v[1], v[0]])


def _runs(part, ring):
    """Analytic description of each construction span of a ring, in ring order."""
    pids = ring.primitive; n = len(pids)
    starts = np.flatnonzero(pids != np.roll(pids, 1))
    if not len(starts):
        starts = np.array([0])
    runs = []
    for k, s in enumerate(starts):
        e = starts[(k+1) % len(starts)] if len(starts) > 1 else s+n
        if e <= s:
            e += n
        idx = np.arange(s, e) % n
        points = ring.nominal[np.r_[idx, e % n]]
        p = part.primitives[int(pids[s])-1]
        run = dict(pid=int(pids[s]), fid=int(ring.feature[s]), kind=p['kind'], points=points,
                   params=p['nominal_parameters'])
        if p['kind'] == 'line':
            run.update(start=points[0].copy(), end=points[-1].copy(),
                       length=float(np.linalg.norm(points[-1]-points[0])))
        elif p['kind'] == 'circular_arc':
            c = np.asarray(p['nominal_parameters']['center'], float); r = float(p['nominal_parameters']['radius'])
            a, b = p['nominal_parameters']['start_angle'], p['nominal_parameters']['end_angle']
            s_ = 1. if b > a else -1.
            t0 = np.arctan2(*(points[0]-c)[::-1]); sweep = abs(b-a)
            run.update(center=c, radius=r, sign=s_, theta0=t0, sweep=sweep, length=r*sweep)
        else:
            run['length'] = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        runs.append(run)
    return runs


def _tangent(run, at_end):
    if run['kind'] == 'line':
        return _unit(run['end']-run['start'])
    theta = run['theta0']+run['sign']*(run['sweep'] if at_end else 0.)
    return run['sign']*np.array([-np.sin(theta), np.cos(theta)])


def _offset(run, side, rho):
    """Curve parallel to ``run`` at distance rho on ``side`` (+1 left of travel)."""
    if run['kind'] == 'line':
        t = _unit(run['end']-run['start'])
        return ('line', run['start']+side*rho*_left(t), t)
    toward_center = (run['sign'] > 0) == (side > 0)
    radius = run['radius']-rho if toward_center else run['radius']+rho
    return ('circle', run['center'], radius) if radius > 1e-9 else None


def _intersections(a, b):
    if a[0] == 'line' and b[0] == 'line':
        den = _cross(a[2], b[2])
        if abs(den) < 1e-12:
            return []
        t = _cross(b[1]-a[1], b[2])/den
        return [a[1]+t*a[2]]
    if a[0] == 'circle' and b[0] == 'line':
        a, b = b, a
    if a[0] == 'line':
        p, d = a[1], a[2]; c, r = b[1], b[2]
        f = p-c; B = 2*f@d; C = f@f-r*r; disc = B*B-4*C
        if disc < 0:
            return []
        s = np.sqrt(disc)
        return [p+(-B-s)/2*d, p+(-B+s)/2*d]
    (c1, r1), (c2, r2) = (a[1], a[2]), (b[1], b[2])
    d = np.linalg.norm(c2-c1)
    if d < 1e-12 or d > r1+r2 or d < abs(r1-r2):
        return []
    x = (d*d+r1*r1-r2*r2)/(2*d); h = np.sqrt(max(r1*r1-x*x, 0.))
    u = (c2-c1)/d; m = c1+x*u
    return [m+h*_left(u), m-h*_left(u)]


def _foot(run, point):
    """Closest point of the run's carrier and the distance travelled from the corner end."""
    if run['kind'] == 'line':
        t = _unit(run['end']-run['start'])
        return run['start']+((point-run['start'])@t)*t
    return run['center']+run['radius']*_unit(point-run['center'])


def _trim(run, point, at_end):
    """Arc length removed from ``run`` at one end when it is cut at ``point``."""
    if run['kind'] == 'line':
        ref = run['end'] if at_end else run['start']
        t = _unit(run['end']-run['start'])
        return float(((ref-point)@t) if at_end else ((point-ref)@t))
    c = run['center']
    theta = np.arctan2(*(point-c)[::-1])
    ref = run['theta0']+(run['sign']*run['sweep'] if at_end else 0.)
    delta = (theta-ref+np.pi) % (2*np.pi)-np.pi
    return float((-run['sign']*delta if at_end else run['sign']*delta)*run['radius'])


def _point_back(run, distance, at_end):
    if run['kind'] == 'line':
        t = _unit(run['end']-run['start'])
        return run['end']-distance*t if at_end else run['start']+distance*t
    theta = run['theta0']+run['sign']*(run['sweep']-distance/run['radius'] if at_end else distance/run['radius'])
    return run['center']+run['radius']*np.array([np.cos(theta), np.sin(theta)])


def corner_geometry(a, b, mode, size, corner):
    """Tangent fillet (radius ``size``) or chamfer (setback ``size``) between runs a -> b.

    Returns (T1, T2, trim_a, trim_b, arc) or None if the corner cannot take it.
    ``arc`` is (center, radius, start_angle, end_angle) for a fillet.
    """
    ta, tb = _tangent(a, True), _tangent(b, False)
    turn = np.arctan2(_cross(ta, tb), ta@tb)
    side = 1. if turn > 0 else -1.
    if mode == 'chamfer':
        if size > MAX_TRIM*min(a['length'], b['length']):
            return None
        t1, t2 = _point_back(a, size, True), _point_back(b, size, False)
        return t1, t2, size, size, None
    oa, ob = _offset(a, side, size), _offset(b, side, size)
    if oa is None or ob is None:
        return None
    candidates = _intersections(oa, ob)
    if not candidates:
        return None
    f = min(candidates, key=lambda q: np.linalg.norm(q-corner))
    t1, t2 = _foot(a, f), _foot(b, f)
    trim_a, trim_b = _trim(a, t1, True), _trim(b, t2, False)
    if not (0 < trim_a <= MAX_TRIM*a['length'] and 0 < trim_b <= MAX_TRIM*b['length']):
        return None
    u1, u2 = t1-f, t2-f
    sweep = np.arctan2(_cross(u1, u2), u1@u2)
    if np.sign(sweep) != side or abs(sweep) < 1e-6:
        return None
    a0 = float(np.arctan2(u1[1], u1[0]))
    return t1, t2, trim_a, trim_b, (f, float(size), a0, a0+float(sweep))


def round_corners(part, decide, spacing=None):
    """Rebuild ``part`` with corners chosen by ``decide(info) -> None | (mode, size)``.

    ``info`` holds convex (material corner), turn (radians), the two feature IDs,
    their semantics, run kinds/lengths and the ring role. A rejected corner (no room,
    no tangent solution) stays sharp. The returned part has a fresh, consistent
    primitive list; feature IDs of existing features are unchanged.
    """
    new = Part(part.family, features=deepcopy(part.features), primitives=[],
               relationships=[r for r in deepcopy(part.relationships) if 'primitive_refs' not in r],
               patterns=deepcopy(part.patterns))
    applied = []
    for ri, ring in enumerate(part.rings):
        runs = _runs(part, ring)
        m = len(runs)
        trims = [[0., 0.] for _ in runs]; corners = [None]*m
        for k in range(m if m > 1 else 0):
            a, b = runs[k-1], runs[k]
            if a['kind'] == 'freeform' or b['kind'] == 'freeform':
                continue
            ta, tb = _tangent(a, True), _tangent(b, False)
            turn = float(np.arctan2(_cross(ta, tb), ta@tb))
            if not MIN_TURN < abs(turn) < MAX_TURN:
                continue
            fa, fb = a['fid'], b['fid']
            info = dict(convex=turn > 0, turn=abs(turn), fa=fa, fb=fb, same=fa == fb,
                        semantic_a=new.features[fa-1]['semantic'], semantic_b=new.features[fb-1]['semantic'],
                        kind_a=a['kind'], kind_b=b['kind'], length_a=a['length'], length_b=b['length'],
                        hole=ring.hole, ring=ri)
            choice = decide(info)
            if not choice:
                continue
            mode, size = choice
            corner = b['points'][0]
            for _ in range(8):
                # Each span may also be trimmed at its other end; budgets are per end.
                geometry = corner_geometry(a, b, mode, size, corner)
                if geometry is not None:
                    break
                size *= .6
            if geometry is None:
                continue
            t1, t2, trim_a, trim_b, arc = geometry
            corners[k] = (mode, t1, t2, arc, info)
            trims[k-1][1] = trim_a; trims[k][0] = trim_b
        step = spacing or float(np.median(np.linalg.norm(np.diff(ring.nominal, axis=0), axis=1)))
        chain = Chain(new, ring.hole, spacing=max(step, 1e-3))
        current = None

        def corner_feature(info, mode):
            if info['same']:
                return info['fa']
            parent = new.features[info['fa']-1]['parent'] or info['fa']
            return new.feature('fillet' if mode == 'arc' else 'chamfer', parent, generated='corner_detail')

        for k, run in enumerate(runs):
            if corners[k] is not None:
                mode, t1, t2, arc, info = corners[k]
                fid = corner_feature(info, mode)
                if mode == 'arc':
                    c, r, a0, a1 = arc
                    chain.arc(c, r, a0, a1, fid)
                    chain.points[-1][0] = current if current is not None else chain.points[-1][0]
                else:
                    chain.line(current if current is not None else t1, t2, fid)
                current = chain.points[-1][-1].copy()
                applied.append(dict(ring=ri, mode=mode, size=float(arc[1]) if arc else float(np.linalg.norm(t2-t1)),
                                    convex=bool(info['convex']), turn_deg=float(np.degrees(info['turn']))))
            s0, s1 = trims[k]
            if run['kind'] == 'line':
                t = _unit(run['end']-run['start'])
                a = current if current is not None else run['start']+s0*t
                b = run['end']-s1*t
                chain.line(a, b, run['fid'])
            elif run['kind'] == 'circular_arc':
                c, r, sg = run['center'], run['radius'], run['sign']
                a0 = run['theta0']+sg*s0/r; a1 = run['theta0']+sg*(run['sweep']-s1/r)
                chain.arc(c, r, a0, a1, run['fid'])
                if current is not None:
                    chain.points[-1][0] = current
            else:
                points = run['points'].copy()
                if current is not None:
                    points[0] = current
                chain.add(points, run['fid'], 'freeform', **{k2: v for k2, v in run['params'].items()})
            current = chain.points[-1][-1].copy()
        # The closing span ends exactly where the first span starts.
        chain.points[-1][-1] = chain.points[0][0].copy()
        chain.close()
    new.imperfections = dict(corner_details=applied)
    return new


def loguniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


REPEATED = ('thread', 'tooth_profile', 'groove', 'notch')


def corner_style(rng, scale=1.):
    """Per-part manufacturing style: consistent radii, varying coverage.

    Sizes are in millimetres for parts of roughly 15–160 mm. ``scale`` shrinks sizes
    for small or finely detailed parts. Coverage probabilities include "never", so
    sharp-cornered designs remain represented.
    """
    return dict(
        inner=float(rng.choice([0., .5, .9, 1.], p=[.15, .15, .3, .4])),
        inner_radius=loguniform(rng, .2, 3.)*scale,
        edge_break=float(rng.choice([0., .3, .8, 1.], p=[.25, .25, .25, .25])),
        edge_size=loguniform(rng, .05, .5)*scale,
        edge_arc=float(rng.uniform(.3, .9)),
        design=float(rng.choice([0., .3, 1.], p=[.55, .3, .15])),
        design_radius=loguniform(rng, 1., 6.)*scale,
        repeat=float(rng.choice([0., .7, 1.], p=[.45, .2, .35])),
        repeat_fraction=loguniform(rng, .06, .35),
        jitter=float(rng.uniform(0., .15)))


def style_policy(rng, style):
    """Corner decision function implementing a sampled ``corner_style``."""
    def size(value):
        return value*float(np.exp(rng.normal(0, style['jitter']))) if style['jitter'] else value

    def decide(info):
        short = min(info['length_a'], info['length_b'])
        if info['same'] and info['semantic_a'] in REPEATED:
            # Rounded roots/crests: relative to the local flank/land size.
            if rng.random() < style['repeat']:
                return 'arc', style['repeat_fraction']*short
            return None
        if not info['convex']:
            if rng.random() < style['inner']:
                return 'arc', size(style['inner_radius'])
            return None
        if rng.random() < style['design']:
            return 'arc', size(style['design_radius'])
        if rng.random() < style['edge_break']:
            mode = 'arc' if rng.random() < style['edge_arc'] else 'chamfer'
            return mode, size(style['edge_size'])
        return None
    return decide


def add_corner_details(part, rng, scale=1., style=None):
    """Apply a sampled corner style; keep the sharp original if the result is invalid."""
    style = style or corner_style(rng, scale)
    try:
        detailed = round_corners(part, style_policy(rng, style))
    except ValueError:
        return part, dict(style, applied=0, status='rejected')
    if not valid(detailed, False):
        return part, dict(style, applied=0, status='invalid')
    details = detailed.imperfections.pop('corner_details')
    detailed.imperfections = {}
    return detailed, dict(style, applied=len(details), status='ok', details=details)
