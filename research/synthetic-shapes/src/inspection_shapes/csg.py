"""Grammar-free planar parts: CSG outlines, sketch-style openings and tangent-arc curves.

Generation follows the statistics that CAD sketch corpora such as DeepCAD and
SketchGraphs report for real mechanical profiles: closed loops of lines, arcs and
circles; many horizontal/vertical lines but also arbitrary angles; tangency,
concentricity, equal radii and symmetric or patterned holes; dimensions on round
values. Nothing here reads those datasets.

Booleans use Shapely on finely polygonized circles. Exact labels come from
*carriers*: every source line and circle is recorded, Boolean-output vertices are
snapped back onto them (exact line/circle intersections at junctions), and each
output edge is assigned to the carrier it lies on. Consecutive edges on one carrier
become one analytic line or arc primitive; no curve fitting is involved.
"""
import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon, LineString, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from .geometry import Part, Chain, circle, slot, valid
from .realism import loguniform

QUAD = 128
STANDARD_HOLES = [2., 2.5, 3., 3.4, 4., 4.5, 5., 5.5, 6., 6.6, 8., 9., 10., 11., 12., 13.5, 14., 16., 17.5, 20., 22., 26.]


def rounded(value, rng, step=.5, p=.7):
    return float(np.round(value/step)*step) if rng.random() < p and value > 2*step else float(value)


class Shapes:
    """Shapely geometry plus the analytic carriers of its boundary."""

    def __init__(self):
        self.lines, self.circles = [], []

    def line(self, a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        self.lines.append((a, (b-a)/np.linalg.norm(b-a)))

    def polygon(self, vertices):
        v = np.asarray(vertices, float)
        for a, b in zip(v, np.roll(v, -1, axis=0)):
            self.line(a, b)
        return Polygon(v)

    def rect(self, cx, cy, w, h, angle=0.):
        v = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        c, s = np.cos(angle), np.sin(angle)
        return self.polygon(v@np.array([[c, s], [-s, c]])+[cx, cy])

    def disc(self, center, r):
        self.circles.append((np.asarray(center, float), float(r)))
        return Point(center).buffer(r, quad_segs=QUAD)

    def stadium(self, a, b, r):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = (b-a)/np.linalg.norm(b-a); n = np.array([-d[1], d[0]])
        self.line(a+r*n, b+r*n); self.line(a-r*n, b-r*n)
        self.circles += [(a, float(r)), (b, float(r))]
        return LineString([a, b]).buffer(r, quad_segs=QUAD)

    def regular(self, center, R, n, angle):
        t = angle+np.arange(n)*2*np.pi/n
        return self.polygon(np.c_[R*np.cos(t), R*np.sin(t)]+center)

    def halfplane(self, point, normal, size):
        """Region on the ``normal`` side of the line through ``point``."""
        n = np.asarray(normal, float)/np.linalg.norm(normal); d = np.array([-n[1], n[0]]); p = np.asarray(point, float)
        self.line(p-d, p+d)
        return Polygon([p-size*d, p+size*d, p+size*d+size*n, p-size*d+size*n])


def _sag(r):
    step = np.pi/(2*QUAD)
    return r*(1-np.cos(step/2))*1.05+1e-9


def _intersect(a, b):
    from .realism import _intersections
    def form(c):
        return ('line', c[1][0], c[1][1]) if c[0] == 'line' else ('circle', c[1][0], c[1][1])
    return _intersections(form(a), form(b))


def _carrier_distances(points, shapes, scale):
    """Distances [N, C] from points to every carrier and per-carrier tolerances [C]."""
    points = np.asarray(points, float)
    parts, tol = [], []
    if shapes.lines:
        P = np.array([a for a, _ in shapes.lines]); D = np.array([d for _, d in shapes.lines])
        w = points[:, None, :]-P[None]
        parts.append(np.abs(w[..., 0]*D[None, :, 1]-w[..., 1]*D[None, :, 0])); tol += [1e-7*scale]*len(P)
    if shapes.circles:
        C = np.array([c for c, _ in shapes.circles]); R = np.array([r for _, r in shapes.circles])
        parts.append(np.abs(np.linalg.norm(points[:, None, :]-C[None], axis=-1)-R[None])); tol += [_sag(r) for r in R]
    return np.concatenate(parts, axis=1), np.asarray(tol)


def _project(carrier, v):
    kind, data = carrier
    if kind == 'line':
        p, d = data
        return p+((v-p)@d)*d
    c, r = data
    w = v-c
    return c+r*w/max(np.linalg.norm(w), 1e-15)


def tagged_runs(coords, shapes, scale):
    """Snap a Boolean-output ring onto carriers; return [(carrier, vertices)] runs."""
    carriers = [('line', c) for c in shapes.lines]+[('circle', c) for c in shapes.circles]
    v = np.asarray(coords, float)
    if np.allclose(v[0], v[-1]):
        v = v[:-1]
    dist, tol = _carrier_distances(v, shapes, scale)
    score = dist/tol[None]
    near = score <= 1.5
    if not near.any(1).all():
        raise ValueError('Boolean vertex is on no carrier')
    snapped = v.copy()
    for i in range(len(v)):
        ids = np.flatnonzero(near[i]); ids = ids[np.argsort(score[i, ids])]
        point = _project(carriers[ids[0]], v[i])
        if len(ids) > 1:
            roots = _intersect(carriers[ids[0]], carriers[ids[1]])
            if roots:
                root = min(roots, key=lambda r: np.linalg.norm(r-v[i]))
                if np.linalg.norm(root-v[i]) < 4*tol[ids[:2]].max()+1e-7*scale:
                    point = root
        snapped[i] = point
    n = len(snapped)
    mids = (snapped+np.roll(snapped, -1, axis=0))/2
    mdist, _ = _carrier_distances(mids, shapes, scale)
    # A chord of a polygonized circle sags inward by at most the circle tolerance.
    both = near & np.roll(near, -1, axis=0)
    mscore = np.where(both, mdist/tol[None], np.inf)
    edge_carrier = mscore.argmin(1)
    if not np.isfinite(mscore[np.arange(n), edge_carrier]).all() or (mscore[np.arange(n), edge_carrier] > 1.5).any():
        raise ValueError('Boolean edge is on no carrier')
    starts = np.flatnonzero(edge_carrier != np.roll(edge_carrier, 1))
    if not len(starts):
        return [(carriers[edge_carrier[0]], np.vstack([snapped, snapped[:1]]))]
    runs = []
    for k, s in enumerate(starts):
        e = starts[(k+1) % len(starts)]
        if e <= s:
            e += n
        idx = np.arange(s, e+1) % n
        runs.append((carriers[edge_carrier[s]], snapped[idx]))
    return runs


def emit_runs(part, runs, fid, hole, spacing=.055):
    """Add one closed ring made of analytic runs; the ring orientation must already be correct."""
    chain = Chain(part, hole, spacing=spacing)
    if len(runs) == 1 and runs[0][0][0] == 'circle':
        c, r = runs[0][0][1]
        chain.arc(c, r, 0, -2*np.pi if hole else 2*np.pi, fid)
        chain.close()
        return
    for (kind, data), q in runs:
        if np.linalg.norm(q[-1]-q[0]) < 1e-9 and kind == 'line':
            continue
        if kind == 'line':
            a = chain.points[-1][-1] if chain.points else q[0]
            chain.line(a, q[-1], fid)
        else:
            c, r = data
            angles = np.unwrap(np.arctan2(q[:, 1]-c[1], q[:, 0]-c[0]))
            chain.arc(c, r, angles[0], angles[-1], fid)
            if len(chain.points) > 1:
                chain.points[-1][0] = chain.points[-2][-1]
    chain.points[-1][-1] = chain.points[0][0].copy()
    chain.close()


def add_tagged(part, geometry, shapes, fid, hole, scale):
    g = orient(geometry, -1. if hole else 1.)
    emit_runs(part, tagged_runs(np.asarray(g.exterior.coords), shapes, scale), fid, hole)


# ---------------------------------------------------------------- outlines

def _outline(rng, style, size):
    s = Shapes()
    if style == 'rectilinear':
        w, h = size, size*loguniform(rng, .3, 1.)
        g = s.rect(0, 0, rounded(w, rng), rounded(h, rng))
        for _ in range(int(rng.integers(0, 6))):
            ww, hh = w*rng.uniform(.12, .5), h*rng.uniform(.12, .5)
            side = int(rng.integers(4)); t = rng.uniform(-.45, .45)
            cx, cy = [(t*w, -h/2), (w/2, t*h), (t*w, h/2), (-w/2, t*h)][side]
            piece = s.rect(cx, cy, ww, hh)
            g = g.union(piece) if rng.random() < .45 else g.difference(piece)
        if rng.random() < .35:
            corner = np.array([w/2*rng.choice([-1, 1]), h/2*rng.choice([-1, 1])])
            normal = -corner*np.array([1/w, 1/h])*rng.uniform(.6, 1.6, 2)
            cut = rng.uniform(.12, .35)*min(w, h)
            g = g.difference(s.halfplane(corner-cut*corner/np.linalg.norm(corner)*np.sqrt(2)*.7, -normal, 4*size))
        if rng.random() < .25:
            r = h*rng.uniform(.25, .5); g = g.union(s.disc([w/2*rng.choice([-1, 1]), 0], r))
    elif style == 'union':
        g = None
        for i in range(int(rng.integers(2, 5))):
            kind = str(rng.choice(['rect', 'disc', 'stadium', 'regular'], p=[.35, .3, .2, .15]))
            if g is None:
                center = np.zeros(2); scale = size/2
            else:
                b = g.boundary; center = np.asarray(b.interpolate(rng.uniform(0, b.length)).coords[0]); scale = size/2*rng.uniform(.25, .7)
            angle = float(rng.choice([0., np.pi/2, rng.uniform(0, np.pi)], p=[.4, .2, .4]))
            if kind == 'rect':
                piece = s.rect(*center, rounded(scale*2*rng.uniform(.5, 1.2), rng), rounded(scale*rng.uniform(.4, 1.2), rng), angle)
            elif kind == 'disc':
                piece = s.disc(center, rounded(scale*rng.uniform(.5, 1.), rng))
            elif kind == 'stadium':
                L = scale*rng.uniform(.8, 2.); d = np.array([np.cos(angle), np.sin(angle)])
                piece = s.stadium(center-L/2*d, center+L/2*d, scale*rng.uniform(.25, .6))
            else:
                piece = s.regular(center, scale*rng.uniform(.6, 1.), int(rng.choice([3, 5, 6, 8])), angle)
            g = piece if g is None else g.union(piece)
    elif style == 'link':
        k = int(rng.choice([2, 3], p=[.75, .25])); radii = [size*rng.uniform(.12, .25) for _ in range(k)]
        centers = [np.zeros(2)]
        for i in range(1, k):
            angle = rng.uniform(-.6, .6)+(np.pi if i == 2 else 0.)
            centers.append(centers[0]+size*rng.uniform(.45, .8)*np.array([np.cos(angle), np.sin(angle)]))
        g = unary_union([s.disc(c, rounded(r, rng, .25)) for c, r in zip(centers, radii)])
        for c, r in zip(centers[1:], radii[1:]):
            web = min(r, radii[0])*rng.uniform(.6, 1.6)
            if rng.random() < .5:
                g = g.union(s.stadium(centers[0], c, web))
            else:
                d = c-centers[0]; angle = np.arctan2(d[1], d[0])
                g = g.union(s.rect(*(centers[0]+c)/2, np.linalg.norm(d), 2*web, angle))
    else:  # radial
        R = size/2*rng.uniform(.55, .8)
        base = s.disc([0, 0], R) if rng.random() < .6 else s.regular([0, 0], R, int(rng.choice([4, 6, 8])), rng.uniform(0, np.pi))
        n = int(rng.integers(2, 9)); phase = rng.uniform(0, 2*np.pi)
        arm, width = size/2*rng.uniform(.85, 1.), R*rng.uniform(.15, .4)
        pieces = []
        for t in phase+np.arange(n)*2*np.pi/n:
            d = np.array([np.cos(t), np.sin(t)])
            pieces.append(s.stadium(d*R*.5, d*arm, width) if rng.random() < .6 else s.rect(*(d*(R*.5+arm)/2), arm-R*.5, 2*width, t))
        g = unary_union([base]+pieces) if rng.random() < .7 else base.difference(unary_union(
            [s.disc(d*R, width) for d in [np.array([np.cos(t), np.sin(t)]) for t in phase+np.arange(n)*2*np.pi/n]]))
    if rng.random() < .25 and g is not None:
        b = g.boundary
        for _ in range(int(rng.integers(1, 3))):
            point = np.asarray(b.interpolate(rng.uniform(0, b.length)).coords[0])
            g = g.difference(s.disc(point, size*rng.uniform(.04, .12)))
    return g, s


# ---------------------------------------------------------------- openings

def _opening(rng, kind, center, size, angle):
    """(shapely geometry, emitter) for one opening at ``center`` with nominal ``size``."""
    s = Shapes()
    if kind == 'hole':
        d = min(STANDARD_HOLES, key=lambda x: abs(x-size)) if rng.random() < .7 else size
        return Point(center).buffer(d/2, quad_segs=32), lambda p: circle(p, center, d/2, 'hole', True)
    if kind == 'slot':
        L, W = size*rng.uniform(1.6, 3.5), size*rng.uniform(.6, 1.)
        g = LineString([np.asarray(center)-(L-W)/2*np.array([np.cos(angle), np.sin(angle)]),
                        np.asarray(center)+(L-W)/2*np.array([np.cos(angle), np.sin(angle)])]).buffer(W/2, quad_segs=32)
        return g, lambda p: slot(p, list(center), L, W, angle)
    if kind == 'curved_slot':
        Rm = size*rng.uniform(1.5, 4.); w = size*rng.uniform(.25, .45); span = rng.uniform(.4, 1.4)
        pivot = np.asarray(center)-Rm*np.array([np.cos(angle), np.sin(angle)])
        a0, a1 = angle-span/2, angle+span/2
        def emit(p):
            fid = p.feature('slot', style='curved', mean_radius=float(Rm), width=float(2*w))
            c = Chain(p, True)
            u = lambda t: np.array([np.cos(t), np.sin(t)])
            # Clockwise (hole): outer arc a1->a0, end cap, inner arc a0->a1, end cap.
            c.arc(pivot, Rm+w, a1, a0, fid)
            c.arc(pivot+Rm*u(a0), w, a0, a0-np.pi, fid); c.points[-1][0] = c.points[-2][-1]
            c.arc(pivot, Rm-w, a0, a1, fid); c.points[-1][0] = c.points[-2][-1]
            c.arc(pivot+Rm*u(a1), w, a1+np.pi, a1, fid); c.points[-1][0] = c.points[-2][-1]
            c.points[-1][-1] = c.points[0][0].copy(); c.close()
        t = np.linspace(a0, a1, 64)
        spine = LineString(np.c_[pivot[0]+Rm*np.cos(t), pivot[1]+Rm*np.sin(t)])
        return spine.buffer(w, quad_segs=16), emit
    if kind == 'rounded_rect':
        w, h = size*rng.uniform(1., 2.5), size*rng.uniform(.6, 1.2); r = min(w, h)*rng.uniform(.1, .45)
        g = s.rect(*center, w-2*r, h-2*r, angle).buffer(r, quad_segs=QUAD)
        s.lines.clear()
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            s.circles.append((np.asarray(center)+R@[sx*(w/2-r), sy*(h/2-r)], float(r)))
        for k in range(4):
            p0 = s.circles[k][0]; p1 = s.circles[(k+1) % 4][0]; d = (p1-p0)/np.linalg.norm(p1-p0); n = np.array([d[1], -d[0]])
            s.line(p0+r*n, p1+r*n)
        return g, lambda p: add_tagged(p, g, s, p.feature('rounded_profile', style='rounded_rect'), True, size)
    if kind == 'polygon':
        n = int(rng.choice([3, 4, 6, 8], p=[.2, .35, .35, .1]))
        g = s.regular(center, size*.6, n, angle)
        return g, lambda p: add_tagged(p, g, s, p.feature('polygon', side_count=n), True, size)
    if kind in ('d', 'double_d'):
        r = size/2; s2 = Shapes(); g = s2.disc(center, r)
        d = np.array([np.cos(angle), np.sin(angle)]); flat = r*rng.uniform(.55, .85)
        g = g.difference(s2.halfplane(np.asarray(center)+flat*d, d, 4*r))
        if kind == 'double_d':
            g = g.difference(s2.halfplane(np.asarray(center)-flat*d, -d, 4*r))
        return g, lambda p: add_tagged(p, g, s2, p.feature('rounded_profile', style=kind), True, size)
    # Ellipse: deliberately outside the line/arc/circle taxonomy (unsupported, zero-weight).
    a, b = size*.6, size*.6*rng.uniform(.45, .8)
    t = np.linspace(0, 2*np.pi, 721)[:-1][::-1]
    pts = np.c_[a*np.cos(t), b*np.sin(t)]@np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])+center
    def emit(p):
        fid = p.feature('rounded_profile', style='ellipse', unsupported=True)
        c = Chain(p, True); c.add(np.vstack([pts, pts[:1]]), fid, 'freeform', profile='ellipse', unsupported=True); c.close()
    return Polygon(pts), emit


OPENING_KINDS = ['hole', 'slot', 'curved_slot', 'rounded_rect', 'polygon', 'd', 'double_d', 'ellipse']
OPENING_P = [.5, .15, .05, .1, .08, .04, .04, .04]


def add_openings(part, rng, outline, size):
    """Openings in sketch-like layouts: scatter, grid, line, bolt circle, concentric."""
    count = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 8, 10, 12], p=[.1, .14, .16, .14, .13, .1, .09, .06, .05, .03]))
    if not count:
        return []
    layout = str(rng.choice(['scatter', 'grid', 'line', 'circle'], p=[.45, .2, .15, .2]))
    kind = str(rng.choice(OPENING_KINDS, p=OPENING_P))
    mixed = rng.random() < .35
    unit = size*loguniform(rng, .04, .14)
    minx, miny, maxx, maxy = outline.bounds
    center = np.asarray(outline.representative_point().coords[0]) if not outline.centroid.within(outline) else np.asarray(outline.centroid.coords[0])
    if layout == 'grid':
        cols = int(rng.integers(1, 5)); rows = max(1, int(np.ceil(count/cols)))
        pitch = unit*rng.uniform(2.2, 4.)
        spots = [center+[(i-(cols-1)/2)*pitch, (j-(rows-1)/2)*pitch] for j in range(rows) for i in range(cols)][:count]
    elif layout == 'line':
        pitch = unit*rng.uniform(2.2, 4.); angle = float(rng.choice([0., np.pi/2, rng.uniform(0, np.pi)]))
        d = np.array([np.cos(angle), np.sin(angle)])
        spots = [center+(i-(count-1)/2)*pitch*d for i in range(count)]
    elif layout == 'circle':
        radius = min(maxx-minx, maxy-miny)*rng.uniform(.18, .38); phase = rng.uniform(0, 2*np.pi)
        spots = [center+radius*np.array([np.cos(t), np.sin(t)]) for t in phase+np.arange(count)*2*np.pi/count]
        if rng.random() < .6:
            spots.append(center)
    else:
        spots = [np.array([rng.uniform(minx, maxx), rng.uniform(miny, maxy)]) for _ in range(count*4)]
    margin = max(.6, .25*unit)
    inner = outline.buffer(-margin)
    placed, emitters, refs = [], [], []
    angle0 = float(rng.choice([0., np.pi/2, rng.uniform(0, np.pi)]))
    for spot in spots:
        if len(emitters) >= count+1:
            break
        k = str(rng.choice(OPENING_KINDS, p=OPENING_P)) if mixed else kind
        angle = angle0 if layout != 'circle' else float(np.arctan2(*(spot-center)[::-1]))
        if k == 'curved_slot' and layout == 'circle':
            angle = float(np.arctan2(*(spot-center)[::-1]))
        size_k = unit*(1.6 if layout == 'circle' and np.allclose(spot, center) else 1.)
        g, emit = _opening(rng, k, spot, size_k, angle)
        if not g.is_valid or not inner.contains(g) or any(g.distance(o) < margin for o in placed):
            continue
        placed.append(g); emitters.append(emit)
    for emit in emitters:
        before = len(part.features)
        emit(part)
        refs.append(before+1)
    if layout in ('grid', 'line', 'circle') and len(refs) > 1:
        part.patterns.append(dict(type=layout, feature_refs=refs))
    return placed


def csg_part(rng, style=None):
    for _ in range(10):
        p = _csg_part(rng, style)
        if valid(p, False):
            return p
    raise ValueError('Could not build a valid CSG part')


def _csg_part(rng, style=None):
    style = style or str(rng.choice(['rectilinear', 'union', 'link', 'radial'], p=[.35, .35, .15, .15]))
    size = loguniform(rng, 20., 150.)
    for _ in range(20):
        g, shapes = _outline(rng, style, size)
        if g is not None and g.geom_type == 'Polygon' and g.is_valid and g.area > .05*size*size:
            break
    else:
        raise ValueError('Could not build a CSG outline')
    # Remove interior voids from unions; openings are generated explicitly.
    g = Polygon(g.exterior)
    p = Part('csg_part')
    root = p.feature('external_profile', generator='csg-v1', style=style)
    add_tagged(p, g, shapes, root, False, size)
    if style == 'link' or (style == 'radial' and rng.random() < .7):
        # Concentric bores on circular carriers: a dominant sketch constraint.
        for c, r in shapes.circles[:3]:
            hole = Point(c).buffer(r*.5, quad_segs=32)
            if g.buffer(-.6).contains(hole) and rng.random() < .8:
                d = min(STANDARD_HOLES, key=lambda x: abs(x-r*rng.uniform(.6, 1.1)))
                if d/2 < r*.75:
                    circle(p, c, d/2, 'hole', True)
        existing = unary_union([Polygon(ring.nominal) for ring in p.rings[1:]]) if len(p.rings) > 1 else None
        outline = g if existing is None else g.difference(existing.buffer(1.))
        if outline.geom_type != 'Polygon':
            outline = g
    else:
        outline = g
    add_openings(p, rng, outline, size)
    p.patterns.append(dict(type='csg', style=style, size_mm=float(size)))
    return p


# ---------------------------------------------------------------- tangent-arc curved profiles

def _arc_ring(part, arcs, fid, hole=False, spacing=.055):
    """arcs: [(center, r, a0, a1)] continuous, defining a CCW ring (reversed for holes)."""
    chain = Chain(part, hole, spacing=spacing)
    seq = arcs if not hole else [(c, r, a1, a0) for c, r, a0, a1 in reversed(arcs)]
    for c, r, a0, a1 in seq:
        chain.arc(np.asarray(c, float), float(r), float(a0), float(a1), fid)
        if len(chain.points) > 1:
            chain.points[-1][0] = chain.points[-2][-1]
    chain.points[-1][-1] = chain.points[0][0].copy()
    chain.close()


def _ccw(a0, a1):
    return a0, a0+((a1-a0) % (2*np.pi))


def _cw(a0, a1):
    return a0, a0-((a0-a1) % (2*np.pi))


def lobed_profile(rng):
    """3–12 convex lobes joined by tangent concave arcs (coupling/star-knob profile)."""
    n = int(rng.integers(3, 13)); Rp = loguniform(rng, 10., 40.)
    a = Rp*np.sin(np.pi/n)*rng.uniform(.35, .9)
    b = max(Rp*np.sin(np.pi/n)-a, 0)+Rp*rng.uniform(.05, .3)
    D = Rp*np.cos(np.pi/n)+np.sqrt((a+b)**2-(Rp*np.sin(np.pi/n))**2)
    lobes = [Rp*np.array([np.cos(t), np.sin(t)]) for t in 2*np.pi*np.arange(n)/n]
    valleys = [D*np.array([np.cos(t), np.sin(t)]) for t in 2*np.pi*(np.arange(n)+.5)/n]
    touch = lambda c, p: c+a*(p-c)/np.linalg.norm(p-c)
    angle = lambda c, x: float(np.arctan2(*(x-c)[::-1]))
    arcs = []
    for i in range(n):
        c, prev_v, next_v, nxt = lobes[i], valleys[i-1], valleys[i], lobes[(i+1) % n]
        t1, t2 = touch(c, prev_v), touch(c, next_v)
        arcs.append((c, a, *_ccw(angle(c, t1), angle(c, t2))))
        t3 = touch(nxt, next_v)
        arcs.append((next_v, b, *_cw(angle(next_v, t2), angle(next_v, t3))))
    p = Part('lobed_profile')
    _arc_ring(p, arcs, p.feature('cam_profile', style='tangent_lobes', lobes=n))
    bore = Rp*rng.uniform(.15, .4)
    circle(p, [0, 0], min(STANDARD_HOLES, key=lambda x: abs(x-2*bore))/2, 'hole', True)
    return p


def tangent_cam(rng):
    """Base circle and nose circle joined by tangent straight flanks (or tangent arcs)."""
    Rb = loguniform(rng, 8., 35.); rn = Rb*rng.uniform(.25, .7); d = max(Rb*rng.uniform(.6, 1.3), 1.15*Rb-rn)
    phi = np.arccos((Rb-rn)/d)
    nose = np.array([d, 0.])
    p = Part('tangent_cam'); fid = p.feature('cam_profile', style='tangent_cam')
    chain = Chain(p)
    chain.arc(nose, rn, -phi, phi, fid)
    chain.line(chain.points[-1][-1], Rb*np.array([np.cos(phi), np.sin(phi)]), fid)
    chain.arc([0, 0], Rb, phi, 2*np.pi-phi, fid); chain.points[-1][0] = chain.points[-2][-1]
    chain.line(chain.points[-1][-1], chain.points[0][0], fid)
    chain.points[-1][-1] = chain.points[0][0].copy(); chain.close()
    # Bore with an optional keyway: a disc/rectangle union emitted as one hole.
    r = Rb*rng.uniform(.18, .35)
    if rng.random() < .5:
        s = Shapes(); g = s.disc([0, 0], r).union(s.rect(r, 0, r*.9, r*.5))
        add_tagged(p, g, s, p.feature('hole', keyway=True), True, Rb)
    else:
        circle(p, [0, 0], r, 'hole', True)
    return p


def round_groove_pulley(rng):
    """Circle with N circular tooth grooves (timing-pulley / sprocket-like)."""
    R = loguniform(rng, 10., 40.); n = int(rng.integers(12, 61))
    g_r = 2*np.pi*R/n*rng.uniform(.18, .32); depth = rng.uniform(.0, .6)*g_r
    s = Shapes(); g = s.disc([0, 0], R)
    for t in 2*np.pi*np.arange(n)/n:
        g = g.difference(s.disc((R-depth+g_r*.0)*np.array([np.cos(t), np.sin(t)]), g_r))
    p = Part('round_groove_pulley')
    add_tagged(p, g, s, p.feature('tooth_profile', count=n, domain='angular', style='round_groove'), False, R)
    circle(p, [0, 0], R*rng.uniform(.15, .35), 'hole', True)
    return p


CURVED = {'lobed_profile': lobed_profile, 'tangent_cam': tangent_cam, 'round_groove_pulley': round_groove_pulley}
