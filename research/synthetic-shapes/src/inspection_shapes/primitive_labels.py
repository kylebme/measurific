"""Versioned geometric labels, independent of manufacturing feature names.

Nominal construction defines identity; all selections retain the imperfect edges.
Legacy semantic annotations remain available as provenance, never model input.
"""
import numpy as np

LABEL_VERSION = 'primitives-compounds-v2'
BASE_NAMES = ['unlabeled', 'line', 'arc', 'circle']
COMPOUND_NAMES = ['none', 'full_round_slot', 'triangular_thread',
                  'trapezoidal_thread', 'tooth_profile']


def _cross2d(a, b):
    """Scalar/vectorized 2-D determinant, independent of NumPy cross changes."""
    a, b = np.asarray(a), np.asarray(b)
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def _assumed_freeform_class(pid, primitive_ids, contours):
    """Map reference-free curves onto the closest useful primitive contract.

    A freeform that owns a complete ring is treated as a circle. Partial spans
    that are effectively straight are lines; all other partial spans are arcs.
    The nominal curve is used only to build targets and is never model input.
    """
    nominal = np.asarray(contours['nominal_xy_mm'], float)
    for lo, hi in zip(contours['ring_offsets'][:-1], contours['ring_offsets'][1:]):
        selected = np.flatnonzero(primitive_ids[lo:hi] == pid)
        if not len(selected):
            continue
        if len(selected) == hi-lo:
            return 3
        # Primitive IDs are construction-span IDs and therefore contiguous.
        start, end = lo+selected[0], lo+selected[-1]
        endpoint = lo if end+1 == hi else end+1
        q = np.vstack([nominal[start:end+1], nominal[endpoint]])
        chord = q[-1]-q[0]
        length = np.linalg.norm(chord)
        if length <= 1e-12:
            return 2
        distance = np.abs(_cross2d(chord, q-q[0]))/length
        return 1 if distance.max() <= .002*length else 2
    raise ValueError(f'Freeform primitive {pid} has no observed edges')


def labels(metadata, contours):
    primitives = {p['id']: p for p in metadata['primitives']}
    primitive_ids = np.asarray(contours['edge_primitive_id'])
    base = np.zeros(len(primitive_ids), np.uint8)
    instances = primitive_ids.astype(np.int32).copy()
    for pid in np.unique(primitive_ids):
        p = primitives[int(pid)]
        kind = p['kind']
        cls = {'line': 1, 'circular_arc': 2}.get(kind)
        if kind == 'freeform':
            cls = _assumed_freeform_class(pid, primitive_ids, contours)
        elif cls is None:
            raise ValueError(f'Unsupported primitive kind: {kind!r}')
        params = p['nominal_parameters']
        if kind == 'circular_arc' and np.isclose(abs(params['end_angle']-params['start_angle']), 2*np.pi):
            cls = 3
        base[primitive_ids == pid] = cls
    # One physical straight surface can cross multiple construction primitives.
    nominal = contours['nominal_xy_mm']
    for lo, hi in zip(contours['ring_offsets'][:-1], contours['ring_offsets'][1:]):
        pids = instances[lo:hi]
        starts = np.flatnonzero(pids != np.roll(pids, 1))
        for start in starts:
            a, b = lo+(start-1) % (hi-lo), lo+start
            if base[a] != 1 or base[b] != 1:
                continue
            pa, pb = primitives[int(primitive_ids[a])], primitives[int(primitive_ids[b])]
            da = np.subtract(pa['nominal_parameters']['end'], pa['nominal_parameters']['start'])
            db = np.subtract(pb['nominal_parameters']['end'], pb['nominal_parameters']['start'])
            if np.dot(da, db) > 0 and abs(_cross2d(da, db)) <= 1e-8*np.linalg.norm(da)*np.linalg.norm(db):
                old, new = instances[b], instances[a]
                instances[instances == old] = new
    compound = np.zeros(len(base), np.uint8)
    compound_ids = np.zeros(len(base), np.int32)
    groups = []
    for f in metadata['features']:
        semantic, params = f['semantic'], f['nominal_parameters']
        cls = 0
        if semantic == 'slot':
            # Only the analytic two-line, two-semicircle stadium is a full round slot.
            members = [p for p in primitives.values() if p['feature_id'] == f['id']]
            arcs = [p for p in members if p['kind'] == 'circular_arc']
            if len(members) == 4 and len(arcs) == 2 and all(
                np.isclose(abs(p['nominal_parameters']['end_angle']-p['nominal_parameters']['start_angle']), np.pi)
                for p in arcs):
                cls = 1
        elif semantic == 'thread':
            style = params.get('style')
            if style not in ('v', 'trapezoid', 'square'):
                raise ValueError(f'Missing or unsupported thread style: {style!r}')
            cls = 2 if style == 'v' else 3
        elif semantic == 'tooth_profile':
            cls = 4
        if not cls:
            continue
        mask = contours['edge_feature_id'] == f['id']
        if not mask.any():
            continue
        # Compound membership consists of whole observable base instances. A
        # nominal thread endpoint cannot split an otherwise continuous root land.
        mask=np.isin(instances,np.unique(instances[mask]))
        if np.any(compound_ids[mask]):
            raise ValueError('Distinct compound instances overlap the same base instance')
        ident = len(groups)+1
        compound[mask] = cls
        compound_ids[mask] = ident
        groups.append({'id': ident, 'class_id': cls, 'class': COMPOUND_NAMES[cls],
                       'base_instance_ids': np.unique(instances[mask]).tolist()})
    return {'edge_base_class': base, 'edge_base_instance': instances,
            'edge_compound_class': compound, 'edge_compound_instance': compound_ids}, groups


def add_polycurve(chain, vertices, fid, arbitrary=False):
    """Recover exact circular/linear construction spans before dense sampling.

Used for polygonized Shapely outlines. Noncircular smooth spans remain freeform;
harmonic cams explicitly bypass decomposition. Fits use nominal geometry only.
"""
    q = np.asarray(vertices, float)
    if np.allclose(q[0], q[-1]):
        q = q[:-1]
    if arbitrary:
        chain.add(np.vstack([q, q[0]]), fid, 'freeform')
        return
    d = np.roll(q, -1, axis=0)-q
    turn = np.arctan2(_cross2d(np.roll(d, 1, axis=0), d), (np.roll(d, 1, axis=0)*d).sum(1))
    # Begin at a corner or curvature transition, avoiding artificial seam splits.
    at = int(np.argmax(np.abs(turn-np.roll(turn, 1))))
    q = np.vstack([np.roll(q, -at, axis=0), q[at]])
    tol = max(np.ptp(q, axis=0).max(), 1.)*1e-10

    def circle_at(i, minimum=5):
        if i+minimum > len(q):
            return None
        a, b, c = q[i:i+3]
        mat = 2*np.stack([b-a, c-a])
        if abs(np.linalg.det(mat)) < 1e-12:
            return None
        center = np.linalg.solve(mat, np.array([np.dot(b-a, b-a), np.dot(c-a, c-a)]))+a
        radius = np.linalg.norm(a-center)
        if np.max(np.abs(np.linalg.norm(q[i:i+minimum]-center, axis=1)-radius)) > tol:
            return None
        return center, radius

    i = 0
    while i < len(q)-1:
        circ = circle_at(i)
        if circ is not None:
            center, radius = circ
            j = i+4
            while j+1 < len(q) and abs(np.linalg.norm(q[j+1]-center)-radius) <= tol:
                j += 1
            angles = np.unwrap(np.arctan2(q[i:j+1, 1]-center[1], q[i:j+1, 0]-center[0]))
            count=max(4,int(np.ceil(abs(angles[-1]-angles[0])*radius/chain.spacing)))
            t=np.linspace(angles[0],angles[-1],count+1)
            points=center+radius*np.c_[np.cos(t),np.sin(t)]
            points[0],points[-1]=q[i],q[j]
            chain.add(points,fid,'circular_arc',center=center.tolist(),radius=float(radius),
                      start_angle=float(angles[0]),end_angle=float(angles[-1]))
        else:
            direction = q[i+1]-q[i]
            j = i+1
            while j+1 < len(q) and abs(_cross2d(direction, q[j+1]-q[i])) <= tol*np.linalg.norm(direction):
                j += 1
            if j > i+1 or i+2 == len(q):
                chain.line(q[i], q[j], fid)
            else:
                # This path is used only for line/circle constructions from
                # Shapely. Clipping may leave a circular fragment with <5 vertices.
                u, v = q[i+1]-q[i], q[i+2]-q[i+1]
                angle = abs(np.arctan2(_cross2d(u, v), np.dot(u, v)))
                if angle > .15 or circle_at(i+1) is not None:
                    chain.line(q[i], q[j], fid)
                else:
                    # Three arbitrary points always define a circle. Accept this
                    # short-fragment fallback only for equal-length tessellation
                    # chords, never a long straight plus a short rounded join.
                    lengths=np.linalg.norm(np.diff(q[i:i+3],axis=0),axis=1)
                    circ=circle_at(i,3) if lengths.max()<=1.01*lengths.min() else None
                    if circ is None:
                        chain.line(q[i],q[j],fid)
                    else:
                        center,radius=circ;j=i+2
                        angle=np.unwrap(np.arctan2(q[i:j+1,1]-center[1],q[i:j+1,0]-center[0]))
                        chain.add(q[i:j+1],fid,'circular_arc',center=center.tolist(),radius=float(radius),
                                  start_angle=float(angle[0]),end_angle=float(angle[-1]))
        i = j
