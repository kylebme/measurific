"""Training-free decomposition of ordered observations into lines and circular arcs.

Candidate-boundary dynamic programming, not an implementation of Gribov's exact
global optimizer. See GEOMETRIC.md for the research, tolerance contract and limits.
Nothing in this module imports torch, reads nominal geometry, or changes points.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import ring_geometry

BASE_NAMES = ['unlabeled', 'line', 'arc', 'circle']
COMPOUND_NAMES = ['none', 'full_round_slot', 'triangular_thread',
                  'trapezoidal_thread', 'tooth_profile']


@dataclass(frozen=True)
class GeometricConfig:
    # Absolute observation-space distance, NOT a fraction of the part size.
    tolerance: float = .75
    min_arc_angle: float = np.deg2rad(2)
    candidate_hops: int = 48
    refinement_passes: int = 2

    def __post_init__(self):
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError('tolerance must be finite and positive')
        if not np.isfinite(self.min_arc_angle) or not 0 < self.min_arc_angle < np.pi:
            raise ValueError('min_arc_angle must be in (0, pi)')
        for name, minimum in [('candidate_hops', 2), ('refinement_passes', 0)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')


def _weights(q):
    d = np.linalg.norm(np.diff(q, axis=0), axis=1)
    return (np.r_[d, 0] + np.r_[0, d]) / 2


def _fit(q, kind, cfg):
    """Fit in a local frame and check every original vertex and edge interior.

Arc-length weights prevent a densely sampled corner dominating a long surface.
Circular candidates get geometric Gauss-Newton refinement, not just Kasa's fit.
"""
    w = _weights(q)
    length = float(w.sum())
    if length <= 0:
        return None
    w /= length
    origin = np.sum(q*w[:, None], axis=0)
    z = q-origin
    if kind == 'line':
        if np.linalg.norm(q[-1]-q[0]) < 1e-12:
            return None  # a closed loop is never one open line
        _, v = np.linalg.eigh((z*w[:, None]).T@z)
        axis = v[:, -1]
        if (q[-1]-q[0])@axis < 0:
            axis = -axis
        residual = z@np.array([-axis[1], axis[0]])
        along = z@axis
        if np.max(np.maximum.accumulate(along)-along) > 2*cfg.tolerance:
            return None
        values = dict(origin=origin.tolist(), direction=axis.tolist(),
                      start=(origin+along.min()*axis).tolist(), end=(origin+along.max()*axis).tolist())
        error = float(np.max(np.abs(residual)))
    else:
        if len(q) < 6:
            return None
        scale = float(np.max(np.linalg.norm(z, axis=1)))
        if scale <= cfg.tolerance:
            return None
        a = z/scale
        design = np.c_[2*a, np.ones(len(a))]
        try:
            solution, _, rank, _ = np.linalg.lstsq(design*np.sqrt(w[:, None]),
                                                  (a*a).sum(1)*np.sqrt(w), rcond=1e-10)
            if rank < 3:
                return None
            center = solution[:2]
            radius = np.sqrt(max(0., solution[2]+center@center))
            for _ in range(4):
                delta = a-center
                norm = np.linalg.norm(delta, axis=1)
                if norm.min() < 1e-10 or not 0 < radius < 1e5:
                    return None
                jac = np.c_[-delta/norm[:, None], -np.ones(len(a))]
                step = np.linalg.lstsq(jac*np.sqrt(w[:, None]),
                                       (radius-norm)*np.sqrt(w), rcond=1e-10)[0]
                center += step[:2]
                radius += step[2]
                if np.linalg.norm(step) < 1e-9:
                    break
        except np.linalg.LinAlgError:
            return None
        center = center*scale  # still local: avoid cancellation at large coordinates
        radius *= scale
        if not np.isfinite(radius) or radius <= cfg.tolerance:
            return None
        delta = z-center
        residual = np.linalg.norm(delta, axis=1)-radius
        angles = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
        sweep = float(angles[-1]-angles[0])
        if not cfg.min_arc_angle <= abs(sweep) <= 2*np.pi+1e-6:
            return None
        progress = angles*np.sign(sweep)
        if np.max(np.maximum.accumulate(progress)-progress)*radius > 2*cfg.tolerance:
            return None
        # The finite diagnostic arc must contain all angular projections, not
        # merely the endpoint angles (which may have small observation noise).
        closed = np.linalg.norm(q[-1]-q[0]) < 1e-12
        start_angle = float(angles[0] if closed else (angles.min() if sweep > 0 else angles.max()))
        if not closed:
            sweep = float(np.sign(sweep)*np.ptp(angles))
            if abs(sweep) > 2*np.pi:
                return None
        # Radial distance has an interior minimum on a chord. Checking it also
        # rejects a sparse polygon whose vertices happen to lie on a circle.
        edge = np.diff(delta, axis=0)
        t = np.clip(-np.sum(delta[:-1]*edge, axis=1)/np.sum(edge*edge, axis=1), 0, 1)
        inner = np.linalg.norm(delta[:-1]+t[:, None]*edge, axis=1)-radius
        error = float(max(np.max(np.abs(residual)), np.max(np.abs(inner))))
        values = dict(center=(center+origin).tolist(), radius=float(radius), start_angle=start_angle, sweep=sweep)
    if error > cfg.tolerance*(1+1e-7):
        return None
    return dict(kind=kind, max_error=error, rms_error=float(np.sqrt(w@(residual*residual))),
                squared_error=float(length*(w@(residual*residual))), length=length, **values)


def _best(q, cfg):
    # An unresolved arc is deliberately a line, independent of family/context.
    line = _fit(q, 'line', cfg)
    return line if line is not None else _fit(q, 'arc', cfg)


def _seed_candidates(q, cfg):
    """Fit-aware split proposals avoid an O(m^2) search on every tessellated arc.

Every resulting interval already has a valid primitive. The later graph may
merge these intervals; local boundary refinement is not confined to proposals.
This deliberately trades exhaustive breakpoint search for interactive latency.
"""
    keep = {0, len(q)-1}
    stack = [(0, len(q)-1)]
    while stack:
        a, b = stack.pop()
        if _best(q[a:b+1], cfg) is not None:
            continue
        d = q[b]-q[a]
        z = q[a+1:b]-q[a]
        if not len(z):
            raise RuntimeError('A single nonzero edge must admit a line fit')
        t = np.clip(z@d/max(d@d, 1e-30), 0, 1)
        error = np.linalg.norm(z-t[:, None]*d, axis=1)
        at = a+1+int(np.argmax(error))
        keep.add(at)
        stack.extend([(a, at), (at, b)])
    return np.array(sorted(keep), dtype=int)


def _optimize(q, cfg):
    candidates = _seed_candidates(q, cfg)
    n = len(candidates)
    costs = [None]*n
    costs[0] = (0, 0.)
    previous = np.full(n, -1, int)
    fits = [None]*n
    for j in range(1, n):
        # Bounded candidate graph, no invalid assumption that a rejected arc
        # implies every longer arc must fail. Full-span merge is tried later.
        for i in range(max(0, j-cfg.candidate_hops), j):
            penalty = costs[i][0]+2
            if costs[j] is not None and penalty > costs[j][0]:
                continue
            support = q[candidates[i]:candidates[j]+1]
            fit = _fit(support, 'line', cfg)
            if fit is None and (costs[j] is None or costs[i][0]+3 <= costs[j][0]):
                fit = _fit(support, 'arc', cfg)
            if fit is None:
                continue
            score = (costs[i][0]+(2 if fit['kind'] == 'line' else 3),
                     costs[i][1]+fit['squared_error'])
            if costs[j] is None or score < costs[j]:
                costs[j], previous[j], fits[j] = score, i, fit
        if previous[j] < 0:
            raise RuntimeError('Candidate graph lost its guaranteed line covering')
    spans = []
    j = n-1
    while j:
        i = int(previous[j])
        spans.append((int(candidates[i]), int(candidates[j]), fits[j]))
        j = i
    return list(reversed(spans)), n


def _refine(q, spans, cfg):
    """Move proposed joins toward the minimum residual, retaining both models."""
    for _ in range(cfg.refinement_passes):
        for i in range(len(spans)-1):
            a, b, left = spans[i]
            _, c, right = spans[i+1]
            # Search is bounded in arc length, not number/density of observations.
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q[a:c+1], axis=0), axis=1))]
            reach = min(.4*min(left['length'], right['length']),
                        max(8*cfg.tolerance, .1*min(left['length'], right['length'])))
            choices = np.flatnonzero(np.abs(s-s[b-a]) <= reach)+a
            choices = choices[(choices > a) & (choices < c)]
            if len(choices) > 33:
                choices = choices[np.linspace(0, len(choices)-1, 33).astype(int)]
            best = left['squared_error']+right['squared_error']
            for at in choices:
                first = _fit(q[a:at+1], left['kind'], cfg)
                if first is None:
                    continue
                second = _fit(q[at:c+1], right['kind'], cfg)
                if second is None:
                    continue
                error = first['squared_error']+second['squared_error']
                if error < best:
                    best = error
                    spans[i], spans[i+1] = (a, int(at), first), (int(at), c, second)
    return spans


def segment_ring(q, config):
    """Return contiguous raw-edge runs, including a correctly merged cyclic seam."""
    q = np.asarray(q, float)
    n = len(q)
    whole = _fit(np.vstack([q, q[0]]), 'arc', config)
    if whole is not None and abs(abs(whole['sweep'])-2*np.pi) < 1e-5:
        return [(np.arange(n), dict(whole, kind='circle'))], dict(candidates=1)
    # Geometry-defined seam, independent of input array's starting vertex.
    edge = np.roll(q, -1, axis=0)-q
    edge /= np.linalg.norm(edge, axis=1)[:, None]
    turn = np.abs(np.arctan2(edge[:, 0]*np.roll(edge[:, 1], 1)-edge[:, 1]*np.roll(edge[:, 0], 1),
                            np.sum(edge*np.roll(edge, 1, axis=0), axis=1)))
    start = int(np.argmax(turn))
    order = (start+np.arange(n)) % n
    closed = np.vstack([q[order], q[start]])
    spans, count = _optimize(closed, config)
    spans = _refine(closed, spans, config)
    result = [(order[a:b], fit) for a, b, fit in spans]
    # Merge any compatible neighbors, including across the computational seam.
    changed = True
    while changed and len(result) > 2:
        changed = False
        for i in range(len(result)):
            j = (i+1) % len(result)
            ids = np.r_[result[i][0], result[j][0]]
            fit = _best(q[np.r_[ids, (ids[-1]+1) % n]], config)
            if fit is not None:
                if j == 0:
                    result = [(ids, fit)]+result[1:i]
                else:
                    result[i:j+1] = [(ids, fit)]
                changed = True
                break
    return result, dict(candidates=count)


def predict(contours, *, units='px', config=None, compounds=True):
    """Same raw-edge handoff schema as the learned backend; no checkpoint required."""
    if units not in ('mm', 'px'):
        raise ValueError('units must be mm or px')
    cfg = config or GeometricConfig(tolerance=.75 if units == 'px' else .01)
    # Only observed geometry and explicit topology are allowed across this boundary.
    offsets = np.asarray(contours['ring_offsets'])
    if offsets.ndim != 1 or offsets.dtype.kind not in 'iu':
        raise ValueError('ring_offsets must be a one-dimensional integer array')
    raw, offsets, holes = ring_geometry(dict(actual_xy_mm=contours['actual_xy_mm'], ring_offsets=offsets))
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        if (np.linalg.norm(np.roll(raw[lo:hi], -1, axis=0)-raw[lo:hi], axis=1) <= 1e-12).any():
            raise ValueError('Duplicate consecutive contour vertices')
    features, relations, diagnostics = [], [], []
    owner = np.full(len(raw), -1, int)
    classes = np.zeros(len(raw), np.uint8)
    for ri, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        parts, diagnostic = segment_ring(raw[lo:hi], cfg)
        diagnostics.append(dict(ring_index=ri, **diagnostic))
        for local, fit in parts:
            ids = local+lo
            cls = BASE_NAMES.index(fit['kind'])
            span = dict(ring_index=ri, start_edge=int(local[0]), edge_count=len(ids),
                        wraps=bool(local[-1] < local[0]), closed_ring=bool(len(ids) == hi-lo))
            feature = dict(id=f'base_{len(features)+1:05d}', semantic_id=cls, semantic=fit['kind'],
                           label_level='base', source='geometry', status='accepted',
                           measurement_route=fit['kind'], raw_edge_indices=ids.tolist(),
                           contour_spans=[span], defect_edge_indices=[], boundary_search_regions=[],
                           parent='part', units=units, fit_geometry='original_observations',
                           grouping='geometric_contiguous_span', fit_evidence=fit,
                           recognition_tolerance=cfg.tolerance)
            owner[ids] = len(features)
            classes[ids] = cls
            features.append(feature)
        for at in np.flatnonzero(owner[lo:hi] != np.roll(owner[lo:hi], 1)):
            left, right = owner[lo+(at-1) % (hi-lo)], owner[lo+at]
            relations.append(dict(type='contour_adjacency', feature_a=features[left]['id'],
                                  feature_b=features[right]['id'], ring_index=ri, at_edge=int(at)))
    compound_features = []
    if compounds:
        from .geometric_compounds import recognize_compounds
        compound_features = recognize_compounds(raw, offsets, holes, features, cfg, units)
        for f in compound_features:
            for member in f['base_instance_ids']:
                relations.append(dict(type='compound_membership', feature_a=member, feature_b=f['id']))
    result = dict(schema_version=2, label_version='primitives-compounds-v2', units=units,
                  recognition_backend='geometric-v1', configuration=asdict(cfg),
                  coordinate_hash=hashlib.sha256(raw.tobytes()+offsets.tobytes()).hexdigest(),
                  part=dict(id='part', ring_count=len(holes), ring_is_hole=holes.tolist(),
                            bounds=[raw.min(0).tolist(), raw.max(0).tolist()]),
                  features=features, compound_features=compound_features, relations=relations,
                  base_names=BASE_NAMES, compound_names=COMPOUND_NAMES,
                  compound_grouping_trained=False, unselected_edge_fraction=0.,
                  measurement_policy='Fit original observations; recognition approximations are diagnostic only',
                  diagnostics=diagnostics)
    return result, dict(classes=classes, owner=owner)


def predict_file(contours, output, units='mm', tolerance=None, compounds=True):
    with np.load(contours, allow_pickle=False) as a:
        raw = dict(actual_xy_mm=a['actual_xy_mm'].copy(), ring_offsets=a['ring_offsets'].copy())
    cfg = GeometricConfig(tolerance=tolerance) if tolerance is not None else None
    result, _ = predict(raw, units=units, config=cfg, compounds=compounds)
    result['source_contours'] = str(Path(contours).resolve())
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"Wrote {len(result['features'])} geometric primitives to {target}")
    return result
