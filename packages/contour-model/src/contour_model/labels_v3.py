"""Observable reference labels: construction identity limited by image resolution.

The construction labels (``inspection_shapes.primitive_labels``) describe the
nominal drawing. Some of that is invisible in a silhouette image: a 1° tooth
crown arc on a 20 mm gear bulges 0.001 mm (a few hundredths of a pixel), and a
0.05 mm edge break spans under a pixel. Training on those targets teaches
arbitrary line/arc swaps. ``observable_labels`` applies, at a given pixel scale:

* an arc whose sagitta is below ``SAGITTA_PX`` becomes a line;
* an instance shorter than ``MIN_FEATURE_PX``, or shorter than ``CORNER_SPAN_PX``
  while deviating less than ``CORNER_DEPTH_PX`` from the sharp corner its
  neighbours would form, dissolves into its neighbours (first half to the previous
  instance, second half to the next): a tiny edge break is part of the corner;
* adjacent lines whose junction deviates by less than ``SAGITTA_PX`` merge.

Reference-free freeform spans use the existing intent policy. Features marked
``unsupported`` (for example ellipses) and other unobservable regions receive a
zero loss weight in ``edge_label_weight`` instead of an arbitrary label.

The clean-contour corpus uses ``REFERENCE_EXTENT_PX`` across the part; rendered
observations are relabelled with their actual pixels per millimetre.
"""
import numpy as np
from inspection_shapes.primitive_labels import labels as construction_labels

POLICY = 'observable-v1'
REFERENCE_EXTENT_PX = 1000.
# Calibrated on paired PAE/edge-model extractions of the v3 pilot
# (scripts/calibrate_observability.py): arc curvature is invisible below ~0.35 px
# sagitta and 50% visible near 0.5 px; extraction already rounds sharp corners by
# 0.3-0.8 px (95th percentile), so shallower corner details are indistinguishable.
SAGITTA_PX = .5
MIN_FEATURE_PX = 2.5
CORNER_SPAN_PX = 8.
CORNER_DEPTH_PX = .5
FIELDS = ('edge_base_class', 'edge_base_instance', 'edge_compound_class', 'edge_compound_instance')


def reference_px_per_mm(contours):
    return REFERENCE_EXTENT_PX/float(np.ptp(np.asarray(contours['nominal_xy_mm']), axis=0).max())


def _ring_runs(inst):
    """Cyclic runs of equal instance IDs: list of index arrays (ring-local)."""
    n = len(inst)
    starts = np.flatnonzero(inst != np.roll(inst, 1))
    if not len(starts):
        return [np.arange(n)]
    runs = []
    for k, s in enumerate(starts):
        e = starts[(k+1) % len(starts)]
        if e <= s:
            e += n
        runs.append(np.arange(s, e) % n)
    return runs


def _sagitta(points):
    chord = points[-1]-points[0]
    length = np.linalg.norm(chord)
    if length < 1e-12:
        return float(np.linalg.norm(points-points[0], axis=1).max())
    w = points-points[0]
    return float(np.abs(w[:, 0]*chord[1]-w[:, 1]*chord[0]).max()/length)


def _shallow_arcs(q, cls, inst, ppm, sagitta_px):
    n = len(q)
    for run in _ring_runs(inst):
        if cls[run[0]] == 2 and _sagitta(q[np.r_[run, (run[-1]+1) % n]])*ppm < sagitta_px:
            cls[run] = 1


def _merge_lines(q, cls, inst, ppm, sagitta_px):
    """Merge adjacent line instances whose junction deviation is unresolvable."""
    n = len(q)
    runs = _ring_runs(inst)
    for k in range(len(runs) if len(runs) > 1 else 0):
        a, b = runs[k-1], runs[k]
        if cls[a[0]] != 1 or cls[b[0]] != 1 or inst[a[0]] == inst[b[0]]:
            continue
        da = q[(a[-1]+1) % n]-q[a[0]]; db = q[(b[-1]+1) % n]-q[b[0]]
        la, lb = np.linalg.norm(da), np.linalg.norm(db)
        if la < 1e-12 or lb < 1e-12:
            continue
        angle = abs(np.arctan2(da[0]*db[1]-da[1]*db[0], da@db))
        if angle*min(la, lb)*ppm < sagitta_px:
            inst[inst == inst[b[0]]] = inst[a[0]]


def _corner_depth(q, run, prev, nxt):
    """Largest distance of a span from the sharp corner its neighbours would form."""
    n = len(q)
    s, e = run[0], (run[-1]+1) % n
    k1, k2 = min(3, len(prev)), min(3, len(nxt))
    d1 = q[s]-q[(s-k1) % n]; d2 = q[(e+k2) % n]-q[e]
    d1 = d1/max(np.linalg.norm(d1), 1e-15); d2 = d2/max(np.linalg.norm(d2), 1e-15)
    pts = q[np.r_[run, e]]
    den = d1[0]*d2[1]-d1[1]*d2[0]
    if abs(den) < 1e-6:
        # Parallel neighbours: a step; its height is the offset between the lines.
        w = q[e]-q[s]
        return float(abs(w[0]*d1[1]-w[1]*d1[0]))
    t = ((q[e]-q[s])[0]*d2[1]-(q[e]-q[s])[1]*d2[0])/den
    apex = q[s]+t*d1

    def to_segment(a, b):
        ab = b-a; L = ab@ab
        u = np.clip(((pts-a)@ab)/L, 0, 1) if L > 1e-24 else np.zeros(len(pts))
        return np.linalg.norm(pts-(a+u[:, None]*ab), axis=1)
    return float(np.minimum(to_segment(q[s], apex), to_segment(apex, q[e])).max())


def _dissolve_short(q, cls, inst, length, ppm, min_px, span_px=CORNER_SPAN_PX, depth_px=CORNER_DEPTH_PX):
    """Give each unresolvable short instance to its neighbours, shortest first."""
    while True:
        runs = _ring_runs(inst)
        if len(runs) < 3:
            return
        sizes = np.array([length[r].sum() for r in runs])*ppm
        candidates = []
        for k in np.argsort(sizes):
            if sizes[k] >= span_px:
                break
            if sizes[k] < min_px or _corner_depth(q, runs[k], runs[k-1], runs[(k+1) % len(runs)])*ppm < depth_px:
                candidates.append(k)
                break
        if not candidates:
            return
        k = candidates[0]
        run, prev, nxt = runs[k], runs[k-1], runs[(k+1) % len(runs)]
        half = (len(run)+1)//2
        for part, donor in ((run[:half], prev), (run[half:], nxt)):
            if len(part):
                cls[part] = cls[donor[0]]; inst[part] = inst[donor[0]]


def _observe_ring(q, cls, inst, length, ppm, sagitta_px, min_px):
    """Apply the observability rules to one ring in place. q: nominal vertices."""
    _shallow_arcs(q, cls, inst, ppm, sagitta_px)
    _merge_lines(q, cls, inst, ppm, sagitta_px)
    _dissolve_short(q, cls, inst, length, ppm, min_px)
    _merge_lines(q, cls, inst, ppm, sagitta_px)


def observable_labels(metadata, contours, px_per_mm=None, sagitta_px=SAGITTA_PX, min_feature_px=MIN_FEATURE_PX):
    """Return (annotations, compound groups, edge_label_weight) at ``px_per_mm``."""
    annotations, groups = construction_labels(metadata, contours)
    primitives = metadata['primitives']
    if any(p['kind'] == 'freeform' for p in primitives):
        from .recache_intent import corrected_annotations
        annotations, groups = corrected_annotations(dict(metadata, compound_instances=groups),
                                                    dict(contours, **annotations))
    annotations = {k: np.asarray(annotations[k]).copy() for k in FIELDS}
    ppm = float(px_per_mm or reference_px_per_mm(contours))
    q_all = np.asarray(contours['nominal_xy_mm'], float)
    offsets = np.asarray(contours['ring_offsets'])
    weight = np.ones(len(q_all), np.float32)
    unsupported = [f['id'] for f in metadata['features'] if f.get('nominal_parameters', {}).get('unsupported')]
    if unsupported:
        weight[np.isin(contours['edge_feature_id'], unsupported)] = 0.
    cls, inst = annotations['edge_base_class'], annotations['edge_base_instance']
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        q = q_all[lo:hi]
        length = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
        c, i = cls[lo:hi].copy(), inst[lo:hi].copy()
        _observe_ring(q, c, i, length, ppm, sagitta_px, min_feature_px)
        cls[lo:hi], inst[lo:hi] = c, i
    # Compound membership remains whole base instances: majority by length.
    seg = np.zeros(len(q_all))
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        q = q_all[lo:hi]; seg[lo:hi] = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
    cid, ccls = annotations['edge_compound_instance'], annotations['edge_compound_class']
    new_cid, new_ccls = np.zeros_like(cid), np.zeros_like(ccls)
    for ident in np.unique(inst):
        mask = inst == ident
        votes = np.bincount(cid[mask], weights=seg[mask])
        winner = int(np.argmax(votes))
        new_cid[mask] = winner
        new_ccls[mask] = ccls[mask][cid[mask] == winner][0] if winner else 0
    annotations['edge_compound_instance'], annotations['edge_compound_class'] = new_cid, new_ccls
    kept = []
    for g in groups:
        mask = new_cid == g['id']
        if not mask.any():
            continue
        members = np.unique(inst[mask]).tolist()
        if g['class_id'] >= 2 and len(members) < 4:
            # A repeated profile whose teeth are no longer resolvable is ambiguous.
            weight[mask] = 0.
        kept.append(dict(g, base_instance_ids=members))
    return annotations, kept, weight


def weighted_targets(metadata, contours, prepared, annotations, groups, weight):
    """Training targets whose token weights exclude zero-weight (unobservable) edges."""
    from .corpus_primitives import targets
    t, _, _ = targets(metadata, contours, prepared, annotation_override=(annotations, groups))
    keep = np.asarray(prepared.overlap@np.asarray(weight, np.float32)).ravel()
    token_weights = (prepared.weights*np.clip(keep, 0, 1)).astype(np.float32)
    if token_weights.sum() <= 0:
        raise ValueError('No supervised contour length remains')
    return t, token_weights
