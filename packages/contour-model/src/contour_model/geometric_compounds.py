"""Conservative, observable compound rules; no nominal family or neural scores.

Only verified stadium holes, radial repeats, and opposing half-pitch-offset
thread-like runs are promoted. Ambiguous repeats remain individually measurable
primitives. These rules do not establish a manufacturing standard or 3D helix.
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from .geometric import COMPOUND_NAMES


def _runs(mask):
    if mask.all():
        return [np.arange(len(mask))]
    result = []
    for at in np.flatnonzero(mask & ~np.roll(mask, 1)):
        ids = []
        while mask[at]:
            ids.append(int(at))
            at = (at+1) % len(mask)
        result.append(np.asarray(ids))
    return result


def _slot(members, cfg):
    lines = [f['fit_evidence'] for f in members if f['semantic'] == 'line']
    arcs = [f['fit_evidence'] for f in members if f['semantic'] == 'arc']
    if len(members) != 4 or len(lines) != 2 or len(arcs) != 2:
        return None
    a, b = arcs
    radius = (a['radius']+b['radius'])/2
    tolerance = max(3*cfg.tolerance, .03*radius)
    if abs(a['radius']-b['radius']) > tolerance or any(abs(abs(f['sweep'])-np.pi) > .2 for f in arcs):
        return None
    centers = np.asarray([a['center'], b['center']])
    axis = centers[1]-centers[0]
    length = np.linalg.norm(axis)
    if length <= 4*cfg.tolerance:
        return None
    axis /= length
    normal = np.array([-axis[1], axis[0]])
    for line in lines:
        if abs(np.dot(line['direction'], axis)) < .995:
            return None
        if np.max(np.abs(np.abs((centers-line['origin'])@normal)-radius)) > tolerance:
            return None
    # Both parallel lines must be on opposite sides of the centerline.
    if np.prod([(np.asarray(f['origin'])-centers[0])@normal for f in lines]) >= 0:
        return None
    return dict(rule='stadium_two_parallel_lines_two_equal_semicircles', radius=radius)


def _radial(q, cfg):
    # A robust center estimate for a full radial outline, followed by angular
    # periodicity checks. Eccentric/non-star-shaped outlines are not inferred.
    center = (q.min(0)+q.max(0))/2
    v = q-center
    angle = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    if angle[-1] < angle[0]:
        angle, q, v = -angle, q, v
    if np.min(np.diff(angle)) < -.01 or angle[-1]-angle[0] < 6:
        return None
    radius = np.linalg.norm(v, axis=1)
    grid = np.linspace(angle[0], angle[0]+2*np.pi, 2048, endpoint=False)
    y = np.interp(grid, np.r_[angle, angle[0]+2*np.pi], np.r_[radius, radius[0]])
    y = gaussian_filter1d(y, 1, mode='wrap')
    depth = float(np.ptp(y))
    if depth < 6*cfg.tolerance:
        return None
    peaks, _ = find_peaks(np.tile(y, 3), prominence=max(3*cfg.tolerance, depth*.35))
    peaks = peaks[(peaks >= len(y)) & (peaks < 2*len(y))]-len(y)
    if len(peaks) < 6:
        return None
    pitches = np.diff(np.r_[peaks, peaks[0]+len(y)])
    if np.std(pitches)/np.mean(pitches) > .12:
        return None
    period = int(round(np.median(pitches)))
    correlation = float(np.corrcoef(y, np.roll(y, period))[0, 1])
    if correlation < .9:
        return None
    return dict(rule='closed_radial_periodicity', count=len(peaks), correlation=correlation,
                scope='Observed radial repetition; gear standard and tooth form are not inferred')


def _side_candidates(q, axis, cfg):
    """Find regular crest trains on monotone opposing silhouette halves."""
    normal = np.array([-axis[1], axis[0]])
    x = q@axis
    left, right = int(x.argmin()), int(x.argmax())
    sides = []
    for start, stop in [(left, right), (right, left)]:
        ids = np.arange(start, stop if stop > start else stop+len(q)) % len(q)
        ids = np.r_[ids, stop]
        if x[ids[-1]] < x[ids[0]]:
            ids = ids[::-1]
        u, v = x[ids], q[ids]@normal
        if len(ids) < 12 or np.max(np.maximum.accumulate(u)-u) > 2*cfg.tolerance:
            return []
        # Interpolate only for candidate detection, never for measurement support.
        unique, first = np.unique(u, return_index=True)
        if len(unique) < 12:
            return []
        grid = np.linspace(unique[0], unique[-1], min(4096, max(512, len(ids))))
        sign = 1 if np.mean(v) > np.mean(q@normal) else -1
        heights = sign*np.interp(grid, unique, v[first])
        heights = gaussian_filter1d(heights, 1)
        amplitude = float(np.ptp(heights))
        peaks, props = find_peaks(heights, prominence=max(3*cfg.tolerance, amplitude*.08))
        candidates = []
        if len(peaks) >= 4:
            # Break at changes in pitch, allowing multiple separated thread zones.
            pitch = np.diff(grid[peaks])
            breaks = np.flatnonzero(np.maximum(pitch[1:]/pitch[:-1], pitch[:-1]/pitch[1:]) > 1.25)+1
            for group in np.split(np.arange(len(peaks)), breaks):
                if len(group) < 4:
                    continue
                indices = peaks[group]
                spacing = np.diff(grid[indices])
                period = float(np.median(spacing))
                if np.std(spacing)/period > .10:
                    continue
                depth = float(np.median(props['prominences'][group]))
                if depth < 4*cfg.tolerance or not .08 < depth/period < 1.5:
                    continue
                a = max(0, int(indices[0]-(indices[1]-indices[0])/2))
                b = min(len(grid)-1, int(indices[-1]+(indices[-1]-indices[-2])/2))
                chosen = ids[(u >= grid[a]) & (u <= grid[b])]
                if len(chosen) < 12:
                    continue
                slopes = np.diff(heights[a:b+1])/np.diff(grid[a:b+1])
                active = np.abs(slopes)
                active = active[active > .25]
                if not len(active):
                    continue
                flank_angle = float(np.degrees(np.arctan(np.median(active))))
                # Shallow sinusoidal/scalloped bands do not establish a thread.
                if flank_angle < 35:
                    continue
                candidates.append(dict(ids=chosen, pitch=period, crests=grid[indices],
                                       interval=[float(grid[a]), float(grid[b])], depth=depth,
                                       side=float(np.mean(q[chosen]@normal)), angle=flank_angle))
        sides.append(candidates)
    result = []
    for a in sides[0]:
        for b in sides[1]:
            period = (a['pitch']+b['pitch'])/2
            if abs(a['pitch']-b['pitch']) > .1*period:
                continue
            overlap = min(a['interval'][1], b['interval'][1])-max(a['interval'][0], b['interval'][0])
            extent = max(np.ptp(a['interval']), np.ptp(b['interval']))
            if overlap < .7*extent or abs(a['side']-b['side']) < 2*max(a['depth'], b['depth']):
                continue
            phases = ((a['crests'][:, None]-b['crests'][None, :])/period) % 1
            phase_error = float(np.median(np.min(np.abs(phases-.5), axis=1)))
            if phase_error > .15:
                continue
            result.append((np.unique(np.r_[a['ids'], b['ids']]),
                           2 if (a['angle']+b['angle'])/2 < 67.5 else 3,
                           dict(rule='opposing_periodic_flanks_half_pitch_phase', pitch=period,
                                depth=max(a['depth'], b['depth']),
                                phase_error_cycles=phase_error, flank_angle_degrees=(a['angle']+b['angle'])/2,
                                scope='Single-start-like 2D silhouette; form category is heuristic, not a standard match')))
    return result


def recognize_compounds(raw, offsets, holes, features, cfg, units):
    compounds, used = [], set()

    def add(members, cls, evidence):
        if any(f['id'] in used for f in members):
            return
        ids = np.unique(np.concatenate([f['raw_edge_indices'] for f in members]))
        spans = []
        for ri, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
            for run in _runs(np.isin(np.arange(lo, hi), ids)):
                spans.append(dict(ring_index=ri, start_edge=int(run[0]), edge_count=len(run),
                                  wraps=bool(run[-1] < run[0]), closed_ring=bool(len(run) == hi-lo)))
        compounds.append(dict(id=f'compound_{len(compounds)+1:04d}', semantic_id=cls,
                              semantic=COMPOUND_NAMES[cls], label_level='compound', source='geometry',
                              base_instance_ids=[f['id'] for f in members], status='accepted',
                              raw_edge_indices=ids.tolist(), contour_spans=spans,
                              parent='part', units=units, fit_geometry='original_observations',
                              measurement_route={1:'slot', 2:'thread_pitch', 3:'thread_pitch', 4:'repeated_profile'}[cls],
                              grouping='verified_geometric_relationships', rule_evidence=evidence,
                              classification_policy='conservative_geometry_rules'))
        used.update(f['id'] for f in members)

    for ri, (lo, hi) in enumerate(zip(offsets[:-1], offsets[1:])):
        members = [f for f in features if f['contour_spans'][0]['ring_index'] == ri]
        if len(members) < 2:
            continue
        evidence = _slot(members, cfg) if holes[ri] else None
        if evidence:
            add(members, 1, evidence)
            continue
        q = raw[lo:hi]
        evidence = _radial(q, cfg)
        if evidence:
            add(members, 4, evidence)
            continue
        if holes[ri]:
            continue
        _, vectors = np.linalg.eigh(np.cov(q.T))
        # Try both principal axes: a short bolt need not be longer than its diameter.
        for axis in (vectors[:, -1], vectors[:, 0]):
            for local, cls, evidence in _side_candidates(q, axis, cfg):
                chosen = [f for f in members if np.mean(np.isin(f['raw_edge_indices'], lo+local)) > .5
                          and not (f['semantic'] == 'line'
                                   and abs(np.dot(f['fit_evidence']['direction'], axis)) < .2
                                   and f['fit_evidence']['length'] > 1.5*evidence['depth'])]
                if len(chosen) < 8:
                    continue
                # Whole base membership must still form exactly two open sides.
                ids = np.unique(np.concatenate([f['raw_edge_indices'] for f in chosen]))-lo
                runs = _runs(np.isin(np.arange(hi-lo), ids))
                if len(runs) == 2:
                    add(chosen, cls, evidence)
            if any(f['id'] in used for f in members):
                break
    return compounds
