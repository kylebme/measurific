"""Interactive metrology contract shared by desktop and other front ends."""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from .engine import evaluate
from .contours import select


MEASUREMENT_SPECS = {
    'distance_construction': {'label': 'Distance', 'count': 2, 'accepts': ['any', 'any']},
    'thread': {'label': 'Thread measurements', 'count': 1, 'accepts': ['thread']},
    'angle': {'label': 'Included angle', 'count': 2, 'accepts': ['line', 'line']},
    'datum': {'label': 'Define datum', 'count': 1, 'accepts': ['datum']},
    'concentricity': {'label': 'Concentricity', 'count': 2, 'accepts': ['center', 'center']},
    'perpendicularity': {'label': 'Perpendicularity', 'count': 2, 'accepts': ['line', 'line']},
    'parallelism': {'label': 'Parallelism', 'count': 2, 'accepts': ['line', 'line']},
    'circularity_ls': {'label': 'Circularity', 'count': 1, 'accepts': ['circle']},
    'straightness': {'label': 'Straightness', 'count': 1, 'accepts': ['line']},
    'symmetry': {'label': 'Symmetry', 'count': 3, 'accepts': ['center', 'center', 'line']},
    'least_squares_diameter': {'label': 'Diameter', 'count': 1, 'accepts': ['circle']},
    'radius': {'label': 'Radius', 'count': 1, 'accepts': ['round']},
    'length': {'label': 'Length', 'count': 1, 'accepts': ['line']},
    'angular_span': {'label': 'Arc angle', 'count': 1, 'accepts': ['arc']},
    'slot': {'label': 'Slot measurements', 'count': 1, 'accepts': ['slot']},
    'tooth_profile': {'label': 'Tooth profile measurements', 'count': 1, 'accepts': ['tooth_profile']},
}

CENTER_TYPES = {'point', 'circle', 'arc', 'ellipse', 'elliptical_arc', 'slot', 'rounded_rectangle', 'polygon'}


def _endpoint_features(report, contours):
    """Open fitted segments with their retained run endpoints.

    An endpoint is trimmed when it borders the segment's own corner-excluded
    edges. The observed corner there is unavailable, so its vertex stays at the
    retained endpoint rather than at a junction synthesized across the gap.
    """
    features = []
    spacings = []
    for feature in report['features']:
        if feature.get('status') != 'fitted' or feature.get('label_level', 'base') != 'base':
            continue
        if feature.get('type') not in {'line', 'arc', 'elliptical_arc'}:
            continue
        gaps = np.empty((0, 2))
        excluded = np.asarray(feature.get('excluded_edge_indices', []), dtype=np.int64)
        if len(excluded):
            gaps = np.array([end for run in select(contours, {'edge_indices': excluded})[0]
                             for end in (run[0], run[-1])])
        runs = feature.get('support', [])
        closed = feature.get('closed', [])
        for run, is_closed in zip(runs, closed):
            q = np.asarray(run, float)
            if is_closed or len(q) < 2:
                continue
            ends = [(end, bool(len(gaps)) and float(np.min(np.linalg.norm(gaps-end, axis=1))) <= 1e-9)
                    for end in (q[0], q[-1])]
            features.append((feature, ends, q))
            spacing = np.linalg.norm(np.diff(q, axis=0), axis=1)
            spacings.extend(spacing[spacing > 1e-9].tolist())
    return features, spacings


def _local_line(run, at_end, length):
    """Total-least-squares line through a run's support within ``length`` of one end."""
    q = run[::-1] if at_end else run
    arclength = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    q = q[arclength <= max(length, arclength[min(2, len(q)-1)])]
    center = q.mean(0)
    return center, np.linalg.svd(q-center)[2][0]


def find_junction_vertices(report, contours):
    """Find selectable vertices where fitted open segments meet.

    Untrimmed endpoints that meet form one junction vertex. Where corner
    exclusion trimmed a segment, each segment keeps its own vertex at its
    retained endpoint, so two vertices may sit close together at one corner.
    """
    segments, spacings = _endpoint_features(report, contours)
    threshold = float(np.clip(3*np.median(spacings) if spacings else 2., 1.5, 8.))
    # A trimmed corner leaves a gap of up to the exclusion radius on each side.
    gap = threshold+2*float(report.get('corner_exclusion', {}).get('radius_px', 0.))
    candidates, junctioned, separate = [], set(), set()
    for i, (a, a_ends, a_run) in enumerate(segments):
        separate.update((i, k) for k, (_, trimmed) in enumerate(a_ends) if trimmed)
        for j in range(i+1, len(segments)):
            b, b_ends, b_run = segments[j]
            (ka, (pa, a_trimmed)), (kb, (pb, b_trimmed)) = min(
                ((x, y) for x in enumerate(a_ends) for y in enumerate(b_ends)),
                key=lambda pair: np.linalg.norm(pair[0][1][0]-pair[1][1][0]))
            distance = np.linalg.norm(pa-pb)
            if a_trimmed or b_trimmed:
                # An untrimmed end facing a trimmed neighbour also keeps its own vertex.
                if distance <= gap:
                    separate.update({(i, ka), (j, kb)})
                continue
            if distance <= threshold:
                junctioned.update({(i, ka), (j, kb)})
                point = (pa+pb)/2
                if a['type'] == b['type'] == 'line':
                    # Long observed lines are rarely perfectly straight. Only the
                    # vertex uses lines fitted near the junction, which keeps the
                    # sharp corner on the local contour; the features keep their
                    # whole-segment fits.
                    ca, ua = _local_line(a_run, ka == 1, 8*threshold)
                    cb, ub = _local_line(b_run, kb == 1, 8*threshold)
                    matrix = np.c_[ua, -ub]
                    if abs(np.linalg.det(matrix)) > 1e-8:
                        sharp = ca + np.linalg.solve(matrix, cb-ca)[0]*ua
                        if max(np.linalg.norm(sharp-pa), np.linalg.norm(sharp-pb)) <= 2*threshold:
                            point = sharp
                candidates.append({'point': point, 'members': {a['id'], b['id']}})
    clusters = []
    for candidate in candidates:
        found = next((cluster for cluster in clusters
                      if np.linalg.norm(cluster['point']-candidate['point']) <= threshold), None)
        if found:
            found['points'].append(candidate['point'])
            found['members'].update(candidate['members'])
            found['point'] = np.mean(found['points'], axis=0)
        else:
            clusters.append({'point': candidate['point'], 'points': [candidate['point']],
                             'members': set(candidate['members'])})
    for i, k in sorted(separate-junctioned):
        feature, ends, _ = segments[i]
        clusters.append({'point': ends[k][0], 'members': {feature['id']}})
    return [{
        'id': f'vertex_{index:04d}', 'type': 'point', 'semantic': 'vertex',
        'source': 'user', 'label_level': 'base', 'point': cluster['point'].tolist(),
        'adjacent_feature_ids': sorted(cluster['members']), 'units': 'px',
    } for index, cluster in enumerate(clusters, 1)]


def _nice_factor(value):
    if value <= 2:
        return 2.
    magnitude = 10**math.floor(math.log10(value))
    for multiplier in (1, 2, 5, 10):
        candidate = multiplier*magnitude
        if candidate >= value:
            return float(min(candidate, 500.))
    return 500.


def prepare_interactive_report(contours, prediction):
    """Evaluate a prediction and add UI-neutral selection/display metadata."""
    request = deepcopy(prediction)
    request.setdefault('corner_exclusion_px', 1.0)
    # Junction vertices are derived from the current retained support. Rebuild
    # them when measurement settings or user geometry change.
    request['features'] = [
        feature for feature in request.get('features', [])
        if not (feature.get('type') == 'point' and feature.get('semantic') == 'vertex')
    ]
    initial = evaluate(contours, request)
    request.setdefault('features', []).extend(find_junction_vertices(initial, contours))
    report = evaluate(contours, request)
    base = [f for f in report['features'] if f.get('label_level', 'base') == 'base']
    maximum = max((f.get('deformation_max', 0.) for f in base), default=0.)
    extent = np.ptp(contours.xy, axis=0)
    target = max(6., .015*float(np.linalg.norm(extent)))
    max_factor = 1. if maximum <= 1e-12 else _nice_factor(1+target/maximum)
    report['interaction'] = {
        'measurement_specs': deepcopy(MEASUREMENT_SPECS),
        'deformation': {'max_factor': max_factor, 'maximum_observed_deviation': maximum},
    }
    return report, request


def _nearest_contour_point(contours, point, ring_index=None):
    """Project an image-space point onto the nearest observed contour edge."""
    p = np.asarray(point, float)
    if p.shape != (2,) or not np.isfinite(p).all():
        raise ValueError('Selection point must be one finite [x, y] coordinate')
    best = None
    rings = range(len(contours.closed)) if ring_index is None else [ring_index]
    for ring in rings:
        q = contours.path(ring)
        count = len(q) if contours.closed[ring] else len(q)-1
        starts = q[:count]
        ends = q[(np.arange(count)+1) % len(q)]
        vectors = ends-starts
        squared = np.einsum('ij,ij->i', vectors, vectors)
        fractions = np.clip(np.einsum('ij,ij->i', p-starts, vectors)/squared, 0., 1.)
        projected = starts+fractions[:, None]*vectors
        distances = np.einsum('ij,ij->i', projected-p, projected-p)
        edge = int(np.argmin(distances))
        candidate = (float(distances[edge]), int(ring), edge, float(fractions[edge]), projected[edge])
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError('No contour is available for selection')
    distance, ring, edge, fraction, snapped = best
    return {'ring_index': ring, 'edge': edge, 't': fraction,
            'point': snapped.tolist(), 'distance': math.sqrt(distance)}


def _feature_edge_ids(contours, feature):
    try:
        if 'selection' in feature:
            return select(contours, feature['selection'])[2]
        if 'raw_edge_indices' in feature:
            return np.asarray(feature['raw_edge_indices'], dtype=np.int64)
        if 'contour_spans' in feature:
            return select(contours, {'spans': feature['contour_spans']})[2]
    except (ValueError, TypeError, KeyError):
        pass
    return np.empty(0, dtype=np.int64)


def _runs_as_spans(contours, ids):
    mask = np.zeros(len(contours.xy), bool)
    mask[np.asarray(ids, dtype=np.int64)] = True
    spans = []
    for ring, (lo, hi) in enumerate(zip(contours.offsets[:-1], contours.offsets[1:])):
        selected = mask[lo:hi]
        if not contours.closed[ring]:
            selected[-1] = False
        if not selected.any():
            continue
        if selected.all():
            spans.append((np.arange(lo, hi), {'ring_index': ring, 'start_edge': 0,
                          'edge_count': int(hi-lo), 'closed_ring': True}))
            continue
        starts = np.flatnonzero(selected & ~np.roll(selected, 1))
        for start in starts:
            length = 1
            while length < len(selected) and selected[(start+length) % len(selected)]:
                length += 1
            local = (start+np.arange(length)) % len(selected)
            spans.append((lo+local, {'ring_index': ring, 'start_edge': int(start),
                          'edge_count': int(length), 'closed_ring': False,
                          'wraps': bool(local[-1] < local[0])}))
    return spans


def _replace_contour_support(contours, request, kind, selection, selected_ids):
    """Replace prior base classifications on selected edges with user geometry."""
    rewritten = deepcopy(request)
    # Junction vertices are derived report metadata and must be regenerated.
    bases = [f for f in rewritten.get('features', [])
             if not (f.get('type') == 'point' and f.get('semantic') == 'vertex')]
    touched, retained = set(), []
    reserved_ids = {f['id'] for f in bases}
    emitted_ids = set()
    selected_set = set(int(v) for v in selected_ids)
    for feature in bases:
        ids = _feature_edge_ids(contours, feature)
        overlap = selected_set.intersection(int(v) for v in ids)
        if not overlap:
            retained.append(feature)
            emitted_ids.add(feature['id'])
            continue
        touched.add(feature['id'])
        remaining = [int(v) for v in ids if int(v) not in selected_set]
        for fragment_index, (fragment_ids, span) in enumerate(_runs_as_spans(contours, remaining)):
            fragment = deepcopy(feature)
            if fragment_index == 0:
                fragment_id = feature['id']
            else:
                serial = 2
                fragment_id = f"{feature['id']}_fragment_{serial}"
                while fragment_id in reserved_ids or fragment_id in emitted_ids:
                    serial += 1
                    fragment_id = f"{feature['id']}_fragment_{serial}"
            fragment['id'] = fragment_id
            fragment.pop('selection', None)
            fragment['raw_edge_indices'] = fragment_ids.tolist()
            fragment['contour_spans'] = [span]
            retained.append(fragment)
            emitted_ids.add(fragment_id)
    prefix = f'user_{kind}_'
    serials = [int(f['id'][len(prefix):]) for f in retained
               if f.get('id', '').startswith(prefix) and f['id'][len(prefix):].isdigit()]
    ident = prefix+f'{max(serials, default=0)+1:04d}'
    retained.append({
        'id': ident, 'type': kind, 'semantic': kind, 'measurement_route': kind,
        'label_level': 'base', 'source': 'user', 'accept': True, 'status': 'accepted',
        'units': 'px', 'selection': selection,
    })
    rewritten['features'] = retained
    compounds = rewritten.get('compound_features', [])
    removed_compounds = {f['id'] for f in compounds if touched.intersection(f.get('base_instance_ids', []))}
    rewritten['compound_features'] = [f for f in compounds if f['id'] not in removed_compounds]
    removed = touched | removed_compounds
    rewritten['relations'] = [r for r in rewritten.get('relations', [])
                              if r.get('feature_a') not in removed and r.get('feature_b') not in removed]
    rewritten.pop('overrides', None)
    rewritten.pop('measurements', None)
    return rewritten, ident, removed


def _cursor_ring_positions(contours, ring_index, cursor_path):
    """Project cursor samples to arclength positions on one ordered ring."""
    trail = np.asarray(cursor_path, float)
    if trail.ndim != 2 or trail.shape[1:] != (2,) or not np.isfinite(trail).all():
        raise ValueError('Cursor path must contain finite [x, y] coordinates')
    q = contours.path(ring_index)
    starts = q
    vectors = np.roll(q, -1, axis=0)-q
    squared = np.einsum('ij,ij->i', vectors, vectors)
    lengths = np.sqrt(squared)
    arc = np.r_[0., np.cumsum(lengths)]
    positions, distances = [], []
    for chunk_start in range(0, len(trail), 64):
        chunk = trail[chunk_start:chunk_start+64]
        relative = chunk[:, None, :]-starts[None, :, :]
        fractions = np.clip(np.einsum('csi,si->cs', relative, vectors)/squared, 0., 1.)
        projected = starts[None, :, :]+fractions[:, :, None]*vectors[None, :, :]
        distance = np.einsum('csi,csi->cs', projected-chunk[:, None, :], projected-chunk[:, None, :])
        edges = np.argmin(distance, axis=1)
        rows = np.arange(len(chunk))
        positions.extend((arc[edges]+fractions[rows, edges]*lengths[edges]).tolist())
        distances.extend(distance[rows, edges].tolist())
    return np.asarray(positions), np.asarray(distances), arc[-1], arc, lengths


def _cursor_route_score(contours, ring_index, cursor_path, start, end, direction):
    """Score how well cursor progress follows one directed interval of a ring."""
    if not cursor_path:
        return 0., 0.
    positions, distances, perimeter, arc, lengths = _cursor_ring_positions(
        contours, ring_index, cursor_path)
    start_position = arc[start['edge']]+start['t']*lengths[start['edge']]
    end_position = arc[end['edge']]+end['t']*lengths[end['edge']]
    coordinates = (direction*(positions-start_position)) % perimeter
    extent = (direction*(end_position-start_position)) % perimeter
    # Samples projected outside this route are separated from it by one of its
    # endpoints along the ring. Samples far from the contour receive less say.
    outside = coordinates > extent
    route_distance = np.zeros_like(coordinates)
    route_distance[outside] = np.minimum(coordinates[outside]-extent,
                                         perimeter-coordinates[outside])
    spacing = float(np.median(lengths))
    weights = 1/(1+distances/max(4., 4*spacing)**2)
    outside_error = float(np.average(route_distance**2, weights=weights))
    # Penalize a route whose projected cursor progress repeatedly runs backward.
    inside_coordinates = coordinates[~outside]
    backward_error = 0.
    if len(inside_coordinates) > 1:
        backward = np.maximum(-np.diff(inside_coordinates), 0.)
        backward_error = float(np.mean(backward**2))
    return outside_error+.1*backward_error, float(extent)


def _contour_interval(contours, first, second, cursor_path=None):
    """Directed interval between two snapped points on one ring.

    On a closed ring the cursor trail between the clicks chooses the way around.
    """
    interval = {
        'ring_index': first['ring_index'],
        'start': {'edge': first['edge'], 't': first['t']},
        'end': {'edge': second['edge'], 't': second['t']},
        'direction': 1,
    }
    if contours.closed[first['ring_index']] and not cursor_path:
        # Without a trail, the shorter way around is the evident intent.
        q = contours.path(first['ring_index'])
        lengths = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
        arc = np.r_[0., np.cumsum(lengths)]
        start = arc[first['edge']]+first['t']*lengths[first['edge']]
        end = arc[second['edge']]+second['t']*lengths[second['edge']]
        if (end-start) % arc[-1] > (start-end) % arc[-1]:
            interval['direction'] = -1
    elif contours.closed[first['ring_index']]:
        trail = cursor_path
        forward_score, forward_length = _cursor_route_score(
            contours, first['ring_index'], trail, interval['start'], interval['end'], 1)
        reverse_score, reverse_length = _cursor_route_score(
            contours, first['ring_index'], trail, interval['start'], interval['end'], -1)
        # Ordered cursor progress decides; route length is only a tie break.
        interval['direction'] = -1 if (reverse_score, reverse_length) < (forward_score, forward_length) else 1
    return interval


def redefine_contour_geometry(contours, session_request, kind, points, cursor_path=None):
    """Snap user selections and replace contour classifications with Line/Arc/Circle.

    One point previews a snap for Line and Arc. A complete selection returns a
    regenerated request/report; callers should commit it atomically.
    """
    if kind not in {'line', 'arc', 'circle'}:
        raise ValueError('Geometry override type must be line, arc, or circle')
    expected = 1 if kind == 'circle' else 2
    if not isinstance(points, list) or not 1 <= len(points) <= expected:
        raise ValueError(f'{kind.title()} redefinition requires {expected} selection(s)')
    first = _nearest_contour_point(contours, points[0])
    snapped = [first]
    if len(points) < expected:
        return {'complete': False, 'snapped_points': [first['point']],
                'ring_index': first['ring_index']}
    if kind == 'circle':
        ring = first['ring_index']
        if not contours.closed[ring]:
            raise ValueError('A circle can only redefine a closed contour ring')
        selection = {'path': ring}
    else:
        second = _nearest_contour_point(contours, points[1], first['ring_index'])
        snapped.append(second)
        selection = {'contour_interval': _contour_interval(contours, first, second, cursor_path)}
    selected_ids = select(contours, selection)[2]
    rewritten, ident, invalidated_ids = _replace_contour_support(
        contours, session_request, kind, selection, selected_ids)
    report, prepared_request = prepare_interactive_report(contours, rewritten)
    feature = next(f for f in report['features'] if f['id'] == ident)
    if feature['status'] != 'fitted':
        raise ValueError(feature.get('reason', f'{kind.title()} could not be fitted'))
    return {'complete': True, 'snapped_points': [item['point'] for item in snapped],
            'ring_index': first['ring_index'], 'feature_id': ident,
            'invalidated_feature_ids': sorted(invalidated_ids),
            'request': prepared_request, 'report': report}


# Compound classes of the primitives-compounds-v2 taxonomy and their measurement routes.
COMPOUND_REDEFINITIONS = {
    'full_round_slot': {'semantic_id': 1, 'measurement_route': 'slot'},
    'triangular_thread': {'semantic_id': 2, 'measurement_route': 'thread_pitch'},
    'trapezoidal_thread': {'semantic_id': 3, 'measurement_route': 'thread_pitch'},
    'tooth_profile': {'semantic_id': 4, 'measurement_route': 'repeated_profile'},
}


def _bridge_short_gaps(contours, selected, max_gap, max_fraction=.25):
    """Return unselected edges lying in short gaps between selected edges.

    Clicking every constituent is impractical when segments are pixel-scale or
    have no retained support to display. A gap is bridged when it lies between
    selected edges of one ring and is no longer than ``max_gap`` (scaled by the
    caller to the selected segments) nor ``max_fraction`` of the selected
    arclength on that ring, so separate sides and lands between them stay out.
    """
    mask = np.zeros(len(contours.xy), bool)
    mask[np.asarray(sorted(selected), dtype=np.int64)] = True
    bridged = []
    for ring, (lo, hi) in enumerate(zip(contours.offsets[:-1], contours.offsets[1:])):
        q = contours.path(ring)
        local = mask[lo:hi].copy()
        lengths = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
        if not contours.closed[ring]:
            local[-1], lengths[-1] = False, 0.
        if not local.any() or local.all():
            continue
        limit = min(max_gap, max_fraction*float(lengths[local].sum()))
        count = len(local)
        # Walk gaps starting after a selected edge so closed rings wrap correctly.
        start = int(np.flatnonzero(local)[-1]) if contours.closed[ring] else int(np.flatnonzero(local)[0])
        stop = count if contours.closed[ring] else int(np.flatnonzero(local)[-1])-start
        gap = []
        for step in range(1, stop+1):
            index = (start+step) % count
            if not local[index]:
                gap.append(index)
                continue
            if gap and float(lengths[gap].sum()) <= limit:
                bridged.extend(lo+index for index in gap)
            gap = []
    return set(int(v) for v in bridged)


def _arclength(contours, ids):
    lengths = np.zeros(len(contours.xy))
    for ring, (lo, hi) in enumerate(zip(contours.offsets[:-1], contours.offsets[1:])):
        q = contours.path(ring)
        lengths[lo:hi] = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
    return float(lengths[np.asarray(ids, dtype=np.int64)].sum())


def redefine_compound(contours, session_request, semantic, feature_ids, bridge_gaps=True):
    """Group complete base instances into a user-declared compound.

    A base instance belongs to at most one compound, so compounds sharing any
    selected member are replaced. The compound support is the union of its
    members' observed edges, matching model compounds. With ``bridge_gaps``,
    short unselected gaps between selected members are included; a gap may be
    at most 1.5 times the longest selected member.
    """
    if semantic not in COMPOUND_REDEFINITIONS:
        raise ValueError('Compound type must be one of: ' + ', '.join(COMPOUND_REDEFINITIONS))
    if not isinstance(feature_ids, list) or not feature_ids or len(set(feature_ids)) != len(feature_ids):
        raise ValueError('Select one or more different base features for the compound')
    rewritten = deepcopy(session_request)
    bases = {f['id']: f for f in rewritten.get('features', [])
             if not (f.get('type') == 'point' and f.get('semantic') == 'vertex')}
    missing = [ident for ident in feature_ids if ident not in bases]
    if missing:
        raise ValueError('Compounds can only group base contour features')
    ids = [_feature_edge_ids(contours, bases[ident]) for ident in feature_ids]
    if any(not len(edges) for edges in ids):
        raise ValueError('A selected feature has no observed contour support')
    selected = set(int(v) for v in np.concatenate(ids))
    gap_edges = (_bridge_short_gaps(contours, selected, 1.5*max(_arclength(contours, edges) for edges in ids))
                 if bridge_gaps else set())
    bridged_ids = []
    if gap_edges:
        for ident, feature in bases.items():
            edges = set(int(v) for v in _feature_edge_ids(contours, feature))
            if ident not in feature_ids and edges and edges <= gap_edges:
                bridged_ids.append(ident)
                ids.append(np.asarray(sorted(edges), dtype=np.int64))
        # Edges owned by no base instance still close the observed support.
        ids.append(np.asarray(sorted(gap_edges), dtype=np.int64))
    feature_ids = list(feature_ids)+bridged_ids
    members = set(feature_ids)
    compounds = rewritten.get('compound_features', [])
    removed = {f['id'] for f in compounds if members.intersection(f.get('base_instance_ids', []))}
    prefix = f'user_{semantic}_'
    serials = [int(f['id'][len(prefix):]) for f in compounds
               if f.get('id', '').startswith(prefix) and f['id'][len(prefix):].isdigit()]
    ident = prefix+f'{max(serials, default=0)+1:04d}'
    rewritten['compound_features'] = [f for f in compounds if f['id'] not in removed] + [{
        'id': ident, 'semantic': semantic, **COMPOUND_REDEFINITIONS[semantic],
        'label_level': 'compound', 'base_instance_ids': list(feature_ids),
        'source': 'user', 'accept': True, 'status': 'accepted', 'confidence_calibrated': False,
        'units': 'px', 'parent': 'part', 'fit_geometry': 'original_observations',
        'raw_edge_indices': np.unique(np.concatenate(ids)).astype(int).tolist(),
    }]
    rewritten['relations'] = [r for r in rewritten.get('relations', [])
                              if r.get('feature_a') not in removed and r.get('feature_b') not in removed]
    rewritten['relations'] += [{'type': 'compound_membership', 'feature_a': member, 'feature_b': ident}
                               for member in feature_ids]
    rewritten.pop('overrides', None)
    rewritten.pop('measurements', None)
    report, prepared_request = prepare_interactive_report(contours, rewritten)
    feature = next(f for f in report['features'] if f['id'] == ident)
    if feature['status'] != 'fitted':
        name = semantic.replace('_', ' ').capitalize()
        if semantic == 'full_round_slot' and feature.get('reason') == 'Operator requires one closed contour':
            raise ValueError(f'{name} needs every segment of one closed opening')
        raise ValueError(f"{name} could not be fitted: {feature.get('reason', 'unsupported geometry')}")
    return {'complete': True, 'snapped_points': [], 'feature_id': ident,
            'bridged_feature_ids': bridged_ids,
            'invalidated_feature_ids': sorted(removed),
            'request': prepared_request, 'report': report}

def redefine_compound_sides(contours, session_request, semantic, points, cursor_paths=None):
    """Define a two-sided compound, such as a thread, from start/end clicks per side.

    Clicks are side A start, side A end, side B start, side B end; each pair
    snaps to one ring. Fewer than four points previews the snapped clicks.
    Base instances with most of their arclength inside a side's interval become
    members, so the compound keeps whole base instances.
    """
    if semantic not in COMPOUND_REDEFINITIONS:
        raise ValueError('Compound type must be one of: ' + ', '.join(COMPOUND_REDEFINITIONS))
    if not isinstance(points, list) or not 1 <= len(points) <= 4:
        raise ValueError('Select the start and end of each side: four points')
    snapped = []
    for index, point in enumerate(points):
        ring = snapped[index-1]['ring_index'] if index % 2 else None
        snapped.append(_nearest_contour_point(contours, point, ring))
    paths = list(cursor_paths or [])+[[], []]
    sides = []
    # Validate each side once its end is clicked, so a bad pick is reported immediately.
    for side in range(len(points)//2):
        interval = _contour_interval(contours, snapped[2*side], snapped[2*side+1], paths[side])
        try:
            sides.append(set(int(v) for v in select(contours, {'contour_interval': interval})[2]))
        except ValueError as error:
            raise ValueError(f"Side {'AB'[side]}: start and end must be different points along the contour") from error
    if len(points) < 4:
        return {'complete': False, 'snapped_points': [item['point'] for item in snapped],
                'ring_index': snapped[-1]['ring_index']}
    if sides[0] & sides[1]:
        raise ValueError('The two sides overlap; select each side separately')
    lengths = np.zeros(len(contours.xy))
    for ring, (lo, hi) in enumerate(zip(contours.offsets[:-1], contours.offsets[1:])):
        q = contours.path(ring)
        lengths[lo:hi] = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
    bases = [f for f in session_request.get('features', [])
             if not (f.get('type') == 'point' and f.get('semantic') == 'vertex')]
    members = []
    for side, edges in enumerate(sides):
        chosen = []
        for feature in bases:
            ids = _feature_edge_ids(contours, feature)
            if not len(ids):
                continue
            inside = sum(lengths[i] for i in ids if int(i) in edges)
            if inside > .5*float(lengths[ids].sum()):
                chosen.append(feature['id'])
        if not chosen:
            raise ValueError(f"No contour segments lie between the points selected on side {'AB'[side]}")
        members.extend(chosen)
    # Each side is already one contiguous interval; bridging could only join the
    # sides across a land such as a bolt's end face.
    result = redefine_compound(contours, session_request, semantic, members, bridge_gaps=False)
    result['snapped_points'] = [item['point'] for item in snapped]
    return result


def remove_compound(contours, session_request, feature_id):
    """Ungroup one compound back into its primitives.

    The base instances stay as they are; only the compound grouping and its
    relations are removed.
    """
    compounds = session_request.get('compound_features', [])
    compound = next((f for f in compounds if f.get('id') == feature_id), None)
    if compound is None:
        raise ValueError('Select a compound feature to remove')
    rewritten = deepcopy(session_request)
    ident = compound['id']
    rewritten['compound_features'] = [f for f in compounds if f['id'] != ident]
    rewritten['relations'] = [r for r in rewritten.get('relations', [])
                              if ident not in (r.get('feature_a'), r.get('feature_b'))]
    rewritten.pop('overrides', None)
    rewritten.pop('measurements', None)
    report, prepared_request = prepare_interactive_report(contours, rewritten)
    return {'complete': True, 'snapped_points': [], 'feature_id': ident,
            'restored_feature_ids': list(compound.get('base_instance_ids', [])),
            'invalidated_feature_ids': [ident],
            'request': prepared_request, 'report': report}


def _accepts(feature, requirement):
    if requirement == 'thread':
        return feature.get('type') == 'repeated_profile' and feature.get('semantic') in {
            'thread', 'triangular_thread', 'trapezoidal_thread', 'square_thread'}
    if requirement == 'tooth_profile':
        return feature.get('type') == 'repeated_profile' and feature.get('semantic') == 'tooth_profile'
    if requirement == 'round':
        return feature.get('type') in ('circle', 'arc')
    if requirement == 'any':
        return True
    if requirement == 'center':
        return feature.get('type') in CENTER_TYPES and 'center' in feature.get('geometry', {})
    if requirement == 'datum':
        return feature.get('type') == 'line' or (
            feature.get('type') in CENTER_TYPES and 'center' in feature.get('geometry', {}))
    return feature.get('type') == requirement


def evaluate_interactive_measurement(contours, session_request, session_report, request):
    """Validate and evaluate one user-authored planar measurement."""
    operator = request.get('operator')
    if operator not in MEASUREMENT_SPECS:
        raise ValueError('Unsupported interactive measurement operator')
    refs = request.get('featureRefs')
    spec = MEASUREMENT_SPECS[operator]
    if not isinstance(refs, list) or len(refs) != spec['count'] or len(set(refs)) != len(refs):
        raise ValueError(f"{spec['label']} requires {spec['count']} different feature selection(s)")
    records = {feature['id']: feature for feature in session_report['features']}
    for index, (ref, requirement) in enumerate(zip(refs, spec['accepts'])):
        feature = records.get(ref)
        if not feature or feature.get('status') != 'fitted':
            raise ValueError('Selected feature is unavailable for measurement')
        if not _accepts(feature, requirement):
            expected = ('a feature with a fitted center' if requirement == 'center' else
                        'a fitted line or center feature' if requirement == 'datum' else
                        'a circle or arc' if requirement == 'round' else
                        'a full round slot compound' if requirement == 'slot' else
                        'a tooth profile compound' if requirement == 'tooth_profile' else
                        f'a {requirement}')
            raise ValueError(f"Selection {index+1} for {spec['label']} must be {expected}")
    measurement = {
        'id': request.get('id', f'user_{operator}_' + '_'.join(refs)),
        'operator': operator, 'feature_refs': refs, 'params': request.get('params', {}),
    }
    if operator == 'distance_construction':
        # Preview reuses the current fitted report, including retained support
        # after corner exclusions. Slider edits must not refit every ellipse and
        # compound in the image. Geometry/corner edits replace this report.
        from .geometry import Geometry
        from .operators import measure
        from .engine import jsonable
        if session_report.get('coordinate_hash') != contours.coordinate_hash:
            raise ValueError('Measurement report is stale; analyze the current image again')
        params = measurement['params']
        if not isinstance(params, dict):
            raise ValueError('params must be an object')
        datum_ref = params.get('datum_feature_ref')
        if datum_ref:
            datum = records.get(datum_ref)
            if not datum or datum.get('status') != 'fitted' or datum['type'] != 'line':
                raise ValueError('Section datum must reference a fitted line')
            datum_axis = params.get('datum_axis', 'normal')
            if datum_axis not in ('normal', 'along'):
                raise ValueError('datum_axis must be normal or along')
            params = {**params, 'direction_mode': 'custom',
                      'direction': datum['geometry']['direction' if datum_axis == 'along' else 'normal']}
        selected = [Geometry(ref, records[ref]['type'],
                             [np.asarray(q, float) for q in records[ref].get('support', [])],
                             records[ref].get('closed', []),
                             {key: np.asarray(value, float) if isinstance(value, list) else value
                              for key, value in records[ref]['geometry'].items()}) for ref in refs]
        result = jsonable(measure(operator, selected, params))
        if datum_ref:
            result['definition'].update(datum_feature_ref=datum_ref, datum_axis=datum_axis)
        return dict(result, id=measurement['id'], operator=operator, feature_refs=refs,
                    status='measured', pass_fail=None, uncertainty=None, uncertainty_status='not_evaluated')
    evaluation_request = deepcopy(session_request)
    evaluation_request['measurements'] = [measurement]
    result = evaluate(contours, evaluation_request)['measurements'][0]
    if result['status'] != 'measured':
        raise ValueError(result.get('reason', 'Measurement could not be evaluated'))
    return result
