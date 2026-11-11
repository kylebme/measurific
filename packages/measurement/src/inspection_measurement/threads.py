"""Visible external-thread metrology on two ordered silhouette sides.

No helix, pitch diameter, or nominal template is inferred. Radii come only from
arc primitives reported at roots or crests; otherwise they remain unavailable.
"""
import numpy as np
from scipy.signal import find_peaks
from .geometry import direction


def statistics(samples, units='px', reason=None):
    if not samples:
        return dict(status='unavailable', minimum=None, maximum=None, average=None,
                    units=units, count=0, reason=reason or 'No resolved observations')
    values = np.asarray([s[0] for s in samples])
    lo, hi = int(values.argmin()), int(values.argmax())
    return dict(status='measured', minimum=float(values[lo]), maximum=float(values[hi]),
                average=float(values.mean()), units=units, count=len(samples),
                values=values.tolist(), minimum_witness=samples[lo][1],
                maximum_witness=samples[hi][1])


def _arc_radii(arcs, runs, u, v, heights):
    """Root/crest radii from arc primitives the contour model reported.

    A radius is reported only for a fitted arc member reaching a root or crest:
    concave (center outside the silhouette) and touching the root band, or
    convex and touching the crest band. Arcs on flanks, and polyline turns
    without a reported arc, are not radii.
    """
    result = {'root_radius': [], 'crest_radius': []}
    midline = np.mean([np.mean(q@v) for q in runs])
    for arc in arcs:
        center, radius = np.asarray(arc.values['center'], float), float(arc.values['radius'])
        support = arc.points
        side = int(np.mean(support@v) > midline)
        sign = [-1, 1][side]
        low, high = np.percentile(heights[side], [10, 90])
        band = (high-low)*.25
        height = sign*(support@v)
        observed = support[len(support)//2]
        convex = sign*(center@v) < sign*(observed@v)
        if convex and height.max() >= high-band:
            kind = 'crest_radius'
        elif not convex and height.min() <= low+band:
            kind = 'root_radius'
        else:
            continue
        p = center+radius*(observed-center)/np.linalg.norm(observed-center)
        result[kind].append((radius, [center.tolist(), p.tolist()]))
    return result


def _flank_crossings(q, u, v, sign, heights):
    """Corresponding same-direction flanks at one fixed radial section.

    Hysteresis prevents noisy crossings of that section from becoming additional
    thread forms. The crossing coordinates interpolate original edge segments.
    """
    low, high = np.percentile(heights, [10, 90])
    level, band = (low+high)/2, (high-low)*.15
    crossings = []
    for polarity in [1, -1]:
        z = polarity*(heights-level)
        armed = z[0] <= -band
        candidate = None
        points = []
        for i in range(1, len(q)):
            if z[i] <= -band:
                armed, candidate = True, None
            if armed and z[i-1] <= 0 < z[i]:
                t = -z[i-1]/(z[i]-z[i-1])
                point = q[i-1]+t*(q[i]-q[i-1])
                # Project only floating-point roundoff back to the section.
                candidate = (point@u)*u+sign*level*v
            if armed and z[i] >= band and candidate is not None:
                points.append(candidate)
                armed, candidate = False, None
        crossings.append(np.asarray(points))
    return crossings


def thread_measurements(g, params):
    if len(g.runs) == 1 and not any(g.closed):
        raise ValueError('Thread measurement needs both opposing sides; group the segments of both thread sides into one thread')
    if len(g.runs) != 2 or any(g.closed):
        raise ValueError('Thread measurement requires two connected open opposing silhouette sides')
    runs = [np.asarray(q, float) for q in g.runs]
    if min(map(len, runs)) < 7:
        raise ValueError('Insufficient thread support')
    if 'direction' in params:
        u = direction(params['direction'])
    else:
        # Center each side separately: shaft diameter must not rotate the axis.
        centered = np.concatenate([q-q.mean(0) for q in runs])
        u = np.linalg.svd(centered, full_matrices=False)[2][0]
    if u[np.argmax(np.abs(u))] < 0:
        u = -u
    v = np.array([-u[1], u[0]])
    runs.sort(key=lambda q: float(np.mean(q@v)))
    sides, heights = [], []
    for q, sign in zip(runs, [-1, 1]):
        if (q[-1]-q[0])@u < 0:
            q = q[::-1]
        x, y = q@u, sign*(q@v)
        # PAE observations can reverse locally around pixel-scale corners and
        # vertical flanks. Bound the entire reverse excursion, rather than each
        # step: sustained overhangs must not pass as many small backward steps.
        axial_tolerance = max(1e-7, 2*float(np.median(np.linalg.norm(np.diff(q, axis=0), axis=1))))
        reverse_excursion = float(np.max(np.maximum.accumulate(x)-x))
        if reverse_excursion > axial_tolerance:
            raise ValueError('Thread side backtracks along its axis; supply params.direction or select complete sides')
        depth = float(np.ptp(y))
        prominence = float(params.get('prominence', depth*.25))
        if not np.isfinite(prominence) or prominence <= 0:
            raise ValueError('Thread prominence must be finite and positive')
        y = np.round(y, 9)  # suppress floating-point plateau splitting after rotation
        peaks, _ = find_peaks(y, prominence=prominence)
        roots, _ = find_peaks(-y, prominence=prominence)
        if len(peaks) < 2 or len(roots) < 1:
            raise ValueError('At least two resolved crests and one root are required on each side')
        flat_tolerance = float(params.get('plateau_tolerance', depth*.002))
        if not np.isfinite(flat_tolerance) or flat_tolerance < 0 or flat_tolerance >= prominence:
            raise ValueError('Plateau tolerance must be nonnegative and smaller than prominence')
        def extrema(indices, heights):
            points = []
            for i in indices:
                a = b = int(i)
                while a > 0 and heights[i]-heights[a-1] <= flat_tolerance:
                    a -= 1
                while b < len(q)-1 and heights[i]-heights[b+1] <= flat_tolerance:
                    b += 1
                # Center a flat land axially even when optical/form noise makes
                # one isolated sample its maximum. Retain the extreme height.
                points.append(((x[a]+x[b])/2)*u + (q[i]@v)*v)
            return np.asarray(points)
        crests, valleys = extrema(peaks, y), extrema(roots, -y)
        if any(np.any(np.diff(points@u) <= 0) for points in (crests, valleys)):
            raise ValueError('Resolved thread crests and roots must progress along the axis')
        sides.append({'crest': crests, 'root': valleys, 'sign': sign,
                      'flanks': _flank_crossings(q, u, v, sign, y)})
        heights.append(y)
    metrics = {}
    envelope_runs = []
    for key, role in [('major_diameter', 'crest'), ('minor_diameter', 'root')]:
        a, b = sides[0][role], sides[1][role]
        ax, bx = a@u, b@u
        stations = np.unique(np.r_[ax, bx])
        stations = stations[(stations >= max(ax[0], bx[0])) & (stations <= min(ax[-1], bx[-1]))]
        samples = []
        for x in stations:
            ay, by = np.interp(x, ax, a@v), np.interp(x, bx, b@v)
            if by <= ay:
                raise ValueError('Thread sides overlap or do not oppose each other')
            samples.append((by-ay, [(x*u+ay*v).tolist(), (x*u+by*v).tolist()]))
        metrics[key] = statistics(samples, reason='Opposing envelopes have no common axial span')
        envelope_runs.extend([a.tolist(), b.tolist()])
        if samples and len(stations) > 1:
            from .distances import distance_construction
            from .geometry import Geometry
            field = distance_construction(Geometry('envelope_a', 'contour', [a], [False]),
                                          Geometry('envelope_b', 'contour', [b], [False]),
                                          {'construction': 'normal_gap', 'direction_mode': 'custom', 'direction': v})
            for name in ('minimum', 'maximum', 'average', 'minimum_witness', 'maximum_witness'):
                value = field[name]
                metrics[key][name] = np.asarray(value).tolist() if name.endswith('witness') else value
            widths = np.diff(stations)
            metrics[key]['sample_weights'] = (np.r_[widths, 0]/2+np.r_[0, widths]/2).tolist()
            metrics[key]['method'] = 'axis_normal_gap_between_virtual_envelopes_interval_weighted_mean'
    pitch, counts = [], []
    for side in sides:
        peaks = side['crest']
        for flanks in side['flanks']:
            if len(flanks) > 1 and np.any(np.diff(flanks@u) <= 0):
                raise ValueError('Corresponding thread flanks must progress along the axis')
            pitch.extend((float((b-a)@u), [a.tolist(), b.tolist()]) for a, b in zip(flanks[:-1], flanks[1:]))
        counts.append((len(peaks), [peaks[0].tolist(), peaks[-1].tolist()]))
    if not pitch:
        raise ValueError('Pitch requires at least two corresponding flanks at the reference section')
    metrics['pitch'] = statistics(pitch)
    metrics['pitch']['method'] = 'axial_spacing_of_corresponding_same_direction_flanks_at_fixed_radial_section'
    metrics['number_of_threads'] = statistics(counts, 'count')
    for key, samples in _arc_radii([m for m in g.members if m.kind == 'arc'], runs, u, v, heights).items():
        metrics[key] = statistics(samples, reason='No arc reported at a thread root or crest')
        metrics[key]['method'] = 'reported_root_or_crest_arc_radius'
    return dict(value=metrics['pitch']['average'], units='px', method='opposing_visible_thread_envelopes',
                thread_metrics=metrics, axis_direction=u.tolist(), envelope_runs=envelope_runs,
                axis_witness=[(x*u+np.mean(np.concatenate(runs)@v)*v).tolist()
                              for x in [min(np.min(q@u) for q in runs),
                                        max(np.max(q@u) for q in runs)]],
                scope='External 2D silhouette; count is resolved crests per side, excluding partial end crests. '
                      'Pitch is axial spacing of corresponding same-direction flanks at a fixed radial section. '
                      'Diameters interpolate opposing crest/root envelopes. Radii are reported only for arcs '
                      'at a root or crest; otherwise they are unavailable. No nominal thread standard or 3D lead is inferred.')
