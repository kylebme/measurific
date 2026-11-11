"""Explicit measurement definitions; no semantics-dependent numeric shortcuts."""
import numpy as np
import shapely
from scipy.signal import find_peaks
from shapely.geometry import LineString, Polygon
from shapely.ops import nearest_points
from .geometry import (center, direction, line, polygon, single,
                       minimum_zone_straightness, fit_geometry, Geometry)


# Public registry is also used for request validation and documentation.
OPERATORS = {
    'length', 'orientation', 'radius', 'least_squares_diameter', 'arc_length',
    'angular_span', 'chord_length', 'sagitta', 'major_axis', 'minor_axis',
    'width', 'overall_length', 'end_radius', 'corner_radius', 'end_center_spacing',
    'rms_residual', 'peak_to_valley_residual', 'max_abs_residual',
    'straightness', 'circularity_ls', 'area', 'perimeter', 'centroid', 'extent',
    'center', 'x', 'y', 'distance', 'center_distance', 'center_offset',
    'minimum_separation', 'distance_construction', 'angle', 'parallelism', 'perpendicularity', 'angularity',
    'intersection', 'tangency', 'symmetry', 'common_axis', 'profile',
    'true_position_2d', 'pattern', 'repetition', 'compound', 'local_size',
    'curvature', 'wall_thickness', 'datum', 'concentricity', 'thread',
    'slot', 'tooth_profile',
}


def result(value, units='px', method=None, **details):
    return {'value': value, 'units': units, 'method': method, **details}


def require_count(gs, count):
    if len(gs) != count:
        raise ValueError(f'Operator requires {count} feature references')


def scalar(value, name, positive=False):
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(float(value)):
        raise ValueError(f'{name} must be finite')
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def intersect(a, b):
    ca, ua, _ = line(a)
    cb, ub, _ = line(b)
    matrix = np.c_[ua, -ub]
    if abs(np.linalg.det(matrix)) < 1e-8:
        raise ValueError('Lines are parallel or intersection is ill-conditioned')
    return ca+np.linalg.solve(matrix, cb-ca)[0]*ua


def angle(a, b):
    _, ua, _ = line(a)
    _, ub, _ = line(b)
    return float(np.degrees(np.arccos(np.clip(abs(ua@ub), 0, 1))))


def width_at(a, b, station):
    ca, ua, na = line(a)
    cb, ub, _ = line(b)
    if abs(ua@ub) < 1e-8:
        raise ValueError('Width is undefined at perpendicular flanks')
    anchor = ca+station*ua
    t = (anchor-cb)@ua/(ub@ua)
    return float((cb+t*ub-anchor)@na), anchor, cb+t*ub


def _point_distances(points, target):
    """Exact point-to-segment distances, indexed for large support sets."""
    coordinates, owners = shapely.get_coordinates(shapely.get_parts(target), return_index=True)
    if len(coordinates) < 64 or len(points)*len(coordinates) < 65536:
        return shapely.distance(points, target)
    # Index individual segments so each query can prune distant support. Adjacent
    # components must never acquire an artificial connecting segment.
    same_part = owners[:-1] == owners[1:]
    segments = shapely.linestrings(np.stack([coordinates[:-1][same_part], coordinates[1:][same_part]], axis=1))
    indices, distances = shapely.STRtree(segments).query_nearest(points, return_distance=True, all_matches=False)
    result = np.empty(len(points))
    result[indices[0]] = distances
    return result


def nominal_profile(g, params):
    q = g.points
    nominal = np.asarray(params['nominal_points'], float)
    if nominal.ndim != 2 or nominal.shape[1] != 2 or len(nominal) < 2 or not np.isfinite(nominal).all():
        raise ValueError('nominal_points must be finite [N, 2]')
    closed = params.get('closed', False)
    if not isinstance(closed, bool):
        raise ValueError('closed must be boolean')
    boundary = LineString(np.vstack([nominal, nominal[0]]) if closed else nominal)
    if not boundary.is_simple or boundary.length <= 1e-12:
        raise ValueError('Nominal profile must be simple and nonzero length')
    points = shapely.points(q)
    distances = _point_distances(points, boundary)
    details = {'distances': distances, 'worst_point': q[np.argmax(distances)],
               'rms': float(np.sqrt(np.mean(distances**2))), 'maximum': float(distances.max()),
               'alignment': 'as_supplied_no_registration'}
    if closed:
        region = Polygon(nominal)
        if not region.is_valid or region.area <= 1e-12:
            raise ValueError('Nominal region is invalid')
        signed = distances*np.where(shapely.contains(region, points), -1, 1)
        details.update(signed_distances=signed, inward=float(max(0, -signed.min())),
                       outward=float(max(0, signed.max())), peak_to_valley=float(np.ptp(signed)))
    # Continuous measured segments against nominal vertices in the reverse direction.
    # Forward direction samples observed vertices: deliberately named sampled Hausdorff.
    reverse = _point_distances(shapely.points(nominal), g.shape()).max()
    details['sampled_hausdorff'] = float(max(reverse, distances.max()))
    return result(float(distances.max()), method='observed_vertex_to_nominal_segment_distance', **details)


def size_witness(operator, g):
    """Dimension endpoints for fitted line length, radius and diameter."""
    if operator == 'length' and g.kind == 'line':
        return np.asarray(g.values['endpoints'], float)
    if operator not in ('radius', 'least_squares_diameter') or g.kind not in ('circle', 'arc'):
        return None
    c, r = np.asarray(g.values['center'], float), g.values['radius']
    if g.kind == 'arc':
        q = g.points
        angles = np.unwrap(np.arctan2(q[:, 1]-c[1], q[:, 0]-c[0]))
        middle = (angles[0]+angles[-1])/2
    else:
        # Drawing convention for a full circle: a diagonal toward the upper right.
        middle = -np.pi/4
    d = r*np.array([np.cos(middle), np.sin(middle)])
    return np.array([c, c+d] if operator == 'radius' else [c-d, c+d])


def pattern(gs, params):
    if len(gs) == 1 and gs[0].members:
        gs = gs[0].members
    if len(gs) < 2:
        raise ValueError('Pattern needs at least two members')
    points = np.array([center(g) for g in gs])
    domain = params.get('domain', 'linear')
    details = {'count': len(points), 'member_order': [g.id for g in gs]}
    if domain == 'linear':
        u = direction(params['direction'])
        v = np.array([-u[1], u[0]])
        order = np.argsort(points@u)
        projected = points[order]@u
        pitch = np.diff(projected)
        if (pitch <= 1e-10).any():
            raise ValueError('Pattern members overlap along the supplied axis')
        nominal = scalar(params.get('nominal_pitch', pitch.mean()), 'nominal_pitch', True)
        error = projected-projected[0]-np.arange(len(points))*nominal
        details.update(member_order=[gs[i].id for i in order], local_pitch=pitch,
                       local_pitch_error=pitch-nominal, cumulative_pitch_error=error,
                       row_straightness=float(np.ptp(points@v)),
                       orientation=float(np.degrees(np.arctan2(u[1], u[0]))),
                       nominal_pitch=nominal, nominal_source='supplied' if 'nominal_pitch' in params else 'measured_mean')
        return result(float(pitch.mean()), method='projected_center_spacing', **details)
    if domain == 'polar':
        if 'center' in params:
            origin = np.asarray(params['center'], float)
            if origin.shape != (2,) or not np.isfinite(origin).all():
                raise ValueError('Invalid pattern center')
        else:
            if len(points) < 3:
                raise ValueError('Polar center fitting requires at least three members')
            # Sort only to create valid connected full-circle fitting support.
            order = np.argsort(np.arctan2(*(points-points.mean(0))[:, ::-1].T))
            fitted = fit_geometry(Geometry('pattern_center', 'circle', [points[order]], [True]))
            origin = center(fitted)
        radial = np.linalg.norm(points-origin, axis=1)
        if (radial < 1e-9).any():
            raise ValueError('Pattern member lies at polar origin')
        angles = np.mod(np.arctan2(points[:, 1]-origin[1], points[:, 0]-origin[0]), 2*np.pi)
        order = np.argsort(angles)
        angles = angles[order]
        gaps = np.diff(np.r_[angles, angles[0]+2*np.pi])
        if gaps.min() < 1e-10:
            raise ValueError('Polar members have coincident angles')
        nominal = scalar(params.get('nominal_pitch_degrees', 360/len(points)), 'nominal_pitch_degrees', True)
        phase = scalar(params.get('phase_degrees', np.degrees(angles[0])), 'phase_degrees')
        errors = np.degrees(angles)-phase-np.arange(len(points))*nominal
        details.update(center=origin, member_order=[gs[i].id for i in order], angular_pitch=np.degrees(gaps),
                       local_pitch_error=np.degrees(gaps)-nominal, indexing_error=errors,
                       phase=phase, radial_error=radial[order]-scalar(params.get('nominal_radius', radial.mean()), 'nominal_radius', True),
                       mean_radius=float(radial.mean()), tangential_error=np.radians(errors)*radial[order])
        return result(float(np.degrees(gaps).mean()), 'deg', 'polar_center_spacing', **details)
    if domain == 'grid':
        u = direction(params['direction'])
        v = np.array([-u[1], u[0]])
        cells = np.asarray(params['cells'])
        if cells.shape != (len(points), 2) or cells.dtype.kind not in 'iu' or len(np.unique(cells, axis=0)) != len(points):
            raise ValueError('Grid needs one unique integer [column, row] per member')
        design = np.c_[np.ones(len(points)), cells]
        if np.linalg.matrix_rank(design) < 3:
            raise ValueError('Grid requires at least two rows and columns with independent support')
        coefficients = np.linalg.lstsq(design, points, rcond=None)[0]
        a, b = coefficients[1:]
        residual = points-design@coefficients
        orthogonality = abs(90-np.degrees(np.arccos(np.clip(direction(a)@direction(b), -1, 1))))
        return result(float(np.max(np.linalg.norm(residual, axis=1))), method='affine_grid_center_fit',
                      count=len(points), origin=coefficients[0], column_pitch=float(np.linalg.norm(a)),
                      row_pitch=float(np.linalg.norm(b)), orthogonality_degrees=float(orthogonality),
                      residual_vectors=residual, requested_axis_error_degrees=float(np.degrees(np.arccos(np.clip(abs(direction(a)@u), 0, 1)))))
    if domain == 'arbitrary':
        nominal = np.asarray(params['nominal_centers'], float)
        if nominal.shape != points.shape or not np.isfinite(nominal).all():
            raise ValueError('nominal_centers must match ordered members')
        errors = points-nominal
        return result(float(np.max(np.linalg.norm(errors, axis=1))), method='as_supplied_member_correspondence',
                      error_vectors=errors, **details)
    raise ValueError('Unknown pattern domain')


def repetition(g, params):
    q = single(g)
    domain = params.get('domain', 'linear')
    if domain == 'linear':
        u = direction(params['direction'])
        v = np.array([-u[1], u[0]])
        x, y = q@u, q@v
        if np.all(np.diff(x) < 0):
            x, y = x[::-1], y[::-1]
        if (np.diff(x) <= 1e-10).any():
            raise ValueError('Linear repetition requires a strictly monotone selected side')
        units = 'px'
    elif domain == 'angular':
        origin = np.asarray(params['center'], float)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError('Invalid repetition center')
        d = q-origin
        x, y = np.unwrap(np.arctan2(d[:, 1], d[:, 0])), np.linalg.norm(d, axis=1)
        if np.all(np.diff(x) < 0):
            x, y = -x, y
        if (np.diff(x) <= 1e-10).any():
            raise ValueError('Angular profile must be single valued about supplied center')
        if g.closed[0]:
            x, y = np.r_[x, x[0]+2*np.pi], np.r_[y, y[0]]
        x = np.degrees(x)
        units = 'deg'
    else:
        raise ValueError('Repetition domain must be linear or angular')
    grid = np.linspace(x[0], x[-1], max(512, len(q)*4))
    profile = np.interp(grid, x, y)
    signal = profile-np.polyval(np.polyfit(grid, profile, 1), grid) if domain == 'linear' else profile-profile.mean()
    prominence = scalar(params['prominence'], 'prominence', True)
    sign = params.get('extrema', 'maxima')
    if sign not in ('maxima', 'minima'):
        raise ValueError('extrema must be maxima or minima')
    signal *= 1 if sign == 'maxima' else -1
    periodic = domain == 'angular' and g.closed[0]
    if periodic:
        signal, grid = signal[:-1], grid[:-1]
        extended = np.tile(signal, 3)
        peaks, _ = find_peaks(extended, prominence=prominence)
        peaks = peaks[(peaks >= len(signal)) & (peaks < 2*len(signal))]-len(signal)
    else:
        peaks, _ = find_peaks(signal, prominence=prominence)
    positions = grid[peaks]
    if len(positions) < 3:
        raise ValueError('At least three resolved extrema are required')
    local = np.diff(np.r_[positions, positions[0]+360]) if periodic else np.diff(positions)
    pitch = float(local.mean())
    nominal = scalar(params.get('nominal_pitch', pitch), 'nominal_pitch', True)
    cumulative = positions-positions[0]-np.arange(len(positions))*nominal
    return result(pitch, units, 'uniform_resampling_prominent_extrema', count=len(peaks),
                  local_pitch=local, local_pitch_error=local-nominal, cumulative_pitch_error=cumulative,
                  extrema_positions=positions, grid_spacing=float(grid[1]-grid[0]),
                  profile_height=float(np.ptp(y)), observed_span=float(x[-1]-x[0]),
                  nominal_source='supplied' if 'nominal_pitch' in params else 'measured_mean')


def compound(g, params):
    members = g.members
    if g.kind in ('width', 'step', 'angle', 'taper'):
        require_count(members, 2)
        a, b = members
        if g.kind == 'angle':
            p = intersect(a, b)
            return result(angle(a, b), 'deg', 'acute_infinite_line_angle', virtual_sharp=p)
        if g.kind == 'step':
            _, _, n = line(a)
            if angle(a, b) > scalar(params.get('parallel_tolerance_degrees', .1), 'parallel_tolerance_degrees'):
                raise ValueError('Step lands must be parallel within the specified angular limit')
            return result(abs((center(b)-center(a))@n), method='parallel_land_separation')
        stations = np.asarray(params.get('stations', [0.]), float)
        if stations.ndim != 1 or len(stations) == 0 or not np.isfinite(stations).all():
            raise ValueError('stations must be a nonempty finite array')
        if g.kind == 'taper' and (len(stations) < 2 or np.ptp(stations) <= 1e-10):
            raise ValueError('Taper requires at least two distinct stations')
        samples = [width_at(a, b, s) for s in stations]
        widths = np.abs([s[0] for s in samples])
        out = result(float(widths.mean()), method='normal_to_first_line_at_stations', widths=widths,
                     stations=stations, flank_angle=angle(a, b), minimum=float(widths.min()), maximum=float(widths.max()))
        if g.kind == 'taper':
            out['signed_width_slope'] = float(np.polyfit(stations, [s[0] for s in samples], 1)[0])
        return out
    if g.kind in ('groove', 'notch'):
        # Explicit member roles avoid guessing design intent from taxonomy labels.
        roles = params.get('roles', {})
        by_id = {m.id: m for m in members}
        opening, left, right = [by_id[roles[k]] for k in ('opening', 'left', 'right')]
        p, q = intersect(opening, left), intersect(opening, right)
        _, _, n = line(opening)
        values = {'opening_width': float(np.linalg.norm(p-q)),
                  'left_flank_angle': angle(opening, left), 'right_flank_angle': angle(opening, right)}
        if 'bottom' in roles:
            bottom = by_id[roles['bottom']]
            r, s = intersect(bottom, left), intersect(bottom, right)
            values.update(bottom_width=float(np.linalg.norm(r-s)), depth=float(abs(((r+s-p-q)/2)@n)))
        else:
            apex = intersect(left, right)
            values.update(bottom_width=0., depth=float(abs((apex-p)@n)), virtual_sharp=apex)
        return result(values['depth'], method='line_intersection_groove', **values)
    if g.kind == 'pattern':
        return pattern(members, params)
    raise ValueError('Unsupported compound type; rounded profiles use their fitted scalar operators')


def measure(operator, gs, params):
    if operator not in OPERATORS:
        raise ValueError(f'Unsupported operator: {operator}')
    if operator == 'distance_construction':
        from .distances import distance_construction
        require_count(gs, 2)
        return distance_construction(gs[0], gs[1], params)
    if operator == 'thread':
        from .threads import thread_measurements
        require_count(gs, 1)
        return thread_measurements(gs[0], params)
    if operator in ('slot', 'tooth_profile'):
        from .profiles import slot_measurements, tooth_profile_measurements
        require_count(gs, 1)
        return (slot_measurements if operator == 'slot' else tooth_profile_measurements)(gs[0], params)
    if operator == 'pattern':
        return pattern(gs, params)
    unary = OPERATORS-{'distance', 'center_distance', 'center_offset', 'minimum_separation',
                       'angle', 'parallelism', 'perpendicularity', 'angularity', 'intersection',
                       'tangency', 'symmetry', 'common_axis', 'pattern', 'wall_thickness',
                       'concentricity'}
    if operator in unary:
        require_count(gs, 1)
    g = gs[0]
    if operator in ('center', 'x', 'y'):
        c = center(g)
        return result(c if operator == 'center' else float(c[0 if operator == 'x' else 1]), method='associated_center')
    if operator == 'datum':
        if g.kind == 'line':
            c, u, n = line(g)
            return result(g.values['orientation'], 'deg', 'fitted_line_datum',
                          datum={'origin': c, 'direction': u, 'normal': n, 'type': 'line'})
        c = center(g)
        return result(0., method='fitted_center_datum', datum={'origin': c, 'type': 'center'})
    if operator in g.values and operator not in ('compound', 'pattern'):
        unit = 'deg' if operator in ('orientation', 'angular_span') else 'px'
        witness = size_witness(operator, g)
        return result(g.values[operator], unit, g.values['method'],
                      **({} if witness is None else {'witness': witness}))
    if operator == 'straightness':
        line(g)
        return result(minimum_zone_straightness(g.points), method='minimum_parallel_strip_convex_hull')
    if operator == 'circularity_ls':
        if g.kind != 'circle':
            raise ValueError('Circularity requires a complete circle; use arc residuals for partial arcs')
        return result(g.values['peak_to_valley_residual'], method='radial_range_about_least_squares_center')
    if operator in ('area', 'centroid'):
        p = polygon(g)
        return result(float(p.area) if operator == 'area' else np.array(p.centroid.coords[0]),
                      'px^2' if operator == 'area' else 'px', 'observed_polygon')
    if operator in ('perimeter', 'length'):
        if operator == 'perimeter':
            single(g, True)
        return result(float(g.shape().length), method='observed_polyline_length')
    if operator == 'extent':
        u = direction(params['direction'])
        return result(float(np.ptp(g.points@u)), method='projected_observed_extent', direction=u)
    if operator in ('distance', 'center_distance', 'center_offset', 'minimum_separation'):
        require_count(gs, 2)
        a, b = gs
        if operator in ('center_distance', 'center_offset'):
            pa, pb = center(a), center(b)
        elif operator == 'minimum_separation' or params.get('mode') == 'segments':
            pa, pb = [np.array(p.coords[0]) for p in nearest_points(a.shape(), b.shape())]
        elif a.kind == 'line' and b.kind == 'line':
            if angle(a, b) > scalar(params.get('parallel_tolerance_degrees', .1), 'parallel_tolerance_degrees'):
                raise ValueError('Nonparallel line distance requires mode=segments or width at a station')
            ca, _, na = line(a)
            pb = center(b)
            pa = pb-((pb-ca)@na)*na
        elif a.kind == 'line' or b.kind == 'line':
            ln, other = (a, b) if a.kind == 'line' else (b, a)
            c, _, n = line(ln)
            p = center(other)
            foot = p-((p-c)@n)*n
            pa, pb = (foot, p) if a.kind == 'line' else (p, foot)
        else:
            raise ValueError('Use center_distance or minimum_separation for these feature types')
        delta = pb-pa
        value = float(np.linalg.norm(delta))
        if 'direction' in params:
            value = float(delta@direction(params['direction']))
        return result(value, method='finite_segments' if operator == 'minimum_separation' or params.get('mode') == 'segments' else 'associated_geometry',
                      witness=np.array([pa, pb]), offset_vector=delta)
    if operator == 'concentricity':
        require_count(gs, 2)
        pa, pb = center(gs[0]), center(gs[1])
        return result(float(np.linalg.norm(pb-pa)), method='planar_fitted_center_offset',
                      witness=np.array([pa, pb]), offset_vector=pb-pa)
    if operator in ('angle', 'parallelism', 'perpendicularity', 'angularity'):
        require_count(gs, 2)
        theta = angle(*gs)
        value = theta if operator in ('angle', 'parallelism') else abs(theta-(90 if operator == 'perpendicularity' else scalar(params['nominal_angle'], 'nominal_angle')))
        return result(value, 'deg', 'acute_unoriented_line_angle')
    if operator == 'intersection':
        require_count(gs, 2)
        return result(intersect(*gs), method='infinite_fitted_line_intersection')
    if operator == 'tangency':
        require_count(gs, 2)
        a, b = gs
        if a.kind == 'line' or b.kind == 'line':
            ln, arc = (a, b) if a.kind == 'line' else (b, a)
            c, _, n = line(ln)
            if arc.kind not in ('circle', 'arc'):
                raise ValueError('Tangency requires line/circle or circle/circle geometry')
            error = abs(abs((center(arc)-c)@n)-arc.values['radius'])
        else:
            if a.kind not in ('circle', 'arc') or b.kind not in ('circle', 'arc'):
                raise ValueError('Tangency requires circles or arcs')
            mode = params.get('mode', 'external')
            if mode not in ('external', 'internal'):
                raise ValueError('Circle tangency mode must be external or internal')
            target = a.values['radius']+b.values['radius'] if mode == 'external' else abs(a.values['radius']-b.values['radius'])
            error = abs(np.linalg.norm(center(a)-center(b))-target)
        return result(float(error), method='supporting_line_circle_tangency_gap', scope='supporting_geometry_not_contact_on_observed_arc')
    if operator == 'symmetry':
        require_count(gs, 3)
        a, b, datum = gs
        c, _, n = line(datum)
        reflected = center(a)-2*((center(a)-c)@n)*n
        return result(float(np.linalg.norm(reflected-center(b))), method='reflected_center_offset')
    if operator == 'common_axis':
        if len(gs) < 2:
            raise ValueError('Common axis needs an explicit datum line and at least one feature')
        c, _, n = line(gs[0])
        deviations = np.array([(center(a)-c)@n for a in gs[1:]])
        return result(float(np.max(np.abs(deviations))), method='centers_to_datum_line', signed_offsets=deviations)
    if operator == 'profile':
        return nominal_profile(g, params)
    if operator == 'true_position_2d':
        nominal = np.asarray(params['nominal_center'], float)
        origin = np.asarray(params['datum_origin'], float)
        u = direction(params['datum_x_direction'])
        if nominal.shape != (2,) or origin.shape != (2,) or not np.isfinite([nominal, origin]).all():
            raise ValueError('Invalid nominal center or datum origin')
        local = (center(g)-origin)@np.stack([u, [-u[1], u[0]]]).T
        return result(float(2*np.linalg.norm(local-nominal)), method='diametral_2d_position_in_explicit_frame', measured_center=local)
    if operator == 'repetition':
        return repetition(g, params)
    if operator == 'compound':
        return compound(g, params)
    if operator == 'curvature':
        q = single(g)
        if len(q) < 3:
            raise ValueError('Curvature requires at least three points')
        prev, mid, nxt = (np.roll(q, 1, axis=0), q, np.roll(q, -1, axis=0)) if g.closed[0] else (q[:-2], q[1:-1], q[2:])
        a, b = mid-prev, nxt-mid
        denom = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)*np.linalg.norm(nxt-prev, axis=1)
        if (denom < 1e-12).any():
            raise ValueError('Curvature has degenerate support')
        k = 2*(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])/denom
        return result(k, '1/px', 'signed_three_point_circumcircle', points=mid)
    if operator == 'local_size':
        region = polygon(g)
        u = direction(params['direction'])
        v = np.array([-u[1], u[0]])
        origin = np.asarray(params.get('origin', [0, 0]), float)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError('Invalid local-size origin')
        stations = np.asarray(params['stations'], float)
        if stations.ndim != 1 or len(stations) == 0 or not np.isfinite(stations).all():
            raise ValueError('Invalid local-size stations')
        reach = float(np.linalg.norm(g.points-origin, axis=1).max()*2+1)
        sizes = []
        for station in stations:
            p = origin+station*u
            section = region.intersection(LineString([p-reach*v, p+reach*v]))
            if section.geom_type != 'LineString' or section.length <= 1e-10:
                raise ValueError('Section is empty, tangent, or has multiple material intervals')
            sizes.append(section.length)
        return result(float(np.mean(sizes)), method='polygon_cross_sections', stations=stations,
                      sizes=sizes, minimum=float(min(sizes)), maximum=float(max(sizes)))
    if operator == 'wall_thickness':
        require_count(gs, 2)
        q, target = gs[0].points, gs[1].shape()
        distances = _point_distances(shapely.points(q), target)
        return result(float(distances.min()), method='directed_observed_vertex_to_segment_distance',
                      distribution=distances, maximum=float(distances.max()), points=q,
                      scope='directed_distance_distribution_not_medial_axis_thickness')
    raise ValueError(f'{operator} is not supported by {g.kind}')
