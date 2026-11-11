"""Versioned JSON requests, independent feature fitting and tolerance evaluation."""
from copy import deepcopy
import json
import numpy as np
from .contours import Contours, select, integer
from .geometry import Geometry, fit_geometry, idealized_support
from .operators import measure, OPERATORS, scalar
from .corners import corner_exclusions

COMPOUNDS = {'width', 'angle', 'groove', 'notch', 'step', 'taper', 'pattern'}
ROUTES = {'profile': 'contour', 'external_profile': 'contour', 'polygon': 'polygon',
          'rounded_profile': 'contour', 'thread_pitch': 'repeated_profile',
          'tab': 'contour', 'groove': 'contour', 'notch': 'contour'}
DEFAULTS = {
    'line': ['length', 'orientation', 'straightness', 'rms_residual'],
    'circle': ['center', 'least_squares_diameter', 'circularity_ls'],
    'arc': ['radius', 'angular_span', 'arc_length', 'rms_residual'],
    'ellipse': ['center', 'major_axis', 'minor_axis', 'orientation', 'rms_residual'],
    'elliptical_arc': ['major_axis', 'minor_axis', 'angular_span', 'arc_length'],
    'slot': ['width', 'overall_length', 'end_radius', 'orientation', 'rms_residual'],
    'rounded_rectangle': ['width', 'overall_length', 'corner_radius', 'rms_residual'],
    'polygon': ['area', 'perimeter', 'centroid'],
    'contour': ['length'], 'freeform': ['length'], 'repeated_profile': ['length'],
    'point': [],
}


def jsonable(value):
    if isinstance(value, np.ndarray):
        # Numeric arrays already know every element's type. Check finiteness in
        # one pass and let NumPy convert them without Python recursion per pixel.
        if value.dtype.kind in 'biu':
            return value.tolist()
        if value.dtype.kind == 'f' and value.dtype.itemsize <= 8:
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite result cannot be serialized')
            return value.tolist()
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('Nonfinite result cannot be serialized')
    return value


def material_side(contours, runs, samples=64):
    """Which side of contour travel the part material lies on, or None without closed rings.

    Material is the even-odd interior of all closed rings, so holes and islands
    count correctly whatever the ring orientation. A short normal step from
    sampled edge midpoints votes for the side.
    """
    import shapely
    from shapely.geometry import Polygon
    rings = [contours.path(i) for i in range(len(contours.offsets)-1) if contours.closed[i]]
    if not rings:
        return None
    starts, ends = [], []
    for run in runs:
        index = np.unique(np.linspace(0, len(run)-2, min(samples, len(run)-1)).astype(int))
        starts.append(run[index])
        ends.append(run[index+1])
    a, b = np.concatenate(starts), np.concatenate(ends)
    t = b-a
    length = np.linalg.norm(t, axis=1)
    keep = length > 1e-9
    mid, right = ((a+b)/2)[keep], (np.c_[t[:, 1], -t[:, 0]]/np.maximum(length, 1e-12)[:, None])[keep]
    probe = mid+right*np.minimum(.25, length[keep][:, None]/4)
    inside = sum(shapely.contains_xy(Polygon(ring), probe[:, 0], probe[:, 1]).astype(int) for ring in rings)
    return 'right' if np.mean(inside % 2 == 1) > .5 else 'left'


def indexed(items, name):
    if not isinstance(items, list):
        raise ValueError(f'{name} must be a list')
    out = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id']:
            raise ValueError(f'Every {name} entry requires a nonempty string id')
        if item['id'] in out:
            raise ValueError(f'Duplicate {name} id: {item["id"]}')
        out[item['id']] = item
    return out


def normalize(request, contours):
    req = deepcopy(request)
    if not isinstance(req, dict):
        raise ValueError('Request must be an object')
    primitive_prediction=(req.get('schema_version')==2 and
                          req.get('label_version') in ('primitives-compounds-v1','primitives-compounds-v2'))
    if (req.get('schema_version', 1) != 1 and not primitive_prediction) or req.get('units', 'px') != 'px':
        raise ValueError('Expected schema_version=1 or primitive prediction schema_version=2, and units=px')
    if req.get('coordinate_system', 'image') != 'image':
        raise ValueError('Coordinates must be image pixels (+y down)')
    if 'coordinate_hash' in req and req['coordinate_hash'] != contours.coordinate_hash:
        raise ValueError('Request contour hash mismatch: edge selections refer to different observations')
    bases = indexed(req.get('features', []), 'features')
    compounds = indexed(req.get('compound_features', []) if primitive_prediction else [], 'compound_features')
    features = indexed(list(bases.values()) + list(compounds.values()), 'features')
    if primitive_prediction:
        for f in bases.values():
            f['label_level'] = 'base'
        for f in compounds.values():
            f['label_level'] = 'compound'
        relations = req.get('relations', [])
        if not isinstance(relations, list) or any(
            not isinstance(r, dict) or r.get('feature_a') not in features or r.get('feature_b') not in features
            for r in relations
        ):
            raise ValueError('Prediction relations must reference existing features')
    # Overrides deliberately replace specified fields, not whole model documents.
    overrides = indexed(req.get('overrides', []), 'overrides')
    for ident, override in overrides.items():
        if ident not in features:
            raise ValueError(f'Override refers to unknown feature {ident}')
        if override.get('source') != 'user':
            raise ValueError('Overrides require source=user')
        f = features[ident]
        if 'selection' in override:
            f.pop('raw_edge_indices', None)
            f.pop('contour_spans', None)
        f.update(override)
    for f in features.values():
        if 'level' in f and integer(f['level'], 'level', 1) > 4:
            raise ValueError('Feature reporting level must be between 1 and 4')
        if f.get('units', 'px') != 'px':
            raise ValueError('Feature units must be pixels')
        f.setdefault('source', 'model' if 'measurement_route' in f else 'user')
        if f['source'] not in ('user', 'model', 'geometry', 'adapter'):
            raise ValueError('Feature source must be model, geometry, adapter, or user')
        if 'type' not in f:
            route = f.get('measurement_route', 'contour')
            f['type'] = ROUTES.get(route, route)
        if 'selection' not in f and 'raw_edge_indices' in f:
            f['selection'] = {'edge_indices': f['raw_edge_indices']}
        elif 'selection' not in f and 'contour_spans' in f:
            f['selection'] = {'spans': f['contour_spans']}
    req['features'] = list(features.values())
    if 'measurements' not in req:
        req['measurements'] = [{'id': f'{f["id"]}.{op}', 'operator': op, 'feature_refs': [f['id']]}
                               for f in features.values() for op in DEFAULTS.get(f['type'], ['compound'])]
    indexed(req['measurements'], 'measurements')
    return req


def tolerance_evaluation(measured, request):
    out = {'nominal_value': request.get('nominal_value'), 'deviation': None,
           'tolerance': request.get('tolerance'), 'pass_fail': None}
    nominal = request.get('nominal_value')
    tol = request.get('tolerance')
    if nominal is None and tol is None:
        return out
    value = measured['value']
    if isinstance(value, (list, dict, np.ndarray)) or not np.isscalar(value):
        raise ValueError('Numeric tolerance checks require a scalar result')
    if nominal is not None:
        nominal = scalar(nominal, 'nominal_value')
        out['deviation'] = value-nominal
    if tol is not None:
        if not isinstance(tol, dict) or set(tol)-{'lower', 'upper'} or not tol:
            raise ValueError('tolerance requires lower and/or upper absolute value bounds')
        low = scalar(tol['lower'], 'lower') if 'lower' in tol else -np.inf
        high = scalar(tol['upper'], 'upper') if 'upper' in tol else np.inf
        if low > high:
            raise ValueError('Tolerance lower bound exceeds upper bound')
        out['pass_fail'] = 'pass' if low <= value <= high else 'fail'
    return out


def evaluate(contours, request):
    """Return JSON-safe results. Structural request errors raise ValueError;
    local selection/fit/measurement failures become explicit result records.
    """
    if not isinstance(contours, Contours):
        contours = Contours.from_dict(contours)
    req = normalize(request, contours)
    eligible, exclusion = corner_exclusions(
        contours, req.get('corner_exclusion_px', 1.0))
    definitions = indexed(req['features'], 'features')
    geometries, records, active = {}, {}, set()

    def build(ident):
        if ident in records:
            return geometries.get(ident)
        if ident not in definitions:
            raise ValueError(f'Unknown feature reference: {ident}')
        if ident in active:
            raise ValueError('Cyclic compound feature membership')
        f = definitions[ident]
        record = {k: deepcopy(f[k]) for k in (
            'id', 'type', 'source', 'semantic', 'semantic_id', 'confidence', 'parent', 'level',
            'label_level', 'base_instance_ids', 'confidence_calibrated', 'classification_policy',
            'measurement_route', 'boundary_search_regions', 'adjacent_feature_ids',
            'recognition_tolerance', 'fit_evidence', 'rule_evidence',
        ) if k in f}
        record['model_status'] = f.get('status')
        record['selection_confidence'] = f.get('confidence')
        abstained = f.get('status') == 'abstain' and not (f.get('source') == 'user' and f.get('accept') is True)
        if abstained and ('selection' not in f or f['type'] in COMPOUNDS):
            records[ident] = {**record, 'status': 'abstain', 'reason': 'Model selection requires explicit user acceptance'}
            return None
        active.add(ident)
        try:
            if f['type'] == 'point':
                point = np.asarray(f.get('point'), float)
                if point.shape != (2,) or not np.isfinite(point).all():
                    raise ValueError('Point feature requires one finite [x, y] coordinate')
                g = Geometry(ident, 'point')
                g.values = {'center': point, 'point': point, 'method': 'explicit_point'}
                record.update(support=[[point]], closed=[False])
            elif f['type'] in COMPOUNDS:
                if not isinstance(f.get('members'), list) or not f['members']:
                    raise ValueError('Compound geometry needs member feature IDs')
                if len(set(f['members'])) != len(f['members']):
                    raise ValueError('Duplicate compound members')
                members = [build(member) for member in f['members']]
                if any(m is None for m in members):
                    raise ValueError('Compound member unavailable')
                g = Geometry(ident, f['type'], members=members)
                g.values = {'method': 'explicit_member_composition'}
                record['members'] = f['members']
            else:
                if 'selection' not in f:
                    raise ValueError('Feature requires observed contour selection')
                runs, closed, ids = select(contours, f['selection'])
                originally_closed_circle = (
                    f['type'] == 'circle' and len(runs) == 1 and closed == [True]
                )
                record.update(edge_indices=ids, support=runs, closed=closed)
                if 'raw_edge_indices' in f and 'contour_spans' in f:
                    _, _, span_ids = select(contours, {'spans': f['contour_spans']})
                    if not np.array_equal(ids, span_ids):
                        raise ValueError('Model raw edge indices and contour spans disagree')
                if f.get('label_level') == 'compound' and f.get('source') in ('model', 'geometry'):
                    members = f.get('base_instance_ids')
                    if not isinstance(members, list) or not members or not all(isinstance(m, str) for m in members):
                        raise ValueError('Model compound requires base_instance_ids')
                    if len(set(members)) != len(members):
                        raise ValueError('Duplicate model compound base members')
                    selections = []
                    for member in members:
                        base = definitions.get(member)
                        if base is None or base.get('label_level') != 'base':
                            raise ValueError('Model compound references an unknown base instance')
                        selections.append(select(contours, base['selection'])[2])
                    if not np.array_equal(ids, np.unique(np.concatenate(selections))):
                        raise ValueError('Model compound support must equal the union of its base instances')
                if abstained:
                    record.update(status='abstain', reason='Model selection requires explicit user acceptance')
                    records[ident] = jsonable(record)
                    return None
                retained = ids[eligible[ids]]
                record.update(
                    selected_edge_indices=ids,
                    excluded_edge_indices=ids[~eligible[ids]],
                    edge_indices=retained,
                )
                if len(retained) != len(ids):
                    # Clear stale support even when exclusion removes everything.
                    record.update(support=[], closed=[])
                    if not len(retained):
                        raise ValueError('Corner exclusion leaves no complete observed edges')
                    runs, closed, _ = select(contours, {'edge_indices': retained})
                    record.update(
                        support=runs,
                        closed=closed,
                        measurement_note=(
                            'Only retained support is measured; corner gaps are not reconstructed.'
                        ),
                    )
                g = fit_geometry(
                    Geometry(ident, f['type'], runs, closed),
                    allow_circle_gaps=originally_closed_circle and len(retained) != len(ids),
                )
                if f['type'] == 'repeated_profile':
                    record['measurement_note'] = ('Length is retained observed support length. Pitch requires an explicit request. The thread operator measures two opposing '
                                                  'sides, and the tooth_profile operator toothed runs, using original selected observations, '
                                                  'including corners. Generic repetition requires one side and explicit direction/prominence.')
            if 'max_rms_residual' in f:
                limit = scalar(f['max_rms_residual'], 'max_rms_residual', True)
                if 'rms_residual' not in g.values or g.values['rms_residual'] > limit:
                    raise ValueError('Geometry exceeds requested RMS fit residual limit')
            geometries[ident] = g
            record.update(status='fitted', geometry=g.values)
            record['display_center'] = (g.values.get('center', g.points.mean(0))
                                        if g.runs else g.values.get('center'))
            ideal = idealized_support(g)
            if ideal and any(len(run) for run in ideal):
                record['ideal_support'] = ideal
                record['deformation_max'] = float(max(
                    np.linalg.norm(observed-target, axis=1).max(initial=0.)
                    for observed, target in zip(g.runs, ideal)))
        except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
            record.update(status='fit_failed', reason=str(error))
        finally:
            active.remove(ident)
        records[ident] = jsonable(record)
        return geometries.get(ident)

    for ident in definitions:
        build(ident)
    measurements = []
    for m in req['measurements']:
        base = {'id': m['id'], 'operator': m.get('operator'), 'feature_refs': m.get('feature_refs', []),
                'status': 'failed', 'value': None, 'pass_fail': None,
                'uncertainty': None, 'uncertainty_status': 'not_evaluated'}
        try:
            if m.get('operator') not in OPERATORS:
                raise ValueError(f'Unsupported operator: {m.get("operator")}')
            refs = m.get('feature_refs')
            if not isinstance(refs, list) or not refs or not all(isinstance(x, str) for x in refs):
                raise ValueError('feature_refs must be a nonempty list of IDs')
            if any(ref not in records or records[ref]['status'] != 'fitted' for ref in refs):
                base['status'] = 'unavailable'
                raise ValueError('Referenced feature is unknown, abstained, or failed fitting')
            params = m.get('params', {})
            if not isinstance(params, dict):
                raise ValueError('params must be an object')
            if m['operator'] == 'compound' and len(refs) == 1:
                params = {**definitions[refs[0]].get('params', {}), **params}
            if m['operator'] == 'distance_construction' and params.get('datum_feature_ref'):
                datum_ref = params['datum_feature_ref']
                datum = geometries.get(datum_ref)
                if datum is None or datum.kind != 'line' or records[datum_ref]['status'] != 'fitted':
                    raise ValueError('Section datum must reference a fitted line')
                params = {**params, 'direction_mode': 'custom', 'direction': datum.values['normal']}
            selected_geometries = [geometries[r] for r in refs]
            if m['operator'] in ('thread', 'tooth_profile'):
                # Thread and tooth turns require original observations, including corners.
                # Do not bridge the excluded gaps used for primitive line fits.
                selected_geometries = []
                for ref in refs:
                    runs, closed, _ = select(contours, definitions[ref]['selection'])
                    # Radii come only from arc primitives the model reported.
                    arcs = [geometries[b] for b in definitions[ref].get('base_instance_ids', [])
                            if b in geometries and geometries[b].kind == 'arc']
                    selected_geometries.append(Geometry(ref, definitions[ref]['type'], runs, closed, members=arcs))
                if m['operator'] == 'tooth_profile':
                    side = material_side(contours, selected_geometries[0].runs)
                    params = {**({'material_side': side} if side else {}), **params}
            if m['operator'] == 'length' and all(geometries[r].kind == 'line' for r in refs):
                # Edge length runs corner to corner, so corner exclusion must not shorten it.
                selected_geometries = [fit_geometry(Geometry(r, 'line', *select(contours, definitions[r]['selection'])[:2]))
                                       for r in refs]
            if m['operator'] == 'slot':
                # Flats and end arcs are the slot's own base members.
                selected_geometries = [Geometry(g.id, g.kind, g.runs, g.closed, g.values, members=[
                    geometries[b] for b in definitions[g.id].get('base_instance_ids', [])
                    if b in geometries and geometries[b].kind in ('line', 'arc')]) for g in selected_geometries]
            measured = jsonable(measure(m['operator'], selected_geometries, params))
            if m['operator'] in ('thread', 'tooth_profile') or (m['operator'] == 'length' and selected_geometries[0].kind == 'line'):
                measured['support_policy'] = 'original_selected_observations_including_corners'
            if m['operator'] == 'distance_construction' and params.get('datum_feature_ref'):
                measured['definition']['datum_feature_ref'] = params['datum_feature_ref']
            checked = tolerance_evaluation(measured, m)
            base.update(measured, **checked, status='measured')
        except (ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as error:
            base['reason'] = str(error)
        measurements.append(jsonable(base))
    report = {'schema_version': 1, 'units': 'px', 'coordinate_system': 'image',
              'coordinate_hash': contours.coordinate_hash,
              'corner_exclusion': exclusion,
              'features': [records[i] for i in definitions], 'measurements': measurements,
              'summary': {'measured': sum(m['status'] == 'measured' for m in measurements),
                          'failed_or_unavailable': sum(m['status'] != 'measured' for m in measurements),
                          'tolerance_failures': sum(m['pass_fail'] == 'fail' for m in measurements)}}
    if req.get('label_version') in ('primitives-compounds-v1','primitives-compounds-v2'):
        report['label_version'] = req['label_version']
        report['relations'] = deepcopy(req.get('relations', []))
    json.dumps(report, allow_nan=False)
    return report
