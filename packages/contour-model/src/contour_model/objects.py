"""Recognition of several objects in one set of contours.

The recognizers take one part: an outline and the holes directly inside it. An image can
show several parts, dust, or a part lying inside another part's hole. Each object is
recognized on its own, then the results are merged back onto the full contour set, so
measurements and relations work across objects.
"""
import hashlib

import numpy as np


def split_objects(contours):
    """Global ring indices of each object, [outline, *holes], largest outline first.

    Nesting is geometric: rings at even containment depth are outlines, and each ring at
    odd depth is a hole of the ring directly containing it."""
    import shapely

    xy = np.asarray(contours['actual_xy_mm'], np.float64)
    offsets = np.asarray(contours['ring_offsets'], np.int64)
    rings = [xy[lo:hi] for lo, hi in zip(offsets[:-1], offsets[1:])]
    area = np.array([abs(np.sum(q[:, 0]*np.roll(q[:, 1], -1)-np.roll(q[:, 0], -1)*q[:, 1]))/2 for q in rings])
    order = np.argsort(-area, kind='stable')
    first = np.array([q[0] for q in rings])
    parent = np.full(len(rings), -1)
    # Containers are larger, so scanning largest first leaves the smallest container.
    for j in order:
        q = rings[j]
        candidates = np.flatnonzero((area < area[j]) & (first > q.min(0)).all(1) & (first < q.max(0)).all(1))
        if len(candidates):
            polygon = shapely.Polygon(q)
            shapely.prepare(polygon)
            parent[candidates[shapely.contains_xy(polygon, first[candidates, 0], first[candidates, 1])]] = j
    depth = np.zeros(len(rings), int)
    for i in order:
        depth[i] = depth[parent[i]]+1 if parent[i] >= 0 else 0
    return [[int(i), *[int(j) for j in order if parent[j] == i and depth[j] % 2]]
            for i in order if depth[i] % 2 == 0]


def predict_objects(predict_one, contours):
    """Run ``predict_one(contours) -> (result, arrays)`` once per object and merge.

    A single object is passed through unchanged. Otherwise feature ids are renumbered
    across objects, point and ring indices refer to the full contour set, each feature's
    ``parent`` names its object, and ``result['objects']`` lists the objects' rings."""
    objects = split_objects(contours)
    if len(objects) <= 1:
        return predict_one(contours)
    xy = np.asarray(contours['actual_xy_mm'], np.float64)
    offsets = np.asarray(contours['ring_offsets'], np.int64)
    merged, arrays, features, compounds, relations, summaries = None, {}, [], [], [], []
    unselected = 0.
    for k, members in enumerate(objects):
        name = f'object_{k+1}'
        to_global = np.concatenate([np.arange(offsets[r], offsets[r+1]) for r in members])
        sub_offsets = np.r_[0, np.cumsum([offsets[r+1]-offsets[r] for r in members])]
        result, sub_arrays = predict_one({'actual_xy_mm': xy[to_global], 'ring_offsets': sub_offsets})
        merged = merged or result
        ids = {f['id']: f'base_{len(features)+i+1:05d}' for i, f in enumerate(result['features'])}
        ids.update({f['id']: f'compound_{len(compounds)+i+1:04d}' for i, f in enumerate(result['compound_features'])})

        def ring(value):
            return int(members[value])

        def points(values, sort=False):
            mapped = to_global[np.asarray(values, int)]
            return sorted(mapped.tolist()) if sort else mapped.tolist()

        for f in result['features']+result['compound_features']:
            target = compounds if f.get('label_level') == 'compound' else features
            f = dict(f, id=ids[f['id']], parent=name, object_index=k)
            f['raw_edge_indices'] = points(f['raw_edge_indices'], sort=f.get('label_level') == 'compound')
            if 'defect_edge_indices' in f:
                f['defect_edge_indices'] = points(f['defect_edge_indices'])
            if 'contour_spans' in f:
                f['contour_spans'] = [dict(span, ring_index=ring(span['ring_index'])) for span in f['contour_spans']]
            if 'boundary_search_regions' in f:
                f['boundary_search_regions'] = [dict(region, ring_index=ring(region['ring_index']),
                                                     raw_edge_indices=points(region['raw_edge_indices']))
                                                for region in f['boundary_search_regions']]
            if 'base_instance_ids' in f:
                f['base_instance_ids'] = [ids[v] for v in f['base_instance_ids']]
            target.append(f)
        for relation in result.get('relations', []):
            relation = dict(relation, feature_a=ids[relation['feature_a']], feature_b=ids[relation['feature_b']])
            if 'ring_index' in relation:
                relation['ring_index'] = ring(relation['ring_index'])
            relations.append(relation)
        for key, values in (sub_arrays or {}).items():
            values = np.asarray(values)
            if values.shape[:1] == (len(to_global),):
                arrays.setdefault(key, np.zeros((len(xy),)+values.shape[1:], values.dtype))[to_global] = values
        q = xy[to_global]
        summaries.append({'id': name, 'ring_indices': [int(r) for r in members],
                          'bounds': [q.min(0).tolist(), q.max(0).tolist()]})
        unselected += result.get('unselected_edge_fraction', 0.)*len(to_global)
    hole = np.zeros(len(offsets)-1, bool)
    for members in objects:
        hole[members[1:]] = True
    merged = dict(merged, features=features, compound_features=compounds, relations=relations, objects=summaries,
                  coordinate_hash=hashlib.sha256(xy.tobytes()+offsets.tobytes()).hexdigest(),
                  part={'id': 'part', 'ring_count': len(offsets)-1, 'ring_is_hole': hole.tolist(),
                        'bounds': [xy.min(0).tolist(), xy.max(0).tolist()]},
                  unselected_edge_fraction=unselected/len(xy))
    return merged, arrays
