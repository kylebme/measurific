"""Explicit topology and stable edge identity, independent of semantic labels."""
from dataclasses import dataclass
import hashlib
import numpy as np
from shapely.geometry import Polygon


def integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


@dataclass
class Contours:
    """Pixel centers: +x right, +y down. Closed paths omit the repeated endpoint."""
    xy: np.ndarray
    offsets: np.ndarray
    closed: np.ndarray | None = None

    def __post_init__(self):
        self.xy = np.asarray(self.xy, dtype=np.float64).copy()
        offsets = np.asarray(self.offsets)
        if offsets.dtype.kind not in 'iu' or offsets.ndim != 1:
            raise ValueError('offsets must be a one-dimensional integer array')
        self.offsets = offsets.astype(np.int64, copy=True)
        if self.xy.ndim != 2 or self.xy.shape[1] != 2 or not np.isfinite(self.xy).all():
            raise ValueError('xy must contain finite [N, 2] pixel coordinates')
        if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(self.xy) or (np.diff(offsets) < 2).any():
            raise ValueError('offsets must partition xy into paths of at least two vertices')
        self.closed = np.ones(len(offsets)-1, bool) if self.closed is None else np.asarray(self.closed)
        if self.closed.dtype.kind != 'b' or self.closed.shape != (len(offsets)-1,):
            raise ValueError('closed must contain one boolean per path')
        for i in range(len(self.closed)):
            q = self.path(i)
            if (np.linalg.norm(np.diff(q, axis=0), axis=1) <= 1e-12).any():
                raise ValueError('Consecutive duplicate vertices are not allowed')
            if self.closed[i] and (len(q) < 3 or np.linalg.norm(q[0]-q[-1]) <= 1e-12):
                raise ValueError('Closed paths need >=3 vertices and no duplicate closing vertex')
            if self.closed[i] and (not Polygon(q).is_valid or Polygon(q).area <= 1e-12):
                raise ValueError('Closed paths must be simple and have nonzero area')

    def path(self, index):
        index = integer(index, 'path index')
        if index >= len(self.closed):
            raise ValueError('Path index out of range')
        return self.xy[self.offsets[index]:self.offsets[index+1]]

    @property
    def coordinate_hash(self):
        # Matches contour-model's pixel-valued raw-array hash, before any reindexing.
        return hashlib.sha256(self.xy.tobytes()+self.offsets.tobytes()).hexdigest()

    def to_dict(self):
        return {'xy': self.xy.tolist(), 'offsets': self.offsets.tolist(),
                'closed': self.closed.tolist(), 'units': 'px', 'coordinate_system': 'image'}

    @classmethod
    def from_dict(cls, value):
        if value.get('units', 'px') != 'px' or value.get('coordinate_system', 'image') != 'image':
            raise ValueError('Only image coordinates (+y down) in pixels are supported')
        return cls(value['xy'], value['offsets'], value.get('closed'))

    def model_arrays(self):
        """Legacy model key is named mm even when predict(..., units='px') is used."""
        if not self.closed.all():
            raise ValueError('The current model requires closed rings')
        return {'actual_xy_mm': self.xy.copy(), 'ring_offsets': self.offsets.copy()}


def select(contours, spec):
    """Return connected support runs; never join disconnected selected edges."""
    n = len(contours.xy)
    mask = np.zeros(n, bool)
    forms = [k for k in ('path', 'spans', 'edge_indices', 'region', 'contour_interval') if k in spec]
    if len(forms) != 1:
        raise ValueError('Selection requires exactly one of path, spans, edge_indices, region, contour_interval')
    interval_run = None
    if 'path' in spec:
        i = integer(spec['path'], 'path')
        contours.path(i)
        mask[contours.offsets[i]:contours.offsets[i+1]-(not contours.closed[i])] = True
    elif 'spans' in spec:
        for span in spec['spans']:
            i = integer(span['ring_index'], 'ring_index')
            q = contours.path(i)
            start = integer(span['start_edge'], 'start_edge')
            count = integer(span['edge_count'], 'edge_count', 1)
            size = len(q)
            if start >= size or count > size or (not contours.closed[i] and start+count > size-1):
                raise ValueError('Span exceeds path support')
            ids = contours.offsets[i]+(start+np.arange(count)) % size
            if mask[ids].any():
                raise ValueError('Overlapping spans')
            mask[ids] = True
    elif 'edge_indices' in spec:
        ids = np.asarray(spec['edge_indices'])
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= n).any():
            raise ValueError('Invalid or duplicate edge indices')
        mask[ids] = True
    elif 'region' in spec:
        region = spec['region']
        if 'box' in region:
            x0, y0, x1, y1 = np.asarray(region['box'], float)
            if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
                raise ValueError('Invalid region box')
            inside = lambda q: (q[:, 0] >= x0) & (q[:, 0] <= x1) & (q[:, 1] >= y0) & (q[:, 1] <= y1)
        elif 'circle' in region:
            x, y, r = np.asarray(region['circle'], float)
            if not np.isfinite([x, y, r]).all() or r <= 0:
                raise ValueError('Invalid region circle')
            inside = lambda q: np.linalg.norm(q-[x, y], axis=1) <= r
        else:
            raise ValueError('Region requires box or circle')
        for i, (lo, hi) in enumerate(zip(contours.offsets[:-1], contours.offsets[1:])):
            q = contours.path(i)
            hit = inside(q)
            mask[lo:hi] = hit & np.roll(hit, -1)  # both endpoints must be in region
            if not contours.closed[i]:
                mask[hi-1] = False
    else:
        interval = spec['contour_interval']
        if not isinstance(interval, dict):
            raise ValueError('contour_interval must be an object')
        i = integer(interval.get('ring_index'), 'ring_index')
        q = contours.path(i)
        start = interval.get('start', {})
        end = interval.get('end', {})
        start_edge = integer(start.get('edge'), 'start edge')
        end_edge = integer(end.get('edge'), 'end edge')
        edge_count = len(q) if contours.closed[i] else len(q)-1
        if start_edge >= edge_count or end_edge >= edge_count:
            raise ValueError('Contour interval edge is out of range')
        start_t = float(start.get('t'))
        end_t = float(end.get('t'))
        direction = interval.get('direction', 1)
        if isinstance(direction, (bool, np.bool_)) or direction not in (-1, 1):
            raise ValueError('Contour interval direction must be 1 or -1')
        if not np.isfinite([start_t, end_t]).all() or not (0 <= start_t <= 1 and 0 <= end_t <= 1):
            raise ValueError('Contour interval fractions must be between zero and one')
        direct = start_edge == end_edge and ((direction == 1 and end_t > start_t) or
                                              (direction == -1 and end_t < start_t))
        if direct:
            edges = np.array([start_edge])
        else:
            delta = direction*(end_edge-start_edge)
            distance = delta % len(q) if contours.closed[i] else delta
            invalid_same = distance == 0 and ((direction == 1 and end_t <= start_t) or
                                               (direction == -1 and end_t >= start_t))
            if distance < 0 or (not contours.closed[i] and invalid_same):
                raise ValueError('End point must follow start point on an open contour')
            edges = (start_edge+direction*np.arange(distance+1)) % len(q)
        start_point = q[start_edge] + start_t*(q[(start_edge+1) % len(q)]-q[start_edge])
        end_point = q[end_edge] + end_t*(q[(end_edge+1) % len(q)]-q[end_edge])
        middle_indices = (edges[:-1]+1) % len(q) if direction == 1 else edges[:-1]
        middle = q[middle_indices] if len(edges) > 1 else np.empty((0, 2))
        interval_run = np.vstack([start_point, middle, end_point])
        keep = np.r_[True, np.linalg.norm(np.diff(interval_run, axis=0), axis=1) > 1e-12]
        interval_run = interval_run[keep]
        if len(interval_run) < 2:
            raise ValueError('Contour interval must have nonzero length')
        lo = contours.offsets[i]
        mask[lo+edges] = True
    if interval_run is not None:
        return [interval_run], [False], np.flatnonzero(mask)
    runs, closure = [], []
    for i, (lo, hi) in enumerate(zip(contours.offsets[:-1], contours.offsets[1:])):
        m = mask[lo:hi]
        if not contours.closed[i] and m[-1]:
            raise ValueError('The last vertex of an open path is not an edge')
        if not m.any():
            continue
        if m.all():
            runs.append(contours.path(i).copy())
            closure.append(True)
            continue
        starts = np.flatnonzero(m & ~np.roll(m, 1))
        ends = np.flatnonzero(m & ~np.roll(m, -1))
        # Each start is paired with the next end cyclically. Include the final
        # edge's endpoint without walking every selected edge in Python.
        matched = ends[np.searchsorted(ends, starts) % len(ends)]
        for start, end in zip(starts, matched):
            count = (end-start) % len(m)+2
            ids = (start+np.arange(count)) % len(m)
            runs.append(contours.xy[lo+ids])
            closure.append(False)
    if not runs:
        raise ValueError('Selection contains no complete observed edges')
    return runs, closure, np.flatnonzero(mask)
