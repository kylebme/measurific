"""Coarse-model-seeded PAE edges with the ordered-ring contract used by the image adapter.

The coarse edge model (512 x 512 frame, about max(H, W)/512 px boundary error) decides
which rings exist and roughly where they are. PAE (``pae.detect_seeded``) then locates the
edge in a band around each coarse ring from the original intensities. Earlier PAE
adapters seeded PAE with a threshold, which merged, split or lost rings under uneven
backlight, glare, contaminants and blur; the coarse model replaces only that step.

Every ring vertex is an unmodified PAE observation, ordered along its coarse ring. Rings
whose PAE support has gaps or self-intersects are reported as extraction failures rather
than bridged with coarse geometry: PAE suits sharp edges, and the full edge model (SDF)
is the alternative for blurred edges and very fine detail.
"""
import time
from pathlib import Path

import numpy as np

COARSE_FORMAT = 'edgesr-coarse-pae-v1'


class PAEExtractionError(ValueError):
    """PAE support failure, with the framed-image points to highlight."""

    def __init__(self, message, points=(), label=None):
        super().__init__(message)
        self.points = np.asarray(points, float).reshape(-1, 2)
        self.label = label or message
        self.coarse_ring = None


def _signed_area(q):
    return np.sum(q[:, 0]*np.roll(q[:, 1], -1)-q[:, 1]*np.roll(q[:, 0], -1)) / 2


def _resolve_local_crossings(q, max_run=4, max_extent=3.):
    """Reverse short runs of observations whose outer links strictly cross (local 2-opt).

    Arclength along a rounded coarse ring cannot reliably order the few observations
    within a pixel or two of a sharp corner, where several project onto the same part of
    the ring; a strict crossing of the links around them supplies the order. Only runs of
    at most ``max_run`` observations within ``max_extent`` px are reversed. Nothing is
    moved or invented, and other crossings remain failures.
    """
    q = q.copy()
    n = len(q)

    def cross(u, v):
        return u[..., 0]*v[..., 1]-u[..., 1]*v[..., 0]
    adjusted = 0
    for _ in range(4):
        changed = False
        for run in range(2, min(max_run, n-2)+1):
            a, b, c, d = (np.roll(q, -k, axis=0) for k in (0, 1, run, run+1))
            crossing = ((cross(b-a, c-a)*cross(b-a, d-a) < -1e-12) &
                        (cross(d-c, a-c)*cross(d-c, b-c) < -1e-12))
            for i in np.flatnonzero(crossing):
                inner = (i+1+np.arange(run)) % n
                # Earlier reversals may have changed this run.
                a, b, c, d = q[i], q[inner[0]], q[inner[-1]], q[(i+run+1) % n]
                if (np.ptp(q[inner], axis=0).max() <= max_extent and
                        cross(b-a, c-a)*cross(b-a, d-a) < -1e-12 and cross(d-c, a-c)*cross(d-c, b-c) < -1e-12):
                    q[inner] = q[inner[::-1]]
                    adjusted += 1
                    changed = True
        if not changed:
            break
    return q, adjusted


def coarse_rings(logit, min_area):
    """Closed coarse rings (framed pixel centers) of the part mask, hole flags and parents.

    Outer rings have positive shoelace area and holes negative, so in both the background
    side of a ring is on the (ty, -tx) side of its tangent. Rings below the minimum areas
    are skipped together with everything inside them. Parents precede their children.
    """
    import cv2
    from scipy.ndimage import gaussian_filter1d
    mask = (logit > 0).astype(np.uint8)
    raster, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return [], [], []
    hierarchy = hierarchy[0]
    depth = np.zeros(len(raster), int)
    skipped = np.zeros(len(raster), bool)
    order = []
    # Parents come before children in a depth-first walk of the hierarchy.
    stack = [i for i in range(len(raster)) if hierarchy[i, 3] < 0][::-1]
    while stack:
        i = stack.pop()
        parent = hierarchy[i, 3]
        if parent >= 0:
            depth[i] = depth[parent] + 1
        hole = bool(depth[i] % 2)
        # findContours traces boundary pixels of the region, i.e. half a pixel inside
        # outlines and outside holes; area thresholds use the pixel count.
        area = cv2.contourArea(raster[i]) + (-1 if hole else 1) * len(raster[i]) / 2
        skipped[i] = (parent >= 0 and skipped[parent]) or len(raster[i]) < 3 or area < min_area[int(hole)]
        if not skipped[i]:
            order.append(i)
        child = hierarchy[i, 2]
        children = []
        while child >= 0:
            children.append(child)
            child = hierarchy[child, 0]
        stack.extend(children[::-1])
    index = {i: k for k, i in enumerate(order)}
    rings, holes, parents = [], [], []
    for i in order:
        q = raster[i][:, 0].astype(np.float64)
        hole = bool(depth[i] % 2)
        # A light smoothing removes raster stair-steps from the ordering guide only.
        if len(q) >= 8:
            q = gaussian_filter1d(q, 1.5, axis=0, mode='wrap')
        if (_signed_area(q) < 0) != hole:
            q = q[::-1]
        rings.append(q)
        holes.append(hole)
        parents.append(index.get(hierarchy[i, 3], -1))
    return rings, holes, parents


def coarse_band(logit, radius):
    """Pixels within ``radius`` px of the coarse part boundary."""
    import cv2
    mask = (logit > 0).astype(np.uint8)
    edge = mask ^ cv2.erode(mask, np.ones((3, 3), np.uint8))
    edge |= cv2.dilate(mask, np.ones((3, 3), np.uint8)) ^ mask
    distance = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return distance <= radius


def assign_observations(points, normals, rings, max_distance, min_alignment=.35):
    """Nearest coarse ring segment whose background side agrees with each PAE normal.

    PAE normals point toward greater intensity, i.e. into the backlight. Returns per
    observation the ring (-1 if none), arclength along it and signed offset (+ toward the
    background).
    """
    from scipy.spatial import cKDTree
    starts = np.concatenate(rings)
    ends = np.concatenate([np.roll(q, -1, axis=0) for q in rings])
    owner = np.repeat(np.arange(len(rings)), [len(q) for q in rings])
    segment = ends-starts
    length = np.maximum(np.linalg.norm(segment, axis=1), 1e-9)
    arc = np.concatenate([np.r_[0, np.cumsum(np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1))[:-1]]
                          for q in rings])
    tangent = segment / length[:, None]
    background = np.c_[tangent[:, 1], -tangent[:, 0]]
    k = min(24, len(starts))
    _, candidates = cKDTree((starts+ends)/2).query(points, k=k)
    candidates = candidates.reshape(len(points), k)
    t = np.clip(np.sum((points[:, None]-starts[candidates])*segment[candidates], axis=2)
                / length[candidates]**2, 0, 1)
    projection = starts[candidates]+t[..., None]*segment[candidates]
    distance = np.linalg.norm(projection-points[:, None], axis=2)
    alignment = np.sum(background[candidates]*normals[:, None], axis=2)
    score = np.where((alignment >= min_alignment) & (distance <= max_distance), distance, np.inf)
    choice = np.argmin(score, axis=1)
    rows = np.arange(len(points))
    best = candidates[rows, choice]
    ring = np.where(np.isfinite(score[rows, choice]), owner[best], -1)
    s = arc[best]+t[rows, choice]*length[best]
    offset = np.sum((points-projection[rows, choice])*background[best], axis=1)
    return ring, s, offset


def order_ring(points, s, offset, max_gap, window=7, tolerance=1.):
    """Order one ring's observations by arclength and drop off-track outliers.

    An observation whose offset from the coarse ring differs from the running median of
    its neighbours by more than ``tolerance`` px belongs to another edge (a reflection, a
    lip, texture) or to a corner where PAE's local model fails.
    """
    from scipy.ndimage import median_filter
    order = np.argsort(s, kind='stable')
    q, offset = points[order], offset[order]
    if len(q) >= window:
        keep = np.abs(offset-median_filter(offset, size=window, mode='wrap')) <= tolerance
        q = q[keep]
    if len(q):
        q = q[np.r_[True, np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-6]]
    if len(q) < 8:
        raise PAEExtractionError('Ring has insufficient PAE edge support', points,
                                 'Insufficient PAE edge support')
    gaps = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
    if gaps.max() > max_gap:
        i = int(np.argmax(gaps))
        raise PAEExtractionError(f'Ring has missing PAE edge support ({gaps.max():.3g} px gap, limit {max_gap:g} px)',
                                 [q[i], q[(i+1) % len(q)]], 'Missing PAE edge support')
    q, adjusted = _resolve_local_crossings(q)
    from shapely.geometry import Polygon
    if not Polygon(q).is_valid:
        raise PAEExtractionError('PAE ring ordering is self-intersecting', q, 'PAE self-intersection')
    return q, adjusted


class PAEEdgeModel:
    """Coarse edge model + PAE, interchangeable with the SDF/CFI models' ``rings``."""

    method = 'coarse_model_pae_rings'
    min_support = .2    # PAE observations per px of coarse ring below which it is no edge

    def __init__(self, coarse_directory, device='cpu', threads=None, smoothing=1, min_area=(50.0, 8.0)):
        from edgesr.coarse_runtime import CoarseRuntime
        from edgesr.runtime import select_providers
        self.path = Path(coarse_directory)
        self.coarse = CoarseRuntime(self.path, select_providers(device), threads)
        self.providers = self.coarse.session.get_providers()
        self.smoothing = int(smoothing)
        self.defaults = tuple(float(a) for a in min_area)

    def describe(self):
        return {'format': COARSE_FORMAT, 'directory': str(self.path), 'providers': self.providers,
                'pae_smoothing': self.smoothing,
                'source_checkpoint_sha256': self.coarse.meta.get('provenance', {}).get('source_checkpoint_sha256')}

    def warmup(self):
        """Compile PAE's Numba kernels so the first analysis is not slowed by them."""
        from pae import warmup
        warmup()

    @staticmethod
    def search_radius(shape):
        """Band half-width: about 1.5 coarse pixels plus PAE's own window."""
        return int(min(32, np.ceil(1.5 * max(shape) / 512) + 2))

    def rings(self, image, max_value=None, valid_mask=None, min_object_area=None, min_hole_area=None,
              apron=16, max_gap=5.):
        import cv2
        from edgesr.runtime import TRUST_MARGIN, backlit_frame, nest_rings, reading
        from pae import detect_seeded
        started = time.perf_counter()
        times = {}
        img = reading(image, max_value)
        valid = np.ones(img.shape, bool) if valid_mask is None else np.asarray(valid_mask, bool)
        if valid.shape != img.shape or not valid.any():
            raise ValueError('valid_mask must match the image and contain pixels')
        framed, backlight, gain = backlit_frame(img, valid, apron)
        min_area = tuple(float(default if value is None else value)
                         for default, value in zip(self.defaults, (min_object_area, min_hole_area)))

        t = time.perf_counter()
        out = self.coarse(framed)
        logit = self.coarse.full_res_logit(out, framed.shape)
        trusted = cv2.dilate(np.pad(valid, apron).astype(np.uint8), np.ones((2*TRUST_MARGIN+1,)*2, np.uint8)) > 0
        logit[~trusted] = np.minimum(logit[~trusted], -8)
        logit[[0, -1], :] = -8
        logit[:, [0, -1]] = -8
        seeds, holes, parents = coarse_rings(logit, min_area)
        times['coarse'] = time.perf_counter() - t
        base = {'apron': apron, 'min_area': min_area, 'backlight': backlight, 'gain': gain,
                'method': self.method}
        if not seeds:
            return {'rings': [], 'is_hole': [], 'depth': [], 'parent': [], 'dropped_rings': 0,
                    'unsupported_rings': 0, 'lattice_crossings': 0, 'stage_seconds': times,
                    'seconds': time.perf_counter() - started, **base}

        t = time.perf_counter()
        # PAE reads the original intensities in 8-bit DN units (the gain only rescales them).
        source = framed.astype(np.float64) * (255. / gain)
        radius = self.search_radius(img.shape)
        band = coarse_band(logit, radius) & trusted
        part = cv2.erode((logit > 4).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        background = cv2.erode((logit < -4).astype(np.uint8) & trusted.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        background_level = float(np.median(source[background])) if background.any() else backlight * 255.
        part_level = float(np.median(source[part])) if part.any() else 0.
        contrast = max(background_level - part_level, 1.)
        residual = (source - cv2.GaussianBlur(source, (0, 0), 1.))[::2, ::2][background[::2, ::2]]
        noise = 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) if len(residual) else 0.
        # A sharp step's central difference peaks at half the contrast and a blurred one
        # lower; noise maxima on either plateau must stay below the threshold.
        threshold = max(.08 * contrast, 4 * noise, 2.)
        y, x = np.nonzero(band)
        pad = 8
        y0, y1 = max(0, y.min() - pad), min(source.shape[0], y.max() + pad + 1)
        x0, x1 = max(0, x.min() - pad), min(source.shape[1], x.max() + pad + 1)
        edges = detect_seeded(source[y0:y1, x0:x1], threshold, seeds=np.c_[x - x0, y - y0].astype(np.int64),
                              search_radius=0, smoothing=self.smoothing)
        points = np.c_[edges.x + x0, edges.y + y0]
        normals = np.c_[edges.nx, edges.ny]
        finite = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
        # The silhouette edge separates the part from the backlight; weaker steps inside
        # the band are reflections, lips or texture.
        strong = finite & (edges.i1 - edges.i0 >= .35 * contrast)
        points, normals = points[strong], normals[strong]
        times['pae'] = time.perf_counter() - t

        t = time.perf_counter()
        ring, s, offset = (assign_observations(points, normals, seeds, radius + 1.) if len(points)
                           else (np.zeros(0, int), np.zeros(0), np.zeros(0)))
        ordered, kept_holes, adjustments = [], [], 0
        dropped = np.zeros(len(seeds), bool)
        unsupported = 0
        for i, seed in enumerate(seeds):
            mine = ring == i
            perimeter = np.linalg.norm(np.roll(seed, -1, axis=0)-seed, axis=1).sum()
            # As the SDF path's seed support check, but reversed: the coarse model may
            # see a part where there is no edge at all (fill seams, vignetting ramps).
            if mine.sum() < self.min_support * perimeter:
                unsupported += not (parents[i] >= 0 and dropped[parents[i]])
                dropped[i] = True
            if dropped[i] or (parents[i] >= 0 and dropped[parents[i]]):
                dropped[i] = True
                continue
            try:
                q, adjusted = order_ring(points[mine], s[mine], offset[mine], max_gap)
            except PAEExtractionError as error:
                if not len(error.points):
                    error.points = seed
                error.points = error.points - apron + .5
                error.coarse_ring = seed - apron + .5
                raise
            ordered.append(q)
            kept_holes.append(holes[i])
            adjustments += adjusted
        # Edge-model coordinates: pixel i covers [i, i + 1).
        nested = nest_rings([q - apron + .5 for q in ordered], kept_holes, min_area)
        nested['dropped_rings'] += int(dropped.sum())
        nested['unsupported_rings'] += unsupported
        times['order'] = time.perf_counter() - t
        return {**nested, **base, 'lattice_crossings': int(len(points)), 'stage_seconds': times,
                'seconds': time.perf_counter() - started,
                'pae': {'threshold_dn8': round(threshold, 3), 'contrast_dn8': round(contrast, 3),
                        'noise_dn8': round(noise, 3), 'search_radius_px': radius, 'smoothing': self.smoothing,
                        'observations': int(len(points)), 'assigned_observations': int((ring >= 0).sum()),
                        'used_observations': int(sum(len(q) for q in ordered)),
                        'local_ordering_adjustments': adjustments}}
