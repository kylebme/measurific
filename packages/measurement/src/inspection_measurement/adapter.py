"""Closed silhouette contours from an image with a sub-pixel edge model.

The deep learning edge model (``edge-model``, run through ONNX Runtime by
``edgesr.runtime``) predicts sub-pixel occupancy or signed distance near the silhouette
and traces it into ordered closed rings whose vertices are crossings of the boundary with
the 1/16 px lattice. ``PAEEdgeModel`` (``pae_edge``) instead seeds PAE with the coarse
edge model and returns PAE observations. This adapter keeps the vertices about
``point_spacing`` px apart (never moving or interpolating them), shifts them to this
package's integer pixel centers and returns ``Contours`` with outer rings of positive
shoelace area and holes negative.
"""
import numpy as np
from shapely.geometry import Polygon
from .contours import Contours
from .pae_edge import PAEExtractionError

_MODELS = {}


class ContourExtractionError(ValueError):
    """Extraction failure with compact geometry suitable for a GUI overlay."""

    def __init__(self, message, diagnostic=None):
        super().__init__(message)
        self.diagnostic = diagnostic or {}


def _sample_points(points, limit=5000):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) > limit:
        points = points[np.linspace(0, len(points)-1, limit, dtype=int)]
    return points.tolist()


def _extraction_failure(message, *, rings=(), issue_points=(), issue_label=None):
    edges = _sample_points(np.concatenate(rings)) if len(rings) else []
    issue = None
    points = np.asarray(issue_points, dtype=float)
    if points.size:
        points = points.reshape(-1, 2)
        low, high = points.min(axis=0), points.max(axis=0)
        margin = 12.
        issue = {
            'label': issue_label or message,
            'bounds': [float(low[0]-margin), float(low[1]-margin),
                       float(high[0]+margin), float(high[1]+margin)],
        }
    raise ContourExtractionError(message, {'edges': edges, 'guides': [], 'issue': issue})


def load_edge_model(path=None, device='cpu', threads=None):
    """Cached CFI/SDF ONNX model or SDF torch checkpoint; default is the CFI export."""
    from edgesr.runtime import DEFAULT_MODEL, EdgeModel
    key = (str(path or DEFAULT_MODEL), device, threads)
    if key not in _MODELS:
        import json
        from pathlib import Path
        source = Path(key[0])
        is_sdf = source.suffix == '.pt' or (
            source.is_dir() and json.loads((source / 'model.json').read_text()).get('format') == 'edgesr-sdf-onnx-v1')
        if is_sdf:
            from .sdf_edge import SDFEdgeModel
            _MODELS[key] = SDFEdgeModel(source, device=device, threads=threads)
        else:
            _MODELS[key] = EdgeModel(source, device=device, threads=threads)
    return _MODELS[key]


def load_pae_edge_model(path, device='cpu', threads=None):
    """Cached coarse model + PAE. ``path`` is a coarse ONNX export or an edge model export
    whose ``model.json`` names its ``coarse_model``, so PAE shares that model's coarse seed."""
    import json
    from pathlib import Path
    source = Path(path)
    if not (source / 'coarse.json').exists():
        source = (source / json.loads((source / 'model.json').read_text())['coarse_model']).resolve()
    key = ('pae', str(source), device, threads)
    if key not in _MODELS:
        from .pae_edge import PAEEdgeModel
        _MODELS[key] = PAEEdgeModel(source, device=device, threads=threads)
    return _MODELS[key]


def _signed_area(q):
    return np.sum(q[:, 0]*np.roll(q[:, 1], -1)-q[:, 1]*np.roll(q[:, 0], -1)) / 2


def _thin(q, spacing):
    """Every crossing that starts a new ``spacing`` interval of arc length."""
    step = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
    s = np.r_[0, np.cumsum(step[:-1])]
    spacing = min(spacing, step.sum()/8)
    keep = np.r_[True, np.diff(np.floor(s/spacing)) > 0]
    # The seam must not leave the last vertex crowding the first.
    last = np.flatnonzero(keep)[-1]
    if keep.sum() > 3 and step.sum()-s[last] < spacing/2:
        keep[last] = False
    return q[keep]


def _simplify(q, spacing):
    """Thin a traced ring, halving the spacing wherever that would make it non-simple."""
    q = q[np.linalg.norm(q-np.roll(q, 1, axis=0), axis=1) > 1e-9]
    while spacing >= 1/64:
        thinned = _thin(q, spacing)
        if len(thinned) >= 3 and Polygon(thinned).is_valid:
            return thinned, spacing
        spacing /= 2
    return q, 0.


def contours_from_image(image, *, polarity='dark-on-light', min_object_area=None,
                        min_hole_area=None, point_spacing=1.1, valid_mask=None,
                        max_value=None, model=None):
    """Return closed silhouette contours and extraction diagnostics.

    Pixel centers are integers (+x right, +y down). Parts cut by the image border or
    by ``valid_mask`` close along it. Regions smaller than ``min_object_area`` /
    ``min_hole_area`` px^2 (edge-model defaults 50 / 8) are dropped with everything
    inside them. ``point_spacing`` (px) thins the 1/16 px crossings (default 1.1, the synthetic
    generator's contour density); 0 keeps them all.
    ``max_value`` is the sensor full scale for data stored in a wider type (e.g. 4095).
    ``model`` is a loaded ``edgesr.runtime.EdgeModel`` (default: ``load_edge_model()``).
    """
    gray = np.asarray(image)
    if gray.ndim != 2 or gray.dtype.kind != 'u' or gray.dtype.itemsize not in (1, 2):
        raise ValueError('Image adapter requires an 8-bit or 16-bit unsigned grayscale image')
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask)
        if valid_mask.shape != gray.shape or valid_mask.dtype.kind != 'b':
            raise ValueError('valid_mask must be a boolean array matching the image')
        if not valid_mask.any():
            raise ValueError('valid_mask contains no image pixels')
    if polarity not in ('dark-on-light', 'light-on-dark'):
        raise ValueError('Unsupported image polarity')
    areas = [a for a in (min_object_area, min_hole_area) if a is not None]
    if not np.isfinite([point_spacing, *areas]).all() or point_spacing < 0 or min(areas, default=0) < 0:
        raise ValueError('Invalid contour adapter thresholds')
    full_scale = int(max_value or np.iinfo(gray.dtype).max)
    bit_depth = int(gray.dtype.itemsize * 8)
    if polarity == 'light-on-dark':
        # The model sees dark parts on a bright backlight.
        gray = (full_scale - np.minimum(gray, full_scale)).astype(gray.dtype)
    if model is None:
        model = load_edge_model()
    whole = [[-.5, -.5], [gray.shape[1]-.5, gray.shape[0]-.5]]
    try:
        traced = model.rings(gray, max_value=full_scale, valid_mask=valid_mask,
                             min_object_area=min_object_area, min_hole_area=min_hole_area)
    except PAEExtractionError as error:
        # Rings and points are in edge-model pixels; this package puts centers at i.
        _extraction_failure(f'PAE edge extraction failed: {error}. Deep Learning edges (Settings) '
                            'handle blurred edges and fine detail.',
                            rings=[error.coarse_ring-.5] if error.coarse_ring is not None else (),
                            issue_points=error.points-.5 if len(error.points) else whole,
                            issue_label=error.label)
    except ValueError as error:
        _extraction_failure(f'Edge model tracing failed: {error}', issue_points=whole,
                            issue_label='Edge tracing failed')
    # Edge-model pixels cover [i, i+1); this package puts pixel centers at i.
    rings = [q-.5 for q in traced['rings']]
    if not rings:
        _extraction_failure('No silhouette contours', issue_points=whole)
    ordered, metadata = [], []
    for i, (q, is_hole, depth, parent) in enumerate(zip(rings, traced['is_hole'], traced['depth'], traced['parent'])):
        crossings = len(q)
        q, spacing = _simplify(q, point_spacing)
        if not Polygon(q).is_valid:
            _extraction_failure(f'Path {i} is self-intersecting', rings=rings, issue_points=q,
                                issue_label=f'Path {i}: self-intersection')
        # Image coordinates: outer rings positive shoelace area, holes negative.
        if (_signed_area(q) < 0) != is_hole:
            q = q[::-1]
        # Start at the top-left vertex so path indices do not depend on tracing order.
        q = np.roll(q, -int(np.lexsort((q[:, 0], q[:, 1]))[0]), axis=0)
        ordered.append(q)
        metadata.append({'path': i, 'depth': int(depth), 'parent': int(parent), 'is_hole': bool(is_hole), 'closed': True,
                         'point_count': len(q), 'lattice_crossings': crossings,
                         'point_spacing': float(spacing), 'area': float(abs(_signed_area(q)))})
    # Thinning must not make neighbouring rings cross.
    polygons = [Polygon(q) for q in ordered]
    low = np.array([q.min(axis=0) for q in ordered])
    high = np.array([q.max(axis=0) for q in ordered])
    for i in range(len(polygons)):
        for j in np.flatnonzero((low[:i] <= high[i]).all(axis=1) & (high[:i] >= low[i]).all(axis=1)):
            if polygons[i].boundary.intersects(polygons[j].boundary):
                _extraction_failure('Subpixel rings cross or touch', rings=rings,
                                    issue_points=np.r_[ordered[i], ordered[j]],
                                    issue_label=f'Rings {j} and {i} cross or touch')
    try:
        contours = Contours(np.concatenate(ordered), np.r_[0, np.cumsum([len(q) for q in ordered])])
    except ValueError as error:
        _extraction_failure(f'Contour construction failed: {error}', rings=rings,
                            issue_points=np.concatenate(ordered), issue_label='Invalid extracted contour')
    extra = {'pae': traced['pae']} if 'pae' in traced else {}
    return contours, {'method': traced.get('method', 'edge_model_rings'),
                      'model': model.describe(),
                      'input_bit_depth': bit_depth,
                      'polarity': polarity,
                      'boundary_closure_padding_px': int(traced['apron']),
                      'backlight_level': round(float(traced['backlight']), 4),
                      'backlight_gain': round(float(traced['gain']), 4),
                      'min_object_area_px2': traced['min_area'][0],
                      'min_hole_area_px2': traced['min_area'][1],
                      'dropped_rings': traced['dropped_rings'],
                      'unsupported_rings': traced['unsupported_rings'],
                      'rings': metadata,
                      'input_edge_count': int(traced['lattice_crossings']),
                      'stage_seconds': {k: round(v, 4) for k, v in traced['stage_seconds'].items()},
                      'interpolated_point_count': 0, **extra}
