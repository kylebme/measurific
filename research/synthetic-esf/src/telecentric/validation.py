"""A diagnostic edge detector and numerical convergence studies.

These measure stability of one estimator, not a certificate of physical accuracy.
"""
from dataclasses import replace
import numpy as np
from scipy.ndimage import map_coordinates
from .render import render


def contour_crossings(result, labels, half_width_px=4., samples=161):
    offsets = np.linspace(-half_width_px, half_width_px, samples)
    rx, ry, width, height = result.metadata['roi_xywh']
    config = result.metadata['config']
    threshold = (config['background']+config['foreground'])/2
    crossings = []
    for label in labels:
        points = np.asarray(label['xy_px'])
        normals = np.asarray(label['normal_xy'])
        visible = np.asarray(label['visible_in_union'])
        for point, normal, is_visible in zip(points, normals, visible):
            xy = point + offsets[:, None]*normal
            local = xy-[rx+0.5, ry+0.5]
            if not is_visible or np.any(local < 0) or np.any(local > [width-1, height-1]):
                crossings.append(np.nan)
                continue
            profile = map_coordinates(result.irradiance, local.T[::-1], order=1, mode='nearest') - threshold
            if not np.isfinite(profile).all():
                crossings.append(np.nan)
                continue
            hits = np.flatnonzero((profile[:-1] <= 0) & (profile[1:] > 0))
            if not len(hits):
                crossings.append(np.nan)
                continue
            positions = offsets[hits]-profile[hits]*(offsets[1]-offsets[0])/(profile[hits+1]-profile[hits])
            crossings.append(positions[np.argmin(np.abs(positions))])
    return np.asarray(crossings)


def convergence(scene, config, roi=None, levels=(8, 16, 32, 64), angular_levels=None, support_levels=None, tolerance_px=0.01, progress=None, contour_count=128):
    from .config import positive
    positive('tolerance_px', tolerance_px)
    levels = list(levels)
    if len(levels) < 2 or levels != sorted(set(levels)):
        raise ValueError('Use at least two increasing supersampling levels')
    from .config import integer
    integer('contour_count', contour_count, 16)
    labels = scene.labels(config.camera, count=contour_count)
    reports = {}
    base = replace(config, sampling=replace(config.sampling, supersample=levels[-1]))
    axes = [('supersample', levels)]
    if angular_levels:
        axes.append(('angular_samples', list(angular_levels)))
    if support_levels:
        axes.append(('psf_support_px', list(support_levels)))
    for axis, values in axes:
        if len(values) < 2 or values != sorted(set(values)):
            raise ValueError(f'{axis} levels must be increasing and have at least two values')
        previous = previous_edges = None
        rows = []
        for value in values:
            if axis == 'supersample':
                local = replace(base, sampling=replace(base.sampling, supersample=value))
            elif axis == 'angular_samples':
                local = replace(base, illumination=replace(base.illumination, samples=value))
            else:
                local = replace(base, optics=replace(base.optics, support_px=value))
            result = render(scene, local, roi, progress)
            edges = contour_crossings(result, labels)
            valid_edges = np.isfinite(edges)
            row = dict(value=value, valid_crossings=int(valid_edges.sum()), total_contour_samples=len(edges))
            if valid_edges.any():
                row['median_threshold_bias_px'] = float(np.median(edges[valid_edges]))
            if previous is not None:
                valid = valid_edges & np.isfinite(previous_edges)
                row['image_max_delta'] = float(np.max(np.abs(result.irradiance-previous.irradiance)))
                row['paired_crossings'] = int(valid.sum())
                row['crossing_validity_changes'] = int(np.count_nonzero(valid_edges != np.isfinite(previous_edges)))
                if valid.any():
                    delta = edges[valid]-previous_edges[valid]
                    row['edge_max_delta_px'] = float(np.max(np.abs(delta)))
                    row['edge_rms_delta_px'] = float(np.sqrt(np.mean(delta**2)))
                row['within_tolerance'] = bool(valid.any() and row.get('edge_max_delta_px', np.inf) <= tolerance_px and row['crossing_validity_changes'] == 0)
            rows.append(row)
            previous, previous_edges = result, edges
        reports[axis] = rows
    return dict(tolerance_px=tolerance_px, studies=reports, config=config.to_dict(), roi_xywh=previous.metadata['roi_xywh'], shapes=[shape.metadata() for shape in scene.shapes],
                interpretation='Successive-level stability of a bilinear 50% crossing diagnostic. Not a bound on absolute rendering or measurement error. Inspect valid/paired counts; test more levels, phases, angular seeds and PSF supports.',
                last_comparisons_within_tolerance=all(rows[-1]['within_tolerance'] for rows in reports.values()))
