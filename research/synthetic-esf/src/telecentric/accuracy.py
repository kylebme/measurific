"""Distortion/threshold attribution and dimension-level diagnostic analysis.

Exact inverse distortion is used ONLY as an oracle error-budget ablation. This
does not claim that a real geometric calibration has zero residual error.
"""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr
from scipy.stats import ncx2
from .config import Camera
from .benchmark import error_stats
from .dataset import versions
from .render import RenderResult
from .validation import contour_crossings


def integrated_gaussian(centers, edge, sigma):
    """Exact unit-pixel integral of a Gaussian-blurred straight step."""
    centers = np.asarray(centers)

    def antiderivative(x):
        t = (x-edge)/sigma
        return (x-edge)*ndtr(t)+sigma*np.exp(-0.5*t*t)/np.sqrt(2*np.pi)

    return antiderivative(centers+0.5)-antiderivative(centers-0.5)


def crossing_1d(centers, values, method='linear'):
    hits = np.flatnonzero((values[:-1] <= 0.5) & (values[1:] > 0.5))
    if len(hits) != 1:
        raise ValueError('Reference profile must have one rising crossing')
    i = hits[0]
    if method == 'linear':
        return centers[i]+(0.5-values[i])*(centers[i+1]-centers[i])/(values[i+1]-values[i])
    interp = (PchipInterpolator if method == 'pchip' else CubicSpline)(centers, values)
    return brentq(lambda x: float(interp(x))-0.5, centers[i], centers[i+1], xtol=1e-13)


def analytic_controls(phase_count=201):
    """No silhouette rasterization: isolates interpolation and curvature bias."""
    phases = np.arange(phase_count)/phase_count
    centers = np.arange(-12, 14)+0.5
    rows = []
    for sigma in (0.4, 0.5, 0.6, 0.8, 1., 1.5, 2.):
        errors = {method: [] for method in ('linear', 'pchip', 'cubic', 'integrated_erf_fit')}
        for phase in phases:
            values = integrated_gaussian(centers, phase, sigma)
            for method in ('linear', 'pchip', 'cubic'):
                errors[method].append(float(crossing_1d(centers, values, method)-phase))
            start = crossing_1d(centers, values)
            fit = least_squares(lambda pars: integrated_gaussian(centers, pars[0], pars[1])-values,
                                [start, 0.8], bounds=([-1, 0.1], [2, 4]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
            errors['integrated_erf_fit'].append(float(fit.x[0]-phase))
        rows.append(dict(sigma_px=sigma, continuous_threshold_bias_px=0., errors={method:error_stats(value) for method,value in errors.items()},
                         phase_px=phases.tolist(), phase_errors=errors))
    # A 2D Gaussian convolved with an opaque disk is the survival probability
    # of a noncentral chi-square variable (df=2). Integrate the unit square
    # pixel with Gauss-Legendre quadrature, then find its continuous 50% point.
    # This reference contains no mesh or high-resolution binary raster.
    curvature = []
    for sigma in (0.6, 1.0):
        for radius in (7., 20., 50., 100., 250.):
            roots = []
            for order in (8, 12):
                nodes, weights = np.polynomial.legendre.leggauss(order)
                dx, dy = np.meshgrid(nodes/2, nodes/2)
                weight = np.outer(weights, weights)/4

                def intensity(u):
                    return float(np.sum(weight*ncx2.sf((radius/sigma)**2, 2, ((u+dx)**2+dy**2)/sigma**2)))

                roots.append(brentq(lambda u: intensity(u)-0.5, radius-2*sigma, radius+sigma, xtol=1e-12))
            curvature.append(dict(sigma_px=sigma, radius_px=radius, continuous_radius_bias_px=roots[-1]-radius,
                                  continuous_diameter_bias_px=2*(roots[-1]-radius), quadrature_root_delta_px=abs(roots[1]-roots[0])))
    return dict(straight_edge=rows, circle_curvature=curvature,
                interpretation='Straight-edge controls use exact pixel-integrated Gaussian data, normalized known plateaus and no noise/distortion. ERF-fit success here is matched-model numerical verification, not evidence of robustness. Circle results are continuous pixel-response threshold bias along an axis; interpolation/noise are excluded.')


def flat_geometry(label, camera):
    """Physical side dimensions and orientation; None for projected circles."""
    shape = label['shape']
    if shape['kind'] == 'box':
        dims = np.asarray(shape['size_mm'][:2])/camera.object_pitch_mm
    elif shape['kind'] == 'cylinder' and shape['axis'] == 'y':
        dims = np.array([2*shape['radius_mm'], shape['length_mm']])/camera.object_pitch_mm
    else:
        return None
    a = np.deg2rad(shape['angle_deg'])
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    center = camera.world_to_ideal_pixel(shape['center_mm'][:2])
    return dims, rotation, center


def stable_mask(label, camera, margin_px=3.):
    geometry = flat_geometry(label, camera)
    if geometry is None:
        return np.ones(len(label['xy_px']), dtype=bool)
    dims, rotation, center = geometry
    local = (np.asarray(label['ideal_xy_px'])-center)@rotation
    corners = np.array([[x, y] for x in (-dims[0]/2, dims[0]/2) for y in (-dims[1]/2, dims[1]/2)])
    distance = np.linalg.norm(local[:, None, :]-corners, axis=-1).min(axis=1)
    return distance >= margin_px


def circle_diameter(points):
    points = np.asarray(points)
    if len(points) < 8:
        return None
    origin = points.mean(axis=0)
    q = points-origin
    matrix = np.column_stack((2*q, np.ones(len(q))))
    initial = np.linalg.lstsq(matrix, np.sum(q*q, axis=1), rcond=None)[0]
    center = initial[:2]+origin
    radius = np.sqrt(max(initial[2]+initial[:2]@initial[:2], 1e-12))
    fit = least_squares(lambda p: np.linalg.norm(points-p[:2], axis=1)-p[2], [*center, radius],
                        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    return float(2*fit.x[2])


def shape_dimensions(label, points, valid, camera, margin_px=3.):
    shape = label['shape']
    geometry = flat_geometry(label, camera)
    if geometry is None:
        if shape['kind'] == 'sphere' or (shape['kind'] == 'cylinder' and shape['axis'] == 'z'):
            diameter = circle_diameter(points[valid])
            if diameter is not None:
                return {'diameter': (diameter, 2*shape['radius_mm']/camera.object_pitch_mm)}
        return {}
    dims, rotation, _ = geometry
    normals = np.asarray(label['normal_world_xy'])@rotation
    mask = valid & stable_mask(label, camera, margin_px)
    output = {}
    for axis, name in enumerate(('width', 'height')):
        plus = mask & (normals[:, axis] > 0.9)
        minus = mask & (normals[:, axis] < -0.9)
        if plus.sum() >= 2 and minus.sum() >= 2:
            projected = points@rotation[:, axis]
            output[name] = (float(projected[plus].mean()-projected[minus].mean()), float(dims[axis]))
    return output


def fit_native_gaussian_edge(image, meta, point, normal, initial_shift):
    """Uncalibrated local straight ERF fit on native square-pixel values.

    Supplied normal/neighborhood, free location, sigma and two plateau levels.
    Pixel area is integrated with 8x8 Gauss-Legendre quadrature. This deliberately
    does not know the renderer's actual PSF, field nodes or distortion parameters.
    """
    rx, ry, w, h = meta['roi_xywh']
    x, y = np.meshgrid(rx+np.arange(w)+0.5, ry+np.arange(h)+0.5)
    delta = np.stack((x-point[0], y-point[1]), axis=-1)
    tangent = np.array([-normal[1], normal[0]])
    d = delta@normal
    t = delta@tangent
    selected = (np.abs(d-initial_shift) <= 4) & (np.abs(t) <= 1.25) & np.isfinite(image)
    if selected.sum() < 12:
        return None
    distances, values = d[selected], image[selected]
    nodes, weights = np.polynomial.legendre.leggauss(8)
    dx, dy = np.meshgrid(nodes/2, nodes/2)
    projected = (dx*normal[0]+dy*normal[1]).ravel()
    weight = (np.outer(weights, weights)/4).ravel()

    def residual(parameters):
        offset, sigma, dark, contrast = parameters
        response = ndtr((distances[:, None]+projected-offset)/sigma)@weight
        return dark+contrast*response-values

    fit = least_squares(residual, [initial_shift, 0.65, 0, 1],
                        bounds=([initial_shift-1.5, 0.15, -0.2, 0.3], [initial_shift+1.5, 3, 0.4, 1.5]),
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=100)
    if not fit.success or np.any(fit.active_mask):
        return None
    return dict(shift_px=float(fit.x[0]), sigma_px=float(fit.x[1]), fit_rms=float(np.sqrt(np.mean(fit.fun**2))))


def analyze_benchmark(dataset, output, margin_px=3., fit_margin_px=4., compare_fit=True, progress=None):
    folder, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError('Accuracy report output already exists')
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Completed benchmark required')
    observations, dimensions, fit_rows = [], [], []
    fit_attempts = {}
    capture_schema = None
    for sample_index, sample in enumerate(manifest['samples']):
        meta = json.loads((folder/sample['metadata']).read_text())
        camera = Camera(**meta['config']['camera'])
        schema = [(c['exposure_ms'], c['repeat']) for c in meta['captures']]
        if capture_schema is not None and schema != capture_schema:
            raise ValueError('Accuracy analysis expects the same exposure/repeat schedule across samples')
        capture_schema = schema
        with np.load(folder/sample['arrays']) as archive:
            images = {'irradiance': archive['irradiance'].astype(float)}
            sensor = meta['config']['sensor']
            for i, capture in enumerate(meta['captures']):
                values = ((archive[f'raw_{i:02d}'].astype(float)-sensor['black_level_adu'])*sensor['electrons_per_adu']/capture['exposure_ms']-sensor['dark_e_per_ms'])/sensor['bright_e_per_ms']
                values[archive[f'saturated_{i:02d}'] | archive[f'clipped_low_{i:02d}']] = np.nan
                images[f'raw_{i:02d}'] = values
        for channel, image in images.items():
            shifts = contour_crossings(RenderResult(image, None, None, meta), meta['labels'])
            offset = 0
            for label_index, label in enumerate(meta['labels']):
                ideal = np.asarray(label['ideal_xy_px']).reshape(-1, 2)
                geometric = np.asarray(label['xy_px']).reshape(-1, 2)
                normals = np.asarray(label['normal_xy']).reshape(-1, 2)
                world_normals = np.asarray(label['normal_world_xy']).reshape(-1, 2)
                local = shifts[offset:offset+len(ideal)]
                offset += len(ideal)
                valid = np.isfinite(local)
                measured = geometric+np.nan_to_num(local)[:, None]*normals
                corrected = camera.pixel_to_ideal(measured)
                keep = stable_mask(label, camera, margin_px)
                for i in np.flatnonzero(valid):
                    n = world_normals[i]
                    distortion = float((geometric[i]-ideal[i])@n)
                    sampling = float((measured[i]-geometric[i])@n)
                    observations.append(dict(sample=sample_index, label=label_index, edge=int(i), group=sample['group_id'], split=sample['split'], channel=channel, stable=bool(keep[i]),
                        total_px=distortion+sampling, distortion_px=distortion, optical_sampling_px=sampling,
                        after_exact_undistortion_px=float((corrected[i]-ideal[i])@n)))
                sources = {'geometric_distortion_only': geometric, 'threshold_uncorrected': measured, 'threshold_exact_undistortion': corrected}
                for method, points in sources.items():
                    if method == 'geometric_distortion_only' and channel != 'irradiance':
                        continue
                    for name, (value, truth) in shape_dimensions(label, points, valid, camera, margin_px).items():
                        dimensions.append(dict(sample=sample_index, label=label_index, group=sample['group_id'], split=sample['split'], channel=channel, method=method, dimension=name,
                            measured_px=value, true_px=truth, error_px=value-truth, error_um=(value-truth)*camera.object_pitch_mm*1000))
                if compare_fit and channel == 'irradiance' and flat_geometry(label, camera) is not None:
                    fit_valid = valid & stable_mask(label, camera, fit_margin_px)
                    fit_attempts[sample['group_id']] = fit_attempts.get(sample['group_id'], 0)+int(fit_valid.sum())
                    fitted = measured.copy()
                    success = np.zeros(len(ideal), bool)
                    for i in np.flatnonzero(fit_valid):
                        fit = fit_native_gaussian_edge(image, meta, geometric[i], normals[i], local[i])
                        if fit is None:
                            continue
                        fitted[i] = geometric[i]+fit['shift_px']*normals[i]
                        success[i] = True
                        corrected_fit = camera.pixel_to_ideal(fitted[i])
                        fit_rows.append(dict(sample=sample_index, group=sample['group_id'], label=label_index, edge=int(i),
                            threshold_error_px=float((corrected[i]-ideal[i])@world_normals[i]), erf_error_px=float((corrected_fit-ideal[i])@world_normals[i]), **fit))
                    for method, points in [('threshold_fit_subset', corrected), ('uncalibrated_erf_fit', camera.pixel_to_ideal(fitted))]:
                        for name, (value, truth) in shape_dimensions(label, points, success, camera, fit_margin_px).items():
                            dimensions.append(dict(sample=sample_index, label=label_index, group=sample['group_id'], split=sample['split'], channel=channel, method=method, dimension=name,
                                measured_px=value, true_px=truth, error_px=value-truth, error_um=(value-truth)*camera.object_pitch_mm*1000))
        if progress:
            progress(sample_index+1, len(manifest['samples']))
    grouped, dimension_groups, fit_groups = {}, {}, {}
    for group in sorted({o['group'] for o in observations}):
        grouped[group] = {}
        for selection in ('all', 'stable'):
            rows = [o for o in observations if o['group'] == group and o['channel'] == 'irradiance' and (selection == 'all' or o['stable'])]
            grouped[group][selection] = {key:error_stats([o[key] for o in rows]) for key in ('total_px', 'distortion_px', 'optical_sampling_px', 'after_exact_undistortion_px')}
        dimension_groups[group] = {}
        for method in sorted({d['method'] for d in dimensions}):
            rows = [d for d in dimensions if d['group'] == group and d['channel'] == 'irradiance' and d['method'] == method]
            if rows:
                dimension_groups[group][method] = error_stats([d['error_px'] for d in rows])
        fits = [row for row in fit_rows if row['group'] == group]
        if fits:
            fit_groups[group] = {key:error_stats([row[key] for row in fits]) for key in ('threshold_error_px', 'erf_error_px')}
            fit_groups[group].update(attempted=fit_attempts[group], accepted=len(fits), rejected=fit_attempts[group]-len(fits))
    # Paired-repeat noise separates frame-to-frame variation from static bias.
    noise = {}
    repeat_pairs = []
    for exposure in sorted({exposure for exposure, _ in capture_schema}):
        indices = [i for i, (e, _) in enumerate(capture_schema) if e == exposure]
        if len(indices) >= 2:
            repeat_pairs.append((exposure, f'raw_{indices[0]:02d}', f'raw_{indices[1]:02d}'))
    for group in grouped:
        noise[group] = {}
        for exposure, channel0, channel1 in repeat_pairs:
            a = {(d['sample'], d['label'], d['dimension']):d['measured_px'] for d in dimensions if d['group'] == group and d['channel'] == channel0 and d['method'] == 'threshold_exact_undistortion'}
            b = {(d['sample'], d['label'], d['dimension']):d['measured_px'] for d in dimensions if d['group'] == group and d['channel'] == channel1 and d['method'] == 'threshold_exact_undistortion'}
            pairs = sorted(a.keys() & b.keys())
            if pairs:
                noise[group][channel0+'_'+channel1] = dict(exposure_ms=exposure, pairs=len(pairs), single_capture_std_estimate_px=float(np.sqrt(np.mean([(a[k]-b[k])**2 for k in pairs])/2)))
    report = dict(versions=versions(), dataset=str(folder.resolve()), stable_corner_margin_px=margin_px, fit_corner_margin_px=fit_margin_px,
                  conventions=['Exact distortion inverse is an oracle ablation, not a fitted calibration.', 'Signed point errors add; RMS components generally do not.', 'Post-distortion residual includes optics, pixel integration, interpolation and finite renderer sampling.', 'Stable masks remove only corner-adjacent straight-edge points; they do not remove curved parts or neighboring edges.', 'Circle centers/radii are fitted; rectangular dimensions use supplied orientation/side correspondence and stable side means.', 'ERF comparison uses native pixel values, known normals/neighborhoods and a free uncalibrated Gaussian model on straight regions only.', 'Repeat noise estimates use only the first two captures at each exposure; not a complete repeatability study.'],
                  groups=grouped, dimension_groups=dimension_groups, fit_groups=fit_groups, paired_dimension_noise=noise, observations=observations, dimensions=dimensions, native_erf_fits=fit_rows,
                  analytic_controls=analytic_controls())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def refine_representatives(dataset, output, groups=None, levels=(64, 128), progress=None, sample_indices=None):
    """Re-render one fixed sample per selected group; preserve every other setting."""
    from dataclasses import replace, fields
    from .config import RenderConfig
    from .geometry import Primitive, Scene
    from .render import render
    folder, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError('Refinement output already exists')
    manifest = json.loads((folder/'manifest.json').read_text())
    groups = groups or ['test_rotated_block', 'test_pin_dimension', 'test_corner_asymmetric', 'test_horizontal_pin']
    report = dict(dataset=str(folder.resolve()), versions=versions(), status='incomplete', cases=[],
                  convention='One unchanged benchmark sample per selected group, exact inverse distortion, 3-pixel corner margin for rectangular side means. Spatial refinement only; angular samples/PSF support held fixed.')
    output.parent.mkdir(parents=True, exist_ok=True)
    for group in groups:
        index = (sample_indices or {}).get(group, 0)
        sample = [s for s in manifest['samples'] if s['group_id'] == group][index]
        meta = json.loads((folder/sample['metadata']).read_text())
        config = RenderConfig.from_dict(meta['config'])
        allowed = {f.name for f in fields(Primitive)}
        shapes = [Primitive(**{k:v for k,v in label['shape'].items() if k in allowed}) for label in meta['labels']]
        scene = Scene(shapes)
        case = dict(group=group, sample_index_within_group=index, metadata=sample['metadata'], source_supersample=config.sampling.supersample, angular_samples=config.illumination.samples, results=[])
        for level in [config.sampling.supersample, *levels]:
            local_config = replace(config, sampling=replace(config.sampling, supersample=level))
            if level == config.sampling.supersample:
                with np.load(folder/sample['arrays']) as arrays:
                    result = RenderResult(arrays['irradiance'], None, None, meta)
            else:
                result = render(scene, local_config, meta['roi_xywh'], progress)
            shifts = contour_crossings(result, meta['labels'])
            offset, dims, edge_errors = 0, {}, []
            for i, label in enumerate(meta['labels']):
                n = len(label['xy_px'])
                local_shifts = shifts[offset:offset+n]
                offset += n
                valid = np.isfinite(local_shifts)
                measured = np.asarray(label['xy_px'])+np.nan_to_num(local_shifts)[:, None]*np.asarray(label['normal_xy'])
                corrected = config.camera.pixel_to_ideal(measured)
                for name, (value, truth) in shape_dimensions(label, corrected, valid, config.camera).items():
                    dims[f'{i}_{name}'] = value-truth
                keep = valid & stable_mask(label, config.camera)
                edge_errors.extend(np.sum((corrected-np.asarray(label['ideal_xy_px']))*np.asarray(label['normal_world_xy']), axis=1)[keep])
            case['results'].append(dict(supersample=level, dimension_errors_px=dims, stable_edge_errors=error_stats(edge_errors)))
        first, second = case['results'][-2:]
        case['last_dimension_change_px'] = {k:second['dimension_errors_px'][k]-v for k,v in first['dimension_errors_px'].items() if k in second['dimension_errors_px']}
        report['cases'].append(case)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    report['status'] = 'complete'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--refine', action='store_true')
    args = parser.parse_args()
    if args.refine:
        result = refine_representatives(args.dataset, args.output, progress=lambda done,total: print(f'Rendered tiles {done}/{total}', flush=True))
        print(json.dumps(result, indent=2))
    else:
        result = analyze_benchmark(args.dataset, args.output, progress=lambda done,total: print(f'Analyzed {done}/{total}', flush=True))
        print(f"Wrote {len(result['observations'])} edge observations and {len(result['dimensions'])} dimension measurements to {args.output}")
