"""Artifact-grouped calibration/test data and a deliberately simple baseline.

The baseline uses supplied contour neighborhoods/normals, not blind detection.
Only calibration labels enter fitting; held-out labels enter scoring only.
"""
from dataclasses import replace
from pathlib import Path
import copy
import hashlib
import json
import re
import numpy as np
from .config import RenderConfig, positive, integer
from .dataset import resolve_psf_paths, versions, write_sample
from .geometry import scene_from_list
from .render import render, RenderResult, validate_roi
from .sensor import SensorModel
from .validation import contour_crossings


def signature(shapes, base):
    """Physical artifact identity independent of naming, pose or tessellation path."""
    if not shapes:
        raise ValueError('A benchmark artifact group needs at least one shape')
    identities = []
    for shape in shapes:
        kind = shape['kind']
        if kind == 'mesh':
            digest = hashlib.sha256((base/shape['path']).read_bytes()).hexdigest()
            identity = dict(kind=kind, sha256=digest, scale=float(shape.get('scale_to_mm', 1.0)))
        else:
            # Validate/default once using the public scene factory.
            obj = scene_from_list([shape], base).shapes[0]
            identity = {'kind': kind}
            if kind == 'box':
                identity['dimensions'] = sorted(map(float, obj.size_mm))
            elif kind == 'sphere':
                identity['radius'] = float(obj.radius_mm)
            else:
                identity.update(radius=float(obj.radius_mm), length=float(obj.length_mm))
        identities.append(json.dumps(identity, sort_keys=True))
    # An assembly is grouped by its constituent artifacts, even if poses differ.
    return hashlib.sha256(json.dumps(sorted(identities)).encode()).hexdigest()


def load_benchmark(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    allowed = {'config', 'groups', 'seed', 'exposures_ms', 'repeat_captures', 'description'}
    if set(spec)-allowed:
        raise ValueError(f'Unknown benchmark keys: {set(spec)-allowed}')
    config = copy.deepcopy(spec.get('config', {}))
    resolve_psf_paths(config.get('optics', {}), path.parent)
    base_config = RenderConfig.from_dict(config)
    integer('seed', spec.get('seed', 2026), 0)
    integer('repeat_captures', spec.get('repeat_captures', 2))
    if not spec.get('exposures_ms', [0.5, 1.0]):
        raise ValueError('exposures_ms cannot be empty')
    for exposure in spec.get('exposures_ms', [0.5, 1.0]):
        positive('exposure_ms', exposure)
    if not spec.get('groups'):
        raise ValueError('Benchmark groups cannot be empty')
    ids, identities, groups = set(), {}, []
    group_keys = {'id', 'split', 'tags', 'shapes', 'placements_px', 'phases_px', 'z_offsets_mm', 'roi_size', 'roi_offset_px', 'optics', 'illumination', 'kernel_provenance'}
    for original in spec['groups']:
        group = copy.deepcopy(original)
        if set(group)-group_keys:
            raise ValueError(f'Unknown benchmark group keys: {set(group)-group_keys}')
        name, split = group['id'], group['split']
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', name) or name in ids:
            raise ValueError('Group ids must be unique safe names')
        ids.add(name)
        if split not in {'calibration', 'held_out'}:
            raise ValueError('split must be calibration or held_out')
        digest = signature(group['shapes'], path.parent)
        if digest in identities and identities[digest] != split:
            raise ValueError(f'Geometry leakage across splits in group {name}')
        identities[digest] = split
        local = base_config.to_dict()
        for key in ('optics', 'illumination'):
            local[key].update(group.get(key, {}))
        resolve_psf_paths(local['optics'], path.parent)
        local = RenderConfig.from_dict(local)
        measured = local.optics.psf == 'measured' or any(n.get('psf', local.optics.psf) == 'measured' for n in local.optics.field_nodes)
        if measured:
            provenance = group.get('kernel_provenance', {})
            if provenance.get('kind') not in {'empirical', 'synthetic'} or not provenance.get('description') or provenance.get('pre_pixel') is not True or provenance.get('registered_origin') is not True:
                raise ValueError('Imported benchmark kernels require kind/description and explicit pre_pixel/registered_origin provenance')
        placements = group.get('placements_px', [[local.camera.width/2, local.camera.height/2]])
        phases = group.get('phases_px', [[0, 0]])
        if not placements or not phases:
            raise ValueError('placements_px and phases_px cannot be empty')
        for entry in placements+phases+[group.get('roi_offset_px', [0, 0])]:
            if len(entry) != 2 or not np.isfinite(entry).all():
                raise ValueError('Placements, phases and ROI offsets must be finite pairs')
        zs = group.get('z_offsets_mm', [0])
        if not zs or not np.isfinite(zs).all():
            raise ValueError('z_offsets_mm must be a nonempty finite list')
        size = group.get('roi_size', [32, 32])
        if len(size) != 2:
            raise ValueError('roi_size must be [width, height]')
        for dimension in size:
            integer('roi_size', dimension)
        for placement in placements:
            for phase in phases:
                roi_center = np.asarray(placement)+phase+group.get('roi_offset_px', [0, 0])
                xy = np.floor(roi_center-np.asarray(size)/2).astype(int)
                validate_roi(local.camera, [int(xy[0]), int(xy[1]), *size])
        groups.append((group, local, digest))
    if {g['split'] for g, _, _ in groups} != {'calibration', 'held_out'}:
        raise ValueError('A benchmark must contain both calibration and held_out groups')
    return spec, groups, path.parent


def generate_benchmark(path, output, progress=None):
    spec, groups, base = load_benchmark(path)
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise ValueError('Benchmark output directory must be empty')
    manifest = dict(schema_version=2, status='incomplete', kind='calibration_benchmark', versions=versions(), source_spec=spec,
                    split_policy='All poses, phases, heights, exposures and repeats of a physical artifact group stay in one split; exact geometry signatures cannot cross splits',
                    qualification='Sampling settings are explicit. Run validation-suite for numerical qualification; generation is not a 0.01-pixel certificate.', samples=[])
    manifest_path = folder/'manifest.json'

    def save():
        temporary = folder/'manifest.tmp'
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        temporary.replace(manifest_path)

    save()
    for group, config, digest in groups:
        camera = config.camera
        sensor = SensorModel(config.sensor, (camera.height, camera.width))
        group_folder = folder/group['split']/group['id']
        group_folder.mkdir(parents=True)
        group_seed = int.from_bytes(hashlib.sha256(group['id'].encode()).digest()[:4], 'little')
        index = 0
        for placement in group.get('placements_px', [[camera.width/2, camera.height/2]]):
            for phase in group.get('phases_px', [[0, 0]]):
                for z in group.get('z_offsets_mm', [0]):
                    ideal_center = np.asarray(placement)+phase
                    center_mm = (ideal_center-[camera.width/2, camera.height/2])*camera.object_pitch_mm+camera.center_xy_mm
                    shapes = copy.deepcopy(group['shapes'])
                    for shape in shapes:
                        key = 'translation_mm' if shape['kind'] == 'mesh' else 'center_mm'
                        shape[key] = (np.asarray(shape.get(key, [0, 0, 0]))+[*center_mm, z]).tolist()
                    size = group.get('roi_size', [32, 32])
                    origin = np.floor(ideal_center+group.get('roi_offset_px', [0, 0])-np.asarray(size)/2).astype(int)
                    roi = [int(origin[0]), int(origin[1]), *size]
                    scene = scene_from_list(shapes, base)
                    result = render(scene, config, roi, progress)
                    exposures = spec.get('exposures_ms', [0.5, 1.0])
                    repeats = spec.get('repeat_captures', 2)
                    seeds = np.random.SeedSequence([spec.get('seed', 2026), group_seed, index]).generate_state(len(exposures)*repeats)
                    captures, capture_meta = [], []
                    for exposure in exposures:
                        for repeat in range(repeats):
                            seed = int(seeds[len(captures)])
                            captures.append(sensor.capture(result.irradiance, exposure, seed, roi))
                            capture_meta.append(dict(exposure_ms=exposure, repeat=repeat, seed=seed))
                    stem = f'sample_{index:06d}'
                    extra = dict(group_id=group['id'], split=group['split'], geometry_signature=digest, tags=group.get('tags', []),
                                 ideal_placement_px=ideal_center.tolist(), phase_px=phase, z_offset_mm=z, captures=capture_meta,
                                 kernel_provenance=group.get('kernel_provenance', {'kind': 'synthetic', 'description': 'Parameterized intensity PSF proxy'}))
                    write_sample(group_folder, stem, result, captures, scene.labels(camera, count=128), extra)
                    prefix = group_folder.relative_to(folder)/stem
                    manifest['samples'].append(dict(group_id=group['id'], split=group['split'], geometry_signature=digest, tags=group.get('tags', []), arrays=str(prefix)+'.npz', metadata=str(prefix)+'.json'))
                    save()
                    index += 1
    manifest['status'] = 'complete'
    save()
    return manifest


def correction_features(xy, normals, camera_size):
    """Normal projection of a cubic vector displacement field (20 features)."""
    x, y = ((np.asarray(xy)-np.asarray(camera_size)/2)/(max(camera_size)/2)).T
    basis = np.column_stack((np.ones(len(x)), x, y, x*x, x*y, y*y, x**3, x*x*y, x*y*y, y**3))
    return np.column_stack((basis*normals[:, [0]], basis*normals[:, [1]]))


def error_stats(errors):
    errors = np.asarray(errors)
    if not len(errors):
        return {'count': 0, 'bias_px': None, 'rms_px': None, 'max_abs_px': None, 'p95_abs_px': None}
    return dict(count=len(errors), bias_px=float(errors.mean()), rms_px=float(np.sqrt(np.mean(errors**2))), max_abs_px=float(np.max(np.abs(errors))), p95_abs_px=float(np.percentile(np.abs(errors), 95)))


def evaluate_benchmark(dataset, output, channel='irradiance', ridge=1e-8):
    """Oracle-neighborhood threshold baseline with calibration-only map fitting.

Measurements have geometric correspondence and supplied normals. This is a
calibration harness, not a production ESF estimator or blind edge detector.
"""
    folder, output = Path(dataset), Path(output)
    positive('ridge', ridge, zero=True)
    if output.exists():
        raise ValueError('Evaluation output already exists')
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest.get('kind') != 'calibration_benchmark' or manifest['status'] != 'complete':
        raise ValueError('A completed calibration benchmark is required')
    assignments = {}
    for sample in manifest['samples']:
        if sample['split'] not in {'calibration', 'held_out'}:
            raise ValueError('Unknown split in benchmark manifest')
        for key in [('group', sample['group_id']), ('geometry', sample['geometry_signature'])]:
            if key in assignments and assignments[key] != sample['split']:
                raise ValueError('Split leakage in benchmark manifest')
            assignments[key] = sample['split']
    observations, sample_rows, camera_size = [], [], None
    for sample_id, sample in enumerate(manifest['samples']):
        meta = json.loads((folder/sample['metadata']).read_text())
        if any(meta[key] != sample[key] for key in ('group_id', 'split', 'geometry_signature')):
            raise ValueError('Sample identity/split disagrees with benchmark manifest')
        c = meta['config']['camera']
        size = [c['width'], c['height']]
        if camera_size is not None and size != camera_size:
            raise ValueError('Baseline assumes one sensor size')
        camera_size = size
        with np.load(folder/sample['arrays']) as arrays:
            values = arrays[channel].astype(float)
            if channel != 'irradiance':
                if not re.fullmatch(r'raw_\d\d', channel):
                    raise ValueError('channel must be irradiance or raw_NN')
                capture = meta['captures'][int(channel[4:])]
                sensor = meta['config']['sensor']
                values = ((values-sensor['black_level_adu'])*sensor['electrons_per_adu']/capture['exposure_ms']-sensor['dark_e_per_ms'])/sensor['bright_e_per_ms']
                # Saturated neighborhoods cannot support this baseline reliably.
                invalid = arrays['saturated_'+channel[4:]] | arrays['clipped_low_'+channel[4:]]
                values[invalid] = np.nan
        result = RenderResult(values, None, None, meta)
        shifts = contour_crossings(result, meta['labels'])
        offset = 0
        count = 0
        for label in meta['labels']:
            ideal = np.asarray(label['ideal_xy_px']).reshape(-1, 2)
            geometric = np.asarray(label['xy_px']).reshape(-1, 2)
            normal = np.asarray(label['normal_xy']).reshape(-1, 2)
            world_normal = np.asarray(label['normal_world_xy']).reshape(-1, 2)
            local_shifts = shifts[offset:offset+len(ideal)]
            offset += len(ideal)
            valid = np.isfinite(local_shifts)
            for i in np.flatnonzero(valid):
                measured = geometric[i]+local_shifts[i]*normal[i]
                observations.append(dict(sample_id=sample_id, group_id=sample['group_id'], split=sample['split'], observed=measured.tolist(), ideal=ideal[i].tolist(), distorted=geometric[i].tolist(), normal=world_normal[i].tolist(), optical_error_px=float(local_shifts[i])))
                count += 1
        sample_rows.append(dict(sample_id=sample_id, group_id=sample['group_id'], split=sample['split'], valid_edges=count, total_contour_samples=len(shifts)))
    if not observations:
        raise ValueError('No measurable contour neighborhoods')
    observed = np.array([o['observed'] for o in observations])
    ideal = np.array([o['ideal'] for o in observations])
    normals = np.array([o['normal'] for o in observations])
    errors = np.sum((observed-ideal)*normals, axis=1)
    features = correction_features(observed, normals, camera_size)
    training = np.array([o['split'] == 'calibration' for o in observations])
    if not training.any() or training.all():
        raise ValueError('Both splits need measurable edges')
    # Equal weight per calibration group, so extra repeats/poses do not silently
    # dominate the map. Held-out coordinates never enter fitting or scaling.
    group_ids = np.array([o['group_id'] for o in observations])
    weights = np.ones(training.sum())
    for group in np.unique(group_ids[training]):
        mask = group_ids[training] == group
        weights[mask] = 1/np.sqrt(mask.sum())
    matrix = features[training]*weights[:, None]
    target = errors[training]*weights
    rank = int(np.linalg.matrix_rank(matrix))
    coefficients = np.linalg.lstsq(np.vstack((matrix, np.sqrt(ridge)*np.eye(features.shape[1]))), np.r_[target, np.zeros(features.shape[1])], rcond=None)[0]
    correction = features@coefficients
    residual = errors-correction
    report = dict(channel=channel, convention='Oracle contour neighborhoods/normals; 50% bilinear crossing, then calibration-only cubic normal-displacement correction. Not blind edge detection. Only normal errors are scored; tangential correspondence is not measured.',
                  coefficients=coefficients.tolist(), feature_count=features.shape[1], calibration_rank=rank, ridge=ridge, versions=versions(), samples=sample_rows, groups={}, splits={}, tags={})
    for split in ('calibration', 'held_out'):
        mask = training if split == 'calibration' else ~training
        report['splits'][split] = dict(before=error_stats(errors[mask]), after=error_stats(residual[mask]))
    for group in np.unique(group_ids):
        mask = group_ids == group
        tags = next(sample.get('tags', []) for sample in manifest['samples'] if sample['group_id'] == group)
        report['groups'][group] = dict(split=observations[int(np.flatnonzero(mask)[0])]['split'], tags=tags, before=error_stats(errors[mask]), after=error_stats(residual[mask]))
    for tag in sorted({tag for sample in manifest['samples'] for tag in sample.get('tags', [])}):
        selected_groups = {sample['group_id'] for sample in manifest['samples'] if tag in sample.get('tags', [])}
        mask = np.isin(group_ids, list(selected_groups))
        report['tags'][tag] = dict(before=error_stats(errors[mask]), after=error_stats(residual[mask]))
    for i, observation in enumerate(observations):
        observation.update(error_before_px=float(errors[i]), correction_px=float(correction[i]), error_after_px=float(residual[i]), corrected_xy=(observed[i]-correction[i]*normals[i]).tolist())
    report['observations'] = observations
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
