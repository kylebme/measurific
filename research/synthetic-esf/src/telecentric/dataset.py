"""Seeded geometry/phase/exposure sweeps with self-describing NPZ + JSON."""
from dataclasses import replace
from pathlib import Path
import copy
import importlib.metadata
import json
import platform
import numpy as np
from PIL import Image
from .config import RenderConfig
from .geometry import scene_from_list
from .render import render, validate_roi
from .sensor import SensorModel


def load_job(path):
    path = Path(path).resolve()
    job = json.loads(path.read_text())
    allowed = {'config', 'shapes', 'dataset', 'roi'}
    if set(job)-allowed:
        raise ValueError(f'Unknown job keys: {set(job)-allowed}')
    config_dict = copy.deepcopy(job.get('config', {}))
    optics = config_dict.get('optics', {})
    resolve_psf_paths(optics, path.parent)
    config = RenderConfig.from_dict(config_dict)
    # Validate shapes/ROI before writing any outputs.
    scene_from_list(job['shapes'], path.parent)
    validate_roi(config.camera, job.get('roi'))
    return job, config, path.parent


def resolve_psf_paths(optics, base):
    for node in [optics, *optics.get('field_nodes', [])]:
        if node.get('measured_path'):
            node['measured_path'] = str(base/node['measured_path'])


def versions():
    result = {'python': platform.python_version(), 'renderer': '0.2.0'}
    for package in ('numpy', 'scipy', 'trimesh', 'shapely', 'embreex'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def write_sample(folder, stem, result, captures, labels, extra):
    arrays = {'irradiance': result.irradiance, 'geometric_coverage': result.geometric_coverage, 'center_mask': result.center_mask}
    for i, capture in enumerate(captures):
        for key, value in capture.items():
            arrays[f'{key}_{i:02d}'] = value
    np.savez_compressed(folder/f'{stem}.npz', **arrays)
    metadata = {**result.metadata, **extra, 'labels': labels}
    (folder/f'{stem}.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    # Preview is explicitly display-only; NPZ retains linear irradiance and RAW.
    image = np.rint(np.clip(result.irradiance/result.metadata['config']['background'], 0, 1)*255).astype(np.uint8)
    Image.fromarray(image).save(folder/f'{stem}.png')


def generate(job_path, output, count=None, progress=None):
    job, config, base = load_job(job_path)
    options = job.get('dataset', {})
    allowed = {'count', 'seed', 'random_phase', 'z_range_mm', 'magnifications', 'exposures_ms', 'repeat_captures'}
    if set(options)-allowed:
        raise ValueError(f'Unknown dataset keys: {set(options)-allowed}')
    count = options.get('count', 1) if count is None else count
    from .config import integer, positive
    integer('count', count)
    seed = options.get('seed', 2026)
    integer('dataset seed', seed, 0)
    repeats = options.get('repeat_captures', 1)
    integer('repeat_captures', repeats)
    exposures = options.get('exposures_ms', [1.0])
    magnifications = options.get('magnifications', [config.camera.magnification])
    if not exposures or not magnifications:
        raise ValueError('Exposure and magnification lists must be nonempty')
    for val in exposures + magnifications:
        positive('exposure/magnification', val)
    zrange = options.get('z_range_mm', [0., 0.])
    if len(zrange) != 2 or not np.isfinite(zrange).all() or zrange[0] > zrange[1]:
        raise ValueError('z_range_mm must be finite [min, max]')
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise ValueError('Output directory must be empty; refusing to overwrite a dataset')
    sensor = SensorModel(config.sensor, (config.camera.height, config.camera.width))
    manifest = dict(schema_version=1, status='incomplete', seed=seed, versions=versions(), source_job=job, samples=[])
    manifest_path = folder/'manifest.json'

    def save_manifest():
        temporary = folder/'manifest.tmp'
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        temporary.replace(manifest_path)

    save_manifest()
    for index in range(count):
        # Index-addressed seeds: changing total count preserves existing samples.
        geometry_seed, sensor_seed = np.random.SeedSequence([seed, index]).spawn(2)
        rng = np.random.default_rng(geometry_seed)
        mag = float(magnifications[index % len(magnifications)])
        local = replace(config, camera=replace(config.camera, magnification=mag))
        phase = rng.uniform(-0.5, 0.5, 2) if options.get('random_phase', True) else np.zeros(2)
        z = float(rng.uniform(*zrange))
        offset = [*(phase*local.camera.object_pitch_mm), z]
        specs = copy.deepcopy(job['shapes'])
        for spec in specs:
            key = 'translation_mm' if spec['kind'] == 'mesh' else 'center_mm'
            spec[key] = (np.asarray(spec.get(key, [0, 0, 0]))+offset).tolist()
        scene = scene_from_list(specs, base)
        result = render(scene, local, job.get('roi'), progress=progress)
        capture_seeds = sensor_seed.generate_state(len(exposures)*repeats).tolist()
        captures, capture_meta = [], []
        for exposure in exposures:
            for repeat in range(repeats):
                capture_seed = capture_seeds[len(captures)]
                captures.append(sensor.capture(result.irradiance, exposure, capture_seed, result.metadata['roi_xywh']))
                capture_meta.append(dict(exposure_ms=exposure, repeat=repeat, seed=capture_seed))
        stem = f'sample_{index:06d}'
        extra = dict(sample_index=index, phase_translation_px=phase.tolist(), z_translation_mm=z, captures=capture_meta)
        write_sample(folder, stem, result, captures, scene.labels(local.camera), extra)
        manifest['samples'].append(dict(index=index, arrays=stem+'.npz', metadata=stem+'.json', preview=stem+'.png'))
        save_manifest()
    manifest['status'] = 'complete'
    save_manifest()
    return manifest
