"""Replayable convergence suites across shapes, phase, depth and source seeds."""
from dataclasses import replace
from pathlib import Path
import copy
import json
import tempfile
import numpy as np
from .config import RenderConfig, integer, positive
from .dataset import resolve_psf_paths, versions
from .geometry import Primitive, Mesh, Scene, scene_from_list
from .render import render, validate_roi
from .validation import convergence, contour_crossings


def mesh_convergence(config, reference, sections, roi=None, tolerance_px=0.01, backend='auto', progress=None):
    """Compare polygonal cylinder silhouettes to exact analytic geometry.

    All mesh levels and the analytic render use the SAME contour probe points.
    Sagitta is also reported independently of the optical edge detector.
    """
    import trimesh
    if reference.kind != 'cylinder':
        raise ValueError('Mesh tolerance study currently supports cylinders')
    if len(sections) < 2 or sections != sorted(set(sections)):
        raise ValueError('Mesh sections require at least two increasing values')
    for count in sections:
        integer('sections', count, 8)
    positive('tolerance_px', tolerance_px)
    scene = Scene([reference])
    labels = scene.labels(config.camera, count=256)
    analytic = render(scene, config, roi, progress)
    analytic_edges = contour_crossings(analytic, labels)
    rows = []
    with tempfile.TemporaryDirectory(prefix='telecentric-mesh-') as temporary:
        for count in sections:
            mesh = trimesh.creation.cylinder(radius=reference.radius_mm, height=reference.length_mm, sections=count)
            if reference.axis == 'y':
                mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(reference.angle_deg), [0, 0, 1]))
            mesh.apply_translation(reference.center_mm)
            path = Path(temporary)/f'cylinder_{count}.ply'
            mesh.export(path)
            shape = Mesh(path, backend=backend)
            result = render(Scene([shape]), config, roi, progress)
            edges = contour_crossings(result, labels)
            valid = np.isfinite(edges) & np.isfinite(analytic_edges)
            sagitta = reference.radius_mm*(1-np.cos(np.pi/count))/config.camera.object_pitch_mm
            # Bound the image displacement using sampled local map Jacobians.
            jac = config.camera.distortion_jacobian(np.asarray(labels[0]['ideal_xy_px']))
            scale = np.linalg.svd(jac, compute_uv=False).max()
            row = dict(sections=count, backend=shape.backend, sagitta_ideal_px=float(sagitta), sagitta_image_px_estimate=float(sagitta*scale), paired_crossings=int(valid.sum()),
                       image_max_delta=float(np.max(np.abs(result.irradiance-analytic.irradiance))), crossing_validity_changes=int(np.count_nonzero(np.isfinite(edges) != np.isfinite(analytic_edges))))
            if valid.any():
                delta = edges[valid]-analytic_edges[valid]
                row.update(edge_max_delta_px=float(np.max(np.abs(delta))), edge_rms_delta_px=float(np.sqrt(np.mean(delta**2))))
            row['within_tolerance'] = bool(valid.any() and row.get('edge_max_delta_px', np.inf) <= tolerance_px and sagitta*scale <= tolerance_px and row['crossing_validity_changes'] == 0)
            rows.append(row)
    return dict(reference=reference.metadata(), rows=rows, last_comparison_within_tolerance=rows[-1]['within_tolerance'],
                interpretation='Mesh versus analytic render at fixed spatial/angular settings, plus ideal sagitta. Spatial/angular convergence must be checked separately. Distorted sagitta uses a sampled Jacobian scale estimate.')


def run_suite(path, output, progress=None):
    path, output = Path(path).resolve(), Path(output)
    if output.exists():
        raise ValueError('Validation output already exists')
    spec = json.loads(path.read_text())
    allowed = {'config', 'cases', 'tolerance_px', 'description'}
    if set(spec)-allowed:
        raise ValueError(f'Unknown suite keys: {set(spec)-allowed}')
    if not spec.get('cases'):
        raise ValueError('Validation cases cannot be empty')
    positive('tolerance_px', spec.get('tolerance_px', 0.01))
    report = dict(status='incomplete', versions=versions(), source_spec=spec, tolerance_px=spec.get('tolerance_px', 0.01), cases=[])
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix(output.suffix+'.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(output)

    save()
    names = set()
    allowed_case = {'id', 'config', 'shapes', 'roi', 'phases_px', 'z_offsets_mm', 'source_seeds', 'levels', 'angular_levels', 'support_levels', 'contour_count', 'mesh_sections', 'mesh_backend'}
    for case in spec['cases']:
        if set(case)-allowed_case:
            raise ValueError(f'Unknown validation case keys: {set(case)-allowed_case}')
        if case['id'] in names:
            raise ValueError('Validation case ids must be unique')
        names.add(case['id'])
        config_dict = copy.deepcopy(spec.get('config', {}))
        for key, value in case.get('config', {}).items():
            if isinstance(value, dict):
                config_dict.setdefault(key, {}).update(value)
            else:
                config_dict[key] = value
        resolve_psf_paths(config_dict.get('optics', {}), path.parent)
        config = RenderConfig.from_dict(config_dict)
        roi = validate_roi(config.camera, case.get('roi'))
        phases = case.get('phases_px', [[0, 0]])
        zs = case.get('z_offsets_mm', [0])
        seeds = case.get('source_seeds', [config.illumination.seed])
        if not phases or not zs or not seeds or not np.isfinite(zs).all():
            raise ValueError('Suite phase, height and seed sweeps must be nonempty and finite')
        for phase in phases:
            if len(phase) != 2 or not np.isfinite(phase).all():
                raise ValueError('Phase must be a finite pair')
            for z in zs:
                shapes = copy.deepcopy(case['shapes'])
                for shape in shapes:
                    key = 'translation_mm' if shape['kind'] == 'mesh' else 'center_mm'
                    shape[key] = (np.asarray(shape.get(key, [0, 0, 0]))+[* (np.asarray(phase)*config.camera.object_pitch_mm), z]).tolist()
                scene = scene_from_list(shapes, path.parent)
                for seed in seeds:
                    local = replace(config, illumination=replace(config.illumination, seed=seed))
                    study = convergence(scene, local, roi, case.get('levels', [16, 32, 64]), case.get('angular_levels'), case.get('support_levels'), report['tolerance_px'], progress, case.get('contour_count', 256))
                    row = dict(id=case['id'], phase_px=phase, z_offset_mm=z, source_seed=seed, convergence=study)
                    passed = study['last_comparisons_within_tolerance']
                    if case.get('mesh_sections'):
                        if len(scene.shapes) != 1 or not isinstance(scene.shapes[0], Primitive):
                            raise ValueError('Mesh study requires one analytic cylinder')
                        finest = replace(local, sampling=replace(local.sampling, supersample=max(case.get('levels', [16, 32, 64]))))
                        row['mesh'] = mesh_convergence(finest, scene.shapes[0], case['mesh_sections'], roi, report['tolerance_px'], case.get('mesh_backend', 'auto'), progress)
                        passed &= row['mesh']['last_comparison_within_tolerance']
                    row['within_tolerance'] = bool(passed)
                    report['cases'].append(row)
                    save()
    report.update(status='complete', all_within_tolerance=all(row['within_tolerance'] for row in report['cases']),
                  interpretation='Per-case diagnostics with explicit failures. Stability is not physical validation; untested configurations and empirical transfer remain unqualified.')
    save()
    return report
