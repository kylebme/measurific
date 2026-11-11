from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from scipy.signal import convolve2d
from telecentric import Camera, Optics, Sampling, Illumination, Primitive, Scene, RenderConfig, render
from telecentric.optics import field_kernels, field_weight
from telecentric.benchmark import load_benchmark, generate_benchmark, evaluate_benchmark, correction_features, signature
from telecentric.studies import mesh_convergence, run_suite


def field_config():
    return RenderConfig(camera=Camera(width=40, height=32, pixel_pitch_um=10, magnification=1, distortion_radial=(0.02, -0.003, 0.001), distortion_tangential=(0.002, -0.001)),
                        illumination=Illumination(kind='collimated'), optics=Optics(psf='gaussian', support_px=3, field_grid=(2, 2),
                          field_nodes=({'sigma_px': 0.4}, {'sigma_px': 0.9, 'sigma_y_px': 0.4, 'angle_deg': 30}, {'psf': 'mixture', 'tail_offset_px': [-0.8, 0.2]}, {'psf': 'mixture', 'tail_offset_px': [0.9, 0.3]})),
                        sampling=Sampling(supersample=8, tile_px=11))


def test_distortion_analytic_value_inverse_and_jacobian():
    c = Camera(width=200, height=100, distortion_radial=(0.02, 0, 0))
    np.testing.assert_allclose(c.ideal_to_pixel([[200, 50], [100, 50]]), [[202, 50], [100, 50]])
    c = field_config().camera
    rng = np.random.default_rng(99)
    xy = rng.uniform([-2, -2], [42, 34], (1000, 2))
    np.testing.assert_allclose(c.pixel_to_ideal(c.ideal_to_pixel(xy)), xy, atol=2e-10)
    h = 1e-4
    numeric = np.stack([(c.ideal_to_pixel(xy+np.eye(2)[i]*h)-c.ideal_to_pixel(xy-np.eye(2)[i]*h))/(2*h) for i in range(2)], axis=-1)
    np.testing.assert_allclose(c.distortion_jacobian(xy), numeric, atol=1e-9)
    with pytest.raises(ValueError, match='folded'):
        Camera(distortion_radial=(-1, 0, 0))


def test_label_normals_and_three_coordinate_conventions():
    camera = field_config().camera
    shape = Primitive('sphere', radius_mm=0.08, center_mm=(0.07, 0.03, 0))
    label = Scene([shape]).labels(camera, count=256)[0]
    ideal = np.asarray(label['ideal_xy_px'])
    distorted = np.asarray(label['distorted_xy_px'])
    np.testing.assert_allclose(camera.ideal_to_pixel(ideal), distorted)
    np.testing.assert_allclose(camera.pixel_to_world(distorted), label['xy_mm'], atol=1e-12)
    assert np.max(np.abs(distorted-ideal)) > 0.05
    # Independent finite-difference tangent must be perpendicular to the mapped
    # normal, including tangential distortion.
    tangent = np.roll(distorted, -1, axis=0)-np.roll(distorted, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    assert np.max(np.abs(np.sum(tangent*np.asarray(label['normal_xy']), axis=1))) < 1e-5


def test_field_is_constant_preserving_and_tile_roi_independent():
    c = field_config()
    scene = Scene([Primitive('sphere', radius_mm=0.093, center_mm=(0.013, 0.009, 0))])
    full = render(scene, c)
    tiled = render(scene, replace(c, sampling=replace(c.sampling, tile_px=7)))
    patch = render(scene, c, [7, 6, 20, 18])
    np.testing.assert_allclose(full.irradiance, tiled.irradiance, atol=1e-7)
    np.testing.assert_allclose(full.irradiance[6:24, 7:27], patch.irradiance, atol=1e-7)
    np.testing.assert_allclose(render(Scene([]), c).irradiance, 1, atol=1e-7)
    assert not np.array_equal(full.center_mask, render(scene, replace(c, camera=replace(c.camera, distortion_radial=(0, 0, 0), distortion_tangential=(0, 0)))).center_mask)


def test_field_operator_against_direct_discrete_convolution():
    # Independent direct convolution reference at S=1, with true off-frame
    # geometry and per-output weights. Tests optical orientation and weighting.
    c = field_config()
    c = replace(c, sampling=Sampling(supersample=1, tile_px=9))
    scene = Scene([Primitive('box', size_mm=(0.12, 0.2, 0.01), angle_deg=17)])
    kernels, _ = field_kernels(c.optics, c.camera, 1)
    pad = kernels[0].shape[0]//2
    x, y = np.meshgrid(np.arange(-pad, c.camera.width+pad)+0.5, np.arange(-pad, c.camera.height+pad)+0.5)
    world = c.camera.pixel_to_world(np.column_stack((x.ravel(), y.ravel())))
    origins = np.column_stack((world, np.zeros(len(world))))
    transmission = (~scene.blocked(origins, np.array([0., 0., 1.]))).reshape(x.shape).astype(float)
    reference = np.zeros((c.camera.height, c.camera.width))
    weights = np.zeros_like(reference)
    for i, kernel in enumerate(kernels):
        weight = field_weight(i, c.optics, c.camera, np.arange(c.camera.width)+0.5, np.arange(c.camera.height)+0.5)
        weights += weight
        reference += convolve2d(transmission, kernel, mode='valid')*weight
    np.testing.assert_allclose(weights, 1, atol=1e-15)
    np.testing.assert_allclose(render(scene, c).irradiance, reference, atol=1e-7)


def test_measured_psf_nodes_resolve_paths(tmp_path):
    from telecentric.dataset import load_job
    np.save(tmp_path/'kernel.npy', np.ones((3, 3)))
    c = field_config().to_dict()
    c['optics']['field_grid'] = [2, 1]
    c['optics']['field_nodes'] = [{'psf': 'measured', 'measured_path': 'kernel.npy', 'measured_pitch_um': 2}, {'psf': 'none'}]
    path = tmp_path/'job.json'
    path.write_text(json.dumps(dict(config=c, shapes=[])))
    _, config, _ = load_job(path)
    assert config.optics.field_nodes[0]['measured_path'] == str(tmp_path/'kernel.npy')
    np.testing.assert_allclose(render(Scene([]), config).irradiance, 1, atol=1e-7)


@pytest.mark.parametrize('kwargs', [{'field_grid': (2, 2)}, {'field_grid': (2, 1), 'field_nodes': ({},)}, {'field_nodes': ({'object_na': 0.1},)}, {'sigma_y_px': -1}])
def test_invalid_field(kwargs):
    with pytest.raises(ValueError):
        Optics(**kwargs)


def tiny_benchmark():
    c = field_config().to_dict()
    c['sampling'] = dict(supersample=4, tile_px=16)
    return dict(config=c, exposures_ms=[0.5, 1.0], repeat_captures=2, seed=7, groups=[
        dict(id='cal', split='calibration', shapes=[dict(kind='cylinder', axis='z', radius_mm=0.06, length_mm=0.01)], placements_px=[[14, 16], [26, 16]], roi_size=[24, 24], phases_px=[[0.1, 0.2], [0.6, 0.7]]),
        dict(id='test', split='held_out', shapes=[dict(kind='sphere', radius_mm=0.075)], placements_px=[[20, 16]], roi_size=[24, 24])])


def test_artifact_leakage_even_when_renamed_and_reposed(tmp_path):
    spec = tiny_benchmark()
    spec['groups'][1]['shapes'] = [dict(kind='cylinder', radius_mm=0.06, length_mm=0.01, name='another_name', angle_deg=45, center_mm=[0, 0, 0.2])]
    path = tmp_path/'spec.json'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='leakage'):
        load_benchmark(path)


def test_artifact_signature_normalizes_numeric_types(tmp_path):
    a = [dict(kind='box', size_mm=[1, 2, 3])]
    b = [dict(kind='box', size_mm=[3.0, 2.0, 1.0], angle_deg=90, name='copy')]
    assert signature(a, tmp_path) == signature(b, tmp_path)


def test_imported_benchmark_psf_requires_honest_provenance(tmp_path):
    spec = tiny_benchmark()
    np.save(tmp_path/'kernel.npy', np.ones((5, 5)))
    spec['groups'][1]['optics'] = dict(psf='measured', field_grid=[1, 1], field_nodes=[], measured_path='kernel.npy', measured_pitch_um=1)
    path = tmp_path/'spec.json'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='provenance'):
        load_benchmark(path)
    spec['groups'][1]['kernel_provenance'] = dict(kind='synthetic', description='Uniform test kernel, not machine data', pre_pixel=True, registered_origin=True)
    path.write_text(json.dumps(spec))
    _, groups, _ = load_benchmark(path)
    assert groups[1][1].optics.measured_path == str(tmp_path/'kernel.npy')


def test_benchmark_grouping_replay_and_calibration_only_fit(tmp_path):
    spec = tiny_benchmark()
    path = tmp_path/'spec.json'
    path.write_text(json.dumps(spec))
    a = generate_benchmark(path, tmp_path/'a')
    b = generate_benchmark(path, tmp_path/'b')
    assert len(a['samples']) == 5
    assert a['status'] == 'complete'
    seen = {}
    for sa, sb in zip(a['samples'], b['samples']):
        seen.setdefault(sa['geometry_signature'], set()).add(sa['split'])
        with np.load(tmp_path/'a'/sa['arrays']) as xa, np.load(tmp_path/'b'/sb['arrays']) as xb:
            for key in xa.files:
                np.testing.assert_array_equal(xa[key], xb[key])
            assert 'raw_03' in xa.files
    assert all(len(splits) == 1 for splits in seen.values())
    first = evaluate_benchmark(tmp_path/'a', tmp_path/'eval1.json')
    # Change ONLY held-out truth: calibration coefficients must remain identical.
    held = a['samples'][-1]
    metadata_path = tmp_path/'a'/held['metadata']
    meta = json.loads(metadata_path.read_text())
    for label in meta['labels']:
        label['ideal_xy_px'] = (np.asarray(label['ideal_xy_px'])+[0.2, 0]).tolist()
    metadata_path.write_text(json.dumps(meta))
    second = evaluate_benchmark(tmp_path/'a', tmp_path/'eval2.json')
    np.testing.assert_array_equal(first['coefficients'], second['coefficients'])
    assert first['splits']['held_out']['before']['rms_px'] != second['splits']['held_out']['before']['rms_px']
    raw = evaluate_benchmark(tmp_path/'b', tmp_path/'raw.json', channel='raw_00')
    assert raw['splits']['held_out']['after']['count'] > 0
    b['samples'][-1]['group_id'] = b['samples'][0]['group_id']
    (tmp_path/'b/manifest.json').write_text(json.dumps(b))
    with pytest.raises(ValueError, match='leakage'):
        evaluate_benchmark(tmp_path/'b', tmp_path/'bad_evaluation.json')


def test_cubic_normal_field_recovery_on_unseen_positions():
    rng = np.random.default_rng(1)
    positions = rng.uniform([0, 0], [200, 100], (1000, 2))
    angles = rng.uniform(0, 2*np.pi, 1000)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    features = correction_features(positions, normals, [200, 100])
    truth = rng.normal(0, 0.05, 20)
    errors = features@truth
    fit = np.linalg.lstsq(features[:500], errors[:500], rcond=None)[0]
    np.testing.assert_allclose(features[500:]@fit, errors[500:], atol=1e-14)


def test_mesh_tolerance_is_measured_against_fixed_analytic_contour():
    c = RenderConfig(camera=Camera(width=32, height=32, pixel_pitch_um=10, magnification=1), illumination=Illumination(kind='collimated'), optics=Optics(support_px=3), sampling=Sampling(supersample=16))
    report = mesh_convergence(c, Primitive('cylinder', axis='z', radius_mm=0.07, length_mm=0.03), [16, 64, 128], backend='auto')
    rows = report['rows']
    assert rows[0]['sagitta_ideal_px'] > 0.1
    assert rows[-1]['sagitta_ideal_px'] < 0.003
    assert rows[0]['edge_max_delta_px'] > rows[-1]['edge_max_delta_px']
    assert rows[-1]['paired_crossings'] > 0


def test_suite_sweeps_and_keeps_failures(tmp_path):
    c = RenderConfig(camera=Camera(width=24, height=24, pixel_pitch_um=10, magnification=1), illumination=Illumination(kind='diffuse', samples=8), sampling=Sampling(supersample=4)).to_dict()
    spec = dict(config=c, tolerance_px=1e-12, cases=[dict(id='sphere', shapes=[dict(kind='sphere', radius_mm=0.0527)], levels=[2, 4], phases_px=[[0.1, 0.2]], z_offsets_mm=[0, 0.04], source_seeds=[3, 5])])
    path = tmp_path/'suite.json'
    path.write_text(json.dumps(spec))
    report = run_suite(path, tmp_path/'report.json')
    assert report['status'] == 'complete'
    assert len(report['cases']) == 4
    assert not report['all_within_tolerance']
